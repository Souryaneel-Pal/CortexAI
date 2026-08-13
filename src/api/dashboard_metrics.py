"""Dashboard metrics derived from the real assessment history.

Everything returned here is computed from rows in `assessments`. Nothing is
invented: with an empty table this returns zeroes and empty lists so the UI can
show an honest "no assessments yet" state, rather than plausible-looking
numbers with nothing behind them.

Uses raw sqlite3 rather than the SQLAlchemy session because these are pure
aggregate reads on the dashboard's hot path, and the ORM round-trip dominated
the query cost for no benefit.
"""
import json
import os
import sqlite3
import time
from collections import OrderedDict
from datetime import datetime

DB_PATH = "./data/cortexai.db"

TREND_MONTHS = 6
_CLASS_TO_RISK = {
    "Healthy": "Healthy",
    "Mild_Stress": "Mild",
    "Moderate_Stress": "Moderate",
    "Severe_Stress": "Severe",
}


def _parse_timestamp(value: str) -> datetime | None:
    """Parse a `completed_at` string tolerantly.

    SQLAlchemy writes naive `YYYY-MM-DD HH:MM:SS[.ffffff]` for a SQLite
    DateTime column, but a row inserted by another client (or a future schema
    change) can carry an ISO `T` separator or a `+00:00` offset. The previous
    implementation used a single hardcoded `strptime` format, so any one of
    those rows raised ValueError and took the entire dashboard down with a 500.
    A single unparseable row should cost that row, not the page.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _empty_payload(start_time: float) -> dict:
    return {
        "totalAssessments": 0,
        "heroStats": [],
        "stressDistribution": [],
        "trend": [],
        "recentAssessments": [],
        "execution_time_ms": round((time.perf_counter() - start_time) * 1000, 3),
    }


def _month_trend(timestamps: list[datetime], now: datetime) -> list[dict]:
    """Assessments per month over the trailing `TREND_MONTHS` months.

    Buckets are keyed by (year, month), not by month name. The previous version
    used a fixed `["Jul".."Dec"]` dict, which silently dropped every assessment
    made in January-June and merged different years into the same bucket -- so
    a July 2023 row and a July 2026 row counted as the same month.
    """
    buckets: "OrderedDict[tuple[int, int], int]" = OrderedDict()
    year, month = now.year, now.month
    for _ in range(TREND_MONTHS):
        buckets[(year, month)] = 0
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    # Oldest first, so the chart reads left-to-right.
    buckets = OrderedDict(reversed(list(buckets.items())))

    for stamp in timestamps:
        key = (stamp.year, stamp.month)
        if key in buckets:
            buckets[key] += 1

    return [
        {"month": datetime(year, month, 1).strftime("%b"), "assessments": count}
        for (year, month), count in buckets.items()
    ]


def _month_over_month_meta(trend: list[dict]) -> str:
    """Real change between the last two trend buckets.

    This replaces a hardcoded `"+12% this month"` string that was rendered on
    the dashboard beside genuinely-derived counts. A fabricated statistic
    presented next to real ones is indistinguishable from them, which is the
    exact failure mode this project is meant to avoid.
    """
    if len(trend) < 2:
        return "No prior month to compare"
    current, previous = trend[-1]["assessments"], trend[-2]["assessments"]
    if previous == 0:
        return "First month with recorded assessments" if current else "No assessments this month"
    change = (current - previous) / previous * 100
    return f"{change:+.0f}% vs last month"


def derive_dashboard_metrics(now: datetime | None = None) -> dict:
    start_time = time.perf_counter()
    now = now or datetime.now()

    if not os.path.exists(DB_PATH):
        # A missing database is a fresh install, not an error -- the UI should
        # show the same empty state it shows before the first assessment.
        return _empty_payload(start_time)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) AS total FROM assessments")
        total = cursor.fetchone()["total"]
        if total == 0:
            return _empty_payload(start_time)

        cursor.execute("SELECT predicted_class, COUNT(*) AS count FROM assessments GROUP BY predicted_class")
        counts = {"Healthy": 0, "Mild_Stress": 0, "Moderate_Stress": 0, "Severe_Stress": 0}
        for row in cursor.fetchall():
            if row["predicted_class"] in counts:
                counts[row["predicted_class"]] = row["count"]

        healthy, mild = counts["Healthy"], counts["Mild_Stress"]
        moderate, severe = counts["Moderate_Stress"], counts["Severe_Stress"]

        pct_healthy = round(healthy / total * 100)
        pct_mild = round(mild / total * 100)
        pct_mod = round(moderate / total * 100)
        # Give the remainder to Severe so the four slices always sum to 100
        # after rounding, and clamp so a rounding overshoot can't go negative.
        pct_sev = max(0, 100 - (pct_healthy + pct_mild + pct_mod))

        cursor.execute("SELECT completed_at FROM assessments")
        timestamps = [t for t in (_parse_timestamp(r["completed_at"]) for r in cursor.fetchall()) if t]
        trend = _month_trend(timestamps, now)

        cursor.execute(
            """
            SELECT patient_id, completed_at, predicted_class, deferred_to_human
            FROM assessments
            ORDER BY completed_at DESC
            LIMIT 10
            """
        )
        recent = []
        for row in cursor.fetchall():
            stamp = _parse_timestamp(row["completed_at"])
            recent.append(
                {
                    "patientId": row["patient_id"] or "—",
                    "dateTime": stamp.strftime("%b %d, %Y · %I:%M %p") if stamp else "Unknown date",
                    "riskLevel": _CLASS_TO_RISK.get(row["predicted_class"], "Healthy"),
                    "status": "AI Reviewing" if row["deferred_to_human"] else "Completed",
                }
            )

        return {
            "totalAssessments": total,
            "heroStats": [
                {"label": "Total Assessments", "value": f"{total:,}", "meta": _month_over_month_meta(trend)},
                {
                    "label": "Healthy Cases",
                    "value": str(healthy),
                    "meta": f"{pct_healthy}% of total",
                    "accentClassName": "border-l-secondary",
                },
                {
                    "label": "Mild Stress",
                    "value": str(mild),
                    "meta": f"{pct_mild}% of total",
                    "accentClassName": "border-l-primary-container",
                },
                {
                    "label": "Moderate Stress",
                    "value": str(moderate),
                    "meta": f"{pct_mod}% of total",
                    "accentClassName": "border-l-tertiary",
                },
                {
                    "label": "Severe Stress",
                    "value": str(severe),
                    "meta": f"{pct_sev}% of total",
                    "accentClassName": "border-l-error",
                },
            ],
            "stressDistribution": [
                {"label": "Healthy", "severity": "Healthy", "pct": pct_healthy},
                {"label": "Mild", "severity": "Mild", "pct": pct_mild},
                {"label": "Mod.", "severity": "Moderate", "pct": pct_mod},
                {"label": "Severe", "severity": "Severe", "pct": pct_sev},
            ],
            "trend": trend,
            "recentAssessments": recent,
            "execution_time_ms": round((time.perf_counter() - start_time) * 1000, 3),
        }
    finally:
        conn.close()


if __name__ == "__main__":
    print(json.dumps(derive_dashboard_metrics(), indent=2))


# ---------------------------------------------------------------------------
# Analytics
#
# Everything below is computed from stored assessments. The previous version
# returned several hardcoded values through this endpoint, which is worse than
# frontend mock data because the client cannot tell them apart from the
# genuinely-derived fields sitting beside them. Removed, and what they were:
#
#   "AI Efficacy Score: 94% -- Prediction accuracy this month"
#       Fabricated, and flatly contradicted by the project's own measurements:
#       the 4-class task scores macro-F1 0.228 against 0.270 for a stratified
#       random guess. Accuracy also cannot be computed at inference time at
#       all -- there is no ground truth for a live assessment. Replaced with
#       the human-deferral rate, which is real and actually actionable.
#
#   "delta: +12%" / "delta: +3" on the patient and alert KPIs
#       Fabricated trend arrows. Now computed month-over-month, or omitted.
#
#   "Avg. Stress Index ... / 10"
#       The value was real but the denominator was wrong: Stress_Score is on a
#       0-39 scale (docs/Dataset_Description.docx), so this rendered e.g.
#       "19.3 / 10". Now labelled with the correct range.
#
#   emotionFrequency: Anxiety 88, Agitation 62, Fatigue 74, ...
#       Hardcoded, and not even in the model's label space -- the encoders emit
#       FER (Angry/Disgust/Fear/Happy/Sad/Surprise/Neutral) and RAVDESS
#       emotions, never "Agitation" or "Apathy". Now aggregated from the stored
#       per-assessment face/speech emotion distributions.
#
#   correlation: Stress-HRV 0.89, Sleep-HRV -0.65, ...
#       Hardcoded ("Static correlation matrix matching mockData.ts"), and
#       contradicted by the data: the largest |r| between any of the 18
#       features and any score is 0.046. Now computed with a real Pearson
#       correlation over stored feature values, which shows the true ~0
#       relationship instead of inventing a strong one.
# ---------------------------------------------------------------------------
HEATMAP_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HEATMAP_HOURS = ["2A", "4A", "6A", "8A", "10A", "12P", "2P", "4P", "6P", "8P", "10P", "12A"]
RISK_SEGMENTS = ("Adults", "Seniors", "Adolescents")
STRESS_SCORE_MAX = 39  # docs/Dataset_Description.docx

# Correlations worth showing a clinician: behavioural/physiological drivers
# against the predicted stress score.
CORRELATION_FEATURES = ("HRV_Index", "GSR_Level", "Skin_Temperature")
CORRELATION_ROWS = ("Stress_Score", "Sleep_Quality", "Social_Engagement")


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r, or None when it is undefined (n < 2, or zero variance)."""
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = sum(d * d for d in dx) ** 0.5
    denom_y = sum(d * d for d in dy) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (denom_x * denom_y)


def _empty_analytics() -> dict:
    return {"kpis": [], "heatmap": [], "emotionFrequency": [], "riskBreakdown": [], "correlation": []}


def derive_analytics_metrics(now: datetime | None = None) -> dict:
    """Cohort analytics computed entirely from stored assessments."""
    now = now or datetime.now()
    if not os.path.exists(DB_PATH):
        return _empty_analytics()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT patient_id, completed_at, demographic, predicted_class, deferred_to_human,
                   stress_score, masked_distress_index, face_emotion_probs, speech_emotion_probs,
                   tabular_features
            FROM assessments
            """
        )
        rows = cursor.fetchall()
        if not rows:
            return _empty_analytics()

        total = len(rows)
        unique_patients = len({r["patient_id"] for r in rows if r["patient_id"]})
        deferred = sum(1 for r in rows if r["deferred_to_human"])

        critical = 0
        stress_values: list[float] = []
        heat = {}
        risk = {seg: {"low": 0, "medium": 0, "high": 0} for seg in RISK_SEGMENTS}
        emotion_totals: dict[str, float] = {}
        emotion_counts: dict[str, int] = {}
        feature_rows: list[dict] = []
        this_month = 0
        last_month = 0
        prev_year, prev_month = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)

        for row in rows:
            if row["stress_score"] is not None:
                stress_values.append(row["stress_score"])

            mdi_flagged = False
            if row["masked_distress_index"]:
                try:
                    mdi_flagged = bool(json.loads(row["masked_distress_index"]).get("flag", False))
                except (json.JSONDecodeError, AttributeError):
                    mdi_flagged = False
            if row["predicted_class"] == "Severe_Stress" or mdi_flagged:
                critical += 1

            stamp = _parse_timestamp(row["completed_at"])
            if stamp:
                key = (stamp.strftime("%a"), HEATMAP_HOURS[min(11, stamp.hour // 2)])
                heat[key] = heat.get(key, 0) + 1
                if (stamp.year, stamp.month) == (now.year, now.month):
                    this_month += 1
                elif (stamp.year, stamp.month) == (prev_year, prev_month):
                    last_month += 1

            segment = row["demographic"] if row["demographic"] in RISK_SEGMENTS else "Adults"
            if row["predicted_class"] in ("Healthy", "Mild_Stress"):
                risk[segment]["low"] += 1
            elif row["predicted_class"] == "Moderate_Stress":
                risk[segment]["medium"] += 1
            elif row["predicted_class"] == "Severe_Stress":
                risk[segment]["high"] += 1

            # Emotion frequency from what the encoders actually emit.
            for column in ("face_emotion_probs", "speech_emotion_probs"):
                if not row[column]:
                    continue
                try:
                    probs = json.loads(row[column])
                except (json.JSONDecodeError, TypeError):
                    continue
                for emotion, value in (probs or {}).items():
                    label = emotion.capitalize()
                    emotion_totals[label] = emotion_totals.get(label, 0.0) + float(value)
                    emotion_counts[label] = emotion_counts.get(label, 0) + 1

            if row["tabular_features"]:
                try:
                    parsed = json.loads(row["tabular_features"])
                    if isinstance(parsed, dict):
                        parsed["Stress_Score"] = row["stress_score"]
                        feature_rows.append(parsed)
                except (json.JSONDecodeError, TypeError):
                    pass

        max_count = max(heat.values()) if heat else 1
        heatmap = [
            {"day": day, "hour": hour, "intensity": round(heat.get((day, hour), 0) / max_count, 2)}
            for day in HEATMAP_DAYS
            for hour in HEATMAP_HOURS
        ]

        risk_breakdown = []
        for segment in RISK_SEGMENTS:
            counts = risk[segment]
            seg_total = sum(counts.values())
            if seg_total == 0:
                continue  # no data for this segment -- omit rather than invent a split
            risk_breakdown.append(
                {
                    "segment": segment,
                    "low": round(counts["low"] / seg_total * 100),
                    "medium": round(counts["medium"] / seg_total * 100),
                    "high": round(counts["high"] / seg_total * 100),
                }
            )

        emotion_frequency = sorted(
            (
                {"emotion": label, "value": round(emotion_totals[label] / emotion_counts[label] * 100)}
                for label in emotion_totals
            ),
            key=lambda item: item["value"],
            reverse=True,
        )[:6]

        correlation = []
        for row_label in CORRELATION_ROWS:
            for col_label in CORRELATION_FEATURES:
                pairs = [
                    (r[row_label], r[col_label])
                    for r in feature_rows
                    if r.get(row_label) is not None and r.get(col_label) is not None
                ]
                r_value = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
                if r_value is None:
                    continue
                correlation.append(
                    {
                        "rowLabel": row_label.replace("_Score", "").replace("_", " "),
                        "colLabel": col_label.replace("_Index", "").replace("_Level", "").replace("_", " "),
                        "value": round(r_value, 2),
                    }
                )

        avg_stress = round(sum(stress_values) / len(stress_values), 1) if stress_values else 0.0
        if last_month > 0:
            patients_delta = f"{(this_month - last_month) / last_month * 100:+.0f}%"
            delta_direction = "up" if this_month >= last_month else "down"
        else:
            patients_delta, delta_direction = None, None

        kpis = [
            {
                "label": "Active Patients",
                "value": f"{unique_patients:,}",
                **({"delta": patients_delta, "deltaDirection": delta_direction} if patients_delta else {}),
                "sub": f"{total:,} assessments",
            },
            {
                "label": "Critical Alerts",
                "value": str(critical),
                "sub": "Severe class or flagged Masked-Distress Index",
            },
            {
                # NOT an accuracy figure: a live assessment has no ground truth
                # to score against. This is how often the uncertainty gate sent
                # a case to a human, which is real and actionable.
                "label": "Deferred to Human",
                "value": f"{round(deferred / total * 100)}%",
                "sub": f"{deferred:,} of {total:,} below the confidence gate",
                "isAi": True,
            },
            {
                "label": "Avg. Stress Index",
                "value": str(avg_stress),
                "sub": f"/ {STRESS_SCORE_MAX}",
            },
        ]

        return {
            "kpis": kpis,
            "heatmap": heatmap,
            "emotionFrequency": emotion_frequency,
            "riskBreakdown": risk_breakdown,
            "correlation": correlation,
        }
    finally:
        conn.close()
