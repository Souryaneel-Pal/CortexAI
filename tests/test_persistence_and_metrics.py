"""Regression tests for the persistence + dashboard-metrics layer.

Each test here pins a defect that was found by static analysis and fixed --
they exist so the specific failure cannot come back quietly.
"""
from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.api import dashboard_metrics
from src.api.dashboard_metrics import (
    _month_over_month_meta,
    _month_trend,
    _parse_timestamp,
    derive_dashboard_metrics,
)
from src.api.database import DEFAULT_SETTINGS, DBAssessment, DBSetting


# ---------------------------------------------------------------------------
# No fabricated history
# ---------------------------------------------------------------------------
def test_bootstrap_seeds_settings_only_and_never_assessments():
    """The old seeder inserted 15 invented assessments -- including narratives
    stamped `report_generator="ollama:llama3.1"`, i.e. hand-written text
    presented as model output. Configuration defaults are legitimate to seed;
    clinical history is not."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import src.api.database as database

    with tempfile.TemporaryDirectory() as tmp:
        engine = create_engine(f"sqlite:///{Path(tmp) / 'test.db'}", connect_args={"check_same_thread": False})
        database.Base.metadata.create_all(bind=engine)
        session = sessionmaker(bind=engine)()

        database.seed_default_settings(session)

        assert session.query(DBAssessment).count() == 0, "bootstrap must not fabricate assessment history"
        assert {s.key for s in session.query(DBSetting).all()} == set(DEFAULT_SETTINGS)
        session.close()


def test_seed_default_settings_does_not_clobber_tuned_values():
    """An operator's tuned threshold must survive a restart."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import src.api.database as database

    with tempfile.TemporaryDirectory() as tmp:
        engine = create_engine(f"sqlite:///{Path(tmp) / 'test.db'}", connect_args={"check_same_thread": False})
        database.Base.metadata.create_all(bind=engine)
        session = sessionmaker(bind=engine)()

        session.add(DBSetting(key="uncertainty_threshold", value="0.90"))
        session.commit()

        database.seed_default_settings(session)
        stored = session.query(DBSetting).filter(DBSetting.key == "uncertainty_threshold").first()
        assert stored.value == "0.90"
        session.close()


# ---------------------------------------------------------------------------
# Trend window
# ---------------------------------------------------------------------------
def test_trend_covers_a_rolling_window_not_a_hardcoded_jul_to_dec():
    """The old implementation bucketed into a fixed ["Jul".."Dec"] dict, so
    every January-June assessment was silently dropped from the chart."""
    now = datetime(2026, 3, 15)
    stamps = [datetime(2026, 1, 5), datetime(2026, 2, 9), datetime(2026, 3, 1)]

    trend = _month_trend(stamps, now)
    months = [bucket["month"] for bucket in trend]

    assert months == ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
    counted = {bucket["month"]: bucket["assessments"] for bucket in trend}
    assert counted["Jan"] == 1 and counted["Feb"] == 1 and counted["Mar"] == 1


def test_trend_does_not_merge_the_same_month_from_different_years():
    """Bucketing by month *name* collapsed Jul-2023 and Jul-2026 together."""
    now = datetime(2026, 8, 1)
    stamps = [datetime(2023, 8, 1), datetime(2026, 8, 2), datetime(2026, 8, 3)]

    trend = _month_trend(stamps, now)
    august = [b for b in trend if b["month"] == "Aug"]

    assert len(august) == 1
    assert august[0]["assessments"] == 2, "the 2023 row must not be counted in the 2026 bucket"


# ---------------------------------------------------------------------------
# No fabricated statistics
# ---------------------------------------------------------------------------
def test_month_over_month_is_computed_not_a_hardcoded_string():
    """The hero card's meta line used to read a literal "+12% this month"
    rendered beside genuinely-derived counts."""
    assert _month_over_month_meta([{"assessments": 10}, {"assessments": 15}]) == "+50% vs last month"
    assert _month_over_month_meta([{"assessments": 20}, {"assessments": 10}]) == "-50% vs last month"
    assert "12%" not in _month_over_month_meta([{"assessments": 1}, {"assessments": 1}])


def test_month_over_month_handles_a_zero_baseline():
    assert "First month" in _month_over_month_meta([{"assessments": 0}, {"assessments": 5}])
    assert "No assessments" in _month_over_month_meta([{"assessments": 0}, {"assessments": 0}])
    assert "No prior month" in _month_over_month_meta([{"assessments": 3}])


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "2026-08-13 13:55:36.455166",  # SQLAlchemy's usual SQLite format
        "2026-08-13 13:55:36",
        "2026-08-13T13:55:36",  # ISO separator
        "2026-08-13 13:55:36+00:00",  # timezone-aware writer
    ],
)
def test_timestamp_parsing_accepts_every_format_a_writer_may_produce(raw):
    """A single hardcoded strptime format meant one oddly-formatted row took
    the whole dashboard down with a 500."""
    assert _parse_timestamp(raw) is not None


def test_unparseable_timestamp_costs_one_row_not_the_page():
    assert _parse_timestamp("not-a-date") is None
    assert _parse_timestamp("") is None


# ---------------------------------------------------------------------------
# Empty history
# ---------------------------------------------------------------------------
def _make_empty_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE assessments (session_id TEXT PRIMARY KEY, completed_at TEXT, patient_id TEXT, "
        "predicted_class TEXT, deferred_to_human INTEGER)"
    )
    conn.commit()
    conn.close()


def test_empty_history_returns_zeroes_rather_than_inventing_numbers(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "empty.db"
        _make_empty_db(db_path)
        monkeypatch.setattr(dashboard_metrics, "DB_PATH", str(db_path))

        payload = derive_dashboard_metrics()

        assert payload["totalAssessments"] == 0
        assert payload["heroStats"] == []
        assert payload["stressDistribution"] == []
        assert payload["recentAssessments"] == []


def test_missing_database_is_an_empty_state_not_a_500(monkeypatch):
    """A fresh install has no database file yet; that is not an error."""
    monkeypatch.setattr(dashboard_metrics, "DB_PATH", "/nonexistent/path/cortexai.db")

    payload = derive_dashboard_metrics()
    assert payload["totalAssessments"] == 0
    assert "error" not in payload


def test_stress_distribution_percentages_sum_to_100(monkeypatch):
    """Rounding four independent percentages can overshoot or undershoot; the
    donut must still add up."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "rounding.db"
        _make_empty_db(db_path)
        conn = sqlite3.connect(db_path)
        # 3 rows of each class: 33.33% each, which naive rounding turns into 132%.
        for i in range(9):
            cls = ["Healthy", "Mild_Stress", "Moderate_Stress"][i % 3]
            conn.execute(
                "INSERT INTO assessments VALUES (?,?,?,?,?)",
                (f"s{i}", "2026-08-13 10:00:00", f"PT-{i}", cls, 0),
            )
        conn.commit()
        conn.close()
        monkeypatch.setattr(dashboard_metrics, "DB_PATH", str(db_path))

        payload = derive_dashboard_metrics()
        assert sum(slice_["pct"] for slice_ in payload["stressDistribution"]) == 100
        assert all(slice_["pct"] >= 0 for slice_ in payload["stressDistribution"])


# ---------------------------------------------------------------------------
# Analytics: no fabricated statistics
# ---------------------------------------------------------------------------
def test_analytics_reports_deferral_rate_not_a_fabricated_accuracy():
    """The KPI row used to carry `"AI Efficacy Score": "94%"` subtitled
    "Prediction accuracy this month". That number was invented, contradicted
    the project's measured macro-F1 of 0.228, and is not even computable --
    a live assessment has no ground truth to score against. It is replaced by
    the human-deferral rate, which is real."""
    from src.api.dashboard_metrics import derive_analytics_metrics

    labels = {kpi["label"] for kpi in derive_analytics_metrics()["kpis"]}
    assert "AI Efficacy Score" not in labels
    assert "Deferred to Human" in labels


def test_analytics_stress_index_uses_the_documented_scale():
    """`Stress_Score` is 0-39; the KPI used to be subtitled "/ 10"."""
    from src.api.dashboard_metrics import STRESS_SCORE_MAX, derive_analytics_metrics

    assert STRESS_SCORE_MAX == 39
    kpi = next(k for k in derive_analytics_metrics()["kpis"] if k["label"] == "Avg. Stress Index")
    assert kpi["sub"] == "/ 39"


def test_analytics_emotions_come_from_the_models_label_space():
    """The hardcoded list contained "Agitation", "Apathy" and "Fatigue" --
    labels no encoder in this system can emit. Real labels come from FER
    (7 classes) and RAVDESS (8 classes)."""
    from src.api.dashboard_metrics import derive_analytics_metrics

    emitted = {e["emotion"].lower() for e in derive_analytics_metrics()["emotionFrequency"]}
    invented = {"agitation", "apathy", "fatigue", "anxiety"}
    assert not (emitted & invented), f"analytics reported labels no model produces: {emitted & invented}"

    valid = {
        "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",  # FER
        "calm", "fearful", "surprised",  # RAVDESS extras
    }
    assert emitted <= valid, f"unexpected emotion labels: {emitted - valid}"


def test_pearson_correlation_is_computed_not_hardcoded():
    from src.api.dashboard_metrics import _pearson

    assert _pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert _pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    # Undefined cases return None rather than a fabricated number.
    assert _pearson([1.0], [2.0]) is None
    assert _pearson([1, 1, 1], [1, 2, 3]) is None


def test_analytics_correlations_are_not_the_old_static_matrix():
    """The old matrix hardcoded Stress-HRV = 0.89, contradicting the measured
    maximum |r| of 0.046 across the whole dataset."""
    from src.api.dashboard_metrics import derive_analytics_metrics

    for cell in derive_analytics_metrics()["correlation"]:
        assert -1.0 <= cell["value"] <= 1.0
        if cell["rowLabel"] == "Stress" and cell["colLabel"] == "HRV":
            assert cell["value"] != 0.89, "the static mock correlation matrix is back"
