"""Sanity-check data/raw/ against the schema and counts documented in docs/1.docx.

Run once the real datasets have been placed under data/raw/:

    python -m src.data.validate_datasets

Exits non-zero with a clear message if anything is missing or doesn't match
the documented shape -- this is the P0 "load and sanity-check all three
datasets" step from PROJECT_PLAN.md, deferred until the data lands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data.schemas import (
    FACIAL_CLASS_COUNTS,
    FACIAL_IMAGE_COUNT,
    RAVDESS_CLASS_COUNTS,
    SPEECH_CLIP_COUNT,
    TABULAR_FEATURE_COLUMNS,
    TABULAR_ROW_COUNT,
    TABULAR_TARGET_CLASS_COLUMN,
    TABULAR_TARGET_SCORE_COLUMNS,
    parse_ravdess_filename,
)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"


def _report(ok: bool, message: str) -> bool:
    print(("[OK]   " if ok else "[FAIL] ") + message)
    return ok


def validate_facial() -> bool:
    root = DATA_ROOT / "facial"
    csv_path = root / "fer2013.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        total = len(df)
        ok = _report(
            total == FACIAL_IMAGE_COUNT,
            f"facial/fer2013.csv row count: {total} (expected {FACIAL_IMAGE_COUNT})",
        )
        return ok
    if not root.exists():
        return _report(False, f"facial data not found at {root} (no fer2013.csv, no class folders)")
    all_ok = True
    for emotion, expected_count in FACIAL_CLASS_COUNTS.items():
        class_dir = root / emotion
        count = len(list(class_dir.glob("*"))) if class_dir.exists() else 0
        all_ok &= _report(
            count == expected_count,
            f"facial/{emotion}: {count} images (expected {expected_count})",
        )
    return all_ok


def validate_speech() -> bool:
    root = DATA_ROOT / "speech"
    if not root.exists():
        return _report(False, f"speech data not found at {root}")
    wavs = sorted(root.glob("*.wav"))
    all_ok = _report(len(wavs) == SPEECH_CLIP_COUNT, f"speech: {len(wavs)} .wav files (expected {SPEECH_CLIP_COUNT})")

    counts: dict[str, int] = {}
    for wav in wavs:
        try:
            meta = parse_ravdess_filename(wav.name)
        except ValueError as exc:
            all_ok &= _report(False, f"speech/{wav.name}: {exc}")
            continue
        counts[meta["emotion_label"]] = counts.get(meta["emotion_label"], 0) + 1

    for emotion, expected_count in RAVDESS_CLASS_COUNTS.items():
        actual = counts.get(emotion, 0)
        all_ok &= _report(
            actual == expected_count, f"speech/{emotion}: {actual} clips (expected {expected_count})"
        )
    return all_ok


def validate_tabular() -> bool:
    csv_path = DATA_ROOT / "numerical.csv"
    if not csv_path.exists():
        return _report(False, f"tabular data not found at {csv_path}")
    df = pd.read_csv(csv_path)
    all_ok = _report(len(df) == TABULAR_ROW_COUNT, f"numerical.csv row count: {len(df)} (expected {TABULAR_ROW_COUNT})")

    required_cols = set(TABULAR_FEATURE_COLUMNS) | {TABULAR_TARGET_CLASS_COLUMN} | set(TABULAR_TARGET_SCORE_COLUMNS)
    missing = required_cols - set(df.columns)
    all_ok &= _report(not missing, f"numerical.csv missing columns: {sorted(missing) if missing else 'none'}")

    if not missing:
        all_ok &= _report(
            df[TABULAR_FEATURE_COLUMNS].isna().sum().sum() == 0,
            "numerical.csv: no missing values in the 18 feature columns",
        )
    return all_ok


def main() -> int:
    results = {
        "facial": validate_facial(),
        "speech": validate_speech(),
        "tabular": validate_tabular(),
    }
    if all(results.values()):
        print("\nAll three datasets validated against docs/1.docx.")
        return 0
    print("\nValidation failed for:", ", ".join(k for k, v in results.items() if not v))
    return 1


if __name__ == "__main__":
    sys.exit(main())
