"""Fusion-vs-single-modality ablation (docs/PROJECT_PLAN.md P6;
docs/MINDSCOPE_Blueprint.pdf Sec. 02/11: "We report each modality alone
vs. the full fusion on the same split. The headline slide: fusion beats
every single modality on macro-F1 and on score RMSE -- the empirical
justification for the whole framework.")

Consumes the per-modality baseline metrics that src/train/train_modality.py
already logs (each baseline run doubles as an ablation entry, per
PROJECT_PLAN.md P1) plus the fusion run's metrics from
src/train/train_fusion.py, and produces one comparison report.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AblationEntry:
    """One row of the ablation table -- either a single-modality baseline
    or the full fusion result, on the same held-out split.
    """

    source: str  # "face_only" | "speech_only" | "tabular_only" | "fusion"
    macro_f1: float
    weighted_f1: float
    rmse_mean: float


@dataclass
class AblationReport:
    entries: list[AblationEntry]

    @property
    def fusion_entry(self) -> AblationEntry | None:
        for entry in self.entries:
            if entry.source == "fusion":
                return entry
        return None

    def fusion_beats_every_modality(self) -> bool:
        """The headline claim this report exists to support: fusion's
        macro-F1 is the highest AND its RMSE is the lowest among all entries.
        Returns False (not raises) when the claim doesn't hold, or when
        there's no fusion entry to compare -- callers decide how to react
        (e.g. don't ship the "fusion wins" slide if this is False).
        """
        fusion = self.fusion_entry
        if fusion is None:
            return False
        others = [e for e in self.entries if e.source != "fusion"]
        if not others:
            return False
        return all(fusion.macro_f1 >= e.macro_f1 for e in others) and all(
            fusion.rmse_mean <= e.rmse_mean for e in others
        )

    def to_markdown_table(self) -> str:
        header = "| Source | Macro-F1 | Weighted-F1 | RMSE (mean) |\n|---|---|---|---|\n"
        rows = "\n".join(
            f"| {e.source} | {e.macro_f1:.3f} | {e.weighted_f1:.3f} | {e.rmse_mean:.3f} |" for e in self.entries
        )
        return header + rows

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([asdict(e) for e in self.entries], indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "AblationReport":
        data = json.loads(Path(path).read_text())
        return cls(entries=[AblationEntry(**row) for row in data])


def build_ablation_report(metrics_by_source: dict[str, dict]) -> AblationReport:
    """`metrics_by_source`: {"face_only": {"macro_f1":, "weighted_f1":,
    "rmse_mean":}, "speech_only": {...}, "tabular_only": {...},
    "fusion": {...}}. Any subset of sources may be provided (e.g. while
    only some P1 baselines have finished training); missing sources are
    simply absent from the report rather than filled with placeholder zeros.
    """
    entries = [AblationEntry(source=source, **values) for source, values in metrics_by_source.items()]
    return AblationReport(entries=entries)
