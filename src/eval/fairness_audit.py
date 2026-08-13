"""Fairness audit split by RAVDESS actor gender metadata (docs/PROJECT_PLAN.md
P6; docs/MINDSCOPE_Blueprint.pdf Sec. 11 "FAIRNESS: Subgroup metrics (gender
via actor metadata; class balance) reported openly, with the known
FER/RAVDESS demographic limits acknowledged").

RAVDESS is gender-balanced by construction (odd actor IDs male, even
female, docs/1.docx), which is exactly what makes it usable for this audit
-- src/data/schemas.ravdess_actor_gender is the single source of truth for
the gender label used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.eval.metrics import ClassificationMetrics, compute_classification_metrics


@dataclass
class FairnessAuditResult:
    per_group: dict[str, ClassificationMetrics]
    macro_f1_gap: float
    accuracy_gap: float
    group_sizes: dict[str, int]

    def summary(self) -> str:
        lines = ["Fairness audit (by RAVDESS actor gender):"]
        for group, metrics in sorted(self.per_group.items()):
            n = self.group_sizes[group]
            lines.append(f"  {group:8s} (n={n:4d}): macro_f1={metrics.macro_f1:.3f}  accuracy={metrics.accuracy:.3f}")
        lines.append(f"  macro_f1 gap: {self.macro_f1_gap:.3f}   accuracy gap: {self.accuracy_gap:.3f}")
        return "\n".join(lines)


def fairness_audit_by_gender(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genders: list[str],
    y_proba: np.ndarray | None = None,
    num_classes: int | None = None,
) -> FairnessAuditResult:
    """`genders`: per-sample "male"/"female" labels, index-aligned with
    y_true/y_pred (from src/data/schemas.ravdess_actor_gender or
    parse_ravdess_filename's "gender" field). Computes the full
    Metrics_Used.docx classification suite per gender subgroup, plus the
    macro-F1 and accuracy gap between groups -- the headline fairness
    signal a health-AI judge will ask about first.
    """
    genders = np.asarray(genders)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_groups = sorted(set(genders.tolist()))
    if len(unique_groups) < 2:
        raise ValueError(f"Need at least 2 gender groups to audit fairness, found: {unique_groups}")

    kwargs = {"num_classes": num_classes} if num_classes is not None else {}
    per_group: dict[str, ClassificationMetrics] = {}
    group_sizes: dict[str, int] = {}
    for group in unique_groups:
        mask = genders == group
        group_proba = y_proba[mask] if y_proba is not None else None
        per_group[group] = compute_classification_metrics(y_true[mask], y_pred[mask], group_proba, **kwargs)
        group_sizes[group] = int(mask.sum())

    macro_f1_values = [m.macro_f1 for m in per_group.values()]
    accuracy_values = [m.accuracy for m in per_group.values()]

    return FairnessAuditResult(
        per_group=per_group,
        macro_f1_gap=float(max(macro_f1_values) - min(macro_f1_values)),
        accuracy_gap=float(max(accuracy_values) - min(accuracy_values)),
        group_sizes=group_sizes,
    )
