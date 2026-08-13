"""End-to-end evaluation driver (PROJECT_PLAN.md P5/P6).

Produces, from the actually-trained checkpoints and the real held-out split:

  1. The full metric suite from docs/Metrics_Used.docx -- classification
     (accuracy, precision, recall, F1, macro-F1, weighted-F1, ROC-AUC,
     confusion matrix) and regression (MAE, MSE, RMSE, R2, explained
     variance) per score.
  2. The proof-of-fusion ablation: each modality alone vs. full fusion, on
     the *same* validation split, by masking modalities through the fusion
     gate rather than training four separate models.
  3. The fairness audit across RAVDESS actor gender, on the speech encoder's
     held-out actors.

Everything is written to artifacts/reports/ as JSON plus a Markdown summary.

Usage:
    python -m src.eval.run_evaluation
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.data.loaders import DEFAULT_SPEECH_ROOT, SpeechEmotionDataset
from src.data.schemas import SPEECH_EMOTIONS, STRESS_LEVEL_NAMES, StressLevel, TABULAR_TARGET_SCORE_COLUMNS
from src.eval.ablation import build_ablation_report
from src.eval.fairness_audit import fairness_audit_by_gender
from src.eval.metrics import compute_classification_metrics, compute_multitarget_regression_metrics
from src.train.train_fusion import FusionLightningModule, build_fusion_dataloaders, load_config
from src.train.train_modality import build_dataloaders_speech, pick_accelerator

REPORTS_DIR = Path("artifacts/reports")

# Which modalities stay visible for each ablation arm. The fusion gate's
# modality_mask replaces a masked modality's embedding with the learned
# "missing" token, which is exactly the missing-modality path the model was
# trained to degrade into -- so every arm is measured on one model and one
# split, with no retraining and no split drift.
ABLATION_ARMS = {
    "face_only": {"face": True, "speech": False, "tabular": False},
    "speech_only": {"face": False, "speech": True, "tabular": False},
    "tabular_only": {"face": False, "speech": False, "tabular": True},
    "fusion": None,  # nothing masked
}


@torch.no_grad()
def _collect_predictions(module, loader, device, mask_spec):
    """Run the fusion model over `loader`, optionally masking modalities."""
    all_logits, all_labels, all_scores, all_score_preds, all_weights = [], [], [], [], []

    for batch in loader:
        features, labels, scores, image, waveform = (b.to(device) for b in batch)
        batch_size = features.shape[0]

        modality_mask = None
        if mask_spec is not None:
            modality_mask = {
                name: torch.full((batch_size,), present, dtype=torch.bool, device=device)
                for name, present in mask_spec.items()
            }

        face_embedding, _ = module.face_encoder(image)
        speech_embedding, _ = module.speech_encoder(waveform)
        tabular_embedding, _, _ = module.tabular_encoder(features)
        fused, weights = module.fusion(
            face_embedding, speech_embedding, tabular_embedding, modality_mask=modality_mask
        )
        class_logits, _probs, score_preds = module.heads(fused)

        all_logits.append(class_logits.float().cpu())
        all_labels.append(labels.cpu())
        all_scores.append(scores.float().cpu())
        all_score_preds.append(score_preds.float().cpu())
        all_weights.append(weights.float().cpu())

    logits = torch.cat(all_logits)
    return {
        "probs": torch.softmax(logits, dim=-1).numpy(),
        "preds": logits.argmax(dim=-1).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "scores": torch.cat(all_scores).numpy(),
        "score_preds": torch.cat(all_score_preds).numpy(),
        "modality_weights": torch.cat(all_weights).numpy(),
    }


def evaluate_fusion(config_path: str = "configs/fusion.yaml", checkpoint: str | None = None) -> dict:
    cfg = load_config(config_path)
    checkpoint = checkpoint or str(Path(cfg.checkpoint_dir) / "best.ckpt")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"No fusion checkpoint at {checkpoint}. Train it first: "
            "python -m src.train.train_fusion --config configs/fusion.yaml"
        )

    _train_loader, val_loader, class_counts, _pairs = build_fusion_dataloaders(cfg)

    module = FusionLightningModule(cfg, class_counts)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    module.load_state_dict(state.get("state_dict", state), strict=False)

    device = torch.device("mps" if pick_accelerator() == "mps" else "cpu")
    module.to(device).eval()

    results: dict = {"arms": {}}
    ablation_inputs: dict[str, dict] = {}

    for arm, mask_spec in ABLATION_ARMS.items():
        out = _collect_predictions(module, val_loader, device, mask_spec)
        cls = compute_classification_metrics(out["labels"], out["preds"], out["probs"])
        reg = compute_multitarget_regression_metrics(out["scores"], out["score_preds"])

        results["arms"][arm] = {
            "classification": asdict(cls),
            "regression": asdict(reg),
            "mean_modality_weights": dict(
                zip(("face", "speech", "tabular"), out["modality_weights"].mean(axis=0).round(4).tolist())
            ),
        }
        ablation_inputs[arm] = {
            "macro_f1": float(cls.macro_f1),
            "weighted_f1": float(cls.weighted_f1),
            "rmse_mean": float(reg.mean_rmse),
        }

    report = build_ablation_report(ablation_inputs)
    results["ablation_markdown"] = report.to_markdown_table()
    results["fusion_beats_every_modality"] = report.fusion_beats_every_modality()
    results["val_class_distribution"] = {
        STRESS_LEVEL_NAMES[StressLevel(int(level))]: int(count)
        for level, count in zip(*np.unique(out["labels"], return_counts=True))
    }
    results["score_targets"] = TABULAR_TARGET_SCORE_COLUMNS
    return results


def evaluate_speech_fairness(config_path: str = "configs/speech.yaml") -> dict:
    """Per-gender speech-emotion metrics on the held-out actors.

    RAVDESS is gender-balanced by construction (odd actor IDs male, even
    female), which is what makes this audit meaningful rather than decorative.
    """
    cfg = load_config(config_path)
    checkpoint = Path(cfg.checkpoint_dir) / "best.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No speech checkpoint at {checkpoint}.")

    from src.models.speech_net import build_speech_encoder

    model = build_speech_encoder(
        backbone=cfg.model.backbone,
        pretrained_checkpoint=cfg.model.get("pretrained_checkpoint", "facebook/wav2vec2-base"),
        freeze_feature_extractor=cfg.model.get("freeze_feature_extractor", True),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = {k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("mps" if pick_accelerator() == "mps" else "cpu")
    model.to(device).eval()

    _train_loader, val_loader, held_out_actors = build_dataloaders_speech(cfg)

    preds, labels, genders, probs = [], [], [], []
    with torch.no_grad():
        for waveforms, batch_labels, meta in val_loader:
            _emb, logits = model(waveforms.to(device))
            p = torch.softmax(logits.float().cpu(), dim=-1)
            probs.append(p)
            preds.append(p.argmax(dim=-1))
            labels.append(batch_labels)
            genders.extend(meta["gender"])

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    probs = torch.cat(probs).numpy()

    audit = fairness_audit_by_gender(
        labels, preds, genders, y_proba=probs, num_classes=len(SPEECH_EMOTIONS)
    )
    overall = compute_classification_metrics(labels, preds, probs, num_classes=len(SPEECH_EMOTIONS))

    return {
        "held_out_actors": held_out_actors,
        "n_val_clips": int(len(labels)),
        "overall": asdict(overall),
        "per_gender": {group: asdict(m) for group, m in audit.per_group.items()},
        "group_sizes": audit.group_sizes,
        "macro_f1_gap": audit.macro_f1_gap,
        "accuracy_gap": audit.accuracy_gap,
        "summary": audit.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-config", default="configs/fusion.yaml")
    parser.add_argument("--speech-config", default="configs/speech.yaml")
    parser.add_argument("--skip-fairness", action="store_true")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output: dict = {}

    output["fusion_evaluation"] = evaluate_fusion(args.fusion_config)
    print(output["fusion_evaluation"]["ablation_markdown"])
    print(f"\nfusion beats every single modality: {output['fusion_evaluation']['fusion_beats_every_modality']}")

    if not args.skip_fairness:
        try:
            output["speech_fairness"] = evaluate_speech_fairness(args.speech_config)
            print("\n" + output["speech_fairness"]["summary"])
        except FileNotFoundError as exc:
            print(f"Skipping fairness audit: {exc}")

    (REPORTS_DIR / "evaluation.json").write_text(json.dumps(output, indent=2, default=float))
    print(f"\nWrote {REPORTS_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
