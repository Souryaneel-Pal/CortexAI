"""Fusion + multi-task training (docs/PROJECT_PLAN.md P2).

Anchored training: the tabular dataset is the only labelled ground truth.
Each batch pairs a tabular row with a weakly-matched face image and speech
clip (src/data/loaders.py FusionPairDataset -- matched-emotion sampling via
src/data/emotion_stress_map.py, never a fabricated identity match).

Two metrics are always reported side by side so the numbers stay honest
(docs/MINDSCOPE_Blueprint.pdf Sec. 02 "tabular-only fallback"):
  - val_macro_f1_fusion / val_rmse_fusion: full face+speech+tabular fusion
  - val_macro_f1_tabular_only / val_rmse_tabular_only: same batch, face and
    speech masked out via the fusion gate's modality_mask, i.e. what the
    system reports when only the labelled tabular signal is available. This
    is also exactly the "coupled inference only when a real trio exists"
    path used at demo/inference time (src/api/inference.py, P5).

Usage:
    python -m src.train.train_fusion --config configs/fusion.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.loaders import (
    DEFAULT_FACIAL_ROOT,
    DEFAULT_SPEECH_ROOT,
    DEFAULT_TABULAR_CSV,
    FacialEmotionDataset,
    FusionPairDataset,
    SpeechEmotionDataset,
    TabularMentalHealthDataset,
)
from src.train.train_modality import pick_accelerator
from src.data.schemas import STRESS_LABEL_NAME_TO_LEVEL
from src.eval.metrics import compute_classification_metrics, compute_multitarget_regression_metrics
from src.models.face_cnn import FaceEmotionEncoder
from src.models.fusion import GatedCrossModalFusion
from src.models.heads import FusionHeads
from src.models.speech_net import build_speech_encoder
from src.models.tabular_ft import TabularEncoder
from src.train.losses import (
    ClassBalancedFocalLoss,
    UncertaintyWeightedMultiTaskLoss,
    class_balanced_weights,
    consistency_loss,
    score_regression_loss,
)

logger = logging.getLogger(__name__)


def _load_or_warn(model: torch.nn.Module, checkpoint_path: str, name: str) -> torch.nn.Module:
    path = Path(checkpoint_path)
    if path.exists():
        state = torch.load(path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        # Lightning checkpoints prefix keys with "model." -- strip it if present.
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded {name} encoder checkpoint from {path}")
    else:
        logger.warning(
            f"No {name} encoder checkpoint at {path} -- using a freshly-initialized encoder. "
            f"Train P1 baselines first (src/train/train_modality.py) for real embeddings."
        )
    return model


class FusionLightningModule(pl.LightningModule):
    def __init__(self, cfg, tabular_class_counts: dict[str, int]):
        super().__init__()
        self.cfg = cfg

        self.face_encoder = FaceEmotionEncoder(backbone=cfg.encoders.face_backbone, pretrained=False)
        self.speech_encoder = build_speech_encoder(backbone=cfg.encoders.speech_backbone)
        self.tabular_encoder = TabularEncoder(backbone=cfg.encoders.tabular_backbone)

        _load_or_warn(self.face_encoder, cfg.encoders.face_checkpoint, "face")
        _load_or_warn(self.speech_encoder, cfg.encoders.speech_checkpoint, "speech")
        _load_or_warn(self.tabular_encoder, cfg.encoders.tabular_checkpoint, "tabular")

        if cfg.encoders.freeze_pretrained:
            for encoder in (self.face_encoder, self.speech_encoder, self.tabular_encoder):
                for param in encoder.parameters():
                    param.requires_grad = False
                encoder.eval()

        self.fusion = GatedCrossModalFusion(
            embed_dim=cfg.encoders.embed_dim,
            hidden_dim=cfg.fusion.hidden_dim,
            n_layers=cfg.fusion.n_layers,
            n_heads=cfg.fusion.n_heads,
            modality_dropout_p=cfg.fusion.modality_dropout_p,
            dropout=cfg.fusion.dropout,
        )
        self.heads = FusionHeads(embed_dim=cfg.encoders.embed_dim)

        weights = class_balanced_weights(tabular_class_counts, beta=0.999)
        self.classification_criterion = ClassBalancedFocalLoss(class_weights=weights, gamma=2.0)
        self.task_weighting = UncertaintyWeightedMultiTaskLoss(num_tasks=2)
        self.consistency_weight = cfg.heads.consistency_loss_weight

        self._val_outputs: list[dict] = []

    def train(self, mode: bool = True):
        """Keep frozen encoders in eval() even when Lightning calls
        model.train() each epoch -- requires_grad=False alone does not stop
        BatchNorm running-stat updates or Dropout from staying active.
        """
        super().train(mode)
        if self.cfg.encoders.freeze_pretrained:
            self.face_encoder.eval()
            self.speech_encoder.eval()
            self.tabular_encoder.eval()
        return self

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.train.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _encode(self, features, image, waveform):
        with torch.set_grad_enabled(not self.cfg.encoders.freeze_pretrained):
            face_embedding, _ = self.face_encoder(image)
            speech_embedding, _ = self.speech_encoder(waveform)
            tabular_embedding, _, _ = self.tabular_encoder(features)
        return face_embedding, speech_embedding, tabular_embedding

    def _label_idx(self, labels, device):
        if isinstance(labels, (list, tuple)):
            return torch.tensor([int(STRESS_LABEL_NAME_TO_LEVEL[l]) for l in labels], device=device)
        return labels

    def _forward_and_loss(self, batch, modality_mask=None):
        features, labels, scores, image, waveform = batch
        label_idx = self._label_idx(labels, features.device)

        face_embedding, speech_embedding, tabular_embedding = self._encode(features, image, waveform)
        fused, modality_weights = self.fusion(
            face_embedding, speech_embedding, tabular_embedding, modality_mask=modality_mask
        )
        class_logits, class_probs, score_preds = self.heads(fused)

        cls_loss = self.classification_criterion(class_logits, label_idx)
        reg_loss = score_regression_loss(score_preds, scores)
        main_loss = self.task_weighting([cls_loss, reg_loss])
        cons_loss = consistency_loss(class_probs, score_preds)
        total_loss = main_loss + self.consistency_weight * cons_loss

        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "consistency_loss": cons_loss.detach(),
            "class_logits": class_logits.detach(),
            "class_probs": class_probs.detach(),
            "score_preds": score_preds.detach(),
            "label_idx": label_idx.detach(),
            "scores": scores.detach(),
            "modality_weights": modality_weights.detach(),
        }

    def training_step(self, batch, batch_idx):
        out = self._forward_and_loss(batch)
        self.log("train_loss", out["loss"], prog_bar=True)
        self.log("train_cls_loss", out["cls_loss"])
        self.log("train_reg_loss", out["reg_loss"])
        self.log("train_consistency_loss", out["consistency_loss"])
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        fusion_out = self._forward_and_loss(batch)
        # Tabular-only fallback: mask face+speech entirely, same batch, for
        # an always-honest side-by-side comparison (see module docstring).
        batch_size = batch[0].shape[0]
        device = batch[0].device
        tabular_only_mask = {
            "face": torch.zeros(batch_size, dtype=torch.bool, device=device),
            "speech": torch.zeros(batch_size, dtype=torch.bool, device=device),
        }
        tabular_only_out = self._forward_and_loss(batch, modality_mask=tabular_only_mask)

        self.log("val_loss", fusion_out["loss"], prog_bar=True)
        self._val_outputs.append({"fusion": fusion_out, "tabular_only": tabular_only_out})

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        for key in ("fusion", "tabular_only"):
            class_logits = torch.cat([o[key]["class_logits"] for o in self._val_outputs])
            label_idx = torch.cat([o[key]["label_idx"] for o in self._val_outputs])
            score_preds = torch.cat([o[key]["score_preds"] for o in self._val_outputs])
            scores = torch.cat([o[key]["scores"] for o in self._val_outputs])

            probs = torch.softmax(class_logits, dim=-1)
            preds = probs.argmax(dim=-1)
            cls_metrics = compute_classification_metrics(
                label_idx.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy()
            )
            reg_metrics = compute_multitarget_regression_metrics(scores.cpu().numpy(), score_preds.cpu().numpy())

            self.log(f"val_macro_f1_{key}", cls_metrics.macro_f1, prog_bar=True)
            self.log(f"val_weighted_f1_{key}", cls_metrics.weighted_f1)
            self.log(f"val_rmse_{key}", reg_metrics.mean_rmse, prog_bar=True)

        mean_modality_weights = torch.cat([o["fusion"]["modality_weights"] for o in self._val_outputs]).mean(dim=0)
        for i, modality in enumerate(("face", "speech", "tabular")):
            self.log(f"val_modality_weight_{modality}", mean_modality_weights[i])

        self._val_outputs.clear()


def load_config(path: str):
    return OmegaConf.load(path)


def build_fusion_dataloaders(
    cfg,
    tabular_csv: str = DEFAULT_TABULAR_CSV,
    facial_root: str = DEFAULT_FACIAL_ROOT,
    speech_root: str = DEFAULT_SPEECH_ROOT,
    val_split: float = 0.15,
    scaler_path: str | Path = "artifacts/checkpoints/tabular/scaler.joblib",
):
    """Build the anchored train/val fusion loaders.

    Train and val are built as two *separate* FusionPairDataset instances
    over disjoint, stratified tabular row indices:

      - train: `pair_by_label=True`  -- matched-emotion weak pairing, the
        documented anchored-training regime, plus media augmentation.
      - val:   `pair_by_label=False` -- face and voice drawn uniformly at
        random, exactly like a real session where the label is unknown, and
        with no augmentation.

    That asymmetry is deliberate and load-bearing. Matched-emotion pairing
    keys the sampled media on the row's ground-truth label, so a val split
    paired the same way would let the model read the answer off the face and
    report an arbitrarily high macro-F1. See the FusionPairDataset docstring.
    """
    import joblib

    scaler = None
    scaler_file = Path(scaler_path)
    if scaler_file.exists():
        # Same StandardScaler the tabular encoder was trained under -- an
        # unscaled fusion input against a scaled-trained encoder silently
        # degrades every downstream number.
        scaler = joblib.load(scaler_file)
    else:
        logger.warning(
            "No tabular scaler at %s -- fusion will train on unscaled features, which will not "
            "match the tabular encoder checkpoint. Train configs/tabular.yaml first.",
            scaler_file,
        )

    full_tabular = TabularMentalHealthDataset(tabular_csv, scaler=scaler)
    labels = full_tabular.labels

    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=val_split, random_state=cfg.train.seed, stratify=labels
    )

    train_tabular = TabularMentalHealthDataset(tabular_csv, scaler=scaler, indices=train_idx)
    val_tabular = TabularMentalHealthDataset(tabular_csv, scaler=scaler, indices=val_idx)

    train_facial = FacialEmotionDataset(facial_root, train=True)
    eval_facial = FacialEmotionDataset(facial_root, train=False)
    train_speech = SpeechEmotionDataset(speech_root, train=True)
    eval_speech = SpeechEmotionDataset(speech_root, train=False)

    train_pairs = FusionPairDataset(
        train_tabular, train_facial, train_speech, seed=cfg.train.seed, pair_by_label=True
    )
    val_pairs = FusionPairDataset(
        val_tabular, eval_facial, eval_speech, seed=cfg.train.seed + 1, pair_by_label=False
    )

    class_counts = train_tabular.df["Mental_Health_Status"].value_counts().to_dict()
    # Every fusion sample decodes a PNG *and* a WAV from disk (and augments
    # the waveform on the train side), so the loader is I/O-bound and starves
    # the GPU at num_workers=0.
    num_workers = int(cfg.train.get("num_workers", 4))
    train_loader = DataLoader(
        train_pairs,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_pairs,
        batch_size=cfg.train.batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, class_counts, train_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tabular-csv", type=str, default=DEFAULT_TABULAR_CSV)
    parser.add_argument("--facial-root", type=str, default=DEFAULT_FACIAL_ROOT)
    parser.add_argument("--speech-root", type=str, default=DEFAULT_SPEECH_ROOT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    pl.seed_everything(cfg.train.seed)

    train_loader, val_loader, class_counts, train_pairs = build_fusion_dataloaders(
        cfg, args.tabular_csv, args.facial_root, args.speech_root
    )

    module = FusionLightningModule(cfg, class_counts)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    class ResamplePairsCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if cfg.pairing.resample_pairs_every_epoch:
                train_pairs.resample()

    accelerator = args.accelerator or pick_accelerator()
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir), filename="best", monitor="val_loss", mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=accelerator,
        devices=1,
        default_root_dir=str(checkpoint_dir),
        callbacks=[
            ResamplePairsCallback(),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.train.early_stopping_patience),
            checkpoint_cb,
        ],
        log_every_n_steps=20,
    )
    trainer.fit(module, train_loader, val_loader)

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    (checkpoint_dir / "metrics.json").write_text(
        json.dumps(
            {
                "stage": "fusion",
                "epochs_run": trainer.current_epoch,
                "accelerator": accelerator,
                "best_checkpoint": checkpoint_cb.best_model_path,
                "val_pairing": "label_independent (honest)",
                "train_pairing": "matched_emotion_weak_pairing",
                "val_metrics": metrics,
            },
            indent=2,
        )
    )
    print(f"[fusion] final val metrics: {metrics}")


if __name__ == "__main__":
    main()
