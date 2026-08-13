"""Single-modality training loop (docs/PROJECT_PLAN.md P1).

Trains one of the three encoders standalone on its native task:
  - facial:  7-way FER emotion classification
  - speech:  8-way RAVDESS emotion classification
  - tabular: 4-class Mental_Health_Status + 3-score regression (the only
             modality with real ground truth)

These per-modality runs double as the ablation study referenced in P6
(src/eval/ablation.py compares them against the full fusion model).

Usage:
    python -m src.train.train_modality --config configs/face.yaml
    python -m src.train.train_modality --config configs/speech.yaml
    python -m src.train.train_modality --config configs/tabular.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

from src.data.augment import class_balanced_sample_weights
from src.data.loaders import FacialEmotionDataset, SpeechEmotionDataset, TabularMentalHealthDataset
from src.data.schemas import FACIAL_CLASS_COUNTS, RAVDESS_CLASS_COUNTS, StressLevel
from src.eval.metrics import compute_classification_metrics, compute_multitarget_regression_metrics
from src.models.face_cnn import NUM_FACIAL_EMOTIONS, FaceEmotionEncoder
from src.models.speech_net import NUM_SPEECH_EMOTIONS, build_speech_encoder
from src.models.tabular_ft import TabularEncoder
from src.train.losses import ClassBalancedFocalLoss, class_balanced_weights, score_regression_loss

STRESS_LABEL_TO_IDX = {
    "Healthy": StressLevel.HEALTHY,
    "Mild_Stress": StressLevel.MILD_STRESS,
    "Moderate_Stress": StressLevel.MODERATE_STRESS,
    "Severe_Stress": StressLevel.SEVERE_STRESS,
}


class _BaseModalityModule(pl.LightningModule):
    """Shared optimizer/scheduler wiring for all three modality trainers."""

    def __init__(self, lr: float, weight_decay: float, epochs: int):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class FaceLightningModule(_BaseModalityModule):
    def __init__(self, cfg):
        super().__init__(cfg.train.lr, cfg.train.weight_decay, cfg.train.epochs)
        self.model = FaceEmotionEncoder(
            backbone=cfg.model.backbone, pretrained=cfg.model.pretrained, dropout=cfg.model.dropout
        )
        weights = class_balanced_weights(FACIAL_CLASS_COUNTS, beta=cfg.imbalance.class_balanced_beta)
        self.criterion = ClassBalancedFocalLoss(class_weights=weights, gamma=cfg.imbalance.focal_gamma)
        self._val_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        _, logits = self.model(images)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        _, logits = self.model(images)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        self._val_outputs.append((logits.detach(), labels.detach()))

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        logits = torch.cat([o[0] for o in self._val_outputs])
        labels = torch.cat([o[1] for o in self._val_outputs])
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        # 7-way FER emotion space, not the 4-tier stress axis -- this baseline
        # run is also the per-modality ablation entry (PROJECT_PLAN.md P1/P6).
        m = compute_classification_metrics(
            labels.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy(), num_classes=NUM_FACIAL_EMOTIONS
        )
        self.log("val_macro_f1", m.macro_f1, prog_bar=True)
        self.log("val_weighted_f1", m.weighted_f1)
        self._val_outputs.clear()


class SpeechLightningModule(_BaseModalityModule):
    def __init__(self, cfg):
        super().__init__(cfg.train.lr, cfg.train.weight_decay, cfg.train.epochs)
        self.model = build_speech_encoder(
            backbone=cfg.model.backbone,
            pretrained_checkpoint=cfg.model.get("pretrained_checkpoint", "facebook/wav2vec2-base"),
            freeze_feature_extractor=cfg.model.get("freeze_feature_extractor", True),
            **(cfg.model.get("fallback", {}) if cfg.model.backbone == "cnn_bilstm" else {}),
        )
        weights = class_balanced_weights(RAVDESS_CLASS_COUNTS, beta=cfg.imbalance.class_balanced_beta)
        self.criterion = ClassBalancedFocalLoss(class_weights=weights, gamma=cfg.imbalance.focal_gamma)
        self._val_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        waveforms, labels, _meta = batch
        _, logits = self.model(waveforms)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, labels, _meta = batch
        _, logits = self.model(waveforms)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        self._val_outputs.append((logits.detach(), labels.detach()))

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        logits = torch.cat([o[0] for o in self._val_outputs])
        labels = torch.cat([o[1] for o in self._val_outputs])
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        # 8-way RAVDESS emotion space -- also the P1/P6 speech ablation entry.
        m = compute_classification_metrics(
            labels.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy(), num_classes=NUM_SPEECH_EMOTIONS
        )
        self.log("val_macro_f1", m.macro_f1, prog_bar=True)
        self.log("val_weighted_f1", m.weighted_f1)
        self._val_outputs.clear()


class TabularLightningModule(_BaseModalityModule):
    """Trains on the real 4-class + 3-score ground truth -- the P1 baseline
    that everything else (weak-pairing anchor, fusion) is validated against.
    """

    def __init__(self, cfg, class_counts: dict[str, int]):
        super().__init__(cfg.train.lr, cfg.train.weight_decay, cfg.train.epochs)
        self.model = TabularEncoder(backbone=cfg.model.backbone, embed_dim=cfg.model.embed_dim)
        weights = class_balanced_weights(class_counts, beta=0.999)
        self.classification_criterion = ClassBalancedFocalLoss(class_weights=weights, gamma=2.0)
        self._val_outputs = []

    def _step(self, batch):
        features, labels, scores = batch
        label_idx = torch.tensor(
            [STRESS_LABEL_TO_IDX[l] for l in labels], device=features.device
        ) if isinstance(labels, (list, tuple)) else labels
        _, class_logits, score_preds = self.model(features)
        cls_loss = self.classification_criterion(class_logits, label_idx)
        reg_loss = score_regression_loss(score_preds, scores)
        return cls_loss + reg_loss, class_logits, score_preds, label_idx, scores

    def training_step(self, batch, batch_idx):
        loss, *_ = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, class_logits, score_preds, label_idx, scores = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self._val_outputs.append((class_logits.detach(), label_idx.detach(), score_preds.detach(), scores.detach()))

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        class_logits = torch.cat([o[0] for o in self._val_outputs])
        label_idx = torch.cat([o[1] for o in self._val_outputs])
        score_preds = torch.cat([o[2] for o in self._val_outputs])
        scores = torch.cat([o[3] for o in self._val_outputs])

        probs = torch.softmax(class_logits, dim=-1)
        preds = probs.argmax(dim=-1)
        cls_metrics = compute_classification_metrics(
            label_idx.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy()
        )
        reg_metrics = compute_multitarget_regression_metrics(scores.cpu().numpy(), score_preds.cpu().numpy())

        self.log("val_macro_f1", cls_metrics.macro_f1, prog_bar=True)
        self.log("val_rmse_mean", reg_metrics.mean_rmse, prog_bar=True)
        self._val_outputs.clear()


def load_config(path: str):
    return OmegaConf.load(path)


def build_dataloaders_tabular(cfg):
    dataset = TabularMentalHealthDataset(cfg.data.csv_path)
    n = len(dataset)
    n_val = int(n * cfg.data.val_split)
    n_test = int(n * cfg.data.test_split)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(cfg.train.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size)

    labels = dataset.df["Mental_Health_Status"] if hasattr(dataset, "df") else None
    class_counts = labels.value_counts().to_dict() if labels is not None else {}
    return train_loader, val_loader, test_loader, class_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg.train.seed)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if cfg.modality == "tabular":
        train_loader, val_loader, _, class_counts = build_dataloaders_tabular(cfg)
        module = TabularLightningModule(cfg, class_counts)
    elif cfg.modality == "facial":
        dataset = FacialEmotionDataset(cfg.data.root)
        n_val = int(len(dataset) * cfg.data.val_split)
        train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
        train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size)
        module = FaceLightningModule(cfg)
    elif cfg.modality == "speech":
        dataset = SpeechEmotionDataset(
            cfg.data.root,
            sample_rate=cfg.data.sample_rate,
            max_duration_sec=cfg.data.max_duration_sec,
        )
        n_val = int(len(dataset) * cfg.data.val_split)
        train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
        train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size)
        module = SpeechLightningModule(cfg)
    else:
        raise ValueError(f"Unknown modality: {cfg.modality!r}")

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        default_root_dir=str(checkpoint_dir),
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.train.early_stopping_patience),
            pl.callbacks.ModelCheckpoint(dirpath=str(checkpoint_dir), filename="best", monitor="val_loss"),
        ],
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
