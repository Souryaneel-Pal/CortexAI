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
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

from src.data.loaders import FacialEmotionDataset, FusionPairDataset, SpeechEmotionDataset, TabularMentalHealthDataset
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


def build_fusion_dataloaders(cfg, tabular_csv: str, facial_root: str, speech_root: str, val_split: float = 0.15):
    tabular_ds = TabularMentalHealthDataset(tabular_csv)
    facial_ds = FacialEmotionDataset(facial_root)
    speech_ds = SpeechEmotionDataset(speech_root)

    pair_ds = FusionPairDataset(tabular_ds, facial_ds, speech_ds, seed=cfg.train.seed)
    n = len(pair_ds)
    n_val = int(n * val_split)
    train_ds, val_ds = random_split(
        pair_ds, [n - n_val, n_val], generator=torch.Generator().manual_seed(cfg.train.seed)
    )

    class_counts = tabular_ds.df["Mental_Health_Status"].value_counts().to_dict()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    return train_loader, val_loader, class_counts, pair_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tabular-csv", type=str, default="data/raw/numerical.csv")
    parser.add_argument("--facial-root", type=str, default="data/raw/facial")
    parser.add_argument("--speech-root", type=str, default="data/raw/speech")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg.train.seed)

    train_loader, val_loader, class_counts, pair_ds = build_fusion_dataloaders(
        cfg, args.tabular_csv, args.facial_root, args.speech_root
    )

    module = FusionLightningModule(cfg, class_counts)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    class ResamplePairsCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if cfg.pairing.resample_pairs_every_epoch:
                pair_ds.resample()

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        default_root_dir=str(checkpoint_dir),
        callbacks=[
            ResamplePairsCallback(),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.train.early_stopping_patience),
            pl.callbacks.ModelCheckpoint(dirpath=str(checkpoint_dir), filename="best", monitor="val_loss"),
        ],
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
