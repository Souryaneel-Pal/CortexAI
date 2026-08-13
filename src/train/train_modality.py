"""Single-modality training loop (PROJECT_PLAN.md P1).

Trains one of the three encoders standalone on its native task:
  - facial:  7-way FER emotion classification
  - speech:  8-way RAVDESS emotion classification
  - tabular: 4-class Mental_Health_Status + 3-score regression (the only
             modality with real ground truth)

These per-modality runs double as the ablation study referenced in P6
(src/eval/ablation.py compares them against the full fusion model).

Three things this file is careful about, because each one silently inflates
metrics if you get it wrong:

  1. **Augmentation is train-split only.** Train and val are built as two
     separate Dataset instances (`train=True` / `train=False`) over disjoint
     index sets, rather than one augmented dataset that is then split -- a
     `random_split` of a single augmented dataset would augment validation
     images too.
  2. **SMOTE runs after the split**, on training rows only. Synthesising
     minority rows first would interpolate validation neighbours into the
     training set.
  3. **The speech split is grouped by actor** (configs/speech.yaml
     `split_by: actor_id`). RAVDESS has 24 actors speaking the same two
     sentences; a random clip-level split puts the same speaker on both
     sides and measures speaker memorisation rather than emotion recognition.

Usage:
    python -m src.train.train_modality --config configs/face.yaml
    python -m src.train.train_modality --config configs/speech.yaml
    python -m src.train.train_modality --config configs/tabular.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.data.augment import class_balanced_sample_weights, tabular_smote
from src.data.loaders import (
    DEFAULT_FERPLUS_CSV,
    FacialEmotionDataset,
    SpeechEmotionDataset,
    TabularMentalHealthDataset,
)
from src.data.schemas import STRESS_LABEL_NAME_TO_LEVEL, STRESS_LEVEL_NAMES, StressLevel
from src.eval.metrics import compute_classification_metrics, compute_multitarget_regression_metrics
from src.models.face_cnn import NUM_FACIAL_EMOTIONS, FaceEmotionEncoder
from src.models.speech_net import NUM_SPEECH_EMOTIONS, build_speech_encoder
from src.models.tabular_ft import TabularEncoder
from src.train.losses import ClassBalancedFocalLoss, class_balanced_weights, score_regression_loss

STRESS_LABEL_TO_IDX = STRESS_LABEL_NAME_TO_LEVEL


def pick_accelerator() -> str:
    """Prefer Apple-silicon MPS, then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


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
        self.save_hyperparameters(ignore=["cfg"])
        self.model = FaceEmotionEncoder(
            backbone=cfg.model.backbone, pretrained=cfg.model.pretrained, dropout=cfg.model.dropout
        )
        weights = class_balanced_weights(
            _facial_class_counts_from_disk(
                cfg.data.root,
                cfg.data.get("label_source", "folders"),
                cfg.data.get("ferplus_csv", DEFAULT_FERPLUS_CSV),
            ),
            beta=cfg.imbalance.class_balanced_beta,
        )
        self.criterion = ClassBalancedFocalLoss(class_weights=weights, gamma=cfg.imbalance.focal_gamma)
        self._val_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

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
        self._val_outputs.append((logits.detach().float().cpu(), labels.detach().cpu()))

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
            labels.numpy(), preds.numpy(), probs.numpy(), num_classes=NUM_FACIAL_EMOTIONS
        )
        self.log("val_accuracy", m.accuracy, prog_bar=True)
        self.log("val_macro_f1", m.macro_f1, prog_bar=True)
        self.log("val_weighted_f1", m.weighted_f1)
        self._val_outputs.clear()


class SpeechLightningModule(_BaseModalityModule):
    def __init__(self, cfg):
        super().__init__(cfg.train.lr, cfg.train.weight_decay, cfg.train.epochs)
        self.save_hyperparameters(ignore=["cfg"])
        fallback = OmegaConf.to_container(cfg.model.fallback) if cfg.model.backbone == "cnn_bilstm" else {}
        if cfg.model.backbone == "cnn_bilstm":
            fallback = {
                **{k: tuple(v) if isinstance(v, list) else v for k, v in fallback.items()},
                "sample_rate": cfg.data.sample_rate,
                "spec_augment": cfg.augment.spec_augment,
                "freq_mask_param": cfg.augment.freq_mask_param,
                "time_mask_param": cfg.augment.time_mask_param,
            }
        self.model = build_speech_encoder(
            backbone=cfg.model.backbone,
            pretrained_checkpoint=cfg.model.get("pretrained_checkpoint", "facebook/wav2vec2-base"),
            freeze_feature_extractor=cfg.model.get("freeze_feature_extractor", True),
            **fallback,
        )
        weights = class_balanced_weights(
            _speech_class_counts_from_disk(cfg.data.root), beta=cfg.imbalance.class_balanced_beta
        )
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
        self._val_outputs.append((logits.detach().float().cpu(), labels.detach().cpu()))

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        logits = torch.cat([o[0] for o in self._val_outputs])
        labels = torch.cat([o[1] for o in self._val_outputs])
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        # 8-way RAVDESS emotion space -- also the P1/P6 speech ablation entry.
        m = compute_classification_metrics(
            labels.numpy(), preds.numpy(), probs.numpy(), num_classes=NUM_SPEECH_EMOTIONS
        )
        self.log("val_accuracy", m.accuracy, prog_bar=True)
        self.log("val_macro_f1", m.macro_f1, prog_bar=True)
        self.log("val_weighted_f1", m.weighted_f1)
        self._val_outputs.clear()


class TabularLightningModule(_BaseModalityModule):
    """Trains on the real 4-class + 3-score ground truth -- the P1 baseline
    that everything else (weak-pairing anchor, fusion) is validated against.
    """

    def __init__(self, cfg, class_counts: dict[str, int]):
        super().__init__(cfg.train.lr, cfg.train.weight_decay, cfg.train.epochs)
        self.save_hyperparameters(ignore=["cfg"])
        self.model = TabularEncoder(backbone=cfg.model.backbone, embed_dim=cfg.model.embed_dim)
        weights = class_balanced_weights(class_counts, beta=0.999)
        self.classification_criterion = ClassBalancedFocalLoss(class_weights=weights, gamma=2.0)
        self._val_outputs: list[tuple[torch.Tensor, ...]] = []

    def _step(self, batch):
        features, labels, scores = batch
        label_idx = (
            torch.tensor([STRESS_LABEL_TO_IDX[l] for l in labels], device=features.device)
            if isinstance(labels, (list, tuple))
            else labels
        )
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
        self._val_outputs.append(
            (
                class_logits.detach().float().cpu(),
                label_idx.detach().cpu(),
                score_preds.detach().float().cpu(),
                scores.detach().float().cpu(),
            )
        )

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        class_logits = torch.cat([o[0] for o in self._val_outputs])
        label_idx = torch.cat([o[1] for o in self._val_outputs])
        score_preds = torch.cat([o[2] for o in self._val_outputs])
        scores = torch.cat([o[3] for o in self._val_outputs])

        probs = torch.softmax(class_logits, dim=-1)
        preds = probs.argmax(dim=-1)
        cls_metrics = compute_classification_metrics(label_idx.numpy(), preds.numpy(), probs.numpy())
        reg_metrics = compute_multitarget_regression_metrics(scores.numpy(), score_preds.numpy())

        self.log("val_accuracy", cls_metrics.accuracy, prog_bar=True)
        self.log("val_macro_f1", cls_metrics.macro_f1, prog_bar=True)
        self.log("val_rmse_mean", reg_metrics.mean_rmse, prog_bar=True)
        self._val_outputs.clear()


def load_config(path: str):
    return OmegaConf.load(path)


# --------------------------------------------------------------------------
# Class counts read from the real data on disk (rather than the documented
# constants), so a partial or re-sampled copy of the dataset still gets
# correctly-sized class-balanced loss weights.
# --------------------------------------------------------------------------
def _facial_class_counts_from_disk(root, label_source: str = "folders", ferplus_csv=DEFAULT_FERPLUS_CSV) -> dict[str, int]:
    from src.data.schemas import FACIAL_EMOTIONS

    dataset = FacialEmotionDataset(root, train=False, label_source=label_source, ferplus_csv=ferplus_csv)
    labels = dataset.labels
    return {name: int((labels == idx).sum()) for idx, name in FACIAL_EMOTIONS.items() if (labels == idx).any()}


def _speech_class_counts_from_disk(root) -> dict[str, int]:
    from src.data.schemas import SPEECH_EMOTIONS

    dataset = SpeechEmotionDataset(root, train=False)
    names = list(SPEECH_EMOTIONS.values())
    labels = dataset.labels
    return {name: int((labels == idx).sum()) for idx, name in enumerate(names) if (labels == idx).any()}


# --------------------------------------------------------------------------
# Split builders -- see the "three things this file is careful about" note.
# --------------------------------------------------------------------------
def _stratified_split_indices(labels: np.ndarray, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=seed, stratify=labels
    )
    return train_idx, val_idx


def build_dataloaders_facial(cfg):
    """Two dataset instances over disjoint stratified index sets: the training
    one augments (minority classes more aggressively), the validation one
    does not.
    """
    label_source = cfg.data.get("label_source", "folders")
    ferplus_csv = cfg.data.get("ferplus_csv", DEFAULT_FERPLUS_CSV)
    train_source = FacialEmotionDataset(
        cfg.data.root, train=True, label_source=label_source, ferplus_csv=ferplus_csv
    )
    eval_source = FacialEmotionDataset(
        cfg.data.root, train=False, label_source=label_source, ferplus_csv=ferplus_csv
    )
    labels = eval_source.labels

    train_idx, val_idx = _stratified_split_indices(labels, cfg.data.val_split, cfg.train.seed)
    train_ds = Subset(train_source, train_idx)
    val_ds = Subset(eval_source, val_idx)

    # Balanced sampling stacks on top of class-balanced focal loss only when
    # the config asks for it (`imbalance.use_balanced_sampler`). For FER it
    # earns its place -- Disgust is a 16.5x minority and loss weighting alone
    # leaves whole epochs where the model barely sees it. See the note in
    # `build_dataloaders_speech` for why it is off by default.
    sampler = None
    if cfg.imbalance.get("use_balanced_sampler", False):
        weights = class_balanced_sample_weights(labels[train_idx], beta=cfg.imbalance.class_balanced_beta)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(), num_samples=len(train_idx), replacement=True
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
    )
    return train_loader, val_loader


def build_dataloaders_speech(cfg):
    """Actor-grouped split (no speaker appears on both sides) with waveform
    augmentation + SpecAugment on the training side only.
    """
    train_source = SpeechEmotionDataset(
        cfg.data.root, sample_rate=cfg.data.sample_rate, max_duration_sec=cfg.data.max_duration_sec, train=True
    )
    eval_source = SpeechEmotionDataset(
        cfg.data.root, sample_rate=cfg.data.sample_rate, max_duration_sec=cfg.data.max_duration_sec, train=False
    )

    actors = eval_source.actor_ids
    unique_actors = np.unique(actors)
    rng = np.random.default_rng(cfg.train.seed)
    shuffled = rng.permutation(unique_actors)
    n_val_actors = max(1, int(round(len(unique_actors) * cfg.data.val_split)))
    # Hold out an even number of actors so the val split stays gender-balanced
    # (odd actor ids are male, even female -- docs/Dataset_Description.docx),
    # which is what makes the P6 fairness audit meaningful.
    if n_val_actors % 2 == 1 and len(unique_actors) > n_val_actors:
        n_val_actors += 1
    val_actors = set(shuffled[:n_val_actors].tolist())

    val_idx = np.where(np.isin(actors, list(val_actors)))[0]
    train_idx = np.where(~np.isin(actors, list(val_actors)))[0]

    train_ds = Subset(train_source, train_idx)
    val_ds = Subset(eval_source, val_idx)

    # Off by default for speech, and this is deliberate. RAVDESS is only
    # 2:1 imbalanced (neutral 96 vs 192 for the other seven emotions), which
    # class-balanced focal loss already handles -- configs/speech.yaml asks
    # for exactly that one strategy. Stacking a balanced sampler on top
    # double-corrects the same imbalance and collapses the model onto the
    # up-weighted minority: an earlier run sat at 6.7% accuracy, i.e. it
    # predicted "neutral" for every clip in the held-out split.
    sampler = None
    if cfg.imbalance.get("use_balanced_sampler", False):
        weights = class_balanced_sample_weights(
            eval_source.labels[train_idx], beta=cfg.imbalance.class_balanced_beta
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(), num_samples=len(train_idx), replacement=True
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.data.num_workers,
        # Workers respawn per epoch otherwise, and on macOS's spawn start
        # method that re-import cost dwarfs the ~6 s the epoch itself takes.
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
    )
    return train_loader, val_loader, sorted(val_actors)


def build_dataloaders_tabular(cfg):
    """Stratified split, StandardScaler fit on train only, then SMOTE on the
    training rows only.

    The scaler is returned so `main()` can persist it next to the checkpoint --
    inference (src/api/inference.py) must apply the identical transform, and
    an un-persisted scaler is the classic way a served model silently sees
    features on a different scale than it was trained on.
    """
    from sklearn.preprocessing import StandardScaler

    dataset = TabularMentalHealthDataset(cfg.data.csv_path)
    labels = dataset.labels
    train_idx, val_idx = _stratified_split_indices(labels, cfg.data.val_split, cfg.train.seed)

    X_train_raw, y_train = dataset.features[train_idx], labels[train_idx]
    X_val_raw, y_val = dataset.features[val_idx], labels[val_idx]
    scores_train, scores_val = dataset.scores[train_idx], dataset.scores[val_idx]

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)

    if cfg.imbalance.strategy == "smote":
        # SMOTE synthesises minority *feature* rows; the 3 regression targets
        # have to travel with them, so class labels and scores are resampled
        # jointly by appending the scores to the feature matrix, running
        # SMOTE once, then splitting them back apart. Interpolating the
        # scores alongside the features keeps each synthetic row internally
        # consistent (a synthetic Severe_Stress row gets Severe-range scores).
        joint = np.hstack([X_train, scores_train])
        joint_resampled, y_train = tabular_smote(
            joint, y_train, random_state=cfg.imbalance.smote_random_state
        )
        X_train = joint_resampled[:, : X_train.shape[1]].astype(np.float32)
        scores_train = joint_resampled[:, X_train.shape[1] :].astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(np.asarray(y_train)).long(),
        torch.from_numpy(scores_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(np.asarray(y_val)).long(),
        torch.from_numpy(scores_val).float(),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size)

    class_counts = {
        STRESS_LEVEL_NAMES[StressLevel(int(level))]: int((y_train == level).sum())
        for level in np.unique(y_train)
    }
    return train_loader, val_loader, class_counts, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override cfg.train.epochs")
    parser.add_argument("--accelerator", type=str, default=None, help="mps | gpu | cpu (default: auto)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    pl.seed_everything(cfg.train.seed)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    split_meta: dict = {}

    if cfg.modality == "tabular":
        train_loader, val_loader, class_counts, scaler = build_dataloaders_tabular(cfg)
        module = TabularLightningModule(cfg, class_counts)
        # Persist the fitted scaler beside the checkpoint -- inference reloads it.
        import joblib

        joblib.dump(scaler, checkpoint_dir / "scaler.joblib")
        split_meta["class_counts_after_smote"] = class_counts
    elif cfg.modality == "facial":
        train_loader, val_loader = build_dataloaders_facial(cfg)
        module = FaceLightningModule(cfg)
    elif cfg.modality == "speech":
        train_loader, val_loader, val_actors = build_dataloaders_speech(cfg)
        module = SpeechLightningModule(cfg)
        split_meta["held_out_actors"] = val_actors
    else:
        raise ValueError(f"Unknown modality: {cfg.modality!r}")

    accelerator = args.accelerator or pick_accelerator()
    # Select the checkpoint on the headline metric from docs/Metrics_Used.docx,
    # not on val_loss. For the two imbalanced emotion tasks that is macro-F1
    # (equal weight to every class); class-balanced focal loss and macro-F1
    # peak at different epochs, and picking on loss quietly ships a checkpoint
    # several points of macro-F1 worse than the one training actually reached.
    # Tabular is multi-task (4-class + 3-score), so its combined val_loss
    # remains the only criterion that accounts for both heads.
    monitor, mode = ("val_loss", "min") if cfg.modality == "tabular" else ("val_macro_f1", "max")
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir), filename="best", monitor=monitor, mode=mode, save_last=False
    )
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=accelerator,
        devices=1,
        default_root_dir=str(checkpoint_dir),
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.train.early_stopping_patience),
            checkpoint_cb,
        ],
        log_every_n_steps=20,
    )
    trainer.fit(module, train_loader, val_loader)

    # Record the final validation metrics next to the checkpoint so the README
    # and the ablation table quote measured numbers, not remembered ones.
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    (checkpoint_dir / "metrics.json").write_text(
        json.dumps(
            {
                "modality": cfg.modality,
                "backbone": cfg.model.backbone,
                "epochs_run": trainer.current_epoch,
                "accelerator": accelerator,
                "best_checkpoint": checkpoint_cb.best_model_path,
                "val_metrics": metrics,
                **split_meta,
            },
            indent=2,
        )
    )
    print(f"[{cfg.modality}] final val metrics: {metrics}")


if __name__ == "__main__":
    main()
