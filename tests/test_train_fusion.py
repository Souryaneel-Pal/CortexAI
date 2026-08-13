import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch

from src.data.loaders import FacialEmotionDataset, FusionPairDataset, SpeechEmotionDataset, TabularMentalHealthDataset
from src.train.train_fusion import FusionLightningModule, load_config


def _build_cfg_for_fixtures():
    cfg = load_config("configs/fusion.yaml")
    cfg.encoders.face_checkpoint = "does/not/exist.ckpt"
    cfg.encoders.speech_checkpoint = "does/not/exist.ckpt"
    cfg.encoders.tabular_checkpoint = "does/not/exist.ckpt"
    cfg.encoders.freeze_pretrained = False  # exercise the trainable path in this fast test
    cfg.train.epochs = 1
    return cfg


def test_fusion_module_end_to_end_fast_dev_run(
    synthetic_tabular_csv, synthetic_facial_dir, synthetic_speech_dir
):
    tabular_ds = TabularMentalHealthDataset(synthetic_tabular_csv)
    facial_ds = FacialEmotionDataset(synthetic_facial_dir)
    speech_ds = SpeechEmotionDataset(synthetic_speech_dir, max_duration_sec=1.0)
    pair_ds = FusionPairDataset(tabular_ds, facial_ds, speech_ds, seed=0)

    train_ds, val_ds = random_split(pair_ds, [len(pair_ds) - 8, 8])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    cfg = _build_cfg_for_fixtures()
    class_counts = tabular_ds.df["Mental_Health_Status"].value_counts().to_dict()
    module = FusionLightningModule(cfg, class_counts)

    trainer = pl.Trainer(max_epochs=1, fast_dev_run=3, enable_progress_bar=False, logger=False)
    trainer.fit(module, train_loader, val_loader)


def test_fusion_module_frozen_encoders_stay_in_eval_after_train_call(
    synthetic_tabular_csv, synthetic_facial_dir, synthetic_speech_dir
):
    tabular_ds = TabularMentalHealthDataset(synthetic_tabular_csv)
    class_counts = tabular_ds.df["Mental_Health_Status"].value_counts().to_dict()

    cfg = _build_cfg_for_fixtures()
    cfg.encoders.freeze_pretrained = True
    module = FusionLightningModule(cfg, class_counts)

    module.train()  # simulate Lightning's per-epoch train() call
    assert not module.face_encoder.training
    assert not module.speech_encoder.training
    assert not module.tabular_encoder.training
    assert module.fusion.training  # the trainable parts should still be in train mode

    for p in module.face_encoder.parameters():
        assert not p.requires_grad
    for p in module.fusion.parameters():
        assert p.requires_grad


def test_fusion_module_tabular_only_fallback_uses_missing_tokens(
    synthetic_tabular_csv, synthetic_facial_dir, synthetic_speech_dir
):
    tabular_ds = TabularMentalHealthDataset(synthetic_tabular_csv)
    facial_ds = FacialEmotionDataset(synthetic_facial_dir)
    speech_ds = SpeechEmotionDataset(synthetic_speech_dir, max_duration_sec=1.0)
    pair_ds = FusionPairDataset(tabular_ds, facial_ds, speech_ds, seed=0)
    loader = DataLoader(pair_ds, batch_size=6)
    batch = next(iter(loader))

    cfg = _build_cfg_for_fixtures()
    class_counts = tabular_ds.df["Mental_Health_Status"].value_counts().to_dict()
    module = FusionLightningModule(cfg, class_counts)
    module.eval()

    with torch.no_grad():
        fusion_out = module._forward_and_loss(batch)
        batch_size = batch[0].shape[0]
        mask = {
            "face": torch.zeros(batch_size, dtype=torch.bool),
            "speech": torch.zeros(batch_size, dtype=torch.bool),
        }
        tabular_only_out = module._forward_and_loss(batch, modality_mask=mask)

    # Masking face+speech should change the fused prediction relative to full fusion.
    assert not torch.allclose(fusion_out["class_probs"], tabular_only_out["class_probs"])
    # Tabular-only modality weight should not be dominated by face/speech being fed real data,
    # since both are replaced by the same learned missing-token regardless of input.
    assert torch.isfinite(tabular_only_out["class_probs"]).all()
