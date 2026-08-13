import torch

from src.explain.ig_audio import integrated_gradients_audio, pool_attribution_to_frames
from src.models.speech_net import CNNBiLSTMEncoder


def test_integrated_gradients_output_shape():
    model = CNNBiLSTMEncoder(sample_rate=16000, n_mels=64)
    waveform = torch.randn(2, 16000)
    attributions = integrated_gradients_audio(model, waveform, n_steps=8)
    assert attributions.shape == waveform.shape
    assert torch.isfinite(attributions).all()


def test_integrated_gradients_differs_by_target_class():
    torch.manual_seed(0)
    model = CNNBiLSTMEncoder(sample_rate=16000, n_mels=64)
    waveform = torch.randn(1, 16000)
    attr_0 = integrated_gradients_audio(model, waveform, target_class=0, n_steps=8)
    attr_1 = integrated_gradients_audio(model, waveform, target_class=1, n_steps=8)
    assert not torch.allclose(attr_0, attr_1)


def test_integrated_gradients_does_not_change_training_mode():
    model = CNNBiLSTMEncoder(sample_rate=16000, n_mels=64)
    model.train()
    waveform = torch.randn(2, 16000)
    integrated_gradients_audio(model, waveform, n_steps=8)
    assert model.training


def test_pool_attribution_to_frames_shape_and_aggregation():
    attribution = torch.tensor([[1.0, -1.0, 2.0, -2.0, 3.0, -3.0]])
    pooled = pool_attribution_to_frames(attribution, frame_size=2)
    assert pooled.shape == (1, 3)
    assert torch.allclose(pooled, torch.tensor([[1.0, 2.0, 3.0]]))
