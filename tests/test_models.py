import torch

from src.models.face_cnn import EMBED_DIM as FACE_EMBED_DIM
from src.models.face_cnn import NUM_FACIAL_EMOTIONS, FaceEmotionEncoder
from src.models.speech_net import EMBED_DIM as SPEECH_EMBED_DIM
from src.models.speech_net import NUM_SPEECH_EMOTIONS, CNNBiLSTMEncoder
from src.models.tabular_ft import EMBED_DIM as TAB_EMBED_DIM
from src.models.tabular_ft import NUM_CLASSES, NUM_FEATURES, NUM_SCORE_TARGETS, TabularEncoder


def test_face_encoder_simple_cnn_backbone_shapes():
    model = FaceEmotionEncoder(backbone="simple_cnn")
    x = torch.randn(4, 1, 48, 48)
    embedding, logits = model(x)
    assert embedding.shape == (4, FACE_EMBED_DIM)
    assert logits.shape == (4, NUM_FACIAL_EMOTIONS)


def test_face_encoder_efficientnet_backbone_shapes_no_pretrained():
    # pretrained=False avoids a network download of timm weights in this sandbox.
    model = FaceEmotionEncoder(backbone="efficientnet_b0", pretrained=False)
    x = torch.randn(2, 1, 48, 48)
    embedding, logits = model(x)
    assert embedding.shape == (2, FACE_EMBED_DIM)
    assert logits.shape == (2, NUM_FACIAL_EMOTIONS)
    assert model.gradcam_target_layer is not None


def test_face_encoder_backward_pass_produces_finite_gradients():
    model = FaceEmotionEncoder(backbone="simple_cnn")
    x = torch.randn(4, 1, 48, 48)
    _, logits = model(x)
    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_speech_cnn_bilstm_shapes():
    model = CNNBiLSTMEncoder(sample_rate=16000, n_mels=64)
    waveform = torch.randn(3, 16000)  # 1 second of audio
    embedding, logits = model(waveform)
    assert embedding.shape == (3, SPEECH_EMBED_DIM)
    assert logits.shape == (3, NUM_SPEECH_EMOTIONS)


def test_tabular_ft_transformer_shapes_and_attention():
    model = TabularEncoder(backbone="ft_transformer")
    x = torch.randn(5, NUM_FEATURES)
    embedding, class_logits, score_preds = model(x)
    assert embedding.shape == (5, TAB_EMBED_DIM)
    assert class_logits.shape == (5, NUM_CLASSES)
    assert score_preds.shape == (5, NUM_SCORE_TARGETS)

    attn = model.encoder.feature_attention_weights(x)
    assert attn.shape == (5, NUM_FEATURES)
    # Non-negative attention mass over the 18 feature tokens; the remainder
    # (up to 1) is the dropped CLS-to-self weight, so this is < 1, not == 1.
    assert (attn >= 0).all()
    assert (attn.sum(dim=1) <= 1.0 + 1e-4).all()
    assert (attn.sum(dim=1) > 0).all()


def test_tabular_residual_mlp_fallback_shapes():
    model = TabularEncoder(backbone="residual_mlp")
    x = torch.randn(6, NUM_FEATURES)
    embedding, class_logits, score_preds = model(x)
    assert embedding.shape == (6, TAB_EMBED_DIM)
    assert class_logits.shape == (6, NUM_CLASSES)
    assert score_preds.shape == (6, NUM_SCORE_TARGETS)


def test_tabular_backward_pass_finite_gradients():
    model = TabularEncoder(backbone="ft_transformer")
    x = torch.randn(8, NUM_FEATURES)
    _, class_logits, score_preds = model(x)
    loss = class_logits.sum() + score_preds.sum()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
