import torch

from src.models.fusion import EMBED_DIM, MODALITIES, GatedCrossModalFusion, ModalityDropout
from src.models.heads import FusionHeads, NUM_CLASSES, NUM_SCORE_TARGETS


def test_modality_dropout_replaces_masked_modality():
    dropout = ModalityDropout(embed_dim=8, p=0.0)  # p=0 isolates the explicit mask path
    embeddings = {"face": torch.ones(3, 8), "speech": torch.ones(3, 8), "tabular": torch.ones(3, 8)}
    mask = {"face": torch.tensor([1, 0, 1])}
    out = dropout(embeddings, modality_mask=mask)
    assert torch.allclose(out["face"][1], dropout.missing_tokens["face"])
    assert torch.allclose(out["face"][0], torch.ones(8))


def test_gated_fusion_output_shapes():
    model = GatedCrossModalFusion(embed_dim=EMBED_DIM, hidden_dim=64, n_layers=1, n_heads=4)
    face = torch.randn(5, EMBED_DIM)
    speech = torch.randn(5, EMBED_DIM)
    tabular = torch.randn(5, EMBED_DIM)
    fused, weights = model(face, speech, tabular)
    assert fused.shape == (5, EMBED_DIM)
    assert weights.shape == (5, len(MODALITIES))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert (weights >= 0).all()


def test_gated_fusion_backward_pass_finite_gradients():
    model = GatedCrossModalFusion(embed_dim=EMBED_DIM, hidden_dim=64, n_layers=1, n_heads=4)
    face = torch.randn(4, EMBED_DIM, requires_grad=True)
    speech = torch.randn(4, EMBED_DIM, requires_grad=True)
    tabular = torch.randn(4, EMBED_DIM, requires_grad=True)
    fused, weights = model(face, speech, tabular)
    (fused.sum() + weights.sum()).backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_gated_fusion_missing_modality_degrades_gracefully():
    model = GatedCrossModalFusion(embed_dim=EMBED_DIM, hidden_dim=64, n_layers=1, n_heads=4)
    model.eval()
    face = torch.randn(2, EMBED_DIM)
    speech = torch.randn(2, EMBED_DIM)
    tabular = torch.randn(2, EMBED_DIM)
    mask = {"face": torch.tensor([0, 0]), "speech": torch.tensor([1, 1])}
    fused, weights = model(face, speech, tabular, modality_mask=mask)
    assert fused.shape == (2, EMBED_DIM)
    assert torch.isfinite(fused).all()


def test_fusion_heads_shapes_and_probs_sum_to_one():
    heads = FusionHeads(embed_dim=EMBED_DIM)
    x = torch.randn(6, EMBED_DIM)
    class_logits, class_probs, score_preds = heads(x)
    assert class_logits.shape == (6, NUM_CLASSES)
    assert class_probs.shape == (6, NUM_CLASSES)
    assert score_preds.shape == (6, NUM_SCORE_TARGETS)
    assert torch.allclose(class_probs.sum(dim=-1), torch.ones(6), atol=1e-5)


def test_regression_head_outputs_within_documented_ranges():
    heads = FusionHeads(embed_dim=EMBED_DIM)
    x = torch.randn(50, EMBED_DIM) * 5  # wide range of activations
    _, _, score_preds = heads(x)
    depression, anxiety, stress = score_preds[:, 0], score_preds[:, 1], score_preds[:, 2]
    assert (depression >= 0).all() and (depression <= 34).all()
    assert (anxiety >= 0).all() and (anxiety <= 24).all()
    assert (stress >= 0).all() and (stress <= 39).all()


def test_mc_dropout_uncertainty_returns_expected_keys_and_shapes():
    heads = FusionHeads(embed_dim=EMBED_DIM)
    heads.eval()
    x = torch.randn(4, EMBED_DIM)
    result = heads.classification_head.predict_with_uncertainty(x, n_passes=10)
    assert result["mean_probs"].shape == (4, NUM_CLASSES)
    assert result["variance"].shape == (4, NUM_CLASSES)
    assert result["confidence"].shape == (4,)
    assert torch.allclose(result["mean_probs"].sum(dim=-1), torch.ones(4), atol=1e-3)


def test_mc_dropout_uncertainty_higher_variance_with_more_dropout():
    torch.manual_seed(0)
    low_dropout_heads = FusionHeads(embed_dim=EMBED_DIM, classification_dropout=0.05)
    high_dropout_heads = FusionHeads(embed_dim=EMBED_DIM, classification_dropout=0.9)
    # Copy weights so only dropout rate differs.
    high_dropout_heads.classification_head.linear.load_state_dict(
        low_dropout_heads.classification_head.linear.state_dict()
    )
    x = torch.randn(8, EMBED_DIM)
    low_var = low_dropout_heads.classification_head.predict_with_uncertainty(x, n_passes=30)["variance"].mean()
    high_var = high_dropout_heads.classification_head.predict_with_uncertainty(x, n_passes=30)["variance"].mean()
    assert high_var > low_var
