import torch

from src.train.losses import (
    ClassBalancedFocalLoss,
    UncertaintyWeightedMultiTaskLoss,
    class_balanced_weights,
    consistency_loss,
    score_regression_loss,
)


def test_class_balanced_weights_favor_rare_classes():
    weights = class_balanced_weights({"a": 1000, "b": 10})
    assert weights[1] > weights[0]


def test_focal_loss_is_finite_and_positive():
    criterion = ClassBalancedFocalLoss(gamma=2.0)
    logits = torch.randn(8, 4, requires_grad=True)
    targets = torch.randint(0, 4, (8,))
    loss = criterion(logits, targets)
    assert loss.item() > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_lower_for_confident_correct_predictions():
    criterion = ClassBalancedFocalLoss(gamma=2.0)
    targets = torch.tensor([0, 1])
    confident_logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    unsure_logits = torch.tensor([[0.1, 0.0], [0.0, 0.1]])
    assert criterion(confident_logits, targets).item() < criterion(unsure_logits, targets).item()


def test_huber_score_loss_zero_for_perfect_predictions():
    preds = torch.tensor([[1.0, 2.0, 3.0]])
    loss = score_regression_loss(preds, preds.clone())
    assert loss.item() == 0.0


def test_uncertainty_weighted_multitask_loss_combines_tasks():
    criterion = UncertaintyWeightedMultiTaskLoss(num_tasks=2)
    cls_loss = torch.tensor(1.5)
    reg_loss = torch.tensor(0.5)
    total = criterion([cls_loss, reg_loss])
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(criterion.log_vars.grad).all()


def test_consistency_loss_low_when_severe_class_matches_high_stress_score():
    # Predicted class strongly Severe, and Stress_Score near the max -- should agree.
    class_probs = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    scores = torch.tensor([[20.0, 15.0, 38.0]])  # Stress_Score near 39 max
    loss_aligned = consistency_loss(class_probs, scores)

    # Predicted class strongly Severe, but Stress_Score near zero -- should disagree.
    scores_mismatched = torch.tensor([[20.0, 15.0, 1.0]])
    loss_mismatched = consistency_loss(class_probs, scores_mismatched)

    assert loss_aligned.item() < loss_mismatched.item()
