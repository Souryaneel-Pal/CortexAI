import torch

from src.explain.gradcam import GradCAM, ScoreCAM
from src.models.face_cnn import FaceEmotionEncoder


def test_gradcam_output_shape_and_range():
    model = FaceEmotionEncoder(backbone="simple_cnn")
    cam_extractor = GradCAM(model, model.gradcam_target_layer)
    x = torch.randn(3, 1, 48, 48)
    heatmap = cam_extractor(x)
    assert heatmap.shape == (3, 48, 48)
    assert (heatmap >= 0).all() and (heatmap <= 1).all()
    cam_extractor.remove_hooks()


def test_gradcam_differs_by_target_class():
    torch.manual_seed(0)
    model = FaceEmotionEncoder(backbone="simple_cnn")
    cam_extractor = GradCAM(model, model.gradcam_target_layer)
    x = torch.randn(1, 1, 48, 48)
    heatmap_class_0 = cam_extractor(x, target_class=0)
    heatmap_class_1 = cam_extractor(x, target_class=1)
    assert not torch.allclose(heatmap_class_0, heatmap_class_1)
    cam_extractor.remove_hooks()


def test_gradcam_does_not_change_model_training_mode():
    model = FaceEmotionEncoder(backbone="simple_cnn")
    model.train()
    cam_extractor = GradCAM(model, model.gradcam_target_layer)
    x = torch.randn(2, 1, 48, 48)
    cam_extractor(x)
    assert model.training  # restored to its original (train) mode
    cam_extractor.remove_hooks()


def test_scorecam_output_shape_and_range():
    model = FaceEmotionEncoder(backbone="simple_cnn")
    cam_extractor = ScoreCAM(model, model.gradcam_target_layer, max_channels=16)
    x = torch.randn(2, 1, 48, 48)
    heatmap = cam_extractor(x)
    assert heatmap.shape == (2, 48, 48)
    assert (heatmap >= 0).all() and (heatmap <= 1).all()
    cam_extractor.remove_hooks()
