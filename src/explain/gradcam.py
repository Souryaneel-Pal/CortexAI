"""Grad-CAM / Score-CAM for the facial encoder (docs/PROJECT_PLAN.md P3, Objective 3
Level 2: "SHAP ranks the 18 tabular signals; Grad-CAM heatmaps the face regions").

Hooks src/models/face_cnn.py's `FaceEmotionEncoder.gradcam_target_layer` (the
CBAM-attended feature map) so heatmaps reflect the attention-focused
regions (eyes/mouth), per docs/MINDSCOPE_Blueprint.pdf Sec. 04.

No dependency beyond torch -- implemented directly via forward/backward
hooks rather than a third-party CAM library, so it works against exactly
the target layer FaceEmotionEncoder exposes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM:
    """Standard Grad-CAM (Selvaraju et al. 2017): weight each channel of the
    target layer's activation map by the average gradient of the target
    class score w.r.t. that channel, then combine and ReLU.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __call__(self, x: torch.Tensor, target_class: int | torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, 1, H, W) input batch. Returns a (B, H, W) heatmap in [0, 1],
        upsampled to the input's spatial size.

        `target_class`: int (same class for the whole batch), a (B,) tensor
        of per-sample class indices, or None (uses each sample's argmax
        prediction).
        """
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()

        with torch.enable_grad():
            x_input = x.clone().detach().requires_grad_(True)
            _embedding, logits = self.model(x_input)
            if target_class is None:
                target_class = logits.argmax(dim=-1)
            elif isinstance(target_class, int):
                target_class = torch.full((x_input.shape[0],), target_class, device=x_input.device, dtype=torch.long)

            selected = logits.gather(1, target_class.unsqueeze(1)).squeeze(1)
            selected.sum().backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = F.relu((weights * self._activations).sum(dim=1))  # (B, H', W')
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        self.model.train(was_training)
        return cam


class ScoreCAM:
    """Score-CAM (Wang et al. 2020): gradient-free alternative. Each channel
    of the target layer's activation is used as a mask over the input; the
    channel's weight is the model's confidence increase when that mask is
    applied. More faithful than Grad-CAM but O(num_channels) forward passes,
    so used as the higher-fidelity option when latency isn't critical
    (docs/MINDSCOPE_Blueprint.pdf Sec. 06: "Grad-CAM / Score-CAM").
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, max_channels: int = 64):
        self.model = model
        self.target_layer = target_layer
        self.max_channels = max_channels
        self._activations: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def remove_hooks(self) -> None:
        self._handle.remove()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, target_class: int | torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, 1, H, W). Returns (B, H, W) heatmap in [0, 1]. Only supports
        batch size 1 semantics per-sample internally (loops over the batch)
        since each sample needs its own set of masked forward passes.
        """
        was_training = self.model.training
        self.model.eval()

        _embedding, base_logits = self.model(x)
        if target_class is None:
            target_class = base_logits.argmax(dim=-1)
        elif isinstance(target_class, int):
            target_class = torch.full((x.shape[0],), target_class, device=x.device, dtype=torch.long)

        batch_cams = []
        for b in range(x.shape[0]):
            activation = self._activations[b : b + 1]  # (1, C, H', W')
            num_channels = min(activation.shape[1], self.max_channels)
            upsampled = F.interpolate(
                activation[:, :num_channels], size=x.shape[-2:], mode="bilinear", align_corners=False
            )  # (1, C, H, W)

            masks = upsampled - upsampled.amin(dim=(2, 3), keepdim=True)
            masks = masks / (masks.amax(dim=(2, 3), keepdim=True) + 1e-8)  # normalize each channel mask to [0,1]

            masked_inputs = x[b : b + 1] * masks.transpose(0, 1)  # (C, 1, H, W)
            _emb, masked_logits = self.model(masked_inputs)
            channel_scores = F.softmax(masked_logits, dim=-1)[:, target_class[b]]  # (C,)

            cam = (channel_scores.view(-1, 1, 1) * masks[0]).sum(dim=0)
            cam = F.relu(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            batch_cams.append(cam)

        self.model.train(was_training)
        return torch.stack(batch_cams, dim=0)
