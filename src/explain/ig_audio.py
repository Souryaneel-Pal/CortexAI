"""Integrated Gradients for the speech encoder (docs/PROJECT_PLAN.md P3,
Objective 3 Level 2: "Integrated Gradients highlights the audio frames that
pushed the prediction" -- docs/MINDSCOPE_Blueprint.pdf Sec. 06).

Primary: Captum's IntegratedGradients (Sundararajan et al. 2017), attributing
the target emotion score back to each input sample of the waveform, then
pooled into per-frame (time-window) attribution so the result reads as
"which time-frequency regions mattered" without needing a spectrogram-shaped
model input.

Fallback (used when Captum is unavailable -- noted per PROJECT_PLAN.md P3
"if blocked" convention, not a silent swap): vanilla gradient saliency
(a 1-step, unintegrated version of the same idea).
"""
from __future__ import annotations

import torch


def _forward_target_score(model: torch.nn.Module, waveform: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
    _embedding, logits = model(waveform)
    return logits.gather(1, target_class.unsqueeze(1)).squeeze(1)


def integrated_gradients_audio(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    target_class: int | torch.Tensor | None = None,
    n_steps: int = 50,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """waveform: (B, T) raw audio. Returns (B, T) per-sample attribution
    (same units as the input signal), positive = pushed toward the target
    class, negative = pushed away.

    `baseline` defaults to silence (all-zero waveform), the natural "absence
    of signal" reference for audio IG.
    """
    was_training = model.training
    model.eval()

    if target_class is None:
        with torch.no_grad():
            _emb, logits = model(waveform)
            target_class = logits.argmax(dim=-1)
    elif isinstance(target_class, int):
        target_class = torch.full((waveform.shape[0],), target_class, device=waveform.device, dtype=torch.long)

    try:
        from captum.attr import IntegratedGradients

        def forward_fn(x):
            return model(x)[1]  # logits

        ig = IntegratedGradients(forward_fn)
        attributions = ig.attribute(
            waveform,
            baselines=baseline if baseline is not None else torch.zeros_like(waveform),
            target=target_class,
            n_steps=n_steps,
        )
    except ImportError:
        # Fallback: vanilla gradient saliency (Simonyan et al. 2014) -- a
        # single-step, unintegrated approximation of the same attribution.
        waveform = waveform.clone().requires_grad_(True)
        score = _forward_target_score(model, waveform, target_class)
        score.sum().backward()
        attributions = waveform.grad.detach()

    model.train(was_training)
    return attributions


def pool_attribution_to_frames(attribution: torch.Tensor, frame_size: int) -> torch.Tensor:
    """Pools a (B, T) sample-level attribution into (B, T // frame_size)
    frame-level attribution (mean absolute attribution per frame), the
    "which time-frequency regions mattered" summary shown to the user
    (docs/Proposal.pdf Sec. 06).
    """
    batch, length = attribution.shape
    n_frames = length // frame_size
    trimmed = attribution[:, : n_frames * frame_size]
    return trimmed.abs().reshape(batch, n_frames, frame_size).mean(dim=-1)
