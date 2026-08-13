"""Wires the P1-P4 model/explainability/RAG code into a single inference
pipeline the API (src/api/main.py) can call (PROJECT_PLAN.md P5).

`CortexAIPipeline` builds one `AgentContext` (src/reasoning/agent_graph.py)
from the trained encoders, the trained fusion+heads, the real explainability
functions, and the real KB retriever.

Checkpoint loading, in priority order:
  1. `artifacts/checkpoints/fusion/best.ckpt` -- a Lightning checkpoint of
     FusionLightningModule, which contains *all five* sub-modules
     (face_encoder / speech_encoder / tabular_encoder / fusion / heads)
     under their attribute prefixes. One file restores the whole graph.
  2. Per-modality `artifacts/checkpoints/{face,speech,tabular}/best.ckpt`
     (keys prefixed `model.`) for anything the fusion checkpoint didn't cover.

`is_demo_untrained_model` is False only when every module the served path
actually uses was restored from a checkpoint. If anything is still randomly
initialised the flag stays True and every response says so, rather than
presenting structurally-valid-but-meaningless numbers as a real result.
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np
import torch

from src.data.loaders import DEFAULT_TABULAR_CSV
from src.data.schemas import STRESS_LEVEL_NAMES, StressLevel, TABULAR_FEATURE_COLUMNS
from src.explain.gradcam import GradCAM
from src.explain.ig_audio import integrated_gradients_audio, pool_attribution_to_frames
from src.explain.masked_distress import PhysiologicalReferenceStats, masked_distress_index
from src.explain.shap_tab import (
    ClassLogitsOnly,
    combine_shap_and_attention,
    rank_features_by_mean_abs_shap,
    shap_values_for_torch_model,
)
from src.models.face_cnn import FACIAL_EMOTIONS, FaceEmotionEncoder
from src.models.fusion import GatedCrossModalFusion
from src.models.heads import FusionHeads
from src.models.speech_net import SPEECH_EMOTIONS, build_speech_encoder
from src.models.tabular_ft import TabularEncoder
from src.reasoning.agent_graph import AgentContext
from src.reasoning.retriever import ClinicalKBRetriever, load_knowledge_base

logger = logging.getLogger(__name__)

CONFIDENCE_DEFER_THRESHOLD = 0.6
SHAP_BACKGROUND_SAMPLES = 64
GRADCAM_RENDER_SIZE = 192  # upscale the 48x48 CAM for a legible UI overlay

# Fallback population reference for the Masked-Distress Index, used only if
# the tabular CSV isn't present to fit real statistics from.
_FALLBACK_PHYSIO_REFERENCE = PhysiologicalReferenceStats(
    mean={"Heart_Rate_BPM": 75.0, "HRV_Index": 50.0, "Skin_Temperature": 33.0, "GSR_Level": 2.0},
    std={"Heart_Rate_BPM": 10.0, "HRV_Index": 15.0, "Skin_Temperature": 0.5, "GSR_Level": 1.0},
)


def _decode_face_image(b64_string: str) -> torch.Tensor:
    if not b64_string:
        raise ValueError("No facial image base64 string provided.")

    from PIL import Image

    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    b64_string = b64_string.strip()
    missing_padding = len(b64_string) % 4
    if missing_padding:
        b64_string += "=" * (4 - missing_padding)

    raw = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(raw)).convert("L").resize((48, 48))
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)  # (1, 1, 48, 48)


class UnsupportedAudioFormatError(ValueError):
    """The uploaded clip is in a container no installed decoder can read.

    Distinct from a generic ValueError so the API layer can answer 400 (the
    caller sent something we cannot use) instead of 500 (we broke).
    """


def _supported_audio_formats() -> list[str]:
    """Formats libsndfile can actually read in this environment."""
    import soundfile as sf

    preferred = ["WAV", "FLAC", "OGG", "MP3", "AIFF", "CAF"]
    available = set(sf.available_formats())
    return [fmt for fmt in preferred if fmt in available]


def _ffmpeg_backend_available() -> bool:
    """Whether torchaudio's FFmpeg-backed decoder can actually load.

    torchcodec ships one shared object per FFmpeg major version and resolves
    the matching `libavutil` at import time, so "torchaudio is installed" does
    not imply "FFmpeg decoding works". Probing it keeps the error message
    honest instead of telling someone to install FFmpeg they already have.
    """
    try:
        from torchcodec._internally_replaced_utils import load_core_libraries

        load_core_libraries()
        return True
    except Exception:
        return False


def _describe_audio_container(raw: bytes) -> str:
    """Name the container from its magic bytes, for the error message.

    "we could not decode your M4A" is actionable; "Format not recognised" is
    not. Sniffing the header costs nothing and is far more reliable than
    trusting a filename we were never sent.
    """
    if len(raw) < 12:
        return "empty or truncated file"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "WAV"
    if raw[:4] == b"fLaC":
        return "FLAC"
    if raw[:4] == b"OggS":
        return "OGG"
    if raw[:3] == b"ID3" or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "MP3"
    if raw[4:8] == b"ftyp":
        brand = raw[8:12].decode("ascii", "replace").strip()
        # M4A, M4B, mp42, isom, qt ... all MPEG-4 containers libsndfile can't read.
        return f"MPEG-4/M4A container (brand {brand!r})"
    if raw[:4] == b"\x1aE\xdf\xa3":
        return "WEBM/Matroska"
    if raw[:4] == b"FORM":
        return "AIFF"
    return f"unrecognised container (starts with {raw[:4]!r})"


def _decode_speech_audio(b64_string: str, sample_rate: int = 16000, max_duration_sec: float = 4.0) -> torch.Tensor:
    if not b64_string:
        raise ValueError("No speech audio base64 string provided.")

    import soundfile as sf  # noqa: F401  (used below; imported early to fail fast)

    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    b64_string = b64_string.strip()
    missing_padding = len(b64_string) % 4
    if missing_padding:
        b64_string += "=" * (4 - missing_padding)

    raw = base64.b64decode(b64_string)
    waveform = None
    sr = None

    try:
        waveform_np, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        waveform = torch.from_numpy(np.atleast_1d(waveform_np))
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)
    except Exception:
        try:
            import torchaudio
            waveform_tensor, sr = torchaudio.load(io.BytesIO(raw))
            if waveform_tensor.ndim > 1:
                waveform = waveform_tensor.mean(dim=0)
            else:
                waveform = waveform_tensor
        except Exception as ta_err:
            # Both decoders are out. Raise something a *user* can act on rather
            # than the ~400-line libtorchcodec loader traceback, which lists a
            # failure for every FFmpeg major version it probes and buries the
            # one line that matters.
            #
            # torchaudio's fallback needs FFmpeg's shared libraries via
            # torchcodec; where those are absent, the only decodable formats
            # are libsndfile's, which exclude M4A/AAC/MP4/WEBM -- exactly what
            # Voice Memos and MediaRecorder produce. The frontend now converts
            # audio to 16 kHz mono WAV in the browser (frontend/src/lib/audio.ts)
            # so this path should only be reachable by direct API callers.
            container = _describe_audio_container(raw)
            if _ffmpeg_backend_available():
                # FFmpeg is present, so the container isn't the problem --
                # the file itself is bad, or has no audio stream.
                remedy = (
                    "FFmpeg is available here, so this is most likely a corrupt file, "
                    "a video with no audio track, or an empty recording."
                )
            else:
                remedy = (
                    f"This server can only read {', '.join(_supported_audio_formats())}; "
                    "M4A/AAC, MP4 and WEBM additionally need FFmpeg, which is not installed. "
                    "Convert the clip to WAV or FLAC, or install FFmpeg "
                    "(`brew install ffmpeg`) and restart the API."
                )
            raise UnsupportedAudioFormatError(f"Could not decode the audio ({container}). {remedy}") from ta_err

    if sr != sample_rate:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    max_samples = int(sample_rate * max_duration_sec)
    if waveform.shape[0] >= max_samples:
        waveform = waveform[:max_samples]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[0]))
    return waveform.unsqueeze(0)  # (1, T)


def _heatmap_to_rgb(cam: np.ndarray) -> np.ndarray:
    """Map a [0,1] CAM to an RGB 'inferno-like' ramp without pulling in
    matplotlib. Dark blue/purple = low attribution, orange/yellow = high.
    """
    cam = np.clip(cam, 0.0, 1.0)
    stops = np.array(
        [
            [0.00, 0.00, 0.02, 0.09],
            [0.25, 0.28, 0.05, 0.43],
            [0.50, 0.66, 0.13, 0.37],
            [0.75, 0.95, 0.41, 0.14],
            [1.00, 0.99, 0.91, 0.15],
        ]
    )
    r = np.interp(cam, stops[:, 0], stops[:, 1])
    g = np.interp(cam, stops[:, 0], stops[:, 2])
    b = np.interp(cam, stops[:, 0], stops[:, 3])
    return np.stack([r, g, b], axis=-1)


def render_gradcam_overlay_png(face_tensor: torch.Tensor, cam: torch.Tensor, alpha: float = 0.5) -> str:
    """Blend a Grad-CAM heatmap over the input face and return a base64 PNG
    the frontend can drop straight into an <img src="data:image/png;base64,...">.

    `face_tensor`: (1, 1, 48, 48) in [0,1]. `cam`: (1, 48, 48) in [0,1].
    """
    from PIL import Image

    face = face_tensor[0, 0].detach().cpu().numpy()
    heat = cam[0].detach().cpu().numpy()

    face_rgb = np.stack([face] * 3, axis=-1)
    heat_rgb = _heatmap_to_rgb(heat)
    # Weight the blend by CAM intensity so cold regions show the face almost
    # unmodified and only genuinely attributed regions get colour.
    weight = (alpha * heat)[..., None]
    blended = np.clip(face_rgb * (1 - weight) + heat_rgb * weight, 0.0, 1.0)

    image = Image.fromarray((blended * 255).astype(np.uint8), mode="RGB")
    image = image.resize((GRADCAM_RENDER_SIZE, GRADCAM_RENDER_SIZE), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ScaledTabularModel(torch.nn.Module):
    """Presents the tabular encoder as if it consumed *raw* feature units.

    Counterfactual search (src/explain/counterfactual.py) reasons in real-world
    units -- "raise Sleep_Quality from 2 to 4" -- but the encoder was trained
    on StandardScaler-transformed features. This wrapper applies the fitted
    scaler inside the forward pass so a raw-unit grid search and a
    scaled-input model can be composed without either side knowing about the
    other. Without it the search silently feeds the encoder out-of-distribution
    values and proposes meaningless levers.
    """

    def __init__(self, tabular_encoder: torch.nn.Module, scaler):
        super().__init__()
        self.tabular_encoder = tabular_encoder
        if scaler is not None:
            self.register_buffer("mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))
        else:
            self.mean = None
            self.scale = None

    def forward(self, x: torch.Tensor):
        if self.mean is not None:
            x = (x - self.mean) / self.scale
        return self.tabular_encoder(x)


class CortexAIPipeline:
    def __init__(
        self,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
        tabular_csv: str | Path = DEFAULT_TABULAR_CSV,
        fusion_config: str | Path = "configs/fusion.yaml",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)

        # Backbones are read from configs/fusion.yaml rather than hardcoded:
        # they must match whatever the checkpoints were actually trained with,
        # and a hardcoded guess silently diverges the moment a config changes
        # (e.g. the speech encoder switching between wav2vec2 and the
        # CNN-BiLSTM fallback, whose embedding projections differ in shape).
        backbones = self._read_backbones(Path(fusion_config))
        self.backbones = backbones

        self.face_encoder = FaceEmotionEncoder(backbone=backbones["face"], pretrained=False)
        self.speech_encoder = build_speech_encoder(backbone=backbones["speech"])
        self.tabular_encoder = TabularEncoder(backbone=backbones["tabular"])
        self.fusion = GatedCrossModalFusion()
        self.heads = FusionHeads()

        self.loaded_modules: dict[str, bool] = self._load_checkpoints()
        self.is_demo_untrained_model = not all(self.loaded_modules.values())
        if self.is_demo_untrained_model:
            missing = [name for name, ok in self.loaded_modules.items() if not ok]
            logger.warning(
                "Serving with randomly-initialized module(s) %s -- no checkpoint found under %s. "
                "Every response will be flagged is_demo_untrained_model=True.",
                missing,
                self.checkpoint_dir,
            )
        else:
            logger.info("Loaded trained checkpoints for all modules from %s", self.checkpoint_dir)

        for module in (self.face_encoder, self.speech_encoder, self.tabular_encoder, self.fusion, self.heads):
            module.eval()

        # StandardScaler fitted on the tabular training split. Inference MUST
        # apply the identical transform the encoder was trained under.
        self.scaler = self._load_scaler()
        # Real population statistics for MDI's physiological z-scoring, and a
        # real SHAP background sample, both fit from the tabular dataset.
        self.physio_reference, self.shap_background = self._fit_dataset_references(Path(tabular_csv))

        # Raw-unit view of the tabular encoder, for counterfactual search.
        self.raw_unit_tabular_model = ScaledTabularModel(self.tabular_encoder, self.scaler).eval()

        documents = load_knowledge_base()
        self.retriever = ClinicalKBRetriever(embedding_backend="auto")
        self.retriever.build_index(documents)

        self.context = AgentContext(
            preprocess_fn=self._preprocess,
            predict_fn=self._predict,
            uncertainty_fn=self._uncertainty,
            explain_fn=self._explain,
            retriever=self.retriever,
        )

    # -- checkpoint / reference loading -----------------------------------

    @staticmethod
    def _read_backbones(config_path: Path) -> dict[str, str]:
        """Encoder backbone names from configs/fusion.yaml, with the
        documented defaults if the file is absent.
        """
        defaults = {"face": "efficientnet_b0", "speech": "cnn_bilstm", "tabular": "ft_transformer"}
        if not config_path.exists():
            logger.warning("No fusion config at %s -- assuming backbones %s", config_path, defaults)
            return defaults
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        encoders = cfg.get("encoders", {})
        return {
            "face": encoders.get("face_backbone", defaults["face"]),
            "speech": encoders.get("speech_backbone", defaults["speech"]),
            "tabular": encoders.get("tabular_backbone", defaults["tabular"]),
        }

    @staticmethod
    def _load_into(module: torch.nn.Module, state_dict: dict, name: str) -> bool:
        """Load `state_dict` into `module`, returning whether it fully matched.

        A checkpoint trained against a different backbone produces tensors of
        the wrong shape, which `load_state_dict` raises on even with
        `strict=False` (that flag forgives missing/unexpected keys, not size
        mismatches). Serving must degrade to `is_demo_untrained_model=True`
        rather than crash the API at startup, so the mismatch is caught,
        logged loudly, and reported as "not loaded".
        """
        try:
            missing, _unexpected = module.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            logger.error(
                "Checkpoint for %s does not match the configured backbone and was ignored: %s",
                name,
                exc,
            )
            return False
        if missing:
            logger.warning("%s: %d parameter(s) missing from checkpoint", name, len(missing))
            return False
        return True

    def _load_checkpoints(self) -> dict[str, bool]:
        """Restore every module, preferring the fusion checkpoint (which holds
        all five) and falling back to per-modality checkpoints.
        """
        targets = {
            "face_encoder": self.face_encoder,
            "speech_encoder": self.speech_encoder,
            "tabular_encoder": self.tabular_encoder,
            "fusion": self.fusion,
            "heads": self.heads,
        }
        loaded = dict.fromkeys(targets, False)

        fusion_ckpt = self.checkpoint_dir / "fusion" / "best.ckpt"
        if fusion_ckpt.exists():
            state = torch.load(fusion_ckpt, map_location="cpu", weights_only=False)
            state_dict = state.get("state_dict", state)
            for name, module in targets.items():
                prefix = f"{name}."
                sub = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
                if sub:
                    loaded[name] = self._load_into(module, sub, f"{name} (fusion checkpoint)")

        # Per-modality fallbacks (Lightning wraps the encoder as `self.model`).
        for name, folder in (("face_encoder", "face"), ("speech_encoder", "speech"), ("tabular_encoder", "tabular")):
            if loaded[name]:
                continue
            path = self.checkpoint_dir / folder / "best.ckpt"
            if not path.exists():
                continue
            state = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = state.get("state_dict", state)
            sub = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
            if sub:
                loaded[name] = self._load_into(targets[name], sub, f"{name} ({folder}/best.ckpt)")

        return loaded

    def _load_scaler(self):
        path = self.checkpoint_dir / "tabular" / "scaler.joblib"
        if not path.exists():
            logger.warning(
                "No tabular scaler at %s -- serving raw (unscaled) features, which will not match "
                "how the encoder was trained.",
                path,
            )
            return None
        import joblib

        return joblib.load(path)

    def _fit_dataset_references(self, tabular_csv: Path):
        """Fit the MDI physiological reference and the SHAP background sample
        from the real tabular dataset. Falls back to documented placeholder
        statistics (and a Gaussian background) if the CSV isn't present.
        """
        if not tabular_csv.exists():
            logger.warning(
                "Tabular CSV not found at %s -- using placeholder physiological reference stats "
                "and a Gaussian SHAP background.",
                tabular_csv,
            )
            return _FALLBACK_PHYSIO_REFERENCE, torch.randn(SHAP_BACKGROUND_SAMPLES, len(TABULAR_FEATURE_COLUMNS))

        import pandas as pd

        df = pd.read_csv(tabular_csv)
        physio_reference = PhysiologicalReferenceStats.fit_from_dataframe(df)

        features = df[TABULAR_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(features), size=min(SHAP_BACKGROUND_SAMPLES, len(features)), replace=False)
        background = features[sample_idx]
        if self.scaler is not None:
            background = self.scaler.transform(background).astype(np.float32)
        return physio_reference, torch.from_numpy(background).float()

    # -- AgentContext stage implementations -------------------------------

    def _preprocess(self, raw_input: dict) -> dict:
        raw_features = np.asarray([raw_input["tabular_features"]], dtype=np.float32)
        scaled = self.scaler.transform(raw_features).astype(np.float32) if self.scaler is not None else raw_features
        tabular_tensor = torch.from_numpy(scaled).float()

        face_tensor = None
        face_mask = False
        if raw_input.get("face_image_base64"):
            face_tensor = _decode_face_image(raw_input["face_image_base64"])
            face_mask = True

        speech_tensor = None
        speech_mask = False
        if raw_input.get("speech_audio_base64"):
            speech_tensor = _decode_speech_audio(raw_input["speech_audio_base64"])
            speech_mask = True

        from src.api.database import get_db_setting
        ignore_face = get_db_setting("ignore_face", False)
        ignore_speech = get_db_setting("ignore_speech", False)
        ignore_tabular = get_db_setting("ignore_tabular", False)

        return {
            "tabular": tabular_tensor,
            # Keep the unscaled row too -- MDI's physiological z-scoring and
            # the counterfactual grid both work in real-world feature units.
            "tabular_raw": torch.from_numpy(raw_features).float(),
            "face": face_tensor if face_tensor is not None else torch.zeros(1, 1, 48, 48),
            "speech": speech_tensor if speech_tensor is not None else torch.zeros(1, 16000 * 4),
            "modality_mask": {
                "face": torch.tensor([face_mask and not ignore_face]),
                "speech": torch.tensor([speech_mask and not ignore_speech]),
                "tabular": torch.tensor([not ignore_tabular]),
            },
        }

    @torch.no_grad()
    def _predict(self, preprocessed: dict) -> dict:
        face_embedding, face_logits = self.face_encoder(preprocessed["face"])
        speech_embedding, speech_logits = self.speech_encoder(preprocessed["speech"])
        tabular_embedding = self.tabular_encoder.encoder(preprocessed["tabular"])

        fused, modality_weights = self.fusion(
            face_embedding, speech_embedding, tabular_embedding, modality_mask=preprocessed["modality_mask"]
        )
        class_logits, class_probs, score_preds = self.heads(fused)

        # MC-dropout gives the uncertainty gate a distribution rather than a
        # single deterministic softmax (docs/MINDSCOPE_Blueprint.pdf Sec. 03).
        mc = self.heads.classification_head.predict_with_uncertainty(fused, n_passes=20)
        mean_probs = mc["mean_probs"]

        # NOTE: there was previously an "AI confidence floor" here that rewrote
        # `mean_probs` so the top class always read >= 0.85. It is deliberately
        # gone, and must not come back. Two reasons:
        #
        #   1. It fabricated the number. On this dataset the honest MC-dropout
        #      confidence sits well below 0.85, so the floor fired on virtually
        #      every request and every assessment reported exactly "85%
        #      confidence" -- a constant presented to clinicians as a model
        #      output.
        #   2. It silently disabled the uncertainty gate, which is the system's
        #      main safety feature. The gate defers to a human when confidence
        #      falls below the configured threshold (default 0.60); a hard floor
        #      at 0.85 means that comparison can never be true, so nothing was
        #      ever routed for human review.
        #
        # If low confidence is undesirable, the fix is a better model or a
        # tuned threshold in Settings -- not overwriting the output.
        predicted_class = int(mean_probs.argmax(dim=-1).item())
        confidence = float(mean_probs.max(dim=-1).values.item())

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probs": mean_probs.squeeze(0).tolist(),
            "deterministic_class_probs": class_probs.squeeze(0).tolist(),
            "predictive_entropy": float(mc["predictive_entropy"].item()),
            "class_probs_variance": mc["variance"].squeeze(0).tolist(),
            "scores": {
                "Depression_Score": float(score_preds[0, 0].item()),
                "Anxiety_Score": float(score_preds[0, 1].item()),
                "Stress_Score": float(score_preds[0, 2].item()),
            },
            "modality_weights": {
                "face": float(modality_weights[0, 0].item()),
                "speech": float(modality_weights[0, 1].item()),
                "tabular": float(modality_weights[0, 2].item()),
            },
            "face_emotion_probs": {
                name: float(p)
                for name, p in zip(FACIAL_EMOTIONS.values(), torch.softmax(face_logits, dim=-1)[0].tolist())
            },
            "speech_emotion_probs": {
                name: float(p)
                for name, p in zip(SPEECH_EMOTIONS.values(), torch.softmax(speech_logits, dim=-1)[0].tolist())
            },
            "_preprocessed": preprocessed,  # carried through for explain_fn without recomputation
        }

    def _uncertainty(self, prediction: dict) -> dict:
        # Full conformal calibration (src/explain/conformal.py) needs a
        # held-out calibration split; MC-dropout confidence drives the gate
        # here, with the predictive entropy reported alongside it.
        from src.api.database import get_db_setting
        threshold = get_db_setting("uncertainty_threshold", 0.60)
        confidence = prediction["confidence"]
        defer = confidence < threshold
        return {
            "defer": defer,
            "reason": "low_confidence" if defer else None,
            "mc_dropout_confidence": confidence,
        }

    def _explain(self, preprocessed: dict, prediction: dict) -> dict:
        explanations: dict = {}

        # -- Level 2a: SHAP over the 18 tabular features, plus the
        # FT-Transformer's own per-feature attention as an independent signal.
        wrapped = ClassLogitsOnly(self.tabular_encoder)
        shap_values = shap_values_for_torch_model(
            wrapped, self.shap_background, preprocessed["tabular"], class_index=prediction["predicted_class"]
        )
        ranked = rank_features_by_mean_abs_shap(shap_values)
        explanations["top_shap_feature"] = ranked[0][0] if ranked else None
        explanations["shap_ranked_features"] = [
            {"feature": name, "mean_abs_shap": value} for name, value in ranked
        ]
        # Signed SHAP for this single row -- the UI's diverging bar chart needs
        # direction (raises vs lowers severity), which mean|SHAP| discards.
        explanations["signed_shap"] = [
            {"feature": name, "shap": float(shap_values[0, i])}
            for i, name in enumerate(TABULAR_FEATURE_COLUMNS)
        ]
        try:
            attention = self.tabular_encoder.encoder.feature_attention_weights(preprocessed["tabular"])
            explanations["shap_with_attention"] = combine_shap_and_attention(shap_values, attention)
        except (AttributeError, RuntimeError) as exc:
            # residual_mlp backbone has no attention to report -- not an error.
            logger.debug("No feature attention available: %s", exc)
            explanations["shap_with_attention"] = None

        face_present = bool(preprocessed["modality_mask"]["face"].item())
        speech_present = bool(preprocessed["modality_mask"]["speech"].item())

        # -- Level 2b: Grad-CAM over the face, rendered as a ready-to-display
        # overlay PNG so the frontend needs no image processing of its own.
        explanations["gradcam"] = None
        if face_present:
            cam_helper = GradCAM(self.face_encoder, self.face_encoder.gradcam_target_layer)
            try:
                face_input = preprocessed["face"].clone().requires_grad_(True)
                cam = cam_helper(face_input)
                explanations["gradcam"] = {
                    "overlay_png_base64": render_gradcam_overlay_png(preprocessed["face"], cam),
                    "heatmap": cam[0].detach().cpu().numpy().round(4).tolist(),
                    "target_layer": "feature_extractor.cbam",
                    "predicted_emotion": max(
                        prediction["face_emotion_probs"], key=prediction["face_emotion_probs"].get
                    ),
                }
            finally:
                cam_helper.remove_hooks()

        # -- Level 2c: Integrated Gradients over the waveform, pooled into
        # frames so the UI can render a time-axis importance strip.
        explanations["audio_integrated_gradients"] = None
        if speech_present:
            with torch.enable_grad():
                attribution = integrated_gradients_audio(
                    self.speech_encoder, preprocessed["speech"].clone(), n_steps=32
                )
            frame_size = 1600  # 100 ms at 16 kHz
            frames = pool_attribution_to_frames(attribution, frame_size)[0]
            peak = float(frames.max()) or 1.0
            explanations["audio_integrated_gradients"] = {
                "frame_importance": (frames / peak).detach().cpu().numpy().round(4).tolist(),
                "frame_ms": frame_size / 16.0,
                "predicted_emotion": max(
                    prediction["speech_emotion_probs"], key=prediction["speech_emotion_probs"].get
                ),
            }

        # -- The signature metric: cross-modal contradiction. Needs both a
        # face and a voice to contradict each other, by construction.
        from src.api.database import get_db_setting
        mdi_thresh = get_db_setting("mdi_threshold", 0.50)
        mdi_result = None
        if face_present and speech_present:
            tabular_row = dict(zip(TABULAR_FEATURE_COLUMNS, preprocessed["tabular_raw"][0].tolist()))
            mdi_result = masked_distress_index(
                prediction["face_emotion_probs"],
                prediction["speech_emotion_probs"],
                tabular_row,
                self.physio_reference,
                threshold=mdi_thresh,
            )
        explanations["masked_distress_index"] = mdi_result or {
            "mdi": 0.0,
            "flag": False,
            "unavailable_reason": "requires both a face image and a speech clip",
        }
        return explanations

    def predicted_class_name(self, predicted_class: int) -> str:
        return STRESS_LEVEL_NAMES[StressLevel(predicted_class)]
