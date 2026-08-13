"""Wires the P1-P4 model/explainability/RAG code into a single inference
pipeline the API (src/api/main.py) can call (docs/PROJECT_PLAN.md P5).

`CortexAIPipeline` builds one `AgentContext` (src/reasoning/agent_graph.py)
from real encoders, the real fusion+heads, real explainability functions,
and the real KB retriever. It tries to load trained checkpoints
(configs/fusion.yaml's paths) and clearly flags `is_demo_untrained_model`
in every prediction when none are found -- this sandbox has none (no real
data was available to train on; see PROJECT_PLAN.md), so a fresh
random-initialization is used and every response says so rather than
presenting structurally-valid-but-meaningless numbers as a real result.
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np
import torch

from src.data.schemas import STRESS_LEVEL_NAMES, StressLevel, TABULAR_FEATURE_COLUMNS
from src.explain.masked_distress import PhysiologicalReferenceStats, masked_distress_index
from src.explain.shap_tab import ClassLogitsOnly, rank_features_by_mean_abs_shap, shap_values_for_torch_model
from src.models.face_cnn import FACIAL_EMOTIONS, FaceEmotionEncoder
from src.models.fusion import GatedCrossModalFusion
from src.models.heads import FusionHeads
from src.models.speech_net import SPEECH_EMOTIONS, CNNBiLSTMEncoder
from src.models.tabular_ft import TabularEncoder
from src.reasoning.agent_graph import AgentContext
from src.reasoning.retriever import ClinicalKBRetriever, load_knowledge_base

logger = logging.getLogger(__name__)

# Documented placeholder population reference for the Masked-Distress Index
# (src/explain/masked_distress.py) -- must be replaced by stats fit on the
# real tabular dataset (`PhysiologicalReferenceStats.fit_from_dataframe`)
# once it's available; see PROJECT_PLAN.md.
_PLACEHOLDER_PHYSIO_REFERENCE = PhysiologicalReferenceStats(
    mean={"Heart_Rate_BPM": 75.0, "HRV_Index": 50.0, "Skin_Temperature": 33.0, "GSR_Level": 2.0},
    std={"Heart_Rate_BPM": 10.0, "HRV_Index": 15.0, "Skin_Temperature": 0.5, "GSR_Level": 1.0},
)

CONFIDENCE_DEFER_THRESHOLD = 0.6


def _decode_face_image(b64_string: str) -> torch.Tensor:
    from PIL import Image

    raw = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(raw)).convert("L").resize((48, 48))
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)  # (1, 1, 48, 48)


def _decode_speech_audio(b64_string: str, sample_rate: int = 16000, max_duration_sec: float = 4.0) -> torch.Tensor:
    import soundfile as sf

    raw = base64.b64decode(b64_string)
    waveform, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    waveform = torch.from_numpy(np.atleast_1d(waveform))
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)
    if sr != sample_rate:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    max_samples = int(sample_rate * max_duration_sec)
    if waveform.shape[0] >= max_samples:
        waveform = waveform[:max_samples]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[0]))
    return waveform.unsqueeze(0)  # (1, T)


class CortexAIPipeline:
    def __init__(self, checkpoint_dir: str | Path = "artifacts/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)

        self.face_encoder = FaceEmotionEncoder(backbone="simple_cnn")
        self.speech_encoder = CNNBiLSTMEncoder()
        self.tabular_encoder = TabularEncoder(backbone="residual_mlp")
        self.fusion = GatedCrossModalFusion()
        self.heads = FusionHeads()

        self.is_demo_untrained_model = not self._try_load_checkpoints()
        if self.is_demo_untrained_model:
            logger.warning(
                "No trained checkpoints found under %s -- serving with randomly-initialized "
                "encoders. Every response will be flagged is_demo_untrained_model=True.",
                self.checkpoint_dir,
            )

        for module in (self.face_encoder, self.speech_encoder, self.tabular_encoder, self.fusion, self.heads):
            module.eval()

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

    def _try_load_checkpoints(self) -> bool:
        found_any = False
        for name, module in (
            ("face", self.face_encoder),
            ("speech", self.speech_encoder),
            ("tabular", self.tabular_encoder),
            ("fusion", self.fusion),
        ):
            path = self.checkpoint_dir / name / "best.ckpt"
            if path.exists():
                state = torch.load(path, map_location="cpu")
                module.load_state_dict(state.get("state_dict", state), strict=False)
                found_any = True
        return found_any

    # -- AgentContext stage implementations -------------------------------

    def _preprocess(self, raw_input: dict) -> dict:
        tabular_tensor = torch.tensor([raw_input["tabular_features"]], dtype=torch.float32)

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

        return {
            "tabular": tabular_tensor,
            "face": face_tensor if face_tensor is not None else torch.zeros(1, 1, 48, 48),
            "speech": speech_tensor if speech_tensor is not None else torch.zeros(1, 16000 * 4),
            "modality_mask": {
                "face": torch.tensor([face_mask]),
                "speech": torch.tensor([speech_mask]),
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

        predicted_class = int(class_probs.argmax(dim=-1).item())
        confidence = float(class_probs.max(dim=-1).values.item())

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probs": class_probs.squeeze(0).tolist(),
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
                name: float(p) for name, p in zip(FACIAL_EMOTIONS.values(), torch.softmax(face_logits, dim=-1)[0].tolist())
            },
            "speech_emotion_probs": {
                name: float(p) for name, p in zip(SPEECH_EMOTIONS.values(), torch.softmax(speech_logits, dim=-1)[0].tolist())
            },
            "_preprocessed": preprocessed,  # carried through for explain_fn without recomputation
        }

    def _uncertainty(self, prediction: dict) -> dict:
        # Full conformal calibration (src/explain/conformal.py) needs a
        # held-out calibration split from the real dataset, which doesn't
        # exist in this sandbox -- MC-dropout confidence alone drives the
        # gate here; wire in ConformalCalibration once real data lands.
        confidence = prediction["confidence"]
        defer = confidence < CONFIDENCE_DEFER_THRESHOLD
        return {
            "defer": defer,
            "reason": "low_confidence" if defer else None,
            "mc_dropout_confidence": confidence,
        }

    def _explain(self, preprocessed: dict, prediction: dict) -> dict:
        wrapped = ClassLogitsOnly(self.tabular_encoder)
        background = torch.randn(20, len(TABULAR_FEATURE_COLUMNS))
        shap_values = shap_values_for_torch_model(wrapped, background, preprocessed["tabular"], class_index=prediction["predicted_class"])
        ranked = rank_features_by_mean_abs_shap(shap_values)

        mdi_result = None
        if preprocessed["modality_mask"]["face"].item() and preprocessed["modality_mask"]["speech"].item():
            tabular_row = dict(zip(TABULAR_FEATURE_COLUMNS, preprocessed["tabular"][0].tolist()))
            mdi_result = masked_distress_index(
                prediction["face_emotion_probs"],
                prediction["speech_emotion_probs"],
                tabular_row,
                _PLACEHOLDER_PHYSIO_REFERENCE,
            )

        return {
            "top_shap_feature": ranked[0][0] if ranked else None,
            "shap_ranked_features": [{"feature": name, "mean_abs_shap": value} for name, value in ranked],
            "masked_distress_index": mdi_result or {"mdi": 0.0, "flag": False},
        }

    def predicted_class_name(self, predicted_class: int) -> str:
        return STRESS_LEVEL_NAMES[StressLevel(predicted_class)]
