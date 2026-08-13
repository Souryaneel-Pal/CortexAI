# CortexAI

Explainable multimodal deep-learning system for psychiatric/mental-health screening,
built for Hack4Health. CortexAI fuses facial expression, speech emotion, and
behavioural/acoustic/physiological signals to (1) classify stress severity, (2) estimate
Depression/Anxiety/Stress scores, and (3) explain every prediction with quantified,
cited evidence — framed throughout as **decision support, not diagnosis**.

> Renamed from "MindScope" (the name of a prior planning pass whose architecture,
> design tokens, and technical approach this project follows). See `PROJECT_PLAN.md`
> for the full phase-by-phase build log and current status.

## Status

Built as a single async engineering session rather than a live hackathon clock — see
`PROJECT_PLAN.md` for the phase-by-phase checklist and what's verified vs. pending.

**All six phases (P0–P6) are code-complete and verified**, but this sandbox had **no GPU
and no dataset files** (the FER images, RAVDESS clips, and 4000-row tabular CSV were
never supplied here — `data/raw/` is empty and git-ignored, ready to receive them).
Every module — data loaders, model architectures, fusion, explainability, RAG/agent,
API — is exercised end-to-end with real forward/backward passes and real training loops
against small synthetic, schema-correct fixtures (see `tests/`), not just import checks.
**No metric numbers in this repo are claimed as real trained results** — the API flags
`is_demo_untrained_model: true` on every response until real checkpoints are trained on
real data.

To go from here to real results:
1. Drop the datasets into `data/raw/facial/`, `data/raw/speech/`, `data/raw/numerical.csv`
   (see layout below) and run `python -m src.data.validate_datasets`.
2. Train the three baselines: `python -m src.train.train_modality --config configs/{face,speech,tabular}.yaml`.
3. Train fusion: `python -m src.train.train_fusion --config configs/fusion.yaml`.
4. Run the full metric suite / ablation / fairness audit (`src/eval/`).

## Architecture

Three modality encoders → an emotion→stress bridge → gated cross-modal attention fusion
→ dual prediction heads → an explainability/trust stack → a grounded RAG report, all
sequenced by a thin agent orchestrator.

| Stage | What | Where |
|---|---|---|
| Facial encoder | EfficientNet-B0 + CBAM (fallback: 4-block CNN) | `src/models/face_cnn.py` |
| Speech encoder | Wav2Vec2-base (fallback: CNN-BiLSTM + SpecAugment) | `src/models/speech_net.py` |
| Tabular encoder | FT-Transformer + LightGBM stack (fallback: residual MLP) | `src/models/tabular_ft.py` |
| Emotion→stress bridge | Two separate modality-specific priors (facial ≠ speech), never merged into one rule | `src/data/emotion_stress_map.py` |
| Fusion | Gated cross-modal attention + modality dropout | `src/models/fusion.py` |
| Heads | 4-class (MC-dropout uncertainty) + 3-score regression + consistency term | `src/models/heads.py`, `src/train/losses.py` |
| Explainability | Grad-CAM/Score-CAM, Integrated Gradients, SHAP + attention, Masked-Distress Index, DiCE counterfactuals, conformal prediction sets | `src/explain/` |
| RAG + agent | FAISS/TF-IDF retrieval over a placeholder clinical KB, cited report generation, LangGraph orchestrator | `src/reasoning/` |
| API | FastAPI: assess / explain / counterfactual / report / follow-up | `src/api/` |
| Frontend | React + Vite + TS + Tailwind, converted 1:1 from the approved Stitch design | `frontend/` |

The three datasets are **not row-paired** — only the 4000-row tabular table carries real
ground truth. Face and voice are trained on their own native emotion labels, projected
onto the shared 4-tier severity axis, and fused onto the labelled tabular rows via
weak-pairing (matched-emotion sampling, `src/data/loaders.py:FusionPairDataset`) — never
a fabricated face+voice+row identity match. A tabular-only fallback metric is always
reported alongside full fusion so results stay honest.

## Responsible AI (non-negotiable, not polish)

- Every prediction-carrying API response and generated report states this is
  decision-support information, not a diagnosis.
- Low-confidence predictions are flagged for human review (`uncertainty.defer`), via
  MC-dropout confidence and (once real calibration data exists) conformal prediction sets.
- Crisis/helpline resources are force-attached whenever severe-distress indicators or a
  high Masked-Distress Index co-occur — never left to retrieval-ranking chance.
- The RAG report layer mechanically rejects any LLM output containing a citation that
  doesn't resolve to an actually-retrieved source, falling back to a templated cached
  report rather than surfacing an unsourced claim.
- `data/knowledge_base/` is placeholder content (see its `README.md`) and must be
  replaced with a licensed, clinically-reviewed knowledge base before any real-world use.

## Repository layout

```
cortexai/
├── docs/                 # source-of-truth requirements, metrics, dataset schema, Stitch design
├── data/                 # raw/ (git-ignored, empty — see Status), knowledge_base/ (placeholder)
├── src/
│   ├── data/             # loaders, schemas, emotion_stress_map, augmentation
│   ├── models/           # face_cnn, speech_net, tabular_ft, fusion, heads
│   ├── train/            # train_modality, train_fusion, losses
│   ├── explain/          # gradcam, ig_audio, shap_tab, masked_distress, counterfactual, conformal
│   ├── reasoning/        # retriever, rag_report, agent_graph
│   ├── eval/              # metrics, ablation, fairness_audit
│   └── api/               # main (FastAPI), schemas, inference
├── frontend/              # React + Vite + TS + Tailwind
├── configs/                # face.yaml, speech.yaml, tabular.yaml, fusion.yaml
├── tests/                   # pytest — 119 tests, all against synthetic fixtures
└── PROJECT_PLAN.md           # phase-by-phase build log
```

## Running it

```bash
# Backend
pip install -r requirements.txt
uvicorn src.api.main:app --reload
# -> http://localhost:8000/health, /docs

# Frontend
cd frontend && npm install && npm run dev
# -> http://localhost:5173

# Tests
pytest
```

`docker-compose.yml` runs both together. Copy `.env.example` to `.env` and set
`ANTHROPIC_API_KEY` for live RAG report generation — without it, the API serves clearly
labeled cached reports (`cached: true`) instead of a live LLM call.

## Metrics

Exact suite from `docs/Metrics_Used.docx`, implemented in `src/eval/metrics.py`:

- **Classification:** Accuracy, Precision, Recall, F1, Macro-F1, Weighted-F1, ROC-AUC,
  Confusion Matrix — headline metric **Macro-F1** (classes are imbalanced).
- **Regression** (per target — Depression/Anxiety/Stress): MAE, MSE, RMSE, R², Explained
  Variance — headline metric **RMSE**.
