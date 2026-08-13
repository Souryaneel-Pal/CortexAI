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

All six phases (P0–P6) are code-complete, and the three modality encoders plus the
fusion stack have been **trained on the real datasets** (Apple-silicon MPS). The API
serves those checkpoints, and every response carries `is_demo_untrained_model: false`
once they are present.

Reproduce end to end:

```bash
python -m src.data.validate_datasets                                  # verify counts/schema
python -m src.train.train_modality --config configs/tabular.yaml
python -m src.train.train_modality --config configs/face.yaml
python -m src.train.train_modality --config configs/speech.yaml
python -m src.train.train_fusion   --config configs/fusion.yaml
python -m src.eval.run_evaluation                                     # metric suite + ablation + fairness
```

### Measured results

Held-out validation splits; augmentation and SMOTE are applied to the training split
only. Headline metrics per `docs/Metrics_Used.docx`: **macro-F1** for classification,
**RMSE** for regression.

| Encoder | Task | Split | Accuracy | Macro-F1 |
|---|---|---|---|---|
| **Face** — EfficientNet-B0 + CBAM | 7-way FER emotion | stratified 15% | 0.615 | **0.604** |
| **Speech** — Wav2Vec2-base | 8-way RAVDESS emotion | 4 held-out **actors** (16/17/19/20) | 0.708 | **0.698** |
| **Tabular** — FT-Transformer | 4-way `Mental_Health_Status` | stratified 15% | 0.283 | **0.228** |

Fairness audit (RAVDESS actor gender, held-out actors only):
female macro-F1 **0.714** (n=120) vs male **0.680** (n=120) — a gap of **0.034**.

**Proof-of-fusion ablation** (`python -m src.eval.run_evaluation`) — one trained fusion
model, one validation split, each arm produced by masking modalities through the fusion
gate rather than training four separate models:

| Source | Macro-F1 | Weighted-F1 | RMSE (mean) |
|---|---|---|---|
| face_only | 0.251 | 0.326 | 11.015 |
| speech_only | 0.251 | 0.294 | 11.900 |
| tabular_only | 0.118 | 0.147 | 9.433 |
| **fusion** | **0.257** | 0.301 | 11.114 |

`fusion_beats_every_modality()` → **False**. Fusion has the best macro-F1 but a *worse*
RMSE than tabular-only, so the headline "fusion wins" claim is not supported here and
the code reports that rather than asserting it.

Two honest caveats on these fusion numbers:

- **Validation pairing is label-independent.** Matched-emotion weak pairing keys the
  sampled face/voice on the row's ground-truth label, so media paired that way encodes
  the answer. That is a defensible *training* prior but would make a validation score
  meaningless, so `FusionPairDataset(pair_by_label=False)` is used for val — media drawn
  at random, exactly like a real session (`tests/test_fusion_pairing.py` pins this).
- **The learned gate collapsed onto face** (face 0.93 / speech 0.06 / tabular 0.01).
  That is the fingerprint of the training-time pairing: during training the face *is* the
  most predictive input, so the gate learned to trust it. Combined with the fact that the
  tabular anchor is noise, the modality-contribution meter should be read as "what the
  gate learned on this data", not as a clinical statement about which channel matters.

### The finding that matters most: the tabular targets are not learnable

Face and speech emotion recognition work — those numbers are real signal. **The 18
tabular features carry essentially no information about the targets they are supposed
to predict**, and the whole point of the honesty scaffolding in this repo is that this
shows up rather than gets papered over:

- The largest absolute correlation between **any** of the 18 features and **any** of the
  three scores is **0.046** (`Head_Motion_Index` vs `Stress_Score`).
- Gradient-boosted trees reach macro-F1 **0.219** on the 4-class target; a
  **stratified random guess scores 0.270**. The models are not beating chance.
- R² is **negative** for all three regression targets (≈ −0.02), i.e. predicting the
  training mean beats every model tried.
- The FT-Transformer lands at macro-F1 0.228 / RMSE 10.53, consistent with the above.

The *targets* are internally coherent — mean Depression rises 9.95 → 18.72 → 25.61 →
30.66 across Healthy → Severe — so class and scores agree with each other. Nothing
predicts either from the features. Treat every stress-class and severity-score number
this system produces as **not clinically meaningful on this dataset**; the pipeline,
explainability, and trust layers around them are what is demonstrable here.

Two consequences worth stating plainly:

1. **Fusion cannot rescue this.** Face/voice enter as auxiliary evidence anchored on
   the labelled tabular rows; if the anchor is noise, fusion has nothing to sharpen.
2. **The "fusion beats every modality" slide is not claimed.**
   `src/eval/ablation.py:fusion_beats_every_modality()` returns a boolean computed from
   the measured numbers rather than an assertion, and `src/eval/run_evaluation.py`
   prints whatever it actually is.

## Dataset Access

The three source datasets are distributed by the Hack4Health organisers and are **not
committed to this repository** — `data/raw/` is git-ignored (see `.gitignore`) so no
raw media or participant-level CSV ever enters version control.

**Official dataset link:**
<https://drive.google.com/drive/folders/1R9ka23jnBsNDyPh6l03f2Zv3d7gyk3tR?usp=sharing>

Download the folder and arrange it under `data/raw/` exactly as below, then run
`python -m src.data.validate_datasets` to verify counts and schema against
`docs/Dataset_Description.docx`:

```
data/raw/
├── Extracted_images/          # FER-style 48x48 grayscale faces, one folder per emotion
│   ├── Angry/  (3,995)   Disgust/ (436)    Fear/     (4,097)
│   ├── Happy/  (7,215)   Neutral/ (4,965)  Sad/      (4,830)
│   └── Surprise/ (3,171)                              # 28,709 total
├── Audios/                    # RAVDESS speech clips, 7-part filenames
│   └── Actor_01/ … Actor_24/  # 1,440 unique .wav (60 per actor)
└── mental_health_multimodal.csv   # 4,000 rows x 18 features + 4 targets
```

Two notes on the archive as distributed:

- `Audios/` also ships a nested `audio_speech_actors_01-24/` directory that is a
  **byte-identical duplicate** of the 24 `Actor_XX/` folders (2,880 files on disk,
  1,440 unique clips — verified by md5). `SpeechEmotionDataset` de-duplicates by
  RAVDESS filename automatically, so either layout loads exactly 1,440 clips.
- The CSV's target column is `Mental_Health_Status`; scores are `Depression_Score`
  (0–34), `Anxiety_Score` (0–24), `Stress_Score` (0–39).

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
| RAG + agent | Local Ollama (`nomic-embed-text` + `llama3.1`) over a placeholder clinical KB, cited report generation, LangGraph orchestrator | `src/reasoning/` |
| API | FastAPI: predict (alias assess) / explain / counterfactual / report / follow-up / health | `src/api/` |
| Frontend | React + Vite + TS + Tailwind, converted 1:1 from the approved Stitch design, wired to the API | `frontend/` |

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
├── data/                 # raw/ (git-ignored — see Dataset Access), knowledge_base/ (placeholder)
├── src/
│   ├── data/             # loaders, schemas, emotion_stress_map, augmentation
│   ├── models/           # face_cnn, speech_net, tabular_ft, fusion, heads
│   ├── train/            # train_modality, train_fusion, losses
│   ├── explain/          # gradcam, ig_audio, shap_tab, masked_distress, counterfactual, conformal
│   ├── reasoning/        # ollama_config, retriever, rag_report, agent_graph
│   ├── eval/              # metrics, ablation, fairness_audit, run_evaluation
│   └── api/               # main (FastAPI), schemas, inference
├── frontend/              # React + Vite + TS + Tailwind
├── configs/                # face.yaml, speech.yaml, tabular.yaml, fusion.yaml
├── tests/                   # pytest — 151 tests against synthetic fixtures + the real API
└── PROJECT_PLAN.md           # phase-by-phase build log
```

## Running it

```bash
# Backend
pip install -r requirements.txt
uvicorn src.api.main:app --reload
# -> http://127.0.0.1:8000/health, /docs
# /health reports is_demo_untrained_model: false once artifacts/checkpoints/ is populated.

# Frontend
cd frontend && npm install && npm run dev
# -> http://localhost:5173
# Point it elsewhere with VITE_API_BASE_URL; defaults to http://127.0.0.1:8000

# Tests
pytest        # 151 tests
```

### Local reasoning stack (Ollama)

The RAG layer runs entirely on-device. This system already handles face, voice, and
physiological data, and the report prompt embeds both the prediction and the retrieved
clinical text — so keeping generation local avoids sending any of it to a third party.

```bash
# Install Ollama (https://ollama.com), then:
ollama pull llama3.1          # writes the cited clinical narratives
ollama pull nomic-embed-text  # embeds the clinical KB for retrieval
```

| Role | Model | Where |
|---|---|---|
| Narrative + follow-up answers | `llama3.1` via `ChatOllama` | `src/reasoning/rag_report.py` |
| KB embeddings → vector index | `nomic-embed-text` via `OllamaEmbeddings` | `src/reasoning/retriever.py` |
| Health probe / config | `OLLAMA_BASE_URL`, `CORTEXAI_OLLAMA_LLM_MODEL`, `CORTEXAI_OLLAMA_EMBED_MODEL` | `src/reasoning/ollama_config.py` |

`GET /health` reports the live state of this stack, so the UI can warn *before* an
assessment that the narrative will be templated rather than surprising the clinician
afterwards:

```json
{"ollama_reachable": true, "llm_model": "llama3.1", "llm_available": true,
 "embedding_model": "nomic-embed-text", "embedding_available": true,
 "retrieval_backend": "ollama", "vector_index": "numpy.exact_inner_product"}
```

**Nothing hard-fails when Ollama is down.** Each degradation is labelled, never silent —
a fallback report carries `cached: true`, `generator: "template"`, and a
`fallback_reason` that the Reports page displays:

| Situation | Retrieval | Report |
|---|---|---|
| Ollama running, models pulled | `nomic-embed-text` + vector index | `ollama:llama3.1`, cited |
| Ollama up, model not pulled | falls back | templated, reason names `ollama pull <model>` |
| Ollama unreachable | sentence-transformers → TF-IDF | templated, reason names `ollama serve` |
| Model returns bad output | — | templated, reason names what was rejected |

A generated narrative is **rejected and replaced** (not patched) if it cites a source
that wasn't retrieved, cites *nothing at all*, or drops the decision-support framing.
That middle case matters more than it looks: `validate_citations` alone is trivially
satisfied by a narrative with zero citations, so an entirely unsourced report would
otherwise pass as "grounded" — observed in practice with `llama3.1`. See
`check_generated_narrative`.

**Vector index note.** `faiss.IndexFlatIP` is used where FAISS can load safely. In the
API process it cannot: faiss-cpu and PyTorch each link their own OpenMP runtime, and
loading both aborts the interpreter (`OMP: Error #15`) in either import order — an
abort, not a catchable exception. There, an exact NumPy inner-product index runs
instead. `IndexFlatIP` *is* brute-force exact search, so this is a numerically identical
substitute, not an approximation — `tests/test_ollama_reasoning.py` pins that they
return identical neighbours and scores. `KMP_DUPLICATE_LIB_OK=TRUE` does suppress the
abort, but the same runs emit overflow/invalid warnings from unrelated matmuls, which is
exactly the "silently produce incorrect results" mode the OpenMP docs warn about — not a
trade worth making here.

### What the UI shows from a live assessment

`New Assessment` uploads a face image and/or an audio clip, maps the form onto the
18-feature vector (`frontend/src/lib/featureMapping.ts` — unfilled columns take the
training-set median, never zero), and POSTs `/assess`. `Explainable Insights` then
renders, from the model's own output rather than sample data:

- the predicted class with MC-dropout confidence and the three severity scores,
- **signed SHAP** bars over the 18 features (direction, not just magnitude),
- the **modality-contribution** donut from the fusion gate's learned weights,
- the **Grad-CAM overlay** rendered server-side, shown beside the submitted face,
- **Integrated-Gradients** frame importance across the waveform,
- the **Masked-Distress Index** with its face-calm / voice-arousal / physio-arousal
  breakdown and which channel drove the contradiction.

`Clinical Report` pulls the RAG-grounded narrative and its citations from `/report`.
Pages with no live session keep rendering illustrative sample data behind a
`SampleDataBadge`; live and sample values are never blended in the same view.

`docker-compose.yml` runs both together. Report generation uses the local Ollama stack
by default (see above) — no API key needed. Setting `ANTHROPIC_API_KEY` in `.env` enables
the optional hosted-Claude generator instead; with neither available the API serves
clearly labelled templated reports (`cached: true` plus a `fallback_reason`).

## Metrics

Exact suite from `docs/Metrics_Used.docx`, implemented in `src/eval/metrics.py`:

- **Classification:** Accuracy, Precision, Recall, F1, Macro-F1, Weighted-F1, ROC-AUC,
  Confusion Matrix — headline metric **Macro-F1** (classes are imbalanced).
- **Regression** (per target — Depression/Anxiety/Stress): MAE, MSE, RMSE, R², Explained
  Variance — headline metric **RMSE**.
