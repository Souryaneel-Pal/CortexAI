# CortexAI

Explainable multimodal deep-learning system for psychiatric / mental-health screening,
built for Hack4Health. CortexAI fuses facial expression, speech emotion, and
behavioural/acoustic/physiological signals to (1) classify stress severity, (2) estimate
Depression / Anxiety / Stress scores, and (3) explain every prediction with quantified,
cited evidence — framed throughout as **decision support, not diagnosis**.

The reasoning layer runs entirely on-device via Ollama, so face, voice, physiological
data, and the generated narrative never leave the machine.

---

## Read this first: what the numbers mean

Two facts govern how every output of this system should be interpreted. Both were
measured, not assumed.

**1. Face and speech emotion recognition work.** The facial encoder reaches macro-F1
**0.604** on 7-way FER; the speech encoder reaches **0.698** on 8-way RAVDESS with an
actor-disjoint split. Those are real signal.

**2. The tabular targets are not learnable from the tabular features.** The largest
absolute correlation between any of the 18 features and any of the three severity scores
is **0.046**. Gradient-boosted trees reach macro-F1 **0.219** on the 4-class target where
a *stratified random guess* scores **0.270**, and R² is **negative** (≈ −0.02) for all
three scores — predicting the training mean beats every model tried.

So: **treat every stress-class and severity-score number this system produces as not
clinically meaningful on this dataset.** The targets are internally coherent (mean
Depression rises 9.95 → 18.72 → 25.61 → 30.66 across Healthy → Severe), but nothing
predicts them from the features. What is demonstrable here is the pipeline, the
explainability, and the trust/provenance machinery around the predictions.

This honesty is enforced in code, not just documented:

- `src/eval/ablation.py:fusion_beats_every_modality()` returns a **computed boolean**,
  not an assertion. It currently returns `False`.
- Every prediction response carries `is_demo_untrained_model`.
- Every report states which generator wrote it, and a templated fallback always carries a
  `fallback_reason`.
- Nothing fabricates history. See [No fabricated data](#no-fabricated-data).

---

## Quick start

```bash
# 1. Backend
pip install -r requirements.txt
uvicorn src.api.main:app --reload          # http://127.0.0.1:8000  (/docs for OpenAPI)

# 2. Local reasoning models (see "Local reasoning stack")
ollama pull llama3.1
ollama pull nomic-embed-text

# 3. Frontend
cd frontend && npm install && npm run dev  # http://localhost:5173

# 4. Tests
pytest                                     # 176 tests
```

### Sign-in

The app opens on a sign-in page at `/`. There is no user store — these are hardcoded
demo accounts, documented here because they are not a secret:

| Role | User ID | Password | Unlocks |
|---|---|---|---|
| **Admin** | `admin` | `admin` | Everything, including **Settings** (model tuning) |
| Clinician | `julian.vance@cortex.ai` | `password` | Everything except writing Settings |

Both roles can *read* settings; only Admin can change them.

---

## Application flow

```
  /  SignIn
  │
  ├─→ /dashboard          Cohort stats from real stored history + latest live result
  ├─→ /assessment/new     Upload face + audio, enter clinical metrics → POST /predict
  │        │
  │        └─→ /results   4-class prediction, DASS scores, Grad-CAM, SHAP, MDI,
  │                       audio Integrated Gradients, modality attribution
  │                       └─ [Generate Clinical Report] → runs Ollama RAG → /reports
  │
  ├─→ /reports            The cited narrative + its sources and provenance
  ├─→ /analytics          Cohort analytics from real stored history
  └─→ /settings           Admin only: uncertainty gate, MDI sensitivity, modality overrides
```

Routes other than `/` are wrapped in `ProtectedRoute` and redirect to sign-in without a
token. The token is a signed JWT held in `sessionStorage` and sent as
`Authorization: Bearer …` by `frontend/src/lib/api.ts`.

---

## Architecture

Three modality encoders → an emotion→stress bridge → gated cross-modal attention fusion
→ dual prediction heads → an explainability/trust stack → a grounded RAG report,
sequenced by a thin agent orchestrator.

| Stage | What | Where |
|---|---|---|
| Facial encoder | EfficientNet-B0 + CBAM (fallback: 4-block CNN) | `src/models/face_cnn.py` |
| Speech encoder | Wav2Vec2-base (fallback: CNN-BiLSTM + SpecAugment) | `src/models/speech_net.py` |
| Tabular encoder | FT-Transformer + LightGBM stack (fallback: residual MLP) | `src/models/tabular_ft.py` |
| Emotion→stress bridge | Two modality-specific priors (facial ≠ speech), never merged | `src/data/emotion_stress_map.py` |
| Fusion | Gated cross-modal attention + modality dropout | `src/models/fusion.py` |
| Heads | 4-class (MC-dropout uncertainty) + 3-score regression + consistency term | `src/models/heads.py`, `src/train/losses.py` |
| Explainability | Grad-CAM/Score-CAM, Integrated Gradients, SHAP + attention, Masked-Distress Index, DiCE counterfactuals, conformal sets | `src/explain/` |
| RAG + agent | Local Ollama (`nomic-embed-text` + `llama3.1`), cited reports, LangGraph orchestrator | `src/reasoning/` |
| Persistence | SQLAlchemy + SQLite: assessments, reports, settings | `src/api/database.py` |
| API | FastAPI, 13 routes | `src/api/main.py` |
| Frontend | React + Vite + TS + Tailwind | `frontend/` |

### API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | public | Liveness + checkpoint state + Ollama status |
| POST | `/api/auth/login` | public | Issue a JWT |
| GET | `/api/settings` | any user | Read inference settings |
| PUT | `/api/settings` | **Admin** | Update inference settings |
| GET | `/api/dashboard` | any user | Cohort metrics from stored history |
| GET | `/api/analytics` | any user | Analytics from stored history |
| POST | `/predict` (alias `/assess`) | any user | Run an assessment; persists it |
| GET | `/explain/{id}` | any user | SHAP, Grad-CAM, audio IG, MDI |
| GET | `/counterfactual/{id}` | any user | Actionable single-feature counterfactual |
| GET | `/report/{id}` | any user | Generate (or serve stored) cited narrative |
| POST | `/follow-up` | any user | Grounded answer to a follow-up question |

---

## Admin model settings

`/settings` (Admin only) writes to the `settings` table, and inference reads it **per
request** — so a change takes effect on the next assessment with no restart.

| Control | Effect | Read at |
|---|---|---|
| Uncertainty gate threshold | Defer to a human when confidence falls below it | `inference.py:_uncertainty` |
| MDI sensitivity | Threshold at which cross-modal contradiction is flagged, and at which crisis resources are force-attached | `inference.py:_explain`, `main.py` report path |
| Ignore face / speech / tabular | Masks that modality through the fusion gate's `modality_mask` | `inference.py:_preprocess` |

The modality overrides reuse the same `modality_mask` the model was *trained* to degrade
into via modality dropout, so ignoring a channel is a supported inference mode rather
than a hack.

---

## Local reasoning stack (Ollama)

```bash
ollama pull llama3.1          # cited clinical narratives + follow-up answers
ollama pull nomic-embed-text  # embeds the clinical KB for retrieval
```

Configure with `OLLAMA_BASE_URL`, `CORTEXAI_OLLAMA_LLM_MODEL`,
`CORTEXAI_OLLAMA_EMBED_MODEL` (`src/reasoning/ollama_config.py`).

`GET /health` reports this stack live, so the UI warns **before** an assessment that the
narrative will be templated rather than surprising the clinician afterwards.

**Nothing hard-fails when Ollama is down**, and every degradation is labelled:

| Situation | Retrieval | Report |
|---|---|---|
| Ollama up, models pulled | `nomic-embed-text` + vector index | `ollama:llama3.1`, cited |
| Ollama up, model missing | falls back | template, reason names `ollama pull <model>` |
| Ollama unreachable | sentence-transformers → TF-IDF | template, reason names `ollama serve` |
| Model output rejected | — | template, reason names what was rejected |

A generated narrative is **rejected and replaced** — not patched — if it cites a source
that wasn't retrieved, cites *nothing at all*, or drops the decision-support framing.
The middle case matters more than it looks: `validate_citations` alone is trivially
satisfied by a narrative with zero citations, so a fully unsourced report would otherwise
pass as "grounded". Observed in practice with `llama3.1`. See `check_generated_narrative`.

### Report provenance

Three orthogonal fields, because conflating them mislabels real output:

- `generator` — `ollama:llama3.1`, `anthropic:<model>`, or `template`.
- `cached` — **this is a template, not model-written.** Drives the UI's "not
  model-generated" warning.
- `from_store` — served from SQLite rather than generated this request. A stored
  `llama3.1` narrative is `from_store: true` with `cached: false`, because it is still
  model-written.

**Vector index note.** `faiss.IndexFlatIP` is used where FAISS can load safely. In the
API process it cannot: faiss-cpu and PyTorch each link their own OpenMP runtime, and
loading both aborts the interpreter (`OMP: Error #15`) in *either* import order — an
abort, not a catchable exception. There an exact NumPy inner-product index runs instead.
`IndexFlatIP` *is* brute-force exact search, so this is a numerically identical
substitute, pinned by a test that compares against real FAISS in a subprocess.
`KMP_DUPLICATE_LIB_OK=TRUE` suppresses the abort, but the same runs emit
overflow/invalid warnings from unrelated matmuls — the "silently produce incorrect
results" mode the OpenMP docs warn about. Not a trade worth making here.

---

## No fabricated data

The dashboard and analytics pages render **only** values computed from stored
assessments. Where there is no data, the UI says so.

An earlier build seeded the database with 15 invented "historical" assessments — fake
patient IDs, fake 2023 dates, invented SHAP and MDI values, and hand-written narratives
stored with `report_generator="ollama:llama3.1"`. That was strictly worse than the
frontend mock data it replaced: mock data sat behind a `SampleDataBadge` and was visibly
not real, whereas the same fabrications *inside the assessments table* are
indistinguishable from genuine output, are served through `/api/dashboard` as real
history, and presented hand-written text as a model-authored clinical narrative.

That seeder is gone (only configuration defaults are seeded), and the rows it wrote were
purged. If you have an older database:

```bash
python -m src.api.database --purge-seed          # dry run: lists what it matched
python -m src.api.database --purge-seed --apply  # delete them
```

Genuine assessments are untouched — the match requires both a known seed patient ID and a
2023 timestamp.

Two pages still show clearly-badged sample data where a single session genuinely cannot
fill a cohort view (`Results` comparison panels, the `Reports` letterhead). They are
marked with `SampleDataBadge` and never blended with live values in the same panel.

---

## Measured results

Held-out validation; augmentation and SMOTE are applied to the **training split only**.
Headline metrics per `docs/Metrics_Used.docx`: macro-F1 for classification, RMSE for
regression.

| Encoder | Task | Split | Accuracy | Macro-F1 |
|---|---|---|---|---|
| **Face** — EfficientNet-B0 + CBAM | 7-way FER emotion | stratified 15% | 0.615 | **0.604** |
| **Speech** — Wav2Vec2-base | 8-way RAVDESS emotion | 4 held-out **actors** | 0.708 | **0.698** |
| **Tabular** — FT-Transformer | 4-way `Mental_Health_Status` | stratified 15% | 0.283 | **0.228** |

Fairness audit (RAVDESS actor gender, held-out actors): female macro-F1 **0.714** (n=120)
vs male **0.680** (n=120) — a gap of **0.034**.

**Proof-of-fusion ablation** — one trained model, one validation split, each arm produced
by masking modalities through the fusion gate:

| Source | Macro-F1 | Weighted-F1 | RMSE (mean) |
|---|---|---|---|
| face_only | 0.251 | 0.326 | 11.015 |
| speech_only | 0.251 | 0.294 | 11.900 |
| tabular_only | 0.118 | 0.147 | 9.433 |
| **fusion** | **0.257** | 0.301 | 11.114 |

`fusion_beats_every_modality()` → **False**. Fusion has the best macro-F1 but a *worse*
RMSE than tabular-only, so the "fusion wins" claim is not supported and the code reports
that rather than asserting it.

Two caveats on the fusion numbers:

- **Validation pairing is label-independent.** Matched-emotion weak pairing keys the
  sampled face/voice on the row's ground-truth label, so media paired that way encodes
  the answer. That is a defensible *training* prior but would make a validation score
  meaningless, so `FusionPairDataset(pair_by_label=False)` is used for val.
- **The learned gate collapsed onto face** (0.93 / 0.06 / 0.01) — the fingerprint of that
  training-time pairing. Read the modality meter as "what the gate learned on this data",
  not as a clinical claim about which channel matters.

Reproduce:

```bash
python -m src.data.validate_datasets
python -m src.train.train_modality --config configs/tabular.yaml
python -m src.train.train_modality --config configs/face.yaml
python -m src.train.train_modality --config configs/speech.yaml
python -m src.train.train_fusion   --config configs/fusion.yaml
python -m src.eval.run_evaluation      # metric suite + ablation + fairness
```

---

## Dataset access

Distributed by the organisers and **not committed** — `data/raw/` is git-ignored, so no
raw media or participant-level CSV enters version control.

<https://drive.google.com/drive/folders/1R9ka23jnBsNDyPh6l03f2Zv3d7gyk3tR?usp=sharing>

```
data/raw/
├── Extracted_images/          # FER-style 48x48 grayscale, one folder per emotion
│   ├── Angry/  (3,995)   Disgust/ (436)    Fear/     (4,097)
│   ├── Happy/  (7,215)   Neutral/ (4,965)  Sad/      (4,830)
│   └── Surprise/ (3,171)                              # 28,709 total
├── Audios/                    # RAVDESS, 7-part filenames
│   └── Actor_01/ … Actor_24/  # 1,440 unique .wav (60 per actor)
└── mental_health_multimodal.csv   # 4,000 rows × 18 features + 4 targets
```

Verify with `python -m src.data.validate_datasets`. Two quirks of the archive:

- `Audios/` also ships `audio_speech_actors_01-24/`, a **byte-identical duplicate** of
  the 24 `Actor_XX/` folders (2,880 files on disk, 1,440 unique — verified by md5).
  `SpeechEmotionDataset` de-duplicates by filename, so either layout loads 1,440 clips.
- Speech **must** be split by actor, not by clip: 24 actors speak the same two sentences,
  so a random clip-level split measures speaker memorisation.

---

## Responsible AI

- Every prediction-carrying response and every generated report states this is
  decision-support information, not a diagnosis.
- Low-confidence predictions are flagged for human review (`uncertainty.defer`) via
  MC-dropout confidence against the Admin-configurable threshold.
- Crisis/helpline resources are **force-attached** whenever severe distress or a high
  Masked-Distress Index co-occur — never left to retrieval-ranking chance.
- The RAG layer mechanically rejects any narrative with an unresolvable citation, no
  citation, or missing framing, falling back to a templated report.
- `data/knowledge_base/` is **placeholder** content (see its `README.md`) and must be
  replaced with a licensed, clinically-reviewed KB before real-world use.
- The Masked-Distress Index formula is CortexAI's own construction and is
  **clinically unvalidated**.

### Security posture (hackathon build)

Not production-ready, and specifically:

- Credentials are hardcoded demo accounts; there is no user store, registration, or
  password reset.
- `CORTEXAI_JWT_SECRET` defaults to a development value. **Set it** before any real
  deployment — with the default, anyone holding the source can mint a valid Admin token.
- CORS is `allow_origins=["*"]`.
- The session store is an in-process dict; assessments persist to SQLite but live session
  state does not survive a restart or scale past one worker.
- Uploaded face/audio base64 is persisted to the local database.

---

## Repository layout

```
cortexai/
├── docs/                  # requirements, metrics, dataset schema, approved design
├── data/                  # raw/ (git-ignored), knowledge_base/ (placeholder), cortexai.db
├── src/
│   ├── data/              # loaders, schemas, emotion_stress_map, augmentation
│   ├── models/            # face_cnn, speech_net, tabular_ft, fusion, heads
│   ├── train/             # train_modality, train_fusion, losses
│   ├── explain/           # gradcam, ig_audio, shap_tab, masked_distress, counterfactual, conformal
│   ├── reasoning/         # ollama_config, retriever, rag_report, agent_graph
│   ├── eval/              # metrics, ablation, fairness_audit, run_evaluation
│   └── api/               # main (FastAPI + auth), schemas, inference, database, dashboard_metrics
├── frontend/src/
│   ├── pages/             # SignIn, Dashboard, NewAssessment, Results, ClinicalReport,
│   │                      # PopulationAnalytics, Settings
│   ├── lib/               # api, assessmentContext, sessionStore, mockData, chartColors
│   └── components/        # layout/ + ui/
├── configs/               # face.yaml, speech.yaml, tabular.yaml, fusion.yaml
├── tests/                 # pytest — 176 tests
└── PROJECT_PLAN.md
```

## Metrics implemented

Exact suite from `docs/Metrics_Used.docx`, in `src/eval/metrics.py`:

- **Classification:** Accuracy, Precision, Recall, F1, Macro-F1, Weighted-F1, ROC-AUC,
  Confusion Matrix — headline **Macro-F1** (classes are imbalanced).
- **Regression** (per target): MAE, MSE, RMSE, R², Explained Variance — headline
  **RMSE**.

## Development

```bash
pytest                              # 176 tests
ruff check src tests                # Python lint
cd frontend && npx tsc -b           # TypeScript typecheck
cd frontend && npx oxlint .         # JS/TS lint
cd frontend && npm run build        # production build
```

`docker-compose.yml` runs backend + frontend together. Report generation uses local
Ollama by default — no API key needed. Setting `ANTHROPIC_API_KEY` enables the optional
hosted-Claude generator instead; with neither, the API serves clearly-labelled templated
reports.
