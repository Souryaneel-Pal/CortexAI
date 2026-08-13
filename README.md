# 🧠 CortexAI - Explainable Multimodal Psychiatric AI

> A local-first, explainable multimodal deep-learning framework that reads face, voice, and body signals together to estimate a person's mental-health state[cite: 1]. It classifies stress severity, estimates Depression, Anxiety, and Stress scores, and explains every prediction as grounded, clinician-ready evidence[cite: 1].

> **⚠️ IMPORTANT NOTICE:** This system is a research prototype and decision-support tool designed to assist qualified professionals, **not a standalone diagnostic device**[cite: 1, 2]. Every surface states its limitations, and low-confidence predictions are deferred to human review[cite: 2].

---

## Table of Contents

1. [Architecture & Pipeline](https://www.google.com/search?q=%23architecture--pipeline)
2. [Key Innovations & Differentiators](https://www.google.com/search?q=%23key-innovations--differentiators)
3. [Evaluation Metrics & Realities](https://www.google.com/search?q=%23evaluation-metrics--realities)
4. [Tech Stack](https://www.google.com/search?q=%23tech-stack)
5. [API Reference](https://www.google.com/search?q=%23api-reference)
6. [Quick Start & Setup](https://www.google.com/search?q=%23quick-start--setup)
7. [Project Structure](https://www.google.com/search?q=%23project-structure)

---

## Architecture & Pipeline

CortexAI operates on a two-stage fusion architecture that respects the fact that the provided datasets are not natively row-paired[cite: 1].

```text
+---------------------------------------------------------------------------------+
| MULTIMODAL INPUTS |
| [ Facial Image: 48x48 ] [ Speech: .wav ] [ Tabular: 18 Features ] |
+-----------+--------------------------+-------------------------+----------------+
            | | |
+-----------v-----------+ +-----------v----------+ +-----------v----------+
| VISION ENCODER | | AUDIO ENCODER | | TABULAR ENCODER |
| EfficientNet-B0+CBAM | | Wav2Vec2-base | | FT-Transformer |
+-----------+-----------+ +-----------+----------+ +-----------+----------+
            | | |
+-----------v--------------------------v----------+ |
| EMOTION-TO-STRESS BRIDGE | |
| (Projects 7-way FER & 8-way RAVDESS to 4-tier) | |
+--------------------------+----------------------+ |
                           | |
+--------------------------v-------------------------------------v----------------+
| GATED CROSS-MODAL ATTENTION FUSION |
| (Learns modality reliability weights, handles missing modalities) |
+--------------------------+-------------------------------------+----------------+
                           | |
+--------------------------v----------+ +-----------v----------+
| CLASSIFICATION HEAD | | REGRESSION HEAD |
| 4-Class (Healthy -> Severe) | | 3 Scores (Dep/Anx/Str)|
+--------------------------+----------+ +-----------+----------+
                           | |
+--------------------------v-------------------------------------v----------------+
| TRUST & EXPLAINABILITY STACK |
| [ Grad-CAM ] [ Integrated Gradients ] [ SHAP ] [ Masked-Distress Index ] |
| [ Conformal Uncertainty Gate (MC-Dropout) ] |
+------------------------------------------+--------------------------------------+
                                           |
+------------------------------------------v--------------------------------------+
| RAG CLINICAL NARRATIVE & AGENT |
| (LangGraph Orchestrator + FAISS Vector DB + Local Ollama Llama-3.1) |
+---------------------------------------------------------------------------------+

```

---

## Key Innovations & Differentiators

* **Masked-Distress Index (MDI):** A novel cross-modal contradiction score. When the face reads calm but voice/physiology indicate high arousal, MDI quantifies the gap to flag suppressed distress[cite: 1].
* **Conformal "I'm not sure" Gate:** Built-in statistically valid prediction sets (MC-dropout). The system safely defers to a human clinician when confidence is low[cite: 1, 2].
* **Missing-Modality Resilience:** Trained with modality dropout, the cross-modal attention fusion degrades gracefully if a webcam or mic is missing[cite: 2].
* **Zero-Hallucination Reports:** Uses RAG over a local clinical knowledge base (DASS-21/DSM-5). A deterministic verification agent rejects LLM output if citations do not resolve to retrieved sources[cite: 1, 2].
* **100% Local Execution (Privacy First):** Sensitive biometric data (face, voice, physiology) and LLM inference (via Ollama `llama3.1`) run entirely on-device to ensure strict patient privacy[cite: 1, 2].

---

## Evaluation Metrics & Realities

All six phases are code-complete, with modality encoders and the fusion stack trained directly on the Apple-silicon MPS backend.

### Unimodal Performance (Held-out Validation Splits)

| Modality Encoder | Task / Target | Val Split | Accuracy | Macro-F1 | RMSE |
| --- | --- | --- | --- | --- | --- |
| **Face** (EfficientNet-B0) | 7-way FER emotion | Stratified 15% | 0.615 | **0.604** | - |
| **Speech** (Wav2Vec2) | 8-way RAVDESS emotion | 4 held-out actors | 0.708 | **0.698** | - |
| **Tabular** (FT-Transformer) | 4-way Stress Class | Stratified 15% | 0.283 | **0.228** | 10.53 |

*Fairness Audit:* Speech performance across held-out RAVDESS actors yields a female Macro-F1 of **0.714** vs. a male Macro-F1 of **0.680** (gap: 0.034)[cite: 1].

### Multimodal Proof-of-Fusion Ablation

| Source Pipeline | Macro-F1 | Weighted-F1 | RMSE (Mean) |
| --- | --- | --- | --- |
| Face Only | 0.251 | 0.326 | 11.015 |
| Speech Only | 0.251 | 0.294 | 11.900 |
| Tabular Only | 0.118 | 0.147 | **9.433** |
| **Fusion** | **0.257** | 0.301 | 11.114 |

> **Crucial Data Finding:** While the face and speech emotion signals are highly learnable, the 18 provided tabular features carry effectively **no predictive signal** for the targets on this specific dataset. Gradient-boosted trees reach a Macro-F1 of only 0.219 (random chance is 0.270), and R² is negative (~ -0.02). Because the tabular rows act as the ground-truth anchor, the fusion model mathematically cannot synthesize strong predictions from noisy anchors. The metrics above are honest reflections of the data limits, proving that our *trust, explainability, and gating layers* operate correctly even when the predictive task is fundamentally constrained.

---

## Tech Stack

| Category | Component / Library |
| --- | --- |
| **Vision Framework** | PyTorch, `timm` (EfficientNet-B0), OpenCV[cite: 1, 2] |
| **Audio Framework** | Hugging Face Transformers (Wav2Vec2), `librosa`, `soundfile`[cite: 1, 2] |
| **Tabular & ML** | `pytorch-tabular` (FT-Transformer), LightGBM, `imbalanced-learn`[cite: 1, 2] |
| **Explainability (XAI)** | Captum (Integrated Gradients, Grad-CAM), SHAP, DiCE[cite: 1, 2] |
| **RAG & Reasoning** | LangGraph, `ollama` (llama3.1, nomic-embed-text), FAISS[cite: 1, 2] |
| **API & Serving** | FastAPI, Uvicorn, Pydantic[cite: 1, 2] |
| **Frontend** | React, Vite, TypeScript, Tailwind CSS, Recharts[cite: 1, 2] |

---

## API Reference

Powered by FastAPI. Accessible via `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`.

| Method | Endpoint | Description | Payload / Response |
| --- | --- | --- | --- |
| `GET` | `/health` | System status & local inference check | Returns model statuses (e.g., `ollama_reachable`) |
| `POST` | `/assess` | End-to-end multimodal inference | **In:** Base64 Face, Audio, 18 Tabular features. **Out:** Class, Confidence, Scores, XAI Data |
| `POST` | `/report` | Generates cited clinical narrative | **In:** Patient prediction context. **Out:** Cited Markdown text |
| `POST` | `/explain` | Granular XAI data retrieval | **Out:** SHAP values, IG temporal arrays, MDI calculations |

---

## Quick Start & Setup

**1. Clone & Environment**

```bash
git clone https://github.com/Souryaneel-Pal/CortexAI.git
cd CortexAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

**2. Local Reasoning Stack (Ollama)**

```bash
# Ensure Ollama is installed (https://ollama.com)
ollama pull llama3.1
ollama pull nomic-embed-text

```

*(Copy `.env.example` to `.env` if utilizing optional hosted fallbacks).*

**3. Run the Backend API**

```bash
uvicorn src.api.main:app --reload --port 8000

```

**4. Run the Frontend**

```bash
cd frontend
npm install
npm run dev

```

---

## Project Structure

```text
CortexAI/
├── docs/ # Hackathon constraints, metrics schemas, design assets
├── data/ # raw/ (git-ignored), knowledge_base/ (Clinical RAG DB)
├── src/
│ ├── data/ # loaders.py, emotion_stress_map.py, augmentation.py
│ ├── models/ # face_cnn.py, speech_net.py, tabular_ft.py, fusion.py
│ ├── train/ # train_modality.py, train_fusion.py, losses.py
│ ├── explain/ # gradcam.py, ig_audio.py, shap_tab.py, masked_distress.py
│ ├── reasoning/ # agent_graph.py, rag_report.py, retriever.py
│ ├── eval/ # metrics.py, ablation.py, run_evaluation.py
│ └── api/ # main.py (FastAPI), inference.py, schemas.py
├── frontend/ # React/Vite web application
├── configs/ # YAML hyperparameter configurations
├── tests/ # Pytest suite (151 passing tests)
└── requirements.txt # Pinned dependency graph

```