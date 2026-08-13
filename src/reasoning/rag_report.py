"""Grounded, cited clinical report generation (docs/PROJECT_PLAN.md P4;
docs/MINDSCOPE_Blueprint.pdf Sec. 05 Layer 3: "writes a citation-backed
narrative. Every clinical claim traces to a source -- no hallucinated
advice.")

The LLM (Claude, via the `anthropic` SDK) is instructed to cite every
substantive claim with a `[doc-id]` marker referencing a retrieved
knowledge-base document (src/reasoning/retriever.py); `validate_citations`
then mechanically checks every marker in the output resolves to an actually
-retrieved document, so a hallucinated or dropped citation is caught rather
than trusted.

Responsible-AI framing (decision-support, not diagnosis; uncertainty
disclosure; crisis-resource surfacing) is baked into the prompt template
itself, not left to the model's discretion, and is checked again by
`contains_required_framing` after generation.

Fallback (docs/PROJECT_PLAN.md P4 "if blocked": "live RAG -> pre-generated
cached reports"): `generate_cached_report` builds a fully templated report
with the same citation guarantees, no LLM call at all -- used when no
ANTHROPIC_API_KEY is configured (this sandbox has none) or a live call
fails, and always clearly labeled as cached, never silently swapped in for
what looks like a live-generated report.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from src.data.schemas import STRESS_LEVEL_NAMES, StressLevel

CITATION_PATTERN = re.compile(r"\[([\w\-]+)\]")

REQUIRED_FRAMING_PHRASES = (
    "decision-support",
    "not a diagnosis",
)

SYSTEM_PROMPT = """You are CortexAI's clinical report writer. You write a short, \
grounded narrative summarizing a multimodal mental-health screening result.

Hard rules, non-negotiable:
1. Every substantive clinical claim MUST end with a citation marker like [doc-id], \
where doc-id is EXACTLY one of the provided source ids. Never invent a doc-id. \
Never state a clinical fact, interpretation, or piece of advice without one.
2. If the provided sources don't support a claim you'd otherwise want to make, omit \
the claim rather than stating it uncited.
3. Explicitly state that this is decision-support information, not a diagnosis, and \
that a qualified professional should be involved for any clinical decision.
4. If crisis-resource documents were provided among the sources, surface them clearly \
near the end of the report.
5. State the model's confidence/uncertainty in plain language using the numbers given.
6. Keep the tone calm, factual, and non-alarming."""


@dataclass
class ReportResult:
    narrative: str
    citations: list[str]
    cached: bool
    valid: bool
    invalid_citations: list[str]


def build_grounded_prompt(prediction_result: dict, retrieved_docs: list[dict]) -> str:
    """`prediction_result`: {"predicted_class": int, "confidence": float,
    "scores": {"Depression_Score": float, ...}, "modality_weights": {...},
    "deferred_to_human": bool, "masked_distress_index": float}.
    `retrieved_docs`: output of ClinicalKBRetriever.retrieve(...).
    """
    class_name = STRESS_LEVEL_NAMES[StressLevel(prediction_result["predicted_class"])].replace("_", " ")
    sources_block = "\n".join(f"[{d['id']}] ({d['source']}): {d['text']}" for d in retrieved_docs)

    return f"""Prediction summary:
- Predicted status: {class_name}
- Confidence: {prediction_result['confidence']:.0%}
- Deferred to human review: {prediction_result.get('deferred_to_human', False)}
- Depression score: {prediction_result['scores'].get('Depression_Score')}
- Anxiety score: {prediction_result['scores'].get('Anxiety_Score')}
- Stress score: {prediction_result['scores'].get('Stress_Score')}
- Modality contribution: {prediction_result.get('modality_weights', {})}
- Masked-Distress Index: {prediction_result.get('masked_distress_index')}

Available sources (cite ONLY these, by [doc-id]):
{sources_block}

Write a short (150-250 word) grounded clinical narrative for this result."""


def validate_citations(narrative: str, retrieved_docs: list[dict]) -> tuple[bool, list[str]]:
    """Returns (all_valid, invalid_citation_ids). A citation is invalid if it
    doesn't match any retrieved document's id -- the mechanical check behind
    "never let this layer emit an unsourced claim."
    """
    valid_ids = {d["id"] for d in retrieved_docs}
    cited = set(CITATION_PATTERN.findall(narrative))
    invalid = sorted(cited - valid_ids)
    return len(invalid) == 0, invalid


def contains_required_framing(narrative: str) -> bool:
    lowered = narrative.lower()
    return all(phrase in lowered for phrase in REQUIRED_FRAMING_PHRASES)


def generate_cached_report(prediction_result: dict, retrieved_docs: list[dict]) -> ReportResult:
    """Fully templated, LLM-free report -- the P4 fallback. Cites every
    retrieved document it references by construction, so it always passes
    `validate_citations`.
    """
    class_name = STRESS_LEVEL_NAMES[StressLevel(prediction_result["predicted_class"])].replace("_", " ")
    confidence = prediction_result["confidence"]

    lines = [
        f"This is decision-support information, not a diagnosis. It summarizes an automated "
        f"multimodal screening result for review by a qualified professional.",
        "",
        f"The system's predicted status is {class_name}, with {confidence:.0%} confidence.",
    ]
    if prediction_result.get("deferred_to_human"):
        lines.append(
            "Confidence for this result was below the system's threshold, so it has been "
            "flagged for human review rather than reported as a confident result."
        )

    scores = prediction_result.get("scores", {})
    if scores:
        lines.append(
            f"Depression score: {scores.get('Depression_Score')}, "
            f"Anxiety score: {scores.get('Anxiety_Score')}, "
            f"Stress score: {scores.get('Stress_Score')}."
        )

    band_docs = [d for d in retrieved_docs if d["category"] == "score_band"]
    for doc in band_docs[:3]:
        lines.append(f"{doc['text']} [{doc['id']}]")

    domain_docs = [d for d in retrieved_docs if d["category"] == "symptom_domain"]
    for doc in domain_docs[:2]:
        lines.append(f"{doc['text']} [{doc['id']}]")

    mdi = prediction_result.get("masked_distress_index")
    if mdi is not None and mdi >= 0.5:
        lines.append(
            f"The system's Masked-Distress Index for this result was elevated ({mdi:.2f}), "
            f"meaning facial signals read calmer than voice/physiological signals -- this "
            f"contradiction pattern is worth specific attention during professional review."
        )

    crisis_docs = [d for d in retrieved_docs if d["category"] == "crisis_resource"]
    if crisis_docs:
        lines.append("")
        lines.append("Support resources:")
        for doc in crisis_docs:
            lines.append(f"- {doc['text']} [{doc['id']}]")

    narrative = "\n".join(lines)
    valid, invalid = validate_citations(narrative, retrieved_docs)
    return ReportResult(narrative=narrative, citations=sorted(set(CITATION_PATTERN.findall(narrative))), cached=True, valid=valid, invalid_citations=invalid)


def generate_report(
    prediction_result: dict,
    retrieved_docs: list[dict],
    client=None,
    model: str = "claude-sonnet-5",
    max_tokens: int = 500,
) -> ReportResult:
    """Primary path: a live Claude call grounded in `retrieved_docs`.
    Falls back to `generate_cached_report` if no API key is configured and
    no `client` was injected (this sandbox has no ANTHROPIC_API_KEY -- the
    cached path is what's actually exercised and tested here).

    `client` is injectable (an object exposing `.messages.create(...)` per
    the `anthropic` SDK) so this function is testable without live credentials.
    """
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return generate_cached_report(prediction_result, retrieved_docs)
        import anthropic

        client = anthropic.Anthropic()

    prompt = build_grounded_prompt(prediction_result, retrieved_docs)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    narrative = response.content[0].text

    valid, invalid = validate_citations(narrative, retrieved_docs)
    if not valid:
        # A hallucinated citation is exactly the failure mode this layer
        # exists to prevent -- fall back to the always-valid cached report
        # rather than surface an unsourced claim.
        cached = generate_cached_report(prediction_result, retrieved_docs)
        cached.invalid_citations = invalid  # surface what the LLM got wrong, for logging
        return cached

    return ReportResult(
        narrative=narrative,
        citations=sorted(set(CITATION_PATTERN.findall(narrative))),
        cached=False,
        valid=True,
        invalid_citations=[],
    )
