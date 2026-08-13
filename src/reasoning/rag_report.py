"""Grounded, cited clinical report generation (docs/PROJECT_PLAN.md P4;
docs/MINDSCOPE_Blueprint.pdf Sec. 05 Layer 3: "writes a citation-backed
narrative. Every clinical claim traces to a source -- no hallucinated
advice.")

Generators, in the order the system prefers them:

  1. `generate_report_ollama` -- **the default**. A local `llama3.1` served by
     Ollama. Local by design: this system already handles face, voice, and
     physiological data, and the prompt embeds both the prediction and the
     retrieved clinical text, so keeping generation on-device avoids sending
     any of it to a third party.
  2. `generate_report` -- hosted Claude via the `anthropic` SDK, for
     deployments that prefer it. Active only with `ANTHROPIC_API_KEY` set or
     an injected client.
  3. `generate_cached_report` -- templated, no LLM, always available. Every
     failure in (1) or (2) lands here.

Whichever runs, the LLM is instructed to cite every substantive claim with a
`[doc-id]` marker referencing a retrieved knowledge-base document
(src/reasoning/retriever.py); `validate_citations` then mechanically checks
every marker resolves to an actually-retrieved document, so a hallucinated
citation is caught rather than trusted -- and a report that fails the check
is *replaced* by the templated one, not patched up and shipped.

Responsible-AI framing (decision-support, not diagnosis; uncertainty
disclosure; crisis-resource surfacing) is baked into the prompt template
itself, not left to the model's discretion, and is checked again by
`contains_required_framing` after generation.

Degradation is always labelled. A fallback report carries `cached=True`,
`generator="template"`, and a human-readable `fallback_reason` that the API
returns and the UI displays -- so a templated summary is never mistaken for
a model-written one.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from src.data.schemas import STRESS_LEVEL_NAMES, StressLevel
from src.reasoning.ollama_config import DEFAULT_REQUEST_TIMEOUT_SECONDS, llm_model, probe_ollama

logger = logging.getLogger(__name__)

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
    #: Which generator produced this, e.g. "ollama:llama3.1" or "template".
    generator: str = "template"
    #: Set when the live generator was skipped or failed. This is what the UI
    #: shows the clinician so a templated fallback is never mistaken for a
    #: model-written narrative.
    fallback_reason: str | None = None


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


def check_generated_narrative(narrative: str, retrieved_docs: list[dict]) -> tuple[bool, str | None, list[str]]:
    """Full acceptance check for an LLM-written narrative.

    Returns `(accepted, rejection_reason, invalid_citation_ids)`.

    `validate_citations` alone is not sufficient, and the gap is easy to miss:
    it asks "does every citation resolve to a retrieved document?", which is
    *trivially true for a narrative containing no citations at all*. A model
    that ignores the citation instruction entirely would therefore pass, and
    ship a fully uncited clinical narrative -- precisely the failure this
    layer exists to prevent. Observed in practice with llama3.1, which
    sometimes writes a fluent summary with no `[doc-id]` markers.

    So acceptance requires all three:
      1. every citation present resolves to a retrieved document,
      2. at least one citation is present,
      3. the responsible-AI framing survived (decision-support, not diagnosis).
    """
    valid, invalid = validate_citations(narrative, retrieved_docs)
    if not valid:
        return False, f"Model cited unknown source id(s): {', '.join(invalid)}.", invalid

    if not CITATION_PATTERN.search(narrative):
        return (
            False,
            "Model returned a narrative with no citations, so no clinical claim in it is traceable to a source.",
            [],
        )

    if not contains_required_framing(narrative):
        return (
            False,
            "Model omitted the required decision-support framing (this is not a diagnosis).",
            [],
        )

    return True, None, []


def generate_cached_report(
    prediction_result: dict,
    retrieved_docs: list[dict],
    fallback_reason: str | None = None,
) -> ReportResult:
    """Fully templated, LLM-free report -- the always-available fallback.
    Cites every retrieved document it references by construction, so it
    always passes `validate_citations`.

    `fallback_reason` explains *why* the live generator was not used and is
    propagated to the API response, so the UI can tell the clinician the
    difference between "the model wrote this" and "the model was unavailable
    and this is a template".
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

    if fallback_reason:
        lines.append("")
        lines.append(f"(Automated narrative generation was unavailable: {fallback_reason} "
                     f"This is a templated summary of the same underlying result and citations.)")

    narrative = "\n".join(lines)
    valid, invalid = validate_citations(narrative, retrieved_docs)
    return ReportResult(
        narrative=narrative,
        citations=sorted(set(CITATION_PATTERN.findall(narrative))),
        cached=True,
        valid=valid,
        invalid_citations=invalid,
        generator="template",
        fallback_reason=fallback_reason,
    )


def generate_report_ollama(
    prediction_result: dict,
    retrieved_docs: list[dict],
    chat_model=None,
    model: str | None = None,
) -> ReportResult:
    """Primary path: a **local** `llama3.1` via Ollama, grounded in `retrieved_docs`.

    Running the narrative layer locally is a deliberate privacy property, not
    a cost decision -- this system already handles face, voice, and
    physiological data, and the report prompt embeds the prediction and the
    retrieved clinical text. Nothing here leaves the machine.

    Every failure mode degrades to `generate_cached_report`, which is
    templated, needs no LLM, and is valid-by-construction. Degradation is
    always *labelled* (`cached=True` plus a `fallback_reason`), never a
    silent swap that looks like a live generation:

      - Ollama unreachable            -> cached, reason names the base URL
      - model not pulled              -> cached, reason names `ollama pull ...`
      - generation error / timeout    -> cached, reason names the exception
      - hallucinated citation         -> cached, reason names the bad ids

    `chat_model` is injectable (anything with `.invoke(messages) -> .content`)
    so this is testable without a running Ollama.
    """
    model = model or llm_model()

    if chat_model is None:
        status = probe_ollama()
        reason = status.unavailable_reason(model)
        if reason is not None:
            return generate_cached_report(prediction_result, retrieved_docs, fallback_reason=reason)

        try:
            try:
                from langchain_ollama import ChatOllama
            except ImportError:  # older split-package layout
                from langchain_community.chat_models import ChatOllama

            chat_model = ChatOllama(
                model=model,
                base_url=status.base_url,
                temperature=0.2,  # a clinical summary should be reproducible, not creative
                timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            return generate_cached_report(
                prediction_result,
                retrieved_docs,
                fallback_reason=f"Could not construct the Ollama client ({type(exc).__name__}: {exc}).",
            )

    prompt = build_grounded_prompt(prediction_result, retrieved_docs)
    try:
        response = chat_model.invoke(
            [("system", SYSTEM_PROMPT), ("human", prompt)]
        )
        narrative = getattr(response, "content", None) or str(response)
    except Exception as exc:
        # Connection reset mid-stream, model evicted, request timeout, OOM.
        logger.warning("Ollama generation failed, serving the cached report: %s", exc)
        return generate_cached_report(
            prediction_result,
            retrieved_docs,
            fallback_reason=f"Ollama generation failed ({type(exc).__name__}: {exc}).",
        )

    accepted, rejection_reason, invalid = check_generated_narrative(narrative, retrieved_docs)
    if not accepted:
        # An unsourced or mis-sourced narrative is exactly the failure mode
        # this layer exists to prevent -- replace it with the always-valid
        # templated report rather than surface it.
        logger.warning("Rejecting generated narrative: %s", rejection_reason)
        cached = generate_cached_report(prediction_result, retrieved_docs, fallback_reason=rejection_reason)
        cached.invalid_citations = invalid  # surface what the LLM got wrong, for logging
        return cached

    return ReportResult(
        narrative=narrative,
        citations=sorted(set(CITATION_PATTERN.findall(narrative))),
        cached=False,
        valid=True,
        invalid_citations=[],
        generator=f"ollama:{model}",
    )


def generate_report(
    prediction_result: dict,
    retrieved_docs: list[dict],
    client=None,
    model: str = "claude-sonnet-5",
    max_tokens: int = 500,
) -> ReportResult:
    """Hosted-Claude path, kept for deployments that prefer a hosted model.

    The default generator is now the local Ollama one
    (`generate_report_ollama`, wired in as `AgentContext.report_fn`); this
    activates only when `ANTHROPIC_API_KEY` is set or a `client` is injected,
    and otherwise falls through to the same templated cached report.
    """
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return generate_cached_report(
                prediction_result,
                retrieved_docs,
                fallback_reason="No ANTHROPIC_API_KEY configured for the hosted-Claude generator.",
            )
        import anthropic

        client = anthropic.Anthropic()

    prompt = build_grounded_prompt(prediction_result, retrieved_docs)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        narrative = response.content[0].text
    except Exception as exc:
        return generate_cached_report(
            prediction_result,
            retrieved_docs,
            fallback_reason=f"Hosted model call failed ({type(exc).__name__}: {exc}).",
        )

    accepted, rejection_reason, invalid = check_generated_narrative(narrative, retrieved_docs)
    if not accepted:
        cached = generate_cached_report(prediction_result, retrieved_docs, fallback_reason=rejection_reason)
        cached.invalid_citations = invalid
        return cached

    return ReportResult(
        narrative=narrative,
        citations=sorted(set(CITATION_PATTERN.findall(narrative))),
        cached=False,
        valid=True,
        invalid_citations=[],
        generator=f"anthropic:{model}",
    )
