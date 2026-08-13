import os
from unittest.mock import patch

from src.reasoning.rag_report import (
    build_grounded_prompt,
    contains_required_framing,
    generate_cached_report,
    generate_report,
    validate_citations,
)


SAMPLE_DOCS = [
    {"id": "band-stress-severe", "text": "Stress_Score in the range 23-31 is interpreted as the Severe band.", "source": "placeholder", "category": "score_band"},
    {"id": "dsm5-domain-sleep", "text": "Sleep disruption is a commonly observed symptom domain.", "source": "placeholder", "category": "symptom_domain"},
    {"id": "crisis-us-988", "text": "988 Suicide & Crisis Lifeline is reachable 24/7.", "source": "placeholder", "category": "crisis_resource"},
]

SAMPLE_PREDICTION = {
    "predicted_class": 3,
    "confidence": 0.55,
    "deferred_to_human": True,
    "scores": {"Depression_Score": 25.0, "Anxiety_Score": 18.0, "Stress_Score": 29.0},
    "modality_weights": {"face": 0.2, "speech": 0.3, "tabular": 0.5},
    "masked_distress_index": 0.7,
}


def test_build_grounded_prompt_includes_all_sources_and_scores():
    prompt = build_grounded_prompt(SAMPLE_PREDICTION, SAMPLE_DOCS)
    for doc in SAMPLE_DOCS:
        assert doc["id"] in prompt
    assert "29.0" in prompt
    assert "Severe" in prompt


def test_validate_citations_accepts_valid_ids():
    narrative = "The result is severe [band-stress-severe] and sleep matters [dsm5-domain-sleep]."
    valid, invalid = validate_citations(narrative, SAMPLE_DOCS)
    assert valid is True
    assert invalid == []


def test_validate_citations_rejects_fabricated_ids():
    narrative = "This claim is unsourced [totally-made-up-doc]."
    valid, invalid = validate_citations(narrative, SAMPLE_DOCS)
    assert valid is False
    assert invalid == ["totally-made-up-doc"]


def test_generate_cached_report_is_always_valid_and_framed():
    report = generate_cached_report(SAMPLE_PREDICTION, SAMPLE_DOCS)
    assert report.cached is True
    assert report.valid is True
    assert report.invalid_citations == []
    assert contains_required_framing(report.narrative)
    assert "crisis-us-988" in report.citations


def test_generate_cached_report_surfaces_high_mdi():
    report = generate_cached_report(SAMPLE_PREDICTION, SAMPLE_DOCS)
    assert "Masked-Distress" in report.narrative


def test_generate_cached_report_omits_mdi_mention_when_low():
    low_mdi_prediction = {**SAMPLE_PREDICTION, "masked_distress_index": 0.1}
    report = generate_cached_report(low_mdi_prediction, SAMPLE_DOCS)
    assert "Masked-Distress" not in report.narrative


def test_generate_report_uses_cached_path_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        report = generate_report(SAMPLE_PREDICTION, SAMPLE_DOCS)
    assert report.cached is True


class _FakeMessage:
    def __init__(self, text):
        self.content = [type("Block", (), {"text": text})()]


class _FakeMessages:
    def __init__(self, response_text):
        self._response_text = response_text
        self.last_call_kwargs = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeMessage(self._response_text)


class _FakeAnthropicClient:
    def __init__(self, response_text):
        self.messages = _FakeMessages(response_text)


def test_generate_report_with_valid_llm_citations_uses_live_path():
    fake_client = _FakeAnthropicClient(
        "This result reads Severe [band-stress-severe], and sleep disruption is relevant "
        "[dsm5-domain-sleep]. This is decision-support information, not a diagnosis."
    )
    report = generate_report(SAMPLE_PREDICTION, SAMPLE_DOCS, client=fake_client)
    assert report.cached is False
    assert report.valid is True
    assert set(report.citations) == {"band-stress-severe", "dsm5-domain-sleep"}
    assert fake_client.messages.last_call_kwargs["system"] is not None


def test_generate_report_falls_back_to_cached_on_hallucinated_citation():
    fake_client = _FakeAnthropicClient(
        "This claim cites a source that was never retrieved [fabricated-source-id]."
    )
    report = generate_report(SAMPLE_PREDICTION, SAMPLE_DOCS, client=fake_client)
    assert report.cached is True  # fell back rather than surfacing the bad citation
    assert report.invalid_citations == ["fabricated-source-id"]
    assert report.valid is True  # the cached fallback itself is always valid
