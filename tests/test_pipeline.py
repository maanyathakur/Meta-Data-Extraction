from dataclasses import replace

import pytest
from pydantic import ValidationError

from metadata_extraction.document_loader import load_bytes
from metadata_extraction.errors import ConfigurationError
from metadata_extraction.normalizer import normalize_metadata
from metadata_extraction.pipeline import ExtractionPipeline, grounding_issues
from metadata_extraction.schemas import FieldResult


def test_schema_rejects_unsupported_found_value():
    with pytest.raises(ValidationError):
        FieldResult(
            value="15000",
            status="found",
            provenance="explicit",
            evidence=[],
            derivation=None,
            explanation=None,
        )


def test_derived_value_requires_explanation():
    with pytest.raises(ValidationError):
        FieldResult(
            value="30",
            status="found",
            provenance="derived",
            evidence=[{"source_id": "body-1", "quote": "one month"}],
            derivation=None,
            explanation=None,
        )


def test_missing_schema_accepts_null(missing_metadata):
    assert missing_metadata.party_one.value is None
    assert missing_metadata.party_one.provenance is None


def test_normalizer_does_not_choose_values(missing_metadata):
    metadata = missing_metadata.model_copy(deep=True)
    metadata.party_one = FieldResult(
        value="  Example   OWNER  ",
        status="found",
        provenance="explicit",
        evidence=[{"source_id": "body-1", "quote": "Example OWNER"}],
        derivation=None,
        explanation=None,
    )
    result = normalize_metadata(metadata)
    assert result.party_one.value == "Example OWNER"
    assert result.agreement_value.value is None


def test_cache_avoids_second_provider_call(settings, fake_provider, docx_bytes):
    pipeline = ExtractionPipeline(settings, fake_provider)
    first = pipeline.extract_bytes(docx_bytes, "contract.docx")
    second = pipeline.extract_bytes(docx_bytes, "renamed.docx")
    assert fake_provider.calls == 1
    assert not first.cached and second.cached
    assert second.filename == "renamed.docx"


def test_model_change_invalidates_cache(settings, fake_provider, docx_bytes):
    ExtractionPipeline(settings, fake_provider).extract_bytes(docx_bytes, "contract.docx")
    ExtractionPipeline(replace(settings, model="other-flash"), fake_provider).extract_bytes(
        docx_bytes, "contract.docx"
    )
    assert fake_provider.calls == 2


def test_grounding_flags_unknown_source(settings, docx_bytes, missing_metadata):
    metadata = missing_metadata.model_copy(deep=True)
    metadata.party_one = FieldResult(
        value="Owner",
        status="found",
        provenance="explicit",
        evidence=[{"source_id": "made-up", "quote": "Owner"}],
        derivation=None,
        explanation=None,
    )
    issues = grounding_issues(metadata, load_bytes(docx_bytes, "contract.docx", settings))
    assert issues == [{"field": "party_one", "source_id": None, "reason": "unknown_source"}]


def test_missing_key_error_is_actionable(settings, docx_bytes):
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        ExtractionPipeline(settings).extract_bytes(docx_bytes, "contract.docx", use_cache=False)


def test_settings_repr_does_not_expose_key(settings):
    assert "secret-key" not in repr(replace(settings, api_key="secret-key"))


def test_generation_settings_invalidate_cache(settings, fake_provider, docx_bytes):
    ExtractionPipeline(settings, fake_provider).extract_bytes(docx_bytes, "a.docx")
    ExtractionPipeline(replace(settings, thinking_level="medium"), fake_provider).extract_bytes(
        docx_bytes, "a.docx"
    )
    ExtractionPipeline(replace(settings, max_output_tokens=32768), fake_provider).extract_bytes(
        docx_bytes, "a.docx"
    )
    assert fake_provider.calls == 3


def test_amount_representation_preserves_numeric_value():
    from metadata_extraction.normalizer import canonical_amount

    for raw, expected in [
        ("4000.00", "4000"),
        ("10,000", "10000"),
        ("12,345.50", "12345.5"),
        ("0.00", "0"),
    ]:
        assert canonical_amount(raw) == expected
    for malformed in ["10,5", "1,00,000", "Rs.1000", "NaN", "1e3", "12.34.5"]:
        assert canonical_amount(malformed) == malformed


def test_cached_amount_is_renormalized_without_model_call(
    settings, fake_provider, docx_bytes, missing_metadata
):
    import json

    pipeline = ExtractionPipeline(settings, fake_provider)
    record = pipeline.extract_bytes(docx_bytes, "a.docx")
    path = next(settings.cache_dir.glob("*.json"))
    data = record.model_dump(mode="json")
    data["metadata"]["agreement_value"] = {
        "value": "10,000.00",
        "status": "found",
        "provenance": "explicit",
        "evidence": [{"source_id": "body-1", "quote": "Agreement starts here."}],
        "derivation": None,
        "explanation": None,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    cached = pipeline.extract_bytes(docx_bytes, "a.docx")
    assert cached.metadata.agreement_value.value == "10000"
    assert cached.raw_response == record.raw_response
    assert fake_provider.calls == 1
