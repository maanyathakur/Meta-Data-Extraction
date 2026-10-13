from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import types

from metadata_extraction.document_loader import load_bytes
from metadata_extraction.errors import ProviderError
from metadata_extraction.prompts import SYSTEM_PROMPT
from metadata_extraction.providers.gemini import GeminiProvider


def test_gemini_sdk_request_and_response(monkeypatch, settings, docx_bytes, missing_metadata):
    context = MagicMock()
    client = context.__enter__.return_value
    client.models.generate_content.return_value = SimpleNamespace(
        text=missing_metadata.model_dump_json(), model_version="test-version", usage_metadata=None
    )
    monkeypatch.setattr("metadata_extraction.providers.gemini.genai.Client", lambda **kw: context)
    provider = GeminiProvider(replace(settings, api_key="synthetic-test-key"))
    result = provider.extract(load_bytes(docx_bytes, "input.docx", settings), SYSTEM_PROMPT)
    assert result.metadata == missing_metadata
    request = client.models.generate_content.call_args.kwargs
    assert request["model"] == settings.model
    assert isinstance(request["config"], types.GenerateContentConfig)
    assert request["config"].response_mime_type == "application/json"
    assert request["config"].max_output_tokens == settings.max_output_tokens
    assert request["config"].thinking_config.thinking_level.value == settings.thinking_level.upper()
    assert request["config"].response_json_schema
    text = " ".join(part.text or "" for part in request["contents"][0].parts)
    assert "body-1" in text and "Example Owner" in text
    assert "input.docx" not in text


def test_bad_schema_response_is_retried_and_fails(monkeypatch, settings, docx_bytes):
    context = MagicMock()
    client = context.__enter__.return_value
    client.models.generate_content.return_value = SimpleNamespace(text="{}")
    monkeypatch.setattr("metadata_extraction.providers.gemini.genai.Client", lambda **kw: context)
    monkeypatch.setattr("metadata_extraction.providers.gemini.time.sleep", lambda seconds: None)
    provider = GeminiProvider(replace(settings, api_key="synthetic-test-key", max_attempts=2))
    with pytest.raises(ProviderError, match="schema-valid"):
        provider.extract(load_bytes(docx_bytes, "input.docx", settings), SYSTEM_PROMPT)
    assert client.models.generate_content.call_count == 2
