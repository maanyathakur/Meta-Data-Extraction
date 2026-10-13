import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai.errors import APIError

from metadata_extraction.document_loader import load_bytes
from metadata_extraction.errors import ProviderError
from metadata_extraction.prompts import SYSTEM_PROMPT
from metadata_extraction.provider_logging import api_diagnostics, write_event
from metadata_extraction.providers.gemini import GeminiProvider


def quota_error():
    return APIError(
        429,
        {
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "message": "secret-key and confidential document content",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaMetric": (
                                    "generativelanguage.googleapis.com/generate_content_requests"
                                ),
                                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                "quotaValue": "20",
                                "quotaDimensions": {
                                    "model": "gemini-test",
                                    "location": "global",
                                    "project": "private-project",
                                },
                                "description": "private document content",
                            }
                        ],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "45.5s"},
                ],
            }
        },
    )


def test_log_allowlists_quota_and_omits_sensitive_response(tmp_path):
    diagnostics = api_diagnostics(quota_error())
    assert diagnostics["retry_delay_seconds"] == 45.5
    assert diagnostics["quota_violations"][0]["quotaValue"] == "20"
    assert write_event(tmp_path, diagnostics)
    saved = (tmp_path / "provider_events.jsonl").read_text(encoding="utf-8")
    assert "secret-key" not in saved
    assert "document content" not in saved
    assert "private-project" not in saved
    assert json.loads(saved)["provider_status"] == "RESOURCE_EXHAUSTED"


def test_provider_records_quota_attempt_and_cooldown(monkeypatch, settings, docx_bytes):
    context = MagicMock()
    client = context.__enter__.return_value
    client.models.generate_content.side_effect = quota_error()
    monkeypatch.setattr("metadata_extraction.providers.gemini.genai.Client", lambda **kw: context)
    monkeypatch.setattr("metadata_extraction.providers.gemini.time.sleep", lambda seconds: None)
    settings = replace(settings, api_key="synthetic-test-key", max_attempts=2)
    with pytest.raises(ProviderError, match="quota-limited"):
        GeminiProvider(settings).extract(
            load_bytes(docx_bytes, "input.docx", settings), SYSTEM_PROMPT
        )
    events = [
        json.loads(line)
        for line in (settings.output_dir / "provider_events.jsonl").read_text().splitlines()
    ]
    assert [e["event"] for e in events] == [
        "request_started",
        "api_error",
        "key_cooldown",
    ]
    assert events[-1]["attempt"] == 1
    assert events[1]["http_status"] == 429
    assert events[0]["request_id"] == events[-1]["request_id"]
    assert "synthetic-test-key" not in json.dumps(events)
    assert "Example Owner" not in json.dumps(events)


def test_malformed_diagnostics_and_log_write_failure(tmp_path):
    exc = SimpleNamespace(code=429, status="RESOURCE_EXHAUSTED", details={"error": None})
    assert api_diagnostics(exc)["quota_violations"] == []
    path = tmp_path / "file"
    path.write_text("not a directory")
    assert write_event(path, {"event": "api_error"}) is False


def test_schema_diagnostics_exclude_response_content(monkeypatch, settings, docx_bytes):
    context = MagicMock()
    client = context.__enter__.return_value
    client.models.generate_content.return_value = SimpleNamespace(
        text="private-document-not-json",
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
    )
    monkeypatch.setattr("metadata_extraction.providers.gemini.genai.Client", lambda **kw: context)
    settings = replace(settings, api_key="synthetic-test-key", max_attempts=1)
    with pytest.raises(ProviderError, match="schema-valid"):
        GeminiProvider(settings).extract(
            load_bytes(docx_bytes, "input.docx", settings), SYSTEM_PROMPT
        )
    text = (settings.output_dir / "provider_events.jsonl").read_text(encoding="utf-8")
    event = json.loads(text.splitlines()[-1])
    assert event["validation_types"] == ["json_invalid"]
    assert event["finish_reason"] == "MAX_TOKENS"
    assert event["response_characters"] == len("private-document-not-json")
    assert "private-document-not-json" not in text
    assert "synthetic-test-key" not in text
