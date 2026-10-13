import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai.errors import APIError

from metadata_extraction.document_loader import load_bytes
from metadata_extraction.errors import ProviderError, ProviderQuotaError
from metadata_extraction.key_pool import KeyPool
from metadata_extraction.pipeline import ExtractionPipeline
from metadata_extraction.prompts import SYSTEM_PROMPT
from metadata_extraction.providers.gemini import GeminiProvider


def quota(delay=3600, daily=True):
    return APIError(
        429,
        {
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaId": "RequestsPerDay" if daily else "RequestsPerMinute",
                                "quotaValue": "20",
                            }
                        ],
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": f"{delay}s",
                    },
                ],
            }
        },
    )


def install_clients(monkeypatch, outcomes, missing_metadata):
    used = []

    def factory(**kwargs):
        key = kwargs["api_key"]
        used.append(key)
        context = MagicMock()
        client = context.__enter__.return_value
        outcome = outcomes[key]
        if isinstance(outcome, Exception):
            client.models.generate_content.side_effect = outcome
        else:
            client.models.generate_content.return_value = SimpleNamespace(
                text=missing_metadata.model_dump_json(),
                model_version="fixture",
                usage_metadata=None,
            )
        return context

    monkeypatch.setattr("metadata_extraction.providers.gemini.genai.Client", factory)
    monkeypatch.setattr("metadata_extraction.providers.gemini.time.sleep", lambda seconds: None)
    return used


def test_priority_failover_and_persisted_cooldown(
    monkeypatch, settings, docx_bytes, missing_metadata
):
    settings = replace(settings, api_key="key-one", api_key_2="key-two", api_key_3="key-three")
    used = install_clients(
        monkeypatch, {"key-one": quota(), "key-two": quota(), "key-three": True}, missing_metadata
    )
    provider = GeminiProvider(settings)
    document = load_bytes(docx_bytes, "sample.docx", settings)
    assert provider.extract(document, SYSTEM_PROMPT).metadata == missing_metadata
    assert used == ["key-one", "key-two", "key-three"]
    GeminiProvider(settings).extract(document, SYSTEM_PROMPT)
    assert used[-1] == "key-three" and len(used) == 4
    state = provider.key_pool.path.read_text(encoding="utf-8")
    assert "key-one" not in state and "key-two" not in state
    events = (settings.output_dir / "provider_events.jsonl").read_text()
    assert '"key_slot": 3' in events
    assert all(key not in events for _, key in settings.configured_keys())


def test_all_slots_exhausted_no_more_network(monkeypatch, settings, docx_bytes, missing_metadata):
    settings = replace(settings, api_key="key-one", api_key_2="key-two", api_key_3="key-three")
    used = install_clients(
        monkeypatch, {k: quota() for _, k in settings.configured_keys()}, missing_metadata
    )
    provider = GeminiProvider(settings)
    document = load_bytes(docx_bytes, "sample.docx", settings)
    with pytest.raises(ProviderQuotaError) as failure:
        provider.extract(document, SYSTEM_PROMPT)
    assert 3595 <= failure.value.retry_after_seconds <= 3600
    assert len(used) == 3
    with pytest.raises(ProviderQuotaError):
        GeminiProvider(settings).extract(document, SYSTEM_PROMPT)
    assert len(used) == 3


def test_duplicate_keys_blank_primary_and_secret_repr(settings):
    settings = replace(settings, api_key="", api_key_2=" key-two ", api_key_3="key-two")
    settings.require_key()
    assert settings.configured_keys() == [(2, "key-two")]
    assert "key-two" not in repr(settings)
    status = KeyPool(settings).public_status()
    assert status[1]["available"] and status[2]["duplicate"]
    assert "key-two" not in json.dumps(status)


def test_cooldown_expiry_and_new_key_is_not_blocked(monkeypatch, settings):
    now = [1000.0]
    monkeypatch.setattr("metadata_extraction.key_pool.time.time", lambda: now[0])
    settings = replace(settings, api_key="key-one")
    pool = KeyPool(settings)
    assert pool.block("key-one", {"retry_delay_seconds": 45.5}) == 46
    assert not KeyPool(settings).available_keys()
    assert KeyPool(replace(settings, api_key="different-key")).available_keys()
    now[0] += 46
    assert KeyPool(settings).available_keys() == [(1, "key-one")]


def test_daily_and_unknown_cooldown_defaults(settings):
    settings = replace(settings, api_key="key-one")
    pool = KeyPool(settings)
    assert pool.block("key-one", {"quota_violations": [{"quotaId": "RequestsPerDay"}]}) == 86400
    assert pool.block("key-one", {}) == 60


def test_nonquota_error_does_not_rotate(monkeypatch, settings, docx_bytes, missing_metadata):
    settings = replace(settings, api_key="key-one", api_key_2="key-two")
    used = install_clients(
        monkeypatch,
        {
            "key-one": APIError(
                403,
                {"error": {"status": "PERMISSION_DENIED", "message": "secret provider message"}},
            )
        },
        missing_metadata,
    )
    with pytest.raises(ProviderError, match="403"):
        GeminiProvider(settings).extract(
            load_bytes(docx_bytes, "sample.docx", settings), SYSTEM_PROMPT
        )
    assert used == ["key-one"]
    assert not KeyPool(settings).public_status()[0]["duplicate"]


def test_slot_changes_preserve_extraction_cache(settings, fake_provider, docx_bytes):
    pipeline = ExtractionPipeline(settings, fake_provider)
    pipeline.extract_bytes(docx_bytes, "a.docx")
    changed = ExtractionPipeline(replace(settings, api_key_2="second-project"), fake_provider)
    assert changed.extract_bytes(docx_bytes, "a.docx").cached
    assert fake_provider.calls == 1
