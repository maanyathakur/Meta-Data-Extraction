import io
import json
from dataclasses import replace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from metadata_extraction.api import create_app
from metadata_extraction.document_loader import TextBlock, load_bytes
from metadata_extraction.errors import (
    ConfigurationError,
    DocumentError,
    ProviderError,
    ProviderQuotaError,
)
from metadata_extraction.pipeline import ExtractionPipeline
from metadata_extraction.prompts import SYSTEM_PROMPT
from metadata_extraction.providers import create_provider
from metadata_extraction.providers.compatible import CompatibleProvider
from metadata_extraction.providers.gemini import GeminiProvider


def compatible(settings, **changes):
    return replace(
        settings,
        provider="openai-compatible",
        base_url="https://example.test/v1",
        model="text-model",
        llm_api_key="synthetic-key",
        **changes,
    )


def test_factory_keeps_gemini_default(settings):
    assert isinstance(create_provider(replace(settings, api_key="synthetic")), GeminiProvider)
    assert isinstance(create_provider(compatible(settings)), CompatibleProvider)


def test_text_model_automatically_uses_ocr(monkeypatch, settings, png_bytes):
    seen = []

    def transcript(images):
        seen.extend(images)
        return [TextBlock(images[0].source_id + ":ocr", "Rent is 8000.")]

    monkeypatch.setattr("metadata_extraction.ocr.transcribe_rapid", transcript)
    config = replace(settings, model_supports_vision=False)
    document = load_bytes(png_bytes, "scan.png", config)
    assert seen and document.images and document.text_blocks[0].text == "Rent is 8000."
    assert config.effective_image_input == "text" and config.effective_ocr_mode == "rapid"


def test_native_docx_and_vision_do_not_invoke_ocr(monkeypatch, settings, docx_bytes, png_bytes):
    def unexpected(images):
        pytest.fail("OCR must not run")

    monkeypatch.setattr("metadata_extraction.ocr.transcribe_rapid", unexpected)
    assert load_bytes(docx_bytes, "text.docx", replace(settings, model_supports_vision=False))
    assert load_bytes(png_bytes, "scan.png", settings).images


def test_empty_automatic_ocr_is_rejected(monkeypatch, settings, png_bytes):
    monkeypatch.setattr("metadata_extraction.ocr.transcribe_rapid", lambda images: [])
    with pytest.raises(DocumentError, match="no readable"):
        load_bytes(png_bytes, "scan.png", replace(settings, model_supports_vision=False))


def test_compatible_request_validates_schema_without_filename(
    monkeypatch, settings, docx_bytes, missing_metadata
):
    config = compatible(settings, model_supports_vision=False)
    context = MagicMock()
    client = context.__enter__.return_value
    client.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", config.base_url),
        json={
            "model": "resolved",
            "choices": [{"message": {"content": missing_metadata.model_dump_json()}}],
            "usage": {"total_tokens": 10},
        },
    )
    monkeypatch.setattr(
        "metadata_extraction.providers.compatible.httpx.Client", lambda **kw: context
    )
    result = create_provider(config).extract(
        load_bytes(docx_bytes, "private.docx", config), SYSTEM_PROMPT
    )
    assert result.metadata == missing_metadata and result.model_version == "resolved"
    payload = client.post.call_args.kwargs["json"]
    assert isinstance(payload["messages"][1]["content"], str)
    assert "private.docx" not in str(payload)
    assert payload["response_format"] == {"type": "json_object"}


def test_compatible_text_payload_has_no_images(monkeypatch, settings, png_bytes):
    monkeypatch.setattr(
        "metadata_extraction.ocr.transcribe_rapid",
        lambda images: [TextBlock(images[0].source_id + ":ocr", "Rent 8000")],
    )
    config = compatible(settings, model_supports_vision=False)
    document = load_bytes(png_bytes, "scan.png", config)
    payload = create_provider(config)._payload(document, SYSTEM_PROMPT)
    assert "image_url" not in str(payload) and "Rent 8000" in str(payload)
    vision = create_provider(compatible(settings))._payload(document, SYSTEM_PROMPT)
    assert any(part["type"] == "image_url" for part in vision["messages"][1]["content"])


def test_provider_credentials_and_configuration_are_isolated(settings):
    config = compatible(settings, api_key="gemini-secret")
    assert config.configured_keys() == [(1, "synthetic-key")]
    assert "synthetic-key" not in repr(config) and "gemini-secret" not in repr(config)
    pipeline = ExtractionPipeline(config)
    assert (
        pipeline._configuration_hash()
        != ExtractionPipeline(
            replace(config, base_url="https://other.test/v1")
        )._configuration_hash()
    )
    assert (
        pipeline._configuration_hash()
        != ExtractionPipeline(replace(config, model_supports_vision=False))._configuration_hash()
    )


def test_compatible_quota_fails_over(monkeypatch, settings, docx_bytes, missing_metadata):
    config = compatible(settings, llm_api_key_2="backup-key")
    context = MagicMock()
    client = context.__enter__.return_value
    request = httpx.Request("POST", config.base_url)
    client.post.side_effect = [
        httpx.Response(429, request=request, headers={"Retry-After": "120"}),
        httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": missing_metadata.model_dump_json()}}]},
        ),
    ]
    monkeypatch.setattr(
        "metadata_extraction.providers.compatible.httpx.Client", lambda **kw: context
    )
    provider = create_provider(config)
    provider.extract(load_bytes(docx_bytes, "a.docx", config), SYSTEM_PROMPT)
    assert client.post.call_count == 2
    assert provider.key_pool.available_keys() == [(2, "backup-key")]
    provider.key_pool.block("backup-key", {"retry_delay_seconds": 60})
    with pytest.raises(ProviderQuotaError):
        provider.extract(load_bytes(docx_bytes, "a.docx", config), SYSTEM_PROMPT)


def test_compatible_schema_failures_are_bounded(monkeypatch, settings, docx_bytes):
    config = compatible(settings, max_attempts=2)
    context = MagicMock()
    client = context.__enter__.return_value
    client.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", config.base_url),
        json={"choices": [{"message": {"content": "{}"}}]},
    )
    monkeypatch.setattr(
        "metadata_extraction.providers.compatible.httpx.Client", lambda **kw: context
    )
    monkeypatch.setattr("metadata_extraction.providers.compatible.time.sleep", lambda seconds: None)
    with pytest.raises(ProviderError, match="schema-valid"):
        create_provider(config).extract(load_bytes(docx_bytes, "a.docx", config), SYSTEM_PROMPT)
    assert client.post.call_count == 2


@pytest.mark.parametrize("mode", ["json_schema", "none"])
def test_compatible_response_format_options(settings, docx_bytes, mode):
    config = compatible(settings, response_format=mode, token_parameter="max_completion_tokens")
    payload = create_provider(config)._payload(
        load_bytes(docx_bytes, "a.docx", config), SYSTEM_PROMPT
    )
    assert payload["max_completion_tokens"] == config.max_output_tokens
    assert "max_tokens" not in payload
    if mode == "none":
        assert "response_format" not in payload
    else:
        assert payload["response_format"]["json_schema"]["schema"]


def test_compatible_requires_explicit_endpoint_and_its_own_key(settings):
    with pytest.raises(ConfigurationError, match="LLM_BASE_URL"):
        replace(settings, provider="openai-compatible")
    config = replace(compatible(settings), llm_api_key="", api_key="gemini-secret")
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        create_provider(config)


def test_api_automatic_ocr_and_compatible_provider(
    monkeypatch, settings, png_bytes, missing_metadata
):
    config = compatible(settings, model_supports_vision=False)
    monkeypatch.setattr(
        "metadata_extraction.ocr.transcribe_rapid",
        lambda images: [TextBlock(images[0].source_id + ":ocr", "Rent is 8000.")],
    )
    requests = []

    def upstream(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": missing_metadata.model_dump_json()}}]}
        )

    client_type = httpx.Client
    transport = httpx.MockTransport(upstream)
    monkeypatch.setattr(
        "metadata_extraction.providers.compatible.httpx.Client",
        lambda **kwargs: client_type(transport=transport, **kwargs),
    )
    api = TestClient(create_app(config))
    assert api.get("/health").json()["ocr_mode"] == "rapid"
    for cached in (False, True):
        response = api.post("/extract?debug=true", files={"file": ("scan.png", png_bytes)})
        assert response.status_code == 200 and response.json()["cached"] == cached
    assert len(requests) == 1
    assert "Rent is 8000." in str(requests[0]) and "image_url" not in str(requests[0])


def test_real_rapidocr_text_model_ingestion(settings):
    pytest.importorskip("rapidocr", reason="Optional OCR dependencies are not installed")
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except OSError:
        font = ImageFont.load_default(size=48)
    image = Image.new("RGB", (1000, 180), "white")
    ImageDraw.Draw(image).text((30, 50), "MONTHLY RENT IS 8000", font=font, fill="black")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    document = load_bytes(
        stream.getvalue(), "synthetic.png", replace(settings, model_supports_vision=False)
    )
    assert document.images
    assert any(
        "8000" in block.text and block.source_id.endswith(":ocr") for block in document.text_blocks
    )
