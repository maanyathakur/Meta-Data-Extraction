from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from metadata_extraction.document_loader import TextBlock, load_bytes
from metadata_extraction.errors import ConfigurationError
from metadata_extraction.prompts import SYSTEM_PROMPT
from metadata_extraction.providers.gemini import GeminiProvider


def test_text_mode_requires_ocr_for_image(settings, png_bytes):
    with pytest.raises(ConfigurationError, match="requires OCR_MODE"):
        load_bytes(png_bytes, "image.png", replace(settings, image_input="text"))


def test_text_mode_adds_transcript_but_keeps_original_images(monkeypatch, settings, png_bytes):
    monkeypatch.setattr(
        "metadata_extraction.ocr.transcribe",
        lambda images: [TextBlock(images[0].source_id + ":ocr", "Monthly rent is 8000.")],
    )
    document = load_bytes(
        png_bytes, "image.png", replace(settings, ocr_mode="paddle", image_input="text")
    )
    assert document.images
    assert document.text_blocks[0].text == "Monthly rent is 8000."


def test_provider_text_mode_sends_no_image_bytes(
    monkeypatch, settings, png_bytes, missing_metadata
):
    monkeypatch.setattr(
        "metadata_extraction.ocr.transcribe",
        lambda images: [TextBlock(images[0].source_id + ":ocr", "Monthly rent is 8000.")],
    )
    config = replace(settings, api_key="synthetic-key", ocr_mode="paddle", image_input="text")
    document = load_bytes(png_bytes, "image.png", config)
    context = MagicMock()
    client = context.__enter__.return_value
    client.models.generate_content.return_value = SimpleNamespace(
        text=missing_metadata.model_dump_json(), model_version="fixture", usage_metadata=None
    )
    monkeypatch.setattr("metadata_extraction.providers.gemini.genai.Client", lambda **kw: context)
    GeminiProvider(config).extract(document, SYSTEM_PROMPT)
    parts = client.models.generate_content.call_args.kwargs["contents"][0].parts
    assert all(part.inline_data is None for part in parts)
    assert any("Monthly rent is 8000." in (part.text or "") for part in parts)
