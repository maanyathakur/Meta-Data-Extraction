"""Format ingestion, page coverage and provider routing without live API calls."""

import io
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from metadata_extraction.api import create_app
from metadata_extraction.dataset import document_id, list_documents
from metadata_extraction.document_loader import TextBlock, load_bytes
from metadata_extraction.errors import DocumentError
from metadata_extraction.pipeline import ExtractionPipeline
from metadata_extraction.prompts import SYSTEM_PROMPT
from metadata_extraction.providers.compatible import CompatibleProvider
from metadata_extraction.providers.gemini import GeminiProvider


def photo_bytes():
    image = Image.new("RGB", (1000, 180), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except OSError:
        font = ImageFont.load_default(size=48)
    ImageDraw.Draw(image).text((30, 50), "MONTHLY RENT IS 8000", font=font, fill="black")
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    return stream.getvalue()


def pdf_bytes(kind="native"):
    with pymupdf.open() as pdf:
        if kind in {"native", "mixed"}:
            page = pdf.new_page(width=500, height=300)
            page.insert_text((30, 60), "Owner Example Owner. Monthly rent is 8000.")
        if kind in {"scanned", "mixed"}:
            page = pdf.new_page(width=500, height=90)
            page.insert_image(page.rect, stream=photo_bytes())
        return pdf.tobytes()


@pytest.mark.parametrize("suffix", ["jpg", "jpeg", "JPG"])
def test_jpeg_is_converted_to_provider_png(settings, suffix):
    document = load_bytes(photo_bytes(), f"agreement.{suffix}", settings)
    assert document.filename == f"agreement.{suffix}"
    assert document.images and not document.text_blocks
    for image in document.images:
        assert image.mime_type == "image/png"
        assert image.data.startswith(b"\x89PNG")


def test_jpeg_exif_orientation_is_applied(settings):
    stream = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (60, 100)).save(stream, format="JPEG", exif=exif)
    document = load_bytes(stream.getvalue(), "photo.jpg", settings)
    assert Image.open(io.BytesIO(document.images[0].data)).size == (100, 60)


@pytest.mark.parametrize("kind,pages", [("native", 1), ("scanned", 1), ("mixed", 2)])
def test_pdf_vision_preserves_every_page(settings, kind, pages):
    document = load_bytes(pdf_bytes(kind), "agreement.pdf", settings)
    assert {image.source_id.split(":")[0] for image in document.images} == {
        f"pdf-page-{i}" for i in range(1, pages + 1)
    }
    assert bool(document.text_blocks) == (kind != "scanned")
    if document.text_blocks:
        assert "8000" in document.text_blocks[0].text


def test_native_pdf_for_text_model_avoids_ocr(monkeypatch, settings):
    def unexpected(images):
        pytest.fail("Native PDF text should not require OCR")

    monkeypatch.setattr("metadata_extraction.ocr.transcribe_rapid", unexpected)
    document = load_bytes(pdf_bytes(), "native.pdf", replace(settings, model_supports_vision=False))
    assert not document.images
    assert "8000" in document.text_blocks[0].text


@pytest.mark.parametrize("kind", ["scanned", "mixed", "jpg", "jpeg"])
def test_text_model_routes_visual_content_through_ocr(monkeypatch, settings, kind):
    seen = []

    def transcript(images):
        seen.extend(images)
        return [TextBlock(part.source_id + ":ocr", "MONTHLY RENT IS 8000") for part in images]

    monkeypatch.setattr("metadata_extraction.ocr.transcribe_rapid", transcript)
    config = replace(settings, model_supports_vision=False)
    data = photo_bytes() if kind in {"jpg", "jpeg"} else pdf_bytes(kind)
    suffix = kind if kind in {"jpg", "jpeg"} else "pdf"
    document = load_bytes(data, f"agreement.{suffix}", config)
    assert seen and any(block.source_id.endswith(":ocr") for block in document.text_blocks)
    if kind == "mixed":
        assert {part.source_id.split(":")[0] for part in seen} == {"pdf-page-2"}
        assert any(block.source_id == "pdf-page-1:text" for block in document.text_blocks)
    config = replace(
        config,
        provider="openai-compatible",
        base_url="https://example.test/v1",
        llm_api_key="synthetic-key",
    )
    payload = CompatibleProvider(config)._payload(document, SYSTEM_PROMPT)
    assert "image_url" not in str(payload)
    assert "8000" in payload["messages"][1]["content"]


@pytest.mark.parametrize("suffix,data", [("pdf", b"bad"), ("jpg", b"bad"), ("gif", b"bad")])
def test_invalid_new_formats_are_rejected(settings, suffix, data):
    with pytest.raises(DocumentError):
        load_bytes(data, f"agreement.{suffix}", settings)


def test_mislabelled_png_as_jpeg_is_rejected(settings, png_bytes):
    with pytest.raises(DocumentError, match="extension"):
        load_bytes(png_bytes, "photo.jpg", settings)


def test_password_protected_pdf_is_rejected(settings):
    with pymupdf.open(stream=pdf_bytes(), filetype="pdf") as pdf:
        data = pdf.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="reader"
        )
    with pytest.raises(DocumentError, match="password protected"):
        load_bytes(data, "locked.pdf", settings)


def test_pdf_limits_reject_instead_of_partial_results(settings):
    with pytest.raises(DocumentError, match="MAX_IMAGE_PARTS"):
        load_bytes(pdf_bytes("mixed"), "mixed.pdf", replace(settings, max_image_parts=1))
    with pytest.raises(DocumentError, match="MAX_DOCUMENT_CHARS"):
        load_bytes(pdf_bytes(), "native.pdf", replace(settings, max_document_chars=10))
    with pymupdf.open() as pdf:
        pdf.new_page(width=10000, height=10000)
        data = pdf.tobytes()
    with pytest.raises(DocumentError, match="too large"):
        load_bytes(data, "huge.pdf", settings)


def test_pdf_filled_form_values_are_included(settings):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        widget = pymupdf.Widget()
        widget.field_name = "Monthly rent"
        widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
        widget.field_value = "8000"
        widget.rect = pymupdf.Rect(30, 30, 200, 60)
        page.add_widget(widget)
        data = pdf.tobytes()
    document = load_bytes(data, "form.pdf", replace(settings, model_supports_vision=False))
    assert any("8000" in block.text for block in document.text_blocks)


@pytest.mark.parametrize("suffix", ["pdf", "jpg", "jpeg"])
def test_new_formats_extract_through_api(settings, fake_provider, suffix):
    data = pdf_bytes("mixed") if suffix == "pdf" else photo_bytes()
    api = TestClient(create_app(settings, ExtractionPipeline(settings, fake_provider)))
    result = api.post("/extract?debug=true", files={"file": (f"agreement.{suffix}", data)})
    assert result.status_code == 200
    assert result.json()["filename"] == f"agreement.{suffix}"
    batch = api.post("/extract-batch", files=[("files", (f"agreement.{suffix}", data))])
    assert batch.status_code == 200 and "agreement," in batch.text
    assert fake_provider.calls == 1


def test_discovery_and_evaluation_identifiers_use_all_formats(tmp_path):
    for suffix in ("docx", "pdf", "png", "jpg", "jpeg"):
        name = f"agreement-{suffix}.{suffix.upper()}"
        (tmp_path / name).write_bytes(b"fixture")
        assert document_id(name) == f"agreement-{suffix}"
    (tmp_path / "ignored.gif").write_bytes(b"fixture")
    assert len(list_documents(tmp_path)) == 5
    assert document_id("original.pdf.docx") == "original"
    (tmp_path / "agreement-jpg.pdf").write_bytes(b"fixture")
    with pytest.raises(DocumentError, match="same identifier"):
        list_documents(tmp_path)


def test_discovery_ignores_word_lock_files_and_rejects_case_collisions(tmp_path):
    (tmp_path / "~$Agreement.docx").write_bytes(b"lock")
    (tmp_path / "Agreement.docx").write_bytes(b"fixture")
    assert [p.name for p in list_documents(tmp_path)] == ["Agreement.docx"]
    (tmp_path / "agreement.png").write_bytes(b"fixture")
    with pytest.raises(DocumentError, match="same identifier"):
        list_documents(tmp_path)


@pytest.mark.parametrize("kind", ["jpg", "scanned"])
def test_real_ocr_recognizes_new_image_formats(settings, kind):
    pytest.importorskip("rapidocr")
    data = photo_bytes() if kind == "jpg" else pdf_bytes("scanned")
    document = load_bytes(
        data,
        "agreement.jpg" if kind == "jpg" else "agreement.pdf",
        replace(settings, model_supports_vision=False),
    )
    assert any("8000" in block.text for block in document.text_blocks)


@pytest.mark.parametrize("vision", [True, False])
def test_gemini_receives_pdf_text_and_only_vision_images(
    monkeypatch, settings, missing_metadata, vision
):
    context = MagicMock()
    client = context.__enter__.return_value
    client.models.generate_content.return_value = SimpleNamespace(
        text=missing_metadata.model_dump_json(), model_version="fixture", usage_metadata=None
    )
    monkeypatch.setattr("metadata_extraction.providers.gemini.genai.Client", lambda **kw: context)
    monkeypatch.setattr(
        "metadata_extraction.ocr.transcribe_rapid",
        lambda images: [
            TextBlock(part.source_id + ":ocr", "MONTHLY RENT IS 8000") for part in images
        ],
    )
    config = replace(settings, api_key="synthetic-key", model_supports_vision=vision)
    document = load_bytes(pdf_bytes("mixed"), "agreement.pdf", config)
    GeminiProvider(config).extract(document, SYSTEM_PROMPT)
    parts = client.models.generate_content.call_args.kwargs["contents"][0].parts
    assert sum(part.inline_data is not None for part in parts) == (2 if vision else 0)
    assert "8000" in " ".join(part.text or "" for part in parts)


@pytest.mark.parametrize("suffix", ["pdf", "jpg"])
def test_frontend_new_format_preview_and_extraction(monkeypatch, settings, fake_provider, suffix):
    import httpx
    from streamlit.testing.v1 import AppTest

    from metadata_extraction.config import PROJECT_ROOT

    data = pdf_bytes("mixed") if suffix == "pdf" else photo_bytes()
    filename = f"agreement.{suffix}"
    record = ExtractionPipeline(settings, fake_provider).extract_bytes(
        data, filename, use_cache=False
    )
    result = record.model_dump(mode="json", exclude={"raw_response"})
    result["values"] = record.metadata.values()
    selected = [
        SimpleNamespace(name=filename, type="application/octet-stream", getvalue=lambda: data)
    ]
    allowed = []

    def uploader(*args, **kwargs):
        if kwargs.get("key") == "ground_truth_uploads":
            return []
        allowed.extend(kwargs["type"])
        return selected

    monkeypatch.setattr("streamlit.file_uploader", uploader)
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = httpx.Response(200, json=result)
    monkeypatch.setattr("httpx.Client", lambda **kwargs: client)
    application = AppTest.from_file(str(PROJECT_ROOT / "frontend" / "app.py")).run(timeout=15)
    assert set(allowed) == {"docx", "pdf", "png", "jpg", "jpeg"}
    next(c for c in application.checkbox if c.label == "Preview a document").check().run(timeout=15)
    assert not application.exception and not application.error
    if suffix == "pdf":
        assert "8000" in application.text_area[0].value
    next(b for b in application.button if b.label == "Extract details").click().run(timeout=15)
    assert not application.exception and not application.error
    assert application.session_state["run_document_ids"] == ["agreement"]
    assert application.session_state["results"][0]["filename"] == filename
    assert client.post.call_args.kwargs["files"]["file"][1] == data


def test_same_pdf_page_with_native_text_and_scan_uses_both(monkeypatch, settings):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=500, height=300)
        page.insert_text((30, 30), "Owner Example Owner")
        page.insert_image(pymupdf.Rect(0, 80, 500, 170), stream=photo_bytes())
        data = pdf.tobytes()
    monkeypatch.setattr(
        "metadata_extraction.ocr.transcribe_rapid",
        lambda images: [
            TextBlock(part.source_id + ":ocr", "MONTHLY RENT IS 8000") for part in images
        ],
    )
    document = load_bytes(data, "mixed.pdf", replace(settings, model_supports_vision=False))
    assert any(
        "Owner" in block.text and block.source_id.endswith(":text")
        for block in document.text_blocks
    )
    assert any(
        "8000" in block.text and block.source_id.endswith(":ocr") for block in document.text_blocks
    )
