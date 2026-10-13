from fastapi.testclient import TestClient

from metadata_extraction.api import create_app
from metadata_extraction.pipeline import ExtractionPipeline


def client(settings, fake_provider):
    return TestClient(create_app(settings, ExtractionPipeline(settings, fake_provider)))


def test_health_without_key(settings, fake_provider):
    response = client(settings, fake_provider).get("/health")
    assert response.status_code == 200
    assert response.json()["credentials_configured"] is False
    assert fake_provider.calls == 0


def test_single_and_debug_outputs(settings, fake_provider, docx_bytes):
    api = client(settings, fake_provider)
    response = api.post("/extract", files={"file": ("a.docx", docx_bytes)})
    assert response.status_code == 200
    assert len(response.json()) == 6
    assert response.json()["party_one"] is None
    debug = api.post("/extract?debug=true", files={"file": ("a.docx", docx_bytes)})
    assert debug.status_code == 200
    assert "raw_response" not in debug.json()
    assert debug.json()["cached"] is True


def test_batch_returns_csv(settings, fake_provider, docx_bytes, png_bytes):
    response = client(settings, fake_provider).post(
        "/extract-batch",
        files=[
            ("files", ("a.docx", docx_bytes)),
            ("files", ("b.png", png_bytes)),
        ],
    )
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "Extracted Metadata.csv" in response.headers["content-disposition"]
    assert len(response.text.splitlines()) == 3


def test_invalid_batch_spends_no_provider_calls(settings, fake_provider, docx_bytes):
    response = client(settings, fake_provider).post(
        "/extract-batch",
        files=[
            ("files", ("a.docx", docx_bytes)),
            ("files", ("bad.pdf", b"invalid")),
        ],
    )
    assert response.status_code == 422
    assert fake_provider.calls == 0


def test_missing_credentials_returns_503(settings, docx_bytes):
    response = TestClient(create_app(settings)).post(
        "/extract", files={"file": ("a.docx", docx_bytes)}
    )
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_upload_size_limit(settings, fake_provider):
    from dataclasses import replace

    settings = replace(settings, max_upload_mb=1)
    response = client(settings, fake_provider).post(
        "/extract", files={"file": ("large.docx", b"x" * (1024 * 1024 + 1))}
    )
    assert response.status_code == 413
    assert fake_provider.calls == 0


def test_batch_duplicate_ids_and_count_limit(settings, fake_provider, docx_bytes):
    from dataclasses import replace

    api = client(replace(settings, max_batch_files=2), fake_provider)
    duplicate = api.post(
        "/extract-batch",
        files=[("files", ("same.docx", docx_bytes)), ("files", ("same.pdf.docx", docx_bytes))],
    )
    assert duplicate.status_code == 422
    too_many = api.post(
        "/extract-batch", files=[("files", (f"{i}.docx", docx_bytes)) for i in range(3)]
    )
    assert too_many.status_code == 422
    assert fake_provider.calls == 0


def test_case_insensitive_batch_ids_are_rejected_before_extraction(
    settings, fake_provider, docx_bytes
):
    response = client(settings, fake_provider).post(
        "/extract-batch",
        files=[("files", ("Same.docx", docx_bytes)), ("files", ("same.pdf.docx", docx_bytes))],
    )
    assert response.status_code == 422 and fake_provider.calls == 0

