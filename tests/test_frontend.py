import csv
import io
import socket
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import uvicorn
from streamlit.testing.v1 import AppTest

from metadata_extraction.api import create_app
from metadata_extraction.config import PROJECT_ROOT
from metadata_extraction.evaluator import csv_text
from metadata_extraction.pipeline import ExtractionPipeline


def uploads(docx_bytes, names):
    return [
        SimpleNamespace(name=name, type="application/octet-stream", getvalue=lambda: docx_bytes)
        for name in names
    ]


def app():
    return AppTest.from_file(str(PROJECT_ROOT / "frontend" / "app.py"), default_timeout=15)


def extract_button(application):
    return next(b for b in application.button if b.label == "Extract details")


def test_frontend_initial_render():
    app = AppTest.from_file(str(PROJECT_ROOT / "frontend" / "app.py")).run(timeout=15)
    assert not app.exception
    assert app.title[0].value == "Agreement Metadata Extraction"
    assert not any("Checks connectivity" in caption.value for caption in app.caption)
    assert next(button for button in app.button if button.label == "Extract details").disabled


def test_duplicate_upload_ids_disable_extraction(monkeypatch, docx_bytes):
    monkeypatch.setattr(
        "streamlit.file_uploader",
        lambda *a, **kw: uploads(docx_bytes, ["same.docx", "same.pdf.docx"]),
    )
    application = app().run(timeout=15)
    assert extract_button(application).disabled
    assert application.error


def test_preview_without_provider(monkeypatch, docx_bytes):
    monkeypatch.setattr(
        "streamlit.file_uploader", lambda *a, **kw: uploads(docx_bytes, ["preview.docx"])
    )
    application = app().run(timeout=15)
    next(c for c in application.checkbox if c.label == "Preview a document").check().run(timeout=15)
    assert not application.exception
    assert "Example Owner" in application.text_area[0].value


def test_preview_respects_configured_document_limit(monkeypatch, docx_bytes):
    monkeypatch.setenv("MAX_DOCUMENT_CHARS", "1")
    monkeypatch.setattr(
        "streamlit.file_uploader", lambda *a, **kw: uploads(docx_bytes, ["preview.docx"])
    )
    application = app().run(timeout=15)
    next(c for c in application.checkbox if c.label == "Preview a document").check().run(timeout=15)
    assert not application.exception
    assert any("MAX_DOCUMENT_CHARS" in message.value for message in application.error)


def test_frontend_timeout_allows_backend_retries(monkeypatch, docx_bytes):
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("MAX_ATTEMPTS", "3")
    monkeypatch.setattr("streamlit.file_uploader", lambda *a, **kw: uploads(docx_bytes, ["a.docx"]))
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = httpx.ConnectError("synthetic failure")
    captured = {}

    def create_client(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setattr("httpx.Client", create_client)
    application = app().run(timeout=15)
    extract_button(application).click().run(timeout=15)
    assert not application.exception
    # The UI must wait for three full provider attempts, even with a single key.
    assert captured["timeout"].read >= 900
    assert captured["timeout"].connect == 10


def test_partial_results_quota_stop_and_errors_persist(
    monkeypatch, settings, fake_provider, docx_bytes
):
    record = ExtractionPipeline(settings, fake_provider).extract_bytes(docx_bytes, "a.docx")
    result = record.model_dump(mode="json", exclude={"raw_response"})
    result["values"] = record.metadata.values()
    monkeypatch.setattr(
        "streamlit.file_uploader",
        lambda *a, **kw: uploads(docx_bytes, ["a.docx", "b.docx", "c.docx"]),
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = [
        httpx.Response(200, json=result),
        httpx.Response(429, json={"detail": "All slots quota-limited"}),
    ]
    monkeypatch.setattr("httpx.Client", lambda **kw: client)
    application = app().run(timeout=15)
    extract_button(application).click().run(timeout=15)
    assert not application.exception
    assert client.post.call_count == 2
    assert len(application.session_state["results"]) == 1
    assert len(application.session_state["failures"]) == 2
    assert "Skipped" in application.session_state["failures"][1]["error"]
    assert len(application.get("download_button")) == 3
    assert application.dataframe[0].value.iloc[0]["Agreement Value"] == ""
    next(c for c in application.checkbox if c.label == "Show source quotes").uncheck().run()
    assert len(application.error) == 2
    assert client.post.call_count == 2


def test_backend_connection_failure_shows_error_report(monkeypatch, docx_bytes):
    monkeypatch.setattr("streamlit.file_uploader", lambda *a, **kw: uploads(docx_bytes, ["a.docx"]))
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = httpx.ConnectError("synthetic failure")
    monkeypatch.setattr("httpx.Client", lambda **kw: client)
    application = app().run(timeout=15)
    extract_button(application).click().run(timeout=15)
    assert not application.exception
    assert not application.session_state["results"]
    assert len(application.error) == 1
    assert len(application.get("download_button")) == 1


def test_frontend_with_real_local_api(monkeypatch, settings, fake_provider, docx_bytes, png_bytes):
    data_dir = settings.output_dir.parent / "data"
    data_dir.mkdir()
    settings = replace(settings, api_key="synthetic-local-test-key", data_dir=data_dir)
    backend = create_app(settings, ExtractionPipeline(settings, fake_provider))
    connection = socket.socket()
    connection.bind(("127.0.0.1", 0))
    connection.listen(128)
    address = f"http://127.0.0.1:{connection.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(backend, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [connection]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        files = [
            ("files", ("agreement.docx", docx_bytes)),
            ("files", ("scan.png", png_bytes)),
        ]
        response = httpx.post(address + "/extract-batch", files=files, timeout=15)
        assert response.status_code == 200
        rows = list(csv.DictReader(io.StringIO(response.text)))
        assert [row["File Name"] for row in rows] == ["agreement", "scan"]
        labels = (
            response.text
            + csv_text([rows[0], {**rows[0], "File Name": "not_in_run"}]).split("\n", 1)[1]
        ).encode("utf-8")
        (data_dir / "labels.csv").write_bytes(labels)
        assert fake_provider.calls == 2
        monkeypatch.setenv("API_BASE_URL", address)
        selected = [
            SimpleNamespace(name=name, type="application/octet-stream", getvalue=lambda b=b: b)
            for name, b in [("agreement.docx", docx_bytes), ("scan.png", png_bytes)]
        ]
        monkeypatch.setattr(
            "streamlit.file_uploader",
            lambda *args, **kwargs: [] if kwargs.get("key") == "ground_truth_uploads" else selected,
        )
        application = app().run(timeout=15)
        next(b for b in application.button if b.label == "Check connection").click().run(timeout=15)
        assert application.success
        extract_button(application).click().run(timeout=15)
        assert not application.exception and not application.error
        assert len(application.session_state["results"]) == 2
        assert all(result["cached"] for result in application.session_state["results"])
        assert len(application.dataframe[0].value) == 2
        assert len(application.get("download_button")) == 2
        assert len(application.tabs) == 4
        assert fake_provider.calls == 2
        next(b for b in application.button if b.label.startswith("Find reference")).click().run()
        application.multiselect[0].select("labels.csv").run()
        # A presentation-only version difference must not block a compatible API report.
        monkeypatch.setattr(
            "metadata_extraction.evaluator.EVALUATION_VERSION", "name-comparison-v1"
        )
        next(b for b in application.button if b.label == "Evaluate results").click().run(timeout=20)
        assert not application.exception and not application.error
        assert (
            application.session_state["evaluation"]["report"]["evaluation_version"]
            == "name-comparison-v1"
        )
        assert application.session_state["evaluation"]["report"]["overall"]["strict"]["recall"] == 1
        assert {metric.label for metric in application.metric}.issuperset({
            "Strict recall", "Normalized recall", "Tolerant recall",
        })
        assert (
            application.session_state["evaluation"]["report"]["overall"]["labelled_documents"] == 2
        )
        assert (
            application.session_state["evaluation"]["report"]["overall"]["missing_documents"] == 0
        )
        assert len(application.get("download_button")) == 3
        assert fake_provider.calls == 2
        monkeypatch.setattr(
            "streamlit.file_uploader",
            lambda *args, **kwargs: (
                [
                    SimpleNamespace(
                        name="uploaded.csv",
                        getvalue=lambda: labels,
                    )
                ]
                if kwargs.get("key") == "ground_truth_uploads"
                else selected
            ),
        )
        application.multiselect[0].unselect("labels.csv").run()
        assert "evaluation" not in application.session_state
        next(b for b in application.button if b.label == "Evaluate results").click().run(timeout=20)
        assert not application.exception and not application.error
        assert application.session_state["evaluation"]["report"]["overall"]["strict"]["recall"] == 1
        assert fake_provider.calls == 2
        assert any(
            "Duplicate reference row ignored" in message.value
            and "agreement" in message.value
            and "Source:" in message.value
            for message in application.info
        )
        assert any(expander.label == "Scoring criteria" for expander in application.expander)
        assert not any(expander.label == "How scores work" for expander in application.expander)
        assert any(
            "- Only documents in the current run" in text.value for text in application.markdown
        )
        application.session_state["evaluation"]["report"].pop("evaluation_version")
        application.run()
        assert not application.exception and "evaluation" not in application.session_state
        assert fake_provider.calls == 2
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        connection.close()
    assert not thread.is_alive()


def test_frontend_evaluates_failed_uploads_and_excludes_extra_references(
    monkeypatch, settings, fake_provider, docx_bytes
):
    import json

    from fastapi.testclient import TestClient

    record = ExtractionPipeline(settings, fake_provider).extract_bytes(docx_bytes, "a.docx")
    result = record.model_dump(mode="json", exclude={"raw_response"})
    result["values"] = record.metadata.values()
    labels = csv_text([record.metadata.csv_row(key) for key in ("a", "failed", "outside")])
    reference = SimpleNamespace(name="reference.csv", getvalue=lambda: labels.encode())
    backend = TestClient(create_app(settings))
    monkeypatch.setattr(
        "streamlit.file_uploader",
        lambda *args, **kwargs: [reference] if kwargs.get("key") == "ground_truth_uploads"
        else uploads(docx_bytes, ["a.docx", "failed.docx"]),
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = [
        httpx.Response(200, json=result),
        httpx.Response(502, json={"detail": "Synthetic provider failure"}),
    ]
    monkeypatch.setattr("httpx.Client", lambda **kwargs: client)

    def evaluate_request(url, *, files, data, timeout):
        assert json.loads(data["document_ids"]) == ["a", "failed"]
        return backend.post("/evaluate", files=files, data=data)

    monkeypatch.setattr("httpx.post", evaluate_request)
    application = app().run(timeout=15)
    extract_button(application).click().run(timeout=15)
    next(b for b in application.button if b.label == "Evaluate results").click().run(timeout=15)
    assert not application.exception
    report = application.session_state["evaluation"]["report"]
    assert report["excluded_label_rows"] == 1
    assert report["overall"]["evaluated_documents"] == 2
    assert report["overall"]["missing_documents"] == 1
    metrics = {metric.label: metric.value for metric in application.metric}
    for label in ("Strict recall", "Normalized recall", "Tolerant recall"):
        assert metrics[label] == "50.0%"


def test_result_filters_and_changed_uploads_clear_stale_results(
    monkeypatch, settings, fake_provider, docx_bytes
):
    selected = uploads(docx_bytes, ["a.docx", "b.docx"])
    monkeypatch.setattr(
        "streamlit.file_uploader",
        lambda *a, **kw: [] if kw.get("key") == "ground_truth_uploads" else selected,
    )
    pipeline = ExtractionPipeline(settings, fake_provider)
    records = []
    for name in ("a.docx", "b.docx"):
        record = pipeline.extract_bytes(docx_bytes, name)
        payload = record.model_dump(mode="json", exclude={"raw_response"})
        payload["values"] = record.metadata.values()
        records.append(payload)
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = [httpx.Response(200, json=r) for r in records]
    monkeypatch.setattr("httpx.Client", lambda **kw: client)
    application = app().run(timeout=15)
    extract_button(application).click().run(timeout=15)
    assert len(application.dataframe[0].value) == 2
    assert "Status" not in application.dataframe[0].value.columns
    assert [metric.label for metric in application.metric] == ["Processed", "Not processed"]
    issues = application.dataframe[1].value
    assert list(issues.columns) == ["Document", "Field", "Issue", "Explanation"]
    assert len(issues) == 12 and set(issues["Issue"]) == {"Missing"}
    next(t for t in application.text_input if t.label == "Find a document").set_value("b").run()
    assert application.dataframe[0].value.iloc[0]["Document"] == "b"
    assert len(application.session_state["results"]) == 2
    assert client.post.call_count == 2
    next(t for t in application.text_input if t.label == "Find a document").set_value(
        "absent"
    ).run()
    assert not application.exception
    assert any("No documents match" in message.value for message in application.info)
    selected[:] = uploads(docx_bytes + b"changed", ["a.docx", "b.docx"])
    application.run()
    assert not application.exception
    assert "results" not in application.session_state
    assert "evaluation" not in application.session_state
    assert not application.dataframe
    assert client.post.call_count == 2


def test_images_are_not_flagged_and_ambiguous_fields_are_listed(
    monkeypatch, settings, fake_provider, png_bytes
):
    monkeypatch.setattr(
        "streamlit.file_uploader",
        lambda *args, **kwargs: (
            [] if kwargs.get("key") == "ground_truth_uploads" else uploads(png_bytes, ["scan.png"])
        ),
    )
    record = ExtractionPipeline(settings, fake_provider).extract_bytes(png_bytes, "scan.png")
    payload = record.model_dump(mode="json", exclude={"raw_response"})
    payload["values"] = record.metadata.values()
    assert not payload["warnings"]
    values = ("8000", "01.01.2020", "31.12.2020", "30", "Owner", "Tenant")
    for item, value in zip(payload["metadata"].values(), values, strict=True):
        item.update(
            status="found", value=value, provenance="explicit", explanation=None,
            evidence=[{"source_id": "image-1:y0-1800", "quote": "Example quotation"}],
        )
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = httpx.Response(200, json=payload)
    monkeypatch.setattr("httpx.Client", lambda **kwargs: client)
    application = app().run(timeout=15)
    extract_button(application).click().run(timeout=15)
    assert not application.exception
    assert len(application.dataframe) == 1
    assert not application.warning
    assert "Status" not in application.dataframe[0].value.columns
    assert [metric.label for metric in application.metric] == ["Processed", "Not processed"]
    payload["metadata"]["agreement_value"].update(
        status="ambiguous", value=None, provenance=None,
        explanation="Two different monthly rents are stated."
    )
    client.post.return_value = httpx.Response(200, json=payload)
    extract_button(application).click().run(timeout=15)
    assert not application.exception
    issues = application.dataframe[1].value
    assert len(issues) == 1
    assert issues.iloc[0].to_dict() == {
        "Document": "scan",
        "Field": "Agreement Value",
        "Issue": "Ambiguous",
        "Explanation": "Two different monthly rents are stated.",
    }


def test_preview_is_reused_until_input_or_settings_change(monkeypatch, docx_bytes):
    from metadata_extraction.document_loader import load_bytes

    loader = MagicMock(wraps=load_bytes)
    monkeypatch.setattr("metadata_extraction.document_loader.load_bytes", loader)
    selected = uploads(docx_bytes, ["preview.docx"])
    monkeypatch.setattr("streamlit.file_uploader", lambda *a, **kw: selected)
    application = app().run()
    next(c for c in application.checkbox if c.label == "Preview a document").check().run()
    application.run()
    assert loader.call_count == 1
    monkeypatch.setenv("MAX_DOCUMENT_CHARS", "2")
    application.run()
    assert loader.call_count == 2
    assert any("MAX_DOCUMENT_CHARS" in error.value for error in application.error)


def test_case_insensitive_duplicates_and_upload_limit(monkeypatch, docx_bytes):
    selected = uploads(docx_bytes, ["Same.docx", "same.pdf.docx"])
    monkeypatch.setattr("streamlit.file_uploader", lambda *a, **kw: selected)
    application = app().run()
    assert extract_button(application).disabled
    assert any("same document ID" in error.value for error in application.error)
    selected[:] = uploads(docx_bytes, ["a.docx", "b.docx"])
    monkeypatch.setenv("MAX_BATCH_FILES", "1")
    application.run()
    assert extract_button(application).disabled
    assert any("at most 1" in error.value for error in application.error)


def test_malformed_success_response_becomes_actionable_failure(monkeypatch, docx_bytes):
    monkeypatch.setattr("streamlit.file_uploader", lambda *a, **kw: uploads(docx_bytes, ["a.docx"]))
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = httpx.Response(200, json={"values": {}})
    monkeypatch.setattr("httpx.Client", lambda **kwargs: client)
    application = app().run()
    extract_button(application).click().run()
    assert not application.exception and not application.session_state["results"]
    assert "invalid result" in application.session_state["failures"][0]["error"]


