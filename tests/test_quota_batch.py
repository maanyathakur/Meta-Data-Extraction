from dataclasses import replace

from fastapi.testclient import TestClient

from metadata_extraction.api import create_app
from metadata_extraction.batch import predict_folder
from metadata_extraction.errors import ProviderQuotaError
from metadata_extraction.pipeline import ExtractionPipeline


class ExhaustedProvider:
    calls = 0

    def extract(self, document, prompt):
        self.calls += 1
        raise ProviderQuotaError(100)


def test_cli_batch_stops_after_all_slots_exhausted(settings, tmp_path, docx_bytes):
    folder = tmp_path / "inputs"
    folder.mkdir()
    for name in ["a.docx", "b.docx", "c.docx"]:
        (folder / name).write_bytes(docx_bytes)
    provider = ExhaustedProvider()
    payload = predict_folder(
        ExtractionPipeline(settings, provider), folder, settings.output_dir / "results.csv"
    )
    assert provider.calls == 1
    assert payload["run"]["stopped_for_quota"]
    assert payload["run"]["skipped_documents"] == 2
    assert len(payload["errors"]) == 3
    assert payload["errors"][1]["attempted"] is False


def test_api_quota_status_and_retry_header(settings, docx_bytes):
    provider = ExhaustedProvider()
    client = TestClient(create_app(settings, ExtractionPipeline(settings, provider)))
    response = client.post("/extract", files={"file": ("a.docx", docx_bytes)})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "100"
    assert "quota-limited" in response.json()["detail"]


def test_health_accepts_backup_only_and_never_reveals_keys(settings):
    settings = replace(settings, api_key_2="synthetic-backup-secret")
    response = TestClient(create_app(settings)).get("/health")
    assert response.json()["credentials_configured"] is True
    assert response.json()["key_slots"][1]["available"]
    assert "synthetic-backup-secret" not in response.text
