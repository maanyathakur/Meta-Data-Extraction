import hashlib

import pytest

from metadata_extraction.config import PROJECT_ROOT
from metadata_extraction.dataset import inspect_dataset
from metadata_extraction.document_loader import load_document

ROOT = PROJECT_ROOT / "data"
DOCUMENTS = sorted(ROOT.rglob("*.docx")) + sorted(ROOT.rglob("*.png"))


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: p.name)
def test_supplied_document_is_fully_ingested_without_modification(path, settings):
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    content = load_document(path, settings)
    assert content.text_blocks or content.images
    assert content.sha256 == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.skipif(not ROOT.exists(), reason="Supplied dataset not included in checkout")
def test_dataset_audit_identifies_known_mapping_issues():
    audit = inspect_dataset(ROOT)
    assert audit["splits"]["train"]["documents"] == 10
    assert audit["splits"]["test"]["unmatched_labels"] == []
    assert audit["label_overlap"] == ["24158401-Rental-Agreement"]
    assert len(audit["splits"]["train"]["invalid_date_labels"]) == 3
