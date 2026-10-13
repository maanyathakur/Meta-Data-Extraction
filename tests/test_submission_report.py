import importlib.util
from dataclasses import replace
from zipfile import ZipFile

import pytest
from openpyxl import Workbook

from metadata_extraction.batch import predict_folder
from metadata_extraction.config import PROJECT_ROOT
from metadata_extraction.errors import DocumentError
from metadata_extraction.evaluator import csv_text, evaluate
from metadata_extraction.pipeline import ExtractionPipeline
from metadata_extraction.schemas import CSV_COLUMNS


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reporter = load_script("report_submission")
builder = load_script("build_submission")


def test_exact_recall_keeps_reference_formatting_and_counts_missing_rows():
    first = dict.fromkeys(CSV_COLUMNS, "")
    first.update({"File Name": "a", "Party One": "Mr. JANE DOE "})
    second = {**first, "File Name": "b"}
    outside = {**first, "File Name": "outside"}
    extracted = {**first, "Party One": "JANE DOE"}
    labels = [first, second, outside]
    report, _ = evaluate(labels, [extracted], document_ids=["a", "b"])
    exact = {column: score["strict"] for column, score in report["fields"].items()}
    assert exact["Party One"] == {"correct": 0, "incorrect": 2, "total": 2, "recall": 0}
    assert exact["Aggrement Value"]["recall"] == 0.5
    app, _ = evaluate(labels, [extracted], document_ids=["a", "b"])
    assert app["fields"]["Party One"]["normalized"]["recall"] == 0.5


def prepare_project(tmp_path, metadata, docx_bytes):
    (tmp_path / "data").mkdir()
    for split in ("train", "test"):
        folder = tmp_path / "data" / split
        folder.mkdir()
        (folder / f"{split}.docx").write_bytes(docx_bytes)
        (tmp_path / "data" / f"{split}.csv").write_text(
            csv_text([metadata.csv_row(split)]), encoding="utf-8"
        )
    book = Workbook()
    sheet = book.active
    sheet.append(CSV_COLUMNS)
    for split in ("train", "test"):
        row = metadata.csv_row(split)
        sheet.append([row[column] for column in CSV_COLUMNS])
    book.save(tmp_path / "data/results_corrected.xlsx")
    (tmp_path / "data/DATASET_NOTES.md").write_text("Reference notes", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "Instructions\n" + reporter.START + "\nPending\n" + reporter.END, encoding="utf-8"
    )


def test_missing_runs_remain_pending_without_fabricated_results(
    tmp_path, settings, missing_metadata, docx_bytes
):
    prepare_project(tmp_path, missing_metadata, docx_bytes)
    report = reporter.main(tmp_path, settings)
    content = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert not report["ready"]
    assert "Pending: 0/1 documents" in content
    assert "100.00%" not in content and "0.00%" not in content
    assert "quota" not in content
    assert not builder.submission_report_complete(tmp_path)


def test_complete_report_packaging_and_stale_input_checks(
    tmp_path, settings, fake_provider, docx_bytes
):
    from metadata_extraction.evaluator import evaluate_files

    prepare_project(tmp_path, fake_provider.metadata, docx_bytes)
    settings = replace(settings, output_dir=tmp_path / "outputs")
    pipeline = ExtractionPipeline(settings, fake_provider)
    for split in ("train", "test"):
        table = settings.output_dir / f"{split}_predictions.csv"
        predict_folder(pipeline, tmp_path / "data" / split, table)
        evaluate_files(
            tmp_path / "data" / f"{split}.csv", table,
            settings.output_dir / f"{split}_evaluation", table.with_suffix(".records.json")
        )
    report = reporter.main(tmp_path, settings)
    assert report["ready"]
    assert report["splits"]["test"]["scores"]["corrected"]["fields"]["Party One"]["strict"][
        "recall"
    ] == 1
    assert builder.submission_report_complete(tmp_path)
    archive = builder.build_submission(tmp_path, final=True, settings=settings)
    with ZipFile(archive) as package:
        assert "data/DATASET_NOTES.md" in package.namelist()
        assert "outputs/submission_metrics.json" in package.namelist()
        assert not any(name.startswith("docs/") for name in package.namelist())
    readme = tmp_path / "README.md"
    content = readme.read_text(encoding="utf-8")
    readme.write_text(content.replace("100.00%", "99.00%"), encoding="utf-8")
    assert not builder.submission_report_complete(tmp_path)
    readme.write_text(content, encoding="utf-8")
    document = tmp_path / "data/train/train.docx"
    document.write_bytes(docx_bytes + b"changed")
    assert not builder.submission_report_complete(tmp_path)
    with pytest.raises(DocumentError, match="no longer matches"):
        reporter.collect_results(tmp_path, settings)
