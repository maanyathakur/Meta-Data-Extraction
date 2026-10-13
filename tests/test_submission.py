import importlib.util
from zipfile import ZipFile

import pytest

from metadata_extraction.config import PROJECT_ROOT

spec = importlib.util.spec_from_file_location(
    "submission_builder", PROJECT_ROOT / "scripts/build_submission.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_draft_excludes_secrets_and_runtime_artifacts(tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=private-key")
    (tmp_path / ".env.example").write_text("GEMINI_API_KEY=")
    (tmp_path / "README.md").write_text("Instructions")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/DATASET_NOTES.md").write_text("Reference corrections")
    (tmp_path / "old_results").mkdir()
    (tmp_path / "old_results/old_report.xlsx").write_bytes(b"historical-artifact")
    for directory in ["src", ".venv", ".cache"]:
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "file.py").write_text("pass")
    archive = builder.build_submission(tmp_path)
    with ZipFile(archive) as zipfile:
        names = zipfile.namelist()
        assert ".env" not in names and ".env.example" in names
        assert "data/DATASET_NOTES.md" in names
        assert not any(name.startswith("old_results/") for name in names)
        assert not any(name.startswith((".venv/", ".cache/")) for name in names)
        assert "DRAFT ONLY" in zipfile.read("SUBMISSION_STATUS.md").decode()
        assert "private-key" not in b"".join(zipfile.read(n) for n in names).decode()


def test_final_package_refuses_incomplete_test_run(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        builder.build_submission(tmp_path, final=True)
    assert not (tmp_path / "outputs/submission_final.zip").exists()


def test_final_requires_fresh_scores(tmp_path, settings, fake_provider, docx_bytes):
    from dataclasses import replace

    from metadata_extraction.batch import predict_folder
    from metadata_extraction.evaluator import csv_text, evaluate_files
    from metadata_extraction.pipeline import ExtractionPipeline

    folder = tmp_path / "data/test"
    folder.mkdir(parents=True)
    (folder / "a.docx").write_bytes(docx_bytes)
    settings = replace(settings, output_dir=tmp_path / "outputs")
    pipeline = ExtractionPipeline(settings, fake_provider)
    predictions = settings.output_dir / "test_predictions.csv"
    predict_folder(pipeline, folder, predictions)
    labels = tmp_path / "data/test.csv"
    labels.write_text(csv_text([fake_provider.metadata.csv_row("a")]), encoding="utf-8")
    records = predictions.with_suffix(".records.json")
    evaluate_files(labels, predictions, settings.output_dir / "test_evaluation", records)
    assert builder.test_run_complete(tmp_path, settings)
    with pytest.raises(ValueError, match="README results"):
        builder.build_submission(tmp_path, final=True, settings=settings)
    import json

    score_path = settings.output_dir / "test_evaluation/evaluation_results.json"
    current_scores = score_path.read_text(encoding="utf-8")
    stale_scores = json.loads(current_scores)
    stale_scores.pop("scoring_policy")
    score_path.write_text(json.dumps(stale_scores), encoding="utf-8")
    assert not builder.test_run_complete(tmp_path, settings)
    score_path.write_text(current_scores, encoding="utf-8")
    stale_scores = json.loads(current_scores)
    stale_scores["evaluation_version"] = "old-version"
    score_path.write_text(json.dumps(stale_scores), encoding="utf-8")
    assert not builder.test_run_complete(tmp_path, settings)
    score_path.write_text(current_scores, encoding="utf-8")
    assert not builder.test_run_complete(tmp_path, replace(settings, model="changed-model"))
    document = folder / "a.docx"
    document.write_bytes(docx_bytes + b"changed")
    assert not builder.test_run_complete(tmp_path, settings)
    document.write_bytes(docx_bytes)
    predictions.write_text(predictions.read_text() + "\n", encoding="utf-8")
    assert not builder.test_run_complete(tmp_path, settings)
    with pytest.raises(ValueError, match="incomplete"):
        builder.build_submission(tmp_path, final=True)
