"""Build a local draft/final ZIP from an explicit allowlist; never include local secrets."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from metadata_extraction.config import Settings
from metadata_extraction.dataset import document_id, list_documents
from metadata_extraction.errors import ExtractionError
from metadata_extraction.evaluator import EVALUATION_VERSION, SCORING_POLICY
from metadata_extraction.formats import SUPPORTED_EXTENSIONS
from metadata_extraction.pipeline import ExtractionPipeline
from metadata_extraction.schemas import AgreementMetadata

ROOT = Path(__file__).resolve().parents[1]


def submission_report_complete(root):
    try:
        metrics = json.loads((root / "outputs/submission_metrics.json").read_text(encoding="utf-8"))
        if (
            not metrics["ready"]
            or metrics.get("scoring_policy") != SCORING_POLICY
            or metrics.get("evaluation_version") != EVALUATION_VERSION
        ):
            return False
        required = {
            "data/train.csv", "data/test.csv", "data/results_corrected.xlsx",
            "outputs/train_predictions.csv", "outputs/train_predictions.records.json",
            "outputs/test_predictions.csv", "outputs/test_predictions.records.json",
        }
        required.update(
            path.relative_to(root).as_posix()
            for split in ("train", "test")
            for path in list_documents(root / "data" / split)
        )
        hashes = metrics["input_sha256"]
        if set(hashes) != required or any(
            hashlib.sha256((root / name).read_bytes()).hexdigest() != hashes[name]
            for name in required
        ):
            return False
        content = (root / "README.md").read_text(encoding="utf-8")
        start, end = "<!-- submission-results:start -->", "<!-- submission-results:end -->"
        if content.count(start) != 1 or content.count(end) != 1:
            return False
        section = content.split(start)[1].split(end)[0].strip()
        return hashlib.sha256(section.encode()).hexdigest() == metrics["readme_section_sha256"]
    except (OSError, ValueError, KeyError, TypeError, ExtractionError):
        return False


def test_run_complete(root, settings=None):
    try:
        labels = list(csv.DictReader((root / "data/test.csv").open(encoding="utf-8-sig")))
        predictions = list(
            csv.DictReader((root / "outputs/test_predictions.csv").open(encoding="utf-8-sig"))
        )
        records = json.loads(
            (root / "outputs/test_predictions.records.json").read_text(encoding="utf-8")
        )
        scores = json.loads(
            (root / "outputs/test_evaluation/evaluation_results.json").read_text(encoding="utf-8")
        )
        identifiers = [r["File Name"] for r in predictions]
        expected = {r["File Name"] for r in labels}
        documents = {document_id(p.name): p for p in list_documents(root / "data/test")}
        current_hash = ExtractionPipeline(settings or Settings.from_env())._configuration_hash()
        record_map = {r["document_id"]: r for r in records["records"]}
        if set(record_map) != expected or set(documents) != expected:
            return False
        for prediction in predictions:
            key = prediction["File Name"]
            record = record_map[key]
            if (
                record["document_sha256"] != hashlib.sha256(documents[key].read_bytes()).hexdigest()
                or record["configuration_hash"] != current_hash
                or AgreementMetadata.model_validate(record["metadata"]).csv_row(key) != prediction
            ):
                return False
        return (
            records["run"]["configuration_hash"] == current_hash
            and bool(expected)
            and set(identifiers) == expected
            and (
                len(identifiers) == len(expected)
                and not records["errors"]
                and len(records["records"]) == len(expected)
                and records["run"]["successful_documents"] == len(expected)
                and scores["run"] == records["run"]
                and scores.get("scoring_policy") == SCORING_POLICY
                and scores.get("evaluation_version") == EVALUATION_VERSION
                and scores.get("input_sha256")
                == {
                    "labels": hashlib.sha256((root / "data/test.csv").read_bytes()).hexdigest(),
                    "predictions": hashlib.sha256(
                        (root / "outputs/test_predictions.csv").read_bytes()
                    ).hexdigest(),
                    "records": hashlib.sha256(
                        (root / "outputs/test_predictions.records.json").read_bytes()
                    ).hexdigest(),
                }
                and scores["missing_prediction_rows"] == []
            )
        )
    except (OSError, ValueError, KeyError, TypeError, ExtractionError):
        return False


def build_submission(root=ROOT, final=False, settings=None):
    complete = test_run_complete(root, settings)
    if final and not complete:
        raise ValueError(
            "Final submission refused: test predictions/evaluation are incomplete or stale."
        )
    if final and not submission_report_complete(root):
        raise ValueError(
            "Final submission refused: README results are incomplete or stale. "
            "Run scripts/report_submission.py after completing extraction."
        )
    target = root / "outputs" / ("submission_final.zip" if final else "submission_draft.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in [
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements-ocr.txt",
        ".env.example",
        ".gitignore",
    ]:
        path = root / name
        if path.is_file() and not path.is_symlink():
            files[path] = name
    for directory in ["src", "frontend", "scripts", "tests", "data"]:
        suffixes = (
            SUPPORTED_EXTENSIONS | {".csv", ".xlsx", ".md"}
            if directory == "data" else {".py"}
        )
        for path in (root / directory).rglob("*"):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in suffixes:
                files[path] = path.relative_to(root).as_posix()
    for name in [
        "dataset_audit.json",
        "run_summary.json",
        "submission_metrics.json",
        "document_timings.json",
        "train_predictions.csv",
        "train_predictions.records.json",
    ]:
        path = root / "outputs" / name
        if path.is_file():
            files[path] = "outputs/" + name
    for directory in ["train_evaluation", "test_evaluation", "test_corrected_evaluation"]:
        for path in (root / "outputs" / directory).glob("*"):
            if path.is_file() and path.suffix in {".csv", ".json", ".xlsx"}:
                prefix = (
                    "outputs/"
                    if complete or directory == "train_evaluation"
                    else ("outputs/incomplete_test_attempt/")
                )
                files[path] = prefix + directory + "/" + path.name
    for name in ["test_predictions.csv", "test_predictions.records.json"]:
        path = root / "outputs" / name
        if path.is_file():
            if complete:
                files[path] = "outputs/" + name
            elif name.endswith(".records.json"):
                files[path] = "outputs/incomplete_test_attempt/" + name
    status = (
        "Final Extracted Metadata and evaluation are populated. See README.md."
        if final
        else "DRAFT ONLY. "
        + (
            "Test predictions are populated, but this archive is still a draft."
            if complete
            else "Final test predictions and meaningful test recall remain pending. "
            "Failure records are under outputs/incomplete_test_attempt; "
            "no final prediction CSV is included."
        )
    )
    manifest = io.StringIO()
    writer = csv.writer(manifest, lineterminator="\n")
    writer.writerow(["archive_path", "bytes"])
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for path, name in sorted(files.items(), key=lambda item: item[1]):
            archive.write(path, name)
            writer.writerow([name, path.stat().st_size])
        archive.writestr("SUBMISSION_STATUS.md", status + "\n")
        archive.writestr("MANIFEST.csv", manifest.getvalue())
    print(f"Saved {target.name}: {len(files)} files; test complete={complete}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    try:
        build_submission(final=args.final)
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")

