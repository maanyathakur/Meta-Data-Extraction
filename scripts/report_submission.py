"""Refresh README result tables from saved extractions, without model calls."""

import hashlib
import json

from metadata_extraction.config import PROJECT_ROOT, Settings
from metadata_extraction.dataset import document_id, list_documents
from metadata_extraction.errors import DocumentError
from metadata_extraction.evaluation_io import read_table
from metadata_extraction.evaluator import (
    EVALUATION_VERSION,
    RECALL_TIERS,
    SCORING_POLICY,
    evaluate,
    evaluation_id,
    matched_rows,
)
from metadata_extraction.pipeline import ExtractionPipeline, write_json
from metadata_extraction.schemas import CSV_COLUMNS, AgreementMetadata

START = "<!-- submission-results:start -->"
END = "<!-- submission-results:end -->"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_results(root, settings):
    references = {
        "corrected": root / "data/results_corrected.xlsx",
        "train": root / "data/train.csv",
        "test": root / "data/test.csv",
    }
    labels = {name: read_table(path.read_bytes(), path.name) for name, path in references.items()}
    hashes = {path.relative_to(root).as_posix(): file_hash(path) for path in references.values()}
    configuration = ExtractionPipeline(settings)._configuration_hash()
    splits = {}
    for split in ("train", "test"):
        documents = {evaluation_id(p.name): p for p in list_documents(root / "data" / split)}
        hashes.update({p.relative_to(root).as_posix(): file_hash(p) for p in documents.values()})
        table = root / "outputs" / f"{split}_predictions.csv"
        records_path = table.with_suffix(".records.json")
        rows, payload = [], None
        if table.exists() and records_path.exists():
            rows = read_table(table.read_bytes(), table.name, allow_empty=True)
            payload = json.loads(records_path.read_text(encoding="utf-8"))
            hashes.update({
                p.relative_to(root).as_posix(): file_hash(p) for p in (table, records_path)
            })
            records = {evaluation_id(r["document_id"]): r for r in payload["records"]}
            by_id, _ = matched_rows(rows, predictions=True)
            run_scope = {
                evaluation_id(r["document_id"])
                for r in payload["records"] + payload["errors"]
            }
            if (
                run_scope != set(documents)
                or set(records) != set(by_id)
                or payload["run"]["configuration_hash"] != configuration
                or payload["run"]["total_documents"] != len(documents)
                or payload["run"]["successful_documents"] != len(records)
                or payload["run"]["failed_documents"] != len(payload["errors"])
            ):
                raise DocumentError(f"{split}: saved run is incomplete or stale; rerun extraction.")
            for key, record in records.items():
                expected_row = AgreementMetadata.model_validate(record["metadata"]).csv_row(
                    document_id(documents[key].name)
                )
                if (
                    record["configuration_hash"] != configuration
                    or record["document_sha256"] != file_hash(documents[key])
                    or any(expected_row[c] != by_id[key][c] for c in CSV_COLUMNS[1:])
                ):
                    raise DocumentError(f"{split}: saved result no longer matches {key}.")
        complete = bool(payload) and not payload["errors"] and len(rows) == len(documents)
        scores = {}
        for source in ("original", "corrected"):
            reference = labels[split] if source == "original" else labels["corrected"]
            app_report, _ = evaluate(reference, rows, document_ids=list(documents))
            scores[source] = {
                "fields": app_report["fields"],
                "reference_documents": app_report["label_rows"],
                "evaluated_documents": app_report["overall"]["evaluated_documents"],
                "unlabelled_documents": app_report["overall"]["unlabelled_documents"],
            }
        splits[split] = {
            "complete": complete,
            "documents": len(documents),
            "results": rows,
            "run": payload["run"] if payload else None,
            "scores": scores,
        }
    return {
        "evaluation_version": EVALUATION_VERSION,
        "scoring_policy": SCORING_POLICY,
        "ready": all(s["complete"] for s in splits.values()),
        "input_sha256": hashes,
        "splits": splits,
    }


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_results(report):
    lines = []
    if not report["ready"]:
        lines.append(
            "**Final extraction results are pending.** "
            "Incomplete runs are not reported as extraction-quality scores. Rerun the batch "
            "and refresh this section with the commands above."
        )
        if any(s["run"] and s["run"].get("stopped_for_quota") for s in report["splits"].values()):
            lines.append("\nThe last batch attempt stopped because provider quota was exhausted.")
    for split in ("test", "train"):
        data = report["splits"][split]
        lines.extend(["", f"### {split.title()} results", ""])
        if not data["complete"]:
            lines.append(
                f"Pending: {len(data['results'])}/{data['documents']} documents extracted."
            )
            continue
        run = data["run"]
        lines.append(
            f"{len(data['results'])}/{data['documents']} documents extracted. "
            f"Provider: `{run['provider']}`. Model: `{run['model']}`. "
            f"Input: `{run['effective_image_input']}`. OCR: `{run['effective_ocr_mode']}`. "
            f"Run timestamp (UTC): `{run['created_at']}`."
        )
        if split == "test":
            lines.extend([
                "", "| Document | Rent | Start | End | Notice (days) | Party One | Party Two |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ])
            for row in data["results"]:
                lines.append("| " + " | ".join(markdown_cell(row[c]) for c in CSV_COLUMNS) + " |")
        for source in ("original", "corrected"):
            lines.extend([
                "", f"**{source.title()} reference answers**", "",
                "| Field | Strict recall | Normalized recall | Tolerant recall |",
                "| --- | --- | --- | --- |",
            ])
            for column in CSV_COLUMNS[1:]:
                values = []
                for tier in RECALL_TIERS:
                    metric = data["scores"][source]["fields"][column][tier]
                    score = metric["recall"]
                    values.append(f"{metric['correct']}/{metric['total']} ({score:.2%})"
                                  if score is not None else "N/A")
                cells = [column.replace("Aggrement", "Agreement"), *values]
                lines.append("| " + " | ".join(cells) + " |")
        count = data["scores"]["original"]["reference_documents"]
        lines.extend(["", f"Original reference coverage: {count}/{data['documents']} documents. "
                      "Documents without reference answers cannot be scored."])
    return "\n".join(lines).strip()


def main(root=PROJECT_ROOT, settings=None):
    report = collect_results(root, settings or Settings.from_env())
    path = root / "README.md"
    content = path.read_text(encoding="utf-8")
    if content.count(START) != 1 or content.count(END) != 1:
        raise DocumentError("README must contain one submission-results marker pair.")
    before, remainder = content.split(START)
    _, after = remainder.split(END)
    section = render_results(report)
    path.write_text(before + START + "\n\n" + section + "\n\n" + END + after, encoding="utf-8")
    report["readme_section_sha256"] = hashlib.sha256(section.encode()).hexdigest()
    write_json(root / "outputs/submission_metrics.json", report)
    print(f"README results refreshed; complete={report['ready']}. No model calls made.")
    return report


if __name__ == "__main__":
    main()
