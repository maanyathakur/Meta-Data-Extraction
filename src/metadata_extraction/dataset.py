import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

from .errors import DocumentError
from .formats import SUPPORTED_EXTENSIONS
from .schemas import CSV_COLUMNS, LEGACY_NOTICE_COLUMN


def document_id(filename: str) -> str:
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        name = name[: -len(suffix)]
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name


def evaluation_id(value: str) -> str:
    """Canonical identity shared by discovery, uploads and reference matching."""
    return document_id(value.strip().replace("\\", "/")).strip().casefold()


def list_documents(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise DocumentError(f"Document directory does not exist: {folder}")
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith("~$")
    )
    identifiers = [evaluation_id(p.name) for p in files]
    if len(identifiers) != len(set(identifiers)):
        raise DocumentError("Multiple documents map to the same identifier.")
    return files


def read_labels(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames:
                reader.fieldnames = [
                    "Termination Notice (Days)" if c == LEGACY_NOTICE_COLUMN else c
                    for c in reader.fieldnames
                ]
            if reader.fieldnames != CSV_COLUMNS:
                raise DocumentError(f"Unexpected label columns in {path.name}.")
            rows = list(reader)
    except OSError as exc:
        raise DocumentError(f"Cannot read labels: {path}") from exc
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise DocumentError("Malformed CSV row; verify quoting and number of columns.")
    ids = [row["File Name"] for row in rows]
    if len(ids) != len(set(ids)):
        raise DocumentError("Duplicate label identifiers.")
    return rows


def inspect_dataset(root: Path) -> dict:
    report = {"splits": {}}
    all_ids = {}
    for split in ("train", "test"):
        paths = list_documents(root / split)
        rows = read_labels(root / f"{split}.csv")
        file_ids = {document_id(p.name) for p in paths}
        label_ids = {row["File Name"] for row in rows}
        invalid_dates, whitespace, missing = [], [], []
        for row in rows:
            for column, value in row.items():
                info = {"document_id": row["File Name"], "field": column, "value": value}
                if not value:
                    missing.append(info)
                if value != value.strip():
                    whitespace.append(info)
                if column in {"Aggrement Start Date", "Aggrement End Date"} and value.strip():
                    try:
                        datetime.strptime(value.strip(), "%d.%m.%Y")
                    except ValueError:
                        invalid_dates.append(info)
        report["splits"][split] = {
            "documents": len(paths),
            "label_rows": len(rows),
            "formats": dict(Counter(p.suffix.lower() for p in paths)),
            "unmatched_labels": sorted(label_ids - file_ids),
            "unlabelled_documents": sorted(file_ids - label_ids),
            "invalid_date_labels": invalid_dates,
            "whitespace_labels": whitespace,
            "missing_labels": missing,
        }
        all_ids[split] = label_ids
    report["label_overlap"] = sorted(all_ids["train"] & all_ids["test"])
    report["notes"] = [
        "Audit never changes documents or labels.",
        "Filename matching removes document extensions and the optional .pdf suffix only.",
        "Consult data/DATASET_NOTES.md for manually verified content anomalies "
        "and related contracts.",
    ]
    return report
