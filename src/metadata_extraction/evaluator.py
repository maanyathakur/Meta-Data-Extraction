import csv
import hashlib
import io
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .dataset import evaluation_id
from .errors import DocumentError
from .evaluation_io import MAX_ROWS, excel_report, read_table
from .name_matching import NAME_COLUMNS, clean_name, fuzzy_conflicts, name_outcome
from .schemas import CSV_COLUMNS, FIELD_COLUMNS

EVALUATION_VERSION = "recall-v5-three-tiers"
SCORING_POLICY = "three-tiers-all-labelled-run-documents-v1"
RECALL_TIERS = ("strict", "normalized", "tolerant")


def recall_score(correct: int, total: int) -> dict:
    return {
        "correct": correct, "incorrect": total - correct, "total": total,
        "recall": correct / total if total else None,
    }


def normalized(value: str) -> str:
    return " ".join(value.split())


def comparable(value: str, column: str) -> tuple[str, bool]:
    if column in NAME_COLUMNS:
        cleaned = clean_name(value)
        return cleaned, not value.strip() or bool(cleaned)
    text = normalized(value).casefold()
    if not text:
        return "", True
    if column in {"Aggrement Start Date", "Aggrement End Date"}:
        for pattern in ("%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, pattern).date().isoformat(), True
            except ValueError:
                continue
        return text, False
    if column in {"Aggrement Value", "Termination Notice (Days)"}:
        try:
            number_text = text
            if "," in text:
                groups = text.split(".")[0].split(",")
                if not (1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])):
                    return text, False
                number_text = text.replace(",", "")
            number = Decimal(number_text)
            if not number.is_finite() or number < 0:
                return text, False
            if abs(number.adjusted()) > 1000:
                return text, False
            if column == "Termination Notice (Days)" and number != number.to_integral_value():
                return text, False
            canonical = format(number, "f")
            return canonical.rstrip("0").rstrip(".") if "." in canonical else canonical, True
        except InvalidOperation:
            return text, False
    return text, True


def matched_rows(rows, predictions=False):
    by_id, issues = {}, []
    if len(rows) > MAX_ROWS:
        raise DocumentError(f"Evaluation exceeds {MAX_ROWS} rows.")
    for original in rows:
        row = dict(original)
        key = evaluation_id(row["File Name"])
        if not key:
            raise DocumentError("Missing document identifier.")
        row["File Name"] = key
        if key in by_id:
            if predictions:
                raise DocumentError("Duplicate prediction identifiers.")
            same = all(row[c] == by_id[key][c] for c in CSV_COLUMNS[1:])
            if not same:
                raise DocumentError(
                    f"Conflicting ground truth for {key}: "
                    f"{by_id[key].get('ground_truth_source', 'first row')} and "
                    f"{row.get('ground_truth_source', 'duplicate row')}. "
                    "Select non-conflicting files or correct the labels."
                )
            issues.append(
                {
                    "type": "identical duplicate label",
                    "filename": key,
                    "details": row.get("ground_truth_source", "duplicate row"),
                }
            )
            continue
        by_id[key] = row
    return by_id, issues


def csv_text(rows: list[dict], columns=CSV_COLUMNS) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def evaluate(
    labels: list[dict], predictions: list[dict], evidence=None, document_ids=None
) -> tuple[dict, list]:
    scope = None
    excluded_label_rows = 0
    if document_ids is not None:
        if (
            not isinstance(document_ids, list)
            or not document_ids
            or len(document_ids) > MAX_ROWS
            or any(not isinstance(value, str) or not evaluation_id(value) for value in document_ids)
        ):
            raise DocumentError(
                "document_ids must be a nonempty list of document filenames or IDs."
            )
        scope = {evaluation_id(value) for value in document_ids}
        selected_labels = [row for row in labels if evaluation_id(row["File Name"]) in scope]
        excluded_label_rows = len(labels) - len(selected_labels)
        labels = selected_labels
        if any(evaluation_id(row["File Name"]) not in scope for row in predictions):
            raise DocumentError("Predictions include documents outside the selected run.")
    by_id, _ = matched_rows(predictions, predictions=True)
    label_map, issues = matched_rows(labels)
    labels = list(label_map.values())
    evaluated_documents = len(label_map)
    completed_documents = len(set(label_map) & set(by_id))
    evidence = {evaluation_id(k): v for k, v in (evidence or {}).items()}
    details = []
    scores = {}
    for name, column in FIELD_COLUMNS.items():
        counts = dict.fromkeys(RECALL_TIERS, 0)
        reviews = 0
        for label in labels:
            identifier = label["File Name"]
            expected = label[column]  # Intentionally retain original whitespace/case/errors.
            supplied = by_id.get(identifier)
            predicted = supplied.get(column, "") if supplied is not None else None
            field = evidence.get(identifier, {}).get(name, {})
            exact = predicted is not None and predicted == expected
            clean_expected, valid_expected = comparable(expected, column)
            clean_predicted, valid_predicted = comparable(predicted or "", column)
            match = exact or (
                supplied is not None
                and valid_expected
                and valid_predicted
                and (clean_expected == clean_predicted)
            )
            outcome = "Match" if match else "Mismatch"
            similarity, initial_conflict = None, False
            if column in NAME_COLUMNS and supplied is not None:
                outcome, similarity, initial_conflict = name_outcome(
                    clean_expected, clean_predicted
                )
                if not valid_expected or not valid_predicted:
                    outcome = "Mismatch"
            if supplied is None:
                outcome = "Missing document"
            elif not valid_expected or not valid_predicted:
                outcome = "Invalid value"
            accepted = match or outcome == "Fuzzy match"
            conflicts = (
                fuzzy_conflicts(clean_expected, clean_predicted)
                if column in NAME_COLUMNS and supplied is not None and not match else []
            )
            if exact:
                outcome = "Strict match"
            elif match:
                outcome = "Normalized match"
            reviews += int(outcome == "Review")
            for kind, value, valid in (
                ("invalid ground truth", expected, valid_expected),
                ("invalid prediction", predicted, valid_predicted),
            ):
                if not valid:
                    issues.append(
                        {"type": kind, "filename": identifier, "field": column, "details": value}
                    )
            for tier, matched in zip(RECALL_TIERS, (exact, match, accepted), strict=True):
                counts[tier] += int(matched)
            details.append(
                {
                    "filename": identifier,
                    "field": column,
                    "ground_truth": expected,
                    "ground_truth_source": label.get("ground_truth_source", ""),
                    "prediction": predicted if predicted is not None else "",
                    "prediction_row_missing": supplied is None,
                    "excluded_from_metrics": False,
                    "evidence": json.dumps(field.get("evidence", []), ensure_ascii=False),
                    "provenance": field.get("provenance", ""),
                    "status": field.get("status", ""),
                    "exact_match": exact,
                    "normalized_ground_truth": clean_expected,
                    "normalized_prediction": clean_predicted,
                    "normalized_match": match,
                    "tolerant_match": accepted,
                    "match": accepted,
                    "comparison_status": outcome,
                    "fuzzy_score": similarity,
                    "conflicting_initials": initial_conflict,
                    "fuzzy_conflicts": conflicts,
                }
            )
        total = evaluated_documents
        scores[column] = {
            "total": total,
            **{tier: recall_score(counts[tier], total) for tier in RECALL_TIERS},
            "review_count": reviews,
        }
    report = {
        "evaluation_version": EVALUATION_VERSION,
        "requested_document_ids": sorted(scope) if scope is not None else None,
        "excluded_label_rows": excluded_label_rows,
        "metric": "per-field recall",
        "scoring_policy": SCORING_POLICY,
        "fields": scores,
        "label_rows": len(labels),
        "prediction_rows": len(predictions),
        "missing_prediction_rows": [r["File Name"] for r in labels if r["File Name"] not in by_id],
        "extra_prediction_rows": sorted(set(by_id) - {r["File Name"] for r in labels}),
        "empty_ground_truth_policy": (
            "Included as supplied. Empty prediction matches empty label only when its row exists."
        ),
    }
    for key in report["missing_prediction_rows"]:
        issues.append({"type": "missing prediction", "filename": key})
    for key in report["extra_prediction_rows"]:
        issues.append(
            {"type": "unlabelled prediction", "filename": key, "details": "Excluded from metrics."}
        )
        for column in CSV_COLUMNS[1:]:
            details.append(
                {
                    "filename": key,
                    "field": column,
                    "ground_truth": "",
                    "prediction": by_id[key][column],
                    "excluded_from_metrics": True,
                    "comparison_status": "unlabelled; excluded from metrics",
                }
            )
    if scope is not None:
        for key in sorted(scope - set(label_map) - set(by_id)):
            issues.append(
                {
                    "type": "missing ground truth",
                    "filename": key,
                    "details": "No reference answer for this run document; excluded from metrics.",
                }
            )
            for column in CSV_COLUMNS[1:]:
                details.append({
                    "filename": key, "field": column, "ground_truth": "", "prediction": "",
                    "excluded_from_metrics": True,
                    "comparison_status": "unlabelled; excluded from metrics",
                })
    total_fields = evaluated_documents * len(FIELD_COLUMNS)
    report["overall"] = {
        "labelled_documents": len(labels),
        "evaluated_documents": evaluated_documents,
        "completed_documents": completed_documents,
        "run_documents": len(scope if scope is not None else set(label_map) | set(by_id)),
        "prediction_documents": len(by_id),
        "evaluated_fields": total_fields,
        "review_fields": sum(s["review_count"] for s in scores.values()),
        **{
            tier: recall_score(sum(s[tier]["correct"] for s in scores.values()), total_fields)
            for tier in RECALL_TIERS
        },
        "missing_documents": len(report["missing_prediction_rows"]),
        "unlabelled_documents": len((scope if scope is not None else set(by_id)) - set(label_map)),
    }
    report["definitions"] = {
        "recall": "Each recall = correct / all labelled documents in the selected run. "
        "All three tiers use the same denominator. Overall scores pool the six fields. "
        "Two empty values match only when an extracted row exists.",
        "strict": "Strict recall uses literal equality, including case, spaces and punctuation. "
        "It measures agreement with the reference, not the validity of the reference itself.",
        "normalized": "Normalized recall includes strict matches and equality after the "
        "documented formatting normalization. Different spellings are not accepted here.",
        "tolerant": "Tolerant recall includes normalized matches plus guarded fuzzy name "
        "matches. Review cases receive no automatic credit.",
        "missing": "Failed/skipped documents with reference answers count as incorrect in all "
        "three tiers, even when the reference value is empty.",
        "scope": "When a run is selected, only reference rows for its documents are scored. "
        "Failed or skipped documents with references remain in the denominator. "
        "Other reference "
        "rows are excluded. Run documents without ground truth cannot be scored.",
        "normalization": "Names ignore case, repeated spaces, formatting punctuation and leading "
        "personal titles. Letters, initials, numbers and joint-party separators are retained; "
        "titles are retained for names with company markers such as Ltd/LLC. "
        "SRI is always retained because it may be part of a company name. "
        "Numeric values are standardized. "
        "Dates accept DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY and ISO YYYY-MM-DD; day first. "
        "Invalid values are flagged; only literal equality can give them credit.",
        "fuzzy": "Only names can match fuzzily: full-name similarity must be strictly above "
        "97%, with at most one changed component and that component at least 85% similar. "
        "Initials, numeric identifiers, party separators, meaningful symbols, company markers "
        "and component counts must agree. Other similarities above 90% require review; "
        "90% or below is a mismatch. Thresholds use unrounded scores.",
    }
    report["data_issues"] = issues
    return report, details


def evaluate_files(labels_path: Path, predictions_path: Path, output: Path, records_path=None):
    paths = [Path(labels_path)] if isinstance(labels_path, (str, Path)) else list(labels_path)
    labels = [row for path in paths for row in read_table(path.read_bytes(), path.name)]
    predictions = read_table(predictions_path.read_bytes(), predictions_path.name, allow_empty=True)
    evidence = {}
    run_metadata = None
    # Without a run manifest, all supplied reference rows are the evaluation scope.
    # Inferring scope from successes would silently omit failed documents.
    scope = None
    if records_path is not None:
        try:
            payload = json.loads(Path(records_path).read_text(encoding="utf-8"))
            evidence = {r["document_id"]: r["metadata"] for r in payload["records"]}
            run_metadata = payload["run"]
            scope = [
                r["document_id"] for r in payload["records"] + payload.get("errors", [])
            ] or scope
        except (OSError, ValueError, KeyError) as exc:
            raise DocumentError("Cannot read extraction records for evaluation.") from exc
    report, details = evaluate(labels, predictions, evidence, document_ids=scope)
    report["run"] = run_metadata
    report["input_sha256"] = {
        "labels": hashlib.sha256(paths[0].read_bytes()).hexdigest()
        if len(paths) == 1
        else {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "predictions": hashlib.sha256(Path(predictions_path).read_bytes()).hexdigest(),
        "records": hashlib.sha256(Path(records_path).read_bytes()).hexdigest()
        if records_path is not None
        else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    from .pipeline import write_json

    write_json(output / "evaluation_results.json", report)
    columns = list(details[0]) if details else ["filename", "field", "ground_truth", "prediction"]
    (output / "error_analysis.csv").write_text(csv_text(details, columns), encoding="utf-8")
    (output / "evaluation_report.xlsx").write_bytes(excel_report(report, details))
    return report


def print_metrics(report):
    print(f"{'Field':28} {'Documents':10} {'Strict':12} {'Normalized':12} Tolerant")
    for name, values in report["fields"].items():
        recalls = [
            f"{values[tier]['recall']:.2%}" if values[tier]["recall"] is not None else "N/A"
            for tier in RECALL_TIERS
        ]
        print(
            f"{name.replace('Aggrement', 'Agreement'):28} {values['total']:<10} "
            f"{recalls[0]:12} {recalls[1]:12} {recalls[2]}"
        )
    print(f"Data issues: {len(report['data_issues'])}; see evaluation_results.json")

