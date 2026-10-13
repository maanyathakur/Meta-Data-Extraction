"""Scoring contract: three tiers, one denominator and conservative name tolerance."""

import io

import pytest
from openpyxl import load_workbook

from metadata_extraction.evaluation_io import excel_report
from metadata_extraction.evaluator import RECALL_TIERS, csv_text, evaluate, evaluate_files
from metadata_extraction.name_matching import clean_name, fuzzy_conflicts, name_outcome
from metadata_extraction.schemas import CSV_COLUMNS


def row(identifier, name="Christopher Alexander Johnson"):
    return {"File Name": identifier, **dict.fromkeys(CSV_COLUMNS[1:], ""), "Party One": name}


def test_three_scores_share_uploaded_scope_and_failure_denominator():
    labels = [row(key) for key in ("exact", "format", "typo", "failed", "outside")]
    # Conflicting answers outside the uploaded run do not affect evaluation.
    labels.append(row("outside", "Different reference"))
    predictions = [
        row("exact"), row("format", "  MR. CHRISTOPHER   ALEXANDER JOHNSON  "),
        row("typo", "Christopher Alexander Jonson"), row("unlabelled"),
    ]
    scope = ["exact", "format", "typo", "failed", "unlabelled", "unlabelled-failed"]
    report, details = evaluate(labels, predictions, document_ids=scope)
    score = report["fields"]["Party One"]
    assert [score[tier]["recall"] for tier in RECALL_TIERS] == [0.25, 0.5, 0.75]
    assert all(score[tier]["total"] == 4 for tier in RECALL_TIERS)
    assert report["excluded_label_rows"] == 2
    assert report["overall"]["run_documents"] == 6
    assert report["overall"]["unlabelled_documents"] == 2
    assert report["overall"]["missing_documents"] == 1
    assert report["overall"]["completed_documents"] == 3
    book = load_workbook(io.BytesIO(excel_report(report, details)))
    names = [r for r in book["Comparisons"].values if r[1] == "Party One"]
    assert [r[4:7] for r in names[:4]] == [
        ("Yes", "Yes", "Yes"), ("No", "Yes", "Yes"),
        ("No", "No", "Yes"), ("No", "No", "No"),
    ]
    assert all(r[4:7] == ("Not scored",) * 3 for r in names[4:])
    assert next(r for r in book["Summary"].values if r[0] == "Party One")[5:] == (0.25, 0.5, 0.75)


def test_run_without_reference_answers_is_explicitly_unscored():
    report, details = evaluate(
        [row("outside")], [row("uploaded")], document_ids=["uploaded", "failed"]
    )
    assert report["overall"]["unlabelled_documents"] == 2
    assert report["overall"]["evaluated_documents"] == 0
    assert len(details) == 12 and all(d["excluded_from_metrics"] for d in details)
    for tier in RECALL_TIERS:
        assert report["overall"][tier]["recall"] is None
    assert excel_report(report, details)


def test_csv_without_manifest_does_not_silently_drop_missing_predictions(tmp_path):
    labels, predictions = tmp_path / "labels.csv", tmp_path / "predictions.csv"
    labels.write_text(csv_text([row("success"), row("failed")]), encoding="utf-8")
    predictions.write_text(csv_text([row("success")]), encoding="utf-8")
    report = evaluate_files(labels, predictions, tmp_path / "reports")
    assert report["missing_prediction_rows"] == ["failed"]
    assert all(report["overall"][tier]["recall"] == 0.5 for tier in RECALL_TIERS)


@pytest.mark.parametrize("expected,predicted", [
    ("Example Property Holdings 1234 Private Limited",
     "Example Property Holdings 1235 Private Limited"),
    ("Christopher Alexander Johnson & Margaret Alexandra Johnson",
     "Christopher Alexander Johnson / Margaret Alexandra Johnson"),
    ("Example Property Management Holdings LLC", "Example Property Management Holdings LLP"),
    ("Christopher Alexander Johnson", "Christopher Johnson"),
    ("Christopher Alexander Johnson", "Christophe Alexander Jonson"),
    ("Christopher Alexander Johnson + Holdings", "Christopher Alexander Johnson Holdings"),
])
def test_fuzzy_matching_rejects_identity_changes(expected, predicted):
    left, right = clean_name(expected), clean_name(predicted)
    assert left != right
    assert fuzzy_conflicts(left, right)
    assert name_outcome(left, right)[0] != "Fuzzy match"
    report, _ = evaluate([row("a", expected)], [row("a", predicted)])
    assert all(report["fields"]["Party One"][tier]["recall"] == 0 for tier in RECALL_TIERS)


def test_normalized_values_keep_meaningful_symbols_and_separated_letters():
    for left, right in [("C++ Holdings", "C Holdings"), ("Jo hn", "John"), ("A & B", "A / B")]:
        assert clean_name(left) != clean_name(right)
    label = row("a", "Mr. ALICE  SMITH")
    label.update({"Aggrement Value": "1,000.00", "Aggrement Start Date": "01/04/2020"})
    prediction = row("a", "alice smith")
    prediction.update({"Aggrement Value": "1000", "Aggrement Start Date": "2020-04-01"})
    report, _ = evaluate([label], [prediction])
    for column in ("Party One", "Aggrement Value", "Aggrement Start Date"):
        assert [report["fields"][column][tier]["recall"] for tier in RECALL_TIERS] == [0, 1, 1]
