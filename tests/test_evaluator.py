import pytest

from metadata_extraction.dataset import document_id
from metadata_extraction.errors import DocumentError
from metadata_extraction.evaluator import evaluate
from metadata_extraction.schemas import CSV_COLUMNS


def row(identifier, value):
    return {"File Name": identifier, **{column: value for column in CSV_COLUMNS[1:]}}


def test_cleaning_counts_as_match_but_invalid_dates_do_not():
    report, details = evaluate([row("a", " Owner ")], [row("a", "Owner")])
    assert report["fields"]["Party One"]["normalized"]["recall"] == 1
    assert details[0]["exact_match"] is False
    report, _ = evaluate([row("a", "31.02.2011")], [row("a", "28.02.2011")])
    assert report["fields"]["Aggrement End Date"]["normalized"]["recall"] == 0


def test_missing_extraction_row_is_incorrect_even_when_label_empty():
    report, _ = evaluate([row("a", "")], [])
    for tier in ("strict", "normalized", "tolerant"):
        assert report["fields"]["Party One"][tier] == {
            "correct": 0, "incorrect": 1, "total": 1, "recall": 0,
        }
    assert report["missing_prediction_rows"] == ["a"]


def test_exact_counts_and_extra_predictions():
    report, _ = evaluate(
        [row("a", "x"), row("b", "y")], [row("a", "x"), row("b", "z"), row("extra", "x")]
    )
    score = report["fields"]["Party One"]["strict"]
    assert (score["correct"], score["total"], score["recall"]) == (1, 2, 0.5)
    assert report["extra_prediction_rows"] == ["extra"]


def test_duplicate_predictions_rejected():
    with pytest.raises(DocumentError, match="Duplicate"):
        evaluate([row("a", "x")], [row("a", "x"), row("a", "x")])


def test_empty_evaluation_is_not_perfect_score():
    report, _ = evaluate([], [])
    assert report["fields"]["Party One"]["strict"]["recall"] is None


@pytest.mark.parametrize(
    "name,expected",
    [
        ("contract.pdf.docx", "contract"),
        ("contract.png", "contract"),
        ("contract.docx", "contract"),
        ("with.dots.docx", "with.dots"),
    ],
)
def test_filename_mapping(name, expected):
    assert document_id(name) == expected


def test_failed_documents_and_missing_fields_count_as_incorrect():
    blank = dict.fromkeys(CSV_COLUMNS[1:], "")
    labels = [
        {"File Name": name, **blank, "Aggrement Value": "1000"}
        for name in ("a", "b", "c", "failed")
    ]
    extracted = [
        labels[0],
        {**labels[1], "Aggrement Value": "2000"},
        {**labels[2], "Aggrement Value": ""},
    ]
    report, details = evaluate(labels, extracted, document_ids=[r["File Name"] for r in labels])
    rent = report["fields"]["Aggrement Value"]
    for tier in ("strict", "normalized", "tolerant"):
        assert rent[tier] == {"correct": 1, "incorrect": 3, "total": 4, "recall": 0.25}
    assert report["overall"]["evaluated_documents"] == 4
    assert report["overall"]["completed_documents"] == 3
    assert report["overall"]["missing_documents"] == 1
    assert report["overall"]["evaluated_fields"] == 24
    failed = [detail for detail in details if detail["filename"] == "failed"]
    assert len(failed) == 6 and not any(detail["excluded_from_metrics"] for detail in failed)
