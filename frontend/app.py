import base64
import hashlib
import json
import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

from metadata_extraction.config import PROJECT_ROOT, Settings
from metadata_extraction.dataset import document_id
from metadata_extraction.document_loader import LOADER_VERSION, load_bytes
from metadata_extraction.errors import ExtractionError
from metadata_extraction.evaluation_io import evaluation_issue_message
from metadata_extraction.evaluator import csv_text, evaluation_id
from metadata_extraction.formats import IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS
from metadata_extraction.pipeline import ExtractionRecord
from metadata_extraction.schemas import FIELD_COLUMNS

load_dotenv(PROJECT_ROOT / ".env", override=False)
API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
settings = Settings.from_env()
# Defined in this script so Streamlit reruns refresh it without cached package imports.
EVALUATION_VIEW_VERSION = "three-recall-v6-all-labelled-run-documents"
# Allow the backend to finish its bounded retries and key failover before timing out.
extraction_timeout = httpx.Timeout(
    connect=10,
    read=max(
        600,
        (settings.timeout_seconds + 8)
        * settings.max_attempts
        * max(1, len(settings.configured_keys()))
        + 120,
    ),
    write=60,
    pool=10,
)

st.set_page_config(
    page_title="Agreement Metadata Extraction", page_icon="\U0001f4c4", layout="wide"
)
st.title("Agreement Metadata Extraction")
st.caption(
    "Extract key details from your agreements, review the results, and evaluate their accuracy."
)
st.markdown(
    "<style>.block-container {max-width: 1280px; padding-top: 2rem;}"
    "div[data-testid='stMetric'] {border: 1px solid #9aa5b433; border-radius: 12px;"
    "padding: 16px;} div[data-baseweb='tab-list'] {gap: 1rem;}"
    "button[data-baseweb='tab'] {font-size: 1rem;}"
    "h1 {font-size: clamp(1.7rem, 4vw, 2.4rem) !important;}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Connection")
    if st.button("Check connection", width="stretch"):
        try:
            response = httpx.get(f"{API_URL}/health", timeout=5)
            response.raise_for_status()
            st.session_state["health"] = response.json()
            st.session_state.pop("health_error", None)
        except (httpx.HTTPError, ValueError):
            st.session_state.pop("health", None)
            st.session_state["health_error"] = "Cannot connect. Start the API and try again."
    health = st.session_state.get("health")
    if health:
        if health["credentials_configured"]:
            st.success("Connected")
        else:
            st.warning("Add an API key in .env and restart the API.")
        slots = health.get("key_slots", [])
        if slots and not any(slot["available"] for slot in slots):
            st.warning("Usage limit reached. Saved results are still available.")
    elif st.session_state.get("health_error"):
        st.error(st.session_state["health_error"])
    else:
        st.caption("Check before you begin.")
    with st.expander("Connection details"):
        st.code(API_URL)
        if health:
            st.caption(f"Model: {health['model']}")

upload_tab, results_tab, evidence_tab, evaluation_tab = st.tabs(
    ["Upload", "Results", "Sources", "Evaluation"]
)

with upload_tab:
    st.subheader("Upload agreements")
    st.caption("DOCX, PDF, PNG or JPG/JPEG · Choose one or more files")
    uploads = st.file_uploader(
        "Choose files",
        type=sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        max_upload_size=settings.max_upload_mb,
    )
    fingerprint = hashlib.sha256()
    for item in uploads:
        fingerprint.update(item.name.encode("utf-8"))
        fingerprint.update(hashlib.sha256(item.getvalue()).digest())
    selection = fingerprint.hexdigest()
    if st.session_state.get("upload_signature") != selection:
        for key in ("results", "failures", "evaluation", "run_document_ids", "preview"):
            st.session_state.pop(key, None)
        st.session_state["upload_signature"] = selection
    if uploads:
        st.caption(f"{len(uploads)} file(s) selected")
    else:
        st.info("Choose files to get started.")
    if uploads and st.checkbox("Preview a document", value=False):
        selected = st.selectbox(
            "Document to preview", range(len(uploads)), format_func=lambda i: uploads[i].name
        )
        upload = uploads[selected]
        try:
            preview_key = (
                selection, selected, LOADER_VERSION, settings.effective_image_input,
                settings.effective_ocr_mode, settings.max_upload_mb, settings.max_image_parts,
                settings.max_document_chars, settings.tile_height, settings.tile_overlap,
            )
            preview = st.session_state.get("preview")
            if preview is None or preview[0] != preview_key:
                content = load_bytes(upload.getvalue(), upload.name, settings)
                st.session_state["preview"] = (preview_key, content)
            else:
                content = preview[1]
            if Path(upload.name).suffix.lower() in IMAGE_EXTENSIONS:
                for image in content.images:
                    st.image(image.data, caption=upload.name, width="stretch")
            else:
                preview_text = "\n\n".join(block.text for block in content.text_blocks)
                st.text_area("Document preview", preview_text[:20000], height=300, disabled=True)
                if len(preview_text) > 20000:
                    st.caption(
                        "Preview shows the first 20,000 characters; extraction uses the full input."
                    )
                if content.images:
                    with st.expander("Images in this document"):
                        for image in content.images[:3]:
                            st.image(image.data, caption=image.source_id, width="stretch")
                        if len(content.images) > 3:
                            st.caption(
                                "Preview shows three sections; extraction includes all sections."
                            )
        except ExtractionError as exc:
            st.error(f"Cannot preview: {exc}")

    def explain_error(response):
        try:
            return response.json().get("detail", "Request failed.")
        except ValueError:
            return "Backend returned an unreadable error response."

    duplicate_ids = len({evaluation_id(upload.name) for upload in uploads}) != len(uploads)
    if duplicate_ids:
        st.error("Some files have the same document ID. Upload them separately.")
    too_many_files = len(uploads) > settings.max_batch_files
    if too_many_files:
        st.error(f"Choose at most {settings.max_batch_files} files per run.")

    if st.button(
        "Extract details", type="primary",
        disabled=not uploads or duplicate_ids or too_many_files, width="stretch"
    ):
        st.session_state.pop("results", None)
        st.session_state.pop("failures", None)
        st.session_state.pop("evaluation", None)
        st.session_state["run_document_ids"] = [document_id(upload.name) for upload in uploads]
        results, failures = [], []
        with st.status("Reading agreements...", expanded=True) as status:
            progress = st.progress(0.0)
            quota_blocked = False
            with httpx.Client(timeout=extraction_timeout) as client:
                for index, upload in enumerate(uploads, 1):
                    st.write(f"{index}/{len(uploads)}: {upload.name}")
                    if quota_blocked:
                        failures.append(
                            {
                                "filename": upload.name,
                                "error": "Skipped: API usage limit reached.",
                            }
                        )
                        progress.progress(index / len(uploads))
                        continue
                    try:
                        response = client.post(
                            f"{API_URL}/extract",
                            params={"debug": "true"},
                            files={"file": (upload.name, upload.getvalue(), upload.type)},
                        )
                        if response.is_success:
                            payload = response.json()
                            record = ExtractionRecord.model_validate(
                                {**payload, "raw_response": ""}
                            )
                            if record.filename != upload.name:
                                raise ValueError("Response does not match the requested document")
                            result = record.model_dump(mode="json", exclude={"raw_response"})
                            result["values"] = record.metadata.values()
                            results.append(result)
                        else:
                            failures.append(
                                {"filename": upload.name, "error": explain_error(response)}
                            )
                            if response.status_code == 429:
                                quota_blocked = True
                    except httpx.HTTPError:
                        failures.append(
                            {
                                "filename": upload.name,
                                "error": "Connection failed. Check the connection and try again.",
                            }
                        )
                    except (ValueError, KeyError, TypeError):
                        failures.append({
                            "filename": upload.name,
                            "error": (
                                "Backend returned an invalid result. Restart the API and retry."
                            ),
                        })
                    progress.progress(index / len(uploads))
            status.update(
                label=f"Finished: {len(results)}/{len(uploads)} documents",
                state="error" if failures else "complete",
            )
        st.session_state["results"] = results
        st.session_state["failures"] = failures
        st.info("Your results are ready. Open Results to review them.")

results = st.session_state.get("results", [])
failures = st.session_state.get("failures", [])
rows = [
    {
        "File Name": document_id(r["filename"]),
        **{column: r["values"][field] or "" for field, column in FIELD_COLUMNS.items()},
    }
    for r in results
]
DISPLAY_COLUMNS = {c: c.replace("Aggrement", "Agreement") for c in FIELD_COLUMNS.values()}


with results_tab:
    st.subheader("Your results")
    if results or failures:
        for column, label, value in zip(
            st.columns(2),
            ["Processed", "Not processed"],
            [len(results), len(failures)],
            strict=True,
        ):
            column.metric(label, value)
    else:
        st.info("Upload and extract agreements to see results here.")
    if failures:
        st.warning(
            f"{len(failures)} file(s) could not be processed. "
            "Available results can still be downloaded."
        )
        for failure in failures:
            st.error(f"{failure['filename']}: {failure['error']}")
        st.download_button(
            "Download error report",
            csv_text(failures, ["filename", "error"]),
            file_name="extraction_errors.csv",
            mime="text/csv",
        )
        st.caption("Try again from Upload. Completed results will be reused.")
    if results:
        search = st.text_input("Find a document", placeholder="Search by filename")
        display_rows, missing_rows = [], []
        for result, row in zip(results, rows, strict=True):
            if search.casefold() not in result["filename"].casefold():
                continue
            display = {"Document": row["File Name"]}
            for field, column in FIELD_COLUMNS.items():
                display[DISPLAY_COLUMNS[column]] = row[column]
                item = result["metadata"][field]
                if item["status"] in {"missing", "ambiguous"}:
                    missing_rows.append(
                        {
                            "Document": row["File Name"],
                            "Field": DISPLAY_COLUMNS[column],
                            "Issue": "Missing" if item["status"] == "missing" else "Ambiguous",
                            "Explanation": item.get("explanation") or "",
                        }
                    )
            display_rows.append(display)
        if display_rows:
            st.dataframe(display_rows, hide_index=True, width="stretch")
        else:
            st.info("No documents match your search.")
        st.caption(
            "Empty cells mean missing or unclear details. Check Sources to learn more. "
            "Downloads include all results."
        )
        if missing_rows:
            st.subheader("Missing or ambiguous details")
            st.caption("These fields were left empty. Check the source for more information.")
            st.dataframe(missing_rows, hide_index=True, width="stretch")
        left, right = st.columns(2)
        left.download_button(
            "Download Extracted Metadata (CSV)",
            csv_text(rows).encode("utf-8"),
            file_name="Extracted Metadata.csv",
            mime="text/csv",
            width="stretch",
        )
        right.download_button(
            "Download sources (JSON)",
            json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="evidence.json",
            mime="application/json",
            width="stretch",
        )

with evaluation_tab:
    if not results and not failures:
        st.info("Extract agreements first, then compare results with your reference answers.")
    else:
        st.subheader("Evaluate results")
        st.caption("Choose reference answers in CSV or Excel format.")
        run_ids = st.session_state.get("run_document_ids") or [
            document_id(item["filename"]) for item in results + failures
        ]
        st.caption(f"Only the {len(run_ids)} document(s) in this run will be evaluated.")
        if st.button("Find reference files"):
            try:
                response = httpx.get(f"{API_URL}/evaluation-files", timeout=10)
                response.raise_for_status()
                st.session_state["evaluation_files"] = response.json()["files"]
            except (httpx.HTTPError, ValueError, KeyError):
                st.error("Cannot load reference files. Check the connection.")
        selected_labels = st.multiselect(
            "Reference files in data folder", st.session_state.get("evaluation_files", [])
        )
        uploaded_labels = st.file_uploader(
            "Or upload reference answers",
            type=["xlsx", "csv"],
            accept_multiple_files=True,
            key="ground_truth_uploads",
        )
        signature = hashlib.sha256(csv_text(rows).encode("utf-8"))
        signature.update(json.dumps(sorted(run_ids)).encode("utf-8"))
        signature.update(json.dumps(selected_labels).encode("utf-8"))
        for label in uploaded_labels:
            signature.update(label.name.encode("utf-8"))
            signature.update(label.getvalue())
        evaluation_signature = signature.hexdigest()
        previous = st.session_state.get("evaluation")
        if previous and (
            previous.get("selection_signature") != evaluation_signature
            or previous.get("view_version") != EVALUATION_VIEW_VERSION
            or not previous["report"].get("evaluation_version")
        ):
            st.session_state.pop("evaluation", None)
            st.caption("Your selection changed. Evaluate again to refresh the report.")
        if st.button("Evaluate results", disabled=not (selected_labels or uploaded_labels)):
            st.session_state.pop("evaluation", None)
            files = [
                (
                    "predictions",
                    ("Extracted Metadata.csv", csv_text(rows).encode("utf-8"), "text/csv"),
                )
            ]
            files.extend(
                ("ground_truth", (u.name, u.getvalue(), "application/octet-stream"))
                for u in uploaded_labels
            )
            try:
                with st.spinner("Comparing results..."):
                    response = httpx.post(
                        f"{API_URL}/evaluate",
                        files=files,
                        data={
                            "local_files": json.dumps(selected_labels),
                            "document_ids": json.dumps(run_ids),
                        },
                        timeout=120,
                    )
                if response.is_success:
                    payload = response.json()
                    report = payload.get("report", {})
                    if report.get("scoring_policy") != "three-tiers-all-labelled-run-documents-v1":
                        raise ValueError("Restart the API to load the current matching policy.")
                    if report.get("requested_document_ids") != sorted(
                        {evaluation_id(v) for v in run_ids}
                    ):
                        raise ValueError(
                            "Restart the API to enable evaluation for this run's documents."
                        )
                    required_scores = {
                        "total",
                        "strict", "normalized", "tolerant",
                        "review_count",
                    }
                    required_overall = {
                        "strict", "normalized", "tolerant",
                        "evaluated_documents",
                        "unlabelled_documents",
                        "missing_documents",
                        "review_fields",
                    }
                    if (
                        not report.get("evaluation_version")
                        or not required_overall.issubset(report.get("overall", {}))
                        or not set(FIELD_COLUMNS.values()).issubset(report.get("fields", {}))
                        or any(
                            not required_scores.issubset(score)
                            for score in report.get("fields", {}).values()
                        )
                        or not {"definitions", "data_issues"}.issubset(report)
                        or not {"excel_base64", "report_id"}.issubset(payload)
                    ):
                        raise ValueError("Incomplete evaluation response")
                    metric_keys = {"correct", "incorrect", "total", "recall"}
                    for scores in [report["overall"], *report["fields"].values()]:
                        for tier in ("strict", "normalized", "tolerant"):
                            if not isinstance(scores[tier], dict) or not metric_keys.issubset(
                                scores[tier]
                            ):
                                raise ValueError("Incomplete recall metric")
                    payload["selection_signature"] = evaluation_signature
                    payload["view_version"] = EVALUATION_VIEW_VERSION
                    st.session_state["evaluation"] = payload
                else:
                    st.error(explain_error(response))
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                st.error(
                    str(exc)
                    if str(exc).startswith("Restart the API")
                    else "Evaluation failed. Check the connection and try again."
                )
            except httpx.HTTPError:
                st.error("Evaluation failed. Check the connection and try again.")
        evaluation = st.session_state.get("evaluation")
        if evaluation:
            report = evaluation["report"]
            overall = report["overall"]

            def score_text(value):
                return f"{value:.1%}" if value is not None else "N/A"

            for column, tier in zip(
                st.columns(3), ("strict", "normalized", "tolerant"), strict=True
            ):
                column.metric(f"{tier.title()} recall", score_text(overall[tier]["recall"]))
            st.info(
                "**Recall metrics**\n\n"
                "- **Strict:** Values match exactly, including case, spaces and punctuation.\n"
                "- **Normalized:** Also accepts equivalent values after formatting cleanup.\n"
                "- **Tolerant:** Also accepts highly similar names that pass identity checks.\n\n"
                "All three use the same set of scored documents. Review cases receive no credit."
            )
            st.caption(
                f"{overall['evaluated_documents']} document(s) with reference answers scored, "
                f"including {overall['missing_documents']} without extraction results. "
                f"{overall['unlabelled_documents']} document(s) without references are unscored. "
                f"{report['excluded_label_rows']} reference row(s) outside this run excluded."
            )
            st.dataframe(
                [
                    {
                        "Field": DISPLAY_COLUMNS[name],
                        "Documents": score["total"],
                        **{
                            f"{tier.title()} recall": score_text(score[tier]["recall"])
                            for tier in ("strict", "normalized", "tolerant")
                        },
                    }
                    for name, score in report["fields"].items()
                ],
                hide_index=True,
                width="stretch",
            )
            if report["data_issues"]:
                st.subheader("Evaluation data notes")
                for issue in report["data_issues"]:
                    message = evaluation_issue_message(issue)
                    if issue["type"] == "identical duplicate label":
                        st.info(message)
                    else:
                        st.warning(message)
            if overall["review_fields"]:
                st.warning(
                    f"{overall['review_fields']} field(s) need review and receive no credit."
                )
            with st.expander("Scoring criteria"):
                st.markdown(
                    "- Only documents in the current run are evaluated against their reference "
                    "answers. Failed/skipped documents with references count as incorrect.\n"
                    + "\n".join(f"- {value}" for value in report["definitions"].values())
                )
            st.download_button(
                "Download report (Excel)",
                base64.b64decode(evaluation["excel_base64"]),
                file_name="evaluation_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

with evidence_tab:
    st.subheader("Check the source")
    if not results:
        st.info("Extract an agreement to see where each detail came from.")
    else:
        chosen = st.selectbox(
            "Choose a document", range(len(results)), format_func=lambda i: results[i]["filename"]
        )
        result = results[chosen]
        with st.expander("Processing details"):
            st.caption(f"Model: {result['model']}")
            st.caption("Reused result" if result.get("cached") else "New result")
        for warning in result["warnings"]:
            st.warning(warning)
        show_evidence = st.checkbox("Show source quotes", value=True)
        for pair in range(0, len(FIELD_COLUMNS), 2):
            for column, (field, label) in zip(
                st.columns(2), list(FIELD_COLUMNS.items())[pair : pair + 2], strict=True
            ):
                item = result["metadata"][field]
                with column.container(border=True):
                    st.caption(DISPLAY_COLUMNS[label])
                    st.text(result["values"][field] or "")
                    st.caption(
                        {"found": "Found", "missing": "Not found", "ambiguous": "Unclear"}[
                            item["status"]
                        ]
                        + (
                            " · "
                            + {"explicit": "From document", "derived": "Calculated"}[
                                item["provenance"]
                            ]
                            if item["provenance"]
                            else ""
                        )
                    )
                    if show_evidence:
                        for evidence in item["evidence"]:
                            st.text(evidence["quote"])
                            st.caption(evidence["source_id"])
                    if item["derivation"]:
                        st.write(item["derivation"])
                    if item["explanation"]:
                        st.info(item["explanation"])
        st.caption(
            "Preview the original in Upload to check these quotes. Review image quotes manually."
        )
