from datetime import UTC, datetime
from pathlib import Path

from .dataset import document_id, list_documents
from .errors import ExtractionError, ProviderQuotaError
from .evaluator import csv_text
from .pipeline import ExtractionPipeline, write_json


def predict_folder(pipeline: ExtractionPipeline, folder: Path, output: Path, use_cache=True):
    paths = list_documents(folder)
    if not paths:
        from .errors import DocumentError

        raise DocumentError("No supported documents found (DOCX, PDF, PNG, JPG or JPEG).")
    rows, records, errors = [], [], []
    stopped_for_quota = False
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        identifier = document_id(path.name)
        if stopped_for_quota:
            errors.append(
                {
                    "document_id": identifier,
                    "error": "Skipped because every configured key slot is quota-limited.",
                    "attempted": False,
                }
            )
            continue
        try:
            record = pipeline.extract_path(path, use_cache)
            rows.append(record.metadata.csv_row(identifier))
            records.append({"document_id": identifier, **record.model_dump(mode="json")})
        except ExtractionError as exc:
            errors.append({"document_id": identifier, "error": str(exc)})
            print(f"  FAILED: {exc}", flush=True)
            if isinstance(exc, ProviderQuotaError):
                stopped_for_quota = True
                print(
                    "  Stopping fresh requests: all configured key slots are cooling down.",
                    flush=True,
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(csv_text(rows), encoding="utf-8")
    payload = {
        "run": {
            "created_at": datetime.now(UTC).isoformat(),
            "input_folder": str(folder.resolve()),
            "total_documents": len(paths),
            "successful_documents": len(records),
            "failed_documents": len(errors),
            "configuration_hash": pipeline._configuration_hash(),
            "model": pipeline.settings.model,
            "provider": pipeline.settings.provider,
            "ocr_mode": pipeline.settings.ocr_mode,
            "image_input": pipeline.settings.image_input,
            "effective_image_input": pipeline.settings.effective_image_input,
            "effective_ocr_mode": pipeline.settings.effective_ocr_mode,
            "model_supports_vision": pipeline.settings.model_supports_vision,
            "thinking_level": pipeline.settings.thinking_level
            if pipeline.settings.provider == "gemini"
            else None,
            "max_output_tokens": pipeline.settings.max_output_tokens,
            "stopped_for_quota": stopped_for_quota,
            "skipped_documents": sum(e.get("attempted") is False for e in errors),
        },
        "records": records,
        "errors": errors,
    }
    write_json(output.with_suffix(".records.json"), payload)
    return payload

