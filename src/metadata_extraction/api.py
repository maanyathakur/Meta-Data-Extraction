import base64
import json
import threading
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .dataset import document_id, evaluation_id
from .document_loader import load_bytes
from .errors import ConfigurationError, DocumentError, ProviderError, ProviderQuotaError
from .evaluation_io import excel_report, read_table
from .evaluator import csv_text, evaluate, print_metrics
from .key_pool import KeyPool
from .pipeline import ExtractionPipeline


def create_app(settings: Settings | None = None, pipeline=None):
    settings = settings or Settings.from_env()
    pipeline = pipeline or ExtractionPipeline(settings)
    app = FastAPI(title="Agreement Metadata Extractor", version="0.1.0")
    # Serialize provider requests to limit quota bursts.
    model_lock = threading.Lock()

    def ground_truth_files():
        root = settings.data_dir.resolve()
        return {
            p.relative_to(root).as_posix(): p
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and p.suffix.lower() in {".xlsx", ".csv"}
            and not p.name.startswith("~$")
            and p.resolve().is_relative_to(root)
        }

    @app.get("/evaluation-files")
    def evaluation_files():
        return {"files": list(ground_truth_files())}

    @app.post("/evaluate")
    async def evaluation(
        predictions: Annotated[UploadFile, File()],
        local_files: Annotated[str, Form()] = "[]",
        ground_truth: Annotated[list[UploadFile] | None, File()] = None,
        document_ids: Annotated[str | None, Form()] = None,
    ):
        try:
            selected = json.loads(local_files)
            scope = json.loads(document_ids) if document_ids is not None else None
            if not isinstance(selected, list) or any(not isinstance(v, str) for v in selected):
                raise DocumentError("local_files must be a JSON list of filenames.")
            uploads = ground_truth or []
            if not 1 <= len(selected) + len(uploads) <= settings.max_batch_files:
                raise DocumentError(f"Select 1-{settings.max_batch_files} ground-truth files.")
            available = ground_truth_files()
            if any(name not in available for name in selected):
                raise DocumentError("Select ground-truth files listed by /evaluation-files.")
            prediction_data = await read_upload(predictions)
            labels = []
            for name in selected:
                path = available[name]
                if path.stat().st_size > settings.max_upload_mb * 1024 * 1024:
                    raise DocumentError("Ground-truth file exceeds upload limit.")
                data = await run_in_threadpool(path.read_bytes)
                labels.extend(await run_in_threadpool(read_table, data, name))
            for file in uploads:
                data = await read_upload(file)
                labels.extend(await run_in_threadpool(read_table, data, file.filename or "upload"))

            def build():
                rows = read_table(
                    prediction_data,
                    predictions.filename or "Extracted Metadata.csv",
                    allow_empty=True,
                )
                report, details = evaluate(labels, rows, document_ids=scope)
                workbook = excel_report(report, details)
                folder = settings.output_dir / "evaluations" / uuid4().hex
                folder.mkdir(parents=True, exist_ok=False)
                (folder / "evaluation_report.xlsx").write_bytes(workbook)
                (folder / "evaluation_results.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print_metrics(report)
                return {
                    "report": report,
                    "report_id": folder.name,
                    "excel_base64": base64.b64encode(workbook).decode("ascii"),
                }

            return await run_in_threadpool(build)
        except (DocumentError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            await predictions.close()
            for file in ground_truth or []:
                await file.close()

    async def read_upload(file: UploadFile):
        try:
            data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB.")
            return data
        finally:
            await file.close()

    def parse(data, name):
        try:
            return load_bytes(data, name, settings)
        except DocumentError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ConfigurationError as exc:
            raise HTTPException(503, str(exc)) from exc

    def extract(document):
        try:
            with model_lock:
                return pipeline.extract_document(document)
        except DocumentError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ConfigurationError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ProviderQuotaError as exc:
            raise HTTPException(
                429, str(exc), headers={"Retry-After": str(exc.retry_after_seconds)}
            ) from exc
        except ProviderError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "provider": settings.provider,
            "model": settings.model,
            "model_supports_vision": settings.model_supports_vision,
            "image_input": settings.effective_image_input,
            "ocr_mode": settings.effective_ocr_mode,
            "credentials_configured": bool(settings.configured_keys()),
            "key_slots": KeyPool(settings).public_status(),
            "provider_check": "Health does not make a live provider request.",
        }

    @app.post("/extract")
    async def single(file: Annotated[UploadFile, File()], debug: bool = Query(False)):
        name = file.filename or "upload"
        data = await read_upload(file)
        document = await run_in_threadpool(parse, data, name)
        record = await run_in_threadpool(extract, document)
        if debug:
            result = record.model_dump(mode="json", exclude={"raw_response"})
            result["values"] = record.metadata.values()
            return result
        return record.metadata.values()

    @app.post("/extract-batch")
    async def batch(files: Annotated[list[UploadFile], File()]):
        try:
            if not files or len(files) > settings.max_batch_files:
                raise HTTPException(422, f"Upload 1-{settings.max_batch_files} files.")
            ids = [document_id(f.filename or "upload") for f in files]
            if len(ids) != len({evaluation_id(identifier) for identifier in ids}):
                raise HTTPException(422, "Batch contains duplicate document identifiers.")
            # Validate the whole batch before spending any model calls.
            inputs = []
            for file in files:
                name = file.filename or "upload"
                data = await read_upload(file)
                document = await run_in_threadpool(parse, data, name)
                inputs.append(document)
            rows = []
            for identifier, document in zip(ids, inputs, strict=True):
                record = await run_in_threadpool(extract, document)
                rows.append(record.metadata.csv_row(identifier))
            return Response(
                csv_text(rows),
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="Extracted Metadata.csv"'},
            )
        finally:
            for file in files:
                await file.close()

    return app


app = create_app()
