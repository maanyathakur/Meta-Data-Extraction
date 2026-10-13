"""Run both dataset splits and save evaluation reports and timings."""

import argparse
import time
from datetime import UTC, datetime

from metadata_extraction.batch import predict_folder
from metadata_extraction.config import PROJECT_ROOT, Settings
from metadata_extraction.dataset import document_id
from metadata_extraction.evaluator import evaluate_files, print_metrics
from metadata_extraction.pipeline import ExtractionPipeline, write_json


class TimedPipeline(ExtractionPipeline):
    def __init__(self, settings):
        super().__init__(settings)
        self.timings = []
        self.split = ""

    def extract_path(self, path, use_cache=True):
        started = time.perf_counter()
        result, error = None, None
        try:
            result = super().extract_path(path, use_cache)
            return result
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            elapsed = round(time.perf_counter() - started, 3)
            self.timings.append(
                {
                    "split": self.split,
                    "document_id": document_id(path.name),
                    "filename": path.name,
                    "seconds": elapsed,
                    "status": "success" if result else "failed",
                    "cached": result.cached if result else False,
                    "error_type": error,
                }
            )
            write_json(
                self.settings.output_dir / "document_timings.json", {"documents": self.timings}
            )
            print(f"  {elapsed:.2f}s; {'success' if result else 'failed'}", flush=True)


def main(clear_cache=False):
    settings = Settings.from_env()
    if clear_cache:
        cache = settings.cache_dir.resolve()
        if not cache.is_relative_to(PROJECT_ROOT.resolve()) or cache == PROJECT_ROOT.resolve():
            raise ValueError("Refusing cache cleanup outside the project.")
        removed = 0
        if cache.exists():
            for path in cache.glob("*.json"):
                path.unlink()
                removed += 1
        print(f"Cleared {removed} extraction cache files; quota cooldowns preserved.", flush=True)
    settings.require_key()
    pipeline = TimedPipeline(settings)
    started = time.perf_counter()
    summary = {
        "started_at": datetime.now(UTC).isoformat(),
        "provider": settings.provider,
        "model": settings.model,
        "configuration_hash": pipeline._configuration_hash(),
        "effective_image_input": settings.effective_image_input,
        "effective_ocr_mode": settings.effective_ocr_mode,
        "splits": {},
    }
    for split in ("train", "test"):
        pipeline.split = split
        split_started = time.perf_counter()
        predictions = settings.output_dir / f"{split}_predictions.csv"
        result = predict_folder(pipeline, settings.data_dir / split, predictions)
        extraction_seconds = time.perf_counter() - split_started
        report = evaluate_files(
            settings.data_dir / f"{split}.csv",
            predictions,
            settings.output_dir / f"{split}_evaluation",
            predictions.with_suffix(".records.json"),
        )
        print_metrics(report)
        summary["splits"][split] = {
            "extraction_seconds": round(extraction_seconds, 3),
            "run": result["run"],
            "fields": report["fields"],
            "missing_prediction_rows": report["missing_prediction_rows"],
            "extra_prediction_rows": report["extra_prediction_rows"],
        }
        summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_json(settings.output_dir / "run_summary.json", summary)
    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary["total_seconds"] = round(time.perf_counter() - started, 3)
    write_json(settings.output_dir / "run_summary.json", summary)
    print(f"Extraction plus evaluation: {summary['total_seconds']:.2f}s", flush=True)
    return 1 if any(s["run"]["failed_documents"] for s in summary["splits"].values()) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-cache", action="store_true")
    raise SystemExit(main(parser.parse_args().clear_cache))
