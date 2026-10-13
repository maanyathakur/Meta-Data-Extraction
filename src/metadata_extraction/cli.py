import argparse
import json
import sys
from pathlib import Path

from .batch import predict_folder
from .config import Settings
from .dataset import inspect_dataset
from .errors import ExtractionError
from .evaluator import evaluate_files, print_metrics
from .pipeline import ExtractionPipeline, write_json


def main():
    parser = argparse.ArgumentParser(description="Agreement metadata extraction and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Audit supplied files/labels without model calls")
    inspect.add_argument("--data-dir", type=Path)
    extract = sub.add_parser("extract", help="Extract one DOCX, PDF, PNG or JPG/JPEG")
    extract.add_argument("file", type=Path)
    extract.add_argument("--output", type=Path)
    extract.add_argument("--no-cache", action="store_true")
    predict = sub.add_parser("predict", help="Predict files only; never reads ground-truth labels")
    predict.add_argument("--input-dir", type=Path)
    predict.add_argument("--split", choices=["train", "test"], default="test")
    predict.add_argument("--output", type=Path)
    predict.add_argument("--no-cache", action="store_true")
    score = sub.add_parser("evaluate", help="Score saved CSV predictions without model calls")
    score.add_argument("--labels", type=Path, nargs="+", required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--records", type=Path)
    score.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        if args.command == "inspect":
            report = inspect_dataset(args.data_dir or settings.data_dir)
            write_json(settings.output_dir / "dataset_audit.json", report)
            print(json.dumps(report, indent=2, ensure_ascii=False))
        elif args.command == "extract":
            record = ExtractionPipeline(settings).extract_path(args.file, not args.no_cache)
            if args.output:
                write_json(args.output, record.model_dump(mode="json"))
            print(record.model_dump_json(indent=2))
        elif args.command == "predict":
            folder = args.input_dir or settings.data_dir / args.split
            output = args.output or settings.output_dir / f"{args.split}_predictions.csv"
            result = predict_folder(ExtractionPipeline(settings), folder, output, not args.no_cache)
            print(f"Saved {len(result['records'])} predictions to {output}")
            if result["errors"]:
                print("Run incomplete: review .records.json errors.", file=sys.stderr)
                return 1
        else:
            report = evaluate_files(
                args.labels, args.predictions, args.output_dir or settings.output_dir, args.records
            )
            print_metrics(report)
        return 0
    except (ExtractionError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

