"""Allowlisted provider diagnostics; never log response bodies, prompts or credentials."""

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path

_LOCK = threading.Lock()
_TOKEN = re.compile(r"[A-Za-z0-9_./:-]{1,200}")
_SECONDS = re.compile(r"[0-9]+(?:\.[0-9]+)?s")


def safe_token(value):
    return value if isinstance(value, str) and _TOKEN.fullmatch(value) else None


def api_diagnostics(exc):
    result = {
        "http_status": getattr(exc, "code", None),
        "provider_status": safe_token(getattr(exc, "status", None)),
        "quota_violations": [],
    }
    payload = getattr(exc, "details", {})
    if not isinstance(payload, dict):
        return result
    payload = payload.get("error", payload)
    if not isinstance(payload, dict):
        return result
    details = payload.get("details", [])
    if not isinstance(details, list):
        return result
    for item in details:
        if not isinstance(item, dict):
            continue
        if item.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
            delay = item.get("retryDelay")
            if isinstance(delay, str) and _SECONDS.fullmatch(delay):
                result["retry_delay_seconds"] = float(delay[:-1])
        if item.get("@type") == "type.googleapis.com/google.rpc.QuotaFailure":
            violations = item.get("violations", [])
            if not isinstance(violations, list):
                continue
            for violation in violations[:20]:
                if not isinstance(violation, dict):
                    continue
                fields = {key: safe_token(violation.get(key)) for key in ("quotaMetric", "quotaId")}
                value = violation.get("quotaValue")
                if isinstance(value, (int, float)) or (
                    isinstance(value, str) and value.isascii() and value.isdigit()
                ):
                    fields["quotaValue"] = value
                dimensions = violation.get("quotaDimensions", {})
                if isinstance(dimensions, dict):
                    fields["dimensions"] = {
                        key: safe_token(dimensions.get(key)) for key in ("model", "location")
                    }
                result["quota_violations"].append(fields)
    return result


def write_event(output_dir: Path, event: dict):
    """Call only with explicitly constructed metadata, never raw exception text."""
    path = output_dir / "provider_events.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"timestamp": datetime.now(UTC).isoformat(), **event}) + "\n"
        with _LOCK, path.open("a", encoding="utf-8") as stream:
            stream.write(line)
    except OSError:
        # Diagnostics must not turn a valid extraction into an unavailable result.
        return False
    return True
