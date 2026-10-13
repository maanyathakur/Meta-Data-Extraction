"""Priority key failover with persistent cooldowns; stored state contains no keys."""

import hashlib
import json
import math
import os
import tempfile
import threading
import time

from .config import Settings

_LOCK = threading.Lock()


class KeyPool:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.cache_dir.parent / "provider_cooldowns.json"
        self._blocked = {}

    def _identity(self, key):
        identity = f"{key}:{self.settings.model}"
        if self.settings.provider != "gemini":
            identity += f":{self.settings.provider}:{self.settings.base_url.rstrip('/')}"
        return hashlib.sha256(identity.encode()).hexdigest()

    def _refresh(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for identifier, until in data.items():
                    if isinstance(until, (int, float)) and math.isfinite(until):
                        self._blocked[identifier] = max(until, self._blocked.get(identifier, 0))
        except (OSError, ValueError):
            pass

    def available_keys(self):
        with _LOCK:
            self._refresh()
            now = time.time()
            return [
                (slot, key)
                for slot, key in self.settings.configured_keys()
                if self._blocked.get(self._identity(key), 0) <= now
            ]

    def block(self, key, diagnostics):
        daily = any(
            "PerDay" in (v.get("quotaId") or "") for v in diagnostics.get("quota_violations", [])
        )
        delay = diagnostics.get("retry_delay_seconds", 86400 if daily else 60)
        if not isinstance(delay, (int, float)) or not math.isfinite(delay):
            delay = 86400 if daily else 60
        delay = max(1, delay)
        with _LOCK:
            self._refresh()
            now = time.time()
            self._blocked[self._identity(key)] = now + delay
            active = {k: v for k, v in self._blocked.items() if v > now}
            temporary = None
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.path.parent, suffix=".tmp", delete=False
                ) as stream:
                    temporary = stream.name
                    json.dump(active, stream)
                os.replace(temporary, self.path)
            except OSError:
                pass  # In-process cooldown still applies if persistence is unavailable.
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
        return math.ceil(delay)

    def retry_after(self):
        with _LOCK:
            self._refresh()
            now = time.time()
            delays = [
                self._blocked.get(self._identity(key), 0) - now
                for _, key in self.settings.configured_keys()
            ]
            return max(1, math.ceil(min(delays))) if delays else 60

    def public_status(self):
        available = {slot for slot, _ in self.available_keys()}
        unique = {slot for slot, _ in self.settings.configured_keys()}
        raw = self.settings.credential_slots()
        return [
            {
                "slot": slot,
                "configured": bool(key.strip()),
                "available": slot in available,
                "duplicate": bool(key.strip()) and slot not in unique,
            }
            for slot, key in enumerate(raw, 1)
        ]
