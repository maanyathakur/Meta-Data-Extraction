"""Adapter for configurable OpenAI-compatible Chat Completions endpoints."""

import base64
import json
import math
import time
from uuid import uuid4

import httpx
from pydantic import ValidationError

from ..config import Settings
from ..document_loader import DocumentContent
from ..errors import ProviderError, ProviderQuotaError
from ..key_pool import KeyPool
from ..provider_logging import safe_token, write_event
from ..schemas import AgreementMetadata
from .base import LLMProvider, ProviderResult


class CompatibleProvider(LLMProvider):
    def __init__(self, settings: Settings):
        settings.require_key()
        self.settings = settings
        self.key_pool = KeyPool(settings)

    def _payload(self, document, system_prompt):
        schema = AgreementMetadata.model_json_schema()
        text = "Extract from all source sections below, in order.\n" + "\n\n".join(
            f"[{block.source_id}]\n{block.text}" for block in document.text_blocks
        )
        content = [{"type": "text", "text": text}]
        if self.settings.effective_image_input == "vision":
            for image in document.images:
                encoded = base64.b64encode(image.data).decode("ascii")
                content.extend(
                    [
                        {"type": "text", "text": f"[Image source_id: {image.source_id}]"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
                        },
                    ]
                )
        payload = {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                    + "\nReturn only a JSON object matching this schema:\n"
                    + json.dumps(schema),
                },
                {"role": "user", "content": content if len(content) > 1 else text},
            ],
            self.settings.token_parameter: self.settings.max_output_tokens,
        }
        if self.settings.response_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "agreement_metadata", "schema": schema},
            }
        elif self.settings.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def extract(self, document: DocumentContent, system_prompt: str) -> ProviderResult:
        payload = self._payload(document, system_prompt)
        context = {
            "request_id": uuid4().hex,
            "provider": self.settings.provider,
            "document_sha256": document.sha256,
            "model": safe_token(self.settings.model),
        }
        available = self.key_pool.available_keys()
        if not available:
            raise ProviderQuotaError(self.key_pool.retry_after())
        with httpx.Client(timeout=self.settings.timeout_seconds) as client:
            for slot, key in available:
                for attempt in range(self.settings.max_attempts):
                    started = time.perf_counter()
                    event = {**context, "key_slot": slot, "attempt": attempt + 1}
                    write_event(self.settings.output_dir, {**event, "event": "request_started"})
                    try:
                        response = client.post(
                            self.settings.base_url.rstrip("/") + "/chat/completions",
                            headers={"Authorization": f"Bearer {key}"},
                            json=payload,
                        )
                        response.raise_for_status()
                        result = response.json()
                        raw = result["choices"][0]["message"]["content"]
                        if not isinstance(raw, str):
                            raise ValueError("Missing JSON response")
                        metadata = AgreementMetadata.model_validate_json(raw)
                        resolved = result.get("model")
                        usage = result.get("usage", {})
                        write_event(
                            self.settings.output_dir,
                            {
                                **event,
                                "event": "request_succeeded",
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                                "model_version": safe_token(resolved),
                            },
                        )
                        return ProviderResult(
                            metadata,
                            raw,
                            resolved if isinstance(resolved, str) else None,
                            usage if isinstance(usage, dict) else {},
                        )
                    except httpx.HTTPStatusError as exc:
                        code = exc.response.status_code
                        diagnostics = {"http_status": code}
                        if code == 429:
                            try:
                                delay = float(exc.response.headers.get("Retry-After", "60"))
                                if math.isfinite(delay):
                                    diagnostics["retry_delay_seconds"] = max(1, delay)
                            except ValueError:
                                pass
                        write_event(
                            self.settings.output_dir,
                            {
                                **event,
                                "event": "api_error",
                                **diagnostics,
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                            },
                        )
                        if code == 429:
                            delay = self.key_pool.block(key, diagnostics)
                            write_event(
                                self.settings.output_dir,
                                {**event, "event": "key_cooldown", "retry_delay_seconds": delay},
                            )
                            break
                        if code not in {500, 502, 503, 504}:
                            raise ProviderError(
                                f"Provider rejected the request (HTTP {code}); check endpoint, "
                                "key, model and supported request settings."
                            ) from exc
                        if attempt + 1 == self.settings.max_attempts:
                            raise ProviderError(f"Provider unavailable (HTTP {code}).") from exc
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        validation = (
                            [
                                safe_token(e["type"])
                                for e in exc.errors(
                                    include_input=False, include_context=False, include_url=False
                                )
                            ]
                            if isinstance(exc, ValidationError)
                            else ["invalid_response"]
                        )
                        write_event(
                            self.settings.output_dir,
                            {
                                **event,
                                "event": "schema_error",
                                "validation_types": validation,
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                            },
                        )
                        if attempt + 1 == self.settings.max_attempts:
                            raise ProviderError(
                                "Provider did not return schema-valid metadata."
                            ) from exc
                    except httpx.RequestError as exc:
                        write_event(
                            self.settings.output_dir,
                            {
                                **event,
                                "event": "connection_error",
                                "exception_type": safe_token(type(exc).__name__),
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                            },
                        )
                        raise ProviderError(
                            "Provider connection failed; check endpoint/network."
                        ) from exc
                    time.sleep(min(2**attempt, 8))
        raise ProviderQuotaError(self.key_pool.retry_after())
