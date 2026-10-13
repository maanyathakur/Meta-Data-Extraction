import time
from uuid import uuid4

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from ..config import Settings
from ..document_loader import DocumentContent
from ..errors import ProviderError, ProviderQuotaError
from ..key_pool import KeyPool
from ..provider_logging import api_diagnostics, safe_token, write_event
from ..schemas import AgreementMetadata
from .base import LLMProvider, ProviderResult


class GeminiProvider(LLMProvider):
    def __init__(self, settings: Settings):
        settings.require_key()
        self.settings = settings
        self.key_pool = KeyPool(settings)

    def extract(self, document: DocumentContent, system_prompt: str) -> ProviderResult:
        parts = [types.Part.from_text(text="Extract from all source sections below, in order.")]
        for block in document.text_blocks:
            parts.append(types.Part.from_text(text=f"[{block.source_id}]\n{block.text}"))
        for image in document.images if self.settings.effective_image_input == "vision" else ():
            parts.extend(
                [
                    types.Part.from_text(text=f"[Image source_id: {image.source_id}]"),
                    types.Part.from_bytes(data=image.data, mime_type=image.mime_type),
                ]
            )
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_json_schema=AgreementMetadata.model_json_schema(),
            temperature=0,
            max_output_tokens=self.settings.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.settings.thinking_level),
        )
        request_id = uuid4().hex
        context = {
            "request_id": request_id,
            "provider": self.settings.provider,
            "document_sha256": document.sha256,
            "model": safe_token(self.settings.model),
        }
        available = self.key_pool.available_keys()
        if not available:
            raise ProviderQuotaError(self.key_pool.retry_after())
        # Quota failures advance to the next configured project; other errors keep their meaning.
        for slot, key in available:
            context["key_slot"] = slot
            with genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=self.settings.timeout_seconds * 1000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            ) as client:
                for attempt in range(self.settings.max_attempts):
                    started = time.perf_counter()
                    attempt_context = {**context, "attempt": attempt + 1}
                    write_event(
                        self.settings.output_dir, {**attempt_context, "event": "request_started"}
                    )
                    response = None
                    try:
                        response = client.models.generate_content(
                            model=self.settings.model,
                            contents=[types.Content(role="user", parts=parts)],
                            config=config,
                        )
                        raw = response.text or ""
                        metadata = AgreementMetadata.model_validate_json(raw)
                        usage = response.usage_metadata
                        write_event(
                            self.settings.output_dir,
                            {
                                **attempt_context,
                                "event": "request_succeeded",
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                                "model_version": safe_token(response.model_version),
                            },
                        )
                        return ProviderResult(
                            metadata,
                            raw,
                            response.model_version,
                            usage.model_dump(mode="json") if usage else {},
                        )
                    except errors.APIError as exc:
                        diagnostics = api_diagnostics(exc)
                        write_event(
                            self.settings.output_dir,
                            {
                                **attempt_context,
                                "event": "api_error",
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                                **diagnostics,
                            },
                        )
                        code = getattr(exc, "code", None)
                        if code == 429:
                            delay = self.key_pool.block(key, diagnostics)
                            write_event(
                                self.settings.output_dir,
                                {
                                    **attempt_context,
                                    "event": "key_cooldown",
                                    "retry_delay_seconds": delay,
                                },
                            )
                            break  # Never immediately retry a quota-exhausted slot.
                        if code not in {500, 502, 503, 504}:
                            raise ProviderError(
                                f"Gemini rejected the request (HTTP {code}); "
                                "check key/model access."
                            ) from exc
                        if attempt + 1 == self.settings.max_attempts:
                            raise ProviderError(
                                f"Gemini unavailable (HTTP {code}); retry later."
                            ) from exc
                    except ValidationError as exc:
                        write_event(
                            self.settings.output_dir,
                            {
                                **attempt_context,
                                "event": "schema_error",
                                "validation_types": [
                                    safe_token(e["type"])
                                    for e in exc.errors(
                                        include_input=False,
                                        include_context=False,
                                        include_url=False,
                                    )
                                ],
                                "finish_reason": safe_token(
                                    str(response.candidates[0].finish_reason)
                                )
                                if response and getattr(response, "candidates", None)
                                else None,
                                "response_characters": len(response.text or "") if response else 0,
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                            },
                        )
                        if attempt + 1 == self.settings.max_attempts:
                            raise ProviderError(
                                "Gemini did not return consistent schema-valid metadata."
                            ) from exc
                    except Exception as exc:
                        write_event(
                            self.settings.output_dir,
                            {
                                **attempt_context,
                                "event": "connection_error",
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                                "exception_type": safe_token(type(exc).__name__),
                            },
                        )
                        raise ProviderError(
                            "Gemini connection failed; check network and timeout."
                        ) from exc
                    time.sleep(min(2**attempt, 8))
        raise ProviderQuotaError(self.key_pool.retry_after())
