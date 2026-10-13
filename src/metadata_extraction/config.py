import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

from .errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _positive_int(name: str, default: int, *, allow_zero=False) -> int:
    description = "a nonnegative integer" if allow_zero else "a positive integer"
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be {description}.") from exc
    if value < (0 if allow_zero else 1):
        raise ConfigurationError(f"{name} must be {description}.")
    return value


def _path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false"}:
        raise ConfigurationError(f"{name} must be true or false.")
    return value == "true"


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    api_key_2: str = field(default="", repr=False)
    api_key_3: str = field(default="", repr=False)
    llm_api_key: str = field(default="", repr=False)
    llm_api_key_2: str = field(default="", repr=False)
    llm_api_key_3: str = field(default="", repr=False)
    provider: str = "gemini"
    base_url: str = ""
    model_supports_vision: bool = True
    response_format: str = "json_object"
    token_parameter: str = "max_tokens"
    model: str = "gemini-3.8-flash"
    ocr_mode: str = "off"
    image_input: str = "auto"
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"
    cache_dir: Path = PROJECT_ROOT / ".cache" / "extractions"
    thinking_level: str = "low"
    max_output_tokens: int = 16384
    timeout_seconds: int = 240
    max_attempts: int = 3
    max_upload_mb: int = 20
    max_batch_files: int = 20
    max_image_parts: int = 40
    max_document_chars: int = 150000
    tile_height: int = 1800
    tile_overlap: int = 150

    def __post_init__(self):
        if self.provider not in {"gemini", "openai-compatible"}:
            raise ConfigurationError("LLM_PROVIDER must be gemini or openai-compatible.")
        if self.provider == "openai-compatible":
            url = urlsplit(self.base_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ConfigurationError("LLM_BASE_URL must be an HTTP(S) API base URL.")
        if self.response_format not in {"json_object", "json_schema", "none"}:
            raise ConfigurationError(
                "LLM_RESPONSE_FORMAT must be json_object, json_schema or none."
            )
        if self.token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ConfigurationError(
                "LLM_TOKEN_PARAMETER must be max_tokens or max_completion_tokens."
            )
        if self.ocr_mode not in {"off", "paddle", "rapid"}:
            raise ConfigurationError("OCR_MODE must be off, paddle, or rapid.")
        if self.image_input not in {"auto", "vision", "text"}:
            raise ConfigurationError("IMAGE_INPUT must be auto, vision or text.")
        if self.thinking_level not in {"low", "medium", "high"}:
            raise ConfigurationError("LLM_THINKING_LEVEL must be low, medium, or high.")
        if not 1 <= self.max_output_tokens <= 65536:
            raise ConfigurationError("LLM_MAX_OUTPUT_TOKENS must be between 1 and 65536.")
        if not self.model.strip():
            raise ConfigurationError("LLM_MODEL cannot be empty.")
        if not 0 <= self.tile_overlap < self.tile_height:
            raise ConfigurationError("IMAGE_TILE_OVERLAP must be smaller than IMAGE_TILE_HEIGHT.")

    def configured_keys(self):
        """Private credentials in slot order; blank/duplicate keys are skipped."""
        result, seen = [], set()
        for slot, key in enumerate(self.credential_slots(), 1):
            key = key.strip()
            if key and key not in seen:
                result.append((slot, key))
                seen.add(key)
        return result

    def credential_slots(self):
        if self.provider == "gemini":
            return self.api_key, self.api_key_2, self.api_key_3
        return self.llm_api_key, self.llm_api_key_2, self.llm_api_key_3

    @property
    def effective_image_input(self):
        if not self.model_supports_vision or self.image_input == "text":
            return "text"
        return "vision"

    @property
    def effective_ocr_mode(self):
        if not self.model_supports_vision and self.ocr_mode == "off":
            return "rapid"
        return self.ocr_mode

    def require_key(self):
        if not self.configured_keys():
            name = "GEMINI_API_KEY" if self.provider == "gemini" else "LLM_API_KEY"
            raise ConfigurationError(f"Add {name} to the project's .env file first.")

    @classmethod
    def from_env(cls):
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            api_key_2=os.getenv("GEMINI_API_KEY_2", ""),
            api_key_3=os.getenv("GEMINI_API_KEY_3", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_api_key_2=os.getenv("LLM_API_KEY_2", ""),
            llm_api_key_3=os.getenv("LLM_API_KEY_3", ""),
            provider=os.getenv("LLM_PROVIDER", "gemini").strip(),
            base_url=os.getenv("LLM_BASE_URL", "").strip().rstrip("/"),
            model_supports_vision=_boolean("MODEL_SUPPORTS_VISION", True),
            response_format=os.getenv("LLM_RESPONSE_FORMAT", "json_object").strip(),
            token_parameter=os.getenv("LLM_TOKEN_PARAMETER", "max_tokens").strip(),
            model=os.getenv("LLM_MODEL", "gemini-3.8-flash").strip(),
            ocr_mode=os.getenv("OCR_MODE", "off").strip(),
            image_input=os.getenv("IMAGE_INPUT", "auto").strip(),
            data_dir=_path("DATA_DIR", "data"),
            output_dir=_path("OUTPUT_DIR", "outputs"),
            cache_dir=_path("CACHE_DIR", ".cache/extractions"),
            thinking_level=os.getenv("LLM_THINKING_LEVEL", "low").strip(),
            max_output_tokens=_positive_int("LLM_MAX_OUTPUT_TOKENS", 16384),
            timeout_seconds=_positive_int("REQUEST_TIMEOUT_SECONDS", 240),
            max_attempts=_positive_int("MAX_ATTEMPTS", 3),
            max_upload_mb=_positive_int("MAX_UPLOAD_MB", 20),
            max_batch_files=_positive_int("MAX_BATCH_FILES", 20),
            max_image_parts=_positive_int("MAX_IMAGE_PARTS", 40),
            max_document_chars=_positive_int("MAX_DOCUMENT_CHARS", 150000),
            tile_height=_positive_int("IMAGE_TILE_HEIGHT", 1800),
            tile_overlap=_positive_int("IMAGE_TILE_OVERLAP", 150, allow_zero=True),
        )
