import hashlib
import json
import os
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel, Field

from .config import Settings
from .document_loader import LOADER_VERSION, DocumentContent, load_bytes, load_document
from .errors import ProviderError
from .normalizer import NORMALIZATION_VERSION, normalize_metadata
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT
from .provider_logging import write_event
from .providers.base import LLMProvider
from .schemas import FIELD_COLUMNS, VALIDATION_VERSION, AgreementMetadata

GROUNDING_VERSION = "verbatim-unique-text-citations-v2"


class ExtractionRecord(BaseModel):
    filename: str
    document_sha256: str
    created_at: str
    provider: str
    model: str
    model_version: str | None
    prompt_version: str
    configuration_hash: str
    normalization_version: str = NORMALIZATION_VERSION
    cached: bool = False
    metadata: AgreementMetadata
    warnings: list[str] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    raw_response: str


def write_json(path: Path, value: dict):
    """Atomic writes prevent partial cache/results after interrupted requests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def grounding_issues(metadata: AgreementMetadata, document: DocumentContent) -> list[dict]:
    issues = []
    blocks = {b.source_id: b.text for b in document.text_blocks}
    for name in FIELD_COLUMNS:
        for evidence in getattr(metadata, name).evidence:
            if evidence.source_id not in document.source_ids:
                issues.append({"field": name, "source_id": None, "reason": "unknown_source"})
            elif evidence.source_id in blocks:
                quote = " ".join(evidence.quote.split())
                if quote not in " ".join(blocks[evidence.source_id].split()):
                    issues.append({
                        "field": name, "source_id": evidence.source_id, "reason": "quote_mismatch"
                    })
    return issues


def correct_text_citations(metadata: AgreementMetadata, document: DocumentContent):
    """Relocate a verbatim quote only when exactly one supplied text section contains it."""
    blocks = {block.source_id: " ".join(block.text.split()) for block in document.text_blocks}
    result = metadata.model_copy(deep=True)
    corrections = []
    for name in FIELD_COLUMNS:
        field = getattr(result, name)
        for index, evidence in enumerate(field.evidence):
            quote = " ".join(evidence.quote.split())
            if evidence.source_id not in blocks or not quote or quote in blocks[evidence.source_id]:
                continue
            matches = [source for source, text in blocks.items() if quote in text]
            if len(matches) == 1:
                corrections.append({
                    "field": name, "original_source_id": evidence.source_id,
                    "source_id": matches[0],
                })
                field.evidence[index] = evidence.model_copy(update={"source_id": matches[0]})
    return result, corrections


def validate_grounding(metadata: AgreementMetadata, document: DocumentContent, output_dir=None):
    issues = grounding_issues(metadata, document)
    if issues:
        if output_dir is not None:
            write_event(output_dir, {
                "event": "evidence_validation_failed",
                "document_sha256": document.sha256,
                "issues": issues,
            })
        reasons = [
            f"{FIELD_COLUMNS[issue['field']].replace('Aggrement', 'Agreement')}: " + (
                "source ID was not supplied to the model" if issue["reason"] == "unknown_source"
                else f"quotation differs from source {issue['source_id']}"
            )
            for issue in issues[:3]
        ]
        suffix = f"; {len(issues) - 3} more issue(s)" if len(issues) > 3 else ""
        raise ProviderError(
            "Evidence validation failed: " + "; ".join(reasons) + suffix + ". "
            "No result was cached or exported; retry extraction."
        )
    return []


class ExtractionPipeline:
    def __init__(self, settings: Settings, provider: LLMProvider | None = None):
        self.settings = settings
        self.provider = provider

    def _configuration_hash(self):
        config = {
            "provider": self.settings.provider,
            "model": self.settings.model,
            "prompt": SYSTEM_PROMPT,
            "schema": AgreementMetadata.model_json_schema(),
            "validation": VALIDATION_VERSION,
            "grounding": GROUNDING_VERSION,
            "loader": LOADER_VERSION,
            "ocr": self.settings.effective_ocr_mode,
            "ocr_engine": f"{self.settings.effective_ocr_mode}-cpu-v1",
            "image_input": self.settings.effective_image_input,
            "tile_height": self.settings.tile_height,
            "tile_overlap": self.settings.tile_overlap,
            "temperature": 0 if self.settings.provider == "gemini" else None,
            "max_output_tokens": self.settings.max_output_tokens,
            "thinking_level": self.settings.thinking_level
            if self.settings.provider == "gemini"
            else None,
            "sdk_version": version(
                "google-genai" if self.settings.provider == "gemini" else "httpx"
            ),
        }
        if self.settings.provider != "gemini":
            config.update(
                base_url=self.settings.base_url.rstrip("/"),
                response_format=self.settings.response_format,
                token_parameter=self.settings.token_parameter,
            )
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def extract_path(self, path: Path, use_cache=True):
        return self.extract_document(load_document(path, self.settings), use_cache)

    def extract_bytes(self, data: bytes, filename: str, use_cache=True):
        return self.extract_document(load_bytes(data, filename, self.settings), use_cache)

    def extract_document(self, document: DocumentContent, use_cache=True):
        evidence_document = (
            document
            if self.settings.effective_image_input == "vision"
            else replace(document, images=())
        )
        configuration_hash = self._configuration_hash()
        cache_key = hashlib.sha256(f"{document.sha256}:{configuration_hash}".encode()).hexdigest()
        cache_file = self.settings.cache_dir / f"{cache_key}.json"
        if use_cache and cache_file.exists():
            try:
                record = ExtractionRecord.model_validate_json(
                    cache_file.read_text(encoding="utf-8")
                )
                if (
                    record.document_sha256 == document.sha256
                    and record.configuration_hash == configuration_hash
                ):
                    warnings = validate_grounding(
                        record.metadata, evidence_document, self.settings.output_dir
                    )
                    return record.model_copy(
                        update={
                            "filename": document.filename,
                            "cached": True,
                            "metadata": normalize_metadata(record.metadata),
                            "normalization_version": NORMALIZATION_VERSION,
                            "warnings": warnings,
                        }
                    )
            except (ValueError, OSError, ProviderError):
                pass
        if self.provider is None:
            from .providers import create_provider

            self.provider = create_provider(self.settings)
        response = self.provider.extract(document, SYSTEM_PROMPT)
        metadata = normalize_metadata(response.metadata)
        metadata, corrections = correct_text_citations(metadata, evidence_document)
        if corrections:
            write_event(self.settings.output_dir, {
                "event": "evidence_source_corrected",
                "document_sha256": document.sha256,
                "corrections": corrections,
            })
        record = ExtractionRecord(
            filename=document.filename,
            document_sha256=document.sha256,
            created_at=datetime.now(UTC).isoformat(),
            provider=self.settings.provider,
            model=self.settings.model,
            model_version=response.model_version,
            prompt_version=PROMPT_VERSION,
            configuration_hash=configuration_hash,
            metadata=metadata,
            warnings=validate_grounding(metadata, evidence_document, self.settings.output_dir),
            usage=response.usage,
            raw_response=response.raw_response,
        )
        if use_cache:
            write_json(cache_file, record.model_dump(mode="json"))
        return record

