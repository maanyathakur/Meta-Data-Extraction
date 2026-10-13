from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..document_loader import DocumentContent
from ..schemas import AgreementMetadata


@dataclass(frozen=True)
class ProviderResult:
    metadata: AgreementMetadata
    raw_response: str
    model_version: str | None
    usage: dict


class LLMProvider(ABC):
    @abstractmethod
    def extract(self, document: DocumentContent, system_prompt: str) -> ProviderResult:
        """Extract validated metadata with source evidence."""
