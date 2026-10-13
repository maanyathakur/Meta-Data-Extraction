class ExtractionError(Exception):
    """An actionable pipeline error without credentials or provider response bodies."""


class ConfigurationError(ExtractionError):
    pass


class DocumentError(ExtractionError):
    pass


class ProviderError(ExtractionError):
    pass


class ProviderQuotaError(ProviderError):
    """Every configured slot is cooling down; callers should stop fresh requests."""

    def __init__(self, retry_after_seconds: int):
        self.retry_after_seconds = max(1, retry_after_seconds)
        super().__init__(
            "All configured provider key slots are quota-limited; "
            f"retry in approximately {self.retry_after_seconds} seconds."
        )
