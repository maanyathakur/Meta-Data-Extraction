from .gemini import GeminiProvider


def create_provider(settings):
    """Create the configured provider adapter."""
    if settings.provider == "gemini":
        return GeminiProvider(settings)
    from .compatible import CompatibleProvider

    return CompatibleProvider(settings)


__all__ = ["GeminiProvider", "create_provider"]
