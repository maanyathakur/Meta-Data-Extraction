from decimal import Decimal, InvalidOperation

from .schemas import FIELD_COLUMNS, AgreementMetadata

NORMALIZATION_VERSION = "whitespace-decimal-v2"


def canonical_amount(value: str) -> str:
    """Format an already-selected number; malformed/ambiguous strings stay unchanged."""
    numeric = value
    if "," in numeric:
        whole, dot, fraction = numeric.partition(".")
        groups = whole.split(",")
        if not (
            1 <= len(groups[0]) <= 3
            and all(g.isascii() and g.isdigit() for g in groups)
            and all(len(g) == 3 for g in groups[1:])
        ):
            return value
        numeric = "".join(groups) + (dot + fraction if dot else "")
    # Keep units and unsupported numeric representations unchanged.
    pieces = numeric.split(".")
    if len(pieces) > 2 or not all(p.isascii() and p.isdigit() for p in pieces):
        return value
    try:
        amount = Decimal(numeric)
    except InvalidOperation:
        return value
    formatted = format(amount, "f")
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def normalize_metadata(metadata: AgreementMetadata) -> AgreementMetadata:
    """Normalize extracted values for storage and export."""
    result = metadata.model_copy(deep=True)
    for name in FIELD_COLUMNS:
        item = getattr(result, name)
        if item.value is not None:
            item.value = " ".join(item.value.split())
    if result.agreement_value.value is not None:
        result.agreement_value.value = canonical_amount(result.agreement_value.value)
    return AgreementMetadata.model_validate(result.model_dump())
