from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Match the supplied agreement-column spelling; clarify the notice header in exports.
LEGACY_NOTICE_COLUMN = "Renewal Notice (Days)"
FIELD_COLUMNS = {
    "agreement_value": "Aggrement Value",
    "agreement_start_date": "Aggrement Start Date",
    "agreement_end_date": "Aggrement End Date",
    "termination_notice_days": "Termination Notice (Days)",
    "party_one": "Party One",
    "party_two": "Party Two",
}
CSV_COLUMNS = ["File Name", *FIELD_COLUMNS.values()]
VALIDATION_VERSION = "calendar-numeric-grounding-v1"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(description="Exact source identifier supplied with the document.")
    quote: str = Field(
        description="Faithful quotation of source text; never invented or paraphrased."
    )


class FieldResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str | None
    status: Literal["found", "missing", "ambiguous"]
    provenance: Literal["explicit", "derived"] | None
    evidence: list[Evidence]
    derivation: str | None = Field(
        description="Brief calculation for derived values, otherwise null."
    )
    explanation: str | None = Field(
        description="Reason for missing/ambiguous status, otherwise null."
    )

    @model_validator(mode="after")
    def check_consistency(self):
        if self.status == "found":
            if not self.value or not self.value.strip() or not self.evidence or not self.provenance:
                raise ValueError(
                    "Found fields require a nonempty value, evidence, and provenance."
                )
            if any(not e.quote.strip() or not e.source_id.strip() for e in self.evidence):
                raise ValueError("Evidence requires a quote and source identifier.")
            if self.provenance == "derived" and not self.derivation:
                raise ValueError("Derived fields require a calculation or explanation.")
        else:
            if self.value is not None or self.provenance is not None:
                raise ValueError(
                    "Missing/ambiguous fields must have null value and provenance."
                )
        if self.status != "found" and not self.explanation:
            raise ValueError("Missing/ambiguous fields require an explanation.")
        return self


class AgreementMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agreement_value: FieldResult
    agreement_start_date: FieldResult
    agreement_end_date: FieldResult
    termination_notice_days: FieldResult = Field(
        description="Ordinary notice required to terminate the tenancy or vacate, in days."
    )
    party_one: FieldResult
    party_two: FieldResult

    @model_validator(mode="after")
    def validate_values(self):
        """Validate date, rent and notice-period representations."""
        for name in ("agreement_start_date", "agreement_end_date"):
            value = getattr(self, name).value
            if value is not None:
                try:
                    parsed = datetime.strptime(value, "%d.%m.%Y")
                except ValueError as exc:
                    raise ValueError(f"{name} must be a real date in DD.MM.YYYY format.") from exc
                if value != f"{parsed.day:02}.{parsed.month:02}.{parsed.year:04}":
                    raise ValueError(f"{name} must use DD.MM.YYYY format.")
        notice = self.termination_notice_days.value
        if notice is not None and not (notice.isascii() and notice.isdigit()):
            raise ValueError("termination_notice_days must be a nonnegative integer.")
        amount = self.agreement_value.value
        if amount is not None:
            whole, dot, fraction = amount.strip().partition(".")
            groups = whole.split(",")
            if not (
                all(g.isascii() and g.isdigit() for g in groups)
                and (
                    len(groups) == 1
                    or (1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]))
                )
                and (not dot or (fraction.isascii() and fraction.isdigit()))
            ):
                raise ValueError("agreement_value must be a nonnegative decimal amount.")
        return self

    def values(self) -> dict[str, str | None]:
        return {name: getattr(self, name).value for name in FIELD_COLUMNS}

    def csv_row(self, identifier: str) -> dict[str, str]:
        return {
            "File Name": identifier,
            **{
                column: getattr(self, name).value or ""
                for name, column in FIELD_COLUMNS.items()
            },
        }
