"""Evaluation-only name formatting and conservative initial comparison."""

import re
import unicodedata
from difflib import SequenceMatcher

from rapidfuzz.fuzz import ratio

NAME_COLUMNS = {"Party One", "Party Two"}
# SRI is retained conservatively: text alone cannot establish whether it names a company.
TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "shri", "shree", "smt"}
COMPANY_WORDS = {
    "ltd",
    "limited",
    "private",
    "pvt",
    "llc",
    "llp",
    "inc",
    "incorporated",
    "company",
    "corp",
    "corporation",
}


def clean_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"\band\s*/\s*or\b", " / ", text)
    text = text.replace("'", "").replace("’", "")
    tokens = re.findall(r"[^\W_]+|[&/+@#%]", text)
    tokens = ["&" if token == "and" else token for token in tokens]
    cleaned, party = [], []

    def append_party():
        # Company prefixes may be part of the registered name, including SRI.
        if not COMPANY_WORDS.intersection(party):
            while party and party[0] in TITLES:
                party.pop(0)
        cleaned.extend(party)
        party.clear()

    for token in tokens:
        if token in {"&", "/", "or"}:
            append_party()
            cleaned.append(token)
        else:
            party.append(token)
    append_party()
    return " ".join(cleaned)


def conflicting_initials(expected: str, predicted: str) -> bool:
    def initial(token):
        return len(token) == 1 and token.isalpha()

    left, right = expected.split(), predicted.split()
    for operation, a, b, c, d in SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        if operation == "equal":
            continue
        old, new = left[a:b], right[c:d]
        for first, second in zip(old, new, strict=False):
            if (initial(first) or initial(second)) and first[0] != second[0]:
                return True
        # Missing initials cannot be assumed equivalent to an unrelated name component.
        remainder = old[len(new) :] + new[len(old) :]
        if any(initial(token) for token in remainder):
            return True
    return False


def fuzzy_conflicts(expected: str, predicted: str) -> list[str]:
    """Protect identity-bearing tokens before allowing one small spelling difference."""
    left, right = expected.split(), predicted.split()
    conflicts = []
    if conflicting_initials(expected, predicted):
        conflicts.append("conflicting or missing initials")
    if [t for t in left if any(c.isdigit() for c in t)] != [
        t for t in right if any(c.isdigit() for c in t)
    ]:
        conflicts.append("different numeric identifiers")
    protected = COMPANY_WORDS | {"&", "/", "or", "+", "@", "#", "%"}
    if [t for t in left if t in protected] != [t for t in right if t in protected]:
        conflicts.append("different party separators, symbols or company markers")
    if len(left) != len(right):
        conflicts.append("missing or additional name components")
    else:
        changed = [(a, b) for a, b in zip(left, right, strict=True) if a != b]
        if len(changed) > 1:
            conflicts.append("multiple changed name components")
        elif changed and ratio(*changed[0]) < 85:
            conflicts.append("name component similarity below 85%")
    return conflicts


def name_outcome(expected: str, predicted: str) -> tuple[str, float, bool]:
    if expected == predicted:
        return "Match", 100.0, False
    similarity = ratio(expected, predicted)
    conflict = conflicting_initials(expected, predicted)
    if expected and predicted and similarity > 97 and not fuzzy_conflicts(expected, predicted):
        return "Fuzzy match", similarity, False
    if expected and predicted and similarity > 90:
        return "Review", similarity, conflict
    return "Mismatch", similarity, conflict
