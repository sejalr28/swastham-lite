

from __future__ import annotations
import re
from dataclasses import dataclass


CRISIS_PATTERNS = [
    r"\bsuicid",
    r"\bself[- ]?harm",
    r"\bself[- ]?injur",
    r"\bkill(ing)? myself\b",
    r"\bend(ing)? (it all|my life)\b",
    r"\bdon'?t want to (live|be alive)\b",
    r"\bwant to die\b",
    r"\bno reason to live\b",
    r"\bcan't breathe\b",
    r"\bchest pain\b",
    r"\boverdose\b",
]

DIAGNOSIS_PATTERNS = [
    r"\bdo i have\b", r"\bam i (developing|showing signs of)\b",
    r"\bdiagnos", r"\bwhat medication\b", r"\bwhat dose\b", r"\bhow much .* should i take\b",
]

CRISIS_RESPONSE = (
    "I'm not able to help with this directly, and I want to make sure you get the "
    "right support. If this is a medical emergency, please contact local emergency "
    "services right away. If you're going through a mental health crisis, please "
    "reach out to a crisis helpline or a trusted person immediately. This assistant "
    "only provides general sleep-hygiene information and isn't equipped to help in "
    "a crisis."
)


@dataclass
class SafetyCheckResult:
    is_crisis: bool
    is_diagnosis_seeking: bool
    fixed_response: str | None = None


def check(query: str) -> SafetyCheckResult:
    q = query.lower()

    for pat in CRISIS_PATTERNS:
        if re.search(pat, q):
            return SafetyCheckResult(is_crisis=True, is_diagnosis_seeking=False,
                                      fixed_response=CRISIS_RESPONSE)

    is_diag = any(re.search(pat, q) for pat in DIAGNOSIS_PATTERNS)
    return SafetyCheckResult(is_crisis=False, is_diagnosis_seeking=is_diag)
