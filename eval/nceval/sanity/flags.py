"""A sanity flag: one suspected bug, with its evidence and a suspected cause.

Flags are bugs to root-cause, not results to disclose. Severity orders the
report: CRITICAL means the comparison or the encoder is plainly broken,
HIGH a large unexplained loss, MEDIUM a moderate one or a missing check,
INFO something worth knowing that is not a violation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum


class Severity(IntEnum):
    INFO = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

    @classmethod
    def parse(cls, name: str) -> Severity:
        return cls[name.upper()]


@dataclass(frozen=True)
class Flag:
    check: str            # ordering | outliers | tools | decay | harness
    severity: Severity
    subject: str          # e.g. "bodycam_airport: nanocodec-av1 vs svtav1-m12"
    finding: str          # one line: what is wrong
    evidence: dict = field(default_factory=dict)
    suspected_cause: str = ""

    def to_json(self) -> dict:
        data = asdict(self)
        data["severity"] = self.severity.name
        return data


def sort_flags(flags: list[Flag]) -> list[Flag]:
    """Most severe first, then by check and subject for a stable report."""
    return sorted(flags, key=lambda f: (-f.severity, f.check, f.subject))


def severity_by_excess(excess: float, high: float, critical: float) -> Severity:
    """MEDIUM, HIGH or CRITICAL by how far a value is past its limit."""
    if excess >= critical:
        return Severity.CRITICAL
    if excess >= high:
        return Severity.HIGH
    return Severity.MEDIUM
