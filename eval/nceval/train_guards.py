"""The commit-message contract of a train verdict that tripped a guard.

`nceval.train` reports a passing arm whose screens tripped a guard (worst-1 % BD,
flicker, class quality, pulse) as PASS-GUARD, and its commit block carries a
`TRAIN_GUARD_LINE` line listing them. Guards are judgement calls (CLAUDE.md), but the
judgement must be written down: land.sh (`tools/dev/land_train_guards.py`) refuses a
commit carrying that line unless it also has a non-empty `JUSTIFICATION_PREFIX` line.
2026-11-23: H.264 veryfast 327c20052 and two passing arms at faster and fast had their
worst-1 % frames worse on most clips (pan +23 %); the train printed PASS and the guard
went unread.
"""

from __future__ import annotations

#: The line nceval.train's commit block writes for a tripped guard.
TRAIN_GUARD_LINE = "- nceval.train guards tripped:"
#: The line a lander adds to accept them, with the reason.
JUSTIFICATION_PREFIX = "Guard-justification:"
#: The placeholder the train writes after the prefix: not a justification.
PLACEHOLDER = "<required:"


def unjustified(message: str) -> bool:
    """Whether `message` reports tripped train guards and gives no justification."""
    lines = [line.strip() for line in message.splitlines()]
    tripped = any(line.startswith(TRAIN_GUARD_LINE) for line in lines)
    reasons = [line[len(JUSTIFICATION_PREFIX):].strip() for line in lines
               if line.startswith(JUSTIFICATION_PREFIX)]
    justified = any(r and not r.startswith(PLACEHOLDER) for r in reasons)
    return tripped and not justified
