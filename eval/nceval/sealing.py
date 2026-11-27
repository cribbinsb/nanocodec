"""Release-only clip sets: sealed from every tuning, screening and sweep tool.

Why (docs/analysis/results-audit-2026-11-23.md finding 8): the earlier held-out
set (`independent`) was kept out of the screening pool, but it was re-measured at
seven or more builds and one sweep tuned a rate-control parameter on it. A
release-only set is read for one purpose: the measurement of a release or of a
published chart. Its card is docs/analysis/heldout-2026-11-23.md.

The rule, enforced in three places:

  - nceval.clips.load_clipset refuses a release-only set (SealedSetError) unless
    the process runs a release measurement: RELEASE_ENV=1 in its environment, set
    by hand on the release or publish command line (docs/PROCEDURES.md "Sealed
    held-out sets"), never by a tool;
  - its families are sealed from the screening pool and the learned-search data
    (nceval.pool.SEALED_SETS), read through `exclusion_only` loads, which return
    the clips for that purpose only;
  - tools/ci/check_sealed_sets.py fails when code outside the allowed files names
    a release-only set, sets RELEASE_ENV or makes an `exclusion_only` load.
"""

from __future__ import annotations

import os

#: The sealed sets (clip set names, as in their JSON "name").
RELEASE_ONLY_SETS = frozenset({"heldout_2026-11-23", "heldout_2026-11-23b"})
RELEASE_ENV = "NC_RELEASE_MEASURE"


class SealedSetError(RuntimeError):
    """A tool asked for a release-only set outside a release measurement.

    Not a ValueError: tools that skip unreadable clip set files must not skip this."""


def release_measurement() -> bool:
    return os.environ.get(RELEASE_ENV) == "1"


def check_read(set_name: str, path: object) -> None:
    """Refuse a release-only set outside a release measurement."""
    if set_name in RELEASE_ONLY_SETS and not release_measurement():
        raise SealedSetError(
            f"{path}: clip set {set_name!r} is sealed for release measurement only (nceval."
            f"sealing). No tuning, screening or sweep may read it; a release or publish "
            f"measurement runs with {RELEASE_ENV}=1 (docs/PROCEDURES.md \"Sealed held-out sets\").")


def remote_env() -> str:
    """The shell prefix that carries a release measurement to a rented box's command."""
    return f"{RELEASE_ENV}=1 " if release_measurement() else ""
