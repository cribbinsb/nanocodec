"""Timing-box CPU placement shared by the benchmark's box.py and the dev CPU A/B (cpu_ab_box.py).

Standard library only: both are copied to the rented box beside their callers. The encoder
runs on one timing CPU; the harness runs on a CPU of another physical core, never on the
timing CPU's SMT sibling (measured on c6a.large, where the only other vCPU is the sibling:
+7.6 % encode time, 4-12 % by encoder; benchmark procedure response 2, T15).
"""

from __future__ import annotations

import os
from pathlib import Path


def cpus(spec: str) -> list[int]:
    """A kernel CPU list ("0-2,5") as CPU numbers."""
    out = []
    for part in spec.split(","):
        first, _, last = part.partition("-")
        out += range(int(first), int(last or first) + 1)
    return out


def harness_cpu(cpu: int) -> int | None:
    """The first CPU this process may use that is not on `cpu`'s physical core, else None."""
    siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    shared = set(cpus(siblings.read_text().strip()))
    others = sorted(os.sched_getaffinity(0) - shared)
    return others[0] if others else None


def pin_harness(cpu: int, single_cpu: bool = False) -> int:
    """Pin this process to `harness_cpu(cpu)`; returns it. Refuses (RuntimeError) a box whose
    every other CPU shares the timing CPU's core, unless `single_cpu` and the box has only that
    one CPU: on a one-vCPU box (c8g.medium) the harness stays on the timing CPU, where it only
    waits (wait4) while an encode runs (2026-11-24, docs/aws-cost-guide.md: 0.4-0.7 % per-encode
    sd, wall/CPU 1.008, as quiet as c8g.2xlarge)."""
    other = harness_cpu(cpu)
    if other is None and single_cpu and os.sched_getaffinity(0) == {cpu}:
        return cpu
    if other is None:
        raise RuntimeError(f"every other CPU shares CPU {cpu}'s core: use a larger instance")
    os.sched_setaffinity(0, {other})
    return other


def siblings(cpu: int) -> set[int]:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    return set(cpus(path.read_text().strip()))


def idle_core_cpus(timing: int, harness: int) -> list[int]:
    """The online CPUs on physical cores other than the timing CPU's and the harness's: where
    a packing test may put background load (owner, 2026-11-24)."""
    online = cpus(Path("/sys/devices/system/cpu/online").read_text().strip())
    return sorted(set(online) - siblings(timing) - siblings(harness))
