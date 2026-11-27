"""tools/bench/pinning.py: harness placement on a one-vCPU box (affinity faked)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
import pinning  # noqa: E402


@pytest.fixture
def one_cpu(monkeypatch):
    """A c8g.medium: CPU 0 only, its own core."""
    pinned = []
    monkeypatch.setattr(pinning, "harness_cpu", lambda cpu: None)
    monkeypatch.setattr(pinning.os, "sched_getaffinity", lambda pid: {0})
    monkeypatch.setattr(pinning.os, "sched_setaffinity", lambda pid, cpus: pinned.append(cpus))
    return pinned


def test_a_one_cpu_box_is_refused_unless_asked_for(one_cpu):
    with pytest.raises(RuntimeError, match="larger instance"):
        pinning.pin_harness(0)
    assert pinning.pin_harness(0, single_cpu=True) == 0 and one_cpu == []


def test_single_cpu_never_shares_an_smt_sibling(monkeypatch, one_cpu):
    """c8i.large: two vCPUs, one core; the harness may not share it even when asked."""
    monkeypatch.setattr(pinning.os, "sched_getaffinity", lambda pid: {0, 1})
    with pytest.raises(RuntimeError):
        pinning.pin_harness(1, single_cpu=True)
