"""Shared pytest setup for the eval suite.

Unit tests score with fake tools and models, so they pin the CPU metric
backend; tests of the VMAFx backend opt in with monkeypatch.
"""

import tempfile

import pytest

from nceval import metrics, tempdirs


def pytest_configure(config):
    """Every temporary path of the suite (tmp_path, tempfile) under nanocodec's own
    temporary root (nceval.tempdirs; owner, 2026-11-21: nothing of ours under another
    name in /tmp), not /tmp/pytest-of-USER."""
    tempfile.tempdir = str(tempdirs.root())


@pytest.fixture(autouse=True)
def cpu_metrics_backend(monkeypatch):
    monkeypatch.setenv(metrics.METRICS_BACKEND_ENV, metrics.LIBVMAF_CPU_BACKEND)


@pytest.fixture(autouse=True)
def private_lever_spool(monkeypatch, tmp_path):
    """Tools append to the lever ledger's spool; tests never touch the machine's."""
    monkeypatch.setenv("NC_LEVERS_SPOOL", str(tmp_path / "levers-spool.jsonl"))


@pytest.fixture(autouse=True)
def measure_here(monkeypatch):
    """Entry points offload themselves to a rented box (nceval.cloud) unless they run on
    one: tests run them here, as if on the box; tests of the offload unset it."""
    from nceval import cloud
    monkeypatch.setenv(cloud.ON_BOX_ENV, "1")
