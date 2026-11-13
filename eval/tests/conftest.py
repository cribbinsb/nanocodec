"""Shared pytest setup for the eval suite.

Unit tests score with fake tools and models, so they pin the CPU metric
backend; tests of the VMAFx backend opt in with monkeypatch.
"""

import pytest

from nceval import metrics


@pytest.fixture(autouse=True)
def cpu_metrics_backend(monkeypatch):
    monkeypatch.setenv(metrics.METRICS_BACKEND_ENV, metrics.LIBVMAF_CPU_BACKEND)
