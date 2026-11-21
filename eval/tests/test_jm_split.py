"""nceval.jm_split runs as a script (nceval.conformance starts it with `python3 -I`)."""

import subprocess
import sys

from nceval import jm_split


def test_runs_as_an_isolated_script(tmp_path):
    """Its imports resolve outside the package: a missing ldecod fails the decode,
    not the import (2026-11-21: `from . import tempdirs` broke every JM resize check)."""
    stream = tmp_path / "one.264"
    stream.write_bytes(b"\x00\x00\x00\x01\x67\x42")
    result = subprocess.run([sys.executable, "-I", jm_split.__file__,
                             str(tmp_path / "no-ldecod"), str(stream), str(tmp_path / "out.yuv")],
                            capture_output=True, text=True, check=False)
    assert "ImportError" not in result.stderr
    assert "FileNotFoundError" in result.stderr
