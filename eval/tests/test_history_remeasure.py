"""tools/dev/history_remeasure.py: the wrapped old builds and the frames' family names."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))

import history_remeasure as hr  # noqa: E402


def test_the_compat_script_unpacks_exactly_its_build(tmp_path, monkeypatch):
    binary = tmp_path / "nanocodec"
    binary.write_bytes(b"\x7fELF" + bytes(range(256)) * 50)
    monkeypatch.setattr(hr, "usage", lambda path: "--input --output")  # an old CLI
    script = hr.compat_script(binary, "abc123")
    offset = int(re.search(rb"tail -c \+(\d+)", script).group(1))
    assert script[offset - 1:] == binary.read_bytes()
    text = script[:offset - 1].decode()
    assert text.rstrip().endswith("exit $status")
    assert "--display-recon" in text and "--temporal-layers" in text  # adapted


def test_a_stored_key_is_named_as_its_build_named_the_preset(tmp_path):
    presets = tmp_path / "presets"
    (presets / "h264").mkdir(parents=True)
    (presets / "h264" / "fast.toml").write_text("")  # before H.264 fast -> faster
    (presets / "h264" / "faster.toml").write_text("")
    assert hr.own_name("h264", presets) == "nanocodec-h264"
    assert hr.own_name("h264@retired-faster", presets) == "nanocodec-h264@faster"
