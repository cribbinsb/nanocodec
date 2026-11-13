"""VP9 reference adapter and conformance integration, without codec executables."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval.clips import Clip
from nceval.codecs import Codec
from nceval.conformance import DECODERS
from nceval.encoders.anchors import LibvpxRealtime
from nceval.encoders.base import CodingStructure, EncodeJob
from nceval.sweep import payload_bytes


def test_realtime_job_and_unsupported_layers():
    clip = Clip("synthetic", Path("/in.yuv"), 64, 48, 30, 5, 1000, "synthetic", "train")
    encoder = LibvpxRealtime(8)
    with patch("nceval.encoders.anchors.find_tool", return_value=Path("/vpxenc")):
        cmd = encoder.command(EncodeJob(clip, 40, CodingStructure(3), 5, Path("/out.ivf")), None)
        assert "--lag-in-frames=0" in cmd and "--auto-alt-ref=0" in cmd
        assert "--passes=1" in cmd and "--threads=1" in cmd
        assert "--min-q=40" in cmd and "--max-q=40" in cmd
        with pytest.raises(ValueError, match="temporal layers"):
            encoder.command(EncodeJob(clip, 40, CodingStructure(4, 2), 5, Path("/out.ivf")), None)
        with pytest.raises(ValueError, match="quantizer"):
            encoder.command(EncodeJob(clip, 64, CodingStructure(3), 5, Path("/out.ivf")), None)


def test_gate_uses_independent_native_ffmpeg_decoder():
    fake = lambda tool, _: Path("/" + tool)  # noqa: E731
    with patch("nceval.conformance.find_tool", side_effect=fake), \
            patch("nceval.conformance.find_anchor_tool", side_effect=fake):
        first, second = DECODERS[Codec.VP9]
        cmd = first.command(Path("/in.ivf"), Path("/out.yuv"), Codec.VP9, None)
        assert cmd[0] == "/ffmpeg-decode"
        assert cmd[cmd.index("-c:v") + 1] == "vp9"
        assert "libvpx-vp9" not in cmd
        assert second.command(Path("/in.ivf"), Path("/out.yuv"), Codec.VP9, None)[0] == "/vpxdec"


def test_vp9_bitrate_excludes_ivf_container(tmp_path):
    packet = tmp_path / "packet.ivf"
    frame = (22).to_bytes(4, "little") + bytes(8 + 22)
    packet.write_bytes(bytes(32) + 2 * frame)
    assert payload_bytes(packet, Codec.VP9, 2) == 100 - 32 - 24
