"""VVC uses the normal streamed adapters and both independent decoder gates."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval.codecs import Codec
from nceval.conformance import DECODERS, VVC_MIN_FFMPEG_VERSION
from nceval.encoders.nanocodec import nanocodec_encoders
from nceval.tune import quantizer_bounds


def test_vvc_registration_and_capabilities():
    encoder = next(e for e in nanocodec_encoders() if e.codec == Codec.VVC)
    assert encoder.name == "nanocodec-vvc"
    assert encoder.quantizer_range == quantizer_bounds(Codec.VVC) == (10, 63)
    assert encoder.supports_recon() and encoder.verify_recon()
    assert encoder.supports_temporal_layers(1)
    assert encoder.supports_temporal_layers(2)
    assert encoder.supports_temporal_layers(3)
    assert not encoder.supports_temporal_layers(4)
    assert Codec.VVC.bitstream_suffix == ".vvc"


def test_vvc_gates_use_native_ffmpeg_and_vvdec_single_thread():
    def fake(tool, _):
        return Path("/" + tool)
    with patch("nceval.conformance.find_tool", side_effect=fake), \
            patch("nceval.conformance.find_anchor_tool", side_effect=fake), \
            patch("nceval.conformance.require_ffmpeg_version") as version_check:
        first, second = DECODERS[Codec.VVC]
        cmd = first.command(Path("/in.vvc"), Path("/dev/fd/7"), Codec.VVC, None)
        assert cmd[0] == "/ffmpeg-decode"
        version_check.assert_called_with(Path("/ffmpeg-decode"), VVC_MIN_FFMPEG_VERSION)
        assert cmd[cmd.index("-c:v") + 1] == "vvc"
        assert cmd[cmd.index("-f") + 1] == "vvc"
        assert "-xerror" in cmd
        cmd = second.command(Path("/in.vvc"), Path("/dev/fd/7"), Codec.VVC, None)
        assert cmd[0] == "/vvdecapp"
        assert cmd[cmd.index("-t") + 1] == "1"
        assert cmd[cmd.index("-o") + 1] == "/dev/fd/7"


def test_vvdec_resolves_from_anchors_only(tmp_path, monkeypatch):
    from nceval.config import Paths, find_anchor_tool
    import pytest
    paths = Paths(data_root=tmp_path, anchors=tmp_path / "anchors", nanocodec_bin=tmp_path / "bin")
    on_path = tmp_path / "path"
    on_path.mkdir()
    (on_path / "vvdecapp").write_text("#!/bin/sh\n")
    (on_path / "vvdecapp").chmod(0o755)
    monkeypatch.setenv("PATH", str(on_path))
    with pytest.raises(FileNotFoundError):
        find_anchor_tool("vvdecapp", paths)
    (tmp_path / "anchors" / "bin").mkdir(parents=True)
    pinned = tmp_path / "anchors" / "bin" / "vvdecapp"
    pinned.write_text("#!/bin/sh\n")
    pinned.chmod(0o755)
    assert DECODERS[Codec.VVC][1].binary(paths) == pinned


def _fake_ffmpeg(directory: Path, banner: str) -> Path:
    binary = directory / "ffmpeg-decode"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(f"#!/bin/sh\necho '{banner}'\n")
    binary.chmod(0o755)
    return binary


def test_vvc_gate_refuses_ffmpeg_before_8(tmp_path):
    import pytest
    from nceval.config import Paths
    paths = Paths(data_root=tmp_path, anchors=tmp_path / "anchors", nanocodec_bin=tmp_path / "bin")
    _fake_ffmpeg(tmp_path / "anchors" / "bin", "ffmpeg version 7.1.1 Copyright (c) 2000-2025")
    with pytest.raises(RuntimeError, match=r"FFmpeg 7\.1; the VVC gate needs FFmpeg >= 8\.0"):
        DECODERS[Codec.VVC][0].binary(paths)
    # Other codecs' gates are unaffected by the VVC-only bug.
    assert DECODERS[Codec.HEVC][0].binary(paths) == tmp_path / "anchors" / "bin" / "ffmpeg-decode"


def test_vvc_gate_accepts_pinned_and_distribution_ffmpeg_8(tmp_path):
    from nceval.conformance import require_ffmpeg_version
    for index, banner in enumerate(("ffmpeg version n8.0.1 Copyright (c) 2000-2025",
                                    "ffmpeg version 8.0.1-3ubuntu2 Copyright (c) 2000-2025")):
        require_ffmpeg_version(_fake_ffmpeg(tmp_path / str(index), banner), VVC_MIN_FFMPEG_VERSION)


def test_vvc_gate_refuses_unversioned_ffmpeg(tmp_path):
    import pytest
    from nceval.conformance import require_ffmpeg_version
    binary = _fake_ffmpeg(tmp_path, "ffmpeg version N-120000-gdeadbeef")
    with pytest.raises(RuntimeError, match="unrecognised version"):
        require_ffmpeg_version(binary, VVC_MIN_FFMPEG_VERSION)


def test_gate_ffmpeg_never_comes_from_path(tmp_path, monkeypatch):
    import pytest
    from nceval.config import Paths
    paths = Paths(data_root=tmp_path, anchors=tmp_path / "anchors", nanocodec_bin=tmp_path / "bin")
    _fake_ffmpeg(tmp_path / "path", "ffmpeg version 8.0.1")
    monkeypatch.setenv("PATH", str(tmp_path / "path"))
    with pytest.raises(FileNotFoundError):
        DECODERS[Codec.H264][0].binary(paths)


def test_untyped_vvc_frames_take_their_packet_key_flag():
    from nceval.framestats import frame_type
    # FFmpeg 7.1's VVC decoder leaves frames untyped; the parser flags IDR packets.
    packet_keys = [True, False]
    assert frame_type({"key_frame": 0, "pict_type": "?"}, packet_keys, 0) == "I"
    assert frame_type({"key_frame": 0, "pict_type": "?"}, packet_keys, 1) == "P"
    # Typed frames (H.264/HEVC/AV1) keep the decoder's answer.
    assert frame_type({"key_frame": 1, "pict_type": "I"}, [False], 0) == "I"
    assert frame_type({"key_frame": 0, "pict_type": "P"}, [True], 0) == "P"
