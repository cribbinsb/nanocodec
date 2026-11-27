"""nceval.clipframes: the reference frames of encoded clips come from one pinned decoder."""

import os

import pytest

from nceval import clipframes

#: The real probe (conftest pins clip_decoder for every other test).
PROBE = clipframes.clip_decoder


def fake_ffmpeg(directory, version):
    directory.mkdir()
    binary = directory / "ffmpeg"
    binary.write_text(f"#!/bin/sh\necho 'ffmpeg version {version} Copyright'\n")
    binary.chmod(0o755)
    return binary


def test_clips_decode_only_with_the_pinned_ffmpeg_release(tmp_path, monkeypatch):
    """Results audit 2026-11-23 (metric validation): another release decoded VC-1K clips
    to other reference frames; a clip is decoded only by CLIP_DECODER_VERSION."""
    old = fake_ffmpeg(tmp_path / "conda", "7.1.1")
    pinned = fake_ffmpeg(tmp_path / "system", f"{clipframes.CLIP_DECODER_VERSION}-3ubuntu2")
    monkeypatch.setattr(clipframes, "CLIP_DECODER_CANDIDATES", ())
    monkeypatch.setenv("PATH", f"{old.parent}{os.pathsep}{pinned.parent}")
    PROBE.cache_clear()
    assert PROBE() == pinned
    monkeypatch.setenv("PATH", str(old.parent))
    PROBE.cache_clear()
    with pytest.raises(FileNotFoundError, match="7.1.1|no ffmpeg"):
        PROBE()
    PROBE.cache_clear()


def test_the_decoder_runs_without_the_python_prefix_libraries(monkeypatch):
    lib = f"{clipframes.sys.prefix}/lib"
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{lib}{os.pathsep}/usr/local/cuda/lib64")
    assert clipframes.decoder_environment()["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64"


def test_frame_sizes_probe_without_the_python_prefix_libraries(tmp_path, monkeypatch):
    """Rebaseline 2026-11-25: ffprobe in framestats inherited cv2's conda library path and
    exited 127 (libopenmpt against conda's libmpg123), failing every GPU-scored screen."""
    from nceval import framestats
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return clipframes.subprocess.CompletedProcess(cmd, 0, stdout='{"packets": []}')

    monkeypatch.setenv("LD_LIBRARY_PATH", f"{clipframes.sys.prefix}/lib")
    monkeypatch.setattr(framestats, "find_tool", lambda name, paths: name)
    monkeypatch.setattr(framestats.subprocess, "run", fake_run)
    stream = tmp_path / "s.264"
    stream.write_bytes(b"\0" * 32)
    assert framestats.probe_frames(stream, None) == []
    assert clipframes.sys.prefix not in seen["env"]["LD_LIBRARY_PATH"]
