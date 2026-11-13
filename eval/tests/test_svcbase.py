"""Spatial SVC streams: per-layer bytes (AV1 OBUs, VP9 superframes), base-only
streams, the rate split and the two-layer setting test."""

from __future__ import annotations

from pathlib import Path

from nceval import svcbase
from nceval.codecs import Codec


def obu(obu_type: int, payload: bytes, spatial_id: int | None = None) -> bytes:
    """An OBU with a size field and, for a layer's OBU, an extension header."""
    header = (obu_type << 3) | svcbase.OBU_HAS_SIZE
    extension = b""
    if spatial_id is not None:
        header |= svcbase.OBU_HAS_EXTENSION
        extension = bytes([spatial_id << 3])
    return bytes([header]) + extension + bytes([len(payload)]) + payload


def test_base_layer_bytes_by_spatial_id():
    delimiter = obu(2, b"")  # no extension: counted with the base
    base, enhancement = obu(6, b"x" * 10, 0), obu(6, b"y" * 20, 1)
    assert svcbase.base_layer_bytes(delimiter + base + enhancement, Codec.AV1) == (
        len(delimiter) + len(base))


def test_operating_point_bytes_of_three_layers():
    delimiter = obu(2, b"")
    s0, s1, s2 = obu(6, b"x" * 10, 0), obu(6, b"y" * 20, 1), obu(6, b"z" * 40, 2)
    unit = delimiter + s0 + s1 + s2
    assert svcbase.av1_base_bytes(unit, 0) == len(delimiter) + len(s0)
    assert svcbase.av1_base_bytes(unit, 1) == len(delimiter) + len(s0) + len(s1)
    assert svcbase.av1_base_bytes(unit, 2) == len(unit)


def superframe(frames: list[bytes]) -> bytes:
    """A VP9 superframe (Annex B.3): the frames, then an index of 2-byte sizes."""
    marker = svcbase.SUPERFRAME_MARKER | (1 << 3) | (len(frames) - 1)
    index = b"".join(len(f).to_bytes(2, "little") for f in frames)
    return b"".join(frames) + bytes([marker]) + index + bytes([marker])


def test_vp9_base_bytes_are_the_base_only_stream():
    # A shown S0 (libvpx) goes alone; a unit of one frame is the base's.
    s0, s1 = b"\x82" + b"a" * 300, b"\x86" + b"b" * 40
    unit = superframe([s0, s1])
    assert svcbase.vp9_superframe_sizes(unit) == [len(s0), len(s1)]
    assert svcbase.base_layer_bytes(unit, Codec.VP9) == len(s0)
    assert svcbase.base_layer_bytes(s0, Codec.VP9) == len(s0)
    # A hidden S0 (ours) is shown by a one-byte show_existing_frame in a
    # superframe of two (index: marker, two 2-byte sizes, marker), as a unit
    # coding S0 alone is.
    hidden = b"\x84" + b"a" * 300
    alone = superframe([hidden, b"\x88"])
    assert svcbase.base_layer_bytes(superframe([hidden, s1]), Codec.VP9) == len(alone)
    assert svcbase.base_layer_bytes(alone, Codec.VP9) == len(alone)


def test_vp9_first_frames_is_the_libvpx_base_stream(tmp_path: Path):
    source, base = tmp_path / "s.ivf", tmp_path / "b.ivf"
    s0, s1 = b"\x86" + b"a" * 30, b"\x86" + b"b" * 50
    svcbase.write_ivf(source, b"DKIF" + bytes(28), [(0, superframe([s0, s1])), (1, s0)])
    svcbase.vp9_first_frames(source, base)
    assert svcbase.ivf_frames(base.read_bytes()) == [(0, s0), (1, s0)]


def test_split_rate3_keeps_the_total_at_the_three_layer_shares():
    for total in (40, 150, 301, 1200):
        s0, s1, s2 = svcbase.split_rate3(total)
        assert s0 + s1 + s2 == total
        for rate, share in zip((s0, s1), svcbase.THREE_LAYER_RATE_SHARES):
            assert abs(rate - total * share) <= 0.5


def test_layer_size_cascades_even_halves():
    from nceval.clips import Clip
    clip = Clip(name="c", yuv=Path("c.yuv"), width=1280, height=720, fps=30.0, frames=150,
                source_kbps=10000, content_class="pan", split="val")
    assert svcbase.layer_size(clip, 0) == (1280, 720)
    assert svcbase.layer_size(clip, 1) == svcbase.base_size(clip) == (640, 360)
    assert svcbase.layer_size(clip, 2) == (320, 180)


def test_split_rate_keeps_the_total():
    for total in (40, 150, 301, 1200):
        base, full = svcbase.split_rate(total)
        assert base + full == total
        assert abs(base - total * svcbase.BASE_RATE_SHARE) <= 0.5




def test_is_spatial_reads_the_layers_param():
    assert svcbase.is_spatial(["--param", "spatial.layers=2"])
    assert svcbase.is_spatial(["--param", "spatial.layers=2.0", "--param", "quant.trellis=0"])
    assert not svcbase.is_spatial(["--param", "spatial.layers=1"])
    assert not svcbase.is_spatial(["--param", "quant.trellis=0"])


def test_base_size_rounds_up_to_even_as_the_encoder():
    from nceval.clips import Clip
    clip = lambda w, h: Clip(name="c", yuv=Path("c.yuv"), width=w, height=h, fps=30.0,  # noqa: E731
                             frames=150, source_kbps=1e4, content_class="pan", split="val")
    assert svcbase.base_size(clip(1280, 720)) == (640, 360)
    assert svcbase.base_size(clip(1884, 720)) == (942, 360)
    assert svcbase.base_size(clip(854, 480)) == (428, 240)
