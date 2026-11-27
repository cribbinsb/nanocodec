"""nceval.streamstrip: SEI and metadata units leave, picture units stay byte for byte."""
from nceval.codecs import Codec
from nceval.streamstrip import _leb128_bytes, strip_annexb, strip_ivf_av1, strip_ivf_av2

SPS = b"\x00\x00\x00\x01\x67\x42\x00\x1e"
SEI = b"\x00\x00\x01\x06\x05\x10" + b"x" * 16
IDR = b"\x00\x00\x01\x65\x88\x84\x80"  # a NAL never ends in a zero byte (rbsp trailing bits)
P = b"\x00\x00\x01\x41\x9a\x02"


def test_h264_every_sei_removed_in_a_multi_frame_stream():
    data = SPS + SEI + IDR + SEI + P + P
    out, removed = strip_annexb(data, Codec.H264)
    assert out == SPS + IDR + P + P
    assert removed == 2 * len(SEI)


def test_hevc_prefix_and_suffix_sei():
    vps = b"\x00\x00\x00\x01\x40\x01\x0c"
    prefix = b"\x00\x00\x01\x4e\x01\x05\x02ab"   # type 39
    suffix = b"\x00\x00\x01\x50\x01\x05\x02cd"   # type 40
    idr = b"\x00\x00\x01\x26\x01\xaf"           # type 19
    out, removed = strip_annexb(vps + prefix + idr + suffix, Codec.HEVC)
    assert out == vps + idr and removed == len(prefix) + len(suffix)


def _obu(kind: int, payload: bytes) -> bytes:
    return bytes([(kind << 3) | 0x2]) + _leb128_bytes(len(payload)) + payload


def test_av1_metadata_obu_removed_and_ivf_sizes_rewritten():
    td, seq = _obu(2, b""), _obu(1, b"seqhdr")
    meta, frame = _obu(5, b"\x01" * 40), _obu(6, b"f" * 300)
    payload = td + seq + meta + frame
    ivf = b"DKIF" + b"\x00" * 28 + len(payload).to_bytes(4, "little") + b"\x00" * 8 + payload
    out, removed = strip_ivf_av1(ivf)
    kept = td + seq + frame
    assert removed == len(meta)
    assert out == b"DKIF" + b"\x00" * 28 + len(kept).to_bytes(4, "little") + b"\x00" * 8 + kept


def _av2_obu(kind: int, payload: bytes) -> bytes:
    return _leb128_bytes(len(payload) + 1) + bytes([kind << 2]) + payload


def test_av2_short_and_group_metadata_removed():
    td, seq = _av2_obu(2, b""), _av2_obu(1, b"seqhdr")
    short, group = _av2_obu(8, b"\x00\x0a" + b"u" * 200), _av2_obu(9, b"\x00" * 4)
    key = _av2_obu(4, b"k" * 150)
    payload = td + seq + short + group + key
    ivf = b"DKIF" + b"\x00" * 28 + len(payload).to_bytes(4, "little") + b"\x00" * 8 + payload
    out, removed = strip_ivf_av2(ivf)
    kept = td + seq + key
    assert removed == len(short) + len(group)
    assert out == b"DKIF" + b"\x00" * 28 + len(kept).to_bytes(4, "little") + b"\x00" * 8 + kept
