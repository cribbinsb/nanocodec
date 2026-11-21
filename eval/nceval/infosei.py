"""Remove an encoder's information SEI from a one-frame H.264 / HEVC stream.

x264 and x265 open every stream with a user_data_unregistered SEI naming the
program and its options (about 0.6 kB and 1.5-2 kB). Over a 150-frame point that is
noise; a key frame at a real-time ladder's low rungs is 3-6 kB, so the key-frame
studies (`nceval.fixedqp --key-share`) drop it: it is not picture data, a decoder
ignores it, and a product would not send it. Our encoder writes none.
"""

from __future__ import annotations

import re
from pathlib import Path

from .codecs import Codec

START_CODE = re.compile(b"\x00\x00\x01")
#: SEI NAL unit types: H.264 6 (H.264 Table 7-1); HEVC prefix SEI 39 (HEVC Table 7-1).
H264_SEI = 6
HEVC_PREFIX_SEI = 39
#: payloadType of user_data_unregistered (H.264 / HEVC Annex D).
USER_DATA_UNREGISTERED = 5


def _is_info_sei(nal: bytes, codec: Codec) -> bool:
    """Whether `nal` (from its header byte) is an SEI whose first message is
    user_data_unregistered."""
    if codec is Codec.H264:
        return (nal[0] & 0x1F) == H264_SEI and nal[1] == USER_DATA_UNREGISTERED
    return ((nal[0] >> 1) & 0x3F) == HEVC_PREFIX_SEI and nal[2] == USER_DATA_UNREGISTERED


def strip_info_sei(stream: Path, codec: Codec) -> int:
    """Rewrite Annex-B `stream` without its information SEI units; returns the
    bytes removed (0 for the other codecs)."""
    if codec not in (Codec.H264, Codec.HEVC):
        return 0
    data = stream.read_bytes()
    starts = [m.start() for m in START_CODE.finditer(data)]
    kept, removed = bytearray(), 0
    for index, start in enumerate(starts):
        # a four-byte start code's leading zero belongs to the unit it starts
        begin = start - 1 if start > 0 and data[start - 1] == 0 else start
        end = starts[index + 1] if index + 1 < len(starts) else len(data)
        if index + 1 < len(starts) and data[end - 1] == 0:
            end -= 1
        unit = data[begin:end]
        if _is_info_sei(data[start + 3:end], codec):
            removed += len(unit)
        else:
            kept += unit
    if removed:
        stream.write_bytes(bytes(kept))
    return removed
