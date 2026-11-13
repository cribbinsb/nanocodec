"""Check 3 driver: pick comparable streams, extract their activity, compare.

For each clip, the reference encoder's middle-rate point is compared with
our point nearest to it in rate (same codec only: an HEVC stream is checked
for never-active tools, not against SVT's AV1). Extracted activity is cached
by stream hash, so a rerun only parses new streams.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..cachefile import write_atomic
from ..codecs import Codec
from ..config import Paths
from .activity import Activity, ToolLimits, compare_activity
from .curves import CurveStore, Point
from .flags import Flag, Severity

# A larger key frame is acceptable if it buys this much more Y-PSNR.
KEYFRAME_PSNR_ALLOWANCE_DB = 0.5

# In-loop filter activity that must not be (near) zero, per codec.
FILTERS_THAT_MUST_RUN = {
    Codec.AV1: ("cdef_active_share",),
    Codec.H264: ("deblocking_enabled_share",),
    Codec.HEVC: ("sao_luma_share", "deblocking_enabled_share"),
}


def stream_activity(stream: Path, codec: Codec, paths: Paths, cache: Path) -> dict[str, Activity]:
    digest = hashlib.sha256(stream.read_bytes()).hexdigest()
    cached = cache / f"{digest}.json"
    if cached.is_file():
        return {k: Activity.from_json(v) for k, v in json.loads(cached.read_text()).items()}
    if codec is Codec.AV1:
        from .av1_tools import av1_activity
        activity = av1_activity(stream, paths)
    else:
        from .avc_tools import avc_activity
        activity = avc_activity(stream, codec, paths)
    write_atomic(cached, json.dumps({k: v.to_json() for k, v in activity.items()}))
    return activity


def keyframe_flag(store: CurveStore, ours: Point, theirs: Point, ratio_limit: float,
                  subject: str) -> Flag | None:
    """Our first key frame much larger than the reference's at no better quality."""
    mine, ref = store.frames(ours), store.frames(theirs)
    if not mine or not ref or mine[0].frame_type != "I" or ref[0].frame_type != "I":
        return None
    ratio = mine[0].bytes / max(1, ref[0].bytes)
    psnr_gain = (mine[0].psnr_y or 0.0) - (ref[0].psnr_y or 0.0)
    if ratio <= ratio_limit or psnr_gain > KEYFRAME_PSNR_ALLOWANCE_DB:
        return None
    return Flag("tools", Severity.HIGH, subject,
                f"key frame {ratio:.1f}x the reference's size "
                f"({mine[0].bytes} vs {ref[0].bytes} bytes) for {psnr_gain:+.2f} dB Y-PSNR",
                {"our_key_share": round(mine[0].bytes / sum(r.bytes for r in mine), 3),
                 "reference_key_share": round(ref[0].bytes / sum(r.bytes for r in ref), 3)},
                "intra coding inefficient (partition/transform sizes capped, intra modes) "
                "or a key-frame quantizer offset far from the reference's")


def middle_point(points: list[Point]) -> Point | None:
    return sorted(points, key=lambda p: p.kbps)[len(points) // 2] if points else None


def check_tools(store: CurveStore, clips: list[str], references: dict[str, str],
                encoders: tuple[str, ...], limits: ToolLimits, paths: Paths,
                label: str) -> tuple[list[Flag], dict]:
    flags, table = [], {}
    cache = paths.work / "activity-cache"
    for clip in clips:
        for test, reference in references.items():
            if test not in encoders:
                continue
            codec = store.codec(test)
            same_codec = reference in encoders and store.codec(reference) is codec
            ref_curve = store.curve(reference, clip) if same_codec else None
            test_curve = store.curve(test, clip)
            anchor = middle_point(ref_curve.points) if ref_curve else middle_point(test_curve.points)
            if anchor is None:
                continue
            ours = test_curve.nearest(anchor.kbps)
            if ours is None:
                continue
            ours_stream = store.bitstream(test, clip, ours)
            ref_stream = store.bitstream(reference, clip, anchor) if ref_curve else None
            subject = f"{label}/{clip}: {test} q{ours.quantizer} ({ours.kbps:.0f} kbps)"
            if ours_stream is None or (ref_curve is not None and ref_stream is None):
                flags.append(Flag("tools", Severity.MEDIUM, subject,
                                  "kept bitstream missing, tool activity unchecked"))
                continue
            mine = stream_activity(ours_stream, codec, paths, cache)
            theirs = stream_activity(ref_stream, codec, paths, cache) if ref_stream else {}
            table[f"{test}/{clip}"] = {
                "ours": {k: v.to_json() for k, v in mine.items()}, "ours_kbps": ours.kbps,
                "reference": reference if theirs else None,
                "reference_kbps": anchor.kbps if theirs else None,
                "theirs": {k: v.to_json() for k, v in theirs.items()}}
            if ref_curve is not None:
                found = keyframe_flag(store, ours, anchor,
                                      limits.keyframe_size_ratio,
                                      subject + f" vs {reference} ({anchor.kbps:.0f} kbps)")
                if found:
                    flags.append(found)
            if "P" in mine:
                flags += compare_activity(
                    mine["P"], theirs.get("P"), limits,
                    subject + (f" vs {reference} ({anchor.kbps:.0f} kbps), P frames"
                               if theirs else ", P frames"),
                    FILTERS_THAT_MUST_RUN[codec])
    return flags, table
