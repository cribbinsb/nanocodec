#!/usr/bin/env python3
"""Build a conformance manifest for tools/ci/conformance.py: probe each
fetched stream (profile, chroma format, bit depths, cropped size), keep the
selected ones, and record the reference decoder's output MD5, whole and per
picture, streamed through a FIFO (no YUV is written).

The streams are untrusted data (tools/dev/conformance_fetch.py fetches
them); nothing in them is executed. Per suite, `SUITES` names the codec's
probe and the reference decoder's command line.

Usage:
    python3 -I tools/dev/conformance_manifest.py --suite hevc-v1|hevc-rext|vvc-v1|vvc-rext|av1-libaom|h264|h264-x264|h264-jm|h264-chroma|h264-x264-chroma|vp9-libvpx
        [--profiles "Main,Main 10,Main Still Picture"] [--max-bit-depth 10]
        [--chroma 4:2:0,4:0:0] [--only A,B] [--jobs 4]

Entries already in the manifest for streams not rebuilt are kept.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import tempdirs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
from conformance import SUITES as RUNNER_SUITES, Hashes  # noqa: E402

DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
BITSTREAM_SUFFIXES = (".bit", ".bin", ".ivf", ".264", ".jsv", ".avc", ".26l", ".h264", ".jvt")
#: Decoded YUVs some suites ship instead of MD5s (H.264.1): hashed for the
#: cross-check up to this size.
SHIPPED_YUV_SUFFIXES = (".yuv", ".qcif")
MAX_SHIPPED_YUV_BYTES = 1 << 30
#: Largest file read for the MD5s a suite ships.
MAX_SHIPPED_MD5_BYTES = 1 << 20
CHUNK = 1 << 20
#: Seconds one reference decode may take.
REFERENCE_TIMEOUT = 1800


class Bits:
    """MSB-first reader of an RBSP with Exp-Golomb codes."""

    def __init__(self, data):
        self.data, self.pos = data, 0

    def u(self, n):
        value = 0
        for _ in range(n):
            byte = self.data[self.pos >> 3]
            value = (value << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return value

    def ue(self):
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
        return (1 << zeros) - 1 + self.u(zeros)

    def se(self):
        code = self.ue()
        return (code + 1) // 2 if code & 1 else -(code // 2)


def annex_b_units(data):
    """NAL units of an Annex-B byte stream, emulation prevention removed."""
    for unit in re.split(b"\x00\x00\x01", data):
        unit = unit.rstrip(b"\x00")
        if unit:
            yield re.sub(b"\x00\x00\x03", b"\x00\x00", unit)


HEVC_PROFILES = {1: "Main", 2: "Main 10", 3: "Main Still Picture", 4: "Format Range Extensions",
                 9: "Screen Content Coding"}
HEVC_SPS = 33
#: SubWidthC, SubHeightC by chroma_format_idc (H.265 Table 6-1).
SUBSAMPLING = {0: (1, 1), 1: (2, 2), 2: (2, 1), 3: (1, 1)}
CHROMA_NAMES = {0: "4:0:0", 1: "4:2:0", 2: "4:2:2", 3: "4:4:4"}


#: Format range extensions and high throughput (H.265 §A.3.5, §A.3.6).
HEVC_REXT_PROFILE_IDCS = {4: "Main", 5: "High Throughput"}


def hevc_rext_profile(profile_idc, constraints):
    """The range-extensions profile name from the ten general constraint
    flags max_12bit .. max_14bit (H.265 Tables A.2, A.3): e.g. "Main 4:2:2
    10", "Monochrome 12", "Main 4:4:4 16 Intra", "Main 4:4:4 Still Picture"."""
    flag = [constraints >> (9 - i) & 1 for i in range(10)]
    max12, max10, max8, max422, max420, mono, intra, one_picture, _lower_rate, max14 = flag
    depth = "" if max8 else "10" if max10 else "12" if max12 else "14" if max14 else "16"
    if mono:
        return " ".join(filter(None, ("Monochrome", depth)))
    chroma = "" if max420 else "4:2:2" if max422 else "4:4:4"
    kind = "Still Picture" if one_picture else "Intra" if intra else ""
    name = " ".join(filter(None, (HEVC_REXT_PROFILE_IDCS[profile_idc], chroma, depth, kind)))
    return "Main" if name == "Main" else name


def probe_hevc(path):
    """Profile, chroma format, depths and cropped size from the first SPS
    (H.265 §7.3.2.2), or None."""
    for nal in annex_b_units(path.read_bytes()):
        if len(nal) < 3 or (nal[0] >> 1) & 63 != HEVC_SPS:
            continue
        r = Bits(nal[2:])
        r.u(4)
        sub_layers = r.u(3)
        r.u(1)
        r.u(2), r.u(1)
        profile_idc = r.u(5)
        compatibility = r.u(32)
        r.u(4)  # progressive, interlaced, non-packed, frame-only source flags
        constraints = r.u(10)  # general_max_12bit_constraint_flag .. max_14bit
        r.u(34), r.u(8)  # the remaining constraint flags, general_level_idc
        present = [(r.u(1), r.u(1)) for _ in range(sub_layers)]
        if sub_layers:
            r.u(2 * (8 - sub_layers))
        for profile, level in present:
            r.u(88 if profile else 0), r.u(8 if level else 0)
        r.ue()
        chroma = r.ue()
        if chroma == 3:
            r.u(1)
        width, height = r.ue(), r.ue()
        crop = [r.ue() for _ in range(4)] if r.u(1) else [0, 0, 0, 0]
        depths = [r.ue() + 8, r.ue() + 8]
        if profile_idc == 0:
            # Some streams signal the profile only in the compatibility flags.
            profile_idc = next((j for j in range(1, 32) if compatibility >> (31 - j) & 1), 0)
        profile = HEVC_PROFILES.get(profile_idc, f"profile_idc {profile_idc}")
        if profile_idc in HEVC_REXT_PROFILE_IDCS:
            profile = hevc_rext_profile(profile_idc, constraints)
        sub_w, sub_h = SUBSAMPLING[chroma]
        w = width - sub_w * (crop[0] + crop[1])
        h = height - sub_h * (crop[2] + crop[3])
        chroma_samples = 0 if chroma == 0 else 2 * (w // sub_w) * (h // sub_h)
        sample_bytes = 2 if max(depths) > 8 else 1
        return {"profile": profile, "chroma_format": CHROMA_NAMES[chroma], "bit_depth": depths,
                "width": w, "height": h, "frame_bytes": (w * h + chroma_samples) * sample_bytes}
    return None


H264_SPS = 7
#: profile_idc with the FRExt SPS fields (H.264 §7.3.2.1.1).
H264_HIGH_PROFILE_IDCS = {100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135}
#: profile_idc -> name; constraint_set flags refine some (H.264 Annex A.2).
H264_PROFILES = {66: "Baseline", 77: "Main", 88: "Extended", 100: "High", 110: "High 10",
                 122: "High 4:2:2", 244: "High 4:4:4 Predictive", 44: "CAVLC 4:4:4 Intra"}
#: `aspect_ratio_idc` of an explicit SAR and `matrix_coefficients` of GBR
#: (H.264 Tables E-1, E-5).
H264_EXTENDED_SAR = 255
H264_MATRIX_GBR = 0
#: JM's output plane holding each coded plane of a GBR picture: it writes
#: R (Cr), G (Y), B (Cb).
JM_RGB_PLANE_ORDER = [1, 2, 0]
#: The intra profiles signalled by constraint_set3_flag (Annex A.2.8..A.2.10).
H264_INTRA_PROFILES = {110: "High 10 Intra", 122: "High 4:2:2 Intra", 244: "High 4:4:4 Intra"}
#: Table 7-6: slice_type % 5.
H264_SLICE_P, H264_SLICE_B, H264_SLICE_I, H264_SLICE_SP, H264_SLICE_SI = range(5)
H264_PARTITION_NAL_TYPES = {2, 3, 4}
H264_EXTENSION_NAL_TYPES = {14, 15, 20}
H264_TRACE_FIELD = re.compile(r"^\[trace_headers @ [^\]]+\] \d+\s+(\S+)\s+\S+ = (-?\d+)$")


def skip_h264_scaling_list(r, size):
    last = following = 8
    for _ in range(size):
        if following:
            following = (last + r.se() + 256) % 256
        last = following or last


def h264_sps(nal):
    """Profile, chroma format, depths, cropped frame size and VUI
    `matrix_coefficients` (None when absent) of an SPS RBSP (H.264
    §7.3.2.1.1, cropping §7.4.2.1.1, VUI §E.1.1)."""
    r = Bits(nal[1:])
    profile_idc, constraints, _level = r.u(8), r.u(8), r.u(8)
    r.ue()
    chroma, depths = 1, [8, 8]
    if profile_idc in H264_HIGH_PROFILE_IDCS:
        chroma = r.ue()
        if chroma == 3:
            r.u(1)
        depths = [r.ue() + 8, r.ue() + 8]
        r.u(1)
        if r.u(1):
            for i in range(8 if chroma != 3 else 12):
                if r.u(1):
                    skip_h264_scaling_list(r, 16 if i < 6 else 64)
    r.ue()
    poc_type = r.ue()
    if poc_type == 0:
        r.ue()
    elif poc_type == 1:
        r.u(1), r.se(), r.se()
        for _ in range(r.ue()):
            r.se()
    r.ue(), r.u(1)
    width, height_units = r.ue() + 1, r.ue() + 1
    frame_mbs_only = r.u(1)
    if not frame_mbs_only:
        r.u(1)
    r.u(1)
    crop = [r.ue() for _ in range(4)] if r.u(1) else [0, 0, 0, 0]
    matrix = None
    if r.u(1):  # vui_parameters_present_flag
        if r.u(1) and r.u(8) == H264_EXTENDED_SAR:  # aspect_ratio_info_present_flag
            r.u(32)
        if r.u(1):  # overscan_info_present_flag
            r.u(1)
        if r.u(1):  # video_signal_type_present_flag
            r.u(4)
            if r.u(1):  # colour_description_present_flag
                r.u(16)
                matrix = r.u(8)
    sub_w, sub_h = SUBSAMPLING[chroma] if chroma else (1, 1)
    unit_x, unit_y = sub_w, sub_h * (2 - frame_mbs_only)
    w = width * 16 - unit_x * (crop[0] + crop[1])
    h = height_units * 16 * (2 - frame_mbs_only) - unit_y * (crop[2] + crop[3])
    name = H264_PROFILES.get(profile_idc, f"profile_idc {profile_idc}")
    if profile_idc == 66 and constraints & 0x40:
        name = "Constrained Baseline"
    elif profile_idc in H264_INTRA_PROFILES and constraints & 0x10:
        name = H264_INTRA_PROFILES[profile_idc]
    return name, chroma, depths, w, h, matrix


def h264_trace(path):
    """Every syntax element ffmpeg's trace_headers prints, as one dict per
    NAL unit grouped by packet; empty when ffmpeg cannot parse the stream."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0", "-c:v", "copy",
         "-bsf:v", "trace_headers", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace", check=False, timeout=600)
    packets, nal = [], None
    for line in result.stderr.splitlines():
        if "] Packet:" in line:
            packets.append([])
            nal = None
            continue
        match = H264_TRACE_FIELD.match(line)
        if not match:
            continue
        name, value = match.group(1).split("[")[0], int(match.group(2))
        if name == "forbidden_zero_bit":
            nal = {}
            if packets:
                packets[-1].append(nal)
        elif nal is not None:
            nal.setdefault(name, set()).add(value)
    return packets


def h264_features(path):
    """The coding tools beyond Constrained High a stream uses, from its
    headers (H.264 Annex A.2), and whether it is interlaced (field pictures
    or MBAFF frames)."""
    packets = h264_trace(path)
    if not packets:
        # ffmpeg parses neither slice groups nor separate colour planes.
        return ["headers ffmpeg cannot parse (FMO, separate colour planes)"], "progressive"
    seen = {}
    aso = False
    for packet in packets:
        first_mbs = []
        for nal in packet:
            for name, values in nal.items():
                seen.setdefault(name, set()).update(values)
            if nal.get("nal_unit_type", set()) & {1, 5} and "first_mb_in_slice" in nal:
                first_mbs.append(min(nal["first_mb_in_slice"]))
        aso |= any(b <= a for a, b in zip(first_mbs, first_mbs[1:]))

    def has(name, value=1):
        return value in seen.get(name, set())

    slice_types = {t % 5 for t in seen.get("slice_type", set())}
    nal_types = seen.get("nal_unit_type", set())
    features = []
    if H264_SLICE_B in slice_types:
        direct = {"temporal" if v == 0 else "spatial"
                  for v in seen.get("direct_spatial_mv_pred_flag", set())}
        features.append("B slices (" + "+".join(sorted(direct)) + " direct)")
        if has("weighted_bipred_idc", 1):
            features.append("explicit bi-pred weights")
        if has("weighted_bipred_idc", 2):
            features.append("implicit bi-pred weights")
    if slice_types & {H264_SLICE_SP, H264_SLICE_SI}:
        features.append("SP/SI slices")
    if has("field_pic_flag"):
        features.append("field pictures")
    frame_pictures = has("field_pic_flag", 0) or "field_pic_flag" not in seen
    mbaff = has("mb_adaptive_frame_field_flag") and frame_pictures
    if mbaff:
        features.append("MBAFF")
    if seen.get("num_slice_groups_minus1", {0}) - {0}:
        features.append("FMO")
    if aso:
        features.append("ASO")
    if has("redundant_pic_cnt_present_flag"):
        features.append("redundant pictures")
    if nal_types & H264_PARTITION_NAL_TYPES:
        features.append("data partitioning")
    if nal_types & H264_EXTENSION_NAL_TYPES:
        features.append("SVC/MVC NAL units")
    if has("qpprime_y_zero_transform_bypass_flag"):
        features.append("lossless")
    if seen.get("bit_depth_luma_minus8", {0}) - {0} or seen.get("bit_depth_chroma_minus8", {0}) - {0}:
        features.append("bit depth above 8")
    interlaced = has("field_pic_flag") or mbaff
    return features, "interlaced" if interlaced else "progressive"


def probe_h264(path):
    """Profile, chroma format, depths, cropped size (first SPS), and the
    tools and scan type the stream uses, or None."""
    for nal in annex_b_units(path.read_bytes()):
        if not nal or nal[0] & 31 != H264_SPS:
            continue
        name, chroma, depths, w, h, matrix = h264_sps(nal)
        features, scan = h264_features(path)
        # Monochrome pictures are output as 4:2:0 with mid-grey chroma (JM, ffmpeg).
        chroma_samples = 2 * ((w + 1) // 2) * ((h + 1) // 2) if chroma in (0, 1) else \
            2 * (w // SUBSAMPLING[chroma][0]) * (h // SUBSAMPLING[chroma][1])
        sample_bytes = 2 if max(depths) > 8 else 1
        info = {"profile": name, "chroma_format": CHROMA_NAMES[chroma], "bit_depth": depths,
                "width": w, "height": h, "frame_bytes": (w * h + chroma_samples) * sample_bytes,
                "scan": scan, "features": features}
        if chroma == 3 and matrix == H264_MATRIX_GBR:
            # JM writes GBR (matrix_coefficients 0) 4:4:4 pictures in R, G, B
            # plane order (ldecod output.c, `rgb_output`); the hashes are of
            # the decoded planes in their coded order, Y (G), Cb (B), Cr (R).
            info["reference_plane_order"] = JM_RGB_PLANE_ORDER
        return info
    return None


VVC_PROFILES = {"Main 10": "Main 10", "65": "Main 10 Still Picture",
                "Main 10 Still Picture": "Main 10 Still Picture",
                "Main 10 4:4:4": "Main 10 4:4:4", "33": "Main 10 4:4:4",
                "97": "Main 10 4:4:4 Still Picture",
                "Main 10 4:4:4 Still Picture": "Main 10 4:4:4 Still Picture",
                # The range extension profiles, by general_profile_idc
                # (H.266 Table A.1), which ffprobe prints as numbers.
                "2": "Main 12", "10": "Main 12 Intra", "66": "Main 12 Still Picture",
                "34": "Main 12 4:4:4", "42": "Main 12 4:4:4 Intra",
                "98": "Main 12 4:4:4 Still Picture", "35": "Main 16 4:4:4",
                "43": "Main 16 4:4:4 Intra", "99": "Main 16 4:4:4 Still Picture"}
#: ffprobe pixel formats: chroma format and bit depth.
PIXEL_FORMATS = {"gray": ("4:0:0", 8), "yuv420p": ("4:2:0", 8), "gray10le": ("4:0:0", 10),
                 "yuv420p10le": ("4:2:0", 10), "yuv422p": ("4:2:2", 8),
                 "yuv422p10le": ("4:2:2", 10), "yuv444p10le": ("4:4:4", 10),
                 "yuv444p": ("4:4:4", 8),
                 **{f"{name}{depth}le": (chroma, depth) for depth in (12, 14, 16)
                    for name, chroma in (("gray", "4:0:0"), ("yuv420p", "4:2:0"),
                                         ("yuv422p", "4:2:2"), ("yuv444p", "4:4:4"))}}


#: chroma_format_idc names (H.266 Table 2).
VVC_CHROMA = {0: "4:0:0", 1: "4:2:0", 2: "4:2:2", 3: "4:4:4"}
#: nal_unit_type of an SPS (H.266 Table 5).
VVC_SPS_NUT = 15
#: Fixed bits of general_constraints_info before gci_num_additional_bits.
VVC_GCI_FIXED_BITS = 71


def vvc_sps_format(path):
    """chroma_format_idc's name, the bit depth and the cropped size of the
    first SPS (H.266 §7.3.2.4), for what ffprobe does not describe (16
    bits, or streams it gives up on); None for an SPS with subpictures,
    which this reader does not follow."""
    for unit in annex_b_units(path.read_bytes()):
        if len(unit) < 3 or unit[1] >> 3 != VVC_SPS_NUT:
            continue
        r = Bits(unit[2:])
        r.u(8)  # sps_seq_parameter_set_id, sps_video_parameter_set_id
        sublayers = r.u(3)
        chroma = r.u(2)
        r.u(2)  # sps_log2_ctu_size_minus5
        if r.u(1):  # sps_ptl_dpb_hrd_params_present_flag: profile_tier_level()
            r.u(7 + 1 + 8 + 1 + 1)  # profile, tier, level, frame only, multilayer
            if r.u(1):  # gci_present_flag
                r.u(VVC_GCI_FIXED_BITS)
                r.u(r.u(8))
            r.pos = (r.pos + 7) // 8 * 8
            present = [r.u(1) for _ in range(sublayers)]
            r.pos = (r.pos + 7) // 8 * 8
            r.u(8 * sum(present))
            r.u(32 * r.u(8))  # ptl_num_sub_profiles, general_sub_profile_idc
        r.u(1)  # sps_gdr_enabled_flag
        if r.u(1):  # sps_ref_pic_resampling_enabled_flag
            r.u(1)
        width, height = r.ue(), r.ue()
        crop = [r.ue() for _ in range(4)] if r.u(1) else [0] * 4
        if r.u(1):  # sps_subpic_info_present_flag
            return None
        sub_w, sub_h = {0: (1, 1), 1: (2, 2), 2: (2, 1), 3: (1, 1)}[chroma]
        size = (width - sub_w * (crop[0] + crop[1]), height - sub_h * (crop[2] + crop[3]))
        return VVC_CHROMA[chroma], r.ue() + 8, size
    return None


def vvc_profile_idc(path):
    """general_profile_idc of the first SPS, as ffprobe prints it."""
    for unit in annex_b_units(path.read_bytes()):
        if len(unit) > 4 and unit[1] >> 3 == VVC_SPS_NUT:
            r = Bits(unit[2:])
            r.u(8 + 3 + 2 + 2 + 1)
            return str(r.u(7))
    return None


def probe_vvc(path):
    """Profile, chroma format, depth and cropped size from ffprobe (its VVC
    parser reads the SPS, subpictures included), or None."""
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "stream=profile,pix_fmt,width,height", "-of", "json", str(path)],
                                capture_output=True, text=True, timeout=60, check=False)
        streams = json.loads(result.stdout or "{}").get("streams", [])
    except subprocess.TimeoutExpired:
        streams = []
    info = streams[0] if len(streams) == 1 else {}
    if info.get("pix_fmt") in PIXEL_FORMATS:
        chroma, depth = PIXEL_FORMATS[info["pix_fmt"]]
    elif (format_ := vvc_sps_format(path)) is not None:
        chroma, depth, (info["width"], info["height"]) = format_
        info.setdefault("profile", vvc_profile_idc(path))
    else:
        return None
    w, h = info["width"], info["height"]
    sub_w, sub_h = {"4:0:0": (1, 1), "4:2:0": (2, 2), "4:2:2": (2, 1), "4:4:4": (1, 1)}[chroma]
    chroma_samples = 0 if chroma == "4:0:0" else 2 * (w // sub_w) * (h // sub_h)
    profile = str(info.get("profile"))
    return {"profile": VVC_PROFILES.get(profile, f"profile {profile}"),
            "chroma_format": chroma, "bit_depth": [depth, depth], "width": w, "height": h,
            "frame_bytes": (w * h + chroma_samples) * (2 if depth > 8 else 1)}


AV1_PROFILES = {0: "Main", 1: "High", 2: "Professional"}
AV1_OBU_SEQUENCE_HEADER = 1
IVF_HEADER_BYTES = 32
IVF_FRAME_HEADER_BYTES = 12


def leb128(data, pos):
    value = 0
    for i in range(8):
        byte = data[pos + i]
        value |= (byte & 0x7F) << (7 * i)
        if not byte & 0x80:
            return value, pos + i + 1
    raise ValueError("leb128 longer than 8 bytes")


def av1_sequence_header(path):
    """The payload of the first sequence header OBU of an IVF file."""
    data = path.read_bytes()
    pos = IVF_HEADER_BYTES
    while pos + IVF_FRAME_HEADER_BYTES <= len(data):
        size = int.from_bytes(data[pos:pos + 4], "little")
        frame, pos = data[pos + IVF_FRAME_HEADER_BYTES:pos + IVF_FRAME_HEADER_BYTES + size], \
            pos + IVF_FRAME_HEADER_BYTES + size
        at = 0
        while at < len(frame):
            header = frame[at]
            obu_type, extension, has_size = (header >> 3) & 15, (header >> 2) & 1, (header >> 1) & 1
            at += 1 + extension
            length, at = leb128(frame, at) if has_size else (len(frame) - at, at)
            if obu_type == AV1_OBU_SEQUENCE_HEADER:
                return frame[at:at + length]
            at += length
    return None


def probe_av1(path):
    """Profile, chroma format, depths and maximum frame size from the first
    sequence header (AV1 §5.5), or None. The per-picture size assumes every
    picture has the maximum size (streams that resize get whole-output
    hashes only)."""
    payload = av1_sequence_header(path)
    if payload is None:
        return None
    r = Bits(payload)
    profile, _still, reduced = r.u(3), r.u(1), r.u(1)
    if reduced:
        r.u(5)
        decoder_model = initial_delay = False
    else:
        decoder_model, delay_bits = False, 0
        if r.u(1):  # timing_info_present_flag
            r.u(32), r.u(32)
            if r.u(1):  # equal_picture_interval: uvlc
                zeros = 0
                while not r.u(1):
                    zeros += 1
                r.u(zeros)
            decoder_model = bool(r.u(1))
            if decoder_model:
                delay_bits = r.u(5) + 1
                r.u(32), r.u(5), r.u(5)
        initial_delay = bool(r.u(1))
        for _ in range(r.u(5) + 1):
            r.u(12)
            if r.u(5) > 7:
                r.u(1)
            if decoder_model and r.u(1):
                r.u(delay_bits), r.u(delay_bits), r.u(1)
            if initial_delay and r.u(1):
                r.u(4)
    width_bits, height_bits = r.u(4) + 1, r.u(4) + 1
    width, height = r.u(width_bits) + 1, r.u(height_bits) + 1
    if not reduced and r.u(1):  # frame_id_numbers_present_flag
        r.u(4), r.u(3)
    r.u(3)  # use_128x128_superblock, enable_filter_intra, enable_intra_edge_filter
    if not reduced:
        r.u(4)  # interintra, masked compound, warped motion, dual filter
        order_hint = r.u(1)
        if order_hint:
            r.u(2)
        force_screen = 2 if r.u(1) else r.u(1)
        if force_screen and not r.u(1):
            r.u(1)
        if order_hint:
            r.u(3)
    r.u(3)  # superres, cdef, restoration
    high = r.u(1)
    depth = (12 if r.u(1) else 10) if profile == 2 and high else (10 if high else 8)
    mono = 0 if profile == 1 else r.u(1)
    subsampling = av1_subsampling(r, profile, depth, mono)
    sample_bytes = 2 if depth > 8 else 1
    # Monochrome pictures are output with grey 4:2:0 chroma planes too.
    chroma_w = (width + subsampling[0]) >> subsampling[0]
    chroma_h = (height + subsampling[1]) >> subsampling[1]
    return {"profile": AV1_PROFILES.get(profile, f"seq_profile {profile}"),
            "chroma_format": "4:0:0" if mono else AV1_CHROMA_NAMES[subsampling],
            "bit_depth": [depth, depth], "width": width, "height": height,
            "frame_bytes": (width * height + 2 * chroma_w * chroma_h) * sample_bytes}


AV1_CHROMA_NAMES = {(1, 1): "4:2:0", (1, 0): "4:2:2", (0, 0): "4:4:4"}
#: color_description: BT.709 primaries, sRGB transfer, identity matrix (sRGB).
AV1_SRGB = (1, 13, 0)


def av1_subsampling(r, profile, depth, mono):
    """(subsampling_x, subsampling_y) from the rest of color_config()
    (AV1 §5.5.2); monochrome is output as 4:2:0."""
    description = (r.u(8), r.u(8), r.u(8)) if r.u(1) else None
    if mono:
        return (1, 1)
    if description == AV1_SRGB:
        return (0, 0)
    r.u(1)  # color_range
    if profile == 0:
        return (1, 1)
    if profile == 1:
        return (0, 0)
    if depth == 12:
        x = r.u(1)
        return (x, r.u(1) if x else 0)
    return (1, 0)


VP9_KEY_FRAME = 0
VP9_SYNC_CODE = 0x498342
VP9_CS_RGB = 7


def ivf_first_frame(path):
    """The first frame of an IVF file (a VP9 superframe starts with its
    first frame), or None."""
    data = path.read_bytes()
    if len(data) < IVF_HEADER_BYTES + IVF_FRAME_HEADER_BYTES:
        return None
    size = int.from_bytes(data[IVF_HEADER_BYTES:IVF_HEADER_BYTES + 4], "little")
    start = IVF_HEADER_BYTES + IVF_FRAME_HEADER_BYTES
    return data[start:start + size]


def probe_vp9(path):
    """Profile, chroma format, depth and frame size from the first frame's
    uncompressed header (VP9 bitstream specification §6.2), or None. The
    per-picture size is the first frame's (streams that resize get
    whole-output hashes only)."""
    frame = ivf_first_frame(path)
    if not frame:
        return None
    r = Bits(frame)
    if r.u(2) != 2:  # frame_marker
        return None
    low = r.u(1)
    profile = (r.u(1) << 1) | low
    if profile == 3:
        r.u(1)
    if r.u(1):  # show_existing_frame
        return None
    frame_type, show_frame, error_resilient = r.u(1), r.u(1), r.u(1)
    intra_only = frame_type != VP9_KEY_FRAME and not show_frame and r.u(1)
    if frame_type != VP9_KEY_FRAME:
        if not intra_only:
            return None
        if not error_resilient:
            r.u(2)  # reset_frame_context
    if r.u(24) != VP9_SYNC_CODE:
        return None
    depth, subsampling = 8, (1, 1)
    if frame_type == VP9_KEY_FRAME or profile > 0:
        if profile >= 2:
            depth = 12 if r.u(1) else 10
        rgb = r.u(3) == VP9_CS_RGB
        if not rgb:
            r.u(1)  # color_range
        if profile in (1, 3):
            subsampling = (0, 0) if rgb else (r.u(1), r.u(1))
            r.u(1)
    if intra_only:
        r.u(8)  # refresh_frame_flags
    width, height = r.u(16) + 1, r.u(16) + 1
    chroma_names = {(1, 1): "4:2:0", (1, 0): "4:2:2", (0, 1): "4:4:0", (0, 0): "4:4:4"}
    chroma_w = (width + subsampling[0]) >> subsampling[0]
    chroma_h = (height + subsampling[1]) >> subsampling[1]
    sample_bytes = 2 if depth > 8 else 1
    return {"profile": f"Profile {profile}", "chroma_format": chroma_names[subsampling],
            "bit_depth": [depth, depth], "width": width, "height": height,
            "frame_bytes": (width * height + 2 * chroma_w * chroma_h) * sample_bytes}


# suite -> (probe, reference decoder command for (bitstream, output path)).
SUITES = {
    "vvc-v1": (probe_vvc, lambda bitstream, out: [
        str(DATA / "work/standards-eval/src/VTM/bin/DecoderAppStatic"), "-b", str(bitstream),
        "-o", str(out), "-d", "0"]),
    "hevc-v1": (probe_hevc, lambda bitstream, out: [
        str(DATA / "anchors/x86_64/bin/TAppDecoder"), "-b", str(bitstream), "-o", str(out), "-d", "0"]),
    # HM built with high bit depth (tools/anchors/build_anchors.sh hmhbd):
    # the default build refuses extended precision processing and 16-bit
    # samples. It writes each picture in its own chroma format (4:0:0 as
    # one plane) and depth.
    "hevc-rext": (probe_hevc, lambda bitstream, out: [
        str(DATA / "anchors/x86_64/bin/TAppDecoder-hbd"), "-b", str(bitstream), "-o", str(out),
        "-d", "0"]),
    # aomdec --i420 writes each picture at its own depth (16-bit
    # little-endian beyond 8 bits), monochrome with mid-grey chroma planes
    # (what libaom's per-frame MD5s hash; --rawvideo would drop them), film
    # grain applied.
    "av1-libaom": (probe_av1, lambda bitstream, out: [
        str(DATA / "anchors/x86_64/bin/aomdec"), "--i420", "-o", str(out), str(bitstream)]),
    # The High and Professional profile streams of
    # tools/dev/av1_profile_streams.sh: 4:2:2 and 4:4:4 in their own format
    # (--rawvideo; --i420 would convert them), 4:2:0 and monochrome as
    # above.
    "av1-profiles": (probe_av1, lambda bitstream, out: [
        str(DATA / "anchors/x86_64/bin/aomdec"),
        "--i420" if probe_av1(bitstream)["chroma_format"] in ("4:2:0", "4:0:0") else "--rawvideo",
        "-o", str(out), str(bitstream)]),
    # vpxdec built with high bit depth (tools/anchors/build_anchors.sh
    # vpxhbd): shown frames at their own size, 8-bit samples as bytes and
    # deeper ones as 16-bit little-endian (the form of libvpx's .md5 files);
    # 4:2:2, 4:4:0 and 4:4:4 in their own format (--rawvideo: --i420 would
    # convert them).
    "vp9-libvpx": (probe_vp9, lambda bitstream, out: [
        str(DATA / "anchors/x86_64/bin/vpxdec-hbd"),
        "--i420" if probe_vp9(bitstream)["chroma_format"] == "4:2:0" else "--rawvideo",
        "-o", str(out), str(bitstream)]),
    # JM 19 ldecod: frames in output order, cropped, 4:0:0 written as 4:2:0
    # with mid-grey chroma, deeper samples as 16-bit little-endian.
    "h264": (probe_h264, lambda bitstream, out: [
        str(DATA / "anchors/x86_64/bin/ldecod"), "-p", f"InputFile={bitstream}",
        "-p", f"OutputFile={out}"]),
}

SUITES["h264-x264"] = SUITES["h264"]
# The JM streams' verified decode, shipped beside each stream
# (tools/dev/h264_conformance_jm.sh: lencod's reconstruction where JM's or
# FFmpeg's decode equals it).
SUITES["h264-jm"] = (probe_h264, lambda bitstream, out: [
    "cp", str(bitstream.with_suffix(".yuv")), str(out)])
SUITES["av1-supplementary"] = SUITES["av1-libaom"]
SUITES["h264-chroma"] = SUITES["h264"]
SUITES["h264-x264-chroma"] = SUITES["h264"]

#: ffmpeg's raw output format of JM's: 4:0:0 as 4:2:0, the stream's format
#: otherwise, 16-bit little-endian beyond 8 bits (10 for 9-bit streams, as
#: before the 4:2:2 / 4:4:4 suite).
H264_PIXEL_FORMATS = {"4:0:0": "yuv420p", "4:2:0": "yuv420p", "4:2:2": "yuv422p",
                      "4:4:4": "yuv444p"}


def h264_pixel_format(info):
    depth = max(info["bit_depth"])
    suffix = "" if depth == 8 else f"{max(depth, 10)}le"
    return H264_PIXEL_FORMATS[info["chroma_format"]] + suffix

# VTM built with ENABLE_HIGH_BITDEPTH (32-bit samples and coefficients),
# which extended precision and 16-bit streams need.
SUITES["vvc-rext"] = (probe_vvc, lambda bitstream, out: [
    str(DATA / "work/standards-eval/src/VTM-hbd/bin/DecoderAppStatic"), "-b", str(bitstream),
    "-o", str(out), "-d", "0"])

# suite -> a second decoder's command for (bitstream, output path, probe
# info): its whole-output MD5 is recorded as a cross-check of the reference.
CROSS_CHECKS = {
    "h264": lambda bitstream, out, info: [
        "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(bitstream), "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", h264_pixel_format(info), str(out)],
}
CROSS_CHECKS["h264-x264"] = CROSS_CHECKS["h264"]
CROSS_CHECKS["h264-jm"] = CROSS_CHECKS["h264"]
CROSS_CHECKS["h264-chroma"] = CROSS_CHECKS["h264"]
CROSS_CHECKS["h264-x264-chroma"] = CROSS_CHECKS["h264"]
# FFmpeg's native VP9 decoder; it scales resized pictures to the first
# size, so those streams record a disagreement.
VP9_PIXEL_FORMATS = {"4:2:0": "yuv420p", "4:2:2": "yuv422p", "4:4:0": "yuv440p",
                     "4:4:4": "yuv444p"}
VP9_DEPTH_SUFFIX = {8: "", 10: "10le", 12: "12le"}
# dav1d writes the native format (monochrome as luma only, so those
# streams record a disagreement), deeper samples as 16-bit little-endian.
CROSS_CHECKS["av1-profiles"] = lambda bitstream, out, info: [
    str(DATA / "anchors/x86_64/bin/dav1d"), "-q", "-i", str(bitstream), "-o", str(out),
    "--muxer", "yuv"]
CROSS_CHECKS["vp9-libvpx"] = lambda bitstream, out, info: [
    "ffmpeg", "-v", "error", "-nostdin", "-y", "-c:v", "vp9", "-i", str(bitstream),
    "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt",
    VP9_PIXEL_FORMATS[info["chroma_format"]] + VP9_DEPTH_SUFFIX[max(info["bit_depth"])], str(out)]


def bitstream(directory):
    found = [p for p in directory.rglob("*")
             if p.suffix.lower() in BITSTREAM_SUFFIXES and "__MACOSX" not in p.parts]
    if len(found) > 1:
        # Extra streams beside the conformance one (alphaconformanceG): the
        # one named after the directory.
        found = [p for p in found if p.stem == directory.name]
    return found[0] if len(found) == 1 else None


def shipped_md5s(directory):
    """The MD5s a stream's directory ships: listed in its MD5 files, or of
    the decoded YUVs it ships (H.264.1)."""
    found = set()
    for p in directory.rglob("*"):
        if not p.is_file():
            continue
        if "md5" in p.name.lower() and p.stat().st_size < MAX_SHIPPED_MD5_BYTES:
            found.update(m.lower() for m in re.findall(r"\b[0-9a-fA-F]{32}\b", p.read_text("latin-1")))
        elif p.suffix.lower() in SHIPPED_YUV_SUFFIXES and p.stat().st_size < MAX_SHIPPED_YUV_BYTES:
            digest = hashlib.md5()
            with open(p, "rb") as f:
                while chunk := f.read(CHUNK):
                    digest.update(chunk)
            found.add(digest.hexdigest())
    return found


class PlaneOrderHashes:
    """[`Hashes`] of a reference output whose 4:4:4 planes come in another
    order: each picture is buffered, its three equal planes put back in
    coded order (`order[k]`: the output plane holding coded plane k)."""

    def __init__(self, frame_bytes, order):
        self.inner, self.order = Hashes(frame_bytes), order
        self.frame_bytes, self.pending = frame_bytes, b""

    def update(self, chunk):
        self.pending += chunk
        plane = self.frame_bytes // 3
        while len(self.pending) >= self.frame_bytes:
            frame, self.pending = self.pending[:self.frame_bytes], self.pending[self.frame_bytes:]
            self.inner.update(b"".join(frame[k * plane:(k + 1) * plane] for k in self.order))

    def __getattr__(self, name):
        # Trailing bytes of a partial picture are hashed as they came.
        if self.pending:
            self.inner.update(self.pending)
            self.pending = b""
        return getattr(self.inner, name)


def reference_hashes(command, frame_bytes, plane_order=None):
    """Hash the reference decoder's output, written into a FIFO (its 4:4:4
    planes put back in coded order when `plane_order` is given); also its
    exit status (VTM exits 1 on a picture-hash SEI mismatch, as after a GDR
    start, though its output is the conformance output)."""
    with tempdirs.temp_dir("conformance") as scratch:
        fifo = Path(scratch) / "out.yuv"
        os.mkfifo(fifo)
        hashes = PlaneOrderHashes(frame_bytes, plane_order) if plane_order else Hashes(frame_bytes)

        def drain():
            with open(fifo, "rb") as pipe:
                while chunk := pipe.read(CHUNK):
                    hashes.update(chunk)

        reader = threading.Thread(target=drain)
        reader.start()
        try:
            # In the scratch directory: JM writes its logs into the current one.
            status = subprocess.run(command(fifo), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, check=False, cwd=scratch,
                                    timeout=REFERENCE_TIMEOUT).returncode
        finally:
            # A decoder that never opened its output leaves the reader
            # waiting for a writer: open (and close) one without blocking.
            # When the reader has already finished there is no reader, the
            # open fails, and nothing is waiting.
            try:
                os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            except OSError:
                pass
            reader.join()
    return hashes, status


def build_entry(suite, streams, directory):
    probe, reference = SUITES[suite]
    path = bitstream(directory)
    info = probe(path) if path else None
    if info is None:
        return directory.name, None
    try:
        hashes, status = reference_hashes(lambda out: reference(path, out), info["frame_bytes"],
                                          info.get("reference_plane_order"))
    except subprocess.TimeoutExpired:
        print(f"skip {directory.name}: reference decoder timed out", file=sys.stderr)
        return directory.name, None
    cross_check = None
    if suite in CROSS_CHECKS:
        try:
            other, _ = reference_hashes(lambda out: CROSS_CHECKS[suite](path, out, info), None)
            cross_check = other.whole.hexdigest() == hashes.whole.hexdigest()
        except subprocess.TimeoutExpired:
            pass
    shipped = shipped_md5s(directory)
    md5 = hashes.whole.hexdigest()
    if status != 0 and not hashes.frames:
        print(f"skip {directory.name}: reference decoder failed ({status})", file=sys.stderr)
        return directory.name, None
    frames = hashes.frames
    agrees = (md5 in shipped or bool(frames) and set(frames) <= shipped) if shipped else None
    split_wrong = len(frames) != len(shipped) or hashes.size % info["frame_bytes"]
    if agrees is False and len(shipped) > 1 and split_wrong:
        # Per-picture MD5s shipped for pictures whose size changes within
        # the stream (libaom's and libvpx's resize vectors): the probe's one
        # picture size does not split the output, so only the whole MD5 is
        # kept.
        info = {**info, "frame_bytes": None}
        frames, agrees = [], None
    elif hashes.size % info["frame_bytes"]:
        # The same, with only a whole MD5 shipped: the output is no whole
        # number of first-picture sizes (HEVC VPSSPSPPS_A_MainConcept_1:
        # six pictures of six sizes were split into 72 pieces of the first).
        info = {**info, "frame_bytes": None}
        frames = []
    return directory.name, {
        "name": directory.name, "bitstream": str(path.relative_to(streams)), **info,
        "md5": md5, "frames": len(hashes.frames) if frames else None, "frame_md5": frames,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shipped_md5_agrees": agrees,
        **({"cross_check_agrees": cross_check} if cross_check is not None else {}),
        **({"reference_status": status} if status else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--profiles", default="Main,Main 10,Main Still Picture")
    parser.add_argument("--max-bit-depth", type=int, default=10)
    parser.add_argument("--chroma", default="4:2:0,4:0:0")
    parser.add_argument("--only", default="")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    _, manifest, streams = RUNNER_SUITES[args.suite]
    probe, _ = SUITES[args.suite]
    profiles = set(args.profiles.split(","))
    chroma = set(args.chroma.split(","))
    names = set(args.only.split(",")) if args.only else None
    chosen = []
    for directory in sorted(p for p in streams.iterdir() if p.is_dir()):
        if names and directory.name not in names:
            continue
        path = bitstream(directory)
        info = probe(path) if path else None
        if (info and info["profile"] in profiles and info["chroma_format"] in chroma
                and max(info["bit_depth"]) <= args.max_bit_depth):
            chosen.append(directory)
        else:
            print(f"skip {directory.name}: {info and (info['profile'], info['chroma_format'], info['bit_depth'])}",
                  file=sys.stderr)
    existing = {}
    if manifest.exists():
        data = json.loads(manifest.read_text())
        existing = {e["name"]: e for e in (data["streams"] if isinstance(data, dict) else data)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name, entry in pool.map(lambda d: build_entry(args.suite, streams, d), chosen):
            if entry:
                existing[name] = entry
                print(f"{name}: {entry['profile']} {entry['bit_depth']} {entry['md5']}"
                      f" shipped={entry['shipped_md5_agrees']}", file=sys.stderr)
    out = {"suite": args.suite, "reference": "the suite's reference decoder (SUITES)",
           "streams": sorted(existing.values(), key=lambda e: e["name"].lower())}
    manifest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out['streams'])} streams -> {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
