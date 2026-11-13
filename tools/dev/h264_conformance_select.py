#!/usr/bin/env python3
"""Select the ITU-T H.264.1 conformance streams a Constrained High decoder must decode.

Constrained High (H.264 Annex A.2.4.2) is High without B slices and without
interlace. A decoder of that profile decodes 8-bit 4:2:0 / 4:0:0 progressive
streams made of I and P slices with CAVLC or CABAC, 8x8 transforms, scaling
matrices, explicit weighted prediction, multiple slices per picture (in
raster order: no ASO, no FMO, no redundant pictures, no data partitioning),
long-term references and memory management operations.

The script classifies every stream under the conformance directory from the
syntax it actually uses (ffmpeg's `trace_headers` bitstream filter prints
every parameter set and slice header), not from its profile signalling, and
writes the manifest consumed by `tools/ci/h264_conformance.py`. Each suite is
a directory of one-stream directories; `itu` is the official suite and
`supplementary` the x264 streams of `h264_conformance_supplement.sh` (the
official suite has no I/P-only High-profile stream):

    python3 -I tools/dev/h264_conformance_select.py tools/ci/h264_conformance.json \
        itu=~/nanocodec-data/conformance/h264/streams \
        supplementary=~/nanocodec-data/conformance/h264/supplementary

The streams are untrusted downloads: they are only read, never executed.
"""

import json
import os
import re
import subprocess
import sys
from collections import defaultdict

BITSTREAM_SUFFIXES = (".264", ".jsv", ".avc", ".26l", ".h264", ".jvt")
YUV_SUFFIXES = (".yuv", ".qcif")

# Table 7-6: slice_type % 5.
SLICE_P, SLICE_B, SLICE_I, SLICE_SP, SLICE_SI = range(5)
# Table 7-1: data partitions A/B/C.
PARTITION_NAL_TYPES = {2, 3, 4}
# Table 7-1: SVC / MVC extension NAL units.
EXTENSION_NAL_TYPES = {14, 15, 20}

FIELD = re.compile(r"^\[trace_headers @ [^\]]+\] \d+\s+(\S+)\s+\S+ = (-?\d+)$")


def bitstream_of(directory):
    """The coded stream of a conformance directory (one per directory)."""
    candidates = [
        name
        for name in sorted(os.listdir(directory))
        if name.lower().endswith(BITSTREAM_SUFFIXES)
    ]
    return candidates[0] if len(candidates) == 1 else None


def reference_yuv_of(directory):
    """The decoded reference of a directory. Some streams also ship a
    display version: CVSE*'s plain YUV repeats frames as their SEI
    pic_struct asks (`-norpt` is the decoder output), CVFC1's pads the
    cropped area with grey (`_nogray` is the cropped decoder output)."""
    candidates = [
        name for name in sorted(os.listdir(directory)) if name.lower().endswith(YUV_SUFFIXES)
    ]
    decoder_output = [name for name in candidates if "norpt" in name or "nogray" in name]
    if decoder_output:
        candidates = decoder_output
    return candidates[0] if len(candidates) == 1 else None


def trace(path):
    """Every syntax element ffmpeg traces, grouped by NAL unit and packet."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", path, "-map", "0:v:0",
         "-c:v", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace", check=False)
    packets = []
    nal = None
    for line in result.stderr.splitlines():
        if "] Packet:" in line:
            packets.append([])
            nal = None
            continue
        match = FIELD.match(line)
        if not match:
            continue
        # Array elements are traced as `name[i]`.
        name, value = match.group(1).split("[")[0], int(match.group(2))
        if name == "forbidden_zero_bit":
            nal = {}
            if packets:
                packets[-1].append(nal)
            continue
        if nal is not None:
            nal.setdefault(name, value)
            nal.setdefault("_all", defaultdict(set))[name].add(value)
    return packets


def classify(packets):
    """Syntax facts and the reasons (if any) the stream is outside Constrained High."""
    facts = defaultdict(set)
    multi_slice = aso = False
    for packet in packets:
        first_mbs = []
        for nal in packet:
            nal_type = nal.get("nal_unit_type")
            facts["nal_types"].add(nal_type)
            for name, values in nal.get("_all", {}).items():
                if name in TRACKED:
                    facts[name] |= values
            if nal_type in (1, 5) and "first_mb_in_slice" in nal:
                first_mbs.append(nal["first_mb_in_slice"])
        multi_slice |= len(first_mbs) > 1
        aso |= any(b <= a for a, b in zip(first_mbs, first_mbs[1:]))
    excluded = []
    slice_types = {t % 5 for t in facts["slice_type"]}
    if SLICE_B in slice_types:
        excluded.append("B slices")
    if slice_types & {SLICE_SP, SLICE_SI}:
        excluded.append("SP/SI slices")
    if 0 in facts["frame_mbs_only_flag"] or 1 in facts["field_pic_flag"]:
        excluded.append("interlace (field/MBAFF)")
    if facts["chroma_format_idc"] - {0, 1}:
        excluded.append("chroma format above 4:2:0")
    if facts["bit_depth_luma_minus8"] - {0} or facts["bit_depth_chroma_minus8"] - {0}:
        excluded.append("bit depth above 8")
    if 1 in facts["qpprime_y_zero_transform_bypass_flag"]:
        excluded.append("lossless transform bypass")
    if facts["num_slice_groups_minus1"] - {0}:
        excluded.append("FMO (slice groups)")
    if 1 in facts["redundant_pic_cnt_present_flag"]:
        excluded.append("redundant pictures")
    if facts["nal_types"] & PARTITION_NAL_TYPES:
        excluded.append("data partitioning")
    if facts["nal_types"] & EXTENSION_NAL_TYPES:
        excluded.append("SVC/MVC extension NAL units")
    if aso:
        excluded.append("arbitrary slice order")
    summary = {
        "profile_idc": sorted(facts["profile_idc"]),
        "level_idc": sorted(facts["level_idc"]),
        "chroma_format_idc": sorted(facts["chroma_format_idc"] or {1}),
        "entropy": sorted({"CABAC" if v else "CAVLC" for v in facts["entropy_coding_mode_flag"]}),
        "slice_types": sorted("PBISX"[t] if t < 4 else "SI" for t in slice_types),
        "multi_slice": multi_slice,
        "transform_8x8": 1 in facts["transform_8x8_mode_flag"],
        "scaling_matrices": bool(
            {1} & (facts["seq_scaling_matrix_present_flag"] | facts["pic_scaling_matrix_present_flag"])),
        "weighted_pred": 1 in facts["weighted_pred_flag"],
        "constrained_intra_pred": 1 in facts["constrained_intra_pred_flag"],
        "max_num_ref_frames": max(facts["max_num_ref_frames"] or {0}),
        "poc_types": sorted(facts["pic_order_cnt_type"]),
        "cropping": 1 in facts["frame_cropping_flag"],
        "gaps_in_frame_num": 1 in facts["gaps_in_frame_num_allowed_flag"],
        "list_modification": 1 in facts["ref_pic_list_modification_flag_l0"],
        "mmco": sorted(v for v in facts["memory_management_control_operation"] if v),
        "long_term": 1 in facts["long_term_reference_flag"] or bool(
            {3, 6} & facts["memory_management_control_operation"]),
        "num_ref_idx_override": 1 in facts["num_ref_idx_active_override_flag"],
        "nal_types": sorted(t for t in facts["nal_types"] if t is not None),
    }
    return summary, excluded


TRACKED = {
    "profile_idc", "level_idc", "chroma_format_idc", "bit_depth_luma_minus8",
    "bit_depth_chroma_minus8", "frame_mbs_only_flag", "field_pic_flag",
    "entropy_coding_mode_flag", "num_slice_groups_minus1",
    "redundant_pic_cnt_present_flag", "transform_8x8_mode_flag",
    "seq_scaling_matrix_present_flag", "pic_scaling_matrix_present_flag",
    "weighted_pred_flag", "constrained_intra_pred_flag", "max_num_ref_frames",
    "pic_order_cnt_type", "frame_cropping_flag", "gaps_in_frame_num_allowed_flag",
    "ref_pic_list_modification_flag_l0", "memory_management_control_operation",
    "long_term_reference_flag", "num_ref_idx_active_override_flag", "slice_type",
    "qpprime_y_zero_transform_bypass_flag",
}


def reason(summary):
    """Why a selected stream matters: the tools it exercises."""
    tools = [f"profile {summary['profile_idc']}", "+".join(summary["entropy"]),
             "slices " + "".join(summary["slice_types"])]
    if summary["chroma_format_idc"] == [0]:
        tools.append("4:0:0")
    for key, label in (("multi_slice", "multiple slices"), ("transform_8x8", "8x8 transform"),
                       ("scaling_matrices", "scaling matrices"), ("weighted_pred", "weighted prediction"),
                       ("constrained_intra_pred", "constrained intra"), ("cropping", "cropping"),
                       ("gaps_in_frame_num", "frame_num gaps"), ("list_modification", "list modification"),
                       ("long_term", "long-term refs"), ("num_ref_idx_override", "ref count override")):
        if summary[key]:
            tools.append(label)
    if summary["mmco"]:
        tools.append("MMCO " + ",".join(map(str, summary["mmco"])))
    if summary["max_num_ref_frames"] > 1:
        tools.append(f"{summary['max_num_ref_frames']} refs")
    if summary["poc_types"] and summary["poc_types"] != [0]:
        tools.append("POC type " + ",".join(map(str, summary["poc_types"])))
    return "; ".join(tools)


def main():
    manifest_path, suites = sys.argv[1], [arg.split("=", 1) for arg in sys.argv[2:]]
    selected, rejected = [], []
    for suite, root in suites:
        for name in sorted(os.listdir(root), key=str.lower):
            directory = os.path.join(root, name)
            stream, yuv = bitstream_of(directory), reference_yuv_of(directory)
            if stream is None or yuv is None:
                rejected.append({"suite": suite, "name": name,
                                 "excluded": ["no single stream + reference YUV"]})
                continue
            packets = trace(os.path.join(directory, stream))
            if not packets:
                # ffmpeg refuses FMO streams outright (FM2_SVA_C).
                rejected.append({"suite": suite, "name": name,
                                 "excluded": ["ffmpeg cannot parse it (FMO)"]})
                continue
            summary, excluded = classify(packets)
            entry = {"suite": suite, "name": name, "stream": f"{name}/{stream}",
                     "yuv": f"{name}/{yuv}", **summary}
            if excluded:
                rejected.append({"suite": suite, "name": name, "excluded": excluded})
            else:
                entry["reason"] = reason(summary)
                selected.append(entry)
    with open(manifest_path, "w", encoding="utf-8") as out:
        json.dump({"profile": "Constrained High (H.264 A.2.4.2), 8-bit",
                   "suites": {"itu": "https://www.itu.int/wftp3/av-arch/jvt-site/draft_conformance/"
                                     " (AVCv1 and FRExt, without Hi422)",
                              "supplementary": "tools/dev/h264_conformance_supplement.sh"},
                   "selected": selected, "excluded": rejected}, out, indent=1)
        out.write("\n")
    print(f"{len(selected)} selected, {len(rejected)} excluded")


if __name__ == "__main__":
    main()
