#!/usr/bin/env python3
"""VP9 decode yardstick: our decoder against libvpx's vpxdec, per stream and per stage.

For every `*.ivf` stream in a directory it checks that our decoded output is
MD5-identical to vpxdec's (piped, nothing written to disk) and counts the
instructions of both decodes under callgrind (single thread; vpxdec runs
`--threads=1 --noblit`). vpxdec's counts are cached by stream hash, since the
anchor does not change. With `--stages` it also groups each decoder's self
cost into decode stages (inlined code counts in the function it is inlined
into, or in its source file for our block decoder) and prints ours, libvpx's
and the excess per stage: the stage with the largest excess is the work list.

    tools/dev/vp9_dec_yardstick.py --bin target/profiling/nanocodec [--stages]

Build `--profile profiling` (release with debug info) so that stages resolve.
The default stream set is ours at 720p / 1080p and SVC L2T1 / L2T3, libvpx
real-time 720p and good-quality 1080p, and the official scaled-reference
vector `vp90-2-21-resize_inter_1920x1080_7_1-2`
(`~/nanocodec-data/work/vp9-dec-parity/streams`, see docs/handoff/vp9.md).
Jetson: time the same streams with `tools/dev/jperf.py` (cycles).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DATA = Path.home() / "nanocodec-data"
DEFAULT_STREAMS = DATA / "work/vp9-dec-parity/streams"
DEFAULT_VPXDEC = DATA / "anchors/x86_64/bin/vpxdec"
CACHE = DATA / "work/vp9-dec-parity/vpxdec-cache"
TIMEOUT_S = 1500

# (stage, pattern over "source file:function"); the first match wins.
OURS_STAGES = [
    ("loop filter kernels", r":nc_dsp::vp9_loop_filter|:nc_dsp::lpf"),
    ("loop filter driver", r"Deblocker|loopfilter\.rs|Units|lf_mask"),
    ("MC kernels", r":nc_dsp::vp9_mc|convolve|emulate_edges"),
    ("inverse + add", r":nc_dsp::vp9_inverse|:nc_dsp::vp9_txfm|add_coefficients|transform::inverse|add_block|vp9_inverse_add"),
    ("intra", r":nc_dsp::vp9_intra|predict_intra|intra_pred"),
    ("inter setup", r"predict_inter|predict_from"),
    ("tokens", r"coeff\.rs|tokens\.rs|bool_coder\.rs.*(reconstruct|residual)|reconstruct_inter_plane|residual"),
    ("mode info / mv", r"modes\.rs|mvref\.rs|tree\.rs|tree::read|motion::|read_mode_info|decoder/block\.rs|"
                       r"::search|find_mv_refs|assign_mv|read_skip|read_tx_size|read_segment_id|"
                       r"decoder/header\.rs|bool_coder\.rs"),
    ("probability adaptation / headers", r"probs\.rs:|subexp|FrameContext>::adapt|decoder::header::"),
    ("copies / borders / memset", r"memset|memcpy|memmove|extend|picture|recycl"),
    ("rest of block decode", r"decode_block|decode_partition|frame\.rs"),
]
VPX_STAGES = [
    ("loop filter kernels", r"vpx_lpf|lpf_"),
    ("loop filter driver", r"filter_selectively|build_mask|filter_block_plane|loop_filter|setup_mask|adjust_mask"),
    ("MC kernels", r"convolve|scaled_2d|scaled_horiz|scaled_vert|build_mc_border"),
    ("inverse + add", r"idct|iadst|iwht|iht|inverse_transform|transpose_16bit"),
    ("inter setup", r"dec_build_inter|inter_predictor|setup_pre_planes"),
    ("tokens", r"decode_coefs|decode_block_tokens|vpx_reader|reader_fill"),
    ("mode info / mv", r"read_mode_info|read_inter|read_intra|mv_ref|find_mv|read_mv|inc_mv|mv_class|"
                       r"read_selected_tx|read_.*mode|dec_find|copy_ref_frame|vp9_read|pred_context|left_block_mode|above_block_mode"),
    ("intra", r"intra|_predictor"),
    ("probability adaptation / headers", r"adapt|read_coef_probs|diff_update|tree_merge|read_frame_header|read_compressed|setup_past"),
    ("copies / borders / memset", r"memset|memcpy|memmove|extend|border|yv12"),
    ("rest of block decode", r"decode_partition|read_partition|set_offsets|decode_block|set_skip|dec_reset|"
                             r"setup_dst_planes|decode_tiles|set_plane_n4"),
]
COST_LINE = re.compile(r"\s*([\d,]+)\s+\([^)]*\)\s+(.*)")


def md5_of(cmd: list[str]) -> str:
    digest = hashlib.md5()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(1 << 20), b""):
            digest.update(chunk)
    if proc.returncode != 0:
        raise RuntimeError(f"failed ({proc.returncode}): {' '.join(cmd)}")
    return digest.hexdigest()


def ours_cmd(binary: Path, stream: Path, output: str) -> list[str]:
    return [str(binary), "decode", "--codec", "vp9", "--input", str(stream), "--output", output]


def vpx_cmd(vpxdec: Path, stream: Path) -> list[str]:
    return [str(vpxdec), "--threads=1", "--noblit", str(stream), "-o", "/dev/null"]


def callgrind(cmd: list[str], out: Path) -> int:
    proc = subprocess.run(
        ["valgrind", "--tool=callgrind", f"--callgrind-out-file={out}", *cmd],
        capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
    match = re.search(r"Collected : (\d+)", proc.stderr)
    if proc.returncode != 0 or not match:
        raise RuntimeError(f"callgrind failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return int(match.group(1))


def recorded_total(profile: Path) -> int:
    for line in profile.read_text(errors="replace").splitlines():
        if line.startswith(("totals:", "summary:")):
            return int(line.split()[1])
    raise RuntimeError(f"no totals in {profile}")


def self_costs(profile: Path) -> list[tuple[int, str]]:
    text = subprocess.run(
        ["callgrind_annotate", "--threshold=100", "--auto=no", "--inclusive=no", str(profile)],
        capture_output=True, text=True, check=True).stdout
    rows = []
    for line in text.splitlines():
        match = COST_LINE.match(line)
        if match and "PROGRAM TOTALS" not in line:
            rows.append((int(match.group(1).replace(",", "")), match.group(2)))
    return rows


def stages(profile: Path, rules: list[tuple[str, str]]) -> dict[str, int]:
    totals = {name: 0 for name, _ in rules} | {"other": 0}
    for cost, name in self_costs(profile):
        # Inlined standard-library code counts by the function it is in.
        if name.startswith("/rustc/"):
            name = ":" + name.split(":", 1)[1]
        stage = next((s for s, pattern in rules if re.search(pattern, name)), "other")
        totals[stage] += cost
    return totals


def stream_hash(stream: Path) -> str:
    return hashlib.sha256(stream.read_bytes()).hexdigest()[:16]


def measure(stream: Path, args: argparse.Namespace, work: Path) -> dict:
    expected = md5_of([str(args.vpxdec), "--threads=1", "--i420", "-o", "-", str(stream)])
    got = md5_of(ours_cmd(args.bin, stream, "/dev/stdout"))
    ours_cg = work / f"ours.{stream.stem}.cg"
    ours = callgrind(ours_cmd(args.bin, stream, "/dev/null"), ours_cg)
    CACHE.mkdir(parents=True, exist_ok=True)
    vpx_cg = CACHE / f"vpx.{stream_hash(stream)}.cg"
    if vpx_cg.exists():
        vpx = recorded_total(vpx_cg)
    else:
        vpx = callgrind(vpx_cmd(args.vpxdec, stream), vpx_cg)
    row = {"stream": stream.name, "identical": got == expected, "ours": ours, "vpx": vpx}
    if args.stages:
        row["stages_ours"] = stages(ours_cg, OURS_STAGES)
        row["stages_vpx"] = stages(vpx_cg, VPX_STAGES)
    if args.keep:
        row["profile"] = str(args.keep / ours_cg.name)
        args.keep.mkdir(parents=True, exist_ok=True)
        ours_cg.replace(args.keep / ours_cg.name)
    return row


def print_stages(row: dict) -> None:
    ours, vpx = row["stages_ours"], row["stages_vpx"]
    print(f"  {'stage':34} {'ours M':>9} {'vpx M':>9} {'excess':>9}")
    for stage in sorted(ours, key=lambda s: vpx.get(s, 0) - ours[s]):
        a, b = ours[stage] / 1e6, vpx.get(stage, 0) / 1e6
        print(f"  {stage:34} {a:9.1f} {b:9.1f} {a - b:+9.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin", type=Path, required=True, help="our nanocodec binary")
    parser.add_argument("--streams", type=Path, default=DEFAULT_STREAMS)
    parser.add_argument("--vpxdec", type=Path, default=DEFAULT_VPXDEC)
    parser.add_argument("--filter", default="", help="only streams whose name contains this")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--stages", action="store_true", help="per-stage table per stream")
    parser.add_argument("--keep", type=Path, help="keep our callgrind profiles here")
    parser.add_argument("--json", type=Path, help="write the rows here")
    args = parser.parse_args()
    streams = sorted(p for p in args.streams.glob("*.ivf") if args.filter in p.name)
    if not streams:
        print(f"no streams in {args.streams}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="vp9yard-") as tmp, \
            concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda s: measure(s, args, Path(tmp)), streams))
    print(f"{'stream':40} {'same':>5} {'ours G':>8} {'vpx G':>8} {'ours/vpx':>9}")
    for row in rows:
        print(f"{row['stream']:40} {'yes' if row['identical'] else 'NO':>5} "
              f"{row['ours'] / 1e9:8.3f} {row['vpx'] / 1e9:8.3f} {row['ours'] / row['vpx']:9.3f}")
        if args.stages:
            print_stages(row)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 0 if all(row["identical"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
