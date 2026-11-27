#!/usr/bin/env python3
"""Train a PGO-instrumented reference encoder on the benchmark's own command lines.

build_anchors.sh --opt runs this between a tool's instrumented build and its
profile-use build (docs/PROCEDURES.md "Reference builds"). Every charted family of the
tool (nceval.fairness.families) codes each training clip twice, with the exact argv the
benchmark runs (nceval.fairness.command): CBR at the chart's middle rate, and Fixed QP at
the middle of the tool's CTC quantizer table (docs/benchmark-procedure.md section 4).
Encodes run side by side: gcc's profile runtime merges concurrent processes' counters
under a file lock.

    pgo_train.py --tool TOOL --bin DIR [--cfg DIR] --work DIR [--jobs N] [--frames N] CLIP...
    pgo_train.py --smoke ...   one family per mode (ours: per codec) on the first clip:
                               the installed binaries' check

TOOL is the record's tool name (x264, x265, openh264-harness, SvtAv1EncApp, vpxenc,
svc_encoder_rtc for libaom: its CBR program, aomenc in Fixed QP, kvazaar, vvencFFapp), or
nanocodec for --smoke only (ours trains through tools/dev/pgo_build.sh).
Clips are 1280x720 I420 (.yuv). The last line printed describes the training for the
build record (build_record.py --pgo).
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import fairness  # noqa: E402

WIDTH, HEIGHT = 1280, 720
#: The chart's middle CBR rung at 720p30 (docs/benchmark-fairness.toml [shared] rate_kbps),
#: and the training clips' nominal rate.
TRAIN_KBPS, FPS = 600, 30
#: The middle of each scale's CTC table (nceval/ctc/conditions.py POINTS): QP for the
#: H.264/HEVC/VVC encoders, the 0-63 quantizer for libvpx, libaom and SVT-AV1.
MID_QP = {"x264": 32, "x265": 32, "openh264-harness": 32, "kvazaar": 32, "vvencFFapp": 32,
          "SvtAv1EncApp": 40, "vpxenc": 40, "aomenc": 40, "svc_encoder_rtc": 40}
#: Ours per codec: the middle of its scale's CTC table (the 0-255 qindex for AV1, VP9, AV2).
OURS_MID_QP = {"h264": 32, "hevc": 32, "vvc": 32, "av1": 160, "vp9": 160, "av2": 160}
#: The record's x86-only option pairs, left out on other hosts: x265's AVX-512 tier
#: (docs/benchmark-fairness.toml, x265 "--asm avx512") is not an aarch64 one, and x265
#: refuses it there; on aarch64 x265's own run-time dispatch picks its NEON/SVE/SVE2 kernels.
X86_ONLY_OPTIONS = (("--asm", "avx512"),)
#: One key per clip, as the benchmark (frames < keyint).
KEYINT = 300


def jobs_for(tool: str, clips: list[Path], frames: int, work: Path, bin_dir: Path,
             cfg: Path | None, smoke: bool) -> list[list[str]]:
    """Every family x mode x clip argv; with `smoke`, each mode's first family on one clip
    (ours: the first of each codec).
    The '+drop' rows are left out: they differ from their family only in frame dropping,
    and openh264's harness exits on a skipped frame (it codes every frame for the metrics)."""
    families = {f: v for f, (t, v) in fairness.families().items()
                if t == tool and not f.endswith(fairness.DROP_SUFFIX)}
    if not families:
        raise SystemExit(f"no charted family runs {tool}")
    argvs = []
    for mode in ("cbr", "qp"):
        smoked: set[str | None] = set()
        for family, values in families.items():
            if smoke and values.get("CODEC") in smoked:
                continue
            qp = OURS_MID_QP[values["CODEC"]] if tool == "nanocodec" else MID_QP[tool]
            try:
                argvs += [fairness.command(family, mode, {
                    "BIN": str(bin_dir), "CFG": str(cfg or ""), "INPUT": str(clip),
                    "OUTPUT": str(work / f"{family}-{mode}-{index}.bin"), "W": WIDTH,
                    "H": HEIGHT, "FPS_NUM": FPS, "FPS_DEN": 1, "N": frames, "KEYINT": KEYINT,
                    "KBPS": TRAIN_KBPS, "QP": qp}) for index, clip in enumerate(clips)]
            except fairness.NotCharted:
                continue
            smoked.add(values.get("CODEC"))
    return [host_argv(argv) for argv in argvs]


def host_argv(argv: list[str]) -> list[str]:
    """`argv` without X86_ONLY_OPTIONS when this host is not x86_64."""
    if platform.machine() == "x86_64":
        return argv
    out: list[str] = []
    for word in argv:
        if out and (out[-1], word) in X86_ONLY_OPTIONS:
            out.pop()
        else:
            out.append(word)
    return out


def run(argv: list[str], log: Path) -> int:
    with log.open("a") as out:
        out.write("+ " + " ".join(argv) + "\n")
        out.flush()
        return subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, check=False).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tool", required=True)
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--cfg", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--frames", type=int, help="frames per clip (default: the whole clip)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("clips", type=Path, nargs="+")
    args = ap.parse_args()
    frame_bytes = WIDTH * HEIGHT * 3 // 2
    frames = args.frames or min(c.stat().st_size // frame_bytes for c in args.clips)
    clips = args.clips[:1] if args.smoke else args.clips
    args.work.mkdir(parents=True, exist_ok=True)
    argvs = jobs_for(args.tool, clips, frames, args.work, args.bin.resolve(), args.cfg,
                     args.smoke)
    log = args.work / "train.log"
    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(lambda a: run(a, log), argvs))
    failed = [" ".join(a) for a, c in zip(argvs, codes) if c]
    if failed:
        raise SystemExit(f"{len(failed)} of {len(argvs)} encodes failed (log {log}); first: {failed[0]}")
    qp = MID_QP.get(args.tool, "at the middle of each codec's table")
    print(f"the benchmark record's commands (nceval.fairness.command) for every charted "
          f"{args.tool} family but the +drop rows: CBR {TRAIN_KBPS} kbps and Fixed QP {qp}, "
          f"{len(clips)} clips x {frames} frames, {len(argvs)} encodes")


if __name__ == "__main__":
    main()
