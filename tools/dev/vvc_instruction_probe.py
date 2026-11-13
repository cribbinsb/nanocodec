#!/usr/bin/env python3
"""Bounded VVC/HEVC instruction profiles through the shared streamed full gate.

One key and one key+two-P encode per codec; subtraction estimates mean P cost.
These are execution-cost diagnostics, not matched-quality speed comparisons.
No raw video is written and no wall-clock performance result is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import point
from nceval.codecs import Codec
from nceval.config import load_paths
from nceval.diskguard import check_disk
from nceval.metrics import StreamSpec
from nceval.yuv import YuvFormat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip", default="static_chirla_s000c3_360p")
    parser.add_argument("--qp", type=int, default=48)
    parser.add_argument("--param", action="append", default=[], help="Shared/backend parameter override name=value")
    parser.add_argument("--codecs", nargs="+", choices=("vvc", "hevc"), default=["vvc", "hevc"])
    args = parser.parse_args()
    if not 0 <= args.qp <= 51:
        parser.error("shared diagnostic QP must be in 0..51")
    paths = load_paths()
    check_disk(paths.work)
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.cli.resolve()
    manifest = json.loads((paths.clips / "quick360.json").read_text())
    clip = next(c for c in manifest["clips"] if c["name"] == args.clip)
    reference = paths.data_root / clip["yuv"]
    fingerprint = hashlib.sha256(binary.read_bytes()).hexdigest()
    metadata = {
        "binary": str(binary), "binary_sha256": fingerprint, "clip": clip,
        "qp": args.qp, "codecs": args.codecs, "params": args.param,
        "note": "Instruction counts only; same QP is not matched quality. Mean P is (3f - 1f)/2, including frame I/O and harness-facing recon writes.",
        "valgrind": subprocess.check_output(["valgrind", "--version"], text=True).strip(),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # The shared point engine owns subprocess cleanup; bound each encode well
    # below the project's 30-minute ceiling, including under instrumentation.
    point.ENCODE_TIMEOUT_SECONDS = 240
    rows = []
    for name in args.codecs:
        codec = Codec(name)
        key_count = None
        for frames in (1, 3):
            label = f"{name}-q{args.qp}-{frames}f"
            stream = args.output / (label + codec.bitstream_suffix)
            profile = args.output / (label + ".callgrind")
            spec = StreamSpec(reference, YuvFormat(clip["width"], clip["height"]), frames, clip["fps"], paths)

            def command(recon):
                return ["valgrind", "--tool=callgrind", f"--callgrind-out-file={profile}",
                        str(binary), "encode", "--codec", name, "--input", str(reference),
                        "--width", str(clip["width"]), "--height", str(clip["height"]),
                        "--fps", str(clip["fps"]), "--frames", str(frames), "--qp", str(args.qp),
                        "--intra-period", "300", "--output", str(stream), "--recon", recon,
                        *(part for value in args.param for part in ("--param", value))]

            result = point.run_point(command, stream, codec, spec, paths,
                                     own_decoder=binary, serial_gate=True)
            if not result.gate.passed:
                raise RuntimeError(f"{label}: {result.gate.reason}")
            count = next(int(line.split()[1]) for line in profile.read_text().splitlines()
                         if line.startswith("summary:"))
            if frames == 1:
                key_count = count
            annotation = subprocess.check_output(
                ["callgrind_annotate", "--inclusive=no", "--threshold=95", str(profile)], text=True)
            (args.output / (label + ".self.txt")).write_text(annotation)
            row = {"codec": name, "frames": frames, "instructions": count,
                   "mean_p_instructions": (count - key_count) / 2 if frames == 3 else None,
                   "stream_bytes": stream.stat().st_size, "gate": result.gate.level,
                   "bitstream_sha256": hashlib.sha256(stream.read_bytes()).hexdigest(),
                   "recon_sha256": result.recon_sha256}
            rows.append(row)
            (args.output / "instructions.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
