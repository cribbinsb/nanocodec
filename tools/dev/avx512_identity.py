#!/usr/bin/env python3
"""Compare AVX2 and emulated AVX-512 streams via the shared streamed engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from nceval.diskguard import check_disk
from nceval.framepipe import FrameHashSink
from nceval.point import encode_streamed
from nceval.slots import cpu_slots
from nceval.yuv import YuvFormat

CODECS = ("h264", "hevc", "av1", "vp9", "av2", "vvc")
CLIPS = ("crowd_mot17_09", "bodycam_bc1", "static_meva_hospital")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Optional pre-change native binary")
    parser.add_argument("--sde", type=Path, required=True)
    parser.add_argument("--model", choices=("spr", "icx"), default="spr")
    parser.add_argument("--codec", choices=CODECS, required=True)
    parser.add_argument("--data", type=Path, default=Path.home() / "nanocodec-data")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--clip", choices=CLIPS, help="Run one clip to bound slower emulated cases")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    check_disk(args.work)
    clipset = json.loads((args.data / "clips/main.json").read_text())
    clips = {c["name"]: c for c in clipset["clips"]}
    results = []
    for name in ((args.clip,) if args.clip else CLIPS):
        clip = clips[name]
        for preset in ("default", f"{args.codec}-fast"):
            outputs = []
            label = f"{args.codec}-{preset}-{name}"
            for isa in (("avx2", "avx512", "baseline") if args.baseline else ("avx2", "avx512")):
                print(f"Checking {label}: {isa}", flush=True)
                stream = args.work / f"{label}-{isa}.bin"
                binary = args.baseline if isa == "baseline" else args.binary
                command = [str(binary.resolve()), "encode", "--codec", args.codec,
                           "--input", str(args.data / clip["yuv"]),
                           "--width", str(clip["width"]), "--height", str(clip["height"]),
                           "--fps", str(clip["fps"]), "--frames", "30",
                           "--intra-period", "300", "--qp",
                           "32" if args.codec in ("h264", "hevc", "vvc") else "140",
                           "--output", str(stream)]
                if preset != "default":
                    command += ["--preset", preset]
                if isa == "avx512":
                    command = [str(args.sde.resolve()), f"-{args.model}", "--", *command]
                sink = FrameHashSink()
                with cpu_slots(1):
                    frames = encode_streamed(
                        lambda out: command + ["--recon", out],
                        YuvFormat(clip["width"], clip["height"]), [sink],
                        env={**os.environ, "NC_DSP_ISA": "avx2" if isa == "baseline" else isa},
                    )
                outputs.append((hashlib.sha256(stream.read_bytes()).hexdigest(), sink.digests, frames))
            if any(output != outputs[0] for output in outputs[1:]) or outputs[0][2] != 30:
                raise RuntimeError(f"ISA identity failed: {label}")
            results.append({"case": label, "sha256": outputs[0][0], "recon": outputs[0][1]})
            print(f"PASS {label}: 30 frames, stream and reconstruction identical", flush=True)
    suffix = f"-{args.clip}" if args.clip else ""
    (args.work / f"identity-{args.codec}-{args.model}{suffix}.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
