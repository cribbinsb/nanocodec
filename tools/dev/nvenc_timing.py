#!/usr/bin/env python3
"""NVENC encode speed for the chart's hardware encoders (eval/nceval/chart.py HARDWARE).

NVENC is fixed-function GPU hardware, so its speed is not comparable with the
chart's single-core CPU fps: it is shown beside the CPU encoders, never on
their Pareto front. Every family is timed with the chart's own encoder
arguments (cbr.Family.command: low-delay CBR) at the CPU timing's rung, 720p,
on its clips (anchor_chart_timing RUNG_KBPS, CLIPS), the stream discarded.
Three measures (`--measure`):

  pipeline   the earlier figure: one ffmpeg process reads the raw clip, converts
             and uploads every frame and encodes it. Pipeline-bound (2-4x below
             NVIDIA's published encoder rates), kept for transparency.
  encode     encode only, one stream: the clip's FRAMES frames are uploaded to GPU
             memory once (hwupload_cuda) and looped there (the loop filter), so
             the encoder reads frames already in GPU memory.
  aggregate  the same with 1, 2, 4 and 8 streams at once (the GPU has several
             NVENC engines): total frames per second of all streams, on one clip.

Each time is a difference, fps = frames / (t(LONG_LOOPS) - t(SHORT_LOOPS)), so
start-up, session setup, the read and the upload cancel; the median of ROUNDS
rounds per clip. `--reference` also times presets P1/P5/P7 (tune ll and hq)
of H.264 and HEVC at 1080p (the clip scaled on the GPU), single stream, for a
sanity check against NVIDIA's published NVENC rates.

The runs hold NVENC slots (nceval.slots.nvenc_slots): one per stream, or the
whole pool for the parallel runs, whose up to 8 sessions are an owner-approved
exception to the 2-encode rule (2026-11-12).

    cd eval && PYTHONPATH=. python3 ../tools/dev/nvenc_timing.py \\
        --out ../docs/results/bd-vs-speed-timing-nvenc.json [--measure encode,aggregate]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anchor_chart_timing import CLIPS, FRAMES, INTRA_PERIOD, RUNG_KBPS  # noqa: E402

ROUNDS = 3
MEASURES = ("pipeline", "encode", "aggregate")
#: Loops of the clip's frames in GPU memory: the timed difference is
#: (LONG_LOOPS - SHORT_LOOPS) x FRAMES encoded frames (3000, ~3 s at 1000 fps).
LONG_LOOPS, SHORT_LOOPS = 21, 1
#: The pipeline measure's subtracted run: ffmpeg start-up and session setup.
SETUP_FRAMES = 1
#: Parallel streams of the aggregate measure, and its clip.
STREAMS = (1, 2, 4, 8)
AGGREGATE_CLIP = "crowd_mot17_09"
#: The 1080p sanity check: encoders, presets, tunes and rate (the 720p rung scaled).
REFERENCE_ENCODERS = ("nvenc-h264", "nvenc-hevc")  # their codec's encoder
REFERENCE_PRESETS = ("p1", "p5", "p7")
REFERENCE_TUNES = ("ll", "hq")
REFERENCE_SIZE = (1920, 1080)
REFERENCE_KBPS = 1200


def run_seconds(commands: list[list[str]]) -> float:
    """Wall time until every command (run at once) has finished."""
    began = time.perf_counter()
    procs = [subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE) for cmd in commands]
    for proc, cmd in zip(procs, commands):
        _, err = proc.communicate()
        if proc.returncode:
            raise RuntimeError(f"{' '.join(cmd)}: {err.decode(errors='replace')[-300:]}")
    return time.perf_counter() - began


def gpu_resident(cmd: list[str], loops: int, scale: tuple[int, int] | None = None) -> list[str]:
    """`cmd` (cbr.Family.command) with its FRAMES input frames uploaded to GPU memory
    once and looped `loops` times there; `scale` resizes them on the GPU first."""
    head = cmd.index("-f")  # the raw input's options start here
    frames = cmd.index("-frames:v")
    count = int(cmd[frames + 1])
    chain = ["hwupload_cuda", *([f"scale_cuda={scale[0]}:{scale[1]}"] if scale else []),
             f"loop=loop={loops - 1}:size={count}:start=0"]
    return [*cmd[:head], "-init_hw_device", "cuda=gpu", "-filter_hw_device", "gpu",
            *cmd[head:frames], "-vf", ",".join(chain), "-frames:v", str(count * loops),
            *cmd[frames + 2:]]


def encode_fps(command, streams: int = 1, scale: tuple[int, int] | None = None) -> float:
    """Total encode-only fps of `streams` concurrent streams (`command(loops)`: one
    stream's argv), median of ROUNDS rounds."""
    from nceval.slots import nvenc_slots

    def timed(loops: int) -> float:
        return run_seconds([gpu_resident(command(), loops, scale)] * streams)

    rounds = []
    with nvenc_slots(streams):
        timed(SHORT_LOOPS)  # warm-up: page cache, driver
        for _ in range(ROUNDS):
            seconds = timed(LONG_LOOPS) - timed(SHORT_LOOPS)
            rounds.append(streams * (LONG_LOOPS - SHORT_LOOPS) * FRAMES / seconds)
    return statistics.median(rounds)


def pipeline_fps(family, clip, kbps: int) -> float:
    """The earlier measure: the raw clip read, converted and uploaded by one ffmpeg
    process, fps = (FRAMES - 1) / (t(FRAMES) - t(1))."""
    from nceval.slots import nvenc_slots

    def timed(frames: int) -> float:
        return run_seconds([family.command(clip, frames, INTRA_PERIOD,
                                           pathlib.Path("/dev/null"), None, kbps=kbps)])

    rounds = []
    with nvenc_slots():
        timed(FRAMES)
        for _ in range(ROUNDS):
            rounds.append((FRAMES - SETUP_FRAMES) / (timed(FRAMES) - timed(SETUP_FRAMES)))
    return statistics.median(rounds)


def chart_command(family, clip, kbps: int):
    return lambda: family.command(clip, FRAMES, INTRA_PERIOD, pathlib.Path("/dev/null"),
                                  None, kbps=kbps)


def reference_1080p(paths, clip) -> dict[str, float]:
    """Single-stream encode-only fps of each REFERENCE preset and tune at 1080p: the
    REFERENCE_ENCODERS families' command with another preset and tune."""
    from nceval.cbr import nvenc_family

    fps = {}
    for name in REFERENCE_ENCODERS:
        family = nvenc_family(name, paths)
        encoder = family.extra[family.extra.index("-c:v") + 1]
        for preset in REFERENCE_PRESETS:
            for tune in REFERENCE_TUNES:
                variant = replace(family, extra=("-c:v", encoder, "-preset", preset,
                                                 "-tune", tune))
                key = f"{encoder} {preset.upper()}-{tune.upper()}"
                fps[key] = round(encode_fps(chart_command(variant, clip, REFERENCE_KBPS),
                                            scale=REFERENCE_SIZE), 1)
                print(key, fps[key], flush=True)
    return fps


def platform() -> str:
    from nceval.config import find_tool

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                          "--format=csv,noheader", "--id=0"],
                         capture_output=True, text=True, check=True).stdout.strip()
    ffmpeg = subprocess.run([str(find_tool("ffmpeg")), "-version"], capture_output=True,
                            text=True, check=True).stdout.split()[2]
    return f"{gpu.replace(', ', ' NVENC, driver ')}, ffmpeg {ffmpeg}"


def main() -> None:
    from nceval.chart import HARDWARE
    from nceval.cbr import clip_targets, nvenc_family
    from nceval.config import load_paths
    from nceval.realuse import main_clips

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="timing JSON for chart render "
                                                     "--timing-hardware")
    parser.add_argument("--points", help="comma-separated hardware families (default: all)")
    parser.add_argument("--measure", default="encode,aggregate",
                        help=f"comma-separated, of {', '.join(MEASURES)}")
    parser.add_argument("--reference", action="store_true",
                        help="also the 1080p preset sanity check")
    args = parser.parse_args()
    measures = args.measure.split(",") if args.measure else []
    paths = load_paths()
    clips = {name: main_clips(paths)[name] for name in CLIPS}
    names = args.points.split(",") if args.points else [name for name, _ in HARDWARE]
    out = pathlib.Path(args.out)
    timing = json.loads(out.read_text()) if out.is_file() else {"points": {}}
    timing.update(platform=platform(), rung_kbps=RUNG_KBPS, frames=FRAMES, rounds=ROUNDS,
                  aggregate_clip=AGGREGATE_CLIP)
    for name in names:
        family, entry = nvenc_family(name, paths), timing["points"].setdefault(name, {})
        rate = {c: clip_targets(clip, (RUNG_KBPS,))[0] for c, clip in clips.items()}
        if "pipeline" in measures:
            entry["pipeline_fps"] = {c: round(pipeline_fps(family, clip, rate[c]), 1)
                                     for c, clip in clips.items()}
        if "encode" in measures:
            entry["encode_fps"] = {c: round(encode_fps(chart_command(family, clip, rate[c])), 1)
                                   for c, clip in clips.items()}
        if "aggregate" in measures:
            clip = clips[AGGREGATE_CLIP]
            command = chart_command(family, clip, rate[AGGREGATE_CLIP])
            entry["aggregate_fps"] = {str(n): round(encode_fps(command, n), 1) for n in STREAMS}
        print(name, {k: v for k, v in entry.items()}, flush=True)
        out.write_text(json.dumps(timing, indent=1) + "\n")
    if args.reference:
        timing["reference_1080p"] = {"kbps": REFERENCE_KBPS, "fps": reference_1080p(
            paths, clips[AGGREGATE_CLIP])}
        out.write_text(json.dumps(timing, indent=1) + "\n")


if __name__ == "__main__":
    main()
