#!/usr/bin/env python3
"""WASM vs native decode/encode speed benchmark for the browser codec module.

Encodes a fixed stream set with the native CLI at HEAD (crowd/pan/static,
720p and 360p, CBR targets per resolution (`--targets`), 150 frames), then times

* native decode (`nanocodec decode`, output to /dev/null),
* the reference decoder of each codec, single-threaded (`REFERENCE_DECODERS`:
  FFmpeg's own H.264, HEVC, VP9 and VVC decoders with `-threads 1`, dav1d
  with `--threads 1` for AV1, the pinned avmdec anchor for AV2), the yardstick our native
  decoder should be at or under, and
* WASM decode under node for each module (`--modules`: `simd128` is `--pkg`,
  `relaxed` its `relaxed/` build, which the page loads where supported) with
  `bench-runner.mjs decode`: the demo's `BrowserDecoder.decode` plus the RGBA
  hand-off out of WASM memory,

checking that every module's I420 output hashes equal the native decoder's.
It also times WASM encode at the demo's settings (15 fps, CBR with 2x peak,
1 s buffer, 30 s keyframes; `ENCODE_KBPS` per resolution, `--encode-res`)
for each preset the page offers (`--presets`: Fast only where the demo has
it) against the native CLI at the same settings.

Every timing is single-threaded wall time. Each round (`--rounds`) runs every
job once per module and natively, interleaved, each run the best of
`--repeat` passes; the best over rounds is reported. `--cpu N` pins every timed
process to one core: for absolute fps run the script under
tools/dev/host-timed.sh with `--cpu 14` (a reserved timing core).
Streams live under the work directory (bitstreams only, keyed by the encoder
binary); no YUV is written.

    tools/dev/host-timed.sh python3 tools/web/bench.py --bin $CARGO_TARGET_DIR/release [--pkg web/pkg] \
        [--codecs h264,hevc,av1,vp9,av2,vvc] [--targets 720p=600,360p=300] \
        [--rounds 3] [--cpu 14] [--skip-encode] [--json out.json] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
RUNNER = REPO / "tools/web/bench-runner.mjs"
CODECS = ("h264", "hevc", "av1", "vp9", "av2", "vvc")
# Annex-B stream extensions; the other codecs are stored as IVF.
ANNEX_B_EXTENSIONS = {"h264": "264", "hevc": "hevc", "vvc": "vvc"}
FRAMES = 150
# Single-threaded reference decoders by codec: argv before the stream path
# and after it. Output is discarded (no YUV written).
REFERENCE_DECODERS = {
    **{codec: ("ffmpeg", ["ffmpeg", "-nostdin", "-loglevel", "error", "-threads", "1", "-i"],
               ["-f", "null", "-"]) for codec in ("h264", "hevc", "vp9", "vvc")},
    "av1": ("dav1d", ["dav1d", "--threads", "1", "--muxer", "null", "-o", "/dev/null", "-q", "-i"], []),
    # AV2's reference decoder is not on PATH: the pinned anchor build.
    "av2": ("avmdec", [str(DATA / "anchors/x86_64/bin/avmdec"), "--threads=1", "--noblit"], []),
}
# (label, clip directory, clip names): one clip per content class.
RESOLUTIONS = {
    "720p": ("quick", ("crowd_mot17_09", "pan_jaad_0117", "static_chirla_s000c3")),
    "360p": (
        "quick360",
        ("crowd_mot17_09_360p", "pan_jaad_0117_360p", "static_chirla_s000c3_360p"),
    ),
}
# Default CBR decode targets per resolution. 360p stays at <= 50 % of its
# sources' bitrate (crowd_mot17_09_360p is ~750 kbps), so its high target is
# 300, not 1200.
TARGETS_KBPS = {"720p": (300, 1200), "360p": (150, 300)}
PEAK_FACTOR = 2
BUFFER_MS = 1000
# The demo's defaults (web/index.html, web/panel.js).
DEMO = {"fps": 15, "buffer_ms": 1000, "key_s": 30}
# Encode targets: the demo's 300 kbps at its default 640x360, twice that at 720p.
ENCODE_KBPS = {"720p": 600, "360p": 300}
# Codecs whose demo panel offers the Fast preset (web/codecs.js).
WEB_FAST_CODECS = ("h264", "hevc", "av1", "vvc")
# `--modules` names: the module directory under `--pkg` (tools/web/build.sh).
MODULE_DIRS = {"simd128": ".", "relaxed": "relaxed"}


@dataclass(frozen=True)
class Clip:
    name: str
    yuv: Path
    width: int
    height: int
    fps: float
    res: str


def load_clips() -> list[Clip]:
    clips = []
    for res, (setname, names) in RESOLUTIONS.items():
        index = {c["name"]: c for c in json.loads((DATA / f"clips/{setname}.json").read_text())["clips"]}
        for name in names:
            c = index[name]
            clips.append(Clip(name, DATA / c["yuv"], c["width"], c["height"], c["fps"], res))
    return clips


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream_path(work: Path, codec: str, clip: Clip, kbps: int) -> Path:
    ext = ANNEX_B_EXTENSIONS.get(codec, "ivf")
    return work / f"{clip.name}.{kbps}k.{codec}.{ext}"


def encode_args(nanocodec: Path, codec: str, clip: Clip, kbps: int, frames: int, fps, out) -> list[str]:
    return [
        str(nanocodec), "encode", "--codec", codec, "--input", str(clip.yuv),
        "--width", str(clip.width), "--height", str(clip.height), "--fps", str(fps),
        "--bitrate", str(kbps), "--peak-bitrate", str(PEAK_FACTOR * kbps),
        "--buffer-ms", str(BUFFER_MS), "--frames", str(frames), "--output", str(out),
    ]


def parse_targets(text: str) -> dict[str, tuple[int, ...]]:
    """`720p=300+1200,360p=150+300` -> {"720p": (300, 1200), "360p": (150, 300)}."""
    targets = {}
    for item in text.split(","):
        res, rates = item.split("=")
        targets[res] = tuple(int(r) for r in rates.split("+"))
    return targets


def make_streams(nanocodec: Path, work: Path, codecs, clips, targets, workers: int) -> list[tuple]:
    jobs = []
    for codec in codecs:
        for clip in clips:
            for kbps in targets[clip.res]:
                jobs.append((codec, clip, kbps, stream_path(work, codec, clip, kbps)))
    todo = [j for j in jobs if not j[3].exists()]
    print(f"streams: {len(jobs)} ({len(todo)} to encode)", file=sys.stderr)

    def run(job):
        codec, clip, kbps, out = job
        tmp = out.with_suffix(out.suffix + ".part")
        subprocess.run(["nice", "-n", "19", *encode_args(nanocodec, codec, clip, kbps, FRAMES, clip.fps, tmp)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp.rename(out)

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, todo))
    return jobs


def native_decode_hash(nanocodec: Path, codec: str, stream: Path) -> str:
    proc = subprocess.Popen([str(nanocodec), "decode", "--codec", codec, "--input", str(stream),
                             "--output", "/dev/stdout"], stdout=subprocess.PIPE)
    digest = hashlib.sha256()
    for chunk in iter(lambda: proc.stdout.read(1 << 20), b""):
        digest.update(chunk)
    if proc.wait() != 0:
        raise RuntimeError(f"native decode failed: {stream}")
    return digest.hexdigest()


def pinned(command: list[str], cpu: int | None) -> list[str]:
    return command if cpu is None else ["taskset", "-c", str(cpu), *command]


def wall(command: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - start


def node(cpu: int | None, *args: str) -> dict:
    out = subprocess.run(pinned(["node", str(RUNNER), *args], cpu), check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def better(old: dict | None, new: dict) -> dict:
    """The faster of two runner results (each already the best of its passes)."""
    return new if old is None or new["ms_per_frame"] < old["ms_per_frame"] else old


def reference_command(codec: str, stream: Path) -> list[str] | None:
    """The reference decoder's command line for `stream`, if `codec` has one."""
    if codec not in REFERENCE_DECODERS:
        return None
    _, before, after = REFERENCE_DECODERS[codec]
    return [*before, str(stream), *after]


def bench_decode(nanocodec: Path, modules: dict[str, Path], jobs, opts) -> list[dict]:
    """Interleaved rounds: each round runs every stream once per module, natively and
    with the codec's reference decoder."""
    wasm: dict[tuple, dict] = {}
    native: dict[Path, float] = {}
    reference: dict[Path, float] = {}
    for round_index in range(opts.rounds):
        for codec, clip, kbps, stream in jobs:
            for label, pkg in modules.items():
                extra = ["--hash"] if round_index == 0 else []
                out = node(opts.cpu, "decode", "--pkg", str(pkg), "--codec", codec, "--stream", str(stream),
                           "--repeat", str(opts.repeat), *extra)
                if round_index == 0 and out["i420_sha256"] != native_decode_hash(nanocodec, codec, stream):
                    raise SystemExit(f"WASM ({label}) decode differs from native: {stream}")
                wasm[(stream, label)] = better(wasm.get((stream, label)), out)
            command = pinned([str(nanocodec), "decode", "--codec", codec, "--input", str(stream),
                              "--output", "/dev/null"], opts.cpu)
            native[stream] = min(native.get(stream, float("inf")), *(wall(command) for _ in range(opts.repeat)))
            if (command := reference_command(codec, stream)) is not None:
                command = pinned(command, opts.cpu)
                reference[stream] = min(reference.get(stream, float("inf")),
                                        *(wall(command) for _ in range(opts.repeat)))
        print(f"decode round {round_index + 1}/{opts.rounds} done", file=sys.stderr)
    rows = []
    for codec, clip, kbps, stream in jobs:
        for label in modules:
            best = wasm[(stream, label)]
            rows.append({
                "codec": codec, "clip": clip.name, "res": clip.res, "kbps_target": kbps, "module": label,
                "kbps_actual": round(stream.stat().st_size * 8 * clip.fps / best["frames"] / 1000, 1),
                "frames": best["frames"], "wasm_ms_per_frame": best["ms_per_frame"],
                "wasm_fps": best["fps"], "wasm_max_frame_ms": best["max_frame_ms"],
                "native_fps": round(best["frames"] / native[stream], 2), "bit_exact": True,
                "reference": REFERENCE_DECODERS.get(codec, (None,))[0],
                "reference_fps": round(best["frames"] / reference[stream], 2) if stream in reference else None,
                # Split of wasm_ms_per_frame, when the module separates the hand-off.
                "wasm_decode_ms": best.get("decode_ms_per_frame"),
                "wasm_display_ms": best.get("display_ms_per_frame"),
            })
    return rows


def bench_encode(nanocodec: Path, modules: dict[str, Path], work: Path, codecs, clips, opts) -> list[dict]:
    """WASM and native encode at the demo's settings, interleaved like bench_decode."""
    jobs = [(codec, preset, clip) for codec in codecs for preset in opts.presets
            if preset == "default" or codec in WEB_FAST_CODECS
            for clip in clips if clip.res in opts.encode_res]
    wasm: dict[tuple, dict] = {}
    native: dict[tuple, float] = {}
    for round_index in range(opts.rounds):
        for job in jobs:
            codec, preset, clip = job
            kbps = ENCODE_KBPS[clip.res]
            for label, pkg in modules.items():
                out = node(opts.cpu, "encode", "--pkg", str(pkg), "--codec", codec, "--yuv", str(clip.yuv),
                           "--width", str(clip.width), "--height", str(clip.height),
                           "--frames", str(FRAMES), "--fps", str(DEMO["fps"]), "--kbps", str(kbps),
                           "--peak-kbps", str(PEAK_FACTOR * kbps), "--buffer-ms", str(DEMO["buffer_ms"]),
                           "--key-seconds", str(DEMO["key_s"]), "--repeat", str(opts.repeat),
                           *(["--fast"] if preset == "fast" else []))
                wasm[(job, label)] = better(wasm.get((job, label)), out)
            out_file = work / f"native-encode.{codec}.out"
            command = pinned([str(nanocodec), "encode", "--codec", codec, "--input", str(clip.yuv),
                              "--width", str(clip.width), "--height", str(clip.height),
                              "--fps", str(DEMO["fps"]), "--bitrate", str(kbps),
                              "--peak-bitrate", str(PEAK_FACTOR * kbps), "--buffer-ms", str(DEMO["buffer_ms"]),
                              "--intra-period", str(DEMO["key_s"] * DEMO["fps"]),
                              *(["--preset", f"{codec}-fast"] if preset == "fast" else []),
                              "--frames", str(FRAMES), "--output", str(out_file)], opts.cpu)
            native[job] = min(native.get(job, float("inf")), *(wall(command) for _ in range(opts.repeat)))
            out_file.unlink(missing_ok=True)
        print(f"encode round {round_index + 1}/{opts.rounds} done", file=sys.stderr)
    rows = []
    for job in jobs:
        codec, preset, clip = job
        for label in modules:
            best = wasm[(job, label)]
            rows.append({"codec": codec, "preset": preset, "clip": clip.name, "res": clip.res,
                         "kbps_target": ENCODE_KBPS[clip.res], "module": label, "frames": FRAMES,
                         "wasm_fps": best["fps"], "wasm_ms_per_frame": best["ms_per_frame"],
                         "wasm_kbps": best["achieved_kbps"], "native_fps": round(FRAMES / native[job], 2)})
    return rows


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Frame-weighted fps per group: total frames over total time."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, members in groups.items():
        frames = sum(r["frames"] for r in members)
        wasm = frames / sum(r["frames"] / r["wasm_fps"] for r in members)
        native = frames / sum(r["frames"] / r["native_fps"] for r in members)
        row = {**dict(zip(keys, key)), "wasm_fps": wasm, "native_fps": native,
               "wasm_ms": 1000 / wasm, "native_ms": 1000 / native, "ratio": native / wasm}
        if all(r.get("reference_fps") for r in members):
            row["reference_ms"] = sum(r["frames"] / r["reference_fps"] for r in members) * 1000 / frames
        if all(r.get("wasm_decode_ms") is not None for r in members):
            row["decode_ms"] = sum(r["frames"] * r["wasm_decode_ms"] for r in members) / frames
            row["display_ms"] = row["wasm_ms"] - row["decode_ms"]
            row["ratio"] = row["decode_ms"] / row["native_ms"]
        out.append(row)
    return out


def markdown(decode_rows, encode_rows) -> str:
    # WASM fps/ms: the demo's per-frame cost (decode + RGBA hand-off). The
    # ratio compares like with like: native decode against WASM decode alone
    # when the module separates the hand-off, else against the whole cost.
    # Native/reference: our native decoder's time over the codec's
    # single-threaded reference decoder (FFmpeg, dav1d or avmdec); at or under 1.0x
    # is the goal.
    lines = ["| Codec | Res | Target kbps | Module | WASM fps | WASM ms/f | of which display | Native fps | WASM/native "
             "| Reference ms/f | Native/reference |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in aggregate(decode_rows, ("codec", "res", "kbps_target", "module")):
        display = f"{a['display_ms']:.2f}" if "display_ms" in a else "n/a"
        reference = (f"{a['reference_ms']:.2f} | {a['native_ms'] / a['reference_ms']:.2f}x"
                     if "reference_ms" in a else "n/a | n/a")
        lines.append(f"| {a['codec']} | {a['res']} | {a['kbps_target']} | {a['module']} | {a['wasm_fps']:.0f} | "
                     f"{a['wasm_ms']:.2f} | {display} | {a['native_fps']:.0f} | {a['ratio']:.2f}x | {reference} |")
    if encode_rows:
        lines += ["", "| Codec | Preset | Res | Module | WASM encode fps | WASM ms/f | Native fps | WASM/native time |",
                  "|---|---|---|---|---:|---:|---:|---:|"]
        for a in aggregate(encode_rows, ("codec", "preset", "res", "module")):
            lines.append(f"| {a['codec']} | {a['preset']} | {a['res']} | {a['module']} | {a['wasm_fps']:.1f} | "
                         f"{a['wasm_ms']:.1f} | {a['native_fps']:.1f} | {a['ratio']:.2f}x |")
    return "\n".join(lines)


def write_csv(path: Path, decode_rows, encode_rows) -> None:
    """One line per (direction, stream or clip, module); fps are the best over rounds."""
    fields = ["direction", "codec", "preset", "res", "clip", "kbps_target", "module", "frames",
              "wasm_fps", "wasm_decode_ms", "wasm_display_ms", "native_fps", "reference", "reference_fps"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        for direction, rows in (("decode", decode_rows), ("encode", encode_rows)):
            for r in rows:
                writer.writerow({"direction": direction, "preset": "", **r})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bin", type=Path, required=True, help="native release dir with `nanocodec`")
    parser.add_argument("--stream-bin", type=Path, help="encoder release dir for the streams (default --bin)")
    parser.add_argument("--pkg", type=Path, default=REPO / "web/pkg")
    parser.add_argument("--modules", default="simd128", help=f"comma list of {', '.join(MODULE_DIRS)}")
    parser.add_argument("--work", type=Path, default=DATA / "work/wasm-bench")
    parser.add_argument("--codecs", default=",".join(CODECS))
    parser.add_argument("--res", default=",".join(RESOLUTIONS))
    parser.add_argument("--targets", type=parse_targets, default=TARGETS_KBPS,
                        help="decode CBR targets, e.g. 720p=300+1200,360p=150+300")
    parser.add_argument("--encode-res", default="360p", help="resolutions to time encode at")
    parser.add_argument("--presets", default="default", help="comma list of default, fast")
    parser.add_argument("--repeat", type=int, default=3, help="passes per timed process")
    parser.add_argument("--rounds", type=int, default=1, help="interleaved rounds (best over rounds)")
    parser.add_argument("--cpu", type=int, help="pin timed processes to this core (14 under host-timed.sh)")
    parser.add_argument("--workers", type=int, default=8, help="parallel stream encodes (timing is serial)")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    args.presets = args.presets.split(",")
    args.encode_res = args.encode_res.split(",")

    nanocodec = args.bin / "nanocodec"
    encoder = (args.stream_bin or args.bin) / "nanocodec"
    codecs = args.codecs.split(",")
    modules = {m: (args.pkg / MODULE_DIRS[m]).resolve() for m in args.modules.split(",")}
    clips = [c for c in load_clips() if c.res in args.res.split(",")]
    work = args.work / sha256_file(encoder)[:16]
    work.mkdir(parents=True, exist_ok=True)

    load_start = os.getloadavg()
    decode_rows = encode_rows = []
    if not args.skip_decode:
        jobs = make_streams(encoder, work, codecs, clips, args.targets, args.workers)
        decode_rows = bench_decode(nanocodec, modules, jobs, args)
    if not args.skip_encode:
        encode_rows = bench_encode(nanocodec, modules, work, codecs, clips, args)
    report = {"date": time.strftime("%Y-%m-%d %H:%M"), "load_start": load_start, "load_end": os.getloadavg(),
              "cpu": args.cpu, "rounds": args.rounds, "repeat": args.repeat,
              "native_sha256": sha256_file(nanocodec),
              "wasm_sha256": {m: sha256_file(pkg / "nc_web_bg.wasm") for m, pkg in modules.items()},
              "node": subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip(),
              "decode": decode_rows, "encode": encode_rows}
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    if args.csv:
        write_csv(args.csv, decode_rows, encode_rows)
    print(markdown(decode_rows, encode_rows))
    print(f"\nload {load_start[0]:.1f} -> {report['load_end'][0]:.1f}, node {report['node']}")


if __name__ == "__main__":
    main()
