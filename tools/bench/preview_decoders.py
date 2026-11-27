"""The results site's decoder page data: native and browser decode speed of one release.

    import preview_decoders; preview_decoders.decoder_data(RECORD_DIR)   (preview.py)

Reads one decoder-speed record directory (docs/results/decoder-speed-<date>/), written by
tools/dev/aws_decode_speed.py on each timing machine: `<machine>/decoders-cpu*.json`
(tools/dev/decoder_speed.py: every timed pass of our decoder and of the in-process reference
harnesses), `<machine>/yardstick.json` (tools/web/ffmpeg_wasm_yardstick.py: our browser module
and ffmpeg.wasm under node, the earlier 13-stream browser table), `<machine>/wasm-yardsticks-cpu*.json`
(tools/web/wasm_yardsticks.py: our browser module against WebAssembly builds of the references,
every stream), `<machine>/webcodecs-cpu*.json` and `pc/webcodecs.json`
(tools/web/webcodecs_yardstick.py: our module against the browser's own decoders in Chrome),
the platform notes, and `streams.json` (each stream's size and the encoder that made it). Returns the page's data (decoders.html, decoders.js): per stream
the medians and ranges, and per format and reference the geometric means the page leads with.
Every figure is a median of the record's passes or a geometric mean of those medians; nothing
else is computed here.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RECORD = REPO / "docs" / "results" / "decoder-speed-2026-11-27"
DATA_NAME = "decoders.json"
OURS = "ours"
#: The plain (non-PGO) release build of ours, timed in the same rounds (a secondary figure).
OURS_PLAIN = "ours-plain"
#: The records' reference keys (decoder_speed.REFERENCES), as the page names them, in its order.
REFERENCES = {
    "ffmpeg": "FFmpeg n8.0.1 (libavcodec)",
    "dav1d": "dav1d 1.5.4 (through libavcodec)",
    "vpxdec": "libvpx v1.16.0 (vpxdec's library)",
    "vvdec": "VVdeC d261469",
    "dav2d": "dav2d 237229e (with its assembly)",
    "avmdec": "AVM v1.0.0 (avmdec's library, the AV2 research reference)",
}
SHORT = {"ffmpeg": "FFmpeg", "dav1d": "dav1d", "vpxdec": "libvpx", "vvdec": "VVdeC",
         "dav2d": "dav2d", "avmdec": "AVM"}
#: The timing machines, x86 first (the page opens on it), with how each was timed.
MACHINES = {
    "c8i": {"label": "x86 (Intel Granite Rapids)", "name": "c8i.2xlarge (Intel Granite Rapids)",
            "type": "c8i.2xlarge",
            "anchors": "Each built from its pinned source without -march, keeping its own run-time "
                       "ISA dispatch (AVX2 and AVX-512 kernels where it has them)."},
    "c8g": {"label": "Arm (Graviton 4)", "name": "c8g.2xlarge (Graviton 4)",
            "type": "c8g.2xlarge",
            "anchors": "Each built from its pinned source for generic aarch64 (on the project's "
                       "Jetson, without -march), with its NEON assembly and its own run-time "
                       "dispatch; none was built with Graviton 4 specific options."},
}
#: Our encoder's streams, by the stream's encoder (streams.json); the rest are other encoders'.
OWN_ENCODER = "nanocodec"
WASM_OURS, WASM_FFMPEG = "ours-wasm", "ffmpeg.wasm"
#: The WebAssembly yardsticks (wasm_yardsticks.py), as the page names them, strongest first.
WASM_REFERENCES = {
    "ffmpeg-wasm-simd": "FFmpeg n8.0.1, built by us for WebAssembly (-O3 -msimd128)",
    "dav1d-wasm-simd": "dav1d 1.5.4, built by us for WebAssembly (-O3 -msimd128; through libavcodec)",
    "libvpx-wasm-simd": "libvpx v1.16.0, built by us for WebAssembly (-O3 -msimd128)",
    WASM_FFMPEG: "ffmpeg.wasm 0.12.10, the official build (FFmpeg 5.1.4, its SIMD kernels off)",
}
WASM_SHORT = {"ffmpeg-wasm-simd": "FFmpeg (WASM SIMD)", "dav1d-wasm-simd": "dav1d (WASM SIMD)",
              "libvpx-wasm-simd": "libvpx (WASM SIMD)", WASM_FFMPEG: "ffmpeg.wasm (official, no SIMD)"}
#: Our module in Chrome, and Chrome's own decoders through WebCodecs.
CHROME_OURS = "ours-chrome"
WEBCODECS = {"webcodecs-sw": "Chrome's software decoders (WebCodecs, prefer-software)",
             "webcodecs-hw": "Chrome's hardware decoders (WebCodecs, prefer-hardware)"}
WEBCODECS_SHORT = {"webcodecs-sw": "WebCodecs software", "webcodecs-hw": "WebCodecs hardware"}
#: The PC's record directory (WebCodecs with and without the GPU), and how the page names it.
PC = {"dir": "pc", "name": "the project's PC (AMD Ryzen 9 5950X, NVIDIA RTX 5090)"}


def geomean(values: list[float]) -> float | None:
    return math.exp(statistics.fmean(math.log(v) for v in values)) if values else None


def timing(seconds: list[float], frames: int) -> dict:
    """Milliseconds per frame: the median pass and the range of passes."""
    ms = sorted(s * 1000 / frames for s in seconds)
    return {"ms": statistics.median(ms), "lo": ms[0], "hi": ms[-1], "passes": len(ms)}


def load_entries(paths: list[Path], key: str | None = None) -> dict[str, dict]:
    """The merged {"stream|codec|decoder": entry} records of the files (under `key` if given)."""
    entries: dict[str, dict] = {}
    for path in paths:
        record = json.loads(path.read_text())
        entries.update(record[key] if key else record)
    return entries


def stream_rows(entries: dict[str, dict], streams: dict, ours: str) -> list[dict]:
    """One row per stream: each decoder's ms per frame. Every decoder's time is divided by the
    same frame count (ours' where it ran), so the ms columns' ratios are the time ratios."""
    rows: dict[str, dict] = {}
    for key, entry in entries.items():
        name, codec, decoder = key.split("|")
        info = streams[name]
        row = rows.setdefault(name, {"stream": name, "codec": codec, "width": info["width"],
                                     "height": info["height"], "encoder": info["encoder"],
                                     "own": info["encoder"] == OWN_ENCODER, "frames": entry["frames"],
                                     "seconds": {}, "decoders": {}})
        row["seconds"][decoder] = entry["seconds"]
        if decoder == ours:
            row["frames"] = entry["frames"]
    for row in rows.values():
        row["decoders"] = {d: timing(sec, row["frames"]) for d, sec in row.pop("seconds").items()}
    return sorted(rows.values(), key=lambda r: (r["codec"], r["width"] * r["height"], r["stream"]))


def native_rows(machine_dir: Path, streams: dict) -> list[dict]:
    """One row per stream: ours and each reference's ms per frame."""
    return stream_rows(load_entries(sorted(machine_dir.glob("decoders-cpu*.json"))), streams, OURS)


def mpx_per_s(row: dict, decoder: str) -> float:
    return row["width"] * row["height"] / (row["decoders"][decoder]["ms"] * 1000)


def summaries(rows: list[dict], ours: str = OURS, references=REFERENCES) -> list[dict]:
    """Per format and reference: the geometric mean time ratio (ours / reference) over all
    streams, our encoder's and other encoders', our plain build's over all streams where it
    was timed, and both decoders' geometric mean Mpx/s."""
    out = []
    for codec in sorted({r["codec"] for r in rows}):
        for ref in references:
            both = [r for r in rows if r["codec"] == codec and ref in r["decoders"] and ours in r["decoders"]]
            if not both:
                continue
            ratio = lambda rs: geomean([r["decoders"][ours]["ms"] / r["decoders"][ref]["ms"] for r in rs])  # noqa: E731
            own = [r for r in both if r["own"]]
            other = [r for r in both if not r["own"]]
            plain = [r for r in both if OURS_PLAIN in r["decoders"]]
            out.append({"codec": codec, "reference": ref, "streams": len(both),
                        "ratio_plain": geomean([r["decoders"][OURS_PLAIN]["ms"] / r["decoders"][ref]["ms"]
                                                for r in plain]),
                        "own_streams": len(own), "other_streams": len(other),
                        "ratio": ratio(both), "ratio_own": ratio(own), "ratio_other": ratio(other),
                        "ours_mpx_s": geomean([mpx_per_s(r, ours) for r in both]),
                        "ref_mpx_s": geomean([mpx_per_s(r, ref) for r in both])})
    return out


def wasm_view(machine_dir: Path, streams: dict, native: list[dict]) -> dict | None:
    """Our module against the WebAssembly yardsticks under node, every stream: rows (with our
    native decoder's time beside), per-format summaries, the runs that failed, the build."""
    paths = sorted(machine_dir.glob("wasm-yardsticks-cpu*.json"))
    if not paths:
        return None
    rows = stream_rows(load_entries(paths, "res"), streams, WASM_OURS)
    by_stream = {r["stream"]: r for r in native}
    for row in rows:
        row["native"] = by_stream[row["stream"]]["decoders"][OURS] if row["stream"] in by_stream else None
    return {"rows": rows, "summary": summaries(rows, WASM_OURS, WASM_REFERENCES),
            "errors": load_entries(paths, "errors")}


def webcodecs_view(paths: list[Path], streams: dict) -> dict | None:
    """Our module against Chrome's own decoders, in the same browser: rows, per-format
    summaries, what each mode could not decode, the browser and chrome://gpu's video lines."""
    if not paths:
        return None
    records = [json.loads(p.read_text()) for p in paths]
    rows = stream_rows(load_entries(paths, "res"), streams, CHROME_OURS)
    unsupported = {k: v for r in records for k, v in r["unsupported"].items()}
    # a decode that failed on single streams: counted per format and mode, with its reason
    codec_of = {r["stream"]: r["codec"] for r in rows}
    totals: dict[str, int] = {}
    for row in rows:
        totals[row["codec"]] = totals.get(row["codec"], 0) + 1
    failed: dict[tuple[str, str, str], int] = {}
    for key in [k for k in unsupported if k.split("|")[0] in codec_of]:
        name, decoder = key.split("|")
        group = (codec_of[name], decoder, unsupported.pop(key))
        failed[group] = failed.get(group, 0) + 1
    for (codec, decoder, reason), count in failed.items():
        unsupported[f"{codec}|{decoder}"] = f"failed on {count} of {totals[codec]} streams: {reason}"
    gpu = paths[0].with_suffix(".gpu.txt")
    video = []
    if gpu.is_file():  # chrome://gpu: its video decode status, renderer and VA-API errors
        lines = gpu.read_text().splitlines()
        video = [f"{line} {lines[i + 1]}" for i, line in enumerate(lines[:-1]) if line == "Video Decode:"]
        video += [f"active GPU {line.split('DEVICE=', 1)[-1].split(', DRIVER_VENDOR')[0]}"
                  for line in lines if line.endswith("*ACTIVE*")]
        video += sorted({line.split("] ", 1)[-1].strip(": ") for line in lines if "vaInitialize" in line})
    return {"rows": rows, "summary": summaries(rows, CHROME_OURS, WEBCODECS), "unsupported": unsupported,
            "browser": records[0].get("browser"), "channel": records[0].get("channel"),
            "flags": records[0].get("flags", []), "headed": records[0].get("headed", False), "gpu_video": video}


def platform(machine_dir: Path) -> dict:
    """The CPU model, our build's version line, node and ffmpeg.wasm versions."""
    def first(name: str, prefix: str = "") -> str | None:
        path = machine_dir / name
        if not path.is_file():
            return None
        lines = [line for line in path.read_text().splitlines() if line.startswith(prefix)]
        return lines[0][len(prefix):].strip() if lines else None
    yardstick = machine_dir / "yardstick.json"
    record = json.loads(yardstick.read_text()) if yardstick.is_file() else {}
    return {"cpu": first("lscpu.txt", "Model name:"), "nanocodec": first("nanocodec-version.txt"),
            "pinning": first("pinning.txt"), "node": record.get("node"),
            "ffmpeg_core": record.get("ffmpeg_core"), "date": record.get("date")}


def decoder_data(record: Path = DEFAULT_RECORD) -> dict:
    """The page's data from one record directory; a machine without records is listed as
    missing."""
    meta = json.loads((record / "about.json").read_text())
    streams = json.loads((record / "streams.json").read_text())
    views, missing = {}, []
    for key in MACHINES:
        machine_dir = record / key
        if not any(machine_dir.glob("decoders-cpu*.json")):
            missing.append(key)
            continue
        native = native_rows(machine_dir, streams)
        views[key] = {"native": native, "summary": summaries(native),
                      "wasm": wasm_view(machine_dir, streams, native),
                      "webcodecs": webcodecs_view(sorted(machine_dir.glob("webcodecs-cpu*.json")), streams),
                      "platform": platform(machine_dir)}
    return {"about": meta, "machines": {k: m for k, m in MACHINES.items() if k in views},
            "missing": missing, "references": REFERENCES, "short": SHORT,
            "wasm_references": WASM_REFERENCES, "wasm_short": WASM_SHORT,
            "webcodecs": WEBCODECS, "webcodecs_short": WEBCODECS_SHORT,
            "pc": {**PC, "webcodecs": webcodecs_view(sorted((record / PC["dir"]).glob("webcodecs*.json")), streams)},
            "codec_order": ["h264", "hevc", "vvc", "av1", "vp9", "av2"],
            "record": str(record.relative_to(REPO)) if record.is_relative_to(REPO) else record.name,
            "views": views}
