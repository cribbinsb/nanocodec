"""The results site's decoder page data: native and browser decode speed of one release.

    import preview_decoders; preview_decoders.decoder_data(RECORD_DIR)   (preview.py)

Reads one decoder-speed record directory (docs/results/decoder-speed-<date>/), written by
tools/dev/aws_decode_speed.py on each timing machine: `<machine>/decoders-cpu*.json`
(tools/dev/decoder_speed.py: every timed pass of our decoder and of the in-process reference
harnesses), `<machine>/yardstick.json` (tools/web/ffmpeg_wasm_yardstick.py: our browser module
and ffmpeg.wasm under node), the platform notes, and `streams.json` (each stream's size and
the encoder that made it). Returns the page's data (decoders.html, decoders.js): per stream
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
#: Yardstick stream names that are links to a decoder-table stream.
ALIASES = {"axis720.cam.hevc": "cam150.axis-m5074-720p.hevc"}


def geomean(values: list[float]) -> float | None:
    return math.exp(statistics.fmean(math.log(v) for v in values)) if values else None


def timing(seconds: list[float], frames: int) -> dict:
    """Milliseconds per frame: the median pass and the range of passes."""
    ms = sorted(s * 1000 / frames for s in seconds)
    return {"ms": statistics.median(ms), "lo": ms[0], "hi": ms[-1], "passes": len(ms)}


def native_rows(machine_dir: Path, streams: dict) -> list[dict]:
    """One row per stream: ours and each reference's ms per frame."""
    entries: dict[str, dict] = {}
    for path in sorted(machine_dir.glob("decoders-cpu*.json")):
        entries.update(json.loads(path.read_text()))
    rows: dict[str, dict] = {}
    for key, entry in entries.items():
        name, codec, decoder = key.split("|")
        info = streams[name]
        row = rows.setdefault(name, {"stream": name, "codec": codec, "width": info["width"],
                                     "height": info["height"], "encoder": info["encoder"],
                                     "own": info["encoder"] == OWN_ENCODER, "frames": entry["frames"],
                                     "decoders": {}})
        row["decoders"][decoder] = timing(entry["seconds"], entry["frames"])
    return sorted(rows.values(), key=lambda r: (r["codec"], r["width"] * r["height"], r["stream"]))


def mpx_per_s(row: dict, decoder: str) -> float:
    return row["width"] * row["height"] / (row["decoders"][decoder]["ms"] * 1000)


def summaries(rows: list[dict]) -> list[dict]:
    """Per format and reference: the geometric mean time ratio (ours / reference) over all
    streams, our encoder's and other encoders', our plain build's over all streams where it
    was timed, and both decoders' geometric mean Mpx/s."""
    out = []
    for codec in sorted({r["codec"] for r in rows}):
        for ref in REFERENCES:
            both = [r for r in rows if r["codec"] == codec and ref in r["decoders"] and OURS in r["decoders"]]
            if not both:
                continue
            ratio = lambda rs: geomean([r["decoders"][OURS]["ms"] / r["decoders"][ref]["ms"] for r in rs])  # noqa: E731
            own = [r for r in both if r["own"]]
            other = [r for r in both if not r["own"]]
            plain = [r for r in both if OURS_PLAIN in r["decoders"]]
            out.append({"codec": codec, "reference": ref, "streams": len(both),
                        "ratio_plain": geomean([r["decoders"][OURS_PLAIN]["ms"] / r["decoders"][ref]["ms"]
                                                for r in plain]),
                        "own_streams": len(own), "other_streams": len(other),
                        "ratio": ratio(both), "ratio_own": ratio(own), "ratio_other": ratio(other),
                        "ours_mpx_s": geomean([mpx_per_s(r, OURS) for r in both]),
                        "ref_mpx_s": geomean([mpx_per_s(r, ref) for r in both])})
    return out


def browser_rows(machine_dir: Path, streams: dict, native: list[dict]) -> list[dict]:
    """One row per browser stream: ours (decode calls) and ffmpeg.wasm (its decode_video
    calls) in ms per frame, and our native decoder's on the same machine and stream."""
    path = machine_dir / "yardstick.json"
    if not path.is_file():
        return []
    record = json.loads(path.read_text())
    by_stream = {r["stream"]: r for r in native}
    rows: dict[str, dict] = {}
    for key, runs in record["res"].items():
        kind, *rest = key.split("|")
        if kind != "dec":
            continue
        codec, name, decoder = rest
        # the yardstick's name for a camera recording the decoder table times under its own
        source = ALIASES.get(name, name)
        info, frames = streams[source], runs[0]["frames"]
        row = rows.setdefault(name, {"stream": source, "codec": codec, "width": info["width"],
                                     "height": info["height"], "encoder": info["encoder"],
                                     "frames": frames, WASM_OURS: None, WASM_FFMPEG: None,
                                     "native": by_stream[source]["decoders"][OURS] if source in by_stream else None})
        seconds = [r["decode_ms_per_frame"] * frames / 1000 if decoder == WASM_OURS else r["decode_ms"] / 1000
                   for r in runs]
        row[decoder] = timing(seconds, frames)
    return sorted(rows.values(), key=lambda r: (r["codec"], r["width"] * r["height"], r["stream"]))


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
                      "browser": browser_rows(machine_dir, streams, native),
                      "platform": platform(machine_dir)}
    return {"about": meta, "machines": {k: m for k, m in MACHINES.items() if k in views},
            "missing": missing, "references": REFERENCES, "short": SHORT,
            "codec_order": ["h264", "hevc", "vvc", "av1", "vp9", "av2"],
            "record": str(record.relative_to(REPO)) if record.is_relative_to(REPO) else record.name,
            "views": views}
