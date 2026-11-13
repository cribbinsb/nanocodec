#!/usr/bin/env python3
"""Build the rotating clip pool's new clips as high-bitrate encodes (no raw YUV).

    python3 clipset/build_pool.py clipset/pool.json
    # -> ~/nanocodec-data/clips/pool/<clip>.mp4 and ~/nanocodec-data/clips/pool.json

The stored file IS the clip: the harness decodes it on the fly
(nceval.clipframes) and its bitrate is the clip's source bitrate, so the
"test at <= 50% of source bitrate" rule applies as to any camera stream,
which also arrives compressed. The pool (nceval.pool) is these clips plus
the usable raw clips of the existing sets.

Forms, all H.264 (x264 preset medium, closed 1 s key intervals so a
rotated start frame decodes from a key at most one second earlier):

  camera     <= 720p (360p as a second form of pristine sources): CRF 16
             capped at CAMERA_KBPS_720P scaled by pixel count (720p 8 Mbps,
             360p ~2 Mbps): a camera's own encode. Its source kbps is the
             stored rate, or the original's (scaled to the output pixel
             rate) if lower.
  pristine   lossless excerpts only: near-lossless CRF NEAR_LOSSLESS_CRF at
             the excerpt's size. Source kbps = stored rate (no practical cap).

Camera entries cut `seconds` from `start` of a /mldata source (whole 20-60 s
clips, so the pool's daily start-frame rotation has room), at most
`height` lines (default MAX_HEIGHT; a mediocre source is stored downscaled
2-3x, eval/clipset/grade.py); pristine entries use the whole excerpt.
After a build, run clipset/grade.py pool: ungraded clips are not used. Frame rates above MAX_FPS are halved (dropping
frames, never duplicating them). Every output is checked for scene cuts
(ffmpeg scdet, as extract.py): a clip with one is cut short before it, or
dropped if what is left is under MIN_SECONDS. Outputs whose recorded
selection and SHA-256 still match are not rebuilt. Never writes under /mldata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_DATA_ROOT = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
DEFAULT_INVENTORY = DEFAULT_DATA_ROOT / "inventory" / "inventory.json"
PARALLEL_ENCODES = 6
ENCODE_THREADS = 2  # 6 x 2 threads: inside one agent's 8-worker share, plus decode
MAX_HEIGHT = 720
SECOND_FORM_HEIGHT = 360  # pristine sources also get a camera-like 360p form
MAX_FPS = 30.0
CAMERA_KBPS_720P = 8000
CAMERA_CRF = 16
NEAR_LOSSLESS_CRF = 6
PIXELS_720P = 1280 * 720
MIN_SECONDS = 4.0  # ~120 frames at 30 fps: short pristine excerpts with a cut keep their first shot
SCENE_CUT_THRESHOLD = 12  # as extract.py
CUT_MARGIN_SECONDS = 0.2
_SCDET_TIME = re.compile(r"lavfi\.scd\.time:\s*([\d.]+)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    inventory = {r["path"]: r for r in json.loads(args.inventory.read_text())["clips"]}
    clips_root = args.data_root / "clips"
    out_dir = clips_root / selection["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    set_path = clips_root / f"{selection['name']}.json"
    previous = ({c["name"]: c for c in json.loads(set_path.read_text())["clips"]}
                if set_path.exists() else {})
    jobs = [job for spec in selection["clips"] for job in forms(spec)]

    def build(job: dict) -> dict | None:
        out = out_dir / f"{job['name']}.mp4"
        old = previous.get(job["name"])
        if old and out.exists() and old.get("job") == job and old.get("sha256") == sha256(out):
            return old
        return encode(job, out, inventory, clips_root.parent)

    with ThreadPoolExecutor(PARALLEL_ENCODES) as pool:
        entries = [e for e in pool.map(build, jobs) if e is not None]
    set_path.write_text(json.dumps({"name": selection["name"], "clips": entries}, indent=1) + "\n")
    size = sum((clips_root.parent / e["yuv"]).stat().st_size for e in entries)
    print(f"wrote {set_path}: {len(entries)} clips, {size / 2**30:.2f} GiB")
    return 0


def forms(spec: dict) -> list[dict]:
    """The stored forms of one selection entry."""
    spec = {**spec, "source": os.path.expanduser(spec["source"])}
    if not spec.get("pristine"):
        return [{**spec, "form": "camera", "max_height": spec.get("height", MAX_HEIGHT)}]
    return [{**spec, "form": "camera", "max_height": MAX_HEIGHT},
            {**spec, "name": f"{spec['name']}_360p", "form": "camera",
             "max_height": SECOND_FORM_HEIGHT},
            {**spec, "name": f"{spec['name']}_pristine", "form": "pristine",
             "max_height": MAX_HEIGHT}]


def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    num, den = (int(x) for x in stream["r_frame_rate"].split("/"))
    return {"width": stream["width"], "height": stream["height"], "fps": num / den}


def output_size(width: int, height: int, max_height: int) -> tuple[int, int]:
    """Never upscale; keep the aspect ratio; even dimensions."""
    out_h = min(height, max_height)
    out_w = round(width * out_h / height / 2) * 2
    return out_w, out_h - out_h % 2


def encode(job: dict, out: Path, inventory: dict, data_root: Path,
           seconds: float | None = None) -> dict | None:
    source = probe(job["source"])
    width, height = output_size(source["width"], source["height"], job["max_height"])
    fps = source["fps"] / 2 if source["fps"] > MAX_FPS else source["fps"]
    seconds = seconds or job.get("seconds")
    keyint = round(fps)
    filters = ([f"fps={fps:g}"] if fps != source["fps"] else []) + \
        [f"scale={width}:{height}:flags=lanczos"]
    if job["form"] == "camera":
        cap = round(CAMERA_KBPS_720P * width * height / PIXELS_720P)
        rate = ["-crf", str(CAMERA_CRF), "-maxrate", f"{cap}k", "-bufsize", f"{cap}k"]
    else:
        rate = ["-crf", str(NEAR_LOSSLESS_CRF)]
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y",
           *(["-ss", str(job["start"])] if job.get("start") else []), "-i", job["source"],
           *(["-t", str(seconds)] if seconds else []), "-an", "-vf", ",".join(filters),
           "-c:v", "libx264", "-threads", str(ENCODE_THREADS), "-preset", "medium", *rate,
           "-g", str(keyint),
           "-keyint_min", str(keyint), "-sc_threshold", "0", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    frames = count_frames(out)
    if frames < MIN_SECONDS * fps:
        out.unlink()
        print(f"{job['name']}: only {frames} frames from {job.get('start', 0)}s, dropped",
              flush=True)
        return None
    cuts = scene_cuts(out)
    if cuts:
        keep = cuts[0] - CUT_MARGIN_SECONDS
        out.unlink()
        if keep < MIN_SECONDS:
            print(f"{job['name']}: scene cut at {cuts[0]:.1f}s, dropped", flush=True)
            return None
        print(f"{job['name']}: scene cut at {cuts[0]:.1f}s, cut to {keep:.1f}s", flush=True)
        return encode(job, out, inventory, data_root, keep)
    stored_kbps = out.stat().st_size * 8 / (frames / fps) / 1000
    source_kbps = stored_kbps
    original = inventory.get(job["source"])
    if job["form"] == "camera" and original and original.get("source_kbps"):
        ratio = min(1.0, width * height * fps /
                    (original["width"] * original["height"] * original["fps"]))
        source_kbps = min(stored_kbps, original["source_kbps"] * ratio)
    entry = {"name": job["name"], "yuv": str(out.relative_to(data_root)),
             "width": width, "height": height, "fps": round(fps, 3), "frames": frames,
             "source_kbps": round(source_kbps, 1), "stored_kbps": round(stored_kbps, 1),
             "content_class": job["content_class"], "split": "train",
             "family": job.get("family") or job["source"], "form": job["form"],
             "source": job["source"], "start": job.get("start", 0.0),
             "sha256": sha256(out), "cut_checked": True, "job": job}
    print(f"{job['name']}: {width}x{height} {frames} frames, {stored_kbps:.0f} kbps stored, "
          f"source {source_kbps:.0f} kbps", flush=True)
    return entry


def count_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return int(out.strip() or 0)


def scene_cuts(path: Path) -> list[float]:
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-vf",
                          f"scdet=threshold={SCENE_CUT_THRESHOLD}", "-f", "null", "-"],
                         capture_output=True, text=True, check=True).stderr
    return [float(t) for t in _SCDET_TIME.findall(log)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
