#!/usr/bin/env python3
"""Extract raw YUV test clips from source videos and write an nceval clip set.

Input: a selection JSON, e.g.

    {
      "name": "smoke",
      "clips": [
        {"name": "hospital_static",
         "source": "/mldata/tracking_original/meva/video/....mp4",
         "start": 30.0, "frames": 60,
         "width": 1280, "height": 720,
         "fps": 30,                      # optional: convert frame rate
         "content_class": "static", "split": "train"}
      ]
    }

Output (under $NC_DATA_ROOT/clips, default ~/nanocodec-data/clips):

    <set>/<clip>.yuv     raw 8-bit I420, scaled with lanczos
    <set>.json           the clip set file read by nceval (nceval/clips.py)

Source bitrate: taken from the inventory (eval/inventory). When a clip is
downscaled or its frame rate lowered, the source bitrate is scaled down
linearly by pixel rate. Linear is deliberately conservative: it makes the
"test at <= 50% of source bitrate" rule stricter, never looser.

Existing outputs whose SHA-256 matches the recorded value are not rebuilt.
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
PARALLEL_EXTRACTIONS = 6
# Frames dropped on each side of a listed cut, and the shortest segment kept.
CUT_MARGIN_FRAMES = 2
DEFAULT_MIN_SEGMENT_SECONDS = 8.0
# ffmpeg scdet score above which a frame change counts as a likely cut.
SCENE_CUT_THRESHOLD = 12
_SCDET_TIME = re.compile(r"lavfi\.scd\.time:\s*([\d.]+)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("selection", type=Path, help="selection JSON")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    for spec in selection["clips"]:  # sources under the home directory are written ~/...
        spec["source"] = os.path.expanduser(spec["source"])
    inventory = {rec["path"]: rec for rec in json.loads(args.inventory.read_text())["clips"]}
    clips_root = args.data_root / "clips"
    out_dir = clips_root / selection["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    clipset_path = clips_root / f"{selection['name']}.json"
    previous = _previous_hashes(clipset_path)
    cut_checked = _cut_checked_hashes(clipset_path)

    specs = [segment for spec in selection["clips"]
             for segment in _split_at_cuts(spec, inventory.get(spec["source"]))]
    for spec in specs:
        source = inventory.get(spec["source"])
        if source is None or source.get("error"):
            print(f"{spec['name']}: source not in inventory (or unreadable): {spec['source']}",
                  file=sys.stderr)
            return 1

    def prepare(spec: dict) -> dict:
        source = inventory[spec["source"]]
        yuv = out_dir / f"{spec['name']}.yuv"
        digest = _sha256_if_exists(yuv)
        if digest is None or digest != previous.get(spec["name"]):
            _extract(spec, source, yuv)
            digest = _sha256_if_exists(yuv)
        if digest not in cut_checked:
            _check_no_scene_cut(spec, yuv, float(spec.get("fps", source["fps"])))
        entry = _clip_entry(spec, source, yuv, clips_root.parent, digest)
        entry["cut_checked"] = True
        print(f"{spec['name']}: {entry['frames']} frames, "
              f"source {entry['source_kbps']:.0f} kbps (at output size)", flush=True)
        return entry

    # ffmpeg decodes and hashing are independent per clip: run several at once.
    with ThreadPoolExecutor(max_workers=PARALLEL_EXTRACTIONS) as pool:
        entries = list(pool.map(prepare, specs))

    clipset_path.write_text(json.dumps({"name": selection["name"], "clips": entries}, indent=2))
    print(f"wrote {clipset_path}")
    return 0


def _extract(spec: dict, source: dict, yuv: Path) -> None:
    filters = [f"scale={spec['width']}:{spec['height']}:flags=lanczos"]
    if "fps" in spec:
        filters.insert(0, f"fps={spec['fps']}")
    start = float(spec.get("start", 0.0))
    seek = ["-ss", str(start)]
    if source.get("container") == "elementary":
        # Raw .264/.265 streams carry no timestamps, so input seeking finds
        # nothing; skip whole frames instead (fps is the inventory's estimate).
        seek = []
        filters.insert(0, f"trim=start_frame={round(start * source['fps'])},"
                          "setpts=PTS-STARTPTS")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y",
           *seek, "-i", spec["source"],
           "-frames:v", str(spec["frames"]), "-vf", ",".join(filters),
           "-pix_fmt", "yuv420p", "-f", "rawvideo", str(yuv)]
    subprocess.run(cmd, check=True)
    frame_bytes = spec["width"] * spec["height"] * 3 // 2
    got = yuv.stat().st_size // frame_bytes
    if got != spec["frames"]:
        raise RuntimeError(f"{spec['name']}: extracted {got} frames, wanted {spec['frames']}")


def _clip_entry(spec: dict, source: dict, yuv: Path, data_root: Path, digest: str) -> dict:
    fps = float(spec.get("fps", source["fps"]))
    pixel_rate_ratio = (spec["width"] * spec["height"] * fps) / (
        source["width"] * source["height"] * source["fps"])
    return {
        "name": spec["name"],
        "yuv": str(yuv.relative_to(data_root)),
        "width": spec["width"], "height": spec["height"], "fps": fps,
        "frames": spec["frames"],
        "source_kbps": round(source["source_kbps"] * min(1.0, pixel_rate_ratio), 1),
        "content_class": spec["content_class"], "split": spec["split"],
        "source": spec["source"], "start": spec.get("start", 0.0),
        "sha256": digest,
    }


def _split_at_cuts(spec: dict, source: dict | None) -> list[dict]:
    """Expand a clip with known scene cuts into cut-free segment clips.

    A selection entry may list "cuts": times in seconds from the clip's start
    (e.g. a compilation switching cameras). Each stretch between cuts becomes
    its own clip "<name>_segN" (N from 1, in order), trimmed by CUT_MARGIN_FRAMES
    on each side of a cut so no transition frame is included. Segments shorter
    than "min_segment_seconds" (default DEFAULT_MIN_SEGMENT_SECONDS) are
    dropped. Every segment still goes through the scene-cut guard, so an
    unlisted cut fails extraction.
    """
    cuts = spec.get("cuts")
    if not cuts or source is None:
        return [spec]
    fps = float(spec.get("fps", source["fps"]))
    duration = spec["frames"] / fps
    bounds = [0.0, *sorted(cuts), duration]
    min_seconds = spec.get("min_segment_seconds", DEFAULT_MIN_SEGMENT_SECONDS)
    segments = []
    for index, (lo, hi) in enumerate(zip(bounds, bounds[1:]), start=1):
        first = round(lo * fps) + (CUT_MARGIN_FRAMES if lo > 0 else 0)
        last = round(hi * fps) - (CUT_MARGIN_FRAMES if hi < duration else 0)
        if (last - first) / fps < min_seconds:
            print(f"{spec['name']}: segment {index} ({lo:.1f}-{hi:.1f}s) shorter than "
                  f"{min_seconds}s, dropped", flush=True)
            continue
        segment = {k: v for k, v in spec.items() if k not in ("cuts", "min_segment_seconds")}
        segment.update({"name": f"{spec['name']}_seg{index}",
                        "start": round(spec.get("start", 0.0) + first / fps, 3),
                        "frames": last - first})
        segments.append(segment)
    return segments


def _check_no_scene_cut(spec: dict, yuv: Path, fps: float) -> None:
    """Fail if the clip contains a likely scene cut (live cameras have none).

    Uses ffmpeg's scdet. A clip with a verified single-camera global change
    (auto-exposure or lighting jump) can opt out with
    "allow_global_change": true plus a "global_change_note" saying why.
    """
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-f", "rawvideo",
           "-s", f"{spec['width']}x{spec['height']}", "-pix_fmt", "yuv420p", "-r", f"{fps:g}",
           "-i", str(yuv), "-vf", f"scdet=threshold={SCENE_CUT_THRESHOLD}", "-f", "null", "-"]
    log = subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
    cuts = [float(t) for t in _SCDET_TIME.findall(log)]
    if not cuts:
        return
    if spec.get("allow_global_change"):
        if not spec.get("global_change_note"):
            raise RuntimeError(f"{spec['name']}: allow_global_change needs a global_change_note")
        print(f"{spec['name']}: global change at {', '.join(f'{t:.1f}s' for t in cuts)} "
              f"allowed ({spec['global_change_note']})", flush=True)
        return
    yuv.unlink()  # never leave a rejected clip where the harness could use it
    raise RuntimeError(
        f"{spec['name']}: likely scene cut(s) at {', '.join(f'{t:.1f}s' for t in cuts)} "
        f"(scdet threshold {SCENE_CUT_THRESHOLD}). Live camera test clips must not contain "
        f"cuts: pick another window or source. If it is a verified single-camera exposure/"
        f"lighting jump, set \"allow_global_change\": true with a \"global_change_note\".")


def _cut_checked_hashes(clipset_path: Path) -> set[str]:
    if not clipset_path.is_file():
        return set()
    return {c.get("sha256") for c in json.loads(clipset_path.read_text())["clips"]
            if c.get("cut_checked")}


def _previous_hashes(clipset_path: Path) -> dict[str, str]:
    if not clipset_path.is_file():
        return {}
    return {c["name"]: c.get("sha256") for c in json.loads(clipset_path.read_text())["clips"]}


def _sha256_if_exists(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
