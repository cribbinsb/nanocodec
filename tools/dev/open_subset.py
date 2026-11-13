#!/usr/bin/env python3
"""The open subset: a clip set anyone can rebuild from this repository alone.

    python3 tools/dev/open_subset.py --out ~/nanocodec-data/clips/open_subset.json

docs/results/reproducibility.md runs the independent test's pipeline on it end
to end. The clips are the browser demo's 720p sample clips (web/clips/, in the
repository) whose licences allow redistribution with attribution or have no
terms at all (web/clips/ATTRIBUTION.md): Blender open movies (CC BY 3.0), SVT
Open Content 2022 (CC BY 4.0), Xiph.org's Vidyo / JCT-VC conferencing
sequences (public domain). Left out: the Netflix (CC BY-NC-ND) and MOTChallenge
(CC BY-NC-SA) clips, and every clip whose source is sealed for milestones
(adv12: CrowdRun, ParkJoy, FourPeople, Johnny, DinnerScene).

Each stored file IS the clip, as for the screening pool (eval/clipset/
build_pool.py): the harness decodes it on the fly, and its own bitrate is its
source bitrate, so the ladder stays within half of it. The demo's clips are
H.264 CRF 22-28 encodes, so this subset tests coding of already-compressed
video at its ladder's low end; it reproduces the pipeline, not the chart.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLIPS_DIR = REPO / "web/clips"
#: (demo clip id, content class (nceval.clips.CONTENT_CLASSES) for the class
#: balance, licence).
OPEN_CLIPS = (("bbb-tree", "pan", "CC BY 3.0"), ("tos-bridge", "pan", "CC BY 3.0"),
              ("tos-canal", "pan", "CC BY 3.0"), ("svt-midnight-sun", "pan", "CC BY 4.0"),
              ("svt-smoke-sauna", "texture", "CC BY 4.0"),
              ("svt-waterfall", "texture", "CC BY 4.0"),
              ("xiph-kristen-sara", "static", "public domain"),
              ("xiph-vidyo1", "static", "public domain"),
              ("xiph-vidyo3", "static", "public domain"),
              ("xiph-vidyo4", "static", "public domain"))
SIZE = "720p"


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets",
                          "-show_entries", "format=duration,size", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream, fmt = data["streams"][0], data["format"]
    num, den = (int(x) for x in stream["avg_frame_rate"].split("/"))
    return {"width": stream["width"], "height": stream["height"], "fps": num / den,
            "frames": int(stream["nb_read_packets"]),
            "kbps": int(fmt["size"]) * 8 / float(fmt["duration"]) / 1000}


def entry(clip_id: str, content_class: str, licence: str) -> dict:
    path = CLIPS_DIR / f"{clip_id}-{SIZE}.mp4"
    info = probe(path)
    return {"name": f"open_{clip_id.replace('-', '_')}", "yuv": str(path),
            "width": info["width"], "height": info["height"], "fps": round(info["fps"], 3),
            "frames": info["frames"], "source_kbps": round(info["kbps"], 1),
            "content_class": content_class, "split": "val", "family": clip_id,
            "licence": licence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    clips = [entry(*spec) for spec in OPEN_CLIPS]
    args.out.write_text(json.dumps({"name": args.out.stem, "clips": clips}, indent=1) + "\n")
    print(f"{len(clips)} clips -> {args.out}")


if __name__ == "__main__":
    main()
