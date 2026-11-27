#!/usr/bin/env python3
"""Decode a stored clip set into BP-3's raw benchmark clips (`<name>.yuv.zst`, the first
BENCH_FRAMES frames as 8-bit I420), hashing each; for VC-1K also write the committed manifest.

    python3 eval/clipset/bench_raw.py vc1k --out ~/nanocodec-data/clips/bench_raw/vc1k
    NC_RELEASE_MEASURE=1 python3 eval/clipset/bench_raw.py heldout \\
        --out ~/nanocodec-data/clips/bench_raw/heldout_2026-11-23b

`tools/bench/fleet.py stage --clips DIR` reads these files (docs/benchmark-procedure.md
section 3). The frames come from the decode the scorer uses (`nceval.clipframes.decode_command`:
the pinned ffmpeg, yuv420p, from the stored file's first frame).

  - vc1k: the clips of VC1K_CLIPSET (`clips/vc1k_2026-11-18.json`, as it is); writes
    VC1K_MANIFEST with each clip's `decoded_sha256`.
  - heldout: the clips of HELDOUT_MANIFEST (sealed, release-measure only); each file's
    SHA-256 must equal the manifest's `decoded_sha256`, else the run is refused. Nothing
    else is read from these clips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))

from nceval.clipframes import CLIP_DECODER_VERSION, clip_decoder  # noqa: E402

DATA = Path.home() / "nanocodec-data"
#: The benchmark's frames per clip (BP section 3).
BENCH_FRAMES = 150
VC1K_CLIPSET = DATA / "clips" / "vc1k_2026-11-18.json"
VC1K_MANIFEST = REPO / "eval" / "clipset" / "vc1k_2026-11-18.manifest.json"
HELDOUT_MANIFEST = REPO / "eval" / "clipset" / "heldout_2026-11-23b.manifest.json"
VC1K_LICENCE = "VC-1K (github.com/facebookresearch/VC-1K): CC BY-NC 4.0; numbers and names only"
#: The clip-set fields the manifest keeps (the rest is the selection's bookkeeping).
VC1K_FIELDS = ("name", "source", "start", "width", "height", "fps", "content_class", "family",
               "stratum", "source_kbps", "stored_kbps", "sha256")
ZSTD_LEVEL = "-3"


def decode(stored: Path, width: int, height: int, out: Path) -> str:
    """Store the first BENCH_FRAMES frames as `out` (zstd) and return their SHA-256."""
    frame_bytes = width * height * 3 // 2
    argv = [str(clip_decoder()), "-nostdin", "-v", "error", "-threads", "1", "-i", str(stored),
            "-frames:v", str(BENCH_FRAMES), "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]
    source = subprocess.Popen(argv, stdout=subprocess.PIPE)
    store = subprocess.Popen(["zstd", "-q", "-f", ZSTD_LEVEL, "-o", str(out)],
                             stdin=subprocess.PIPE)
    digest = hashlib.sha256()
    for _ in range(BENCH_FRAMES):
        frame = source.stdout.read(frame_bytes)
        if len(frame) != frame_bytes:
            raise SystemExit(f"refused: {stored} decoded to fewer than {BENCH_FRAMES} frames")
        digest.update(frame)
        store.stdin.write(frame)
    store.stdin.close()
    if source.wait() or store.wait():
        raise SystemExit(f"refused: decoding {stored} failed")
    return digest.hexdigest()


def vc1k(out: Path, jobs: int) -> None:
    clipset = json.loads(VC1K_CLIPSET.read_text())

    def one(clip: dict) -> dict:
        row = {k: clip[k] for k in VC1K_FIELDS}
        row["stored"] = clip["yuv"]
        row["frames"] = BENCH_FRAMES
        row["licence"] = VC1K_LICENCE
        row["decoded_sha256"] = decode(DATA / clip["yuv"], clip["width"], clip["height"],
                                       out / f"{clip['name']}.yuv.zst")
        return row

    with ThreadPoolExecutor(jobs) as pool:
        clips = list(pool.map(one, clipset["clips"]))
    manifest = {
        "name": clipset["name"],
        "_comment": ("The committed manifest of VC-1K as it is (BP-3 section 3), from "
                     "clips/vc1k_2026-11-18.json by eval/clipset/bench_raw.py. 'decoded_sha256' "
                     f"is the SHA-256 of the first {BENCH_FRAMES} frames decoded by ffmpeg "
                     f"{CLIP_DECODER_VERSION} as yuv420p from 'stored' (relative to the data "
                     "root); 'sha256' is the stored file's. Sealed (nceval.pool SEALED_SETS)."),
        "seed": clipset["seed"], "strata": clipset["strata"],
        "clips": clips}
    VC1K_MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"vc1k: {len(clips)} clips -> {out}, manifest {VC1K_MANIFEST}")


def heldout(out: Path, jobs: int) -> None:
    clips = json.loads(HELDOUT_MANIFEST.read_text())["clips"]

    def one(clip: dict) -> str | None:
        got = decode(DATA / clip["stored"], clip["width"], clip["height"],
                     out / f"{clip['name']}.yuv.zst")
        return None if got == clip["decoded_sha256"] else clip["name"]

    with ThreadPoolExecutor(jobs) as pool:
        bad = [name for name in pool.map(one, clips) if name]
    if bad:
        raise SystemExit(f"refused: {len(bad)} clips differ from decoded_sha256: {bad}")
    print(f"heldout: {len(clips)} clips -> {out}, every decoded_sha256 matches")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("set", choices=("vc1k", "heldout"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.set == "heldout" and os.environ.get("NC_RELEASE_MEASURE") != "1":
        raise SystemExit("refused: heldout_2026-11-23b is release-measure only "
                         "(set NC_RELEASE_MEASURE=1 on this command line)")
    a.out.mkdir(parents=True, exist_ok=True)
    (vc1k if a.set == "vc1k" else heldout)(a.out, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
