#!/usr/bin/env python3
"""Build BP-3's AOM-CTC dataset: verify the downloaded CTC sources, convert each to the
benchmark's 130-frame I420 8-bit clip, check the 10-bit conversion, write the manifest.

    python3 -I eval/clipset/aomctc_bp3.py --src ~/nanocodec-data/clips/aomctc_bp3/src \\
        --out ~/nanocodec-data/clips/aomctc_bp3 \\
        --manifest eval/clipset/aomctc_bp3.manifest.json [--jobs 4]

The dataset (docs/benchmark-procedure.md section 3, row "AOM-CTC"):
  - the 12 low-delay sequences of classes A3, A4 and A5 (`nceval.ctc.sequences.AOM_LD`, the
    standards lens's set), each checked against the CTC's MD5;
  - class A2: every .y4m listed at A2_INDEX on 2026-11-23 (22), each checked against the MD5
    of the folder's own `a2_2k.md5`. A2 is reported separately (review 3 C2).

Each clip is the first CLIP_FRAMES frames decoded by the pinned ffmpeg
(`nceval.clipframes.CLIP_DECODER_VERSION`) with `-pix_fmt yuv420p -sws_dither none`, stored as
`<name>.yuv.zst` (what `tools/bench/fleet.py stage --clips` reads), and hashed
(`decoded_sha256`).

The 10-bit check (BP section 3): for TENBIT_CHECK, the 8-bit frames are also computed
independently from the .y4m's 10-bit samples as min((x + 2) >> 2, 255) and compared with
ffmpeg's output frame by frame; the manifest records the result.

`source_kbps` is the uncompressed rate of the clip the encoders receive (8-bit I420), so the
"no more than half the source bitrate" rule never removes a rung of a raw source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))

from nceval.clipframes import CLIP_DECODER_CANDIDATES, CLIP_DECODER_VERSION  # noqa: E402
from nceval.ctc.sequences import AOM_LD  # noqa: E402

#: The CTC's frame count for the low-delay configuration (BP section 3).
CLIP_FRAMES = 130
AOM_BASE = "https://media.xiph.org/video/aomctc/test_set"
A2_INDEX = f"{AOM_BASE}/a2_2k/"
A2_MD5_FILE = "a2_2k.md5"
#: The 10-bit sequence whose conversion is checked against (x + 2) >> 2 (the first 10-bit
#: A2 sequence by name).
TENBIT_CHECK = "Aerial3200_1920x1080_5994_10bit_420"
CONVERSION = "ffmpeg {version} -pix_fmt yuv420p -sws_dither none (first {frames} frames)"
ZSTD_LEVEL = "-3"
CHUNK = 1 << 20

_NETFLIX = ("noncommercial", "Netflix 2020: CC BY-NC-ND 4.0")
_SVT_2006 = ("standards-only", "SVT HD Multi Format Test Set 2006: \"may only be used for the "
             "purpose of developing, testing and presenting technology standards\"")
_TUM = ("free", "TU München / Taurus Media 2001: \"Restrictions of use: No restrictions\"")
_NTIA = ("research", "NTIA/ITS: research use only, acknowledgement kept")


def _cc_by_sa(author: str) -> tuple[str, str]:
    return ("free", f"CC BY-SA 4.0, (c) {author}")


#: A2 licences, read from each sequence's notice in the A2 folder on 2026-11-23.
A2_LICENCES: dict[str, tuple[str, str]] = {
    "Aerial3200_1920x1080_5994_10bit_420": _NETFLIX,
    "Boat_1920x1080_5994_10bit_420": _NETFLIX,
    "CrowdRun_1920x1080p50": _SVT_2006,
    "DinnerSceneCropped_1920x1080_2997fps_10bit_420": _NETFLIX,
    "FoodMarket_1920x1080_5994_10bit_420": _NETFLIX,
    "MeridianTalk_sdr_1920x1080p_5994_10bit": _NETFLIX,
    "Motorcycle_1920x1080_30fps_8bit": _cc_by_sa("Neil Birkbeck"),
    "MountainBike_1920x1080_30fps_8bit": _cc_by_sa("Neil Birkbeck"),
    "OldTownCross_1920x1080p50": _SVT_2006,
    "PedestrianArea_1920x1080p25": _TUM,
    "RitualDance_1920x1080_5994_10bit_420": _NETFLIX,
    "Riverbed_1920x1080p25": _TUM,
    "RushFieldCuts_1920x1080_2997": _NTIA,
    "Skater227_1920x1080_30fps": _cc_by_sa("James Leslie Skinner"),
    "ToddlerFountain_1920x1080_2997fps_10bit_420": _NETFLIX,
    "TreesAndGrass_1920_1080_30fps_8bit": _cc_by_sa("Hangyu Kuang"),
    "TunnelFlag_1920x1080_5994_10bit_420": _NETFLIX,
    "Vertical_Carnaby_1080x1920_5994": _cc_by_sa("James Leslie Skinner"),
    "Vertical_bees_1080x1920_2997": _cc_by_sa("James Leslie Skinner"),
    "WalkingInStreet_1920x1080_30fps": _cc_by_sa("Victor Cherepanov"),
    "WorldCup_1920x1080_30p": _cc_by_sa("James Leslie Skinner"),
    "WorldCup_far_1920x1080_30p": _cc_by_sa("James Leslie Skinner"),
}
_NOTICE_SUFFIX = {"CrowdRun_1920x1080p50": "_Copyright copy.pdf",
                  "OldTownCross_1920x1080p50": "_Copyright.pdf"}


def ffmpeg() -> str:
    for candidate in CLIP_DECODER_CANDIDATES:
        out = subprocess.run([str(candidate), "-version"], capture_output=True, text=True)
        if out.returncode == 0 and f"version {CLIP_DECODER_VERSION}" in out.stdout.split("\n")[0]:
            return str(candidate)
    raise SystemExit(f"refused: no ffmpeg {CLIP_DECODER_VERSION} in {CLIP_DECODER_CANDIDATES}")


def file_digests(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def y4m_header(path: Path) -> dict:
    """The stream header's size, rate and bit depth (YUV4MPEG2 W H F C tags)."""
    with path.open("rb") as f:
        tags = f.readline().decode("ascii").split()
    if tags[0] != "YUV4MPEG2":
        raise SystemExit(f"refused: {path} is not a y4m file")
    fields = {t[0]: t[1:] for t in tags[1:]}
    colour = fields.get("C", "420")
    if not colour.startswith("420"):
        raise SystemExit(f"refused: {path} is {colour}, not 4:2:0")
    depth = int(colour.removeprefix("420p")) if colour.startswith("420p") else 8
    num, den = fields["F"].split(":")
    return {"width": int(fields["W"]), "height": int(fields["H"]), "fps": f"{num}/{den}",
            "bit_depth": depth}


def y4m_frames(path: Path, header: dict, count: int):
    """The first `count` frames as uint16 (10-bit) or uint8 arrays, read independently of ffmpeg."""
    w, h = header["width"], header["height"]
    sample = np.dtype("<u2") if header["bit_depth"] > 8 else np.dtype("u1")
    size = w * h * 3 // 2
    with path.open("rb") as f:
        f.readline()
        for _ in range(count):
            marker = f.readline()
            if not marker.startswith(b"FRAME"):
                raise SystemExit(f"refused: {path} ends before {count} frames")
            yield np.frombuffer(f.read(size * sample.itemsize), dtype=sample)


def convert(source: Path, out: Path, frame_bytes: int, keep: bool) -> tuple[str, list[bytes]]:
    """Decode the first CLIP_FRAMES frames to 8-bit I420, store them zstd-compressed and hash
    them; returns the hash and, when `keep`, the frames (for the 10-bit check only: 130
    1080p frames are 400 MB)."""
    decode = subprocess.Popen(
        [ffmpeg(), "-v", "error", "-nostdin", "-i", str(source), "-frames:v", str(CLIP_FRAMES),
         "-pix_fmt", "yuv420p", "-sws_dither", "none", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE)
    store = subprocess.Popen(["zstd", "-q", "-f", ZSTD_LEVEL, "-o", str(out)],
                             stdin=subprocess.PIPE)
    whole, frames = hashlib.sha256(), []
    for _ in range(CLIP_FRAMES):
        frame = decode.stdout.read(frame_bytes)
        if len(frame) != frame_bytes:
            raise SystemExit(f"refused: {source} decoded to fewer than {CLIP_FRAMES} frames")
        whole.update(frame)
        if keep:
            frames.append(frame)
        store.stdin.write(frame)
    if decode.stdout.read(1):
        raise SystemExit(f"refused: ffmpeg gave more than {CLIP_FRAMES} frames for {source}")
    store.stdin.close()
    if decode.wait() or store.wait():
        raise SystemExit(f"refused: converting {source} failed")
    return whole.hexdigest(), frames


def tenbit_check(source: Path, header: dict, converted: list[bytes]) -> dict:
    """Compare ffmpeg's 8-bit frames with min((x + 2) >> 2, 255) of the 10-bit samples."""
    differing, worst, total = [], 0, 0
    for index, samples in enumerate(y4m_frames(source, header, CLIP_FRAMES)):
        expected = np.minimum((samples.astype(np.int32) + 2) >> 2, 255)
        got = np.frombuffer(converted[index], dtype=np.uint8).astype(np.int32)
        diff = np.abs(got - expected)
        count = int(np.count_nonzero(diff))
        if count:
            differing.append({"frame": index, "samples": count, "max_abs": int(diff.max())})
            worst = max(worst, int(diff.max()))
            total += count
    return {"sequence": source.stem, "rule": "min((x + 2) >> 2, 255) on the y4m's 10-bit samples",
            "frames": CLIP_FRAMES, "match": not differing,
            "frames_differing": len(differing), "samples_differing": total, "max_abs_diff": worst,
            "per_frame": differing}


def sequences(src: Path) -> list[dict]:
    """Every expected sequence with its URL, class, licence and published MD5."""
    rows = [{"name": s.name, "content_class": s.sequence_class, "url": s.url,
             "published_md5": s.md5, "licence_tier": s.licence.tier.value,
             "licence": s.licence.summary, "notice_url": s.licence.notice_url, "a2": False}
            for s in AOM_LD]
    published = dict(reversed(line.split()) for line in (src / A2_MD5_FILE).read_text().splitlines()
                     if line.strip())
    for name, (tier, summary) in sorted(A2_LICENCES.items()):
        notice = _NOTICE_SUFFIX.get(name, "_Copyright.txt")
        rows.append({"name": name, "content_class": "A2", "url": f"{A2_INDEX}{name}.y4m",
                     "published_md5": published[f"{name}.y4m"], "licence_tier": tier,
                     "licence": summary, "notice_url": f"{A2_INDEX}{name}{notice}", "a2": True})
    if len(published) != len(A2_LICENCES):
        raise SystemExit(f"refused: {A2_MD5_FILE} lists {len(published)} sequences, "
                         f"expected {len(A2_LICENCES)}")
    return rows


def build_one(row: dict, src: Path, out: Path) -> dict:
    source = src / f"{row['name']}.y4m"
    md5, sha = file_digests(source)
    if md5 != row["published_md5"]:
        raise SystemExit(f"refused: {source} MD5 {md5} != published {row['published_md5']}")
    header = y4m_header(source)
    frame_bytes = header["width"] * header["height"] * 3 // 2
    decoded, frames = convert(source, out / f"{row['name']}.yuv.zst", frame_bytes,
                              keep=row["name"] == TENBIT_CHECK)
    num, den = map(int, header["fps"].split("/"))
    clip = {"name": row["name"], "width": header["width"], "height": header["height"],
            "fps": header["fps"], "frames": CLIP_FRAMES, "start": 0,
            "source_kbps": round(frame_bytes * 8 * num / den / 1000, 1),
            "content_class": row["content_class"], "a2": row["a2"],
            "source_bit_depth": header["bit_depth"],
            "conversion": (CONVERSION.format(version=CLIP_DECODER_VERSION, frames=CLIP_FRAMES)
                           if header["bit_depth"] > 8 else
                           f"none: 8-bit source, first {CLIP_FRAMES} frames as stored"),
            "url": row["url"], "md5": md5, "sha256": sha,
            "licence_tier": row["licence_tier"], "licence": row["licence"],
            "notice_url": row["notice_url"],
            "stored": f"clips/aomctc_bp3/{row['name']}.yuv.zst", "decoded_sha256": decoded}
    if row["name"] == TENBIT_CHECK:
        clip["tenbit_check"] = tenbit_check(source, header, frames)
    print(f"{row['name']}: {header['width']}x{header['height']} {header['fps']} "
          f"{header['bit_depth']}-bit ok", flush=True)
    return clip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True, help="the downloaded .y4m files")
    ap.add_argument("--out", type=Path, required=True, help="where <name>.yuv.zst go")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    rows = sequences(a.src)
    with ThreadPoolExecutor(a.jobs) as pool:
        clips = list(pool.map(lambda row: build_one(row, a.src, a.out), rows))
    check = next(c.pop("tenbit_check") for c in clips if "tenbit_check" in c)
    manifest = {
        "name": "aomctc_bp3",
        "_comment": ("BP-3 AOM-CTC dataset (docs/benchmark-procedure.md section 3), built by "
                     "eval/clipset/aomctc_bp3.py. 'md5'/'sha256' are the downloaded .y4m's (MD5 "
                     "checked against the CTC's); 'decoded_sha256' is the SHA-256 of the first "
                     "130 frames as 8-bit I420 ('stored', relative to the data root). "
                     "'source_kbps' is the uncompressed 8-bit I420 rate. A2 ('a2': true) is "
                     "reported separately. Not comparable with published CTC numbers."),
        "a2_index": A2_INDEX, "a2_listed": "2026-11-23",
        "decoder": f"ffmpeg {CLIP_DECODER_VERSION}",
        "tenbit_check": check,
        "clips": clips}
    a.manifest.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(clips)} clips; 10-bit check {'match' if check['match'] else 'DIFFERS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
