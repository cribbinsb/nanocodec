#!/usr/bin/env python3
"""Fetch the JCT-VC HEVC_v1 conformance set and build the Main-profile manifest.

The bitstreams (ITU-T H.265.1, draft_conformance/HEVC_v1) are untrusted data:
each zip is extracted into its own directory under
``~/nanocodec-data/conformance/hevc/streams/<name>/`` and nothing in them is
executed. Streams are selected by the profile in their SPS (Main and Main Still
Picture, 8-bit 4:2:0); Main 10, RExt, SCC, MV-HEVC and SHVC are excluded.

The expected output of each stream is the MD5 of HM's (TAppDecoder) cropped
8-bit output in output order. HM's output is written to a scratch file, hashed
and deleted at once. The manifest records that MD5 and, where the zip ships a
32-hex yuv MD5, whether the two agree.

Usage: hevc_conformance_fetch.py [--data DIR] [--hm PATH] [--manifest OUT]
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www.itu.int"
INDEX = BASE + "/wftp3/av-arch/jctvc-site/bitstream_exchange/draft_conformance/HEVC_v1/"
# Profiles a Main decoder must decode (H.265 A.3.2, A.3.5).
MAIN_PROFILES = {"Main", "Main Still Picture"}
BITSTREAM_SUFFIXES = (".bit", ".bin")
CHUNK = 1 << 20


def download(data: Path) -> list[Path]:
    zips = data / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(INDEX, timeout=60) as response:
        index = response.read().decode("latin-1")
    paths = []
    for href in re.findall(r'HREF="([^"]+\.zip)"', index, re.IGNORECASE):
        target = zips / Path(href).name
        if not target.exists():
            with urllib.request.urlopen(BASE + href, timeout=300) as response:
                target.write_bytes(response.read())
        paths.append(target)
    return paths


def extract(archive: Path, streams: Path) -> Path:
    target = streams / archive.stem
    if not target.exists():
        target.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                resolved = (target / member.filename).resolve()
                if not str(resolved).startswith(str(target.resolve()) + os.sep):
                    raise ValueError(f"{archive}: unsafe member {member.filename}")
            z.extractall(target)
    return target


def bitstream(directory: Path) -> Path | None:
    found = [
        p
        for p in directory.rglob("*")
        if p.suffix.lower() in BITSTREAM_SUFFIXES and "__MACOSX" not in p.parts
    ]
    return found[0] if len(found) == 1 else None


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=profile,pix_fmt,width,height,level", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    streams = json.loads(out.stdout or "{}").get("streams", [])
    return streams[0] if streams else {}


def hm_md5(hm: str, path: Path, scratch: Path) -> tuple[str, int]:
    out = scratch / (path.stem + ".yuv")
    try:
        subprocess.run([hm, "-b", str(path), "-o", str(out), "-d", "8"],
                       capture_output=True, check=True, timeout=1800)
        digest = hashlib.md5()
        size = 0
        with out.open("rb") as f:
            while chunk := f.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size
    finally:
        out.unlink(missing_ok=True)


def shipped_md5s(directory: Path) -> set[str]:
    found = set()
    for p in directory.rglob("*"):
        if p.is_file() and ("md5" in p.name.lower()) and p.stat().st_size < 4096:
            found.update(re.findall(r"\b[0-9a-fA-F]{32}\b", p.read_text("latin-1")))
    return {m.lower() for m in found}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    home = Path.home()
    parser.add_argument("--data", type=Path, default=home / "nanocodec-data/conformance/hevc")
    parser.add_argument("--hm", default=str(home / "nanocodec-data/anchors/x86_64/bin/TAppDecoder"))
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "ci/hevc_conformance_main.json")
    args = parser.parse_args()
    streams = args.data / "streams"
    scratch = home / "nanocodec-data/work/hevc-conformance"
    scratch.mkdir(parents=True, exist_ok=True)
    entries = []
    for archive in sorted(download(args.data)):
        directory = extract(archive, streams)
        path = bitstream(directory)
        if path is None:
            print(f"skip {archive.stem}: no single bitstream", file=sys.stderr)
            continue
        info = probe(path)
        if info.get("profile") not in MAIN_PROFILES or info.get("pix_fmt") != "yuv420p":
            print(f"skip {archive.stem}: {info.get('profile')} {info.get('pix_fmt')}", file=sys.stderr)
            continue
        md5, size = hm_md5(args.hm, path, scratch)
        frame = int(info["width"]) * int(info["height"]) * 3 // 2
        shipped = shipped_md5s(directory)
        entries.append({
            "name": archive.stem,
            "bitstream": str(path.relative_to(streams)),
            "md5": md5,
            "frames": size // frame if frame else 0,
            "width": int(info["width"]),
            "height": int(info["height"]),
            "profile": info["profile"],
            "level_idc": int(info.get("level", 0)),
            "shipped_md5_agrees": (md5 in shipped) if shipped else None,
        })
        print(f"{archive.stem}: {md5}", file=sys.stderr)
    entries.sort(key=lambda e: e["name"].lower())
    args.manifest.write_text(json.dumps(entries, indent=1) + "\n")
    print(f"{len(entries)} Main-profile streams -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
