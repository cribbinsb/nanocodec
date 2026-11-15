#!/usr/bin/env python3
"""Fetch an official conformance suite into one directory per stream, for
tools/dev/conformance_manifest.py and tools/ci/conformance.py.

Two kinds of suite:
- zip indexes (JCT-VC, JVT): each zip listed on the suite's index pages and
  not excluded by its name pattern is extracted (member paths checked) into
  ``<data>/streams/<zip name>/``;
- a listed set (libaom, libvpx test data): each file named in a list with
  its SHA-1 and its per-frame ``.md5`` is downloaded from the mirror,
  checked against both SHA-1s and kept as ``<data>/streams/<name>/``;
  Matroska/WebM vectors are rewrapped as IVF (tools/conformance/webm_to_ivf.py).

The bitstreams are untrusted data: nothing in them is executed. Files
already fetched or extracted are kept.

Usage: conformance_fetch.py --suite hevc-v1|h264|av1-libaom [--data DIR]
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
ROOT = Path(__file__).resolve().parents[2]
ITU = "https://www.itu.int"
AOM_TEST_DATA = "https://storage.googleapis.com/aom-test-data/"
#: Container rewrap of Matroska/WebM vectors: suffix -> IVF fourcc.
REWRAP = {".mkv": "AV01", ".webm": "VP90"}

JVT = ITU + "/wftp3/av-arch/jvt-site/draft_conformance/"

# suite -> ("zips", index pages listing the zips, zip names to skip (regex or
#           None), data directory) or
#          ("listed", mirror, list of "sha1 md5-sha1 name" lines, data directory).
SUITES = {
    "hevc-v1": ("zips", (ITU + "/wftp3/av-arch/jctvc-site/bitstream_exchange/draft_conformance/HEVC_v1/",),
                None, DATA / "conformance/hevc"),
    # ITU-T H.264.1: AVCv1 (Baseline, Extended, Main), FRExt (High) and the
    # High 10 streams of the professional profiles. 4:2:2 and 4:4:4 (High
    # 4:2:2 / 4:4:4 profiles) are not fetched; nor are MVC and SVC.
    "h264": ("zips", (JVT + "AVCv1/", JVT + "FRExt/", JVT + "Professional_profiles/"),
             r"^(Hi422|PPH422|PPH444|PPCV444)", DATA / "conformance/h264"),
    "av1-libaom": ("listed", AOM_TEST_DATA, ROOT / "tools/conformance/av1_manifest.txt",
                   DATA / "conformance/av1"),
}


def download(index_urls: tuple[str, ...], skip: str | None, data: Path) -> list[Path]:
    zips = data / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    hrefs = []
    for index_url in index_urls:
        with urllib.request.urlopen(index_url, timeout=60) as response:
            index = response.read().decode("latin-1")
        hrefs += re.findall(r'HREF="([^"]+\.zip)"', index, re.IGNORECASE)
    paths = []
    for href in hrefs:
        if skip and re.search(skip, Path(href).name):
            continue
        target = zips / Path(href).name
        if not target.exists():
            url = href if href.startswith("http") else ITU + href
            with urllib.request.urlopen(url, timeout=300) as response:
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


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def fetch_listed(mirror: str, listing: Path, data: Path) -> int:
    """Each listed vector and its .md5 into ``streams/<stem>/``, SHA-1 checked."""
    count = 0
    for line in listing.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, md5_digest, name = line.split()
        stem = Path(name).stem
        target = data / "streams" / stem
        target.mkdir(parents=True, exist_ok=True)
        for file, expected in ((name, digest), (name + ".md5", md5_digest)):
            path = target / file
            if not path.exists():
                with urllib.request.urlopen(mirror + file, timeout=300) as response:
                    path.write_bytes(response.read())
            if sha1(path) != expected:
                path.unlink()
                raise ValueError(f"{file}: SHA-1 differs from {listing.name}")
        suffix = Path(name).suffix
        if suffix in REWRAP and not (target / f"{stem}.ivf").exists():
            subprocess.run([sys.executable, "-I", str(ROOT / "tools/conformance/webm_to_ivf.py"),
                            str(target / name), str(target / f"{stem}.ivf"), REWRAP[suffix]],
                           check=True)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--data", type=Path, help="override the suite's data directory")
    args = parser.parse_args()
    kind, *source, data = SUITES[args.suite]
    data = args.data or data
    if kind == "listed":
        print(f"{fetch_listed(*source, data)} streams in {data / 'streams'}")
        return 0
    archives = download(*source, data)
    for archive in archives:
        extract(archive, data / "streams")
    print(f"{len(archives)} zips in {data / 'streams'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
