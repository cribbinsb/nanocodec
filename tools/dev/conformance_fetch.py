#!/usr/bin/env python3
"""Fetch an official conformance suite's zips and extract each into its
own directory, for tools/dev/conformance_manifest.py and
tools/ci/conformance.py.

The bitstreams are untrusted data: each zip is extracted (member paths
checked) into ``<data>/streams/<zip name>/`` and nothing in them is
executed. Zips already fetched or extracted are kept.

Usage: conformance_fetch.py --suite hevc-v1 [--data DIR]
"""

import argparse
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
ITU = "https://www.itu.int"

# suite -> (index page listing the zips, data directory).
SUITES = {
    "hevc-v1": (ITU + "/wftp3/av-arch/jctvc-site/bitstream_exchange/draft_conformance/HEVC_v1/",
                DATA / "conformance/hevc"),
}


def download(index_url: str, data: Path) -> list[Path]:
    zips = data / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(index_url, timeout=60) as response:
        index = response.read().decode("latin-1")
    paths = []
    for href in re.findall(r'HREF="([^"]+\.zip)"', index, re.IGNORECASE):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--data", type=Path, help="override the suite's data directory")
    args = parser.parse_args()
    index_url, data = SUITES[args.suite]
    data = args.data or data
    archives = download(index_url, data)
    for archive in archives:
        extract(archive, data / "streams")
    print(f"{len(archives)} zips in {data / 'streams'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
