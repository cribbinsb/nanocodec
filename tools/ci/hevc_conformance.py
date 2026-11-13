#!/usr/bin/env python3
"""Decode the HEVC Main-profile conformance streams with our decoder and
compare the MD5 of the output (output order, cropped) with HM's.

The manifest (`tools/ci/hevc_conformance_main.json`) and the streams are made
by `tools/dev/hevc_conformance_fetch.py`. Decoded output is streamed into the
hash; nothing is written to disk.

Usage: hevc_conformance.py [--bin NANOCODEC] [--streams DIR] [--jobs N]
                           [--only NAME,...] [--json OUT]
Exit status 0 when every selected stream matches.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHUNK = 1 << 20
TIMEOUT_SECONDS = 600


def run_one(binary: str, path: Path, expected: str) -> dict:
    start = time.monotonic()
    proc = subprocess.Popen(
        [binary, "decode", "--codec", "hevc", "--input", str(path), "--output", "/dev/stdout"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    digest = hashlib.md5()
    size = 0
    while chunk := proc.stdout.read(CHUNK):
        digest.update(chunk)
        size += len(chunk)
    try:
        _, err = proc.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, err = proc.communicate()
    md5 = digest.hexdigest()
    status = "pass" if proc.returncode == 0 and md5 == expected else (
        "mismatch" if proc.returncode == 0 else "error")
    return {"status": status, "seconds": round(time.monotonic() - start, 2),
            "bytes": size, "error": err.decode(errors="replace").strip()[-300:]}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin", default=os.environ.get(
        "NC_NANOCODEC_BIN_FILE", str(root / "target/release/nanocodec")))
    parser.add_argument("--streams", type=Path,
                        default=Path.home() / "nanocodec-data/conformance/hevc/streams")
    parser.add_argument("--manifest", type=Path, default=root / "tools/ci/hevc_conformance_main.json")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--only", default="")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        entries = [e for e in entries if e["name"] in wanted]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs, 8)) as pool:
        futures = {pool.submit(run_one, args.bin, args.streams / e["bitstream"], e["md5"]): e
                   for e in entries}
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            results[entry["name"]] = future.result()
    passed = 0
    for name in sorted(results, key=str.lower):
        r = results[name]
        passed += r["status"] == "pass"
        line = f"{r['status']:8} {name:34} {r['seconds']:7.2f}s"
        if r["status"] != "pass":
            line += "  " + r["error"].splitlines()[-1] if r["error"] else ""
        print(line)
    print(f"{passed}/{len(results)} streams match HM")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
