#!/usr/bin/env python3
"""Run our H.264 decoder over the Constrained High conformance manifest.

Every selected stream of tools/ci/h264_conformance.json is decoded with
`nanocodec decode --codec h264`; the output is streamed through a pipe and
compared frame by frame (MD5) with the reference YUV shipped with the
stream. Nothing decoded is written to disk.

    python3 -I tools/ci/h264_conformance.py --bin <target>/release \
        [--root ~/nanocodec-data/conformance/h264] [--jobs 4] [--only REGEX] [--json out.json]

Exit status: 0 when every stream matches, 1 otherwise. The conformance
streams are untrusted downloads: they are only read.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h264_conformance.json")
SUITE_DIRS = {"itu": "streams", "supplementary": "supplementary"}
TIMEOUT_SECONDS = 300
READ_CHUNK = 1 << 20


def frame_size(stream):
    """Visible luma size of the first picture, from ffprobe (4:2:0 frames)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", stream],
        capture_output=True, text=True, check=False)
    width, height = map(int, probe.stdout.strip().split(",")[:2])
    return width * height * 3 // 2


def run_stream(binary, root, entry):
    stream = os.path.join(root, SUITE_DIRS[entry["suite"]], entry["stream"])
    reference = os.path.join(root, SUITE_DIRS[entry["suite"]], entry["yuv"])
    result = {"name": entry["name"], "suite": entry["suite"]}
    try:
        size = frame_size(stream)
    except (ValueError, IndexError):
        return {**result, "status": "error", "error": "ffprobe failed"}
    process = subprocess.Popen(
        [os.path.join(binary, "nanocodec"), "decode", "--codec", "h264", "--input", stream,
         "--output", "/dev/stdout"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = mismatch = 0
    first_mismatch = None
    with open(reference, "rb") as ref:
        while True:
            got = process.stdout.read(size)
            want = ref.read(size)
            if not got or not want:
                extra_got, extra_want = len(got), len(want)
                break
            if hashlib.md5(got).digest() != hashlib.md5(want).digest():
                mismatch += 1
                if first_mismatch is None:
                    first_mismatch = frames
            frames += 1
        reference_left = extra_want + len(ref.read())
    decoded_left = extra_got + len(process.stdout.read())
    try:
        _, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        return {**result, "status": "timeout"}
    error = stderr.decode(errors="replace").strip().splitlines()[-1:] if process.returncode else []
    ok = process.returncode == 0 and mismatch == 0 and reference_left == 0 and decoded_left == 0
    return {**result, "status": "pass" if ok else "fail", "frames": frames,
            "mismatched_frames": mismatch, "first_mismatch": first_mismatch,
            "missing_bytes": reference_left, "extra_bytes": decoded_left,
            "error": error[0] if error else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bin", required=True, help="directory with the nanocodec binary")
    parser.add_argument("--root", default=os.path.expanduser("~/nanocodec-data/conformance/h264"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--only", help="regex over stream names")
    parser.add_argument("--json", help="write the per-stream results here")
    args = parser.parse_args()
    with open(MANIFEST, encoding="utf-8") as f:
        entries = json.load(f)["selected"]
    if args.only:
        entries = [e for e in entries if re.search(args.only, e["name"])]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda e: run_stream(args.bin, args.root, e), entries))
    for r in results:
        detail = ""
        if r["status"] == "fail":
            detail = (f"frames {r['frames']}, mismatched {r['mismatched_frames']} "
                      f"(first {r['first_mismatch']}), missing {r['missing_bytes']} B, "
                      f"extra {r['extra_bytes']} B")
            if r["error"]:
                detail += f": {r['error']}"
        elif r["status"] == "error":
            detail = r["error"]
        print(f"{r['status']:7} {r['suite']:13} {r['name']:30} {detail}")
    passed = sum(r["status"] == "pass" for r in results)
    print(f"\n{passed}/{len(results)} streams pass")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
