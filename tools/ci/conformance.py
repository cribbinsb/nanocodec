#!/usr/bin/env python3
"""Run our decoder over an official conformance suite: every selected stream
must decode MD5-exact to the reference decoder's output.

One runner for every codec's suite (`SUITES` below; a suite is a manifest,
a stream directory and our CLI's codec name). The manifest
(`tools/dev/conformance_manifest.py` builds it from the reference decoder)
lists per stream the whole-output MD5 and, where recorded, one MD5 per
picture and the picture's byte size, so a mismatch names the first wrong
picture. Decoded output is streamed into the hashes through a pipe; no
YUV is written.

Output format (the reference decoders' raw output): pictures in output
order, cropped to the conformance window, planes Y, Cb, Cr; 8-bit samples
as bytes, deeper ones as 16-bit little-endian.

Usage:
    python3 -I tools/ci/conformance.py --suite hevc-v1 [--bin NANOCODEC]
        [--profiles "Main,Main 10"] [--max-bit-depth 10] [--scan progressive]
        [--only A,B] [--jobs 4] [--timeout 600] [--json OUT]

Where the manifest records a stream's scan type and the tools it uses
(H.264), the summary also counts per profile and scan and groups the
failures by tool.

Exit status 0 when every selected stream matches, or with `--must-pass
FILE` (one stream name per line: a decoder still on its way to the whole
suite) when every listed stream does.

Before any decode, every selected stream must be in the stream directory with
the manifest's SHA-256: a partial or damaged corpus fails at once, naming the
streams (exit 2), rather than as one decode error per stream or, under
`--must-pass`, not at all for the streams not listed. `--check-corpus` runs
only that check over the whole manifest (tools/ci/wasm_conformance.sh, and
after syncing a corpus to the build box).
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))

# suite -> (our CLI's --codec, manifest, stream directory). Add a codec's
# suite here and build its manifest with tools/dev/conformance_manifest.py.
SUITES = {
    "hevc-v1": ("hevc", ROOT / "tools/ci/hevc_conformance_v1.json", DATA / "conformance/hevc/streams"),
    "vvc-v1": ("vvc", ROOT / "tools/ci/vvc_conformance_v1.json", DATA / "conformance/vvc/streams"),
    "av1-libaom": ("av1", ROOT / "tools/ci/av1_conformance_libaom.json", DATA / "conformance/av1/streams"),
    # libaom / SVT-AV1 streams for Main-profile tools the libaom vectors lack
    # (tools/dev/av1_conformance_supplement.sh).
    "av1-supplementary": ("av1", ROOT / "tools/ci/av1_conformance_supplementary.json",
                          DATA / "conformance/av1/supplementary"),
    "vp9-libvpx": ("vp9", ROOT / "tools/ci/vp9_conformance_libvpx.json", DATA / "conformance/vp9/streams"),
    # ITU-T H.264.1 (AVCv1, FRExt, High 10): conformance_fetch.py --suite h264.
    "h264": ("h264", ROOT / "tools/ci/h264_conformance.json", DATA / "conformance/h264/streams"),
    # x264 streams for I/P-only High tools (tools/dev/h264_conformance_supplement.sh).
    "h264-x264": ("h264", ROOT / "tools/ci/h264_conformance_x264.json",
                  DATA / "conformance/h264/supplementary"),
    # JM streams for Baseline / Extended tools the suite lacks
    # (tools/dev/h264_conformance_jm.sh).
    "h264-jm": ("h264", ROOT / "tools/ci/h264_conformance_jm.json", DATA / "conformance/h264/jm"),
}

CHUNK = 1 << 20
MAX_JOBS = 8
#: Bytes of the decoder's error output kept for the report.
ERROR_TAIL = 2000


class Hashes:
    """Whole-output MD5 and, given the picture size, one MD5 per picture."""

    def __init__(self, frame_bytes):
        self.whole = hashlib.md5()
        self.frame_bytes = frame_bytes
        self.frames = []
        self.current = hashlib.md5()
        self.filled = 0
        self.size = 0

    def update(self, chunk):
        self.whole.update(chunk)
        self.size += len(chunk)
        if not self.frame_bytes:
            return
        view = memoryview(chunk)
        while view:
            take = min(len(view), self.frame_bytes - self.filled)
            self.current.update(view[:take])
            self.filled += take
            view = view[take:]
            if self.filled == self.frame_bytes:
                self.frames.append(self.current.hexdigest())
                self.current = hashlib.md5()
                self.filled = 0


def decode_hashed(command, frame_bytes, timeout):
    """Run `command`, hashing its stdout; returns (hashes, return code or
    None on timeout, error tail)."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    hashes = Hashes(frame_bytes)
    errors = bytearray()
    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "pixels")
            selector.register(process.stderr, selectors.EVENT_READ, "errors")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                for key, _ in selector.select(min(remaining, 1)):
                    chunk = key.fileobj.read1(CHUNK)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif key.data == "pixels":
                        hashes.update(chunk)
                    else:
                        errors.extend(chunk)
                        del errors[:-ERROR_TAIL]
        if not timed_out:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
    code = None if timed_out else process.returncode
    return hashes, code, errors.decode(errors="replace").strip()


def run_one(binary, codec, streams, entry, timeout):
    start = time.monotonic()
    command = [str(binary), "decode", "--codec", codec, "--input",
               str(streams / entry["bitstream"]), "--output", "/dev/stdout"]
    hashes, code, error = decode_hashed(command, entry.get("frame_bytes"), timeout)
    md5 = hashes.whole.hexdigest()
    if code is None:
        status = "timeout"
    elif code != 0:
        status = "error"
    else:
        status = "pass" if md5 == entry["md5"] else "mismatch"
    result = {"status": status, "seconds": round(time.monotonic() - start, 2),
              "bytes": hashes.size, "md5": md5, "error": error.splitlines()[-1] if error else ""}
    expected = entry.get("frame_md5")
    if status != "pass" and expected:
        wrong = next((i for i, (a, b) in enumerate(zip(hashes.frames, expected)) if a != b), None)
        if wrong is None and len(hashes.frames) != len(expected):
            wrong = min(len(hashes.frames), len(expected))
        result["first_wrong_picture"] = wrong
        result["pictures"] = f"{len(hashes.frames)}/{len(expected)}"
    return result


def corpus_problems(entries, streams):
    """The entries' streams missing from `streams` or differing from their manifest
    SHA-256 (2026-11-27: the build box's H.264 folders held a stream's notes but not
    the stream, after a partial copy)."""
    problems = []
    for entry in entries:
        path = streams / entry["bitstream"]
        if not path.is_file():
            problems.append(f"{entry['name']}: missing {path}")
            continue
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                problems.append(f"{entry['name']}: {path} differs from the manifest's SHA-256")
    return problems


def selected(entries, args):
    """The manifest entries the profile, depth and name filters keep."""
    profiles = {p.strip() for p in args.profiles.split(",")} if args.profiles else None
    names = set(args.only.split(",")) if args.only else None
    keep = []
    for entry in entries:
        depth = max(entry.get("bit_depth", [8, 8]))
        if profiles and entry.get("profile") not in profiles:
            continue
        if args.max_bit_depth and depth > args.max_bit_depth:
            continue
        if names and entry["name"] not in names:
            continue
        if args.scan and entry.get("scan", "progressive") != args.scan:
            continue
        keep.append(entry)
    if names and names - {e["name"] for e in keep}:
        raise SystemExit(f"unknown or filtered streams: {sorted(names - {e['name'] for e in keep})}")
    return keep


def print_breakdown(entries, results):
    """Pass counts per profile and scan type, and the failures grouped by
    the tools beyond the decoder's base profile that each failing stream
    uses (the manifest's `features`)."""
    tally, failing = {}, {}
    for entry in entries:
        passed = results[entry["name"]]["status"] == "pass"
        count = tally.setdefault((entry.get("profile"), entry.get("scan")), [0, 0])
        count[0] += passed
        count[1] += 1
        if not passed:
            key = ", ".join(entry.get("features") or ["no tool beyond Constrained High"])
            failing.setdefault(key, []).append(entry["name"])
    print()
    for (profile, scan), (n, m) in sorted(tally.items(), key=str):
        print(f"{profile:24} {scan:12} {n}/{m}")
    if failing:
        print("\nfailing, by the tools they use:")
        for key, names in sorted(failing.items(), key=lambda item: -len(item[1])):
            print(f"{len(names):4}  {key}: {' '.join(sorted(names, key=str.lower))}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--bin", default=os.environ.get(
        "NC_NANOCODEC_BIN_FILE", str(ROOT / "target/release/nanocodec")))
    parser.add_argument("--manifest", type=Path, help="override the suite's manifest")
    parser.add_argument("--streams", type=Path, help="override the suite's stream directory")
    parser.add_argument("--profiles", default="", help="comma-separated profile names to keep")
    parser.add_argument("--max-bit-depth", type=int, default=0, help="keep streams up to this depth")
    parser.add_argument("--scan", choices=("progressive", "interlaced"), help="keep this scan type")
    parser.add_argument("--only", default="", help="comma-separated stream names")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--must-pass", type=Path,
                        help="only these streams (one name per line) must match")
    parser.add_argument("--check-corpus", action="store_true",
                        help="only check that the whole corpus is present and intact")
    args = parser.parse_args()
    codec, manifest, streams = SUITES[args.suite]
    manifest = args.manifest or manifest
    streams = args.streams or streams
    data = json.loads(manifest.read_text())
    listed = data["streams"] if isinstance(data, dict) else data
    entries = listed if args.check_corpus else selected(listed, args)
    if not entries:
        parser.error("empty conformance selection")
    if problems := corpus_problems(entries, streams):
        print("\n".join(problems))
        print(f"{args.suite}: {len(problems)} of {len(entries)} streams missing or damaged under "
              f"{streams} (tools/dev/conformance_fetch.py, or nc-remote run --sync on the box)")
        return 2
    if args.check_corpus:
        print(f"{args.suite}: all {len(entries)} streams present with the manifest's SHA-256")
        return 0
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs, MAX_JOBS)) as pool:
        futures = {pool.submit(run_one, args.bin, codec, streams, e, args.timeout): e for e in entries}
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]["name"]] = future.result()
    passed = 0
    by_depth = {}
    depth_of = {e["name"]: max(e.get("bit_depth", [8, 8])) for e in entries}
    for name in sorted(results, key=str.lower):
        r = results[name]
        passed += r["status"] == "pass"
        tally = by_depth.setdefault(depth_of[name], [0, 0])
        tally[0] += r["status"] == "pass"
        tally[1] += 1
        line = f"{r['status']:8} {name:40} {r['seconds']:7.2f}s"
        if r["status"] != "pass":
            if "first_wrong_picture" in r:
                line += f"  first wrong picture {r['first_wrong_picture']} ({r['pictures']} decoded)"
            if r["error"]:
                line += "  " + r["error"]
        print(line)
    if any("scan" in e for e in entries):
        print_breakdown(entries, results)
    depths = ", ".join(f"{bits}-bit {n}/{m}" for bits, (n, m) in sorted(by_depth.items()))
    by_profile = {}
    for entry in entries:
        tally = by_profile.setdefault(entry.get("profile") or "unknown profile", [0, 0])
        tally[0] += results[entry["name"]]["status"] == "pass"
        tally[1] += 1
    profiles = "; ".join(f"{name} {n}/{m}" for name, (n, m) in sorted(by_profile.items()))
    print(f"{args.suite}: {passed}/{len(results)} streams match the reference decoder ({depths})"
          f" [{profiles}]")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    if args.must_pass:
        required = {line.strip() for line in args.must_pass.read_text().splitlines()
                    if line.strip() and not line.startswith("#")}
        failed = sorted(n for n in required & set(results) if results[n]["status"] != "pass")
        if failed:
            print(f"required streams that do not match: {' '.join(failed)}")
        return 0 if not failed else 1
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
