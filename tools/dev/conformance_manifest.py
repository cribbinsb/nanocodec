#!/usr/bin/env python3
"""Build a conformance manifest for tools/ci/conformance.py: probe each
fetched stream (profile, chroma format, bit depths, cropped size), keep the
selected ones, and record the reference decoder's output MD5, whole and per
picture, streamed through a FIFO (no YUV is written).

The streams are untrusted data (tools/dev/conformance_fetch.py fetches
them); nothing in them is executed. Per suite, `SUITES` names the codec's
probe and the reference decoder's command line.

Usage:
    python3 -I tools/dev/conformance_manifest.py --suite hevc-v1
        [--profiles "Main,Main 10,Main Still Picture"] [--max-bit-depth 10]
        [--chroma 4:2:0,4:0:0] [--only A,B] [--jobs 4]

Entries already in the manifest for streams not rebuilt are kept.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
from conformance import SUITES as RUNNER_SUITES, Hashes  # noqa: E402

DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
BITSTREAM_SUFFIXES = (".bit", ".bin")
CHUNK = 1 << 20
#: Seconds one reference decode may take.
REFERENCE_TIMEOUT = 1800


class Bits:
    """MSB-first reader of an RBSP with Exp-Golomb codes."""

    def __init__(self, data):
        self.data, self.pos = data, 0

    def u(self, n):
        value = 0
        for _ in range(n):
            byte = self.data[self.pos >> 3]
            value = (value << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return value

    def ue(self):
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
        return (1 << zeros) - 1 + self.u(zeros)


def annex_b_units(data):
    """NAL units of an Annex-B byte stream, emulation prevention removed."""
    for unit in re.split(b"\x00\x00\x01", data):
        unit = unit.rstrip(b"\x00")
        if unit:
            yield re.sub(b"\x00\x00\x03", b"\x00\x00", unit)


HEVC_PROFILES = {1: "Main", 2: "Main 10", 3: "Main Still Picture", 4: "Format Range Extensions",
                 9: "Screen Content Coding"}
HEVC_SPS = 33
#: SubWidthC, SubHeightC by chroma_format_idc (H.265 Table 6-1).
SUBSAMPLING = {0: (1, 1), 1: (2, 2), 2: (2, 1), 3: (1, 1)}
CHROMA_NAMES = {0: "4:0:0", 1: "4:2:0", 2: "4:2:2", 3: "4:4:4"}


def probe_hevc(path):
    """Profile, chroma format, depths and cropped size from the first SPS
    (H.265 §7.3.2.2), or None."""
    for nal in annex_b_units(path.read_bytes()):
        if len(nal) < 3 or (nal[0] >> 1) & 63 != HEVC_SPS:
            continue
        r = Bits(nal[2:])
        r.u(4)
        sub_layers = r.u(3)
        r.u(1)
        r.u(2), r.u(1)
        profile_idc = r.u(5)
        compatibility = r.u(32)
        r.u(48), r.u(8)  # constraint flags, general_level_idc
        present = [(r.u(1), r.u(1)) for _ in range(sub_layers)]
        if sub_layers:
            r.u(2 * (8 - sub_layers))
        for profile, level in present:
            r.u(88 if profile else 0), r.u(8 if level else 0)
        r.ue()
        chroma = r.ue()
        if chroma == 3:
            r.u(1)
        width, height = r.ue(), r.ue()
        crop = [r.ue() for _ in range(4)] if r.u(1) else [0, 0, 0, 0]
        depths = [r.ue() + 8, r.ue() + 8]
        if profile_idc == 0:
            # Some streams signal the profile only in the compatibility flags.
            profile_idc = next((j for j in range(1, 32) if compatibility >> (31 - j) & 1), 0)
        sub_w, sub_h = SUBSAMPLING[chroma]
        w = width - sub_w * (crop[0] + crop[1])
        h = height - sub_h * (crop[2] + crop[3])
        chroma_samples = 0 if chroma == 0 else 2 * (w // sub_w) * (h // sub_h)
        sample_bytes = 2 if max(depths) > 8 else 1
        return {"profile": HEVC_PROFILES.get(profile_idc, f"profile_idc {profile_idc}"),
                "chroma_format": CHROMA_NAMES[chroma], "bit_depth": depths,
                "width": w, "height": h, "frame_bytes": (w * h + chroma_samples) * sample_bytes}
    return None


VVC_PROFILES = {"Main 10": "Main 10", "65": "Main 10 Still Picture",
                "Main 10 Still Picture": "Main 10 Still Picture"}
#: ffprobe pixel formats: chroma format and bit depth.
PIXEL_FORMATS = {"gray": ("4:0:0", 8), "yuv420p": ("4:2:0", 8), "gray10le": ("4:0:0", 10),
                 "yuv420p10le": ("4:2:0", 10), "yuv422p10le": ("4:2:2", 10),
                 "yuv444p10le": ("4:4:4", 10), "yuv444p": ("4:4:4", 8)}


def probe_vvc(path):
    """Profile, chroma format, depth and cropped size from ffprobe (its VVC
    parser reads the SPS, subpictures included), or None."""
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "stream=profile,pix_fmt,width,height", "-of", "json", str(path)],
                            capture_output=True, text=True, timeout=60, check=False)
    streams = json.loads(result.stdout or "{}").get("streams", [])
    if len(streams) != 1 or streams[0].get("pix_fmt") not in PIXEL_FORMATS:
        return None
    info = streams[0]
    chroma, depth = PIXEL_FORMATS[info["pix_fmt"]]
    w, h = info["width"], info["height"]
    sub_w, sub_h = {"4:0:0": (1, 1), "4:2:0": (2, 2), "4:2:2": (2, 1), "4:4:4": (1, 1)}[chroma]
    chroma_samples = 0 if chroma == "4:0:0" else 2 * (w // sub_w) * (h // sub_h)
    profile = str(info.get("profile"))
    return {"profile": VVC_PROFILES.get(profile, f"profile {profile}"),
            "chroma_format": chroma, "bit_depth": [depth, depth], "width": w, "height": h,
            "frame_bytes": (w * h + chroma_samples) * (2 if depth > 8 else 1)}


# suite -> (probe, reference decoder command for (bitstream, output path)).
SUITES = {
    "vvc-v1": (probe_vvc, lambda bitstream, out: [
        str(DATA / "work/standards-eval/src/VTM/bin/DecoderAppStatic"), "-b", str(bitstream),
        "-o", str(out), "-d", "0"]),
    "hevc-v1": (probe_hevc, lambda bitstream, out: [
        str(DATA / "anchors/x86_64/bin/TAppDecoder"), "-b", str(bitstream), "-o", str(out), "-d", "0"]),
}


def bitstream(directory):
    found = [p for p in directory.rglob("*")
             if p.suffix.lower() in BITSTREAM_SUFFIXES and "__MACOSX" not in p.parts]
    return found[0] if len(found) == 1 else None


def shipped_md5s(directory):
    found = set()
    for p in directory.rglob("*"):
        if p.is_file() and "md5" in p.name.lower() and p.stat().st_size < 4096:
            found.update(re.findall(r"\b[0-9a-fA-F]{32}\b", p.read_text("latin-1")))
    return {m.lower() for m in found}


def reference_hashes(command, frame_bytes):
    """Hash the reference decoder's output, written into a FIFO; also its
    exit status (VTM exits 1 on a picture-hash SEI mismatch, as after a GDR
    start, though its output is the conformance output)."""
    with tempfile.TemporaryDirectory(prefix="conformance-") as scratch:
        fifo = Path(scratch) / "out.yuv"
        os.mkfifo(fifo)
        hashes = Hashes(frame_bytes)

        def drain():
            with open(fifo, "rb") as pipe:
                while chunk := pipe.read(CHUNK):
                    hashes.update(chunk)

        reader = threading.Thread(target=drain)
        reader.start()
        try:
            status = subprocess.run(command(fifo), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, check=False,
                                    timeout=REFERENCE_TIMEOUT).returncode
        finally:
            # A decoder that failed before opening its output: release the reader.
            if reader.is_alive():
                with open(fifo, "wb"):
                    pass
            reader.join()
    return hashes, status


def build_entry(suite, streams, directory):
    probe, reference = SUITES[suite]
    path = bitstream(directory)
    info = probe(path) if path else None
    if info is None:
        return directory.name, None
    try:
        hashes, status = reference_hashes(lambda out: reference(path, out), info["frame_bytes"])
    except subprocess.TimeoutExpired:
        print(f"skip {directory.name}: reference decoder timed out", file=sys.stderr)
        return directory.name, None
    shipped = shipped_md5s(directory)
    md5 = hashes.whole.hexdigest()
    if status != 0 and not hashes.frames:
        print(f"skip {directory.name}: reference decoder failed ({status})", file=sys.stderr)
        return directory.name, None
    return directory.name, {
        "name": directory.name, "bitstream": str(path.relative_to(streams)), **info,
        "md5": md5, "frames": len(hashes.frames), "frame_md5": hashes.frames,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shipped_md5_agrees": (md5 in shipped) if shipped else None,
        **({"reference_status": status} if status else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--profiles", default="Main,Main 10,Main Still Picture")
    parser.add_argument("--max-bit-depth", type=int, default=10)
    parser.add_argument("--chroma", default="4:2:0,4:0:0")
    parser.add_argument("--only", default="")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    _, manifest, streams = RUNNER_SUITES[args.suite]
    probe, _ = SUITES[args.suite]
    profiles = set(args.profiles.split(","))
    chroma = set(args.chroma.split(","))
    names = set(args.only.split(",")) if args.only else None
    chosen = []
    for directory in sorted(p for p in streams.iterdir() if p.is_dir()):
        if names and directory.name not in names:
            continue
        path = bitstream(directory)
        info = probe(path) if path else None
        if (info and info["profile"] in profiles and info["chroma_format"] in chroma
                and max(info["bit_depth"]) <= args.max_bit_depth):
            chosen.append(directory)
        else:
            print(f"skip {directory.name}: {info and (info['profile'], info['chroma_format'], info['bit_depth'])}",
                  file=sys.stderr)
    existing = {}
    if manifest.exists():
        data = json.loads(manifest.read_text())
        existing = {e["name"]: e for e in (data["streams"] if isinstance(data, dict) else data)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name, entry in pool.map(lambda d: build_entry(args.suite, streams, d), chosen):
            if entry:
                existing[name] = entry
                print(f"{name}: {entry['profile']} {entry['bit_depth']} {entry['md5']}"
                      f" shipped={entry['shipped_md5_agrees']}", file=sys.stderr)
    out = {"suite": args.suite, "reference": "the suite's reference decoder (SUITES)",
           "streams": sorted(existing.values(), key=lambda e: e["name"].lower())}
    manifest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out['streams'])} streams -> {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
