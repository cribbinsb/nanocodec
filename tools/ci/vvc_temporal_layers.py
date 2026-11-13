#!/usr/bin/env python3
"""VVC temporal extraction conformance, using shared streamed encode/decode gates.

Sources live in a bounded anonymous memfd; only bitstreams and hashes reach disk.
Every extracted layer must match its full-stream reconstruction in our decoder,
VVdeC and the pinned FFmpeg >= 8.0. Run via temporal_layers_conformance.sh vvc.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
from nceval.codecs import Codec
from nceval.config import load_paths
from nceval.conformance import gate_stream
from nceval.diskguard import check_disk
from nceval.framepipe import FrameHashSink
from nceval.point import encode_streamed
from nceval.slots import cpu_slots
from nceval.yuv import YuvFormat

# Cropping, interrupted mini-GOPs, long prediction chains and 8-bit POC wrap.
SHAPES = [(352, 288, 23, 11), (200, 120, 13, 6), (128, 96, 45, 300),
          (64, 64, 262, 300)]


def memory_file() -> int:
    """Linux anonymous source storage, including Python builds lacking os.memfd_create."""
    if hasattr(os, "memfd_create"):
        return os.memfd_create("vvc-temporal-source")
    # Some Conda Python builds omit the wrapper despite running on modern Linux.
    import ctypes
    create = ctypes.CDLL(None, use_errno=True).memfd_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(b"vvc-temporal-source", 1)  # MFD_CLOEXEC
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def source(fd: int, fmt: YuvFormat, frames: int, adaptive: bool) -> None:
    """Moving textured planes, with a still first half for adaptive layering."""
    with os.fdopen(os.dup(fd), "wb") as output:
        for frame in range(frames):
            shift = max(0, frame - frames // 2) * 8 if adaptive else frame * 2
            for plane in range(3):
                scale = 1 if plane == 0 else 2
                y, x = np.indices((fmt.height // scale, fmt.width // scale))
                x = x + shift // scale
                pixels = (16 + ((x * 3 + y * 5 + (x // 16 ^ y // 16) * 37
                                 + plane * 53) % 220)).astype(np.uint8)
                output.write(pixels.tobytes())
    os.lseek(fd, 0, os.SEEK_SET)


def check_case(binary: Path, paths, work: Path, layers: int, shape, adaptive: bool):
    w, h, frames, period = shape
    fmt = YuvFormat(w, h)
    label = f"L{layers}-{w}x{h}-{frames}-key{period}-{'adaptive' if adaptive else 'fixed'}"
    stream = work / f"{label}.vvc"
    fd = memory_file()
    try:
        source(fd, fmt, frames, adaptive)
        command = [str(binary), "encode", "--codec", "vvc", "--input",
                   f"/proc/{os.getpid()}/fd/{fd}", "--width", str(w), "--height", str(h),
                   "--fps", "30", "--qp", "30", "--frames", str(frames),
                   "--intra-period", str(period), "--temporal-layers", str(layers),
                   "--param", f"layers.adaptive={int(adaptive)}", "--output", str(stream)]
        recon = FrameHashSink()
        with cpu_slots(1):
            count = encode_streamed(lambda output: command + ["--recon", output], fmt, [recon])
        assert count == frames, (label, count)
    finally:
        os.close(fd)
    extract = [str(binary), "extract", "--codec", "vvc", "--temporal-layers", str(layers),
               "--input", str(stream)]
    ids = list(map(int, subprocess.check_output(extract + ["--list"], text=True).split()))
    assert len(ids) == frames and min(ids) == 0 and max(ids) == layers - 1, (label, ids)
    results = []
    for maximum in range(layers):
        kept = work / f"{label}-T{maximum}.vvc"
        subprocess.run(extract + ["--max-layer", str(maximum), "--output", str(kept)], check=True)
        expected = [digest for digest, layer in zip(recon.digests, ids) if layer <= maximum]
        gate = gate_stream(kept, Codec.VVC, fmt, len(expected), paths,
                           expected_hashes=expected, own_decoder=binary, serial=True)
        if not gate.passed:
            raise RuntimeError(f"{label}/T{maximum}: {gate.reason}")
        results.append({"maximum": maximum, "frames": len(expected), "gate": gate.level})
        print(f"PASS {label}/T{maximum}: {len(expected)} frames", flush=True)
    return {"case": label, "layers": ids, "operating_points": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one cropped, interrupted-GOP case")
    args = parser.parse_args()
    paths = load_paths()
    binary = Path(os.environ.get("NANOCODEC", paths.nanocodec_bin / "nanocodec")).resolve()
    work = paths.work / "vvc-temporal-layers-conformance"
    check_disk(work)
    cases = [(shape, False) for shape in (SHAPES[1:2] if args.quick else SHAPES)]
    if not args.quick:
        cases += [((200, 120, 40, 300), True), ((200, 120, 30, 13), True)]
    results = []
    for layers in (2, 3):
        for shape, adaptive in cases:
            results.append(check_case(binary, paths, work, layers, shape, adaptive))
    (work / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"VVC temporal-layer conformance passed: {len(results)} cases; {work / 'report.json'}")


if __name__ == "__main__":
    main()
