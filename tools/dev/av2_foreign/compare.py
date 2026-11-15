#!/usr/bin/env python3
"""Decode other encoders' AV2 streams with avmdec, dav2d and nanocodec and
compare per-frame MD5s (docs/handoff/av2.md "Other encoders' streams").

    tools/dev/av2_foreign/compare.py [--write-hashes] [STREAM...]

Without streams, the whole corpus: the 8-bit dav2d test vectors
(`git clone https://code.videolan.org/videolan/dav2d-test-data` into
CORPUS/vectors) and the avmenc streams of encode.sh (CORPUS/streams), where
CORPUS is $NC_AV2_FOREIGN or ~/nanocodec-data/work/av2-foreign. Prints one
table row per stream: name, result, first error. `--write-hashes` rewrites
crates/nc-cli/tests/data/av2-foreign.md5 (the regression test's list) with
avmdec's frame hashes of every stream nanocodec decodes bit-exactly.

Decoded pictures go through pipes into hashes; nothing is written.
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
ROOT = Path(__file__).resolve().parents[3]
CORPUS = Path(os.environ.get("NC_AV2_FOREIGN", HOME / "nanocodec-data/work/av2-foreign"))
BIN = HOME / "nanocodec-data/anchors/vp9-av2/bin"
NC = Path(os.environ.get("NC_NANOCODEC_BIN", HOME / "nanocodec-data/targets/av2-dec/gate/nanocodec"))
HASHES = ROOT / "crates/nc-cli/tests/data/av2-foreign.md5"


def y4m_hashes(cmd):
    """Run cmd (Y4M on stdout); return (tags, [md5 per frame], last stderr line, rc)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    header = p.stdout.readline().decode(errors="replace").strip()
    tags, hashes = {}, []
    if header.startswith("YUV4MPEG2"):
        tags = {t[0]: t[1:] for t in header.split()[1:]}
        w, h = int(tags["W"]), int(tags["H"])
        deep = "p1" in tags.get("C", "420")
        size = (w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2)) * (2 if deep else 1)
        while p.stdout.readline().startswith(b"FRAME"):
            data = p.stdout.read(size)
            if len(data) < size:
                break
            hashes.append(hashlib.md5(data).hexdigest())
    else:
        p.stdout.read()
    err = p.stderr.read().decode(errors="replace").strip().splitlines()
    return tags, hashes, err[-1] if err else "", p.wait()


def ours(path):
    """nanocodec's frame hashes, last stderr line and exit code."""
    p = subprocess.run([str(NC), "decode", "--codec", "av2", "--input", str(path), "--frame-md5", "/dev/stdout"],
                       capture_output=True, text=True)
    err = p.stderr.strip().splitlines()
    return p.stdout.split(), err[-1].removeprefix("nanocodec: ") if err else "", p.returncode


def verdict(ref, got, err, rc):
    if rc != 0 or (not got and ref):
        ok = got == ref[:len(got)]
        return False, f"rejected after {len(got)} frames{'' if ok else ' (mismatch before)'}", err or f"rc={rc}"
    for i, (a, b) in enumerate(zip(ref, got)):
        if a != b:
            return False, f"MISMATCH at frame {i} of {len(ref)}", ""
    if len(got) != len(ref):
        return False, f"{len(got)} frames, {len(ref)} expected", ""
    return True, f"bit-exact ({len(ref)} frames)", ""


def corpus():
    return sorted((CORPUS / "vectors/dav2d-test-data/avm").glob("*.obu")) + sorted((CORPUS / "streams").glob("*.ivf"))


def main():
    args = sys.argv[1:]
    write = "--write-hashes" in args
    paths = [Path(a) for a in args if a != "--write-hashes"] or corpus()
    passing, lines = 0, []
    print("| stream | result | first error |\n|---|---|---|")
    for path in paths:
        tags, ref, err, rc = y4m_hashes([str(BIN / "avmdec"), "-o", "-", str(path)])
        _, second, err2, rc2 = y4m_hashes([str(BIN / "dav2d"), "-q", "--threads", "1", "-i", str(path),
                                           "--muxer", "yuv4mpeg2", "-o", "-"])
        if rc or not ref or second != ref:
            print(f"| {path.name} | references disagree or fail | avmdec rc={rc} {err}; dav2d rc={rc2} {err2} |")
            continue
        if "p1" in tags.get("C", ""):
            print(f"| {path.name} | not run | not 8-bit |")
            continue
        ok, result, error = verdict(ref, *ours(path))
        passing += ok
        if ok:
            lines.append(f"{path.name} {' '.join(ref)}\n")
        print(f"| {path.name} | {result} | {error} |")
    print(f"\n{passing} of {len(paths)} bit-exact with avmdec and dav2d")
    if write:
        HASHES.write_text("".join(lines))


main()
