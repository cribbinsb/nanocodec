#!/usr/bin/env python3
"""Decode other encoders' AV2 streams with avmdec, dav2d and nanocodec and
compare per-frame MD5s (docs/handoff/av2.md "Other encoders' streams").

    tools/dev/av2_foreign/compare.py [--write-hashes] [STREAM...]

Without streams, the whole corpus: the dav2d test vectors, 8-bit (`avm/`)
and deeper (`avm-10bit/`: 10- and 12-bit) (`git clone
https://code.videolan.org/videolan/dav2d-test-data` into CORPUS/vectors) and
the avmenc streams of encode.sh (CORPUS/streams), where CORPUS is
$NC_AV2_FOREIGN or ~/nanocodec-data/work/av2-foreign. The references are
avmdec and dav2d (`dav2d-hbd`, its 16-bit build, where installed), from the
vp9-av2 anchor prefix or the platform's (the build box keeps them there).
Prints one
table row per stream: name, result, first error. `--write-hashes` rewrites
crates/nc-cli/tests/data/av2-foreign.md5 (the regression test's list) with
avmdec's frame hashes of every stream nanocodec decodes bit-exactly.

Decoded pictures go through pipes into hashes; nothing is written.
"""
import hashlib
import os
import subprocess
import tempfile
import sys
from pathlib import Path

HOME = Path.home()
ROOT = Path(__file__).resolve().parents[3]
CORPUS = Path(os.environ.get("NC_AV2_FOREIGN", HOME / "nanocodec-data/work/av2-foreign"))
ANCHOR_DIRS = [HOME / "nanocodec-data/anchors/vp9-av2/bin",
               HOME / "nanocodec-data/anchors" / os.uname().machine / "bin"]
NC = Path(os.environ.get("NC_NANOCODEC_BIN", HOME / "nanocodec-data/targets/av2-dec/gate/nanocodec"))
HASHES = ROOT / "crates/nc-cli/tests/data/av2-foreign.md5"


def y4m_hashes(cmd):
    """Run cmd (Y4M on stdout); return (tags, [md5 per frame], last stderr line, rc).

    stderr goes to a temporary file, and stdout is read to its end, so a
    decoder blocked on either pipe never deadlocks the comparison (a stream
    whose frames change size misaligns the frame reads)."""
    errors = tempfile.TemporaryFile()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors)
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
    p.stdout.read()
    rc = p.wait()
    errors.seek(0)
    err = errors.read().decode(errors="replace").strip().splitlines()
    return tags, hashes, err[-1] if err else "", rc


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


def anchor(*names):
    """The first of `names` installed in an anchor directory."""
    for name in names:
        for directory in ANCHOR_DIRS:
            if (directory / name).is_file():
                return str(directory / name)
    sys.exit(f"none of {names} in {ANCHOR_DIRS}")


def corpus():
    vectors = CORPUS / "vectors/dav2d-test-data"
    return (sorted((vectors / "avm").glob("*.obu")) + sorted((vectors / "avm-10bit").glob("*.obu"))
            + sorted((CORPUS / "streams").glob("*.ivf")))


def main():
    args = sys.argv[1:]
    write = "--write-hashes" in args
    paths = [Path(a) for a in args if a != "--write-hashes"] or corpus()
    passing, lines = 0, []
    avmdec, dav2d = anchor("avmdec"), anchor("dav2d-hbd", "dav2d")
    print("| stream | result | first error |\n|---|---|---|")
    for path in paths:
        _, ref, err, rc = y4m_hashes([avmdec, "-o", "-", str(path)])
        _, second, err2, rc2 = y4m_hashes([dav2d, "-q", "--threads", "1", "-i", str(path),
                                           "--muxer", "yuv4mpeg2", "-o", "-"])
        if rc or not ref or second != ref:
            print(f"| {path.name} | references disagree or fail | avmdec rc={rc} {err}; dav2d rc={rc2} {err2} |")
            continue
        ok, result, error = verdict(ref, *ours(path))
        passing += ok
        if ok:
            lines.append(f"{path.name} {' '.join(ref)}\n")
        print(f"| {path.name} | {result} | {error} |")
    print(f"\n{passing} of {len(paths)} bit-exact with avmdec and dav2d")
    if write:
        HASHES.write_text("".join(lines))


if __name__ == "__main__":
    main()
