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
the frame hashes of every stream nanocodec decodes bit-exactly: first those
where avmdec and dav2d agree, then, under a `#` note, those where only avmdec
does (the pinned dav2d predates some test vectors; avmdec is the reference):
there nanocodec's own frame hashes, its pictures having matched avmdec's MD5.

dav2d writes each decoded picture to its own file in a temporary directory
(`nanocodec-av2-compare-*`, removed at once), hashed in output order:
frames may change size within a stream (reference scaling), which one Y4M
stream cannot carry. avmdec agrees when its MD5 of all pictures (`--md5`)
is that of dav2d's pictures in order; where it does not, nanocodec's
pictures are held against avmdec's MD5 alone. nanocodec's frame hashes come
through a pipe.
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


def last_line(stderr):
    lines = stderr.decode(errors="replace").strip().splitlines()
    return lines[-1] if lines else ""


def frame_hashes(cmd):
    """Run cmd(directory) (it writes one file per picture, numbered in
    output order); return ([md5 per frame], md5 of all, last stderr line, rc)."""
    whole = hashlib.md5()
    with tempfile.TemporaryDirectory(prefix="nanocodec-av2-compare-") as directory:
        p = subprocess.run(cmd(directory), capture_output=True)
        number = lambda f: int("".join(c for c in f.stem if c.isdigit()) or 0)
        hashes = []
        for f in sorted(Path(directory).iterdir(), key=number):
            data = f.read_bytes()
            whole.update(data)
            hashes.append(hashlib.md5(data).hexdigest())
    return hashes, whole.hexdigest(), last_line(p.stderr), p.returncode


def stream_md5(cmd):
    """Run cmd (it prints the MD5 of all its pictures first); return (md5,
    last stderr line, rc)."""
    p = subprocess.run(cmd, capture_output=True)
    words = p.stdout.decode(errors="replace").split()
    return (words[0] if words else ""), last_line(p.stderr), p.returncode


def ours_md5(path):
    """The MD5 of all nanocodec's pictures of `path`, last stderr line and
    exit code."""
    with tempfile.TemporaryDirectory(prefix="nanocodec-av2-compare-") as directory:
        out = Path(directory) / "out.yuv"
        p = subprocess.run([str(NC), "decode", "--codec", "av2", "--input", str(path), "--output", str(out)],
                           capture_output=True)
        md5 = hashlib.md5(out.read_bytes()).hexdigest() if out.exists() else ""
        return md5, last_line(p.stderr).removeprefix("nanocodec: "), p.returncode


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
    passing, avmdec_only, lines, avmdec_lines = 0, 0, [], []
    avmdec, dav2d = anchor("avmdec"), anchor("dav2d-hbd", "dav2d")
    print("| stream | result | first error |\n|---|---|---|")
    for path in paths:
        avm, err, rc = stream_md5([avmdec, "--rawvideo", "--md5", str(path)])
        ref, whole, err2, rc2 = frame_hashes(lambda d: [dav2d, "-q", "--threads", "1", "-i", str(path),
                                                        "--muxer", "frameyuv", "-o", f"{d}/f_%n.yuv"])
        if rc or not avm:
            print(f"| {path.name} | avmdec fails | avmdec rc={rc} {err} |")
            continue
        if rc2 or whole != avm:
            # The pinned dav2d predates some test vectors; avmdec (AVM
            # v1.0.0) is the reference alone there, for the whole stream.
            md5, our_err, our_rc = ours_md5(path)
            ok = our_rc == 0 and md5 == avm
            avmdec_only += ok
            if ok:
                avmdec_lines.append(f"{path.name} {' '.join(ours(path)[0])}\n")
            # A stream we refuse (its error, a nonzero exit) is rejected, not
            # a mismatch, whatever pictures came before the refusal.
            result = "bit-exact" if ok else "MISMATCH" if our_rc == 0 else "rejected"
            print(f"| {path.name} | {result} (avmdec only: dav2d rc={rc2} disagrees) | {our_err or err2} |")
            continue
        ok, result, error = verdict(ref, *ours(path))
        passing += ok
        if ok:
            lines.append(f"{path.name} {' '.join(ref)}\n")
        print(f"| {path.name} | {result} | {error} |")
    print(f"\n{passing} of {len(paths)} bit-exact with avmdec and dav2d; "
          f"{avmdec_only} more with avmdec where the pinned dav2d disagrees")
    if write:
        HASHES.write_text("# Bit-exact with avmdec v1.0.0 and dav2d 237229e (compare.py --write-hashes).\n"
                          + "".join(lines)
                          + "# avmdec v1.0.0 only: the pinned dav2d 237229e decodes these differently;\n"
                          + "# later dav2d agrees with avmdec (docs/handoff/av2.md).\n"
                          + "".join(avmdec_lines))


if __name__ == "__main__":
    main()
