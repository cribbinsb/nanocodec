#!/usr/bin/env python3
"""Hold nanocodec's AV2 frame-header parse against dav2d's, checkpoint by
checkpoint (docs/handoff/av2.md "Other encoders' streams").

    tools/dev/av2_foreign/header_trace.py [STREAM...]

dav2d built with DEBUG_FRAME_HDR (DAV2D_TRACE, default
~/nanocodec-data/work/av2-dec/dav2d-debug/build-hdr/tools/dav2d) prints
`HDR: post-<field>[values]: off=<bit offset>` at each checkpoint of
`parse_frame_hdr`; `cargo run -p nanocodec-av2 --example dump_headers --
--trace` (DUMP_HEADERS, a built example binary) prints the same checkpoints
of our parse. Both are reduced to `OBU type=N` and `HDR: <checkpoint>:
off=N` lines and compared; the first difference of each stream is printed.
Without streams, the whole corpus of compare.py.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import corpus  # noqa: E402

HOME = Path.home()
DAV2D = Path(os.environ.get("DAV2D_TRACE", HOME / "nanocodec-data/work/av2-dec/dav2d-debug/build-hdr/tools/dav2d"))
DUMP = Path(os.environ.get("DUMP_HEADERS", HOME / "nanocodec-data/targets/av2-dec2/gate/examples/dump_headers"))
HDR = re.compile(r"^HDR: (post-[^\[:]*).*: off=(\d+)$")
OBU = re.compile(r"^OBU type=(\d+)")
# The reference slots of each rank, as dav2d prints them (`post-refs`, and
# `post-refs2` once ranked with the frame size) and as dump_headers does.
DAV2D_REFS = re.compile(r"^HDR: post-refs2?\[(?:explicit:\d+,)?refs:([\d,]+)\]")
REFS = re.compile(r"^REFS: ([\d,]+)$")


def reduce(text):
    """The OBU types, checkpoints and offsets, and each frame's final
    reference slots (after its checkpoints)."""
    lines, refs = [], None
    for line in text.splitlines() + ["OBU type=end"]:
        if m := DAV2D_REFS.match(line):
            refs = m[1]
        if m := HDR.match(line):
            lines.append(f"HDR: {m[1]}: off={m[2]}")
        elif m := REFS.match(line):
            refs = m[1]
        elif m := OBU.match(line):
            if refs is not None:
                lines.append(f"REFS: {refs}")
                refs = None
            lines.append(f"OBU type={m[1]}")
    return lines[:-1]


def main():
    paths = [Path(a) for a in sys.argv[1:]] or corpus()
    agreed = 0
    for path in paths:
        theirs = reduce(subprocess.run([str(DAV2D), "-i", str(path), "-o", "/dev/null"],
                                       capture_output=True, text=True).stdout)
        run = subprocess.run([str(DUMP), "--trace", str(path)], capture_output=True, text=True)
        ours = reduce(run.stdout)
        error = run.stderr.strip().splitlines()[-1:] if run.returncode else []
        first = next((i for i, (a, b) in enumerate(zip(theirs, ours)) if a != b), None)
        checkpoints = sum(line.startswith("HDR") for line in theirs)
        if first is None and len(ours) >= len(theirs):
            agreed += 1
            print(f"{path.name}: {checkpoints} checkpoints agree")
        elif first is None:
            print(f"{path.name}: ours stops after {len(ours)} of {len(theirs)} lines {error}")
        else:
            print(f"{path.name}: line {first}: dav2d `{theirs[first]}` ours `{ours[first]}`")
    print(f"\n{agreed} of {len(paths)} streams: every frame-header checkpoint agrees with dav2d")


if __name__ == "__main__":
    main()
