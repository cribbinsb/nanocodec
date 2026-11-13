"""Frame/slice header fields of any of our codecs, from ffmpeg's trace_headers.

ffmpeg's coded-bitstream layer prints every syntax element it parses; the
header elements (loop filter levels, CDEF strengths, SAO and deblocking
flags) say which in-loop tools a frame enables, decoder-side and for any
encoder. One dict per frame header (AV1) or slice header (H.264/HEVC), in
decode order; parameter-set fields that a slice inherits are kept in
`params`.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Paths, find_tool

# One syntax element: "[trace_headers @ 0x..] <bit pos> <name> <bits> = <value>".
_ELEMENT = re.compile(r"\]\s+\d+\s+(\S+)\s+[01]+\s+=\s+(-?\d+)\s*$")
_SECTION = re.compile(r"\]\s+([A-Z][A-Za-z ]+(?:Header|Set|Segment Header))\s*$")
FRAME_SECTIONS = ("Frame Header", "Slice Header", "Slice Segment Header")
PARAM_SECTIONS = ("Picture Parameter Set", "Sequence Parameter Set", "Sequence Header")


@dataclass
class HeaderTrace:
    frames: list[dict[str, int]] = field(default_factory=list)
    params: dict[str, int] = field(default_factory=dict)


def parse_trace(lines: list[str]) -> HeaderTrace:
    trace, current, section = HeaderTrace(), None, ""
    for line in lines:
        heading = _SECTION.search(line)
        if heading:
            section = heading.group(1).strip()
            current = {} if section in FRAME_SECTIONS else None
            if current is not None:
                trace.frames.append(current)
            continue
        element = _ELEMENT.search(line)
        if not element:
            continue
        name, value = element.group(1), int(element.group(2))
        if current is not None:
            current[name] = value
        elif section in PARAM_SECTIONS:
            trace.params[name] = value
    # AV1 repeats show_existing_frame headers that code nothing.
    trace.frames = [f for f in trace.frames if f.get("show_existing_frame", 0) == 0]
    return trace


def trace_headers(stream: Path, paths: Paths) -> HeaderTrace:
    cmd = [str(find_tool("ffmpeg", paths)), "-nostdin", "-v", "trace", "-i", str(stream),
           "-c", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          check=True)
    return parse_trace([line for line in proc.stderr.splitlines() if "trace_headers" in line])
