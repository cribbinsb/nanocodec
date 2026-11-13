#!/usr/bin/env python3
"""Pack fuzz inputs (files or directories of files) into a replay archive
on stdout: each input as a little-endian u32 length then its bytes, in
sorted path order; the names go to stderr, one per index."""
import struct
import sys
from pathlib import Path

paths = []
for arg in sys.argv[1:]:
    p = Path(arg)
    paths += sorted(f for f in p.iterdir() if f.is_file()) if p.is_dir() else [p]
out = sys.stdout.buffer
for index, path in enumerate(paths):
    data = path.read_bytes()
    out.write(struct.pack("<I", len(data)) + data)
    print(index, path, file=sys.stderr)
