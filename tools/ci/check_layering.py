#!/usr/bin/env python3
"""Enforce the one-way crate layering from CLAUDE.md.

Each workspace crate has a layer. A crate may depend only on crates in a
strictly lower layer, so codec crates can use the shared engine, but the
engine can never reach a codec (and codecs never depend on each other).

    0  nc-common
    1  nc-dsp
    2  nc-analysis
    3  nc-encoder
    4  nc-h264, nc-hevc, nc-av1, nc-vp9, nc-av2
    5  nc-capi, nc-cli, nc-checkasm   (front ends and tools)

A new crate must be added to LAYERS, or this check fails, so the layering is
always a deliberate decision.
"""

import json
import subprocess
import sys

LAYERS = {
    "nc-common": 0,
    "nc-dsp": 1,
    "nc-analysis": 2,
    "nc-encoder": 3,
    "nc-h264": 4,
    "nc-hevc": 4,
    "nc-vvc": 4,
    "nc-av1": 4,
    "nc-vp9": 4,
    "nc-av2": 4,
    "nc-capi": 5,
    "nc-cli": 5,
    "nc-web": 5,
    "nc-checkasm": 5,
}


def main() -> int:
    metadata = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        check=True, capture_output=True, text=True).stdout)
    workspace = {p["name"]: p for p in metadata["packages"]}
    errors = []
    for name, package in sorted(workspace.items()):
        if name not in LAYERS:
            errors.append(f"{name}: not assigned a layer in tools/ci/check_layering.py")
            continue
        for dep in package["dependencies"]:
            target = dep["name"]
            if target not in workspace:
                continue  # external crate
            if LAYERS.get(target, 99) >= LAYERS[name]:
                errors.append(f"{name} (layer {LAYERS[name]}) must not depend on "
                              f"{target} (layer {LAYERS.get(target, '?')})")
    for e in errors:
        print(f"layering: {e}")
    if not errors:
        print(f"layering ok ({len(workspace)} crates)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
