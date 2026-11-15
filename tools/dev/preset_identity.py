#!/usr/bin/env python3
"""Whether two nanocodec builds encode every shipped preset of every codec identically.

The proof an exact refactor needs (docs/PROCEDURES.md "Exact changes"): for each
codec, its Default and every `presets/<codec>-*.toml` (built into each binary),
on three 720p clips (crowd, static, pan), in CBR at two rates (short
key period, so key frames and their rate-control paths recur) and at fixed QP,
the two builds' streams are compared by SHA-256. Streams are written to a
temporary directory and deleted as soon as they are hashed; no YUVs are written.

    python3 tools/dev/preset_identity.py OLD_BINARY NEW_BINARY [--codecs h264,av1]
        [--frames N] [--jobs N]

Exit status 0 when every stream is identical, 1 otherwise (the differing
encodes are listed).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLIP_DIR = Path.home() / "nanocodec-data/clips/main"
#: One clip per motion regime the engine branches on (moving crowd, fixed camera, pan).
CLIPS = ("crowd_mot17_09", "static_otw_homes390", "pan_bdd_00c12bd0")
WIDTH, HEIGHT, FPS = 1280, 720, 30
CODECS = ("h264", "hevc", "vvc", "av1", "vp9", "av2")
#: Rate-controlled modes (kbps, key period) and the fixed-QP mode's quantizer.
CBR_MODES = ((300, 300), (1200, 20))
FIXED_QP = 32


def presets(codec: str, preset_dir: Path) -> list[str | None]:
    """Default (None) and every shipped preset of `codec` by name."""
    names = sorted(p.stem for p in preset_dir.glob(f"{codec}-*.toml"))
    return [None, *names]


def commands(codec: str, preset: str | None, frames: int) -> list[tuple[str, list[str]]]:
    """(label, arguments) of every checked encode of one preset."""
    base = ["encode", "--codec", codec, "--width", str(WIDTH), "--height", str(HEIGHT),
            "--fps", str(FPS), "--frames", str(frames)]
    if preset:
        base += ["--preset", preset]
    out = []
    for clip in CLIPS:
        src = ["--input", str(CLIP_DIR / f"{clip}.yuv")]
        for kbps, period in CBR_MODES:
            out.append((f"{clip} cbr{kbps}/k{period}", base + src + [
                "--bitrate", str(kbps), "--buffer-ms", "1000", "--intra-period", str(period)]))
        out.append((f"{clip} qp{FIXED_QP}", base + src + ["--qp", str(FIXED_QP),
                                                          "--intra-period", "300"]))
    return out


def stream_hash(binary: Path, args: list[str], work: Path) -> str:
    """SHA-256 of the stream `binary` encodes with `args` (or the failure)."""
    out = Path(tempfile.mkstemp(dir=work, suffix=".bin")[1])
    try:
        cmd = [str(binary), *args, "--output", str(out)]
        run = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if run.returncode:
            return "FAILED: " + run.stderr.strip()[-200:]
        return hashlib.sha256(out.read_bytes()).hexdigest()
    finally:
        out.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--codecs", default=",".join(CODECS))
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    jobs = []
    for codec in a.codecs.split(","):
        names = presets(codec, REPO / "presets")
        for preset in names:
            for label, args in commands(codec, preset, a.frames):
                jobs.append((f"{codec} {preset or 'default'} {label}", args))
    with tempfile.TemporaryDirectory(prefix="preset-identity-") as tmp:
        work = Path(tmp)
        with ThreadPoolExecutor(a.jobs) as pool:
            old = pool.map(lambda j: stream_hash(a.old, j[1], work), jobs)
            new = pool.map(lambda j: stream_hash(a.new, j[1], work), jobs)
            pairs = list(zip(jobs, old, new))
    differ = [(name, o, n) for (name, _), o, n in pairs if o != n or o.startswith("FAILED")]
    for name, o, n in differ:
        print(f"DIFF {name}: {o[:16]} vs {n[:16]}")
    print(f"{len(pairs) - len(differ)}/{len(pairs)} streams identical")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
