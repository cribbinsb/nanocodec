#!/usr/bin/env python3
"""Whether two nanocodec builds encode every shipped preset of every codec identically.

The proof an exact refactor needs (docs/PROCEDURES.md "Exact changes"): for each
codec, its Default (`medium`) and every `presets/<codec>/*.toml` (built into each
binary; `l2-` / `l3-` files coded with `--scalability L2T1` / `L3T1`),
on three 720p clips (crowd, static, pan), in CBR at two rates (short
key period, so key frames and their rate-control paths recur) and at fixed QP,
the two builds' streams are compared by SHA-256. Streams are written to a
temporary directory and deleted as soon as they are hashed; no YUVs are written.

    python3 tools/dev/preset_identity.py OLD_BINARY NEW_BINARY [--codecs h264,av1]
        [--frames N] [--jobs N] [--old-names]

`--old-names`: OLD_BINARY predates the speed ladder (2026-11-15) and is given
each preset by its codec-prefixed name (`h264-fastest` for H.264 `veryfast`,
`av1-svc3-fast` for AV1 `--scalability L3T1 --preset fast`).

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


#: The codec-prefixed name each ladder rung had before the ladder (2026-11-15).
OLD_RUNG_NAMES = {"fast": "fast", "faster": "faster", "veryfast": "fastest"}
#: The codec-prefixed name of each layered file before the ladder.
OLD_LAYERED_NAMES = {"l2-medium": "svc", "l2-fast": "svc-fast", "l3-medium": "svc3",
                     "l3-fast": "svc3-fast"}


def preset_args(stem: str | None) -> list[str]:
    """The CLI arguments of one preset file (None: medium, the Default)."""
    if stem is None:
        return []
    layers, _, speed = stem.rpartition("-")
    scalability = ["--scalability", f"{layers.upper()}T1"] if layers else []
    return [*scalability, "--preset", speed]


def old_preset_args(codec: str, stem: str | None) -> list[str]:
    """The same preset's arguments for a build from before the ladder."""
    if stem is None:
        return []
    old = OLD_LAYERED_NAMES.get(stem) or OLD_RUNG_NAMES[stem]
    return ["--preset", f"{codec}-{old}"]


def presets(codec: str, preset_dir: Path) -> list[str | None]:
    """Default (None) and every shipped preset file of `codec` by stem."""
    names = sorted(p.stem for p in (preset_dir / codec).glob("*.toml"))
    return [None, *names]


def commands(codec: str, preset: list[str], frames: int) -> list[tuple[str, list[str]]]:
    """(label, arguments) of every checked encode of one preset (its CLI arguments)."""
    base = ["encode", "--codec", codec, "--width", str(WIDTH), "--height", str(HEIGHT),
            "--fps", str(FPS), "--frames", str(frames), *preset]
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
    ap.add_argument("--old-names", action="store_true",
                    help="OLD_BINARY takes the presets' names from before the ladder")
    a = ap.parse_args()

    jobs = []  # (name, old arguments, new arguments)
    for codec in a.codecs.split(","):
        for stem in presets(codec, REPO / "presets"):
            old_args = old_preset_args(codec, stem) if a.old_names else preset_args(stem)
            old = commands(codec, old_args, a.frames)
            new = commands(codec, preset_args(stem), a.frames)
            for (label, o), (_, n) in zip(old, new):
                jobs.append((f"{codec} {stem or 'medium'} {label}", o, n))
    with tempfile.TemporaryDirectory(prefix="preset-identity-") as tmp:
        work = Path(tmp)
        with ThreadPoolExecutor(a.jobs) as pool:
            old = pool.map(lambda j: stream_hash(a.old, j[1], work), jobs)
            new = pool.map(lambda j: stream_hash(a.new, j[2], work), jobs)
            pairs = list(zip(jobs, old, new))
    differ = [(job[0], o, n) for job, o, n in pairs if o != n or o.startswith("FAILED")]
    for name, o, n in differ:
        print(f"DIFF {name}: {o[:16]} vs {n[:16]}")
    print(f"{len(pairs) - len(differ)}/{len(pairs)} streams identical")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
