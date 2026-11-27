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
        [--frames N] [--jobs N] [--old-names] [--renamed OLD=NEW ...] [--added STEM ...]
        [--clip-dir DIR]

`--old-names`: OLD_BINARY predates the speed ladder (2026-11-15) and is given
each preset by its codec-prefixed name (`h264-fastest` for H.264 `veryfast`,
`av1-svc3-fast` for AV1 `--scalability L3T1 --preset fast`).

`--renamed OLD=NEW` (repeatable): a rung renamed between the builds (2026-11-21:
`slow=slower`); OLD_BINARY codes the new tree's NEW file under its OLD name. A new
tree's file named OLD is then a rung OLD_BINARY does not have: it is listed as new,
not compared, and the count of checked encodes leaves it out.

`--added STEM` (repeatable): a rung the new tree adds (`slow`, 2026-11-21) is listed as
new and not compared, as above.

`--old-env K=V` / `--new-env K=V` (repeatable): environment for one side only. With the
same binary on both sides this checks that DSP tiers code identical streams
(tools/ci/isa_identity.sh: `NC_DSP_ISA=scalar` against `avx2`, `avx2` against `avx512`;
`neon` on aarch64), which every timing machine relies on (docs/benchmark-procedure.md).

`--strip-metadata`: each stream loses its SEI NAL units and metadata OBUs
(`nceval.streamstrip`, as the benchmark strips every encoder's) before it is
hashed, so a change that adds or alters only informative units (the encoder
information message) is shown to leave the picture data identical; the bytes
removed are summed per codec and side. `--new-param NAME=VALUE` (repeatable)
adds `--param NAME=VALUE` to NEW_BINARY's encodes only (e.g.
`stream.encoder_info=0`).

`--clip-dir`: where the clips are (`<clip>.yuv`, 1280x720 I420), for a machine
without the PC's clip sets (the Jetson keeps crowd_mot17_09 in clips/quick).
A clip missing there is left out, and the count of checked encodes says so.

Exit status 0 when every stream is identical, 1 otherwise (the differing
encodes are listed).

It runs on the build box by default (owner, 2026-11-21: the PC carries no material
compute): `nc-remote run` syncs this tree, both builds and the clips, runs it there
with `--local`, and prints its output; `--local` runs it here.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import streamstrip, tempdirs  # noqa: E402
from nceval.codecs import Codec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NC_REMOTE = Path.home() / "nanocodec-data/bin/nc-remote"
#: Set in every nc-remote job (tools/dev/nc_remote_job.py).
ON_BUILD_BOX_ENV = "NC_REMOTE_BOX"
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


def renamed_stem(stem: str | None, renamed: dict[str, str]) -> str | None:
    """The stem OLD_BINARY knows the new tree's `stem` by (`--renamed`), layers kept."""
    if stem is None:
        return None
    layers, dash, speed = stem.rpartition("-")
    old = {new: old for old, new in renamed.items()}.get(speed, speed)
    return f"{layers}{dash}{old}"


def presets(codec: str, preset_dir: Path) -> list[str | None]:
    """Default (None) and every shipped preset file of `codec` by stem."""
    names = sorted(p.stem for p in (preset_dir / codec).glob("*.toml"))
    return [None, *names]


def commands(codec: str, preset: list[str], frames: int,
             clip_dir: Path = CLIP_DIR) -> list[tuple[str, list[str]]]:
    """(label, arguments) of every checked encode of one preset (its CLI arguments)."""
    base = ["encode", "--codec", codec, "--width", str(WIDTH), "--height", str(HEIGHT),
            "--fps", str(FPS), "--frames", str(frames), *preset]
    out = []
    for clip in (c for c in CLIPS if (clip_dir / f"{c}.yuv").is_file()):
        src = ["--input", str(clip_dir / f"{clip}.yuv")]
        for kbps, period in CBR_MODES:
            out.append((f"{clip} cbr{kbps}/k{period}", base + src + [
                "--bitrate", str(kbps), "--buffer-ms", "1000", "--intra-period", str(period)]))
        out.append((f"{clip} qp{FIXED_QP}", base + src + ["--qp", str(FIXED_QP),
                                                          "--intra-period", "300"]))
    return out


def stream_hash(binary: Path, args: list[str], work: Path,
                env: dict[str, str] | None = None,
                strip: Codec | None = None) -> tuple[str, int]:
    """SHA-256 of the stream `binary` encodes with `args` (and `env` added), or the
    failure, and the bytes `strip` removed (its SEI / metadata units, before hashing)."""
    out = Path(tempfile.mkstemp(dir=work, suffix=".bin")[1])
    try:
        cmd = [str(binary), *args, "--output", str(out)]
        run = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                             env={**os.environ, **(env or {})})
        if run.returncode:
            return "FAILED: " + run.stderr.strip()[-200:], 0
        removed = streamstrip.strip(out, strip) if strip else 0
        return hashlib.sha256(out.read_bytes()).hexdigest(), removed
    finally:
        out.unlink(missing_ok=True)


def on_build_box(a: argparse.Namespace) -> int:
    """This check on the build box: the tree, both builds and the clips go up
    (nc-remote run --sync / --clip), the same command runs there with --local."""
    builds = [a.old.resolve(), a.new.resolve()]
    if not all(b.is_relative_to(Path.home()) for b in builds):
        raise SystemExit("the builds must be under ~ to go to the build box (or --local)")
    clips = [a.clip_dir.resolve() / f"{clip}.yuv" for clip in CLIPS]
    box = lambda path: "$HOME/" + str(path.relative_to(Path.home()))
    command = ["python3", "tools/dev/preset_identity.py", box(builds[0]), box(builds[1]),
               "--codecs", a.codecs, "--frames", str(a.frames), "--jobs", str(a.jobs),
               "--clip-dir", box(a.clip_dir.resolve()), "--local",
               *(["--old-names"] if a.old_names else []),
               *[x for r in a.renamed for x in ("--renamed", r)],
               *[x for r in a.added for x in ("--added", r)],
               *[x for e in a.old_env for x in ("--old-env", e)],
               *[x for e in a.new_env for x in ("--new-env", e)],
               *[x for p in a.new_param for x in ("--new-param", p)],
               *(["--strip-metadata"] if a.strip_metadata else [])]
    print("preset_identity on the build box (nc-remote; --local runs it here)", flush=True)
    return subprocess.call([str(NC_REMOTE), "run", "--worktree", str(REPO),
                            "--name", f"preset-identity-{REPO.name}",
                            *[x for b in builds for x in ("--sync", str(b))],
                            *[x for c in clips if c.is_file() for x in ("--clip", str(c))],
                            "--", " ".join(shlex.quote(x) if not x.startswith("$HOME") else
                                           f'"{x}"' for x in command)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--codecs", default=",".join(CODECS))
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--old-names", action="store_true",
                    help="OLD_BINARY takes the presets' names from before the ladder")
    ap.add_argument("--renamed", action="append", default=[], metavar="OLD=NEW",
                    help="a rung renamed between the builds: OLD_BINARY codes NEW as OLD")
    ap.add_argument("--added", action="append", default=[], metavar="STEM",
                    help="a rung the new tree adds: listed, not compared")
    ap.add_argument("--old-env", action="append", default=[], metavar="K=V",
                    help="environment for OLD_BINARY's encodes only (e.g. NC_DSP_ISA=scalar)")
    ap.add_argument("--new-env", action="append", default=[], metavar="K=V",
                    help="environment for NEW_BINARY's encodes only")
    ap.add_argument("--new-param", action="append", default=[], metavar="NAME=VALUE",
                    help="a --param for NEW_BINARY's encodes only")
    ap.add_argument("--strip-metadata", action="store_true",
                    help="strip SEI / metadata units (nceval.streamstrip) before hashing")
    ap.add_argument("--clip-dir", type=Path, default=CLIP_DIR,
                    help="the clips' directory (default: the PC's main set)")
    ap.add_argument("--local", action="store_true",
                    help="encode here (default: on the build box, nc-remote)")
    a = ap.parse_args()
    if not a.local and not os.environ.get(ON_BUILD_BOX_ENV):
        return on_build_box(a)

    renamed = dict(r.split("=", 1) for r in a.renamed)
    old_env = dict(e.split("=", 1) for e in a.old_env)
    new_env = dict(e.split("=", 1) for e in a.new_env)
    jobs = []  # (name, old arguments, new arguments)
    added = []  # new rungs OLD_BINARY does not have (`--renamed`, `--added`)
    new_rungs = {*renamed, *a.added}
    for codec in a.codecs.split(","):
        for stem in presets(codec, REPO / "crates/nc-encoder/presets"):
            if stem is not None and stem.rpartition("-")[2] in new_rungs:
                added.append(f"{codec} {stem}")
                continue
            old_stem = renamed_stem(stem, renamed)
            old_args = old_preset_args(codec, old_stem) if a.old_names else preset_args(old_stem)
            old = commands(codec, old_args, a.frames, a.clip_dir)
            new_params = [x for p in a.new_param for x in ("--param", p)]
            new = commands(codec, [*preset_args(stem), *new_params], a.frames, a.clip_dir)
            for (label, o), (_, n) in zip(old, new):
                jobs.append((f"{codec} {stem or 'medium'} {label}", o, n, codec))
    with tempdirs.temp_dir("preset-identity") as tmp:
        work = Path(tmp)
        with ThreadPoolExecutor(a.jobs) as pool:
            strip = (lambda codec: Codec(codec)) if a.strip_metadata else (lambda _: None)
            old = pool.map(lambda j: stream_hash(a.old, j[1], work, old_env, strip(j[3])), jobs)
            new = pool.map(lambda j: stream_hash(a.new, j[2], work, new_env, strip(j[3])), jobs)
            hashed = list(zip(jobs, old, new))
    pairs = [(job, o, n) for job, (o, _), (n, _) in hashed]
    if a.strip_metadata:
        for codec in a.codecs.split(","):
            sides = [sum(r[side][1] for r in hashed if r[0][3] == codec) for side in (1, 2)]
            print(f"{codec}: metadata bytes stripped old {sides[0]}, new {sides[1]}")
    if not pairs:
        print(f"no clip of {', '.join(CLIPS)} in {a.clip_dir}: nothing checked")
        return 1
    differ = [(job[0], o, n) for job, o, n in pairs if o != n or o.startswith("FAILED")]
    for name, o, n in differ:
        print(f"DIFF {name}: {o[:16]} vs {n[:16]}")
    if added:
        print(f"new rungs, not compared: {', '.join(added)}")
    print(f"{len(pairs) - len(differ)}/{len(pairs)} streams identical")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
