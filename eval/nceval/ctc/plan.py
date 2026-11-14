"""Turn the conditions into a job list for `ctc/worker.py`, and stage a run directory.

A job is one encode at one quantiser point (or, for our CBR extra, one target
rate), with the command lines the worker runs, all relative to the run
directory: tools in `bin/`, JVET configuration files in `cfg/`, sources in
`SOURCE_DIR`, outputs under `out/`. Command lines come from the same adapters
the rest of the harness uses (`nceval.encoders`) and the conformance gate's
decoder table (`nceval.conformance.DECODERS`), so no flag is restated here.

Placeholders the worker fills: `{RECON}` / `{DISPLAY}` (our coded and shown
pictures, into pipes), `{BITSTREAM}` / `{OUT}` (a decoder's input and output
pipe), `{KBPS}` (a CBR job's target).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ..clips import Clip
from ..codecs import Codec
from ..config import Paths
from ..conformance import DECODERS
from ..dav2d_crop import CODED_MULTIPLE, dav2d_command
from ..encoders.base import CodingStructure, EncodeJob
from .conditions import CONDITIONS, POINTS, Condition, CtcEncoder
from .reference_encoders import HM_CONFIG, VTM_CONFIG, VVENC_CONFIG
from .sequences import Sequence, by_name

#: Raw sources on the worker (a tmpfs the worker mounts).
SOURCE_DIR = Path("/mnt/ctc")
#: Bitstreams live outside out/ (what checkpoints copy home): the worker deletes
#: each one when its job ends, which would race with a checkpoint's file list.
STREAMS = Path("streams")
#: The CBR extra's buffer, the project's evaluation setting (eval/nceval/cbr.py BUFFER_MS).
CBR_BUFFER_MS = 1000
DAV2D_CROP_SCRIPT = "dav2d_crop.py"
JVET_CONFIG_FILES = (HM_CONFIG, VTM_CONFIG, VVENC_CONFIG)
#: Relative CPU per pixel-frame of a job (encode + scoring), a fast encoder = 1:
#: measured at 270p on our PC (2026-11-14: VMAF ~17 s per 130 frames, HM ~180 s,
#: VTM and libaom speed 0 ~4 s per frame). Used to start the slowest jobs first
#: and to balance shards.
COST = {"vtm-ldp": 41.0, "aom-good-s0-ld": 39.0, "hm-ldp": 10.0, "vvenc-faster-ld": 2.0,
        "x265-medium": 1.3, "kvazaar-medium": 1.2}
DEFAULT_COST = 1.0


def clip_for(sequence: Sequence, frames: int, source_dir: Path) -> Clip:
    """The harness's clip record for a sequence's raw source on the worker."""
    return Clip(name=sequence.name, yuv=source_dir / f"{sequence.name}.yuv",
                width=sequence.width, height=sequence.height, fps=sequence.fps, frames=frames,
                source_kbps=0.0, content_class=sequence.sequence_class, split="ctc")


def job_id(condition: Condition, sequence: Sequence, encoder: CtcEncoder, point: str) -> str:
    return f"{condition.name}/{sequence.name}/{encoder.name}/{point}"


def remote_tool(argv: list[str]) -> list[str]:
    """`argv` with its executable taken from the run directory's `bin/`."""
    return [f"bin/{Path(argv[0]).name}", *argv[1:]]


def decoder_argv(decoder, codec: Codec, sequence: Sequence, paths: Paths) -> list[str]:
    """A standard decoder's command with `{BITSTREAM}` / `{OUT}` placeholders.

    The gate's commands for JM and dav2d look into the bitstream to choose a
    form; ours have one picture size, so the choice is made from the size."""
    if decoder.tool == "ldecod":
        return ["bin/ldecod", "-i", "{BITSTREAM}", "-o", "{OUT}"]
    if decoder.tool == "dav2d":
        if sequence.width % CODED_MULTIPLE or sequence.height % CODED_MULTIPLE:
            return ["python3", "-I", DAV2D_CROP_SCRIPT, "bin/dav2d", "{BITSTREAM}", "{OUT}"]
        return dav2d_command("bin/dav2d", "{BITSTREAM}", "{OUT}")
    return remote_tool(decoder.command("{BITSTREAM}", "{OUT}", codec, paths))


def encode_argv(condition: Condition, sequence: Sequence, encoder: CtcEncoder, quantizer: int,
                bitstream: Path, paths: Paths, source_dir: Path) -> list[str]:
    job = EncodeJob(clip=clip_for(sequence, condition.frames, source_dir), quantizer=quantizer,
                    structure=CodingStructure(intra_period_frames=condition.frames),
                    frames=condition.frames, bitstream=bitstream,
                    recon=Path("{DISPLAY}") if encoder.own else None)
    argv = remote_tool(encoder.adapter.command(job, paths))
    return argv + ["--recon", "{RECON}"] if encoder.own else argv


def to_cbr(argv: list[str]) -> list[str]:
    """Our fixed-QP command turned into the CBR extra's (`--bitrate`, 1 s buffer)."""
    at = argv.index("--qp")
    return [*argv[:at], "--bitrate", "{KBPS}", "--buffer-ms", str(CBR_BUFFER_MS), *argv[at + 2:]]


def jobs_for(condition: Condition, paths: Paths, source_dir: Path) -> list[dict]:
    jobs = []
    for sequence in condition.sequences:
        for encoder in condition.encoders:
            if not condition.runs(encoder, sequence):
                continue
            decoders = DECODERS[encoder.codec] if encoder.own else DECODERS[encoder.codec][:1]
            common = {"condition": condition.name, "sequence": sequence.name,
                      "encoder": encoder.name, "own": encoder.own,
                      "source": str(source_dir / f"{sequence.name}.yuv"),
                      "frames": condition.frames, "width": sequence.width,
                      "height": sequence.height,
                      "frame_bytes": sequence.frame_bytes,
                      "fps": [sequence.fps_num, sequence.fps_den],
                      "container": "ivf" if encoder.codec.bitstream_suffix == ".ivf" else "annexb",
                      "decoders": [decoder_argv(d, encoder.codec, sequence, paths)
                                   for d in decoders],
                      "cost": COST.get(encoder.name, DEFAULT_COST) * sequence.width
                      * sequence.height * condition.frames}
            for quantizer in POINTS[encoder.scale]:
                key = job_id(condition, sequence, encoder, f"q{quantizer}")
                bitstream = STREAMS / f"{key}{encoder.codec.bitstream_suffix}"
                argv = encode_argv(condition, sequence, encoder, quantizer, bitstream, paths,
                                   source_dir)
                jobs.append({**common, "id": key, "kind": "qp", "point": quantizer,
                             "bitstream": str(bitstream), "encode": argv})
                if condition.cbr_extra and encoder.own:
                    cbr_key = job_id(condition, sequence, encoder, f"cbr{quantizer}")
                    cbr_stream = STREAMS / f"{cbr_key}{encoder.codec.bitstream_suffix}"
                    jobs.append({**common, "id": cbr_key, "kind": "cbr", "point": quantizer,
                                 "rate_of": key, "bitstream": str(cbr_stream),
                                 "encode": to_cbr(encode_argv(condition, sequence, encoder,
                                                              quantizer, cbr_stream, paths,
                                                              source_dir))})
    return jobs


def plan(condition_names: list[str], paths: Paths, source_dir: Path = SOURCE_DIR,
         sequences: list[str] | None = None, encoders: list[str] | None = None,
         points: int | None = None) -> list[dict]:
    """Every job of the named conditions, optionally narrowed (the pipe-clean's subset:
    some sequences, some encoders, the first `points` quantiser points of each)."""
    jobs = []
    for name in condition_names:
        condition = CONDITIONS[name]
        if sequences:
            condition = replace(condition, sequences=tuple(
                s for s in condition.sequences if s.name in sequences))
        if encoders:
            condition = replace(condition, encoders=tuple(
                e for e in condition.encoders if e.name in encoders))
        jobs += jobs_for(condition, paths, source_dir)
    if points is not None:
        jobs = [j for j in jobs if _point_rank(j) < points]
    return sorted(jobs, key=lambda j: (j["kind"] == "cbr", -j["cost"]))


def shard(jobs: list[dict], index: int, count: int) -> list[dict]:
    """Shard `index` (1-based) of `count`, balanced on cost: whole (condition,
    sequence, encoder) groups, so a CBR job stays with its fixed-QP twin, each
    group given in turn (costliest first) to the least-loaded shard."""
    groups: dict[tuple, list[dict]] = {}
    for job in jobs:
        groups.setdefault((job["condition"], job["sequence"], job["encoder"]), []).append(job)
    loads = [0.0] * count
    chosen = []
    for key in sorted(groups, key=lambda k: (-sum(j["cost"] for j in groups[k]), k)):
        target = loads.index(min(loads))
        loads[target] += sum(j["cost"] for j in groups[key])
        if target == index - 1:
            chosen += groups[key]
    return sorted(chosen, key=lambda j: (j["kind"] == "cbr", -j["cost"]))


def _point_rank(job: dict) -> int:
    condition = CONDITIONS[job["condition"]]
    return POINTS[condition.encoder(job["encoder"]).scale].index(job["point"])


def tools_of(jobs: list[dict]) -> set[str]:
    """The executables the jobs name in `bin/`."""
    names = set()
    for job in jobs:
        for argv in [job["encode"], *job["decoders"]]:
            names |= {Path(a).name for a in argv if a.startswith("bin/")}
    return names


def source_manifest(jobs: list[dict]) -> list[dict]:
    """What the worker downloads: each sequence once, with the most frames any job reads."""
    frames: dict[str, int] = {}
    for job in jobs:
        frames[job["sequence"]] = max(frames.get(job["sequence"], 0), job["frames"])
    return [{"name": s.name, "url": s.url, "md5": s.md5, "frames": frames[s.name],
             "frame_bytes": s.frame_bytes,
             "raw": next(j["source"] for j in jobs if j["sequence"] == s.name)}
            for s in (by_name(name) for name in sorted(frames))]


def stage(run_dir: Path, jobs: list[dict], tool_paths: dict[str, Path],
          files: list[Path]) -> None:
    """Write a run directory: jobs.json, sources.json, bin/ (symlinks), cfg/ and
    `files` (the worker, the dav2d crop wrapper, the VMAF v1 model)."""
    (run_dir / "bin").mkdir(parents=True)
    (run_dir / "cfg").mkdir()
    for name in sorted(tools_of(jobs)) + ["vmaf"]:
        (run_dir / "bin" / name).symlink_to(tool_paths[name].resolve())
    for config in JVET_CONFIG_FILES:
        if config.name in {Path(a).name for job in jobs for a in job["encode"]}:
            (run_dir / "cfg" / config.name).symlink_to(tool_paths[config.name].resolve())
    for extra in files:
        (run_dir / extra.name).symlink_to(extra.resolve())
    (run_dir / "jobs.json").write_text(json.dumps(jobs, indent=1) + "\n")
    (run_dir / "sources.json").write_text(json.dumps(source_manifest(jobs), indent=1) + "\n")
