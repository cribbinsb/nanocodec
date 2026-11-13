#!/usr/bin/env python3
"""The bitrate chart re-measured on a large held-out clip set: encodes rented, scoring here.

    python3 tools/dev/aws_heldout_chart.py select --clipset FILE [--count 72]
    python3 tools/dev/aws_heldout_chart.py run --clipset FILE [--instance-type c7a.metal-48xl]
    python3 tools/dev/aws_heldout_chart.py score --clipset FILE [--minutes 25]
    python3 tools/dev/aws_heldout_chart.py page --clipset FILE --out docs/results/heldout-chart-DATE

An independent check of the published chart (docs/results/bd-vs-speed.md)
against overfitting: every chart family, ours and the reference encoders,
on the chart's real-use CBR ladder and rules (nceval.chart: 150 frames, one
key frame, ladder within 50% of the source rate, the low rungs) on clips no
tuning or chart decision has used.

  select  ~COUNT clips of the screening pool's stored encodes (clips/pool.json)
          whose source family is in no raw clip set (main's TRAIN/VAL/TEST and
          the chart's VAL clips, the broad VAL set, the layer-gate fits, long,
          adv12), clean sources only (nceval.pool rules), one per family, at
          least MIN_RUNGS ladder rungs, spread over every class and dataset;
          written as a clip set (eval/nceval/clips.py format).
  run     every chart point (ladder and low rungs, every clip) encoded on one
          large spot instance (tools/dev/aws_session.py): the encoders and the
          clips' stored encodes go up once, the sources are decoded there, our
          encoder's displayed reconstruction is reduced to per-frame digests,
          and only bitstreams, rate logs, coded-frame times and digests come
          back (--streams). Fail-fast stages: readiness (tools run, our version
          stamp, every clip decodes to its size), a pipe-cleaner (one point per
          family), then the rest, slowest first, watched for stalls and errors.
          Two clips' remote decodes are checked against local decodes.
  score   the same points scored here by the shared point engine
          (nceval.realuse.Runner) from those streams (cbr.Bench.prebuilt: the
          anchors' path, metrics from a standard decoder on the GPU slots, our
          decodes proved equal to the encoder's pictures), cached like any
          chart point. The ladder first, then the low rungs on the clips that need
          them; each run stops starting points after --minutes and a rerun resumes.
  page    BD-J vs x264 medium per family on the held-out set (nceval.chart's own
          quality_points) beside the published chart's, ranks and gaps, flags
          moves over FLAG_POINTS / FLAG_RANKS; writes OUT.md and OUT.csv.

`--families`, `--clips N` and `--rungs I,J` cut every stage to a pipe-cleaner
(e.g. 3 families x 2 clips x 2 rungs on a small instance) with the same code
path as the full run.

The independent test (docs/results/independent-*.md) runs the same stages with
`--build DIR` (a scoring build of the commit under test instead of the
chart's), `--references` (more reference families by name, e.g. x264-faster)
and `--extras N`: on N clips spread over the classes (`extras_subset`), the
audit's variants (EXTRA_VARIANTS: our Default of each codec with strict
buffer settings and with frame hold off; the anchors' ANCHOR_PROFILES) on
the ladder and low rungs, and every family on the HIGH_LADDER_KBPS rungs.
The analysis is eval/nceval/independent.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "eval"))
from aws_session import say  # noqa: E402
from nceval.cbr import Bench, Point  # noqa: E402
from nceval.cbr import Family  # noqa: E402
from nceval.chart import (ANCHOR, OURS, REFERENCES, chart_families, low_ladder,  # noqa: E402
                          low_rung_clips, quality_points)
from nceval.clipframes import decode_command  # noqa: E402
from nceval.clips import Clip, load_clipset  # noqa: E402
from nceval.config import load_paths  # noqa: E402
from nceval.gatepolicy import EVALUATION_GATE  # noqa: E402
from nceval.pool import POOL_SETS, SEALED_SETS, clean_source, family_key, load_pool  # noqa: E402
from nceval.realuse import (DEFAULT_FRAMES, LADDER_KBPS, Runner,  # noqa: E402
                            clipset_clips)
from nceval.cbr import (ANCHOR_PRESET_PREFIXES, PROFILE_SEPARATOR, clip_targets,  # noqa: E402
                        preset_anchor)

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
BUILDS_FILE = REPO / "docs/results/bd-vs-speed-builds.json"
TIMING_X86_FILE = REPO / "docs/results/bd-vs-speed-timing-x86.json"
PUBLISHED_CSV = REPO / "docs/results/bd-vs-speed.csv"
BUILDS_ROOT = DATA / "charts/builds"
SCORING_BINARY = "nanocodec"
DEFAULT_STREAMS = DATA / "work/heldout-chart/streams"
#: The cloud scoring's result files as they came back (also the next session's seed).
CLOUD_RESULTS = DATA / "work/heldout-chart/cloud-results"
#: The stored-encode clip set the held-out clips come from, and the raw sets
#: whose source families they must avoid (every TRAIN/VAL/TEST and chart clip).
SOURCE_SET = "pool"
RAW_SETS = tuple(s for s in POOL_SETS if s != SOURCE_SET)
SELECT_SEED = "heldout-chart"
MIN_RUNGS = 3
#: Instance choices: (region, cores used for encodes = vCPUs minus two).
INSTANCES = {"c7a.metal-48xl": ("eu-north-1", 190), "c7a.48xlarge": ("eu-north-1", 190),
             "c7a.24xlarge": ("eu-north-1", 94), "c7a.4xlarge": ("eu-north-1", 14)}
REMOTE_DIR = "heldout"
#: The decoded sources live in a tmpfs of their own (the instance's RAM, never its disk):
#: not /dev/shm, whose files logind removes when the user's last SSH session ends.
REMOTE_CLIPS = "/mnt/clips"
#: The tmpfs's size over the decoded sources' total.
TMPFS_MARGIN = 1.2
#: Stage limits (minutes) and the failures a full run may record before it stops.
PREPARE_MINUTES = 10
PIPE_MINUTES = 5
RUN_MINUTES = 60
MAX_FAILED_SHARE = 0.01
#: A scoring run stops starting points after this long (the 30-minute run budget),
#: checking between batches of this many points per worker.
SCORE_MINUTES = 25
SCORE_BATCH_PER_WORKER = 1
#: The cloud score stage's stall limit: a shard reports after each batch, ~1.5 min
#: on a saturated machine (2026-11-12: batches of 4 per worker stayed silent for 5 min,
#: a false stall).
SCORE_STALL_MINUTES = 10
#: Cloud scoring runs in rounds of this length, each round's results brought back
#: before the next, and the marker file (in the results directory) that tracks them.
CHECKPOINT_MINUTES = 10
CHECKPOINT_MARKER = ".checkpoint"
#: Cloud scoring: the CPU metrics backend, one shard (a `score` of one clip) per
#: clip with this many workers, the machine's cores as the CPU slots; the
#: pipe-cleaner's workers; where streams and results live under the data root.
#: The work root (point cache, scratch) is a tmpfs: nceval.diskguard wants 200 GiB
#: free, which a minimal root volume never has, and nothing there must outlive the
#: instance but the result files that come back.
REMOTE_WORK = "/mnt/work"
REMOTE_WORK_GB = 220
METRICS_ENV = f"NC_METRICS_BACKEND=libvmaf-cpu NC_CPU_SLOTS=$(nproc) NC_WORK={REMOTE_WORK}"
SHARD_WORKERS = 8
#: At most this many shards run at once (several clips each beyond it).
MAX_SHARDS = 96
PIPE_WORKERS = 16
STREAMS_REL = "work/heldout-chart/streams"
RESULTS_REL = "realuse"
REPO_ARCHIVE = "repo.tar"
SEED_REL = "work/heldout-chart/seed"
SCORE_ROOT_GB = 24
#: Anchor tools scoring never runs (a CUDA build, a formatter).
UNNEEDED_TOOLS = ("vmaf-cuda", "clang-format")
#: Clips whose remote decode is compared with a local one, every run.
DECODE_CHECKS = 2
#: The independent test's extras (`--extras`): our Default of every codec with
#: these --param overrides (after the preset's), each a family `<family>+<name>`:
#: strict keeps every frame, the first key included, inside the strict-CBR
#: receiver's 1 s buffer (re-code at 1 buffer, keys capped at 1 buffer, a key's
#: debt repaid within 1 s); nohold codes every input frame (no frame hold).
STRICT_PARAMS = ("rate.recode_overflow=1", "rate.recode_overflow_inter=1",
                 "rate.key_cap_buffer=1", "rate.key_repay_seconds=1")
EXTRA_VARIANTS = {"strict": STRICT_PARAMS, "nohold": ("hold.enable=0",)}
OWN_DEFAULTS = ("nanocodec-h264", "nanocodec-hevc", "nanocodec-av1", "nanocodec-vp9",
                "nanocodec-av2", "nanocodec-vvc")
#: The anchors in other settings than the chart's: (family, cbr.ANCHOR_PROFILES name),
#: each a family `<family>~<profile>`.
ANCHOR_VARIANTS = (("x264-medium", "lookahead"), ("x264-medium", "threads4"),
                   ("x265-medium", "lookahead"), ("x265-medium", "threads4"),
                   ("svtav1-m8", "hier"), ("svtav1-m12", "hier"),
                   ("libvpx-vp9-cpu7", "webrtc"), ("libvpx-vp9-cpu2", "good"),
                   ("libaom-av1-cpu8", "webrtc"), ("openh264-baseline", "aq"))
#: Rungs above the real-use ladder (720p30 kbps, scaled like it; each within the
#: clip's 50%-of-source cap), on the extras' clips: how gaps move with rate.
HIGH_LADDER_KBPS = (2400, 4000)
#: Page flags: a family whose BD-J vs x264 medium moves by more than this many
#: percentage points, or whose rank moves by more than this many places.
FLAG_POINTS = 2.0
FLAG_RANKS = 2


# --- clip selection ----------------------------------------------------------------------

def select(args: argparse.Namespace) -> None:
    """`--source-set` (default the pool) with its clean, usable clips, one per family
    outside every raw and sealed set; `--include` puts an earlier clip set's clips
    first (the independent test: yesterday's held-out clips, then the fresh set)."""
    paths = load_paths()
    source = args.source_set
    pool_names = {e["name"] for e in json.loads((paths.clips / f"{source}.json")
                                                .read_text())["clips"]}
    excluded = set()
    for name in (*RAW_SETS, *SEALED_SETS):
        if (paths.clips / f"{name}.json").exists():
            excluded |= {family_key(c.family) for c in
                         load_clipset(paths.clips / f"{name}.json", paths.data_root).clips}
    usable = lambda clip: len(clip_targets(clip, LADDER_KBPS)) >= MIN_RUNGS  # noqa: E731
    clips = load_pool(paths, usable) if source == SOURCE_SET else \
        [c for c in load_clipset(paths.clips / f"{source}.json", paths.data_root).clips
         if usable(c)]
    candidates = [c for c in clips
                  if c.name in pool_names and family_key(c.family) not in excluded
                  and clean_source(c)]
    chosen = spread(candidates, args.count)
    records = {e["name"]: e for e in json.loads((paths.clips / f"{source}.json")
                                                .read_text())["clips"]}
    included = json.loads(Path(args.include).read_text())["clips"] if args.include else []
    out = Path(args.clipset)
    out.write_text(json.dumps({"name": out.stem, "clips": included
                               + [records[c.name] for c in chosen]}, indent=1) + "\n")
    by_class: dict[str, int] = {}
    for clip in chosen:
        by_class[clip.content_class] = by_class.get(clip.content_class, 0) + 1
    print(f"{len(chosen)} of {len(candidates)} candidates, {len({c.dataset for c in chosen})} "
          f"datasets, by class {by_class}; wrote {out}")


def spread(candidates: list[Clip], count: int) -> list[Clip]:
    """One clip per family, round robin over the classes and, within a class, over its
    datasets, in a fixed hashed order (reproducible)."""
    rank = lambda c: hashlib.sha256(f"{SELECT_SEED}/{c.name}".encode()).hexdigest()  # noqa
    queues: dict[str, dict[str, list[Clip]]] = {}
    for clip in sorted(candidates, key=rank):
        queues.setdefault(clip.content_class, {}).setdefault(clip.dataset, []).append(clip)
    chosen, families = [], set()
    while len(chosen) < count and any(any(q.values()) for q in queues.values()):
        for by_dataset in queues.values():
            for dataset in sorted(by_dataset, key=lambda d: -len(by_dataset[d])):
                queue = by_dataset[dataset]
                while queue and queue[0].family in families:
                    queue.pop(0)
                if queue:
                    clip = queue.pop(0)
                    chosen.append(clip)
                    families.add(clip.family)
                    break
            if len(chosen) == count:
                break
    return sorted(chosen, key=lambda c: c.name)


# --- the points -----------------------------------------------------------------------

def chart_build(override: str | None = None) -> Path:
    """The scoring build: `override` (--build) or the published chart's single build."""
    if override:
        return Path(override).resolve()
    ids = set(json.loads(BUILDS_FILE.read_text())["families"].values())
    if len(ids) != 1:
        raise SystemExit(f"the chart's families are on several builds {sorted(ids)}")
    return BUILDS_ROOT / ids.pop()


def high_ladder(clip: Clip) -> list[int]:
    return clip_targets(clip, HIGH_LADDER_KBPS)


def extras_subset(clips: list[Clip], count: int) -> list[Clip]:
    """`count` clips, round robin over the classes, each class in a fixed hashed order."""
    rank = lambda c: hashlib.sha256(f"{SELECT_SEED}/extras/{c.name}".encode()).hexdigest()  # noqa
    queues: dict[str, list[Clip]] = {}
    for clip in sorted(clips, key=rank):
        queues.setdefault(clip.content_class, []).append(clip)
    chosen: list[Clip] = []
    while len(chosen) < count and any(queues.values()):
        for name in sorted(queues):
            if queues[name] and len(chosen) < count:
                chosen.append(queues[name].pop(0))
    return sorted(chosen, key=lambda c: c.name)


def extra_families(families: list[Family]) -> list[Family]:
    """The audit's variants (EXTRA_VARIANTS of OWN_DEFAULTS, ANCHOR_VARIANTS)."""
    by_name = {f.name: f for f in families}
    paths = load_paths()
    variants = [replace(by_name[name], name=f"{name}+{variant}",
                        extra=by_name[name].extra + tuple(x for p in params
                                                          for x in ("--param", p)))
                for name in OWN_DEFAULTS if name in by_name
                for variant, params in EXTRA_VARIANTS.items()]
    for name, profile in ANCHOR_VARIANTS:
        full = f"{name}{PROFILE_SEPARATOR}{profile}"
        if name in by_name:
            variants.append(replace(by_name[name], name=full, profile=profile))
        elif name.startswith(tuple(ANCHOR_PRESET_PREFIXES.values())):  # a preset not run
            variants.append(preset_anchor(full, paths))
    return variants


@dataclass
class Setup:
    """The run's families (the anchor first) and clips, the runner that scores
    from --streams, and the extras (`--extras`): their clips and variant families."""
    families: list[Family]
    clips: list[Clip]
    runner: Runner
    extra_clips: list[Clip]
    variants: list[Family]

    def extras(self) -> list:
        """The variants' ladder and low rungs, and every family's high rungs, on the
        extras' clips (the main families' ladder and low rungs are `ladder`/`low_rungs`)."""
        if not self.extra_clips:
            return []
        return (self.runner.points(self.variants, self.extra_clips, DEFAULT_FRAMES, None)
                + self.runner.points(self.variants, self.extra_clips, DEFAULT_FRAMES, None,
                                     ladder=low_ladder)
                + self.runner.points(self.families + self.variants, self.extra_clips,
                                     DEFAULT_FRAMES, None, ladder=high_ladder))

    def ladder(self, rungs: tuple[int, ...] | None, clips: list[Clip] | None = None) -> list:
        return self.runner.points(self.families, self.clips if clips is None else clips,
                                  DEFAULT_FRAMES, rungs)

    def low_rungs(self, clips: list[Clip] | None = None) -> list:
        return self.runner.points(self.families, self.clips if clips is None else clips,
                                  DEFAULT_FRAMES, None, ladder=low_ladder)


def setup(args: argparse.Namespace, workers: int = 1) -> Setup:
    build = chart_build(args.build)
    names = args.families.split(",") if args.families else None
    if args.references:
        names = (names or [n for n, _ in OURS + REFERENCES]) + args.references.split(",")
    families = chart_families(build / SCORING_BINARY, build / "presets", names)
    all_clips = clipset_clips(args.clipset, load_paths())
    clips = all_clips[:args.clips]
    extra_clips = extras_subset(clips, args.extras) if args.extras else []
    if getattr(args, "session", None):  # scoring sessions' clips (I/N, comma-separated)
        parts = [tuple(int(x) for x in part.split("/")) for part in args.session.split(",")]
        chosen = {c.name for index, count in parts for c in clips[index - 1::count]}
        clips = [c for c in clips if c.name in chosen]
        extra_clips = [c for c in extra_clips if c in clips]
    if getattr(args, "shard", None):
        index, count = (int(x) for x in args.shard.split("/"))
        clips = clips[index - 1::count]
        extra_clips = [c for c in extra_clips if c in clips]
    if any(c.start_frame for c in clips):
        raise SystemExit("held-out clips are measured from frame 0")
    return Setup(families, clips, Runner(load_paths(), EVALUATION_GATE, workers,
                                         over_budget=True, prebuilt=Path(args.streams)),
                 extra_clips, extra_families(families) if args.extras else [])


def remote_job(bench: Bench, point: Point) -> dict:
    """One remote encode: argv relative to the remote directory, outputs under out/."""
    key = bench.prebuilt_key(point)
    family = point.family
    remote = replace(point, clip=replace(point.clip, yuv=Path(f"{REMOTE_CLIPS}/"
                                                              f"{point.clip.name}.yuv")))
    argv = bench.command(remote, Path(f"out/{key}{family.codec.bitstream_suffix}"),
                         "{RECON}" if family.own else None,
                         Path(f"out/{key}.rate.csv") if family.own and point.kbps else None,
                         Path(f"out/{key}.times.txt") if family.own else None)
    argv = [f"bin/{family.tool.name}", *argv[1:]]
    if any(str(Path.home()) in a for a in argv):
        raise SystemExit(f"{key}: a local path in the remote command: {shlex.join(argv)}")
    return {"id": key, "argv": argv, "recon": {
        "frame_bytes": point.clip.frame_bytes, "frames": bench.frames} if family.own else None}


def slowest_first(jobs: list) -> list:
    """The jobs in descending order of expected encode time (the stored x86 time of the
    family x the clip's pixel count), so all cores finish together."""
    stored = json.loads(TIMING_X86_FILE.read_text())["points"]
    cost = lambda job: (sum(stored.get(job[1].family.name, {"ms_per_frame": {"x": 1e3}})
                            ["ms_per_frame"].values())
                        * job[1].clip.width * job[1].clip.height)  # noqa: E731
    return sorted(jobs, key=cost, reverse=True)


# --- run (remote encodes) ----------------------------------------------------------------

def stage(stage_dir: Path, jobs: list, clips: list[Clip], tools: dict[str, Path]) -> None:
    (stage_dir / "bin").mkdir(parents=True)
    (stage_dir / "clips").mkdir()
    (stage_dir / "out").mkdir()
    for name, path in tools.items():
        (stage_dir / "bin" / name).symlink_to(path.resolve())
    for clip in clips:
        (stage_dir / "clips" / clip.yuv.name).symlink_to(clip.yuv.resolve())
    (stage_dir / "remote_jobs.py").symlink_to(Path(__file__).with_name("remote_jobs.py"))
    ordered = [remote_job(b, p) for b, p in slowest_first(jobs)]
    first_clip = {}
    for bench, point in sorted(jobs, key=lambda j: (j[1].kbps or 0)):  # one quick job a family
        first_clip.setdefault(point.family.name, remote_job(bench, point))
    (stage_dir / "jobs.json").write_text(json.dumps(ordered) + "\n")
    (stage_dir / "pipe.json").write_text(json.dumps(list(first_clip.values())) + "\n")
    decodes = [(c.name, c.yuv.name, min(DEFAULT_FRAMES, c.frames), c.frame_bytes) for c in clips]
    (stage_dir / "decode.txt").write_text("".join(f"{n} {f} {k} {b}\n" for n, f, k, b in decodes))


def unstage(stage_dir: Path) -> None:
    """Remove the staging directory entry by entry (symlinks and small job lists)."""
    for sub in ("bin", "clips", "out"):
        for entry in (stage_dir / sub).iterdir():
            entry.unlink()
        (stage_dir / sub).rmdir()
    for name in ("remote_jobs.py", "jobs.json", "pipe.json", "decode.txt"):
        (stage_dir / name).unlink(missing_ok=True)
    stage_dir.rmdir()


def prepare_command(version: str, tools: list[str], clips: list[Clip]) -> str:
    """Readiness: ffmpeg and xxhash installed, every tool runs (our version stamp
    matches), every clip decodes (in parallel, to RAM) to exactly its size."""
    decode = (f"ffmpeg -nostdin -v error -i clips/$2 -frames:v $3 -f rawvideo -pix_fmt yuv420p "
              f"{REMOTE_CLIPS}/$1.yuv && test $(stat -c %s {REMOTE_CLIPS}/$1.yuv) = $(($3 * $4))"
              f" && sha256sum {REMOTE_CLIPS}/$1.yuv")
    return " && ".join([
        f"cd ~/{REMOTE_DIR}",
        "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq update",
        "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg python3-xxhash "
        "> /dev/null",
        "! ldd bin/* | grep 'not found'",
        f"test \"$(bin/nanocodec --version)\" = {shlex.quote(version)}",
        *[f"bin/{t} --version > /dev/null" for t in tools if t in ("x264", "x265",
                                                                     "SvtAv1EncApp")],
        f"sudo mkdir -p {REMOTE_CLIPS}",
        f"sudo mount -t tmpfs -o size={tmpfs_mb(clips)}m,mode=1777 tmpfs {REMOTE_CLIPS}",
        f"xargs -P $(nproc) -L 1 bash -c {shlex.quote(decode)} _ < decode.txt "
        "> out/decoded.sha256",
        "echo decoded $(wc -l < out/decoded.sha256) of $(wc -l < decode.txt) clips",
        "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs | tee out/platform.txt",
    ])


def tmpfs_mb(clips: list[Clip]) -> int:
    total = sum(min(DEFAULT_FRAMES, c.frames) * c.frame_bytes for c in clips)
    return round(total * TMPFS_MARGIN / 1e6) + 64


def check_decodes(streams: Path, clips: list[Clip]) -> None:
    """The remote decodes of the first DECODE_CHECKS clips equal local decodes."""
    remote = {Path(line.split()[1]).stem: line.split()[0]
              for line in (streams / "decoded.sha256").read_text().splitlines()}
    for clip in clips[:DECODE_CHECKS]:
        command = decode_command(clip, min(DEFAULT_FRAMES, clip.frames))
        local = hashlib.sha256(subprocess.run(command, capture_output=True, check=True)
                               .stdout).hexdigest()
        if local != remote.get(clip.name):
            raise SystemExit(f"{clip.name}: the remote decode differs from the local one")
    say(f"remote decodes of {DECODE_CHECKS} clips equal local decodes")


def run(args: argparse.Namespace) -> None:
    from aws_session import Session, SessionSpec  # boto3: only where sessions start

    region, workers = INSTANCES[args.instance_type]
    run_setup = setup(args)
    clips = run_setup.clips
    # every clip's low rungs too: which clips need them is only known once scored here
    jobs = run_setup.ladder(args.rungs) + run_setup.low_rungs() + run_setup.extras()
    streams = Path(args.streams)
    # a rerun encodes only what an earlier, stopped session did not bring back
    todo = [j for j in jobs if not Path(f"{streams / j[0].prebuilt_key(j[1])}.done").exists()]
    if len(todo) < len(jobs):
        say(f"{len(jobs) - len(todo)} of {len(jobs)} encodes already here: skipped")
        jobs = todo
    tools = {p.family.tool.name: p.family.tool for _, p in jobs}
    version = subprocess.run([str(tools["nanocodec"]), "--version"], capture_output=True,
                             text=True, check=True).stdout.strip() if "nanocodec" in tools else ""
    stage_dir = streams.parent / f"stage-{time.strftime('%Y%m%d-%H%M%S')}"
    stage(stage_dir, jobs, clips, tools)
    try:
        upload_mb = sum(p.resolve().stat().st_size for p in stage_dir.rglob("*")
                        if p.is_symlink() or p.is_file()) / 1e6
        say(f"{len(jobs)} encodes, {len(clips)} clips, {len(tools)} tools, "
            f"upload {upload_mb:.0f} MB")
        stages = [("prepare", prepare_command(version, list(tools), clips), PREPARE_MINUTES, 0),
                  ("pipe-cleaner", f"cd ~/{REMOTE_DIR} && python3 remote_jobs.py --jobs "
                   f"pipe.json --workers {workers}", PIPE_MINUTES, 0),
                  ("encode", f"cd ~/{REMOTE_DIR} && python3 remote_jobs.py --jobs jobs.json "
                   f"--workers {workers}", args.minutes,
                   max(2, round(MAX_FAILED_SHARE * len(jobs))))]
        if args.dry_run:
            for name, command, minutes, _ in stages:
                print(f"{name} ({minutes} min): {command[:600]}")
            return
        spec = SessionSpec(region=region, instance_type=args.instance_type, arch="amd64",
                           purpose=args.purpose or "heldout-chart", lifetime_minutes=args.lifetime,
                           budget_usd=args.budget, root_gb=args.root_gb)
        timings = {}
        with Session(spec) as session:
            started = time.monotonic()
            session.upload(stage_dir, f"~/{REMOTE_DIR}")
            timings["upload"] = time.monotonic() - started
            try:
                for name, command, minutes, max_errors in stages:
                    started = time.monotonic()
                    say(f"remote: {name}")
                    session.run_detached(command, f"~/{REMOTE_DIR}/{name}.log", minutes,
                                         max_errors=max_errors)
                    timings[name] = time.monotonic() - started
            finally:  # whatever finished comes back before the instance goes
                started = time.monotonic()
                session.download(f"~/{REMOTE_DIR}/out", streams)
                timings["download"] = time.monotonic() - started
                say("stage minutes: " + ", ".join(f"{k} {v / 60:.1f}"
                                                  for k, v in timings.items()))
    finally:
        unstage(stage_dir)
    check_decodes(streams, clips)
    failed = sorted(streams.rglob("*.failed"))
    say(f"{len(failed)} failed encodes" + "".join(f"\n  {p.relative_to(streams)}"
                                                  for p in failed[:20]))


# --- score and page (here) ---------------------------------------------------------------

def score(args: argparse.Namespace) -> None:
    """The real-use ladder first; once it is all scored, the low rungs on the clips
    that need them (the chart's rule, nceval.chart.low_rung_clips). Stops starting
    points after --minutes; a rerun resumes (scored points are cached)."""
    run = setup(args, args.workers)
    uncached = lambda jobs: [j for j in jobs if not j[0].cached(j[1])]  # noqa: E731
    deadline = time.monotonic() + args.minutes * 60
    for phase in args.phases.split(","):
        if phase == "ladder":
            todo = uncached(run.ladder(args.rungs))
        elif phase == "extras":
            todo = uncached(run.extras())
        elif uncached(run.ladder(args.rungs)):
            break  # which clips need low rungs is judged on the whole ladder
        else:
            curves = run.runner.measure("held-out ladder", run.families, run.clips,
                                        DEFAULT_FRAMES, args.rungs, budgeted=False)
            low = low_rung_clips(curves, run.families, run.clips)
            say(f"low rungs on {len(low)} of {len(run.clips)} clips")
            todo = uncached(run.low_rungs(low))
        if not score_points(run.runner, todo, phase, args.workers, deadline):
            break
    say(f"refused: {run.runner.refused or 'none'}")


def score_points(runner: Runner, todo: list, phase: str, workers: int,
                 deadline: float) -> bool:
    """Score `todo` in batches until done or past `deadline`; whether all were scored."""
    say(f"{phase}: {len(todo)} points to score on {workers} workers")
    started, done = time.monotonic(), 0
    batch = workers * SCORE_BATCH_PER_WORKER
    while done < len(todo) and time.monotonic() < deadline:
        runner.run_jobs(todo[done:done + batch])
        done = min(len(todo), done + batch)
        minutes = (time.monotonic() - started) / 60
        say(f"{phase}: {done}/{len(todo)} scored, {done / minutes:.0f} points/min, "
            f"ETA {(len(todo) - done) / done * minutes:.0f} min")
    say(f"{phase}: {len(todo) - done} points left"
        + ("" if done == len(todo) else " (run score again)"))
    return done == len(todo)


# --- score-cloud (scoring rented too) ------------------------------------------------------

def score_cloud(args: argparse.Namespace) -> None:
    """`score` on a rented machine: the streams, clips, tools, chart build and this
    repository's eval code go up (inbound is free), one `score` shard per clip runs
    there with the CPU metrics backend (nceval.metrics: libvmaf + nceval.msssim,
    equal to VMAFx at its stored precision, and the backend is in no cache key),
    and only the result files come back, into this machine's point cache."""
    from aws_session import Session, SessionSpec  # boto3: only where sessions start

    region, cores = INSTANCES[args.instance_type]
    CLOUD_RESULTS.mkdir(parents=True, exist_ok=True)  # the seed (none on a first session)
    run = setup(args)
    if subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "eval", "tools"],
                      capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit("commit eval/ and tools/ first: the remote runs this repository's HEAD")
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"  # sessions run side by side
    stage_dir = DATA / "work/heldout-chart" / f"score-stage-{stamp}"
    try:
        stage_score(stage_dir, args, run.clips, run.families)
    except BaseException:
        unstage_score(stage_dir)
        raise
    remote_clipset = f"~/nanocodec-data/clips/{Path(args.clipset).name}"
    score_command = (f"cd ~/repo && {METRICS_ENV} python3 tools/dev/aws_heldout_chart.py score "
                     f"--clipset {remote_clipset} --streams ~/nanocodec-data/{STREAMS_REL}"
                     + remote_options(args))
    shards = min(len(run.clips), MAX_SHARDS)
    # rounds of CHECKPOINT_MINUTES, each brought back before the next (a spot reclaim
    # loses at most one round: 2026-11-13, two sessions lost 50 min of points)
    shard = (f"set -o pipefail; {score_command} --shard $1/{shards} --workers "
             f"{SHARD_WORKERS} --minutes {CHECKPOINT_MINUTES} 2>&1 | sed -u \"s/^/[$1] /\" "
             f"|| echo \"FAILED shard $1\"")
    stages = [
        ("prepare", " && ".join([
            f"mkdir -p ~/repo && tar -C ~/repo -xf ~/{REPO_ARCHIVE}",
            f"sudo mkdir -p {REMOTE_WORK}",
            f"sudo mount -t tmpfs -o size={REMOTE_WORK_GB}g,mode=1777 tmpfs {REMOTE_WORK}",
            # results an earlier, stopped session brought back: those points are cached
            f"mkdir -p {REMOTE_WORK}/{RESULTS_REL}",
            f"cp -r ~/nanocodec-data/{SEED_REL}/. {REMOTE_WORK}/{RESULTS_REL}/",
            "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq update",
            "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg python3-numpy "
            "python3-scipy python3-opencv python3-xxhash > /dev/null",
            "! ldd ~/nanocodec-data/anchors/x86_64/bin/* | grep 'not found'",
            "~/nanocodec-data/anchors/x86_64/bin/vmaf --version",
            f"cd ~/repo && {METRICS_ENV} python3 -c 'import sys; sys.path.insert(0, \"eval\"); "
            "import nceval.chart, nceval.metrics as m; print(m.metrics_backend())'",
            "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs"]), PREPARE_MINUTES),
        # every family on the first clip's first rung: every tool, decoder and the whole
        # scoring path, before the shards
        ("pipe-cleaner", f"{score_command} --clips 1 --rungs 0 --phases ladder "
         f"--workers {PIPE_WORKERS}", PIPE_MINUTES),
    ]
    score_round = (f"seq 1 {shards} | xargs -P {shards} -I{{}} bash -c "
                   f"{shlex.quote(shard)} _ {{}}")
    if args.dry_run:
        for name, command, minutes in stages:
            print(f"{name} ({minutes} min): {command}")
        print(f"score rounds ({CHECKPOINT_MINUTES} min each, to {args.minutes} min): {score_round}")
        unstage_score(stage_dir)
        return
    spec = SessionSpec(region=region, instance_type=args.instance_type, arch="amd64",
                       purpose=args.purpose or "heldout-score", lifetime_minutes=args.lifetime,
                       budget_usd=args.budget, root_gb=SCORE_ROOT_GB)
    results = CLOUD_RESULTS
    results.mkdir(parents=True, exist_ok=True)
    timings = {}
    try:
        with Session(spec) as session:
            started = time.monotonic()
            session.upload(stage_dir, "~")
            timings["upload"] = time.monotonic() - started
            try:
                for name, command, minutes in stages:
                    started = time.monotonic()
                    say(f"remote: {name}")
                    session.run_detached(command, f"~/{name}.log", minutes,
                                         stall_minutes=SCORE_STALL_MINUTES)
                    timings[name] = time.monotonic() - started
                score_rounds(session, score_round, args.minutes, results)
            finally:
                started = time.monotonic()
                session.download_new(f"{REMOTE_WORK}/{RESULTS_REL}", results, CHECKPOINT_MARKER)
                timings["download"] = time.monotonic() - started
                say("stage minutes: " + ", ".join(f"{k} {v / 60:.1f}" for k, v in
                                                  timings.items()))
    finally:
        unstage_score(stage_dir)
    merge_results(results, load_paths().work / RESULTS_REL)


def score_rounds(session, command: str, minutes: float, results: Path) -> None:
    """Score in rounds until no shard has points left or `minutes` have passed, each
    round's new results brought back and merged here (a checkpoint)."""
    started = time.monotonic()
    for round_index in itertools.count(1):
        log = f"~/score-{round_index}.log"
        say(f"remote: score round {round_index}")
        session.run_detached(command, log, CHECKPOINT_MINUTES + PIPE_MINUTES,
                             stall_minutes=SCORE_STALL_MINUTES)
        moved = session.download_new(f"{REMOTE_WORK}/{RESULTS_REL}", results, CHECKPOINT_MARKER)
        merge_results(results, load_paths().work / RESULTS_REL)
        left = session.ssh(f"grep -c 'run score again' {log} || true", quiet=True,
                           check=False).stdout.strip()
        say(f"round {round_index}: {moved / 1e6:.0f} MB back, {left or 0} shard phases unfinished")
        if left in ("", "0") or time.monotonic() - started > minutes * 60:
            return


def remote_options(args: argparse.Namespace) -> str:
    """The run's own options for the remote `score` (the build at its remote path)."""
    options = ""
    if args.build:
        options += f" --build ~/nanocodec-data/charts/builds/{chart_build(args.build).name}"
    if args.references:
        options += f" --references {shlex.quote(args.references)}"
    if args.families:
        options += f" --families {shlex.quote(args.families)}"
    if args.clips:
        options += f" --clips {args.clips}"
    if args.extras:
        options += f" --extras {args.extras}"
    if args.session:
        options += f" --session {args.session}"
    return options


def stage_score(stage_dir: Path, args: argparse.Namespace, clips: list[Clip],
                families: list[Family]) -> None:
    """The remote home: repo.tar (this repository at HEAD: eval code, this tool, the
    chart's records, the anchors' pinned versions) and, as symlinks, ~/nanocodec-data
    (clip set, stored clips, anchor tools, chart build, streams)."""
    data = stage_dir / "nanocodec-data"
    stage_dir.mkdir(parents=True)
    subprocess.run(["git", "-C", str(REPO), "archive", "-o", str(stage_dir / REPO_ARCHIVE),
                    "HEAD"], check=True)
    links = {data / "clips" / Path(args.clipset).name: Path(args.clipset),
             data / "anchors/x86_64/share/vmaf": DATA / "anchors/x86_64/share/vmaf",
             data / "charts/builds" / chart_build(args.build).name: chart_build(args.build),
             data / STREAMS_REL: Path(args.streams),
             data / SEED_REL: CLOUD_RESULTS}
    for clip in clips:
        links[data / clip.yuv.relative_to(DATA)] = clip.yuv
    # every family's tool where the families expect it (openh264's lives in the work dir)
    for family in families:
        if family.tool.is_relative_to(DATA) and \
                not family.tool.is_relative_to(chart_build(args.build)):
            links[data / family.tool.relative_to(DATA)] = family.tool
    for tool in (DATA / "anchors/x86_64/bin").iterdir():
        if tool.name not in UNNEEDED_TOOLS:
            links[data / "anchors/x86_64/bin" / tool.name] = tool
    # the manifest first: a staging that fails half-way is still removed
    (stage_dir / "manifest.txt").write_text("".join(f"{p}\n" for p in links))
    for link, target in links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target.resolve())


def unstage_score(stage_dir: Path) -> None:
    """Remove the staging tree: the archive, its symlinks (listed in the manifest),
    then its directories."""
    manifest = stage_dir / "manifest.txt"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            Path(line).unlink(missing_ok=True)
        manifest.unlink()
    (stage_dir / REPO_ARCHIVE).unlink(missing_ok=True)
    for directory in sorted((p for p in stage_dir.rglob("*") if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
        directory.rmdir()
    stage_dir.rmdir()


def merge_results(results: Path, cache: Path) -> None:
    """Every remote result file into the local point cache; reports how many points
    the cache already held and whether those were bit-identical."""
    same = differ = new = 0
    for remote in sorted(results.rglob("*.json")):
        local = cache / remote.relative_to(results)
        if local.exists():
            if local.read_bytes() == remote.read_bytes():
                same += 1
                continue
            differ += 1
        else:
            new += 1
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(remote.read_bytes())
    say(f"merged {new} new results; of {same + differ} already scored here, {same} were "
        f"bit-identical and {differ} differed (replaced by the cloud's)")


def published() -> dict[str, dict]:
    return {row["family"]: row for row in csv.DictReader(PUBLISHED_CSV.open())}


def ranks(values: dict[str, float | None]) -> dict[str, int]:
    order = sorted((v, k) for k, v in values.items() if v is not None)
    return {k: i + 1 for i, (_, k) in enumerate(order)}


def page(args: argparse.Namespace) -> None:
    run = setup(args, args.workers)
    clips = run.clips
    quality = quality_points(run.runner, run.families, clips, DEFAULT_FRAMES, args.rungs)
    chart = published()
    held = {p.family: p for p in quality.points}
    held_rank = ranks({n: p.bd_j for n, p in held.items()})
    chart_rank = ranks({n: float(chart[n]["bd_j_pct"]) if n in chart and chart[n]["bd_j_pct"]
                        else None for n in held})
    before = {n: float(chart[n]["bd_j_pct"]) if n in chart and chart[n]["bd_j_pct"] else None
              for n in held}
    gaps = {n: p.bd_j - before[n] for n, p in held.items()
            if p.bd_j is not None and before[n] is not None and n != ANCHOR}
    # every BD is against x264 medium, so a clip set on which the anchor does
    # better moves every gap alike; the overfitting signal is a move apart from it
    common = statistics.median(gaps.values())
    rows = []
    for name, point in held.items():
        gap = gaps.get(name)
        relative = gap - common if gap is not None else None
        moved = (held_rank.get(name, 0) - chart_rank[name]) if name in chart_rank and \
            name in held_rank else None
        flag = (relative is not None and abs(relative) > FLAG_POINTS) or \
            (moved is not None and abs(moved) > FLAG_RANKS)
        rows.append({"family": name, "label": point.label, "ours": int(point.ours),
                     "bd_j_heldout_pct": fmt(point.bd_j), "bd_j_chart_pct": fmt(before[name]),
                     "delta_pts": fmt(gap), "delta_vs_median_pts": fmt(relative),
                     "rank_heldout": held_rank.get(name, ""),
                     "rank_chart": chart_rank.get(name, ""), "rank_move": moved if moved
                     is not None else "", "bd_psnr_heldout_pct": fmt(point.bd_psnr),
                     "bd_vmaf_heldout_pct": fmt(point.bd_vmaf),
                     "clips_scored": point.scored, "clips": point.clips,
                     "flag": "FLAG" if flag else ""})
    rows.sort(key=lambda r: float(r["bd_j_heldout_pct"]) if r["bd_j_heldout_pct"] else 1e9)
    out = Path(args.out)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    out.with_suffix(".csv").write_text(buffer.getvalue())
    out.with_suffix(".md").write_text(page_text(rows, clips, quality, args, common))
    print(out.with_suffix(".md").read_text())


def fmt(value: float | None) -> str:
    return f"{value:+.2f}" if value is not None else ""


def shift_summary(rows: list[dict], ours: bool) -> str:
    moves = [float(r["delta_vs_median_pts"]) for r in rows
             if r["ours"] == int(ours) and r["delta_vs_median_pts"]]
    return (f"median {statistics.median(moves):+.2f} (range {min(moves):+.2f} to "
            f"{max(moves):+.2f})")


def page_text(rows: list[dict], clips: list[Clip], quality, args, common: float) -> str:
    classes: dict[str, int] = {}
    for clip in clips:
        classes[clip.content_class] = classes.get(clip.content_class, 0) + 1
    flagged = [r for r in rows if r["flag"]]
    lines = [
        f"# Held-out chart check ({Path(args.out).name.removeprefix('heldout-chart-')})", "",
        "The published bitrate chart ([bd-vs-speed.md](bd-vs-speed.md)) re-measured on "
        f"{len(clips)} clips that no tuning, screen gate or chart decision has used, as a "
        "check against overfitting. Same families, builds, real-use CBR ladder (150 frames, "
        "one key frame, rungs within 50% of the source rate, the low rungs), the same "
        "BD-J vs x264 medium (nceval.chart), encoded on a rented machine and scored here "
        "(`tools/dev/aws_heldout_chart.py`).", "",
        f"Clips: {len(clips)} of the screening pool's stored encodes whose source family is "
        "in no raw clip set (TRAIN, VAL, TEST, the chart's VAL clips, adv12), one per "
        f"family, {len({c.dataset for c in clips})} datasets; by class "
        + ", ".join(f"{k} {v}" for k, v in sorted(classes.items())) + ".", "",
        f"Every family's BD-J vs x264 medium moved by a median of {common:+.2f} points: "
        "on these clips the anchor itself does relatively better, which moves every gap "
        "alike. The overfitting signal is a move apart from that common shift (\"vs "
        f"median\"): ours {shift_summary(rows, True)}, the reference encoders "
        f"{shift_summary(rows, False)}.", "",
        f"Flag: a move more than {FLAG_POINTS:g} points from the common shift, or a rank "
        f"move of more than {FLAG_RANKS} places. Flagged: "
        + (", ".join(f"{r['label']} ({r['delta_vs_median_pts']} pts vs median, rank "
                     f"{r['rank_chart']} -> {r['rank_heldout']})" for r in flagged)
           or "none") + ".", "",
        "| Family | BD-J held-out | BD-J chart | Δ pts | Δ vs median | rank held-out | "
        "rank chart | BD-PSNR | BD-VMAF | clips | flag |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['bd_j_heldout_pct']} | {r['bd_j_chart_pct']} | "
                     f"{r['delta_pts']} | {r['delta_vs_median_pts']} | "
                     f"{r['rank_heldout']} | {r['rank_chart']} | "
                     f"{r['bd_psnr_heldout_pct']} | {r['bd_vmaf_heldout_pct']} | "
                     f"{r['clips_scored']}/{r['clips']} | {r['flag']} |")
    if quality.excluded:
        lines += ["", "Clips left out for every point (no direct overlap with the anchor): "
                  + ", ".join(e.clip for e in quality.excluded) + "."]
    lines += ["", "Clips: " + ", ".join(c.name for c in clips) + ".", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("--clipset", required=True, help="the clip set file to write")
    s.add_argument("--count", type=int, default=72)
    s.add_argument("--source-set", default=SOURCE_SET, help="the clip set to draw from")
    s.add_argument("--include", help="a clip set file whose clips come first")
    for name in ("run", "score", "score-cloud", "page"):
        p = sub.add_parser(name)
        p.add_argument("--clipset", required=True)
        p.add_argument("--streams", default=str(DEFAULT_STREAMS),
                       help="where the encodes land (and are scored from)")
        p.add_argument("--families", help="comma-separated chart families (default: all)")
        p.add_argument("--clips", type=int, help="the first N clips only")
        p.add_argument("--rungs", type=lambda v: tuple(int(i) for i in v.split(",")),
                       help="real-use ladder indices (default: all)")
        p.add_argument("--workers", type=int, default=8, help="local scoring workers")
        p.add_argument("--build", help="the scoring build directory (nanocodec, presets/; "
                       "default: the published chart's build)")
        p.add_argument("--references", help="more reference families by name (preset anchors)")
        p.add_argument("--extras", type=int, default=0,
                       help="the audit's extras on this many clips (0: none)")
    r = sub.choices["run"]
    r.add_argument("--instance-type", default="c7a.metal-48xl", choices=sorted(INSTANCES))
    r.add_argument("--budget", type=float, default=15.0)
    r.add_argument("--lifetime", type=int, default=90, help="hard instance lifetime, min")
    r.add_argument("--root-gb", type=int, default=16)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--purpose", help="the session's Purpose tag (default heldout-chart)")
    r.add_argument("--minutes", type=float, default=RUN_MINUTES,
                   help="the encode stage's limit (minutes)")
    sc = sub.choices["score"]
    sc.add_argument("--minutes", type=float, default=SCORE_MINUTES,
                    help="start no point after this long (resumable)")
    sc.add_argument("--phases", default="ladder,low,extras",
                    help="ladder, low (rungs) and/or extras (--extras)")
    sc.add_argument("--shard", help="I/N: every Nth clip from the Ith")
    sc.add_argument("--session", help="I/N: the clips of score-cloud's session I of N")
    c = sub.choices["score-cloud"]
    c.add_argument("--instance-type", default="c7a.metal-48xl", choices=sorted(INSTANCES))
    c.add_argument("--budget", type=float, default=15.0)
    c.add_argument("--lifetime", type=int, default=90, help="hard instance lifetime, min")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--purpose", help="the session's Purpose tag (default heldout-score)")
    c.add_argument("--session", help="I/N[,J/N]: this session scores every Nth clip from the "
                   "Ith (and Jth) (sessions run side by side)")
    c.add_argument("--minutes", type=float, default=RUN_MINUTES,
                   help="each shard starts no point after this long (resumable)")
    sub.choices["page"].add_argument("--out", required=True,
                                     help="output stem (OUT.md, OUT.csv)")
    args = parser.parse_args(argv)
    {"select": select, "run": run, "score": score, "score-cloud": score_cloud,
     "page": page}[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
