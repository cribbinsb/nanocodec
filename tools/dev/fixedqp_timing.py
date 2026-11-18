#!/usr/bin/env python3
"""Encode speed for the results site's Fixed QP charts, timed on the charts' own encodes.

    python3 tools/dev/fixedqp_timing.py plan --binary BIN --out PLAN.json
        [--machine x86|zen3|graviton|jetson] [--expect-version LINE] [--encoders a,b]
        [--sequences a,b] [--points 27,42] [--thin hm-ldp:27,42] [--rounds 1]
        [--shard-minutes 20]
    python3 tools/dev/fixedqp_timing.py plan-pool --binary BIN --gap GAP.json [--gap ...]
        --out PLAN.json
    python3 tools/dev/fixedqp_timing.py plan-from --from TIMED.json --machine zen3 --out PLAN.json
        [--encoders a,b] [--binary BIN]
    python3 tools/dev/fixedqp_timing.py run --plan PLAN.json --out DIR
        [--shards 0,3] [--parallel 6] [--dry-run]
    python3 tools/dev/fixedqp_timing.py run-jetson --plan PLAN.json --out DIR
    python3 tools/dev/fixedqp_timing.py record --plan PLAN.json --out DIR [--plan --out ...]
        --build COMMIT --record JSON
    python3 tools/dev/fixedqp_timing.py merge --record SITE.json --from NEW.json --encoders a,b

The site has two Fixed QP charts. Their quality comes from the fixed-QP lens
(AOM CTC v8 low delay, 12 public sequences, 130 frames, six quantizers per
encoder: nceval.ctc) and from `nceval.fixedqp gap --clipset` on held-out
clips. Their speed axes are timed here on the same encodes: every encoder runs
the quality run's own command lines (nceval.ctc.plan, or the pool engine's
cbr.Bench.command at the quantizers each curve used, so no flag is restated;
our recon outputs are dropped, as in the CBR chart's timing), one encode at a
time pinned to one core, on the x86 timing machine (aws_timing.TIMING_MACHINE,
c8a.large: Zen 5, no SMT, its second core), on the x86 one without AVX-512
(aws_timing.ZEN3_MACHINE, c6a.large: Zen 3, the same way), on the server Arm one
(aws_timing.GRAVITON_MACHINE, c8g.large: Graviton4, the same way, with the Jetson's
aarch64 builds of our encoder and the references) or on the Jetson's timing core
(core 3, under the board's timing lock).

- `plan-from` times another machine's plan again on this one: the same jobs
  (command lines, quantizers, rounds), e.g. the c8a series' on Zen 3.
- `plan` / `plan-pool` write the jobs (one encode each) and pack them into
  shards (an instance, or one turn of the Jetson's lock, each). Every encode's
  time is estimated from the lens's own fps (shared-load PC figures, only for
  packing); an encode estimated under REPEAT_MAX_SECONDS is timed in
  REPEATED_ROUNDS rotated rounds after a warm-up round (the median is kept), a
  slower one once (or every encode `--rounds` times). Shards are filled
  sequence by sequence, so an instance only receives the sequences it codes.
- `run` launches the shards side by side (aws_session: on demand, tagged,
  budgeted, terminated and ledgered on every exit), each uploading its tools,
  its sequences (zstd) and tools/dev/fixedqp_timing_remote.py, and downloads
  out/times.json per shard into DIR/shard-N; `run-jetson` runs them one after
  another on the board with its own tool builds.
- `record` writes the site's timing record: per encoder and sequence the
  CPU milliseconds per frame (sum over the timed quantizers of the median
  encode time, over their frames); the chart's fps is 1000 over their
  geometric mean across sequences, each sequence weighted equally as in the
  quality means.
- `merge` puts the points of encoders re-timed later (a `record` of their own
  plan, same machine and content) into the site's record, each with its build.

Sources: lens sequences as raw I420 `<sequence>.yuv` files (--source-dir, as
`nceval.ctc local`); held-out clips decoded by the harness's own command
(nceval.clipframes.decode_command). Reference tools come from the anchors'
bin directory, HM and VVenC (and their configuration files) from --refs-dir.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

DATA = Path.home() / "nanocodec-data"
DEFAULT_SOURCES = DATA / "work/ctc-improve/src"
ANCHOR_TOOLS = DATA / "anchors/x86_64/bin"
#: The Jetson's aarch64 reference builds, copied here (aws_timing.AARCH64_TOOLS).
AARCH64_TOOLS = DATA / "anchors/aarch64/bin"
#: HM and VVenC as the lens ran them (built from source, BSD; nceval.ctc.reference_encoders).
DEFAULT_REFS = DATA / "work/standards-eval/refs-bin"
REMOTE_SCRIPT = Path(__file__).resolve().with_name("fixedqp_timing_remote.py")
SITE_LENS = REPO / "web/results/data/fixed-qp-lens.json"
CONDITION = "aom-ld"
#: Our encode's outputs the timing leaves out (the recon pipes the lens scores).
OWN_OUTPUT_FLAGS = ("--recon", "--display-recon")
#: Encodes estimated under this many seconds are timed REPEATED_ROUNDS times
#: (median) after a warm-up round; slower ones once (a minutes-long encode's
#: run-to-run spread on a quiet core is far below the differences the chart shows).
REPEAT_MAX_SECONDS = 20.0
REPEATED_ROUNDS = 3
#: The lens's fps is PC CPU time under shared load; c8a's Zen 5 core is about this
#: much faster (the chart's own timing: x264 medium 1.6x, SVT M10 1.95x). Packing only.
ZEN5_OVER_PC = 1.6
#: A Graviton4 core's encode time over a Zen 5 core's (packing only; the pipe-cleaner of
#: 2026-11-17 on Johnny: 1.8-1.9x for AV1 Faster, 2.3-2.7x for SVT-AV1 M12).
GRAVITON_OVER_ZEN5 = 2.0
#: A Zen 3 (c6a, EPYC Milan) core's encode time over a Zen 5 core's (packing only; an estimate:
#: no AVX-512 and an older core).
ZEN3_OVER_ZEN5 = 1.5
#: Estimated minutes of timing per shard (the box boots and uploads in 2-4 more).
SHARD_MINUTES = 20.0
#: The instance's lifetime and its budget's guard, per shard.
LIFETIME_MINUTES = 75
SHARD_BUDGET_USD = 0.5
#: Per machine: the timed core, the time of an encode over its PC estimate (packing only;
#: the chart's own timing: Jetson core about 4x a Zen 5 core's time; Graviton4 GRAVITON_OVER_ZEN5)
#: and, for a rented one, its instance type (aws_timing.INSTANCES), how the record names it and
#: where the timed thread sits (aws_timing.pinning_command's wording).
PHYSICAL_CORE = "one thread pinned to one physical core"
MACHINES = {"x86": {"core": 1, "over_pc": 1 / ZEN5_OVER_PC, "instance": "c8a.large",
                    "aws": "AWS c8a.large", "pinning": PHYSICAL_CORE},
            "zen3": {"core": 1, "over_pc": ZEN3_OVER_ZEN5 / ZEN5_OVER_PC, "instance": "c6a.large",
                     "aws": "AWS c6a.large, Zen 3",
                     "pinning": "one thread pinned to vCPU 1, the SMT sibling of vCPU 0 "
                                "(left to the system)"},
            "graviton": {"core": 1, "over_pc": GRAVITON_OVER_ZEN5 / ZEN5_OVER_PC,
                         "instance": "c8g.large", "aws": "AWS c8g.large, Graviton4",
                         "pinning": PHYSICAL_CORE},
            "jetson": {"core": 3, "over_pc": 4 / ZEN5_OVER_PC}}
#: The Jetson side (tools/dev/anchor_chart_timing.py's board setup): the reference
#: tools built for it, the timing lock and the platform as the chart states it.
JETSON_TOOLS = "~/nanocodec-data/anchors/aarch64/bin"
JETSON_LOCKED = "~/nanocodec-data/jetson-timed.sh"
JETSON_DIR = "nanocodec-data/work/lenstime"
JETSON_PLATFORM = "Jetson Orin Nano, Cortex-A78AE core 3 at 1.728 GHz, one thread"
REMOTE_DIR = "~/lenstime"
ZSTD_LEVEL = 9


def log(line: str) -> None:
    print(f"[fixedqp-timing {datetime.datetime.now():%H:%M:%S}] {line}", flush=True)


# --- plan -----------------------------------------------------------------------------------

def chart_encoders() -> list[str]:
    """The lens tests the Fixed QP chart shows (its site data), and our ladder's rungs."""
    from nceval.ctc.conditions import AOM_LD
    shown = {row["test"] for row in json.loads(SITE_LENS.read_text())["rows"]}
    return [e.name for e in AOM_LD.encoders if e.name in shown or e.own]


def lens_pixel_rates() -> dict[str, float]:
    """Each lens test's pixels per CPU second on the PC (its fps over the mean frame)."""
    from nceval.ctc.conditions import AOM_LD
    mean_pixels = statistics.mean(s.width * s.height for s in AOM_LD.sequences)
    rates = {}
    for row in json.loads(SITE_LENS.read_text())["rows"]:
        if row.get("fps"):
            rates.setdefault(row["test"], row["fps"] * mean_pixels)
    return rates


def estimate_rate(encoder: str, rates: dict[str, float]) -> float:
    """A lens test's pixel rate; a rung the lens has not run yet takes its Fast rung's."""
    if encoder in rates:
        return rates[encoder]
    base = encoder.partition("@")[0]
    return rates.get(f"{base}@fast") or rates[base]


def timing_argv(encode: list[str], sequence_yuv: str, bitstream: str) -> list[str]:
    """A lens encode as a timing command: tools under {BIN}, configuration files under
    {CFG}, the source as {INPUT}, the bitstream as {OUTPUT}, our recon outputs dropped."""
    argv, skip = [], False
    for arg in encode:
        if skip:
            skip = False
            continue
        if arg in OWN_OUTPUT_FLAGS:
            skip = True
            continue
        if arg.startswith("bin/"):
            arg = "{BIN}/" + arg.removeprefix("bin/")
        elif arg.startswith("cfg/"):
            arg = "{CFG}/" + arg.removeprefix("cfg/")
        elif arg == sequence_yuv:
            arg = "{INPUT}"
        elif arg == bitstream:
            arg = "{OUTPUT}"
        argv.append(arg)
    return argv


def plan_jobs(encoders: list[str], sequences: list[str] | None,
              points: set[int] | None, thin: dict[str, set[int]], over_pc: float) -> list[dict]:
    from dataclasses import replace

    from nceval.config import load_paths
    from nceval.ctc import plan as planner
    from nceval.ctc.conditions import CONDITIONS
    condition = CONDITIONS[CONDITION]
    condition = replace(
        condition, encoders=tuple(e for e in condition.encoders if e.name in encoders),
        sequences=tuple(s for s in condition.sequences if not sequences or s.name in sequences))
    rates = lens_pixel_rates()
    source_dir = Path("/SOURCES")
    jobs = []
    for job in planner.jobs_for(condition, load_paths(), source_dir):
        if job["kind"] != "qp" or (points and job["point"] not in points):
            continue
        if job["encoder"] in thin and job["point"] not in thin[job["encoder"]]:
            continue
        pixels = job["width"] * job["height"] * job["frames"]
        estimate = pixels * over_pc / estimate_rate(job["encoder"], rates)
        jobs.append({"id": job["id"], "encoder": job["encoder"], "sequence": job["sequence"],
                     "point": job["point"], "frames": job["frames"],
                     "argv": timing_argv(job["encode"], job["source"], job["bitstream"]),
                     "estimate_s": round(estimate, 2),
                     "rounds": REPEATED_ROUNDS if estimate < REPEAT_MAX_SECONDS else 1})
    return jobs


def job_seconds(job: dict) -> float:
    """A job's estimated share of its shard: every timed round plus the warm-up."""
    return job["estimate_s"] * (job["rounds"] + (job["rounds"] > 1))


def pack(jobs: list[dict], shard_minutes: float) -> list[list[str]]:
    """Job ids in shards of about `shard_minutes`, sequence by sequence (largest first),
    so a shard receives few sequences; a shard is closed once full."""
    by_sequence: dict[str, list[dict]] = {}
    for job in jobs:
        by_sequence.setdefault(job["sequence"], []).append(job)
    order = sorted(by_sequence, key=lambda s: -sum(map(job_seconds, by_sequence[s])))
    shards, current, load = [], [], 0.0
    for sequence in order:
        for job in sorted(by_sequence[sequence], key=job_seconds, reverse=True):
            if current and load + job_seconds(job) > shard_minutes * 60:
                shards.append(current)
                current, load = [], 0.0
            current.append(job["id"])
            load += job_seconds(job)
    if current:
        shards.append(current)
    return shards


def plan(args: argparse.Namespace) -> None:
    encoders = args.encoders.split(",") if args.encoders else chart_encoders()
    sequences = args.sequences.split(",") if args.sequences else None
    points = {int(p) for p in args.points.split(",")} if args.points else None
    thin = {name: {int(p) for p in values.split(",")}
            for name, _, values in (t.partition(":") for t in args.thin)}
    machine = MACHINES[args.machine]
    jobs = plan_jobs(encoders, sequences, points, thin, machine["over_pc"])
    if args.rounds:  # one pass where the machine's time is short (the Jetson's single core)
        jobs = [{**j, "rounds": args.rounds} for j in jobs]
    missing = sorted(set(encoders) - {j["encoder"] for j in jobs})
    if missing:
        raise SystemExit(f"no lens jobs for {', '.join(missing)}")
    shards = pack(jobs, args.shard_minutes)
    if args.machine == "jetson":  # the board's own build (nceval.jetson.ensure_build)
        binary = args.binary
        version = jetson(f"{binary} --version").strip()
    elif args.expect_version:  # a build for another architecture: it cannot run here
        binary, version = str(Path(args.binary).resolve()), args.expect_version
    else:
        binary = str(Path(args.binary).resolve())
        version = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                 check=True).stdout.strip()
    from nceval.ctc.sequences import by_name
    source_dir = Path(args.source_dir).resolve()
    sources = {name: {"raw": str(source_dir / f"{name}.yuv"),
                      "frame_bytes": by_name(name).frame_bytes}
               for name in {j["sequence"] for j in jobs}}
    write_plan(Path(args.out), {
        "condition": CONDITION, "machine": args.machine, "binary": binary,
        "nanocodec_version": version, "sources": sources,
        "refs_dir": str(Path(args.refs_dir).resolve()), "core": machine["core"],
        "shards": shards, "jobs": jobs})


def write_plan(path: Path, plan_data: dict) -> None:
    path.write_text(json.dumps(plan_data, indent=1) + "\n")
    jobs, shards = plan_data["jobs"], plan_data["shards"]
    total = sum(map(job_seconds, jobs)) / 3600
    encoders = len({j["encoder"] for j in jobs})
    log(f"{len(jobs)} encodes of {encoders} encoders in {len(shards)} shards, "
        f"about {total:.1f} instance-hours of timing (estimate)")
    for i, shard in enumerate(shards):
        own = [j for j in jobs if j["id"] in set(shard)]
        log(f"  shard {i}: {len(own)} encodes, {sum(map(job_seconds, own)) / 60:.0f} min, "
            f"sequences {', '.join(sorted({j['sequence'] for j in own}))}")


def plan_pool(args: argparse.Namespace) -> None:
    """The fixed-QP encodes of `nceval.fixedqp gap --clipset` results (each family at the
    quantizers its curve used on each clip), with the harness's own command lines."""
    from dataclasses import replace

    from nceval.cbr import Point
    from nceval.clipframes import decode_command
    from nceval.config import load_paths
    from nceval.fixedqp import Benches, family_of, held_out_clips
    from nceval.site import LENS_FAMILIES
    results = [json.loads(Path(f).read_text()) for f in args.gap]
    clipset, frames = results[0]["clipset"], results[0]["frames"]
    if not clipset or any(r["clipset"] != clipset or r["frames"] != frames for r in results):
        raise SystemExit("the gap results must share one --clipset run and frame count")
    clips = {c.name: c for c in held_out_clips(clipset, len(results[0]["clips"]), frames)}
    if set(clips) != {c["clip"] for c in results[0]["clips"]}:
        raise SystemExit("the clip set's rule picks other clips than the gap run used")
    paths, binary = load_paths(), Path(args.binary).resolve()
    benches, machine = Benches(paths, frames), MACHINES[args.machine]
    lens_name = {family: test for test, family in LENS_FAMILIES.items()}
    rates = lens_pixel_rates()
    curves = {key: qps for r in results for key, qps in r["quantizers"].items()}
    jobs = []
    for key, quantizers in sorted(curves.items()):
        name, _, clip_name = key.partition("|")
        clip = clips[clip_name]
        if clip.start_frame:
            raise SystemExit(f"{clip_name}: a rotated start is not supported here")
        family = family_of(name, binary, paths, Path(args.presets_dir))
        estimate_rate_ = estimate_rate(lens_name.get(name, name), rates)
        for q in quantizers:
            argv = benches.of(clip).command(Point(family, replace(clip, yuv=Path("/IN.yuv")),
                                                  qp=q), Path("/OUT.bin"), None)
            argv = ["{BIN}/" + Path(argv[0]).name,
                    *({"/IN.yuv": "{INPUT}", "/OUT.bin": "{OUTPUT}"}.get(a, a) for a in argv[1:])]
            estimate = clip.width * clip.height * frames * machine["over_pc"] / estimate_rate_
            jobs.append({"id": f"{name}/{clip_name}/q{q}", "encoder": name,
                         "sequence": clip_name, "point": q, "frames": frames, "argv": argv,
                         "estimate_s": round(estimate, 2),
                         "rounds": REPEATED_ROUNDS if estimate < REPEAT_MAX_SECONDS else 1})
    sources = {c.name: {"decode": decode_command(c, frames),
                        "frame_bytes": c.width * c.height * 3 // 2} for c in clips.values()}
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                             check=True).stdout.strip()
    write_plan(Path(args.out), {
        "condition": Path(clipset).stem, "machine": args.machine, "binary": str(binary),
        "nanocodec_version": version, "sources": sources, "refs_dir": str(DEFAULT_REFS),
        "core": machine["core"], "shards": pack(jobs, args.shard_minutes), "jobs": jobs})


def plan_from(args: argparse.Namespace) -> None:
    """Another machine's timed plan for this one: its jobs as they were (command lines,
    quantizers, rounds), so both machines time the same encodes; the estimates are rescaled
    for packing. `--binary` stands in for the plan's own build of ours (the same commit
    rebuilt, its version line checked on the instance)."""
    source = json.loads(Path(args.source).read_text())
    if source["machine"] in ("graviton", "jetson") or args.machine in ("graviton", "jetson"):
        raise SystemExit("plan-from moves a plan between x86 machines (same tool builds)")
    encoders = set(args.encoders.split(",")) if args.encoders else None
    scale = MACHINES[args.machine]["over_pc"] / MACHINES[source["machine"]]["over_pc"]
    jobs = [{**j, "estimate_s": round(j["estimate_s"] * scale, 2)}
            for j in source["jobs"] if not encoders or j["encoder"] in encoders]
    missing = sorted((encoders or set()) - {j["encoder"] for j in jobs})
    if missing:
        raise SystemExit(f"not in {args.source}: {', '.join(missing)}")
    binary = str(Path(args.binary).resolve()) if args.binary else source["binary"]
    version = subprocess.run([binary, "--version"], capture_output=True, text=True,
                             check=True).stdout.strip()
    if not args.binary and version != source["nanocodec_version"]:
        raise SystemExit(f"{binary} is now {version!r}, not the plan's build: pass --binary")
    sequences = {j["sequence"] for j in jobs}
    write_plan(Path(args.out), {
        **source, "machine": args.machine, "binary": binary, "nanocodec_version": version,
        "sources": {n: v for n, v in source["sources"].items() if n in sequences},
        "core": MACHINES[args.machine]["core"],
        "shards": pack(jobs, args.shard_minutes), "jobs": jobs})


# --- run ------------------------------------------------------------------------------------

def tool_source(name: str, plan_data: dict) -> Path:
    """A tool's local file: our binary, HM / VVenC from the refs (x86 builds only), or
    the machine's reference builds; a missing one stops the run before any launch."""
    if name == "nanocodec":
        return Path(plan_data["binary"])
    if plan_data["machine"] == "graviton":
        source = AARCH64_TOOLS / name
    else:
        refs = Path(plan_data["refs_dir"]) / "bin" / name
        source = refs if refs.exists() else ANCHOR_TOOLS / name
    if not source.is_file():
        raise SystemExit(f"no {name} for the {plan_data['machine']} machine: {source}")
    return source


def sequence_archive(name: str, frames: int, frame_bytes: int, read, cache: Path) -> Path:
    """A sequence's first `frames` frames (`read(nbytes)` returns them raw), zstd-compressed,
    cached by name and length. Runs side by side may share the cache (a release times four
    machines at once): each writes its own partial file, and the last rename wins with the
    same bytes."""
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{name}-{frames}f.yuv.zst"
    if not archive.is_file():
        partial = archive.with_suffix(f".{os.getpid()}.partial")
        data = read(frame_bytes * frames)
        if len(data) != frame_bytes * frames:
            raise SystemExit(f"{name}: fewer than {frames} frames in the source")
        subprocess.run(["zstd", "-q", "-T8", f"-{ZSTD_LEVEL}", "-f", "-o", str(partial)],
                       input=data, check=True)
        partial.rename(archive)
    return archive


def archives(plan_data: dict, cache: Path) -> dict[str, Path]:
    """Every planned sequence's archive, by sequence: a lens sequence from its raw file
    in the source directory, one of our clips decoded by the harness's own command
    (plan "decode", nceval.clipframes.decode_command)."""
    frames = max(j["frames"] for j in plan_data["jobs"])
    out = {}
    for name in sorted({j["sequence"] for j in plan_data["jobs"]}):
        source = plan_data["sources"][name]
        if "decode" in source:
            def read(size: int, argv=source["decode"]) -> bytes:
                return subprocess.run(argv, capture_output=True, check=True).stdout[:size]
        else:
            def read(size: int, path=source["raw"]) -> bytes:
                with open(path, "rb") as raw:
                    return raw.read(size)
        out[name] = sequence_archive(name, frames, source["frame_bytes"], read, cache)
    return out


def stage(stage_dir: Path, plan_data: dict, jobs: list[dict], cache: Path) -> None:
    for sub in ("bin", "cfg", "clips"):
        (stage_dir / sub).mkdir(parents=True)
    tools = {j["argv"][0].removeprefix("{BIN}/") for j in jobs}
    for tool in tools:
        (stage_dir / "bin" / tool).symlink_to(tool_source(tool, plan_data).resolve())
    configs = {a.removeprefix("{CFG}/") for j in jobs for a in j["argv"] if a.startswith("{CFG}/")}
    for config in configs:
        (stage_dir / "cfg" / config).symlink_to(
            (Path(plan_data["refs_dir"]) / "cfg" / config).resolve())
    ready = archives(plan_data, cache)
    for sequence in sorted({j["sequence"] for j in jobs}):
        (stage_dir / "clips" / f"{sequence}.yuv.zst").symlink_to(ready[sequence])
    (stage_dir / "fixedqp_timing_remote.py").symlink_to(REMOTE_SCRIPT)
    (stage_dir / "plan.json").write_text(json.dumps(
        {"core": plan_data["core"], "jobs": jobs}, indent=1) + "\n")


def unstage(stage_dir: Path) -> None:
    """Remove the staging directory entry by entry (symlinks, the plan, empty dirs)."""
    for sub in ("bin", "cfg", "clips"):
        if (stage_dir / sub).is_dir():
            for entry in (stage_dir / sub).iterdir():
                entry.unlink()
            (stage_dir / sub).rmdir()
    for name in ("fixedqp_timing_remote.py", "plan.json"):
        (stage_dir / name).unlink(missing_ok=True)
    stage_dir.rmdir()


#: Shared libraries a reference build needs that a fresh Ubuntu image may lack, by package
#: (the Jetson's x265 links libnuma).
RUNTIME_PACKAGES = {"libnuma.so.1": "libnuma1"}


def prepare_command(plan_data: dict, jobs: list[dict]) -> str:
    """The instance's readiness stage: missing runtime libraries installed, every tool
    loadable, our binary the planned build, the sequences unpacked and the CPU's model
    line in out/platform.txt (/proc/cpuinfo on x86, lscpu's on Arm, which has none)."""
    installs = " && ".join(
        f"{{ ldconfig -p | grep -q {lib} || {{ sudo apt-get -qq update && "
        f"sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y {package} > /dev/null; }}; }}"
        for lib, package in RUNTIME_PACKAGES.items())
    ours = ""
    if any(j["argv"][0] == "{BIN}/nanocodec" for j in jobs):
        ours = f"test \"$(bin/nanocodec --version)\" = {shlex.quote(plan_data['nanocodec_version'])} && "
    return (f"cd {REMOTE_DIR} && {installs} && ! ldd bin/* | grep 'not found' && {ours}"
            "for c in clips/*.zst; do zstd -d -q --rm \"$c\"; done && mkdir -p out && "
            "{ grep -m1 'model name' /proc/cpuinfo || lscpu | grep -m1 'Model name'; } "
            "> out/platform.txt && echo ready")


def check_shard(shard_out: Path, jobs: list[dict]) -> None:
    """Fail loudly unless every job of the shard was timed (a failed encode is reported
    by the remote side and leaves its job without times)."""
    times = json.loads((shard_out / "times.json").read_text())
    untimed = [j["id"] for j in jobs if not times.get(j["id"], {}).get("cpu_s")]
    if untimed:
        raise RuntimeError(f"{len(untimed)} encodes untimed, e.g. {untimed[0]}")


def run_shard(index: int, plan_data: dict, out: Path, dry_run: bool) -> str:
    from aws_session import Session, SessionSpec
    from aws_timing import INSTANCES
    ids = set(plan_data["shards"][index])
    jobs = [j for j in plan_data["jobs"] if j["id"] in ids]
    shard_out = out / f"shard-{index}"
    if (shard_out / "DONE").exists():
        return f"shard {index}: already done"
    stage_dir = out / f"stage-{index}"
    if stage_dir.exists():  # left by a run stopped while staging
        unstage(stage_dir)
    stage(stage_dir, plan_data, jobs, out.parent / "cache")
    minutes = sum(map(job_seconds, jobs)) / 60
    if dry_run:
        unstage(stage_dir)
        return f"shard {index}: staged {len(jobs)} encodes ({minutes:.0f} min), dry run"
    instance = MACHINES[plan_data["machine"]]["instance"]
    region, arch = INSTANCES[instance]
    spec = SessionSpec(region=region, instance_type=instance, arch=arch,
                       purpose=f"fixedqp-timing-{index}", lifetime_minutes=LIFETIME_MINUTES,
                       budget_usd=SHARD_BUDGET_USD, on_demand=True)
    try:
        with Session(spec) as session:
            session.upload(stage_dir, REMOTE_DIR)
            try:
                session.run_detached(prepare_command(plan_data, jobs),
                                     f"{REMOTE_DIR}/prepare.log", 10)
                session.run_detached(f"cd {REMOTE_DIR} && python3 -I fixedqp_timing_remote.py",
                                     f"{REMOTE_DIR}/time.log",
                                     min(minutes * 3 + 10, LIFETIME_MINUTES - 10),
                                     stall_minutes=max(10.0, 3 * max(
                                         j["estimate_s"] for j in jobs) / 60))
            finally:
                session.download(f"{REMOTE_DIR}/out", shard_out)
    finally:
        unstage(stage_dir)
    check_shard(shard_out, jobs)
    (shard_out / "DONE").write_text("")
    return f"shard {index}: done"


def run(args: argparse.Namespace) -> None:
    """Each shard in its own process (aws_session installs signal handlers, which only
    the main thread may), `--parallel` at a time; its log in DIR/shard-N.log."""
    plan_data = json.loads(Path(args.plan).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = ([int(s) for s in args.shards.split(",")] if args.shards
              else range(len(plan_data["shards"])))
    archives(plan_data, out.parent / "cache")  # once, before the shards share them

    def one(index: int) -> int:
        with open(out / f"shard-{index}.log", "a") as shard_log:
            command = [sys.executable, __file__, "run-shard", "--plan", args.plan,
                       "--out", str(out), "--index", str(index)]
            command += ["--dry-run"] if args.dry_run else []
            status = subprocess.run(command, stdout=shard_log, stderr=subprocess.STDOUT).returncode
        log(f"shard {index}: {'done' if status == 0 else f'FAILED (exit {status})'}")
        return status

    with ThreadPoolExecutor(args.parallel) as pool:
        failed = [i for i, status in zip(shards, pool.map(one, shards)) if status]
    if failed:
        raise SystemExit(f"failed shards: {','.join(map(str, failed))} (rerun with --shards)")


def run_one(args: argparse.Namespace) -> None:
    plan_data = json.loads(Path(args.plan).read_text())
    log(run_shard(args.index, plan_data, Path(args.out), args.dry_run))


def jetson(command: str) -> str:
    """A command on the board (NC_JETSON), its standard output."""
    from nceval.jetson import ssh
    return ssh(command)


def run_jetson(args: argparse.Namespace) -> None:
    """The shards one after another on the Jetson's timing core, each under the board's
    timing lock (jetson-timed.sh); the sequences go up once, as archives."""
    from nceval.localenv import require
    plan_data = json.loads(Path(args.plan).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    host = require("NC_JETSON")
    ready = archives(plan_data, out.parent / "cache")
    jetson(f"mkdir -p ~/{JETSON_DIR}/clips")
    for name, archive in ready.items():
        subprocess.run(["rsync", "-a", str(archive), f"{host}:{JETSON_DIR}/clips/{name}.yuv.zst"],
                       check=True)
    jetson(f"cd ~/{JETSON_DIR}/clips && for c in *.yuv.zst; do [ -s \"${{c%.zst}}\" ] || "
           "zstd -d -q -f \"$c\"; done")
    shards = ([int(s) for s in args.shards.split(",")] if args.shards
              else range(len(plan_data["shards"])))
    for index in shards:
        shard_out = out / f"shard-{index}"
        if (shard_out / "DONE").exists():
            continue
        ids = set(plan_data["shards"][index])
        jobs = [j for j in plan_data["jobs"] if j["id"] in ids]
        work = f"{JETSON_DIR}/shard-{index}"
        tools = sorted({j["argv"][0].removeprefix("{BIN}/") for j in jobs} - {"nanocodec"})
        links = " && ".join([f"ln -sf {plan_data['binary']} bin/nanocodec",
                             *(f"ln -sf {JETSON_TOOLS}/{t} bin/{t}" for t in tools)])
        jetson(f"rm -rf ~/{work} && mkdir -p ~/{work}/bin ~/{work}/out && cd ~/{work} && "
               f"ln -s ~/{JETSON_DIR}/clips clips && {links}")
        payload = json.dumps({"core": plan_data["core"], "jobs": jobs})
        subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"cat > ~/{work}/plan.json"],
                       input=payload, text=True, check=True)
        subprocess.run(["rsync", "-a", str(REMOTE_SCRIPT), f"{host}:{work}/"], check=True)
        log(f"shard {index}: {len(jobs)} encodes, about "
            f"{sum(map(job_seconds, jobs)) / 60:.0f} min under the timing lock")
        subprocess.run(["ssh", "-o", "BatchMode=yes", host,
                        f"cd ~/{work} && {JETSON_LOCKED} python3 -I fixedqp_timing_remote.py"],
                       check=True)
        shard_out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{host}:{work}/out/", f"{shard_out}/"], check=True)
        (shard_out / "platform.txt").write_text(f"model name\t: {JETSON_PLATFORM}\n")
        jetson(f"rm -rf ~/{work}")
        check_shard(shard_out, jobs)
        (shard_out / "DONE").write_text("")
    log(f"Jetson shards done; the decoded sequences stay in ~/{JETSON_DIR}/clips")


# --- record ---------------------------------------------------------------------------------

def record(args: argparse.Namespace) -> None:
    """The site's timing record from one or more timed plans of one machine and build
    (--plan / --out in pairs, e.g. our encoders and the references timed apart)."""
    if len(args.plan) != len(args.out):
        raise SystemExit("--plan and --out come in pairs")
    plans = [json.loads(Path(p).read_text()) for p in args.plan]
    if len({(p["machine"], p["condition"], p["nanocodec_version"]) for p in plans}) != 1:
        raise SystemExit("the plans differ in machine, content or build")
    jobs = list({j["id"]: j for p in plans for j in p["jobs"]}.values())  # a shared anchor once
    times, platforms = {}, set()
    for plan_data, out in zip(plans, map(Path, args.out)):
        for index in range(len(plan_data["shards"])):
            shard = out / f"shard-{index}"
            times.update(json.loads((shard / "times.json").read_text()))
            platforms.add((shard / "platform.txt").read_text().split(":", 1)[1].strip())
    per: dict[str, dict[str, list[tuple[float, int]]]] = {}
    missing = []
    for job in jobs:
        cpu = times.get(job["id"], {}).get("cpu_s")
        if not cpu:
            missing.append(job["id"])
            continue
        per.setdefault(job["encoder"], {}).setdefault(job["sequence"], []).append(
            (statistics.median(cpu), job["frames"]))
    if missing:
        raise SystemExit(f"untimed encodes: {', '.join(missing[:10])} ({len(missing)})")
    points = {}
    for encoder, sequences in sorted(per.items()):
        ms = {s: round(1000 * sum(t for t, _ in v) / sum(f for _, f in v), 4)
              for s, v in sequences.items()}
        points[encoder] = {"ms_per_frame": ms,
                           "fps": round(1000 / math.exp(statistics.mean(
                               math.log(m) for m in ms.values())), 1)}
    first = plans[0]
    Path(args.record).write_text(json.dumps({
        "platform": JETSON_PLATFORM if first["machine"] == "jetson" else
        f"{' / '.join(sorted(platforms))}, {MACHINES[first['machine']]['pinning']} "
        f"({MACHINES[first['machine']]['aws']}, one encode at a time)",
        "condition": first["condition"], "frames": jobs[0]["frames"],
        "sequences": sorted({j["sequence"] for j in jobs}),
        # per encoder: the quantizers timed on each sequence (the most on any one)
        "quantizers_timed": {e: max(len(v) for v in per[e].values()) for e in sorted(per)},
        "nanocodec_version": first["nanocodec_version"], "build": args.build,
        "rounds": {"per_encode": sorted({j["rounds"] for j in jobs}),
                   "repeat_max_estimate_s": REPEAT_MAX_SECONDS},
        "note": "CPU time (user + system) of each encode, median of its rounds; "
                "ms_per_frame per sequence over all its quantizers; fps = 1000 / the "
                "geometric mean of ms_per_frame over the sequences",
        "points": points}, indent=1) + "\n")
    for encoder, entry in sorted(points.items(), key=lambda kv: -kv[1]["fps"]):
        print(f"{encoder:28s} {entry['fps']:8.1f} fps")


def merge(args: argparse.Namespace) -> None:
    """Some encoders' points of a newer timing record (`record` of a re-timed subset, same
    machine and content) into the site's record; each merged point names its build, so
    the record keeps one footing per encoder and says which build each was timed on."""
    site_record = json.loads(Path(args.record).read_text())
    newer = json.loads(Path(args.source).read_text())
    for key in ("condition", "frames", "sequences"):
        if newer[key] != site_record[key]:
            raise SystemExit(f"the records differ in {key}")
    if newer["platform"].split(",")[0] != site_record["platform"].split(",")[0]:
        raise SystemExit("the records differ in machine")
    encoders = args.encoders.split(",")
    missing = [e for e in encoders if e not in newer["points"]]
    if missing:
        raise SystemExit(f"not in {args.source}: {', '.join(missing)}")
    for encoder in encoders:
        site_record["points"][encoder] = {**newer["points"][encoder], "build": newer["build"]}
        site_record["quantizers_timed"][encoder] = newer["quantizers_timed"][encoder]
    Path(args.record).write_text(json.dumps(site_record, indent=1) + "\n")
    for encoder in encoders:
        print(f"{encoder:28s} {site_record['points'][encoder]['fps']:8.1f} fps "
              f"(build {newer['build']})")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--binary", required=True,
                   help="our encoder (on the Jetson: its path there, e.g. a chart build)")
    p.add_argument("--machine", choices=sorted(MACHINES), default="x86")
    p.add_argument("--expect-version", help="our binary's `--version` line, for a build of "
                   "another architecture than this PC's (graviton: the Jetson's chart build)")
    p.add_argument("--out", required=True)
    p.add_argument("--encoders")
    p.add_argument("--sequences")
    p.add_argument("--points", help="quantizer points to time (default: all six)")
    p.add_argument("--thin", action="append", default=[], metavar="ENCODER:P,P",
                   help="time ENCODER at these quantizer points only (HM: hours per sequence)")
    p.add_argument("--shard-minutes", type=float, default=SHARD_MINUTES)
    p.add_argument("--rounds", type=int, help="every encode this many timed rounds "
                   "(default: by its estimate, REPEATED_ROUNDS or one)")
    p.add_argument("--source-dir", default=str(DEFAULT_SOURCES))
    p.add_argument("--refs-dir", default=str(DEFAULT_REFS))
    p.set_defaults(handler=plan)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--shards")
    r.add_argument("--parallel", type=int, default=6)
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(handler=run)
    o = sub.add_parser("run-shard", help="one shard (run's worker process)")
    o.add_argument("--plan", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--index", type=int, required=True)
    o.add_argument("--dry-run", action="store_true")
    o.set_defaults(handler=run_one)
    q = sub.add_parser("plan-pool", help="plan the encodes of fixedqp gap --clipset results")
    q.add_argument("--binary", required=True)
    q.add_argument("--gap", action="append", required=True, help="a gap result JSON")
    q.add_argument("--out", required=True)
    q.add_argument("--presets-dir", default=str(REPO / "crates/nc-encoder/presets"),
                   help="our presets for --binary, as the gap run used (nceval.fixedqp gap)")
    # not the Graviton: only the public lens sequences and the chart clips go to it
    q.add_argument("--machine", choices=("jetson", "x86"), default="x86")
    q.add_argument("--shard-minutes", type=float, default=SHARD_MINUTES)
    q.set_defaults(handler=plan_pool)
    f = sub.add_parser("plan-from", help="another x86 machine's timed plan, for this one")
    f.add_argument("--from", dest="source", required=True, help="the timed plan (`plan`)")
    f.add_argument("--machine", choices=sorted(MACHINES), required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--encoders", help="only these encoders' jobs, a,b")
    f.add_argument("--binary", help="our build of the plan's commit (its own path is gone)")
    f.add_argument("--shard-minutes", type=float, default=SHARD_MINUTES)
    f.set_defaults(handler=plan_from)
    j = sub.add_parser("run-jetson", help="time a --machine jetson plan on the board")
    j.add_argument("--plan", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--shards")
    j.set_defaults(handler=run_jetson)
    c = sub.add_parser("record")
    c.add_argument("--plan", action="append", required=True)
    c.add_argument("--out", action="append", required=True)
    c.add_argument("--record", required=True)
    c.add_argument("--build", required=True,
                   help="the variance-integration commit our binary's code is (8 characters)")
    c.set_defaults(handler=record)
    m = sub.add_parser("merge", help="re-timed encoders of a newer record into the site's")
    m.add_argument("--record", required=True, help="the site's record, updated in place")
    m.add_argument("--from", dest="source", required=True, help="the newer record (`record`)")
    m.add_argument("--encoders", required=True, help="the encoders to take from it, a,b")
    m.set_defaults(handler=merge)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main(sys.argv[1:])
