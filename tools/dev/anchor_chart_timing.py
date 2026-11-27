"""Encode speed for the bitrate-vs-speed chart (eval/nceval/chart.py).

The chart's speed axis times a subset of the very encodes its quality points run
(nceval.chart_timing): every encoder, ours and every reference, in its own real-time
CBR mode, on the timing clips (three of the chart's VAL clips, each at one rung of the
real-use ladder), DEFAULT_FRAMES frames, one key frame, single-threaded. Until
2026-11-23 the plan held fixed-QP commands matched to the 600 kbps rung's bytes on three
other clips (the `scan` step): frames timed before then are fixed-QP speeds.

  plan  (from eval/ with PYTHONPATH=.): each point's timing commands, built by
        nceval.chart_timing from the quality path's own job list and command, with
        {BIN}/{INPUT}/{OUTPUT} placeholders, and each stream's size (one encode per
        point and clip; our streams are bit-exact across platforms, so the timing steps
        check theirs against it), into plan.json. `--bytes-from` takes a `sizes` file: an
        encode whose quality stream is listed there is not encoded again.
  sizes (here): {prebuilt key: bytes} of a chart run's stored quality streams
        (aws_heldout_chart --streams), for plan --bytes-from.
  time  (Jetson, under ~/nanocodec-data/jetson-timed.sh, one point per lock):
        core 3 at its fixed 1.728 GHz, a warm-up run of the point and the anchor
        (x264 medium) per clip, then three rounds in rotated order (`--rounds`,
        `--no-warm-up`: an every-encode plan runs each once); encode CPU
        time (rusage) per frame, median per clip. `--anchor-from t-X.json`
        reuses that file's anchor times instead of re-timing the anchor. The clips
        are JETSON_CLIP_DIR/<clip>.yuv, the plan's frames of each.
  time-host (PC, from eval/ with PYTHONPATH=., under tools/dev/host-timed.sh):
        the same streams on this machine (x86) on its reserved timing cores.
        Each run is pinned to one logical CPU (one per physical core, `--cpus`,
        e.g. 14,15). Every point runs back to back with the anchor and the
        second anchor (SVT-AV1 M13) on the same CPU and clip, HOST_ROUNDS
        rounds in rotated order (`--rounds`); per clip it keeps the median
        per-round time ratio to each anchor (the cores' boost clock moves with
        the machine's load within a session). The (point, clip) pairs share the
        CPUs, so a single point's clips are timed side by side.
  time-cloud (a rented machine, tools/dev/aws_timing.py): time-host's paired rounds
        without the eval package: tools from --bin-dir (named as in the plan's
        argv), clips from --clips-dir (<clip>.yuv), `--cpus physical` takes one
        logical CPU per physical core. A stream that differs from the plan's
        (another CPU's SIMD paths) is recorded, not fatal, and a failed pair is
        recorded in errors.json while the others go on.
  merge (either): collect the per-point timing files into one timing.json for
        `nceval.chart render --timing`.

    python3 tools/dev/anchor_chart_timing.py plan --binary BIN --presets-dir DIR --plan plan.json
    jetson-timed.sh python3 anchor_chart_timing.py time --plan plan.json --point x265-medium \
        --nanocodec BIN --out t-x265-medium.json
    PYTHONPATH=. ../tools/dev/host-timed.sh python3 ../tools/dev/anchor_chart_timing.py \
        time-host --plan plan.json --binary BIN --presets-dir DIR --points a,b --cpus 14,15 \
        --out-dir host/
    python3 tools/dev/anchor_chart_timing.py merge --out timing.json [--machine M] [--build C] \
        [--build-for POINT=C ...] t-*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import queue
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

#: What the plan's commands are (the records and the chart say so).
RATE_MODE = "cbr"
JETSON_WORK = pathlib.Path.home() / "nanocodec-data/work"
#: The timing clips on the Jetson, each cut to the plan's frames (chart_refresh copies them).
JETSON_CLIP_DIR = JETSON_WORK / "chart-timing-clips"
JETSON_TOOLS = pathlib.Path.home() / "nanocodec-data/anchors/aarch64/bin"
#: Tools the chart's quality points take from elsewhere than JETSON_TOOLS: the openh264
#: harness with CBR (--bitrate), built on the board from tools/openh264-harness (the
#: anchors' 2026-11-06 copy has none; x86 and Graviton time the --opt builds' harness).
JETSON_OWN_TOOLS = {"openh264-harness":
                    pathlib.Path.home() / "nanocodec-data/work/ratecontrol/bin/openh264-harness"}
TIMING_CORE = 3
JETSON_PLATFORM = "Jetson Orin Nano, Cortex-A78AE core 3 at 1.728 GHz, one thread"
CORE_KHZ = "1728000"
ROUNDS = 3
QUIET_LOAD = 1.5  # jetson-timed.sh's own refusal threshold
#: Host timing: most rounds per clip (the median per-round ratio is kept) and its
#: platform note. 5 fixed rounds mis-timed points at load ~35 (2026-11-12: AV1 Fast
#: level with AV1 Faster), so a pair runs up to 11.
HOST_ROUNDS = 11
#: A pair stops after HOST_MIN_ROUNDS once the 95 % interval of its mean log time
#: ratio to the anchor is within +-HOST_TARGET_HALF_WIDTH. Stored sessions
#: (2026-11-15, 246 pairs): per-round ratio sd 2.2 % median (CI +-1.9 % at 5
#: rounds), 7.7 % at p90, so most pairs stop at 5 and the noisy ones run on.
HOST_MIN_ROUNDS = 5
HOST_TARGET_HALF_WIDTH = 0.02
#: z of a two-sided 95 % interval.
Z95 = 1.96
#: Host timing's second anchor, timed in the same rounds: the real-time reference our
#: fast presets are compared with, as a direct same-session time ratio.
SECOND_ANCHOR = "svtav1-m13"
HOST_PLATFORM = "Ryzen 9 5950X (x86-64, AVX2), one thread pinned to one core"
#: (point, clip) streams sized side by side while planning (each holds one CPU slot).
PLAN_WORKERS = 3


# --- plan ----------------------------------------------------------------------------

def plan_clips(plan: dict) -> list[str]:
    """The plan's entries' clip keys (a clip, or clip@kbps), in its order."""
    return list(plan["clips"])


def plan_input(plan: dict, key: str) -> str:
    """The clip an entry key encodes."""
    return plan["inputs"][key]


def input_frames(plan: dict, name: str) -> int:
    """The frames the plan's entries on input clip `name` code: the plan's frames, or
    fewer for a clip shorter than that (its quality encodes code the clip's own length)."""
    return max(entry["frames"] for entries in plan["points"].values()
               for key, entry in entries.items() if plan_input(plan, key) == name)


def stream_bytes(bench, point) -> dict:
    """The stream `point`'s timing command writes ({bytes, sha256}: one encode, its clip's
    frames served as its quality point reads them)."""
    from nceval.slots import cpu_slots

    with tempfile.TemporaryDirectory(dir=pathlib.Path.home() / "nanocodec-data/work") as tmp, \
            bench.frames_of(point.clip) as frames:
        out = pathlib.Path(tmp) / "s.bin"
        cmd = bench.command(point, out, None, source=frames)
        with cpu_slots(1):
            subprocess.run(["nice", "-n", "19", *cmd], check=True, capture_output=True,
                           stdin=subprocess.DEVNULL)
        return {"bytes": out.stat().st_size, "sha256": sha256(out)}


def stream_sizes(streams: pathlib.Path) -> dict[str, dict]:
    """{prebuilt key: {bytes, sha256}} of every finished stream under `streams` (a rented
    chart run's
    stream directory, aws_heldout_chart --streams: <prebuilt key><codec suffix>, finished
    when its .done marker is there)."""
    from nceval.codecs import Codec

    suffixes = {c.bitstream_suffix for c in Codec}
    sizes = {}
    for done in streams.glob("*/*/*.done"):
        base = done.with_suffix("")
        for suffix in suffixes:
            stream = base.with_name(base.name + suffix)
            if stream.is_file():
                sizes[str(base.relative_to(streams))] = {"bytes": stream.stat().st_size,
                                                         "sha256": sha256(stream)}
    return sizes


def write_sizes(args: argparse.Namespace) -> None:
    sizes = stream_sizes(pathlib.Path(args.streams))
    pathlib.Path(args.out).write_text(json.dumps(sizes, sort_keys=True) + "\n")
    print(f"{len(sizes)} stream sizes -> {args.out}")


def entry_key(entry: dict, clip, tool_digest: str) -> str:
    """What a planned stream's size depends on: the tool's content, the command and the
    clip. An entry with the same key keeps its stored size (the encodes are bit-exact
    reruns)."""
    parts = [tool_digest, entry["argv"], clip.name, clip.yuv.stat().st_size]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]


def write_plan(path: pathlib.Path, plan: dict) -> None:
    """The plan with one line per entry: an every-encode plan holds thousands of commands,
    and a refresh's diff should name the entries it changed."""
    header = {k: v for k, v in plan.items() if k != "points"}
    lines = ["{", *(f" {json.dumps(k)}: {json.dumps(v)}," for k, v in header.items()),
             ' "points": {']
    names = list(plan["points"])
    for i, name in enumerate(names):
        entries = plan["points"][name]
        lines.append(f"  {json.dumps(name)}: {{")
        keys = list(entries)
        lines += [f"   {json.dumps(k)}: {json.dumps(entries[k])}"
                  + ("," if j + 1 < len(keys) else "") for j, k in enumerate(keys)]
        lines.append("  }" + ("," if i + 1 < len(names) else ""))
    path.write_text("\n".join([*lines, " }", "}"]) + "\n")


def make_plan(args: argparse.Namespace) -> None:
    from nceval.chart import ANCHOR, binary_overrides, chart_clips, chart_families
    from nceval.chart_timing import (PLAN_STORE, entry_key as job_key, load_plan, plan_digest,
                                     plan_entry, plan_jobs, plan_record)
    from nceval.config import load_paths
    from nceval.gatepolicy import EVALUATION_GATE
    from nceval.realuse import DEFAULT_FRAMES, Runner

    families = chart_families(pathlib.Path(args.binary).resolve(),
                              pathlib.Path(args.presets_dir),
                              args.points.split(",") if args.points else None,
                              binary_overrides(args.binary_for), binary_overrides(args.presets_for))
    runner = Runner(load_paths(), EVALUATION_GATE, workers=1, over_budget=True)
    clips = chart_clips(args.clipset)
    header = {"all_encodes": args.all_encodes,
              "low_clips": args.low_clips.split(",") if args.low_clips else []}
    jobs = plan_jobs(runner, families, clips, header)
    keyed = [(job_key(p, args.all_encodes), b, p) for b, p in jobs]
    plan_path = pathlib.Path(args.plan)
    store = pathlib.Path(args.store) if args.store else PLAN_STORE
    try:
        stored = load_plan(plan_path, store) if plan_path.is_file() else {}
    except (FileNotFoundError, ValueError):
        stored = {}  # a plan not here is made afresh
    plan = {"anchor": ANCHOR, "rate_mode": RATE_MODE, "frames": DEFAULT_FRAMES,
            "clipset": args.clipset, **header,
            "clips": {k: p.kbps for k, _, p in keyed},
            "inputs": {k: p.clip.name for k, _, p in keyed}}
    same = all(stored.get(k) == plan[k] for k in
               ("rate_mode", "frames", "clipset", "all_encodes", "low_clips", "clips"))
    plan["points"] = stored.get("points", {}) if same else {}
    digest = hashlib.sha256(pathlib.Path(args.binary).read_bytes()).hexdigest()
    plan["nanocodec_sha256"] = {**stored.get("nanocodec_sha256", {}),
                                **{f.name: digest for f in families if f.own}}
    tool_digests = {f.tool: sha256(f.tool) for f in families}
    # the timing command is the quality command without its reconstruction and logs,
    # which change no stream byte: a quality stream's size is its planned size
    quality_sizes = json.loads(pathlib.Path(args.bytes_from).read_text()) if args.bytes_from else {}
    if any(not isinstance(v, dict) for v in quality_sizes.values()):
        raise SystemExit(f"{args.bytes_from}: sizes without stream hashes (written before "
                         "644e71b24); write it again with `anchor_chart_timing.py sizes`")
    with ThreadPoolExecutor(args.workers) as pool:
        work = {}
        for clip_key, bench, point in keyed:
            entry = plan_entry(bench, point)
            key = entry_key(entry, point.clip, tool_digests[point.family.tool])
            old = plan["points"].get(point.family.name, {}).get(clip_key, {})
            known = ({"bytes": old["bytes"], "sha256": old["sha256"]}
                     if old.get("key") == key and "sha256" in old
                     else quality_sizes.get(bench.prebuilt_key(point)))
            work[(point.family.name, clip_key)] = (
                {**entry, "key": key},
                known if known is not None else pool.submit(stream_bytes, bench, point))
        for (name, clip), (entry, stream) in work.items():
            cached = isinstance(stream, dict)
            entry.update(stream if cached else stream.result())
            plan["points"].setdefault(name, {})[clip] = entry
            print(f"{name:26s} {clip:22s} {entry['kbps']:5d} kbps {entry['bytes']:9d} bytes"
                  f"{' (cached)' if cached else ''}", flush=True)
    for name in plan["points"]:  # a point planned on other clips is planned afresh
        if set(plan["points"][name]) != set(plan["clips"]):
            raise SystemExit(f"{name}: planned on {sorted(plan['points'][name])}, not the "
                             f"timing clips; plan it again (--points {name})")
    store.mkdir(parents=True, exist_ok=True)
    plan_file = f"{plan_path.stem}-{plan_digest(plan)[:12]}.json"
    write_plan(store / plan_file, plan)
    command = ["anchor_chart_timing.py", *(a.replace(str(pathlib.Path.home()), "~")
                                           for a in sys.argv[1:])]
    plan_path.write_text(json.dumps(plan_record(plan, plan_file, command),
                                    indent=1) + "\n")


# --- time (Jetson) ---------------------------------------------------------------------

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def concrete(argv: list[str], clip_input: str, out: pathlib.Path, nanocodec: str) -> list[str]:
    tool = argv[0].removeprefix("{BIN}/")
    head = nanocodec if tool == "nanocodec" else str(JETSON_OWN_TOOLS.get(tool, JETSON_TOOLS / tool))
    return [head] + [str(JETSON_CLIP_DIR / f"{clip_input}.yuv") if a == "{INPUT}" else
                     str(out) if a == "{OUTPUT}" else a for a in argv[1:]]


def rate_fraction(entry: dict, stream_bytes: int) -> float:
    """A stream's average rate over its command's CBR target."""
    return stream_bytes * 8 * entry["fps"] / entry["frames"] / 1000 / entry["kbps"]


def cpu_ms_per_frame(cmd: list[str], frames: int) -> float:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    seconds = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    return seconds * 1000 / frames


def time_point(args: argparse.Namespace) -> None:
    os.sched_setaffinity(0, {TIMING_CORE})
    freq = pathlib.Path(f"/sys/devices/system/cpu/cpu{TIMING_CORE}/cpufreq")
    for bound in ("scaling_min_freq", "scaling_max_freq"):
        if (freq / bound).read_text().strip() != CORE_KHZ:
            raise SystemExit(f"core {TIMING_CORE} is not pinned at {CORE_KHZ} kHz")
    plan = json.loads(pathlib.Path(args.plan).read_text())
    labels = {"point": args.point, "anchor": plan["anchor"]}
    # The core's clock is locked and the anchor repeats within ~0.3 ms/frame
    # across lock windows (~20 windows, 2026-11-09), so one fresh anchor
    # timing may stand in for the paired one.
    cached_anchor = None
    if args.anchor_from:
        cached_anchor = json.loads(pathlib.Path(args.anchor_from).read_text())["clips"]
        del labels["anchor"]
    result = {"point": args.point, "anchor": plan["anchor"], "platform": JETSON_PLATFORM,
              "rate_mode": plan["rate_mode"], "frames": plan["frames"], "nanocodec_sha256": sha256(
              pathlib.Path(args.nanocodec)), "load_at_start": os.getloadavg()[0], "clips": {},
              "anchor_from": args.anchor_from}
    frames = plan["frames"]
    for clip in plan_clips(plan):
        with tempfile.TemporaryDirectory(dir=JETSON_WORK) as tmp:
            out = pathlib.Path(tmp) / "s.bin"
            cmds = {role: concrete(plan["points"][name][clip]["argv"], plan_input(plan, clip), out,
                                   args.nanocodec)
                    for role, name in labels.items()}
            streams = {}
            for role, cmd in cmds.items() if args.warm_up else ():  # its stream identifies the run
                subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
                streams[role] = {"sha256": sha256(out), "bytes": out.stat().st_size}
            samples: dict[str, list[float]] = {role: [] for role in cmds}
            for round_index in range(args.rounds):
                order = list(cmds) if round_index % 2 == 0 else list(reversed(cmds))
                for role in order:
                    samples[role].append(cpu_ms_per_frame(cmds[role], frames))
                    run = {"sha256": sha256(out), "bytes": out.stat().st_size}
                    if streams.setdefault(role, run)["sha256"] != run["sha256"]:
                        raise SystemExit(f"{labels[role]} {clip}: stream changed between runs")
        planned = plan["points"][args.point][clip]
        if streams["point"]["sha256"] != planned.get("sha256", streams["point"]["sha256"]) \
                or streams["point"]["bytes"] != planned["bytes"]:
            # recorded; merge_timing refuses the point (the timed stream is not the scored one)
            streams["point"].update(planned_bytes=planned["bytes"],
                                    planned_sha256=planned.get("sha256"))
        result["clips"][clip] = {
            "ms_per_frame": {r: statistics.median(s) for r, s in samples.items()},
            "samples": samples, "streams": streams, "kbps": plan["clips"][clip],
            "rate_fraction": rate_fraction(plan["points"][args.point][clip],
                                           streams["point"]["bytes"])}
        if cached_anchor:
            result["clips"][clip]["ms_per_frame"]["anchor"] = \
                cached_anchor[clip]["ms_per_frame"]["anchor"]
        med = result["clips"][clip]["ms_per_frame"]
        print(f"{args.point} {clip}: {med['point']:.1f} ms/f, anchor {med['anchor']:.1f} ms/f, "
              f"{result['clips'][clip]['rate_fraction']:.3f} of the CBR target", flush=True)
        pathlib.Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


# --- time-host (PC) -------------------------------------------------------------------

def pinned_cpu_ms_per_frame(cmd: list[str], cpu: int, frames: int,
                            env: dict[str, str] | None = None) -> float:
    """CPU time per frame of one run pinned to `cpu` (this child's own rusage,
    so concurrent timing threads do not mix), with `env` added to the environment."""
    # taskset execs the encoder in place, so the rusage is the encoder's
    child = subprocess.Popen(["taskset", "-c", str(cpu), *cmd], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                             env={**os.environ, **env} if env else None)
    _, status, usage = os.wait4(child.pid, 0)
    if status != 0:
        raise SystemExit(f"{cmd[0]} failed (status {status})")
    return (usage.ru_utime + usage.ru_stime) * 1000 / frames


def host_time_clip(plan: dict, name: str, clip: str, tools: dict[str, str],
                   clip_paths: dict[str, pathlib.Path], cpu: int, rounds: int,
                   work_dir: pathlib.Path, strict: bool = True,
                   anchor_per_second: float | None = None) -> dict:
    """One point and the second anchor on one clip and one CPU: a warm-up, then
    `rounds` rounds back to back in rotated order. The anchor (x264 medium, the
    slowest program by far: ~60 % of every round) runs in the rounds only without
    `anchor_per_second`, the session's anchor / second-anchor time ratio on this clip
    (`host_anchor_ratio`); with it the point's ratio to the anchor is its per-round
    ratio to the second anchor divided by that (a pair took ~76 s with the anchor in
    every round, ~25 s without).

    The boosting cores' clock follows the whole package's load and moves within a
    session, so each round's point time is divided by the anchors' times of that same
    round and the median per-round ratio is kept (2026-11-12: the x264 anchor's
    minimum, taken from other rounds than the point's, varied 1.4-1.6x within one
    session). Medians of the raw times are kept for reference. Not `strict`, a stream
    of another size than the plan's is recorded (`planned_bytes`), not fatal."""
    labels = {"point": name, SECOND_ANCHOR: SECOND_ANCHOR}
    if anchor_per_second is None:
        labels["anchor"] = plan["anchor"]
    reference = "anchor" if anchor_per_second is None else SECOND_ANCHOR
    # a point may name a partner timed in the same rounds on the same core (e.g. the
    # same encoder on another SIMD tier): its time ratio is then a direct pairing
    partner = plan["points"][name][clip].get("pair_with")
    if partner:
        labels["partner"] = partner
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        out = pathlib.Path(tmp) / "s.bin"
        cmds = {role: [tools[n]] + [str(clip_paths[clip]) if a == "{INPUT}" else
                                    str(out) if a == "{OUTPUT}" else a
                                    for a in plan["points"][n][clip]["argv"][1:]]
                for role, n in labels.items()}
        # a plan entry's "env" (e.g. NC_DSP_ISA=avx2: the same binary on another SIMD tier)
        envs = {role: plan["points"][n][clip].get("env") for role, n in labels.items()}
        streams = {}
        for role, cmd in cmds.items():  # warm-up; the stream must be the planned one
            pinned_cpu_ms_per_frame(cmd, cpu, plan["frames"], envs[role])
            streams[role] = {"sha256": sha256(out), "bytes": out.stat().st_size}
            planned_entry = plan["points"][labels[role]][clip]
            planned = planned_entry["bytes"]
            # the timed stream must be the scored one: its hash where the plan has it
            differs = (streams[role]["sha256"] != planned_entry["sha256"]
                       if "sha256" in planned_entry else streams[role]["bytes"] != planned)
            if differs:
                streams[role]["planned_bytes"] = planned
                streams[role]["planned_sha256"] = planned_entry.get("sha256")
            if differs and strict:
                raise SystemExit(f"{labels[role]} {clip}: {streams[role]['bytes']} bytes, "
                                 f"planned {planned}")
        roles = list(cmds)
        samples: dict[str, list[float]] = {role: [] for role in roles}
        for round_index in range(rounds):
            shift = round_index % len(roles)
            for role in roles[shift:] + roles[:shift]:
                samples[role].append(pinned_cpu_ms_per_frame(cmds[role], cpu, plan["frames"],
                                                             envs[role]))
            if ratio_settled(samples["point"], samples[reference]):
                break
    ms_per_frame = {r: statistics.median(v) for r, v in samples.items()}
    time_ratio = {labels[r]: statistics.median(p / a for p, a in zip(samples["point"], samples[r]))
                  for r in roles if r != "point"}
    if anchor_per_second is not None:
        time_ratio[plan["anchor"]] = time_ratio[SECOND_ANCHOR] / anchor_per_second
        ms_per_frame["anchor"] = ms_per_frame[SECOND_ANCHOR] * anchor_per_second
    return {"ms_per_frame": ms_per_frame, "time_ratio": time_ratio,
            "anchor_per_second": anchor_per_second,
            "samples": samples, "streams": streams, "cpu": cpu,
            "kbps": plan["clips"][clip],
            "rate_fraction": rate_fraction(plan["points"][name][clip], streams["point"]["bytes"])}


def host_anchor_ratio(plan: dict, clip: str, tools: dict[str, str],
                      clip_paths: dict[str, pathlib.Path], cpu: int, rounds: int,
                      work_dir: pathlib.Path) -> float:
    """The anchor's time over the second anchor's on `clip`, this session, this CPU:
    their median per-round ratio (warm-up first, rounds until it settles)."""
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        out = pathlib.Path(tmp) / "s.bin"
        cmds = [[tools[n]] + [str(clip_paths[clip]) if a == "{INPUT}" else
                              str(out) if a == "{OUTPUT}" else a
                              for a in plan["points"][n][clip]["argv"][1:]]
                for n in (plan["anchor"], SECOND_ANCHOR)]
        for cmd in cmds:
            pinned_cpu_ms_per_frame(cmd, cpu, plan["frames"])
        anchor, second = [], []
        for round_index in range(rounds):
            for cmd in (cmds if round_index % 2 == 0 else cmds[::-1]):
                (anchor if cmd is cmds[0] else second).append(
                    pinned_cpu_ms_per_frame(cmd, cpu, plan["frames"]))
            if ratio_settled(anchor, second):
                break
    return statistics.median(a / b for a, b in zip(anchor, second))


def ratio_settled(point: list[float], anchor: list[float]) -> bool:
    """Whether the per-round point/anchor time ratios are tight enough to stop
    (HOST_MIN_ROUNDS, HOST_TARGET_HALF_WIDTH)."""
    if len(point) < HOST_MIN_ROUNDS:
        return False
    logs = [math.log(p / a) for p, a in zip(point, anchor)]
    return Z95 * statistics.stdev(logs) / math.sqrt(len(logs)) <= HOST_TARGET_HALF_WIDTH


def time_host(args: argparse.Namespace) -> None:
    """Every (point, clip) pair on the next free CPU: the pairing with the anchor is
    per clip and CPU, so one point's clips run side by side on the CPUs."""
    from nceval.chart import chart_clips, chart_families

    plan = json.loads(pathlib.Path(args.plan).read_text())
    names = args.points.split(",")
    families = chart_families(pathlib.Path(args.binary).resolve(),
                              pathlib.Path(args.presets_dir), [*names, SECOND_ANCHOR])
    clips = {c.name: c for c in chart_clips(plan["clipset"])}
    tools = {f.name: str(f.tool) for f in families}
    # the reserved cores are this job's alone (host-timed.sh), so it takes no shared CPU slots
    time_pairs(plan, names, tools, {c: clips[plan_input(plan, c)].yuv for c in plan_clips(plan)},
               [int(c) for c in args.cpus.split(",")], args.rounds,
               pathlib.Path(args.out_dir), pathlib.Path.home() / "nanocodec-data/work",
               HOST_PLATFORM, strict=True)


def time_pairs(plan: dict, names: list[str], tools: dict[str, str],
               clip_paths: dict[str, pathlib.Path], cpus: list[int], rounds: int,
               out_dir: pathlib.Path, work_dir: pathlib.Path, platform: str,
               strict: bool) -> None:
    """Every (point, clip) pair, in `names` order, on the next free CPU; a point's
    record is written when its last clip is done. Not `strict`, a failed pair is
    recorded in errors.json and the others go on."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n in names if not (out_dir / f"t-{n}.json").is_file()]
    pending: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
    clips = plan_clips(plan)
    for name in names:
        for clip in clips:
            pending.put((name, clip))
    # The anchor's ratio to the second anchor, once per clip this session, the
    # clips side by side on the CPUs: the points' rounds then leave the anchor out.
    with ThreadPoolExecutor(len(cpus)) as pool:
        per_second = dict(zip(clips, pool.map(
            lambda job: host_anchor_ratio(plan, job[0], tools, clip_paths, job[1], rounds, work_dir),
            [(clip, cpus[i % len(cpus)]) for i, clip in enumerate(clips)])))
    print("anchor / second anchor: " + ", ".join(f"{c} {r:.3f}" for c, r in per_second.items()),
          flush=True)
    timed: dict[str, dict] = {name: {} for name in names}
    errors: dict[str, str] = {}
    lock = threading.Lock()
    total, started = len(names) * len(clips), time.monotonic()

    def progress(name: str, clip: str, outcome: str) -> None:
        """One line per finished pair (the cloud launcher's stall watch reads it)."""
        done = sum(map(len, timed.values())) + len(errors)
        minutes = (time.monotonic() - started) / 60
        print(f"pair {done}/{total} {outcome} {name} {clip}: {minutes:.1f} min, "
              f"ETA {minutes / done * (total - done):.1f} min", flush=True)

    def worker(cpu: int) -> None:
        while True:
            try:
                name, clip = pending.get_nowait()
            except queue.Empty:
                return
            try:
                entry = host_time_clip(plan, name, clip, tools, clip_paths, cpu, rounds,
                                       work_dir, strict, per_second[clip])
            except (Exception, SystemExit) as error:
                if strict:
                    raise
                with lock:
                    errors[f"{name}/{clip}"] = str(error)
                    (out_dir / "errors.json").write_text(json.dumps(errors, indent=1) + "\n")
                    progress(name, clip, f"FAILED ({error})")
                continue
            with lock:
                timed[name][clip] = entry
                progress(name, clip, "ok")
                if len(timed[name]) == len(clips):
                    write_host_record(out_dir, plan, name, timed[name], platform)

    with ThreadPoolExecutor(len(cpus)) as pool:
        for job in [pool.submit(worker, cpu) for cpu in cpus]:
            job.result()


def physical_cpus(skip: int) -> list[int]:
    """One logical CPU per physical core (the first of its SMT siblings), the first
    `skip` cores left to the system."""
    firsts = set()
    for topology in pathlib.Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology"):
        siblings = (topology / "thread_siblings_list").read_text().strip()
        firsts.add(int(siblings.replace("-", ",").split(",")[0]))
    return sorted(firsts)[skip:]


def time_cloud(args: argparse.Namespace) -> None:
    plan = json.loads(pathlib.Path(args.plan).read_text())
    names = args.points.split(",")
    bin_dir = pathlib.Path(args.bin_dir).resolve()
    # a point's pairing partner (pair_with) runs in its rounds, timed or not itself
    first = plan_clips(plan)[0]
    partners = [plan["points"][n][first].get("pair_with") for n in names]
    tools = {n: str(bin_dir / plan["points"][n][first]["argv"][0].removeprefix("{BIN}/"))
             for n in [*names, *filter(None, partners), plan["anchor"], SECOND_ANCHOR]}
    clips_dir = pathlib.Path(args.clips_dir).resolve()
    cpus = (physical_cpus(args.skip_cores) if args.cpus == "physical"
            else [int(c) for c in args.cpus.split(",")])
    if not cpus:  # e.g. a single-core box: name its timed CPU with --cpus
        raise SystemExit(f"no physical core left after skipping {args.skip_cores}")
    print(f"{len(names)} points x {len(plan['clips'])} clips on {len(cpus)} CPUs: {cpus}",
          flush=True)
    work_dir = pathlib.Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    time_pairs(plan, names, tools,
               {c: clips_dir / f"{plan_input(plan, c)}.yuv" for c in plan_clips(plan)}, cpus,
               args.rounds, pathlib.Path(args.out_dir), work_dir, args.platform, strict=False)


def write_host_record(out_dir: pathlib.Path, plan: dict, name: str, clips: dict,
                      platform: str) -> None:
    record = {"point": name, "anchor": plan["anchor"], "platform": platform,
              "rate_mode": plan["rate_mode"], "frames": plan["frames"],
              "load_at_start": os.getloadavg()[0], "clips": {c: clips[c] for c in plan_clips(plan)}}
    (out_dir / f"t-{name}.json").write_text(json.dumps(record, indent=1) + "\n")
    ratios = {ref: statistics.geometric_mean([c["time_ratio"][ref] for c in clips.values()])
              for ref in (plan["anchor"], SECOND_ANCHOR)}
    print(f"{name:28s} cpus {','.join(str(clips[c]['cpu']) for c in plan_clips(plan))} "
          + ", ".join(f"{r:.3f}x {ref}'s time" for ref, r in ratios.items()), flush=True)


# --- merge -----------------------------------------------------------------------------

def merge(args: argparse.Namespace) -> None:
    """One timing.json: per point, its median ms/frame per clip; the anchor's
    from its own entry, else the mean of its paired runs in every point's lock."""
    points, anchor_runs = {}, {}
    for path in map(pathlib.Path, args.files):
        record = json.loads(path.read_text())
        clips = record["clips"]
        points[record["point"]] = {"ms_per_frame": {c: v["ms_per_frame"]["point"]
                                                    for c, v in clips.items()},
                                   "rate_fraction": {c: v["rate_fraction"]
                                                     for c, v in clips.items()},
                                   "anchor_ms_per_frame": {c: v["ms_per_frame"]["anchor"]
                                                           for c, v in clips.items()}}
        for clip, value in clips.items():
            anchor_runs.setdefault(clip, []).append(value["ms_per_frame"]["anchor"])
        anchor = record["anchor"]
        platform = record.get("platform", JETSON_PLATFORM)
        settings = {"rate_mode": record["rate_mode"], "frames": record["frames"],
                    "clips": {c: v["kbps"] for c, v in clips.items()}}
    points.setdefault(anchor, {"ms_per_frame": {c: statistics.mean(v)
                                                for c, v in anchor_runs.items()}})
    if args.machine:  # the rented machine the time-cloud records do not name
        platform = f"{platform} ({args.machine})"
    meta = {"platform": platform, **settings, "anchor": anchor}
    if args.build:  # one session timed every point of ours at one build
        meta["build"] = args.build
    for point, build in (b.split("=", 1) for b in args.build_for):
        if point not in points:  # a typo would leave the point on the record's build
            raise SystemExit(f"--build-for {point}: not among the merged points")
        points[point]["build"] = build  # timed on another build than the record's
    out = pathlib.Path(args.out)
    if args.update:  # these points re-timed into an existing record; the rest stay as timed
        record = json.loads(out.read_text())
        for point, entry in points.items():
            if point == anchor:  # the record's anchor keeps its own session's time
                continue
            ours = point.startswith("nanocodec-")  # a reference tool has no build of ours
            record["points"][point] = ({**entry, "build": entry.get("build", args.build)}
                                       if ours and args.build else entry)
        out.write_text(json.dumps(record, indent=1) + "\n")
        return
    out.write_text(json.dumps({**meta, "points": points}, indent=1) + "\n")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--binary", required=True)
    s.add_argument("--presets-dir", required=True)
    s.add_argument("--plan", required=True, help="the plan's record (the repository's); the "
                   "full plan goes to --store")
    s.add_argument("--store", help="where the full plan is written (default: "
                   "nceval.chart_timing.PLAN_STORE; a build-box job writes into its tree and "
                   "fetches it)")
    s.add_argument("--all-encodes", action="store_true",
                   help="every quality encode (all clips and rungs, low rungs on --low-clips): "
                        "the published speed axis; without it one rung on three clips")
    s.add_argument("--bytes-from", help="a `sizes` file of the chart's quality streams: an "
                        "encode found there is planned at its size, the rest are encoded")
    s.add_argument("--low-clips", help="comma-separated clips with the chart's low rungs "
                   "(its render's coverage note)")
    s.add_argument("--clipset", help="the chart's clip set (nceval.chart --clipset; default: "
                   "the main chart's VAL clips)")
    s.add_argument("--points", help="comma-separated families (default: every chart point)")
    s.add_argument("--binary-for", action="append", default=[], metavar="FAMILY=PATH",
                   help="a family on another build than --binary (a chart's record)")
    s.add_argument("--presets-for", action="append", default=[], metavar="FAMILY=DIR",
                   help="that build's presets directory")
    s.add_argument("--workers", type=int, default=PLAN_WORKERS,
                   help=f"streams sized side by side (default {PLAN_WORKERS}; a rented box "
                        "takes more)")
    t = sub.add_parser("time")
    t.add_argument("--plan", required=True)
    t.add_argument("--point", required=True)
    t.add_argument("--nanocodec", required=True, help="our aarch64 build of the same commit")
    t.add_argument("--out", required=True)
    t.add_argument("--rounds", type=int, default=ROUNDS,
                   help="timed runs of each command per clip (an every-encode plan: 1, its "
                        "interval over the entries instead)")
    t.add_argument("--no-warm-up", dest="warm_up", action="store_false",
                   help="no untimed first run (the first timed run's stream identifies it)")
    t.add_argument("--anchor-from", help="a timing file whose anchor times to reuse "
                   "instead of timing the anchor in this lock window")
    h = sub.add_parser("time-host")
    h.add_argument("--plan", required=True)
    h.add_argument("--binary", required=True, help="our x86 build (the plan's)")
    h.add_argument("--presets-dir", required=True)
    h.add_argument("--points", required=True, help="comma-separated families")
    h.add_argument("--cpus", required=True,
                   help="reserved timing CPUs (host-timed.sh), one per physical core")
    h.add_argument("--out-dir", required=True)
    h.add_argument("--rounds", type=int, default=HOST_ROUNDS,
                   help=f"most interleaved rounds per clip (default {HOST_ROUNDS}; a pair stops "
                        f"earlier once its ratio settles, ratio_settled; "
                        "more on a loaded PC)")
    c = sub.add_parser("time-cloud")
    c.add_argument("--plan", required=True)
    c.add_argument("--bin-dir", required=True, help="every tool the points' argv name")
    c.add_argument("--clips-dir", required=True, help="<clip>.yuv, the plan's frames each")
    c.add_argument("--points", required=True, help="comma-separated plan points")
    c.add_argument("--cpus", default="physical",
                   help="comma-separated CPUs, or 'physical' (one per physical core)")
    c.add_argument("--skip-cores", type=int, default=2,
                   help="with --cpus physical: the first cores left to the system")
    c.add_argument("--rounds", type=int, default=HOST_ROUNDS)
    c.add_argument("--platform", required=True, help="the platform note of the records")
    c.add_argument("--work-dir", required=True, help="scratch for the streams")
    c.add_argument("--out-dir", required=True)
    z = sub.add_parser("sizes")
    z.add_argument("--streams", required=True, help="a chart run's stream directory")
    z.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("--machine", help="appended to the platform, e.g. 'AWS c8g.large, one pair "
                   "at a time'")
    m.add_argument("--build", help="the commit (8 characters) our points were all timed at")
    m.add_argument("--build-for", action="append", default=[], metavar="POINT=COMMIT",
                   help="a point of ours timed on another build than --build's")
    m.add_argument("--update", action="store_true",
                   help="replace these points (each on --build) in the existing --out record, "
                        "keeping its other points, anchor and description")
    m.add_argument("files", nargs="+")
    args = parser.parse_args(argv)
    {"plan": make_plan, "time": time_point, "time-host": time_host, "time-cloud": time_cloud,
     "merge": merge, "sizes": write_sizes}[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
