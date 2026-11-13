"""Encode speed for the bitrate-vs-speed chart (eval/nceval/chart.py).

Every chart point is timed single-threaded at a fixed quantizer that lands it
near the 600 kbps rung of the real-use ladder (720p30, 150 frames) on three
clips (low light, crowd, bodycam), so all encoders are timed doing the same
amount of compression work. The fixed-QP command lines are the harness's own
(cbr.Family.command: x264/x265 `--qp` with zero latency and no B-frames,
SVT-AV1 `--rtc --rc 0`, openh264 realtime, nanocodec `--qp` at its preset).

  scan  (PC, from eval/ with PYTHONPATH=.): binary-search each point's
        quantizer to the rung's byte budget and write plan.json (command lines
        with {BIN}/{INPUT}/{OUTPUT} placeholders). Our streams are bit-exact
        across platforms, so our QPs carry to the Jetson unchanged; the anchors'
        Jetson stream sizes are recorded beside their timings.
  time  (Jetson, under ~/nanocodec-data/jetson-timed.sh, one point per lock):
        core 3 at its fixed 1.728 GHz, a warm-up run of the point and the anchor
        (x264 medium) per clip, then three rounds in rotated order; encode CPU
        time (rusage) per frame, median per clip. `--anchor-from t-X.json`
        reuses that file's anchor times instead of re-timing the anchor.
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

    python3 tools/dev/anchor_chart_timing.py scan --binary BIN --presets-dir DIR --plan plan.json
    jetson-timed.sh python3 anchor_chart_timing.py time --plan plan.json --point x265-medium \
        --nanocodec BIN --out t-x265-medium.json
    PYTHONPATH=. ../tools/dev/host-timed.sh python3 ../tools/dev/anchor_chart_timing.py \
        time-host --plan plan.json --binary BIN --presets-dir DIR --points a,b --cpus 14,15 \
        --out-dir host/
    python3 tools/dev/anchor_chart_timing.py merge --out timing.json t-*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from dataclasses import replace

#: The rung the speed axis is timed at (kbps at 720p30) and the clip length.
RUNG_KBPS = 600
FRAMES = 150
INTRA_PERIOD = 300
#: Timing clips (the scoreboard's Jetson timing set) and their 150-frame copies on the Jetson.
CLIPS = ("lowlight_city_night", "crowd_mot17_09", "bodycam_bc1")
JETSON_WORK = pathlib.Path.home() / "nanocodec-data/work"
JETSON_CLIPS = {
    "lowlight_city_night": JETSON_WORK / "av1-cdef-joint-native15/lowlight_city_night-150f.yuv",
    "crowd_mot17_09": JETSON_WORK / "adopted-fast-reference-native/crowd_mot17_09-150f.yuv",
    "bodycam_bc1": JETSON_WORK / "adopted-fast-reference-native/bodycam_bc1-150f.yuv",
}
JETSON_TOOLS = pathlib.Path.home() / "nanocodec-data/anchors/aarch64/bin"
TIMING_CORE = 3
JETSON_PLATFORM = "Jetson Orin Nano, Cortex-A78AE core 3 at 1.728 GHz, one thread"
CORE_KHZ = "1728000"
ROUNDS = 3
QUIET_LOAD = 1.5  # jetson-timed.sh's own refusal threshold
#: Host timing: rounds per clip (the median per-round ratio is kept) and its
#: platform note. 5 rounds mis-timed points at load ~35 (2026-11-12: AV1 Fast level
#: with AV1 Faster), so 11.
HOST_ROUNDS = 11
#: Host timing's second anchor, timed in the same rounds: the real-time reference our
#: fast presets are compared with, as a direct same-session time ratio.
SECOND_ANCHOR = "svtav1-m13"
HOST_PLATFORM = "Ryzen 9 5950X (x86-64, AVX2), one thread pinned to one core"
SCAN_WORKERS = 3  # one per timing clip
#: Concurrent quantizer probes per scanned point (each holds one CPU slot).
SCAN_PROBES = 3


# --- scan (PC) -------------------------------------------------------------------------

def target_bytes(fps: float) -> int:
    return round(RUNG_KBPS * 1000 / 8 * FRAMES / fps)


def scan_point(family, clip) -> dict:
    """The quantizer whose stream size is closest to the rung's budget (sizes fall with QP)."""
    from nceval.slots import cpu_slots

    grid = scan_grid(family)
    sizes: dict[int, int] = {}
    goal = target_bytes(clip.fps)
    with tempfile.TemporaryDirectory(dir=pathlib.Path.home() / "nanocodec-data/work") as tmp:

        def size(index: int) -> int:
            out = pathlib.Path(tmp) / f"s{index}.bin"
            cmd = family.command(clip, FRAMES, INTRA_PERIOD, out, None, qp=grid[index])
            with cpu_slots(1):
                subprocess.run(["nice", "-n", "19", *cmd], check=True, capture_output=True,
                               stdin=subprocess.DEVNULL)
            return out.stat().st_size

        with ThreadPoolExecutor(SCAN_PROBES) as pool:
            def probe(indices: list[int]) -> None:
                wanted = [i for i in dict.fromkeys(indices) if i not in sizes]
                sizes.update(zip(wanted, pool.map(size, wanted)))

            # Bisection with SCAN_PROBES probes a round: the same adjacent pair
            # (size(lo) >= goal > size(hi)) in log_(probes+1) rounds, not log_2.
            lo, hi = 0, len(grid) - 1
            while hi - lo > 1:
                step = (hi - lo) / (SCAN_PROBES + 1)
                cuts = sorted({lo + max(1, round(step * k)) for k in range(1, SCAN_PROBES + 1)}
                              - {hi})
                probe(cuts)
                for cut in cuts:
                    lo, hi = (cut, hi) if sizes[cut] >= goal else (lo, min(hi, cut))
                    if cut >= hi:
                        break
            probe([lo, hi])
        best = min((lo, hi), key=lambda i: abs(sizes[i] - goal))
        cmd = family.command(clip, FRAMES, INTRA_PERIOD, pathlib.Path("{OUTPUT}"), None,
                             qp=grid[best])
    argv = ["{BIN}/" + pathlib.Path(cmd[0]).name if i == 0 else
            "{INPUT}" if a == str(clip.yuv) else a for i, a in enumerate(cmd)]
    return {"qp": grid[best], "bytes": sizes[best], "target_bytes": goal, "argv": argv,
            "fps": clip.fps}


def plan_family(family):
    """A chart family in the timing plan's form: ours by `--preset NAME`, so every
    build encodes with its own compiled preset. The plan used to store the preset's
    values as a frozen --param list from the scanning build, which went stale when a
    preset gained a value (2026-11-12: vp9-fast and av2-fast without
    planner.lambda_scale=2, timed as something other than the shipped preset)."""
    if not family.own:
        return family
    preset = family.name.partition("@")[2]
    return replace(family, extra=("--preset", preset) if preset else ())


def scan_grid(family) -> list[int]:
    """The quantizers a point's encoder takes."""
    return (list(range(10, 52)) if family.own and family.codec.value in ("h264", "hevc")
            else list(range(10, 64)) if family.own and family.codec.value == "vvc"
            else list(range(1, 256)) if family.own
            else list(range(1, 64)) if family.kind in ("svt", "vpx", "aom")
            else list(range(10, 52)))


def scan_key(family, clip, tool_digest: str) -> str:
    """What a scanned quantizer depends on: the encoder binary's content, its
    settings, the clip and the budget. A plan entry with the same key is reused
    (the scan is deterministic: our streams and the anchors' are bit-exact reruns)."""
    parts = [tool_digest, family.extra, family.preset, clip.name, clip.yuv.stat().st_size,
             FRAMES, INTRA_PERIOD, target_bytes(clip.fps), scan_grid(family)]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]


def scan(args: argparse.Namespace) -> None:
    from nceval.chart import ANCHOR, chart_families
    from nceval.config import load_paths
    from nceval.realuse import main_clips

    families = [plan_family(f) for f in chart_families(
        pathlib.Path(args.binary).resolve(), pathlib.Path(args.presets_dir),
        args.points.split(",") if args.points else None)]
    clips = [main_clips(load_paths())[name] for name in CLIPS]
    plan_path = pathlib.Path(args.plan)
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {"points": {}}
    plan.update(anchor=ANCHOR, rung_kbps=RUNG_KBPS, frames=FRAMES)
    digest = hashlib.sha256(pathlib.Path(args.binary).read_bytes()).hexdigest()
    plan.setdefault("nanocodec_sha256", {}).update(
        {f.name: digest for f in families if f.own})  # per point: builds may differ
    tool_digests = {f.tool: sha256(f.tool) for f in families}
    keys = {(f.name, c.name): scan_key(f, c, tool_digests[f.tool]) for f in families for c in clips}
    with ThreadPoolExecutor(SCAN_WORKERS) as pool:
        jobs = {}
        for f in families:
            for c in clips:
                stored = plan["points"].get(f.name, {}).get(c.name, {})
                jobs[(f.name, c.name)] = (None if stored.get("key") == keys[(f.name, c.name)]
                                          else pool.submit(scan_point, f, c))
        for (name, clip), job in jobs.items():
            entry = plan["points"][name][clip] if job is None else {
                **job.result(), "key": keys[(name, clip)]}
            plan["points"].setdefault(name, {})[clip] = entry
            print(f"{name:26s} {clip:20s} qp {entry['qp']:3d} "
                  f"{entry['bytes'] / entry['target_bytes']:.3f} of the budget"
                  f"{' (cached)' if job is None else ''}", flush=True)
    plan_path.write_text(json.dumps(plan, indent=1) + "\n")


# --- time (Jetson) ---------------------------------------------------------------------

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def concrete(argv: list[str], clip: str, out: pathlib.Path, nanocodec: str) -> list[str]:
    tool = argv[0].removeprefix("{BIN}/")
    head = nanocodec if tool == "nanocodec" else str(JETSON_TOOLS / tool)
    return [head] + [str(JETSON_CLIPS[clip]) if a == "{INPUT}" else
                     str(out) if a == "{OUTPUT}" else a for a in argv[1:]]


def cpu_ms_per_frame(cmd: list[str]) -> float:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    seconds = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    return seconds * 1000 / FRAMES


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
    result = {"point": args.point, "anchor": plan["anchor"], "nanocodec_sha256": sha256(
              pathlib.Path(args.nanocodec)), "load_at_start": os.getloadavg()[0], "clips": {},
              "anchor_from": args.anchor_from}
    for clip in CLIPS:
        with tempfile.TemporaryDirectory(dir=JETSON_WORK) as tmp:
            out = pathlib.Path(tmp) / "s.bin"
            cmds = {role: concrete(plan["points"][name][clip]["argv"], clip, out, args.nanocodec)
                    for role, name in labels.items()}
            streams = {}
            for role, cmd in cmds.items():  # warm-up; the stream identifies the run
                subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
                streams[role] = {"sha256": sha256(out), "bytes": out.stat().st_size}
            samples: dict[str, list[float]] = {role: [] for role in cmds}
            for round_index in range(ROUNDS):
                order = list(cmds) if round_index % 2 == 0 else list(reversed(cmds))
                for role in order:
                    samples[role].append(cpu_ms_per_frame(cmds[role]))
                    if sha256(out) != streams[role]["sha256"]:
                        raise SystemExit(f"{labels[role]} {clip}: stream changed between runs")
        expected = plan["points"][args.point][clip]["bytes"]
        if args.point.startswith("nanocodec") and streams["point"]["bytes"] != expected:
            raise SystemExit(f"{args.point} {clip}: stream differs from the PC scan")
        result["clips"][clip] = {
            "ms_per_frame": {r: statistics.median(s) for r, s in samples.items()},
            "samples": samples, "streams": streams,
            "budget_fraction": streams["point"]["bytes"]
            / plan["points"][args.point][clip]["target_bytes"]}
        if cached_anchor:
            result["clips"][clip]["ms_per_frame"]["anchor"] = \
                cached_anchor[clip]["ms_per_frame"]["anchor"]
        med = result["clips"][clip]["ms_per_frame"]
        print(f"{args.point} {clip}: {med['point']:.1f} ms/f, anchor {med['anchor']:.1f} ms/f, "
              f"{result['clips'][clip]['budget_fraction']:.3f} of the byte budget", flush=True)
        pathlib.Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


# --- time-host (PC) -------------------------------------------------------------------

def pinned_cpu_ms_per_frame(cmd: list[str], cpu: int, env: dict[str, str] | None = None
                            ) -> float:
    """CPU time per frame of one run pinned to `cpu` (this child's own rusage,
    so concurrent timing threads do not mix), with `env` added to the environment."""
    # taskset execs the encoder in place, so the rusage is the encoder's
    child = subprocess.Popen(["taskset", "-c", str(cpu), *cmd], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                             env={**os.environ, **env} if env else None)
    _, status, usage = os.wait4(child.pid, 0)
    if status != 0:
        raise SystemExit(f"{cmd[0]} failed (status {status})")
    return (usage.ru_utime + usage.ru_stime) * 1000 / FRAMES


def host_time_clip(plan: dict, name: str, clip: str, tools: dict[str, str],
                   clip_paths: dict[str, pathlib.Path], cpu: int, rounds: int,
                   work_dir: pathlib.Path, strict: bool = True) -> dict:
    """One point, the anchor and the second anchor on one clip and one CPU: a warm-up,
    then `rounds` rounds of the three back to back in rotated order.

    The boosting cores' clock follows the whole package's load and moves within a
    session, so each round's point time is divided by the anchors' times of that same
    round and the median per-round ratio is kept (2026-11-12: the x264 anchor's
    minimum, taken from other rounds than the point's, varied 1.4-1.6x within one
    session). Medians of the raw times are kept for reference. Not `strict`, a stream
    of another size than the plan's is recorded (`planned_bytes`), not fatal."""
    labels = {"point": name, "anchor": plan["anchor"], SECOND_ANCHOR: SECOND_ANCHOR}
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
            pinned_cpu_ms_per_frame(cmd, cpu, envs[role])
            streams[role] = {"sha256": sha256(out), "bytes": out.stat().st_size}
            planned = plan["points"][labels[role]][clip]["bytes"]
            if streams[role]["bytes"] != planned:
                streams[role]["planned_bytes"] = planned
            if streams[role]["bytes"] != planned and strict:
                raise SystemExit(f"{labels[role]} {clip}: {streams[role]['bytes']} bytes, "
                                 f"planned {planned}")
        roles = list(cmds)
        samples: dict[str, list[float]] = {role: [] for role in roles}
        for round_index in range(rounds):
            shift = round_index % len(roles)
            for role in roles[shift:] + roles[:shift]:
                samples[role].append(pinned_cpu_ms_per_frame(cmds[role], cpu, envs[role]))
    return {"ms_per_frame": {r: statistics.median(v) for r, v in samples.items()},
            "time_ratio": {labels[r]: statistics.median(p / a for p, a in
                                                        zip(samples["point"], samples[r]))
                           for r in roles if r != "point"},
            "samples": samples, "streams": streams, "cpu": cpu,
            "budget_fraction": streams["point"]["bytes"]
            / plan["points"][name][clip]["target_bytes"]}


def time_host(args: argparse.Namespace) -> None:
    """Every (point, clip) pair on the next free CPU: the pairing with the anchor is
    per clip and CPU, so one point's clips run side by side on the CPUs."""
    from nceval.chart import chart_families
    from nceval.config import load_paths
    from nceval.realuse import main_clips

    plan = json.loads(pathlib.Path(args.plan).read_text())
    names = args.points.split(",")
    families = chart_families(pathlib.Path(args.binary).resolve(),
                              pathlib.Path(args.presets_dir), [*names, SECOND_ANCHOR])
    clips = main_clips(load_paths())
    sample = clips[CLIPS[0]]
    tools = {f.name: str(f.command(sample, FRAMES, INTRA_PERIOD, pathlib.Path("o"), None,
                                   qp=1)[0]) for f in families}
    # the reserved cores are this job's alone (host-timed.sh), so it takes no shared CPU slots
    time_pairs(plan, names, tools, {c: clips[c].yuv for c in CLIPS},
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
    for name in names:
        for clip in CLIPS:
            pending.put((name, clip))
    timed: dict[str, dict] = {name: {} for name in names}
    errors: dict[str, str] = {}
    lock = threading.Lock()
    total, started = len(names) * len(CLIPS), time.monotonic()

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
                                       work_dir, strict)
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
                if len(timed[name]) == len(CLIPS):
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
    tools = {n: str(bin_dir / plan["points"][n][CLIPS[0]]["argv"][0].removeprefix("{BIN}/"))
             for n in [*names, plan["anchor"], SECOND_ANCHOR]}
    clips_dir = pathlib.Path(args.clips_dir).resolve()
    cpus = (physical_cpus(args.skip_cores) if args.cpus == "physical"
            else [int(c) for c in args.cpus.split(",")])
    print(f"{len(names)} points x {len(CLIPS)} clips on {len(cpus)} CPUs: {cpus}", flush=True)
    work_dir = pathlib.Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    time_pairs(plan, names, tools, {c: clips_dir / f"{c}.yuv" for c in CLIPS}, cpus,
               args.rounds, pathlib.Path(args.out_dir), work_dir, args.platform, strict=False)


def write_host_record(out_dir: pathlib.Path, plan: dict, name: str, clips: dict,
                      platform: str) -> None:
    record = {"point": name, "anchor": plan["anchor"], "platform": platform,
              "load_at_start": os.getloadavg()[0], "clips": {c: clips[c] for c in CLIPS}}
    (out_dir / f"t-{name}.json").write_text(json.dumps(record, indent=1) + "\n")
    ratios = {ref: statistics.geometric_mean([c["time_ratio"][ref] for c in clips.values()])
              for ref in (plan["anchor"], SECOND_ANCHOR)}
    print(f"{name:28s} cpus {','.join(str(clips[c]['cpu']) for c in CLIPS)} "
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
                                   "budget_fraction": {c: v["budget_fraction"]
                                                       for c, v in clips.items()},
                                   "anchor_ms_per_frame": {c: v["ms_per_frame"]["anchor"]
                                                           for c, v in clips.items()}}
        for clip, value in clips.items():
            anchor_runs.setdefault(clip, []).append(value["ms_per_frame"]["anchor"])
        anchor = record["anchor"]
        platform = record.get("platform", JETSON_PLATFORM)
    points.setdefault(anchor, {"ms_per_frame": {c: statistics.mean(v)
                                                for c, v in anchor_runs.items()}})
    meta = {"platform": platform,
            "rung_kbps": RUNG_KBPS, "frames": FRAMES, "clips": list(CLIPS), "anchor": anchor}
    pathlib.Path(args.out).write_text(json.dumps({**meta, "points": points}, indent=1) + "\n")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--binary", required=True)
    s.add_argument("--presets-dir", required=True)
    s.add_argument("--plan", required=True)
    s.add_argument("--points", help="comma-separated families (default: every chart point)")
    t = sub.add_parser("time")
    t.add_argument("--plan", required=True)
    t.add_argument("--point", required=True)
    t.add_argument("--nanocodec", required=True, help="our aarch64 build of the same commit")
    t.add_argument("--out", required=True)
    t.add_argument("--anchor-from", help="a timing file whose anchor times to reuse "
                   "instead of timing the anchor in this lock window")
    h = sub.add_parser("time-host")
    h.add_argument("--plan", required=True)
    h.add_argument("--binary", required=True, help="our x86 build (the scan's)")
    h.add_argument("--presets-dir", required=True)
    h.add_argument("--points", required=True, help="comma-separated families")
    h.add_argument("--cpus", required=True,
                   help="reserved timing CPUs (host-timed.sh), one per physical core")
    h.add_argument("--out-dir", required=True)
    h.add_argument("--rounds", type=int, default=HOST_ROUNDS,
                   help=f"interleaved rounds per clip, minimum taken (default {HOST_ROUNDS}; "
                        "more on a loaded PC)")
    c = sub.add_parser("time-cloud")
    c.add_argument("--plan", required=True)
    c.add_argument("--bin-dir", required=True, help="every tool the points' argv name")
    c.add_argument("--clips-dir", required=True, help="<clip>.yuv, FRAMES frames each")
    c.add_argument("--points", required=True, help="comma-separated plan points")
    c.add_argument("--cpus", default="physical",
                   help="comma-separated CPUs, or 'physical' (one per physical core)")
    c.add_argument("--skip-cores", type=int, default=2,
                   help="with --cpus physical: the first cores left to the system")
    c.add_argument("--rounds", type=int, default=HOST_ROUNDS)
    c.add_argument("--platform", required=True, help="the platform note of the records")
    c.add_argument("--work-dir", required=True, help="scratch for the streams")
    c.add_argument("--out-dir", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("files", nargs="+")
    args = parser.parse_args(argv)
    {"scan": scan, "time": time_point, "time-host": time_host, "time-cloud": time_cloud,
     "merge": merge}[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
