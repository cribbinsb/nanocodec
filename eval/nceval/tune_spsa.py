"""Fishtest-style tuning: SPSA over a few parameters on fresh draws, then an SPRT against the rung.

Why (docs/handoff/stop-network.md, 2026-11-22): `nceval.tune search` judges every trial on
one fixed set of 16-23 long clips and keeps the best of tens of trials, so it keeps that
set's noise (about 0.2 % of J per trial, 0.3-0.5 % for the best of 30), the size of most
levers: six studies of learned terms found nothing that survived fresh clips. Stockfish's
fishtest tunes and adopts differently, and so does this module:

- **Steering, `spsa`** (nceval.spsa): each iteration draws FRESH samples (below) from the
  steering half of the pool, measures the two perturbed points u +- c_k delta on the same
  samples (common random numbers) and takes one SPSA step. No point is ever chosen for its
  measured score: the walk averages the noise over iterations. CPU for steering is
  callgrind's x86 count of the draw's own encodes (their declared subset at each key
  share, nceval.steering_cpu), priced at the rung's front slope (nceval.front_k).
- **Adoption, `sprt`**: the final point (or any point) against the shipped rung, on samples
  of the OTHER half of the pool that no iteration used, in rounds, until the sequential test
  decides (adopt: it gains `--delta` or more; reject: it gains nothing) or the rounds run
  out (undecided: not adopted). Its CPU is then measured on rented cores
  (tools/dev/aws_cpu_ab.py: Graviton 3 and 4 and Zen 5) on the SPRT's own encodes (the
  rounds' samples, both shares; nceval.decision_cpu) and the verdict is restated with
  it, each sample at its own share's CPU.

Samples. A sample is one clip, one key share (nceval.key_shares: 1 key in 150 frames, 600
frames from its start; 1 in 300, 900 frames) and one start (0, then every window length
while the clip holds another whole window). Clips are the usable pool's; a family is in the
steering or the testing half by a hash of the run's seed, so the two never share content.
Per sample the score is J (single point against the anchor's cached curve, nceval.singlepoint)
plus k * 100 * (CPU ratio - 1) at that share. A draw is stratified over the pool's strata and
takes each (clip, share) at most once.

State: one JSON per run, `$NC_DATA_ROOT/tuning/<name>.spsa.json` (it goes up to the box and
comes back with each slice, as a study's storage does); rerun the same command to resume.

    python3 -m nceval.tune spsa --name hevc-slow-ctx-1 --codec hevc --preset slow \\
        --binary BIN --space tune_spaces/X.txt --iterations 12 --per-draw 8
    python3 -m nceval.tune sprt --name hevc-slow-ctx-1 --delta 0.3 --per-round 8 --max-rounds 4
    python3 -m nceval.tune spsa-report --name hevc-slow-ctx-1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from . import (cloud, decision_cpu, front_k, key_shares, levels, pool, realuse, spsa,
               steering_cpu)
from .clips import Clip
from .codecs import Codec
from .config import load_paths
from .encoders.nanocodec import preset_values
from .gatepolicy import EVALUATION_GATE, file_sha256
from .singlepoint import assign_rungs, balanced_j, rung_ladder, single_points
from .tune import REPO_ROOT, TUNING_DIR, specs_of

STATE_SUFFIX = ".spsa.json"
#: SPRT round r's rung assignment offset (assign_rungs), apart from the iterations'.
SPRT_OFFSET = 1000
STEER, TEST = "steer", "test"
#: Iterations per cloud slice (each slice is one warm-box job, its state up and back).
SLICE_ITERATIONS = 3
#: Minutes a slice may take per iteration on the box, and its fixed part.
MINUTES_PER_ITERATION = 12
SLICE_MINUTES = 10
#: Default gains: perturbation and first step in normalised units, and the gradient
#: (score percent per normalised unit) the first step is sized for.
DEFAULT_C = 0.2
DEFAULT_FIRST_STEP = 0.25
DEFAULT_FIRST_GRADIENT = 1.0
#: Default ceiling of a run's spend.
DEFAULT_MAX_USD = 2.0


@dataclass(frozen=True)
class Sample:
    """One clip at one key share from one start frame."""
    clip: str
    period: int
    start: int

    @property
    def key(self) -> str:
        return f"{self.clip}@{self.start}/{self.period}"


def state_path(name: str) -> Path:
    return TUNING_DIR / f"{name}{STATE_SUFFIX}"


def load_state(name: str) -> dict | None:
    path = state_path(name)
    return json.loads(path.read_text()) if path.exists() else None


def save_state(state: dict) -> None:
    path = state_path(state["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    tmp.replace(path)


# --- samples --------------------------------------------------------------------------

def half_of(family: str, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}:{family}".encode()).digest()
    return STEER if digest[0] % 2 == 0 else TEST


def pool_clips() -> dict[str, Clip]:
    return {c.name: c for c in pool.load_pool(load_paths(), levels.usable)}


def samples_of(clips: dict[str, Clip], half: str, seed: str) -> list[tuple[Sample, str]]:
    """Every (sample, stratum) of `half`."""
    out = []
    for share in key_shares.shares():
        for clip in sorted(clips.values(), key=lambda c: c.name):
            if half_of(clip.family, seed) != half:
                continue
            for start in range(0, clip.frames - share.window + 1, share.window):
                out.append((Sample(clip.name, share.period, start), pool.stratum(clip)))
    return out


def draw(candidates: list[tuple[Sample, str]], count: int, rng: random.Random,
         exclude: set[str] = frozenset()) -> list[Sample]:
    """`count` samples, strata in turn, at most one per (clip, share), none of `exclude`."""
    by_stratum: dict[str, list[Sample]] = {}
    for sample, stratum in candidates:
        if sample.key not in exclude:
            by_stratum.setdefault(stratum, []).append(sample)
    for queue in by_stratum.values():
        rng.shuffle(queue)
    chosen, taken = [], set()
    strata = sorted(by_stratum)
    rng.shuffle(strata)
    while len(chosen) < count and any(by_stratum.values()):
        for stratum in strata:
            queue = by_stratum[stratum]
            while queue:
                sample = queue.pop()
                if (sample.clip, sample.period) not in taken:
                    taken.add((sample.clip, sample.period))
                    chosen.append(sample)
                    break
            if len(chosen) == count:
                break
    return chosen


def iteration_samples(state: dict, k: int, clips: dict[str, Clip]) -> list[Sample]:
    """Iteration `k`'s fresh draw from the steering half (deterministic in the run's seed)."""
    rng = random.Random(f"{state['seed']}:iteration:{k}")
    return draw(samples_of(clips, STEER, state["seed"]), state["per_draw"], rng)


def round_samples(state: dict, r: int, clips: dict[str, Clip], per_round: int) -> list[Sample]:
    """SPRT round `r`'s draw from the testing half, none a previous round used."""
    used = {s for done in state.get("sprt", {}).get("rounds", [])[:r] for s in done["samples"]}
    rng = random.Random(f"{state['seed']}:sprt:{r}")
    return draw(samples_of(clips, TEST, state["seed"]), per_round, rng, used)


def direction(state: dict, k: int) -> list[int]:
    return spsa.direction(random.Random(f"{state['seed']}:direction:{k}"), len(state["names"]))


# --- measurement ----------------------------------------------------------------------

class Measurer:
    """Scores of parameter settings on samples, against the rung as shipped."""

    def __init__(self, state: dict, workers: int):
        self.state = state
        self.codec = Codec(state["codec"])
        self.binary = Path(state["binary"])
        self.preset = state["preset_values"]
        self.k = state["k"]
        self.runner = realuse.Runner(load_paths(), EVALUATION_GATE, workers, over_budget=False)
        self.clips = pool_clips()
        #: The preset's callgrind counts by job (steering_cpu.key_of): u+ and u- share a draw.
        self.anchor_counts: dict[str, int] = {}

    def cpu(self, values: dict[str, float], samples: list[Sample], offset: int
            ) -> dict[int, float]:
        """Per key period, `values`' CPU over the preset's on these samples' own encodes
        (callgrind on their declared subset, nceval.steering_cpu)."""
        from .tune_pool import family, settings_label
        anchor = family(self.codec, self.binary, self.preset, f"nanocodec-{self.codec.value}")
        jobs = decision_cpu.declared_subset(decision_cpu.cpu_jobs(
            sample_jobs(self.runner, anchor, self.clips, samples, offset), anchor.name))
        test = family(self.codec, self.binary, values,
                      f"nanocodec-{self.codec.value}@{settings_label(values)}")
        todo = [j for j in jobs if steering_cpu.key_of(j) not in self.anchor_counts]
        self.anchor_counts |= steering_cpu.instructions(self.binary, anchor.extra, todo,
                                                        self.clips)
        counts = steering_cpu.instructions(self.binary, test.extra, jobs, self.clips)
        return steering_cpu.ratios_by_period(counts, self.anchor_counts, jobs)

    def scores(self, values: dict[str, float], samples: list[Sample], offset: int
               ) -> dict[str, dict]:
        """Per sample {"j", "cpu", "score"}; CPU from callgrind on the draw's own encodes,
        at the sample's share."""
        from .tune_pool import family, settings_label
        anchor = family(self.codec, self.binary, self.preset, f"nanocodec-{self.codec.value}")
        test = family(self.codec, self.binary, values,
                      f"nanocodec-{self.codec.value}@{settings_label(values)}")
        with ThreadPoolExecutor(1) as background:
            cost = background.submit(self.cpu, values, samples, offset)
            rows = {}
            for share, chosen, clips in share_clips(self.clips, samples):
                found = single_points(self.runner, anchor, test, clips,
                                      assign_rungs(clips, offset), share.window,
                                      key_frames=share.period)
                for s in chosen:
                    if s.clip in found:
                        rows[s.key] = {**found[s.clip], "period": share.period}
            ratio = cost.result()
        out = {}
        for key, row in rows.items():
            if row["quality"] is None:
                continue
            cpu = ratio[row["period"]]
            out[key] = {"j": row["quality"], "cpu": cpu, "class": row["class"],
                        "score": row["quality"] + self.k * 100 * (cpu - 1)}
        return out


def share_clips(clips: dict[str, Clip], samples: list[Sample]
                ) -> list[tuple[key_shares.KeyShare, list[Sample], list[Clip]]]:
    """Per key share, its samples and their clips from their start frames, as `scores`
    codes them."""
    out = []
    for share in key_shares.shares():
        chosen = [s for s in samples if s.period == share.period]
        if chosen:
            out.append((share, chosen,
                        [replace(clips[s.clip], start_frame=s.start) for s in chosen]))
    return out


def sample_jobs(runner: realuse.Runner, test, clips: dict[str, Clip], samples: list[Sample],
                offset: int) -> list:
    """The candidate's encodes of one draw (`scores`' single points), for its CPU plan."""
    jobs = []
    for share, _, chosen in share_clips(clips, samples):
        jobs += runner.points([test], chosen, share.window, None, key_frames=share.period,
                              ladder=rung_ladder(assign_rungs(chosen, offset)))
    return jobs


def mean_score(rows: dict[str, dict]) -> float:
    """The class-balanced mean score of a draw (as J is balanced elsewhere)."""
    value = balanced_j(rows, "score")
    if value is None:
        raise ValueError("no sample measured")
    return value


# --- the run --------------------------------------------------------------------------

def values_of(state: dict, u: list[float]) -> dict[str, float]:
    values = dict(state["preset_values"])
    for name, x in zip(state["names"], u):
        d = state["domains"][name]
        value = spsa.to_value(x, d["low"], d["high"])
        values[name] = float(round(value)) if d["kind"] == "int" else value
    return values


def gains_of(state: dict) -> spsa.Gains:
    return spsa.Gains(**state["gains"])


def create(args: argparse.Namespace) -> dict:
    from .tune_pool import read_space
    from .buildinfo import clean_commit
    from . import levels as levels_module
    codec = Codec(args.codec)
    binary, commit = levels_module.build(args.binary, load_paths(), "tune")
    commit = commit or clean_commit(binary)
    specs = specs_of(binary, codec)
    presets = binary.parent / "presets"
    presets = presets if presets.is_dir() else REPO_ROOT / "crates/nc-encoder/presets"
    preset = preset_values(presets, codec, args.preset)
    domains = read_space(Path(args.space).read_text(), specs, preset)
    start = json.loads(Path(args.start).read_text()) if args.start else {}
    names = sorted(domains)
    u0 = [spsa.to_unit(start.get(n, preset.get(n, specs[n][0])), domains[n]["low"],
                       domains[n]["high"]) for n in names]
    k, k_source = front_k.resolve(codec.value, args.preset, args.k)
    gains = spsa.Gains.plan(args.iterations, args.c, args.first_step, args.first_gradient)
    state = {"name": args.name, "codec": codec.value, "preset": args.preset,
             "binary": str(binary), "binary_sha256": file_sha256(binary), "commit": commit or "",
             "preset_values": preset, "names": names, "domains": domains, "u0": u0, "u": u0,
             "k": k, "k_source": k_source, "seed": args.seed or args.name,
             "per_draw": args.per_draw, "iterations_planned": args.iterations,
             "gains": {"a": gains.a, "c": gains.c, "stability": gains.stability},
             "iterations": []}
    save_state(state)
    print(f"spsa {args.name}: {codec.value} {args.preset}, {len(names)} parameters "
          f"({', '.join(names)}), k {k:.4g} ({k_source}), {args.per_draw} samples a draw",
          flush=True)
    return state


def run_iterations(state: dict, until: int, workers: int) -> dict:
    measurer = Measurer(state, workers)
    gains = gains_of(state)
    while len(state["iterations"]) < until:
        k = len(state["iterations"])
        started = time.monotonic()
        samples = iteration_samples(state, k, measurer.clips)
        delta = direction(state, k)
        plus, minus = spsa.sides(state["u"], delta, gains.perturbation(k))
        rows = [measurer.scores(values_of(state, side), samples, k) for side in (plus, minus)]
        common = sorted(set(rows[0]) & set(rows[1]))
        if not common:
            raise ValueError(f"iteration {k}: no sample measured on both sides")
        both = [{key: r[key] for key in common} for r in rows]
        f_plus, f_minus = (mean_score(r) for r in both)
        u = spsa.update(state["u"], delta, f_plus, f_minus, k, gains)
        state["iterations"].append({
            "k": k, "u": state["u"], "delta": delta, "plus": plus, "minus": minus,
            "f_plus": f_plus, "f_minus": f_minus, "samples": common,
            "seconds": round(time.monotonic() - started)})
        state["u"] = u
        save_state(state)
        print(f"iteration {k}: f+ {f_plus:+.3f} f- {f_minus:+.3f} on {len(common)} samples; "
              f"u -> {', '.join(f'{x:+.3f}' for x in u)} "
              f"({time.monotonic() - started:.0f} s)", flush=True)
    return state


def candidate_values(state: dict, spec: str | None) -> tuple[list[float] | None, dict]:
    """The SPRT's candidate: the run's current point (default), its start (`start`), or a
    JSON file of {parameter: value} over the preset."""
    if spec in (None, "current"):
        return state["u"], values_of(state, state["u"])
    if spec == "start":
        return state["u0"], values_of(state, state["u0"])
    return None, {**state["preset_values"], **json.loads(Path(spec).read_text())}


def run_sprt(state: dict, args: argparse.Namespace) -> dict:
    u, values = candidate_values(state, args.candidate)
    test = state.setdefault("sprt", {"candidate": values, "delta": args.delta, "rounds": []})
    if test["candidate"] != values or test["delta"] != args.delta:
        raise SystemExit("an SPRT of another candidate or delta is stored: --name a new run "
                         "or remove the run's sprt entry")
    measurer = Measurer(state, args.workers)
    sprt = spsa.Sprt(args.delta, args.alpha, args.beta,
                     [v for r in test["rounds"] for v in r["scores"]])
    while sprt.decision() is None and len(test["rounds"]) < args.max_rounds:
        r = len(test["rounds"])
        samples = round_samples(state, r, measurer.clips, args.per_round)
        if not samples:
            break
        rows = measurer.scores(values, samples, SPRT_OFFSET + r)
        keys = sorted(rows)
        scores = [rows[key]["score"] for key in keys]
        sprt.add(scores)
        # each list in the samples' order (box_cpu_verdict restates each at its share)
        test["rounds"].append({"samples": keys, "scores": scores, "aligned": True,
                               "drawn": [asdict(s) for s in samples],
                               "j": [rows[key]["j"] for key in keys],
                               "cpu": [rows[key]["cpu"] for key in keys]})
        test["summary"] = sprt.summary()
        save_state(state)
        print(f"sprt round {r}: {len(scores)} samples; {describe(sprt.summary())}", flush=True)
    test["summary"] = sprt.summary()
    save_state(state)
    return state


def describe(summary: dict) -> str:
    mean = "n/a" if summary["mean"] is None else f"{summary['mean']:+.3f}"
    return (f"n {summary['n']}, mean score {mean} % (sd {summary['sd']:.2f}), "
            f"LLR {summary['llr']:+.2f} in [{summary['bounds'][0]:+.2f}, "
            f"{summary['bounds'][1]:+.2f}] -> {summary['decision'] or 'undecided'}")


def box_cpu_verdict(state: dict) -> dict:
    """The SPRT's candidate timed on rented cores on the SPRT's own encodes (every
    round's samples; nceval.decision_cpu), and its scores restated with that CPU, each
    sample at its own share's, in place of callgrind's."""
    from .tune_pool import family, settings_label
    test = state["sprt"]
    if not all(r.get("aligned") for r in test["rounds"]):
        raise SystemExit("rounds recorded before 2026-11-23 do not keep each sample's J "
                         "beside its share: run a new SPRT")
    codec, binary = Codec(state["codec"]), Path(state["binary"])
    anchor = family(codec, binary, state["preset_values"], f"nanocodec-{codec.value}")
    candidate = family(codec, binary, test["candidate"],
                       f"nanocodec-{codec.value}@{settings_label(test['candidate'])}")
    runner = realuse.Runner(load_paths(), EVALUATION_GATE, 1, over_budget=True)
    clips = pool_clips()
    jobs = [job for r, done in enumerate(test["rounds"])
            for job in sample_jobs(runner, candidate, clips,
                                   [Sample(**s) for s in done["drawn"]], SPRT_OFFSET + r)]
    entry = decision_cpu.plan_entry("sprt", codec.value, state["preset"], None, jobs, anchor,
                                    candidate, state["commit"], state["commit"])
    summary = decision_cpu.measure([entry], TUNING_DIR / f"{state['name']}-sprt-cpu",
                                   "tune-sprt")
    found = decision_cpu.load_charge(summary, "sprt")
    periods = [int(key.rpartition("/")[2]) for r in test["rounds"] for key in r["samples"]]
    js = [j for r in test["rounds"] for j in r["j"]]
    restated = [j + state["k"] * found.dcpu(key_shares.KeyShare(p).window, p)
                for j, p in zip(js, periods)]
    sprt = spsa.Sprt(test["delta"], samples=restated)
    test["box_cpu"] = {"cpu": decision_cpu.record(found), "summary": sprt.summary()}
    save_state(state)
    print(f"rented cores on the SPRT's own encodes: dCPU {decision_cpu.describe(found)}; "
          f"restated: {describe(sprt.summary())}", flush=True)
    return test["box_cpu"]


# --- commands -------------------------------------------------------------------------

def pin_binary(state: dict, given: str | None) -> None:
    """Measure with `given` (a box's copy of the run's build) when it is the same build."""
    if given and Path(given).is_file():
        state["binary"] = str(Path(given).resolve())
    if file_sha256(Path(state["binary"])) != state["binary_sha256"]:
        raise SystemExit(f"{state['binary']} is not the build the run began with: start a new run")


def spsa_command(args: argparse.Namespace) -> None:
    if not cloud.runs_here(args.local):
        return cloud_spsa(args)
    state = load_state(args.name) or create(args)
    pin_binary(state, args.binary)
    run_iterations(state, args.iterations, args.workers)
    if args.json:
        Path(args.json).write_text(json.dumps({"status": "done", "name": args.name,
                                               "iterations": len(state["iterations"])}) + "\n")


def sprt_command(args: argparse.Namespace) -> None:
    if not cloud.runs_here(args.local):
        return cloud_sprt(args)
    state = load_state(args.name)
    if state is None:
        raise SystemExit(f"no run {args.name}")
    pin_binary(state, args.binary)
    run_sprt(state, args)
    if args.json:
        Path(args.json).write_text(json.dumps({"status": "done", "name": args.name,
                                               "sprt": state["sprt"]["summary"]}) + "\n")


def cloud_spsa(args: argparse.Namespace) -> None:
    """Slices of SLICE_ITERATIONS iterations, each one warm-box job (the state up and back),
    until the iterations are done or the run has cost --max-usd."""
    from .tune_pool import offload
    if load_state(args.name) is None:
        create(args)  # here: the slices' draws are staged from the stored run
    state = load_state(args.name)
    args.binary = args.binary or state["binary"]
    spent = 0.0
    passed = replace_option([a for a in sys.argv[2:] if a != "--local"], "--binary",
                            args.binary)
    while True:
        state = load_state(args.name)
        done = len(state["iterations"]) if state else 0
        if done >= args.iterations or spent >= args.max_usd:
            break
        until = min(args.iterations, done + args.slice_iterations)
        argv = replace_option(passed, "--iterations", str(until))
        started = time.monotonic()
        code, result = offload("spsa", argv,
                               SLICE_MINUTES + MINUTES_PER_ITERATION * (until - done))
        cost = (result.get("cloud") or {}).get("estimated_usd") or 0.0
        spent += cost
        state = load_state(args.name)
        if state:
            state["binary"] = str(Path(args.binary).resolve())
            save_state(state)
        now = len(state["iterations"]) if state else 0
        minutes = (time.monotonic() - started) / 60
        print(f"spsa {args.name}: slice {now - done} iterations in {minutes:.1f} min, "
              f"${cost:.2f}; {now} of {args.iterations} done, ${spent:.2f} spent", flush=True)
        if code != 0 or now == done:
            raise SystemExit(f"spsa {args.name}: the slice failed (exit {code}) or made no "
                             f"progress (resume: same command)")
    print(f"spsa {args.name}: ${spent:.2f} this run; next: `nceval.tune sprt --name "
          f"{args.name}`", flush=True)


def cloud_sprt(args: argparse.Namespace) -> None:
    """The test's rounds as one warm-box job; then the candidate's CPU on rented cores."""
    from .tune_pool import offload
    state = load_state(args.name)
    if state is None:
        raise SystemExit(f"no run {args.name}")
    stored = state.get("sprt", {})
    finished = stored.get("summary", {}).get("decision") or \
        len(stored.get("rounds", [])) >= args.max_rounds
    if not finished:  # a finished test only takes its rented-core CPU (--box-cpu)
        passed = replace_option([a for a in sys.argv[2:] if a not in ("--local", "--box-cpu")],
                                "--binary", args.binary or state["binary"])
        code, result = offload("sprt", passed,
                               SLICE_MINUTES + MINUTES_PER_ITERATION * args.max_rounds)
        cost = (result.get("cloud") or {}).get("estimated_usd") or 0.0
        print(f"sprt {args.name}: ${cost:.2f}", flush=True)
        if code != 0:
            raise SystemExit(f"sprt {args.name}: the job failed (exit {code})")
        state = load_state(args.name)
        state["binary"] = str(Path(args.binary or state["binary"]).resolve())
        save_state(state)
    print(describe(state["sprt"]["summary"]), flush=True)
    if args.box_cpu:
        box_cpu_verdict(state)


def replace_option(argv: list[str], option: str, value: str) -> list[str]:
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == option:
            skip = True
            continue
        if a.startswith(option + "="):
            continue
        out.append(a)
    return out + [option, value]


def report_command(args: argparse.Namespace) -> None:
    state = load_state(args.name)
    if state is None:
        raise SystemExit(f"no run {args.name}")
    print(f"{state['name']}: {state['codec']} {state['preset']}, k {state['k']:.4g}")
    for it in state["iterations"]:
        print(f"  {it['k']:>3}  f+ {it['f_plus']:+.3f}  f- {it['f_minus']:+.3f}  "
              f"n {len(it['samples']):>2}  {it['seconds']:>4} s  u "
              + " ".join(f"{x:+.3f}" for x in it["u"]))
    print("  now: " + ", ".join(f"{n} = {v:.4g}" for n, v in
                                zip(state["names"], (values_of(state, state["u"])[n]
                                                     for n in state["names"]))))
    if "sprt" in state:
        print("  sprt: " + describe(state["sprt"]["summary"]))
        if "box_cpu" in state["sprt"]:
            print("  sprt with rented-core CPU: " + describe(state["sprt"]["box_cpu"]["summary"]))


def slice_samples(command_get, sub: str) -> list[Sample]:
    """The samples a cloud job of `sub` can measure (for aws_measure's staging): the next
    iterations' draws, or the SPRT's remaining rounds."""
    state = load_state(command_get("--name"))
    clips = pool_clips()
    if sub == "spsa":
        if state is None:
            raise SystemExit("an spsa run is created here first (`--local` for its first "
                             "iteration), so its draws can be staged")
        start, until = len(state["iterations"]), int(command_get("--iterations"))
        return [s for k in range(start, until) for s in iteration_samples(state, k, clips)]
    rounds = int(command_get("--max-rounds") or 1)
    per_round = int(command_get("--per-round") or 8)
    done = len(state.get("sprt", {}).get("rounds", []))
    chosen, used = [], set()
    for r in range(done, rounds):
        rng = random.Random(f"{state['seed']}:sprt:{r}")
        previous = {s for d in state.get("sprt", {}).get("rounds", []) for s in d["samples"]}
        batch = draw(samples_of(clips, TEST, state["seed"]), per_round, rng, previous | used)
        used |= {s.key for s in batch}
        chosen += batch
    return chosen


def staged_clips(command_get, sub: str) -> list[Clip]:
    """slice_samples' clips at their start frames (one entry per clip and start)."""
    clips = pool_clips()
    return list({(s.clip, s.start): replace(clips[s.clip], start_frame=s.start)
                 for s in slice_samples(command_get, sub)}.values())


def add_commands(sub) -> None:
    s = sub.add_parser("spsa", help="SPSA steering on fresh draws (module docs)")
    s.add_argument("--name", required=True)
    s.add_argument("--codec", choices=[c.value for c in Codec])
    s.add_argument("--preset")
    s.add_argument("--binary", help="nanocodec binary, or a commit (built and cached)")
    s.add_argument("--space", help="space file (`name [low high] [int|log]`), a few parameters")
    s.add_argument("--start", help="JSON {parameter: value}: the walk's start (default: preset)")
    s.add_argument("--iterations", type=int, default=12)
    s.add_argument("--per-draw", type=int, default=8, help="samples per iteration's draw")
    s.add_argument("--c", type=float, default=DEFAULT_C, help="perturbation (normalised units)")
    s.add_argument("--first-step", type=float, default=DEFAULT_FIRST_STEP)
    s.add_argument("--first-gradient", type=float, default=DEFAULT_FIRST_GRADIENT)
    s.add_argument("--k", type=float, help="CPU price (default: the rung's front slope)")
    s.add_argument("--seed", help="draw seed (default: the run's name)")
    s.add_argument("--workers", type=int, default=realuse.max_workers())
    s.add_argument("--slice-iterations", type=int, default=SLICE_ITERATIONS)
    s.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    s.add_argument("--rotation", help="unused (aws_measure passes the day's): draws follow --seed")
    s.add_argument("--local", action="store_true")
    s.add_argument("--json")
    s.set_defaults(func=spsa_command)
    t = sub.add_parser("sprt", help="sequential test of a point against the shipped rung")
    t.add_argument("--name", required=True)
    t.add_argument("--binary", help="the run's build at another path (a box's)")
    t.add_argument("--candidate", help="current (default), start, or a JSON {param: value}")
    t.add_argument("--delta", type=float, default=0.3, help="the gain H1 claims (score %%)")
    t.add_argument("--alpha", type=float, default=0.05)
    t.add_argument("--beta", type=float, default=0.05)
    t.add_argument("--per-round", type=int, default=8)
    t.add_argument("--max-rounds", type=int, default=4)
    t.add_argument("--workers", type=int, default=realuse.max_workers())
    t.add_argument("--box-cpu", action="store_true",
                   help="then measure the candidate's CPU on rented cores and restate")
    t.add_argument("--rotation", help="unused (aws_measure passes the day's): draws follow --seed")
    t.add_argument("--local", action="store_true")
    t.add_argument("--json")
    t.set_defaults(func=sprt_command)
    r = sub.add_parser("spsa-report", help="a run's iterations and test")
    r.add_argument("--name", required=True)
    r.set_defaults(func=report_command)
