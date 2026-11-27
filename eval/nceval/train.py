"""Stages 2 and 3 of the three-stage checks: the candidate queue and its parallel checks.

    python3 -m nceval.train add --codec av1 --preset fast --param a.b=1 [--param ...] \\
        --idea SLUG --mechanism subpel-refine --quick-json QUICK.json
    python3 -m nceval.train list [--codec av1] [--all]
    python3 -m nceval.train drop ID [--reason TEXT]
    python3 -m nceval.train run [--codec av1 --preset fast | --target av1:fast ...] \\
        [--local] [--plan]

A candidate is a preset-level parameter set whose code has already landed
default-off (identity 306/306), with its stage-1 numbers (`nceval.quick
--json`), its CPU cost and a mechanism tag. The queue is an append-only JSONL
of events under ~/nanocodec-data/results (NC_TRAIN_QUEUE overrides), written
under a file lock like the lever spool.

`run` (owner, 2026-11-21: no forced serialisation) checks every queued
candidate of each codec x preset against ONE base, the head build of the
landed-commit cache with the preset as it is:

  - Candidates with different mechanism tags are independent: each is its
    own unit, checked alone against the base. Two candidates that set the
    same parameter join one unit (they change the same mechanism).
  - Candidates sharing a tag form a group. Candidates that set any of the same
    parameters are alternatives, never bundled: each is checked alone against the
    base and at most one passes (the best net J, ties to the cheaper; the others
    stay queued). On rented boxes the group runs in ONE round: the whole bundle
    plus each leave-one-out bundle (groups of up to LOO_MAX_MEMBERS), so a
    member's marginal contribution in context is full - leave-one-out on the
    same clips; a member whose marginal interval is above 0 is ejected and
    the matching leave-one-out arm is the verified bundle. Locally (--local)
    the group runs serially: a failing bundle ejects the member whose stage-1
    profile best matches the failing guard (pulse, a class, else J) or,
    when that is ambiguous, keeps its better half (the rest stays queued),
    and runs again, up to --max-rounds.

Each arm is two standard screens on the day's draw, run side by side:

  final-check  `nceval.screen --clips adopt --levels final --final-frames 600`:
               the L3 families' long clips at every key share (nceval.key_shares:
               600 f with keys every 150 f, 900 f with keys every 300 f), each
               charged at the preset's k with the lone candidate's CPU at that
               share, not WORSE at either; the periodic-key pulse passes; the
               measurement of `nceval.levels final-check`, as a screen so that
               any preset, medium included, runs as --param pairs and
               tools/dev/aws_measure.py can take it.
  L3           `nceval.screen --clips adopt --levels final` at 150 frames,
               decided at the preset's k with a lone candidate's CPU (a bundle's CPU
               is not measured: judged on quality, measure it before landing).

An arm passes when L3 is BETTER and the final check is not WORSE, or L3 is NEUTRAL
(identical streams: the change touches only periodic keys) and the final check BETTER (a
periodic-key pulse makes it WORSE). By default every screen runs on its own
rented spot box (aws_measure.py: CPU scoring, cloud points kept in the cloud
cache), all at once; --local runs them here through the slot engine.

On a pass the unit's candidates are marked bundled-pass, the exact preset
diff and a commit-message block are printed; nothing lands here: the caller
lands with tools/dev/land.sh and runs preset_identity, static_recovery and
landing conformance once for what it lands. Every arm's screens record their
lever rows; each member's idea record is revised with those rows and the
verdict (nceval.levers_ideas.revise). The candidate preset of every arm is
written to a frozen directory under ~/nanocodec-data/results/train/bundles.
"""

from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .screen import RUNG_PARAMS_FLAG
from . import (checkrun, cloud, decision_cpu, interval, jreal, key_shares, levels, levers,
               levers_ideas, pool)
from .cbr import DEFAULT_INTRA_PERIOD
from .budget import MAX_MINUTES
from .cloud import RUN_CAP_MINUTES, pinned_tree
from .config import load_paths
from .encoders.nanocodec import MEDIUM, PRESETS_PATH, param_text, preset_path, read_preset
from .gatepolicy import EVALUATION_GATE, GatePolicy
from .realuse import DEFAULT_FRAMES, DEFAULT_WORKERS, Runner
from .train_guards import JUSTIFICATION_PREFIX, PLACEHOLDER, TRAIN_GUARD_LINE

QUEUE_ENV = "NC_TRAIN_QUEUE"
QUEUED, PASSED, EJECTED, DROPPED = "queued", "bundled-pass", "ejected", "dropped"
FINAL_CHECK, L3 = "final-check", "L3"
#: Each check's estimator. The final check is always the full ladder: only it computes the
#: worst-1 % BD and flicker guards (a single-point final check left them uncomputed and
#: H.264 veryfast 327c20052 landed with its worst frames worse on 27/36 clips, 2026-11-23).
#: L3 (150 f, one key, 108 clips) stays single-point: the ranker, not the guard.
ESTIMATOR = {FINAL_CHECK: levels.LADDER, L3: levels.SINGLE}
#: The verdict of a passing arm whose screens tripped a guard: it lands only with a
#: `Guard-justification:` line (tools/dev/land_train_guards.py).
PASS, PASS_GUARD = "PASS", "PASS-GUARD"
#: The largest group whose leave-one-out arms run in the same round (owner, 2026-11-21).
LOO_MAX_MEMBERS = 6
#: Local rounds of eject/bisect per group.
DEFAULT_ROUNDS = 3
#: A stage-1 profile "matches" a failure only this far ahead of the next member's
#: (class J in %, pulse in dB, J in %); closer than that, the group is bisected.
BLAME_MARGIN = {"class": 1.0, "pulse": 0.1, "j": 0.3}
#: Wall-time limits (minutes): a screen here gets the 30-min budget plus a margin for
#: its last points; a box run its aws_measure lifetime plus the download.
LOCAL_LIMIT_MINUTES = MAX_MINUTES + 10
#: A box run's limit: its planned run (box_minutes) times BOX_LIMIT_MARGIN, plus the most
#: a launch may wait for quota (aws_session.SessionSpec.launch_wait_minutes) and the setup
#: (CLOUD_FIXED_MINUTES). A driver stopped at it, or killed, leaves a resumable round
#: (finished screens and the CPU summary are reused by a rerun on the same base).
BOX_LIMIT_MARGIN = 2.0
QUOTA_WAIT_MINUTES = 90
#: Tries per box of a round (a failed box is run once more, on another box).
BOX_ATTEMPTS = 2
#: A box run's fixed part (PROCEDURES "On a rented box", measured 2026-11-21 for L1):
#: launch, upload and setup.
CLOUD_FIXED_MINUTES = 4.0
#: Box packing: a box runs BOX_WORKERS points at once (48 vCPU, 3 a point: the smallest
#: Graviton box aws_measure's auto sizing allows a train batch, which it no longer moves
#: to a smaller warm box) at about POINT_SECONDS a 150-frame point (PROCEDURES "On a rented
#: box": an L1's 117 new points took 0.6 min on 64 workers); a 600-frame point counts
#: LONG_POINT_FACTOR. Screens are packed while a box's estimate stays under PACK_MINUTES.
#: 2026-11-24: packed for 96 vCPU, a final check and an L3 on one Graviton box ran 30-35
#: min (estimate 17) and one hit the 35-min run cap.
BOX_WORKERS = 16
POINT_SECONDS = 18.0
LONG_POINT_FACTOR = 4
PACK_MINUTES = 20.0
AWS_MEASURE = checkrun.REPO / "tools" / "dev" / "aws_measure.py"
#: Writing one screen's CPU plan (a draw and its job list, no encode; measure_cpu).
PLAN_LIMIT_MINUTES = 5
EVAL_DIR = checkrun.REPO / "eval"
CLASS_GUARD = re.compile(r"^class (\S+) quality")


# --- the queue ---------------------------------------------------------------------------

@dataclass
class Candidate:
    id: str
    codec: str
    preset: str
    params: dict[str, float]
    idea: str
    mechanism: str
    #: The stage-1 report's parts, with L1's CPU change measured on L1's own encodes
    #: ("dcpu", None when not measured; nceval.decision_cpu). Entries queued before
    #: 2026-11-23 carried a CPU measured on other encodes, which is no longer read.
    quick: dict | None
    added: str
    status: str = QUEUED
    note: str = ""

    @property
    def dcpu(self) -> float | None:
        """Stage 1's CPU change on L1's own encodes (ranks; stage 3 measures its own)."""
        return (self.quick or {}).get("dcpu")

    def stage1_j(self, k: float) -> float:
        """Stage-1 J: L1's CPU-free quality plus k * dCPU (inf when unmeasured)."""
        quality = ((self.quick or {}).get("l1") or {}).get("quality")
        return math.inf if quality is None else quality + k * (self.dcpu or 0.0)

    def class_quality(self, name: str) -> float | None:
        values = [((self.quick or {}).get(part) or {}).get("classes", {}).get(name)
                  for part in ("l1", "probe")]
        values = [v for v in values if v is not None]
        return max(values) if values else None

    def pulse(self) -> float | None:
        return ((self.quick or {}).get("probe") or {}).get("pulse")


def queue_path() -> Path:
    override = os.environ.get(QUEUE_ENV)
    return Path(override) if override else load_paths().data_root / "results" / "train-queue.jsonl"


def append_events(events: list[dict]) -> None:
    path = queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        fcntl.flock(out, fcntl.LOCK_EX)
        for event in events:
            out.write(json.dumps({"at": now(), **event}, default=str) + "\n")


def load_queue() -> list[Candidate]:
    """Every candidate with its latest status, in the order added."""
    path = queue_path()
    candidates: dict[str, Candidate] = {}
    if not path.is_file():
        return []
    with path.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        lines = stream.read().splitlines()
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if event["event"] == "add":
            candidates[event["id"]] = Candidate(**{k: event[k] for k in Candidate.__dataclass_fields__
                                                   if k in event})
        elif event["id"] in candidates:
            candidates[event["id"]].status = event["status"]
            candidates[event["id"]].note = event.get("note", "")
    return list(candidates.values())


def set_status(candidates: list[Candidate], status: str, note: str) -> None:
    append_events([{"event": "status", "id": c.id, "status": status, "note": note}
                   for c in candidates])
    for candidate in candidates:
        candidate.status, candidate.note = status, note


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def add(args: argparse.Namespace) -> Candidate:
    params = levers.pairs_to_dict(args.param)
    quick = json.loads(Path(args.quick_json).read_text())
    if quick.get("status") != "done":
        raise SystemExit(f"{args.quick_json}: not a finished nceval.quick report")
    stated = (quick["codec"], quick["preset"], levers.pairs_to_dict(quick["params"]))
    if stated != (args.codec, args.preset, params):
        raise SystemExit(f"{args.quick_json} screened {stated}, not this candidate")
    if quick["verdict"] != "CONTINUE" and not args.force:
        raise SystemExit(f"stage 1 said {quick['verdict']} ({'; '.join(quick['reasons'])}); "
                         f"--force queues it anyway")
    if not any(i["idea"] == args.idea for i in levers_ideas.load_ideas()):
        raise SystemExit(f"no idea {args.idea!r} in the ledger: `python3 -m nceval.levers "
                         f"ideas add --name ...` first")
    identity = json.dumps([args.codec, args.preset, sorted(params.items()), args.idea, now()])
    candidate = Candidate(hashlib.sha256(identity.encode()).hexdigest()[:10], args.codec,
                          args.preset, params, args.idea, args.mechanism,
                          {part: {k: v for k, v in quick[part].items() if k != "per_clip"}
                           for part in ("l1", "probe")}
                          | {"verdict": quick["verdict"], "rotation": quick["rotation"],
                             "dcpu": quick.get("dcpu") if quick.get("cpu") else None}, now())
    append_events([{"event": "add", **asdict(candidate)}])
    print(f"queued {candidate.id}: {args.codec}@{args.preset} "
          f"{' '.join(checkrun.pairs(params))} ({args.mechanism}, idea {args.idea})")
    return candidate


def describe(candidate: Candidate, k: float) -> str:
    quick = candidate.quick or {}
    l1, probe = quick.get("l1") or {}, quick.get("probe") or {}
    pct = lambda v, unit="%": "n/a" if v is None else f"{v:+.2f}{unit}"
    worst = max((probe.get("classes") or {}).items(), key=lambda kv: kv[1], default=None)
    return (f"{candidate.id} {candidate.status:12s} {candidate.mechanism:14s} "
            f"{' '.join(checkrun.pairs(candidate.params))}  idea {candidate.idea}; "
            f"stage-1 dCPU {pct(candidate.dcpu)} (L1's own encodes); "
            f"L1 J {pct(l1.get('mean'))} [{pct(l1.get('lo'))}, {pct(l1.get('hi'))}] "
            f"n={l1.get('n')}; probe J {pct(probe.get('mean'))} pulse "
            f"{pct(probe.get('pulse'), ' dB')}"
            + (f", worst class {worst[0]} {worst[1]:+.2f}%" if worst else "")
            + (f"  ({candidate.note})" if candidate.note else ""))


def listing(args: argparse.Namespace) -> None:
    shown = [c for c in load_queue() if (args.all or c.status == QUEUED)
             and (not args.codec or c.codec == args.codec)]
    targets = sorted({(c.codec, c.preset) for c in shown})
    for codec, preset in targets:
        k, source = checkrun.preset_k(codec, preset)
        print(f"== {codec}@{preset} (k {k:.4g}: {source})")
        for candidate in sorted((c for c in shown if (c.codec, c.preset) == (codec, preset)),
                                key=lambda c: (c.mechanism, c.stage1_j(k))):
            print("  " + describe(candidate, k))
    print(f"{len(shown)} candidate(s) in {queue_path()}")


def drop(args: argparse.Namespace) -> None:
    found = [c for c in load_queue() if c.id == args.id]
    if not found:
        raise SystemExit(f"no candidate {args.id}")
    set_status(found, DROPPED, args.reason or "dropped by hand")
    print(f"dropped {args.id}")


# --- units, groups and arms --------------------------------------------------------------

def units(candidates: list[Candidate]) -> list[list[Candidate]]:
    """Independent units: one per mechanism tag, merged where two set one parameter."""
    groups: list[list[Candidate]] = []
    for candidate in candidates:
        joined = [g for g in groups if any(m.mechanism == candidate.mechanism or
                                           set(m.params) & set(candidate.params) for m in g)]
        merged = [candidate] + [m for g in joined for m in g]
        groups = [g for g in groups if g not in joined] + [merged]
    return [sorted(g, key=lambda c: c.added) for g in groups]


def compose(group: list[Candidate], k: float) -> tuple[list[Candidate], list[Candidate]]:
    """(members, alternatives): the bundle, best stage-1 J first, of candidates that set
    disjoint parameters; a candidate setting any parameter a member sets (another value,
    or the same value with more beside it) is an alternative to it, never bundled with
    it (2026-11-21: `gate` and `gate + frame period 2` were bundled as the second's
    parameters with both CPU costs summed, read WORSE, and both were ejected, though
    each passed alone). Each alternative is checked alone against the base."""
    members, alternatives, taken = [], [], set()
    for candidate in sorted(group, key=lambda c: c.stage1_j(k)):
        if taken & set(candidate.params):
            alternatives.append(candidate)
            continue
        members.append(candidate)
        taken |= set(candidate.params)
    return members, alternatives


@dataclass
class Arm:
    """One candidate preset to check: `members`' parameters on the base preset."""
    codec: str
    preset: str
    members: list[Candidate]
    role: str  # "full", "single" or "loo:<left-out id>"
    directory: Path | None = None
    params: dict[str, float] = field(default_factory=dict)
    diff: str = ""
    #: Per check, the CPU change measured on that screen's own encodes (measure_cpu;
    #: decision_cpu.Charge), bundles included.
    cpu: dict[str, decision_cpu.Charge] = field(default_factory=dict)

    @property
    def change(self) -> dict[str, float]:
        return {n: v for c in self.members for n, v in c.params.items()}

    @property
    def dcpu(self) -> float | None:
        """The arm's CPU change on L3's own encodes (150 frames, one key)."""
        charge = self.cpu.get(L3)
        return None if charge is None else charge.dcpu(DEFAULT_FRAMES, DEFAULT_INTRA_PERIOD)

    @property
    def key(self) -> str:
        ids = ",".join(sorted(c.id for c in self.members))
        return hashlib.sha256(f"{self.codec}@{self.preset}:{ids}".encode()).hexdigest()[:10]

    def label(self) -> str:
        return f"{self.codec}@{self.preset} {self.role} [{', '.join(c.id for c in self.members)}]"


def arms_of(members: list[Candidate], cloud: bool) -> list[Arm]:
    """The arms of one round: the unit alone, or a group's bundle plus (on boxes, up to
    LOO_MAX_MEMBERS members) each leave-one-out bundle."""
    codec, preset = members[0].codec, members[0].preset
    if len(members) == 1:
        return [Arm(codec, preset, members, "single")]
    arms = [Arm(codec, preset, members, "full")]
    if cloud and len(members) <= LOO_MAX_MEMBERS:
        arms += [Arm(codec, preset, [m for m in members if m is not out], f"loo:{out.id}")
                 for out in members]
    return arms


# --- freezing an arm's preset ------------------------------------------------------------

def write_preset(text: str, values: dict[str, float], note: str) -> str:
    """`text` (a preset file) with `values` set: lines naming them rewritten in place
    (every occurrence: a later line wins), the rest appended under `note`."""
    seen, out = set(), []
    for line in text.splitlines():
        code, _, comment = line.partition("#")
        name = code.split("=")[0].strip() if "=" in code else None
        if name in values:
            seen.add(name)
            line = f"{name} = {param_text(values[name])}" + (f"  #{comment}" if comment
                                                                     else "")
        out.append(line)
    missing = {n: v for n, v in values.items() if n not in seen}
    if missing:
        out += ["", f"# {note}"] + [f"{n} = {param_text(v)}" for n, v in sorted(missing.items())]
    return "\n".join(out) + "\n"


def freeze(arm: Arm, base_presets: Path, root: Path) -> None:
    """The arm's candidate preset in its own directory (outside any worktree): a copy
    of the base's presets/ with the change written in (medium: `candidate.params`,
    the engine-default change), its diff, and the parameter set the screens run."""
    arm.directory = root / f"{arm.codec}-{arm.preset}-{arm.key}"
    if arm.directory.exists():
        shutil.rmtree(arm.directory / "presets", ignore_errors=True)
    arm.directory.mkdir(parents=True, exist_ok=True)
    base = checkrun.preset_params(base_presets, arm.codec, arm.preset)
    arm.params = {**base, **arm.change}
    note = f"nceval.train {arm.role}: " + ", ".join(f"{c.idea} ({c.id})" for c in arm.members)
    if arm.preset == MEDIUM:
        arm.diff = "".join(f"+ {arm.codec} DEFAULT_PARAMS: {n} = {param_text(v)}\n"
                           for n, v in sorted(arm.change.items()))
    else:
        shutil.copytree(base_presets, arm.directory / "presets")
        target = preset_path(arm.directory / "presets", arm.codec, arm.preset)
        old = target.read_text()
        target.write_text(write_preset(old, arm.change, note))
        written = read_preset(target)
        if written != arm.params:
            raise SystemExit(f"{target}: wrote {written}, expected {arm.params}")
        relative = f"{PRESETS_PATH}/{arm.codec}/{arm.preset}.toml"
        arm.diff = "".join(difflib.unified_diff(old.splitlines(True),
                                                target.read_text().splitlines(True),
                                                f"a/{relative}", f"b/{relative}"))
    (arm.directory / "candidate.params").write_text(
        "".join(f"{p}\n" for p in checkrun.pairs(arm.params)))
    (arm.directory / "arm.json").write_text(json.dumps(
        {"codec": arm.codec, "preset": arm.preset, "role": arm.role,
         "members": [c.id for c in arm.members], "change": arm.change,
         "params": arm.params}, indent=1) + "\n")


# --- measuring ---------------------------------------------------------------------------

@dataclass
class Base:
    binary: Path
    commit: str
    presets: Path
    rotation: str
    #: An explicit `train run --k`, in place of each rung's front slope (warned loudly).
    k: float | None = None
    #: The commit of this tree every box of the run stages (nceval.cloud.pinned_tree,
    #: taken when the run starts); None for a run here.
    tree: str | None = None


def screen_command(arm: Arm, base: Base, check: str, json_out: Path,
                   local: bool = False) -> list[str]:
    """The standard screen that measures `check` (FINAL_CHECK or L3) of `arm`; `local`
    keeps it on this machine (nceval.cloud would otherwise offload it)."""
    command = ["python3", "-m", "nceval.screen", "--codec", arm.codec, "--binary",
               str(base.binary), "--clips", "adopt", "--levels", pool.FINAL,
               "--final-estimator", ESTIMATOR[check], "--rotation", base.rotation,
               "--json", str(json_out)] + (["--local"] if local else [])
    base_params = checkrun.preset_params(base.presets, arm.codec, arm.preset)
    command += [x for p in checkrun.pairs(arm.params) for x in ("--param", p)]
    command += [x for p in checkrun.pairs(base_params) for x in ("--anchor-param", p)]
    if check == FINAL_CHECK:  # at every key share (nceval.key_shares), CPU at each share
        command += ["--final-frames", str(levels.LONG_FRAMES)]
    # the rung's values are in the --param / --anchor-param lists already
    command += [RUNG_PARAMS_FLAG]
    command += ["--preset", arm.preset] if base.k is None else ["--k", f"{base.k:g}"]
    if check in arm.cpu:
        command += ["--cpu-charge", arm.cpu[check].text()]
    return command


def measure_cpu(arms: list[Arm], base: Base) -> None:
    """Every arm's CPU on each of its screens' own encodes (nceval.decision_cpu): each
    screen writes its plan (`--cpu-plan`, the screen's own job list), aws_cpu_ab times
    them all at once, and each screen is then given its verified charge."""
    directory = cpu_directory(arms, base)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for arm in arms:
        for check in (FINAL_CHECK, L3):
            plan = directory / f"{arm.key}-{check}.json"
            run_logged(screen_command(arm, base, check, directory / "unused.json", local=True)
                       + ["--cpu-plan", str(plan), "--cpu-decision", cpu_decision(arm, check)],
                       directory / f"{arm.key}-{check}-plan.log", PLAN_LIMIT_MINUTES)
            entries += json.loads(plan.read_text())["decisions"]
    print(f"  CPU of {len(arms)} arm(s) on their screens' own encodes (aws_cpu_ab, "
          f"{directory})", flush=True)
    summary = directory / CPU_SUMMARY
    if reusable_cpu(summary, arms):
        print(f"  CPU: reusing {summary} (a run of this round that stopped after its CPU "
              f"was measured)", flush=True)
    else:
        summary = decision_cpu.measure(entries, directory, "train-cpu")
    for arm in arms:
        arm.cpu = {check: decision_cpu.load_charge(summary, cpu_decision(arm, check))
                   for check in (FINAL_CHECK, L3)}
        print(f"  {arm.label()}: " + "; ".join(
            f"{c} {decision_cpu.describe(v)}" for c, v in arm.cpu.items()), flush=True)


#: The CPU summary aws_cpu_ab writes in a round's CPU directory (decision_cpu.measure).
CPU_SUMMARY = "cpu-summary.json"


def cpu_directory(arms: list[Arm], base: Base) -> Path:
    """A round's CPU directory, named by what decides its encodes (the base build, the
    draw and the arms), so a rerun of the same round after the driver died finds it."""
    arms_key = hashlib.sha256(",".join(sorted(a.key for a in arms)).encode()).hexdigest()[:10]
    return arms[0].directory.parent / f"cpu-{base.rotation}-{base.commit[:10]}-{arms_key}"


def reusable_cpu(summary: Path, arms: list[Arm]) -> bool:
    """Whether `summary` already charges every screen of `arms` (each charge is checked
    against its screen's own digest again when the screen runs, `--cpu-charge`)."""
    if not summary.exists():
        return False
    try:
        for arm in arms:
            for check in (FINAL_CHECK, L3):
                decision_cpu.load_charge(summary, cpu_decision(arm, check))
    except (decision_cpu.CpuMismatch, KeyError, ValueError):
        return False
    return True


def cpu_decision(arm: Arm, check: str) -> str:
    """The decision name of one arm's screen in the round's CPU plan."""
    return f"{arm.key}-{check}"


def run_logged(command: list[str], log: Path, limit_minutes: float) -> None:
    """One command to completion, loudly on failure or when it outlives its limit."""
    with log.open("w") as out:
        process = subprocess.Popen(command, cwd=EVAL_DIR, stdout=out, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=limit_minutes * 60)
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError(f"STALLED: no result after {limit_minutes} min, killed: {log}")
    if code != 0:
        tail = "\n".join(log.read_text().splitlines()[-15:])
        raise RuntimeError(f"FAILED (exit {code}): {' '.join(command[:6])}... log {log}:\n{tail}")


Measure = Callable[[list[Arm], Base], dict[str, dict]]


@dataclass
class Box:
    """Screens packed onto one rented box (one aws_measure --batch session)."""
    screens: list[tuple[Arm, str]] = field(default_factory=list)
    kinds: set[tuple[str, str, str]] = field(default_factory=set)  # anchors measured
    points: int = 0


def pack(arms: list[Arm], sizes: dict[str, tuple[int, int]]) -> list[Box]:
    """The round's screens on as few boxes as keep each box's estimated run under
    PACK_MINUTES. `sizes[check]` is (anchor points, candidate points) of one screen; the
    anchor of a codec x preset and check is measured once per box, which is why the
    screens of one target go together."""
    boxes: list[Box] = []
    for arm in sorted(arms, key=lambda a: (a.codec, a.preset)):
        box = boxes[-1] if boxes else None
        added = lambda b: sum(sizes[c][1] + (0 if (arm.codec, arm.preset, c) in b.kinds
                                             else sizes[c][0]) for c in (FINAL_CHECK, L3))
        if box is None or box_minutes(box.points + added(box)) > PACK_MINUTES:
            box = Box()
            boxes.append(box)
        box.points += added(box)
        for check in (FINAL_CHECK, L3):
            box.kinds.add((arm.codec, arm.preset, check))
            box.screens.append((arm, check))
    return boxes


def box_limit_minutes(box: Box) -> float:
    """How long the driver waits for one box run (BOX_LIMIT_MARGIN)."""
    return QUOTA_WAIT_MINUTES + CLOUD_FIXED_MINUTES + BOX_LIMIT_MARGIN * box_minutes(box.points)


def box_minutes(points: int) -> float:
    """A box run's estimate: its points in waves of BOX_WORKERS at POINT_SECONDS each."""
    return points / BOX_WORKERS * POINT_SECONDS / 60


def screen_sizes(base: Base) -> dict[str, tuple[int, int]]:
    """(anchor points, candidate points) of a final-check and an L3 screen on the draw,
    a long point counted LONG_POINT_FACTOR times per LONG_FRAMES of its key share's window."""
    from .cbr import clip_targets
    from .realuse import LADDER_KBPS
    final = level_clips(base)
    ladder = lambda clips: sum(len(clip_targets(c, LADDER_KBPS)) for c in clips)
    weight = lambda share: LONG_POINT_FACTOR * share.window / levels.LONG_FRAMES
    shares = share_clips(final)
    return {FINAL_CHECK: (round(sum(weight(s) * ladder(c) for s, c in shares)),
                          round(sum(weight(s) * len(c) for s, c in shares))),
            L3: (ladder(final), len(final))}


def share_clips(final: list) -> list[tuple[key_shares.KeyShare, list]]:
    """Each key share of the final check with the L3 clips long enough for its window."""
    return [(s, levels.long_clips(final, s.window)) for s in key_shares.shares()]


def level_clips(base: Base) -> list:
    """The draw's L3 clips (the final check measures the long ones, share_clips)."""
    paths = load_paths()
    draw = pool.draw(pool.load_pool(paths, levels.usable), base.rotation, DEFAULT_FRAMES)
    return draw.levels[pool.FINAL]


def measure_arms(arms: list[Arm], base: Base, cloud: bool, budget: float,
                 workers: int) -> dict[str, dict]:
    """{arm key: {FINAL_CHECK: result, L3: result, "cost": USD}}: every arm's CPU on its
    screens' own encodes first (measure_cpu), then on boxes the round's screens packed
    (`pack`) and every box at once; here, two screens at a time."""
    measure_cpu(arms, base)
    if not cloud:
        jobs = [(arm, check) for arm in arms for check in (FINAL_CHECK, L3)]

        def one(job):
            arm, check = job
            out = arm.directory / f"{check}.json"
            run_logged(screen_command(arm, base, check, out, local=True),
                       arm.directory / f"{check}.log",
                       LOCAL_LIMIT_MINUTES)
            return arm, check, json.loads(out.read_text()), 0.0
        with ThreadPoolExecutor(2) as threads:
            done = list(threads.map(one, jobs))
    else:
        boxes = pack(arms, screen_sizes(base))
        print(f"  {len(boxes)} box(es): " + "; ".join(
            f"{len(b.screens)} screens ~{box_minutes(b.points):.0f} min" for b in boxes),
              flush=True)
        for index, box in enumerate(boxes):  # a lone screen too big for any box
            if box_minutes(box.points) > RUN_CAP_MINUTES:
                print(f"WARNING: box {index} is estimated at ~{box_minutes(box.points):.0f} "
                      f"min, over the {RUN_CAP_MINUTES:g}-min run cap: it will keep "
                      f"only the points it finishes", file=sys.stderr, flush=True)

        def run_box(numbered, attempt: int = 1):
            """One box's screens; a box that fails (no screen finished: an rsync to it,
            the box lost) is run again once, as a new batch (aws_measure takes another
            warm box), and then only its own units go unjudged."""
            index, box = numbered
            directory = box.screens[0][0].directory
            suffix = "" if attempt == 1 else f"-retry{attempt - 1}"
            batch = directory / f"box-{index}{suffix}.json"
            todo = [(arm, check) for arm, check in box.screens
                    if not finished(arm.directory / f"{check}.json", base)]
            batch.write_text(json.dumps([screen_command(arm, base, check,
                                                        arm.directory / f"{check}.json")
                                         for arm, check in todo], indent=1))
            log = directory / f"box-{index}{suffix}.log"  # every attempt keeps its log
            try:
                run_logged(["python3", str(AWS_MEASURE), "--budget", f"{budget:g}", "--batch",
                            str(batch), *(["--tree-commit", base.tree] if base.tree else [])],
                           log, box_limit_minutes(box))
            except RuntimeError as error:  # the screens that finished still count
                print(f"  box {index}: {str(error).splitlines()[0]}", flush=True)
            results = [finished(arm.directory / f"{check}.json", base)
                       for arm, check in box.screens]
            if not all(results) and attempt < BOX_ATTEMPTS:
                print(f"  box {index}: {results.count(None)} screen(s) did not finish (log "
                      f"{log}): once more, on another box", flush=True)
                return run_box(numbered, attempt + 1)
            if not any(results):
                print(f"  box {index}: FAILED, no screen finished (log {log}): its units "
                      f"stay queued", flush=True)
                return []
            cloud = next(r for r in results if r).get("cloud") or {}
            print(f"  box {index}: {describe_cloud(cloud)}", flush=True)
            share = (cloud.get("estimated_usd") or 0.0) / len(box.screens)
            return [(arm, check, result, share)
                    for (arm, check), result in zip(box.screens, results) if result]
        with ThreadPoolExecutor(len(boxes)) as threads:
            done = [s for screens in threads.map(run_box, enumerate(boxes)) for s in screens]
    measured: dict[str, dict] = {}
    for arm, check, result, cost in done:
        entry = measured.setdefault(arm.key, {"cost": 0.0})
        entry[check] = result
        entry["cost"] += cost
    return measured


def finished(path: Path, base: Base) -> dict | None:
    """A screen's result when it finished (written, status done) on `base`'s build, else
    None: a rerun after the driver died reuses what its boxes brought back, and a round
    on another base measures again."""
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    done = result.get("status", "done") == "done" and "levels" in result
    return result if done and Path(result.get("binary", "")) == base.binary else None


def describe_cloud(cloud: dict) -> str:
    minutes = cloud.get("stage_minutes") or {}
    return (f"{cloud.get('instance_type')}, {sum(minutes.values()):.1f} min "
            f"({', '.join(f'{k} {v}' for k, v in minutes.items())}), CPUs busy "
            f"{cloud.get('busy_pct')}%, ${cloud.get('estimated_usd')}")


def plan_minutes(arms: list[Arm], base: Base, cloud: bool) -> str:
    """The round's expected wall time: on boxes the packed boxes' estimates (all at once,
    plus CLOUD_FIXED_MINUTES and any wait for quota), here the uncached points priced on
    the slot engine, two screens at a time."""
    if cloud:
        boxes = pack(arms, screen_sizes(base))
        lines = [f"  box {i}: " + ", ".join(f"{arm.role} {check}" for arm, check in b.screens)
                 + f" (~{b.points} point-equivalents, ~{box_minutes(b.points):.0f} min run)"
                 for i, b in enumerate(boxes)]
        worst = CLOUD_FIXED_MINUTES + max(box_minutes(b.points) for b in boxes)
        return "\n".join(lines + [f"  {len(boxes)} rented box(es) at once: ~{worst:.0f} min "
                                  f"wall expected, plus any wait for spot quota"])
    paths = load_paths()
    runner = Runner(paths, GatePolicy(EVALUATION_GATE.value), DEFAULT_WORKERS, False)
    final = level_clips(base)
    lines, total = [], 0.0
    for arm in arms:
        anchor, test = checkrun.families(arm.codec, base.binary,
                                         checkrun.preset_params(base.presets, arm.codec,
                                                                arm.preset), arm.params)
        priced = [checkrun.price(runner, anchor, test, clips, s.window, key_frames=s.period)
                  for s, clips in share_clips(final)]
        cost_fc, points_fc = sum(c for c, _ in priced), sum(n for _, n in priced)
        cost_l3, points_l3 = checkrun.price(runner, anchor, test, final, DEFAULT_FRAMES,
                                            test_ladder=checkrun.single_rung(final))
        fc, l3 = checkrun.minutes(cost_fc), checkrun.minutes(cost_l3)
        total += max(fc, l3)
        lines.append(f"  {arm.label()}: final-check at key shares "
                     f"{', '.join(s.label for s in key_shares.shares())} "
                     f"({points_fc} new points) ~{fc:.0f} min; L3 {len(final)} clips "
                     f"({points_l3} new) ~{l3:.0f} min")
    return "\n".join(lines + [f"  here, two screens at a time: ~{total:.0f} min wall expected"])


# --- judging -----------------------------------------------------------------------------

def level_of(result: dict) -> dict:
    return result["levels"][pool.FINAL]


def clip_js(result: dict, cpu: decision_cpu.Charge | None, k: float
            ) -> dict[str, tuple[str, float]]:
    """Per clip (class, J + k dCPU) of a screen's last stage, at that stage's own CPU."""
    stage = result["stages"][-1]
    rows = stage["per_clip"]
    dcpu = None if cpu is None else cpu.dcpu(
        stage["frames"], stage.get("key_period") or DEFAULT_INTRA_PERIOD)
    out = {}
    for name, row in rows.items():
        j = interval.clip_j(row, dcpu, k)
        if j is not None:
            out[name] = (row["class"], j)
    return out


def passes(entry: dict) -> bool:
    """L3 BETTER and the final check not WORSE; or, where L3's streams are identical
    (NEUTRAL: the change touches only periodic keys, which the 150-frame one-key L3
    does not have), the final check BETTER."""
    l3, final = level_of(entry[L3])["decision"], level_of(entry[FINAL_CHECK])["decision"]
    if l3 == interval.NEUTRAL:
        return final == interval.BETTER
    return l3 == interval.BETTER and final != interval.WORSE


def tripped_guards(entry: dict) -> list[str]:
    """Every guard either check tripped, labelled with the check."""
    return [f"{check}: {g}" for check in (L3, FINAL_CHECK) for g in level_of(entry[check])["guards"]]


def pass_verdict(entry: dict) -> str:
    """PASS, or PASS-GUARD when a guard tripped (guards are judgement calls, CLAUDE.md:
    the lander looks at what it flags and justifies it in the commit message)."""
    return PASS_GUARD if tripped_guards(entry) else PASS


def failing(entry: dict) -> list[str]:
    """What failed, most specific first: "pulse", "class:<name>", else "j"."""
    guards = [g for check in (FINAL_CHECK, L3) for g in level_of(entry[check])["guards"]]
    found = ["pulse"] if any(g.startswith(jreal.PULSE_TERMS) for g in guards) else []
    found += [f"class:{m.group(1)}" for g in guards if (m := CLASS_GUARD.match(g))]
    return found + ["j"]


def score(candidate: Candidate, dimension: str, k: float) -> float:
    kind, _, name = dimension.partition(":")
    value = (candidate.pulse() if kind == "pulse" else candidate.class_quality(name)
             if kind == "class" else candidate.stage1_j(k))
    return -math.inf if value is None else value


def blame(members: list[Candidate], dimensions: list[str], k: float) -> tuple[Candidate | None, str]:
    """The member whose stage-1 profile best matches the first failing dimension it
    tells apart, or None (ambiguous: bisect)."""
    if len(members) == 1:
        return members[0], "the only member"
    for dimension in dimensions:
        ranked = sorted(members, key=lambda c: score(c, dimension, k), reverse=True)
        top, second = score(ranked[0], dimension, k), score(ranked[1], dimension, k)
        margin = BLAME_MARGIN[dimension.partition(":")[0]]
        if top > 0 and top - second >= margin:
            return ranked[0], f"worst stage-1 {dimension} ({top:+.2f} vs next {second:+.2f})"
    return None, f"no member stands out on {', '.join(dimensions)}"


def marginal(full: dict, loo: dict, check: str, k: float, cpu_full, cpu_loo) -> dict:
    """The left-out member's contribution in context: per-clip J (each bundle at its
    own CPU) of the full bundle minus the leave-one-out bundle's, on the clips both
    scored, with its interval."""
    a, b = clip_js(full[check], cpu_full, k), clip_js(loo[check], cpu_loo, k)
    rows = {n: {"class": a[n][0], "quality": a[n][1] - b[n][1],
                "quality_low": a[n][1] - b[n][1]} for n in sorted(set(a) & set(b))}
    return interval.interval(rows, None, 0.0)


def summary_line(entry: dict) -> str:
    parts = []
    for check in (L3, FINAL_CHECK):
        level = level_of(entry[check])
        parts.append(f"{check} {interval.describe(level)} -> {level['decision'].upper()}"
                     + (f" (guards: {'; '.join(level['guards'])})" if level["guards"] else ""))
    return "; ".join(parts)


# --- recording ---------------------------------------------------------------------------

def record(arm: Arm, entry: dict, verdict: str, outcome: str | None) -> None:
    """The arm's lever rows (as its screens recorded them) on each member's idea."""
    rows = [r for check in (FINAL_CHECK, L3)
            for r in levers.screen_rows({"status": "done", **entry[check]},
                                        str(arm.directory / f"{check}.json"))]
    levers.record(rows, f"nceval.train {arm.label()}")
    for member in arm.members:
        levers_ideas.revise(member.idea, [r["id"] for r in rows],
                            f"nceval.train {arm.codec}@{arm.preset} {arm.role} {arm.key}: "
                            f"{verdict}; {summary_line(entry)}", "nceval.train", outcome,
                            rung=f"{arm.codec}@{arm.preset}")


def commit_block(arm: Arm, entry: dict, base: Base, k: float, deferred: list) -> str:
    lines = [f"{arm.codec} {arm.preset}: " + ", ".join(sorted(arm.change)),
             "", f"Checked by nceval.train ({arm.role}, {arm.key}) against {base.commit[:12]} "
                 f"on draw {base.rotation}, k {k:.4g} "
                 f"({checkrun.preset_k(arm.codec, arm.preset, base.k)[1]}):"]
    for member in arm.members:
        lines.append(f"- {member.idea} ({member.mechanism}): "
                     f"{' '.join(checkrun.pairs(member.params))}; stage 1: "
                     + describe(member, k).split("; ", 2)[-1])
    lines += [f"- dCPU {check} (its own encodes): {decision_cpu.describe(arm.cpu.get(check))}"
              for check in (L3, FINAL_CHECK)]
    lines.append("- " + summary_line(entry))
    if guards := tripped_guards(entry):
        lines.append(f"{TRAIN_GUARD_LINE} {'; '.join(guards)}")
        lines.append(f"{JUSTIFICATION_PREFIX} {PLACEHOLDER} what the guards flag and why it "
                     f"is acceptable, or do not land>")
    lines += [f"- deferred: {c.id} {c.idea}: {why}" for c, why in deferred]
    lines.append("- streams change: <preset_identity listing>")
    return "\n".join(lines)


def report_pass(arm: Arm, entry: dict, base: Base, k: float, deferred: list) -> dict:
    verdict = pass_verdict(entry)
    set_status(arm.members, PASSED, f"{arm.role} {arm.key}")
    record(arm, entry, verdict, None)
    print(f"\n{verdict} {arm.label()}\n--- preset diff (frozen in {arm.directory}) ---\n"
          f"{arm.diff}--- commit message ---\n{commit_block(arm, entry, base, k, deferred)}\n",
          flush=True)
    return {"arm": arm.label(), "directory": str(arm.directory), "diff": arm.diff}


def proposal(arm: Arm, entry: dict) -> dict:
    """A passing arm, reported (report_pass) once its rivals are ranked (run_units)."""
    return {"arm": arm, "entry": entry}


def net_j(found: dict) -> tuple[float, float]:
    """Rank of a passing arm among alternatives: L3's J (CPU charged), then its CPU."""
    return (level_of(found["entry"][L3])["mean"], found["arm"].dcpu or 0.0)


def eject(candidate: Candidate, why: str) -> None:
    set_status([candidate], EJECTED, why)
    levers_ideas.revise(candidate.idea, [], f"nceval.train {candidate.codec}@{candidate.preset}: "
                        f"ejected: {why}", "nceval.train", "rejected",
                        rung=f"{candidate.codec}@{candidate.preset}")
    print(f"EJECT {candidate.id} ({candidate.idea}): {why}", flush=True)


# --- running a unit ----------------------------------------------------------------------

def run_round(arms: list[Arm], base: Base, measure: Measure, cloud: bool) -> dict[str, dict]:
    root = load_paths().data_root / "results" / "train" / "bundles"
    for arm in arms:
        freeze(arm, base.presets, root)
    print(f"round of {len(arms)} arm(s), base {base.commit[:12]} draw {base.rotation}:\n"
          + plan_minutes(arms, base, cloud), flush=True)
    started = time.monotonic()
    measured = measure(arms, base)
    cost = sum(e.get("cost", 0.0) for e in measured.values())
    print(f"round done in {(time.monotonic() - started) / 60:.1f} min"
          + (f", ${cost:.2f}" if cost else ""), flush=True)
    for arm in arms:
        entry = measured.get(arm.key, {})
        print(f"  {arm.label()}: " + (summary_line(entry) if FINAL_CHECK in entry and L3 in entry
                                      else "a screen failed"), flush=True)
    return measured


def judge_cloud(members: list[Candidate], arms: list[Arm], measured: dict, base: Base,
                k: float, deferred: list) -> dict | None:
    """A group's one parallel round, measured: the bundle passes, or members whose
    marginal contribution is worse are ejected and a passing leave-one-out arm is the
    verified bundle."""
    full = arms[0]
    if len(arms) == 1:
        if passes(measured[full.key]):
            return proposal(full, measured[full.key])
        for candidate in (full.members if len(full.members) == 1 else []):
            eject(candidate, f"{full.role}: {summary_line(measured[full.key])}")
        record(full, measured[full.key], "FAIL", None)
        return None
    by_out = {arm.role.split(":")[1]: arm for arm in arms[1:]}
    worse = []
    for member in members:
        loo = by_out[member.id]
        effects = {check: marginal(measured[full.key], measured[loo.key], check, k,
                                   full.cpu.get(check), loo.cpu.get(check))
                   for check in (L3, FINAL_CHECK)}
        print(f"  marginal {member.id} ({member.idea}): " + "; ".join(
            f"{c} {interval.describe(s)}" for c, s in effects.items()), flush=True)
        if any(s["lo"] is not None and s["lo"] > 0 for s in effects.values()):
            worse.append((member, effects))
    for arm in arms:
        record(arm, measured[arm.key],
               pass_verdict(measured[arm.key]) if passes(measured[arm.key]) else "FAIL", None)
    if not worse and passes(measured[full.key]):
        return proposal(full, measured[full.key])
    for member, effects in worse:
        eject(member, "marginal contribution worse in context: " + "; ".join(
            f"{c} {interval.describe(s)}" for c, s in effects.items()))
    if len(worse) == 1 and passes(measured[by_out[worse[0][0].id].key]):
        arm = by_out[worse[0][0].id]
        return proposal(arm, measured[arm.key])
    if not worse:
        passing = [a for a in arms[1:] if passes(measured[a.key])]
        if passing:
            best = min(passing, key=lambda a: level_of(measured[a.key][L3])["mean"])
            out = next(m for m in members if m.id == best.role.split(":")[1])
            eject(out, f"the bundle failed and only leaving it out passes ({best.key})")
            return proposal(best, measured[best.key])
    print(f"no arm of the group passes; {len(members) - len(worse)} member(s) stay queued",
          flush=True)
    return None


def run_group_local(members: list[Candidate], base: Base, measure: Measure, k: float,
                    deferred: list, rounds: int) -> dict | None:
    """Serial rounds: a failing bundle ejects its best-matching member or keeps its
    better half (the rest stays queued), then runs again."""
    for _ in range(rounds):
        if not members:
            return None
        arm = arms_of(members, cloud=False)[0]
        entry = run_round([arm], base, measure, cloud=False)[arm.key]
        if passes(entry):
            return proposal(arm, entry)
        record(arm, entry, "FAIL", None)
        culprit, why = blame(members, failing(entry), k)
        if culprit is not None:
            eject(culprit, f"{arm.role} {arm.key} failed ({summary_line(entry)}); {why}")
            members = [m for m in members if m is not culprit]
        else:
            ranked = sorted(members, key=lambda c: c.stage1_j(k))
            kept = ranked[:math.ceil(len(ranked) / 2)]
            for candidate in ranked[len(kept):]:
                deferred.append((candidate, f"bisected out of {arm.key} ({why})"))
                print(f"DEFER {candidate.id}: bisected ({why})", flush=True)
            members = kept
    print("rounds exhausted; the remaining members stay queued", flush=True)
    return None


@dataclass
class Unit:
    """An independent unit of a target, composed: its members, deferred, arms and k."""
    group: list[Candidate]
    members: list[Candidate]
    deferred: list[tuple[Candidate, str]]
    k: float
    arms: list[Arm]
    #: Units sharing it are alternatives (they set the same parameters): at most one
    #: of them passes, the best net J.
    rivals: str = ""


def prepare(group: list[Candidate], cloud: bool, base: Base) -> list[Unit]:
    """The group's bundle and, for each alternative, a unit of its own (rivals), at
    the rung's CPU charge (its front slope, or the run's --k)."""
    k, source = checkrun.preset_k(group[0].codec, group[0].preset, base.k)
    print(f"{group[0].codec}@{group[0].preset}: k {k:.4g} ({source})", flush=True)
    members, alternatives = compose(group, k)
    rivals = min(c.id for c in group) if alternatives else ""
    for candidate in alternatives:
        print(f"ALTERNATIVE {candidate.id} ({candidate.idea}): sets parameters of "
              f"{', '.join(m.id for m in members if set(m.params) & set(candidate.params))}"
              f"; checked alone, ranked against them", flush=True)
    return [Unit(members, members, [], k, arms_of(members, cloud), rivals)] + [
        Unit([c], [c], [], k, arms_of([c], cloud), rivals) for c in alternatives]


def outcome(unit: Unit, passed: dict | None, judged: bool = True) -> dict:
    return {"members": [c.id for c in unit.group], "passed": passed, "judged": judged,
            "deferred": [c.id for c, _ in unit.deferred]}


def run_units(units_: list[Unit], base: Base, measure: Measure, cloud: bool,
              rounds: int) -> list[dict]:
    """On boxes: every unit's arms in ONE round (packed onto boxes), then each judged;
    here: unit after unit, each in serial rounds."""
    if not cloud:
        found = [run_group_local(u.members, base, measure, u.k, u.deferred, rounds)
                 for u in units_]
    else:
        measured = run_round([a for u in units_ for a in u.arms], base, measure, cloud=True)
        found = []
        for unit in units_:
            missing = [f"{a.label()} {c}" for a in unit.arms for c in (FINAL_CHECK, L3)
                       if c not in measured.get(a.key, {})]
            if missing:
                print(f"NOT JUDGED (a screen failed, its members stay queued): "
                      f"{'; '.join(missing)}", flush=True)
                found.append("unjudged")
                continue
            found.append(judge_cloud(unit.members, unit.arms, measured, base, unit.k,
                                     unit.deferred))
    judged = [f if f != "unjudged" else None for f in found]
    return [outcome(u, settle(u, j, units_, judged, base), f != "unjudged")
            for u, f, j in zip(units_, found, judged)]


def settle(unit: Unit, found: dict | None, units_: list[Unit], everything: list,
           base: Base) -> dict | None:
    """Report a unit's passing arm, unless an alternative to it passed with a better net
    J (ties: the cheaper); a passing arm not chosen stays queued."""
    if found is None:
        return None
    rivals = [f for u, f in zip(units_, everything)
              if f is not None and unit.rivals and u.rivals == unit.rivals]
    best = min(rivals, key=net_j) if rivals else found
    if best is not found:
        note = f"alternative of {best['arm'].key}, which passed with a better net J"
        set_status(found["arm"].members, QUEUED, note)
        print(f"ALTERNATIVE NOT CHOSEN {found['arm'].label()}: {note}", flush=True)
        return None
    return report_pass(found["arm"], found["entry"], base, unit.k, unit.deferred)


def run_unit(group: list[Candidate], base: Base, measure: Measure, cloud: bool,
             rounds: int) -> dict:
    """The group's outcome (with alternatives: the first unit's, the bundle's)."""
    return run_units(prepare(group, cloud, base), base, measure, cloud, rounds)[0]


@contextmanager
def target_lock(codec: str, preset: str):
    """One `train run` per codec x preset at a time (refused, not queued)."""
    path = queue_path().with_name(f"train-{codec}-{preset}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another `train run` holds {codec}@{preset} ({path})")
        yield


def target_units(codec: str, preset: str, base: Base, cloud: bool) -> list[Unit]:
    queued = [c for c in load_queue() if (c.codec, c.preset, c.status) == (codec, preset, QUEUED)]
    checkrun.check_known(base.binary, codec, {n: v for c in queued for n, v in c.params.items()})
    groups = units(queued)
    print(f"== {codec}@{preset}: {len(queued)} candidate(s) in {len(groups)} independent "
          f"unit(s): " + "; ".join("+".join(f"{c.id}/{c.mechanism}" for c in g)
                                   for g in groups), flush=True)
    return [unit for g in groups for unit in prepare(g, cloud, base)]


def targets_of(args: argparse.Namespace) -> list[tuple[str, str]]:
    """--target codec:preset ..., --codec/--preset, else every target with queued work."""
    if args.target:
        return [tuple(t.split(":", 1)) for t in args.target]
    queued = sorted({(c.codec, c.preset) for c in load_queue() if c.status == QUEUED})
    if args.codec:
        queued = [t for t in queued if t[0] == args.codec and (not args.preset or t[1] == args.preset)]
    return queued


def run(args: argparse.Namespace, measure: Measure | None = None) -> list[dict]:
    targets = targets_of(args)
    if not targets:
        raise SystemExit("nothing queued for that")
    binary = Path(args.binary).resolve() if args.binary else checkrun.landed_binary(args.base)
    from . import buildinfo
    cloud = not args.local
    # refused here, in seconds, before anything is measured or rented
    tree = pinned_tree() if cloud and not args.plan else None
    base = Base(binary, buildinfo.info_of(binary).get("commit") or args.base,
                checkrun.presets_beside(binary), pool.rotation_key(args.rotation), args.k, tree)
    if measure is None:
        def measure(arms, base_):
            return measure_arms(arms, base_, cloud, args.budget, args.workers)
    with ExitStack() as locks:
        for target in targets:
            locks.enter_context(target_lock(*target))
        prepared = [u for target in targets for u in target_units(*target, base, cloud)]
        if args.plan:
            print_plan(prepared, base, cloud)
            return []
        results = run_units(prepared, base, measure, cloud, args.max_rounds)
    passed = [r for r in results if r["passed"]]
    unjudged = [r for r in results if not r.get("judged", True)]
    if unjudged:
        print(f"train run: {len(unjudged)} unit(s) NOT JUDGED (their box failed twice; "
              f"they stay queued): " + "; ".join("+".join(r["members"]) for r in unjudged),
              flush=True)
    print(f"train run: {len(passed)} of {len(results)} unit(s) passed; land each with "
          f"tools/dev/land.sh after preset_identity, static_recovery and landing "
          f"conformance", flush=True)
    return results


def print_plan(prepared: list[Unit], base: Base, cloud: bool) -> None:
    """The round's arms frozen under plans/, its expected time and exact commands."""
    root = load_paths().data_root / "results" / "train" / "plans"
    arms = [a for u in prepared for a in u.arms]
    for arm in arms:
        freeze(arm, base.presets, root)
    print(plan_minutes(arms, base, cloud))
    for arm in arms:
        for check in (FINAL_CHECK, L3):
            command = screen_command(arm, base, check, arm.directory / f"{check}.json",
                                     local=not cloud)
            print("  " + " ".join((["python3", str(AWS_MEASURE), "--"] if cloud else [])
                                  + command))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python3 -m nceval.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add", help="queue a candidate that passed nceval.quick")
    a.add_argument("--codec", required=True)
    a.add_argument("--preset", required=True)
    a.add_argument("--param", action="append", required=True)
    a.add_argument("--idea", required=True, help="the idea's ledger slug (levers ideas)")
    a.add_argument("--mechanism", required=True,
                   help="what it changes (subpel-refine, static-skip, partition, rate-control, "
                        "intra-search, loop-filter, mv-prediction, analysis, ...): candidates "
                        "sharing a tag are checked together, others independently")
    a.add_argument("--quick-json", required=True, help="the candidate's nceval.quick --json")
    a.add_argument("--force", action="store_true", help="queue a stage-1 KILL anyway")
    a.set_defaults(run=add)
    ls = sub.add_parser("list", help="the queue per codec x preset with stage-1 numbers")
    ls.add_argument("--codec")
    ls.add_argument("--all", action="store_true", help="also passed, ejected and dropped")
    ls.set_defaults(run=listing)
    d = sub.add_parser("drop", help="take a candidate off the queue")
    d.add_argument("id")
    d.add_argument("--reason")
    d.set_defaults(run=drop)
    r = sub.add_parser("run", help="check the queued candidates (default: every target)")
    r.add_argument("--codec")
    r.add_argument("--preset")
    r.add_argument("--target", action="append", help="codec:preset (repeatable)")
    r.add_argument("--local", action="store_true",
                   help="measure here (serial eject/bisect) instead of on rented boxes")
    r.add_argument("--plan", action="store_true",
                   help="print the arms, expected times and exact commands; measure nothing")
    r.add_argument("--base", default="HEAD", help="commit of the base build (default HEAD)")
    r.add_argument("--binary", help="base build (default: the landed cache's --base)")
    r.add_argument("--rotation", help="pool rotation key (default: today, UTC)")
    r.add_argument("--k", type=float,
                   help="CPU charge for every target (default: each rung's front slope, "
                        "nceval.front_k; an explicit k is warned about loudly)")
    r.add_argument("--budget", type=float, default=4.0, help="USD cap per box (aws_measure)")
    r.add_argument("--max-rounds", type=int, default=DEFAULT_ROUNDS, help="--local rounds")
    r.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    r.set_defaults(run=run)
    args = p.parse_args(argv)
    cloud.check_fresh()
    results = args.run(args)
    # every finished unit is reported above; a unit left unjudged fails the run, last
    if args.command == "run" and any(not r.get("judged", True) for r in results or []):
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
