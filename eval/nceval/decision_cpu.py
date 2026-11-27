"""A decision's CPU, measured on the very encodes its quality judges.

Owner rule (2026-11-23): never combine measurements of different things. An
acceptance or tuning decision charges J + k dCPU; until 2026-11-23 its dCPU came
from other encodes than its J (aws_cpu_ab at an absolute 600 kbps on its own four
clips, or callgrind at a fixed QP on 13 frames), while J was CBR at each clip's
ladder rates on the screen's draw. Here the CPU commands are derived from the
decision's own quality jobs and nowhere else:

    quality jobs (Runner.points, as the decision measures them)
      -> cpu_jobs()          the test family's commands (Bench.command), with the
                             binary, input and stream as placeholders and the side's
                             own arguments (preset, --param) split off
      -> declared_subset()   per protocol group (frames and key period, or one key;
                             layers): one clip per content class, the longest, at the
                             first rung the decision codes for it (a subset of the
                             same commands)
      -> plan_entry()        what tools/dev/aws_cpu_ab.py --decision-plan measures:
                             both sides on the same core, interleaved rounds
      -> charge()            the summary's change per group, refused unless its
                             digest equals the decision's own (`digest`)

So rate mode, per-clip rates, buffer, clips (with their rotated start frames),
frames, key period and layers are the decision's own by construction; tests
(eval/tests/test_decision_cpu.py) fail on any difference. Each quality level is
charged the CPU of its own group: the 150-frame one-key levels are no longer
charged the 1/150 share's 600-frame CPU.

What the CPU commands leave out: `--display-recon` and `--rate-log`, which only
the scoring reads (the published charts' timing drops them too, nceval.chart_timing).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .cbr import Bench, Point
from .buildinfo import clean_commit
from .checkrun import combined_dcpu

PLAN_VERSION = 1
#: Placeholders for the encoder binary, the input clip and the stream in a job's argv.
BIN, INPUT, OUTPUT = "{BIN}", "{INPUT}", "{OUTPUT}"
#: The rented CPU targets a decision's charge needs (cpu_ab_shard.DEFAULT_TARGETS): the
#: benchmark's two timing machines (BP-3.10), Granite Rapids (c8i.2xlarge) for x86 and
#: Graviton 4 (c8g.2xlarge) for Arm, weighted equally (checkrun.combined_dcpu). The optional
#: zen5 target is reported beside them, never charged.
X86_TARGETS = ("graniterapids",)
ARM_TARGETS = ("graviton4",)
#: Separates the decision's name from its group in aws_cpu_ab's pair names.
GROUP_MARK = "@"
AWS_CPU_AB = Path(__file__).resolve().parents[2] / "tools" / "dev" / "aws_cpu_ab.py"
#: aws_cpu_ab's boxes per target for one plan, and its wall limit (minutes).
PLAN_BOXES = 3
MEASURE_LIMIT_MINUTES = 60


#: How the ledger names a charge measured this way (levers rows' x86_how).
LEDGER_HOW = ("decision CPU: own encodes, c8i Granite Rapids + c8g Graviton 4 "
              "(nceval.decision_cpu)")


class CpuMismatch(SystemExit):
    """A CPU measurement of other encodes than the decision's: refused."""


@dataclass(frozen=True)
class CpuJob:
    """One encode of the decision, as aws_cpu_ab runs it (both sides)."""
    clip: str
    content_class: str
    start_frame: int
    frames: int
    intra: int
    layers: int
    kbps: int
    #: The command without the side's own arguments; {BIN}/{INPUT}/{OUTPUT} placeholders.
    argv: tuple[str, ...]

    @property
    def group(self) -> str:
        """The protocol group: what one quality level codes (frames, key period, layers)."""
        return group_name(self.frames, self.intra, self.layers)

    def record(self) -> dict:
        return {**asdict(self), "argv": list(self.argv)}

    @staticmethod
    def from_record(record: dict) -> CpuJob:
        return CpuJob(**{**record, "argv": tuple(record["argv"])})


#: The group of a decision's one-key encodes (`group_name`).
ONE_KEY = "key1"


def group_name(frames: int, intra: int, layers: int = 1) -> str:
    """A protocol group: frames and key period, or ONE_KEY for every encode with one key
    (frames <= key period: the 150-frame levels, clips shorter than that included; a
    decision codes them at one frame count)."""
    base = ONE_KEY if frames <= intra else f"f{frames}-k{intra}"
    return base + (f"-l{layers}" if layers != 1 else "")


def job_of(bench: Bench, point: Point) -> CpuJob:
    """The CPU job of one quality point: its own command (Bench.command), the family's
    tool, extra arguments and the clip's file replaced by placeholders."""
    bare = replace(point.family, tool=Path(BIN), extra=())
    clip = replace(point.clip, yuv=Path(INPUT))
    argv = bench.command(replace(point, family=bare, clip=clip), Path(OUTPUT), None)
    if point.kbps is None:
        raise CpuMismatch(f"{point.family.name} {point.clip.name}: a fixed-QP point; decisions "
                          f"judge CBR quality")
    return CpuJob(point.clip.name, point.clip.content_class, point.clip.start_frame,
                  bench.frames, bench.intra, bench.layers, point.kbps, tuple(argv))


def cpu_jobs(quality: list[tuple[Bench, Point]], family_name: str) -> list[CpuJob]:
    """The CPU jobs of `family_name`'s quality points (the candidate's), in order, once."""
    seen, jobs = set(), []
    for bench, point in quality:
        if point.family.name != family_name:
            continue
        job = job_of(bench, point)
        if job not in seen:
            seen.add(job)
            jobs.append(job)
    return jobs


def declared_subset(jobs: list[CpuJob]) -> list[CpuJob]:
    """The CPU subset of a decision's jobs: per group and content class, the longest clip
    (then the first by name), at the first of its rates the decision codes."""
    chosen: dict[tuple[str, str], CpuJob] = {}
    for job in sorted(jobs, key=lambda j: (j.group, j.content_class, -j.frames, j.clip)):
        chosen.setdefault((job.group, job.content_class), job)
    return sorted(chosen.values(), key=lambda j: (j.group, j.content_class, j.clip))


def digest(jobs: list[CpuJob], base_extra: list[str], cand_extra: list[str]) -> str:
    """The identity of what a CPU measurement measures: its jobs and both sides'
    arguments (not the binaries' paths, which differ between this PC and a box)."""
    text = json.dumps({"version": PLAN_VERSION, "jobs": [j.record() for j in jobs],
                       "base": list(base_extra), "cand": list(cand_extra)}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def plan_entry(name: str, codec: str, rung: str | None, rotation: str | None,
               quality: list[tuple[Bench, Point]], anchor, test,
               base_build: str, cand_build: str) -> dict:
    """One decision of an aws_cpu_ab --decision-plan: the candidate's quality jobs'
    declared subset, both sides' builds and arguments, and their digest."""
    jobs = declared_subset(cpu_jobs(quality, test.name))
    if not jobs:
        raise CpuMismatch(f"{name}: the decision has no candidate encodes to time")
    sources = {p.clip.name: p.clip for _, p in quality}
    return {"version": PLAN_VERSION, "name": name, "codec": codec, "rung": rung,
            "rotation": rotation,
            "digest": digest(jobs, list(anchor.extra), list(test.extra)),
            "base": {"build": base_build, "extra": list(anchor.extra)},
            "cand": {"build": cand_build, "extra": list(test.extra)},
            "clips": {j.clip: clip_source(sources[j.clip]) for j in jobs},
            "jobs": [j.record() for j in jobs]}


def build_of(binary: Path) -> str:
    """What aws_cpu_ab builds a side from: the binary's clean commit (built for every
    target's architecture), else the binary itself (one architecture)."""
    return clean_commit(binary) or str(binary)


def clip_source(clip) -> dict:
    """What aws_cpu_ab needs to cut a clip's frames (nceval.clipframes serves them)."""
    return {"name": clip.name, "yuv": str(clip.yuv), "width": clip.width,
            "height": clip.height, "fps": clip.fps, "frames": clip.frames,
            "source_kbps": clip.source_kbps, "content_class": clip.content_class,
            "split": clip.split}


def write_plan(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": PLAN_VERSION, "decisions": entries}, indent=1) + "\n")
    for e in entries:
        groups = sorted({CpuJob.from_record(j).group for j in e["jobs"]})
        print(f"CPU plan {e['name']}: {len(e['jobs'])} encodes per side and round "
              f"({', '.join(groups)}), digest {e['digest']} -> {path}", flush=True)


# --- the measured charge ------------------------------------------------------------------

@dataclass(frozen=True)
class Charge:
    """A decision's measured CPU change in percent per group, with where it came from."""
    digest: str
    groups: dict[str, float]
    source: str
    #: Per group, the x86 and Arm changes the combined value is made of.
    parts: dict[str, dict[str, float]] = field(default_factory=dict)

    def text(self) -> str:
        """The compact form a screen offloaded to a box receives (`parse`)."""
        return f"{self.digest}:" + ",".join(f"{g}={v:.4f}" for g, v in sorted(self.groups.items()))

    def dcpu(self, frames: int, intra: int, layers: int = 1) -> float:
        group = group_name(frames, intra, layers)
        if group not in self.groups:
            raise CpuMismatch(f"no CPU measured for {group} (measured: "
                              f"{', '.join(sorted(self.groups))}; {self.source})")
        return self.groups[group]

    def mean(self, part: str | None = None) -> float | None:
        """The groups' equal-weight mean (a long window's key shares, as their verdicts
        are weighted), of the combined change or of one `part` (x86, arm)."""
        values = [self.groups[g] if part is None else self.parts.get(g, {}).get(part)
                  for g in self.groups]
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None


def parse(text: str) -> Charge:
    found, _, rest = text.partition(":")
    groups = {g: float(v) for g, _, v in (p.partition("=") for p in rest.split(",") if p)}
    return Charge(found, groups, f"verified charge {found}")


def charge(summary: dict, name: str, source: str) -> Charge:
    """The decision `name`'s charge from an aws_cpu_ab summary: per group, the x86 and
    Arm changes (cycles, else CPU time) combined at equal weight."""
    decision = (summary.get("decisions") or {}).get(name)
    if decision is None:
        raise CpuMismatch(f"{source}: no decision {name!r} (an aws_cpu_ab run without "
                          f"--decision-plan measures other encodes than any screen)")
    by_group: dict[str, dict[str, float]] = {}
    for row in summary["rows"]:
        pair, _, group = row["pair"].rpartition(GROUP_MARK)
        if pair == name and row["change_pct"]:
            by_group.setdefault(group, {})[row["target"]] = row["change_pct"][0]
    groups, parts = {}, {}
    for group in decision["groups"]:
        targets = by_group.get(group, {})
        x86 = [targets[t] for t in X86_TARGETS if t in targets]
        arm = [targets[t] for t in ARM_TARGETS if t in targets]
        if not x86 or not arm:
            raise CpuMismatch(f"{source}: {name} {group} lacks an x86 or Arm target "
                              f"(measured {sorted(targets)})")
        parts[group] = {"x86": sum(x86) / len(x86), "arm": sum(arm) / len(arm)}
        groups[group] = combined_dcpu(parts[group]["x86"], parts[group]["arm"])
    return Charge(decision["digest"], groups, f"{source} ({name})", parts)


def measure(entries: list[dict], directory: Path, purpose: str) -> Path:
    """Time every decision of `entries` (plan_entry) in one aws_cpu_ab run; the summary's
    path (each decision's charge: `load_charge(summary, name)`)."""
    directory.mkdir(parents=True, exist_ok=True)
    plan, summary = directory / "cpu-plan.json", directory / "cpu-summary.json"
    write_plan(plan, entries)
    subprocess.run([sys.executable, str(AWS_CPU_AB), "--decision-plan", str(plan), "--json",
                    str(summary), "--boxes", str(PLAN_BOXES), "--purpose", purpose],
                   check=True, timeout=MEASURE_LIMIT_MINUTES * 60)
    return summary


def load_charge(path: Path, name: str) -> Charge:
    return charge(json.loads(path.read_text()), name, str(path))


def plan_rotation(path: Path, name: str) -> str | None:
    """The pool draw the summary's decision `name` was planned on."""
    return json.loads(path.read_text())["decisions"][name]["rotation"]


def verify(found: Charge, expected_digest: str, what: str) -> Charge:
    """`found` if it measured exactly the decision's CPU subset, else refuse."""
    if found.digest != expected_digest:
        raise CpuMismatch(
            f"{what}: the CPU was measured on other encodes (digest {found.digest}, this "
            f"decision's is {expected_digest}; {found.source}). Write this decision's plan "
            f"(--cpu-plan) and measure it: tools/dev/aws_cpu_ab.py --decision-plan PLAN")
    return found


def without_local_cpu(argv: list[str]) -> list[str]:
    """A command line for a box: without the local --cpu-summary file (the verified charge
    travels as --cpu-charge) and --rotation (the charge's draw is added with it)."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg in ("--cpu-summary", "--rotation"):
            skip = True
        elif not arg.startswith(("--cpu-summary=", "--rotation=")):
            out.append(arg)
    return out


def record(found: Charge | None) -> dict | None:
    """A result file's record of the charge (`from_record` reads it back)."""
    return None if found is None else asdict(found)


def from_record(stored: dict | None) -> Charge | None:
    return None if stored is None else Charge(**stored)


def describe(found: Charge | None) -> str:
    if found is None:
        return "no CPU charge"
    return ", ".join(f"{g} {v:+.2f}%" for g, v in sorted(found.groups.items())) + \
        f" ({found.source})"
