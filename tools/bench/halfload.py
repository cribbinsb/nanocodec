#!/usr/bin/env python3
"""The box side of a BP-4 timing run: one encoder family at a time on half the machine.

    sudo python3 -I halfload.py --plan plan.json --checkset checkset.json \\
        --manifest manifest.json --series c8g --clips /mnt/nanocodec-clips --out out/

Runs on one whole bare-metal machine (docs/benchmark-procedure.md sections 2 and 6, D13), beside
box.py, whose per-encode checks it reuses (V2, V4, V5, V7, V8, V9, V19, V21). Standard library
only.

  - **Slots:** the first hardware thread of every second physical core from core 1 (`lscpu`);
    core 0 holds this harness; SMT siblings and the other cores stay idle. Each encode is
    pinned to its slot by CPU affinity, inherited at the fork (box.timed_encode without a
    cgroup: 48 cgroup moves at once queued on the kernel's migration lock), and V8 checks it
    from Cpus_allowed_list.
  - **Families one at a time:** `plan.json` lists this machine's families in order, each with
    its jobs. A family's timed jobs go longest first to the next free slot. A slot with no timed
    job left while others still run codes an untimed copy of one of the family's jobs (filler),
    so every timed encode runs under the same load, made only of its own family. A job refused
    on a timing check (V7) is coded again once at the end of its pass; a family with more than
    MAX_REFUSED_SHARE of its jobs refused is marked refused (the report re-codes it whole).
  - **Mixed queue (`--queue mixed`, the Mac and Jetson series, owner 2026-11-26):** one pass over
    all the plan's timed jobs, longest first across families, with no filler: every slot always
    runs a timed job until the queue empties, so each encode is loaded by the run's own mixed
    jobs. Each family's `family_done` is written when its last job settles, so a restart
    resumes by family; the driver also leaves out job ids already recorded.
  - **Check set (V6, D13):** `checkset.json` holds the check families' jobs; each runs as its
    own family pass (CHECK_ROUNDS times) once when the machine starts and once at its end
    (section 6; owner, 2026-11-25: whole machines do not drift, minimal overhead).
    `check_done` marks each complete set. A family settles the moment it finishes, so a
    reclaim loses only the family in progress. The driver compares the start set with the
    series' other machines; the end set is drift, reported. Nothing here rescales a time.
  - **Records** (`out/encodes.jsonl`, appended as each encode ends): a header with the layout,
    each timed encode (with `load`, the slots busy when it started), the check encodes
    (`check`: start or end), and a marker when a family ends (`family_done` or
    `family_refused`). The driver copies `out/` every half minute.
  - **Spot notice / STOP:** between encodes; the machine then starts no new encode, lets the
    running ones end, writes `interrupted` and exits. The interrupted family has no
    `family_done`, so the driver codes it again whole on a replacement machine.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
import box  # noqa: E402  (shipped beside this file)

#: The report's V7 clock rule (report.MIN_CLOCK_SHARE, MIN_CLOCK_WALL_S; a test keeps them
#: equal), applied here as well, so a timing it would refuse is coded again at once on the
#: same, now warm core instead of leaving a job for another machine (owner, 2026-11-26).
CLOCK_MIN_SHARE, CLOCK_MIN_WALL_S = 0.95, 0.25
#: Timed encodes seen before the running median clock is trusted.
CLOCK_MIN_SAMPLES = 20

#: D13: slots on every second physical core, starting at core 1 (core 0 holds the harness).
SLOT_START, SLOT_STRIDE = 1, 2
#: The check set's repeats at each check point (section 6).
CHECK_ROUNDS = 3
#: A family pass with more than this share of its jobs refused on timing is refused whole.
MAX_REFUSED_SHARE = 0.01
#: The driver's STOP file (any stop of a machine: a cancel, a spend ceiling).
STOP_FILE = "STOP"
#: The spot notice is read at most this often (every slot asks before each encode).
NOTICE_EVERY_S = 2.0


def physical_cores(lscpu_parse: str) -> list[int]:
    """The first hardware thread of each physical core, in core order, from `lscpu -p=CPU,CORE,SOCKET`."""
    seen, cpus = set(), []
    for line in lscpu_parse.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu, core, socket = (int(x) for x in line.split(",")[:3])
        if (socket, core) not in seen:
            seen.add((socket, core))
            cpus.append(cpu)
    return cpus


def half_slots(cores: list[int]) -> tuple[int, list[int]]:
    """(harness CPU, slot CPUs): core 0 for the harness, every second core from core 1."""
    return cores[0], cores[SLOT_START::SLOT_STRIDE]


class Stopped(Exception):
    """A spot notice or the driver's STOP."""


class FamilyPass:
    """One family's timed jobs on every slot, longest first, with same-family filler.

    `encode(job, cpu, timed)` codes one job on one slot and returns its record (raising
    box.TimingRefusal on V7, box.JobFailure on a failed encode). `stopped()` says whether to
    stop taking work. Records go to `write`."""

    def __init__(self, family: str, jobs: list[dict], slots: list[int],
                 encode: Callable[[dict, int, bool], dict], write: Callable[[dict], None],
                 stopped: Callable[[], bool], extra: dict | None = None,
                 filler: bool = True) -> None:
        self.family, self.slots, self.encode, self.write = family, slots, encode, write
        self.stopped, self.extra = stopped, extra or {}
        self.queue = sorted(jobs, key=lambda j: -j.get("expected_s", 0.0))
        self.total = len(self.queue)
        self.filler = list(self.queue) if filler else []
        #: Jobs still unsettled per family: a mixed pass writes each family's end as it comes.
        self.open = {}
        for job in self.queue:
            self.open[job["family"]] = self.open.get(job["family"], 0) + 1
        self.lock = threading.Lock()
        self.busy = self.settled = self.refused = 0
        self.retried: set[str] = set()
        self.clocks: list[float] = []  # median clock of each timed encode so far (MHz)
        self.interrupted = False
        self.error: BaseException | None = None

    def next_job(self, cpu: int) -> tuple[dict, bool, int] | None:
        """(job, timed, timed encodes running with it) for a slot, or None when the pass is
        over for it."""
        with self.lock:
            if self.error or self.interrupted:
                return None
            if self.stopped():
                self.interrupted = True
                return None
            if self.queue:
                job = self.queue.pop(0)
                self.busy += 1
                return job, True, self.busy
            if self.settled + self.refused < self.total and self.filler:
                filler = self.filler[cpu % len(self.filler)]
                return {**filler, "id": f"filler{cpu}|{filler['id']}"}, False, self.busy
            return None

    def worker(self, cpu: int) -> None:
        while True:
            taken = self.next_job(cpu)
            if taken is None:
                return
            job, timed, load = taken
            try:
                rec = self.encode(job, cpu, timed)
            except box.TimingRefusal as why:
                if not timed:
                    continue
                self.timing_refused(job, why)
                continue
            except box.JobFailure as why:
                if timed:
                    self.done_timed({"id": job["id"], "failed": str(why), **self.extra},
                                    job["family"])
                continue
            except BaseException as error:  # noqa: BLE001 - any other refusal ends the pass
                with self.lock:
                    self.error = error
                return
            if timed and self.clock_too_low(rec):
                self.timing_refused(job, box.TimingRefusal(
                    f"V7: clock {rec['mhz_median']:.0f} MHz under {CLOCK_MIN_SHARE:.0%} of the "
                    f"run's median"), now=True)
                continue
            if timed:
                self.done_timed({**rec, "load": load, **self.extra}, job["family"])

    def clock_too_low(self, rec: dict) -> bool:
        """Whether the report would refuse this timing for a low clock (V7)."""
        mhz = rec.get("mhz_median")
        if not mhz:
            return False
        with self.lock:
            self.clocks.append(float(mhz))
            ordered = sorted(self.clocks)
            median = ordered[len(ordered) // 2]
            enough = len(ordered) >= CLOCK_MIN_SAMPLES
        return enough and rec.get("wall_s", 0) >= CLOCK_MIN_WALL_S and mhz < CLOCK_MIN_SHARE * median

    def timing_refused(self, job: dict, why: Exception, now: bool = False) -> None:
        """A refused timing is coded again once: at the end of the pass, or with `now` next
        (a low clock: the core that just finished is warm)."""
        with self.lock:
            self.busy -= 1
            if job["id"] in self.retried:
                self.refused += 1
                self.write({"id": job["id"], "timing_refused": str(why), **self.extra})
                self.settle(job["family"])
            else:
                self.retried.add(job["id"])
                if now:
                    self.queue.insert(0, job)
                else:
                    self.queue.append(job)
                self.write({"id": job["id"], "retry": str(why), **self.extra})

    def done_timed(self, rec: dict, family: str) -> None:
        with self.lock:
            self.busy -= 1
            self.settled += 1
            self.write(rec)
            self.settle(family)

    def settle(self, family: str) -> None:
        """One more of `family`'s jobs settled (lock held); a filler-free pass (several families)
        marks the family's end with its last job."""
        self.open[family] -= 1
        if not self.open[family] and not self.filler:
            self.write({"family_done": family, **self.extra})

    def run(self) -> str:
        """"done", "refused" (too many timing refusals) or "interrupted"; raises a refusal."""
        threads = [threading.Thread(target=self.worker, args=(cpu,)) for cpu in self.slots]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if self.error:
            raise self.error
        if self.interrupted:
            return "interrupted"
        return "refused" if self.refused > MAX_REFUSED_SHARE * self.total else "done"


class Machine:
    """This machine's slots, groups and records."""

    def __init__(self, a, slots: list[int]) -> None:
        self.a, self.slots = a, slots
        self.box = {"bin": HERE / "bin", "cfg": HERE / "cfg", "clips": a.clips,
                    "series": a.series, "streams": a.out / "streams"}
        self.box["hashes"] = box.check_binaries(json.loads(a.manifest.read_text()), self.box["bin"])
        self.scratch = {cpu: a.out.parent / "scratch" / str(cpu) for cpu in slots}
        for d in (*self.scratch.values(), self.box["streams"]):
            d.mkdir(parents=True, exist_ok=True)
        self.log = open(a.out / "encodes.jsonl", "a")
        self.log_lock = threading.Lock()
        self.warmed: set[str] = set()
        self.notice_at, self.notice = 0.0, False

    def write(self, rec: dict) -> None:
        with self.log_lock:
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()

    def encode(self, job: dict, cpu: int, timed: bool) -> dict:
        """One encode on one slot (box.encode_job with this slot's scratch); a filler's stream
        is deleted."""
        slot_box = {**self.box, "scratch": self.scratch[cpu]}
        rec = box.encode_job(job, cpu, None, slot_box, self.a.out, checked=timed)
        if not timed:
            (self.box["streams"] / Path(rec["stream"]).name).unlink(missing_ok=True)
        return rec

    def warm(self, jobs: list[dict]) -> None:
        """One untimed encode per tool before its first timed one (box.Warm's reason), with the
        tool's shortest job: the warm-up only has to touch the binary and its code paths, and
        jobs come longest first, so the first job made the other slots wait out a family's
        longest encode (coordinator, 2026-11-25; docs/analysis/time-audit-2026-11-25.md)."""
        for job in sorted(jobs, key=lambda j: j.get("expected_s", 0.0)):
            if job["argv"][0] not in self.warmed:
                self.warmed.add(job["argv"][0])
                self.encode({**job, "id": f"warmup|{job['id']}"}, self.slots[0], False)

    def stopped(self) -> bool:
        if (self.a.out.parent / STOP_FILE).exists():
            return True
        if time.monotonic() - self.notice_at >= NOTICE_EVERY_S:
            self.notice_at, self.notice = time.monotonic(), bool(box.spot_notice())
        return self.notice

    def run_pass(self, family: str, jobs: list[dict], extra: dict, filler: bool = True) -> str:
        self.warm(jobs)
        outcome = FamilyPass(family, jobs, self.slots, self.encode, self.write, self.stopped,
                             extra, filler).run()
        print(f"{family} {extra.get('check', '')}: {outcome}", flush=True)
        return outcome

    def check_set(self, where: str, checks: dict[str, list[dict]]) -> bool:
        """The check set at `where` (start or end); False when stopped."""
        for family, jobs in checks.items():
            rounds = [{**j, "id": f"check-{where}-{r}|{j['id']}"} for r in range(CHECK_ROUNDS)
                      for j in jobs]
            if self.run_pass(family, rounds, {"check": where}) == "interrupted":
                return False
        self.write({"check_done": where})
        return True

    def serve(self, families: list[dict], checks: dict[str, list[dict]],
              check_at: tuple[str, ...] = ("start", "end"), queue: str = "family") -> int:
        if "start" in check_at and not self.check_set("start", checks):
            self.write({"interrupted": "during the start check set"})
            return 0
        if queue == "mixed":
            jobs = [job for entry in families for job in entry["jobs"]]
            if jobs and self.run_pass("mixed", jobs, {"pass": "mixed"}, filler=False) == "interrupted":
                self.write({"interrupted": "mixed"})
                return 0
            families = []
        for entry in families:
            family = entry["family"]
            outcome = self.run_pass(family, entry["jobs"], {"pass": family})
            if outcome == "interrupted":
                self.write({"interrupted": family})
                return 0
            self.write({f"family_{outcome}": family})
        if "end" in check_at and not self.check_set("end", checks):
            self.write({"interrupted": "during the end check set"})
        return 0


def main() -> int:
    box.heartbeat()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", type=Path, required=True, help="this machine's families, in order")
    ap.add_argument("--checkset", type=Path, required=True, help="the check families' jobs")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--series", required=True)
    ap.add_argument("--clips", type=Path, required=True, help="the clips' tmpfs directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check-at", choices=("both", "start", "end", "none"), default="both",
                    help="where this invocation runs the check set: a machine that serves its "
                         "families in several invocations (the Jetson's clip batches, which do "
                         "not fit its tmpfs at once) runs the start set only in its first and "
                         "the end set only in its last")
    ap.add_argument("--queue", choices=("family", "mixed"), default="family",
                    help="family: one family at a time with same-family filler (BP-4, D13); "
                         "mixed: one queue of every timed job, longest first, no filler")
    ap.add_argument("--slots", type=int, default=2,
                    help="macOS only: the number of concurrent encodes (no pinning there)")
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    if box.DARWIN:
        # macOS cannot pin: the slots are concurrent encodes the scheduler places, and the
        # harness is not pinned either (TIMING-METHOD.md of the Mac series measured the count).
        harness, slots = None, list(range(a.slots))
    else:
        lscpu = subprocess.run(["lscpu", "-p=CPU,CORE,SOCKET"], capture_output=True, text=True,
                               check=True).stdout
        harness, slots = half_slots(physical_cores(lscpu))
        os.sched_setaffinity(0, {harness})
    machine = Machine(a, slots)
    plan = json.loads(a.plan.read_text())
    checks = json.loads(a.checkset.read_text())
    machine.write({"box": {"layout": "unpinned" if box.DARWIN else "half", "queue": a.queue, "harness_cpu": harness, "slots": slots,
                           "cpu_model": box.cpu_model(), "series": a.series,
                           "instance_type": plan.get("instance_type")}})
    try:
        check_at = {"both": ("start", "end"), "none": ()}.get(a.check_at, (a.check_at,))
        return machine.serve(plan["families"], checks, check_at, a.queue)
    except box.Refusal as why:
        machine.write({"refused": str(why), "timing": isinstance(why, box.TimingRefusal)})
        print(f"refused: {why}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
