"""BP-4 timing on whole bare-metal machines: the plan and the driver (`fleet.py plan-timing`,
`fleet.py run --layout half`).

docs/benchmark-procedure.md sections 2 and 6 (D13, D14, D15). The box side is `halfload.py`.

  - **One machine type per chip, bare metal** (D14, V28): MACHINE_TYPES. Capacity is sought
    for CAPACITY_WAIT_S: each round tries every zone of the home region, then every spill
    region whose cheapest spot price for the type is within SPILL_PRICE_FACTOR of the home
    region's; a new round starts every CAPACITY_RETRY_S. Never another size, never non-metal.
  - **The plan** (`timing_plan`): each job's expected seconds (fleet.expected_seconds), the
    families ordered by total expected time, longest first, and assigned whole to machines,
    each to the machine with the least planned work. A machine's expected wall time is the sum
    over its families of max(longest job, family total / slots), times the chip's measured
    half-load slowdown, plus setup. The plan, with its expected wall time and cost, goes into
    the run (`timing-plan.json`) before the first encode (principle 4).
  - **The check set** (V6, D13): CHECK_FAMILIES on the run's first CHECK_CLIPS clips at
    CHECK_RUNG, coded once when a machine starts and once at its end if it survives. The
    driver compares the start set with the series' other machines (`start_set_ratio`): a
    machine more than CHECK_SET_LIMIT off does no work and is replaced. The end set is drift,
    recorded and reported, never a refusal (owner, 2026-11-25: whole bare-metal machines do
    not drift, measured < 0.35 % over 40 min and half-load repeats <= 0.28 %).
  - **Designed for failure:** each machine's `out/` comes back every POLL_S; a family settles
    the moment its `family_done` marker is on the PC (after the machine's start set), so a
    reclaim loses only the family in progress. A machine lost or interrupted gives its
    unsettled families to a replacement (at most MAX_ATTEMPTS machines in a row that settle
    nothing).
"""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path

#: D14: the one bare-metal timing type per chip (the series names the chip).
MACHINE_TYPES = {"c8i": "c8i.metal-48xl", "c8g": "c8g.metal-24xl"}
#: Every second physical core of a 96-core socket, core 0 excepted (halfload.half_slots).
SLOTS = 48
#: The measured mean half-load time against alone over 17 families (2026-11-25,
#: docs/analysis/whole-box-timing-2026-11-25.md): Intel x1.00-1.07, Graviton x1.01-1.23.
HALF_LOAD_SLOWDOWN = {"c8i": 1.03, "c8g": 1.11}
#: A machine's launch, upload, clip decompression, warm-ups and its two check sets.
SETUP_S = 600
#: D14: capacity is sought this long, a round every CAPACITY_RETRY_S. Owner, 2026-11-26: with
#: no spot capacity, keep waiting and retrying (never on demand, never another size), so the
#: search runs for hours, not 10 minutes; progress shows in each machine's log.
CAPACITY_WAIT_S, CAPACITY_RETRY_S = 4 * 3600, 60
HOME_REGION = "eu-north-1"
#: The regions a capacity search may try beyond home, each only within SPILL_PRICE_FACTOR of
#: home's price (owner, 2026-11-26: every cheap region, not only the build box's spill pair;
#: metal spot capacity was absent in eu-north-1 and eu-central-1 for hours).
CAPACITY_REGIONS = ("eu-west-1", "eu-central-1", "eu-west-3", "eu-west-2", "eu-south-2",
                    "us-east-1", "us-east-2", "us-west-2", "ca-central-1", "ap-northeast-1",
                    "ap-southeast-2", "ap-south-1")
#: A spill region is tried only within this factor of the home price (aws_choose's rule).
SPILL_PRICE_FACTOR = 1.5
#: Owner, 2026-11-26 ("of course we will pay more"): the metal timing machines are taken at a spot
#: price up to this many USD/h in any region, still spot and under half the on-demand price
#: (c8g.metal-24xl $3.8-4.4, c8i.metal-48xl $9-10.3); 1.5x eu-north-1 had left Graviton one region.
BENCH_SPOT_CAP_USD = 2.0
#: V6 (section 6): the check set.
CHECK_FAMILIES = ("x264-medium", "x264-ultrafast", "nanocodec-av1@medium", "nanocodec-h264@medium")
CHECK_CLIPS, CHECK_RUNG = 2, 500
#: V6 (D13): a machine's start check set within this of the series' other machines.
CHECK_SET_LIMIT = 0.01
#: How often the driver copies a machine's records back (rule 1 of the cost guide); a
#: rehearsal polls faster, so its forced reclaim lands mid-run on families of seconds.
POLL_S, REHEARSAL_POLL_S = 30, 2
#: Machines in a row that settle nothing before a machine slot gives up.
MAX_ATTEMPTS = 3
#: A machine's hard lifetime is its planned wall time times this, plus setup, and at least
#: MIN_LIFETIME_S. 2026-11-26: two c8g.metal-24xl were ended by their own 32-34 min lifetime
#: part-way through (17 and 30 families left): 16 min of it went waiting for our binary and the
#: plan's times underestimate Graviton. A finished machine is ended at once, so a long
#: lifetime costs nothing; the lifetime only bounds a runaway.
LIFETIME_FACTOR = 2.0
MIN_LIFETIME_S = 3 * 3600
PLAN_FILE = "timing-plan.json"


def machine_type(series: str) -> str:
    try:
        return MACHINE_TYPES[series]
    except KeyError:
        raise SystemExit(f"refused: no BP-4 timing type for series {series!r} "
                         f"(D14: {', '.join(MACHINE_TYPES)})") from None


def families_of(jobs: list[dict]) -> dict[str, list[dict]]:
    """{family: its jobs, longest first}."""
    out: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        out[job["family"]].append(job)
    return {f: sorted(js, key=lambda j: -j["expected_s"]) for f, js in out.items()}


def pass_seconds(jobs: list[dict], slots: int) -> float:
    """A family pass's expected wall time on `slots`: its longest job, or its work spread."""
    return max(jobs[0]["expected_s"], sum(j["expected_s"] for j in jobs) / slots)


def assign(families: dict[str, list[dict]], machines: int, slots: int) -> list[list[str]]:
    """Whole families to machines: longest total first, each to the least-loaded machine."""
    order = sorted(families, key=lambda f: -sum(j["expected_s"] for j in families[f]))
    load, out = [0.0] * machines, [[] for _ in range(machines)]
    for family in order:
        target = min(range(machines), key=lambda m: load[m])
        out[target].append(family)
        load[target] += pass_seconds(families[family], slots)
    return [m for m in out if m]


def timing_plan(jobs: list[dict], series: str, machines: int, price_per_h: float,
                slots: int = SLOTS) -> dict:
    """The run's BP-4 timing plan (each job carries `expected_s`)."""
    families = families_of(jobs)
    slowdown = HALF_LOAD_SLOWDOWN[series]
    plan_machines = []
    for index, names in enumerate(assign(families, machines, slots)):
        wall = SETUP_S + slowdown * sum(pass_seconds(families[f], slots) for f in names)
        plan_machines.append({"index": index, "families": names, "expected_s": round(wall)})
    wall = max(m["expected_s"] for m in plan_machines)
    hours = sum(m["expected_s"] for m in plan_machines) / 3600
    return {"series": series, "instance_type": machine_type(series), "slots": slots,
            "half_load_slowdown": slowdown, "machines": plan_machines,
            "expected_wall_s": wall, "expected_machine_hours": round(hours, 2),
            "price_per_h": price_per_h, "expected_cost_usd": round(hours * price_per_h, 2)}


def check_jobs(jobs: list[dict]) -> dict[str, list[dict]]:
    """V6's check set: CHECK_FAMILIES on the run's first CHECK_CLIPS clips at CHECK_RUNG,
    planned like every job (the record's commands), {family: jobs}."""
    import plan
    clips, seen = [], set()
    for job in sorted(jobs, key=lambda j: j["input"]):
        if job["input"] not in seen:
            seen.add(job["input"])
            clips.append({"name": job["input"], "width": job["w"], "height": job["h"],
                          "fps": job["fps"], "source_kbps": float("inf")})
    frames = {j["input"]: j["frames"] for j in jobs}
    out: dict[str, list[dict]] = defaultdict(list)
    for clip in clips[:CHECK_CLIPS]:
        for job in plan.plan([clip], list(CHECK_FAMILIES), "cbr", [CHECK_RUNG],
                             frames[clip["name"]], plan.KEY_INTERVAL):
            if not job["anchor_extra"]:
                out[job["family"]].append({**job, "expected_s": 1.0})
    return dict(out)


def settled_families(records: list[dict]) -> set[str]:
    """The families a machine's records show settled: each `family_done` marker after the
    machine's complete start check set (`check_done` start; D13, V24)."""
    started, settled = False, set()
    for r in records:
        if r.get("check_done") == "start":
            started = True
        elif started and "family_done" in r:
            settled.add(r["family_done"])
    return settled


def check_medians(records: list[dict]) -> dict[tuple[str, str, str], float]:
    """{(machine, check point, check job): the median of its rounds' CPU times}."""
    times: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in records:
        if r.get("check") and "cpu_s" in r and "stream_sha256" in r:
            times[r.get("box", ""), r["check"], r["id"].split("|", 1)[1]].append(r["cpu_s"])
    return {k: statistics.median(v) for k, v in times.items()}


def check_set_levels(records: list[dict]) -> dict[tuple[str, str], float]:
    """{(machine, check point): the check set's level}: the geometric mean over its jobs of the
    median time there over the job's median across the series' machines and check points."""
    medians = check_medians(records)
    by_job: dict[str, list[float]] = defaultdict(list)
    for (_, _, job), t in medians.items():
        by_job[job].append(t)
    centre = {job: statistics.median(ts) for job, ts in by_job.items()}
    logs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (machine, where, job), t in medians.items():
        logs[machine, where].append(math.log(t / centre[job]))
    return {k: math.exp(statistics.fmean(v)) for k, v in logs.items()}


def start_set_ratio(machine: str, records: list[dict]) -> float | None:
    """The machine's start check set level among the start sets in `records` (the series'
    machines so far, as `check_set_levels`, the report's measure); None while it has no start
    set or no other machine has one (the first machine is judged by the report)."""
    starts = [r for r in records if r.get("check") == "start"]
    if not {r.get("box", "") for r in starts} - {machine}:
        return None
    return check_set_levels(starts).get((machine, "start"))


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line cut by a checkpoint in progress
            continue
    return out


class Assignments:
    """The families still to settle, per machine slot; a lost machine's go to its replacement."""

    def __init__(self, plan: dict) -> None:
        self.lock = threading.Lock()
        self.todo = {m["index"]: list(m["families"]) for m in plan["machines"]}

    def remaining(self, index: int) -> list[str]:
        with self.lock:
            return list(self.todo[index])

    def settle(self, index: int, families: set[str]) -> int:
        with self.lock:
            before = len(self.todo[index])
            self.todo[index] = [f for f in self.todo[index] if f not in families]
            return before - len(self.todo[index])


#: A region or zone priced above this share of the bid is tried only after the cheaper ones.
SAFE_PRICE_SHARE = 0.8


def capacity_regions(price_of, instance_type: str) -> list[str]:
    """The regions a launch round tries: home, then spill regions within SPILL_PRICE_FACTOR of
    home's price (`price_of(region, type)`: the cheapest spot price there, or None)."""
    home = price_of(HOME_REGION, instance_type)
    priced = []
    for region in CAPACITY_REGIONS:
        price = price_of(region, instance_type)
        if home and price and price <= max(SPILL_PRICE_FACTOR * home, BENCH_SPOT_CAP_USD):
            priced.append((price, region))
    # cheapest first, and regions priced near the bid last: a machine there is reclaimed as
    # soon as the price passes the cap (2026-11-26: two eu-west-1c machines at $1.917 of $2.00)
    priced.sort(key=lambda pr: (pr[0] > SAFE_PRICE_SHARE * BENCH_SPOT_CAP_USD, pr[0]))
    return [HOME_REGION, *(region for _, region in priced)]


def open_session(spec_for, regions: list[str], stop: threading.Event, clients,
                 clock=time.monotonic, sleep=None):
    """A Session on the first region with capacity, retried every CAPACITY_RETRY_S for up to
    CAPACITY_WAIT_S (D14); raises NoCapacity after that. `spec_for(region)` gives the spec.
    A region whose price is over the session budget is left out of the search: a price does
    not fall within the minutes of the search, and its BudgetExceeded once ended the whole
    search at once, counted as a lost machine (2026-11-25, eu-central-1 at $1.03 against $1.00:
    three refusals in 2 min, the run given up and relaunched by hand)."""
    from aws_session import BudgetExceeded, NoCapacity, Session
    sleep = sleep or (lambda s: stop.wait(s))
    deadline = clock() + CAPACITY_WAIT_S
    regions = list(regions)
    while True:
        for region in list(regions):
            session = Session(spec_for(region), stop, clients)
            try:
                return session.__enter__()
            except NoCapacity as why:
                print(f"{region}: {why}")
            except BudgetExceeded as why:
                print(f"{region}: {why}: left out of the search")
                regions.remove(region)
        if not regions:
            raise BudgetExceeded("every capacity region is over the session budget")
        if clock() + CAPACITY_RETRY_S > deadline:
            raise NoCapacity(f"no {spec_for(regions[0]).instance_type} capacity in "
                             f"{', '.join(regions)} for {CAPACITY_WAIT_S // 60} min")
        if sleep(CAPACITY_RETRY_S):
            raise NoCapacity("stopped while waiting for capacity")
