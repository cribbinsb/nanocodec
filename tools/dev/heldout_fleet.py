#!/usr/bin/env python3
"""A chart's measure on rented machines as many small sessions: plan, run, relaunch, report.

    python3 tools/dev/heldout_fleet.py measure --state FILE [--boxes 5] [--lane-minutes 10]
        [--first NAMES|@FILE] [--budget 3] [--dry-run] -- <aws_heldout_chart.py options>
    python3 tools/dev/heldout_fleet.py terminate --state FILE --lane K

`measure` splits the clips that still have work into lanes (`plan_lanes`: each about
--lane-minutes of box time, from the rates measured on 0.6.0's run, `ENCODES_PER_MINUTE`,
`POINTS_PER_MINUTE`), the clips other consumers wait on first (--first, default the
history page's slice: tools/dev/history_chart.py). Each lane is `aws_heldout_chart.py run
--clip-list` then `score-cloud --clip-list`, each its own small session (c7i.24xlarge:
up in seconds), and --boxes lanes run side by side. A session that never launched (exit
aws_heldout_chart.EXIT_NO_LAUNCH: no room within a minute, or nothing within budget) is
launched again after LAUNCH_RETRY_S; a session that ends any other way (a spot reclaim,
a stall, an error) is relaunched at once on its lane's clips, and since finished encodes
and results come back as they finish (aws_session.Session.pulling) and both tools skip
what is done, only that lane's unfinished work is redone. Every STATUS_SECONDS one line
says where the fleet is (lanes done / running / queued, encodes and points done, ETA,
any lane waiting for room or relaunched), and --state keeps each lane's session (its
instance and log) for `terminate`, which ends a lane's instance by hand (the loss test).

0.6.0 (2026-11-21) measured one 209-clip held-out set in one encode and two scoring
sessions: a spot reclaim took one scoring session and every point it held, and four
`--session k/8`-style sessions then finished in 7-10 minutes each; this is that shape
by default, with the loss contained.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aws_heldout_chart as heldout  # noqa: E402
from aws_session import say  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TOOL = Path(__file__).with_name("aws_heldout_chart.py")
#: The clips the history page re-measures each day (history_chart.py's slice): they go
#: first, so its frame is not held up behind a whole chart.
HISTORY_DATA = REPO / "web" / "results" / "data" / "history.json"
#: Box throughput in 720p 150-frame points per box-minute on a c7a.24xlarge (96 vCPU),
#: from 0.6.0's held-out run (2026-11-21, 209 clips averaging 0.64 of a 720p point):
#: 39,468 encodes in an 18.8-minute encode stage (1,340); four scoring sessions of
#: 26-27 clips each brought back 1,039-1,452 results in 3.5-5.9 min of scoring
#: (~240 a minute, ~150 at 720p; tail-bound, one 8-worker shard a clip). Each session
#: adds its boot, upload (0.7 min), readiness (1.4), pipe-cleaner (0.4) and teardown.
ENCODES_PER_MINUTE = 1300.0
POINTS_PER_MINUTE = 150.0
SESSION_OVERHEAD_MINUTES = 3.5
REFERENCE_PIXELS = 1280 * 720 * heldout.DEFAULT_FRAMES
#: The default lane size (box-minutes of work, without the sessions' overhead) and lanes side by
#: side: two held-out chains at once stay within eu-north-1's 1,200 spot vCPU (less the
#: build box's 96) at five 96-vCPU boxes each.
LANE_MINUTES = 10.0
BOXES = 5
INSTANCE = "c7i.24xlarge"  # docs/aws-cost-guide.md
#: Each session's spending cap, and its hard lifetime: a margin plus this many times its
#: lane's estimated work (a lane is larger than --lane-minutes when the boxes are few);
#: a scoring session starts no point after its lifetime less SCORE_STOP_MARGIN.
SESSION_BUDGET = 3.0
LIFETIME_BASE_MINUTES = 25
LIFETIME_PER_WORK = 1.5
SCORE_STOP_MARGIN = 10
#: A session waits this long for spot room before it exits unlaunched; its lane then
#: tries again after LAUNCH_RETRY_S, at most LAUNCH_TRIES times (then fails loudly).
LAUNCH_WAIT_MINUTES = 1.0
LAUNCH_RETRY_S = 30
LAUNCH_TRIES = 20
#: A lane's session may be lost and relaunched this many times per step.
RELAUNCHES = 3
STATUS_SECONDS = 60
STEPS = ("run", "score-cloud")
#: The line of a session's log that names its instance (aws_session.Session.await_running).
LAUNCHED = re.compile(r"launched (i-[0-9a-f]+) \((\S+)\) in (\S+) ")


@dataclass
class ClipWork:
    """One clip's remaining work: its encodes' done markers, its points' result files,
    and the box-minutes they take."""
    name: str
    encodes: list[Path]
    points: list[Path]
    minutes: float


@dataclass
class Lane:
    index: int
    clips: list[str]
    #: Estimated box-minutes of work (ClipWork.minutes summed).
    minutes: float = 0.0
    #: "queued", "run", "score-cloud", "waiting", "done" or "failed: ..."
    status: str = "queued"
    log: Path | None = None
    instance: str | None = None
    region: str | None = None
    relaunched: int = 0
    history: list[str] = field(default_factory=list)


def weight(point) -> float:
    clip = point.clip
    return clip.width * clip.height * min(heldout.DEFAULT_FRAMES, clip.frames) / REFERENCE_PIXELS


def clip_work(options: list[str]) -> list[ClipWork]:
    """Every clip's remaining encodes (aws_heldout_chart.encode_todo) and points (the
    ladder, the extras and the predicted low rungs not yet scored), in clip-set order."""
    args = heldout.main_parser().parse_args(["run", *options])
    run = heldout.setup(args)
    streams = Path(args.streams)
    encodes = heldout.encode_todo(run, args.rungs, streams)
    scored = [j for j in run.ladder(args.rungs) + run.extras()
              + run.low_rungs(run.predicted_low()) if not j[0].cached(j[1])]
    works = {c.name: ClipWork(c.name, [], [], 0.0) for c in run.clips}
    for bench, point in encodes:
        work = works[point.clip.name]
        work.encodes.append(Path(f"{streams / bench.prebuilt_key(point)}.done"))
        work.minutes += weight(point) / ENCODES_PER_MINUTE
    for bench, point in scored:
        work = works[point.clip.name]
        work.points.append(bench.result_path(point))
        work.minutes += weight(point) / POINTS_PER_MINUTE
    return [w for w in works.values() if w.encodes or w.points]


def plan_lanes(minutes: dict[str, float], first: list[str], lane_minutes: float,
               boxes: int) -> list[list[str]]:
    """The clips with work in lanes, each about `lane_minutes` of box work (its sessions'
    own boot and upload on top), at most `boxes` of them per group so that all run in
    one wave: more lanes than boxes would add sessions, not speed, now that a lost
    session costs only the minutes since its last pull. The `first` clips get lanes of
    their own, which start first; the rest's lanes that start behind them on the same
    boxes take that much less work (longest clip first onto the lane that ends soonest).
    """
    def pack(names: list[str], starts: list[float]) -> list[list[str]]:
        lanes: list[list[str]] = [[] for _ in starts]
        ends = list(starts)
        for name in sorted(names, key=lambda n: (-minutes[n], n)):
            soonest = ends.index(min(ends))
            lanes[soonest].append(name)
            ends[soonest] += minutes[name]
        return [lane for lane in lanes if lane]

    def count(names: list[str]) -> int:
        return min(len(names), boxes, max(1, math.ceil(sum(minutes[n] for n in names)
                                                         / lane_minutes)))
    wanted = set(first)
    early = [n for n in minutes if n in wanted]
    rest = [n for n in minutes if n not in wanted]
    early_lanes = pack(early, [0.0] * count(early)) if early else []
    # the executor gives the rest's lanes the free boxes first, then the first lanes'
    # boxes as those finish: the last of the rest's lanes start that much later
    behind = [sum(minutes[n] for n in lane) + 2 * SESSION_OVERHEAD_MINUTES
              for lane in early_lanes]
    rest_count = max(count(rest), len(behind)) if rest else 0
    starts = [0.0] * (rest_count - len(behind)) + behind[:rest_count]
    return early_lanes + (pack(rest, starts) if rest else [])


def expected_minutes(works: list[ClipWork], lanes: list[list[str]], boxes: int) -> float:
    """The wall time of `lanes` on `boxes` (in submission order, each lane its work plus
    two sessions' overhead)."""
    by_name = {w.name: w.minutes for w in works}
    ends = [0.0] * min(boxes, len(lanes))
    for lane in lanes:
        soonest = ends.index(min(ends))
        ends[soonest] += sum(by_name[n] for n in lane) + 2 * SESSION_OVERHEAD_MINUTES
    return max(ends, default=0.0)


def history_slice() -> list[str]:
    """The history page's slice clips (history_chart.py), none if it has no data yet."""
    if not HISTORY_DATA.exists():
        return []
    return [c["name"] for c in json.loads(HISTORY_DATA.read_text())["slice"]["clips"]]


class Fleet:
    """The lanes of one measure, run side by side, each session relaunched as needed."""

    def __init__(self, args: argparse.Namespace, works: list[ClipWork], lanes: list[list[str]]):
        self.args, self.works = args, {w.name: w for w in works}
        self.lanes = [Lane(i, names, sum(self.works[n].minutes for n in names))
                      for i, names in enumerate(lanes, 1)]
        self.state = Path(args.state)
        self.logs = self.state.parent / f"{self.state.stem}-logs"
        self.lock = threading.Lock()
        self.started = time.monotonic()

    def event(self, line: str) -> None:
        say(f"[fleet] {line}")

    def command(self, lane: Lane, step: str) -> list[str]:
        clips = self.logs / f"lane-{lane.index}.clips"
        clips.write_text("".join(f"{n}\n" for n in lane.clips))
        purpose = f"{self.args.purpose}-{'encode' if step == 'run' else 'score'}"
        lifetime = round(LIFETIME_BASE_MINUTES + LIFETIME_PER_WORK * lane.minutes)
        extra = []
        if step == "score-cloud":
            extra += ["--minutes", str(lifetime - SCORE_STOP_MARGIN)]
            extra += ["--cloud-results", self.args.cloud_results] if self.args.cloud_results else []
        return [sys.executable, str(TOOL), step, *self.args.options, *extra,
                "--clip-list", f"@{clips}",
                "--instance-type", self.args.instance_type, "--budget", str(self.args.budget),
                "--lifetime", str(lifetime), "--launch-wait", str(LAUNCH_WAIT_MINUTES),
                "--purpose", purpose]

    def run_lane(self, lane: Lane) -> None:
        for step in STEPS:
            launches = relaunches = 0
            while True:
                attempt = launches + relaunches + 1
                with self.lock:
                    lane.status, lane.instance, lane.region = step, None, None
                    lane.log = self.logs / f"lane-{lane.index}-{step}-{attempt}.log"
                with lane.log.open("w") as out:
                    status = subprocess.run(self.command(lane, step), cwd=REPO, stdout=out,
                                            stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL).returncode
                if status == 0:
                    break
                if status == heldout.EXIT_NO_LAUNCH:
                    launches += 1
                    if launches >= LAUNCH_TRIES:
                        raise RuntimeError(f"lane {lane.index} {step}: no launch in "
                                           f"{launches} tries; see {lane.log}")
                    with self.lock:
                        lane.status = "waiting"
                    self.event(f"lane {lane.index} {step}: no room to launch; again in "
                               f"{LAUNCH_RETRY_S} s ({lane.log.name})")
                    time.sleep(LAUNCH_RETRY_S)
                    continue
                relaunches += 1
                with self.lock:
                    lane.relaunched += 1
                    lane.history.append(f"{step} lost (exit {status}, {lane.instance})")
                if relaunches > RELAUNCHES:
                    raise RuntimeError(f"lane {lane.index} {step}: lost {relaunches} times; "
                                       f"see {lane.log}")
                self.event(f"lane {lane.index} {step}: session ended (exit {status}, "
                           f"{lane.instance or 'no instance'}): relaunching its "
                           f"{len(lane.clips)} clips' remaining work ({lane.log.name})")

    def lane_task(self, lane: Lane) -> None:
        """A lane to its end, its outcome in its status at once (not when its turn to be
        waited for comes)."""
        try:
            self.run_lane(lane)
            outcome = "done"
        except Exception as error:  # the fleet goes on; it fails at the end, naming it
            outcome = f"failed: {error}"
        with self.lock:
            lane.status = outcome
        self.event(f"lane {lane.index}: {outcome}")

    def measure(self) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        stop = threading.Event()
        reporter = threading.Thread(target=self.report_every, args=(stop,), daemon=True)
        reporter.start()
        try:
            with ThreadPoolExecutor(self.args.boxes) as pool:
                for future in [pool.submit(self.lane_task, lane) for lane in self.lanes]:
                    future.result()
        finally:
            stop.set()
            reporter.join()
            self.report()
        failed = [lane for lane in self.lanes if lane.status != "done"]
        if failed:
            raise SystemExit("fleet: lanes failed: " + "; ".join(
                f"{lane.index} ({lane.status})" for lane in failed))

    def report_every(self, stop: threading.Event) -> None:
        while not stop.wait(STATUS_SECONDS):
            self.report()

    def report(self) -> None:
        """One status line, and the state file (each lane's session, for `terminate`)."""
        with self.lock:
            for lane in self.lanes:
                if lane.log and not lane.instance and lane.log.exists():
                    found = LAUNCHED.search(lane.log.read_text(errors="replace"))
                    if found:
                        lane.instance, lane.region = found[1], found[3][:-1]
            lanes = list(self.lanes)
        encodes = [p for w in self.works.values() for p in w.encodes]
        points = [p for w in self.works.values() for p in w.points]
        done_encodes = sum(p.exists() for p in encodes)
        done_points = sum(p.exists() for p in points)
        minutes = (time.monotonic() - self.started) / 60
        done_share = ((done_encodes / ENCODES_PER_MINUTE + done_points / POINTS_PER_MINUTE)
                      / max(1e-9, len(encodes) / ENCODES_PER_MINUTE
                            + len(points) / POINTS_PER_MINUTE))
        eta = f"{minutes * (1 - done_share) / done_share:.0f} min" if done_share > 0 else "?"
        counts = {s: sum(lane.status == s for lane in lanes)
                  for s in ("done", "queued", "waiting", *STEPS)}
        running = ", ".join(f"{lane.index}:{'enc' if lane.status == 'run' else 'score'}"
                            for lane in lanes if lane.status in STEPS)
        notes = [f"lane {lane.index} waiting for room" for lane in lanes if lane.status == "waiting"]
        notes += [f"lane {lane.index} relaunched {lane.relaunched}x" for lane in lanes
                  if lane.relaunched]
        notes += [f"lane {lane.index} {lane.status}" for lane in lanes
                  if lane.status.startswith("failed")]
        say(f"[fleet] {minutes:.0f} min: lanes {counts['done']}/{len(lanes)} done, "
            f"running [{running}], {counts['queued']} queued; encodes {done_encodes}/"
            f"{len(encodes)}, points {done_points}/{len(points)}; ETA {eta}"
            + (f"; {'; '.join(notes)}" if notes else ""))
        self.state.write_text(json.dumps({"lanes": [
            {"lane": lane.index, "status": lane.status, "clips": lane.clips,
             "log": str(lane.log) if lane.log else None, "instance": lane.instance,
             "region": lane.region, "history": lane.history} for lane in lanes]},
            indent=1) + "\n")


def measure(args: argparse.Namespace) -> None:
    # the scoring sessions run this repository's HEAD: refuse once here, before any lane
    # (each session would refuse alone, and its lane read that as a lost session)
    if subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "eval", "tools"],
                      capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit("commit eval/ and tools/ first: the remote runs this repository's HEAD")
    works = clip_work(args.options)
    if not works:
        say("[fleet] every encode and point is done: no session")
        return
    names = {w.name for w in works}
    first = heldout.clip_names(args.first) if args.first else history_slice()
    first = [n for n in first if n in names]
    lanes = plan_lanes({w.name: w.minutes for w in works}, first, args.lane_minutes,
                       args.boxes)
    total = sum(w.minutes for w in works)
    say(f"[fleet] {sum(len(w.encodes) for w in works)} encodes and "
        f"{sum(len(w.points) for w in works)} points on {len(works)} clips, ~{total:.0f} "
        f"box-min of work: {len(lanes)} lanes ({len(first)} clips first), {args.boxes} "
        f"side by side, ~{expected_minutes(works, lanes, args.boxes):.0f} min expected")
    for index, lane in enumerate(lanes, 1):
        load = sum(w.minutes for w in works if w.name in lane)
        say(f"[fleet]   lane {index}: {len(lane)} clips, ~{load:.1f} box-min")
    if args.dry_run:
        return
    Fleet(args, works, lanes).measure()


def terminate(args: argparse.Namespace) -> None:
    """End one lane's running instance by hand (the loss test): only an instance of ours
    (tagged Project=nanocodec); its session sees it gone and the fleet relaunches it."""
    import boto3

    from aws_session import PROJECT_TAG

    lane = next(entry for entry in json.loads(Path(args.state).read_text())["lanes"]
                if entry["lane"] == args.lane)
    if not lane["instance"]:
        raise SystemExit(f"lane {args.lane} has no instance now ({lane['status']})")
    ec2 = boto3.client("ec2", region_name=lane["region"])
    found = [i for r in ec2.describe_instances(InstanceIds=[lane["instance"]])["Reservations"]
             for i in r["Instances"]]
    tags = {t["Key"]: t["Value"] for t in (found[0].get("Tags", []) if found else [])}
    if tags.get("Project") != PROJECT_TAG:
        raise SystemExit(f"{lane['instance']} is not tagged Project={PROJECT_TAG}: left alone")
    ec2.terminate_instances(InstanceIds=[lane["instance"]])
    say(f"terminated {lane['instance']} (lane {args.lane}, {lane['status']})")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--state", required=True, help="the fleet's state file (its logs beside it)")
    m.add_argument("--boxes", type=int, default=BOXES, help="lanes side by side")
    m.add_argument("--lane-minutes", type=float, default=LANE_MINUTES,
                   help="box-minutes of work per lane")
    m.add_argument("--first", help="clips that go first: names or @FILE (default: the "
                   "history page's slice)")
    m.add_argument("--budget", type=float, default=SESSION_BUDGET, help="USD cap a session")
    m.add_argument("--instance-type", default=INSTANCE)
    m.add_argument("--purpose", default="heldout-fleet", help="the sessions' Purpose prefix")
    m.add_argument("--cloud-results", help="score-cloud's --cloud-results (a scratch store)")
    m.add_argument("--dry-run", action="store_true", help="the plan only")
    m.add_argument("options", nargs=argparse.REMAINDER,
                   help="-- then aws_heldout_chart.py's clip-set options (both steps')")
    t = sub.add_parser("terminate")
    t.add_argument("--state", required=True)
    t.add_argument("--lane", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "measure":
        args.options = args.options[1:] if args.options[:1] == ["--"] else args.options
        measure(args)
    else:
        terminate(args)


if __name__ == "__main__":
    main(sys.argv[1:])
