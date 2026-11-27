#!/usr/bin/env python3
"""Stage a BP-3 run, run its encode and score stages on rented boxes, collect the records.

    python3 tools/bench/fleet.py stage RUN --jobs JOBS.json --bin DIR --cfg DIR --clips DIR
    systemd-run --user --scope -p MemoryMax=2G python3 tools/bench/fleet.py run RUN \
        --series c8i --boxes K [--budget USD] [--fleet-budget USD]

The driver of section 13 (docs/benchmark-procedure.md). `stage` freezes a run directory:
`jobs.json` (from `plan.py`), `manifest.json` (every binary's SHA-256, the record's phase, the
procedure commit) and `stage/` (`bin/`, `cfg/`, the clips as `<name>.yuv.zst` holding exactly
the jobs' frames, `box.py`, `score.py` and the `nceval` modules they import).

`run` (BP-3.11, designed for failure) drives K slots as threads of this one process (which
must run in a memory-capped scope). The jobs wait in one queue (`WorkQueue`); each slot
launches a box (c8i.2xlarge spot by default, its zones taken in turn), which takes small
units of one clip's jobs from its inbox. Every POLL_S the driver checkpoints the box's new
records and streams to the PC and tops up its inbox with about INBOX_SECONDS of work. A job is
settled only when its record, with its stream's SHA-256 (and the stream itself, unless a
re-timing), is on the PC and a timing reference follows it on the same box, so a double run
is harmless. A box that is reclaimed, stalls, gets a spot notice (box.py then stops and codes
its references) or is refused on timing gives its unsettled jobs back to the queue, and its
slot launches a replacement while the fleet budget allows. Any other refusal stops the run.
Each box's pool and outcome go to `boxes.jsonl`; `missing` is the last resort.

`score` (after `run`) shards the stored streams over K spot scoring boxes (many workers each;
not timed), copies each box's scores to the PC every POLL_S, and merges them into the series'
`scores.jsonl`; each score record keeps its encode's series and stream hash (V23).
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "tools" / "dev"))
from nceval import fairness  # noqa: E402
import localscore  # noqa: E402  (beside this file)
from aws_session import (BudgetExceeded, NoCapacity, RemoteFailure, Session,  # noqa: E402
                         SessionSpec, SharedClients, say)

#: The nceval modules the box side imports (standard library, numpy, xxhash).
BOX_MODULES = ("__init__.py", "codecs.py", "streamstrip.py", "display.py", "yuv.py")
BOX_SCRIPTS = ("box.py", "pinning.py", "score.py", "halfload.py", "streamcheck.py")
TMPFS = "/mnt/nanocodec-clips"
#: The timing CPU (section 5); vCPU 0 keeps the system and the harness.
TIMING_CPU = 1
APT = "ffmpeg python3-numpy python3-xxhash util-linux zstd"
PURPOSE = "bench"


def plan_purpose(a) -> str:
    """The boxes' Purpose tag: `bench-<plan>` (RCA R6: each plan's spend is its own ledger
    rows, `aws_session.py ledger --by purpose`); every benchmark purpose starts `bench`, which
    the vCPU cap counts together."""
    return f"{PURPOSE}-{a.plan}" if getattr(a, "plan", None) else PURPOSE
#: BP-3.10: the timing machine, on spot.
TIMING_TYPE, TIMING_REGION = "c8i.2xlarge", "eu-north-1"
#: Untimed boxes (scoring, packed encodes) follow docs/aws-cost-guide.md rows 3 and 4: c7i or
#: c6i .24xlarge spot in eu-north-1, whichever is cheaper per vCPU at launch (the session
#: orders its candidate types that way). c7a is never used (owner: double-pumped AVX-512).
UNTIMED_TYPES = ("c7i.24xlarge", "c6i.24xlarge")
#: vCPUs per box type the fleets use (for the cap and fleet sizing).
BOX_VCPUS = {"c8i.2xlarge": 8, "c8g.2xlarge": 8, "c8g.8xlarge": 32, "c7i.24xlarge": 96,
             "c6i.24xlarge": 96}
#: Scoring runs on Graviton first (docs/aws-cost-guide.md rule 1: never x86 where Arm is
#: cheaper and will do): c8g.8xlarge scored 1,113 1080p streams per box-hour at $0.164/h
#: ($0.147 per 1,000, against $0.22 on c6i.24xlarge) with scores identical to x86 per frame
#: (24 of 24 streams, six codecs; arm-parity.json). x86 (UNTIMED_TYPES) is the fallback when
#: no Graviton capacity is free. Packed encodes stay on x86: the references' quality builds
#: are x86 (BP-3).
ARM_SCORE_TYPES = ("c8g.8xlarge",)
#: A scoring box's session budget (c8g.8xlarge is up to $0.25/h, c7i.24xlarge about $0.6/h).
SCORE_BOX_BUDGET = 1.0
#: The aarch64 decoders and libvmaf 3.2.1 (tools/anchors/build_anchors.sh), and the run's own
#: encoder built for aarch64 (its decoder cross-checks our streams).
ARM_TOOLS = Path.home() / "nanocodec-data" / "anchors" / "aarch64"
ARM_NANOCODEC = Path.home() / "nanocodec-data" / "anchors" / "aarch64-graviton-opt-bp3" / "nanocodec"
ARM_SCORE_TOOLS = ("vmaf", "aomdec", "vpxdec", "avmdec")
VMAF_MODEL_NAME = "vmaf_v1.0.16_3d0h.json"
#: The vCPU all benchmark boxes together may hold in the region (coordinator, 2026-11-24: of the
#: 1200 spot quota, the rest is for the dev screens); a launch waits for room under it.
BENCH_VCPU_CAP = 1000
#: Zones the timing fleets avoid: on 2026-11-24, c8i.2xlarge spot in eu-north-1a was reclaimed
#: 35 times in 7.7 box-hours (4.6 per box-hour) against 0.12 in 1b and 0.17 in 1c, and cost
#: more there ($0.075/h against $0.058-0.061). Scoring does not avoid it: c8g.8xlarge there was
#: reclaimed 1.55 times per box-hour but cost $0.164/h against $0.238-0.249 in 1b/1c, and a
#: replaced scoring box loses only its setup and the streams in progress (about 6 min), so 1a
#: is still the cheapest (docs/analysis/bp3-run-lessons-2026-11-24.md; `preflight` prints both).
AVOID_ZONES = ("eu-north-1a",)
EXCLUDED_FAMILIES = ("c7a",)
#: The account's stop line (owner, 2026-11-26: the AWS cap is $900, raised by $100 from $800;
#: the tools stop at $890):
#: no box is launched, and no preflight passes, whose worst case would take the ledger past it.
ACCOUNT_STOP_USD = 890.0
LEDGER_READ_TRIES = 5


def account_spent() -> float:
    """The ledger's booked total for the whole account (its "Spent so far" line)."""
    from aws_session import LEDGER
    for _ in range(LEDGER_READ_TRIES):  # another driver may be rewriting it this moment
        found = re.search(r"\*\*Spent so far: \$([0-9.]+) of", LEDGER.read_text())
        if found:
            return float(found.group(1))
        time.sleep(1)
    raise SystemExit(f"refused: {LEDGER} has no 'Spent so far' line")


#: The project's disk rule (eval/nceval/diskguard.py): no download below this much free.
MIN_FREE_BYTES = int(os.environ.get("NC_MIN_FREE_GIB", "200")) << 30


def check_free_disk(path: Path) -> None:
    """Refuse a run or scoring pass that would download onto a disk under the project's
    free-space floor (NC_MIN_FREE_GIB, default 200)."""
    free = shutil.disk_usage(path).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"refused: {free >> 30} GiB free under {path}, below the "
                         f"{MIN_FREE_BYTES >> 30} GiB floor (NC_MIN_FREE_GIB)")
#: The last line of a box's log: done, or failed with its reason.
BOX_DONE, BOX_FAILED = "box done", "box failed: "
#: BP-3.11 work units: the driver keeps each box holding about INBOX_SECONDS of expected work
#: not yet coded (a longer unit alone), checkpoints its records every POLL_S, and stops feeding
#: it LIFETIME_MARGIN_MIN before its lifetime ends.
#: A job failed on this many boxes is quarantined: listed, not retried (review #5).
QUARANTINE_AFTER = 2
#: A unit's expected encode seconds (RCA R10); a longer job is a unit of its own.
UNIT_SECONDS = 120
#: The share of a timing box's time spent on jobs (setup, references and canaries are the
#: rest): 77 % on "ours" CBR, 2026-11-24.
BOX_BUSY_SHARE = 0.77
INBOX_SECONDS = 120
POLL_S = 30
LIFETIME_MARGIN_MIN = 10
#: A box whose log (box.py prints a heartbeat every minute) stops growing this long is given
#: up and replaced. Not its records: a 1080p encode can take longer than this (2026-11-24: the
#: AOM-CTC Fixed QP fleets lost boxes in the middle of long encodes on a records rule).
BOX_STALL_MIN = 10
#: The clips a box keeps on its tmpfs (c8i.2xlarge: 16 GiB of memory), least recently used
#: evicted first; the reference clip and clips with work in flight are never evicted.
TMPFS_CLIP_BYTES = 6 << 30
#: Launch attempts per slot (the first box and its replacements), so a systematic launch
#: failure cannot loop.
MAX_ATTEMPTS = 6
CHECKPOINT = ".nanocodec-checkpoint"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


#: The benchmark's gate records (RCA R1): a passed rehearsal and an approved preflight envelope,
#: each for one version of the tools.
GATE_LOG = Path.home() / "nanocodec-data" / "bench" / "gate.jsonl"
#: The files a gate record certifies: everything a run executes or uploads (tools/bench, the
#: nceval modules and the AWS session it imports).
TOOLS_PATHS = ("tools/bench", "tools/dev/aws_session.py", "eval/nceval")


def tools_version() -> str:
    """The version of the tools a gate record names: a hash of TOOLS_PATHS' git trees at HEAD,
    so a commit elsewhere (a doc, a crate, another tool) keeps the rehearsal valid; keyed on
    HEAD, every unrelated landing forced a new metal rehearsal or `--ungated` (8 ungated
    launches by 2026-11-25; docs/analysis/time-audit-2026-11-25.md). Refused when the tools
    differ from HEAD (a rehearsal of uncommitted tools certifies nothing)."""
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *TOOLS_PATHS],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"refused: the tools have uncommitted changes:\n{dirty}")
    trees = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                            *(f"HEAD:{path}" for path in TOOLS_PATHS)],
                           capture_output=True, text=True, check=True).stdout.split()
    return hashlib.sha256(" ".join(trees).encode()).hexdigest()


def record_gate(kind: str, **fields) -> None:
    GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(GATE_LOG, "a") as f:
        f.write(json.dumps({"kind": kind, "tools": tools_version(), "commit": commit(),
                            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(
                                timespec="seconds"), **fields}) + "\n")


def gate_refusal(tools: str, log: Path = GATE_LOG) -> str | None:
    """Why a full launch on tools version `tools` is refused (RCA R1), or None: it needs a
    passed rehearsal and an approved preflight envelope recorded for that same version."""
    records = [json.loads(line) for line in (log.read_text().splitlines() if log.exists()
                                             else []) if line.strip()]
    mine = [r for r in records if r.get("tools") == tools]
    missing = [what for kind, what in (("rehearsal", "a passed rehearsal (`fleet.py rehearse`)"),
                                       ("approval", "an approved preflight envelope "
                                                    "(`fleet.py preflight --approve USD`)"))
               if not any(r["kind"] == kind for r in mine)]
    return f"no {' and no '.join(missing)} for tools {tools[:12]}" if missing else None


def stage(run: Path, jobs_path: Path, bin_dir: Path, cfg_dir: Path, clips: Path) -> None:
    jobs = json.loads(jobs_path.read_text())
    out = run / "stage"
    if out.exists():
        raise SystemExit(f"refused: {out} exists (a staged run is frozen)")
    (out / "nceval").mkdir(parents=True)
    shutil.copytree(bin_dir, out / "bin", symlinks=False)
    shutil.copytree(cfg_dir, out / "cfg", symlinks=False)
    (out / "clips").mkdir()
    for name in sorted({j["input"] for j in jobs}):
        job = next(j for j in jobs if j["input"] == name)
        source = (clips / f"{name}.yuv.zst").resolve()  # zstd skips symbolic links
        frames_bytes = job["w"] * job["h"] * 3 // 2 * job["frames"]
        raw = subprocess.run(["zstd", "-dcq", str(source)], capture_output=True, check=True).stdout
        if len(raw) < frames_bytes:
            raise SystemExit(f"refused: {name} holds fewer than {job['frames']} frames")
        target = out / "clips" / f"{name}.yuv.zst"
        if len(raw) == frames_bytes:  # already exactly the jobs' frames: no copy (disk)
            target.symlink_to(source)
        else:
            subprocess.run(["zstd", "-q", "-o", str(target)], input=raw[:frames_bytes], check=True)
    for module in BOX_MODULES:
        target = out / "nceval" / module
        if module == "__init__.py":
            target.write_text('"""The nceval modules the box side imports (copied by fleet)."""\n')
        else:
            shutil.copy(REPO / "eval" / "nceval" / module, target)
    for script in BOX_SCRIPTS:
        shutil.copy(REPO / "tools" / "bench" / script, out / script)
    shutil.copy(jobs_path, run / "jobs.json")
    shutil.copy(jobs_path, out / "jobs.json")
    manifest = {"procedure_commit": commit(),
                "phase": {k: int(v) if v.isdigit() else v
                          for k, v in fairness.phase_values(fairness.load()).items()},
                "phase_name": fairness.load()["phase"]["current"],
                "binaries": {p.name: sha256(p) for p in sorted((out / "bin").iterdir())
                             if p.is_file()}}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1))
    shutil.copy(run / "manifest.json", out / "manifest.json")
    say(f"staged {len(jobs)} jobs over {len(list((out / 'clips').iterdir()))} clips in {out}")


def prepare_command() -> str:
    """A box's setup: packages, the clips' tmpfs, and any clips staged with it."""
    return (f"sudo apt-get -qq update && sudo DEBIAN_FRONTEND=noninteractive apt-get -qq "
            f"install -y {APT} >/dev/null && sudo mkdir -p {TMPFS} && "
            f"sudo mount -t tmpfs -o size=90% tmpfs {TMPFS} && sudo chmod 777 {TMPFS} && "
            f"for c in bench/clips/*.yuv.zst; do [ -e \"$c\" ] || continue; "
            f"zstd -dq --rm \"$c\" -o {TMPFS}/$(basename \"$c\" .zst); "
            # streams go to the tmpfs too: a metal box's 15 GB root filled with a run's clips
            # and streams and killed its halfload mid-run (2026-11-26)
            f"done && mkdir -p bench/inbox {TMPFS}/out && ln -sfn {TMPFS}/out bench/out && "
            f"lscpu > bench/lscpu.txt && echo ready")


def serve_command(series: str, discard: bool, packed: int, background: bool = False) -> str:
    """box.py taking units from its inbox (BP-3.11); `background`: the packing test's load."""
    return (f"(cd bench && sudo python3 -I box.py --inbox inbox --manifest manifest.json "
            f"{'--background background.json ' if background else ''}"
            f"--cpu {TIMING_CPU} --series {series} --clips {TMPFS} --canaries canaries.json "
            f"--references references.json {'--discard-streams ' if discard else ''}"
            f"{f'--packed {packed} ' if packed else ''}--out out)")


class WorkQueue:
    """BP-3.11: the run's jobs, pending or in flight on a box, until settled (their record,
    with its stream's SHA-256, on the PC, and its time bracketed by references). A lost box's
    unsettled jobs go back to pending. Thread-safe.

    Longest first (RCA R10, the standard for parallel queues): jobs are handed out in
    descending expected duration (`expected_s`, from `expected_seconds`), and units are sized
    in expected seconds, not jobs: a job longer than UNIT_SECONDS is a unit of its own, shorter
    ones fill a unit from one clip (the box's own clips first). A short queue's units shrink so
    every slot gets work. On 2026-11-24 a fill-in's last five 1080p VVenC encodes (up to 14 min
    each) came last, on 2 boxes, and set its end."""

    def __init__(self, jobs: list[dict], slots: int = 1) -> None:
        self.lock, self.slots = threading.Lock(), slots
        self.pending = sorted(jobs, key=lambda j: -j.get("expected_s", 1.0))
        self.by_id = {j["id"]: j for j in jobs}
        self.inflight: dict[str, dict[str, dict]] = {}
        self.settled: set[str] = set()
        self.total = len(jobs)
        self.failures: dict[str, list[str]] = defaultdict(list)  # id -> one reason per box
        self.quarantined: dict[str, str] = {}

    def fail(self, box: str, job_id: str, reason: str) -> None:
        """A job failed on `box` (V9, V21, its timeout): back to pending, or quarantined once
        QUARANTINE_AFTER boxes have failed it (review #5: a poison job must not stop the fleet,
        nor loop for ever)."""
        with self.lock:
            job = self.inflight.get(box, {}).pop(job_id, None)
            if job is None:
                return
            self.failures[job_id].append(f"{box}: {reason}")
            if len(self.failures[job_id]) >= QUARANTINE_AFTER:
                self.quarantined[job_id] = "; ".join(self.failures[job_id])
            else:
                self.pending = sorted(self.pending + [job], key=lambda j: -j.get("expected_s", 1.0))

    def take(self, box: str, resident: set[str]) -> list[dict] | None:
        """The next unit for `box`, or None when nothing is pending."""
        with self.lock:
            if not self.pending:
                return None
            head = self.pending[0]
            if head.get("expected_s", 1.0) >= UNIT_SECONDS:
                unit = [head]
            else:
                left = sum(j.get("expected_s", 1.0) for j in self.pending)
                target = min(UNIT_SECONDS, left / self.slots)
                here = [j for j in self.pending if j["input"] in resident]
                clip = (here[0] if here else head)["input"]
                unit, total = [], 0.0
                for job in self.pending:  # longest first, from one clip
                    if job["input"] != clip:
                        continue
                    unit.append(job)
                    total += job.get("expected_s", 1.0)
                    if total >= target:
                        break
            taken = {j["id"] for j in unit}
            self.pending = [j for j in self.pending if j["id"] not in taken]
            self.inflight.setdefault(box, {}).update((j["id"], j) for j in unit)
            return unit

    def settle(self, box: str, ids: set[str]) -> None:
        with self.lock:
            self.settled |= ids
            for job_id in ids:
                self.inflight.get(box, {}).pop(job_id, None)

    def unsettle(self, ids: set[str]) -> int:
        """Jobs a box settled that its later records refuse (review #4: the end canaries or the
        V6 moving median can refuse a box's earlier times) back to pending; how many."""
        with self.lock:
            back = [self.by_id[i] for i in ids if i in self.settled and i in self.by_id]
            self.settled -= {j["id"] for j in back}
            self.pending = sorted(self.pending + back, key=lambda j: -j.get("expected_s", 1.0))
            return len(back)

    def release(self, box: str) -> int:
        """`box`'s unsettled jobs back to pending, in their longest-first place; how many."""
        with self.lock:
            jobs = [j for j in self.inflight.pop(box, {}).values() if j["id"] not in self.settled]
            self.pending = sorted(self.pending + jobs, key=lambda j: -j.get("expected_s", 1.0))
            return len(jobs)

    def held(self, box: str) -> list[dict]:
        with self.lock:
            return list(self.inflight.get(box, {}).values())

    def pending_count(self) -> int:
        with self.lock:
            return len(self.pending)

    def pending_seconds(self) -> float:
        with self.lock:
            return sum(j.get("expected_s", 1.0) for j in self.pending)

    def finished(self) -> bool:
        with self.lock:
            return len(self.settled) + len(self.quarantined) >= self.total and not self.pending


#: Where measured encode times are kept for the queue's estimates (one line per job id).
DURATIONS = Path.home() / "nanocodec-data" / "bench" / "durations.json"


def measured_durations(bench: Path) -> dict[str, float]:
    """The median measured wall time of every job id coded on a timing box so far (all runs'
    merged records), cached in DURATIONS."""
    walls: dict[str, list[float]] = defaultdict(list)
    # BP-3 runs live at bench/<run>/run/, BP-4 runs at bench/<run>/ (records per machine box
    # until the run's merge); callers pass a run's grandparent, which for BP-4 is bench's parent
    if (bench / "bench").is_dir():
        bench = bench / "bench"
    paths = [*bench.glob("bp3-*/run/series/*/encodes.jsonl"),
             *bench.glob("bp4-*/series/*/box*/encodes.jsonl")]
    for path in paths:
        with open(path) as f:
            for line in f:
                if '"wall_s"' not in line or '"untimed"' in line or '"canary"' in line \
                        or '"reference"' in line or '"warmup"' in line:
                    continue
                r = json.loads(line)
                if "id" in r and "stream_sha256" in r:
                    walls[r["id"]].append(r["wall_s"])
    return {k: sorted(v)[len(v) // 2] for k, v in walls.items()}


#: The window every measured (BP-3) job time was coded at.
MEASURED_FRAMES = 150


def expected_seconds(jobs: list[dict], measured: dict[str, float]) -> None:
    """Each job's `expected_s`: its own measured time, else the family's measured seconds per
    megapixel-frame times the job's frames x size, else the median of all (RCA R10)."""
    per_px: dict[str, list[float]] = defaultdict(list)
    sizes = {j["id"]: j["w"] * j["h"] * j["frames"] for j in jobs if "w" in j}
    for job in jobs:
        if job["id"] in measured and job["id"] in sizes:
            per_px[job["family"]].append(measured[job["id"]] / sizes[job["id"]])
    family_rate = {f: sorted(v)[len(v) // 2] for f, v in per_px.items()}
    fallback = sorted(measured.values())[len(measured) // 2] if measured else 1.0
    for job in jobs:
        if job["id"] in measured:
            # the measured runs coded 150-frame windows (BP-3); a BP-4 long window (D7) codes
            # the same id at 600 frames: time scales with frames
            job["expected_s"] = measured[job["id"]] * max(1.0, job.get("frames", 0) / MEASURED_FRAMES)
        elif job.get("family") in family_rate and job["id"] in sizes:
            job["expected_s"] = family_rate[job["family"]] * sizes[job["id"]]
        else:
            job["expected_s"] = fallback


def fleet_size(jobs: list[dict], target_minutes: float, vcpu_cap: int, box_vcpus: int) -> int:
    """Boxes for a run from its expected seconds, not its job count (RCA R10): enough that the
    work spread longest-first ends in about `target_minutes` (no fewer than the longest job
    allows), within the benchmark's vCPU cap."""
    total = sum(j.get("expected_s", 1.0) for j in jobs) / BOX_BUSY_SHARE
    boxes = int(-(-total // (target_minutes * 60)))
    return max(1, min(boxes, len(jobs), vcpu_cap // box_vcpus))


def box_records(box_dir: Path) -> list[dict]:
    """A box's downloaded records (a line still being written is skipped)."""
    path = box_dir / "encodes.jsonl"
    records = []
    for line in (path.read_text().splitlines() if path.exists() else []):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def settled_ids(box_dir: Path, keep_streams: bool) -> set[str]:
    """The jobs a box's downloaded records settle: valid by the report's own rule, checked as
    the records arrive (BP-3.11: bracketed by references, outside V6-refused windows, not
    V7-slow, the box's canaries in line), with the stream itself on the PC when the run keeps
    streams. A refused job never settles, so it goes back to the queue when its box ends and
    is coded again while the fleet is still up: the report needs no fill-in pass."""
    import report
    records = box_records(box_dir)
    if not records:
        return set()
    try:
        valid, _ = report.timed_records(box_dir.name, records)
    except report.ReportRefusal:  # not judgeable yet (e.g. before its canaries are in)
        return set()
    return {r["id"] for r in valid if not keep_streams or stream_arrived(box_dir, r)}


def stream_arrived(box_dir: Path, rec: dict) -> bool:
    """The record's stream is on the PC whole: its size and SHA-256 equal the record's (review
    #10: a checkpoint caught mid-write must not settle a job). A verified stream is marked by a
    sidecar, so each stream is hashed once."""
    stream = box_dir / rec["stream"]
    mark = box_dir / ".verified" / stream.name  # beside, not among, the streams
    if mark.exists():
        return True
    if not stream.exists() or ("bytes" in rec and stream.stat().st_size != rec["bytes"]):
        return False
    if sha256(stream) != rec["stream_sha256"]:
        return False
    mark.parent.mkdir(exist_ok=True)
    mark.write_text(rec["stream_sha256"])
    return True


class BoxLost(RuntimeError):
    """The box stopped answering: reclaimed, stalled or past its lifetime."""


class Fleet:
    """The shared state of one run's boxes: the queue, the costs and the stop."""

    def __init__(self, a, jobs: list[dict], stop: threading.Event,
                 planned: list[dict] | None = None) -> None:
        self.a, self.queue, self.stop = a, WorkQueue(jobs, slots=a.boxes), stop
        self.planned = planned if planned is not None else jobs  # the canaries' pool
        self.base = a.run / "series" / a.series
        self.clients = SharedClients()
        self.lock = threading.Lock()
        self.spent, self.running = 0.0, 0
        self.failure: str | None = None
        # the rehearsal's forced reclaim: the first box is terminated once it has settled a job
        self.chaos_pending = bool(getattr(a, "chaos", False))
        self.started = datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S")
        # RCA R7: once the allowed zones have no capacity, the avoided zone is used too (a
        # re-timed reclaim there costs less than waiting; 2026-11-24: two slots spent their
        # launch attempts on "no zone could launch")
        self.zone_fallback = False
        ec2 = self.clients("ec2", a.region)
        self.zones = sorted(z["ZoneName"] for z in ec2.describe_availability_zones(
            Filters=[{"Name": "state", "Values": ["available"]}])["AvailabilityZones"]
            if z["ZoneName"] not in AVOID_ZONES)

    def may_launch(self) -> bool:
        """Within the fleet's budget and under the account's stop line, counting each running
        box at its own cap (the ledger books a box when it ends)."""
        with self.lock:
            if self.spent + (self.running + 1) * self.a.budget > self.a.fleet_budget:
                return False
            if account_spent() + (self.running + 1) * self.a.budget > ACCOUNT_STOP_USD:
                print(f"the account's stop line ${ACCOUNT_STOP_USD:.0f} would be passed: no "
                      "more boxes")
                return False
            self.running += 1
            return True

    def ended(self, cost: float) -> None:
        with self.lock:
            self.running -= 1
            self.spent += cost

    def record(self, row: dict) -> None:
        """The pool and outcome of each box (BP-3.11 rule 4), for the audit."""
        with self.lock, open(self.base / "boxes.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


def put_clip(s: Session, run: Path, name: str) -> None:
    """A clip onto the box's tmpfs, streamed and decompressed there (written whole, then
    renamed, so box.py never sees a partial clip)."""
    with open((run / "stage" / "clips" / f"{name}.yuv.zst").resolve(), "rb") as f:
        s.ssh(f"zstd -dqf -o {TMPFS}/{name}.yuv.part && mv {TMPFS}/{name}.yuv.part "
              f"{TMPFS}/{name}.yuv", stdin=f)


def clip_bytes(job: dict) -> int:
    return job["w"] * job["h"] * 3 // 2 * job["frames"]


class BoxFeed:
    """What one box holds: its clips (least recently used first) and units sent."""

    def __init__(self, fleet: Fleet, s: Session, name: str, pinned: str) -> None:
        self.fleet, self.s, self.name, self.pinned = fleet, s, name, pinned
        self.clips: dict[str, int] = {}  # name -> bytes, in use order
        self.units, self.ended = 0, False

    def ensure_clip(self, job: dict) -> None:
        name = job["input"]
        if name in self.clips:
            self.clips[name] = self.clips.pop(name)  # most recently used
            return
        busy = {j["input"] for j in self.fleet.queue.held(self.name)} | {self.pinned}
        while sum(self.clips.values()) + clip_bytes(job) > TMPFS_CLIP_BYTES:
            victim = next((c for c in self.clips if c not in busy), None)
            if victim is None:
                break
            self.s.ssh(f"rm -f {TMPFS}/{victim}.yuv")
            del self.clips[victim]
        put_clip(self.s, self.fleet.a.run, name)
        self.clips[name] = clip_bytes(job)

    def send(self, unit: list[dict]) -> None:
        self.ensure_clip(unit[0])
        self.write(f"bench/inbox/unit-{self.units:05d}.json", json.dumps(unit))
        self.units += 1

    def write(self, path: str, text: str) -> None:
        """A file on the box, written whole then renamed (box.py never reads a partial one)."""
        subprocess.run([*self.s.ssh_base(), f"cat > {path}.part && mv {path}.part {path}"],
                       input=text.encode(), check=True)

    def top_up(self, recorded: set[str]) -> None:
        """Keep the box holding about INBOX_SECONDS of expected work not yet coded (one long
        unit alone: a deep inbox of long jobs would leave other boxes idle at the end); END
        when the queue is empty or the box nears its lifetime. Coded jobs still waiting for
        their next references are not work the box holds (2026-11-24: counting them starved
        boxes whose units were shorter than a reference interval)."""
        if self.ended:
            return
        near_end = utc_minutes_since(self.s.launched) > \
            self.fleet.a.lifetime - LIFETIME_MARGIN_MIN

        def uncoded_s() -> float:
            return sum(j.get("expected_s", 1.0) for j in self.fleet.queue.held(self.name)
                       if j["id"] not in recorded)

        while not near_end and uncoded_s() < INBOX_SECONDS:
            unit = self.fleet.queue.take(self.name, set(self.clips))
            if unit is None:
                break
            self.send(unit)
        if near_end or self.fleet.queue.pending_count() == 0:
            self.write("bench/inbox/END", "")
            self.ended = True


def utc_minutes_since(moment) -> float:
    import datetime
    return (datetime.datetime.now(datetime.timezone.utc) - moment).total_seconds() / 60


def recorded_ids(box_dir: Path) -> set[str]:
    """The jobs a box has coded so far (settled or still waiting for references)."""
    path = box_dir / "encodes.jsonl"
    ids = set()
    for line in (path.read_text().splitlines() if path.exists() else []):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "stream_sha256" in r and "canary" not in r and "reference" not in r:
            ids.add(r["id"])
    return ids


def loaded_slot(name: str, share: float) -> bool:
    """The packing test (owner, 2026-11-24): every round(1/share)-th slot's boxes carry
    background load on their idle cores; none at share 0."""
    if share <= 0:
        return False
    slot_index = int(name.removeprefix("box").split("-")[0])
    return slot_index % max(1, round(1 / share)) == 0


#: Canaries are short jobs (each is coded 3 times at 3 points of a box's run): the ones whose
#: expected time is nearest CANARY_TARGET_S, never the queue's head (review #1: with longest
#: first, the head is the longest job, and its canaries would take a 14-minute VVenC encode 9
#: times, about 2 hours of a 90-minute box).
CANARY_TARGET_S = 3.0


#: A canary is coded 9 times per box; a planned job longer than this is never one (2026-11-24:
#: a 6-job fill-in had only 1080p x265 jobs of ~8 min, and its boxes spent their lifetime on
#: canaries). With none short enough, the box's timing references serve as its canaries.
CANARY_MAX_S = 20.0


def canary_jobs(planned: list[dict], clip: str, count: int,
                fallback: list[dict] = ()) -> list[dict]:
    """`count` short jobs of the box's first clip (on its tmpfs already) for V6's canaries, or
    `fallback` (the box's references) when no planned job is short enough."""
    pool = [j for j in planned if j["input"] == clip] or planned
    short = [j for j in pool if j.get("expected_s", CANARY_TARGET_S) <= CANARY_MAX_S]
    if not short:
        return list(fallback)[:count]
    return sorted(short, key=lambda j: abs(j.get("expected_s", CANARY_TARGET_S)
                                           - CANARY_TARGET_S))[:count]


def image_arch(instance_type: str) -> str:
    """The AMI architecture of an instance type: Graviton families carry a "g" after their
    generation (c8g, c7gn), others are x86."""
    family = instance_type.split(".")[0]
    return "arm64" if family[2:].startswith("g") else "amd64"


def serve_box(fleet: Fleet, name: str, zone: str) -> str:
    """One box from launch to end, fed from the queue: "done", "interrupted" (a spot notice:
    it stopped and referenced), or raises (BoxLost, a refusal). Its records are downloaded
    every POLL_S, so a loss costs only the jobs it had not yet settled."""
    a, stop = fleet.a, fleet.stop
    spec = SessionSpec(region=a.region, instance_type=a.types[0], fallback_types=a.types[1:],
                       purpose=plan_purpose(a),
                       lifetime_minutes=a.lifetime, budget_usd=a.budget,
                       on_demand=a.on_demand and not a.packed, root_gb=a.root_gb, spill=False,
                       preferred_zone=zone, launch_wait_minutes=10, arch=image_arch(a.types[0]),
                       purpose_vcpu_cap=a.vcpu_cap, cap_purpose=PURPOSE,
                       excluded_zones=() if fleet.zone_fallback else AVOID_ZONES,
                       tags={"Owner": "bench", "Series": a.series})
    local = fleet.base / name
    keep = not a.discard_streams
    first = fleet.queue.take(name, set())
    if first is None:
        return "done"
    row: dict = {"box": name, "type": a.types[0], "preferred_zone": zone}
    session: list[Session] = []
    try:
        with Session(spec, stop, fleet.clients) as s:
            session.append(s)
            row.update(zone=s.zone, instance=s.instance_id, price=s.price)
            s.upload(box_stage(a.run, "queue"), "bench")
            s.run_detached(prepare_command(), "bench/prepare.log", 15, max_errors=0)
            feed = BoxFeed(fleet, s, name, first[0]["input"])
            refs = references(first, a.run / "stage" / "bin")
            feed.write("bench/canaries.json", json.dumps(canary_jobs(
                fleet.planned, first[0]["input"], a.canaries, refs)))
            feed.write("bench/references.json", json.dumps(refs))
            background = loaded_slot(name, a.background_share)
            row["background"] = background
            if background:
                feed.write("bench/background.json", json.dumps(first))
            feed.send(first)
            s.start_detached(serve_command(a.series, a.discard_streams, a.packed, background),
                             "bench/encode.log")
            outcome = watch_box(fleet, s, feed, name, local, keep)
            row["outcome"] = outcome
            return outcome
    except BaseException as error:
        row["outcome"] = f"lost: {str(error).strip().splitlines()[0] if str(error) else error!r}"
        raise
    finally:
        s = session[0] if session else None
        cost = (s.price or 0.0) * utc_minutes_since(s.launched) / 60 \
            if s is not None and s.launched else 0.0  # the ledger books the exact cost
        final = settled_ids(local, keep)
        fleet.queue.settle(name, final)
        coded = {r["id"] for r in box_records(local) if "stream_sha256" in r
                 and "canary" not in r and "reference" not in r}
        row["unsettled"] = fleet.queue.unsettle(coded - final)
        row["released"] = fleet.queue.release(name)
        row["cost_estimate"] = round(cost, 4)
        fleet.record(row)
        fleet.ended(cost)


def watch_box(fleet: Fleet, s: Session, feed: BoxFeed, name: str, local: Path,
              keep: bool) -> str:
    """Poll the box every POLL_S: checkpoint its new records, settle them, keep it fed."""
    last_growth, size = utc_minutes_since(s.launched), ""
    while True:
        if fleet.stop.wait(POLL_S):
            close_box(s, local, keep, fleet, name)
            raise BoxLost("stopped by the driver")
        probe = s.ssh("cat bench/encode.log.status 2>/dev/null; echo ---; "
                      "stat -c %s bench/encode.log; echo ---; tail -n 3 bench/encode.log",
                      check=False, quiet=True)
        if probe.returncode != 0:
            state = s.state()
            if state != "running":
                raise BoxLost(f"the instance is {state}")
            s.follow_host_address()
            continue
        s.download_new("bench/out", local, CHECKPOINT)
        settled = settled_ids(local, keep)
        fleet.queue.settle(name, settled)
        for r in box_records(local):
            if "failed" in r:
                fleet.queue.fail(name, r["id"], r["failed"])
        if fleet.chaos_pending and settled:
            fleet.chaos_pending = False
            print(f"{name}: rehearsal: terminating {s.instance_id} (a forced reclaim)")
            s.ec2.terminate_instances(InstanceIds=[s.instance_id])
        status, _, rest = probe.stdout.partition("---\n")
        log_size, _, tail = rest.partition("---\n")
        if status.strip():
            return box_outcome(local, status.strip(), tail)
        now = utc_minutes_since(s.launched)
        if log_size.strip() != size:
            size, last_growth = log_size.strip(), now
        elif now - last_growth > BOX_STALL_MIN:
            raise BoxLost(f"its log was silent for {BOX_STALL_MIN} min")
        feed.top_up(recorded_ids(local))


#: How long a stopped box gets to finish its encode in progress and code its closing references.
STOP_GRACE_MIN = 12


def close_box(s: Session, local: Path, keep: bool, fleet: Fleet, name: str) -> None:
    """Any stop of a box (a cancel, a spend ceiling, a fleet stop) closes it like a spot notice:
    STOP in its inbox, box.py finishes the encode in progress and codes its references, and its
    records come back before the box is terminated, so a stop never costs finished work (owner,
    2026-11-24: a stopped tail left 214 coded jobs without a valid time)."""
    deadline = utc_minutes_since(s.launched) + STOP_GRACE_MIN
    try:
        subprocess.run([*s.ssh_base(), "touch bench/inbox/STOP"], check=True, timeout=60)
        while utc_minutes_since(s.launched) < deadline:
            done = s.ssh("cat bench/encode.log.status 2>/dev/null", check=False, quiet=True)
            if done.returncode == 0 and done.stdout.strip():
                break
            time.sleep(POLL_S / 3)
        s.download_new("bench/out", local, CHECKPOINT)
        fleet.queue.settle(name, settled_ids(local, keep))
    except (subprocess.SubprocessError, RuntimeError, OSError) as error:
        print(f"{name}: closing failed ({error}); its unsettled jobs go back to the queue")


def box_outcome(local: Path, status: str, tail: str) -> str:
    """How box.py ended, from its last records: done, interrupted, or a refusal (raised)."""
    last = {}
    path = local / "encodes.jsonl"
    for line in (path.read_text().splitlines() if path.exists() else []):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "refused" in r or "interrupted" in r:
            last = r
    if "interrupted" in last:
        return "interrupted"
    if "refused" in last:
        raise (BoxRefusedTiming if last.get("timing") else BoxRefused)(last["refused"])
    if status != "0":
        raise BoxRefused(f"box.py exit {status}: {tail.strip()[-300:]}")
    return "done"


class BoxRefused(RuntimeError):
    """A check other than timing refused (V2, V4, V9, V21...): the run stops, never retried."""


class BoxRefusedTiming(RuntimeError):
    """V7/V8 failed twice on one box: the box is given up and replaced (section 6)."""


class StartSetRefused(BoxRefusedTiming):
    """V6 (D13): a machine's start check set is off the series' other machines: it does no
    work and is replaced."""


#: Errors a box's session can raise that mean "this box is lost", not "the run is wrong"
#: (review #2: a transient SSH or API error used to end its slot for good).
LOST_BOX_ERRORS = (BoxLost, BoxRefusedTiming, RemoteFailure, NoCapacity, RuntimeError,
                   subprocess.SubprocessError, OSError)


def slot(fleet: Fleet, index: int) -> None:
    """One slot of the fleet: a box, then a replacement for each box lost (reclaimed, refused on
    timing, or any SSH or API error), while the queue has work, the fleet's budget allows and
    no box has refused on a non-timing check (BP-3.11 rule 3). It gives up after MAX_ATTEMPTS
    boxes in a row that settled nothing (review #12: a lifetime end or a reclaim after useful
    work is normal and does not count). Each box gets a zone in turn (rule 4). Box names carry
    the driver's start time, so a restarted driver never reuses (and overwrites) a box's
    directory (review #8)."""
    from botocore.exceptions import BotoCoreError, ClientError
    barren, attempt = 0, 0
    while barren < MAX_ATTEMPTS:
        if fleet.stop.is_set() or fleet.queue.pending_count() == 0:
            return
        if not fleet.may_launch():
            print(f"slot {index}: the fleet budget ${fleet.a.fleet_budget:.2f} is reached")
            return
        name = f"box{index}-{fleet.started}-{attempt}"
        zone = fleet.zones[(index + attempt) % len(fleet.zones)]
        attempt += 1
        before = len(fleet.queue.settled)
        try:
            outcome = serve_box(fleet, name, zone)
            print(f"{name}: {outcome}")
        except BoxRefused as why:
            fleet.failure = f"{name}: {why}"
            fleet.stop.set()
            raise
        except (*LOST_BOX_ERRORS, ClientError, BotoCoreError) as why:
            print(f"{name}: {type(why).__name__}: {(str(why).splitlines() or ['?'])[0]}; replacing")
            if isinstance(why, NoCapacity) and not fleet.zone_fallback:
                fleet.zone_fallback = True
                print(f"no capacity in the allowed zones: {', '.join(AVOID_ZONES)} allowed too")
        barren = 0 if len(fleet.queue.settled) > before else barren + 1
    print(f"slot {index}: {MAX_ATTEMPTS} boxes in a row settled nothing; giving up")


class ThreadLogs(io.TextIOBase):
    """The driver's stdout: each box thread's prints go to that box's own log, anything else
    to the driver's."""

    def __init__(self, fallback) -> None:
        super().__init__()
        self.fallback, self.local = fallback, threading.local()

    def bind(self, log) -> None:
        self.local.log = log

    def write(self, text: str) -> int:
        target = getattr(self.local, "log", None) or self.fallback
        target.write(text)
        target.flush()
        return len(text)

    def flush(self) -> None:
        self.fallback.flush()


def drive(logs: dict[int, Path], work: Callable[[int, threading.Event], None],
          stop: threading.Event | None = None) -> list[int]:
    """Run `work(index, stop)` for every box as a thread of this process, each printing to its
    own log, which ends with BOX_DONE or BOX_FAILED. SIGTERM or SIGHUP set `stop`, which fails
    every session's waits so each terminates its box. Returns the failed indices."""
    stop, failed = stop or threading.Event(), []
    router = ThreadLogs(sys.stdout)

    def one(index: int) -> None:
        with open(logs[index], "a") as log:
            router.bind(log)
            try:
                work(index, stop)
                print(BOX_DONE)
            except BaseException as error:  # noqa: BLE001 - every failure is the box's result
                traceback.print_exc(file=log)
                print(BOX_FAILED + (str(error).strip().splitlines() or [type(error).__name__])[0])
                failed.append(index)

    previous = {sig: signal.signal(sig, lambda *_: stop.set())
                for sig in (signal.SIGTERM, signal.SIGHUP)}
    sys.stdout, original = router, sys.stdout
    try:
        threads = [threading.Thread(target=one, args=(index,)) for index in logs]
        for thread in threads:
            thread.start()
        for thread in threads:
            while thread.is_alive():  # a timed join, so the main thread still takes signals
                thread.join(1)
    finally:
        sys.stdout = original
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return sorted(failed)


def score_box(run: Path, series: str, index: int, a, stop: threading.Event | None = None,
              clients: SharedClients | None = None, attempt: int = 0) -> None:
    """One spot scoring box: the stage (decoders, clips, scripts), this shard's encode records
    and their streams; score.py with many workers; the scores back."""
    base = run / "series" / series
    # one record per id: a job coded twice (re-coded after a refused time) has two records with
    # the same stream (V11), and scoring both at once made two workers share one work dir
    # (2026-11-24: score.py crashed on every attempt of one slot)
    encodes = list({r["id"]: r for r in map(json.loads,
                                            (base / "encodes.jsonl").read_text().splitlines())
                    if "stream" in r and "canary" not in r and "reference" not in r}.values())
    if a.ids:  # exactly these streams, scored or not (an Arm parity check)
        wanted = set(json.loads(a.ids.read_text()))
        encodes = [r for r in encodes if r["id"] in wanted]
    else:
        done = scored(base)
        encodes = [r for r in encodes if needs_score(r, done, a.rescore)]
    clips = sorted({r["id"].split("|")[1] for r in encodes})
    mine = set(clips[index::a.boxes])  # whole clips per box: only its clips are uploaded
    # a replacement box skips what this slot's earlier boxes already scored (checkpointed)
    earlier = pass_scored(base, a.tag, index)
    shard = [r for r in encodes if r["id"].split("|")[1] in mine and r["id"] not in earlier]
    if not shard:
        return
    work = base / f"score{a.tag}{index}" if attempt == 0 else \
        base / f"score{a.tag}{index}-{attempt}"
    (work / "out" / "streams").mkdir(parents=True, exist_ok=True)
    (work / "out" / "encodes.jsonl").write_text("\n".join(map(json.dumps, shard)) + "\n")
    for r in shard:
        target = work / "out" / r["stream"]
        if not target.exists():
            target.symlink_to((base / r["stream"]).resolve())
    staged = box_stage(run, f"score{index}", mine)
    # scoring runs the current score.py, not the run's frozen copy: it does not change a
    # score (the same decoders, metrics and checks), only how a box does the work (BP-3.11:
    # decodes in memory, workers capped by it); each pass records its commit
    for name in ("score.py", "streamcheck.py"):  # streamcheck: new in BP-4 (V29, V30)
        current = staged / name
        if current.is_symlink() or current.exists():
            current.unlink()
        current.symlink_to(REPO / "tools" / "bench" / name)
    choices = [a.types] if a.type_given else [ARM_SCORE_TYPES, UNTIMED_TYPES]
    for i, types in enumerate(choices):
        spec = SessionSpec(region=a.region, instance_type=types[0], fallback_types=types[1:],
                           purpose=plan_purpose(a), lifetime_minutes=a.lifetime,
                           budget_usd=a.budget,
                           on_demand=False, root_gb=a.root_gb, purpose_vcpu_cap=a.vcpu_cap,
                           cap_purpose=PURPOSE,
                           launch_wait_minutes=a.lifetime if i else 10,
                           arch=image_arch(types[0]),
                           tags={"Owner": "bench", "Series": series})
        link_score_tools(run, staged, spec.arch, a.score_bin)
        try:
            return score_on(spec, staged, work, a, stop, clients)
        except (NoCapacity, BudgetExceeded) as why:  # no Graviton room, or none in budget
            if i + 1 == len(choices):
                raise
            print(f"no {types[0]} capacity ({str(why).splitlines()[0]}): x86 fallback")


def link_score_tools(run: Path, staged: Path, arch: str, given: Path | None) -> None:
    """The scoring tools for the box's architecture: an Arm box gets the aarch64 decoders and
    libvmaf (scores identical to x86 per frame), an x86 box the run's stage/bin."""
    tools = staged / "bin"
    if tools.is_symlink() or tools.exists():
        tools.unlink()
    if given:
        tools.symlink_to(given.resolve())
    elif arch == "arm64":
        tools.symlink_to(arm_score_bin(run).resolve())
    else:
        tools.symlink_to((run / "stage" / "bin").resolve())


def arm_score_bin(run: Path) -> Path:
    """The aarch64 scoring tools as one directory (links), beside the run's box stages."""
    target = run / "boxstage" / "scorebin-aarch64"
    target.mkdir(parents=True, exist_ok=True)
    sources = {**{t: ARM_TOOLS / "bin" / t for t in ARM_SCORE_TOOLS},
               "nanocodec": ARM_NANOCODEC,
               VMAF_MODEL_NAME: ARM_TOOLS / "share" / "vmaf" / "model" / "vmaf_v1.0.16" /
               VMAF_MODEL_NAME}
    for name, source in sources.items():
        if not source.exists():
            raise SystemExit(f"refused: the Arm scoring tool {source} is missing")
        link = target / name
        try:  # scoring slots make these links at once: the first wins
            link.symlink_to(source)
        except FileExistsError:
            pass
    return target


def score_on(spec: SessionSpec, staged: Path, work: Path, a, stop, clients) -> None:
    """One scoring box: upload, prepare, score.py with many workers, scores checkpointed."""
    with Session(spec, stop, clients) as s:
        s.upload(staged, "bench")
        s.upload(work / "out", "bench/out")
        s.run_detached(prepare_command(), "bench/prepare.log", 15, max_errors=0)
        checkpoint = ScoreCheckpoint(s, work / "scores.jsonl")
        try:
            s.run_detached(f"(cd bench && python3 -I score.py --encodes out/encodes.jsonl "
                           f"--jobs jobs.json --clips {TMPFS} --out out --workers {a.score_workers})",
                           "bench/score.log", a.lifetime - 10, max_errors=0)
        finally:
            checkpoint.close()


class ScoreCheckpoint:
    """BP-3.11 for scoring: the box's scores.jsonl copied to the PC every POLL_S (a small
    text file, copied whole), so a reclaimed scoring box loses only the scores in progress."""

    def __init__(self, s: Session, local: Path) -> None:
        self.s, self.local, self.stop = s, local, threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def pull(self) -> None:
        got = self.s.ssh("cat bench/out/scores.jsonl 2>/dev/null", check=False, quiet=True)
        if got.returncode == 0 and got.stdout:
            text = got.stdout[:got.stdout.rfind("\n") + 1]  # whole lines only
            if len(text) >= (self.local.stat().st_size if self.local.exists() else 0):
                self.local.write_text(text)

    def loop(self) -> None:
        while not self.stop.wait(POLL_S):
            self.pull()

    def close(self) -> None:
        self.stop.set()
        self.thread.join()
        self.pull()


def needs_score(encode: dict, done: set[str], rescore: str | None) -> bool:
    """A stream is scored unless the series already has a valid score for it; `rescore`
    (fnmatch on the family) scores those families again (the later record wins in report)."""
    if rescore and fnmatch.fnmatchcase(encode["family"], rescore):
        return True
    return encode["id"] not in done


def score_hashes(base: Path) -> dict[str, str]:
    """{id: stream SHA-256} of the series' valid score records, read without their per-frame
    metrics (the full records run a driver out of memory)."""
    path = base / "scores.jsonl"
    out = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip() and '"refused"' not in line:
                    rec = json.loads(line[:line.index(', "frame_bytes"')] + "}")
                    out[rec["id"]] = rec["stream_sha256"]
    return out


def scored(base: Path) -> set[str]:
    """The ids the series already has a valid score for."""
    path = base / "scores.jsonl"
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:  # a score record carries per-frame metrics: read only its id
                if line.strip() and '"refused"' not in line:
                    try:
                        done.add(json.loads(line[:line.index(', "')] + "}")["id"])
                    except (ValueError, KeyError):
                        continue  # a line cut short (a killed writer): its stream is rescored
    return done


def pass_scored(base: Path, tag: str, index: int) -> set[str]:
    """The ids a scoring slot's boxes (the first and its replacements) have scored so far."""
    ids = set()
    for d in [base / f"score{tag}{index}", *base.glob(f"score{tag}{index}-*")]:
        path = d / "scores.jsonl"
        for line in (path.read_text().splitlines() if path.exists() else []):
            try:
                ids.add(json.loads(line)["id"])
            except json.JSONDecodeError:
                continue
    return ids


def score_slot(run: Path, series: str, index: int, a, stop: threading.Event,
               clients: SharedClients) -> None:
    """A scoring slot: its box, and a replacement for each box lost (spot reclaim, stall) while
    its streams are unscored (BP-3.11, as for the encode fleets)."""
    for attempt in range(MAX_ATTEMPTS):
        if stop.is_set():
            return
        try:
            score_box(run, series, index, a, stop, clients, attempt)
            return
        except (RemoteFailure, NoCapacity) as why:
            print(f"scoring box {index}-{attempt}: {str(why).splitlines()[0]}; replacing")
    raise RuntimeError(f"scoring slot {index}: {MAX_ATTEMPTS} attempts used")


def merge_scores(run: Path, series: str, tag: str, boxes: int) -> None:
    """This scoring pass's records appended to the series' (a later record for an id wins)."""
    base = run / "series" / series
    dirs = [d for i in range(boxes)
            for d in [base / f"score{tag}{i}", *sorted(base.glob(f"score{tag}{i}-*"))]]
    lines = [line for d in dirs if (d / "scores.jsonl").exists()
             for line in (d / "scores.jsonl").read_text().splitlines() if line.strip()]
    with open(base / "scores.jsonl", "a") as out:
        out.write("\n".join(lines) + "\n")


#: BP-3.5 retention (coordinator, pending the owner): a Fixed QP stream is deleted from the PC
#: once it is scored and its SHA-256 and size are recorded (it is re-generable from its
#: recorded command, build and clip); CBR streams are kept.
PRUNE_OWNER = "tools.bench.fleet"


def prune(run: Path, series: str) -> tuple[int, int]:
    """Delete the series' scored Fixed QP streams, path by path, each through the retention
    guard; returns (files, bytes) removed."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.retention import guard
    base = run / "series" / series
    jobs = {j["id"]: j for j in json.loads((run / "jobs.json").read_text())}
    encodes = {}
    for line in (base / "encodes.jsonl").read_text().splitlines():
        r = json.loads(line) if line.strip() else {}
        if "stream" in r and "canary" not in r and "reference" not in r:
            encodes[r["id"]] = r
    good = score_hashes(base)
    files = size = 0
    for job_id, enc in encodes.items():
        scored_hash = good.get(job_id)
        path = base / enc["stream"]
        if jobs.get(job_id, {}).get("mode") != "qp" or not scored_hash or not path.exists():
            continue
        if scored_hash != enc["stream_sha256"] or enc.get("bytes") is None:
            continue
        refused = guard(path, owner=PRUNE_OWNER)
        if refused:
            raise SystemExit(f"refused by retention: {refused}")
        size += path.stat().st_size
        path.unlink()
        files += 1
    return files, size


def dedupe_stage(run: Path, clips: Path) -> int:
    """Replace a staged clip copy by a link to its source when both decode to the same
    frames (BP-3.5: the stage keeps no second copy of a kept clip); returns bytes freed."""
    sys.path.insert(0, str(REPO / "eval"))
    from nceval.retention import guard
    decoded = lambda p: hashlib.sha256(subprocess.run(  # noqa: E731
        ["zstd", "-dcq", str(p)], capture_output=True, check=True).stdout).hexdigest()
    freed = 0
    for copy in sorted((run / "stage" / "clips").glob("*.yuv.zst")):
        source = (clips / copy.name).resolve()
        if copy.is_symlink() or not source.exists() or decoded(copy) != decoded(source):
            continue
        refused = guard(copy, owner=PRUNE_OWNER)
        if refused:
            raise SystemExit(f"refused by retention: {refused}")
        freed += copy.stat().st_size
        copy.unlink()
        copy.symlink_to(source)
    return freed


def valid_ids(runs: list[Path], series: str) -> set[str]:
    """The jobs with a valid record in any of the runs, merged or still in a box's directory:
    what the report accepts (`report.timed_records`). `missing` and `status` both count by it,
    so they agree (RCA R5)."""
    import report
    valid = set()
    for run in runs:
        base = run / "series" / series
        sources = [(base.name, report.read_jsonl(base / "encodes.jsonl"))] \
            if (base / "encodes.jsonl").exists() else []
        sources += [(box.name, box_records(box)) for box in base.glob("box*") if box.is_dir()]
        for name, records in sources:
            try:
                valid |= {r["id"] for r in report.timed_records(name, records)[0]}
            except report.ReportRefusal:  # not judgeable yet
                continue
    return valid


def missing(runs: list[Path], series: str) -> list[dict]:
    """The first run's planned jobs without a valid record in any run (lost or refused boxes,
    times after a box's last references): the jobs of a follow-up run, with the commands they
    were planned with."""
    valid = valid_ids(runs, series)
    return [j for j in json.loads((runs[0] / "jobs.json").read_text()) if j["id"] not in valid]


def tag_box(rec: dict, name: str) -> dict:
    """A record with its box's name; the box's header record keeps its facts as `box_info`."""
    if isinstance(rec.get("box"), dict):
        rec["box_info"] = rec["box"]
    return {**rec, "box": name}


def merge(run: Path, series: str) -> None:
    """The boxes' records into the series' `encodes.jsonl` and `scores.jsonl` (report input)."""
    base = run / "series" / series
    base.mkdir(parents=True, exist_ok=True)
    boxes = sorted(b for b in base.glob("box*") if b.is_dir())
    # each encode record names its box, so V6 can compare a box's canaries with its own
    encodes = []
    for box in boxes:
        if not (box / "encodes.jsonl").exists():
            continue
        for line in (box / "encodes.jsonl").read_text().splitlines():
            try:
                encodes.append(json.dumps(tag_box(json.loads(line), box.name)))
            except ValueError:
                continue  # blank, or a line cut short when its machine was lost
    (base / "encodes.jsonl").write_text("\n".join(encodes) + "\n")
    scores = [line for box in boxes if (box / "scores.jsonl").exists()
              for line in (box / "scores.jsonl").read_text().splitlines() if line.strip()]
    if scores:  # scored on the timing boxes: appended (a later record for an id wins)
        with open(base / "scores.jsonl", "a") as out:
            out.write("\n".join(scores) + "\n")
    for box in sorted(base.glob("box*")):
        streams = box / "streams"
        if streams.exists():
            (base / "streams").mkdir(exist_ok=True)
            for f in streams.iterdir():
                f.rename(base / "streams" / f.name)


#: BP-3.6: the two timing references, one lighter on the memory system (x264 medium) and one
#: heavier (our AV1 medium), coded on the box's first clip at the 500 rung; neighbour
#: interference may hit them differently, and the report refuses a box whose two references
#: diverge. (x264 ultrafast, the lightest, codes a clip in well under 0.1 s: too short to time
#: to the residual target.)
REFERENCE_FAMILIES = ("x264-medium", "nanocodec-av1@medium")
REFERENCE_RUNG = 500


#: A run whose encoder under test is not today's (the history series' daily builds) carries
#: the benchmark's own build as this binary: the timing references always run the record's
#: commands on one fixed build (a day's build may not know them: 2026-11-06 had no CBR).
REFERENCE_BINARY = "nanocodec-ref"


def references(shard: list[dict], bin_dir: Path | None = None) -> list[dict]:
    sys.path.insert(0, str(REPO / "tools" / "bench"))
    import plan
    first = shard[0]
    clip = {"name": first["input"], "width": first["w"], "height": first["h"], "fps": first["fps"],
            "source_kbps": float("inf")}  # a reference, not a ladder point: no source cap
    jobs = [j for j in plan.plan([clip], list(REFERENCE_FAMILIES), "cbr", [REFERENCE_RUNG],
                                 first["frames"], 300) if not j["anchor_extra"]]
    if bin_dir is not None and (bin_dir / REFERENCE_BINARY).exists():
        for job in jobs:
            if job["argv"][0] == "{BIN}/nanocodec":
                job["argv"] = [f"{{BIN}}/{REFERENCE_BINARY}", *job["argv"][1:]]
    return jobs


def box_stage(run: Path, key: str, clips: set[str] = frozenset()) -> Path:
    """The stage as a box sees it: everything but the clips, and the named clips (links); a
    queue box gets its clips from the driver one at a time (`put_clip`)."""
    stage_dir, target = run / "stage", run / "boxstage" / key
    (target / "clips").mkdir(parents=True, exist_ok=True)
    for entry in stage_dir.iterdir():
        link = target / entry.name
        if link.is_symlink() and not link.exists():  # dangling: its worktree was removed
            link.unlink()
        if entry.name != "clips" and not link.exists():
            link.symlink_to(entry.resolve())
    for name in sorted(clips):
        link = target / "clips" / f"{name}.yuv.zst"
        if not link.exists():
            link.symlink_to((stage_dir / "clips" / link.name).resolve())
    # The box's harness is this driver's own, never the stage's copy: a stage frozen before a
    # harness option existed got commands it rejected, and the box ended at once (2026-11-26)
    for name in HARNESS_FILES:
        link = target / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(REPO / "tools" / "bench" / name)
    return target


#: The box-side harness scripts, taken from the driver's tools (`box_stage`).
HARNESS_FILES = ("halfload.py", "box.py", "pinning.py", "streamcheck.py")


def resolve_types(a) -> tuple[str, ...]:
    """The instance types a step launches: as given, else timing or untimed by role."""
    if a.type:
        types = (a.type,)
    elif a.step == "score":
        types = ARM_SCORE_TYPES
    elif a.packed:
        types = UNTIMED_TYPES
    else:
        types = (TIMING_TYPE,)
    for t in types:
        if t.split(".")[0] in EXCLUDED_FAMILIES:
            raise SystemExit(f"refused: {t} is excluded by docs/aws-cost-guide.md")
    return types


def require_capped_scope() -> None:
    """Every local driver runs in a memory-capped scope (owner, 2026-11-24: the PC is not to
    be loaded): refuse to start outside one."""
    group = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[-1]
    limit = Path("/sys/fs/cgroup") / group.lstrip("/") / "memory.max"
    if not limit.exists() or limit.read_text().strip() == "max":
        raise SystemExit("refused: run me in a memory-capped scope, e.g. "
                         "systemd-run --user --scope -p MemoryMax=2G python3 tools/bench/fleet.py ...")


def score(a) -> None:
    """Spot scoring boxes as threads of this process, then the scores merged; with `--local`,
    this PC's GPU scoring service instead (`localscore`, BP-4.0 D16)."""
    require_capped_scope()
    check_free_disk(a.run)
    base = a.run / "series" / a.series
    with open(base / "score-passes.jsonl", "a") as f:
        f.write(json.dumps({"tag": a.tag, "commit": commit(),
                            "backend": "vmafx" if a.local else "libvmaf",
                            "types": None if a.local else a.types,
                            "score_bin": str(a.score_bin or "stage/bin")}) + "\n")
    if a.local:
        localscore.score_local(a, scored, needs_score, score)
        return
    logs = {i: base / f"score{a.tag}{i}.log" for i in range(a.boxes)}
    clients = SharedClients()
    failed = drive(logs, lambda index, stop: score_slot(a.run, a.series, index, a, stop, clients))
    if not a.no_merge:
        merge_scores(a.run, a.series, a.tag, a.boxes)
        # BP-3.5: a scored Fixed QP stream is not kept (its hash and size are): pruned as each
        # pass ends, so a long series of runs stays above the disk floor (2026-11-24: the
        # AOM-CTC Fixed QP runs held 90 GB of scored streams and took the PC under it)
        files, size = prune(a.run, a.series)
        print(f"pruned {files} scored Fixed QP streams, {size / 1e9:.1f} GB (BP-3.5)")
    if failed:
        raise SystemExit(f"refused: {len(failed)} of {a.boxes} scoring boxes failed "
                         "(their unscored streams are picked up by the next `score`)")


def check_gate(a) -> None:
    """RCA R1: a full launch needs a passed rehearsal and an approved envelope on the same tools
    version; `--ungated REASON` overrides, printed and recorded."""
    if getattr(a, "rehearsal", False):
        return
    tools = tools_version()
    why = gate_refusal(tools)
    if why is None:
        return
    if not a.ungated:
        raise SystemExit(f"refused: {why}. Rehearse and approve first, or pass --ungated REASON")
    say(f"UNGATED LAUNCH ({why}): {a.ungated}")
    record_gate("ungated", run=str(a.run), reason=a.ungated, refusal=why)


def run(a) -> None:
    """BP-3.11: `--boxes` slots as threads of this process, each serving queue units and
    replacing its lost boxes; then the merge. Jobs left unsettled (budget reached, a refusal)
    are listed by `missing`."""
    require_capped_scope()
    check_gate(a)
    check_free_disk(a.run)
    base = a.run / "series" / a.series
    base.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((a.run / "jobs.json").read_text())
    done = valid_ids([a.run], a.series)
    expected_seconds(jobs, measured_durations(a.run.parent.parent))
    todo = [j for j in jobs if j["id"] not in done]
    if not a.boxes:  # RCA R10: from the expected seconds, never the job count
        a.boxes = fleet_size(todo, a.target_minutes, a.vcpu_cap,
                             BOX_VCPUS.get(a.types[0], 8))
    if a.fleet_budget is None:  # after sizing (review #9: before it, 0 boxes gave $0)
        a.fleet_budget = 1.5 * a.boxes * a.budget
    say(f"{len(todo)} jobs, {sum(j['expected_s'] for j in todo) / 3600:.1f} expected encode-hours "
        f"(longest {max((j['expected_s'] for j in todo), default=0):.0f} s) on {a.boxes} boxes")
    stop = threading.Event()
    fleet = Fleet(a, todo, stop, planned=jobs)
    logs = {i: base / f"slot{i}.log" for i in range(a.boxes)}
    failed_slots = drive(logs, lambda index, _stop: slot(fleet, index), stop)
    merge(a.run, a.series)
    # the final count is the report's own (review #4: a job settled early can be refused later,
    # by its box's end canaries or the V6 moving median)
    left = len(missing([a.run], a.series))
    say(f"{a.series}: {len(jobs) - left} of {len(jobs)} jobs valid; fleet spend about "
        f"${fleet.spent:.2f} (the ledger books the exact cost); records in {base}")
    if fleet.failure:
        raise SystemExit(f"refused: {fleet.failure}")
    if failed_slots:
        raise SystemExit(f"refused: slots {failed_slots} failed (their logs say why)")
    if fleet.queue.quarantined:
        (base / "quarantined.json").write_text(json.dumps(fleet.queue.quarantined, indent=1))
        raise SystemExit(f"refused: {len(fleet.queue.quarantined)} jobs failed on "
                         f"{QUARANTINE_AFTER} boxes each, listed in {base / 'quarantined.json'}")
    if left:
        raise SystemExit(f"refused: {left} jobs unsettled (fleet budget or attempts); "
                         "`missing` lists them")


# --- BP-4: whole machines at half load (wholebox.py, halfload.py) -------------------------


def plan_timing(a) -> dict:
    """D13/D15: the run's timing plan (`timing-plan.json`), its expected wall time and cost."""
    import wholebox
    jobs = json.loads((a.run / "jobs.json").read_text())
    done = valid_ids([a.run], a.series)
    todo = [j for j in jobs if j["id"] not in done]
    measured = measured_durations(a.run.parent.parent)
    if not measured:  # every job would weigh 1 s and families run in name order
        raise SystemExit("refused: no measured job times found to order the families by")
    expected_seconds(todo, measured)
    price = a.price if a.price else wholebox_price(wholebox.machine_type(a.series))
    plan = wholebox.timing_plan(todo, a.series, a.machines, price)
    plan["jobs"] = {j["id"]: round(j["expected_s"], 3) for j in todo}
    (a.run / wholebox.PLAN_FILE).write_text(json.dumps(plan, indent=1))
    say(f"{len(todo)} jobs over {len(plan['machines'])} {plan['instance_type']}: expected "
        f"{plan['expected_wall_s'] / 3600:.2f} h wall, {plan['expected_machine_hours']} "
        f"machine-hours, about ${plan['expected_cost_usd']} at ${price}/h")
    return plan


#: A staged run whose binary is still being built carries this file in stage/bin; `set-binary`
#: replaces the binary, records its hash in the manifest and removes the file.
PENDING_MARK = "PENDING"


def await_pending_binaries(staged: Path, stop: threading.Event, poll_s: float = 15) -> None:
    """Hold a machine obtained early until the run's pending binary is set (owner, 2026-11-26:
    hunt capacity while our build finishes). Nothing is uploaded or coded before it."""
    mark = staged / "bin" / PENDING_MARK
    if mark.exists():
        say(f"machine held: waiting for the pending binary ({mark.read_text().strip()})")
    while mark.exists():
        if stop.wait(poll_s):
            raise RuntimeError("stopped while waiting for the pending binary")


def set_binary(run: Path, name: str, binary: Path) -> None:
    """Replace a staged binary before any machine uses it, with its hash in the manifest (V2)."""
    stage_dir = run / "stage"
    target = stage_dir / "bin" / name
    shutil.copy2(binary.resolve(), target)
    for manifest_path in (run / "manifest.json", stage_dir / "manifest.json"):
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["binaries"][name] = sha256(target)
            manifest_path.write_text(json.dumps(manifest, indent=1))
    (stage_dir / "bin" / PENDING_MARK).unlink(missing_ok=True)
    say(f"{run.name}: {name} set to {binary} ({sha256(target)[:12]})")


def region_quota_fits(region: str, instance_type: str) -> bool:
    """Whether the region's spot vCPU quota can hold one machine of the type at all (several
    regions have a 5-vCPU quota: a capacity search there only waits on the quota)."""
    import boto3
    from aws_session import spot_quota
    vcpus = boto3.client("ec2", region_name=region).describe_instance_types(
        InstanceTypes=[instance_type])["InstanceTypes"][0]["VCpuInfo"]["DefaultVCpus"]
    fits = spot_quota(region) >= vcpus
    if not fits:
        say(f"{region}: spot quota below one {instance_type} ({vcpus} vCPU): left out")
    return fits


def spot_price(region: str, instance_type: str) -> float | None:
    """The cheapest current spot price of a type in a region (None: not offered, or a region
    this account has not enabled: AuthFailure)."""
    import boto3
    from botocore.exceptions import ClientError
    try:
        history = boto3.client("ec2", region_name=region).describe_spot_price_history(
            InstanceTypes=[instance_type], ProductDescriptions=["Linux/UNIX"],
            StartTime=datetime.datetime.now(datetime.timezone.utc))["SpotPriceHistory"]
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != "AuthFailure":
            raise
        return None
    return min((float(h["SpotPrice"]) for h in history), default=None)


def wholebox_price(instance_type: str) -> float:
    import wholebox
    price = spot_price(wholebox.HOME_REGION, instance_type)
    if price is None:
        raise SystemExit(f"refused: {instance_type} is not offered in {wholebox.HOME_REGION}")
    return price


#: Owner, 2026-11-26 (D13 amended): one longest-first queue of all the machine's timed jobs,
#: no filler. One family at a time kept a 48-slot machine about 25 % busy (most families are
#: seconds of work; the per-family overhead dominated), so a run took 2-3x its machine time.
HALF_QUEUE = "mixed"


def half_command(series: str, remote: str = "bench", check_at: str = "both") -> str:
    return (f"(cd {remote} && sudo python3 -I halfload.py --plan plan.json --checkset "
            f"checkset.json --manifest manifest.json --series {series} --clips {TMPFS} "
            f"--out out --queue {HALF_QUEUE} --check-at {check_at})")


def chain_marker(box: str, first_run: str) -> str:
    """The record a chained run's machine starts with: its start check set ran (and passed,
    or the machine would have stopped) in `first_run`, where its check records are (V6 and
    `settled_families` read it; the report takes the check records from there)."""
    return json.dumps({"check_done": "start", "box": box, "chained_from": first_run})


def next_stage_command(remote: str, marker: str = "") -> str:
    """A machine taking another run of its chain: the previous run's clips leave the tmpfs,
    this run's are decompressed into it, its records get their own tmpfs directory, and the
    box's own hard stop moves on (MACHINE_EXTEND_MINUTES from now)."""
    return (f"sudo find {TMPFS} -maxdepth 1 -name '*.yuv' -delete && "
            f"for c in {remote}/clips/*.yuv.zst; do [ -e \"$c\" ] || continue; "
            f"zstd -dq --rm \"$c\" -o {TMPFS}/$(basename \"$c\" .zst); done && "
            f"mkdir -p {remote}/inbox {TMPFS}/out-{remote} && ln -sfn {TMPFS}/out-{remote} "
            f"{remote}/out && "
            + (f"echo {shlex.quote(marker)} >> {TMPFS}/out-{remote}/encodes.jsonl && "
               if marker else "") +
            f"(sudo shutdown -c || true) && "
            f"sudo shutdown -h +{MACHINE_EXTEND_MINUTES} && echo ready")


#: A machine taking another run of its chain lives this long from then (its own shutdown).
MACHINE_EXTEND_MINUTES = 180
#: A run's claim: the driver and machine serving it (O_EXCL; stale once its PID is gone).
CLAIM_FILE = "claim"


def claim_run(run: Path, series: str, box: str) -> bool:
    """Claim `run` for this driver's machine `box`, atomically; a claim whose driver PID is
    gone is stale and taken over. True when this driver holds the claim."""
    path = run / "series" / series / CLAIM_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = run_claim(run, series)
            if holder and holder.get("pid") == os.getpid():
                return True
            if holder and pid_alive(holder.get("pid", 0)):
                return False
            path.unlink(missing_ok=True)  # stale: its driver is gone
            continue
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "box": box, "since": time.time()}, f)
        return True
    return False


def run_claim(run: Path, series: str) -> dict | None:
    try:
        return json.loads((run / "series" / series / CLAIM_FILE).read_text())
    except (OSError, ValueError):
        return None


def release_run(run: Path, series: str) -> None:
    holder = run_claim(run, series)
    if holder and holder.get("pid") == os.getpid():
        (run / "series" / series / CLAIM_FILE).unlink(missing_ok=True)


def claimed_elsewhere(run: Path, series: str) -> bool:
    holder = run_claim(run, series)
    return bool(holder and holder.get("pid") != os.getpid() and pid_alive(holder.get("pid", 0)))


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return pid > 0


def run_remaining(run: Path, series: str) -> tuple[dict, dict, list[str]]:
    """A run's timing plan, its jobs and the planned families not yet settled by any of its
    machines (its records under series/<series>/box*)."""
    import wholebox
    plan = json.loads((run / wholebox.PLAN_FILE).read_text())
    jobs = {j["id"]: {**j, "expected_s": plan["jobs"].get(j["id"], 1.0)}
            for j in json.loads((run / "jobs.json").read_text()) if j["id"] in plan["jobs"]}
    done: set[str] = set()
    for box in sorted((run / "series" / series).glob("box*")):
        done |= wholebox.settled_families(wholebox.read_records(box / "encodes.jsonl"))
    families = [f for m in plan["machines"] for f in m["families"] if f not in done]
    return plan, jobs, families


def stage_run(run: Path, plan: dict, jobs: dict[str, dict], checks: dict, families: list[str],
              name: str) -> Path:
    """The box stage of `run` for machine `name`: its families' jobs and its check set."""
    import wholebox
    by_family = wholebox.families_of([jobs[i] for i in jobs if jobs[i]["family"] in families])
    clips = {j["input"] for js in by_family.values() for j in js} | \
        {j["input"] for js in checks.values() for j in js}
    staged = box_stage(run, name, clips)
    (staged / "plan.json").write_text(json.dumps({
        "instance_type": plan["instance_type"],
        "families": [{"family": f, "jobs": by_family[f]} for f in families if f in by_family]}))
    (staged / "checkset.json").write_text(json.dumps(checks))
    return staged


def serve_machine(a, plan: dict, jobs: dict[str, dict], checks: dict, families: list[str],
                  name: str, stop: threading.Event, clients) -> set[str]:
    """One whole machine from launch to end: its own run's families (whole families, the
    plan's order) and then, with `--chain`, every other unfinished, unclaimed run of its chip
    on the same machine (capacity is the bottleneck: owner, 2026-11-26). The check set runs
    at the machine's start and at its final end. Returns the families of ITS OWN run it
    settled; a loss raises after the last copy."""
    import wholebox
    staged = stage_run(a.run, plan, jobs, checks, families, name)
    expected = max(m["expected_s"] for m in plan["machines"])
    lifetime = int(max(wholebox.MIN_LIFETIME_S,
                       wholebox.SETUP_S + wholebox.LIFETIME_FACTOR * expected) / 60) + 1

    def spec_for(region: str) -> SessionSpec:
        return SessionSpec(region=region, instance_type=plan["instance_type"],
                           purpose=plan_purpose(a), lifetime_minutes=lifetime,
                           budget_usd=a.budget, on_demand=False, root_gb=a.root_gb,
                           spill=False, launch_wait_minutes=1,
                           spot_cap_usd=wholebox.BENCH_SPOT_CAP_USD,
                           arch=image_arch(plan["instance_type"]),
                           tags={"Owner": "bench", "Series": a.series, "Layout": "half"})

    regions = [r for r in wholebox.capacity_regions(spot_price, plan["instance_type"])
               if region_quota_fits(r, plan["instance_type"])]
    adopt = getattr(a, "adopt", None)
    s = adopt_session(adopt, spec_for, regions) if adopt else \
        wholebox.open_session(spec_for, regions, stop, clients)
    chain = [Path(r) for r in getattr(a, "chain", None) or []]
    own: set[str] = set()
    served_any = False
    try:
        await_pending_binaries(staged, stop)
        # an adopted machine is already set up (tmpfs, packages): it takes its run like a
        # chained one, in its own directory
        first = f"bench-{a.run.name}" if adopt else "bench"
        s.upload(staged, first)
        s.run_detached(next_stage_command(first) if adopt else prepare_command(),
                       f"{first}/prepare.log", 20, max_errors=0)
        last_remote, last_run = first, a.run
        first_served: Path | None = None  # the run whose records hold the start check set
        if claim_run(a.run, a.series, name):
            own = serve_run(a, s, a.run, staged, first, name, stop,
                            "both" if not chain else "start")
            served_any, first_served = True, a.run
            release_run(a.run, a.series)
        for run in chain:
            if stop.is_set():
                break
            if claimed_elsewhere(run, a.series) or not (run / wholebox.PLAN_FILE).exists():
                continue
            plan_c, jobs_c, families_c = run_remaining(run, a.series)
            if not families_c or not claim_run(run, a.series, name):
                continue
            remote = f"bench-{run.name}"
            staged_c = stage_run(run, plan_c, jobs_c, {}, families_c, name)
            await_pending_binaries(staged_c, stop)
            print(f"{name}: taking {run.name} ({len(families_c)} families) on the same machine")
            s.upload(staged_c, remote)
            marker = chain_marker(name, first_served.name) if first_served else ""
            s.run_detached(next_stage_command(remote, marker), f"{remote}/prepare.log", 20,
                           max_errors=0)
            serve_run(a, s, run, staged_c, remote, name, stop,
                      "start" if not served_any else "none")
            served_any, last_remote, last_run = True, remote, run
            first_served = first_served or run
            release_run(run, a.series)
            if not run_remaining(run, a.series)[2]:
                merge(run, a.series)
        if chain and served_any and not stop.is_set():
            # the machine's final check set, in its last run's directory
            s.ssh(f"echo '{{\"families\": []}}' > {last_remote}/plan-end.json", check=False,
                  quiet=True)
            s.run_detached(half_command(a.series, last_remote, "end").replace(
                "--plan plan.json", "--plan plan-end.json"), f"{last_remote}/end.log", 30,
                max_errors=0)
            s.download_new(f"{last_remote}/out", run_local(last_run, a.series, name), CHECKPOINT)
        return own
    finally:
        for run in [a.run, *chain]:
            release_run(run, a.series)
        s.__exit__(None, None, None)


def adopt_session(instance_id: str, spec_for, regions: list[str]):
    """The Session of a live machine whose driver is gone (`run --adopt`), from its instance
    and the key pair and known-hosts files its driver left in WORK_DIR."""
    import boto3
    from aws_session import WORK_DIR, Session, iso
    for region in regions:
        found = boto3.client("ec2", region_name=region).describe_instances(
            Filters=[{"Name": "instance-id", "Values": [instance_id]}])["Reservations"]
        if not found:
            continue
        i = found[0]["Instances"][0]
        if i["State"]["Name"] != "running":
            raise SystemExit(f"refused: {instance_id} is {i['State']['Name']}, not running")
        key = i["KeyName"]
        if not (WORK_DIR / f"{key}.pem").exists():
            raise SystemExit(f"refused: no key {WORK_DIR / (key + '.pem')} to adopt it with")
        spec = spec_for(region)
        return Session.attach({
            "name": key, "key_name": key, "group_id": i["SecurityGroups"][0]["GroupId"],
            "instance_id": instance_id, "host": i["PublicIpAddress"],
            "zone": i["Placement"]["AvailabilityZone"],
            "price": spot_price(region, i["InstanceType"]),
            "spot_request": i.get("SpotInstanceRequestId"),
            "key_path": str(WORK_DIR / f"{key}.pem"),
            "known_hosts": str(WORK_DIR / f"{key}.known_hosts"),
            "launched": iso(i["LaunchTime"]),
            "spec": {**spec.__dict__, "tags": dict(spec.tags),
                     "fallback_types": list(spec.fallback_types)}})
    raise SystemExit(f"refused: no instance {instance_id} in {', '.join(regions)}")


def run_local(run: Path, series: str, name: str) -> Path:
    return run / "series" / series / name


def serve_run(a, s, run: Path, staged: Path, remote: str, name: str, stop: threading.Event,
              check_at: str) -> set[str]:
    """One run's families on an open machine: halfload in `remote`, its records copied back
    every POLL_S under the run's own series directory, its start set judged once back.
    Returns the families it settled."""
    import wholebox
    local = run_local(run, a.series, name)
    s.start_detached(half_command(a.series, remote, check_at), f"{remote}/half.log")
    chaos = getattr(a, "chaos", False)
    poll = wholebox.REHEARSAL_POLL_S if chaos else wholebox.POLL_S
    judged = check_at not in ("start", "both")
    while True:
        if stop.wait(poll):
            s.ssh(f"touch {remote}/STOP", check=False, quiet=True)
        s.download_new(f"{remote}/out", local, CHECKPOINT)
        records = wholebox.read_records(local / "encodes.jsonl")
        if not judged and any(r.get("check_done") == "start" for r in records):
            judged = True
            why = judge_start_set(name, local.parent)
            if why:
                s.ssh(f"touch {remote}/STOP", check=False, quiet=True)
                raise StartSetRefused(why)
        if chaos and wholebox.settled_families(records):
            a.chaos = False  # one forced reclaim per rehearsal; the replacement finishes
            print(f"{name}: rehearsal: terminating {s.instance_id} (a forced reclaim)")
            s.ec2.terminate_instances(InstanceIds=[s.instance_id])
            raise BoxLost("rehearsal reclaim")
        # by its wrapper's PID: `pgrep -f halfload.py` would match this very ssh command
        status = s.ssh(f"cat {remote}/half.log.status 2>/dev/null; "
                       f"kill -0 $(cat {remote}/half.log.pid) 2>/dev/null || echo DEAD",
                       check=False, quiet=True)
        if status.returncode != 0:
            if s.state() != "running":
                raise BoxLost(f"the instance is {s.state()}")
            continue
        if status.stdout.strip() == "DEAD":  # halfload died without its exit status
            raise BoxLost("halfload died on the box (see half.log)")
        if status.stdout.strip():
            s.download_new(f"{remote}/out", local, CHECKPOINT)
            records = wholebox.read_records(local / "encodes.jsonl")
            refused = [r["refused"] for r in records if "refused" in r]
            if refused:
                raise BoxRefused(refused[-1])
            interrupted = [r["interrupted"] for r in records if "interrupted" in r]
            if interrupted:
                # a spot notice or STOP ended halfload early: the machine is going, so it must
                # not chain on (2026-11-26: two reclaimed machines took a next run and died);
                # the slot replaces it, and the finally releases every claim at once
                raise BoxLost(f"halfload interrupted ({interrupted[-1]})")
            return wholebox.settled_families(records)


def judge_start_set(name: str, series_dir: Path) -> str | None:
    """V6 (D13): why machine `name`'s start check set refuses it, or None: its level among the
    start sets of the series' machines so far (records under `series_dir`, the report's
    measure) is more than CHECK_SET_LIMIT off. The first machine of a series has no
    comparison here; the report judges it against them all."""
    import wholebox
    records = [{**r, "box": path.parent.name}
               for path in sorted(series_dir.glob("box*/encodes.jsonl"))
               for r in wholebox.read_records(path)]
    ratio = wholebox.start_set_ratio(name, records)
    if ratio is None or abs(ratio - 1) <= wholebox.CHECK_SET_LIMIT:
        return None
    return f"V6: {name}'s start check set is {ratio:.4f} of the series' other machines"


def run_half(a) -> None:
    """BP-4 `run --layout half`: one thread per planned machine; a machine lost or interrupted
    hands its unsettled families to a replacement; then the merge (report input)."""
    import wholebox
    require_capped_scope()
    check_gate(a)
    check_free_disk(a.run)
    plan_path = a.run / wholebox.PLAN_FILE
    if not plan_path.exists():
        raise SystemExit(f"refused: no {wholebox.PLAN_FILE}: run `fleet.py plan-timing` first")
    plan = json.loads(plan_path.read_text())
    if plan["series"] != a.series:
        raise SystemExit(f"refused: the plan is for series {plan['series']}, not {a.series}")
    # a machine's session budget covers its lifetime at the spot cap (1.5 x today's price)
    # in the dearest region the capacity search may take (D14: up to SPILL_PRICE_FACTOR x
    # home's price; a budget sized on home's price refused eu-central-1, 2026-11-25)
    longest = max(m["expected_s"] for m in plan["machines"])
    life_s = max(wholebox.MIN_LIFETIME_S, wholebox.SETUP_S + wholebox.LIFETIME_FACTOR * longest)
    # the bid is the flat owner cap (BENCH_SPOT_CAP_USD); 10 % headroom over its worst case
    need = round(1.1 * wholebox.BENCH_SPOT_CAP_USD * life_s / 3600 + 0.05, 2)
    if a.budget < need:
        say(f"each machine's session budget raised from ${a.budget} to ${need} (its lifetime "
            f"at the spot cap)")
        a.budget = need
    jobs = {j["id"]: {**j, "expected_s": plan["jobs"].get(j["id"], 1.0)}
            for j in json.loads((a.run / "jobs.json").read_text()) if j["id"] in plan["jobs"]}
    checks = wholebox.check_jobs(list(jobs.values()))
    work = wholebox.Assignments(plan)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S")
    base = a.run / "series" / a.series
    base.mkdir(parents=True, exist_ok=True)
    # A relaunched driver (its predecessor killed on the PC) keeps the families its earlier
    # machines already settled, instead of coding them again.
    for m in plan["machines"]:
        for box in sorted(base.glob(f"box{m['index']}-*")):
            done = wholebox.settled_families(wholebox.read_records(box / "encodes.jsonl"))
            if done:
                say(f"{box.name}: {work.settle(m['index'], done)} families already settled")
    from aws_session import SharedClients
    clients = SharedClients()

    def machine_slot(index: int, stop: threading.Event) -> None:
        from botocore.exceptions import BotoCoreError, ClientError
        barren, attempt = 0, 0
        while work.remaining(index) and barren < wholebox.MAX_ATTEMPTS and not stop.is_set():
            if claimed_elsewhere(a.run, a.series):
                # another machine's chain took this run: no second machine for it
                print(f"slot {index}: {a.run.name} is served by {run_claim(a.run, a.series)}; "
                      f"this driver stops hunting")
                return
            name = f"box{index}-{started}-{attempt}"
            attempt += 1
            settled: set[str] = set()
            try:
                settled = serve_machine(a, plan, jobs, checks, work.remaining(index), name,
                                        stop, clients)
            except BoxRefused:
                stop.set()
                raise
            except (*LOST_BOX_ERRORS, ClientError, BotoCoreError) as why:
                print(f"{name}: {type(why).__name__}: {(str(why).splitlines() or ['?'])[0]}")
                settled = set() if isinstance(why, StartSetRefused) else \
                    wholebox.settled_families(wholebox.read_records(base / name / "encodes.jsonl"))
            got = work.settle(index, settled)
            barren = 0 if got else barren + 1
            print(f"{name}: settled {got}; {len(work.remaining(index))} families left")

    logs = {m["index"]: base / f"machine{m['index']}.log" for m in plan["machines"]}
    failed = drive(logs, machine_slot)
    if claimed_elsewhere(a.run, a.series):
        say(f"{a.run.name}: served by another machine's chain ({run_claim(a.run, a.series)})")
        return
    merge(a.run, a.series)
    left = {i: work.remaining(i) for i in logs if work.remaining(i)}
    if failed or left:
        raise SystemExit(f"refused: machines {failed} failed; unsettled families {left}")
    say(f"{a.series}: every planned family settled; records in {base}")


#: `status` measures each stage's rate over this window.
RATE_WINDOW_S = 600
#: The shortest span a box's rate is taken over (one record is not a rate).
MIN_RATE_SPAN_S = 120


def group_rate(records: list[dict], now: float) -> float:
    """One box's rate per hour over the last RATE_WINDOW_S, counted from its first record, not
    from its launch: setup is not work (RCA R5: a box 2 minutes into scoring after 8 minutes
    of setup showed 18 per hour)."""
    ends = [r["end_unix"] for r in records if r.get("end_unix")]
    if not ends:
        return 0.0
    start = max(now - RATE_WINDOW_S, min(ends))
    recent = sum(1 for t in ends if t > start)
    return recent / max(now - start, MIN_RATE_SPAN_S) * 3600 if now - max(ends) < RATE_WINDOW_S \
        else 0.0


def stage_progress(groups: dict[str, list[dict]], total: int, done: set[str],
                   now: float) -> dict:
    """Done/total, the rate (the sum of each box's own rate, setup excluded) and the ETA at it."""
    rate = sum(group_rate(records, now) for records in groups.values())
    left = max(0, total - len(done))
    return {"done": len(done), "total": total, "per_hour": round(rate),
            "eta_min": round(left / rate * 60) if rate else None}


def running_boxes(base: Path) -> list[tuple[float, float]]:
    """(price per hour, hours so far) of each box a series' logs show launched and not ended."""
    return [box for log in base.glob("*.log") for box in running_boxes_in(log)]


def running_boxes_in(log: Path) -> list[tuple[float, float]]:
    """(price per hour, hours so far) of each box one log shows launched and not ended."""
    boxes = []
    for session in log.read_text(errors="replace").split("] launched ")[1:]:
        if "session cost" in session:
            continue
        price = re.search(r"spot \$([0-9.]+)/h", session)
        started = re.search(r"\[aws (\d\d):(\d\d):(\d\d)\] .*running at", session)
        if price and started:
            now = datetime.datetime.now(datetime.timezone.utc)
            h, m, sec = (int(x) for x in started.groups())
            began = now.replace(hour=h, minute=m, second=sec)
            boxes.append((float(price.group(1)), max(0.0, (now - began).total_seconds() / 3600)))
    return boxes


def purpose_spend(ledger_text: str, purpose: str, since: str) -> float:
    """USD booked by boxes of one purpose since an ISO time, from the ledger's own rows (RCA R6:
    the ledger total is the whole account; other agents book into it too)."""
    total = 0.0
    for line in ledger_text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7 or not cells[0].startswith("20") or cells[0] < since:
            continue
        event = cells[3]
        name = event.split(" (")[0]
        if not (name == purpose or name.startswith(f"{purpose}-")) or "started" in event:
            continue
        cost = re.findall(r"\$([0-9.]+)", cells[6])
        if cost and "cap" not in cells[6]:
            total += float(cost[-1])
    return total


def status(runs: list[Path], since: str | None, group: bool = False) -> None:
    """BP-3.11 (owner, 2026-11-24): measured progress, not hand estimates. Per run and stage:
    done/total, the rate over the last 10 minutes, the ETA at that rate; the boxes running and
    their unbooked cost; spend against the plan and the projected total."""
    import time as clock
    import report
    from aws_session import LEDGER
    now, hourly, unbooked, worst_eta = clock.time(), 0.0, 0.0, 0.0
    measured = measured_durations(runs[0].parent.parent)
    for run in runs:
        planned = json.loads((run / "jobs.json").read_text())
        jobs = {j["id"] for j in planned}
        for base in sorted((run / "series").iterdir()):
            by_box = {box.name: box_records(box) for box in base.glob("box*") if box.is_dir()}
            records = [r for recs in by_box.values() for r in recs]
            valid = valid_ids([run], base.name)
            enc = stage_progress({b: [r for r in recs if report.is_job(r)]
                                  for b, recs in by_box.items()}, len(jobs), valid & jobs, now)
            # RCA R10: the encode ETA from the remaining expected seconds over the boxes up
            left = [j for j in planned if j["id"] not in valid]
            expected_seconds(left, measured)
            encode_boxes = max(1, sum(1 for log in base.glob("slot*.log")
                                      for _ in running_boxes_in(log)))
            enc["eta_min"] = round(sum(j["expected_s"] for j in left) / BOX_BUSY_SHARE
                                   / encode_boxes / 60) if left else 0
            score_groups = {f.parent.name: [json.loads(line) for line in f.read_text().splitlines()
                                            if line.strip().endswith("}")]
                            for f in base.glob("score*/scores.jsonl")}
            scores = [r for recs in score_groups.values() for r in recs]
            have = {r["id"] for r in scores if "pooled" in r} | scored(base)
            coded = {r["id"] for r in records if report.is_job(r)} | (
                {r["id"] for r in report.read_jsonl(base / "encodes.jsonl") if report.is_job(r)}
                if (base / "encodes.jsonl").exists() else set())
            sc = stage_progress(score_groups, len(coded & jobs), have & jobs, now)
            boxes = running_boxes(base)
            hourly += sum(p for p, _ in boxes)
            unbooked += sum(p * h for p, h in boxes)
            worst_eta = max(worst_eta, enc["eta_min"] or 0, sc["eta_min"] or 0)
            print(f"{run.parent.name}/{base.name}: encode {enc['done']}/{enc['total']} valid, "
                  f"{enc['per_hour']}/h, ETA {enc['eta_min']} min | score {sc['done']}/"
                  f"{sc['total']}, {sc['per_hour']}/h, ETA {sc['eta_min']} min | "
                  f"{len(boxes)} boxes up")
    if group:  # one dataset's runs: the counts `missing` gives
        series = sorted(p.name for p in (runs[0] / "series").iterdir())[0]
        planned = json.loads((runs[0] / "jobs.json").read_text())
        left = missing(runs, series)
        print(f"all {len(runs)} runs ({series}): {len(planned) - len(left)} of {len(planned)} "
              f"planned jobs valid; {len(left)} missing")
    if since is not None:
        mine = purpose_spend(LEDGER.read_text(), PURPOSE, since)  # every bench-* purpose
        print(f"benchmark ({PURPOSE}) booked since {since}: ${mine:.2f}; running boxes about "
              f"${unbooked:.2f} unbooked at ${hourly:.2f}/h; projected "
              f"${mine + unbooked + hourly * worst_eta / 60:.2f} at the slowest ETA "
              f"({worst_eta:.0f} min)")


#: Measured unit costs (docs/analysis/bp3-run-lessons-2026-11-24.md section 3), USD per job:
#: encode-and-time on c8i.2xlarge, score on c8g.8xlarge; by the job's picture size.
UNIT_COST = {"encode_720p": 0.03e-3, "encode_1080p": 0.35e-3,
             "score_720p": 0.067e-3, "score_1080p": 0.147e-3}
#: The envelope's margin over the unit-cost estimate: today's run came in 25 % over its plan
#: after all fixes (reclaims, setup, re-codes).
ENVELOPE_MARGIN = 0.25
#: The lowest AWS spot placement score (1-10) preflight accepts for a fleet's size in its
#: allowed zones.
MIN_PLACEMENT_SCORE = 5
HD_PIXELS = 1920 * 1080


def reclaim_rates(bench: Path) -> dict[tuple[str, str], tuple[int, float]]:
    """(reclaims, box-hours) per (type, zone) from every run's box logs."""
    rates: dict[tuple[str, str], list] = defaultdict(lambda: [0, 0.0])
    for log in bench.glob("bp3-*/run/series/*/*.log"):
        text = log.read_text(errors="replace")
        for session in re.split(r"(?=\[aws \S+\] launched )", text)[1:]:
            launched = re.match(r"\[aws \S+\] launched \S+ \((\S+)\) in (\S+)", session)
            cost = re.search(r"session cost \$[0-9.]+ \(([0-9.]+) min", session)
            if launched and cost:
                key = (launched.group(1), launched.group(2))
                rates[key][1] += float(cost.group(1)) / 60
                # a reclaim: the instance was terminated under it, or box.py stopped on the
                # spot notice (the slot log's "boxN-M: interrupted")
                rates[key][0] += "SpotInstanceTermination" in session or \
                    re.search(r"box\d+-[\d-]+: interrupted", session) is not None
    return {k: (v[0], v[1]) for k, v in rates.items()}


def placement_scores(ec2, instance_type: str, boxes: int, region: str) -> dict:
    """AWS spot placement scores for `boxes` instances; the account caps the capacity it may
    explore, so the request halves until AWS accepts it (the reply names the size scored)."""
    from botocore.exceptions import ClientError
    while True:
        try:
            reply = ec2.get_spot_placement_scores(
                InstanceTypes=[instance_type], TargetCapacity=boxes,
                SingleAvailabilityZone=True, RegionNames=[region])
            return {**reply, "boxes": boxes}
        except ClientError as error:
            if error.response["Error"]["Code"] != "TargetCapacityLimitExceeded" or boxes <= 1:
                raise
            boxes //= 2


#: Fixed QP streams' bytes per coded pixel, for the disk check: the mean over the AOM-CTC
#: Fixed QP run of 2026-11-24 (0.0156; "ours" Fixed QP 0.0101), the larger of the two.
QP_BYTES_PER_PIXEL = 0.0156
#: A CBR stream's bytes over its target's (the strict bound allows a little overshoot).
CBR_STREAM_MARGIN = 1.1


def expected_stream_bytes(job: dict) -> float:
    """A job's stream size before it is coded, for the preflight's disk check (the PC keeps
    every stream until it is scored)."""
    num, den = (int(x) for x in job["fps"].split("/"))
    if job["mode"] == "cbr":
        return job["kbps"] * 1000 / 8 * job["frames"] * den / num * CBR_STREAM_MARGIN
    return job["w"] * job["h"] * job["frames"] * QP_BYTES_PER_PIXEL


def preflight(a) -> None:
    """BP-3.11 (owner, 2026-11-24): everything a full run needs, checked once before launch, so
    nothing holds it midway: disk, quota and the benchmark cap, the PC's address, spot prices
    and reclaim rates by zone, and the cost envelope (unit-cost estimate + margin) to approve
    once. Refuses on any failed check."""
    from aws_session import public_ip, purpose_vcpus, spot_quota
    import boto3
    failures = []
    free = shutil.disk_usage(a.runs[0]).free
    print(f"disk: {free >> 30} GiB free (floor {MIN_FREE_BYTES >> 30})")
    if free < MIN_FREE_BYTES:
        failures.append("disk below the floor")
    ec2 = boto3.client("ec2", region_name=a.region)
    quota, mine = spot_quota(a.region), purpose_vcpus(ec2, PURPOSE)
    print(f"spot quota {quota} vCPU; benchmark holds {mine} of its {a.vcpu_cap} cap")
    from aws_session import LEDGER
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    print(f"benchmark ({PURPOSE}) booked today: ${purpose_spend(LEDGER.read_text(), PURPOSE, today):.2f}")
    try:
        print(f"PC address {public_ip()} (each box's security group admits it)")
    except OSError as error:
        failures.append(f"no public address: {error}")
    zone_names = {z["ZoneId"]: z["ZoneName"]
                  for z in ec2.describe_availability_zones()["AvailabilityZones"]}
    for instance_type, boxes in ((TIMING_TYPE, a.timing_boxes), (ARM_SCORE_TYPES[0], a.score_boxes)):
        # RCA R7: spot capacity in each allowed zone (AWS placement score, 1-10). Printed as the
        # largest fleet that scores MIN_PLACEMENT_SCORE somewhere allowed: a warning, not a
        # refusal (on 2026-11-24 the score was 1/10 for 50 c8i while the fleets held 100+)
        size, best = boxes, 0
        while size >= 1:
            reply = placement_scores(ec2, instance_type, size, a.region)
            size = reply["boxes"]
            scores = {zone_names.get(r.get("AvailabilityZoneId"), r.get("AvailabilityZoneId")):
                      r["Score"] for r in reply["SpotPlacementScores"]}
            allowed = {z: v for z, v in scores.items()
                       if instance_type != TIMING_TYPE or z not in AVOID_ZONES}
            best = max(allowed.values(), default=0)
            if best >= MIN_PLACEMENT_SCORE:
                break
            size //= 2
        print(f"capacity {instance_type}: {size} boxes score {best}/10 in an allowed zone "
              f"({', '.join(f'{z} {v}' for z, v in sorted(scores.items()))})"
              + ("" if size >= boxes else f"; WARNING: below the planned {boxes}, and the zone "
                 "fallback (RCA R7) will use the avoided zone"))
    for instance_type in (TIMING_TYPE, *ARM_SCORE_TYPES, "c8g.2xlarge"):
        history = ec2.describe_spot_price_history(
            InstanceTypes=[instance_type], ProductDescriptions=["Linux/UNIX"],
            StartTime=datetime.datetime.now(datetime.timezone.utc))["SpotPriceHistory"]
        prices = {h["AvailabilityZone"]: float(h["SpotPrice"]) for h in history}
        print(f"{instance_type}: " + ", ".join(f"{z} ${p:.4f}/h" + (" (avoided)" if z in AVOID_ZONES else "")
                                               for z, p in sorted(prices.items())))
    for (instance_type, zone), (n, hours) in sorted(reclaim_rates(a.runs[0].parent.parent).items()):
        if hours >= 1:
            print(f"reclaims {instance_type} {zone}: {n} in {hours:.1f} box-h ({n / hours:.2f}/h)")
    import wholebox
    estimate = 0.0
    for run in a.runs:
        plan_path = run / wholebox.PLAN_FILE
        if plan_path.exists():
            # BP-4: the timing plan prices the whole machines (D13); scoring is local (D16)
            plan = json.loads(plan_path.read_text())
            estimate += plan["expected_cost_usd"]
            print(f"{run.name}: timing plan {plan['instance_type']}, "
                  f"{plan['expected_machine_hours']:.2f} machine-h, "
                  f"{plan['expected_wall_s'] / 3600:.2f} h wall, ${plan['expected_cost_usd']:.2f}")
            continue
        for job in json.loads((run / "jobs.json").read_text()):
            size = "1080p" if job["w"] * job["h"] >= HD_PIXELS * 0.9 else "720p"
            estimate += UNIT_COST[f"encode_{size}"] + UNIT_COST[f"score_{size}"]
    envelope = estimate * (1 + ENVELOPE_MARGIN)
    print(f"cost estimate ${estimate:.2f}; envelope to approve ${envelope:.2f} "
          f"(+{ENVELOPE_MARGIN:.0%})")
    spent = account_spent()
    print(f"account: ${spent:.2f} booked; with the envelope ${spent + envelope:.2f} "
          f"(stop line ${ACCOUNT_STOP_USD:.0f})")
    if spent + envelope > ACCOUNT_STOP_USD:
        failures.append("the envelope passes the account's stop line")
    streams = sum(expected_stream_bytes(job) for run in a.runs
                  for job in json.loads((run / "jobs.json").read_text()))
    print(f"streams: about {streams / 2**30:.0f} GiB expected; {int(free - streams) >> 30} GiB "
          f"free after them (floor {MIN_FREE_BYTES >> 30})")
    if free - streams < MIN_FREE_BYTES:
        failures.append("the runs' streams would take the disk below the floor")
    if failures:
        raise SystemExit("refused: " + "; ".join(failures))
    if a.approve is not None:
        if a.approve < envelope:
            raise SystemExit(f"refused: the approval ${a.approve:.2f} is below the envelope")
        record_gate("approval", runs=[str(r) for r in a.runs], envelope=round(envelope, 2),
                    approved_usd=a.approve, approved_by=a.approved_by)
        print(f"approval of ${a.approve:.2f} by {a.approved_by} recorded for this tools commit")


#: The rehearsal adds the run's longest jobs, so the tail of long encodes is rehearsed too.
REHEARSAL_LONG_JOBS = 4


def rehearsal_jobs(jobs: list[dict], clips: int) -> list[dict]:
    """The rehearsal's sample of a run: every job of `clips` clips (whole clips, so the report
    can draw every curve): the largest clip (the slowest encodes and the heaviest scoring) and
    the smallest (the shortest encodes, where V7's clock rule bit); plus the run's
    REHEARSAL_LONG_JOBS longest jobs by expected seconds (RCA R10: a tail of long jobs)."""
    sizes = {j["input"]: j["w"] * j["h"] for j in jobs}
    order = sorted(sizes, key=lambda c: (sizes[c], c))
    chosen = {order[-1], order[0], *order[1:max(0, clips - 1)]}
    longest = {j["id"] for j in sorted(jobs, key=lambda j: -j.get("expected_s", 0.0))
               [:REHEARSAL_LONG_JOBS]}
    return [j for j in jobs if j["input"] in chosen or j["id"] in longest]


def rehearse_half(a) -> None:
    """T22 for BP-4: a tiny sample of each run on one whole machine of the series' type, with a
    forced reclaim after its first settled family (a replacement codes the rest), then BP-4's
    V6/V24 on the merged records. Encoding and timing only (scoring is its own step)."""
    import argparse as ap_
    import report
    import wholebox
    for source in a.runs:
        target = a.out / source.parent.name
        planned = json.loads((source / "jobs.json").read_text())
        expected_seconds(planned, measured_durations(source.parent.parent))
        jobs = rehearsal_jobs(planned, a.clips)
        sample = a.out / f"{source.parent.name}-jobs.json"
        a.out.mkdir(parents=True, exist_ok=True)
        sample.write_text(json.dumps(jobs))
        stage(target, sample, source / "stage" / "bin", source / "stage" / "cfg",
              source / "stage" / "clips")
        common = dict(run=target, series=a.series, machines=1, price=None)
        plan_timing(ap_.Namespace(**common))
        run_half(ap_.Namespace(**common, budget=a.budget, root_gb=40, plan="rehearsal",
                               chaos=True, rehearsal=True, ungated=None))
        records = [json.loads(line) for line in
                   (target / "series" / a.series / "encodes.jsonl").read_text().splitlines()]
        valid, refused = report.half_records(a.series, records)
        if refused:
            raise SystemExit(f"refused: rehearsal of {source.parent.name}: {refused}")
        say(f"rehearsal of {source.parent.name}: {len(valid)} valid times of {len(jobs)} jobs "
            f"over {len({r['box'] for r in valid})} machines")
    record_gate("rehearsal", runs=[str(r) for r in a.runs], out=str(a.out), layout="half")


def rehearse(a) -> None:
    """BP-3.11 (owner, 2026-11-24): the whole pipeline on a sample before any full run, on the
    same commit of the tools: stage, encode and time (2 boxes, one forced reclaim), validity as
    records arrive, scoring (one Arm box), and the report, for every run given. A full run is
    launched only after this passes; every failure of 2026-11-24 would have shown here."""
    import argparse as ap_
    import report
    if getattr(a, "layout", "queue") == "half":
        return rehearse_half(a)
    for source in a.runs:
        target = a.out / source.parent.name
        planned = json.loads((source / "jobs.json").read_text())
        expected_seconds(planned, measured_durations(source.parent.parent))
        jobs = rehearsal_jobs(planned, a.clips)
        sample = a.out / f"{source.parent.name}-jobs.json"
        a.out.mkdir(parents=True, exist_ok=True)
        sample.write_text(json.dumps(jobs))
        stage(target, sample, source / "stage" / "bin", source / "stage" / "cfg",
              source / "stage" / "clips")
        common = dict(run=target, series=a.series, region=TIMING_REGION, budget=0.3,
                      lifetime=45, root_gb=40, on_demand=False, packed=0,
                      discard_streams=False, vcpu_cap=a.vcpu_cap, background_share=0.0,
                      canaries=3, score_workers=0, rescore=None, ids=None, no_merge=False,
                      verify_scorer=False,
                      score_bin=None, tag="", type=None, type_given=False)
        run(ap_.Namespace(**common, step="run", boxes=2, chaos=True, rehearsal=True, ungated=None,
                          target_minutes=15,
                          types=(TIMING_TYPE,), fleet_budget=None))
        # a scoring box's own budget (an Arm box in its dearer zones, or the x86 fallback,
        # passes the timing boxes' $0.30: seven history rehearsals launched no scoring box)
        score(ap_.Namespace(**{**common, "budget": SCORE_BOX_BUDGET}, step="score", boxes=1,
                            types=ARM_SCORE_TYPES))
        # a run of ours alone (competitors reused from earlier runs) has no zero line: its
        # rehearsal checks every point (V6-V13, V23-V25) without the BD
        alone = not any(j["family"] in report.ZERO_FAMILY.values() for j in jobs)
        for mode in sorted({j.get("mode") for j in jobs}):  # a run of both modes: each alone
            result = report.report(target, checks_only=alone, mode=mode)
        say(f"rehearsal of {source.parent.name}: {len(jobs)} jobs passed; report series "
            f"{sorted(result['series'])}")
    record_gate("rehearsal", runs=[str(r) for r in a.runs], out=str(a.out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    st = sub.add_parser("stage")
    st.add_argument("run", type=Path)
    st.add_argument("--jobs", type=Path, required=True)
    st.add_argument("--bin", type=Path, required=True)
    st.add_argument("--cfg", type=Path, required=True)
    st.add_argument("--clips", type=Path, required=True, help="<name>.yuv.zst per clip")
    pr = sub.add_parser("prune")
    pr.add_argument("run", type=Path)
    pr.add_argument("--series", required=True)
    ms = sub.add_parser("missing")
    ms.add_argument("runs", type=Path, nargs="+")
    ms.add_argument("--series", required=True)
    ms.add_argument("--out", type=Path, required=True)
    pf = sub.add_parser("preflight", help="the checks and cost envelope before a run (T18)")
    pf.add_argument("runs", type=Path, nargs="+")
    pf.add_argument("--region", default=TIMING_REGION)
    pf.add_argument("--vcpu-cap", type=int, default=BENCH_VCPU_CAP)
    pf.add_argument("--timing-boxes", type=int, default=100, help="the timing fleet's size")
    pf.add_argument("--score-boxes", type=int, default=20, help="the scoring fleet's size")
    pf.add_argument("--approve", type=float,
                    help="record the approved envelope (USD) for this tools commit (RCA R1)")
    pf.add_argument("--approved-by", default="the owner",
                    help="who approved it (as relayed), recorded with the approval")
    rh = sub.add_parser("rehearse", help="the whole pipeline on a sample of each run (BP-3.11)")
    rh.add_argument("runs", type=Path, nargs="+", help="staged runs to rehearse")
    rh.add_argument("--out", type=Path, required=True)
    rh.add_argument("--series", default="c8i")
    rh.add_argument("--clips", type=int, default=2, help="clips per run (whole clips)")
    rh.add_argument("--vcpu-cap", type=int, default=BENCH_VCPU_CAP)
    rh.add_argument("--layout", choices=("queue", "half"), default="queue",
                    help="half: BP-4 whole machines (encode and timing only)")
    rh.add_argument("--budget", type=float, default=1.0, help="(half) USD per machine")
    stt = sub.add_parser("status", help="measured progress, ETAs and spend (BP-3.11)")
    stt.add_argument("runs", type=Path, nargs="+")
    stt.add_argument("--since", help="the plan's start (ISO, UTC): spend from the benchmark's "
                                     "own ledger rows since then")
    stt.add_argument("--group", action="store_true",
                     help="the runs are one dataset's: print their combined count (= missing)")
    pt = sub.add_parser("plan-timing", help="BP-4: the whole-machine timing plan (D13-D15)")
    pt.add_argument("run", type=Path)
    pt.add_argument("--series", required=True, help="c8i or c8g (the chip)")
    pt.add_argument("--machines", type=int, default=1)
    pt.add_argument("--price", type=float, help="USD/h (default: today's home spot price)")
    sb = sub.add_parser("set-binary", help="replace a pending staged binary (V2 hash updated)")
    sb.add_argument("run", type=Path)
    sb.add_argument("--name", required=True)
    sb.add_argument("--binary", type=Path, required=True)
    dd = sub.add_parser("dedupe-stage")
    dd.add_argument("run", type=Path)
    dd.add_argument("--clips", type=Path, required=True)
    for name in ("run", "score"):
        add_run_args(sub.add_parser(name))
    a = ap.parse_args()
    if a.step == "score" and not a.boxes:
        a.boxes = 1
    if hasattr(a, "type"):
        a.type_given = bool(a.type)
        a.types = resolve_types(a)
        a.type = a.types[0]
    if a.step == "stage":
        stage(a.run, a.jobs, a.bin, a.cfg, a.clips)
    elif a.step == "missing":
        jobs = missing(a.runs, a.series)
        a.out.write_text(json.dumps(jobs, indent=0))
        say(f"{len(jobs)} jobs over {len({j['input'] for j in jobs})} clips -> {a.out}")
    elif a.step == "preflight":
        preflight(a)
    elif a.step == "rehearse":
        require_capped_scope()
        rehearse(a)
    elif a.step == "status":
        status(a.runs, a.since, a.group)
    elif a.step == "plan-timing":
        plan_timing(a)
    elif a.step == "set-binary":
        set_binary(a.run, a.name, a.binary)
    elif a.step == "dedupe-stage":
        say(f"dedupe: {dedupe_stage(a.run, a.clips) / 1e9:.1f} GB freed (BP-3.5)")
    elif a.step == "prune":
        files, size = prune(a.run, a.series)
        say(f"pruned {files} scored Fixed QP streams, {size / 1e9:.1f} GB (BP-3.5)")
    elif a.step == "score":
        score(a)
    elif getattr(a, "layout", "queue") == "half":
        run_half(a)
    else:
        run(a)


def add_run_args(rn) -> None:
    rn.add_argument("run", type=Path)
    rn.add_argument("--fleet-budget", type=float,
                    help="(run) USD for all boxes, replacements included (default 1.5 x boxes x "
                         "budget)")
    rn.add_argument("--score-bin", type=Path,
                    help="(score) the decoders and libvmaf for the scoring boxes' architecture "
                         "(default: the run's stage/bin)")
    rn.add_argument("--ids", type=Path, help="(score) score exactly these stream ids (JSON list)")
    rn.add_argument("--no-merge", action="store_true",
                    help="(score) keep this pass's scores in its own dirs (a parity check)")
    rn.add_argument("--ungated", metavar="REASON",
                    help="(run) launch without a rehearsal and approval for this tools commit; "
                         "printed and recorded (RCA R1)")
    rn.add_argument("--plan", help="the plan's name: boxes are tagged Purpose bench-<plan>, so "
                                   "its spend is its own ledger rows (RCA R6)")
    rn.add_argument("--vcpu-cap", type=int, default=BENCH_VCPU_CAP,
                    help="the benchmark's share of the region's spot quota, all fleets together "
                         f"(default {BENCH_VCPU_CAP})")
    rn.add_argument("--background-share", type=float, default=0.0,
                    help="(run) the packing test: this share of slots load their idle cores")
    rn.add_argument("--canaries", type=int, default=3,
                    help="(run) canary jobs per box, from its first unit (V6)")
    rn.add_argument("--on-demand", action="store_true",
                    help="(run) timing boxes on demand; the default is spot (BP-3.10: reclaims "
                         "are filled by `missing`)")
    rn.add_argument("--packed", type=int, default=0,
                    help="(run) an untimed quality run, this many encodes at once (BP-3.8)")
    rn.add_argument("--discard-streams", action="store_true",
                    help="(run) a re-timing run: hashes and sizes only, no streams kept")
    rn.add_argument("--rescore", help="(score) score these families again (fnmatch)")
    rn.add_argument("--local", action="store_true",
                    help="(score) on this PC's GPU scoring service (BP-4.0 D16); refuses a "
                         "scorer build not verified by tools/bench/verify_scorer.py (D19)")
    rn.add_argument("--follow", type=float, default=0, metavar="MINUTES",
                    help="(score --local) repeat passes while streams arrive; stop after this "
                         "many minutes with nothing new")
    rn.add_argument("--verify-scorer", action="store_true", help=argparse.SUPPRESS)
    rn.add_argument("--tag", default="", help="(score) a name for this scoring pass's dirs")
    rn.add_argument("--series", required=True)
    rn.add_argument("--type", help=f"default {TIMING_TYPE} for timing, {ARM_SCORE_TYPES[0]} "
                                   f"(x86 {' or '.join(UNTIMED_TYPES)} as fallback) for scoring, "
                                   f"{' or '.join(UNTIMED_TYPES)} for --packed (the cost guide)")
    rn.add_argument("--region", default=TIMING_REGION, help=f"default {TIMING_REGION}")
    rn.add_argument("--boxes", type=int, default=0,
                    help="boxes; 0 (default): from the expected seconds and --target-minutes "
                         "(run), 1 (score)")
    rn.add_argument("--target-minutes", type=float, default=30,
                    help="(run) the wall time the fleet is sized for (RCA R10)")
    rn.add_argument("--score-workers", type=int, default=0,
                    help="scoring workers per box; 0 (default): as many as the box's CPUs and "
                         "memory allow")
    rn.add_argument("--layout", choices=("queue", "half"), default="queue",
                    help="(run) half: BP-4 whole bare-metal machines at half load, from "
                         "`plan-timing` (D13); queue: BP-3's one encode per small box")
    rn.add_argument("--budget", type=float, default=1.0)
    rn.add_argument("--adopt", metavar="INSTANCE_ID",
                    help="(run, half) serve with a live machine whose driver is gone, instead of "
                         "launching one (its key pair file must still be in work/aws)")
    rn.add_argument("--chain", type=lambda v: [x for x in v.split(",") if x], default=[],
                    help="(run, half) the chip's other runs, in priority order: a machine "
                         "whose own run is done (or served elsewhere) takes the next unfinished, "
                         "unclaimed one instead of ending")
    rn.add_argument("--lifetime", type=int, default=90)
    rn.add_argument("--root-gb", type=int, default=16)


if __name__ == "__main__":
    main()
