#!/usr/bin/env python3
"""The box side of a BP-3 encode stage: one timed encode per job, every check refusing.

    python3 -I box.py --inbox inbox/ --manifest manifest.json --cpu 1 --out out/

Runs on a timing box (docs/benchmark-procedure.md sections 6 and 10), standard library
only, beside `bin/` (the manifest's binaries), `cfg/`, the clips as raw YUV on tmpfs, and
a copy of `nceval/streamstrip.py` and `nceval/codecs.py` (`fleet` uploads them).

The single-run rule: each job's one encode gives both its time and its stream. Per encode:

  - V2  every binary's SHA-256 equals the manifest's (once, before the first encode);
  - V4  the input's SHA-256 equals the job's, when the plan recorded one, and it holds exactly
        the job's frames (V9: libaom's example program codes every frame it is given);
  - V5  the input is on tmpfs and wholly resident (`fincore`) before the encode;
  - V8  the encoder runs in a cpuset cgroup holding one CPU, checked from its own
        Cpus_allowed_list while it runs: its threads, however many, share that one CPU. A
        sampler records how many threads did work (SVT-AV1 at --lp 1 runs several, in turn);
        the plan stage's unpinned audit (CPU <= 1.02 x wall) is what shows a family is serial;
  - V7  steal on that CPU under MAX_STEAL, wall over CPU time at most MAX_WALL_OVER_CPU,
        no major page fault;
  - V9  exit status 0 and a non-empty stream;
  - V21 the encoder's banner matches the job's `banner` pattern, when the record gives one;
  - V19 after the timed span, SEI and metadata are stripped (`nceval.streamstrip`), the bytes
        removed recorded, and the stream's SHA-256 taken from the stripped stream.

BP-3.11, work from a queue: the box takes its jobs as small units that the driver (`fleet.py`)
drops into `--inbox` (`unit-*.json`, each one clip's jobs; the driver puts the clip on tmpfs
first), until the driver writes `inbox/END`. Each record is appended to `out/encodes.jsonl` as
its job ends, and the driver downloads `out/` every half minute, so a lost box costs only the
job in progress. Before each job the box reads the spot interruption notice (instance metadata;
about two minutes' warning): on a notice it takes no more work, codes its timing references
(so the jobs since the last ones keep a valid time) and exits.

Before a tool's first timed encode, it codes one untimed warm-up encode (`Warm`), so no
binary is faulted in from disk during a timed one. V6's canary sets run at the start, after
CANARY_MIDDLE jobs and at the end (`--canaries`); the report compares them. A job whose timing
check fails (V7, V8) is re-run once at the end of its unit; a second failure, or any other
check, refuses the run: the box exits non-zero, and every record written so far stays for the
audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
from nceval.codecs import Codec  # noqa: E402  (shipped beside this file)
from nceval.streamstrip import strip  # noqa: E402
import pinning  # noqa: E402  (shipped beside this file)

#: V7 (section 10): steal share of the encode CPU, wall over CPU time. The wall time includes
#: the fork, exec and cgroup move, a fixed cost that a short encode cannot hide in 3 %
#: (0.18 s encodes of 30 frames measured 1.05x), so up to MAX_WALL_SLACK_S of it is allowed.
MAX_STEAL = 0.005
MAX_WALL_OVER_CPU = 1.03
MAX_WALL_SLACK_S = 0.02
#: V8: a thread "did work" when its CPU time grew by more than this during the encode
#: (the encoder's own idle helper threads, if any, stay under it).
THREAD_WORK_S = 0.01
SAMPLE_S = 0.05
CGROUP_ROOT = Path("/sys/fs/cgroup")
CGROUP_NAME = "nanocodec-bench"
STDERR_KEPT = 2048
#: Each canary is coded this many times at each point; the report takes the median, so one
#: noisy short encode does not trip V6 (the pipe-clean's 0.2 s canaries moved 5 % alone).
CANARY_REPEATS = 3
#: BP-3.6: both timing references are coded after every REFERENCE_EVERY jobs, so the report
#: can normalise each job by the box's speed around it (neighbour interference changes the
#: cycles per instruction on a shared host: response 2's drift diagnosis).
REFERENCE_EVERY = 20
#: ... and whenever this much encode time has passed since the last pair, so the time a refused
#: window can cost is bounded in minutes, not jobs (owner, 2026-11-24: with 1080p encodes, 20
#: jobs could be an hour unprotected).
REFERENCE_MAX_S = 300.0
PROGRESS_EVERY = 10
#: V6's middle canary set runs after this many jobs (a queue box does not know its total).
CANARY_MIDDLE = 150
#: How often an idle box looks in its inbox.
INBOX_POLL_S = 1.0
#: The spot interruption notice (IMDSv2): present about two minutes before a reclaim.
IMDS = "http://169.254.169.254/latest"
IMDS_TIMEOUT_S = 0.5
CLK_TCK = os.sysconf("SC_CLK_TCK")
STEAL_FIELD = 7  # /proc/stat cpuN: user nice system idle iowait irq softirq steal
#: macOS (the Mac series, 2026-11-26): no /proc, no CPU pinning (the kernel offers only affinity
#: hints), bare metal with no steal. The clips sit on a RAM disk (`hdiutil attach ram://`), not
#: tmpfs. V8 cannot hold there: the record states that encodes are not pinned.
DARWIN = sys.platform == "darwin"
#: macOS: the P-cluster clock moves 1.7-4.0 GHz with load while an encode's cycles stay constant
#: (the Mac series' TIMING-METHOD.md), so an encode's time there is its cycles at this fixed clock
#: (the M4's measured single-encode P clock), its raw CPU time recorded beside it.
DARWIN_REF_HZ = 4.0e9
#: macOS V7: an encode that spent less than this share of its CPU time on P-cores is coded again.
DARWIN_MIN_P_SHARE = 0.98
#: proc_pid_rusage's RUSAGE_INFO_V6 (sys/resource.h), up to the fields read here.
DARWIN_RUSAGE_FIELDS = (
    "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size "
    "phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time "
    "child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime "
    "diskio_bytesread diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance "
    "cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy "
    "cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time "
    "serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles "
    "billed_energy serviced_energy interval_max_phys_footprint runnable_time flags user_ptime "
    "system_ptime pinstructions pcycles energy_nj penergy_nj secure_time_in_system "
    "secure_ptime_in_system neural_footprint lifetime_max_neural_footprint "
    "interval_max_neural_footprint").split()
RUSAGE_INFO_V6 = 6


class Refusal(RuntimeError):
    """A check failed: the run stops (section 10, "each refuses")."""


class JobFailure(Refusal):
    """One job failed (V9 exit or empty stream, V21 banner, its timeout): the box records it
    and goes on; the driver counts its attempts and quarantines it after two boxes (review #5:
    one failing job used to stop the whole fleet)."""


class TimingRefusal(Refusal):
    """A timing check failed (V7, V8): the job is re-run once before the run refuses."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_binaries(manifest: dict, bin_dir: Path) -> dict[str, str]:
    """V2: the binaries are the manifest's; their SHA-256 by name (each record carries its own
    binary's, V25)."""
    for name, digest in manifest["binaries"].items():
        if sha256(bin_dir / name) != digest:
            raise Refusal(f"V2: {name}'s SHA-256 is not the manifest's")
    return dict(manifest["binaries"])


def mount_type(path: Path) -> str:
    best, kind = "", ""
    for line in Path("/proc/mounts").read_text().splitlines():
        _, point, fstype, *_ = line.split()
        if str(path).startswith(point.rstrip("/") + "/") and len(point) > len(best):
            best, kind = point, fstype
    return kind


def on_ram_disk(path: Path) -> bool:
    """macOS: `path`'s volume is a RAM disk (`hdiutil attach ram://`), so wholly in memory."""
    device = subprocess.run(["df", str(path)], capture_output=True, text=True,
                            check=True).stdout.splitlines()[-1].split()[0]
    info = subprocess.run(["hdiutil", "info"], capture_output=True, text=True, check=True).stdout
    return any("ram://" in block and device.removeprefix("/dev/").rstrip("s0123456789") in block
               for block in info.split("=" * 10))


def check_resident(path: Path) -> None:
    """V5: on tmpfs and every byte in memory (macOS: on a RAM disk)."""
    if DARWIN:
        if not on_ram_disk(path):
            raise Refusal(f"V5: {path} is not on a RAM disk")
        return
    if mount_type(path) != "tmpfs":
        raise Refusal(f"V5: {path} is not on tmpfs")
    def resident() -> int:
        out = subprocess.run(["fincore", "--bytes", "--noheadings", "--output", "RES", str(path)],
                             capture_output=True, text=True, check=True).stdout.split()
        return int(out[0])
    # fincore counts whole pages: a file whose size is not a page multiple shows its last page.
    # A small machine (the Jetson, 8 GB with zram) may swap a few tmpfs pages out under load:
    # read the file once to bring them back, then judge (2026-11-26: two batches refused for
    # ~100 KB of a 200 MB clip)
    if resident() < path.stat().st_size:
        with open(path, "rb") as f:
            while f.read(1 << 24):
                pass
    if resident() < path.stat().st_size:
        raise Refusal(f"V5: {path} is not wholly resident ({resident()} of {path.stat().st_size} bytes)")


def pin_harness(cpu: int) -> int:
    """V8: this harness (and its sampler) runs on a CPU of another physical core than the
    timing CPU, never on its SMT sibling (pinning.pin_harness). Returns its CPU."""
    try:
        return pinning.pin_harness(cpu)
    except RuntimeError as error:
        raise Refusal(f"V8: {error}") from error


def cgroup(cpu: int) -> Path:
    """V8: a cgroup v2 cpuset holding only `cpu`."""
    (CGROUP_ROOT / "cgroup.subtree_control").write_text("+cpuset")
    group = CGROUP_ROOT / f"{CGROUP_NAME}-{cpu}"
    group.mkdir(exist_ok=True)
    (group / "cpuset.cpus").write_text(str(cpu))
    (group / "cpuset.mems").write_text("0")
    if (group / "cpuset.cpus.effective").read_text().strip() != str(cpu):
        raise Refusal(f"V8: cgroup {group} does not hold CPU {cpu} alone")
    return group


def concrete(job: dict, paths: dict[str, Path]) -> list[str]:
    out = []
    for arg in job["argv"]:
        for key, value in paths.items():
            arg = arg.replace("{" + key + "}", str(value))
        out.append(arg)
    return out


def cpu_ticks(cpu: int) -> list[int]:
    if DARWIN:
        return [0] * (STEAL_FIELD + 1)  # bare metal, no steal to count
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith(f"cpu{cpu} "):
            return [int(x) for x in line.split()[1:]]
    raise Refusal(f"cpu{cpu} not in /proc/stat")


def darwin_reap(pid: int) -> tuple[int, dict]:
    """macOS: wait for `pid` without reaping it, read its RUSAGE_INFO_V6 from the zombie (CPU
    and P-core time, cycles, instructions), then reap it. Returns (wait status, measures)."""
    import ctypes

    class RusageInfoV6(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + \
            [(name, ctypes.c_uint64) for name in DARWIN_RUSAGE_FIELDS] + \
            [("reserved", ctypes.c_uint64 * 9)]

    libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
    timebase = (ctypes.c_uint32 * 2)()
    libc.mach_timebase_info(timebase)
    os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
    info = RusageInfoV6()
    if libc.proc_pid_rusage(pid, RUSAGE_INFO_V6, ctypes.byref(info)) != 0:
        raise Refusal(f"proc_pid_rusage failed for {pid}")
    _, status = os.waitpid(pid, 0)
    seconds = lambda ticks: ticks * timebase[0] / timebase[1] / 1e9  # noqa: E731
    cpu = seconds(info.user_time + info.system_time)
    return status, {"user_s": seconds(info.user_time), "sys_s": seconds(info.system_time),
                    "cpu_time_s": cpu, "cycles": info.cycles, "instructions": info.instructions,
                    "p_share": seconds(info.user_ptime + info.system_ptime) / cpu if cpu else 0.0,
                    "ghz": info.cycles / cpu / 1e9 if cpu else 0.0, "pageins": info.pageins}


def thread_times(pid: int) -> dict[int, float]:
    """CPU seconds of each thread of `pid` (utime + stime, /proc/PID/task/TID/stat)."""
    times = {}
    try:
        for task in os.scandir(f"/proc/{pid}/task"):
            fields = Path(task.path, "stat").read_text().rsplit(")", 1)[1].split()
            times[int(task.name)] = (int(fields[11]) + int(fields[12])) / CLK_TCK
    except (FileNotFoundError, ProcessLookupError):
        pass
    return times


def cpu_mhz(cpu: int) -> float | None:
    """The CPU's current clock (cpufreq, else /proc/cpuinfo); None where neither reports it."""
    if DARWIN:
        return None
    freq = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
    if freq.exists():
        return int(freq.read_text()) / 1000
    for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
        lines = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
        if lines.get("processor\t", "").strip() == str(cpu) and "cpu MHz\t\t" in lines:
            return float(lines["cpu MHz\t\t"])
    return None


def cpu_model() -> str:
    """The CPU model (V6): x86 'model name', or lscpu's 'Model name' on Arm (macOS: sysctl)."""
    if DARWIN:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, check=True).stdout.strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    return next((l.split(":", 1)[1].strip() for l in out.splitlines()
                 if l.startswith("Model name")), "unknown")


class ThreadSampler(threading.Thread):
    """Samples every thread's CPU time (V8) and the timing CPU's clock (V7) while the encoder
    runs."""

    def __init__(self, pid: int, cpu: int) -> None:
        super().__init__(daemon=True)
        self.pid, self.cpu, self.first, self.last, self.mhz = pid, cpu, {}, {}, []
        self.allowed: set[str] = set()
        self.done = threading.Event()

    def run(self) -> None:
        while not self.done.is_set():
            try:
                for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                    if line.startswith("Cpus_allowed_list:"):
                        self.allowed.add(line.split(":", 1)[1].strip())
            except (FileNotFoundError, ProcessLookupError):
                pass
            mhz = cpu_mhz(self.cpu)
            if mhz:
                self.mhz.append(mhz)
            now = thread_times(self.pid)
            for tid, t in now.items():
                self.first.setdefault(tid, t)
            self.last.update(now)
            self.done.wait(SAMPLE_S)

    def working(self) -> int:
        return sum(self.last[t] - self.first.get(t, 0.0) > THREAD_WORK_S for t in self.last)


def timed_encode(job: dict, cpu: int, group: Path | None, paths: dict[str, Path],
                 scratch: Path) -> dict:
    """The job's one encode, with V7 and V8 measured around it (no hardware counters: D19). With
    `group` None the encoder is pinned by CPU affinity instead of a cpuset move: moving a
    process into a cgroup takes the kernel's cgroup migration lock, and 48 slots spawning at
    once (BP-4, halfload.py) waited up to 0.14 s on it per spawn (rehearsal, 2026-11-25), which
    V7 counts as wall time. The calling thread takes `cpu` only for the fork, so the child holds
    it from its first instruction (`taskset` left a window on the harness's CPU that V8 saw),
    and the sampler stays on the harness's CPU. V8 checks the confinement either way, from the
    encoder's Cpus_allowed_list."""
    stream = paths["OUTPUT"]
    stream.unlink(missing_ok=True)
    stderr_path = scratch / "stderr.txt"
    ticks0, t0, start_unix = cpu_ticks(cpu), time.monotonic(), time.time()
    move = (lambda: (group / "cgroup.procs").write_text(str(os.getpid()))) if group else None
    harness = None if group or DARWIN else os.sched_getaffinity(0)
    with open(stderr_path, "wb") as err:
        if harness is not None:
            os.sched_setaffinity(0, {cpu})  # this thread only, for the fork
        try:
            proc = subprocess.Popen(concrete(job, paths), cwd=scratch,
                                    stdout=subprocess.DEVNULL, stderr=err, preexec_fn=move)
        finally:
            if harness is not None:
                os.sched_setaffinity(0, harness)
        sampler = ThreadSampler(proc.pid, cpu)
        sampler.start()
        NOW["job"], NOW["since"] = job["id"], time.monotonic()
        # a hung encoder is killed at JOB_TIMEOUT_FACTOR x its expected time (review #5)
        limit = max(JOB_TIMEOUT_MIN_S, JOB_TIMEOUT_FACTOR * job.get("expected_s", 0.0))
        killer = threading.Timer(limit, lambda: proc.kill())
        killer.start()
        if DARWIN:
            status, darwin = darwin_reap(proc.pid)
        else:
            _, status, usage = os.wait4(proc.pid, 0)
        killer.cancel()
        NOW["job"] = None
        sampler.done.set()
    wall, ticks1 = time.monotonic() - t0, cpu_ticks(cpu)
    sampler.join()
    total = sum(ticks1) - sum(ticks0)
    if DARWIN:
        return {"start_unix": start_unix, "end_unix": round(time.time(), 1), "wall_s": wall,
                "steal": 0.0,
                "cpu_s": darwin["cycles"] / DARWIN_REF_HZ, "time_measure": "cycles at 4.0 GHz",
                **darwin, "major_faults": 0, "working_threads": sampler.working(),
                "cpus_allowed": [], "mhz_median": None, "mhz_min": None, "status": status,
                "stderr_head": stderr_path.read_bytes()[:STDERR_KEPT].decode(errors="replace")}
    cpu_s = usage.ru_utime + usage.ru_stime
    return {"start_unix": start_unix,  # with end_unix, each slot's gaps between encodes
            "end_unix": round(time.time(), 1),  # for the driver's measured rates
            "cpu_s": cpu_s, "user_s": usage.ru_utime, "sys_s": usage.ru_stime, "wall_s": wall,
            "steal": (ticks1[STEAL_FIELD] - ticks0[STEAL_FIELD]) / max(1, total),
            "major_faults": usage.ru_majflt, "working_threads": sampler.working(),
            "cpus_allowed": sorted(sampler.allowed),
            "mhz_median": sorted(sampler.mhz)[len(sampler.mhz) // 2] if sampler.mhz else None,
            "mhz_min": min(sampler.mhz) if sampler.mhz else None,
            "status": status,
            "stderr_head": stderr_path.read_bytes()[:STDERR_KEPT].decode(errors="replace")}


def check_timing(rec: dict) -> None:
    """V7 and V8 on one encode's measurements."""
    if rec["steal"] >= MAX_STEAL:
        raise TimingRefusal(f"V7: steal {rec['steal']:.4f}")
    if DARWIN:
        # macOS: the time is the encode's cycles, which neither the clock (DVFS) nor time spent
        # waiting for a core changes, so wall over CPU says nothing about it; an encode is
        # judged by where it ran (TIMING-METHOD.md of the Mac series).
        if rec["p_share"] < DARWIN_MIN_P_SHARE:
            raise TimingRefusal(f"V7: only {rec['p_share']:.3f} of the CPU time on P-cores")
        return
    if rec["wall_s"] > max(MAX_WALL_OVER_CPU * rec["cpu_s"], rec["cpu_s"] + MAX_WALL_SLACK_S):
        raise TimingRefusal(f"V7: wall {rec['wall_s']:.3f} s over CPU {rec['cpu_s']:.3f} s")
    if rec["major_faults"]:
        raise TimingRefusal(f"V7: {rec['major_faults']} major page faults")
    # An encode shorter than the sampler's first read leaves `cpus_allowed` empty: not seen,
    # not wrong (the affinity was set at the fork, pinning.py). A short SVT-AV1 job refused a
    # whole run this way (2026-11-26); only an observed wrong set refuses.
    if not DARWIN and rec["cpus_allowed"] and rec["cpus_allowed"] != [str(rec["cpu"])]:
        raise Refusal(f"V8: the encoder could run on CPUs {rec['cpus_allowed']}, not CPU {rec['cpu']} alone")


#: The sidecar an encoder writes beside its stream when the stream itself carries no
#: timestamps: one input frame index per coded picture, in stream order (openh264's '+drop'
#: row, `--coded-frames {OUTPUT}.frames`; procedure task T16).
CODED_FRAMES_SUFFIX = ".frames"


def coded_frames_path(stream: Path) -> Path:
    return stream.with_name(stream.name + CODED_FRAMES_SUFFIX)


def take_coded_frames(stream: Path) -> list[int] | None:
    """The input frame index of each coded picture, if the encoder wrote the sidecar (which
    is then removed: the record keeps the list); None otherwise."""
    sidecar = coded_frames_path(stream)
    if not sidecar.exists():
        return None
    frames = [int(line) for line in sidecar.read_text().split()]
    sidecar.unlink()
    return frames


def encode_job(job: dict, cpu: int, group: Path | None, box: dict, out: Path,
               checked: bool = True) -> dict:
    """One job: input checks, the timed encode, its checks (unless a warm-up), then the
    strip (V19)."""
    clip = box["clips"] / f"{job['input']}.yuv"
    if clip.stat().st_size != job["w"] * job["h"] * 3 // 2 * job["frames"]:
        raise Refusal(f"V9: {clip.name} does not hold exactly {job['frames']} frames "
                      "(libaom's example has no working frame-count option)")
    if job.get("input_sha256") and sha256(clip) != job["input_sha256"]:
        raise Refusal(f"V4: {clip.name} is not the manifest's clip")
    check_resident(clip)
    codec = Codec(job["codec"])
    name = f"{job['id'].replace('/', '_').replace('|', '_')}{codec.bitstream_suffix}"
    stream = box["streams"] / name
    paths = {"BIN": box["bin"], "CFG": box["cfg"], "INPUT": clip, "OUTPUT": stream}
    coded_frames_path(stream).unlink(missing_ok=True)
    rec = {"id": job["id"], "family": job["family"], "series": box["series"], "cpu": cpu,
           "binary_sha256": box.get("hashes", {}).get(Path(job["argv"][0]).name),
           **timed_encode(job, cpu, group, paths, box["scratch"])}
    coded = take_coded_frames(stream)
    if rec["status"] or not stream.exists() or not stream.stat().st_size:
        raise JobFailure(f"V9: {job['id']} exit {rec['status']}: {rec['stderr_head'][-300:]}")
    if coded is not None:
        rec["coded_frames"] = coded
    if job.get("banner") and not re.search(job["banner"], rec["stderr_head"]):
        raise JobFailure(f"V21: {job['family']}'s banner lacks /{job['banner']}/")
    if checked:
        try:
            check_timing(rec)
        except Refusal as why:
            raise type(why)(f"{job['id']}: {why}") from None
    rec["sei_bytes_removed"] = strip(stream, codec)
    rec["bytes"], rec["stream_sha256"] = stream.stat().st_size, sha256(stream)
    rec["stream"] = f"streams/{name}"  # under `out` (a re-timing's streams live in scratch)
    return rec


class Warm:
    """One untimed encode per tool before its first timed one, so each binary and its
    libraries are in the page cache (V7 refuses a major page fault: on a fresh box the first
    exec of a binary faults its pages in from disk)."""

    def __init__(self, cpu: int, group: Path, box: dict, out: Path, log) -> None:
        self.cpu, self.group, self.box, self.out, self.log = cpu, group, box, out, log
        self.tools: set[str] = set()

    def __call__(self, jobs: list[dict]) -> None:
        for job in jobs:
            if job["argv"][0] in self.tools:
                continue
            self.tools.add(job["argv"][0])
            rec = encode_job({**job, "id": f"warmup|{job['id']}"}, self.cpu, self.group,
                             self.box, self.out, checked=False)
            self.log.write(json.dumps({"warmup": job["argv"][0],
                                       "major_faults": rec["major_faults"]}) + "\n")
            (self.box["streams"] / Path(rec["stream"]).name).unlink()


def run_references(references: list[dict], seq: int, cpu: int, group: Path, box: dict,
                   out: Path, log) -> None:
    """BP-3.6: the two timing references (one cache-light, one memory-heavy) at position
    `seq` of the box's queue; their streams are not kept."""
    for ref in references:
        reference = {**ref, "id": f"reference|{ref['id']}|{seq}"}
        try:
            rec = encode_job(reference, cpu, group, box, out)
        except TimingRefusal as why:  # re-run once, as a job is
            log.write(json.dumps({"id": reference["id"], "retry": str(why)}) + "\n")
            rec = encode_job(reference, cpu, group, box, out)
        (box["streams"] / Path(rec["stream"]).name).unlink()
        log.write(json.dumps({**rec, "reference": ref["family"], "seq": seq}) + "\n")
        log.flush()


def run_canaries(canaries: list[dict], where: str, cpu: int, group: Path, box: dict,
                 out: Path, log, seq: int = 0) -> None:
    """V6: the canary set at one point of the run; the report checks the drift. A canary
    whose timing check fails is re-run once, as a job is."""
    for job in canaries * CANARY_REPEATS:
        canary = {**job, "id": f"canary-{where}|{job['id']}"}
        try:
            rec = encode_job(canary, cpu, group, box, out)
        except TimingRefusal as why:
            log.write(json.dumps({"id": canary["id"], "retry": str(why)}) + "\n")
            rec = encode_job(canary, cpu, group, box, out)
        log.write(json.dumps({**rec, "canary": where, "seq": seq}) + "\n")
        log.flush()


def run_packed(jobs: list[dict], workers: int, box: dict, out: Path) -> int:
    """BP-3.8 (A): an untimed quality run, `workers` encodes at once, each in its own one-CPU
    cgroup on CPUs 1..workers. Times are not recorded as speed (packing is invalid for
    timing); the streams are deterministic, and a timed subset run checks their hashes."""
    import concurrent.futures
    groups = [cgroup(cpu) for cpu in range(1, workers + 1)]
    free = list(range(workers))
    lock = threading.Lock()

    def one(job: dict) -> dict:
        with lock:
            slot = free.pop()
        try:
            scratch = box["scratch"] / str(slot)  # each encode its own stderr and work files
            scratch.mkdir(parents=True, exist_ok=True)
            rec = encode_job(job, slot + 1, groups[slot], {**box, "scratch": scratch}, out,
                             checked=False)
        finally:
            with lock:
                free.append(slot)
        return {**rec, "untimed": True}

    refused = 0
    with open(out / "encodes.jsonl", "a") as log, \
            concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, job): job for job in jobs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                log.write(json.dumps(future.result()) + "\n")
            except Refusal as why:
                refused += 1
                log.write(json.dumps({"id": futures[future]["id"], "refused": str(why)}) + "\n")
            log.flush()
            if done % PROGRESS_EVERY == 0 or done == len(jobs):
                print(f"encoded {done}/{len(jobs)}", flush=True)
    return 1 if refused else 0


#: Seconds between heartbeat lines: the driver stops a box that prints nothing for 5 min, and
#: one VVenC encode of a 1080p clip can take longer than that.
HEARTBEAT_S = 60


#: The job being encoded, for the heartbeat (the box's one encode at a time).
NOW: dict = {"job": None, "since": 0.0}
#: A job is killed after this many times its expected seconds (at least JOB_TIMEOUT_MIN_S).
JOB_TIMEOUT_FACTOR = 5.0
JOB_TIMEOUT_MIN_S = 600.0


def heartbeat() -> None:
    """Print a line every HEARTBEAT_S seconds (on the harness's CPU, outside every timing)."""
    def beat() -> None:
        while True:
            time.sleep(HEARTBEAT_S)
            job = NOW["job"]
            print(f"alive: {job} for {time.monotonic() - NOW['since']:.0f} s" if job
                  else "alive: between jobs", flush=True)
    threading.Thread(target=beat, daemon=True).start()


class BackgroundLoad:
    """The packing test (owner, 2026-11-24): a constant load on the idle physical cores, one
    encoder per CPU, each looping over the box's background jobs in its own shuffled order,
    streams discarded. The timed encodes and their checks are unchanged, so the references'
    records show whether packed boxes keep timing accuracy."""

    def __init__(self, jobs: list[dict], cpus_: list[int], box: dict) -> None:
        self.stop, self.encodes, self.failures = threading.Event(), 0, 0
        self.threads = [threading.Thread(target=self.loop, args=(jobs, cpu, box), daemon=True)
                        for cpu in cpus_]
        for thread in self.threads:
            thread.start()

    def loop(self, jobs: list[dict], cpu: int, box: dict) -> None:
        import random
        work = box["scratch"] / f"background{cpu}"
        work.mkdir(parents=True, exist_ok=True)
        order = random.Random(cpu)
        while not self.stop.is_set():
            job = order.choice(jobs)
            paths = {"BIN": box["bin"], "CFG": box["cfg"],
                     "INPUT": box["clips"] / f"{job['input']}.yuv", "OUTPUT": work / "stream"}
            done = subprocess.run(concrete(job, paths), cwd=work, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
            if done.returncode:
                self.failures += 1
                self.stop.wait(1.0)  # a failing encoder must not spin
            else:
                self.encodes += 1

    def close(self) -> dict:
        self.stop.set()
        for thread in self.threads:
            thread.join()
        return {"encodes": self.encodes, "failures": self.failures}


def spot_notice() -> str | None:
    """The spot interruption notice, if one is posted (instance metadata, IMDSv2); None off
    EC2 or with no notice."""
    if DARWIN:
        return None
    import urllib.error
    import urllib.request
    try:
        token = urllib.request.urlopen(urllib.request.Request(
            f"{IMDS}/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}),
            timeout=IMDS_TIMEOUT_S).read().decode()
        return urllib.request.urlopen(urllib.request.Request(
            f"{IMDS}/meta-data/spot/instance-action",
            headers={"X-aws-ec2-metadata-token": token}), timeout=IMDS_TIMEOUT_S).read().decode()
    except (urllib.error.URLError, OSError):
        return None


def next_unit(inbox: Path) -> tuple[str, list[dict]] | None:
    """The oldest unit the driver has dropped in the inbox, taken (renamed) so it runs once."""
    for unit in sorted(inbox.glob("unit-*.json")):
        taken = unit.with_suffix(".taken")
        unit.rename(taken)
        return unit.stem, json.loads(taken.read_text())
    return None


class Interrupted(Exception):
    """A spot interruption notice: stop taking work."""


class Server:
    """BP-3.11: the timed jobs of a queue box, unit by unit from its inbox."""

    def __init__(self, a, box: dict, group: Path, log, canaries: list[dict],
                 references: list[dict]) -> None:
        self.a, self.box, self.group, self.log = a, box, group, log
        self.canaries, self.references = canaries, references
        self.warm = Warm(a.cpu, group, box, a.out, log)
        self.done, self.referenced, self.retried = 0, -1, set()
        self.since_reference_s = 0.0

    def serve(self) -> None:
        self.warm(self.canaries + self.references)
        run_canaries(self.canaries, "start", self.a.cpu, self.group, self.box, self.a.out, self.log)
        try:
            while True:
                unit = next_unit(self.a.inbox)
                if unit is None:
                    if (self.a.inbox / "END").exists():
                        break
                    self.check_notice()
                    self.reference()  # idle: bracket the jobs coded so far (they settle)
                    time.sleep(INBOX_POLL_S)
                    continue
                self.run_unit(*unit)
        except Interrupted as notice:
            self.log.write(json.dumps({"interrupted": str(notice), "seq": self.done}) + "\n")
            print(f"interrupted: {notice}", flush=True)
            self.reference()  # the jobs since the last references keep a valid time
            return
        self.reference()
        run_canaries(self.canaries, "end", self.a.cpu, self.group, self.box, self.a.out,
                     self.log, self.done)

    def check_notice(self) -> None:
        """A spot notice, or the driver's STOP (any stop of a box: a cancel, a spend ceiling,
        a fleet stop), ends the work the same way: references first, so coded jobs stay valid."""
        if (self.a.inbox / "STOP").exists():
            raise Interrupted("stopped by the driver")
        notice = spot_notice()
        if notice:
            raise Interrupted(notice)

    def reference(self) -> None:
        if self.references and self.done != self.referenced:
            run_references(self.references, self.done, self.a.cpu, self.group, self.box,
                           self.a.out, self.log)
            self.referenced, self.since_reference_s = self.done, 0.0

    def run_unit(self, name: str, jobs: list[dict]) -> None:
        queue = list(jobs)
        while queue:
            self.check_notice()
            if self.done % REFERENCE_EVERY == 0 or self.since_reference_s >= REFERENCE_MAX_S:
                self.reference()
            job = queue.pop(0)
            self.warm([job])
            try:
                rec = encode_job(job, self.a.cpu, self.group, self.box, self.a.out)
            except JobFailure as why:
                self.log.write(json.dumps({"id": job["id"], "failed": str(why)}) + "\n")
                self.log.flush()
                continue
            except TimingRefusal as why:
                if job["id"] in self.retried:
                    raise
                self.retried.add(job["id"])
                self.log.write(json.dumps({"id": job["id"], "retry": str(why)}) + "\n")
                queue.append(job)
                continue
            if self.a.discard_streams:
                (self.box["streams"] / Path(rec["stream"]).name).unlink()
            self.log.write(json.dumps({**rec, "seq": self.done}) + "\n")
            self.log.flush()
            self.done += 1
            self.since_reference_s += rec["wall_s"]
            if self.since_reference_s >= REFERENCE_MAX_S:  # e.g. after one long encode
                self.reference()
            if self.done % PROGRESS_EVERY == 0:
                print(f"encoded {self.done}", flush=True)  # the driver's stall watch
            if self.done == CANARY_MIDDLE:
                run_canaries(self.canaries, "middle", self.a.cpu, self.group, self.box,
                             self.a.out, self.log, self.done)
        self.log.write(json.dumps({"unit": name, "seq": self.done}) + "\n")
        self.log.flush()


def serve_packed(a, box: dict) -> int:
    """BP-3.8 untimed quality encodes from the queue: each unit packed over `a.packed` CPUs."""
    refused = 0
    while True:
        if spot_notice():
            print("interrupted", flush=True)
            return refused
        unit = next_unit(a.inbox)
        if unit is None:
            if (a.inbox / "END").exists():
                return refused
            time.sleep(INBOX_POLL_S)
            continue
        refused += run_packed(unit[1], a.packed, box, a.out)
        with open(a.out / "encodes.jsonl", "a") as log:
            log.write(json.dumps({"unit": unit[0]}) + "\n")


def main() -> int:
    heartbeat()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inbox", type=Path, required=True, help="the driver's units (BP-3.11)")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cpu", type=int, required=True, help="the one CPU the encodes run on")
    ap.add_argument("--series", required=True, help="the machine series (V23), e.g. c8i")
    ap.add_argument("--clips", type=Path, required=True, help="the clips' tmpfs directory")
    ap.add_argument("--canaries", type=Path, help="the canary jobs (V6)")
    ap.add_argument("--references", type=Path, help="the two timing references (BP-3.6)")
    ap.add_argument("--packed", type=int, default=0,
                    help="an untimed quality run with this many encodes at once (BP-3.8)")
    ap.add_argument("--discard-streams", action="store_true",
                    help="a re-timing run: record each stream's SHA-256 and size, keep no stream")
    ap.add_argument("--background", type=Path,
                    help="the packing test: these jobs looped on the idle cores' CPUs")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out = a.out.resolve()  # the encoders run in a scratch directory: absolute paths only
    scratch = a.out.parent / "scratch"  # outside `out`: the driver downloads `out` as it grows
    box = {"bin": HERE / "bin", "cfg": HERE / "cfg", "clips": a.clips, "series": a.series,
           "scratch": scratch,
           "streams": scratch / "streams" if a.discard_streams else a.out / "streams"}
    for d in (a.out, box["streams"], scratch):
        d.mkdir(parents=True, exist_ok=True)
    box["hashes"] = check_binaries(json.loads(a.manifest.read_text()), box["bin"])
    if a.packed:
        (CGROUP_ROOT / "cgroup.subtree_control").write_text("+cpuset")
        return serve_packed(a, box)
    canaries = json.loads(a.canaries.read_text()) if a.canaries else []
    references = json.loads(a.references.read_text()) if a.references else []
    harness = pin_harness(a.cpu)
    group = cgroup(a.cpu)
    background_cpus = pinning.idle_core_cpus(a.cpu, harness) if a.background else []
    with open(a.out / "encodes.jsonl", "a") as log:
        log.write(json.dumps({"box": {"timing_cpu": a.cpu, "harness_cpu": harness,
                                      "cpu_model": cpu_model(), "series": a.series,
                                      "background_cpus": background_cpus}}) + "\n")
        load = BackgroundLoad(json.loads(a.background.read_text()), background_cpus, box) \
            if a.background else None
        try:
            Server(a, box, group, log, canaries, references).serve()
        except Refusal as why:
            log.write(json.dumps({"refused": str(why), "timing": isinstance(why, TimingRefusal)})
                      + "\n")
            print(f"refused: {why}", file=sys.stderr)
            return 1
        finally:
            if load is not None:
                log.write(json.dumps({"background": load.close()}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
