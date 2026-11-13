"""Single-core encode speed measurement (eval/README.md "Methodology: Speed").

Method:
  - every run is pinned to one core with taskset;
  - one warm-up run per encoder (page cache, CPU frequency ramp);
  - then `runs` timed rounds, interleaving encoders ABBA-style (alternate the
    order each round) so slow drift in machine state cancels between them;
  - wall time and user+sys CPU time (os.wait4 rusage) per run; medians kept;
  - if perf is usable, one extra run per encoder under `perf stat` for
    instruction and cycle counts (kept separate so perf never skews timing).
  - each warm-up passes the standard decoder gate outside timing; our
    encoder also verifies its reconstruction. Every timed and perf stream
    must hash-match that encoder's verified warm-up, including reference
    encoders, avoiding repeated decoding of identical bytes;
  - actual payload rates must stay below the clip's source-rate ceiling.

Each encoder is timed at the quantiser whose bitrate (from the sweep results)
is closest to a target bitrate, so all encoders are compared at about the same
rate. Results are ms/frame and the ratio to the anchor encoder on the same
machine and clip, the form the speed targets are stated in.

Conditions are reported with the results: a CPU governor other than
"performance" or enabled turbo/boost adds run-to-run noise.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .clips import Clip
from .config import Paths
from .conformance import check_stream
from .diskguard import check_disk
from .encoders.base import CodingStructure, EncodeJob, EncoderAdapter
from .results import ResultsDb, SpeedRecord
from .sweep import payload_bytes
from .yuv import YuvFormat

WARMUP_RUNS = 1
MIN_TIMED_RUNS = 5


@dataclass(frozen=True)
class SpeedResult:
    encoder: str
    quantizer: int
    kbps: float
    ms_per_frame: float
    cpu_ms_per_frame: float
    instructions: float | None
    cycles: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def machine_conditions() -> list[str]:
    """Human-readable warnings about settings that make timings noisy."""
    warnings = []
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor.is_file() and governor.read_text().strip() != "performance":
        warnings.append(f"CPU governor is '{governor.read_text().strip()}', not 'performance'")
    boost = Path("/sys/devices/system/cpu/cpufreq/boost")
    if boost.is_file() and boost.read_text().strip() == "1":
        warnings.append("CPU boost/turbo is enabled")
    if not _perf_usable():
        warnings.append("perf stat unavailable (perf_event_paranoid too high): "
                        "no instruction/cycle counts")
    return warnings


def choose_quantizers(db: ResultsDb, encoders: list[EncoderAdapter], structure_tag: str,
                      clip: Clip, machine: str, target_kbps: float) -> dict[str, tuple[int, float]]:
    """Per encoder, the swept quantiser whose bitrate is nearest the target."""
    return {enc.name: _quantizer_near(db, enc.name, structure_tag, clip.name, machine,
                                      target_kbps, clip.max_test_kbps)
            for enc in encoders}


def measure(encoders: list[EncoderAdapter], clip: Clip, structure: CodingStructure,
            quantizers: dict[str, tuple[int, float]], frames: int, core: int, runs: int,
            paths: Paths) -> list[SpeedResult]:
    """Time each encoder at its given (quantizer, kbps); nothing is stored here."""
    runs = max(runs, MIN_TIMED_RUNS)
    adapters = {enc.name: enc for enc in encoders}
    check_disk(paths.work)
    scratch = Path(tempfile.mkdtemp(prefix="nc-speed-", dir=_fast_tmp()))
    validation: Path | None = None
    try:
        validation = Path(tempfile.mkdtemp(prefix="nc-speed-check-", dir=paths.work))
        # Numeric filenames also support candidate names containing '/'.
        outputs = {name: scratch / f"encode-{i}{enc.codec.bitstream_suffix}"
                   for i, (name, enc) in enumerate(adapters.items())}
        reconstructions = {name: validation / f"recon-{i}.yuv"
                           for i, name in enumerate(adapters)}
        verified_hashes: dict[str, str] = {}
        rates: dict[str, list[float]] = {name: [] for name in adapters}

        def command(name: str, recon: Path | None = None) -> list[str]:
            enc = adapters[name]
            job = EncodeJob(clip=clip, quantizer=quantizers[name][0], structure=structure,
                            frames=frames, bitstream=outputs[name], recon=recon)
            enc.check_job(job)
            return ["taskset", "-c", str(core), *enc.command(job, paths)]

        def verify(name: str, recon: Path | None = None) -> None:
            enc = adapters[name]
            out = outputs[name]
            check_disk(paths.work)
            digest = hashlib.sha256(out.read_bytes()).hexdigest()
            if name not in verified_hashes or recon is not None:
                # Decode each encoder's warm-up once, including references.
                # Timed streams carry the same conformance certificate only
                # when their complete bytes match this verified stream.
                with tempfile.TemporaryDirectory(prefix="speed-decode-", dir=paths.work) as tmp:
                    gate = check_stream(out, enc.codec, YuvFormat(clip.width, clip.height),
                                        frames, Path(tmp), paths, recon)
                    if not gate.passed:
                        raise RuntimeError(f"{name}: speed conformance failed: {gate.reason}")
                verified_hashes[name] = digest
            elif verified_hashes[name] != digest:
                raise RuntimeError(f"{name}: timed stream differs from decoder-verified warmup")
            kbps = payload_bytes(out, enc.codec, frames) * 8 / 1000 / (frames / clip.fps)
            if not clip.allows_kbps(kbps):
                raise ValueError(f"{name}: speed stream {kbps:.0f} kbps exceeds "
                                 f"source cap {clip.max_test_kbps:.0f}")
            rates[name].append(kbps)
            if recon is not None:
                recon.unlink()

        names = list(adapters)
        for _ in range(WARMUP_RUNS):
            for name in names:
                # Request recon only in the untimed warm-up, then require
                # each timed stream to match the bytes checked against it.
                recon = reconstructions[name] if adapters[name].verify_recon() else None
                _timed_run(command(name, recon))
                verify(name, recon)

        wall: dict[str, list[float]] = {n: [] for n in names}
        cpu: dict[str, list[float]] = {n: [] for n in names}
        for round_index in range(runs):
            order = names if round_index % 2 == 0 else list(reversed(names))
            for name in order:
                w, c = _timed_run(command(name))
                verify(name)
                wall[name].append(w)
                cpu[name].append(c)

        results = []
        for name in names:
            q, _ = quantizers[name]
            if _perf_usable():
                counters = _perf_counters(command(name))
                verify(name)
            else:
                counters = (None, None)
            results.append(SpeedResult(
                encoder=name, quantizer=q, kbps=statistics.median(rates[name]),
                ms_per_frame=statistics.median(wall[name]) * 1000 / frames,
                cpu_ms_per_frame=statistics.median(cpu[name]) * 1000 / frames,
                instructions=counters[0], cycles=counters[1]))
        return results
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if validation is not None:
            shutil.rmtree(validation, ignore_errors=True)


def store(results: list[SpeedResult], db: ResultsDb, structure_tag: str, clip: Clip,
          frames: int, machine: str, core: int, runs: int, fingerprints: dict[str, str],
          harness_commit: str) -> None:
    for r in results:
        db.store_speed(SpeedRecord(
            encoder=r.encoder, structure=structure_tag, clip=clip.name, quantizer=r.quantizer,
            frames=frames, platform=machine, core=core, runs=max(runs, MIN_TIMED_RUNS),
            ms_per_frame=r.ms_per_frame, cpu_ms_per_frame=r.cpu_ms_per_frame,
            instructions=r.instructions, cycles=r.cycles,
            fingerprint=fingerprints[r.encoder], harness_commit=harness_commit))


def format_speed(results: list[SpeedResult], anchor: str, clip: Clip,
                 conditions: list[str]) -> str:
    by_name = {r.encoder: r for r in results}
    base = by_name.get(anchor)
    lines = [f"Single-core speed on {clip.name} ({clip.width}x{clip.height}@{clip.fps:g}), "
             f"ratio vs {anchor} (CPU time)", ""]
    lines.append(f"{'encoder':<20}{'q':>4}{'kbps':>8}{'wall ms/f':>11}{'cpu ms/f':>10}"
                 f"{'x anchor':>10}{'Minstr/f':>10}")
    for r in sorted(results, key=lambda r: r.cpu_ms_per_frame):
        ratio = f"{r.cpu_ms_per_frame / base.cpu_ms_per_frame:>9.2f}x" if base else f"{'-':>10}"
        instr = f"{r.instructions / 1e6:>10.1f}" if r.instructions else f"{'-':>10}"
        lines.append(f"{r.encoder:<20}{r.quantizer:>4}{r.kbps:>8.0f}{r.ms_per_frame:>11.2f}"
                     f"{r.cpu_ms_per_frame:>10.2f}{ratio}{instr}")
    if conditions:
        lines += ["", "Measurement conditions (noise sources):"] + [f"  - {c}" for c in conditions]
    return "\n".join(lines)


def _quantizer_near(db: ResultsDb, encoder: str, structure: str, clip: str, machine: str,
                    target_kbps: float, cap_kbps: float) -> tuple[int, float]:
    points = [(q, rate) for q, rate in db.swept_points(encoder, structure, clip, machine)
              if 0 < rate <= cap_kbps]
    if not points:
        raise LookupError(f"no legal sweep results for {encoder}/{clip}/{structure}: "
                          "run 'sweep' first")
    return min(points, key=lambda point: abs(point[1] - target_kbps))


def _timed_run(cmd: list[str]) -> tuple[float, float]:
    """Run once; return (wall seconds, user+sys CPU seconds) of the process."""
    # stderr goes to a file, not a pipe: encoders print progress, and a full
    # pipe would block the child while we sit in wait4.
    with tempfile.TemporaryFile() as err:
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err)
        _, status, usage = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - start
        proc.returncode = os.waitstatus_to_exitcode(status)
        if proc.returncode != 0:
            err.seek(0)
            tail = err.read().decode(errors="replace").strip().splitlines()[-3:]
            raise RuntimeError(f"timed run failed ({proc.returncode}): {' | '.join(tail)}")
    return wall, usage.ru_utime + usage.ru_stime


def _perf_usable() -> bool:
    if not shutil.which("perf"):
        return False
    probe = subprocess.run(["perf", "stat", "-x,", "-e", "instructions", "--", "true"],
                           capture_output=True, text=True)
    return probe.returncode == 0 and "not supported" not in probe.stderr


def _perf_counters(cmd: list[str]) -> tuple[float | None, float | None]:
    with tempfile.NamedTemporaryFile(mode="r", suffix=".csv") as out:
        subprocess.run(["perf", "stat", "-x,", "-e", "instructions,cycles", "-o", out.name,
                        "--", *cmd], capture_output=True, check=True)
        counts: dict[str, float] = {}
        for line in Path(out.name).read_text().splitlines():
            fields = line.split(",")
            if len(fields) > 2 and fields[0].replace(".", "").isdigit():
                counts[fields[2].split(":")[0]] = float(fields[0])
    return counts.get("instructions"), counts.get("cycles")


def _fast_tmp() -> str | None:
    """Write bitstreams to RAM where possible, so disk I/O doesn't enter timings."""
    return "/dev/shm" if os.path.isdir("/dev/shm") else None
