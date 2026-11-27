"""Metric validation: the charts' scorers checked against reference tools on real chart points.

    python3 -m nceval.metricval sample --chart CLIPSET STREAMS RECORD [--chart ...]
        --results DIR --out SAMPLE_DIR [--bd-clips N] [--spread N]
    python3 -m nceval.metricval score --sample SAMPLE_DIR --backend vmafx-cuda|libvmaf-cpu
        --out SCORES.json [--encoder-psnr] [--workers N]
    python3 -m nceval.metricval compare --sample SAMPLE_DIR --cpu A.json --gpu B.json
        --out-csv FILE --out-json FILE

What the charts compute (docs/PROCEDURES.md "Metrics and protocol"):
  - VMAF v1 (model vmaf_v1.0.16_3d0h) at each clip's own size, by VMAFx on CUDA on the
    PC or by libvmaf on the CPU on rented boxes (metrics.vmaf_v1_and_ms_ssim);
  - PSNR from exact per-plane squared errors: our encoder's own `--sse-log` on its
    reconstruction (nceval.point, the main chart's local points), or numpy on a
    standard decoder's output (cbr.Bench.prebuilt, the rented-box points).

`sample` picks stored chart points (the streams a chart refresh kept, every family
on its recorded build) and copies their streams, their encoder digests and their
stored results into SAMPLE_DIR (a few MB): the BD set (every rung of three
our-codec/reference pairs on `--bd-clips` clips) and a spread over resolutions,
classes, families and rungs. `score` decodes each stream with the first standard
decoder (nceval.conformance.gate_stream: our streams must equal the encoder's
reconstruction digests, so their scores are the reconstruction's) and scores it on
one backend: VMAF v1 and MS-SSIM; with libvmaf-cpu also libvmaf's own `psnr`
feature (an independent PSNR tool, same 60 dB cap) beside the harness's numpy PSNR;
with `--encoder-psnr` our points are encoded again by their recorded build with
`--sse-log` (the main chart's PSNR path), and the stream must equal the stored one.
`compare` joins two backends' scores and the stored results: per-frame and
per-sequence differences, and BD-VMAF of the pairs both ways.

No decoded video is written: every decode and reference goes through pipes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from . import tempdirs
from .bdrate import MIN_POINTS, bd_rate
from .cbr import Bench, DEFAULT_FRAMES, DEFAULT_INTRA_PERIOD, Family, Point
from .clipframes import ClipFrames
from .clips import Clip
from .codecs import Codec
from .config import Paths, load_paths
from .conformance import DECODERS, gate_stream
from .framepipe import PSNR_CAP_DB
from .metrics import (METRICS_BACKEND_ENV, LIBVMAF_CPU_BACKEND, VMAFX_CUDA_BACKEND,
                      EncoderPsnrMetric, Measurement, MetricSession, NativeVmafMetric,
                      PsnrMetric, StreamSpec, gpu_slot_need, vmaf_v1_and_ms_ssim,
                      weighted_yuv)
from .point import coded_format
from .realuse import clipset_clips
from .slots import gpu_slots
from .sweep import payload_bytes
from .yuv import YuvFormat

#: The BD pairs (test, anchor): each of our three oldest codecs against its chart
#: reference. Every rung both families have on a BD clip is sampled.
BD_PAIRS = (("nanocodec-av1", "svtav1-m12"), ("nanocodec-hevc", "x265-medium"),
            ("nanocodec-h264", "x264-medium"))
#: Families the spread draws from in turn: our default of each codec and the
#: other references, so every decoder path is exercised.
SPREAD_FAMILIES = ("nanocodec-vp9", "x264-medium", "nanocodec-vvc", "svtav1-m12",
                   "nanocodec-av1", "x265-medium", "nanocodec-av2", "libvpx-vp9-cpu7",
                   "nanocodec-hevc", "libaom-av1-cpu8", "nanocodec-h264", "openh264-baseline")
#: Height bins the spread and BD clips cover (240p to 720p).
HEIGHT_BINS = ((0, 300), (300, 450), (450, 600), (600, 800))
#: Files of a stored point copied into the sample (beside its bitstream).
OUR_SIDE_FILES = (".recon.txt", ".rate.csv", ".times.txt")
SAMPLE_FILE = "sample.json"
PLANES = ("psnr_y", "psnr_u", "psnr_v")
#: libvmaf's psnr feature, under our names prefixed `lib_` (beside numpy's own).
LIBVMAF_PSNR_FEATURES = {"lib_psnr_y": "psnr_y", "lib_psnr_u": "psnr_cb",
                         "lib_psnr_v": "psnr_cr"}


@dataclass(frozen=True)
class SampledPoint:
    """One stored chart point in the sample (its files under SAMPLE_DIR/<key>)."""
    clipset: str
    clip: str
    family: str
    codec: str
    own: bool
    kbps: int
    height: int
    content_class: str
    #: Our points: the encode command with {binary}, {input}, {out}, {recon},
    #: {rate_log}, {times} placeholders, the binary relative to the data root.
    command: tuple[str, ...] = ()
    binary: str = ""
    sets: tuple[str, ...] = ()
    #: The stream's rate (payload bits over its duration): BD's rate where no stored
    #: result gives the chart's own.
    coded_kbps: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.clip}/{self.family}/cbr{self.kbps}"


# --- sample -------------------------------------------------------------------


def stored_points(streams: Path, clips: list[Clip], families: list[str]) -> dict:
    """(clip, family) -> sorted kbps of every stored stream that was coded (.done)."""
    found: dict[tuple[str, str], list[int]] = {}
    for clip in clips:
        for family in families:
            directory = streams / clip.name / family
            if not directory.is_dir():
                continue
            rungs = sorted({int(p.name.split("-")[0].removeprefix("cbr"))
                            for p in directory.glob("cbr*.done")})
            if rungs:
                found[(clip.name, family)] = rungs
    return found


def size_bin(clip: Clip) -> int:
    """The clip's HEIGHT_BINS index by its shorter side (portrait clips too)."""
    side = min(clip.width, clip.height)
    return next(i for i, (low, high) in enumerate(HEIGHT_BINS) if low <= side < high)


def choose(clips: list[Clip], stored: dict, bd_clips: int, spread: int) -> list[tuple]:
    """(clip, family, kbps, set) picks: the BD set on `bd_clips` clips that hold every
    BD family, drawn round-robin over (size bin, class) cells, then `spread` points over
    the other clips, likewise, rotating family and rung (lowest, middle, highest)."""
    bd_families = [f for pair in BD_PAIRS for f in pair]
    picks: list[tuple] = []
    cells: dict[tuple[int, str], list[Clip]] = {}
    for clip in sorted(clips, key=lambda c: c.name):
        cells.setdefault((size_bin(clip), clip.content_class), []).append(clip)
    complete = [c for c in round_robin(cells) if all((c.name, f) in stored for f in bd_families)]
    for clip in complete[:bd_clips]:
        cells[(size_bin(clip), clip.content_class)].remove(clip)
        for test, anchor in BD_PAIRS:
            rungs = sorted(set(stored[(clip.name, test)]) & set(stored[(clip.name, anchor)]))
            picks += [(clip.name, family, kbps, "bd") for family in (test, anchor)
                      for kbps in rungs]
    taken = 0
    while taken < spread and any(cells.values()):
        for cell in sorted(cells):
            if taken == spread or not cells[cell]:
                continue
            clip = cells[cell].pop(0)
            family = SPREAD_FAMILIES[taken % len(SPREAD_FAMILIES)]
            if (clip.name, family) not in stored:
                family = next((f for f in SPREAD_FAMILIES[taken % len(SPREAD_FAMILIES):]
                               + SPREAD_FAMILIES if (clip.name, f) in stored), None)
                if family is None:
                    continue
            rungs = stored[(clip.name, family)]
            kbps = rungs[[0, len(rungs) // 2, -1][taken % 3]]
            picks.append((clip.name, family, kbps, "spread"))
            taken += 1
    return picks


def stored_stream(streams: Path, results: list[Path], label: str, codec: Codec, frames: int,
                  build: str | None) -> tuple[Path, dict | None]:
    """A rung's kept stream (in `streams`) and the stored result (in one of `results`) scored
    from it. A chart keeps the streams of several builds: ours is the recorded `build`'s
    where one is kept, else the newest one with a result."""
    found = [Path(f"{done.with_suffix('')}{codec.bitstream_suffix}")
             for done in streams.glob(f"{label}-*.done")]
    found = sorted((f for f in found if f.exists()), key=lambda f: -f.stat().st_mtime)
    if not found:
        raise SystemExit(f"{streams} {label}: no kept stream")
    scored = [(f, stored_result(results, label, payload_bytes(f, codec, frames) * 8))
              for f in found]
    def rank(item: tuple[Path, dict | None]) -> int:
        result = item[1]
        if result is None:
            return 3
        commit = result.get("binary_commit", "")[:len(build or "")]
        if build is None or commit == build:
            return 0
        return 1 if (load_paths().data_root / "charts" / "builds" / commit).is_dir() else 2
    return min(scored, key=rank)  # stable: the newest of the best rank


def stored_result(directories: list[Path], label: str, stream_bits: int) -> dict | None:
    """The stored result scored from a stream: the one whose frame bits add up to its
    payload (results of other builds stay beside it; their keys hold the file sizes of
    the machine that decoded the clips, so they cannot be recomputed here)."""
    for path in sorted(p for d in directories for p in d.glob(f"{label}-*.json")):
        result = json.loads(path.read_text())
        if sum(result.get("frame_bits", [])) == stream_bits:
            return result
    return None


def round_robin(cells: dict[tuple, list[Clip]]) -> list[Clip]:
    """The cells' clips in turn: first of each cell, then second of each, ..."""
    columns = [cells[cell] for cell in sorted(cells)]
    return [column[i] for i in range(max(map(len, columns), default=0))
            for column in columns if i < len(column)]


def record_families(record: Path, names: list[str]) -> dict[str, Family]:
    """Each family on its recorded build (as tools/dev/aws_heldout_chart.py measures it)."""
    from .chart import chart_families
    data = json.loads(record.read_text())
    builds = load_paths().data_root / "charts" / "builds"
    dirs = {f: builds / b for f, b in data["families"].items()}
    base = builds / data["base"]
    families = chart_families(base / "nanocodec", base / "presets", names,
                              {f: d / "nanocodec" for f, d in dirs.items()},
                              {f: d / "presets" for f, d in dirs.items()})
    return {f.name: f for f in families}


def our_command(bench: Bench, point: Point, paths: Paths) -> tuple[tuple[str, ...], str]:
    """Our encode command as a template (see SampledPoint.command)."""
    marks = {"out": Path("{out}"), "rate_log": Path("{rate_log}"), "times": Path("{times}")}
    cmd = bench.command(point, marks["out"], "{recon}", marks["rate_log"], marks["times"])
    binary = str(point.family.tool)
    template = tuple("{binary}" if part == binary else
                     "{input}" if part == str(point.clip.yuv) else part for part in cmd)
    return template, str(point.family.tool.relative_to(paths.data_root))


def sample(args: argparse.Namespace) -> None:
    paths = load_paths()
    results, out = [Path(r) for r in args.results], Path(args.out)
    names = list(dict.fromkeys([f for pair in BD_PAIRS for f in pair] + list(SPREAD_FAMILIES)))
    points: dict[str, SampledPoint] = {}
    for index, (clipset, streams, record) in enumerate(args.chart):
        clips = clipset_clips(clipset, paths)
        families = record_families(Path(record), names)
        stored = stored_points(Path(streams), clips, list(families))
        picks = choose(clips, stored, args.bd_clips if index == 0 else 0, args.spread)
        by_name = {c.name: c for c in clips}
        for clip_name, family_name, kbps, which in picks:
            entry = sample_point(by_name[clip_name], families[family_name], kbps, which,
                                 copy_clipset(Path(clipset), out), Path(streams), results,
                                 out, paths)
            if entry.key in points:  # a BD point drawn again by the spread
                entry = SampledPoint(**{**points[entry.key].__dict__,
                                        "sets": points[entry.key].sets + entry.sets})
            points[entry.key] = entry
    (out / SAMPLE_FILE).write_text(json.dumps(
        {"points": [p.__dict__ for p in points.values()]}, indent=1) + "\n")
    print(f"{len(points)} points ({sum('bd' in p.sets for p in points.values())} in the BD set) "
          f"into {out}; {sum((out / k / 'stored.json').exists() for k in points)} "
          f"with their stored result")


def copy_clipset(clipset: Path, out: Path) -> str:
    """The clip-set file copied into the sample (its clips' paths are relative to the
    data root, so it serves on any machine): its path relative to `out`."""
    target = Path("clipsets") / clipset.name
    (out / target).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(clipset, out / target)
    return str(target)


def sample_point(clip: Clip, family: Family, kbps: int, which: str, clipset: str,
                 streams: Path, results: list[Path], out: Path, paths: Paths) -> SampledPoint:
    """Copy one stored point's stream (and our side files, its stored result) into `out`."""
    frames = min(DEFAULT_FRAMES, clip.frames)
    bench = Bench(paths, frames, DEFAULT_INTRA_PERIOD, out, metric_set="real")
    point = Point(family, clip, kbps=kbps)
    stream, result = stored_stream(streams / clip.name / family.name,
                                   [r / clip.name / family.name for r in results], point.label,
                                   family.codec, frames,
                                   family.tool.parent.name if family.own else None)
    if family.own and result is not None:  # the build that coded it
        family = replace(family, tool=paths.data_root / "charts" / "builds"
                         / result["binary_commit"][:8] / "nanocodec")
        point = Point(family, clip, kbps=kbps)
    command, binary = our_command(bench, point, paths) if family.own else ((), "")
    coded_kbps = payload_bytes(stream, family.codec, frames) * 8 * clip.fps / frames / 1000.0
    entry = SampledPoint(clipset, clip.name, family.name, family.codec.value, family.own,
                         kbps, clip.height, clip.content_class, command, binary, (which,),
                         coded_kbps)
    target = out / entry.key
    target.mkdir(parents=True, exist_ok=True)
    stem = stream.with_suffix("")
    suffixes = [family.codec.bitstream_suffix] + (list(OUR_SIDE_FILES) if family.own else [])
    for suffix in suffixes:
        if Path(f"{stem}{suffix}").exists():
            shutil.copyfile(f"{stem}{suffix}", target / f"stream{suffix}")
    if result is not None:
        (target / "stored.json").write_text(json.dumps(result) + "\n")
    return entry


def load_sample(directory: Path) -> list[SampledPoint]:
    data = json.loads((directory / SAMPLE_FILE).read_text())
    return [SampledPoint(**{**p, "command": tuple(p["command"]), "sets": tuple(p["sets"])})
            for p in data["points"]]


def sample_clips(points: list[SampledPoint], directory: Path,
                 paths: Paths) -> dict[tuple[str, str], Clip]:
    """(clip set, clip name) -> Clip, each of the sample's clip sets loaded once."""
    clips = {}
    for clipset in {p.clipset for p in points}:
        clips.update({(clipset, c.name): c
                      for c in clipset_clips(str(directory / clipset), paths)})
    return clips


# --- score --------------------------------------------------------------------


def libvmaf_psnr_metric(spec: StreamSpec) -> NativeVmafMetric:
    """libvmaf's own psnr feature (integer, 8-bit cap 60 dB): the independent PSNR."""
    def parse(report: dict) -> Measurement:
        frames = sorted(report["frames"], key=lambda f: f["frameNum"])
        rows = [{ours: float(f["metrics"][theirs]) for ours, theirs in
                 LIBVMAF_PSNR_FEATURES.items()} for f in frames]
        summary = plane_summary([{p: r[f"lib_{p}"] for p in PLANES} for r in rows])
        return Measurement({f"lib_{k}": v for k, v in summary.items()}, rows)
    return NativeVmafMetric(spec, "libvmaf_psnr", ["--feature", "psnr", "--no_prediction"],
                            parse)


def plane_summary(rows: list[dict]) -> dict:
    """Per-plane means over frames, combined 6:1:1 (the charts' PSNR-YUV)."""
    means = {plane: statistics.fmean(r[plane] for r in rows) for plane in PLANES}
    return {**means, "psnr_yuv": weighted_yuv(*means.values())}


class ReferenceDigest:
    """SHA-256 of the reference frames a point is scored against: two machines' scores
    are comparable only where their clip decodes agree (ffmpeg versions differ)."""

    cpu_threads = 0
    uses_gpu = False

    def __init__(self, spec: StreamSpec):
        self._spec = spec
        self._digest = hashlib.sha256()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self, processes, scratch: Path) -> list:
        path = self._spec.reference_path()
        size = self._spec.frames * self._spec.fmt.frame_size

        def read() -> None:
            try:
                with open(path, "rb") as source:
                    left = size
                    while left:
                        chunk = source.read(min(left, 1 << 20))
                        if not chunk:
                            raise EOFError(f"reference ended {left} bytes short")
                        self._digest.update(chunk)
                        left -= len(chunk)
            except BaseException as error:  # reported by measurement()
                self._error = error
        self._thread = threading.Thread(target=read, daemon=True)
        self._thread.start()
        return []

    def measurement(self, processes) -> Measurement:
        self._thread.join()
        if self._error:
            raise RuntimeError(f"reference digest: {self._error}")
        return Measurement({"reference_sha256": self._digest.hexdigest()})


def components(spec: StreamSpec, backend: str) -> list:
    """One backend's pass: on the CPU also numpy's and libvmaf's PSNR. A part with
    per-frame rows comes first (metrics.combine takes the row count from it)."""
    psnr = [PsnrMetric(spec), libvmaf_psnr_metric(spec)] if backend == LIBVMAF_CPU_BACKEND \
        else []
    return [*psnr, *vmaf_v1_and_ms_ssim(spec, ms_ssim=True), ReferenceDigest(spec)]


def score_point(point: SampledPoint, directory: Path, clip: Clip, backend: str,
                encoder_psnr: bool, paths: Paths) -> dict:
    codec = Codec(point.codec)
    base = directory / point.key
    stream = base / f"stream{codec.bitstream_suffix}"
    frames = min(DEFAULT_FRAMES, clip.frames)
    fmt = YuvFormat(clip.width, clip.height)
    expected = (base / "stream.recon.txt").read_text().split() if point.own else None
    rate_log = base / "stream.rate.csv"
    out: dict = {"key": point.key}
    with tempdirs.temp_dir("metricval") as scratch, \
            ClipFrames(clip, frames, Path(scratch)) as source:
        spec = StreamSpec(source, fmt, frames, clip.fps, paths)
        parts = components(spec, backend)
        with gpu_slots(gpu_slot_need(parts)), MetricSession(spec, parts) as session:
            gate = gate_stream(stream, codec, fmt, frames, paths, expected_hashes=expected,
                               extra_sinks=session.sinks, extra_threads=session.cpu_threads,
                               standard=False,
                               coded_sizes=coded_format(rate_log, fmt) if point.own else None)
            if not gate.passed:
                raise RuntimeError(f"{point.key}: decode failed or differs from the "
                                   f"encoder's reconstruction: {gate.reason}")
            measured = session.result(frames)
        out["decoder"] = DECODERS[codec][0].name
        out["reconstruction_checked"] = expected is not None  # the gate compared every frame
        out["summary"] = measured.summary
        out["frames"] = measured.per_frame
        if encoder_psnr and point.own and (paths.data_root / point.binary).is_file():
            out["encoder"] = encoder_pass(point, stream, spec, source, Path(scratch), paths)
    return out


def encoder_pass(point: SampledPoint, stored_stream: Path, spec: StreamSpec,
                 source: ClipFrames, scratch: Path, paths: Paths) -> dict:
    """Encode the point again with `--sse-log` (the main chart's PSNR path)."""
    log = scratch / "sse.csv"
    metric = EncoderPsnrMetric(spec, log)
    out = scratch / f"again{stored_stream.suffix}"
    marks = {"binary": str(paths.data_root / point.binary), "input": str(source.reader()),
             "out": str(out), "recon": os.devnull, "rate_log": str(scratch / "rate.csv"),
             "times": str(scratch / "times.txt")}
    cmd = [part.format(**marks) if part.startswith("{") else part for part in point.command]
    subprocess.run(cmd + metric.encoder_arguments(), check=True, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    measured = metric.measurement(None)
    return {"stream_identical": out.read_bytes() == stored_stream.read_bytes(),
            "summary": measured.summary,
            "frames": [{k: r[k] for k in PLANES} for r in measured.per_frame]}


def score(args: argparse.Namespace) -> None:
    os.environ[METRICS_BACKEND_ENV] = args.backend
    paths = load_paths()
    directory = Path(args.sample)
    points = load_sample(directory)[:args.limit]
    clips = sample_clips(points, directory, paths)

    def one(point: SampledPoint) -> dict:
        try:
            result = score_point(point, directory, clips[(point.clipset, point.clip)],
                                 args.backend, args.encoder_psnr, paths)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            # one point's missing tool or failed decode is reported, not the whole run lost
            print(f"FAILED {point.key}: {error}", flush=True)
            return {"key": point.key, "error": str(error)}
        print(f"scored {point.key}", flush=True)
        return result
    with ThreadPoolExecutor(args.workers) as pool:
        scored = list(pool.map(one, points))
    Path(args.out).write_text(json.dumps({"backend": args.backend, "points": scored}) + "\n")


# --- compare ------------------------------------------------------------------


def frame_diffs(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mean and max absolute per-frame difference."""
    if len(a) != len(b):
        raise ValueError(f"frame counts differ: {len(a)} vs {len(b)}")
    diffs = [abs(x - y) for x, y in zip(a, b)]
    return statistics.fmean(diffs), max(diffs)


def capped(value: float) -> float:
    return min(PSNR_CAP_DB, value)


def compare_point(point: SampledPoint, cpu: dict, gpu: dict, stored: dict | None) -> dict:
    """One CSV row: VMAF and MS-SSIM CPU vs GPU, PSNR between every pair of tools."""
    row: dict = {"clip": point.clip, "height": point.height, "class": point.content_class,
                 "family": point.family, "kbps_target": point.kbps,
                 "sets": "+".join(point.sets),
                 "same_reference": cpu["summary"]["reference_sha256"]
                 == gpu["summary"]["reference_sha256"]}
    vc, vg = ([f["vmaf_v1"] for f in s["frames"]] for s in (cpu, gpu))
    row["vmaf_cpu"], row["vmaf_gpu"] = cpu["summary"]["vmaf_v1"], gpu["summary"]["vmaf_v1"]
    row["vmaf_seq_diff"] = row["vmaf_gpu"] - row["vmaf_cpu"]
    row["vmaf_frame_mean_abs"], row["vmaf_frame_max_abs"] = frame_diffs(vc, vg)
    mc, mg = ([f["ms_ssim"] for f in s["frames"]] for s in (cpu, gpu))
    row["msssim_seq_diff"] = gpu["summary"]["ms_ssim"] - cpu["summary"]["ms_ssim"]
    row["msssim_frame_max_abs"] = frame_diffs(mc, mg)[1]
    lib = {"summary": {p: cpu["summary"][f"lib_{p}"] for p in (*PLANES, "psnr_yuv")},
           "frames": [{p: f[f"lib_{p}"] for p in PLANES} for f in cpu["frames"]]}
    numpy = {"summary": cpu["summary"], "frames": cpu["frames"]}
    encoder = cpu.get("encoder")
    # the encoder's PSNR is of the stream it coded now: compared only where that is the
    # stored stream (a kept stream of another build than the record's is not)
    for name, other in (("numpy", numpy),
                        ("encoder", encoder if encoder and encoder["stream_identical"] else None)):
        if other is None:
            continue
        for plane in PLANES:
            row[f"{name}_vs_libvmaf_{plane}_max"] = frame_diffs(
                [capped(f[plane]) for f in other["frames"]],
                [capped(f[plane]) for f in lib["frames"]])[1]
        row[f"{name}_vs_libvmaf_psnr_yuv"] = other["summary"]["psnr_yuv"] - \
            lib["summary"]["psnr_yuv"]
    if cpu.get("encoder"):
        row["encoder_stream_identical"] = cpu["encoder"]["stream_identical"]
    row["kbps"] = stored["kbps"] if stored is not None else point.coded_kbps
    if stored is not None:
        row["stored_vmaf_vs_cpu"] = stored["vmaf_v1"] - row["vmaf_cpu"]
        row["stored_vmaf_frame_max_abs"] = frame_diffs(stored["frame_vmaf_v1"], vc)[1]
        row["stored_psnr_yuv_vs_libvmaf"] = stored["psnr_yuv"] - lib["summary"]["psnr_yuv"]
        row["stored_psnr_y_frame_max_abs"] = frame_diffs(
            stored["frame_psnr_y"], [f["psnr_y"] for f in lib["frames"]])[1]
    return row


def bd_vmaf(rows: list[dict], which: str) -> list[dict]:
    """BD-rate on VMAF v1 of each BD pair per clip, scored by `which` (cpu/gpu), at
    the stored results' achieved rates."""
    out = []
    bd_rows = [r for r in rows if "bd" in r["sets"].split("+")]
    for clip in sorted({r["clip"] for r in bd_rows}):
        for test, anchor in BD_PAIRS:
            curves = {}
            for family in (test, anchor):
                mine = sorted((r for r in bd_rows if r["clip"] == clip and r["family"] == family),
                              key=lambda r: r["kbps_target"])
                curves[family] = ([r["kbps"] for r in mine], [r[f"vmaf_{which}"] for r in mine])
            if min(len(c[0]) for c in curves.values()) < MIN_POINTS:
                continue  # the pair is not in this clip's BD set
            result = bd_rate(*curves[anchor], *curves[test])
            out.append({"clip": clip, "test": test, "anchor": anchor, "scorer": which,
                        "bd_vmaf": result.bd_rate_percent, "reliable": bool(result.reliable)})
    return out


def compare(args: argparse.Namespace) -> None:
    directory = Path(args.sample)
    points = load_sample(directory)
    cpu, gpu = ({p["key"]: p for p in json.loads(Path(f).read_text())["points"]}
                for f in (args.cpu, args.gpu))
    failed = sorted(k for k in cpu.keys() | gpu.keys()
                    if "error" in cpu.get(k, {}) or "error" in gpu.get(k, {}))
    rows = []
    for point in points:
        if point.key in failed:
            continue
        stored_file = directory / point.key / "stored.json"
        stored = json.loads(stored_file.read_text()) if stored_file.exists() else None
        rows.append(compare_point(point, cpu[point.key], gpu[point.key], stored))
    bd = [b for which in ("cpu", "gpu") for b in bd_vmaf(rows, which)]
    with open(args.out_csv, "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {**summarize(rows, bd), "failed": failed}
    Path(args.out_json).write_text(json.dumps({"summary": summary, "bd": bd}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def summarize(rows: list[dict], bd: list[dict]) -> dict:
    """Worst and mean differences. CPU-vs-GPU scores count only points whose two passes
    read the same reference frames (the clip decodes of two ffmpeg versions can differ)."""
    same = [r for r in rows if r["same_reference"]]

    def worst(key: str) -> float | None:
        values = [abs(r[key]) for r in (same if key.startswith(("vmaf", "msssim")) else rows)
                  if key in r]
        return max(values) if values else None

    def mean_abs(key: str) -> float | None:
        values = [abs(r[key]) for r in same if key in r]
        return statistics.fmean(values) if values else None
    pairs = {}
    for b in bd:
        pairs.setdefault((b["clip"], b["test"]), {})[b["scorer"]] = b["bd_vmaf"]
    differing = {r["clip"] for r in rows if not r["same_reference"]}
    bd_diffs = [v["gpu"] - v["cpu"] for (clip, _), v in pairs.items() if clip not in differing
                if v.get("gpu") is not None and v.get("cpu") is not None]
    return {
        "points": len(rows),
        "same_reference": len(same),
        "clips_decoded_differently": sorted({r["clip"] for r in rows if not r["same_reference"]}),
        "vmaf_seq_mean_abs": mean_abs("vmaf_seq_diff"), "vmaf_seq_max_abs": worst("vmaf_seq_diff"),
        "vmaf_frame_mean_abs": mean_abs("vmaf_frame_mean_abs"),
        "vmaf_frame_max_abs": worst("vmaf_frame_max_abs"),
        "msssim_seq_max_abs": worst("msssim_seq_diff"),
        "msssim_frame_max_abs": worst("msssim_frame_max_abs"),
        **{f"{tool}_vs_libvmaf_{plane}_max": worst(f"{tool}_vs_libvmaf_{plane}_max")
           for tool in ("numpy", "encoder") for plane in ("psnr_y", "psnr_u", "psnr_v")},
        "encoder_vs_libvmaf_psnr_yuv_max": worst("encoder_vs_libvmaf_psnr_yuv"),
        "numpy_vs_libvmaf_psnr_yuv_max": worst("numpy_vs_libvmaf_psnr_yuv"),
        "encoder_streams_identical": sum(bool(r.get("encoder_stream_identical")) for r in rows),
        "encoder_points": sum("encoder_stream_identical" in r for r in rows),
        "stored_vmaf_vs_cpu_max": worst("stored_vmaf_vs_cpu"),
        "stored_vmaf_frame_max_abs": worst("stored_vmaf_frame_max_abs"),
        "stored_psnr_yuv_vs_libvmaf_max": worst("stored_psnr_yuv_vs_libvmaf"),
        "stored_psnr_y_frame_max_abs": worst("stored_psnr_y_frame_max_abs"),
        "bd_vmaf_gpu_minus_cpu_max_abs": max(map(abs, bd_diffs)) if bd_diffs else None,
        "bd_pairs": len(bd_diffs),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--chart", nargs=3, action="append", required=True,
                   metavar=("CLIPSET", "STREAMS", "RECORD"),
                   help="a chart: its clip set, kept streams (prebuilt layout) and build "
                        "record; the BD set comes from the first")
    p.add_argument("--results", action="append", required=True,
                   help="a store of the charts' results (f150-ip300; repeat for several)")
    p.add_argument("--out", required=True)
    p.add_argument("--bd-clips", type=int, default=3)
    p.add_argument("--spread", type=int, default=12, help="spread points per chart")
    p = sub.add_parser("score")
    p.add_argument("--sample", required=True)
    p.add_argument("--backend", required=True, choices=(LIBVMAF_CPU_BACKEND, VMAFX_CUDA_BACKEND))
    p.add_argument("--out", required=True)
    p.add_argument("--encoder-psnr", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="only the first N points (a pipe-cleaner)")
    p = sub.add_parser("compare")
    p.add_argument("--sample", required=True)
    p.add_argument("--cpu", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-json", required=True)
    args = parser.parse_args(argv)
    {"sample": sample, "score": score, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
