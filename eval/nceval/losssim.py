"""Quality under packet loss: our encoder and decoder in a closed loop with a
lossy network and receiver feedback (nceval.lossloop, nceval.lossnet).

    python3 -m nceval.losssim run --codec h264 [--preset default|fast] \\
        [--clips a,b] [--kbps 600] [--loss burst5] [--feedback-delay-ms 50] \\
        [--feedback-loss bernoulli:0.02] [--ltr FRAMES] [--layers 1|2|3] \\
        [--rtx ROUNDS] [--seed 1] [--frames 150] [--receiver transport|bitstream] \\
        [--param name=value]...
    python3 -m nceval.losssim matrix   [--workers 8] [--out NAME]
    python3 -m nceval.losssim startup  [--workers 8] [--out NAME]
    python3 -m nceval.losssim ltr      [--workers 8] [--out NAME]   (LTR interval sweep)
    python3 -m nceval.losssim checks   [--codecs a,b] [--clips a,b]   (mechanism checks)
    python3 -m nceval.losssim table    NAME   (matrix / startup tables, Markdown)

`--kbps` is the 720p30 reference rate, scaled to each clip's pixel rate as
the real-use screens scale theirs (nceval.cbr.clip_targets); it is the
encoder's payload target, the wire rate adds the packet headers.
`--ltr N`: a long-term reference marked every N frames (0: off, every loss
costs a key frame). Results go to ~/nanocodec-data/work/losssim/NAME.jsonl,
one line per run; a run already there (same binary and settings) is not
repeated. Nothing but those lines is left behind: the receiver's pictures
are scored as they stream. See docs/handoff/loss-resilience.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .cbr import BUFFER_MS, clip_targets
from .clipframes import ClipFrames
from .clips import Clip
from .config import Paths, load_paths
from .encoders.base import fps_fraction
from .lossloop import (Feedback, Session, enhancement_recovery_times, first_quality_tick,
                       frame_stats, percentile, recovery_times, run_loop, stall_stats)
from .lossnet import loss_model
from .metrics import MetricSession, PsnrMetric, StreamSpec, gpu_slot_need, real_metrics
from .pool import load_pool
from .slots import cpu_slots, gpu_slots
from .tune import read_preset
from .yuv import YuvFormat

PRESETS_DIR = Path(__file__).resolve().parents[2] / "presets"
#: Seconds between periodic key frames (the live-stream setting).
INTRA_PERIOD_SECONDS = 10
#: VMAF points below the settled quality that still count as a good picture.
QUALITY_MARGIN_VMAF = 6.0
#: The same for PSNR-Y (dB), when a run scores PSNR only.
QUALITY_MARGIN_PSNR_DB = 2.0
#: Seed streams: forward loss, feedback loss, S1 loss, S2 loss.
FORWARD_STREAM, FEEDBACK_STREAM, ENHANCEMENT_STREAM, TOP_ENHANCEMENT_STREAM = 1, 2, 3, 4
#: Clean 720p pool clips of three classes (crowd, pan, low light).
DEFAULT_CLIPS = ("external_nyc_crowd_720p", "fit_pan_jaad_0127", "fit_lowlight_police")
MAX_WORKERS = 8


@dataclass(frozen=True)
class RunConfig:
    codec: str
    preset: str = "default"  # default | fast
    kbps: int = 600
    loss: str = "burst5"
    feedback_delay_ms: float = 50.0
    #: Feedback loss model; "same" takes the forward model's.
    feedback_loss: str = "same"
    ltr: int = 15
    layers: int = 1
    rtx: int = 0
    seed: int = 1
    frames: int = 150
    receiver: str = "transport"
    params: tuple[str, ...] = ()
    metrics: bool = True
    #: "real": PSNR, VMAF v1 and MS-SSIM (GPU); "psnr": PSNR only, enough
    #: for start-up times and cheaper (no GPU slot).
    scores: str = "real"
    scalability: str = ""
    operating_point: int = 0
    enhancement_loss: str = "same"
    top_enhancement_loss: str = "same"


def config_dict(cfg: RunConfig) -> dict:
    """The run's settings as stored (fields added later are left out at
    their defaults, so earlier results keep their identity)."""
    settings = asdict(cfg)
    if settings["scores"] == "real":
        del settings["scores"]
    if not settings["scalability"]:
        del settings["scalability"]
    if settings["operating_point"] == 0:
        del settings["operating_point"]
    if settings["enhancement_loss"] == "same":
        del settings["enhancement_loss"]
    if settings["top_enhancement_loss"] == "same":
        del settings["top_enhancement_loss"]
    return settings


def preset_params(codec: str, preset: str) -> list[str]:
    if preset == "default":
        return []
    values = read_preset(PRESETS_DIR / f"{codec}-{preset}.toml")
    return [f"{k}={v:g}" for k, v in sorted(values.items())]


def session_command(binary: Path, cfg: RunConfig, clip: Clip, source: Path, kbps: int) -> list[str]:
    num, den = fps_fraction(clip.fps)
    params = preset_params(cfg.codec, cfg.preset) + list(cfg.params)
    layer_flags = (["--scalability", cfg.scalability] if cfg.scalability
                   else ["--temporal-layers", str(cfg.layers)])
    op_flags = ["--operating-point", str(cfg.operating_point)] if cfg.operating_point else []
    ltr_flags = ["--ltr-interval", str(cfg.ltr)] if cfg.ltr else []
    return [str(binary), "session", "--codec", cfg.codec, "--input", str(source),
            "--width", str(clip.width), "--height", str(clip.height), "--fps", f"{num}/{den}",
            "--frames", str(cfg.frames),
            "--intra-period", str(round(clip.fps * INTRA_PERIOD_SECONDS)),
            *layer_flags, "--bitrate", str(kbps),
            "--buffer-ms", str(BUFFER_MS), *ltr_flags, *op_flags,
            "--receiver", cfg.receiver, *(x for p in params for x in ("--param", p))]


def clip_kbps(clip: Clip, kbps: int) -> int:
    legal = clip_targets(clip, (kbps,))
    if not legal:
        raise ValueError(f"{clip.name}: {kbps} kbps (720p30) is over half its source rate")
    return legal[0]


def simulate(cfg: RunConfig, clip: Clip, binary: Path, paths: Paths,
             keep_trace: bool = False) -> dict:
    """One closed-loop run of `clip`: its statistics and scores (and with
    `keep_trace`, its Trace under "trace", which is not stored)."""
    if clip.frames < cfg.frames:
        raise ValueError(f"{clip.name} has {clip.frames} frames, fewer than {cfg.frames}")
    fmt = YuvFormat(clip.width, clip.height)
    kbps = clip_kbps(clip, cfg.kbps)
    forward = loss_model(cfg.loss, hash_seed(cfg.seed, FORWARD_STREAM, clip.name))
    enh_forward = (None if cfg.enhancement_loss == "same"
                   else loss_model(cfg.enhancement_loss,
                                   hash_seed(cfg.seed, ENHANCEMENT_STREAM, clip.name)))
    top_forward = (None if cfg.top_enhancement_loss == "same"
                   else loss_model(cfg.top_enhancement_loss,
                                   hash_seed(cfg.seed, TOP_ENHANCEMENT_STREAM, clip.name)))
    feedback_spec = cfg.loss if cfg.feedback_loss == "same" else cfg.feedback_loss
    feedback = Feedback(cfg.feedback_delay_ms,
                        loss_model(feedback_spec, hash_seed(cfg.seed, FEEDBACK_STREAM, clip.name)),
                        cfg.rtx)
    with tempfile.TemporaryDirectory(prefix="nc-losssim-", dir=paths.work) as scratch, \
            ClipFrames(clip, cfg.frames, Path(scratch)) as source:
        spec = StreamSpec(clip.metric_reference if clip.direct else source, fmt, cfg.frames,
                          clip.fps, paths)
        metrics = ([] if not cfg.metrics else
                   real_metrics(spec) if cfg.scores == "real" else [PsnrMetric(spec)])
        with gpu_slots(gpu_slot_need(metrics)), \
                cpu_slots(2 + sum(m.cpu_threads for m in metrics)), \
                MetricSession(spec, metrics) as scoring:
            session = Session(session_command(binary, cfg, clip, source.reader(), kbps), fmt,
                              scoring.sinks)
            try:
                trace = run_loop(session, cfg.frames, clip.fps, forward, feedback,
                                 enhancement_forward=enh_forward,
                                 top_enhancement_forward=top_forward)
                shown = session.close()
            except BaseException:
                session.kill()
                raise
            measurement = scoring.result(shown) if metrics else None
    tick_ms = 1000.0 / clip.fps
    result = {"config": config_dict(cfg), "clip": clip.name, "class": clip.content_class,
              "fps": clip.fps, "kbps_target": kbps, "binary": binary_hash(binary)}
    result.update(stall_stats(trace.screen, tick_ms))
    result.update(frame_stats(trace, clip.fps, round(clip.fps * INTRA_PERIOD_SECONDS)))
    recoveries = recovery_times(trace, tick_ms)
    result["recoveries"] = len(recoveries)
    result["recovery_ms_mean"] = statistics.fmean(recoveries) if recoveries else None
    result["recovery_ms_p95"] = percentile(recoveries, 95) if recoveries else None
    result["recovery_ms_max"] = max(recoveries, default=None)
    if trace.enhancement_screen:
        enh_rec = enhancement_recovery_times(trace, tick_ms)
        result["enhancement_recoveries"] = len(enh_rec)
        result["enhancement_recovery_ms_mean"] = statistics.fmean(enh_rec) if enh_rec else None
        result["enhancement_recovery_ms_p95"] = percentile(enh_rec, 95) if enh_rec else None
        result["enhancement_recovery_ms_max"] = max(enh_rec, default=None)
    first_key = next((f for f in trace.frames if f.kind == "key"), None)
    result["first_key_bytes"] = first_key.bytes if first_key else None
    result["first_key_packets"] = first_key.packets if first_key else None
    result["keys_before_first_picture"] = sum(
        f.kind == "key" for f in trace.frames
        if result["first_picture_ms"] is None or f.tick * tick_ms <= result["first_picture_ms"])
    result["screen"] = "".join(s[0] for s in trace.screen)
    if keep_trace:
        result["trace"] = trace
    if measurement is not None:
        result.update({k: v for k, v in measurement.summary.items()
                       if k in ("psnr_yuv", "vmaf_v1", "ms_ssim")})
        if cfg.scores == "real":
            series = [row["vmaf_v1"] for row in measurement.per_frame]
            margin = QUALITY_MARGIN_VMAF
        else:
            series = [row["psnr_y"] for row in measurement.per_frame]
            margin = QUALITY_MARGIN_PSNR_DB
        good = [v for s, v in zip(trace.screen, series) if s == "good"]
        result["quality_good_frames"] = statistics.fmean(good) if good else None
        quality_tick = first_quality_tick(trace.screen, series, margin)
        result["first_quality_ms"] = None if quality_tick is None else quality_tick * tick_ms
    return result


def hash_seed(*parts) -> int:
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:12], 16)


_BINARY_HASHES: dict[Path, str] = {}


def binary_hash(binary: Path) -> str:
    if binary not in _BINARY_HASHES:
        _BINARY_HASHES[binary] = hashlib.sha256(binary.read_bytes()).hexdigest()[:16]
    return _BINARY_HASHES[binary]


def run_key(cfg: RunConfig, clip: str, binary: Path) -> str:
    return json.dumps([config_dict(cfg), clip, binary_hash(binary)], sort_keys=True)


class ResultFile:
    """Runs of one study, one JSON line each; finished runs are skipped."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.done: set[str] = set()
        if path.exists():
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                self.done.add(json.dumps([entry["config"], entry["clip"], entry["binary"]],
                                         sort_keys=True))

    def add(self, result: dict) -> None:
        with self._lock, self.path.open("a") as out:
            out.write(json.dumps(result, sort_keys=True) + "\n")

    def entries(self) -> list[dict]:
        return [json.loads(line) for line in self.path.read_text().splitlines()]


def run_all(jobs: list[tuple[RunConfig, Clip]], binary: Path, paths: Paths, out: ResultFile,
            workers: int) -> None:
    todo = [(cfg, clip) for cfg, clip in jobs if run_key(cfg, clip.name, binary) not in out.done]
    print(f"{len(todo)} runs ({len(jobs) - len(todo)} already done) -> {out.path}", flush=True)
    failures = []

    def one(job: tuple[RunConfig, Clip]) -> None:
        cfg, clip = job
        try:
            out.add(simulate(cfg, clip, binary, paths))
        except Exception as error:  # reported at the end; the other runs go on
            failures.append(f"{clip.name} {cfg}: {error}")
            print(f"FAILED {clip.name} {cfg}: {error}", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(min(workers, MAX_WORKERS)) as pool:
        list(pool.map(one, todo))
    if failures:
        raise SystemExit(f"{len(failures)} runs failed")


def pool_clips(names: list[str], paths: Paths) -> list[Clip]:
    by_name = {clip.name: clip for clip in load_pool(paths)}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"not in the clip pool: {', '.join(missing)}")
    return [by_name[n] for n in names]


def default_binary(paths: Paths) -> Path:
    return paths.nanocodec_bin / "nanocodec"


# ---------------------------------------------------------------- studies

#: Codecs with a long-term reference (VVC has none yet).
MATRIX_CODECS = ("h264", "hevc", "av1", "vp9", "av2")
MATRIX_LOSSES = ("none", "burst2", "burst5", "uniform5")
MATRIX_DELAYS_MS = (25.0, 100.0)
MATRIX_SEEDS = (1,)


def matrix_jobs(clips: list[Clip]) -> list[tuple[RunConfig, Clip]]:
    """Every codec's Default and Fast (VP9/AV2: Default only for layers),
    LTR on and off, flat; layered (L1T3) on burst5 with LTR on and off."""
    jobs = []
    for codec in MATRIX_CODECS:
        for preset in ("default", "fast"):
            for loss in MATRIX_LOSSES:
                delays = MATRIX_DELAYS_MS if loss != "none" else MATRIX_DELAYS_MS[:1]
                for delay in delays:
                    for ltr in (0, 15):
                        for seed in MATRIX_SEEDS if loss != "none" else (1,):
                            base = RunConfig(codec, preset, loss=loss, feedback_delay_ms=delay,
                                             ltr=ltr, seed=seed)
                            jobs += [(base, clip) for clip in clips]
        for ltr in (0, 15):
            for seed in MATRIX_SEEDS:
                base = RunConfig(codec, "default", loss="burst5", feedback_delay_ms=25.0,
                                 ltr=ltr, layers=3, seed=seed)
                jobs += [(base, clip) for clip in clips]
    return jobs


#: In-band loss detection: the receiver with and without the transport's
#: dependency signal, everything else equal (Default, flat).
DETECT_LOSSES = ("burst5", "uniform5")
DETECT_RECEIVERS = ("transport", "bitstream")


def detect_jobs(clips: list[Clip]) -> list[tuple[RunConfig, Clip]]:
    """Every codec's Default on burst5 and uniform5, both delays, LTR on and
    off, with the transport receiver and the bitstream-only one."""
    return [(RunConfig(codec, loss=loss, feedback_delay_ms=delay, ltr=ltr, receiver=receiver),
             clip)
            for codec in MATRIX_CODECS for loss in DETECT_LOSSES for delay in MATRIX_DELAYS_MS
            for ltr in (0, 15) for receiver in DETECT_RECEIVERS for clip in clips]


#: Startup study: loss rates, rates and the mitigations compared.
STARTUP_LOSSES = ("burst2", "burst5", "burst10")
#: Uniform loss is harder on many-packet keys than bursts of the same rate:
#: it is run at the higher rate (its keys take ~50 packets), without the
#: combined arm.
STARTUP_UNIFORM = ("uniform5", "uniform10")
STARTUP_UNIFORM_KBPS = 1000
STARTUP_KBPS = (350, 1000)
STARTUP_CODECS = ("h264", "hevc", "av1")
STARTUP_SEEDS = (1, 2, 3, 4)
STARTUP_FRAMES = 60
#: Mitigation (a), a smaller key: every key (the first and those answering
#: a loss) at most this share of the 1 s buffer, and the first key without
#: its extra budget (`rate.first_key_boost_scale`); quality then rises
#: through the inter frames.
def key_cap(share: float) -> tuple[str, ...]:
    return (f"rate.key_cap_buffer={share:g}", "rate.first_key_boost_scale=1")


#: name: (encoder parameters, NACK + RTX rounds). (c) is retransmission.
STARTUP_ARMS = {
    "baseline": ((), 0),
    "keycap0.25": (key_cap(0.25), 0),
    "keycap0.1": (key_cap(0.1), 0),
    "rtx": ((), 3),
    "keycap0.1+rtx": (key_cap(0.1), 3),
}


#: LTR mark interval sweep: one slot means a loss reported while the
#: newest mark awaits its acknowledgement (one round trip) needs a key, so
#: the share of losses recovered by a key grows as RTT / interval.
LTR_INTERVALS = (0, 15, 30, 60)


def ltr_jobs(clips: list[Clip]) -> list[tuple[RunConfig, Clip]]:
    return [(RunConfig(codec, loss="burst5", feedback_delay_ms=delay, ltr=ltr, seed=seed), clip)
            for codec in ("h264", "hevc", "av1") for delay in (25.0, 100.0)
            for ltr in LTR_INTERVALS for seed in (1, 2) for clip in clips]


def startup_jobs(clips: list[Clip]) -> list[tuple[RunConfig, Clip]]:
    jobs = []
    for codec in STARTUP_CODECS:
        for kbps in STARTUP_KBPS:
            losses = STARTUP_LOSSES + (STARTUP_UNIFORM if kbps == STARTUP_UNIFORM_KBPS else ())
            for loss in losses:
                for arm, (params, rtx) in STARTUP_ARMS.items():
                    if loss in STARTUP_UNIFORM and params and rtx:
                        continue
                    for seed in STARTUP_SEEDS:
                        cfg = RunConfig(codec, "default", kbps=kbps, loss=loss,
                                        feedback_delay_ms=50.0, ltr=15, rtx=rtx, seed=seed,
                                        frames=STARTUP_FRAMES, params=params, scores="psnr")
                        jobs += [(cfg, clip) for clip in clips]
    return jobs


# ---------------------------------------------------------------- mechanism checks

#: Scenarios of `checks`, per codec: (name, settings).
CHECK_SCENARIOS = (
    ("ltr", RunConfig("", loss="burst5", feedback_delay_ms=50.0, ltr=15)),
    ("key-only", RunConfig("", loss="burst5", feedback_delay_ms=50.0, ltr=0)),
    ("layers3", RunConfig("", loss="burst5", feedback_delay_ms=25.0, ltr=15, layers=3,
                          params=("layers.adaptive=0",))),
    ("slow-lossy-feedback", RunConfig("", loss="burst5", feedback_delay_ms=150.0,
                                      feedback_loss="bernoulli:0.2", ltr=15)),
)
#: (e): most repairs per loss report the encoder acts on, and the longest
#: freeze allowed after a loss, in round trips plus this many frames.
MAX_REPAIRS_PER_LOSS_EVENT = 1.5
STALL_SLACK_FRAMES = 3


def mechanism_checks(codec: str, clips: list[Clip], binary: Path, paths: Paths) -> list[dict]:
    """Per scenario: the (a)-(e) checks of docs/handoff/loss-resilience.md."""
    rows = []
    for name, base in CHECK_SCENARIOS:
        cfg = replace(base, codec=codec, metrics=False, seed=3)
        for clip in clips:
            result = simulate(cfg, clip, binary, paths, keep_trace=True)
            rows.append({"scenario": name, "clip": clip.name,
                         **check_trace(cfg, result.pop("trace"), clip.fps), **{
                             k: result[k] for k in ("frames_lost", "response_keys",
                                                    "recovery_frames", "mismatches",
                                                    "longest_stall_ms")}})
    return rows


def check_trace(cfg: RunConfig, trace, fps: float) -> dict:
    """What the encoder did with each loss report it received."""
    by_tick = {f.tick: f for f in trace.frames}
    by_index = {f.index: f for f in trace.frames}
    responses = {"key": 0, "recovery": 0, "none": 0}
    top_layer_repairs = 0
    top_layer_reports = 0
    acted = 0
    for tick, lost in trace.reports_applied:
        frame = by_tick.get(tick)  # the frame coded right after the report
        if frame is None:
            continue
        response = ("key" if frame.kind == "key" else
                    "recovery" if frame.recovery else "none")
        lost_frame = by_index.get(lost)
        top = (lost_frame is not None and cfg.layers > 1
               and lost_frame.layer == cfg.layers - 1)
        top_layer_reports += top
        if response != "none":
            acted += 1
            # (c) a repair right after a report of a top-layer frame alone
            reports_now = [i for t, i in trace.reports_applied if t == tick]
            if top and all(by_index.get(i) is not None and by_index[i].layer == cfg.layers - 1
                           for i in reports_now):
                top_layer_repairs += 1
        responses[response] += 1
    recovery_ok = all(f.status in ("good", "lost") or not f.arrived
                      for f in trace.frames if f.recovery)
    events = len(recovery_times(trace, 1000.0 / fps)) or 1
    rtt_frames = 2 * cfg.feedback_delay_ms * fps / 1000
    return {
        "reports_applied": len(trace.reports_applied),
        "responses": responses,
        "recovery_frames_decoded_good": sum(f.recovery and f.status == "good"
                                            for f in trace.frames),
        "a_recovery_frames_decode": recovery_ok,
        "b_key_only_sends_keys": cfg.ltr > 0 or responses["recovery"] == 0,
        "top_layer_reports": top_layer_reports,
        "c_top_layer_losses_need_nothing": top_layer_repairs == 0,
        "e_repairs_per_loss_event": acted / events,
        "e_stall_bound_ms": (2 * rtt_frames + STALL_SLACK_FRAMES) * 1000 / fps,
    }


def checks_table(rows: list[dict]) -> str:
    lines = ["| codec | scenario | lost frames | reports acted on: key / recovery / none "
             "| recovery frames good | mismatches | top-layer reports, none repaired "
             "| repairs per loss event "
             "| longest stall ms |", "|" + "---|" * 9]
    for row in rows:
        r = row["responses"]
        lines.append(
            f"| {row['codec']} | {row['scenario']} | {row['frames_lost']} "
            f"| {r['key']} / {r['recovery']} / {r['none']} | {row['recovery_frames_decoded_good']} "
            f"| {row['mismatches']} | {row['top_layer_reports']}, "
            f"{'yes' if row['c_top_layer_losses_need_nothing'] else 'NO'} "
            f"| {row['e_repairs_per_loss_event']:.2f} | {row['longest_stall_ms']:.0f} |")
    return "\n".join(lines)


STUDIES = {"matrix": matrix_jobs, "startup": startup_jobs, "ltr": ltr_jobs,
           "detect": detect_jobs}


# ---------------------------------------------------------------- tables

def mean_of(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def fmt(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def matrix_table(entries: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = {}
    for e in entries:
        c = e["config"]
        key = (c["codec"], c["preset"], c["layers"], c["loss"], c["feedback_delay_ms"], c["ltr"],
               " ".join(c["params"]), c["receiver"])
        groups.setdefault(key, []).append(e)
    lines = ["| codec | preset | layers | loss | one-way ms | LTR | params | receiver | PSNR-YUV "
             "| VMAF v1 | MS-SSIM "
             "| frozen % | longest stall ms | recovery ms mean / p95 | resp. keys | recovery frames "
             "| recovery kbps | wire kbps | mismatches |",
             "|" + "---|" * 19]
    for key in sorted(groups):
        rows = groups[key]
        codec, preset, layers, loss, delay, ltr, params, receiver = key
        lines.append(
            f"| {codec} | {preset} | {layers} | {loss} | {delay:g} | {ltr or 'off'} | {params} "
            f"| {receiver} "
            f"| {fmt(mean_of(rows, 'psnr_yuv'), 2)} | {fmt(mean_of(rows, 'vmaf_v1'), 2)} "
            f"| {fmt(mean_of(rows, 'ms_ssim'), 4)} | {fmt(100 * mean_of(rows, 'frozen_share'))} "
            f"| {fmt(mean_of(rows, 'longest_stall_ms'), 0)} "
            f"| {fmt(mean_of(rows, 'recovery_ms_mean'), 0)} / {fmt(mean_of(rows, 'recovery_ms_p95'), 0)} "
            f"| {fmt(mean_of(rows, 'response_keys'))} | {fmt(mean_of(rows, 'recovery_frames'))} "
            f"| {fmt(mean_of(rows, 'recovery_kbps'))} | {fmt(mean_of(rows, 'wire_kbps'), 0)} "
            f"| {sum(r['mismatches'] for r in rows)} |")
    return "\n".join(lines)


def packets_table(entries: list[dict]) -> str:
    """Packets per frame, inter vs key, per codec and rate (loss-free runs)."""
    groups: dict[tuple, list[dict]] = {}
    for e in entries:
        c = e["config"]
        groups.setdefault((c["codec"], c["preset"], c["kbps"]), []).append(e)
    lines = ["| codec | preset | kbps (720p30) | inter packets mean / p95 / max "
             "| key packets mean / max | first key kB |", "|---|---|---|---|---|---|"]
    for key in sorted(groups):
        rows = groups[key]
        inter = [r["packets_inter"] for r in rows]
        key_packets = [r["packets_key"] for r in rows if r["packets_key"]["max"]]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} "
            f"| {fmt(statistics.fmean(p['mean'] for p in inter), 2)} / "
            f"{fmt(statistics.fmean(p['p95'] for p in inter))} / {max(p['max'] for p in inter)} "
            f"| {fmt(statistics.fmean(p['mean'] for p in key_packets))} / "
            f"{max((p['max'] for p in key_packets), default=0)} "
            f"| {fmt(mean_of(rows, 'first_key_bytes') / 1000 if mean_of(rows, 'first_key_bytes') else None)} |")
    return "\n".join(lines)


def startup_table(entries: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = {}
    for e in entries:
        c = e["config"]
        if c["seed"] not in STARTUP_SEEDS:
            continue
        arm = next((name for name, (params, rtx) in STARTUP_ARMS.items()
                    if tuple(params) == tuple(c["params"]) and rtx == c["rtx"]), "?")
        groups.setdefault((c["codec"], c["kbps"], c["loss"], arm), []).append(e)
    lines = ["| codec | kbps | loss | start | first key packets | first picture ms mean / p90 "
             "| first good picture ms mean / p90 | no picture in run % | keys sent before it "
             "| PSNR-YUV |", "|" + "---|" * 10]
    for key in sorted(groups):
        rows = groups[key]
        horizon = STARTUP_FRAMES * 1000.0 / rows[0]["fps"]
        first = [r["first_picture_ms"] if r["first_picture_ms"] is not None else horizon
                 for r in rows]
        quality = [r.get("first_quality_ms") if r.get("first_quality_ms") is not None else horizon
                   for r in rows]
        never = sum(r["first_picture_ms"] is None for r in rows) / len(rows)
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {fmt(mean_of(rows, 'first_key_packets'))} "
            f"| {fmt(statistics.fmean(first), 0)} / {fmt(percentile(first, 90), 0)} "
            f"| {fmt(statistics.fmean(quality), 0)} / {fmt(percentile(quality, 90), 0)} "
            f"| {fmt(100 * never, 0)} | {fmt(mean_of(rows, 'keys_before_first_picture'))} "
            f"| {fmt(mean_of(rows, 'psnr_yuv'), 2)} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nceval.losssim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="one configuration on some clips")
    run.add_argument("--codec", required=True)
    run.add_argument("--preset", default="default")
    run.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
    run.add_argument("--kbps", type=int, default=600)
    run.add_argument("--loss", default="burst5")
    run.add_argument("--enhancement-loss", default="same")
    run.add_argument("--top-enhancement-loss", default="same")
    run.add_argument("--feedback-delay-ms", type=float, default=50.0)
    run.add_argument("--feedback-loss", default="same")
    run.add_argument("--ltr", type=int, default=None)
    run.add_argument("--layers", type=int, default=1)
    run.add_argument("--scalability", default="")
    run.add_argument("--operating-point", type=int, default=0)
    run.add_argument("--rtx", type=int, default=0)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--frames", type=int, default=150)
    run.add_argument("--receiver", default="transport", choices=("transport", "bitstream"))
    run.add_argument("--param", action="append", default=[])
    run.add_argument("--no-metrics", action="store_true")
    for study in (run, *(sub.add_parser(name) for name in STUDIES)):
        study.add_argument("--workers", type=int, default=MAX_WORKERS)
        study.add_argument("--out", default=None)
        study.add_argument("--binary", type=Path, default=None)
    checks = sub.add_parser("checks", help="mechanism checks (a)-(e), per codec")
    checks.add_argument("--codecs", default=",".join(MATRIX_CODECS))
    checks.add_argument("--clips", default=",".join(DEFAULT_CLIPS))
    checks.add_argument("--binary", type=Path, default=None)
    table = sub.add_parser("table")
    table.add_argument("name")
    args = parser.parse_args(argv)
    paths = load_paths()
    directory = paths.work / "losssim"
    directory.mkdir(parents=True, exist_ok=True)
    if args.command == "table":
        entries = ResultFile(directory / f"{args.name}.jsonl").entries()
        if any(e["config"]["frames"] == STARTUP_FRAMES for e in entries):
            print(startup_table(entries))
        else:
            print(matrix_table(entries))
        print()
        print(packets_table([e for e in entries if e["config"]["loss"] in ("none", "burst2")
                             or e["config"]["frames"] == STARTUP_FRAMES]))
        return
    binary = (args.binary or default_binary(paths)).resolve()
    if args.command == "checks":
        clips = pool_clips(args.clips.split(","), paths)
        rows = []
        for codec in args.codecs.split(","):
            rows += [{"codec": codec, **row}
                     for row in mechanism_checks(codec, clips, binary, paths)]
        (directory / "checks.json").write_text(json.dumps(rows, indent=1))
        print(checks_table(rows))
        return
    out = ResultFile(directory / f"{args.out or args.command}.jsonl")
    if args.command == "run":
        is_spatial = (args.scalability.upper().startswith(("L2", "L3"))
                      or any(p.startswith(("spatial.layers=2", "spatial.layers=3"))
                             for p in args.param))
        ltr = (0 if is_spatial else 15) if args.ltr is None else args.ltr
        cfg = RunConfig(args.codec, args.preset, args.kbps, args.loss, args.feedback_delay_ms,
                        args.feedback_loss, ltr, args.layers, args.rtx, args.seed,
                        args.frames, args.receiver, tuple(args.param), not args.no_metrics,
                        scalability=args.scalability, operating_point=args.operating_point,
                        enhancement_loss=args.enhancement_loss,
                        top_enhancement_loss=args.top_enhancement_loss)
        clips = pool_clips(args.clips.split(","), paths)
        run_all([(cfg, clip) for clip in clips], binary, paths, out, args.workers)
        for entry in out.entries()[-len(clips):]:
            print(json.dumps({k: v for k, v in entry.items() if k != "screen"}, indent=None))
            print("screen:", entry["screen"])
        return
    clips = pool_clips(list(DEFAULT_CLIPS), paths)
    run_all(STUDIES[args.command](clips), binary, paths, out, args.workers)


if __name__ == "__main__":
    main()
