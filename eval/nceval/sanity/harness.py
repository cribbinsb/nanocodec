"""Check 5: harness self-checks.

The anchors and the measurement chain are checked like the encoder is:

  - CBR anchors must hit their rate target (the openh264 CBR harness once
    missed by 100-270% from a QP-range misconfiguration and nobody looked);
  - curves must be usable: monotonic, not saturated, at least three points
    under the source cap, no conformance failures, nothing missing;
  - each anchor's command line must carry its documented settings;
  - every clip set must give each clip a source bitrate, or the 50% rule
    cannot be enforced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..clips import Clip
from ..encoders import resolve_encoder
from ..encoders.base import CodingStructure, EncodeJob
from ..config import Paths
from .curves import Curve, CurveStore
from .flags import Flag, Severity

MIN_LEGAL_POINTS = 3
# VMAF this close to its ceiling, or PSNR spanning less than this, carries no
# rate-quality information.
VMAF_SATURATED = 98.0
MIN_PSNR_SPAN_DB = 0.5
# Our encoder families in nceval.cbr results (the rest are anchors).
OUR_ENCODER_PREFIX = "nanocodec"
# CBR rate misses beyond this multiple of the limit are CRITICAL.
CRITICAL_CBR_MULTIPLE = 3.0
# An intra period / frame count no adapter would choose by itself, to see that
# the structure actually reaches the command line.
PROBE_INTRA_PERIOD = 777
PROBE_FRAMES = 333


# ----------------------------------------------------------------------------
# Curves


def curve_problems(curve: Curve, metrics: tuple[str, ...]) -> list[str]:
    """Why a curve cannot be trusted (empty if it is fine)."""
    problems = []
    if not curve.points and not curve.unscored and not curve.failures:
        return ["missing: no encodes stored"]
    if curve.failures:
        problems.append(f"{len(curve.failures)} conformance failure(s): "
                        + "; ".join(f"q{q}: {r[:80]}" for q, r in curve.failures[:3]))
    if len(curve.points) < MIN_LEGAL_POINTS:
        problems.append(f"only {len(curve.points)} scored point(s) under the source cap "
                        f"({len(curve.unscored)} over it)")
    by_q = sorted(curve.points, key=lambda p: p.quantizer)
    if any(b.kbps >= a.kbps for a, b in zip(by_q, by_q[1:])):
        problems.append("rate does not fall monotonically with quantizer")
    for metric in metrics:
        kbps, quality = curve.series(metric)
        pairs = sorted(zip(kbps, quality))
        if any(q2 <= q1 for (_, q1), (_, q2) in zip(pairs, pairs[1:])):
            problems.append(f"{metric} does not rise monotonically with rate")
        if metric.startswith("vmaf") and sum(q >= VMAF_SATURATED for q in quality) >= 2:
            problems.append(f"{metric} saturated (>= {VMAF_SATURATED:g} at 2+ points)")
        if metric.startswith("psnr") and len(quality) >= 2 and \
                max(quality) - min(quality) < MIN_PSNR_SPAN_DB:
            problems.append(f"{metric} spans under {MIN_PSNR_SPAN_DB} dB")
    return problems


def check_curves(store: CurveStore, clips: list[str], encoders: tuple[str, ...],
                 metrics: tuple[str, ...], label: str) -> list[Flag]:
    flags = []
    for encoder in encoders:
        for clip in clips:
            curve = store.curve(encoder, clip)
            problems = curve_problems(curve, metrics)
            if not problems:
                continue
            failed = bool(curve.failures)
            missing = problems[0].startswith("missing")
            flags.append(Flag(
                check="harness",
                severity=(Severity.CRITICAL if failed else
                          Severity.HIGH if missing else Severity.MEDIUM),
                subject=f"{label}/{clip}: {encoder} curve",
                finding="; ".join(problems),
                evidence={"points": [(p.quantizer, round(p.kbps, 1),
                                      {m: round(v, 3) for m, v in p.metrics.items()
                                       if m in metrics}) for p in curve.points],
                          "over_cap": [(p.quantizer, round(p.kbps, 1)) for p in curve.unscored]},
                suspected_cause=("conformance bug in the encoder" if failed else
                                 "encode error or unsupported clip (see the sweep log)"
                                 if missing else
                                 "QP grid outside the clip's useful range, or a rate model "
                                 "that is not monotonic")))
    return flags


# ----------------------------------------------------------------------------
# CBR rate targets


@dataclass(frozen=True)
class CbrPoint:
    family: str
    clip: str
    target_kbps: float
    kbps: float
    path: Path
    # Our encoder only: the rate error without the planned key-frame loan
    # still outstanding at the end of the clip (nceval.cbr, rate log).
    loan_adjusted_miss: float | None = None

    @property
    def raw_miss(self) -> float:
        return self.kbps / self.target_kbps - 1.0

    @property
    def ours(self) -> bool:
        return self.family.startswith(OUR_ENCODER_PREFIX)

    @property
    def error_magnitude(self) -> float:
        """How far the point is off target. For our encoder the target is a
        ceiling (docs/PROCEDURES.md, "Measure compression"): deliberate
        underspend where bits stop buying quality is not an error, only
        overshoot is. Anchors are judged on both sides."""
        return max(0.0, self.miss) if self.ours else abs(self.miss)

    @property
    def miss(self) -> float:
        """The loan-adjusted error where known (a planned key burst being
        repaid is not an overshoot), else the raw one (anchors)."""
        return self.raw_miss if self.loan_adjusted_miss is None else self.loan_adjusted_miss


def latest_cbr_points(root: Path) -> list[CbrPoint]:
    """The newest result per (family, clip, target) under a cbr.py results tree."""
    newest: dict[tuple, tuple[float, CbrPoint]] = {}
    for path in root.rglob("cbr*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if "target_kbps" not in data:
            continue
        loan = data.get("loan_adjusted_rate_error")
        point = CbrPoint(data["family"], data["clip"], float(data["target_kbps"]),
                         float(data["kbps"]), path,
                         None if loan is None else float(loan))
        key = (point.family, point.clip, point.target_kbps, path.parent.parent.parent.name)
        mtime = path.stat().st_mtime
        if key not in newest or mtime > newest[key][0]:
            newest[key] = (mtime, point)
    return [p for _, p in sorted(newest.values(), key=lambda item: str(item[1].path))]


def max_qp_floor(point: CbrPoint) -> tuple[int, float] | None:
    """(quantizer, kbps) of the coarsest fixed-QP point measured beside a CBR point.

    A variant family (`nanocodec-h264@keycad`) codes CBR only; its fixed-QP
    floor is the base family's (same encoder, same clip).
    """
    best = None
    base = point.path.parent.parent / point.family.split("@")[0]
    for path in [*point.path.parent.glob("qp*.json"), *base.glob("qp*.json")]:
        try:
            q, kbps = int(path.name[2:].split("-")[0]), float(json.loads(path.read_text())["kbps"])
        except (OSError, ValueError, KeyError):
            continue
        if best is None or q > best[0]:
            best = (q, kbps)
    return best


def check_cbr(root: Path, limit: float) -> list[Flag]:
    if not root.is_dir():
        return [Flag("harness", Severity.MEDIUM, f"CBR results {root}",
                     "no CBR results found, so CBR rate accuracy is unchecked")]
    flags = []
    for point in latest_cbr_points(root):
        if point.error_magnitude <= limit:
            continue
        floor = max_qp_floor(point)
        below_floor = floor is not None and floor[1] > point.target_kbps * (1 + limit)
        if below_floor:
            # The encoder is QP-pinned at its coarsest quantizer: the benchmark
            # chose an unreachable target, which is a harness problem.
            severity = Severity.MEDIUM
            cause = (f"target below the codec's floor: fixed QP {floor[0]} already gives "
                     f"{floor[1]:.0f} kbps; the benchmark should not use this target")
        else:
            severity = (Severity.CRITICAL if point.error_magnitude > CRITICAL_CBR_MULTIPLE * limit
                        else Severity.HIGH)
            cause = ("rate controller misses a reachable target (misconfigured min/max QP "
                     "or rate model)" if floor else
                     "rate controller misconfigured, or the target is below the clip's floor "
                     "(no fixed-QP floor measured for this encoder)")
        flags.append(Flag(
            check="harness", severity=severity,
            subject=f"CBR {point.family}/{point.clip} target {point.target_kbps:g} kbps",
            finding=f"achieved {point.kbps:.0f} kbps ({100 * point.miss:+.0f}% vs target"
                    + (", key-loan adjusted)" if point.loan_adjusted_miss is not None else ")"),
            evidence={"result": str(point.path), "limit": limit,
                      "raw_rate_error": round(point.raw_miss, 4),
                      "loan_adjusted_rate_error": point.loan_adjusted_miss,
                      "max_qp_floor": list(floor) if floor else None},
            suspected_cause=cause))
    return flags


# ----------------------------------------------------------------------------
# Anchor command lines


def command_mismatches(command: list[str], expected: dict[str, str]) -> list[str]:
    """Flags whose value differs from (or is missing in) a command line."""
    values = {a: b for a, b in zip(command, command[1:]) if a.startswith("-")}
    return [f"{flag}: expected {want}, got {values.get(flag, 'missing')}"
            for flag, want in expected.items() if values.get(flag) != want]


def probe_command(encoder: str, paths: Paths) -> list[str] | None:
    adapter = resolve_encoder(encoder)
    if adapter is None:
        return None
    clip = Clip(name="probe", yuv=Path("/probe.yuv"), width=1280, height=720, fps=30.0,
                frames=PROBE_FRAMES, source_kbps=1000.0, content_class="static", split="train")
    job = EncodeJob(clip=clip, quantizer=adapter.quantizer_range[0],
                    structure=CodingStructure(PROBE_INTRA_PERIOD), frames=PROBE_FRAMES,
                    bitstream=Path("/probe" + adapter.codec.bitstream_suffix))
    return adapter.command(job, paths)


def structure_flag(encoder: str) -> str:
    """The command-line flag that carries the intra period."""
    if encoder.startswith("svtav1"):
        return "--keyint"
    return "--intra-period" if encoder.startswith(("openh264", "nanocodec")) else ""


def check_commands(documented: dict[str, dict], paths: Paths) -> list[Flag]:
    flags = []
    for encoder, spec in documented.items():
        try:
            command = probe_command(encoder, paths)
        except FileNotFoundError as err:
            command, missing_tool = None, str(err)
        else:
            missing_tool = ""
        if command is None:
            flags.append(Flag("harness", Severity.HIGH, f"anchor {encoder} command line",
                              f"cannot build the command: {missing_tool or 'unknown encoder'}"))
            continue
        expected = dict(spec["flags"])
        period_flag = structure_flag(encoder)
        if period_flag:
            expected[period_flag] = str(PROBE_INTRA_PERIOD)
        mismatches = command_mismatches(command, expected)
        if mismatches:
            flags.append(Flag(
                check="harness", severity=Severity.HIGH,
                subject=f"anchor {encoder} command line",
                finding="differs from the documented settings: " + "; ".join(mismatches),
                evidence={"documented_in": spec["source"], "command": command},
                suspected_cause="adapter changed without updating the documentation, "
                                "or a regression in the adapter"))
    return flags


# ----------------------------------------------------------------------------
# Clip source caps


def check_clip_caps(clip_dir: Path) -> list[Flag]:
    """Every clip in every clip set needs a positive source bitrate (the 50% rule)."""
    flags = []
    for path in sorted(clip_dir.glob("*.json")):
        try:
            clips = json.loads(path.read_text()).get("clips", [])
        except (OSError, ValueError) as err:
            flags.append(Flag("harness", Severity.MEDIUM, f"clip set {path.name}",
                              f"unreadable: {err}"))
            continue
        bad = [c.get("name", "?") for c in clips
               if not isinstance(c.get("source_kbps"), (int, float)) or c["source_kbps"] <= 0]
        if bad:
            flags.append(Flag(
                check="harness", severity=Severity.HIGH, subject=f"clip set {path.name}",
                finding=f"{len(bad)} clip(s) without a source bitrate: the 50% cap cannot "
                        f"be enforced",
                evidence={"clips": bad},
                suspected_cause="clip set written by hand or by an old extractor; "
                                "re-run eval/clipset/extract.py"))
    return flags
