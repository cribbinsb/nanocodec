#!/usr/bin/env python3
"""The report stage of BP-3: checks every stored point, then BD, compliance and speed tables.

    python3 tools/bench/report.py RUN_DIR [RUN_DIR ...] [--out report.json]

RUN_DIR holds the plan's `jobs.json` and `manifest.json`, and one directory per machine series
(`series/<name>/`) with the box stages' `encodes.jsonl` and `scores.jsonl`
(docs/benchmark-procedure.md sections 8-10 and 13). The report reads stored records only: nothing is
encoded, decoded or measured here. Its checks refuse the whole report (section 10):

  - V13 every planned point of every series has an encode record and a score record;
  - V23 every score was taken from a stream that series' boxes coded (the stream's SHA-256 is
        among that series' encode records), so no series borrows another's quality;
  - V6  each series' canary times at the middle and end are within CANARY_SET of the start
        (geometric mean) and each canary encoder within CANARY_ONE;
  - V14 the curve rules of section 9 and the clip sets: a family without a BD on a clip
        loses that clip only, the headline is the common set, and each lost clip is listed
        with its reason;
  - V15 one bucket replay for every encoder (section 7), published as continuous compliance
        figures beside the BD; no encoder's own rate log is read.

Aggregation is section 8's: per-clip BD against the x264 medium zero line with the tune
matched to the metric, the composite BD-J against the default tune, the set mean of the
per-clip log rate ratio with equal weight per clip, a seeded 95 % bootstrap over clips, and
speed as the geometric mean over clips of megapixels per CPU-second.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance  # noqa: E402
import wholebox  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
from nceval import bdj  # noqa: E402
from nceval.metrics import weighted_yuv  # noqa: E402

#: Section 9: curve rules (nceval.bdj, the one definition the dev screens share).
MIN_POINTS = bdj.MIN_POINTS
#: Section 9: an overlap under this fraction of the smaller span is counted as thin beside
#: each family's figure, never excluded (owner, 2026-11-25, BP-3.14).
THIN_OVERLAP = bdj.THIN_OVERLAP_FRACTION
OURS_PREFIX = "nanocodec-"
DROP_SUFFIX = "+drop"
#: Section 8: statistics (a test compares them with the procedure's text, V20).
BOOTSTRAP_RESAMPLES = 10_000
INTERVAL = 0.95
BOOTSTRAP_SEED = 20261123
#: Section 9: the zero line, x264 medium, in its best tune for each metric (owner,
#: 2026-11-25): ~psnr beats the default tune by 20.6 % on VMAF v1 and 12.5 % on J on the
#: BP-3 "ours" CBR points (docs/analysis/bp4-adversarial-review-2026-11-25.md, finding 1).
ZERO_FAMILY = {"psnr_yuv": "x264-medium~psnr", "ms_ssim": "x264-medium~ssim",
               "vmaf_v1": "x264-medium~psnr", "j": "x264-medium~psnr"}
METRICS = bdj.METRICS


def psnr_yuv(pooled: dict) -> float | None:
    """The weighted PSNR of a point's pooled planes; None when a plane has no figure."""
    planes = [pooled.get(k) for k in ("psnr_y", "psnr_u", "psnr_v")]
    return None if None in planes else weighted_yuv(*planes)


#: A point's raw value per metric; None when the stored score has none (the point is then not
#: a point of that metric's curve, nceval.bdj.has_metric, and is counted in the report).
RAW = {"psnr_yuv": psnr_yuv, "vmaf_v1": lambda p: p.get("vmaf_v1"),
       "ms_ssim": lambda p: p.get("ms_ssim")}
#: BP-3.6: a box whose two timing references disagree on its speed by more than this at any
#: reference point cannot be normalised, and is refused (V6).
REFERENCE_DIVERGENCE = 0.02
#: V6 (section 10): canary drift limits.
CANARY_SET, CANARY_ONE = 0.02, 0.04
BUCKET_S = 1.0


#: `--speed-unchecked`: a V6 failure (canary drift) does not refuse the report; every speed
#: figure is then labelled provisional and the failures are listed. BD does not depend on
#: timing (the streams are the same), so the quality front can be read while the timing is
#: re-run.
SPEED_UNCHECKED = False
V6_FAILURES: list[str] = []
#: V6: the boxes (canaries, missing references) and reference points (diverging references)
#: refused; their jobs are coded again. Reported.
REFUSED_BOXES: dict[str, str] = {}


class ReportRefusal(RuntimeError):
    """A check failed: no report is written."""


#: V31: an encode's CPU time over its wall time at most this (one thread on its one core;
#: x265's and Kvazaar's idle helper threads measured 1.02-1.04 in the 2026-11-25 audit), with
#: ONE_CORE_SLACK_S for process start-up on very short encodes.
ONE_CORE_MAX = 1.10
ONE_CORE_SLACK_S = 0.02


def check_structure(points: dict[str, dict[str, dict]]) -> dict:
    """V29-V31 on every point whose score record comes from BP-4's score.py (it names its
    metrics backend; BP-3 records carry no structure and are listed as unchecked):
      - V29 no picture coded ahead of its display (streamcheck, AV1/VP9: no hidden frames);
            a failed probe refuses too; H.264/HEVC/VVC reordering is not read (D19: no extra
            decode) and their points are listed per family;
      - V30 the window's first frame is a key; AV1/VP9 keys per stream are reported per family;
      - V31 the encode used one core (CPU over wall time at most ONE_CORE_MAX).
    AV2's frame headers are not parsed (streamcheck "count-only"): listed, not refused."""
    keys: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    unchecked, count_only, order_unread = 0, defaultdict(int), defaultdict(int)
    for series, by_id in points.items():
        for job_id, p in by_id.items():
            if "metrics_backend" not in p:
                unchecked += 1
                continue
            s = p.get("structure")
            if not s or s.get("probe") == "failed":
                raise ReportRefusal(f"V29: {series}: {job_id}: no structure read from the stream "
                                    f"({(s or {}).get('error', 'missing')})")
            if s["probe"] == "count-only":
                count_only[p["family"]] += 1
            else:
                if s["reordered"]:
                    raise ReportRefusal(f"V29: {series}: {job_id}: {s['hidden_frames']} pictures "
                                        f"coded ahead of their display ({s['probe']})")
                if not s["first_is_key"]:
                    raise ReportRefusal(f"V30: {series}: {job_id}: the first frame is no key")
                if s["reordered"] is None:
                    order_unread[p["family"]] += 1
                else:
                    keys[p["family"]][s["keys"]] += 1
            cpu, wall = p.get("cpu_s"), p.get("wall_s")
            if cpu is not None and wall and cpu > ONE_CORE_MAX * wall + ONE_CORE_SLACK_S:
                raise ReportRefusal(f"V31: {series}: {job_id}: {cpu:.2f} CPU-s in {wall:.2f} s "
                                    f"wall: more than one core")
    return {"keys_per_stream": {f: dict(sorted(c.items())) for f, c in sorted(keys.items())},
            "count_only": dict(count_only), "order_unread": dict(order_unread),
            "unchecked_points": unchecked}


def read_jsonl(path: Path) -> list[dict]:
    """A records file; a missing one is no records (V13 then names what is incomplete,
    rather than the report failing on a path)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def chained_check_sets(run: Path, series: str, records: list[dict]) -> list[dict]:
    """The check-set records of this run's machines that served it as a later run of their
    chain (`fleet.py run --chain`): such a machine runs its check sets once, at its start and
    final end, so they sit in the other runs of the chip that machine served (same machine
    name). Returned with their own box name, for V6; nothing else is taken from them."""
    # a machine's own start-set jobs (a chained run holds only its marker record)
    with_start = {r.get("box") for r in records if r.get("check") == "start" and "cpu_s" in r}
    lacking = {r.get("box") for r in records if "family_done" in r
               and isinstance(r.get("box"), str)} - with_start
    if not lacking:
        return []
    found = []
    for other in sorted(run.parent.glob("*/series/" + series)):
        if other.parent.parent == run:
            continue
        files = [other / "encodes.jsonl", *(other / box / "encodes.jsonl" for box in lacking)]
        for f in files:
            if not f.exists():
                continue
            for r in read_jsonl(f):
                box = r.get("box") if isinstance(r.get("box"), str) else None
                box = box or (f.parent.name if f.parent.name in lacking else None)
                if box in lacking and ("check" in r or "check_done" in r) \
                        and "chained_from" not in r:
                    found.append({**r, "box": box})
    return found


def load_series(run: Path, jobs: dict[str, dict],
                check_speed: bool = True) -> dict[str, dict[str, dict]]:
    """{series: {job id: point}} after V23 and V6; V13 (every planned point present) is
    checked by `report` over all the runs together, since a later run may code the points an
    earlier run's refused boxes lost."""
    out = {}
    for series_dir in sorted((run / "series").iterdir()):
        series = series_dir.name
        records = read_jsonl(series_dir / "encodes.jsonl")
        if not any(is_job(r) for r in records):  # an attempt that coded nothing (no series)
            continue
        # first: a machine's start set precedes the families it timed (valid_families)
        records = chained_check_sets(run, series, records) + records
        # a point needs a valid time from these records unless a re-timing gives it (then any
        # coded stream serves)
        if check_speed:
            encodes, refused = timed_records(series, records)
            REFUSED_BOXES.update({f"{series}/{refusal_name(b)}": why
                                  for b, why in refused.items()})
        else:
            encodes = [r for r in records if is_job(r)]
        scores: dict[str, dict] = {}
        for r in read_jsonl(series_dir / "scores.jsonl"):
            # a later record wins, but a refusal never displaces a valid score of the same id
            # (2026-11-26: a pass that raced a merge found the stream moved and recorded it
            # refused after another pass had scored it)
            if "refused" in r and r["id"] in scores and "refused" not in scores[r["id"]]:
                continue
            scores[r["id"]] = r
        coded = {r["id"]: r for r in encodes if r["id"] in jobs}
        if check_speed:
            # owner, 2026-11-26: a point with a valid stream and score but no valid time counts
            # for quality and BD; it is left out of its family's speed only, and listed
            for r in records:
                if is_job(r) and r["id"] in jobs and r["id"] not in coded and "stream_sha256" in r \
                        and not r.get("check") and not r.get("filler"):
                    why = next((x["timing_refused"] for x in records if x.get("id") == r["id"]
                                and "timing_refused" in x), None) or \
                        REFUSED_BOXES.get(f"{series}/{refusal_name(r.get('box', ''))}") or \
                        "no valid time"
                    coded[r["id"]] = {**r, "untimed": True, "untimed_reason": why}
            encodes = list(encodes) + [r for r in coded.values() if r.get("untimed")]
        hashes = {r["stream_sha256"] for r in encodes}
        points = {}
        for job_id, enc in coded.items():
            score = scores.get(job_id)
            if score is None or "refused" in score:
                continue
            if score["series"] != series or score["stream_sha256"] not in hashes:
                raise ReportRefusal(f"V23: {series}: {job_id} was scored from another stream")
            points[job_id] = {**jobs[job_id], **enc, **score}
            if enc.get("untimed"):
                UNTIMED_POINTS.append({"series": series, "id": job_id,
                                       "reason": str(enc["untimed_reason"])[:200]})
        out[series] = points
    return out


#: T8 (coordinator, set after the first c8a.2xlarge re-timing): the divergence check compares
#: a 3-point moving median of each reference, because one 0.5-1 s reference encode carries
#: about 1 % of timing noise (median single-point divergence 1.1 %), while the canaries pass.
REFERENCE_SMOOTHING = 3


def moving_median(values: list[float], width: int = REFERENCE_SMOOTHING) -> list[float]:
    """Each value replaced by the median of the `width` values centred on it (fewer at the
    ends)."""
    half = width // 2
    return [float(np.median(values[max(0, i - half):i + half + 1])) for i in range(len(values))]


def is_job(r: dict) -> bool:
    """A job's encode record (not a canary, reference, warm-up or marker)."""
    return "stream_sha256" in r and "canary" not in r and "reference" not in r


def bracketed(records: list[dict]) -> list[dict]:
    """BP-3.11 rule 6: the job records whose time is valid, those with a timing reference
    after them on the same box (one before is always there: a box codes its references
    before its first job). A box without references (an untimed packed run) keeps all."""
    last: dict[str, int] = {}
    for r in records:
        if "reference" in r and "cpu_s" in r:
            box = r.get("box", "")
            last[box] = max(last.get(box, -1), r["seq"])
    return [r for r in records if is_job(r) and
            (r.get("box", "") not in last or r.get("seq", 0) < last[r.get("box", "")])]


def reference_factors(records: list[dict]) -> tuple[dict, dict[str, str]]:
    """BP-3.6: per box, the box's relative speed at each reference point (seq, factor): the
    geometric mean over the two references of its time there over its median time on the
    box; and V6's refusals, keyed by box (references missing: the whole box) or by
    (box, seq) (BP-3.11: the reference points where the two references, each a 3-point moving
    median of its time over its box median, diverge by more than REFERENCE_DIVERGENCE; the
    jobs in the windows on either side of such a point are refused, not the whole box)."""
    refs: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in records:
        if "reference" in r and "cpu_s" in r:
            refs[r.get("box", "")][r["reference"]][r["seq"]] = r["cpu_s"]
    factors, refused = {}, {}
    for box, by_ref in refs.items():
        if len(by_ref) != 2:
            refused[box] = f"V6: {box} has {len(by_ref)} timing references, not 2"
            continue
        (a, ta), (b, tb) = sorted(by_ref.items())
        seqs = sorted(set(ta) & set(tb))
        ra = [ta[q] / float(np.median(list(ta.values()))) for q in seqs]
        rb = [tb[q] / float(np.median(list(tb.values()))) for q in seqs]
        for q, x, y in zip(seqs, moving_median(ra), moving_median(rb)):
            if abs(x / y - 1) > REFERENCE_DIVERGENCE:
                refused[box, q] = (f"V6: {box} at job {q}: its references {a} and {b} diverge "
                                   f"by {abs(x / y - 1):.3f}")
        factors[box] = (seqs, [math.sqrt(x * y) for x, y in zip(ra, rb)])
    return factors, refused


def speed_factors(records: list[dict]) -> dict[str, tuple[list[int], list[float]]]:
    """`reference_factors`, refusing on the first V6 failure (unless --speed-unchecked)."""
    factors, refused = reference_factors(records)
    for why in refused.values():
        if not SPEED_UNCHECKED:
            raise ReportRefusal(why)
        V6_FAILURES.append(why + " (its times stay raw)")
    return factors


def refusal_name(key: str | tuple[str, int]) -> str:
    """A refusal's key as text: a box, or a box and the reference point (`box@seq`)."""
    return key if isinstance(key, str) else f"{key[0]}@{key[1]}"


def refused_window(r: dict, refused: dict, factors: dict) -> bool:
    """Whether a job's time falls under a V6 refusal: its whole box, or a refused reference
    point at either end of the window it ran in."""
    box, seq = r.get("box", ""), r.get("seq", 0)
    if box in refused:
        return True
    points = factors.get(box, ([], []))[0]
    before = max((q for q in points if q <= seq), default=None)
    after = min((q for q in points if q > seq), default=None)
    return (box, before) in refused or (box, after) in refused


#: V6 under BP-4 (section 6, D13): a machine's start check set within this of the series level.
CHECK_SET_LIMIT = wholebox.CHECK_SET_LIMIT
#: Points with a valid stream and score but no valid time (owner, 2026-11-26): in quality and
#: BD, out of their family's speed, listed at the top of the report.
UNTIMED_POINTS: list[dict] = []
#: BP-4's end check sets: each surviving machine's end level over its start level, minus one
#: (reported, never a refusal: D13).
CHECK_SET_DRIFT: dict[str, float] = {}


def half_layout(records: list[dict]) -> bool:
    """A BP-4 series: its machines ran halfload.py (their header says so)."""
    return any(isinstance(r.get("box_info"), dict) and r["box_info"].get("layout") == "half"
               for r in records)


def valid_families(records: list[dict], levels: dict[tuple[str, str], float]
                   ) -> tuple[set[tuple[str, str]], dict[str, str], dict[str, float]]:
    """V6/V24 per machine (D13): its families count from the moment each finishes
    (`family_done`) after a complete start check set within CHECK_SET_LIMIT of the series.
    Returns ({(machine, family)}, {machine: why} for the machines whose start set fails or is
    missing, {machine: end set over start set - 1} for the machines that reached their end)."""
    done: dict[str, list[str]] = defaultdict(list)
    complete: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if "check_done" in r:
            complete[r.get("box", "")].add(r["check_done"])
        elif "family_done" in r and "start" in complete[r.get("box", "")]:
            done[r.get("box", "")].append(r["family_done"])
    machines = {r.get("box", "") for r in records if "family_done" in r or "check_done" in r}
    valid, why, drift = set(), {}, {}
    for box in machines:
        start = levels.get((box, "start")) if "start" in complete[box] else None
        if start is None or abs(start - 1) > CHECK_SET_LIMIT:
            why[box] = f"V6: {box}'s start check set is {start} of the series level"
            continue
        valid |= {(box, f) for f in done[box]}
        end = levels.get((box, "end")) if "end" in complete[box] else None
        if end is not None:
            drift[box] = end / start - 1
    return valid, why, drift


def half_records(series: str, records: list[dict]) -> tuple[list[dict], dict]:
    """BP-4's V6 and V24: a job's time counts only if its family finished on a machine whose
    start check set passed (`valid_families`) and the job was not refused on timing. Times are
    never rescaled (`cpu_s_normalised` = `cpu_s`). V7's clock check as for BP-3. The second
    value lists the machines with a failing or missing start set; each surviving machine's
    end-set drift goes to CHECK_SET_DRIFT (reported)."""
    _, refused, drift = valid_families(records, wholebox.check_set_levels(records))
    CHECK_SET_DRIFT.update({f"{series}/{box}": round(d, 5) for box, d in drift.items()})
    timing_refused = {(r.get("box", ""), r["id"]) for r in records if "timing_refused" in r}
    slow = check_machine(series, records)
    machines = {r.get("box", "") for r in records if "family_done" in r or "check_done" in r}
    passed = machines - set(refused)
    valid = []
    for r in records:
        if not is_job(r) or r.get("check"):
            continue
        key = (r.get("box", ""), r["id"])
        # D13 as amended 2026-11-26 (owner): machines of one type are equivalent, so a point's
        # time stands on its own (its machine's start check set passed, V7/V8 for the encode),
        # whichever machine coded the rest of its family
        if (SPEED_UNCHECKED or r.get("box", "") in passed) \
                and key not in timing_refused and key not in slow:
            valid.append({**r, "cpu_s_normalised": r["cpu_s"]})
    if SPEED_UNCHECKED:
        V6_FAILURES.extend(f"{why} (its times stay raw)" for why in refused.values())
    return valid, refused


def timed_records(series: str, records: list[dict]) -> tuple[list[dict], dict]:
    """The job records whose times are valid, normalised (BP-3.6), and the refused boxes:
    a box whose references diverge or whose canaries drift is refused (V6), and so is every
    time after a box's last references (V24); their jobs are coded again (`fleet.py
    missing`). V7's clock check and V6's one-CPU-model check refuse the whole series. With
    --speed-unchecked, refused boxes keep their raw times, listed as provisional. A BP-4
    series (whole machines at half load) is judged by `half_records`."""
    if half_layout(records):
        return half_records(series, records)
    factors, refused = reference_factors(records)
    canaries = {i: r for i, r in enumerate(records) if "canary" in r and "cpu_s" in r}
    if factors and not canaries:
        raise ReportRefusal(f"V6: {series} has no canary records")
    if canaries:  # a timed series (an untimed packed run has neither)
        normalise(canaries, factors)
        refused |= canary_failures(list(canaries.values()))
    slow = check_machine(series, records)
    valid = [r for r in bracketed(records)
             if SPEED_UNCHECKED or (not refused_window(r, refused, factors)
                                    and (r.get("box", ""), r["id"]) not in slow)]
    if SPEED_UNCHECKED:
        V6_FAILURES.extend(f"{why} (its times stay raw)" for why in refused.values())
    normalise({i: r for i, r in enumerate(valid)}, factors)
    return valid, refused


def normalise(points: dict[str, dict], factors: dict) -> None:
    """Each point's `cpu_s_normalised`: its CPU time over its box's speed factor, interpolated
    between the reference points around it (BP-3.6). Without references, the raw time."""
    for p in points.values():
        seqs, f = factors.get(p.get("box", ""), ([], []))
        p["cpu_s_normalised"] = p["cpu_s"] / float(np.interp(p.get("seq", 0), seqs, f)) \
            if seqs else p["cpu_s"]


#: V7: a point whose timing CPU ran below this share of its box's median clock is refused.
MIN_CLOCK_SHARE = 0.95
#: V7's clock check needs several clock samples (box.py samples every 50 ms): every encode it
#: refused on 2026-11-24 (69, x264 ultrafast) lasted 0.02-0.04 s, one sample taken while the
#: core's clock was still rising from idle, and the same jobs failed again on every re-run.
MIN_CLOCK_WALL_S = 0.25


def check_machine(series: str, records: list[dict]) -> set[tuple[str, str]]:
    """V6: every box of a series has one CPU model (else the series is refused). V7: the
    encodes whose median clock was below MIN_CLOCK_SHARE of their box's median (where the CPU
    reports a clock), as (box, id): those times are refused and the jobs coded again."""
    models = {r["box_info"]["cpu_model"] for r in records
              if isinstance(r.get("box_info"), dict) and "cpu_model" in r["box_info"]}
    if len(models) > 1:
        raise ReportRefusal(f"V6: {series} ran on more than one CPU model: {sorted(models)}")
    by_box: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("mhz_median") and r.get("wall_s", 0) >= MIN_CLOCK_WALL_S:
            by_box[r.get("box", "")].append(r)
    slow = set()
    for box, recs in by_box.items():
        median = float(np.median([r["mhz_median"] for r in recs]))
        slow |= {(box, r["id"]) for r in recs if r["mhz_median"] < MIN_CLOCK_SHARE * median}
    return slow


def canary_failures(records: list[dict]) -> dict[str, str]:
    """V6: the boxes whose canary set, at its middle or end, drifted from its own start."""
    runs: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        if "canary" in r and "cpu_s" in r:
            runs[r.get("box", "")][r["canary"]][r["id"].split("|", 1)[1]].append(
                r.get("cpu_s_normalised", r["cpu_s"]))
    failures = {}
    for box, by_where in sorted(runs.items()):
        times = {w: {k: float(np.median(v)) for k, v in ids.items()} for w, ids in by_where.items()}
        start = times["start"]
        for where in ("middle", "end"):
            ratios = {k: times[where][k] / start[k] for k in start if k in times.get(where, {})}
            if not ratios:  # a box stopped before this point: its references cover its drift
                continue
            overall = math.exp(np.mean([math.log(v) for v in ratios.values()]))
            worst = max(ratios.items(), key=lambda kv: abs(kv[1] - 1))
            if abs(overall - 1) > CANARY_SET or abs(worst[1] - 1) > CANARY_ONE:
                failures[box] = (f"V6: {box} canaries at {where}: set {overall:.3f}, "
                                 f"{worst[0]} {worst[1]:.3f}")
    return failures


def replay(frame_bytes: list[int], kbps: float, fps: float, init_s: float) -> dict:
    """Section 7's bucket: 1 s at the target, starting `init_s` full on the decoder's side."""
    size, drain = kbps * 1000 * BUCKET_S, kbps * 1000 / fps
    level, peak, over = (BUCKET_S - init_s) * kbps * 1000, 0.0, 0
    for b in frame_bytes:
        level += 8 * b
        peak = max(peak, level)
        over += level > size
        level = max(0.0, level - drain)
    seconds = len(frame_bytes) / fps
    return {"frames_over": over, "frames": len(frame_bytes), "peak_buffers": peak / size,
            "rate_over_target": 8 * sum(frame_bytes) / (kbps * 1000 * seconds)}


def replay_own_rate(frame_bytes: list[int], fps: float, init_s: float) -> dict:
    """The same bucket at the stream's own rate: drained at its achieved average rate and
    holding one second of it (owner, 2026-11-22: a point is plotted at its achieved rate, so
    an encoder that undershoots must not be credited with a larger buffer; the results site's
    buffer page, `nceval.cbr_stats.own_rate_fullness`)."""
    achieved_kbps = 8 * sum(frame_bytes) * fps / len(frame_bytes) / 1000
    return replay(frame_bytes, achieved_kbps, fps, init_s)


#: The bounds compliance is reported against: one buffer, and our default re-code ceiling
#: (docs/analysis/cbr-buffer-references-2026-11-22.md, the owner's decision).
BOUNDS = (1.0, 1.15)


#: The frame-rate bands compliance is also reported in (the 2026-11-25 fairness review: the
#: references' buffer use differs on low-rate static-camera clips).
FPS_BANDS = (("<=20 fps", 0.0, 20.0), (">20 fps", 20.0, 1e9))


def bucket_figures(peaks: list[float], frames_over: int, frames: int) -> dict:
    """Median, p95 and worst peak in buffers, the shares of points over each of BOUNDS, and
    the share of frames over one buffer."""
    peaks = sorted(peaks)
    return {"median_peak_buffers": float(np.median(peaks)),
            "p95_peak_buffers": peaks[min(len(peaks) - 1, int(0.95 * len(peaks)))],
            "worst_peak_buffers": peaks[-1],
            **{f"share_points_over_{b:g}": sum(p > b for p in peaks) / len(peaks)
               for b in BOUNDS},
            "share_frames_over_1": frames_over / frames}


def compliance_figures(points: list[dict], init_s: float) -> dict:
    """Both buckets and the worse of the two per point (section 8), the rate over target, and
    the first key's size in seconds of the stream's own rate."""
    rows = []
    for p in points:
        fps = fraction(p["fps"])
        own = replay_own_rate(p["frame_bytes"], fps, init_s)
        target = replay(p["frame_bytes"], p["kbps"], fps, init_s)
        achieved_bps = 8 * sum(p["frame_bytes"]) * fps / len(p["frame_bytes"])
        rows.append((own, target, 8 * p["frame_bytes"][0] / achieved_bps))
    frames = sum(t["frames"] for _, t, _ in rows)
    ratios = [t["rate_over_target"] for _, t, _ in rows]
    keys = sorted(k for _, _, k in rows)
    worse = [max(o["peak_buffers"], t["peak_buffers"]) for o, t, _ in rows]
    dropped = sum(p.get("dropped_frames", 0) for p in points)
    return {"points": len(rows), "frames": frames, "dropped_frames": dropped,
            "share_frames_dropped": dropped / frames,
            "own_rate": bucket_figures([o["peak_buffers"] for o, _, _ in rows],
                                       sum(o["frames_over"] for o, _, _ in rows), frames),
            "target_rate": bucket_figures([t["peak_buffers"] for _, t, _ in rows],
                                          sum(t["frames_over"] for _, t, _ in rows), frames),
            "worse_of_two": bucket_figures(
                worse, sum(max(o["frames_over"], t["frames_over"]) for o, t, _ in rows), frames),
            "median_rate_over_target": float(np.median(ratios)),
            "mean_rate_over_target": float(np.mean(ratios)),
            "first_key_seconds": {"median": float(np.median(keys)),
                                  "p95": keys[min(len(keys) - 1, int(0.95 * len(keys)))],
                                  "worst": keys[-1]}}


def compliance(points: list[dict], init_s: float) -> dict:
    """Section 8's compliance: overall, per class and per frame-rate band. Primary is the
    bucket at each stream's own rate, then the bucket at the target, then the worse of the two
    per point (BP-3.12)."""
    out = compliance_figures(points, init_s)
    classes = sorted({p.get("class", "?") for p in points})
    out["by_class"] = {c: compliance_figures([p for p in points if p.get("class", "?") == c],
                                             init_s) for c in classes}
    out["by_fps"] = {name: compliance_figures(band, init_s) for name, low, high in FPS_BANDS
                     if (band := [p for p in points if low < fraction(p["fps"]) <= high])}
    out["definition"] = ("own rate: the 1 s bucket drained at the stream's achieved average rate "
                         "(the results site's buffer page, web/results/buffer.html); target "
                         "rate: drained at the target; worse of two: the larger peak per point; "
                         "first key: its bits in seconds of the stream's own rate")
    return out


def fraction(fps: str) -> float:
    num, den = fps.split("/")
    return int(num) / int(den)


#: BP-4.0 windows (D7): a point coded at a long window also serves the short window, its first
#: frames. Its curve on a clip is then keyed `clip@frames` (WINDOW_SEP); `base_clip` undoes it.
WINDOW_SEP = "@"
#: The pooled metrics a window recomputes from the per-frame scores: libvmaf's pooled value is
#: the per-frame mean (score.py), so the mean of a window's first frames is its pooled value.
POOLED_KEYS = ("psnr_y", "psnr_u", "psnr_v", "vmaf_v1", "ms_ssim")


def base_clip(name: str) -> str:
    return name.split(WINDOW_SEP)[0]


def window_point(p: dict, frames: int) -> dict:
    """The point as its first `frames` frames: their bytes and the per-frame mean of each
    metric. Untimed (its time belongs to the whole encode) and out of compliance, which takes
    the whole encode."""
    per_frame = p.get("per_frame")
    if not per_frame or len(p["frame_bytes"]) < frames:
        raise ReportRefusal(f"{p['id']}: no per-frame scores for its {frames}-frame window")
    pooled = {k: float(np.mean(per_frame[k][:frames])) for k in POOLED_KEYS if per_frame.get(k)}
    return {**p, "frame_bytes": p["frame_bytes"][:frames], "frames": frames, "pooled": pooled,
            "untimed": True, "prefix": True}


def with_windows(points: dict[str, dict]) -> dict[str, dict]:
    """Each point once per window it serves (the job's `windows`), keyed `clip@frames`. A run
    in which no point serves more than its own length (BP-3) is returned as it is."""
    served = {pid: p.get("windows") or [p["frames"]] for pid, p in points.items()}
    if all(len(w) == 1 for w in served.values()):
        return points
    out = {}
    for pid, p in points.items():
        for frames in served[pid]:
            q = p if frames == p["frames"] else window_point(p, frames)
            out[f"{pid}{WINDOW_SEP}{frames}"] = {**q, "window": frames,
                                                "input": f"{p['input']}{WINDOW_SEP}{frames}"}
    return out


def once_per_clip(logs: dict[str, dict[str, float]], clips: list[str]) -> dict[str, list[float]]:
    """{metric: per-clip log ratios} with each clip once: its windows' logs averaged (D7)."""
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for c in clips:
        grouped[base_clip(c)].append(logs[c])
    return {m: [float(np.mean([w[m] for w in ws])) for ws in grouped.values()]
            for m in (*METRICS, "j")}


def curves(points: dict[str, dict]) -> tuple[dict, dict]:
    """({family: {clip: points}}, {zero family: {clip: points}}), sorted by rate. The first
    holds the ladder's points only; the second, for the anchor's families, also its extra
    points past each end (BP-3.4), used only as the zero curve of a BD."""
    test: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    zero: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in points.values():
        p["kbps_coded"] = 8 * sum(p["frame_bytes"]) / (len(p["frame_bytes"]) / fraction(p["fps"])) / 1000
        if p["family"] in ZERO_FAMILY.values():
            zero[p["family"]][p["input"]].append(p)
        if not p.get("anchor_extra"):
            test[p["family"]][p["input"]].append(p)
    for table in (test, zero):
        for family in table.values():
            for clip in family:
                family[clip].sort(key=lambda p: p["kbps_coded"])
    return test, zero


class CurveRefusal(ValueError):
    """No BD for a family on a clip (a curve breaks section 9's rules, or its quality range
    does not touch the zero line's): the family loses the clip, with this reason."""


#: BP-3.9 (owner, after seeing the first results): the monotone envelope. Before section 9's
#: rules, a point is dropped when another point of the same encoder and clip has a lower or
#: equal achieved rate and a higher or equal quality on the metric (more bits for less
#: quality), identically for every encoder, ours and the anchor included. `--no-envelope`
#: gives the BP-3.7 view. Dropped points are counted per family and metric.
ENVELOPE = True
DROPPED: set[tuple[str, str, str]] = set()  # (family, metric, point id)


def bdj_curve(curve: list[dict]) -> list[tuple[float, dict[str, float | None]]]:
    """A stored curve as nceval.bdj takes it: coded rate and the pooled metrics."""
    return [(p["kbps_coded"], {m: RAW[m](p["pooled"]) for m in METRICS}) for p in curve]


def metricless(points: dict[str, dict]) -> dict[str, dict[str, int]]:
    """{family: {metric: points}} of the points whose stored score lacks the metric. Such a
    point is in no curve of that metric (BP-3.14); the count is printed so it is seen."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in points.values():
        for metric in METRICS:
            if RAW[metric](p["pooled"]) is None:
                counts[p["family"]][metric] += 1
    return {f: dict(m) for f, m in sorted(counts.items())}


def envelope(curve: list[dict], metric: str) -> list[dict]:
    """The points of `curve` that no other point dominates on `metric` (BP-3.9,
    nceval.bdj.envelope), each dropped one counted."""
    kept = set(bdj.envelope(bdj_curve(curve), metric))
    for i, p in enumerate(curve):
        if i not in kept:
            DROPPED.add((p["family"], metric, p["id"]))
    return [p for i, p in enumerate(curve) if i in kept]


def clip_bd(test: list[dict], zero: list[dict], metric: str) -> bdj.BdResult:
    """BD-rate of `test` against `zero` with its overlap fraction (nceval.bdj.bd). Raises
    CurveRefusal, naming the reason, when there is none: a curve that breaks section 9's rules
    or ranges that do not touch."""
    if ENVELOPE:  # the dropped points are counted for the report
        envelope(test, metric), envelope(zero, metric)
    label = test[0]["family"] if test else "curve"
    try:
        return bdj.bd(bdj_curve(test), bdj_curve(zero), metric, ENVELOPE)
    except bdj.NoBd as refusal:
        raise CurveRefusal(f"{label}: {refusal}") from refusal


def clip_logs(fam: dict[str, dict[str, list[dict]]], zero: dict[str, dict[str, list[dict]]],
              family: str, clip: str) -> tuple[dict[str, float], float]:
    """The family's per-clip log rate ratios on `clip`: each metric against its matched zero,
    and BD-J (the mean of the three BDs against the default tune); and the smallest overlap
    fraction among those BDs (the clip is as thin as its thinnest). Raises CurveRefusal when
    the family has no BD there, with every metric's reason (the lost-clip list is exact)."""
    curve = fam[family].get(clip, [])
    if len(curve) < MIN_POINTS:
        raise CurveRefusal(f"{family}: {len(curve)} points coded")
    bds: dict[tuple[str, str], bdj.BdResult] = {}
    reasons: dict[str, list[str]] = defaultdict(list)  # {why: the zero lines it holds against}
    for metric in METRICS:
        for zero_family in dict.fromkeys((ZERO_FAMILY[metric], ZERO_FAMILY["j"])):
            try:
                bds[metric, zero_family] = clip_bd(curve, zero[zero_family][clip], metric)
            except CurveRefusal as why:
                reasons[str(why)].append(zero_family)
    if reasons:
        raise CurveRefusal("; ".join(f"{why} (against {', '.join(zeros)})"
                                     for why, zeros in reasons.items()))
    logs = {m: math.log1p(bds[m, ZERO_FAMILY[m]].bd_rate_percent / 100) for m in METRICS}
    logs["j"] = math.log1p(float(np.mean([bds[m, ZERO_FAMILY["j"]].bd_rate_percent
                                          for m in METRICS])) / 100)
    return logs, min(bd.overlap_fraction for bd in bds.values())


@dataclass(frozen=True)
class ClipTable:
    """V14's clip sets for one series."""
    #: the clips where the zero line's curves are well formed
    view: list[str]
    #: {clip: why} for the clips a malformed zero-line curve keeps out of the view
    out_of_view: dict[str, str]
    #: {family: {clip: log rate ratios}} where the family has a BD
    logs: dict[str, dict[str, dict]]
    #: {family: {clip: why}} where it has none
    missing: dict[str, dict[str, str]]
    #: {family: {clip: the smallest overlap fraction of its BDs}}
    overlaps: dict[str, dict[str, float]]


def clip_table(fam: dict[str, dict[str, list[dict]]],
               zero: dict[str, dict[str, list[dict]]]) -> ClipTable:
    """V14 (BP-3.2). A clip leaves the view for every family only when a zero line's curve is
    malformed there; any other family without a BD on a clip loses that clip only, counted
    with its reason, and the clip stays for the others."""
    table = ClipTable([], {}, defaultdict(dict), defaultdict(dict), defaultdict(dict))
    for clip in sorted({c for f in fam.values() for c in f}):
        try:
            for family in sorted(set(ZERO_FAMILY.values())):
                for metric in METRICS:
                    bdj.rd_curve(bdj_curve(zero[family].get(clip, [])), metric, ENVELOPE, family)
            table.view.append(clip)
        except bdj.CurveRefusal as refusal:
            table.out_of_view[clip] = str(refusal)
    for family in fam:
        for clip in table.view:
            try:
                table.logs[family][clip], table.overlaps[family][clip] = clip_logs(
                    fam, zero, family, clip)
            except CurveRefusal as why:
                table.missing[family][clip] = str(why)
    return table


def overlap_figures(overlaps: dict[str, float], clips: list[str]) -> dict:
    """How much of the smaller curve's quality span a family's BDs were integrated over, on
    `clips`: shown beside its figure in place of the old exclusion (BP-3.14)."""
    values = [overlaps[c] for c in clips]
    return {"median": float(np.median(values)), "min": float(min(values)),
            "thin_clips": sum(v < THIN_OVERLAP for v in values), "thin_under": THIN_OVERLAP}


def common_clips(view: list[str], logs: dict, families: list[str]) -> list[str]:
    """The sensitivity set: clips where every non-drop family is well formed."""
    return [c for c in view if all(c in logs[f] for f in families if not f.endswith(DROP_SUFFIX))]


def exclusion_check(fam, family: str, missing: dict, logs: dict, init_s: float) -> dict:
    """Whom a family's missing clips favour (BP-3.2): its own rate over target on its missing
    and kept clips (overshoot marks the hard clips), and our encoders' BD-J there and on all
    their clips. Ours doing better on those clips means leaving them out favours the family."""
    gone, kept = sorted(missing), sorted(logs[family])
    # only clips the family coded: a history family was never coded on the clips outside the
    # history subset that the zero line's run covers, so those "missing" clips have no points
    coded = lambda clips: [c for c in clips if fam[family].get(c)]  # noqa: E731
    overshoot = lambda clips: float(np.mean([  # noqa: E731
        compliance_figures(fam[family][c], init_s)["mean_rate_over_target"]
        for c in coded(clips)])) if coded(clips) else None
    ours = [f for f in logs if f.startswith(OURS_PREFIX)]
    ours_j = lambda clips: [logs[f][c]["j"] for f in ours for c in clips  # noqa: E731
                            if c in logs[f]]
    # None where none of ours has a BD on those clips either (nothing to average)
    on = lambda clips: float(np.mean(ours_j(clips))) if ours_j(clips) else None  # noqa: E731
    pct = lambda v: None if v is None else 100 * math.expm1(v)  # noqa: E731
    return {"missing_clips": len(gone), "rate_over_target_missing": overshoot(gone),
            "rate_over_target_kept": overshoot(kept), "ours_bd_j_on_missing": pct(on(gone)),
            "ours_bd_j_on_kept": pct(on(kept))}


def mean_interval(logs: list[float], rng: np.random.Generator) -> dict:
    """Set mean of per-clip log ratios as a percentage, with a bootstrap over clips."""
    x = np.asarray(logs)
    draws = x[rng.integers(0, len(x), size=(BOOTSTRAP_RESAMPLES, len(x)))].mean(axis=1)
    lo, hi = np.quantile(draws, [(1 - INTERVAL) / 2, (1 + INTERVAL) / 2])
    pct = lambda v: 100 * math.expm1(v)  # noqa: E731
    return {"mean": pct(x.mean()), "low": pct(lo), "high": pct(hi)}


#: BP-3.8 (A, coordinator): families timed on a subset of clips only; their other points are
#: coded untimed on packed boxes for quality, and their speed covers the timed clips only.
SUBSET_TIMED = ("vvenc-",)


def speed(fam: dict[str, list[dict]], clips: list[str],
          key: str = "cpu_s_normalised") -> float | None:
    """Geometric mean over clips of megapixels per CPU-second over the curve's points (`key`:
    the BP-3.6 normalised CPU time, or the raw `cpu_s`)."""
    rates = []
    for clip in clips:
        curve = [p for p in fam[clip] if not p.get("untimed")]
        if not curve:
            continue
        pixels = sum(p["w"] * p["h"] * p["frames"] for p in curve)
        rates.append(math.log(pixels / 1e6 / sum(p.get(key, p["cpu_s"]) for p in curve)))
    return math.exp(float(np.mean(rates))) if rates else None


def speed_text(row: dict) -> str:
    v = row.get("speed_mpix_per_cpu_s")
    if v is None:
        return "   untimed"
    subset = f" (timed on {row['speed_clips']} clips)" if row.get("speed_clips", row["clips"]) \
        < row["clips"] else ""
    return f"{v:7.2f} Mpx/CPU-s{subset}"


def front(rows: dict) -> list[str]:
    """The headline speed / BD-J front on the common set: families no other family beats on
    both (faster and lower BD-J). `+drop` rows, on their own clips, are not in it."""
    points = {f: (r["speed_mpix_per_cpu_s"], r["bd"]["j"]["mean"]) for f, r in rows.items()
              if r.get("clips") and r.get("bd") and r["headline_set"] == "common set"
              and r.get("speed_mpix_per_cpu_s") is not None}
    return sorted((f for f, (v, j) in points.items()
                   if not any(v2 >= v and j2 <= j and (v2, j2) != (v, j)
                              for v2, j2 in points.values())),
                  key=lambda f: -points[f][0])


def timing(runs: list[Path]) -> dict[str, dict[str, dict]]:
    """BP-3.7: {series: {id: encode record}} from re-timing runs (no streams kept): their own
    V6 and BP-3.6 normalisation applied."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for run in runs:
        for series_dir in sorted((run / "series").iterdir()):
            records = read_jsonl(series_dir / "encodes.jsonl")
            valid, refused = timed_records(series_dir.name, records)
            REFUSED_BOXES.update({f"{run.parent.name}/{series_dir.name}/{refusal_name(b)}": why
                                  for b, why in refused.items()})
            out[series_dir.name].update((r["id"], r) for r in valid)
    return out


def apply_timing(points: dict[str, dict], timed: dict[str, dict], series: str) -> None:
    """Every point's time from the re-timing run, whose stream must be byte-identical to the
    scored one (V11: the same build codes the same stream)."""
    untimed, different = [], []
    for job_id, p in points.items():
        if p.get("anchor_extra"):  # the anchor's extra points are not timed (BP-3.4)
            continue
        t = timed.get(job_id)
        if t is None and p["family"].startswith(SUBSET_TIMED):
            p["untimed"] = True  # BP-3.8 (A): its speed comes from the timed subset only
            continue
        if t is None:
            untimed.append(job_id)
            continue
        if t["stream_sha256"] != p["stream_sha256"]:
            different.append(job_id)
            continue
        p.update(cpu_s=t["cpu_s"], cpu_s_normalised=t["cpu_s_normalised"], box=t.get("box"),
                 retimed=True)
    if different:
        raise ReportRefusal(f"V11: {series}: {len(different)} points re-timed to a different "
                            f"stream, e.g. {different[0]}")
    if untimed:
        raise ReportRefusal(f"V13: {series}: {len(untimed)} points have no valid time (refused "
                            f"boxes: {len(REFUSED_BOXES)}), e.g. {untimed[0]}; "
                            "`fleet.py missing` lists them")


#: Failures of the machine, not the encoder (2026-11-26: a box's disk filled at 16:25 and 86
#: encodes failed with ENOSPC before being coded again elsewhere): never listed under V26.
INFRASTRUCTURE_ERRORS = ("No space left on device", "os error 28", "ENOSPC")


def infrastructure_failure(message: str) -> bool:
    return any(e in message for e in INFRASTRUCTURE_ERRORS)


def encoder_failures(runs: list[Path], keep=lambda job_id: True) -> dict[str, dict]:
    """V26: points whose stream fails a standard decoder (a V9 score refusal) and is
    deterministic (coded more than once, every time to the same SHA-256) are failures of that
    encoder: listed at the top of the report and counted against it (the point is absent, so
    its clip loses the family's curve where the minimum of points is missed). Never dropped
    silently, and never waived for one encoder without the same record for another (owner,
    2026-11-24, the first case: ours, av1@slow, ControlledBurn qp160). An encode that fails
    with the same error on every box it was given (the queue quarantines it after
    QUARANTINE_AFTER boxes) is the same kind of failure, without a stream (2026-11-24: the
    builds of 11-06 and 11-07 refuse 960x540 in HEVC and AV1)."""
    refusals: dict[str, str] = {}
    hashes: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for run in runs:
        for series_dir in (run / "series").iterdir():
            for r in read_jsonl(series_dir / "scores.jsonl"):
                if str(r.get("refused", "")).startswith("V9") \
                        and not infrastructure_failure(str(r["refused"])):
                    refusals[r["id"]] = r["refused"]
            for r in read_jsonl(series_dir / "encodes.jsonl"):
                if is_job(r) and r["id"] in refusals:
                    hashes[r["id"]].append((r["stream_sha256"], r.get("bytes", 0)))
    out = {}
    for run in runs:  # an encode that fails the same way on every box (the queue's quarantine)
        for listing in (run / "series").glob("*/quarantined.json"):
            for job_id, why in json.loads(listing.read_text()).items():
                reasons = {r.split(": ", 1)[-1].strip() for r in why.split("; ")}
                if keep(job_id) and len(reasons) == 1 and why.count("; ") >= 1:
                    out[job_id] = {"family": job_id.split("|")[0], "refusal": reasons.pop(),
                                   "stream_sha256": None, "bytes": None,
                                   "codings": why.count("; ") + 1}
    for job_id, why in refusals.items():
        if not keep(job_id):
            continue
        coded = hashes.get(job_id, [])
        if len(coded) >= 2 and len({h for h, _ in coded}) == 1:
            out[job_id] = {"family": job_id.split("|")[0], "refusal": why,
                           "stream_sha256": coded[0][0], "bytes": coded[0][1],
                           "codings": len(coded)}
    return out


@dataclass(frozen=True)
class Part:
    """Runs whose points a report takes, with their re-timing runs. A mixed report (the
    re-measure of our families on a new build beside reused competitor points) takes our
    families from one part and every other family from another; each part is checked on its
    own (V6, V11, V23), then V25 per family over all of them."""
    runs: tuple[Path, ...]
    timing_runs: tuple[Path, ...] = ()
    #: "ours" (nanocodec-* families), "others" (every other family) or None (all)
    take: str | None = None
    #: V18 for reused points: the plans, made at the report's tools commit, whose commands the
    #: part's jobs must equal byte for byte (the record did not change under them)
    commands: tuple[Path, ...] = ()
    #: only these clips (empty: all): a view's subset, as the history clips of the "ours" set
    inputs: tuple[str, ...] = ()
    #: only the families matching these patterns (empty: all), as ours and the zero line for
    #: a history point, whose common clip set must be the history days' own
    families: tuple[str, ...] = ()

    def keeps(self, family: str) -> bool:
        if self.families and not any(fnmatch.fnmatchcase(family, f) for f in self.families):
            return False
        if self.take is None:
            return True
        return family.startswith(OURS_PREFIX) == (self.take == "ours")

    def keeps_id(self, job_id: str) -> bool:
        """Whether the part takes this point (an id is family|clip|point)."""
        family, clip = job_id.split("|")[:2]
        return self.keeps(family) and (not self.inputs or clip in self.inputs)


def check_commands(part: Part, jobs: dict[str, dict]) -> None:
    """V18 on reused points: each job's argv is the current plan's for the same id."""
    if not part.commands:
        return
    current = {j["id"]: j["argv"] for path in part.commands
               for j in json.loads(path.read_text())}
    changed = [i for i, j in jobs.items() if current.get(i) != j["argv"]]
    if changed:
        raise ReportRefusal(f"V18: {len(changed)} reused points' commands are not the current "
                            f"plan's, e.g. {changed[0]}")


def part_jobs(part: Part, run: Path, mode: str | None) -> dict[str, dict]:
    """The jobs of one of the part's runs that the part takes."""
    return {j["id"]: j for j in json.loads((run / "jobs.json").read_text())
            if (mode is None or j.get("mode") == mode)  # a run of both modes (history)
            and part.keeps_id(j["id"])}


def part_points(part: Part, timing_series: str | None, mode: str | None,
                replaced: frozenset[str] = frozenset()
                ) -> tuple[dict[str, dict], set[str], str | None]:
    """({series: {id: point}}, the planned ids, the quality series) of one part, without the
    points a later part codes again (`replaced`)."""
    merged: dict[str, dict] = defaultdict(dict)
    planned: set[str] = set()
    taken = [{i: j for i, j in part_jobs(part, run, mode).items() if i not in replaced}
             for run in part.runs]
    loaded = [load_series(run, jobs, not part.timing_runs) for run, jobs in zip(part.runs, taken)]
    for index, (jobs, series_points) in enumerate(zip(taken, loaded)):
        # V18 judges the points the report uses: a point a later run of the part holds again
        # (a fill run's, after the record's command changed) is replaced below, not reused.
        # A later run that only planned the point, without coding it, replaces nothing, so
        # the earlier point is still judged.
        used = {i for series, points in series_points.items() for i in points
                if not any(i in later.get(series, ()) for later in loaded[index + 1:])}
        check_commands(part, {i: jobs[i] for i in used})
        planned |= set(jobs)
        for series, points in series_points.items():
            merged[series].update(points)
    if not part.timing_runs:
        return merged, planned, None
    timed = timing(list(part.timing_runs))
    # BP-3.10: the scored streams may come from one other machine series (c8a), timed on
    # the timing series only because V11 proves the re-timing reproduced every one of them
    # byte for byte. A series never takes another series' times: the timing runs must hold
    # exactly the named timing series, and the quality runs exactly one series.
    if set(timed) != {timing_series}:
        raise ReportRefusal(f"the timing runs hold series {sorted(timed)}, not "
                            f"{timing_series!r}")
    if len(merged) != 1:
        raise ReportRefusal(f"a re-timing applies to one quality series, not {sorted(merged)}")
    (quality_series, points), = merged.items()
    apply_timing(points, timed[timing_series], timing_series)
    return {timing_series: points}, planned, quality_series


def report(*runs: Path, timing_runs: tuple[Path, ...] = (), timing_series: str | None = None,
           mode: str | None = None, checks_only: bool = False,
           parts: tuple[Part, ...] = ()) -> dict:
    """The report over one or more runs of the same dataset and mode (a later run may add
    points, such as the anchor's extra points, BP-3.4); each run is checked on its own. `parts`
    instead of `runs`: a mixed report, each family taken from the part that holds it."""
    parts = parts or (Part(tuple(runs), tuple(timing_runs)),)
    every_run = list(dict.fromkeys(r for p in parts for r in p.runs))
    manifests = [json.loads((run / "manifest.json").read_text()) for run in every_run]
    try:  # V25: one recorded build identity for every binary, quality and timing runs alike
        if len(parts) == 1 and parts[0].take is None and not parts[0].families:
            block = provenance.provenance(list(parts[0].runs), list(parts[0].timing_runs))
        else:
            block = provenance.merged([
                (*provenance.identity(list(p.runs), list(p.timing_runs), keep=p.keeps),
                 p.take != "others") for p in parts])
    except provenance.ProvenanceRefusal as why:
        raise ReportRefusal(str(why)) from None
    block["procedure"] = provenance.procedure(block)
    # the phase is its name and initial fullness (the record's other phase values change with
    # procedure versions; a run's own commands hold them)
    if len({(m.get("phase_name"), m["phase"]["INIT_MS"]) for m in manifests}) != 1:
        raise ReportRefusal("the runs were made in different phases")
    init_s = manifests[0]["phase"]["INIT_MS"] / 1000
    merged: dict[str, dict] = defaultdict(dict)
    planned: set[str] = set()
    quality = set()
    # a later part's point replaces an earlier part's with the same id (a reused point coded
    # again, as the "ours" Fixed QP VVenC points whose command the record changed on 11-23)
    later: list[frozenset[str]] = []
    for part in reversed(parts):
        mine = frozenset(i for run in part.runs for i in part_jobs(part, run, mode))
        later.insert(0, (later[0] | mine) if later else mine)
    later = later[1:] + [frozenset()]
    for part, replaced in zip(parts, later):
        points_of, planned_of, quality_series = part_points(part, timing_series, mode, replaced)
        planned |= planned_of
        quality.add(quality_series)
        for series, points in points_of.items():
            merged[series].update(points)
    quality_series = quality.pop() if len(quality) == 1 else sorted(map(str, quality))
    # a failure is the result only where no run coded the point (a later fill may have, as the
    # quarantined jobs of a lost box can be coded on a later run)
    coded = {i for points in merged.values() for i in points}
    failures = {i: f for p, replaced in zip(parts, later)
                for i, f in encoder_failures(list(p.runs), p.keeps_id).items()
                if i not in replaced and i not in coded}
    for series, points in merged.items():
        missing = sorted(planned - set(points) - set(failures))
        if missing:
            raise ReportRefusal(f"V13: {series}: {len(missing)} points incomplete, e.g. {missing[0]}")
    structure = check_structure(merged)
    out = {"provenance": block, "encoder_failures": failures, "manifests": manifests,
           "untimed_points": list(UNTIMED_POINTS),
           "series": {}, "stream_structure": structure,
           "speed_provisional": bool(V6_FAILURES),
           "envelope": ENVELOPE, "envelope_dropped": DROPPED,
           "v6_failures": V6_FAILURES, "refused_boxes": REFUSED_BOXES,
           "check_set_drift": CHECK_SET_DRIFT,
           "quality_series": quality_series,
           "timing_series": timing_series}
    if checks_only:  # a rehearsal of a run without its zero line: every check, no BD
        out["checked_points"] = {series: len(points) for series, points in merged.items()}
        return out
    for series, points in merged.items():
        points = with_windows(points)
        fam, zero = curves(points)
        table = clip_table(fam, zero)
        view, logs, missing, overlaps = table.view, table.logs, table.missing, table.overlaps
        if not view:
            raise ReportRefusal(f"V14: {series}: the zero line has no well-formed curve")
        common = common_clips(view, logs, sorted(fam))
        if not common:
            raise ReportRefusal(f"V14: {series}: the common set is empty (a finding: report it)")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        rows = {}
        for family in sorted(fam):
            own = sorted(logs[family])
            if not own:
                rows[family] = {"clips": 0, "missing": missing[family]}
                continue
            mode = fam[family][own[0]][0]["mode"]
            drop = family.endswith(DROP_SUFFIX)
            # BP-3.3: the headline averages every family over the SAME clips, the common set;
            # a +drop row, which does not define that set, is shown on its own clips, labelled
            head = own if drop else common
            # D7: each clip counts once in the headline (its windows averaged); each window
            # length is also its own view
            head_logs, own_logs = once_per_clip(logs[family], head), once_per_clip(logs[family], own)
            windows = sorted({int(c.split(WINDOW_SEP)[1]) for c in head if WINDOW_SEP in c})
            rows[family] = {
                "clips": len({base_clip(c) for c in head}), "own_clips": len({base_clip(c) for c in own}),
                "missing": missing[family],
                "headline_set": "own clips (+drop row)" if drop else "common set",
                "bd": {m: mean_interval(head_logs[m], rng) for m in (*METRICS, "j")} if head else None,
                "bd_own_set": {m: mean_interval(own_logs[m], rng) for m in (*METRICS, "j")},
                "bd_by_window": {n: {m: mean_interval([logs[family][c][m] for c in head
                                                       if c.endswith(f"{WINDOW_SEP}{n}")], rng)
                                     for m in (*METRICS, "j")} for n in windows} or None,
                "overlap": overlap_figures(overlaps[family], head) if head else None,
                "overlap_own_set": overlap_figures(overlaps[family], own),
                "speed_mpix_per_cpu_s": speed(fam[family], head) if head else None,
                "speed_clips": sum(any(not p.get("untimed") for p in fam[family][c]) for c in head),
                "speed_raw_mpix_per_cpu_s": speed(fam[family], head, "cpu_s") if head else None,
                "compliance": compliance([p for c in own for p in fam[family][c]
                                          if not p.get("prefix")], init_s)
                if mode == "cbr" else None,
                "exclusion_check": exclusion_check(fam, family, missing[family], logs, init_s)
                if missing[family] and mode == "cbr" else None}
        out["series"][series] = {"view_clips": len(view), "common_clips": len(common),
                                 "curve_rules": {"min_points": MIN_POINTS,
                                                 "thin_overlap_under": THIN_OVERLAP},
                                 "points_without_a_metric": metricless(points),
                                 "out_of_view": table.out_of_view,
                                 "families": rows, "zero_line": ZERO_FAMILY,
                                 "front": front(rows)}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for family, metric, _ in DROPPED:
        counts[family][metric] += 1
    out["envelope_dropped"] = {f: dict(m) for f, m in sorted(counts.items())}
    return out


def lost_clips(s: dict) -> None:
    """Prints each clip the common set lacks, with the families that have no BD there and
    why (V14: nothing leaves the headline unlisted)."""
    lost: dict[str, list[str]] = defaultdict(list)
    for family, row in s["families"].items():
        if not family.endswith(DROP_SUFFIX):
            for clip, why in row["missing"].items():
                lost[clip].append(why)
    for clip, why in sorted(s["out_of_view"].items()):
        print(f"  OUT OF THE VIEW (the zero line has no curve): {clip}: {why}")
    for clip, reasons in sorted(lost.items()):
        print(f"  LOST from the common set: {clip}: " + "; ".join(reasons))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path, nargs="*")
    ap.add_argument("--parts", type=Path,
                    help="a mixed report: a JSON list of {runs, timing, take: ours|others, "
                         "commands, inputs, families}, each family taken from its part (Part)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--timing", type=Path, nargs="*", default=[],
                    help="re-timing runs whose times replace the runs' own (V11 identity)")
    ap.add_argument("--timing-series", help="the machine series of the --timing runs (e.g. c8i)")
    ap.add_argument("--mode", choices=("cbr", "qp"),
                    help="report only this mode's jobs (a run holding both, as the history's)")
    ap.add_argument("--no-envelope", action="store_true",
                    help="the BP-3.7 view, without the BP-3.9 monotone envelope")
    ap.add_argument("--speed-unchecked", action="store_true",
                    help="V6 failures label speed provisional instead of refusing")
    a = ap.parse_args()
    global SPEED_UNCHECKED, ENVELOPE
    SPEED_UNCHECKED = a.speed_unchecked
    ENVELOPE = not a.no_envelope
    try:
        if a.timing and not a.timing_series:
            raise ReportRefusal("--timing needs --timing-series (the timing machine)")
        if bool(a.runs) == bool(a.parts):
            raise ReportRefusal("give either runs or --parts")
        if a.parts and not a.out:
            raise ReportRefusal("--parts needs --out")
        parts = tuple(Part(tuple(map(Path, p["runs"])), tuple(map(Path, p.get("timing", []))),
                           p.get("take"), tuple(map(Path, p.get("commands", []))),
                           tuple(p.get("inputs", [])), tuple(p.get("families", [])))
                      for p in json.loads(a.parts.read_text())) if a.parts else ()
        result = report(*a.runs, timing_runs=tuple(a.timing), timing_series=a.timing_series,
                        mode=a.mode, parts=parts)
    except ReportRefusal as why:
        print(f"refused: {why}", file=sys.stderr)
        return 1
    (a.out or a.runs[0] / "report.json").write_text(json.dumps(result, indent=1) + "\n")
    untimed = result.get("untimed_points", [])
    if untimed:  # owner, 2026-11-26: listed first; in quality and BD, out of speed
        print(f"UNTIMED POINTS ({len(untimed)}: valid stream and score, no valid time; counted "
              f"for quality and BD, left out of their family's speed):")
        for u in untimed:
            print(f"  {u['series']} {u['id']}: {u['reason']}")
    print("\n".join(provenance.text(result["provenance"])))
    st = result.get("stream_structure", {})
    if st:  # V29-V31: read from the streams themselves
        print(f"stream structure (V29-V31): no reordering (AV1, VP9), first frame a key, one "
              f"core; {st['unchecked_points']} points without a structure record (BP-3 scorer)"
              + (f"; H.26x order not read (D19): {sum(st['order_unread'].values())} points"
                 if st.get("order_unread") else "")
              + (f"; AV2 counted only: {sum(st['count_only'].values())} points" if st["count_only"]
                 else ""))
        for family, counts in st["keys_per_stream"].items():
            if set(counts) != {1}:
                print(f"  keys per stream, {family}: {counts}")
    for job_id, f in result["encoder_failures"].items():
        how = (f"the same stream every time ({f['codings']} codings, sha256 "
               f"{f['stream_sha256'][:16]}, {f['bytes']} bytes)" if f["stream_sha256"] else
               f"the same encode error on each of {f['codings']} boxes")
        print(f"ENCODER FAILURE (V26, counted against {f['family']}): {job_id}: {f['refusal']}; "
              f"{how}")
    if result["speed_provisional"]:
        print(f"SPEED PROVISIONAL: V6 failed ({len(result['v6_failures'])} canary checks); "
              "BD is unaffected")
    for series, s in result["series"].items():
        print(f"{series}: headline on the common set of {s['common_clips']} clips "
              f"(of {s['view_clips']} in the view); * = front. Secondary: each family's own set")
        print(f"  curve rules ({result['provenance']['procedure']['report']}): at least "
              f"{s['curve_rules']['min_points']} points after the envelope; BD over whatever "
              f"quality range the curves share; overlap = median fraction of the smaller span "
              f"on the own set, thin = clips under {s['curve_rules']['thin_overlap_under']}")
        for family, metrics in s["points_without_a_metric"].items():
            print(f"  POINTS WITHOUT A METRIC (in no curve of that metric): {family}: "
                  + ", ".join(f"{n} without {m}" for m, n in sorted(metrics.items())))
        for family, row in s["families"].items():
            if not row["clips"]:
                print(f"  {family:32s} no well-formed curve")
                continue
            j, o = row["bd"]["j"], row["bd_own_set"]["j"]
            mark = "*" if family in s["front"] else " "
            if family.startswith(OURS_PREFIX):
                o = row["bd_own_set"]["j"]
                print(f"  {family:32s} ALL anchor-valid clips: BD-J {o['mean']:+7.2f} % "
                      f"[{o['low']:+.2f}, {o['high']:+.2f}] on {row['own_clips']} "
                      f"(the figure for claims; common set below)")
            print(f"{mark} {family:32s} BD-J {j['mean']:+7.2f} % [{j['low']:+.2f}, {j['high']:+.2f}]"
                  f" on {row['clips']:3d}  {speed_text(row)}"
                  f"  | own set {o['mean']:+7.2f} % on {row['own_clips']:3d}, "
                  f"{len(row['missing'])} lost"
                  f"  | overlap {row['overlap_own_set']['median']:.2f}, "
                  f"{row['overlap_own_set']['thin_clips']} thin")
        lost_clips(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
