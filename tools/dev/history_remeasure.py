#!/usr/bin/env python3
"""The history page re-measured: every frame's quality and speed from the same encodes.

Owner rule (2026-11-23): never combine measurements of different things. The page's
stored frames paired CBR quality with fixed-QP speed on three other clips, and the frames
to 2026-11-20 lost their quality points in the 2026-11-21 clean-up. So every frame is
measured again, on its own build, with today's chart job list (nceval.chart.chart_jobs:
the real-use ladder and the upper rungs on every slice clip, the low rungs on the basis's
low-rung clips): its quality points on rented machines (tools/dev/heldout_fleet.py), and
its speed by timing every one of those encodes (anchor_chart_timing.py plan --all-encodes,
aws_timing.py on c8a.large, split by clip). Then tools/dev/history_chart.py's page is
rebuilt from them (`report`).

  stage   the frames to 2026-11-20: each day's build (the plain release build the page
          was first measured on, WORK/fetch/remote-out/hist/<commit>) as a chart build
          directory charts/builds/hist-<commit>/ (nanocodec, nanocodec-timed-release,
          presets/), and the frame's build record docs/results/history-<day>-builds.json
          (its families in that build's own preset names). A build before --display-recon
          (2026-11-11) is wrapped (`compat_script`).
  points  the frame's quality points (heldout_fleet.py measure on the slice, --record).
  plan    the frame's timing plan on the build box: every encode of `points`, sizes from
          its stored streams (anchor_chart_timing.py sizes / plan --bytes-from).
  time    the frame's timing on c8a.large sessions, split by clip (aws_timing --inputs),
          merged into WORK/remeasure/timing-<day>.json (chart_refresh.merge_timing).
  report  every frame's BD-J (on the slice, against the zero line, the basis recomputed
          from every frame's points at once) and fps (1000 / geomean over its entries'
          times, as nceval.chart.with_speed) into history.json (`--out DIR`: a scratch copy).

    python3 tools/dev/history_remeasure.py stage
    python3 tools/dev/history_remeasure.py points|plan|time --day 2026-11-15
    python3 tools/dev/history_remeasure.py report [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import history_chart as hc  # noqa: E402
from nceval import chart  # noqa: E402
from nceval.encoders.nanocodec import _build_stem, presets_in  # noqa: E402

DATA = hc.DATA
#: The frames to 2026-11-20 as first measured: each day's plain release build and tree.
HIST_SOURCE = hc.WORK / "fetch" / "remote-out" / "hist"
BUILDS_ROOT = DATA / "charts" / "builds"
SCORING_BINARY, TIMED_BINARY = "nanocodec", "nanocodec-timed-release"
#: The re-measure's plans, timing sessions, merged timing records and report (durable,
#: tools/dev/retention.toml); the frames' quality streams beside the held-out charts'.
REMEASURE = DATA / "charts" / "history-remeasure"
STREAMS = DATA / "work" / "heldout-chart" / "streams-history-{day}"
#: A frame's build record (the frames from 2026-11-21 have theirs already).
FRAME_RECORD = "history-{day}-builds.json"
#: Records written before this convention: 0.6.0's frame is the fresh chart's builds.
RECORD_OF = {"2026-11-21": "bd-vs-speed-independent_fresh_2026-11-13-builds.json"}
#: The plain release builds the later frames are timed on (their records name scoring
#: builds): 0.6.0's plain build beside its PGO one.
TIMED_OF = {"2026-11-21": DATA / "release/builds/207745d7/x86_64/plain/nanocodec"}
#: CLI flags today's quality command passes that early builds lack (`compat_script`).
DISPLAY_RECON, OLD_RECON = "--display-recon", "--recon"
LAYERS_FLAG, TIMES_FLAG = "--temporal-layers", "--timestamps-out"
LOAN_COLUMN = "key_loan_bits"
#: Rented-session options: the quality fleet's boxes, the timing sessions a frame.
FLEET_BOXES = 4
TIMING_SESSIONS = 6
TIMING_MACHINE = "c8a.large"


def frame_record(day: str) -> Path:
    return hc.RESULTS / RECORD_OF.get(day, FRAME_RECORD.format(day=day))


def build_id(commit: str) -> str:
    return f"hist-{commit[:9]}"


def usage(binary: Path) -> str:
    done = subprocess.run([str(binary), "--help"], capture_output=True, text=True)
    return done.stdout + done.stderr


# --- stage -------------------------------------------------------------------------------

def compat_script(binary: Path, commit: str) -> bytes:
    """A self-contained script that runs `binary` (appended to it, unpacked beside the
    script on first use) with today's quality command adapted to its older CLI: the
    displayed reconstruction as --recon (before 2026-11-11), a piped --input copied to a
    file first (such a build seeks in its input), and before 2026-11-08 no
    --temporal-layers (it coded one layer only), --timestamps-out written here (every
    frame coded at its nominal time: no drops or holds) and the rate log's key_loan_bits
    column added as 0 (rate checks only; BD uses the achieved rate). The timing runs the
    build itself, not this (`time`)."""
    text = usage(binary)
    layers, times = LAYERS_FLAG not in text, TIMES_FLAG not in text
    loan = LOAN_COLUMN.encode() not in binary.read_bytes()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    loop = ""
    if layers:
        loop += (f'  if [ "$previous" = {LAYERS_FLAG} ]; then previous=$a; continue; fi\n'
                 f'  if [ "$a" = {LAYERS_FLAG} ]; then previous=$a; continue; fi\n')
    if times:
        loop += (f'  if [ "$previous" = {TIMES_FLAG} ]; then times=$a; previous=$a; continue; fi\n'
                 f'  if [ "$a" = {TIMES_FLAG} ]; then previous=$a; continue; fi\n'
                 '  [ "$previous" = --frames ] && frames=$a\n'
                 '  [ "$previous" = --fps ] && rate=$a\n')
    if loan:
        loop += '  [ "$previous" = --rate-log ] && log=$a\n'
    after = ""
    if times:
        after += ('if [ -n "$times" ]; then\n'
                  '  awk -v n="$frames" -v r="$rate" \'BEGIN { split(r, f, "/"); '
                  'd = (2 in f) ? f[2] : 1;\n'
                  '    for (i = 0; i < n; i++) printf "%.6f\\n", i * 1000 * d / f[1] }\' '
                  '> "$times"\nfi\n')
    if loan:
        after += ('if [ -n "$log" ]; then\n'
                  f'  awk -F, \'NR == 1 {{ print $0 ",{LOAN_COLUMN}"; next }} '
                  '{ print $0 ",0" }\' "$log" > "$log.loan"\n'
                  '  mv -f -- "$log.loan" "$log"\nfi\n')
    head = f"""#!/usr/bin/env bash
# history_remeasure.py compat_script: nanocodec {commit} (sha256 {digest}) appended below,
# run with today's quality command adapted to its CLI
set -euo pipefail
self=$(readlink -f "$0"); real="$self.real"
if [ ! -x "$real" ]; then
  unpack=$(mktemp "$real.XXXXXX"); tail -c +OFFSET "$self" > "$unpack"
  chmod +x "$unpack"; mv -f -- "$unpack" "$real"
fi
copy=""
trap '[ -z "$copy" ] || rm -f -- "$copy"' EXIT
args=(); previous=""; times=""; log=""; frames=""; rate=""
for a in "$@"; do
{loop}  if [ "$previous" = --input ] && [ ! -f "$a" ]; then
    copy=$(mktemp "${{TMPDIR:-/tmp}}/nanocodec-compat-XXXXXX.yuv"); cat -- "$a" > "$copy"
    previous=$a; a=$copy
  else
    previous=$a
  fi
  [ "$a" = {DISPLAY_RECON} ] && a={OLD_RECON}
  args+=("$a")
done
status=0
"$real" "${{args[@]}}" || status=$?
{after}exit $status
"""
    # OFFSET is the byte after the script; its own digits change the length, so settle it
    offset = len(head.encode())
    while True:
        text_bytes = head.replace("OFFSET", str(offset + 1)).encode()
        if len(text_bytes) == offset:
            return text_bytes + binary.read_bytes()
        offset = len(text_bytes)


def stage_build(commit: str) -> str:
    """charts/builds/hist-<commit>/ from the frame's first-measured build; its id."""
    source = HIST_SOURCE / commit
    target = BUILDS_ROOT / build_id(commit)
    if (target / "COMMIT").is_file():
        return target.name
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    real = source / "nanocodec"
    shutil.copy2(real, partial / TIMED_BINARY)
    if DISPLAY_RECON in usage(real):
        shutil.copy2(real, partial / SCORING_BINARY)
    else:
        (partial / SCORING_BINARY).write_bytes(compat_script(real, commit))
        (partial / SCORING_BINARY).chmod(0o755)
    shutil.copytree(presets_in(source / "tree"), partial / "presets")
    (partial / "COMMIT").write_text(commit + "\n")
    partial.rename(target)
    return target.name


def own_name(key: str, presets: Path) -> str:
    """A stored frame's preset key (today's rung names) as the build named its preset:
    the family name its build record and its points use."""
    codec, _, stem = key.partition("@")
    if stem.startswith(hc.RETIRED):
        return f"nanocodec-{codec}@{stem.removeprefix(hc.RETIRED)}"
    built = _build_stem(presets, codec, stem) if stem else None
    if stem and built is None:
        raise SystemExit(f"{presets}: no {codec} rung for today's {stem!r}")
    return f"nanocodec-{codec}" + (f"@{built}" if built and built != "medium" else "")


def stage_cmd(args: argparse.Namespace) -> None:
    record = hc.stored_record(hc.PAGE_DATA)
    for day in record["days"]:
        path = frame_record(day["date"])
        if path.is_file():
            continue
        build = stage_build(day["commit"])
        presets = BUILDS_ROOT / build / "presets"
        families = {own_name(key, presets): build for key in day["values"]}
        path.write_text(json.dumps({"base": build, "families": families}, indent=1) + "\n")
        print(f"{day['date']}: {build}, {len(families)} families -> {path.relative_to(REPO)}")


# --- points, plan, time ------------------------------------------------------------------

def frame(day: str) -> dict:
    match = [d for d in hc.stored_record(hc.PAGE_DATA)["days"] if d["date"] == day]
    if not match:
        raise SystemExit(f"no stored frame {day}")
    return match[0]


def streams(day: str) -> Path:
    return Path(str(STREAMS).format(day=day))


def slice_options(record: dict) -> list[str]:
    """The slice and the basis's low-rung clips, for the cloud and plan paths."""
    return ["--clipset", str(hc.write_slice_clipset(record)),
            "--low-clips", ",".join(record["low_rung_clips"])]


def own_families(day: str) -> list[str]:
    return list(json.loads(frame_record(day).read_text())["families"])


def reference_names() -> list[str]:
    return [f.name for f in hc.reference_families()]


def points_cmd(args: argparse.Namespace) -> None:
    record = hc.stored_record(hc.PAGE_DATA)
    families = [*reference_names(), *own_families(args.day)]
    store()
    subprocess.run([sys.executable, str(REPO / "tools/dev/heldout_fleet.py"), "measure",
                    "--state", str(REMEASURE / f"fleet-{args.day}.json"),
                    "--boxes", str(FLEET_BOXES), "--purpose", f"history-{args.day}",
                    *(["--dry-run"] if args.dry_run else []), "--",
                    *slice_options(record), "--streams", str(streams(args.day)),
                    "--record", str(frame_record(args.day)), "--families", ",".join(families)],
                   check=True)


def store() -> None:
    """REMEASURE, with its keep marker (tools/dev/retention.toml)."""
    from nceval import retention

    REMEASURE.mkdir(parents=True, exist_ok=True)
    retention.ensure_entry_marker(REMEASURE)


def plan_path(day: str) -> Path:
    return REMEASURE / f"plan-{day}.json"


def plan_cmd(args: argparse.Namespace) -> None:
    """The frame's every-encode plan on the build box (our families, and the references
    with --references), sizes from its quality streams where they are stored."""
    from nceval.chart_timing import PLAN_STORE

    record = hc.stored_record(hc.PAGE_DATA)
    builds = json.loads(frame_record(args.day).read_text())
    store()
    sizes = REMEASURE / f"sizes-{args.day}.json"
    subprocess.run([sys.executable, str(REPO / "tools/dev/anchor_chart_timing.py"), "sizes",
                    "--streams", str(streams(args.day)), "--out", str(sizes)],
                   check=True, cwd=REPO / "eval", env={**os.environ, "PYTHONPATH": "."})
    points = [*own_families(args.day), *(reference_names() if args.references else [])]
    on_box = "$HOME/" + str(BUILDS_ROOT.relative_to(Path.home()))
    base = builds["base"]
    per_family = [a for name, build in builds["families"].items() if build != base
                  for a in (f"--binary-for={name}={on_box}/{build}/{SCORING_BINARY}",
                            f"--presets-for={name}={on_box}/{build}/presets")]
    clipset, low = slice_options(record)[1], slice_options(record)[3]
    remote_store = "~/" + str(PLAN_STORE.relative_to(Path.home()))
    PLAN_STORE.mkdir(parents=True, exist_ok=True)
    relative = plan_path(args.day).relative_to(Path.home())
    plan = shlex.join(["python3", "../tools/dev/anchor_chart_timing.py", "plan",
                       "--all-encodes", "--low-clips", low, "--points", ",".join(points),
                       "--bytes-from", f"sizes-{args.day}.json", "--workers", "32"])
    clips = [c.yuv for c in chart.chart_clips(clipset)]
    box_clipset = "$HOME/" + str(Path(clipset).relative_to(Path.home()))
    command = [str(DATA / "bin/nc-remote"), "run", "--worktree", str(REPO),
               "--name", f"history-plan-{args.day}", "--data", remote_store, "--fetch", remote_store,
               "--data", str(sizes), "--fetch", f"~/{relative}",
               *(a for b in {base, *builds["families"].values()}
                 for a in ("--sync", str(BUILDS_ROOT / b))),
               *(a for tools in (DATA / "work/ratecontrol/bin", DATA / "anchors/x86_64/bin",
                                 DATA / "anchors/vp9-av2/bin") for a in ("--sync", str(tools))),
               "--sync", clipset, *(a for c in clips for a in ("--clip", str(c))), "--",
               f"cp ~/{sizes.relative_to(Path.home())} eval/ && cd eval && PYTHONPATH=. {plan} "
               f"--clipset {box_clipset} --plan ~/{relative} --store {remote_store} "
               f"--binary {on_box}/{base}/{SCORING_BINARY} --presets-dir {on_box}/{base}/presets "
               + " ".join(per_family)]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


def timed_plan(day: str) -> Path:
    """The frame's plan as its timed build takes it: `--temporal-layers 1` left out of our
    commands for a build without that flag (one layer was all it coded)."""
    from nceval.chart_timing import load_plan

    plan = load_plan(plan_path(day))
    binary = timed_binary(day)
    if LAYERS_FLAG not in usage(binary):
        for name, entries in plan["points"].items():
            for entry in entries.values() if name.startswith("nanocodec") else ():
                argv = entry["argv"]
                at = argv.index(LAYERS_FLAG)
                if argv[at + 1] != "1":
                    raise SystemExit(f"{name}: {LAYERS_FLAG} {argv[at + 1]} on a one-layer build")
                entry["argv"] = argv[:at] + argv[at + 2:]
    out = REMEASURE / f"timed-plan-{day}.json"
    out.write_text(json.dumps(plan, indent=1) + "\n")
    return out


def timed_binary(day: str) -> Path:
    if day in TIMED_OF:
        return TIMED_OF[day]
    builds = json.loads(frame_record(day).read_text())
    return BUILDS_ROOT / builds["base"] / TIMED_BINARY


def time_cmd(args: argparse.Namespace) -> None:
    """The frame's plan timed on TIMING_SESSIONS c8a.large sessions, each on its own
    clips (aws_timing --inputs), then merged (chart_refresh.merge_timing)."""
    from chart_refresh import merge_timing
    from nceval.chart_timing import load_plan

    plan_file = timed_plan(args.day)
    plan = load_plan(plan_file)
    inputs = sorted(set(plan["inputs"].values()))
    parts = [inputs[i::TIMING_SESSIONS] for i in range(TIMING_SESSIONS)]
    store()
    root = REMEASURE / f"timing-{args.day}"
    root.mkdir(parents=True, exist_ok=True)
    points = [p for p in plan["points"] if p != plan["anchor"]]
    sessions = []
    for index, part in enumerate(p for p in parts if p):
        out = root / f"part-{index + 1}"
        if (out / "summary.json").is_file():
            continue
        command = [sys.executable, str(REPO / "tools/dev/aws_timing.py"), "--instance-type",
                   TIMING_MACHINE, "--binary", str(timed_binary(args.day)),
                   "--expect-version", usage_version(timed_binary(args.day)),
                   "--plan", str(plan_file), "--points", ",".join(points),
                   "--inputs", ",".join(part), "--repeats", "1", "--rounds", "1",
                   "--lifetime", "150", "--repeat-minutes", "120",
                   "--purpose", f"history-time-{args.day}", "--budget", "1", "--out", str(out),
                   *(["--dry-run"] if args.dry_run else [])]
        log = (root / f"part-{index + 1}.log").open("w")
        sessions.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL))
    failed = [s.args for s in sessions if s.wait()]
    if failed:
        raise SystemExit(f"{len(failed)} timing sessions failed: see {root}/part-*.log")
    if not args.dry_run:
        merge_timing(REMEASURE / f"timing-{args.day}.json",
                     sorted(root.glob("part-*/repeat-1/t-*.json")), rescale=False,
                     build=frame(args.day)["commit"])


def usage_version(binary: Path) -> str:
    """The binary's --version line (empty for a build before its stamp, 2026-11-11)."""
    return subprocess.run([str(binary), "--version"], capture_output=True,
                          text=True).stdout.strip()


# --- report ------------------------------------------------------------------------------

def frame_fps(day: str) -> dict[str, float]:
    """{family: fps} of the frame's timing record: 1000 over the geometric mean of its
    entries' times, as nceval.chart.with_speed."""
    timing = json.loads((REMEASURE / f"timing-{day}.json").read_text())["points"]
    return {name: 1000.0 / chart.geomean(list(entry["ms_per_frame"].values()))
            for name, entry in timing.items()}


def frame_families(day: str) -> list:
    """The frame's families on its builds, each as `<family>#<day>` beside its points'
    own name (curves are cached under the own name)."""
    return hc.record_families(frame_record(day))


def all_curves(run, families: list, clips: list, low: list) -> dict:
    """The families' cached curves on the slice: the real-use ladder, the upper rungs on
    every clip and the low rungs on `low` (nceval.chart.chart_jobs). Refuses to measure
    here: a missing point is the `points` step's."""
    jobs = chart.chart_jobs(run, families, clips, hc.DEFAULT_FRAMES, None, low)
    missing = [p for bench, p in jobs if not bench.cached(p)]
    if missing:
        first = missing[0]
        raise SystemExit(f"{len(missing)} of {len(jobs)} points are not in the store (first: "
                         f"{first.family.name} {first.clip.name} {first.label}): run `points`")
    curves = run.measure("history", families, clips, hc.DEFAULT_FRAMES, budgeted=False)
    return chart.with_extra_rungs(run, "history", curves, families, clips, low,
                                  hc.DEFAULT_FRAMES)


def report_cmd(args: argparse.Namespace) -> None:
    """Every frame's BD-J and fps from its re-measured points and timing; the basis (the
    clips every point overlaps the zero line on) from all frames' points at once."""
    from dataclasses import replace

    record = hc.stored_record(hc.PAGE_DATA)
    clips, low = hc.basis_clips(record)
    run = hc.runner()
    refs = hc.reference_families()
    curves = all_curves(run, refs, clips, low)
    families, keys = list(refs), {}
    for day in record["days"]:
        own = frame_families(day["date"])
        measured = all_curves(run, own, clips, low)
        presets = {f.name: f for f in own}
        for key in day["values"]:
            family = presets[own_name(key, BUILDS_ROOT / json.loads(
                frame_record(day["date"]).read_text())["base"] / "presets")]
            renamed = replace(family, name=f"{family.name}#{day['date']}")
            families.append(renamed)
            keys[renamed.name] = (day["date"], key, family.name)
            for clip in clips:
                curves[(renamed.name, clip.name)] = measured[(family.name, clip.name)]
    labels = {f.name: f.name for f in families}
    points, excluded = chart.anchor_points(curves, families, clips, run, labels)
    by_name = {p.family: p for p in points}
    newest = record["days"][-1]["date"]
    reference_fps = frame_fps(newest)
    for name, stored in record["references"].items():
        stored.update(hc.bd_values(by_name[name]), fps_x86=hc.rounded(reference_fps.get(name)))
    for day in record["days"]:
        fps = frame_fps(day["date"])
        day["values"] = {key: {**hc.bd_values(by_name[name]), "fps_x86": hc.rounded(fps[own])}
                         for name, (date, key, own) in keys.items() if date == day["date"]}
        day["note"] = REMEASURED_NOTE
    record["excluded"] = [{"clip": e.clip, "unscored": e.unscored} for e in excluded]
    record["timing"] = {"machine": f"AWS {TIMING_MACHINE} (Zen 5), one core",
                        "method": TIMING_METHOD, "x264_medium_fps": hc.rounded(
                            reference_fps.get(hc.ANCHOR_TIMED))}
    record.pop("timing_noise", None)
    record["codec_front_base_days"] = hc.add_fronts(record["days"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(record, indent=1) + "\n")
    (out / "README.md").write_text(hc.markdown(record))
    print(f"history re-measured: {len(record['days'])} frames, {len(excluded)} clips left out "
          f"-> {out}")


#: What every re-measured frame is (the page's per-frame note).
REMEASURED_NOTE = ("re-measured 2026-11-23 on its own build: quality and speed from the same "
                   "encodes (every real-use, upper and low rung of the slice, CBR)")
TIMING_METHOD = ("every quality encode of the frame timed once on one core, paired with x264 "
                 "medium on the same encode (aws_timing --rounds 1, split by clip); fps is "
                 "1000 over the geometric mean of the encodes' CPU ms per frame")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage")
    sub.add_parser("report").add_argument("--out", default=str(REMEASURE / "report"),
                                          help="where history.json and README.md go")
    for name in ("points", "plan", "time"):
        p = sub.add_parser(name)
        p.add_argument("--day", required=True)
        p.add_argument("--dry-run", action="store_true")
        if name == "plan":
            p.add_argument("--references", action="store_true",
                           help="plan the references too (once, on one frame)")
    args = parser.parse_args()
    {"stage": stage_cmd, "points": points_cmd, "plan": plan_cmd, "time": time_cmd,
     "report": report_cmd}[
        args.command](args)


if __name__ == "__main__":
    main()
