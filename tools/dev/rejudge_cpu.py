#!/usr/bin/env python3
"""Re-judge a recorded screen with CPU measured on its own encodes.

    python3 tools/dev/rejudge_cpu.py RESULT.json [RESULT.json ...] --preset RUNG | --k K
        [--binary B --anchor-binary A] [--out DIR] [--plan-only]

Owner rule (2026-11-23): never combine measurements of different things. Screens
before 2026-11-23 charged a dCPU measured on other encodes (aws_cpu_ab at 600 kbps on
its own clips, or callgrind at fixed QP) to quality measured in CBR at each clip's
ladder rates on the screen's draw. This keeps each recorded result's quality exactly
as recorded (its per-clip rows) and re-measures only the CPU, consistently:

  1. the screen's own CPU plan: `nceval.screen --cpu-plan` with the recorded codec,
     builds, both sides' parameters, draw (rotation), levels and long window, so the
     plan lists the very encodes the result scored (nceval.decision_cpu);
  2. `tools/dev/aws_cpu_ab.py --decision-plan` times them on the rented CPU targets
     (all results' plans in one run, each its own decision);
  3. nceval.levels.rejudge: every level (and key share) charged its own encodes' dCPU
     at the rung's front slope k, decided as the screen decides.

A result's builds are its recorded binaries (`binary`, `anchor_binary`) unless
--binary / --anchor-binary name others (e.g. tools/dev/binary_for.sh COMMIT, when the
result was measured on a box); each must be a clean commit's build so that aws_cpu_ab
can build it for every target. Writes DIR/rejudge.json and prints recorded against
re-judged decisions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
from nceval import decision_cpu, front_k, interval, levels, pool  # noqa: E402
from nceval.screen import RUNG_PARAMS_FLAG  # noqa: E402

DATA = Path.home() / "nanocodec-data"
#: One plan's screen run writes only a plan (a draw and job list): a few seconds.
PLAN_TIMEOUT_S = 300


def screen_argv(result: dict, binary: str, anchor_binary: str, plan: Path,
                name: str) -> list[str]:
    """The recorded screen's command line, writing its CPU plan as decision `name`."""
    final = next((s for s in result["stages"] if s["stage"] == pool.FINAL), {})
    shares = sorted(int(p) for p in final.get("share_results") or {})
    argv = ["--codec", result["codec"], "--binary", binary, "--anchor-binary", anchor_binary,
            "--clips", "adopt", "--frames", str(result["frames"]),
            "--rotation", result["draw"]["rotation"],
            "--levels", ",".join(result["levels"]), RUNG_PARAMS_FLAG,
            "--cpu-plan", str(plan), "--cpu-decision", name]
    argv += [a for p in result["params"] for a in ("--param", p)]
    argv += [a for p in result["anchor_params"] for a in ("--anchor-param", p)]
    if shares:
        argv += ["--final-frames", str(final["frames"]),
                 "--key-shares", ",".join(map(str, shares))]
    return argv


def plan_of(result: dict, args: argparse.Namespace, out: Path, name: str) -> dict:
    binary = args.binary or result["binary"]
    anchor = args.anchor_binary or result["anchor_binary"]
    plan = out / f"{name}.plan.json"
    subprocess.run([sys.executable, "-m", "nceval.screen",
                    *screen_argv(result, binary, anchor, plan, name)],
                   cwd=REPO / "eval", check=True, timeout=PLAN_TIMEOUT_S)
    return json.loads(plan.read_text())["decisions"][0]


def report(name: str, path: str, judged: dict) -> list[str]:
    lines = [f"{name} {path}: re-judged {str(judged['decision']).upper()} "
             f"(k {judged['k']:.4g}; dCPU {decision_cpu.describe(decision_cpu.Charge(**judged['cpu']))})"]
    for level, stats in judged["levels"].items():
        lines.append(f"  {level}: recorded {stats['recorded'].upper()} -> "
                     f"{stats['decision'].upper()}: {interval.describe(stats)}"
                     + (" (would not have run: an earlier level is WORSE)"
                        if stats["after_worse"] else ""))
    return lines


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results", nargs="+", type=Path, help="nceval.screen --json results")
    rate = p.add_mutually_exclusive_group(required=True)
    rate.add_argument("--preset", help="the rung: k is its front slope (nceval.front_k)")
    rate.add_argument("--k", type=float, help="an explicit CPU weight (warned)")
    p.add_argument("--binary", help="the candidate build (default: the recorded one)")
    p.add_argument("--anchor-binary", help="the anchor build (default: the recorded one)")
    p.add_argument("--out", type=Path, default=DATA / "results" / "rejudge",
                   help="where the plans, the CPU summary and rejudge.json go")
    p.add_argument("--plan-only", action="store_true", help="write the plans and stop")
    args = p.parse_args(argv)
    results = {f"r{i}": (path, json.loads(path.read_text()))
               for i, path in enumerate(args.results)}
    for name, (path, result) in results.items():
        if result.get("status") != "done" or result.get("mode") != "real" or \
                result.get("clips") not in (None, "adopt"):
            raise SystemExit(f"{path}: not a finished adoption screen (real mode, --clips adopt)")
    args.out.mkdir(parents=True, exist_ok=True)
    entries = [plan_of(result, args, args.out, name) for name, (_, result) in results.items()]
    merged, summary = args.out / "plan.json", args.out / "summary.json"
    decision_cpu.write_plan(merged, entries)
    if args.plan_only:
        return
    subprocess.run([sys.executable, str(REPO / "tools/dev/aws_cpu_ab.py"), "--decision-plan",
                    str(merged), "--json", str(summary), "--boxes", "3",
                    "--purpose", "rejudge-cpu"], check=True)
    judged, lines = {}, []
    for entry, (name, (path, result)) in zip(entries, results.items()):
        codec = result["codec"]
        k, source = front_k.resolve(codec, args.preset, args.k)
        found = decision_cpu.verify(decision_cpu.load_charge(summary, name), entry["digest"],
                                    str(path))
        judged[str(path)] = {**levels.rejudge(result, found, k), "k_source": source}
        lines += report(name, str(path), judged[str(path)])
    (args.out / "rejudge.json").write_text(json.dumps(judged, indent=1, default=str) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
