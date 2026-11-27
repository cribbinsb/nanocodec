#!/usr/bin/env python3
"""Fail when a tool launches AWS instances outside the cost guide's rules.

    python3 tools/ci/check_aws_launchers.py        (tools/ci/check.sh runs it)

Why: the BP-3 benchmark RCA (docs/analysis/bp3-benchmark-rca-2026-11-24.md, action R2):
machines and markets were chosen without the guide, and finished work was lost to spot
reclaims in tools that had never been tested for one. Over the tracked Python under tools/
and eval/ (tests excluded), it fails when:

  - a file other than LAUNCH_API_FILES calls an EC2 launch API (run_instances,
    request_spot_instances, create_fleet, request_spot_fleet);
  - a string constant (not a docstring) names an instance type or region that is not in
    docs/aws-cost-guide.md's table "Allowed instance types and regions";
  - `on_demand=True` is written in a file not in ON_DEMAND, which states why;
  - a file that builds an `aws_session.SessionSpec` (it launches instances) has no entry in
    RECLAIM_TESTS: the test that simulates a reclaim mid-run, or the reason it has none.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_names import tracked as python_files  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "docs/aws-cost-guide.md"
GUIDE_TABLE = "## Allowed instance types and regions"
LAUNCH_APIS = {"run_instances", "request_spot_instances", "create_fleet", "request_spot_fleet"}
TYPE = re.compile(r"\b[a-z]\d+[a-z]*\.(?:medium|large|\d*xlarge|metal(?:-\d+xl)?)\b")
REGION = re.compile(r"\b(?:eu|us|ap|sa|ca|me|af)-[a-z]+-\d\b")

#: Files that may call an EC2 launch API, with why.
LAUNCH_API_FILES = {
    "tools/dev/aws_session.py": "the one launcher every session uses (one-time spot, budget, "
                                "hard lifetime, cleanup, ledger)",
    "tools/dev/nc_remote.py": "the build box: a persistent EC2 Fleet over several types "
                              "(price-capacity-optimized, reclaim rule), which aws_session's "
                              "one-type sessions do not offer",
}
#: Files that write `on_demand=True`, with why.
ON_DEMAND = {
    "tools/dev/cpu_ab_shard.py": "the zen5 target: c8a has no eu-north-1 offering and no spot "
                                 "series is kept for it (optional, never a default)",
    "tools/dev/aws_timing.py": "the charts' timing sessions: a reclaimed spot box loses the "
                               "timing (four c8a.large spot sessions reclaimed within 6 min)",
    "tools/dev/fixedqp_timing.py": "timing, as aws_timing",
    "tools/dev/aws_decode_speed.py": "decoder speed timing, as aws_timing",
    "eval/distill/box_run.py": "the study's approved fallback after spot was reclaimed twice "
                               "(2026-11-22)",
}
#: Every file that launches instances: its simulated-reclaim test (FILE::TEST), or why it
#: has none (a reason, not a test path).
RECLAIM_TESTS = {
    "tools/dev/aws_measure.py": "eval/tests/test_aws_measure.py::"
                                "test_a_retry_is_seeded_with_the_points_pulled_from_the_lost_box",
    "tools/dev/aws_warm.py": "eval/tests/test_aws_measure.py::"
                             "test_a_reclaimed_warm_box_s_retry_re_runs_only_the_missing_points",
    "tools/dev/cpu_ab_shard.py": "eval/tests/test_cpu_ab_retry.py::"
                                 "test_a_reclaim_mid_run_loses_only_the_job_in_flight",
    "tools/dev/nc_remote.py": "eval/tests/test_nc_remote_market.py::"
                              "test_a_job_lost_to_a_reclaim_or_refused_on_a_notice_runs_again_on_a_new_box",
    "tools/dev/aws_heldout_chart.py": "eval/tests/test_heldout_fleet.py::"
                                      "test_a_lost_session_is_relaunched_on_its_own_lane_and_an_unlaunched_one_waits",
    "tools/bench/fleet.py": "eval/tests/test_bench_fleet.py::"
                            "test_a_lost_box_gives_back_only_its_unsettled_jobs",
    "eval/nceval/gpuscore.py": "eval/tests/test_gpuscore.py::"
                               "test_a_reclaim_mid_round_loses_only_the_points_in_flight",
    "tools/dev/aws_timing.py": "on demand only (timing): no spot reclaim",
    "tools/dev/fixedqp_timing.py": "on demand only (timing): no spot reclaim",
    "tools/dev/aws_decode_speed.py": "on demand only (timing): no spot reclaim",
    "tools/dev/aws_ctc.py": "rare one-off lens run: download_new checkpoints and a rerun "
                            "resumes from them; no simulated-reclaim test yet",
    "tools/dev/aws_reference_builds.py": "one-off reference builds (unit: a whole build, rerun "
                                         "on failure); the builds are on demand",
    "tools/dev/decoder_yardstick.py": "one-off decoder timing comparison, rerun whole",
    "tools/dev/vp9_efficiency_probe.py": "one-off probe, rerun whole",
    "tools/dev/aws_bake.py": "one-off image bake (unit: one image, minutes): a reclaim leaves "
                             "no image and the next warm box uses the stock image; rerun whole",
    "eval/distill/box_run.py": "the distillation study: Session.pulling copies results off "
                               "as written, a rerun redoes the rest; no simulated-reclaim test",
}


def tracked(root: Path = REPO) -> list[str]:
    """The Python files checked (repository-relative): git's list, or a walk of the tree
    where there is no repository (the build box's synced copy: check_names.tracked)."""
    files = [str(path.relative_to(root)) for path in python_files(".py", root)]
    return [f for f in files if f.startswith(("tools/", "eval/")) and "/tests/" not in f
            and not f.startswith("eval/tests") and f != "tools/ci/check_aws_launchers.py"]


def allowed() -> tuple[set[str], set[str]]:
    """The guide table's types and the regions named after it."""
    text = GUIDE.read_text()
    section = text[text.index(GUIDE_TABLE):]
    section = section[:section.index("\n## ", len(GUIDE_TABLE))]
    return set(TYPE.findall(section)), set(REGION.findall(section))


def docstrings(tree: ast.AST) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant):
            found.add(id(node.body[0].value))
    return found


def problems(path: str, source: str, types: set[str], regions: set[str]) -> list[str]:
    tree = ast.parse(source)
    skip = docstrings(tree)
    found, launches = [], False
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            for kind, pattern, known in (("instance type", TYPE, types),
                                         ("region", REGION, regions)):
                for value in pattern.findall(node.value):
                    if value not in known:
                        found.append(f"{path}:{node.lineno}: {kind} {value} is not in the "
                                     f"cost guide's table ({GUIDE_TABLE[3:]})")
        if isinstance(node, ast.Attribute) and node.attr in LAUNCH_APIS \
                and path not in LAUNCH_API_FILES:
            found.append(f"{path}:{node.lineno}: calls EC2 {node.attr} directly: launch through "
                         f"aws_session")
        if isinstance(node, ast.keyword) and node.arg == "on_demand" \
                and isinstance(node.value, ast.Constant) and node.value.value is True \
                and path not in ON_DEMAND:
            found.append(f"{path}:{node.value.lineno}: on_demand=True without a reason in "
                         f"ON_DEMAND (cost guide rule: no on-demand without a stated reason)")
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) \
                == "SessionSpec":
            launches = True
    if launches and path not in RECLAIM_TESTS and path != "tools/dev/aws_session.py":
        found.append(f"{path}: launches instances (SessionSpec) but has no entry in RECLAIM_TESTS")
    return found


def missing_tests() -> list[str]:
    """RECLAIM_TESTS entries naming a test that does not exist."""
    out = []
    for tool, entry in RECLAIM_TESTS.items():
        if "::" not in entry:
            continue
        file, name = entry.split("::")
        test = REPO / file
        if not test.is_file() or f"def {name}(" not in test.read_text():
            out.append(f"{tool}: its reclaim test {entry} does not exist")
    return out


def main() -> int:
    types, regions = allowed()
    found = [p for f in tracked() for p in problems(f, (REPO / f).read_text(), types, regions)]
    found += missing_tests()
    for line in found:
        print(line)
    if found:
        print(f"check_aws_launchers: {len(found)} problem(s) (docs/aws-cost-guide.md; "
              f"the registries in tools/ci/check_aws_launchers.py)")
        return 1
    print("check_aws_launchers: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
