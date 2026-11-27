#!/usr/bin/env python3
"""Stage the history series: one run per built day (tools/bench/history_build.sh,
tools/bench/history_plan.py), each with its own build registered as a provenance store, so V25
names the day's commit and build on every report.

    python3 tools/bench/history_stage.py --days DAYS.json --plans PLANS --builds BUILDS \\
        --bin BIN --cfg CFG --clips CLIPS --out OUT

Per day with a build: `anchors/history-<day>/` (SHA256SUMS and BUILD.json: the day's commit,
`--version`, how it was built), a binary directory (the run's reference binaries with the day's
nanocodec), and `OUT/<day>/run` staged by `fleet.py stage` from the day's plan.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fleet  # noqa: E402
import provenance  # noqa: E402

BUILT_BY = ("tools/dev/pgo_build.sh at tools commit {tools} (release LTO, x86-64-v3, "
            "-Cprofile-generate / -Cprofile-use) on the build box (tools/bench/history_build.sh)")
TRAINING = ("PGO_FRAMES=20, PGO_CLIPS = the four cuts of eval/clipset/pgo_training_2026-11-23.json "
            "(as the benchmark's own build); the day's codecs: {codecs}")


def version(build: Path, commit: str) -> str:
    """The build's `--version` line; a build from before the flag (to 11-10) printed its usage
    instead, so its version is its crate version (the build's `cargo_version`, else unknown)
    and commit."""
    line = (build / "version").read_text().strip().splitlines()[0]
    if line.startswith("nanocodec "):
        return line
    crate = build / "cargo_version"
    crate_version = crate.read_text().strip() if crate.exists() else "unknown"
    return f"nanocodec {crate_version} {commit} (this build has no --version)"


def register(day: dict, build: Path, tools: str) -> None:
    """The day's build as an anchor store (V25 finds its commit by its hash)."""
    store = provenance.ANCHORS / f"history-{day['day']}"
    store.mkdir(parents=True, exist_ok=True)
    digest = (build / "sha256").read_text().strip()
    (store / "SHA256SUMS").write_text(f"{digest}  nanocodec\n")
    (store / "BUILD.json").write_text(json.dumps({"nanocodec": {
        "commit": day["full"], "day": day["day"],
        "version": version(build, day["full"]),
        "sha256": digest, "build": BUILT_BY.format(tools=tools),
        "training": TRAINING.format(codecs=day["codecs"])}}, indent=1) + "\n")


def bin_dir(base_bin: Path, build: Path, target: Path) -> Path:
    """The run's reference binaries with the day's nanocodec, and the benchmark's own build as
    the timing references' encoder (`fleet.REFERENCE_BINARY`); links, `stage` copies them."""
    target.mkdir(parents=True, exist_ok=True)
    for binary in base_bin.iterdir():
        link = target / binary.name
        if not link.exists():
            link.symlink_to((build / "nanocodec" if binary.name == "nanocodec"
                             else binary).resolve())
    ref = target / fleet.REFERENCE_BINARY
    if not ref.exists():
        ref.symlink_to((base_bin / "nanocodec").resolve())
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("days", "plans", "builds", "bin", "cfg", "clips", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    a = ap.parse_args()
    tools = fleet.commit()
    for day in json.loads(a.days.read_text()):
        build = a.builds / day["commit"]
        if not (build / "sha256").exists():
            print(f"{day['day']} {day['commit']}: no build, not staged")
            continue
        run = a.out / day["day"] / "run"
        if run.exists():
            print(f"{day['day']}: already staged")
            continue
        register(day, build, tools)
        bins = bin_dir(a.bin, build, a.out / day["day"] / "bin")
        fleet.stage(run, a.plans / day["day"] / "jobs.json", bins, a.cfg, a.clips)
        shutil.copy(a.plans / day["day"] / "conditions.json", run / "conditions.json")
        print(f"{day['day']} {day['commit']}: staged with {(build / 'sha256').read_text()[:12]}")


if __name__ == "__main__":
    main()
