"""The distillation study on a rented spot box of its own (tools/dev/aws_session.py), for
fits too heavy for the shared build box (2026-11-22: its load stood at 5x its cores).

    python3 eval/distill/box_run.py --codecs hevc,av1 --contexts twin,plain \\
        --models rule,trial_mlp,gbdt,wide_mlp,cnn [--ablation] --out DIR \\
        [--budget 6] [--lifetime 90] [--threads 60]

Staged and uploaded: this tree's eval/ at HEAD (git archive), nn-lead's eval/learned, and
the nn dataset v0's search twins and plain shards of the fast rungs with their patch
stores and manifests (symlinks, followed by the upload's tar). On the box: a Python
environment with numpy, torch (CPU), lightgbm, scikit-learn and pandas; then per codec and
context, distill.features and distill.models side by side. Results come back to --out as
they are written; the box is terminated on every exit.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/dev"))
from aws_session import BUILD_BOX_REGION, NoCapacity, Session, SessionSpec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path.home() / "nanocodec-data"
VERSION = DATA / "nn-data/v0"
LEARNED = DATA / "worktrees/nn-lead/eval/learned"
#: Spot types tried in order (coordinator, 2026-11-22: eu-north-1's quota is the build
#: boxes'; 64 vCPU families with RAM for the samples, spilling to eu-west-1 / eu-central-1).
#: No c7a (owner: double-pumped AVX-512; docs/aws-cost-guide.md). Arm (r8g.16xlarge $0.52/h,
#: m7g.16xlarge $0.29/h spot on 2026-11-24) is cheaper but its torch-CPU stack is untried.
SPOT_TYPES = ("r7i.16xlarge", "m7i.16xlarge", "c6a.16xlarge")
#: The on-demand fallback (approved for this study: up to $4/h, 1.5 h).
ON_DEMAND_TYPE = "m7i.16xlarge"
RUNGS = {"hevc": "medium,fast,faster", "av1": "medium,fast,faster,veryfast"}
REMOTE = "~/distill"
SETUP = ("sudo apt-get -qq update && sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y "
         "python3-venv libgomp1 > /dev/null && python3 -m venv ~/venv && ~/venv/bin/pip install -q numpy "
         "pandas scikit-learn lightgbm && ~/venv/bin/pip install -q torch "
         "--index-url https://download.pytorch.org/whl/cpu")


def stage(directory: Path, codecs: list[str]) -> None:
    """Symlinks of everything the box needs, under `directory`."""
    subprocess.run(f"git -C {shlex.quote(str(REPO))} archive HEAD eval | tar -C "
                   f"{shlex.quote(str(directory))} -xf -", shell=True, check=True)
    (directory / "learned").symlink_to(LEARNED)
    # The clip sets, so the split's sealed-source check sees every sealed family.
    (directory / "clips").mkdir()
    for clipset in (DATA / "clips").glob("*.json"):
        (directory / "clips" / clipset.name).symlink_to(clipset)
    data = directory / "nn-data/v0"
    data.mkdir(parents=True)
    (data / "manifest.json").symlink_to(VERSION / "manifest.json")
    for part in sorted(p for p in VERSION.iterdir() if p.is_dir()):
        if not any(part.name.startswith(c) for c in codecs):
            continue
        (data / part.name).mkdir()
        for name in ("manifest.json", "patches"):
            if (part / name).exists():
                (data / part.name / name).symlink_to(part / name)
        for job in part.iterdir():
            fields = job.name.split("-")
            if len(fields) < 3 or not job.is_dir():
                continue  # patches/, manifest.json
            codec, rung = fields[:2]
            if codec in codecs and rung in RUNGS[codec].split(","):
                (data / part.name / job.name).symlink_to(job)


def deployable_runs(pairs: list[str]) -> list[str]:
    """distill.deployable for each `codec:rung`, on the plain-context features."""
    runs = []
    for pair in pairs:
        codec, rung = pair.split(":")
        feat = f"{REMOTE}/feat-{codec}-plain"
        out = f"{REMOTE}/out/tree-{codec}-{rung}"
        runs.append(f"(test -d {feat} || ~/venv/bin/python -m distill.features --root "
                    f"{REMOTE}/nn-data/v0 --codec {codec} --rungs {RUNGS[codec]} --context plain "
                    f"--out {feat} --workers 24 > /dev/null; ~/venv/bin/python -m "
                    f"distill.deployable --features {feat} --rung {rung} --out {out}.json) "
                    f"> {out}.log 2>&1 &")
    return runs


def job_command(codecs, contexts, models, ablation, threads, skip=(), deployable=()) -> str:
    runs = deployable_runs(list(deployable)) if deployable else []
    for codec in codecs if not deployable else ():
        for context in contexts:
            if f"{codec}-{context}" in skip:
                continue
            feat = f"{REMOTE}/feat-{codec}-{context}"
            out = f"{REMOTE}/out/{codec}-{context}"
            runs.append(
                f"(~/venv/bin/python -m distill.features --root {REMOTE}/nn-data/v0 --codec {codec} "
                f"--rungs {RUNGS[codec]} --context {context} --out {feat} --workers 24 > /dev/null "
                f"&& ~/venv/bin/python -m distill.models --features {feat} --seed nn-fast-1 "
                f"--out {out}.json --models {models}{' --ablation' if ablation else ''}) "
                f"> {out}.log 2>&1 &")
    threads = max(1, threads // max(1, len(runs)))
    # Setup first, then the runs side by side: each `&` starts its own list, so a setup
    # joined to the first run with `&&` would not hold for the others.
    return (f"cd {REMOTE}/eval; mkdir -p {REMOTE}/out; export OMP_NUM_THREADS={threads} "
            f"NC_DATA_ROOT={REMOTE}; {' '.join(runs)} wait; tail -n 40 {REMOTE}/out/*.log")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--codecs", default="hevc,av1")
    p.add_argument("--contexts", default="twin,plain")
    p.add_argument("--skip", default="", help="codec-context pairs to leave out (e.g. hevc-twin)")
    p.add_argument("--deployable", default="",
                   help="instead of the study, fit distill.deployable for these codec:rung pairs")
    p.add_argument("--models", default="rule,trial_mlp,gbdt,wide_mlp")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--budget", type=float, default=6.0)
    p.add_argument("--lifetime", type=int, default=90)
    p.add_argument("--threads", type=int, default=60, help="OMP threads across all fits")
    p.add_argument("--on-demand", action="store_true",
                   help="straight to on demand (spot was reclaimed twice on 2026-11-22)")
    args = p.parse_args()
    codecs = args.codecs.split(",")
    spot = SessionSpec(region=BUILD_BOX_REGION, instance_type=SPOT_TYPES[0],
                       purpose="nn-fast distillation study", lifetime_minutes=args.lifetime,
                       budget_usd=args.budget, root_gb=120, fallback_types=SPOT_TYPES[1:],
                       launch_wait_minutes=10)
    on_demand = SessionSpec(region=BUILD_BOX_REGION, instance_type=ON_DEMAND_TYPE,
                            purpose="nn-fast distillation study (on demand)",
                            lifetime_minutes=args.lifetime, budget_usd=args.budget,
                            root_gb=120, on_demand=True)
    with tempfile.TemporaryDirectory(prefix="nanocodec-distill-") as tmp:
        stage(Path(tmp), codecs)  # before the box: a staging error costs nothing
        try:
            if args.on_demand:
                raise NoCapacity("on demand asked for")
            with Session(spot) as box:
                run(box, Path(tmp), codecs, args)
        except NoCapacity:
            with Session(on_demand) as box:
                run(box, Path(tmp), codecs, args)


def run(box: Session, staged: Path, codecs: list[str], args) -> None:
    box.upload(staged, REMOTE)
    box.run_detached(SETUP, f"{REMOTE}/setup.log", timeout_minutes=20)
    with box.pulling(f"{REMOTE}/out", args.out, "distill-out"):
        box.run_detached(job_command(codecs, args.contexts.split(","), args.models,
                                     args.ablation, args.threads, args.skip.split(","),
                                     [d for d in args.deployable.split(",") if d]),
                         f"{REMOTE}/run.log", timeout_minutes=args.lifetime - 25,
                         stall_minutes=60)


if __name__ == "__main__":
    main()
