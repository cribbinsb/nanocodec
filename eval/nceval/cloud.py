"""Cloud by default: the measurement entry points run on a warm rented box.

The PC is shared by every agent and the owner's other work (load 55-72 on
2026-11-21, mostly screens' encodes, ffmpeg and libvmaf), so `nceval.quick`,
`nceval.screen`, `nceval.levels final-check`, `nceval.fixedqp ab` and
`nceval.train run` hand themselves to tools/dev/aws_measure.py, which runs the
same command on a warm box (tools/dev/aws_warm.py) and brings back its result
JSON and lever rows. `--local` runs it here instead. On the box (ON_BOX_ENV,
set by aws_measure) the command runs where it is.

What stays on the PC: landing, verify, wall-clock timing and GPU scoring for
published charts (docs/PROCEDURES.md "Three-stage checks").
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ON_BOX_ENV = "NC_ON_BOX"
LOCAL_FLAG = "--local"
AWS_MEASURE = Path(__file__).resolve().parents[2] / "tools" / "dev" / "aws_measure.py"


#: The integration branch, and the cloud tools whose copy in a lagging worktree has
#: run stale (2026-11-21: agents' trees predated fixes to --batch and the quota queue).
INTEGRATION = "variance-integration"
FRESH_FILES = ("tools/dev/aws_measure.py", "tools/dev/aws_session.py",
               "tools/dev/aws_warm.py", "eval/nceval/cloud.py", "eval/nceval/quick.py",
               "eval/nceval/train.py", "eval/nceval/checkrun.py")
STALE_ENV = "NC_ALLOW_STALE_TOOLS"
#: The tool version a job or study started with (tools_version), passed down to its
#: later slices: those are not checked again while they run this same tree at that
#: version (2026-11-21: five tooling landings in an hour failed every running study
#: at its next slice).
PINNED_ENV = "NC_TOOLS_PINNED"
REPO = Path(__file__).resolve().parents[2]


def tools_version() -> str:
    """This tree and the contents of its cloud tools, as one digest."""
    import hashlib
    digest = hashlib.sha256(str(REPO).encode())
    for name in FRESH_FILES:
        path = REPO / name
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()[:16]


def check_fresh() -> None:
    """Refuse to start when the integration branch has commits to the cloud tools that
    this tree lacks (its own unlanded changes to them are fine), with the fix. The
    check applies when a job or study starts: once passed, the version is pinned in
    PINNED_ENV, so the commands it starts later (its slices) run unchecked as long as
    the tree's tools are unchanged; a resume, or a changed tree, is checked again."""
    if os.environ.get(ON_BOX_ENV) or os.environ.get(STALE_ENV) or not (REPO / ".git").exists():
        return
    version = tools_version()
    if os.environ.get(PINNED_ENV) == version:
        return
    found = subprocess.run(["git", "-C", str(REPO), "log", "--format=%h %s",
                            f"HEAD..{INTEGRATION}", "--", *FRESH_FILES],
                           capture_output=True, text=True)
    commits = found.stdout.strip().splitlines() if found.returncode == 0 else []
    if not commits:
        os.environ[PINNED_ENV] = version  # inherited by the slices this process starts
        return
    raise SystemExit(f"!!! this tree's cloud tools are behind {INTEGRATION} by {len(commits)} "
                     f"commit(s) (latest: {commits[0]}).\n    fix: git -C {REPO} rebase "
                     f"{INTEGRATION}    ({STALE_ENV}=1 runs the stale copy anyway)")


#: Set in every nc-remote job (the build box has no AWS credentials to offload with).
BUILD_BOX_ENV = "NC_REMOTE_BOX"


#: The instance-metadata service a box asks for its own type (IMDSv2: a token first).
IMDS = "http://169.254.169.254/latest"
IMDS_TIMEOUT_S = 2


def cpu_family() -> str:
    """Where callgrind's instruction counts come from, as far as they depend on it: a
    rented box's instance family (`c7a`, `c7i`: the CPU, so the kernels the dispatch
    picks; AVX-512 or not), else `host:NAME`. A tune study is pinned to it
    (tune_pool.check_family): its CPU ratios are against an anchor counted there."""
    if not os.environ.get(ON_BOX_ENV):
        import socket
        return f"host:{socket.gethostname()}"
    import urllib.request
    try:
        token = urllib.request.urlopen(urllib.request.Request(
            f"{IMDS}/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}),
            timeout=IMDS_TIMEOUT_S).read().decode()
        kind = urllib.request.urlopen(urllib.request.Request(
            f"{IMDS}/meta-data/instance-type", headers={"X-aws-ec2-metadata-token": token}),
            timeout=IMDS_TIMEOUT_S).read().decode()
    except OSError as error:
        raise SystemExit(f"this box's instance type is unknown ({error}): a tune study's "
                         f"CPU cannot be pinned to its family") from error
    return kind.split(".")[0]


def box_family(family: str | None) -> str | None:
    """A recorded cpu_family() as an instance family to ask aws_measure for (None: a
    family of this machine, which no box has)."""
    return None if not family or family.startswith("host:") else family


def runs_here(local: bool) -> bool:
    return local or bool(os.environ.get(ON_BOX_ENV)) or bool(os.environ.get(BUILD_BOX_ENV))


def offload(module: str, argv: list[str], json_out: str | None,
            default_json: Path) -> None:
    """Run `python3 -m module argv` on a warm box and exit with its status. `argv`
    without --local; a --json is added (aws_measure brings the result back there)
    when the command has none."""
    check_fresh()
    argv = [a for a in argv if a != LOCAL_FLAG]
    if not json_out:
        default_json.parent.mkdir(parents=True, exist_ok=True)
        argv += ["--json", str(default_json)]
    print(f"cloud: {module} on a warm box (tools/dev/aws_measure.py; {LOCAL_FLAG} runs it "
          f"here)", flush=True)
    code = subprocess.call([sys.executable, str(AWS_MEASURE), "--",
                            "python3", "-m", module, *argv])
    if code == 0:
        print(f"cloud: result in {json_out or default_json}", flush=True)
    sys.exit(code)


def default_json(name: str) -> Path:
    """Where an offloaded run without --json leaves its result."""
    from .config import load_paths
    import time
    return load_paths().work / "cloud-results" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-" \
        f"{os.getpid()}.json"
