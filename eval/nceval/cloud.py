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

from .budget import MAX_MINUTES

ON_BOX_ENV = "NC_ON_BOX"
LOCAL_FLAG = "--local"
AWS_MEASURE = Path(__file__).resolve().parents[2] / "tools" / "dev" / "aws_measure.py"
#: aws_measure's run cap (--run-minutes): the project's 30-minute run budget plus a
#: margin for the last points. A run past it is stopped and keeps only the points it
#: finished (pulled back as they land).
RUN_CAP_MINUTES = MAX_MINUTES + 5


#: The integration branch: where the current measurement protocol is defined.
INTEGRATION = "variance-integration"
#: The measurement protocol: what a cloud job's results are comparable on (clip draws,
#: ladder and rates, metrics and their pooling, the screens' estimators and guards, the
#: CPU charge). Bump it ON PURPOSE in the commit that makes results incomparable with
#: earlier ones; a job from a tree at an older protocol is then refused. Any other tools
#: commit leaves running and queued jobs alone: each runs on its own pinned tree
#: (`pinned_tree`; 2026-11-24, a check on every commit to seven tool files refused every
#: agent's queued jobs at each tools landing).
MEASUREMENT_PROTOCOL = 1
PROTOCOL_LINE = "MEASUREMENT_PROTOCOL = "
STALE_ENV = "NC_ALLOW_STALE_TOOLS"
#: The protocol a job or study started with (tools_version), passed down to its later
#: slices: those are not checked again (2026-11-21: five tooling landings in an hour
#: failed every running study at its next slice).
PINNED_ENV = "NC_TOOLS_PINNED"
REPO = Path(__file__).resolve().parents[2]


def tools_version() -> str:
    """The measurement protocol this tree's tools follow (MEASUREMENT_PROTOCOL)."""
    return f"protocol-{MEASUREMENT_PROTOCOL}"


def integration_protocol() -> int | None:
    """MEASUREMENT_PROTOCOL on the integration branch, None when it cannot be read."""
    shown = subprocess.run(["git", "-C", str(REPO), "show",
                            f"{INTEGRATION}:eval/nceval/cloud.py"],
                           capture_output=True, text=True)
    if shown.returncode != 0:
        return None
    lines = [line for line in shown.stdout.splitlines() if line.startswith(PROTOCOL_LINE)]
    return int(lines[0].removeprefix(PROTOCOL_LINE)) if lines else None


def check_fresh() -> None:
    """Refuse to start a job when the integration branch has moved to another measurement
    protocol than this tree's (MEASUREMENT_PROTOCOL), with the fix: its results would not
    be comparable with the others'. A tools commit that keeps the protocol refuses
    nothing. Once passed, the version is pinned in PINNED_ENV for the job's slices."""
    if os.environ.get(ON_BOX_ENV) or os.environ.get(STALE_ENV) or not (REPO / ".git").exists():
        return
    version = tools_version()
    if os.environ.get(PINNED_ENV) == version:
        return
    current = integration_protocol()
    if current is None or current == MEASUREMENT_PROTOCOL:
        os.environ[PINNED_ENV] = version  # inherited by the slices this process starts
        return
    raise SystemExit(f"!!! this tree measures by protocol {MEASUREMENT_PROTOCOL}; "
                     f"{INTEGRATION} is at protocol {current}: results would not be "
                     f"comparable.\n    fix: git -C {REPO} rebase {INTEGRATION}    "
                     f"({STALE_ENV}=1 runs the old protocol anyway)")


#: What a box runs of this tree (aws_measure stages it from git): an uncommitted change
#: there is not what the box would run.
TREE_PATHS = ("eval", "tools", "crates/nc-encoder/presets")


def pinned_tree() -> str:
    """The commit a cloud job's boxes run: HEAD, refused at once (before anything is
    measured or rented) when eval/, tools/ or the presets have uncommitted changes.
    A long job (`nceval.train run`) pins it at its start and hands it to every box it
    launches, so later edits to the tree cannot fail its retries (2026-11-24: two
    trains lost their boxes to a tree edited while they ran)."""
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--",
                            *TREE_PATHS], capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"commit eval/, tools/ and the presets first: the boxes run this "
                         f"tree's HEAD ({dirty.splitlines()[0]} ...)")
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


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
