#!/usr/bin/env python3
"""Our SVC chart points' encodes on one rented spot box; their scoring stays on this PC's GPU.

    python3 -m nceval.svc_chart encode-plan --binary BIN --binary-for POINT=NEW ... \\
        --points POINTS --out PLAN
    python3 tools/dev/aws_svc_encodes.py --plan PLAN --binary NEW --out STREAMS [--budget 3]
    python3 -m nceval.svc_chart site --binary BIN --binary-for POINT=NEW ... --prebuilt STREAMS

The plan (nceval.svc_chart `encode-plan`) is one shell command a line, reading
$NC_SVC_BIN (our encoder), $NC_SVC_CLIPS (`<clip>.yuv`) and $NC_SVC_OUT. This
uploads the encoder, the plan's clips (their first FRAMES frames only: the
chart codes no more) and the plan, runs every command on all the box's cores,
and brings each stream (with its layer table and frame times) back as it
lands (aws_session.Session.pulling). A rerun after a spot reclaim runs only
the commands whose stream is not back yet. The box is terminated on every
exit (aws_session); its spend is in the session ledger.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "eval"))
from aws_session import (THROUGHPUT_X86_TYPES, Session, SessionSpec,  # noqa: E402
                         fallbacks_after, say)
from nceval.svc_chart import FRAMES, chart_clips  # noqa: E402

REGION = "eu-north-1"
#: The first type tried; the rest of the throughput ladder, then these, follow.
INSTANCE_TYPE = "c7i.24xlarge"
SMALLER_TYPES = ("c6i.16xlarge", "c7i.16xlarge", "c6i.12xlarge")
REMOTE = "~/svc-encodes"
CHECKPOINT_MARKER = ".svc-encodes-pulled"
#: A stream path in a plan line (`--output $NC_SVC_OUT/<stem>.ivf`).
OUTPUT = re.compile(r"\$NC_SVC_OUT/(\S+)\.ivf")
CLIP = re.compile(r"\$NC_SVC_CLIPS/(\S+)\.yuv")
#: Run minutes allowed (a full VP9 page plan is a few CPU-hours: minutes on the box).
RUN_MINUTES = 40


def pending(plan: Path, out: Path) -> list[str]:
    """The plan's commands whose stream is not back yet."""
    lines = [line for line in plan.read_text().splitlines() if line.strip()]
    return [line for line in lines
            if not (out / f"{OUTPUT.search(line)[1]}.ivf").is_file()]


def stage(lines: list[str], binary: Path, root: Path) -> None:
    """The upload: our encoder, the clips the commands read (FRAMES frames each) and
    the plan."""
    (root / "clips").mkdir(parents=True)
    shutil.copy2(binary, root / "nanocodec")
    by_name = {c.name: c for c in chart_clips()}
    for name in sorted({CLIP.search(line)[1] for line in lines}):
        clip = by_name[name]
        frame_bytes = clip.width * clip.height * 3 // 2
        with clip.yuv.open("rb") as source, (root / "clips" / f"{name}.yuv").open("wb") as dst:
            dst.write(source.read(frame_bytes * FRAMES))
    (root / "plan.sh").write_text("".join(f"{line}\n" for line in lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True,
                        help="the encoder the plan was made for ($NC_SVC_BIN)")
    parser.add_argument("--out", type=Path, required=True, help="where the streams land")
    parser.add_argument("--budget", type=float, default=3.0, help="USD cap of the session")
    parser.add_argument("--instance-type", default=INSTANCE_TYPE)
    args = parser.parse_args()
    lines = pending(args.plan, args.out)
    if not lines:
        say("every stream of the plan is back")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nanocodec-svc-encodes-") as scratch:
        root = Path(scratch)
        stage(lines, args.binary.resolve(), root)
        upload_gb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e9
        say(f"{len(lines)} encodes, upload {upload_gb:.1f} GB")
        spec = SessionSpec(region=REGION, instance_type=args.instance_type, arch="amd64",
                           purpose="svc-encodes", lifetime_minutes=RUN_MINUTES + 30,
                           budget_usd=args.budget, root_gb=12 + int(upload_gb * 2),
                           fallback_types=(*fallbacks_after(args.instance_type,
                                                             THROUGHPUT_X86_TYPES),
                                           *SMALLER_TYPES))
        command = (f"cd {REMOTE} && mkdir -p out && chmod +x nanocodec && "
                   f"export NC_SVC_BIN=$PWD/nanocodec NC_SVC_CLIPS=$PWD/clips "
                   f"NC_SVC_OUT=$PWD/out && "
                   f"xargs -d '\\n' -P $(nproc) -I CMD bash -c 'CMD && echo done' < plan.sh "
                   f"| awk '{{n++; if (n % 20 == 0) {{print n; fflush()}}}}'")
        with Session(spec) as session:
            session.upload(root, REMOTE)
            with session.pulling(f"{REMOTE}/out", args.out, CHECKPOINT_MARKER):
                session.run_detached(command, "~/svc-encodes.log", RUN_MINUTES)
    missing = pending(args.plan, args.out)
    say(f"{len(lines) - len(missing)} of {len(lines)} streams back")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
