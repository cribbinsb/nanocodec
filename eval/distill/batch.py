"""Write the oracle arms of distill.arms as an aws_measure --batch file of L1 screens.

    python3 -m distill.batch --binary BIN --codec av1 [--rungs fast,faster] --out FILE \\
        --results DIR

Each command is one `nceval.screen` of an arm against its rung (quality only, k 0): the
day's L1 draw, 150 frames, single-point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from distill.arms import FAST_RUNGS, arms
from nceval.encoders.nanocodec import param_text


def pairs(values: dict[str, float]) -> list[str]:
    out = []
    for name, value in sorted(values.items()):
        out += ["--param", f"{name}={param_text(value)}"]
    return out


def commands(binary: Path, codec: str, rungs: list[str], results: Path) -> list[list[str]]:
    out = []
    for rung in rungs:
        sets = arms(binary, codec, rung)
        anchor = [a.replace("--param", "--anchor-param") for a in pairs(sets["base"])]
        for arm, values in sets.items():
            if arm == "base":
                continue
            out.append(["python3", "-m", "nceval.screen", "--codec", codec, "--binary",
                        str(binary), *pairs(values), *anchor, "--clips", "adopt",
                        "--levels", "quick", "--k", "0",
                        "--json", str(results / f"{codec}-{rung}-{arm}.json")])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--codec", required=True)
    p.add_argument("--rungs")
    p.add_argument("--arms", help="only these arms (comma-separated)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    args = p.parse_args()
    rungs = args.rungs.split(",") if args.rungs else list(FAST_RUNGS[args.codec])
    cmds = commands(args.binary.resolve(), args.codec, rungs, args.results.resolve())
    if args.arms:
        wanted = set(args.arms.split(","))
        cmds = [c for c in cmds if Path(c[-1]).stem.rsplit("-", 1)[1] in wanted]
    args.out.write_text(json.dumps(cmds, indent=1))
    print(f"{len(cmds)} screens -> {args.out}")


if __name__ == "__main__":
    main()
