#!/usr/bin/env python3
"""Record how build_anchors.sh built one reference tool, in PREFIX/build-flags.json.

    build_record.py --prefix P --tool T --build BLD --log LOG --source FILE
                    --ref R --commit C --lto TEXT --pgo TEXT --profile opt|plain
    build_record.py collect MACHINE=PREFIX ...   # rewrite tools/anchors/build-flags.json

Called by build_anchors.sh after each recipe that builds an encoder. It writes the
tool's exact configure line (as the recipe ran it, from the tool's build log), the
compile line of one representative source file of the encoder's generic C/C++ code
(from CMake's compile database, else from the make output in the log), the link
line of its program where one can be found, and what LTO and PGO the recipe used.
tools/anchors/build-flags.json is the checked-in copy of these files that the pages
print (docs/PROCEDURES.md "Reference builds").
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path

#: The program each tool installs whose link line is recorded.
PROGRAMS = {"x264": "x264", "x265": "x265", "svtav1": "SvtAv1EncApp", "libaom": "aomenc",
            "libvpx": "vpxenc", "openh264": "h264enc", "harness": "openh264-harness",
            "avm": "avmenc", "vvenc": "vvencFFapp", "kvazaar": "kvazaar"}
#: Flag prefixes that describe code generation (include paths, defines and warnings
#: are left out: they are the project's and do not change how fast it runs).
CODEGEN_PREFIXES = ("-O", "-f", "-m", "-g", "-std=", "-pipe", "-flto", "-DNDEBUG")
#: The checked-in copy of every timed build's record, keyed by timing machine
#: (nceval.config.REFERENCE_BUILDS), that the pages print.
CHECKED_IN = Path(__file__).resolve().with_name("build-flags.json")
#: What every record says about the policy (docs/PROCEDURES.md "Reference builds").
POLICY = ("Each reference encoder is built by tools/anchors/build_anchors.sh --opt with the "
          "link-time (LTO) and profile-guided (PGO) optimisation its own build system offers "
          "and nothing else: its own release flags, no -march, its shipped runtime ISA "
          "dispatch; PGO trains on the release's four public Xiph clips (never a chart clip).")
#: The marker build_anchors.sh's run() puts before each command it logs.
COMMAND_MARK = "+ "


def codegen_flags(command: str) -> list[str]:
    tokens = shlex.split(command)
    return [t for t in tokens[1:] if t.startswith(CODEGEN_PREFIXES)]


def configure_lines(log: list[str]) -> list[str]:
    """The configure / cmake configure commands the recipe ran, in order."""
    return [line[len(COMMAND_MARK):] for line in log if line.startswith(COMMAND_MARK)
            and ("/configure" in line or " -S " in line or "cmake -S" in line)]


def compile_line(build: Path, log: list[str], source: str) -> str | None:
    """The last compile command of `source` (a path suffix)."""
    database = build / "compile_commands.json"
    if database.is_file():
        for entry in reversed(json.loads(database.read_text())):
            if entry["file"].endswith(source):
                return entry.get("command") or shlex.join(entry["arguments"])
    for line in reversed(log):
        tokens = line.split()
        if "-c" in tokens and any(t == source or t.endswith(f"/{source}") for t in tokens):
            return line.strip()
    return None


def link_line(build: Path, log: list[str], program: str) -> str | None:
    """The link command of `program`: CMake's link.txt, else the make output."""
    def links_program(line: str) -> bool:
        tokens = line.split()
        return " -c " not in f" {line} " and any(
            a == "-o" and (b == program or b.endswith(f"/{program}"))
            for a, b in zip(tokens, tokens[1:]))
    for link in sorted(build.rglob("link.txt")):
        text = " ".join(link.read_text().split())
        if links_program(text):
            return text
    return next((line.strip() for line in reversed(log) if links_program(line)), None)


def version(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown"


def collect(pairs: list[str]) -> None:
    """Write CHECKED_IN from each MACHINE=PREFIX's build-flags.json."""
    builds = {}
    for pair in pairs:
        machine, prefix = pair.split("=", 1)
        builds[machine] = json.loads((Path(prefix).expanduser() / "build-flags.json").read_text())
    CHECKED_IN.write_text(json.dumps({"policy": POLICY, "builds": builds}, indent=1,
                                     sort_keys=True) + "\n")


def main() -> None:
    if sys.argv[1:2] == ["collect"]:
        collect(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("prefix", "tool", "build", "log", "source", "ref", "commit", "lto", "pgo",
                 "profile"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    log = Path(args.log).read_text(errors="replace").splitlines()
    build = Path(args.build)
    compiled = compile_line(build, log, args.source)
    if compiled is None:
        raise SystemExit(f"build_record: no compile line of {args.source} for {args.tool}")
    linked = link_line(build, log, PROGRAMS.get(args.tool, args.tool))
    record = {
        "ref": args.ref,
        "commit": args.commit,
        "configure": configure_lines(log),
        "compile_flags": codegen_flags(compiled),
        "compile_source": args.source,
        "link_flags": codegen_flags(linked) if linked else None,
        "lto": args.lto,
        "pgo": args.pgo,
    }
    out = Path(args.prefix) / "build-flags.json"
    data = json.loads(out.read_text()) if out.is_file() else {}
    data.update({
        "profile": args.profile,
        "platform": platform.machine(),
        "compiler": {"cc": version(["gcc", "--version"]), "cxx": version(["g++", "--version"]),
                     "cmake": version(["cmake", "--version"])},
    })
    data.setdefault("tools", {})[args.tool] = record
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
