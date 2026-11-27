"""The timed reference builds' flags record (tools/anchors/build-flags.json) covers every
reference program the charts run, on every timing machine with its own build
(docs/PROCEDURES.md "Reference builds")."""

import json
from pathlib import Path

from nceval.cbr import ANCHOR_TOOLS
from nceval.config import ANCHOR_BUILD_TOOLS, REFERENCE_BUILDS, reference_bin

REPO = Path(__file__).resolve().parents[2]
FLAGS = REPO / "tools/anchors/build-flags.json"
PLAN = REPO / "docs/results/bd-vs-speed-plan.json"
#: Timing machines whose references are the --opt builds (the Jetson keeps the board's).
OPT_MACHINES = ("x86_64", "graviton")
#: Reference kinds that are not built by build_anchors.sh (NVENC is the GPU's).
NOT_BUILT = {"nvenc"}
RECORD_KEYS = ("ref", "commit", "compile_flags", "lto", "pgo")


def build_tool(program: str) -> str:
    return ANCHOR_BUILD_TOOLS.get(program, program)


def recorded() -> dict:
    return json.loads(FLAGS.read_text())["builds"]


def test_every_reference_family_has_a_record_on_every_timed_machine():
    programs = {ANCHOR_TOOLS[k] for k in ANCHOR_TOOLS if k not in NOT_BUILT}
    programs.add("openh264-harness")
    builds = recorded()
    for machine in OPT_MACHINES:
        tools = builds[machine]["tools"]
        missing = sorted(p for p in programs if build_tool(p) not in tools)
        assert not missing, f"{machine}: no build record for {missing}"
        for name, record in tools.items():
            # `configure` is empty for a plain-Makefile project (openh264)
            assert "configure" in record, (machine, name)
            assert all(record.get(k) for k in RECORD_KEYS), (machine, name)


def test_every_timed_reference_program_is_recorded():
    """The timing plan record's reference programs (argv[0]) all have a record."""
    plan = json.loads(PLAN.read_text())
    programs = {point["example"]["argv"][0].removeprefix("{BIN}/")
                for name, point in plan["points"].items() if not name.startswith("nanocodec")}
    for machine in OPT_MACHINES:
        tools = recorded()[machine]["tools"]
        assert {build_tool(p) for p in programs} <= set(tools), machine


def test_records_are_the_opt_builds_with_lto_or_a_reason():
    for machine in OPT_MACHINES:
        build = recorded()[machine]
        assert build["profile"] == "opt"
        for name, record in build["tools"].items():
            lto_on = any(flag.startswith("-flto") for flag in record["compile_flags"])
            assert lto_on or record["lto"].startswith("no ("), (machine, name, record["lto"])


def test_reference_builds_switch_in_one_place():
    assert set(OPT_MACHINES) | {"jetson"} == set(REFERENCE_BUILDS)
    assert reference_bin("x86_64", Path("/d")) == Path("/d/anchors/x86_64-opt/bin")
