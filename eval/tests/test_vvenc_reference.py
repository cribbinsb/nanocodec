"""VVenC as a fixed-QP reference: its tool switches and picture structures (nceval.cbr),
and its commands as tools/dev/fixedqp_timing.py times them (no encodes run)."""
import importlib.util
from pathlib import Path

from nceval.cbr import preset_anchor
from nceval.clips import Clip

TOOL = Path(__file__).resolve().parents[2] / "tools" / "dev" / "fixedqp_timing.py"
spec = importlib.util.spec_from_file_location("fixedqp_timing", TOOL)
fixedqp_timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixedqp_timing)

PREFIX = Path("/anchors")
CLIP = Clip("c", Path("/IN.yuv"), 640, 360, 30, 300, 2000, "static", "val")


def family(name):
    return preset_anchor(name, None, cqp=True,
                         tool=lambda kind, paths: PREFIX / "bin" / "vvencFFapp")


def command(name):
    return family(name).command(CLIP, 150, 300, Path("/OUT.bin"), None, qp=37)


def configs(argv):
    """The files given to VVenC's `-c` (after `sh -c SCRIPT`, which feeds its input)."""
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-c" and i > 1]


def test_the_preset_reads_its_low_delay_file_alone():
    assert configs(command("vvenc-ld-fast")) == [
        "/anchors/share/vvenc/cfg/experimental/lowdelay_fast.cfg"]


def test_switches_follow_the_configuration_files():
    argv = command("vvenc-ld-faster~flat+MaxMTTDepth=1+BIM=0")
    assert configs(argv) == ["/anchors/share/vvenc/cfg/experimental/lowdelay_faster.cfg",
                             "/anchors/share/vvenc/cfg/nanocodec/structure-flat.cfg"]
    last_config = max(i for i, a in enumerate(argv) if a == "-c" and i > 1)
    assert argv.index("--MaxMTTDepth=1") > last_config
    assert argv.index("--BIM=0") > last_config


def test_a_structure_file_is_staged_with_the_program():
    assert family("vvenc-ld-faster~flat4").program_files[-1] == \
        PREFIX / "share/vvenc/cfg/nanocodec/structure-flat4.cfg"


def test_timing_names_the_program_and_its_files_by_placeholder():
    f = family("vvenc-ld-faster~flat+BIM=0")
    argv, tools, cfgs = fixedqp_timing.placeholder_argv(command(f.name), f)
    assert argv[:5] == ["sh", "-c", argv[2], "{INPUT}", "{BIN}/vvencFFapp"]
    assert configs(argv) == ["{CFG}/lowdelay_faster.cfg", "{CFG}/structure-flat.cfg"]
    assert "{OUTPUT}" in argv
    assert set(tools) == {"vvencFFapp"}
    assert set(cfgs) == {"lowdelay_faster.cfg", "structure-flat.cfg"}


def test_a_non_standard_rate_is_given_as_the_nearest_standard_one():
    from nceval.cbr import vvenc_frame_rate
    assert vvenc_frame_rate(29.97) == (30000, 1001)
    assert vvenc_frame_rate(29.68) == (30, 1)
    assert vvenc_frame_rate(25) == (25, 1)
