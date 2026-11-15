"""tools/dev/anchor_chart_timing.py: the timing plan's encoder arguments (no encodes run)."""
import importlib.util
from pathlib import Path

from nceval.cbr import Codec, Family
from nceval.encoders.nanocodec import PRESETS_DIR

TOOL = Path(__file__).resolve().parents[2] / "tools" / "dev" / "anchor_chart_timing.py"
spec = importlib.util.spec_from_file_location("anchor_chart_timing", TOOL)
anchor_chart_timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(anchor_chart_timing)

FROZEN = ("--param", "planner.depth_removal_min_quantizer=0")


def test_our_preset_family_is_planned_by_preset_name_not_frozen_params():
    family = Family("nanocodec-vp9@fast", Codec.VP9, Path("nanocodec"), FROZEN, own=True)
    assert anchor_chart_timing.plan_family(family, PRESETS_DIR).extra == ("--preset", "fast")


def test_a_layered_preset_is_planned_by_scalability_and_speed():
    family = Family("nanocodec-av1@l2-fast", Codec.AV1, Path("nanocodec"), FROZEN, own=True)
    assert anchor_chart_timing.plan_family(family, PRESETS_DIR).extra == (
        "--scalability", "L2T1", "--preset", "fast")


def test_a_frozen_pre_rename_build_gets_its_old_preset_name(tmp_path):
    (tmp_path / "vp9-fastest.toml").write_text("x = 1\n")
    family = Family("nanocodec-vp9@veryfast", Codec.VP9, Path("nanocodec"), FROZEN, own=True)
    assert anchor_chart_timing.plan_family(family, tmp_path).extra == ("--preset", "vp9-fastest")


def test_our_default_family_takes_the_builds_defaults():
    family = Family("nanocodec-vp9", Codec.VP9, Path("nanocodec"), FROZEN, own=True)
    assert anchor_chart_timing.plan_family(family, PRESETS_DIR).extra == ()


def test_anchors_keep_their_arguments():
    family = Family("svtav1-m13", Codec.AV1, Path("SvtAv1EncApp"), ("--x",), kind="svt")
    assert anchor_chart_timing.plan_family(family, PRESETS_DIR) is family
