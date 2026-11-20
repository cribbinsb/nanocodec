"""tools/dev/pulsecheck.py: the two arms are built independently from the preset."""
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from nceval.cbr import Family

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "dev" / "pulsecheck.py"
spec = importlib.util.spec_from_file_location("pulsecheck", SCRIPT)
pulsecheck = importlib.util.module_from_spec(spec)
sys.modules.setdefault("pulsecheck", pulsecheck)
spec.loader.exec_module(pulsecheck)

PRESET = ("--param", "vvc.preset_value=1")


def family(name, binary):
    """own_family's shape without the preset files: one preset value."""
    return Family(name, "vvc", binary, PRESET, own=True, cqp=False)


def test_an_anchor_param_alone_leaves_the_candidate_on_the_preset():
    anchor, candidate = pulsecheck.arms("nanocodec-vvc", Path("bin"), None, [],
                                        ["vvc.tx64=0"], family=family)
    assert anchor.extra == PRESET + ("--param", "vvc.tx64=0")
    assert candidate.extra == PRESET
    assert candidate.name == "nanocodec-vvc" + pulsecheck.CANDIDATE_SUFFIX


def test_the_candidate_never_carries_the_anchor_params():
    anchor, candidate = pulsecheck.arms("nanocodec-vvc", Path("bin"), None, ["vvc.tx64=1"],
                                        ["vvc.tx64=0", "vvc.other=2"], family=family)
    assert "vvc.other=2" not in candidate.extra and "vvc.tx64=0" not in candidate.extra
    assert candidate.extra == PRESET + ("--param", "vvc.tx64=1")


def test_an_anchor_binary_is_its_own_arm():
    anchor, candidate = pulsecheck.arms("nanocodec-vvc", Path("new"), Path("old"), [], [],
                                        family=family)
    assert (anchor.tool, candidate.tool) == (Path("old"), Path("new"))
    assert anchor.extra == candidate.extra == PRESET


def test_identical_arms_are_refused():
    with pytest.raises(SystemExit, match="same encoder"):
        pulsecheck.arms("nanocodec-vvc", Path("bin"), None, [], [], family=family)
    with pytest.raises(SystemExit, match="same encoder"):
        pulsecheck.arms("nanocodec-vvc", Path("bin"), Path("bin"), ["a=1"], ["a=1"],
                        family=family)


@dataclass
class Clip:
    name: str
    content_class: str


def test_the_draw_takes_six_of_each_kind_deterministically():
    clips = [Clip(f"s{i}", "static") for i in range(10)] + [Clip(f"m{i}", "crowd")
                                                            for i in range(10)]
    chosen = pulsecheck.draw(clips)
    assert [c.content_class == "static" for c in chosen] == [True] * 6 + [False] * 6
    assert chosen == pulsecheck.draw(list(reversed(clips)))


def test_means_skip_missing_terms():
    rows = {"a": {"bd_psnr_yuv": 1.0, "bd_vmaf_v1": None}, "b": {"bd_psnr_yuv": 2.0}}
    summary = pulsecheck.mean_terms(rows)
    assert summary["bd_psnr_yuv"] == (1.5, 2)
    assert summary["bd_vmaf_v1"] is None
