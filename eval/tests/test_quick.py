"""Stage 1's verdict and probe draw (nceval.quick)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import checkrun, pool, quick
from nceval.clips import Clip


def test_quick_verdict_kills_on_the_probe_pulse_and_class():
    l1 = {"decision": "continue", "mean": 0.0, "lo": -0.5, "hi": 0.5, "n": 24,
          "confidence": 0.9}
    probe = {"mean": 0.2, "lo": -0.4, "hi": 0.8, "n": 6, "confidence": 0.9, "pulse": 0.1,
             "pulse_max": 0.3, "classes": {"static": 1.0}}
    assert quick.verdict(l1, {150: probe, 300: probe}) == ("CONTINUE", [])
    decision, reasons = quick.verdict(l1, {150: probe, 300: {**probe, "pulse": 0.4,
                                                             "classes": {"lowlight": 7.0}}})
    assert decision == "KILL" and len(reasons) == 2
    assert all(r.startswith("1/300: ") for r in reasons)  # a loss at either share kills


def test_overlay_names_check_against_their_registered_parameter():
    # light.* (light superblocks) is an overlay like nonbase.* and key.*: tuned fast-rung
    # points set it, and stage 1 refused them as unknown until 2026-11-23
    assert checkrun.base_name("light.planner.lambda_scale") == "planner.lambda_scale"
    assert checkrun.base_name("nonbase.vp9.shave_gap") == "vp9.shave_gap"
    assert checkrun.base_name("key.h264.trellis_i8x8") == "h264.trellis_i8x8"
    assert checkrun.base_name("quant.trellis") == "quant.trellis"


def test_probe_worst_of_shares_feeds_the_queue():
    share = lambda pulse, cls: {"pulse": pulse, "pulse_max": pulse, "classes": cls}
    worst = quick.worst_of_shares({150: share(0.1, {"static": 1.0, "crowd": -1.0}),
                                   300: share(0.3, {"static": -2.0})})
    assert worst == {"pulse": 0.3, "pulse_max": 0.3, "classes": {"crowd": -1.0, "static": 1.0}}


def clip(name, cls, frames=900):
    return Clip(name=name, yuv=Path(f"/x/{name}.yuv"), width=64, height=64, fps=30.0,
                frames=frames, source_kbps=1e4, content_class=cls, split="train",
                family=name, start_frame=17)


def test_probe_takes_one_long_clip_per_target_class_from_l2_first():
    adopt = [clip("s1", "static"), clip("s2", "static"), clip("l1", "lowlight"),
             clip("short", "bodycam", frames=300), clip("c1", "crowd"), clip("c2", "crowd")]
    draw = pool.Draw("k", 150, {pool.QUICK: [clip("b1", "bodycam")], pool.ADOPT: adopt,
                                pool.FINAL: [clip("d1", "drone")]})
    chosen = quick.probe_clips(draw, 600)
    classes = [c.content_class for c in chosen]
    assert classes[:3] == ["static", "lowlight", "bodycam"]  # bodycam from L1: L2's is short
    assert "drone" not in classes  # never from L3, which the bundle's final check measures
    assert len(chosen) == quick.PROBE_CLIPS and all(c.start_frame == 0 for c in chosen)
    assert quick.probe_clips(draw, 600) == chosen


def test_probes_run_only_when_l1_does_not_kill():
    """Owner, 2026-11-25: the probes are ~83 % of a quick's frames; an L1 KILL is a KILL
    whatever they show, so they are skipped; a candidate passing L1 gets every key share."""
    calls = []
    probe = lambda period: (lambda: calls.append(period) or {"c": period})
    l1 = lambda decision: (lambda: {"levels": {pool.QUICK: {"decision": decision}}})
    _, rows = quick.staged(l1("worse"), {150: probe(150), 300: probe(300)})
    assert rows == {} and calls == []
    _, rows = quick.staged(l1("neutral"), {150: probe(150), 300: probe(300)})
    assert rows == {150: {"c": 150}, 300: {"c": 300}} and sorted(calls) == [150, 300]
