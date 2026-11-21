"""Stage 1's verdict and probe draw (nceval.quick)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import pool, quick
from nceval.clips import Clip


def test_quick_verdict_kills_on_the_probe_pulse_and_class():
    l1 = {"decision": "continue", "mean": 0.0, "lo": -0.5, "hi": 0.5, "n": 24,
          "confidence": 0.9}
    probe = {"mean": 0.2, "lo": -0.4, "hi": 0.8, "n": 6, "confidence": 0.9, "pulse": 0.1,
             "pulse_max": 0.3, "classes": {"static": 1.0}}
    assert quick.verdict(l1, probe) == ("CONTINUE", [])
    decision, reasons = quick.verdict(l1, {**probe, "pulse": 0.4,
                                           "classes": {"lowlight": 7.0}})
    assert decision == "KILL" and len(reasons) == 2


def clip(name, cls, frames=900):
    return Clip(name=name, yuv=Path(f"/x/{name}.yuv"), width=64, height=64, fps=30.0,
                frames=frames, source_kbps=1e4, content_class=cls, split="train",
                family=name, start_frame=17)


def test_probe_takes_one_long_clip_per_target_class_from_l2_first():
    adopt = [clip("s1", "static"), clip("s2", "static"), clip("l1", "lowlight"),
             clip("short", "bodycam", frames=300), clip("c1", "crowd"), clip("c2", "crowd")]
    draw = pool.Draw("k", 150, {pool.QUICK: [clip("b1", "bodycam")], pool.ADOPT: adopt,
                                pool.FINAL: [clip("d1", "drone")]})
    chosen = quick.probe_clips(draw)
    classes = [c.content_class for c in chosen]
    assert classes[:3] == ["static", "lowlight", "bodycam"]  # bodycam from L1: L2's is short
    assert "drone" not in classes  # never from L3, which the bundle's final check measures
    assert len(chosen) == quick.PROBE_CLIPS and all(c.start_frame == 0 for c in chosen)
    assert quick.probe_clips(draw) == chosen
