"""Flicker calibration (nceval.flicker_reference): window and frame flicker of the real-use anchors and our arms on given clips.

usage: python3 tools/dev/flicker_calibrate.py BINARY CLIPS FRAMES [OUT.json]
Arms: SVT-AV1 M12 rtc, x265 superfast, x264 veryfast (zerolatency), our VVC/HEVC Default and the
candidate (start-up cadence + motion gate). Prints per clip the ladder-mean window and frame
flicker and the matched-rate window-flicker delta of each arm against SVT-AV1 rtc.
"""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import jreal, realuse
from nceval.cbr import Family
from nceval.codecs import Codec
from nceval.config import load_paths
from nceval.gatepolicy import EVALUATION_GATE

CANDIDATE = ("rate.cadence_startup_period=8", "rate.cadence_startup_base_boost=6",
             "rate.cadence_motion_gate=1")

binary, clip_spec, frames, *out = sys.argv[1:]
paths = load_paths()
anchors = {f.name: f for f in realuse.anchor_families(paths, realuse.OPENH264_HARNESS,
                                                       extra=["x264-veryfast", "x265-superfast"])}
families = [anchors["svtav1-m12"], anchors["x265-superfast"], anchors["x264-veryfast"]]
for codec in (Codec.VVC, Codec.HEVC):
    families.append(Family(f"nanocodec-{codec.value}", codec, Path(binary), (), own=True, cqp=False))
    families.append(Family(f"nanocodec-{codec.value}@candidate", codec, Path(binary),
                           realuse.param_args(list(CANDIDATE)), own=True, cqp=False))
clips = realuse.screen_clips(clip_spec, paths)
runner = realuse.Runner(paths, EVALUATION_GATE, 4, False)
curves = runner.measure("flicker calibration", families, clips, int(frames))
rows = {}
print(f"{'clip':26s} {'arm':28s} {'kbps':>14s} {'wflk':>6s} {'fflk':>6s} {'d_wflk vs svt':>13s}")
for clip in clips:
    svt = curves[("svtav1-m12", clip.name)]
    for f in families:
        pts = curves[(f.name, clip.name)]
        if not pts:
            continue
        w = statistics.fmean(p["flicker_psnr_y"] for p in pts)
        fr = statistics.fmean(p["flicker_frame_psnr_y"] for p in pts)
        d = jreal.matched_delta(svt, pts, lambda p: p["flicker_psnr_y"]) if f.name != "svtav1-m12" else 0.0
        rates = f"{min(p['kbps'] for p in pts):.0f}-{max(p['kbps'] for p in pts):.0f}"
        rows.setdefault(f.name, {})[clip.name] = {"wflk": w, "fflk": fr, "d_wflk_vs_svt": d}
        dtxt = "n/a" if d is None else f"{d:+.3f}"
        print(f"{clip.name:26s} {f.name:28s} {rates:>14s} {w:6.3f} {fr:6.3f} {dtxt:>13s}", flush=True)
print("\nmean over clips (wflk / fflk / d_wflk vs svt):")
for name, per in rows.items():
    ds = [v["d_wflk_vs_svt"] for v in per.values() if v["d_wflk_vs_svt"] is not None]
    print(f"  {name:28s} {statistics.fmean(v['wflk'] for v in per.values()):.3f} "
          f"{statistics.fmean(v['fflk'] for v in per.values()):.3f} "
          f"{statistics.fmean(ds) if ds else float('nan'):+.3f}")
if out:
    Path(out[0]).write_text(json.dumps(rows, indent=1))
