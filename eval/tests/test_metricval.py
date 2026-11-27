"""nceval.metricval: sample choice, stored-result matching, and the comparison arithmetic."""
import json
from pathlib import Path

from nceval import metricval
from nceval.clips import Clip


def clip(name, height, content_class, width=None):
    return Clip(name, Path(f"/clips/{name}.yuv"), width or height * 16 // 9, height, 30.0, 150,
                1000.0, content_class, "val")


def test_bd_clips_hold_every_bd_family_and_spread_covers_cells():
    clips = [clip("a720", 720, "pan"), clip("b720", 720, "static"), clip("c240", 240, "crowd"),
             clip("d540", 540, "pan"), clip("portrait", 1280, "pan", width=720)]
    bd_families = [f for pair in metricval.BD_PAIRS for f in pair]
    stored = {(c.name, f): [40, 75, 150, 300] for c in clips for f in bd_families}
    stored.pop(("b720", "x264-medium"))  # b720 cannot carry the BD set
    stored.update({(c.name, "nanocodec-vp9"): [10, 20, 30] for c in clips})
    picks = metricval.choose(clips, stored, bd_clips=2, spread=3)
    bd = {p[0] for p in picks if p[3] == "bd"}
    assert len(bd) == 2 and "b720" not in bd
    assert sum(p[3] == "bd" for p in picks) == 2 * len(bd_families) * 4
    spread = [p for p in picks if p[3] == "spread"]
    assert len(spread) == 3 and not {p[0] for p in spread} & bd
    assert [p[2] for p in spread if p[1] == "nanocodec-vp9"][:1] == [10]  # first: lowest rung


def test_portrait_clips_bin_by_their_shorter_side():
    assert metricval.size_bin(clip("p", 1280, "x", width=720)) == \
        metricval.size_bin(clip("l", 720, "x"))


def test_stored_result_is_the_one_scored_from_the_stream(tmp_path):
    (tmp_path / "cbr40-aaaa.json").write_text(json.dumps({"frame_bits": [100, 20], "kbps": 1}))
    (tmp_path / "cbr40-bbbb.json").write_text(json.dumps({"frame_bits": [100, 30], "kbps": 2}))
    assert metricval.stored_result([tmp_path], "cbr40", 130)["kbps"] == 2
    assert metricval.stored_result([tmp_path], "cbr40", 999) is None


def frames(vmaf, psnr, lib=None):
    rows = []
    for i, v in enumerate(vmaf):
        row = {"vmaf_v1": v, "ms_ssim": 0.99, "psnr_y": psnr[i], "psnr_u": 50.0, "psnr_v": 50.0}
        if lib is not None:
            row.update({"lib_psnr_y": lib[i], "lib_psnr_u": 50.0, "lib_psnr_v": 50.0})
        rows.append(row)
    return rows


def scored(vmaf, psnr, lib=None, reference="r"):
    rows = frames(vmaf, psnr, lib)
    summary = {"vmaf_v1": sum(vmaf) / len(vmaf), "ms_ssim": 0.99, "reference_sha256": reference,
               **metricval.plane_summary(rows)}
    if lib is not None:
        lib_rows = [{p: r[f"lib_{p}"] for p in metricval.PLANES} for r in rows]
        summary.update({f"lib_{k}": v for k, v in metricval.plane_summary(lib_rows).items()})
    return {"summary": summary, "frames": rows}


def test_compare_point_reports_per_frame_and_sequence_differences():
    point = metricval.SampledPoint("clipsets/x.json", "c", "nanocodec-av1", "av1", True, 40,
                                   720, "pan", sets=("bd",), coded_kbps=41.0)
    cpu = scored([80.0, 70.0], [40.0, 61.0], lib=[40.0, 60.0])
    cpu["encoder"] = {"stream_identical": True, **scored([0, 0], [40.0, 60.0])}
    gpu = scored([80.0, 70.5], [0, 0])
    row = metricval.compare_point(point, cpu, gpu, None)
    assert row["same_reference"] and row["kbps"] == 41.0
    assert row["vmaf_frame_max_abs"] == 0.5 and abs(row["vmaf_seq_diff"] - 0.25) < 1e-12
    # numpy's 61 dB frame is capped at 60, as libvmaf caps it
    assert row["numpy_vs_libvmaf_psnr_y_max"] == 0.0
    assert row["encoder_vs_libvmaf_psnr_y_max"] == 0.0 and row["encoder_stream_identical"]


def test_bd_vmaf_scores_each_pair_with_each_scorer():
    rows = []
    for family, scale in (("nanocodec-av1", 0.8), ("svtav1-m12", 1.0)):
        for kbps, vmaf in ((40, 50.0), (75, 60.0), (150, 70.0), (300, 80.0)):
            rows.append({"clip": "c", "family": family, "kbps_target": kbps, "sets": "bd",
                         "kbps": kbps * scale, "vmaf_cpu": vmaf, "vmaf_gpu": vmaf})
    result = {r["test"]: r for r in metricval.bd_vmaf(rows, "cpu")}
    assert abs(result["nanocodec-av1"]["bd_vmaf"] - -20.0) < 1e-6
