"""The standards lens (nceval.ctc): sealing, job plans, the worker's parsers, the BD tables."""

from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path

import pytest

from nceval import pool
from nceval.config import Paths
from nceval.ctc import conditions, local, plan, report, sequences, worker

FAKE_FFMPEG = "#!/bin/sh\necho 'ffmpeg version n8.0.1'\n"


@pytest.fixture
def fake_paths(tmp_path: Path) -> Paths:
    """Every tool a plan names, as a stub (plans never run them)."""
    bin_dir = tmp_path / "anchors" / "bin"
    bin_dir.mkdir(parents=True)
    names = {"nanocodec", "x264", "x265", "kvazaar", "SvtAv1EncApp", "aomenc", "vpxenc",
             "openh264-harness", "ffmpeg-decode", "ldecod", "TAppDecoder", "vvdecapp", "aomdec",
             "dav1d", "vpxdec", "avmdec", "dav2d"}
    for name in names:
        tool = bin_dir / name
        tool.write_text(FAKE_FFMPEG)
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return Paths(data_root=tmp_path, anchors=tmp_path / "anchors", nanocodec_bin=bin_dir)


def test_every_sequence_family_is_sealed_from_the_pool():
    assert sequences.KRISTEN_AND_SARA_FAMILY in pool.SEALED_FAMILIES
    assert {s.family for s in sequences.ALL} <= pool.SEALED_FAMILIES


def test_sequences_are_8bit_and_licensed():
    for s in sequences.ALL:
        assert "10bit" not in s.name and s.licence.notice_url.startswith("https://")
    assert all(s.md5 and len(s.md5) == 32 for s in sequences.AOM_LD)


def test_plan_names_only_run_directory_tools_and_placeholders(fake_paths):
    jobs = plan.plan(list(conditions.CONDITIONS), fake_paths)
    assert jobs and all(j["encode"][0].startswith("bin/") for j in jobs)
    own = [j for j in jobs if j["own"]]
    assert all("{RECON}" in j["encode"] and "{DISPLAY}" in j["encode"] for j in own)
    assert all(len(j["decoders"]) == 2 for j in own)
    cbr = [j for j in jobs if j["kind"] == "cbr"]
    ids = {j["id"] for j in jobs}
    assert cbr and all("{KBPS}" in j["encode"] and j["rate_of"] in ids for j in cbr)
    assert not any(str(fake_paths.anchors) in a for j in jobs for a in j["encode"])


def test_slow_reference_encoders_run_on_small_classes_only(fake_paths):
    jobs = plan.plan(["aom-ld"], fake_paths, encoders=["vtm-ldp", "aom-good-s0-ld"])
    classes = {(j["encoder"], sequences.by_name(j["sequence"]).sequence_class) for j in jobs}
    assert classes == {("vtm-ldp", "A5"), ("aom-good-s0-ld", "A4"), ("aom-good-s0-ld", "A5")}
    assert not plan.plan(["jvet-e"], fake_paths, encoders=["vtm-ldp"])


def test_each_scale_has_six_points_and_the_jvet_four():
    assert all(len(points) == 6 for points in conditions.POINTS.values())
    assert conditions.JVET_CTC_QPS == (22, 27, 32, 37)


def test_ivf_payload_excludes_file_and_frame_headers(tmp_path):
    stream = tmp_path / "s.ivf"
    frames = [b"a" * 10, b"b" * 25]
    stream.write_bytes(b"\0" * worker.IVF_FILE_HEADER + b"".join(
        len(f).to_bytes(4, "little") + b"\0" * 8 + f for f in frames))
    assert worker.payload_bytes(stream, "ivf") == (35, 2)
    assert worker.payload_bytes(stream, "annexb") == (stream.stat().st_size, 0)


def test_shards_partition_the_jobs_and_keep_cbr_twins_together(fake_paths):
    jobs = plan.plan(["aom-ld"], fake_paths)
    shards = [plan.shard(jobs, i, 3) for i in (1, 2, 3)]
    assert sorted(j["id"] for s in shards for j in s) == sorted(j["id"] for j in jobs)
    for part in shards:
        ids = {j["id"] for j in part}
        assert all(j["rate_of"] in ids for j in part if j["kind"] == "cbr")


def test_y4m_frames_become_raw_i420(tmp_path):
    frame_bytes = 4 * 2 * 3 // 2
    y4m = tmp_path / "s.y4m"
    y4m.write_bytes(b"YUV4MPEG2 W4 H2 F30:1 C420jpeg\n" + b"".join(
        b"FRAME\n" + bytes([i]) * frame_bytes for i in range(3)))
    raw = tmp_path / "s.yuv"
    assert worker.y4m_to_raw(y4m, raw, 2, frame_bytes) == 2
    assert raw.read_bytes() == bytes([0]) * frame_bytes + bytes([1]) * frame_bytes


def _record(job_id: str, kbps: float, psnr: float) -> dict:
    means = {m: psnr for m in worker.MEAN_METRICS}
    return {"id": job_id, "kbps": kbps, "user_s": 1.0, "means": means,
            "overall_psnr": {"apsnr_y": psnr, "apsnr_cb": psnr, "apsnr_cr": psnr}}


def test_report_halving_the_rate_at_equal_quality_is_minus_fifty(tmp_path):
    seq = "FourPeople_480x270_60"
    for i, q in enumerate((22, 27, 32, 37)):
        psnr = 40.0 - 3 * i
        for encoder, kbps in (("hm-ldp", 800 / 2 ** i), ("x264-medium", 400 / 2 ** i)):
            path = tmp_path / "aom-ld" / seq / encoder / f"q{q}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_record(f"aom-ld/{seq}/{encoder}/q{q}", kbps, psnr)))
    summary = report.report([tmp_path], tmp_path / "out")
    rows = {(r["anchor"], r["test"], r["metric"]): r for r in summary["rows"]}
    assert rows[("hm-ldp", "x264-medium", "Y")]["bd"] == pytest.approx(-50.0)
    assert rows[("hm-ldp", "x264-medium", "Weighted")]["bd"] == pytest.approx(-50.0)
    assert (tmp_path / "out" / "bd.csv").exists()


def _local_args(**overrides) -> argparse.Namespace:
    values = {"conditions": "aom-ld", "sequences": "FourPeople_480x270_60", "kinds": "qp,cbr",
              "encoder": "nanocodec-av1", "param": ["refs.second=1"], "label": "two",
              "x264": "qp"}
    return argparse.Namespace(**{**values, **overrides})


def test_local_variant_is_renamed_not_decoded_and_carries_its_params(fake_paths, monkeypatch):
    monkeypatch.setattr(local, "load_paths", lambda: fake_paths)
    jobs = local.local_jobs(_local_args(), Path("/src"))
    assert len(jobs) == 12 and all(j["encoder"] == "nanocodec-av1+two" for j in jobs)
    assert all(j["decoders"] == [] and j["encode"][-2:] == ["--param", "refs.second=1"]
               for j in jobs)
    assert all(j["bitstream"] in j["encode"] and "+two" in j["bitstream"] for j in jobs)
    ids = {j["id"] for j in jobs}
    assert all(j["rate_of"] in ids for j in jobs if j["kind"] == "cbr")


def test_local_x264_cbr_twin_uses_the_charts_vbv(fake_paths, monkeypatch):
    monkeypatch.setattr(local, "load_paths", lambda: fake_paths)
    jobs = local.local_jobs(_local_args(encoder="x264-medium", x264="psnr-cbr"), Path("/src"))
    cbr = [j for j in jobs if j["kind"] == "cbr"]
    assert len(cbr) == 6 and len(jobs) == 12
    for job in cbr:
        encode = job["encode"]
        assert "--qp" not in encode and encode.count("{KBPS}") == 3
        assert encode[encode.index("--tune") + 1] == "zerolatency,psnr"
        assert job["bitstream"] in encode and job["rate_of"] in {j["id"] for j in jobs}


def test_lean_points_have_the_weighted_and_j_figures(tmp_path):
    record = _record("aom-ld/FourPeople_480x270_60/x/q22", 100.0, 40.0)
    record["means"] = {k: record["means"][k] for k in ("psnr_y", "psnr_cb", "psnr_cr",
                                                       "float_ms_ssim", "vmaf_v1")}
    quality = report.quality_of(record)
    assert {"Y", "U", "V", "PSNR-YUV", "MS-SSIM", "VMAF-v1"} <= set(quality)
    assert "VMAF" not in quality
