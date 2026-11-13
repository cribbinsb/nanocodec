"""Scoreboard: Jetson log, cached instruction counts, page and STATUS section."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import scoreboard, tune
from nceval.config import Paths
from nceval.report import EncoderSummary


def paths(tmp_path) -> Paths:
    return Paths(tmp_path / "data", tmp_path / "anchors", tmp_path / "bin", tmp_path / "work")


def test_latest_jetson_batch_per_codec_and_preset(tmp_path):
    p = paths(tmp_path)
    log = scoreboard.jetson_file(p)
    log.parent.mkdir(parents=True)
    record = lambda codec, preset, ratio, sha: json.dumps(
        {"date": "2026-11-08", "codec": codec, "preset": preset, "ratio": ratio,
         "binary_sha256": sha, "anchor": "svtav1-m12-rtc"})
    log.write_text("\n".join([
        record("hevc", "hevc-fast", 2.0, "old"),
        record("hevc", "hevc-fast", 1.0, "new"), record("hevc", "Fast", 4.0, "new"),
        "not json",
        record("h264", "default", 0.9, "x")]))
    latest = scoreboard.latest_jetson(p)
    assert latest[("hevc", "fast", "svtav1-m12")]["ratio"] == 2.0  # the new batch only
    assert latest[("hevc", "fast", "svtav1-m12")]["clips"] == 2
    assert latest[("h264", "default", "svtav1-m12")]["ratio"] == 0.9


def test_instruction_counts_are_cached_by_binary_and_params(tmp_path):
    p = paths(tmp_path)
    binary = tmp_path / "nanocodec"
    binary.write_bytes(b"build one")
    row = next(r for r in scoreboard.ROWS if r.encoder == "nanocodec-hevc@hevc-fast")  # overrides
    with patch.object(scoreboard, "find_tool", return_value=binary), \
         patch.object(tune, "instruction_cost", return_value=tune.InstructionCost(12.5, 80.0)) as cost:
        assert scoreboard.instruction_cost(row, p).instructions.p == 12.5
        assert scoreboard.instruction_cost(row, p).instructions.key == 80.0
        assert cost.call_count == 1
        binary.write_bytes(b"build two")
        older = scoreboard.instruction_cost(row, p, measure_new=False)
        assert cost.call_count == 1 and not older.current  # --no-measure: older count, marked
        assert older.instructions.p == 12.5
        scoreboard.instruction_cost(row, p)
        assert cost.call_count == 2  # a new binary is measured again


def test_page_and_status_section(tmp_path):
    summary = EncoderSummary({"crowd": -3.0, "static": 1.0}, -1.0, 2, 0)
    entry = {"metrics": {"PSNR": summary, "VMAF": summary},
             "clips": {"a": {"PSNR": -3.0, "VMAF": -2.0}, "b": {"PSNR": 2.0, "VMAF": 4.0}}}
    tables = {scoreboard.ROWS[0]: entry}
    costs = {scoreboard.ROWS[0]: scoreboard.Cost(tune.InstructionCost(10.0, 50.0), True)}
    jetson = {("h264", "default", "openh264-baseline"): {"ratio": 1.1, "clips": 4, "date": "2026-11-08"}}
    page = scoreboard.render(tables, costs, jetson, ["crowd", "static"], "2026-11-08 18:00")
    av2 = next(r for r in scoreboard.ROWS if r.encoder == "nanocodec-av2")
    pending = scoreboard.render({av2: entry}, {}, {}, ["crowd", "static"], "x",
                                {av2.encoder: "being rebuilt"})
    assert "| av2 | Default | nanocodec-av1 | pending: being rebuilt |" in pending
    assert "| h264 | Default | openh264-baseline | -1.0% | -1.0% | -3.0% / -3.0% |" in page
    assert "10.0 / 50.0 | 1.10x (4 clips, 2026-11-08) | b +3.0% |" in page
    status = tmp_path / "STATUS.md"
    status.write_text("# Status\n\nintro\n")
    scoreboard.rewrite_status(status, page)
    scoreboard.rewrite_status(status, page.replace("-1.0%", "-2.0%"))
    text = status.read_text()
    assert text.count(scoreboard.STATUS_BEGIN) == 1 and "-2.0%" in text and "intro" in text


def test_unchanged_canonical_probe_rekeys_curves_instead_of_encoding(tmp_path):
    import platform as host
    from nceval.clips import Clip, ClipSet
    from nceval.results import EncodeRecord, ResultsDb
    from nceval.framestats import FrameInfo
    from nceval.sweep import SweepConfig
    p = paths(tmp_path)
    db = ResultsDb(tmp_path / "r.sqlite")
    cfg = SweepConfig(10, max_frames=150)
    clip = Clip("static_cam29", tmp_path / "c.yuv", 16, 16, 30, 300, 2000, "static", "val")
    clips = ClipSet("val", (clip,))
    row = scoreboard.ROWS[0]
    for q in (26, 30, 34, 38):
        db.store(EncodeRecord(row.encoder, cfg.structure_tag, clip.name, q, 150, 10, 100, True,
                              "", host.machine(), "old-binary", "c"), {"psnr_yuv": 40.0},
                 [FrameInfo(10, "P")] * 150)
    fingerprint = {"value": "old-binary"}
    sweep = SimpleNamespace(db=db, platform=host.machine(),
                            fingerprint=lambda encoder, cfg, clip: fingerprint["value"])
    scoreboard.save_row_state(row, "sig", clips, sweep, cfg, p)
    fingerprint["value"] = "new-binary"
    assert scoreboard.reuse_identical(row, "other-sig", clips, sweep, cfg, p) == 0
    assert scoreboard.reuse_identical(row, "sig", clips, sweep, cfg, p) == 1
    assert db.curve_is_current(row.encoder, cfg.structure_tag, clip.name, 150, host.machine(),
                               "new-binary")
    db.conn.close()


def test_context_table_and_anchor_costs_render():
    context = {scoreboard.ROWS[0]: {a: {"PSNR": -5.0, "VMAF": 2.5} for a in scoreboard.CONTEXT_ANCHORS}}
    jetson = {("h264", "default", "x264-ultrafast"): {"ratio": 1.4}}
    text = scoreboard.render_context(context, {"x264-veryfast": tune.InstructionCost(30.0, 90.0)},
                                     jetson)
    assert "| h264 | Default | -5.0% / +2.5%; Jetson 1.40x | -5.0% / +2.5% |" in text
    assert "| x264-veryfast | 30.0 / 90.0 |" in text
