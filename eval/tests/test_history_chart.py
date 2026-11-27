"""tools/dev/history_chart.py: stored frames read through the rung renames (no encodes run)."""
import importlib.util
import json

import pytest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "tools" / "dev" / "history_chart.py"
spec = importlib.util.spec_from_file_location("history_chart", TOOL)
history_chart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history_chart)


def test_stored_keys_take_todays_rung_names():
    rename = lambda key: history_chart.renamed_key(key, history_chart.RUNG_RENAMES)  # noqa: E731
    # 0.5.0's slowest rung is today's slower, veryslow on HEVC and VVC
    assert [rename(k) for k in ("av1@slow", "hevc@slow", "vvc@slow")] == \
        ["av1@slower", "hevc@veryslow", "vvc@veryslow"]
    # H.264's and VVC's old fast is today's faster; the old H.264 faster is retired
    assert [rename(k) for k in ("h264@fast", "vvc@fast", "h264@faster")] == \
        ["h264@faster", "vvc@faster", "h264@retired-faster"]
    assert [rename(k) for k in ("h264", "vp9@veryfast", "h264@retired-faster")] == \
        ["h264", "vp9@veryfast", "h264@retired-faster"]


def test_renamed_names_are_not_renamed_again(tmp_path):
    """A record that carries the renames (rung_renames) is read as it is."""
    record = history_chart.stored_record(history_chart.PAGE_DATA)
    (tmp_path / "h.json").write_text(json.dumps(record))
    again = history_chart.stored_record(tmp_path / "h.json")
    assert [sorted(d["values"]) for d in again["days"]] == \
        [sorted(d["values"]) for d in record["days"]]


def test_stored_fronts_recompute_unchanged():
    """Each stored frame's frontier and front changes follow from its values alone, so an
    appended frame leaves them as published."""
    record = history_chart.stored_record(history_chart.PAGE_DATA)
    fields = lambda d: json.dumps({k: d.get(k) for k in history_chart.DERIVED}, sort_keys=True)  # noqa: E731
    before = [fields(d) for d in record["days"]]
    assert history_chart.add_fronts(record["days"]) == record["codec_front_base_days"]
    assert [fields(d) for d in record["days"]] == before


def test_the_key_share_view_reads_rungs_clips_and_paired_cpu_fps(tmp_path, monkeypatch):
    assert history_chart.family_rung("nanocodec-h264") == "h264:medium"
    assert history_chart.family_rung("nanocodec-av1@faster") == "av1:faster"
    clip = lambda name, cls, frames: {"name": name, "content_class": cls, "frames": frames}
    source = [clip("c2", "crowd", 1200), clip("c1", "crowd", 900), clip("c3", "crowd", 950),
              clip("s1", "static", 899), clip("s2", "static", 1000)]
    (tmp_path / history_chart.SHARE_SOURCE).write_text(json.dumps({"clips": source}))
    monkeypatch.setattr(history_chart, "load_paths", lambda: type("P", (), {"clips": tmp_path})())
    # long enough for 900 f with a key every 300; SHARE_PER_CLASS a class, by name
    assert [c["name"] for c in history_chart.share_clip_entries({})] == ["c1", "c2", "s2"]
    assert [c["name"] for c in history_chart.share_clip_entries(
        {history_chart.SHARE_KEY: {"clips": ["s2"]}})] == ["s2"]  # a stored view keeps its clips
    shard = tmp_path / "runs" / "zen5-1"
    shard.mkdir(parents=True)
    runs = [{"job": f"ab@k300|h264:fast|{c}", "variant": v, "user_s": t}
            for c, v, t in (("a", "base", 9.0), ("a", "base", 10.0), ("a", "base", 30.0),
                            ("a", "cand", 4.5), ("b", "base", 90.0), ("b", "cand", 45.0))]
    runs.append({"job": "ab@k150|h264:fast|a", "variant": "base", "user_s": 1.0})
    (shard / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    fps = history_chart.share_fps(tmp_path / "runs", 300)
    assert fps["h264:fast"]["base"] == pytest.approx((900 / 10.0 * 900 / 90.0) ** 0.5)
    assert fps["h264:fast"]["cand"] == pytest.approx((900 / 4.5 * 900 / 45.0) ** 0.5)
