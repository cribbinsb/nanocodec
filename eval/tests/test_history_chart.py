"""tools/dev/history_chart.py: stored frames read through the rung renames (no encodes run)."""
import importlib.util
import json
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
