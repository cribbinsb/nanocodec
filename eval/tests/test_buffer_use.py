"""The buffer-use report (nceval.buffer_use) and the buffer statement (nceval.buffer_note)."""

from nceval import buffer_note, buffer_use
from nceval.cbr_stats import own_rate_fullness, replay_buffer

FPS = 30.0
BUFFER_MS = 1000.0


def test_own_rate_drains_an_undershooting_stream_at_its_own_rate():
    # a 30 kbit key then 1 kbit frames: 35.8 kbps over 5 s, on a 100 kbps target
    bits = [30_000] + [1_000] * 149
    at_target = replay_buffer(bits, 100_000, FPS, BUFFER_MS).max_fullness
    assert abs(at_target - 0.3) < 1e-9  # a third of the target's second
    assert abs(own_rate_fullness(bits, FPS, BUFFER_MS) - 30_000 / 35_800) < 1e-9


def point(kbps, target, at_target, own, key=0.3, window=1.1):
    return {"kbps": kbps, "target_kbps": target, "max_fullness": at_target,
            "own_rate_fullness": own, "key_buffers": key, "max_window_ratio": window}


def test_a_rate_floor_is_left_out_at_the_target_only():
    points = [point(100, 100, 0.5, 0.5), point(98, 100, 1.2, 1.25),
              point(300, 100, 9.0, 0.6)]  # the encoder's floor: 3x the target
    row = buffer_use.row({"a": points})
    assert row["points"] == 3
    assert abs(row["rate_floor_pct"] - 100 / 3) < 1e-9
    assert row["target_max"] == 1.2  # the floor's 9 buffers are not buffer control
    assert row["own_max"] == 1.25
    assert row["target_over"] == {"1": 50.0, "1.15": 50.0, "1.5": 0.0}
    assert abs(row["own_over"]["1"] - 100 / 3) < 1e-9


def test_our_presets_pool_per_codec():
    assert buffer_use.encoder_group("nanocodec-av1@fast") == "nanocodec-av1"
    assert buffer_use.encoder_group("svtav1-m12") == "svtav1-m12"


def test_statement_ranges():
    assert buffer_note.percent_range([2.4, 11.6]) == "2-12 %"
    assert buffer_note.percent_range([42.0]) == "42 %"


def test_every_fact_has_an_interval_over_clips_that_holds_its_value():
    by_clip = {"a": [point(100, 100, 0.5, 0.5, key=0.2), point(100, 100, 1.2, 1.3, key=0.4)],
               "b": [point(100, 100, 0.9, 0.9, key=0.3)],
               "c": [point(300, 100, 9.0, 0.6, key=0.5)]}  # a rate floor only
    facts = buffer_use.row(by_clip)["intervals"]
    assert set(facts) == {name for name, _ in buffer_use.FACTS}
    for name, (value, low, high) in facts.items():
        assert low <= value <= high, name
    assert abs(facts["own_over_1"][0] - 25.0) < 1e-9  # 1 of 4 points, every point counts
    assert abs(facts["target_over_1"][0] - 100 / 3) < 1e-9  # the floor is left out
    assert facts["key_buffers_median"][0] == 0.35


def test_facts_table_shows_every_encoder_with_intervals(monkeypatch):
    rows = {name: {"points": 4, "intervals": {key: [1.0, 0.5, 1.5] for key, _ in
                                               buffer_use.FACTS}}
            for name in ("nanocodec-av1", "svtav1-m12")}
    monkeypatch.setattr(buffer_note, "view", lambda view_id: {"rows": rows})
    table = buffer_note.facts_table()
    assert "| nanocodec-av1 | 4 | 1% (0–2) |" in table
    assert "svtav1-m12" in table and "1.00 buffers (0.50–1.50)" in table
