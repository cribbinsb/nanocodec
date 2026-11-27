"""tools/bench/history_plan.py: each day runs the record's command under the same conditions,
adapted only where the day's switch for a condition differs (coordinator for the owner)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
import history_plan  # noqa: E402

CBR = ["{BIN}/nanocodec", "encode", "--codec", "av1", "--temporal-layers", "1", "--bitrate",
       "300", "--strict-buffer", "--param", "rate.key_cap_buffer=1.0", "--param",
       "hold.enable=0", "--param", "stream.encoder_info=0"]


def test_today_runs_the_record_command_unchanged():
    argv, notes = history_plan.adapt(list(CBR), "2026-11-24", "cbr", "record")
    assert argv == CBR and not notes


def test_before_the_strict_flag_the_bound_is_set_by_its_ceiling_param():
    argv, notes = history_plan.adapt(list(CBR), "2026-11-15", "cbr", "record")
    assert "--strict-buffer" not in argv and argv[-1] == "rate.recode_overflow=1.0"
    assert "stream.encoder_info=0" not in argv and "hold.enable=0" in argv
    argv, _ = history_plan.adapt(list(CBR), "2026-11-15", "cbr", "1.15")
    assert argv[-1] == "rate.recode_overflow=1.15"


def test_the_1_15_bound_drops_the_flag_where_1_15_is_the_default():
    argv, notes = history_plan.adapt(list(CBR), "2026-11-23", "cbr", "1.15")
    assert "--strict-buffer" not in argv and "rate.recode_overflow" not in " ".join(argv)


def test_early_days_drop_switches_for_features_they_did_not_have():
    argv, notes = history_plan.adapt(list(CBR), "2026-11-07", "qp", "record")
    assert "--temporal-layers" not in argv and "hold.enable=0" not in argv
    assert len(notes) == 3


def test_the_default_bound_is_ours_from_the_day_it_became_the_default():
    """BP-3.12 (owner): 1.15 buffers, no strict flag on any day, the key cap at its default."""
    argv, _ = history_plan.adapt(list(CBR), "2026-11-24", "cbr", "1.15")
    assert "--strict-buffer" not in argv and not any("key_cap_buffer" in a for a in argv)
    argv, _ = history_plan.adapt(list(CBR), "2026-11-12", "cbr", "1.15")
    assert "rate.recode_overflow=1.15" in argv and "rate.key_cap_buffer=1.0" in argv


def test_a_build_without_version_flag_records_its_commit_not_its_usage(tmp_path):
    """The builds to 11-10 print their usage for --version: the record names the commit."""
    import history_stage
    (tmp_path / "version").write_text("usage: nanocodec encode --codec av1|h264 ...\n")
    assert history_stage.version(tmp_path, "c" * 40) == \
        f"nanocodec unknown {'c' * 40} (this build has no --version)"
    (tmp_path / "version").write_text("nanocodec 0.6.0 abc\n")
    assert history_stage.version(tmp_path, "c" * 40) == "nanocodec 0.6.0 abc"


def test_days_before_all_skip_output_run_hold_off():
    """D21: hold with all-skip output from 11-23; earlier builds code every frame, hold off."""
    cbr = ["{BIN}/nanocodec", "encode", "--param", "hold.enable=1", "--param", "hold.code_skip=1"]
    argv, notes = history_plan.adapt(list(cbr), "2026-11-20", "cbr", "record")
    assert "hold.enable=0" in argv and not any("code_skip" in a for a in argv)
    argv, _ = history_plan.adapt(list(cbr), "2026-11-24", "cbr", "record")
    assert "hold.enable=1" in argv and "hold.code_skip=1" in argv

