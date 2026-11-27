"""The fairness record (docs/benchmark-fairness.toml): every family's command keeps the
shared constraints, every flag group is explained, and the procedure's tables are its render."""
from pathlib import Path

import pytest

from nceval import fairness

CLIP = {"BIN": "/b", "CFG": "/cfg", "INPUT": "in.yuv", "OUTPUT": "out", "W": 1280, "H": 720,
        "FPS_NUM": 30, "FPS_DEN": 1, "N": 150, "KEYINT": 300, "KBPS": 600, "QP": 32}
FAMILIES = sorted(fairness.families())
PHASE = fairness.phase_values(fairness.load())


def pairs(argv):
    joined = {}
    for i, a in enumerate(argv):
        if "=" in a and a.startswith("--"):
            k, v = a.split("=", 1)
            joined[k] = v
        elif a.startswith("-") and i + 1 < len(argv):
            joined.setdefault(a, argv[i + 1])
    return joined


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["cbr", "qp"])
def test_every_command_fills(family, mode):
    try:
        argv = fairness.command(family, mode, CLIP)
    except fairness.NotCharted:
        assert family.endswith(fairness.DROP_SUFFIX) or "~" in family or \
            (family.startswith("vvenc") and mode == "cbr")
        return
    assert all("{" not in a for a in argv)


def charted(family, mode):
    try:
        return fairness.command(family, mode, CLIP)
    except fairness.NotCharted:
        return None


@pytest.mark.parametrize("family", FAMILIES)
def test_shared_cbr_constraints(family):
    argv = charted(family, "cbr")
    if argv is None:
        return
    tool = argv[0].rsplit("/", 1)[-1]
    f = pairs(argv)
    drop = family.endswith(fairness.DROP_SUFFIX)
    if tool == "nanocodec":
        # BP-4.0 (owner, 2026-11-25, final): ours holds the 1 s buffer strictly
        assert f["--buffer-ms"] == "1000" and f["--peak-bitrate"] == "600"
        assert "--strict-buffer" in argv and not any("key_cap" in a for a in argv)
        for p in ("rate.first_key_lookahead_off=1", "hold.enable=1", "hold.code_skip=1"):
            assert p in argv
        assert f["--intra-period"] == "300"
    elif tool in ("x264", "x265"):
        assert f["--vbv-bufsize"] == f["--vbv-maxrate"] == f["--bitrate"] == "600"
        assert f["--vbv-init"] == PHASE["VBV_INIT"]
        assert f["--rc-lookahead"] == "0" and f["--bframes"] == "0"
        assert f["--keyint"] == "300"
        if tool == "x264":
            # BP-4.0 rule 4: scene-cut keys are x264's own choice (default on)
            assert "--no-scenecut" not in argv and "--min-keyint" not in argv
        else:
            # zerolatency's own scene-cut off, restated (x265 param.cpp:674-677)
            assert "--no-scenecut" in argv and "--asm" not in f
        assert (f["--threads"] == "1") if tool == "x264" else \
            (f["--frame-threads"] == "1" and f["--pools"] == "none" and "--no-wpp" in argv)
    elif tool == "SvtAv1EncApp":
        assert (f["--buf-sz"], f["--buf-initial-sz"], f["--buf-optimal-sz"]) == \
            ("1000", PHASE["INIT_MS"], "600")
        assert f["--lp"] == "1" and f["--scd"] == "0" and f["--keyint"] == "300"
        # BP-4.0 rule 4: SVT's own low-delay CBR structure (2 levels), not forced flat
        assert "--hierarchical-levels" not in f
    elif tool in ("vpxenc", "aomenc"):
        assert (f["--buf-sz"], f["--buf-initial-sz"], f["--buf-optimal-sz"]) == \
            ("1000", PHASE["INIT_MS"], "600")
        assert f["--threads"] == "1" and f["--lag-in-frames"] == "0"
        assert f["--drop-frame"] == (fairness.VPX_AOM_DROP_THRESHOLD if drop else "0")
        assert f["--kf-max-dist"] == f["--kf-min-dist"] == "300"
        # main rows: the full quantizer range (owner, 2026-11-25); +drop: the example's 52
        assert f["--max-q"] == (fairness.SHIPPED_DROP_MAX_Q if drop else fairness.MAIN_ROW_MAX_Q)
        if tool == "vpxenc":
            assert f["--resize-allowed"] == "0"
        else:
            # libaom's main rows only: the +drop rows run the example program
            assert not drop and "--resize-mode" not in f  # aomenc's default: no resize
    elif tool == "svc_encoder_rtc":
        # libaom's '+drop' rows only: its real-time example as shipped, one layer; its own bucket
        # (1000 / 600 / 600 ms) is fixed in the program (record)
        assert drop
        assert f["-th"] == "1" and f["-k"] == "300" and f["-sl"] == f["-tl"] == "1"
        assert f["-lm"] == "0" and f["-b"] == f["-bl"] == "600"
        assert f["--drop-frame"] == (fairness.VPX_AOM_DROP_THRESHOLD if drop else "0")
    elif tool == "openh264-harness":
        assert f["--frame-skip"] == ("1" if drop else "0")
        # BP-4.0 rule 4: the library's defaults (param_svc.h:166, 168)
        assert f["--adaptive-quant"] == "1" and f["--scene-change"] == "1"
        assert f["--threads"] == "1" and f["--intra-period"] == "300"
    elif tool == "kvazaar":
        # Kvazaar's --bitrate is in bits per second (kvazaar --help), not kbps
        assert f["--bitrate"] == "600000"
        # its low-delay P GOP (BP-4.0, record): no reordering; Kvazaar has no lookahead
        assert f["--gop"] == "lp-g8d4t1" and f["--period"] == "300"
        assert f["--threads"] == "0" and f["--owf"] == "0"
        for flag in ("--no-wpp", "--no-psnr", "--no-info", "--no-open-gop"):
            assert flag in argv
    elif tool == "vvencFFapp":
        # low-delay configs: no reordering, past references only (verified, record); the
        # one-pass RC's look-ahead is the stated exception
        assert any(a.endswith(("lowdelay_faster.cfg", "lowdelay_fast.cfg")) for a in argv)
        assert f["--IntraPeriod"] == "300" and f["--TargetBitrate"] == "600000"
        assert f["--MaxBitrate"] == "900000"
        assert "--Threads=1" in argv and "--MaxParallelFrames=0" in argv
    else:
        pytest.fail(f"no shared-constraint rule for {tool}")


def test_the_buffer_starts_full_for_every_encoder():
    """BP-4.0 (owner, 2026-11-25): one phase, the 1 s buffer starting full."""
    record = fairness.load()
    assert record["phase"]["current"] == "full-start"
    assert set(record["phase"]) == {"current", "full-start"}
    assert (PHASE["INIT_MS"], PHASE["VBV_INIT"]) == ("1000", "1.0")


def test_fixed_qp_is_each_encoders_no_rate_control_mode():
    for family in FAMILIES:
        argv = charted(family, "qp")
        if argv is None:
            continue
        assert "--ipratio" not in argv and "rate.constant_qp=1" not in argv, family
        tool = argv[0].rsplit("/", 1)[-1]
        f = pairs(argv)
        if tool == "SvtAv1EncApp":
            # its own default AQ (review 3 C5); inactive in --rtc mode (record: identical streams)
            assert f["--rc"] == "0" and "--aq-mode" not in f
        if tool in ("vpxenc", "aomenc"):
            assert f["--end-usage"] == "q" and "--min-q" not in f and "--max-q" not in f
        if tool == "aomenc":  # the example's real-time tool set (review 3 C4)
            assert f["--enable-global-motion"] == "0" and f["--enable-obmc"] == "0"
            assert f["--coeff-cost-upd-freq"] == "3" and f["--use-intra-default-tx-only"] == "1"
        if tool == "vpxenc":
            assert f["--static-thresh"] == "1" and f["--noise-sensitivity"] == "0"


def test_every_flag_group_is_explained():
    for e in fairness.load()["encoder"]:
        assert e["why"], e["family"]
        for w in e["why"]:
            for key in ("flags", "controls", "value", "fair", "remaining"):
                assert w.get(key, "").strip(), (e["family"], key)


def test_procedure_tables_are_the_render():
    doc = (Path(fairness.RECORD).parent / "benchmark-procedure.md").read_text()
    assert fairness.render() in doc


def test_openh264_has_a_labelled_drop_row_and_its_main_rows_code_every_frame():
    """T16: the '+drop' row runs openh264's own frame skipping with the coded-frame list."""
    drop = fairness.command("openh264-baseline" + fairness.DROP_SUFFIX, "cbr", CLIP)
    assert drop[drop.index("--frame-skip") + 1] == "1"
    assert drop[drop.index("--coded-frames") + 1] == CLIP["OUTPUT"] + ".frames"
    for family in ("openh264-baseline", "openh264-baseline~cabac"):
        main = fairness.command(family, "cbr", CLIP)
        assert main[main.index("--frame-skip") + 1] == "0" and "--coded-frames" not in main
    assert "openh264-baseline~cabac" + fairness.DROP_SUFFIX not in fairness.families()
