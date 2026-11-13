"""Calibrated window flicker of the real-time reference encoder, per clip.

The J_real flicker guard (jreal.FLICKER_GUARD_DB) compares a candidate with
its anchor. On content where our anchor already pumps far less than the
real-time encoders we are judged against, that delta alone would reject a
cadence the reference itself exceeds. So a clip's guarded delta is reduced by
its headroom: how far the anchor's window flicker lies below the reference's
(jreal.compare_clip). Coordinator decision 2026-11-10: a candidate whose
window flicker stays at or below SVT-AV1 M12 --rtc's is acceptable.

Values: SVT-AV1 M12 --rtc CBR (1 s buffer, the real-use ladder, one key),
window flicker (jreal.window_flicker) averaged over the ladder, measured with
tools/dev/flicker_calibrate.py on 2026-11-10 (conference clips at 130 frames,
TRAIN proxy-6 clips at 150). For reference on the same runs (mean of the
clips): conference x265 superfast 0.310, x264 veryfast 0.366, SVT 0.417, our
VVC / HEVC Default 0.129 / 0.126; proxy-6 x265 0.209, x264 0.225, SVT 0.361,
our Default 0.356 / 0.361. Clips without a value get no headroom.
"""

#: SVT-AV1 M12 --rtc window flicker (dB) per clip.
REFERENCE_WINDOW_FLICKER_DB = {
    "adv_johnny_720p": 0.452,
    "adv_fourpeople_270p": 0.531,
    "adv_meridiantalk_1080p": 0.269,
    "crowd_mot20_02": 0.283,
    "crowd_pp22_027": 0.159,
    "pan_jaad_0117": 0.214,
    "static_cam22": 0.327,
    "bodycam_antare_0815": 0.926,
    "pan_jaad_0223": 0.255,
}
