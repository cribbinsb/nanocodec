"""The statement every CBR chart and CBR number carries in 0.6.0 (owner, 2026-11-21): our
buffer is a target, the references' a hard bound. One text, used by the results site
(nceval.site: chart captions, the README's headline), the chart pages (nceval.chart render)
and the release notes; the method page carries the same text by hand. Measured in
docs/analysis/frame-allocation-2026-11-21.md; a later release makes the buffer a hard bound
by default and re-measures the CBR charts."""

HEADING = "Rate control and the buffer"
#: Where the method page carries it.
URL = "https://cribbinsb.github.io/nanocodec/results/method.html#buffer"
#: One line beside a CBR chart or number.
CAPTION = f"nanocodec's buffer is looser than the references' here: see {HEADING}."

STATEMENT = (
    "Every encoder in the CBR charts is asked for a 1 s buffer at the target rate. x264, x265 "
    "and NVENC are held to it by a hard VBV, and SVT-AV1, libaom and libvpx by their 1000 ms "
    "real-time buffer. nanocodec 0.6.0 treats its buffer as a target with a 1.4x stop and sizes "
    "its key frames more generously: in these charts its buffer peak passes one buffer on "
    "10-16 % of points (theirs: 0-1 %). openh264 is given no buffer setting and behaves like "
    "nanocodec in this respect. That is worth quality. On a sample of the held-out clips (24 "
    "clips at 150 frames, 12 at 600 frames; AV1, HEVC and H.264 medium), holding nanocodec's "
    "stream inside one buffer, as the references' is, cost it 3-5.5 % in bits at equal "
    "quality: its lead over SVT-AV1 M12 at 150 frames moved from 31 % to 28 % (AV1) and from "
    "33 % to 29 % (HEVC), H.264 medium moved from level with SVT-AV1 M12 to about 5 % behind, "
    "and the leads over x264 medium moved by 0.6-2.2 points. The Fixed QP charts have no rate "
    "control and are not affected. A later release will make the buffer a hard bound by "
    "default and re-measure these charts.")
