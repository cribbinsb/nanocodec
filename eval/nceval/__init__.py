"""nceval: the nanocodec evaluation harness.

Pipeline for every test point (see eval/README.md "Methodology"):

    clip (raw YUV) -> encoder adapter -> bitstream
        -> conformance gate (two independent standard decoders, bit-exact)
        -> metrics (PSNR-YUV 6:1:1, XPSNR, VMAF, VMAF-NEG, MS-SSIM)
        -> results database -> BD-rate / speed reports

Speed is measured in separate, pinned single-core runs (nceval.speed).
"""
