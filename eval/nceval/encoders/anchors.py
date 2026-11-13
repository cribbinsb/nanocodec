"""Adapters for the third-party anchor encoders (eval/README.md
"Methodology: Anchors").

Each class pins one configuration. Flags are kept explicit even when they
repeat a preset's default, so a preset change upstream cannot silently alter
the comparison.

openh264 is configured through its own adapter (openh264.py), because its
CLI is driven by a config file.
"""

from __future__ import annotations

from ..codecs import Codec
from ..config import Paths, find_tool
from .base import EncodeJob, EncoderAdapter, fps_fraction


# Anchor variants (adversarial review): the registered names above keep their
# exact historical flags (variant ""); each variant changes only the quality
# controls named, to test whether the pinned configuration handicaps an anchor.
#   nopsy  psychovisual RD/trellis off (x264/x265 default them on; they cost PSNR)
#   crf    the encoder's own constant-quality mode with its default AQ (how it is
#          really used; the quantizer becomes the CRF value)
#   qm     SVT quantisation matrices on
# (SVT --crf equals --qp in low-delay RTC, byte for byte: it has no AQ there;
# its --tune 5 VMAF mode refuses low-delay prediction. Neither is a variant.)
X264_VARIANTS = {
    "": lambda q: ["--qp", str(q)],
    "nopsy": lambda q: ["--qp", str(q), "--no-psy"],
    "crf": lambda q: ["--crf", str(q)],
}
X265_VARIANTS = {
    "": lambda q: ["--qp", str(q)],
    "nopsy": lambda q: ["--qp", str(q), "--psy-rd", "0", "--psy-rdoq", "0"],
    "crf": lambda q: ["--crf", str(q)],
}
SVT_VARIANTS = {
    "": lambda q: ["--rc", "0", "--aq-mode", "0", "--qp", str(q)],
    "qm": lambda q: ["--rc", "0", "--aq-mode", "0", "--qp", str(q), "--enable-qm", "1"],
}


class X264(EncoderAdapter):
    """x264, a true High-profile H.264 reference (CABAC + 8x8). GPL: run only."""

    codec = Codec.H264
    quantizer_range = (10, 51)  # below 10 is far outside our bitrate range

    def __init__(self, preset: str, variant: str = ""):
        self.preset = preset
        self.variant = variant
        self.name = f"x264-{preset}" + (f"-{variant}" if variant else "")

    def supports_recon(self) -> bool:
        return True

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        clip, num_den = job.clip, fps_fraction(job.clip.fps)
        cmd = [str(find_tool("x264", paths)),
               "--preset", self.preset, "--tune", "zerolatency",
               "--bframes", "0", "--rc-lookahead", "0", "--no-scenecut",
               "--keyint", str(job.structure.intra_period_frames),
               "--min-keyint", str(job.structure.intra_period_frames),
               "--threads", "1", *X264_VARIANTS[self.variant](job.quantizer),
               "--input-res", f"{clip.width}x{clip.height}",
               "--fps", f"{num_den[0]}/{num_den[1]}",
               "--frames", str(job.frames),
               "--output", str(job.bitstream), str(clip.yuv)]
        if job.recon:
            cmd[1:1] = ["--dump-yuv", str(job.recon)]
        return cmd


class X265(EncoderAdapter):
    """x265 in zero-latency mode. No hierarchical P is possible. GPL: run only."""

    codec = Codec.HEVC
    quantizer_range = (10, 51)  # below 10 is far outside our bitrate range

    def __init__(self, preset: str, variant: str = ""):
        self.preset = preset
        self.variant = variant
        self.name = f"x265-{preset}" + (f"-{variant}" if variant else "")

    def supports_recon(self) -> bool:
        return True

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        clip, num_den = job.clip, fps_fraction(job.clip.fps)
        cmd = [str(find_tool("x265", paths)),
               "--preset", self.preset, "--tune", "zerolatency",
               "--bframes", "0", "--rc-lookahead", "0", "--no-scenecut",
               "--keyint", str(job.structure.intra_period_frames),
               "--min-keyint", str(job.structure.intra_period_frames),
               "--frame-threads", "1", "--pools", "none", "--no-wpp",
               *X265_VARIANTS[self.variant](job.quantizer),
               "--input-res", f"{clip.width}x{clip.height}",
               "--input-depth", "8", "--fps", f"{num_den[0]}/{num_den[1]}",
               "--frames", str(job.frames),
               "--input", str(clip.yuv), "--output", str(job.bitstream)]
        if job.recon:
            cmd += ["--recon", str(job.recon)]
        return cmd


class Kvazaar(EncoderAdapter):
    """kvazaar with a low-delay P GOP (bipred off, so frames really are P)."""

    codec = Codec.HEVC
    quantizer_range = (10, 51)  # below 10 is far outside our bitrate range

    def __init__(self, preset: str):
        self.preset = preset
        self.name = f"kvazaar-{preset}"

    def supports_temporal_layers(self, layers: int) -> bool:
        return layers in (1, 2, 3)

    def supports_recon(self) -> bool:
        return True

    def check_job(self, job: EncodeJob) -> None:
        super().check_job(job)
        # kvazaar 2.3.2 crashes (SIGSEGV) when the intra period is not a
        # multiple of the low-delay GOP length; fail with a clear message.
        span = 2 ** (job.structure.temporal_layers - 1)
        if job.structure.intra_period_frames % span:
            raise ValueError(f"{self.name}: intra period {job.structure.intra_period_frames} "
                             f"must be a multiple of the {span}-frame temporal GOP")

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        clip = job.clip
        layers = job.structure.temporal_layers
        # "--gop 0" is flat P. Hierarchical P uses kvazaar's low-delay GOP syntax
        # lp-g<length>d<QP layers>t<temporal skip> (kvazaar README "LP-GOP
        # syntax"); t = 2^(L-1) makes only every 2^(L-1)-th picture a base-layer
        # picture, i.e. L dyadic temporal layers.
        span = 2 ** (layers - 1)
        gop = "0" if layers == 1 else f"lp-g{span}d{layers}t{span}"
        cmd = [str(find_tool("kvazaar", paths)),
               "--preset", self.preset, "--gop", gop, "--no-bipred",
               "--period", str(job.structure.intra_period_frames),
               "--qp", str(job.quantizer),
               "--threads", "0", "--owf", "0", "--no-wpp",
               "--input-res", f"{clip.width}x{clip.height}",
               "--input-fps", f"{clip.fps:g}", "--frames", str(job.frames),
               "-i", str(clip.yuv), "-o", str(job.bitstream)]
        if job.recon:
            cmd += ["--debug", str(job.recon)]
        return cmd


class SvtAv1(EncoderAdapter):
    """SVT-AV1 in its real-time mode (--rtc 1: low-delay with RTC speed
    settings) at one of its fast presets. Without --rtc, M12+ are silently
    mapped to M11 and use non-RTC tools."""

    codec = Codec.AV1
    # Floor 10, not 20: on near-static clips QP 20 is still far below the
    # 50%-of-source cap (bodycam_airport: 182 kbps vs a 788 kbps cap), which
    # left two legal points and no BD-rate. Part of the fingerprint.
    quantizer_range = (10, 63)

    def __init__(self, preset: int, variant: str = ""):
        self.preset = preset
        self.variant = variant
        self.name = f"svtav1-m{preset}" + (f"-{variant}" if variant else "")

    def supports_temporal_layers(self, layers: int) -> bool:
        return layers in (1, 2, 3)

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        clip, num_den = job.clip, fps_fraction(job.clip.fps)
        return [str(find_tool("SvtAv1EncApp", paths)),
                "--rtc", "1", "--preset", str(self.preset),
                "--pred-struct", "1",  # 1 = low delay (also forced by --rtc)
                # Levels beyond the base layer: 0 = flat P.
                "--hierarchical-levels", str(job.structure.temporal_layers - 1),
                *SVT_VARIANTS[self.variant](job.quantizer),
                "--keyint", str(job.structure.intra_period_frames), "--scd", "0",
                "--lp", "1", "--fps-num", str(num_den[0]), "--fps-denom", str(num_den[1]),
                "-w", str(clip.width), "-h", str(clip.height), "-n", str(job.frames),
                "-i", str(clip.yuv), "-b", str(job.bitstream)]


class LibaomRealtime(EncoderAdapter):
    """libaom in real-time usage: the closest public design to ours."""

    codec = Codec.AV1
    quantizer_range = (20, 63)  # lower values are far above our bitrate range

    def __init__(self, cpu_used: int):
        self.cpu_used = cpu_used
        self.name = f"aom-rt-s{cpu_used}"

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        clip, num_den = job.clip, fps_fraction(job.clip.fps)
        q = str(job.quantizer)
        return [str(find_tool("aomenc", paths)),
                "--rt", f"--cpu-used={self.cpu_used}", "--lag-in-frames=0",
                # min-q == max-q pins the quantiser. aomenc warns about that and
                # would wait for a "continue?" answer; -y prints but never asks.
                "--end-usage=q", f"--cq-level={q}", f"--min-q={q}", f"--max-q={q}",
                "--disable-warning-prompt",
                f"--kf-min-dist={job.structure.intra_period_frames}",
                f"--kf-max-dist={job.structure.intra_period_frames}",
                "--threads=1", "--tile-columns=0", "--row-mt=0", "--aq-mode=0",
                f"--width={clip.width}", f"--height={clip.height}", "--i420",
                f"--fps={num_den[0]}/{num_den[1]}", f"--limit={job.frames}",
                "--ivf", "-o", str(job.bitstream), str(clip.yuv)]


class LibvpxRealtime(EncoderAdapter):
    """VP9 Profile 0, flat forward prediction and constant quantizer."""

    codec = Codec.VP9
    quantizer_range = (0, 63)

    def __init__(self, cpu_used: int):
        if not 0 <= cpu_used <= 9:
            raise ValueError("libvpx realtime speed must be in 0..9")
        self.cpu_used = cpu_used
        self.name = f"vpx-vp9-rt-s{cpu_used}"

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        self.check_job(job)
        clip = job.clip
        num, den = fps_fraction(clip.fps)
        q = str(job.quantizer)
        return [str(find_tool("vpxenc", paths)), "--codec=vp9", "--profile=0",
                "--rt", "--passes=1", f"--cpu-used={self.cpu_used}",
                "--lag-in-frames=0", "--auto-alt-ref=0", "--threads=1",
                "--tile-columns=0", "--tile-rows=0", "--row-mt=0", "--aq-mode=0",
                "--end-usage=q", f"--cq-level={q}", f"--min-q={q}", f"--max-q={q}",
                "--drop-frame=0", "--resize-allowed=0", "--disable-warning-prompt",
                f"--kf-min-dist={job.structure.intra_period_frames}",
                f"--kf-max-dist={job.structure.intra_period_frames}",
                f"--width={clip.width}", f"--height={clip.height}", "--i420",
                f"--fps={num}/{den}", f"--limit={job.frames}",
                "--ivf", "-o", str(job.bitstream), str(clip.yuv)]


class AvmLowDelay(EncoderAdapter):
    """Final AV2 reference in single-thread low-delay mode, not an RTC preset."""

    codec = Codec.AV2
    quantizer_range = (0, 255)

    def __init__(self, cpu_used: int):
        if not 0 <= cpu_used <= 9:
            raise ValueError("AVM speed must be in 0..9")
        self.cpu_used = cpu_used
        self.name = f"avm-ld-s{cpu_used}"

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        self.check_job(job)
        clip = job.clip
        num, den = fps_fraction(clip.fps)
        q = str(job.quantizer)
        return [str(find_tool("avmenc", paths)), "--good", "--passes=1",
                f"--cpu-used={self.cpu_used}", "--threads=1", "--row-mt=0",
                "--tile-columns=0", "--lag-in-frames=0", "--auto-alt-ref=0",
                "--enable-fwd-kf=0", "--aq-mode=0", "--deltaq-mode=0",
                "--end-usage=q", f"--qp={q}", f"--min-qp={q}", f"--max-qp={q}",
                "--disable-warning-prompt", "--bit-depth=8", "--input-bit-depth=8",
                f"--kf-min-dist={job.structure.intra_period_frames}",
                f"--kf-max-dist={job.structure.intra_period_frames}",
                f"--width={clip.width}", f"--height={clip.height}", "--i420",
                f"--fps={num}/{den}", f"--limit={job.frames}",
                "--ivf", "-o", str(job.bitstream), str(clip.yuv)]
