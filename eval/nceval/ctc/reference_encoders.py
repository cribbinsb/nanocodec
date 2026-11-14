"""Adapters for the standards yardsticks the existing registry lacks.

HM and VTM run the JVET low-delay P configuration files unchanged
(`encoder_lowdelay_P_main.cfg`, `encoder_lowdelay_P_vtm.cfg`), VVenC its
low-delay `faster` file, and libaom the AOM CTC low-delay command line
(CWG-F384o §4.3) as far as libaom has its options. Every adapter names its
tool and configuration file relative to the run directory (`bin/`, `cfg/`):
these encoders run only on the rented instance, where the launcher puts them.

Deviations from the CTCs, all forced by the project's 8-bit scope, are kept
on the command line, not in edited configuration files:
`--InternalBitDepth=8` for VTM and VVenC (their CTC files say 10). Picture
sizes that are not whole minimum CUs (270 lines) are padded by the encoder
and cropped by a conformance window (`--ConformanceWindowMode=1`), as the
JVET software does for such inputs.

HM, VTM and VVenC are BSD-licensed and run as external tools only.
"""

from __future__ import annotations

from pathlib import Path

from ..codecs import Codec
from ..config import Paths
from ..encoders.base import EncodeJob, EncoderAdapter, fps_fraction

#: Where the launcher puts the JVET configuration files, relative to the run directory.
CFG_DIR = Path("cfg")
HM_CONFIG = CFG_DIR / "encoder_lowdelay_P_main.cfg"
VTM_CONFIG = CFG_DIR / "encoder_lowdelay_P_vtm.cfg"
VVENC_CONFIG = CFG_DIR / "lowdelay_faster.cfg"
#: The project codes 8-bit video only (owner decision; CTC files say 10 for VVC).
CODING_BIT_DEPTH = 8
#: Pad to whole CUs and signal a conformance window (JVET software option).
AUTO_CONFORMANCE_WINDOW = 1
#: The configuration files name a reconstruction file (rec.yuv in the working
#: directory, shared by parallel jobs); nothing reads it.
NO_RECON = "/dev/null"
QP_RANGE = (0, 51)
AV1_Q_RANGE = (0, 63)


class _JvetSoftware(EncoderAdapter):
    """HM and VTM share one command line apart from the tool and configuration."""

    tool = ""
    config: Path = Path()
    extra: tuple[str, ...] = ()

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        clip = job.clip
        # `-fr` takes an integer; it only labels the stream (the rate is computed here).
        return [self.tool, "-c", str(self.config), "-i", str(clip.yuv),
                "-wdt", str(clip.width), "-hgt", str(clip.height),
                "-fr", str(round(clip.fps)), "-f", str(job.frames), "-q", str(job.quantizer),
                "--InputBitDepth=8", "--OutputBitDepth=8",
                f"--ConformanceWindowMode={AUTO_CONFORMANCE_WINDOW}", *self.extra,
                "-b", str(job.bitstream), "-o", NO_RECON]


class HmLowDelayP(_JvetSoftware):
    """HM-19.0, JVET/JCT-VC low-delay P (GOP 8, 4 references, QP offsets)."""

    name = "hm-ldp"
    codec = Codec.HEVC
    quantizer_range = QP_RANGE
    tool = "TAppEncoder"
    config = HM_CONFIG


class VtmLowDelayP(_JvetSoftware):
    """VTM-24.0, JVET low-delay P, coded at 8 bits (the CTC codes Main 10)."""

    name = "vtm-ldp"
    codec = Codec.VVC
    quantizer_range = QP_RANGE
    tool = "EncoderApp"
    config = VTM_CONFIG
    extra = (f"--InternalBitDepth={CODING_BIT_DEPTH}",)


class VvencLowDelay(EncoderAdapter):
    """VVenC v1.14.0 `lowdelay_faster.cfg`: low delay with B slices that
    predict only from past pictures (LD-B), one thread, 8-bit."""

    name = "vvenc-faster-ld"
    codec = Codec.VVC
    quantizer_range = QP_RANGE

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        clip = job.clip
        num, den = fps_fraction(clip.fps)
        return ["vvencFFapp", "-c", str(VVENC_CONFIG), "-i", str(clip.yuv),
                "-s", f"{clip.width}x{clip.height}", "--FrameRate", str(num),
                "--FrameScale", str(den), "-f", str(job.frames), "-q", str(job.quantizer),
                "--InputBitDepth=8", "--OutputBitDepth=8",
                f"--InternalBitDepth={CODING_BIT_DEPTH}",
                f"--ConformanceWindowMode={AUTO_CONFORMANCE_WINDOW}",
                "--Threads=1", "--MaxParallelFrames=0", "-b", str(job.bitstream),
                "-o", NO_RECON]


class LibaomGoodLowDelay(EncoderAdapter):
    """libaom v3.15.3 at the AOM CTC low-delay settings (CWG-F384o §4.3):
    speed 0, no lag, fixed QP offsets over a 16-frame pyramid, no TPL or
    delta-q. `--subgop-config-str=ld` is AVM-only; libaom's lag-0 GF groups
    are its low-delay form. Quantiser on libaom's 0-63 scale."""

    name = "aom-good-s0-ld"
    codec = Codec.AV1
    quantizer_range = AV1_Q_RANGE

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        clip = job.clip
        num, den = fps_fraction(clip.fps)
        return ["aomenc", "--good", "--cpu-used=0", "--passes=1", "--lag-in-frames=0",
                "--min-gf-interval=16", "--max-gf-interval=16", "--gf-min-pyr-height=4",
                "--gf-max-pyr-height=4", "--kf-min-dist=9999", "--kf-max-dist=9999",
                "--use-fixed-qp-offsets=1", "--deltaq-mode=0", "--enable-tpl-model=0",
                "--end-usage=q", f"--cq-level={job.quantizer}",
                "--enable-keyframe-filtering=0", "--threads=1", "--tile-columns=0",
                "--tile-rows=0", "--row-mt=0", f"--width={clip.width}",
                f"--height={clip.height}", "--i420", f"--fps={num}/{den}",
                f"--limit={job.frames}", "--ivf", "-o", str(job.bitstream), str(clip.yuv)]
