"""Adapter for our own encoder: the `nanocodec` CLI from crates/nc-cli.

Unlike the anchors, nanocodec outputs its reconstruction, so evaluation
scores it directly and decodes nothing (nceval.gatepolicy); an explicit gate
checks it against the decoders.
"""

from __future__ import annotations

from pathlib import Path

from ..codecs import Codec
from ..config import Paths, find_tool
from .base import EncodeJob, EncoderAdapter, fps_fraction

PRESETS_DIR = Path(__file__).resolve().parents[3] / "presets"
#: `nc_encoder::MAX_TEMPORAL_LAYERS`.
MAX_TEMPORAL_LAYERS = 3


class Nanocodec(EncoderAdapter):
    """Native low-delay encoder; VP9/AV2 currently support one temporal layer."""

    def __init__(self, codec: Codec, quantizer_range: tuple[int, int],
                 preset: str | None = None):
        self.codec = codec
        self.quantizer_range = quantizer_range
        self.name = f"nanocodec-{codec.value}" + (f"@{preset}" if preset else "")
        # A preset's values go on the command line as --param pairs (not
        # --params FILE), so the result fingerprint covers the values.
        self.overrides = read_preset(PRESETS_DIR / f"{preset}.toml") if preset else {}

    def supports_temporal_layers(self, layers: int) -> bool:
        # Hierarchical-P structures (docs/handoff/hierarchical-p.md).
        return layers == 1 if self.codec == Codec.VP9 else 1 <= layers <= MAX_TEMPORAL_LAYERS

    def supports_recon(self) -> bool:
        return True

    def verify_recon(self) -> bool:
        return True

    def own_decoder(self, paths: Paths) -> Path:
        return find_tool("nanocodec", paths)

    def command(self, job: EncodeJob, paths: Paths) -> list[str]:
        if not self.supports_temporal_layers(job.structure.temporal_layers):
            raise ValueError(f"{self.name} does not support requested temporal layers")
        clip, (num, den) = job.clip, fps_fraction(job.clip.fps)
        cmd = [str(find_tool("nanocodec", paths)), "encode", "--codec", self.codec.value,
               "--input", str(clip.yuv), "--width", str(clip.width), "--height", str(clip.height),
               "--fps", f"{num}/{den}", "--qp", str(job.quantizer), "--frames", str(job.frames),
               "--intra-period", str(job.structure.intra_period_frames),
               "--output", str(job.bitstream)]
        # Flat streams keep their historical command line (and fingerprint).
        if job.structure.temporal_layers > 1:
            cmd += ["--temporal-layers", str(job.structure.temporal_layers)]
        if job.recon:
            cmd += ["--display-recon", str(job.recon)]
        if job.coded_times:
            cmd += ["--timestamps-out", str(job.coded_times)]
        for name, value in sorted(self.overrides.items()):
            cmd += ["--param", f"{name}={value!r}"]
        return cmd


def read_preset(path: Path) -> dict[str, float]:
    """A parameter file: `name = value` lines, `#` comments."""
    overrides = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, value = line.split("=")
            overrides[name.strip()] = float(value)
    return overrides


def nanocodec_with_preset(name: str) -> Nanocodec | None:
    """`nanocodec-<codec>@<preset>`: our encoder with presets/<preset>.toml."""
    base, _, preset = name.partition("@")
    if not preset or not base.startswith("nanocodec-"):
        return None
    for encoder in nanocodec_encoders():
        if encoder.name == base and (PRESETS_DIR / f"{preset}.toml").is_file():
            return Nanocodec(encoder.codec, encoder.quantizer_range, preset)
    return None


def nanocodec_encoders() -> list[Nanocodec]:
    # AV1 base_q_idx: below ~40 is far above our bitrate range.
    # H.264 and HEVC QP: below 10 likewise.
    return [
        Nanocodec(Codec.AV1, (40, 255)),
        Nanocodec(Codec.H264, (10, 51)),
        Nanocodec(Codec.HEVC, (10, 51)),
        Nanocodec(Codec.VVC, (10, 63)),
        Nanocodec(Codec.VP9, (1, 255)),
        Nanocodec(Codec.AV2, (1, 255)),
    ]
