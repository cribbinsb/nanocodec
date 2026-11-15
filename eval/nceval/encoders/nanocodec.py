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

#: The shipped presets, repository-relative: in the crate that embeds them.
PRESETS_PATH = "crates/nc-encoder/presets"
#: Where they were before 2026-11-15 (commits older builds are made from).
OLD_PRESETS_PATH = "presets"
PRESETS_DIR = Path(__file__).resolve().parents[3] / PRESETS_PATH


def presets_in(tree: Path) -> Path:
    """The presets directory of a checked-out tree of any commit."""
    new = tree / PRESETS_PATH
    return new if new.is_dir() else tree / OLD_PRESETS_PATH


#: The per-codec preset directories (presets/<codec>/).
CODEC_DIRS = frozenset(codec.value for codec in Codec)
#: The codec-independent speed ladder, fastest first. `medium` is each codec's
#: Default (the engine defaults): it has no file and is what no `@stem` means.
LADDER = ("veryfast", "faster", "fast", "medium")
MEDIUM = "medium"
#: A preset file stem `l2-fast` is the ladder name `fast` in a scalability
#: mode: `--scalability <mode> --preset fast`. The mode named here is the one
#: the file is coded at by default; L2T3, L2T1_KEY etc. pick the same file.
SCALABILITY_STEM_MODES = {"l2": "L2T1", "l3": "L3T1"}
#: Frozen-build compatibility: builds recorded before the preset rename
#: (2026-11-15; docs/results/bd-vs-speed-builds.json) keep their presets flat
#: as presets/<codec>-<old suffix>.toml and their binaries know only that old
#: `--preset` name. New stem -> old suffix; a stem not listed kept its name
#: (fast, faster -> <codec>-fast, <codec>-faster). Used only when a presets
#: directory lacks <codec>/<stem>.toml. Delete once every chart family is
#: rebuilt past the rename.
OLD_FLAT_SUFFIX = {"veryfast": "fastest", "l2-medium": "svc", "l2-fast": "svc-fast",
                   "l3-medium": "svc3", "l3-fast": "svc3-fast"}
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
        self.overrides = read_preset(preset_path(PRESETS_DIR, codec, preset)) if preset else {}

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
    """A parameter file: `name = value` lines, `#` comments. A codec's rung file
    (presets/<codec>/<rung>.toml) lies over the rung's shared bundle
    (presets/<rung>.toml) where there is one, as nc_encoder::presets applies it."""
    shared = path.parent.parent / path.name
    overrides = read_preset(shared) if path.parent.name in CODEC_DIRS and shared.is_file() else {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, value = line.split("=")
            overrides[name.strip()] = float(value)
    return overrides


def current_family_name(name: str) -> str:
    """A family name recorded before the rename (`nanocodec-vp9@vp9-fastest`)
    in today's form (`nanocodec-vp9@veryfast`); any other name unchanged. For
    reading dated results, which keep their old names."""
    base, at, rest = name.partition("@")
    codec = base.removeprefix("nanocodec-")
    if not at or base == codec or not rest.startswith(f"{codec}-"):
        return name
    old, plus, variant = rest.removeprefix(f"{codec}-").partition("+")
    new = {suffix: stem for stem, suffix in OLD_FLAT_SUFFIX.items()}.get(old, old)
    return f"{base}@{new}{plus}{variant}"


def _codec_name(codec: Codec | str) -> str:
    return codec.value if isinstance(codec, Codec) else codec


def _old_flat_stem(codec: Codec | str, stem: str) -> str:
    """The pre-rename flat file stem / `--preset` name (OLD_FLAT_SUFFIX)."""
    return f"{_codec_name(codec)}-{OLD_FLAT_SUFFIX.get(stem, stem)}"


def _is_old_layout(presets: Path, codec: Codec | str, stem: str) -> bool:
    """Whether `presets` is a frozen pre-rename build's flat directory for this preset."""
    return (not (presets / _codec_name(codec) / f"{stem}.toml").is_file()
            and (presets / f"{_old_flat_stem(codec, stem)}.toml").is_file())


def preset_path(presets: Path, codec: Codec | str, stem: str) -> Path:
    """presets/<codec>/<stem>.toml, or a frozen build's old flat file."""
    if _is_old_layout(presets, codec, stem):
        return presets / f"{_old_flat_stem(codec, stem)}.toml"
    return presets / _codec_name(codec) / f"{stem}.toml"


def preset_values(presets: Path, codec: Codec | str, stem: str | None) -> dict[str, float]:
    """Preset `stem`'s parameter values ({} for medium, the engine defaults)."""
    return read_preset(preset_path(presets, codec, stem)) if stem and stem != MEDIUM else {}


def preset_cli_args(codec: Codec | str, stem: str | None, presets: Path = PRESETS_DIR,
                    mode: str | None = None) -> tuple[str, ...]:
    """The `nanocodec encode` options selecting preset file `stem` of `codec`.

    `fast` -> `--preset fast`; `l2-fast` -> `--scalability L2T1 --preset fast`
    (l2-medium: `--preset medium`; `mode`, e.g. L2T3_KEY, replaces L2T1);
    medium (or none) -> nothing. A frozen pre-rename build (its `presets` dir
    in the old flat layout) gets its old `--preset <codec>-<suffix>` name.
    """
    if not stem or stem == MEDIUM:
        return ()
    if _is_old_layout(presets, codec, stem):
        return ("--preset", _old_flat_stem(codec, stem))
    layers, _, speed = stem.partition("-")
    if speed and layers in SCALABILITY_STEM_MODES:
        # medium is named explicitly here: the file is l2-medium, not "no preset".
        return ("--scalability", mode or SCALABILITY_STEM_MODES[layers], "--preset", speed)
    return ("--preset", stem)


def shipped_presets(codec: Codec | str, presets: Path = PRESETS_DIR) -> list[str]:
    """Every shipped preset stem of `codec`: medium (no file) and presets/<codec>/*.toml."""
    directory = presets / _codec_name(codec)
    return [MEDIUM, *sorted(p.stem for p in directory.glob("*.toml"))] if directory.is_dir() \
        else [MEDIUM]


def nanocodec_with_preset(name: str) -> Nanocodec | None:
    """`nanocodec-<codec>@<stem>`: our encoder with presets/<codec>/<stem>.toml."""
    base, _, preset = name.partition("@")
    if not preset or not base.startswith("nanocodec-"):
        return None
    for encoder in nanocodec_encoders():
        if encoder.name == base and preset_path(PRESETS_DIR, encoder.codec, preset).is_file():
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
