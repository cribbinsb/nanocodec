"""Adapter for our own encoder: the `nanocodec` CLI from crates/nc-cli.

Unlike the anchors, nanocodec outputs its reconstruction, so evaluation
scores it directly and decodes nothing (nceval.gatepolicy); an explicit gate
checks it against the decoders.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class RungRename:
    """A rung renamed in the ladder (frozen-build compatibility): in a frozen build (a
    presets/ directory beside a COMMIT file, as the build cache keeps them) from before
    it (one without `marker`, a glob under presets/), today's `new` is the build's `old`
    file and `--preset old`, and today's `old` (a rung that took the name later) does
    not exist. `codec` limits it to one codec's directory; `text`, when given, must be
    in the marker file (a rename that kept every file name)."""
    new: str
    old: str
    marker: str
    codec: str | None = None
    text: str | None = None

    def predates(self, presets: Path, codec: str) -> bool:
        """A tree that is not a frozen build (the repository's, a test's) has today's
        names."""
        frozen = (presets.parent / FROZEN_BUILD_MARKER).is_file()
        return frozen and self.codec in (None, codec) and not any(
            self.text is None or self.text in path.read_text() for path in presets.glob(self.marker))


#: The file beside a frozen build's presets/ (tools/dev/build_cache.py).
FROZEN_BUILD_MARKER = "COMMIT"
#: Every rung rename, oldest first (docs/handoff/presets.md "The ladder").
RUNG_RENAMES = (
    RungRename("slower", "slow", "**/slower.toml"),  # 2026-11-21: slow -> slower
    RungRename("faster", "fast", "vvc/faster.toml", "vvc"),  # 2026-11-21: VVC fast -> faster
    # 2026-11-21: H.264 fast -> faster; the old faster is retired (no name today)
    RungRename("faster", "fast", "h264/faster.toml", "h264", "`fast` until 2026-11-21"),
    RungRename("veryslow", "slower", "hevc/veryslow.toml", "hevc"),  # 2026-11-21
    RungRename("veryslow", "slower", "vvc/veryslow.toml", "vvc"),  # 2026-11-21
)
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


#: The quality rungs from the slowest, for "the codec's slowest rung" (nc_encoder::presets).
SLOWEST_FIRST = ("veryslow", "slower", "slow")
SLOWEST_SHARED = "slowest.toml"


def _shared_bundle(path: Path) -> Path | None:
    """The shared bundle under rung file `path` (presets/<codec>/<rung>.toml), as
    nc_encoder::presets applies it: presets/slowest.toml under the codec's slowest rung;
    in a frozen build from before it, the same-named presets/<rung>.toml."""
    if path.parent.name not in CODEC_DIRS:
        return None
    presets = path.parent.parent
    if (presets / SLOWEST_SHARED).is_file():
        slowest = next((r for r in SLOWEST_FIRST if (path.parent / f"{r}.toml").is_file()), None)
        return presets / SLOWEST_SHARED if path.stem == slowest else None
    legacy = presets / path.name
    return legacy if legacy.is_file() else None


def read_preset(path: Path) -> dict[str, float]:
    """A parameter file: `name = value` lines, `#` comments. A codec's rung file
    (presets/<codec>/<rung>.toml) lies over its shared bundle where it has one
    (`_shared_bundle`)."""
    shared = _shared_bundle(path)
    overrides = read_preset(shared) if shared else {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, value = line.split("=")
            overrides[name.strip()] = float(value)
    return overrides


def param_text(value: float) -> str:
    """A parameter value as `--param` text, exactly the value the preset file holds.

    `:g` keeps six significant digits: h264 fast's 1.3333334 became 1.33333, so
    final checks, boards and charts measured a slightly different setting than the
    shipped preset (found 2026-11-21). Values `:g` writes exactly keep that form
    (their cached points stay valid); the rest take Python's round-trip repr."""
    short = f"{value:g}"
    return short if float(short) == value else repr(float(value))


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


def _build_stem(presets: Path, codec: Codec | str, stem: str) -> str | None:
    """The stem build `presets` has for today's `stem` (RUNG_RENAMES, newest first), or
    None when the build has no such rung."""
    name = _codec_name(codec)
    for rename in reversed(RUNG_RENAMES):
        if rename.predates(presets, name):
            if stem == rename.new:
                stem = rename.old
            elif stem == rename.old:
                return None
    return stem


def _today_stem(presets: Path, codec: str, stem: str) -> str | None:
    """Today's name of build `presets`' file `stem` (RUNG_RENAMES, oldest first), or None
    for a rung retired since (a file the build already had under a rename's new name)."""
    return renamed_stem(codec, stem, [r for r in RUNG_RENAMES if r.predates(presets, codec)])


def renamed_stem(codec: str, stem: str, renames=RUNG_RENAMES) -> str | None:
    """`stem` of `codec` named before `renames` (oldest first) by today's name, or None
    for a rung they retire. A dated record names its rungs as of its day: the history
    page's stored frames pass the renames made since they were written."""
    for rename in renames:
        if rename.codec in (None, codec):
            if stem == rename.new:
                return None
            if stem == rename.old:
                stem = rename.new
    return stem


def preset_path(presets: Path, codec: Codec | str, stem: str) -> Path:
    """presets/<codec>/<stem>.toml, or a frozen build's renamed or old flat file (a path
    that does not exist when the build has no such rung). The renames apply first, to
    a flat build too: its h264-fast.toml is today's `faster`, as in every later build."""
    built = _build_stem(presets, codec, stem)
    if built is None:
        return presets / _codec_name(codec) / f"{stem}.absent"
    if _is_old_layout(presets, codec, built):
        return presets / f"{_old_flat_stem(codec, built)}.toml"
    return presets / _codec_name(codec) / f"{built}.toml"


def preset_values(presets: Path, codec: Codec | str, stem: str | None) -> dict[str, float]:
    """Preset `stem`'s parameter values ({} for medium, the engine defaults)."""
    if not stem or stem == MEDIUM:
        return {}
    if _build_stem(presets, codec, stem) is None:
        raise ValueError(f"{presets}: this build has no {_codec_name(codec)} rung {stem!r}")
    return read_preset(preset_path(presets, codec, stem))


def recorded_values(presets: Path, codec: Codec | str, stem: str | None) -> dict[str, float]:
    """Preset `stem`'s values as build `presets` itself named its files, without
    RUNG_RENAMES: what a build record's family meant when its points were measured (a
    record keeps the names of its day: 0.5.0's `nanocodec-av1@slow` is today's
    `slower`, and that build has no rung today's `slow` names)."""
    if not stem or stem == MEDIUM:
        return {}
    path = (presets / f"{_old_flat_stem(codec, stem)}.toml" if _is_old_layout(presets, codec, stem)
            else presets / _codec_name(codec) / f"{stem}.toml")
    if not path.is_file():
        raise ValueError(f"{presets}: this build has no {_codec_name(codec)} preset file {stem!r}")
    return read_preset(path)


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
    built = _build_stem(presets, codec, stem)
    if built is None:
        raise ValueError(f"{presets}: this build has no {_codec_name(codec)} rung {stem!r}")
    if _is_old_layout(presets, codec, built):
        return ("--preset", _old_flat_stem(codec, built))
    layers, _, speed = built.partition("-")
    if speed and layers in SCALABILITY_STEM_MODES:
        # medium is named explicitly here: the file is l2-medium, not "no preset".
        return ("--scalability", mode or SCALABILITY_STEM_MODES[layers], "--preset", speed)
    return ("--preset", built)


def shipped_presets(codec: Codec | str, presets: Path = PRESETS_DIR) -> list[str]:
    """Every shipped preset stem of `codec`: medium (no file) and presets/<codec>/*.toml,
    by today's names (RUNG_RENAMES: a frozen build's `slow` is `slower`)."""
    name = _codec_name(codec)
    directory = presets / name
    if not directory.is_dir():
        return [MEDIUM]
    today = (_today_stem(presets, name, p.stem) for p in directory.glob("*.toml"))
    return [MEDIUM, *sorted(stem for stem in today if stem)]


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
