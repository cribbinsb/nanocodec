"""Locations of tools and data used by the harness.

Everything machine-specific is resolved here, so the rest of the harness never
hard-codes a path. Defaults can be overridden with environment variables:

    NC_DATA_ROOT     root for clips, encodes and results
                     (default: ~/nanocodec-data; /mldata is nearly full)
    NC_ANCHORS       install prefix of tools/anchors/build_anchors.sh
                     (default: $NC_DATA_ROOT/anchors/<machine>)
    NC_NANOCODEC_BIN directory holding our own encoder/decoder binaries
                     (default: the cargo release output, target/release)
    NC_WORK          scratch and kept-bitstream root (default: $NC_DATA_ROOT/work);
                     a private run (e.g. a sanity report) uses its own, so
                     another agent's sweep never replaces its streams
    NC_RESULTS_DB    results database (default: $NC_DATA_ROOT/results/results.sqlite)
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_root: Path
    anchors: Path
    nanocodec_bin: Path
    work_root: Path | None = None
    results_db_path: Path | None = None

    @property
    def clips(self) -> Path:
        return self.data_root / "clips"

    @property
    def work(self) -> Path:
        """Scratch space for bitstreams and decoded output of a run."""
        return self.work_root or self.data_root / "work"

    @property
    def results_db(self) -> Path:
        return self.results_db_path or self.data_root / "results" / "results.sqlite"


def load_paths() -> Paths:
    data_root = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
    anchors = Path(os.environ.get("NC_ANCHORS", data_root / "anchors" / platform.machine()))
    repo_root = Path(__file__).resolve().parents[2]
    nanocodec_bin = Path(os.environ.get("NC_NANOCODEC_BIN", repo_root / "target" / "release"))
    work = os.environ.get("NC_WORK")
    results_db = os.environ.get("NC_RESULTS_DB")
    return Paths(data_root=data_root, anchors=anchors, nanocodec_bin=nanocodec_bin,
                 work_root=Path(work) if work else None,
                 results_db_path=Path(results_db) if results_db else None)


def find_tool(name: str, paths: Paths | None = None) -> Path:
    """Locate a tool binary: anchors prefix first, then our build, then $PATH.

    Raises FileNotFoundError with a hint, rather than letting a subprocess fail
    later with a less helpful message.
    """
    paths = paths or load_paths()
    for directory in (paths.anchors / "bin", paths.nanocodec_bin):
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(
        f"tool '{name}' not found in {paths.anchors / 'bin'}, {paths.nanocodec_bin} "
        f"or $PATH (run tools/anchors/build_anchors.sh?)"
    )


#: The reference-encoder builds the speed charts time, per timing machine, relative to
#: the data root: tools/anchors/build_anchors.sh --opt installs (each encoder with the
#: LTO and PGO its own build offers; docs/PROCEDURES.md "Reference builds"). This is the
#: one place that switches every timing tool (aws_timing.py, fixedqp_timing.py,
#: release.py) to another build. The Jetson keeps the board's own build (its gcc
#: predates x265's SVE2 kernels, which its Cortex-A78AE lacks anyway). Quality encodes
#: (on x86) run QUALITY_MACHINE's build too (cbr.anchor_tool, `openh264_harness`), so a
#: reference's quality and speed come from one binary (owner rule, 2026-11-23: the
#: timed x86 build codes other bytes for x264 medium than the 2026-11-06 one).
REFERENCE_BUILDS = {"x86_64": Path("anchors/x86_64-opt"),
                    "graviton": Path("anchors/aarch64-graviton-opt"),
                    "jetson": Path("anchors/aarch64")}
#: The machine whose reference build the quality encodes run (they run on x86 boxes).
QUALITY_MACHINE = "x86_64"
#: The openh264 driver with --bitrate where QUALITY_MACHINE's build has none.
OPENH264_HARNESS = Path("work/ratecontrol/bin/openh264-harness")


def reference_bin(machine: str, data_root: Path | None = None) -> Path:
    """The `bin` directory of the reference build timed on `machine` (REFERENCE_BUILDS)."""
    root = data_root or Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
    return root / REFERENCE_BUILDS[machine] / "bin"


def quality_reference(name: str, data_root: Path | None = None) -> Path | None:
    """Reference program `name` in QUALITY_MACHINE's timed build, if that build has it."""
    candidate = reference_bin(QUALITY_MACHINE, data_root) / name
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def openh264_harness(data_root: Path | None = None) -> Path:
    """The CBR charts' openh264 driver: the timed build's (`quality_reference`), else
    OPENH264_HARNESS under the data root."""
    root = data_root or Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
    return quality_reference("openh264-harness", root) or root / OPENH264_HARNESS


#: The build_anchors.sh tool that installs an anchor binary, where the names differ.
ANCHOR_BUILD_TOOLS = {"ffmpeg-decode": "ffmpeg", "TAppDecoder": "hm", "ldecod": "jm",
                      "vvdecapp": "vvdec", "aomdec": "libaom", "aomenc": "libaom",
                      "vpxdec": "libvpx", "vpxenc": "libvpx", "avmdec": "avm", "avmenc": "avm",
                      "vvencFFapp": "vvenc",
                      "SvtAv1EncApp": "svtav1", "openh264-harness": "harness"}


def find_anchor_tool(name: str, paths: Paths | None = None) -> Path:
    """Locate a pinned anchor binary in the anchors prefix only, never $PATH.

    For tools whose exact build is part of a gate's meaning (a decoder pinned in
    tools/anchors/versions.env), where a stray system install must not be used.
    """
    paths = paths or load_paths()
    candidate = paths.anchors / "bin" / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    raise FileNotFoundError(
        f"anchor tool '{name}' not found in {paths.anchors / 'bin'} "
        f"(run tools/anchors/build_anchors.sh {ANCHOR_BUILD_TOOLS.get(name, name)}?)"
    )
