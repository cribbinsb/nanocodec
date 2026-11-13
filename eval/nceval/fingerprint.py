"""Fingerprints that decide whether a stored result is still up to date.

A result is reusable only if nothing that could change it has changed. The
fingerprint of an encoder configuration hashes:

  - the pinned commit of the encoder (tools/anchors/versions.env), plus the
    source of tools/openh264-harness for the openh264 anchors;
  - the adapter's command template: its command line for a fixed dummy job,
    so any flag change in an adapter makes its old results stale;
  - the measurement chain: pinned decoders and libvmaf, and
    metrics.METRICS_REVISION;
  - the clip itself (clip_fingerprint): geometry, frame rate, frame count and
    a hash of its YUV data, so two different clips that share a name (e.g. a
    720p clip and its 360p version) can never reuse each other's results.

Results whose stored fingerprint differs are re-measured by
`nceval reference`. Bitstreams are cached under a directory named by the
fingerprint, so a stale bitstream is never reused either.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from .clips import Clip
from .codecs import Codec
from .config import Paths, find_tool, load_paths
from .encoders.base import CodingStructure, EncodeJob, EncoderAdapter
from .metrics import (DISPLAY_METRICS_REVISION, DISPLAY_PROFILE, LEGACY_METRICS_REVISION,
                      LEGACY_PROFILE, METRICS_REVISION, V1_PROFILE, vmaf_v1_protocol)

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_FILE = REPO_ROOT / "tools" / "anchors" / "versions.env"
OPENH264_HARNESS_SOURCE = REPO_ROOT / "tools" / "openh264-harness" / "openh264_harness.c"

# versions.env tool key per adapter-name prefix.
_TOOL_KEYS = {
    "openh264": "OPENH264", "x264": "X264", "x265": "X265", "kvazaar": "KVAZAAR",
    "svtav1": "SVTAV1", "aom": "LIBAOM", "vpx": "LIBVPX", "avm": "AVM",
}
# Adapters named "nanocodec-*" run our own binary (crates/nc-cli).
NANOCODEC_PREFIX = "nanocodec"
NANOCODEC_BINARY = "nanocodec"
_MEASUREMENT_KEYS = ("DAV1D", "JM", "HM", "LIBAOM", "VMAF")

_DUMMY_CLIP = Clip(name="fingerprint", yuv=Path("/in.yuv"), width=1280, height=720, fps=30.0,
                   frames=300, source_kbps=10000.0, content_class="static", split="train")


@lru_cache(maxsize=1)
def pinned_commits() -> dict[str, str]:
    commits = {}
    for line in VERSIONS_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "_COMMIT=" in line:
            key, value = line.split("=", 1)
            commits[key.removesuffix("_COMMIT")] = value
    return commits


def measurement_fingerprint(profile: str = V1_PROFILE, paths: Paths | None = None) -> str:
    commits = pinned_commits()
    parts = [f"{k}={commits.get(k, '?')}" for k in _MEASUREMENT_KEYS]
    if profile == LEGACY_PROFILE:
        parts.append(f"metrics={LEGACY_METRICS_REVISION}")
    elif profile == DISPLAY_PROFILE:
        # Exactly the pre-native V1 fingerprint: display results stay current.
        parts.append(f"metrics={DISPLAY_METRICS_REVISION}")
        parts.append(json.dumps(vmaf_v1_protocol(paths or load_paths(), display=True),
                                sort_keys=True))
        parts.append("default_secondary=psnr-xpsnr;float_ms_ssim=explicit-shortlist-only")
    elif profile == V1_PROFILE:
        parts.append(f"metrics={METRICS_REVISION}")
        parts.append(json.dumps(vmaf_v1_protocol(paths or load_paths()), sort_keys=True))
        parts.append("default_secondary=psnr;xpsnr=opt-in;float_ms_ssim=explicit-shortlist-only")
    else:
        raise ValueError(f"unknown measurement profile {profile!r}")
    return _digest(parts)


def encoder_fingerprint(adapter: EncoderAdapter, structure: CodingStructure,
                        paths: Paths, measurement_profile: str = V1_PROFILE) -> str:
    prefix = adapter.name.split("-")[0]
    if prefix == NANOCODEC_PREFIX:
        # Our own encoder has no pinned version: its identity is the binary.
        parts = [hashlib.sha256(find_tool(NANOCODEC_BINARY, paths).read_bytes()).hexdigest()]
    else:
        tool = _TOOL_KEYS.get(prefix)
        if tool is None:
            raise KeyError(f"no pinned version known for adapter {adapter.name}")
        parts = [f"{tool}={pinned_commits().get(tool, '?')}"]
    if prefix == "openh264":
        parts.append(hashlib.sha256(OPENH264_HARNESS_SOURCE.read_bytes()).hexdigest())
    lo, _ = adapter.quantizer_range
    dummy = EncodeJob(clip=_DUMMY_CLIP, quantizer=lo, structure=structure, frames=300,
                      bitstream=Path("/out" + adapter.codec.bitstream_suffix))
    command = adapter.command(dummy, paths)
    # These newly added references can coexist with an older system install.
    # Include the actual executable, not just the intended source pin.
    if prefix in ("vpx", "avm"):
        parts.append(f"encoder-binary={binary_digest(Path(command[0]))}")
    companions = {Codec.VP9: ("vpxdec", "ffmpeg"), Codec.AV2: ("avmdec", "dav2d"),
                  Codec.VVC: ("vvdecapp", "ffmpeg")}
    for decoder in companions.get(adapter.codec, ()):
        parts.append(f"{decoder}={binary_digest(find_tool(decoder, paths))}")
    parts.append(" ".join(Path(a).name if a.startswith("/") else a for a in command))
    parts.append(measurement_fingerprint(measurement_profile, paths))
    return _digest(parts)


def binary_digest(path: Path) -> str:
    """Cache executable identity while still noticing an in-place rebuild."""
    path = path.resolve()
    stat = path.stat()
    return _binary_digest(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=128)
def _binary_digest(path: Path, size: int, mtime_ns: int, ctime_ns: int) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


# Bytes hashed from each end of a clip: enough to tell different clips apart
# without reading hundreds of megabytes on every run.
_CLIP_SAMPLE_BYTES = 1 << 20


@lru_cache(maxsize=None)
def clip_fingerprint(clip: Clip) -> str:
    size = clip.yuv.stat().st_size
    digest = hashlib.sha256()
    with clip.yuv.open("rb") as f:
        digest.update(f.read(_CLIP_SAMPLE_BYTES))
        f.seek(max(0, size - _CLIP_SAMPLE_BYTES))
        digest.update(f.read(_CLIP_SAMPLE_BYTES))
    return _digest([f"{clip.width}x{clip.height}", f"{clip.fps:g}", str(clip.frames),
                    str(size), digest.hexdigest(), f"source_kbps={clip.source_kbps!r}",
                    clip.content_class, clip.split])


def curve_fingerprint(encoder_fp: str, clip: Clip) -> str:
    """Identity of one RD curve or speed point: encoder configuration + clip."""
    return _digest([encoder_fp, clip_fingerprint(clip)])


def _digest(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
