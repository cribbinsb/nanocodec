"""How much decoding a measured point does, and the verified-stream cache.

Correctness and evaluation are separate tasks (CLAUDE.md "Testing"):
conformance of our streams is checked when a change lands, by
tools/ci/landing_conformance.py (standard decoders and ours, bit-exact per
frame against the encoder's reconstruction). Evaluation (screens, boards,
sweeps, charts, tune) is our encoder plus the quality metrics on its
reconstruction, and decodes nothing by default. Three policies:

  off     the evaluation default (EVALUATION_GATE): no decoder runs on our
          streams. An anchor, which outputs no reconstruction, is still
          decoded once by the first standard decoder, for its metrics only.
  sample  explicit only. Our own decoder hash-checks every point against
          the encoder's reconstruction; the standard decoders run on the
          first point of each curve and one rotating quantizer, chosen per
          (binary, parameters, clip).
  full    explicit only: every stream goes through both standard decoders
          (plus our own decoder when given).

Independently of the policy, a stream whose exact bytes already passed both
standard decoders is not decoded by them again: `VerifiedStreams` keeps the
per-frame hashes (yuv.FRAME_HASH) of every fully verified stream, keyed by
the SHA-256 of the bitstream, its codec and geometry, and the identity of the
standard decoder binaries. A cache hit is compared with the reconstruction like a decode.

Every GateResult records its `level`, stored with the point's results:
`full`, `cached-verified`, `own-decoder` (standard decoders sampled out) or
`unchecked` (not decoded). A cached point is reused by any run whose policy
asks for no more checking than it had (`level_satisfies`).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from enum import Enum, IntEnum
from functools import lru_cache
from pathlib import Path


class GatePolicy(str, Enum):
    OFF = "off"
    SAMPLE = "sample"
    FULL = "full"


#: The policy of every evaluation tool unless asked otherwise (module docstring).
EVALUATION_GATE = GatePolicy.OFF


class GateLevel(str, Enum):
    FULL = "full"
    CACHED = "cached-verified"
    OWN_DECODER = "own-decoder"
    UNCHECKED = "unchecked"


class PointCheck(IntEnum):
    """What one point's gate runs, weakest first."""
    NONE = 0  # no decoder
    OWN = 1  # our own decoder against the reconstruction
    STANDARD = 2  # both standard decoders (and ours, when given)


#: The check each stored level stands for. A level not listed (results
#: stored before levels were recorded) passed both standard decoders.
_LEVEL_CHECK = {GateLevel.FULL: PointCheck.STANDARD, GateLevel.CACHED: PointCheck.STANDARD,
                GateLevel.OWN_DECODER: PointCheck.OWN, GateLevel.UNCHECKED: PointCheck.NONE}


def level_satisfies(level: str, check: PointCheck) -> bool:
    """Whether a point stored with gate `level` had at least `check`."""
    try:
        had = _LEVEL_CHECK[GateLevel(level)]
    except ValueError:
        had = PointCheck.STANDARD
    return had >= check


def point_check(policy: GatePolicy, combination: str, index: int, count: int) -> PointCheck:
    """The check point `index` of a `count`-point curve of our encoder gets."""
    if policy is GatePolicy.OFF:
        return PointCheck.NONE
    return (PointCheck.STANDARD if standard_decoders_needed(policy, combination, index, count)
            else PointCheck.OWN)


def uniform_check(policy: GatePolicy) -> PointCheck:
    """The check of a point outside any curve's sample (e.g. a rate probe)."""
    return {GatePolicy.OFF: PointCheck.NONE, GatePolicy.SAMPLE: PointCheck.OWN,
            GatePolicy.FULL: PointCheck.STANDARD}[policy]


def anchor_check(policy: GatePolicy) -> PointCheck:
    """An anchor's check: it has no reconstruction and no decoder of ours, so
    it is either decoded for its metrics only (off) or fully gated."""
    return PointCheck.NONE if policy is GatePolicy.OFF else PointCheck.STANDARD


def gate_arguments(check: PointCheck, own_decoder: Path | None) -> dict:
    """`own_decoder` and `standard` for nceval.point.run_point / gate_stream."""
    return {"own_decoder": own_decoder if check >= PointCheck.OWN else None,
            "standard": check is PointCheck.STANDARD}


def standard_decoders_needed(policy: GatePolicy, combination: str, index: int,
                             count: int) -> bool:
    """Whether point `index` of a `count`-point curve gets the standard decoders.

    `combination` names the (binary hash, parameters, clip) of the curve; the
    rotating extra point is a hash of it, so successive curves and parameter
    sets exercise different quantizers without any shared state.
    """
    if policy is GatePolicy.OFF:
        return False
    if policy is GatePolicy.FULL or index == 0 or count <= 1:
        return True
    rotating = 1 + int(hashlib.sha256(combination.encode()).hexdigest()[:8], 16) % (count - 1)
    return index == rotating


def combination_key(binary_sha256: str, overrides: dict, clip: str) -> str:
    return json.dumps([binary_sha256, sorted(overrides.items()), clip])


def verified_root() -> Path:
    default = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data")) / "run" / "verified"
    return Path(os.environ.get("NC_VERIFIED_DIR", default))


@lru_cache(maxsize=None)
def _file_sha256(path: Path, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, memoised per (path, size, mtime) for large binaries."""
    stat = path.stat()
    return _file_sha256(path.resolve(), stat.st_size, stat.st_mtime_ns)


class VerifiedStreams:
    """Per-frame hashes of streams that passed both standard decoders."""

    def __init__(self, decoder_identity: str, root: Path | None = None):
        self._identity = decoder_identity
        self._root = root or verified_root()

    def _path(self, stream_sha256: str, codec: str, width: int, height: int,
              frames: int) -> Path:
        key = hashlib.sha256(json.dumps(
            [stream_sha256, codec, width, height, frames, self._identity]).encode()).hexdigest()
        return self._root / codec / key[:2] / f"{key}.json"

    def lookup(self, stream_sha256: str, codec: str, width: int, height: int,
               frames: int, held: bool = False) -> list[str] | None:
        """The verified hashes of a stream of `frames` input frames; `held`:
        frames may have been held, so fewer were decoded (nceval.conformance)."""
        try:
            record = json.loads(self._path(stream_sha256, codec, width, height, frames).read_text())
        except (OSError, ValueError):
            return None
        digests = record.get("frame_hashes")
        if not isinstance(digests, list):
            return None
        return digests if len(digests) == frames or (held and len(digests) < frames) else None

    def store(self, stream_sha256: str, codec: str, width: int, height: int,
              frame_hashes: list[str], frames: int | None = None) -> None:
        """Record `frame_hashes` for a stream of `frames` input frames
        (default: one per decoded frame)."""
        frames = len(frame_hashes) if frames is None else frames
        path = self._path(stream_sha256, codec, width, height, frames)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                         prefix=".pending-") as output:
            json.dump({"stream_sha256": stream_sha256, "codec": codec,
                       "width": width, "height": height,
                       "decoders": self._identity, "frame_hashes": frame_hashes}, output)
        Path(output.name).replace(path)
