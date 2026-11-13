#!/usr/bin/env python3
"""Add realistic synthetic sensor noise to clean clips (docs/ARCHITECTURE.md §1).

Purpose: measure whether an encoder recovers a *clean* picture from noisy
input. The noisy clip is encoded; quality is scored against the clean
original (`clean_reference` in the clip set), so removing noise is rewarded
instead of penalised.

Noise model (per pixel, independent in time):

  1. Luma is mapped to an approximately linear light value
         L = ((Y - 16) / 219) ** GAMMA            (limited range, GAMMA = 2.2)
  2. Sensor noise in linear light: shot noise (variance proportional to the
     signal) plus read noise (constant variance):
         sigma_L(L) = sqrt(shot * L + read ** 2)
  3. Demosaicing makes sensor noise slightly spatially correlated, so the
     unit Gaussian field is smoothed with a [1, 2, 1] / 4 kernel in x and y,
     then rescaled back to unit variance.
  4. L' = L + sigma_L * field, clipped to [0, 1], mapped back to Y.
  5. Chroma noise: an independent smoothed Gaussian field on each chroma
     plane, with a standard deviation of CHROMA_RATIO times the local luma
     noise (in code values), sampled at chroma resolution.

Because of the gamma mapping, dark areas get more noise in code values than
bright ones, which is what real low-light footage looks like.

Gain levels (luma noise standard deviation at mid grey, Y = 100):

  low   shot 0.0007, read 0.0015  ->  about 3 code values
  high  shot 0.0040, read 0.0030  ->  about 7 code values

Usage (after clipset/extract.py has written the clip set):

  python3 clipset/add_noise.py ~/nanocodec-data/clips/main.json \\
      --clip static_meva_hospital --clip pan_jaad_0117 ...

For every listed clip and gain this writes <clip>_noise_<gain>.yuv next to the
original and adds a `synthetic_noise` entry (same split as the source) to the
clip set. Previous synthetic entries are replaced and existing complete
outputs are reused, so the script is idempotent. The random seed is fixed per
(clip, gain), so outputs are reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GAMMA = 2.2
CHROMA_RATIO = 0.6
# The [1,2,1]/4 filter keeps 6/16 of the variance per axis, so after smoothing
# in x and y a unit field has standard deviation sqrt((6/16)^2) = 6/16.
SMOOTHING_STD = 6.0 / 16.0


@dataclass(frozen=True)
class Gain:
    name: str
    shot: float
    read: float


GAINS = (Gain("low", shot=0.0007, read=0.0015), Gain("high", shot=0.0040, read=0.0030))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clipset", type=Path, help="clip set JSON written by extract.py")
    parser.add_argument("--clip", action="append", required=True, help="clean source clip name")
    parser.add_argument("--gain", action="append", choices=[g.name for g in GAINS],
                        help="noise level(s) to generate (default: all)")
    args = parser.parse_args()
    gains = [g for g in GAINS if not args.gain or g.name in args.gain]

    data_root = args.clipset.parent.parent
    clipset = json.loads(args.clipset.read_text())
    clips = [c for c in clipset["clips"] if c["content_class"] != "synthetic_noise"]
    by_name = {c["name"]: c for c in clips}

    for name in args.clip:
        source = by_name.get(name)
        if source is None:
            print(f"clip {name!r} not in {args.clipset}", file=sys.stderr)
            return 1
        for gain in gains:
            clips.append(_make_noisy(source, gain, data_root))
            print(f"{clips[-1]['name']}: ready")

    clipset["clips"] = clips
    args.clipset.write_text(json.dumps(clipset, indent=2))
    return 0


def _make_noisy(source: dict, gain: Gain, data_root: Path) -> dict:
    clean = data_root / source["yuv"]
    noisy = clean.with_name(f"{clean.stem}_noise_{gain.name}.yuv")
    w, h = source["width"], source["height"]
    frame_size = w * h * 3 // 2
    seed = int.from_bytes(hashlib.sha256(f"{source['name']}/{gain.name}".encode()).digest()[:8],
                          "little")
    rng = np.random.default_rng(seed)

    # The noise is deterministic (fixed seed), so an existing complete output
    # is identical to what would be generated: skip it.
    if not (noisy.is_file() and noisy.stat().st_size == clean.stat().st_size):
        frames = np.memmap(clean, dtype=np.uint8, mode="r").reshape(-1, frame_size)
        partial = noisy.with_name("partial-" + noisy.name)
        with partial.open("wb") as out:
            for frame in frames:
                out.write(add_noise_to_frame(frame, w, h, gain, rng).tobytes())
        partial.rename(noisy)

    entry = dict(source)
    entry.update({
        "name": f"{source['name']}_noise_{gain.name}",
        "yuv": str(noisy.relative_to(data_root)),
        "content_class": "synthetic_noise",
        "clean_reference": source["yuv"],
        "noise": {"gain": gain.name, "shot": gain.shot, "read": gain.read, "seed": seed},
        "sha256": None,
    })
    return entry


def add_noise_to_frame(frame: np.ndarray, w: int, h: int, gain: Gain,
                       rng: np.random.Generator) -> np.ndarray:
    luma = frame[: w * h].reshape(h, w).astype(np.float64)
    chroma_size = (w // 2) * (h // 2)
    u = frame[w * h: w * h + chroma_size].reshape(h // 2, w // 2).astype(np.float64)
    v = frame[w * h + chroma_size:].reshape(h // 2, w // 2).astype(np.float64)

    linear = (np.clip((luma - 16.0) / 219.0, 0.0, 1.0)) ** GAMMA
    sigma_linear = np.sqrt(gain.shot * linear + gain.read ** 2)
    noisy_linear = np.clip(linear + sigma_linear * _correlated_field(rng, h, w), 0.0, 1.0)
    noisy_luma = 16.0 + 219.0 * noisy_linear ** (1.0 / GAMMA)

    # Local luma noise in code values, at chroma resolution, drives chroma noise.
    sigma_luma = np.abs(noisy_luma - luma)  # cheap local estimate, smoothed below
    sigma_chroma = CHROMA_RATIO * _downsample2(_smooth(sigma_luma)) * np.sqrt(np.pi / 2)
    noisy_u = u + sigma_chroma * _correlated_field(rng, h // 2, w // 2)
    noisy_v = v + sigma_chroma * _correlated_field(rng, h // 2, w // 2)

    planes = (noisy_luma, noisy_u, noisy_v)
    return np.concatenate([np.clip(np.rint(p), 0, 255).astype(np.uint8).ravel() for p in planes])


def _correlated_field(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Unit-variance Gaussian field with demosaic-like short-range correlation."""
    return _smooth(rng.standard_normal((h, w))) / SMOOTHING_STD


def _smooth(x: np.ndarray) -> np.ndarray:
    """Separable [1, 2, 1] / 4 filter with edge replication."""
    p = np.pad(x, 1, mode="edge")
    x = (p[:, :-2] + 2.0 * p[:, 1:-1] + p[:, 2:]) / 4.0
    x = (x[:-2, :] + 2.0 * x[1:-1, :] + x[2:, :]) / 4.0
    return x


def _downsample2(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0] // 2, 2, x.shape[1] // 2, 2).mean(axis=(1, 3))


if __name__ == "__main__":
    sys.exit(main())
