#!/usr/bin/env python3
"""Syntax-coverage stress test: random legal decisions vs every decoder.

Normal encodes exercise only the syntax their mode decision likes, and our
encoder and decoder share prediction code, so a bug on a rarely chosen path
(e.g. AV1's MV-reference scan next to 8x8 inter blocks) agrees with itself
and passes our own checks. This test encodes small clips with
`debug.random_decisions=<seed>` (random partitions down to the minimum
block, intra/inter mix, cheap/random/far/sub-sample vectors, block QP
offsets) combined with the real-use features of nceval.stress_matrix
(temporal layers, LTR with a reported loss, intra refresh, CBR, AV1 loop
filter), and requires every stream to decode bit-exactly, by per-frame
hash, with both standard decoders and our own decoder, matching the
encoder's reconstruction:

    H.264  JM + ffmpeg     HEVC  HM + ffmpeg     AV1  aomdec + dav1d (ffmpeg)
    VP9    vpxdec + ffmpeg (flat structure)

Clip sizes include ones that are not multiples of 16 or 64, and a 1884-wide
strip (picture-edge and padding paths). Seed n picks one feature
combination per codec (nceval.stress_matrix); a run over as many seeds as
there are combinations covers every one of them at every size.

    tools/ci/syntax_stress.py --quick            # CI: 16 seeds, about a minute
    tools/ci/syntax_stress.py --seeds 1-500      # nightly sweep
    tools/ci/syntax_stress.py --codecs av1 --spatial --seeds 1-48
                                                 # AV1 spatial SVC: every mode, scale and interval

The encoder binary comes from NC_NANOCODEC_BIN (default target/release).
A failure keeps its bitstream under the work directory and prints the
command that reproduces it; the exit status is non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from nceval.codecs import Codec  # noqa: E402
from nceval.config import Paths, find_tool, load_paths  # noqa: E402
from nceval.conformance import (  # noqa: E402
    gate_av1_operating_points,
    gate_vp9_operating_points,
)
from nceval.framepipe import ProcessFailed  # noqa: E402
from nceval.metrics import StreamSpec  # noqa: E402
from nceval.point import run_point  # noqa: E402
from nceval.stress_matrix import (  # noqa: E402
    FRAMES, INTRA_PERIOD, RESIZE_DOWN, RESIZE_UP, StressCase)
from nceval.yuv import YuvFormat  # noqa: E402

QUICK_SEEDS = range(1, 17)
MAX_WORKERS = 8
SOURCE_STEP = 3  # every third source frame: more motion per coded frame
CROP_DRIFT = (8, 4)  # per output frame: a slow pan on top of the scene motion


@dataclass(frozen=True)
class StressSize:
    """A crop of a crowd scene (real motion for search and predicted vectors)."""
    width: int
    height: int
    source: str
    source_size: tuple[int, int]
    fps: float
    origin: tuple[int, int]


SIZES = (
    StressSize(128, 96, "clips/main/crowd_mot17_09.yuv", (1280, 720), 30.0, (400, 300)),
    StressSize(136, 104, "clips/main/crowd_mot17_09.yuv", (1280, 720), 30.0, (400, 300)),
    StressSize(200, 120, "clips/main/crowd_mot17_09.yuv", (1280, 720), 30.0, (400, 300)),
    # Full width of a 1884x720 clip: not a multiple of 8, 16 or 64.
    StressSize(1884, 64, "clips/main/crowd_mot20_06.yuv", (1884, 720), 25.0, (0, 300)),
)


@dataclass
class Outcome:
    case: str
    passed: bool
    reason: str = ""
    repro: str = ""


# Every codec codes any even size.
SIZE_MULTIPLE: dict[Codec, int] = {}


def codes_size(codec: Codec, size: StressSize) -> bool:
    multiple = SIZE_MULTIPLE.get(codec, 2)
    return size.width % multiple == 0 and size.height % multiple == 0


def stress_clip(size: StressSize, paths: Paths, directory: Path) -> Path:
    """The crop as a small YUV file (cached; at most a few MB)."""
    path = directory / f"stress_{size.width}x{size.height}_{FRAMES}f.yuv"
    if path.is_file():
        return path
    sw, sh = size.source_size
    luma, frame = sw * sh, sw * sh * 3 // 2
    drift_x = CROP_DRIFT[0] if size.width + CROP_DRIFT[0] * FRAMES <= sw - size.origin[0] else 0
    out = bytearray()
    with (paths.data_root / size.source).open("rb") as src:
        for i in range(FRAMES):
            src.seek(i * SOURCE_STEP * frame)
            data = src.read(frame)
            x0, y0 = size.origin[0] + drift_x * i, size.origin[1] + CROP_DRIFT[1] * i
            for r in range(size.height):
                start = (y0 + r) * sw + x0
                out += data[start:start + size.width]
            for plane in (luma, luma + luma // 4):
                for r in range(size.height // 2):
                    start = plane + (y0 // 2 + r) * (sw // 2) + x0 // 2
                    out += data[start:start + size.width // 2]
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(bytes(out))
    temporary.replace(path)
    return path


def switched_clip(size: StressSize, small: tuple[int, int], paths: Paths, directory: Path) -> Path:
    """The crop with its frames from RESIZE_DOWN to RESIZE_UP replaced by a
    `small` crop (the "resize" structure's source; cached)."""
    path = directory / f"stress_{size.width}x{size.height}_to_{small[0]}x{small[1]}_{FRAMES}f.yuv"
    if path.is_file():
        return path
    full = stress_clip(size, paths, directory).read_bytes()
    reduced = stress_clip(replace(size, width=small[0], height=small[1]), paths, directory).read_bytes()
    full_bytes = YuvFormat(size.width, size.height).frame_size
    small_bytes = YuvFormat(*small).frame_size
    out = bytearray()
    for i in range(FRAMES):
        if RESIZE_DOWN <= i < RESIZE_UP:
            out += reduced[i * small_bytes:(i + 1) * small_bytes]
        else:
            out += full[i * full_bytes:(i + 1) * full_bytes]
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(bytes(out))
    temporary.replace(path)
    return path


def encode_command(binary: Path, case: StressCase, clip: Path, stream: Path,
                   layer_md5: Path | None = None):
    extra = ["--layer-md5", str(layer_md5)] if layer_md5 is not None else []

    def command(recon: str) -> list[str]:
        return [str(binary), "encode", "--codec", case.codec.value, "--input", str(clip),
                "--width", str(case.width), "--height", str(case.height),
                "--fps", f"{case.fps:g}", "--frames", str(FRAMES),
                "--intra-period", str(INTRA_PERIOD), *case.encoder_flags(), *extra,
                "--output", str(stream), "--display-recon", recon]
    return command


def run_case(case: StressCase, clip: Path, binary: Path, paths: Paths, work: Path) -> Outcome:
    stream = work / f"{case.name}{case.codec.bitstream_suffix}"
    layer_md5 = work / f"{case.name}.layers.csv" if case.is_spatial else None
    command = encode_command(binary, case, clip, stream, layer_md5)
    repro = " ".join(command("recon.yuv"))
    spec = StreamSpec(reference=clip, fmt=case.frame_format,
                      frames=case.coded_frames, fps=case.fps, paths=None)
    try:
        result = run_point(command, stream, case.codec, spec, paths, own_decoder=binary,
                           layer_md5=layer_md5)
    except ProcessFailed as err:
        if layer_md5 is not None:
            layer_md5.unlink(missing_ok=True)
        return Outcome(case.name, False, f"encode: {err}", repro)
    if not result.gate.passed:
        if layer_md5 is not None:
            layer_md5.unlink(missing_ok=True)
        return Outcome(case.name, False, result.gate.reason, repro)
    if case.is_spatial and layer_md5 is not None:
        if case.codec is Codec.AV1:
            op_gate = gate_av1_operating_points(stream, layer_md5, None, paths, own_decoder=binary)
        else:
            op_gate = gate_vp9_operating_points(stream, layer_md5, paths, own_decoder=binary)
        layer_md5.unlink(missing_ok=True)
        if not op_gate.passed:
            return Outcome(case.name, False, op_gate.reason, repro)
    stream.unlink(missing_ok=True)
    return Outcome(case.name, True)


def parse_seeds(text: str) -> range:
    first, _, last = text.partition("-")
    return range(int(first), int(last or first) + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help=f"seeds {QUICK_SEEDS.start}-"
                        f"{QUICK_SEEDS.stop - 1} (the CI target)")
    parser.add_argument("--seeds", default="1-50", help="seed range FIRST-LAST")
    parser.add_argument("--codecs", default="h264,hevc,av1,vp9")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", type=Path, help="write every outcome here")
    parser.add_argument("--spatial", action="store_true",
                        help="AV1 spatial-SVC draws only (every seed codes a spatial mode; "
                             "even seeds sweep scales 1.25x/1.5x/2x x enhancement intervals 2-8)")
    args = parser.parse_args(argv)
    if args.spatial and any(c not in ("av1", "vp9") for c in args.codecs.split(",")):
        parser.error("--spatial needs --codecs av1 and/or vp9")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be 1..{MAX_WORKERS}")
    paths = load_paths()
    binary = find_tool("nanocodec", paths)
    work = paths.work / "syntax-stress"
    work.mkdir(parents=True, exist_ok=True)
    seeds = QUICK_SEEDS if args.quick else parse_seeds(args.seeds)
    # Clips are made before the workers start, so no two write the same file.
    clips = {(s.width, s.height): stress_clip(s, paths, work) for s in SIZES}
    switched = {(s.width, s.height): switched_clip(s, StressCase(Codec.H264, s.width, s.height,
                                                                 s.fps, 0).resized, paths, work)
                for s in SIZES}
    cases = [StressCase(Codec(c), s.width, s.height, s.fps, seed, args.spatial)
             for c in args.codecs.split(",") for s in SIZES for seed in seeds
             if codes_size(Codec(c), s)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(
            lambda case: run_case(case, (switched if case.features.structure == "resize"
                                         else clips)[case.width, case.height],
                                  binary, paths, work),
            cases))
    failures = [o for o in outcomes if not o.passed]
    for failure in failures:
        print(f"FAIL {failure.case}: {failure.reason}\n  repro: {failure.repro}")
    print(f"syntax stress: {len(outcomes) - len(failures)}/{len(outcomes)} streams match "
          f"every decoder ({binary})")
    if args.json:
        args.json.write_text(json.dumps([asdict(o) for o in outcomes], indent=1) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
