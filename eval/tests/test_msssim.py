"""nceval.msssim: libvmaf's float_ms_ssim, recomputed, and its streaming sink."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nceval import msssim  # noqa: E402
from nceval.config import find_tool, load_paths  # noqa: E402
from nceval.yuv import YuvFormat  # noqa: E402

# Smallest size whose fifth scale still holds the 11-tap window, plus odd
# sizes so the ceil(w/2) decimation and odd-width strips are exercised.
FMT = YuvFormat(258, 194)
FRAMES = 6


def synthetic_clip(seed: int, noise: float) -> tuple[bytes, bytes]:
    """Smooth gradients and texture as reference; blurred, noisy distortion."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:FMT.height, :FMT.width]
    reference, distorted = bytearray(), bytearray()
    for frame in range(FRAMES):
        base = 128 + 60 * np.sin(x / (9 + frame)) * np.cos(y / 13) + rng.normal(0, 12, x.shape)
        luma = np.clip(base, 0, 255).astype(np.uint8)
        blurred = (luma.astype(np.float32) + np.roll(luma, 1, axis=1) + np.roll(luma, 1, axis=0)) / 3
        damaged = np.clip(blurred + rng.normal(0, noise, x.shape), 0, 255).astype(np.uint8)
        chroma = bytes(2 * FMT.chroma_size)
        reference += luma.tobytes() + chroma
        distorted += damaged.tobytes() + chroma
    return bytes(reference), bytes(distorted)


def luma(clip: bytes, index: int) -> np.ndarray:
    start = index * FMT.frame_size
    return np.frombuffer(clip, np.uint8, FMT.luma_size, start).reshape(FMT.height, FMT.width)


def libvmaf_scores(reference: Path, distorted: Path) -> list[float]:
    try:
        vmaf = find_tool("vmaf", load_paths())
    except FileNotFoundError:
        pytest.skip("pinned libvmaf not built here")
    report = subprocess.run(
        [str(vmaf), "-r", str(reference), "-d", str(distorted), "-w", str(FMT.width),
         "-h", str(FMT.height), "-p", "420", "-b", "8", "--no_prediction",
         "--feature", "float_ms_ssim", "--json", "-o", "/dev/stdout", "-q"],
        capture_output=True, text=True, check=True).stdout
    frames = sorted(json.loads(report)["frames"], key=lambda f: f["frameNum"])
    return [f["metrics"]["float_ms_ssim"] for f in frames]


@pytest.mark.parametrize("noise", [2.0, 20.0])
def test_matches_libvmaf_float_ms_ssim(tmp_path, noise):
    reference, distorted = synthetic_clip(7, noise)
    (tmp_path / "ref.yuv").write_bytes(reference)
    (tmp_path / "dis.yuv").write_bytes(distorted)
    expected = libvmaf_scores(tmp_path / "ref.yuv", tmp_path / "dis.yuv")
    ours = [msssim.ms_ssim(luma(reference, i), luma(distorted, i)) for i in range(FRAMES)]
    # libvmaf prints 6 decimals; the computations agree to ~1e-6.
    assert ours == pytest.approx(expected, abs=2e-6)


def test_identical_planes_score_one_and_small_planes_are_refused():
    reference, _ = synthetic_clip(3, 0.0)
    assert msssim.ms_ssim(luma(reference, 0), luma(reference, 0)) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="too small"):
        msssim.ms_ssim(np.zeros((150, 400), np.uint8), np.zeros((150, 400), np.uint8))


def test_sink_scores_every_nth_frame_against_the_same_reference_frame(tmp_path):
    reference, distorted = synthetic_clip(5, 10.0)
    (tmp_path / "ref.yuv").write_bytes(reference)
    sink = msssim.MsSsimSink(tmp_path / "ref.yuv", FMT, subsample=4)
    for index in range(FRAMES):
        sink.consume(distorted[index * FMT.frame_size:(index + 1) * FMT.frame_size])
    sink.finish()
    sink.finish()  # idempotent: MetricSession's exit finishes it again
    assert sink.scores == [msssim.ms_ssim(luma(reference, i), luma(distorted, i))
                           for i in (0, 4)]


def test_sink_reraises_a_scoring_error_at_the_end_of_the_stream(tmp_path):
    small = YuvFormat(64, 64)
    (tmp_path / "ref.yuv").write_bytes(bytes(small.frame_size))
    sink = msssim.MsSsimSink(tmp_path / "ref.yuv", small)
    sink.consume(bytes(small.frame_size))
    with pytest.raises(ValueError, match="too small"):
        sink.finish()


#: Luma 224x224 at (608, 192) of frame 6 of x265 ultrafast at 35 kbps on the
#: held-out clip pool_ducks_take_off_pristine (2026-11-12 held-out chart): its
#: structure mean is negative at three scales, which made math.pow raise.
NEGATIVE_STRUCTURE = Path(__file__).parent / "data" / "msssim_negative_structure.npz"
#: VMAFx's CUDA float_ms_ssim on that pair (the default metrics backend).
NEGATIVE_STRUCTURE_GPU_SCORE = 0.32084621079643527
#: CPU and GPU MS-SSIM must agree within this.
BACKEND_TOLERANCE = 1e-5


def test_a_negative_structure_mean_is_taken_by_magnitude_like_the_gpu_backend():
    pair = np.load(NEGATIVE_STRUCTURE)
    x, y = pair["reference"].astype(np.float32), pair["distorted"].astype(np.float32)
    structures = []
    for scale in range(msssim.SCALES):
        structures.append(msssim._ssim_terms(x, y)[2])
        x, y = msssim._decimate(x), msssim._decimate(y)
    assert min(structures) < 0
    score = msssim.ms_ssim(pair["reference"], pair["distorted"])
    assert score == pytest.approx(NEGATIVE_STRUCTURE_GPU_SCORE, abs=BACKEND_TOLERANCE)
