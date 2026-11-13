"""The player model (nceval.display) against the fixture nc_dsp::display pins."""

import numpy as np

from nceval.display import scale_frame, scale_plane, switched_format
from nceval.yuv import SwitchedFormat, YuvFormat


def test_doubling_matches_the_rust_fixture():
    # nc_dsp::display::tests::doubling_matches_the_shared_fixture
    plane = np.array([[10, 50, 200], [90, 0, 255]], dtype=np.uint8)
    assert scale_plane(plane, 4, 6).tolist() == [
        [10, 20, 40, 88, 163, 200],
        [30, 32, 36, 82, 170, 214],
        [70, 56, 27, 70, 184, 241],
        [90, 68, 23, 64, 191, 255],
    ]


def test_flat_frames_stay_flat_and_same_size_is_identity():
    coded, shown = YuvFormat(8, 4), YuvFormat(16, 8)
    frame = bytes([77]) * coded.frame_size
    assert scale_frame(frame, coded, shown) == bytes([77]) * shown.frame_size
    assert scale_frame(frame, coded, coded) is frame


def test_switched_format_follows_the_sizes():
    sizes = [(16, 8)] * 3 + [(8, 4)] * 2 + [(16, 8)]
    fmt = switched_format(sizes)
    assert isinstance(fmt, SwitchedFormat)
    assert [fmt.at(i) for i in range(6)] == [YuvFormat(*s) for s in sizes]
    assert switched_format([(16, 8)] * 4) == YuvFormat(16, 8)
