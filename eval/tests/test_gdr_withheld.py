"""VVdeC's withheld first GDR sweep: the conformance gate's expectation."""

from nceval.conformance import has_gdr_picture, withheld_sweep
from nceval.codecs import Codec


def test_one_contiguous_run_left_out_is_accepted():
    reference = ["a", "b", "c", "d", "e"]
    assert withheld_sweep(reference, reference) == 0
    assert withheld_sweep(reference, ["a", "d", "e"]) == 2
    assert withheld_sweep(reference, ["a", "b", "c"]) == 2


def test_anything_else_is_refused():
    reference = ["a", "b", "c", "d", "e"]
    assert withheld_sweep(reference, ["a", "c", "e"]) is None
    assert withheld_sweep(reference, ["a", "x", "e"]) is None
    assert withheld_sweep(reference, [*reference, "f"]) is None


def test_gdr_nal_units_are_found(tmp_path):
    trail = b"\x00\x00\x00\x01\x00\x01\xaa"
    gdr = b"\x00\x00\x00\x01\x00\x51\xbb"  # nal_unit_type 10, TemporalId 0
    plain, with_gdr = tmp_path / "plain.266", tmp_path / "gdr.266"
    plain.write_bytes(trail * 3)
    with_gdr.write_bytes(trail + gdr + trail)
    assert not has_gdr_picture(plain, Codec.VVC)
    assert has_gdr_picture(with_gdr, Codec.VVC)
    assert not has_gdr_picture(with_gdr, Codec.HEVC)
