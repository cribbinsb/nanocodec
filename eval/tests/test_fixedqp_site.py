"""nceval.fixedqp site --keep-references: a release re-measures ours, keeps the references."""
import pytest

from nceval.fixedqp import keep_references


def record(families: dict, clips=("a", "b")) -> dict:
    return {"clipset": "set.json", "anchor": "x264-medium", "frames": 150, "points": 4,
            "clips": [{"clip": c} for c in clips], "families": families}


def test_the_references_rows_are_kept_and_ours_never_carried_over():
    new = record({"nanocodec-av1": {"j": -40.0}})
    old = record({"nanocodec-av1": {"j": -30.0}, "nanocodec-av1@old": {"j": -1.0},
                  "svtav1-m8": {"j": -45.0}})
    keep_references(new, old)
    assert new["families"] == {"nanocodec-av1": {"j": -40.0}, "svtav1-m8": {"j": -45.0}}


def test_another_view_is_refused():
    with pytest.raises(SystemExit):
        keep_references(record({}), record({"svtav1-m8": {}}, clips=("a", "c")))
