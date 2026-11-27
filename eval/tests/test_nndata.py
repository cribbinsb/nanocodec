"""nceval.nndata: the loader's schema checks, the validator, packing, versions and splits,
on a tiny fixture shard written to the encoder's layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nceval import nndata
from nceval.nndata import schema, shard as shard_io, splits as split_mod, version
from nceval.nndata.validate import validate

SB_LOG2 = 6


def _clip(name: str, family: str, dataset: str = "cams", klass: str = "crowd") -> dict:
    return {"name": name, "family": family, "dataset": dataset, "content_class": klass,
            "sha256": f"sha-{name}", "source_kbps": 1000.0}


def write_fixture(path: Path, clip: dict | None = None, version: int = schema.FORMAT_VERSION) -> Path:
    """One key frame, one 64x64 superblock: a root that tried leaf and split, its split's
    four leaves; the leaf won."""
    clip = clip or _clip("c1", "fam1")
    n_sb = 1
    patch = np.full((n_sb, 32, 32), 100, np.uint8)
    values = {
        "frame": {"index": [0], "key": [1], "temporal_layer": [0], "quantizer": [32],
                  "lambda": [58.0], "noise_sigma": [0.5], "width": [64], "height": [64],
                  "superblock_log2": [SB_LOG2], "min_block_log2": [3]},
        "sb": {"frame": [0], "x": [0], "y": [0], "inside": [1], "cost": [10.0], "ns": [5000],
               "luma_mean": [100.0], "luma_variance": [0.0], "gradient": [[0, 0]],
               "coherence": [0.0], "block_variance": [[0, 0]], "static_share": [0.0],
               "still": [[0, 0]], "zero_sad": [0.0], "best_sad": [0.0], "motion": [[0, 0, 0, 0]],
               "search_vector": [[0, 0]], "residual_energy": [0.0], "patch": patch,
               "residual": np.zeros((n_sb, 32, 32), np.int16), "tool_cost": [[10.0]],
               "tool_ns": [[4000]]},
    }
    nan = np.nan
    costs = [[10.0, 12.0, nan, nan, nan, nan]] + [[3.0, nan, nan, nan, nan, nan]] * 4
    values["node"] = {"frame": [0] * 5, "sb": [0] * 5, "parent": [-1, 0, 0, 0, 0],
                      "x": [0, 0, 32, 0, 32], "y": [0, 0, 0, 32, 32],
                      "log2": [6, 5, 5, 5, 5], "planned": [0, 255, 255, 255, 255],
                      "chosen": [0] * 5, "eligible": [3, 1, 1, 1, 1], "cost": costs,
                      "plan_margin": [1.2] + [nan] * 4, "gates": [15] * 5, "context": [[0, 0]] * 5,
                      "has_inputs": [0] * 5, "inputs": [[0] * 11] * 5,
                      "ns": [[900, 2000, 0, 0, 0, 0]] + [[400, 0, 0, 0, 0, 0]] * 4}
    values["leaf"] = {"frame": [0], "sb": [0], "x": [0], "y": [0], "log2": [[6, 6]], "mode": [0],
                      "skip": [0], "mv": [[0, 0]], "qp_offset": [0]}
    rows = {"frame": 1, "sb": n_sb, "node": 5, "leaf": 1}
    tables = schema.required(version)
    for table, columns in tables.items():
        (path / table).mkdir(parents=True)
        for column in columns:
            shape = (rows[table], *column.shape_for(1))
            value = values[table].get(column.name)
            array = np.zeros(shape, np.dtype(column.dtype)) if value is None else \
                np.asarray(value, dtype=np.dtype(column.dtype))
            np.save(path / table / f"{column.name}.npy", array)
    labels = {"format_version": version, "codec": "hevc", "candidate_slots": 6,
              "patch_side": 32, "tools": ["preset"]}
    if version != 4:
        labels["tables"] = {t: [{"name": c.name, "dtype": c.dtype, "shape": list(c.row_shape)}
                                for c in cols] for t, cols in tables.items()}
    (path / shard_io.LABELS_FILE).write_text(json.dumps(labels))
    (path / shard_io.META_FILE).write_text(json.dumps(
        {"stream_identical": True, "job": {"clip_meta": clip, "clip": clip["name"], "codec": "hevc",
                                           "preset": "medium", "rate": "qp-high", "width": 64,
                                           "height": 64, "start_frame": 0, "frames": 1}}))
    return path


def test_fixture_loads_and_validates(tmp_path):
    shard = shard_io.load(write_fixture(tmp_path / "s"))
    assert shard.rows("node") == 5
    assert validate(shard) == []


def test_format_four_shards_still_load(tmp_path):
    shard = shard_io.load(write_fixture(tmp_path / "s", version=4))
    assert "leaf" not in shard.tables and validate(shard) == []


def test_format_five_shards_still_load_without_the_trial_summaries(tmp_path):
    shard = shard_io.load(write_fixture(tmp_path / "s", version=5))
    assert "nonzero" not in shard["node"] and "positions" not in shard["leaf"]
    assert validate(shard) == []


def test_a_column_the_encoder_adds_loads_without_a_reader_change(tmp_path):
    path = write_fixture(tmp_path / "s")
    np.save(path / "sb" / "extra.npy", np.zeros((1, 3), np.float32))
    labels = json.loads((path / shard_io.LABELS_FILE).read_text())
    labels["tables"]["sb"].append({"name": "extra", "dtype": "<f4", "shape": [3]})
    (path / shard_io.LABELS_FILE).write_text(json.dumps(labels))
    assert shard_io.load(path)["sb"]["extra"].shape == (1, 3)


def test_wrong_dtype_or_version_is_refused(tmp_path):
    path = write_fixture(tmp_path / "s")
    np.save(path / "sb" / "cost.npy", np.zeros(1, np.float64))
    with pytest.raises(shard_io.ShardError, match="sb.cost"):
        shard_io.load(path)
    path = write_fixture(tmp_path / "t")
    (path / shard_io.LABELS_FILE).write_text(json.dumps({"format_version": 99}))
    with pytest.raises(shard_io.ShardError, match="format version"):
        shard_io.load(path)


def _corrupt(tmp_path, table: str, column: str, change) -> list[str]:
    path = write_fixture(tmp_path / "s")
    array = np.load(path / table / f"{column}.npy")
    change(array)
    np.save(path / table / f"{column}.npy", array)
    return validate(shard_io.load(path))


def test_validator_catches_a_wrong_choice(tmp_path):
    def choose_split(a):
        a[0] = 1
    assert any("costlier" in p for p in _corrupt(tmp_path, "node", "chosen", choose_split))


def test_validator_catches_a_broken_tree(tmp_path):
    def reparent(a):
        a[1] = 3
    assert any("parent" in p for p in _corrupt(tmp_path, "node", "parent", reparent))


def test_validator_catches_a_cost_mismatch_and_missing_identity(tmp_path):
    def shift(a):
        a[0] = 11.0
    assert any("root node" in p for p in _corrupt(tmp_path, "sb", "cost", shift))
    path = write_fixture(tmp_path / "u")
    meta = json.loads((path / shard_io.META_FILE).read_text())
    meta["stream_identical"] = False
    (path / shard_io.META_FILE).write_text(json.dumps(meta))
    assert any("identical" in p for p in validate(shard_io.load(path)))


def test_pack_round_trip(tmp_path):
    path = write_fixture(tmp_path / "s")
    before = shard_io.load(path).tables
    shard_io.pack(path)
    after = shard_io.load(path)
    assert not (path / "sb").exists()
    for table, columns in before.items():
        for name, array in columns.items():
            np.testing.assert_array_equal(after[table][name], array)


def test_release_is_immutable_and_verified(tmp_path):
    run = tmp_path / "run"
    write_fixture(run / "shards" / "hevc-medium-qp-high" / "c1@0")
    write_fixture(run / "shards" / "hevc-slow-qp-high" / "c1@0")
    root = tmp_path / "nn-data"
    target = version.release([run], "v0", root)
    stored = list((target / shard_io.PATCH_STORE).iterdir())
    assert len(stored) == 1  # one source patch file for both rungs
    np.testing.assert_array_equal(nndata.load_version("v0", root)[0]["sb"]["patch"],
                                  np.full((1, 32, 32), 100, np.uint8))
    assert version.verify(target) == []
    with pytest.raises(version.VersionError, match="never edited"):
        version.release([run], "v0", root)
    assert len(nndata.load_version("v0", root)) == 2
    labels = next(target.rglob(shard_io.LABELS_FILE))
    labels.write_text(labels.read_text() + " ")
    with pytest.raises(shard_io.ShardError, match="manifest"):
        nndata.load_version("v0", root)


def test_split_keeps_sources_whole_and_refuses_sealed():
    clips = [_clip(f"a{i}", f"fa{i}", "cams_a") for i in range(4)] + \
        [_clip(f"b{i}", f"fb{i}", "cams_b") for i in range(4)] + \
        [_clip(f"x{i}", f"fx{i}", "xiph") for i in range(4)] + \
        [_clip("x0_copy", "fx0_copy", "xiph") | {"sha256": "sha-x0"}]
    sides = split_mod.split(clips, "seed", sealed=set())
    split_mod.check_split(clips, sides)
    assert len({sides[f"a{i}"] for i in range(4)}) == 1  # one dataset, one side
    assert sides["x0_copy"] == sides["x0"]  # the same file under another name
    assert set(sides.values()) == {split_mod.TRAIN, split_mod.HELD_OUT}
    with pytest.raises(split_mod.LeakageError):
        split_mod.split(clips, "seed", sealed={"fa1"})


def test_balance_weights_even_easy_and_hard(tmp_path):
    shard = shard_io.load(write_fixture(tmp_path / "s"))
    assert not nndata.easy_rows(shard).any()  # a key frame is never easy
    assert nndata.balance_weights(shard).tolist() == [1.0]


def test_long_run_rate_spreads_the_key_over_its_period(tmp_path):
    from nceval.nndata import generate
    log = tmp_path / "frames.csv"
    rows = ["tu,spatial_id,temporal_layer,frame_type,width,height,bytes,md5,droppable,op_mask,inter_layer_ref",
            "0,0,0,Key,64,64,30000,x,0,1,0"] + [f"{i},0,0,Inter,64,64,1000,x,0,1,0" for i in range(1, 30)]
    log.write_text("\n".join(rows) + "\n")
    fps = 30.0
    period = generate.KEY_PERIOD_SECONDS * fps
    expected = (30000 / period + 1000 * (1 - 1 / period)) * 8 * fps / 1000
    assert generate.long_run_kbps(log, fps) == pytest.approx(expected)
