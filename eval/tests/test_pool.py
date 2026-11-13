"""The rotating pool: stratified, family-disjoint draws; rotated starts; intervals; fan-out."""

import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from nceval import clipframes, interval, pool
from nceval.clipframes import ClipFrames
from nceval.clips import Clip
from nceval.config import Paths


def clip(name, cls, family=None, frames=300):
    return Clip(name, Path(f"/{name}.yuv"), 64, 32, 30, frames, 2000, cls, "train",
                family=family or name)


def fake_pool():
    clips = []
    for cls in ("crowd", "lowlight", "pan", "static", "bodycam", "texture", "drone"):
        for i in range(20):
            clips.append(clip(f"{cls}{i}", cls))
    # two forms of one source: never in one draw or two levels together
    clips.append(clip("crowd0_360p", "crowd", family="crowd0"))
    return clips


def test_draw_is_stratified_disjoint_and_reproducible():
    clips = fake_pool()
    first = pool.draw(clips, "2026-11-11", 150)
    assert first == pool.draw(clips, "2026-11-11", 150)
    assert first.levels != pool.draw(clips, "2026-11-12", 150).levels
    sizes = {level: len(c) for level, c in first.levels.items()}
    assert sizes[pool.QUICK] == 24 and sizes[pool.ADOPT] == 48
    families = [c.family for level in first.levels.values() for c in level]
    assert len(families) == len(set(families)) == len({c.family for c in clips})
    for level in (pool.QUICK, pool.ADOPT):
        assert {pool.stratum(c) for c in first.levels[level]} == set(pool.STRATA)


def test_datasets_are_dealt_round_robin():
    order = ["a1", "a2", "a3", "b1", "c1"]
    dataset = {f: f[0] for f in order}
    assert pool.spread_datasets(order, dataset) == ["a1", "b1", "c1", "a2", "a3"]


def test_start_frame_rotates_within_the_clip():
    long, short = clip("long", "pan", frames=600), clip("short", "pan", frames=130)
    starts = {pool.rotated(long, f"day{d}", 150).start_frame for d in range(20)}
    assert len(starts) > 10 and all(0 <= s <= 450 for s in starts)
    assert pool.rotated(short, "day1", 150).start_frame == 0
    record = pool.draw([long], "k", 150).record()
    assert record["rotation"] == "k" and "start_frame" in record["levels"][pool.QUICK][0]


def test_sealed_families_and_test_split_stay_out(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()

    def write(name, entries):
        (clips_dir / f"{name}.json").write_text(json.dumps({"name": name, "clips": entries}))
    entry = lambda name, split="train", source=None: {
        "name": name, "yuv": f"/{name}.yuv", "width": 64, "height": 32, "fps": 30,
        "frames": 300, "source_kbps": 2000, "content_class": "static", "split": split,
        "grade": "A", **({"source": source} if source else {})}
    write("main", [entry("a", source="/src/a.mp4"), entry("t", "test", "/src/t.mp4"),
                   entry("t_sibling", source="/src/t.mp4")])
    write("adv12", [entry("adv", "val", "/src/adv.y4m")])
    poor = {**entry("poor", source="/src/p"), "grade": "B", "downscale": 1.0}
    small = {**entry("small", source="/src/s"), "grade": "B", "downscale": 2.5,
             "source_bpp": 0.4}
    write("pool", [entry("adv_camera", source="/src/adv.y4m"), entry("new", source="/src/n"),
                   poor, small, {**entry("bad", source="/src/b"), "grade": "C"}])
    paths = Paths(data_root=tmp_path, anchors=tmp_path, nanocodec_bin=tmp_path)
    assert [c.name for c in pool.load_pool(paths)] == ["a", "new", "small"]


def rows(values):
    return {f"c{i}": {"class": cls, "quality": j, "quality_low": j}
            for i, (cls, j) in enumerate(values)}


def test_interval_stops_only_confident_losses():
    noisy = rows([(cls, j) for cls in ("crowd", "pan", "static") for j in (-3.0, 4.0)])
    stats = interval.interval(noisy, None, 0.6)
    assert stats["n"] == 6 and stats["lo"] < 0.5 < stats["hi"]
    assert interval.decide(stats, final=False) == interval.CONTINUE
    clear = interval.interval(rows([(c, j) for c in ("crowd", "pan") for j in (2.0, 2.4)]),
                              None, 0.6)
    assert interval.decide(clear, final=False) == interval.WORSE
    gain = interval.interval(rows([(c, j) for c in ("crowd", "pan") for j in (-1.0, -0.6)]),
                             None, 0.6)
    assert interval.decide(gain, final=True) == interval.BETTER
    assert interval.decide(stats, final=True) == interval.AMBIGUOUS
    small = interval.interval(rows([(c, j) for c in ("crowd", "pan") for j in (0.2, 0.4)]),
                              None, 0.6)
    assert interval.decide(small, final=False) == interval.CONTINUE
    assert interval.decide(small, final=True) == interval.WORSE


def test_one_clip_per_class_is_not_a_zero_width_interval():
    stats = interval.interval(rows([("crowd", 1.0), ("pan", 3.0), ("static", 2.0)]), None, 0.6)
    assert stats["hi"] - stats["lo"] > 0.5


def test_identical_builds_are_not_better():
    same = interval.interval(rows([("crowd", -1e-12), ("pan", 0.0)]), None, 0.6)
    assert interval.decide(same, final=True) == interval.AMBIGUOUS


def test_dcpu_shifts_every_clip():
    base = rows([("crowd", 1.0), ("crowd", 1.0)])
    assert interval.interval(base, -10.0, 0.1)["mean"] == pytest.approx(0.0)


def read_all(paths):
    out = [b""] * len(paths)

    def read(i):
        out[i] = Path(paths[i]).read_bytes()
    threads = [threading.Thread(target=read, args=(i,)) for i in range(len(paths))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def test_raw_clip_from_a_rotated_start_fans_out(tmp_path):
    frame = 64 * 32 * 3 // 2
    data = bytes(i % 251 for i in range(frame * 10))
    yuv = tmp_path / "c.yuv"
    yuv.write_bytes(data)
    raw = Clip("c", yuv, 64, 32, 30, 10, 2000, "pan", "train", start_frame=3)
    with ClipFrames(raw, 4, tmp_path / "frames") as frames:
        got = read_all([frames.reader() for _ in range(3)])
    assert got == [data[3 * frame:7 * frame]] * 3
    direct = Clip("c", yuv, 64, 32, 30, 10, 2000, "pan", "train")
    with ClipFrames(direct, 4, tmp_path / "frames") as frames:
        assert frames.reader() == yuv


def test_unread_fifo_does_not_hang_close(tmp_path):
    yuv = tmp_path / "c.yuv"
    yuv.write_bytes(bytes(64 * 32 * 3 // 2 * 4))
    raw = Clip("c", yuv, 64, 32, 30, 4, 2000, "pan", "train", start_frame=1)
    with ClipFrames(raw, 2, tmp_path / "frames") as frames:
        frames.reader()  # never opened


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_encoded_clip_decodes_frame_exact_from_any_start(tmp_path):
    stored = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=64x32:rate=30:duration=2", "-c:v", "libx264", "-g", "30",
                    "-pix_fmt", "yuv420p", str(stored)], check=True)
    whole = subprocess.run(["ffmpeg", "-v", "error", "-i", str(stored), "-f", "rawvideo",
                            "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
    frame = 64 * 32 * 3 // 2
    for start in (0, 1, 29, 30, 31, 47):
        encoded = Clip("c", stored, 64, 32, 30, 60, 2000, "pan", "train", start_frame=start)
        with ClipFrames(encoded, 10, tmp_path / "frames") as frames:
            got = read_all([frames.reader(), frames.reader()])
        assert got == [whole[start * frame:(start + 10) * frame]] * 2, start


@pytest.fixture
def encoded_clip(tmp_path):
    """A 2 s 64x32 H.264 clip and its whole decode."""
    stored = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=64x32:rate=30:duration=2", "-c:v", "libx264", "-g", "30",
                    "-pix_fmt", "yuv420p", str(stored)], check=True)
    whole = subprocess.run(["ffmpeg", "-v", "error", "-i", str(stored), "-f", "rawvideo",
                            "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
    return Clip("c", stored, 64, 32, 30, 60, 2000, "pan", "train", start_frame=7), whole


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_concurrent_readers_share_one_decode_and_an_unopened_one_stalls_nobody(
        tmp_path, encoded_clip):
    encoded, whole = encoded_clip
    frame = 64 * 32 * 3 // 2
    with ClipFrames(encoded, 40, tmp_path / "frames") as frames:
        frames.reader()  # built only to inspect a command line: never opened
        got = read_all([frames.reader() for _ in range(3)])
        assert frames.decodes_started == 1
    assert got == [whole[7 * frame:47 * frame]] * 3


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_late_and_stalled_readers_get_the_same_frames(tmp_path, encoded_clip, monkeypatch):
    monkeypatch.setattr(clipframes, "STALL_SECONDS", 0.5)
    encoded, whole = encoded_clip
    expected = whole[7 * 64 * 32 * 3 // 2:47 * 64 * 32 * 3 // 2]
    with ClipFrames(encoded, 40, tmp_path / "frames") as frames:
        first = frames.reader()
        assert Path(first).read_bytes() == expected
        # Frame 0 is gone: later readers share a second decode.
        assert read_all([frames.reader(), frames.reader()]) == [expected] * 2
        assert frames.decodes_started == 2
        # One reader opened but read only after the other finished: it held
        # a full window while the other starved, so it finishes from its own decode.
        with open(frames.reader(), "rb") as lagging:
            assert Path(frames.reader()).read_bytes() == expected
            assert lagging.read() == expected
        assert frames.decodes_started == 4


def test_l1_is_the_chosen_set_and_the_other_levels_avoid_its_families(tmp_path):
    clips = [Clip(name=f"{cls}{i}", yuv=Path(f"/x/{cls}{i}.yuv"), width=640, height=360,
                  fps=30.0, frames=600, source_kbps=4000.0, content_class=cls, split="train",
                  family=f"{cls}{i}", grade="A", dataset=f"d{i % 3}")
             for cls in ("crowd", "pan", "static") for i in range(6)]
    sets = tmp_path / "l1_sets.json"
    sets.write_text(json.dumps({"sets": [["crowd0", "pan0", "static0"]]}))
    draw = pool.draw(clips, "day", 150, {pool.QUICK: 3, pool.ADOPT: 3, pool.FINAL: None}, sets)
    assert [c.name for c in draw.levels[pool.QUICK]] == ["crowd0", "pan0", "static0"]
    others = {c.family for level in (pool.ADOPT, pool.FINAL) for c in draw.levels[level]}
    assert not others & {"crowd0", "pan0", "static0"} and len(others) == 15
    sets.write_text(json.dumps({"sets": [["gone"]]}))  # a clip left the pool: drawn instead
    assert pool.draw(clips, "day", 150, {pool.QUICK: 3}, sets).levels[pool.QUICK][0].name != "gone"
