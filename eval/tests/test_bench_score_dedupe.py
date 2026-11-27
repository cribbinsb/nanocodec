"""Cross-series score dedupe (owner, 2026-11-26): a stream scored in one BP-4 series (same
SHA-256) is copied, marked, into another series instead of being scored again."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bench"))
import localscore  # noqa: E402


def test_a_known_stream_is_copied_and_an_unknown_one_left_to_score(tmp_path, monkeypatch):
    bench = tmp_path / "bench"
    src = bench / "bp4-x-ours-c8i" / "series" / "c8i"
    dst = bench / "bp4-x-ours-c8g" / "series" / "c8g"
    src.mkdir(parents=True)
    dst.mkdir(parents=True)
    rec = {"id": "fam|clip|cbr500", "family": "fam", "series": "c8i", "stream_sha256": "aa",
           "per_frame": {"vmaf_v1": [90.0]}, "pooled": {"vmaf_v1": 90.0}}
    (src / "scores.jsonl").write_text(json.dumps(rec) + "\n" + '{"id": "cut sh')  # a cut line
    monkeypatch.setenv(localscore.SCORE_INDEX_ENV, str(tmp_path / "index"))
    todo = [{"id": "fam|clip|cbr500", "family": "fam", "stream_sha256": "aa"},
            {"id": "fam|clip|cbr800", "family": "fam", "stream_sha256": "bb"}]
    left = localscore.copy_known(dst, "c8g", todo)
    assert [r["stream_sha256"] for r in left] == ["bb"]
    copied = [json.loads(line) for line in (dst / "scores.jsonl").read_text().splitlines()]
    assert len(copied) == 1 and copied[0]["series"] == "c8g"
    assert copied[0]["copied_from"] == "bp4-x-ours-c8i/c8i"
    assert copied[0]["per_frame"] == rec["per_frame"]


def test_a_missing_stream_is_found_on_the_twin_chip_by_sha(tmp_path):
    bench = tmp_path / "bench"
    run = bench / "bp4-x-ours-c8i"
    twin = bench / "bp4-x-ours-c8g" / "series" / "c8g"
    (run / "series" / "c8i").mkdir(parents=True)
    (twin / "streams").mkdir(parents=True)
    (twin / "streams" / "a.ivf").write_bytes(b"same bytes")
    (twin / "encodes.jsonl").write_text(json.dumps(
        {"id": "f|c|cbr500", "stream": "streams/a.ivf", "stream_sha256": "aa"}) + "\n")
    found = localscore.twin_streams(run, run / "series" / "c8i")
    assert found == {"aa": twin / "streams" / "a.ivf"}


def test_a_reused_pass_directory_never_leaves_a_stale_entry(tmp_path):
    """A series' scores.jsonl entry is preferred; a rewritten pass file drops its entries."""
    bench = tmp_path / "bench"
    series = bench / "bp4-x-ours-c8i" / "series" / "c8i"
    pass_out = series / "scorelocal-0" / "out"
    pass_out.mkdir(parents=True)
    rec = {"id": "f|c|cbr500", "stream_sha256": "aa", "per_frame": {"vmaf_v1": [1.0]}}
    (pass_out / "scores.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(
        {**rec, "id": "f|c|cbr800", "stream_sha256": "bb"}) + "\n")
    index = tmp_path / "index"
    assert localscore.refresh_index(index, bench)["aa"][0].endswith("out/scores.jsonl")
    (series / "scores.jsonl").write_text(json.dumps(rec) + "\n")
    (pass_out / "scores.jsonl").write_text("")  # the directory reused by a later pass
    known = localscore.refresh_index(index, bench)
    assert known["aa"][0] == str(series / "scores.jsonl") and "bb" not in known


def test_a_c8g_stream_with_a_c8i_twin_is_left_for_the_c8i_pass(tmp_path, monkeypatch):
    bench = tmp_path / "bench"
    run = bench / "bp4-x-ours-c8g"
    (run / "series" / "c8g").mkdir(parents=True)
    twin = bench / "bp4-x-ours-c8i" / "series" / "c8i"
    twin.mkdir(parents=True)
    (twin / "encodes.jsonl").write_text(json.dumps({"id": "a", "stream_sha256": "aa"}) + "\n")
    monkeypatch.setenv(localscore.SCORE_INDEX_ENV, str(tmp_path / "index"))
    todo = [{"id": "a", "stream_sha256": "aa"}, {"id": "b", "stream_sha256": "x264-differs"}]
    assert [r["id"] for r in localscore.defer_to_twin(run, "c8g", todo)] == ["b"]


def test_a_failed_attempts_empty_file_is_not_the_streams_file(tmp_path, monkeypatch):
    """A zero-byte file left by an ENOSPC attempt under the same name is not the successful
    record's stream: the twin chip's identical file (same SHA and size) is used instead."""
    bench = tmp_path / "bench"
    run = bench / "bp4-x-ours-c8i"
    base = run / "series" / "c8i"
    (base / "streams").mkdir(parents=True)
    (base / "streams" / "a.hevc").write_bytes(b"")
    twin = bench / "bp4-x-ours-c8g" / "series" / "c8g"
    (twin / "streams").mkdir(parents=True)
    (twin / "streams" / "a.hevc").write_bytes(b"12345")
    (twin / "encodes.jsonl").write_text(json.dumps(
        {"id": "f|c|cbr500", "stream": "streams/a.hevc", "stream_sha256": "aa", "bytes": 5}) + "\n")
    (run / "jobs.json").write_text(json.dumps([{"id": "f|c|cbr500", "frames": 1, "w": 1, "h": 1}]))
    work = base / "scorelocal-0"
    monkeypatch.setattr(localscore, "gpu_refusal", lambda: None)

    class Queue:
        def acquire(self): return None
        def release(self): pass
    monkeypatch.setattr(localscore.gpuscore, "GpuQueue", Queue)
    monkeypatch.setattr(localscore.gpuscore, "ensure_slice", lambda: None)
    monkeypatch.setattr(localscore, "stage_clips", lambda *a: None)
    monkeypatch.setattr(localscore, "run_shards", lambda *a: [0])
    monkeypatch.setattr(localscore, "PINNED_FFMPEG_DIR", tmp_path)
    (tmp_path / "ffmpeg").write_text("")
    enc = {"id": "f|c|cbr500", "stream": "streams/a.hevc", "stream_sha256": "aa", "bytes": 5}
    localscore.run_pass(run, base, [enc], work, 1)
    linked = work / "out" / "streams" / "a.hevc"
    assert linked.resolve() == (twin / "streams" / "a.hevc").resolve()


def test_an_out_of_space_failure_is_not_an_encoder_failure():
    import report
    assert report.infrastructure_failure("V9: x exit 256: nanocodec: No space left on device (os error 28)")
    assert not report.infrastructure_failure("V9: x: decoded 149 frames of 150")
