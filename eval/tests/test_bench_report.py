"""tools/bench/report.py on a synthetic run: the BD tables, compliance, and its refusals."""
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "bench"))
import report  # noqa: E402

ZEROS = ("x264-medium", "x264-medium~psnr", "x264-medium~ssim")
RUNGS = (300, 500, 800, 1200)
CLIPS = ("a", "b", "c")


def point(family, clip, kbps, saving):
    """A point whose quality equals the zero line's at `saving` less rate."""
    rate = kbps * (1 - saving)
    q = math.log(kbps)
    frames = [int(rate * 1000 / 8 / 30)] * 30
    pooled = {"psnr_y": 30 + 4 * q, "psnr_u": 40 + q, "psnr_v": 40 + q, "vmaf_v1": 10 * q,
              "ms_ssim": 1 - 0.5 / kbps}
    job = {"id": f"{family}|{clip}|cbr{kbps}", "family": family, "input": clip, "w": 64, "h": 64,
           "frames": 30, "fps": "30/1", "kbps": rate, "mode": "cbr", "argv": ["{BIN}/x264"]}
    enc = {"id": job["id"], "family": family, "series": "c8a", "cpu_s": 0.1,
           "stream_sha256": job["id"]}
    score = {"id": job["id"], "family": family, "series": "c8a", "stream_sha256": job["id"],
             "frame_bytes": frames, "pooled": pooled}
    return job, enc, score


X264 = "a" * 64


@pytest.fixture(autouse=True)
def anchor_store(tmp_path_factory, monkeypatch):
    """A recorded build of the test's one binary (V25 needs one for every encoder used)."""
    store = tmp_path_factory.mktemp("anchors") / "x86_64-test"
    store.mkdir()
    (store / "SHA256SUMS").write_text(f"{X264}  bin/x264\n")
    (store / "build-flags.json").write_text(json.dumps(
        {"compiler": {"cc": "gcc 15.2"}, "profile": "opt",
         "tools": {"x264": {"commit": "b" * 40, "ref": "stable", "lto": True, "pgo": True}}}))
    import provenance
    monkeypatch.setattr(provenance, "ANCHORS", store.parent)


def make_run(tmp_path, families, canary_end=1.0, borrowed=False):
    jobs, encs, scores = [], [], []
    for family, saving in families.items():
        for clip in CLIPS:
            for kbps in RUNGS:
                j, e, s = point(family, clip, kbps, saving)
                jobs.append(j), encs.append(e), scores.append(s)
    if borrowed:
        scores[-1]["stream_sha256"] = "another series' stream"
    for where, t in (("start", 1.0), ("middle", 1.0), ("end", canary_end)):
        encs.append({"id": f"canary-{where}|x", "canary": where, "cpu_s": t, "stream_sha256": "c"})
    series = tmp_path / "series" / "c8a"
    series.mkdir(parents=True)
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"phase": {"INIT_MS": 1000}, "procedure_commit": "f" * 40, "binaries": {"x264": X264}}))
    (series / "encodes.jsonl").write_text("\n".join(map(json.dumps, encs)))
    (series / "scores.jsonl").write_text("\n".join(map(json.dumps, scores)))
    return tmp_path


def test_a_family_with_a_known_saving_reports_it(tmp_path):
    run = make_run(tmp_path, {**{z: 0.0 for z in ZEROS}, "ours": 0.3})
    rows = report.report(run)["series"]["c8a"]
    assert rows["view_clips"] == rows["common_clips"] == 3
    j = rows["families"]["ours"]["bd"]["j"]
    assert j["mean"] == pytest.approx(-30, abs=0.5) and j["low"] <= j["mean"] <= j["high"]
    assert rows["families"]["ours"]["compliance"]["target_rate"]["share_points_over_1"] == 0


def test_in_sample_clips_leave_every_family_and_are_listed(tmp_path):
    run = make_run(tmp_path, {**{z: 0.0 for z in ZEROS}, "ours": 0.3})
    record = tmp_path / "tuned_on.json"
    record.write_text(json.dumps({"sets": {"worst": {"used": "2026-11-18", "clips": [
        {"name": "a"}, {"name": "not-in-this-run"}]}}}))
    result = report.report(run, exclude=report.tuned_on(record))
    rows = result["series"]["c8a"]
    assert rows["view_clips"] == rows["common_clips"] == 2
    assert rows["families"]["ours"]["own_clips"] == 2
    assert result["excluded_clips"] == {"a": "worst (2026-11-18)"}  # only the run's own clips


def test_a_malformed_curve_costs_only_its_own_family_the_clip(tmp_path):
    run = make_run(tmp_path, {**{z: 0.0 for z in ZEROS}, "nanocodec-x": 0.3, "ref": 0.1})
    scores = run / "series" / "c8a" / "scores.jsonl"
    rows = [json.loads(line) for line in scores.read_text().splitlines()]
    for r in rows:  # 'ref' on clip a: quality falls as its rate rises
        if r["id"].startswith("ref|a|"):
            r["pooled"] = {**r["pooled"], "vmaf_v1": 100 - r["pooled"]["vmaf_v1"]}
    scores.write_text("\n".join(map(json.dumps, rows)))
    s = report.report(run)["series"]["c8a"]
    assert s["view_clips"] == 3 and s["common_clips"] == 2
    # BP-3.3: the headline averages every family over the same (common) clips
    assert s["families"]["ref"]["clips"] == s["families"]["nanocodec-x"]["clips"] == 2
    assert s["families"]["nanocodec-x"]["own_clips"] == 3
    assert "a" in s["families"]["ref"]["missing"]
    check = s["families"]["ref"]["exclusion_check"]
    assert check["missing_clips"] == 1 and check["ours_bd_j_on_missing"] < 0


def rewrite_scores(run, change):
    """Applies `change(record)` to every stored score (None: the record is unchanged)."""
    scores = run / "series" / "c8a" / "scores.jsonl"
    rows = [json.loads(line) for line in scores.read_text().splitlines()]
    scores.write_text("\n".join(json.dumps(change(r) or r) for r in rows))


def replace_curve(run, family, clip, rungs, saving):
    """Gives `family` these rungs on `clip` in place of the run's."""
    series = run / "series" / "c8a"
    made = [point(family, clip, kbps, saving) for kbps in rungs]
    other = lambda r: not r["id"].startswith(f"{family}|{clip}|")  # noqa: E731
    jobs = [j for j in json.loads((run / "jobs.json").read_text()) if other(j)]
    (run / "jobs.json").write_text(json.dumps(jobs + [m[0] for m in made]))
    for name, index in (("encodes.jsonl", 1), ("scores.jsonl", 2)):
        rows = [json.loads(line) for line in (series / name).read_text().splitlines()]
        (series / name).write_text("\n".join(map(json.dumps, [
            *(r for r in rows if other(r)), *(m[index] for m in made)])))


def test_two_points_and_a_thin_overlap_keep_the_clip_and_are_shown(tmp_path):
    """BP-3.14 (owner, 2026-11-25): a family with two rungs on a clip, sharing only the top
    of the zero line's range, has a BD there; its thin overlap is counted beside the figure."""
    run = make_run(tmp_path, {**{z: 0.0 for z in ZEROS}, "nanocodec-x": 0.3, "ref": 0.1})
    replace_curve(run, "ref", "a", (1100, 4000), 0.1)
    s = report.report(run)["series"]["c8a"]
    assert s["view_clips"] == s["common_clips"] == 3 and not s["families"]["ref"]["missing"]
    assert s["families"]["ref"]["bd"]["j"]["mean"] == pytest.approx(-10, abs=0.5)
    overlap = s["families"]["ref"]["overlap"]
    assert overlap["thin_clips"] == 1 and overlap["min"] < report.THIN_OVERLAP <= overlap["median"]
    assert s["curve_rules"] == {"min_points": 2, "thin_overlap_under": 0.5}


def test_a_lost_clip_is_listed_with_its_family_and_exact_reason(tmp_path, capsys, monkeypatch):
    run = make_run(tmp_path, {**{z: 0.0 for z in ZEROS}, "nanocodec-x": 0.3, "ref": 0.1})
    jobs = [j for j in json.loads((run / "jobs.json").read_text())
            if not j["id"].startswith("ref|a|") or j["id"].endswith("cbr300")]
    (run / "jobs.json").write_text(json.dumps(jobs))

    def far_above(r):  # nanocodec-x on clip b: every quality above the zero line's range
        if r["id"].startswith("nanocodec-x|b|"):
            r["pooled"] = {k: v + (0.0004 if k == "ms_ssim" else 50) for k, v in r["pooled"].items()}
    rewrite_scores(run, far_above)
    monkeypatch.setattr(sys, "argv", ["report.py", str(run), "--out", str(tmp_path / "r.json")])
    assert report.main() == 0
    text = capsys.readouterr().out
    assert "LOST from the common set: a: ref: 1 points coded" in text
    assert "LOST from the common set: b: nanocodec-x: " in text
    assert "ranges do not touch, the test curve lies above the anchor's" in text
    assert "common set of 1 clips (of 3 in the view)" in text


def test_a_point_without_a_metric_is_counted_and_costs_no_crash(tmp_path):
    """BP-3.14: the stored "ours" CBR runs hold points whose pooled VMAF is None."""
    run = make_run(tmp_path, {**{z: 0.0 for z in ZEROS}, "nanocodec-x": 0.3})

    def no_vmaf(r):
        if r["id"] in ("nanocodec-x|a|cbr500", "x264-medium|b|cbr800"):
            r["pooled"] = {**r["pooled"], "vmaf_v1": None}
    rewrite_scores(run, no_vmaf)
    s = report.report(run)["series"]["c8a"]
    assert s["points_without_a_metric"] == {"nanocodec-x": {"vmaf_v1": 1},
                                            "x264-medium": {"vmaf_v1": 1}}
    assert s["view_clips"] == s["common_clips"] == 3
    assert s["families"]["nanocodec-x"]["bd"]["j"]["mean"] == pytest.approx(-30, abs=0.5)


def test_a_borrowed_score_is_refused(tmp_path):
    run = make_run(tmp_path, {**{z: 0.0 for z in ZEROS}, "ours": 0.3}, borrowed=True)
    with pytest.raises(report.ReportRefusal, match="V23"):
        report.report(run)


def test_canary_drift_refuses_the_box_and_its_points_lose_only_their_speed(tmp_path):
    """V6 refuses the drifting box: its times are discarded. Its points keep their streams
    and scores, so they count for quality and BD, are left out of speed and are listed with
    the refusal (owner, 2026-11-26: V13 refuses only missing streams or scores)."""
    report.REFUSED_BOXES.clear()
    report.UNTIMED_POINTS.clear()
    run = make_run(tmp_path, {z: 0.0 for z in ZEROS}, canary_end=1.05)
    result = report.report(run)
    assert any("canaries" in why for why in report.REFUSED_BOXES.values())
    assert result["untimed_points"] and all("canaries" in u["reason"]
                                             for u in result["untimed_points"])


def test_the_replay_counts_frames_over_the_bucket():
    # a 1 s bucket at 100 kbps, starting full on the decoder side: a 2-buffer first frame
    out = report.replay([25_000] + [400] * 29, 100, 30, 1.0)
    assert out["frames_over"] >= 1 and out["peak_buffers"] == pytest.approx(2.0)


def test_statistics_constants_equal_the_procedure_text():
    """V20: section 8's numbers are the report's constants."""
    text = (REPO / "docs" / "benchmark-procedure.md").read_text()
    assert f"{report.BOOTSTRAP_RESAMPLES:,} resamples" in text
    assert f"{round(report.INTERVAL * 100)} % bootstrap" in text
    assert f"`MIN_POINTS = {report.MIN_POINTS}`" in text
    assert f"`THIN_OVERLAP_FRACTION = {report.THIN_OVERLAP}`" in text


def test_the_anchors_extra_points_serve_only_its_bd_curve(tmp_path):
    run = make_run(tmp_path, {**{z: 0.0 for z in ZEROS}, "ours": 0.3})
    jobs = json.loads((run / "jobs.json").read_text())
    encs = (run / "series/c8a/encodes.jsonl").read_text().splitlines()
    scores = (run / "series/c8a/scores.jsonl").read_text().splitlines()
    for z in ZEROS:
        for clip in CLIPS:
            for kbps in (150, 2400):
                j, e, s = point(z, clip, kbps, 0.0)
                j["id"] = e["id"] = s["id"] = j["id"] + "-anchor"
                e["stream_sha256"] = s["stream_sha256"] = j["id"]
                j["anchor_extra"] = True
                jobs.append(j), encs.append(json.dumps(e)), scores.append(json.dumps(s))
    (run / "jobs.json").write_text(json.dumps(jobs))
    (run / "series/c8a/encodes.jsonl").write_text("\n".join(encs))
    (run / "series/c8a/scores.jsonl").write_text("\n".join(scores))
    rows = report.report(run)["series"]["c8a"]["families"]
    assert rows["ours"]["bd"]["j"]["mean"] == pytest.approx(-30, abs=0.5)
    assert rows["x264-medium"]["compliance"]["points"] == len(CLIPS) * len(RUNGS)


def test_references_normalise_a_step_and_refuse_divergence():
    """BP-3.6: a box that slows by 10 % halfway: both references show it, so a job's time is
    divided by the box's speed around it; references that disagree refuse the box."""
    recs = []
    for seq in (0, 20, 40, 60):
        slow = 1.1 if seq >= 40 else 1.0
        recs += [{"box": "b", "reference": "x264-medium", "seq": seq, "cpu_s": 1.0 * slow},
                 {"box": "b", "reference": "nanocodec-av1@medium", "seq": seq, "cpu_s": 2.0 * slow}]
    factors = report.speed_factors(recs)
    points = {"j": {"box": "b", "seq": 50, "cpu_s": 3.3}}
    report.normalise(points, factors)
    ref_median = (1.0 + 1.1) / 2  # the box's median reference time is between its two speeds
    assert points["j"]["cpu_s_normalised"] == pytest.approx(3.3 / (1.1 / ref_median))
    recs[-1]["cpu_s"] = 2.0 * 1.3  # the memory-heavy reference alone slows: not normalisable
    with pytest.raises(report.ReportRefusal, match="diverge"):
        report.speed_factors(recs)


def test_one_noisy_reference_point_does_not_refuse_a_box():
    """T8: a single 3 % outlier in one reference is smoothed by the 3-point moving median."""
    recs = []
    for seq in range(0, 100, 20):
        recs += [{"box": "b", "reference": "x264-medium", "seq": seq,
                  "cpu_s": 1.03 if seq == 40 else 1.0},
                 {"box": "b", "reference": "nanocodec-av1@medium", "seq": seq, "cpu_s": 2.0}]
    assert "b" in report.speed_factors(recs)


def test_the_envelope_drops_only_dominated_points():
    """BP-3.9: a point with more bits and less quality than another is dropped."""
    def p(rate, q):
        return {"family": "f", "id": f"f|{rate}", "kbps_coded": rate,
                "pooled": {"psnr_y": q, "psnr_u": q, "psnr_v": q, "vmaf_v1": q, "ms_ssim": 0.9}}
    curve = [p(100, 30), p(200, 33), p(250, 32), p(300, 35), p(400, 37)]
    kept = report.envelope(curve, "psnr_yuv")
    assert [x["kbps_coded"] for x in kept] == [100, 200, 300, 400]


def test_v6_refuses_only_the_windows_around_a_diverging_reference_point():
    """BP-3.11: references that diverge at one point refuse the jobs in the windows on either
    side of it; the box's other jobs, bracketed by agreeing references, keep their times."""
    recs = []
    for seq in range(0, 200, 20):  # the memory-heavy reference slows alone at 100 and 120
        heavy = 2.0 * (1.3 if seq in (100, 120) else 1.0)
        recs += [{"box": "b", "reference": "x264-medium", "seq": seq, "cpu_s": 1.0},
                 {"box": "b", "reference": "nanocodec-av1@medium", "seq": seq, "cpu_s": heavy}]
    factors, refused = report.reference_factors(recs)
    flagged = sorted(q for _, q in refused)
    assert flagged == [100, 120]
    jobs = {seq: {"box": "b", "seq": seq} for seq in range(180)}
    bad = sorted(seq for seq, r in jobs.items() if report.refused_window(r, refused, factors))
    assert bad == list(range(80, 140))  # the windows on either side of 100 and of 120


def test_the_report_opens_with_its_provenance_and_refuses_a_mixed_identity(tmp_path):
    """V25 (owner, 2026-11-24): the report says exactly which builds it measured; runs with
    different binaries, or a binary without a build record, are refused."""
    run = make_run(tmp_path / "a", {z: 0.0 for z in ZEROS})
    result = report.report(run)
    assert next(iter(result)) == "provenance"
    x264 = result["provenance"]["encoders"]["x264"]
    assert x264["commit"] == "b" * 40 and x264["sha256"] == X264 and x264["pgo"] is True
    other = make_run(tmp_path / "b", {z: 0.0 for z in ZEROS})
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["binaries"]["x264"] = "c" * 64
    (other / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(report.ReportRefusal, match="V25"):
        report.report(run, other)
    with pytest.raises(report.ReportRefusal, match="V25: x264 .* no build record"):
        report.report(other)


def test_the_own_rate_bucket_is_the_buffer_pages_measure():
    """BP-3.12 (owner): compliance is judged on the bucket at the stream's own rate, the same
    measure as the results site's buffer page (nceval.cbr_stats.own_rate_fullness)."""
    from nceval.cbr_stats import own_rate_fullness
    frames = [20_000] + [1_000] * 29  # a big key, then a stream at 392 kbps of a 600 target
    own = report.replay_own_rate(frames, 30.0, 1.0)
    assert own["peak_buffers"] == pytest.approx(own_rate_fullness([8 * b for b in frames], 30.0, 1000))
    at_target = report.replay(frames, 600, 30.0, 1.0)
    assert own["peak_buffers"] > at_target["peak_buffers"]  # undershoot earns no extra room


def test_compliance_gives_both_buckets_the_worse_of_two_and_the_first_key():
    """The 2026-11-25 fairness review: both buckets per encoder, the worse of the two, the first
    key in seconds of the stream's rate, per class and frame-rate band."""
    low = {"frame_bytes": [20_000] + [1_000] * 29, "kbps": 600, "fps": "15/1", "class": "cctv"}
    high = {"frame_bytes": [40_000] + [3_000] * 29, "kbps": 600, "fps": "30/1", "class": "crowd"}
    c = report.compliance([low, high], 1.0)
    for name in ("own_rate", "target_rate", "worse_of_two"):
        assert set(c[name]) >= {"median_peak_buffers", "p95_peak_buffers", "worst_peak_buffers",
                                "share_points_over_1", "share_points_over_1.15"}
    assert c["worse_of_two"]["worst_peak_buffers"] == max(c["own_rate"]["worst_peak_buffers"],
                                                          c["target_rate"]["worst_peak_buffers"])
    assert set(c["by_fps"]) == {"<=20 fps", ">20 fps"} and set(c["by_class"]) == {"cctv", "crowd"}
    key_s = 8 * 40_000 / (8 * (40_000 + 29 * 3_000) * 30 / 30)
    assert c["by_class"]["crowd"]["first_key_seconds"]["worst"] == pytest.approx(key_s)


def test_a_deterministic_decoder_failure_is_counted_against_its_encoder(tmp_path):
    """V26: a point whose stream fails the standard decoder and was coded twice to the same
    bytes is listed at the top as that encoder's failure, not a missing point; a stream that
    differs between codings is not deterministic and stays a V13 refusal."""
    run = make_run(tmp_path / "a", {**{z: 0.0 for z in ZEROS}, "ours": 0.3})
    series = run / "series" / "c8a"
    scores = [json.loads(l) for l in (series / "scores.jsonl").read_text().splitlines()]
    bad = next(s for s in scores if s["family"] == "ours")
    bad_id = bad["id"]
    scores = [s for s in scores if s["id"] != bad_id] + [{"id": bad_id, "refused": "V9: decoded 81 frames of 130"}]
    (series / "scores.jsonl").write_text("\n".join(map(json.dumps, scores)))
    encs = (series / "encodes.jsonl").read_text().splitlines()
    first = next(json.loads(l) for l in encs if json.loads(l).get("id") == bad_id)
    (series / "encodes.jsonl").write_text("\n".join(encs + [json.dumps(first)]))
    result = report.report(run)
    assert result["encoder_failures"][bad_id]["family"] == "ours"
    assert result["encoder_failures"][bad_id]["codings"] == 2
    (series / "encodes.jsonl").write_text("\n".join(encs + [json.dumps({**first, "stream_sha256": "x"})]))
    with pytest.raises(report.ReportRefusal, match="V13"):
        report.report(run)


OURS_OLD, OURS_NEW = "d" * 64, "e" * 64


def ours_run(tmp_path, saving, digest, competitors=X264):
    """A run of the zero line and one of our families, ours built as `digest`."""
    run = make_run(tmp_path, {**{z: 0.0 for z in ZEROS}, "nanocodec-av1@fast": saving})
    jobs = json.loads((run / "jobs.json").read_text())
    for j in jobs:
        if j["family"].startswith("nanocodec-"):
            j["argv"] = ["{BIN}/nanocodec"]
    (run / "jobs.json").write_text(json.dumps(jobs))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["binaries"] = {"x264": competitors, "nanocodec": digest}
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


@pytest.fixture
def two_builds(anchor_store):
    """Both of our builds recorded, each in its own store (V25 needs a record for each)."""
    import provenance
    for digest, commit in ((OURS_OLD, "1" * 40), (OURS_NEW, "2" * 40)):
        store = provenance.ANCHORS / f"ours-{commit[:4]}"
        store.mkdir()
        (store / "SHA256SUMS").write_text(f"{digest}  bin/nanocodec\n")
        (store / "BUILD.json").write_text(json.dumps(
            {"nanocodec": {"commit": commit, "version": "0.6", "build": "pgo"}}))


def test_a_mixed_report_takes_our_families_from_the_new_runs(tmp_path, two_builds):
    """The re-measure: our families from the new build's run, every other family reused from
    today's run; V25 names one build per family, the reused competitors' binaries must equal
    the new run's, and V18 refuses a reused point whose command the record has since changed."""
    old = ours_run(tmp_path / "old", 0.1, OURS_OLD)
    new = ours_run(tmp_path / "new", 0.3, OURS_NEW)
    parts = (report.Part((old,), take="others"), report.Part((new,), take="ours"))
    result = report.report(parts=parts)
    assert result["provenance"]["encoders"]["nanocodec"]["commit"] == "2" * 40
    row = result["series"]["c8a"]["families"]["nanocodec-av1@fast"]
    assert row["bd"]["j"]["mean"] < -20  # the new run's 30 % saving, not the old run's 10 %
    alone = report.report(new)
    assert alone["series"] == result["series"]
    other_x264 = ours_run(tmp_path / "x", 0.3, OURS_NEW, competitors="c" * 64)
    with pytest.raises(report.ReportRefusal, match="V25: x264 differs"):
        report.report(parts=(parts[0], report.Part((other_x264,), take="ours")))
    plan = tmp_path / "plan.json"
    jobs = json.loads((old / "jobs.json").read_text())
    plan.write_text(json.dumps([{**j, "argv": ["{BIN}/x264", "--new"]} if j["family"] == ZEROS[0]
                                else j for j in jobs]))
    with pytest.raises(report.ReportRefusal, match="V18"):
        report.report(parts=(report.Part((old,), take="others", commands=(plan,)), parts[1]))


def test_an_encode_failing_alike_on_every_box_is_its_encoders_failure(tmp_path):
    """V26 for encodes: a job the queue quarantined after the same error on each box is listed
    as its encoder's failure (an old build refusing an input size), not a V13 hole; a job
    that failed differently on its boxes stays a V13 refusal."""
    run = make_run(tmp_path / "a", {**{z: 0.0 for z in ZEROS}, "ours": 0.3})
    series = run / "series" / "c8a"
    scores = [json.loads(l) for l in (series / "scores.jsonl").read_text().splitlines()]
    bad_id = next(s["id"] for s in scores if s["family"] == "ours")
    (series / "scores.jsonl").write_text("\n".join(json.dumps(s) for s in scores if s["id"] != bad_id))
    error = "V9: x exit 256: nanocodec: 960x540: dimensions must be non-zero multiples of 8"
    (series / "quarantined.json").write_text(json.dumps({bad_id: f"box0: {error}; box1: {error}"}))
    assert report.report(run)["encoder_failures"][bad_id]["codings"] == 2
    (series / "quarantined.json").write_text(json.dumps({bad_id: f"box0: {error}; box1: timeout"}))
    with pytest.raises(report.ReportRefusal, match="V13"):
        report.report(run)


def test_a_later_part_codes_a_reused_point_again(tmp_path, two_builds):
    """A reused point whose command the record has since changed is coded again in the new run:
    the later part's point replaces it, and V18 checks only the reused points that stay."""
    old = ours_run(tmp_path / "old", 0.1, OURS_OLD)
    new = ours_run(tmp_path / "new", 0.3, OURS_NEW)
    plan = tmp_path / "plan.json"
    jobs = json.loads((old / "jobs.json").read_text())
    plan.write_text(json.dumps([{**j, "argv": ["{BIN}/x264", "--new"]} if j["family"] == ZEROS[0]
                                else j for j in jobs]))
    recoded = report.Part((new,), take=None)
    result = report.report(parts=(report.Part((old,), take="others", commands=(plan,)), recoded))
    assert result["provenance"]["encoders"]["nanocodec"]["sha256"] == OURS_NEW


def test_v18_judges_an_old_point_a_fill_run_planned_but_did_not_code(tmp_path):
    """A fill run of the same part replaces the points it holds, and only those: an old-command
    point the fill planned without coding is still the one the report uses, so V18 judges it."""
    first = make_run(tmp_path / "first", {**{z: 0.0 for z in ZEROS}, "ours": 0.3})
    fill = make_run(tmp_path / "fill", {ZEROS[0]: 0.0})
    plan = tmp_path / "plan.json"
    jobs = [{**j, "argv": ["{BIN}/x264", "--new"]} if j["family"] == ZEROS[0] else j
            for j in json.loads((first / "jobs.json").read_text())]
    plan.write_text(json.dumps(jobs))
    (fill / "jobs.json").write_text(json.dumps([j for j in jobs if j["family"] == ZEROS[0]]))
    part = report.Part((first, fill), commands=(plan,))
    assert report.report(parts=(part,))["series"]["c8a"]["view_clips"] == 3
    scores = fill / "series" / "c8a" / "scores.jsonl"
    scores.write_text("\n".join(scores.read_text().splitlines()[1:]))  # one point not coded again
    with pytest.raises(report.ReportRefusal, match="V18: 1 reused points"):
        report.report(parts=(part,))


def test_the_provenance_states_the_procedure_versions(tmp_path):
    """Owner, 2026-11-24: each report states the procedure version its runs were made under
    (from each run's tools commit) and the report code's own, read from the procedure's title."""
    import provenance
    result = report.report(make_run(tmp_path / "a", {z: 0.0 for z in ZEROS}))
    stated = result["provenance"]["procedure"]
    title = (REPO / "docs" / "benchmark-procedure.md").read_text().splitlines()[0]
    assert stated["report"] in title and stated["report"].startswith("BP-")
    assert stated["runs"] == ["unknown"]  # the synthetic run's commit is not in the repository
    assert provenance.procedure_version("HEAD").startswith("BP-")


def test_a_part_can_take_a_subset_of_the_clips(tmp_path):
    """A view's subset (the history clips of the "ours" set): only those clips' points."""
    run = make_run(tmp_path / "a", {**{z: 0.0 for z in ZEROS}, "ours": 0.3})
    result = report.report(parts=(report.Part((run,), inputs=("a", "b")),))
    assert result["series"]["c8a"]["view_clips"] == 2


def test_a_long_window_serves_both_windows_and_counts_each_clip_once(tmp_path):
    """D7: a clip coded at 60 frames also serves its 30-frame window, read from the first 30
    frames' bytes and per-frame scores; the headline counts the clip once, each window is its
    own view, and only the whole encode enters compliance."""
    run = make_run(tmp_path, {**{z: 0.0 for z in ZEROS}, "ours": 0.3})

    def lengthen(r):  # clip "a" coded at 60 frames, serving windows 30 and 60
        if "|a|" in r["id"] and "frame_bytes" in r:
            r["frame_bytes"] = r["frame_bytes"] * 2
            r["per_frame"] = {k: [v] * 60 for k, v in r["pooled"].items()}
        return r
    rewrite_scores(run, lengthen)
    jobs = json.loads((run / "jobs.json").read_text())
    for j in jobs:
        if j["input"] == "a":
            j["frames"], j["windows"] = 60, [30, 60]
    (run / "jobs.json").write_text(json.dumps(jobs))
    s = report.report(run)["series"]["c8a"]
    ours = s["families"]["ours"]
    assert ours["clips"] == 3                               # a, b, c: "a" once
    assert ours["bd"]["j"]["mean"] == pytest.approx(-30, abs=0.5)
    assert set(ours["bd_by_window"]) == {30, 60}
    assert ours["bd_by_window"][60]["j"]["mean"] == pytest.approx(-30, abs=0.5)
    assert ours["compliance"]["points"] == 3 * len(RUNGS)   # whole encodes only
