"""tools/dev/aws_cpu_ab.py launch_shards: a failed shard is retried on spot unless
--retry-on-demand (docs/aws-cost-guide.md rule 5); nothing is launched."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import aws_cpu_ab  # noqa: E402


class Done:
    def __init__(self, code):
        self.returncode = code

    def poll(self):
        return self.returncode


def run(tmp_path, monkeypatch, codes, **options):
    monkeypatch.setattr(aws_cpu_ab.time, "sleep", lambda s: None)
    starts, codes = [], iter(codes)

    def start(label, attempt, demand):
        starts.append((attempt, demand))
        return Done(next(codes))
    failed = aws_cpu_ab.launch_shards({"graviton4-1": tmp_path}, tmp_path, 1, False, "t", 0.1,
                                      start_shard=start, **options)
    return starts, failed


def test_retries_stay_on_spot_and_fail_with_the_flag_named(tmp_path, monkeypatch, capsys):
    starts, failed = run(tmp_path, monkeypatch, [1, 1, 1])
    assert starts == [(1, False), (2, False), (3, False)] and failed == ["graviton4-1"]
    assert "--retry-on-demand" in capsys.readouterr().out


def test_on_demand_retries_only_when_asked(tmp_path, monkeypatch):
    starts, failed = run(tmp_path, monkeypatch, [1, 0], retry_on_demand=True)
    assert starts == [(1, False), (2, True)] and failed == []


def records(job, rounds=2, variants=("base", "cand")):
    return [{"job": job, "variant": v, "round": i, "counts": {"cycles": 100}, "user_s": 1.0,
             "system_s": 0.0, "wall_s": 1.0} for i in range(rounds) for v in variants]


def test_a_reclaim_mid_run_loses_only_the_job_in_flight(tmp_path, monkeypatch):
    """Box 1 finished jobs a and b (pulled while it ran: Session.pulling) and was reclaimed
    half-way through c; box 2 is told to skip a and b, measures c and d, and the summary
    reads all four from the two boxes."""
    import json
    jobs = ["p|av1:medium|a", "p|av1:medium|b", "p|av1:medium|c", "p|av1:medium|d"]
    stage, out = tmp_path / "stage", tmp_path / "graviton4-1"
    stage.mkdir()
    monkeypatch.setattr(aws_cpu_ab.time, "sleep", lambda s: None)
    skipped = []

    def start(label, attempt, demand):
        out.mkdir(exist_ok=True)
        done = stage / aws_cpu_ab.DONE_FILE
        skip = set(json.loads(done.read_text())) if done.is_file() else set()
        skipped.append(sorted(skip))
        if attempt == 1:  # two jobs and half of the third, then the reclaim
            lines = records(jobs[0]) + records(jobs[1]) + records(jobs[2])[:2]
            code = 1
        else:
            lines = [r for j in jobs if j not in skip for r in records(j)]
            code = 0
        (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
        return Done(code)
    failed = aws_cpu_ab.launch_shards({"graviton4-1": stage}, tmp_path, 2, False, "t", 0.1,
                                      start_shard=start)
    assert failed == [] and skipped == [[], jobs[:2]]
    measured = {json.loads(line)["job"] for files in aws_cpu_ab.shard_runs(tmp_path).values()
                for f in files for line in f.read_text().splitlines()}
    assert measured == set(jobs)
    runs = [json.loads(line) for files in aws_cpu_ab.shard_runs(tmp_path).values()
            for f in files for line in f.read_text().splitlines()]
    assert len(runs) == 4 * 4  # no job twice, the lost half-job not at all


def test_finished_jobs_need_every_variant_at_every_round():
    assert aws_cpu_ab.finished_jobs(records("x") + records("y")[:3], 2) == {"x"}
