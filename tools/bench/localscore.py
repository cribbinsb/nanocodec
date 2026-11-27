"""Score a bench run's streams on this PC's GPU (`fleet.py score`, BP-4.0 D16, T25).

The scoring is `score.py` itself, with its `vmafx` backend: the same decodes, V9/V10/V23
checks and record layout as a scoring box. VMAF v1 and MS-SSIM come from the existing GPU
scoring service of the dev tools (`nceval.metrics`, `nceval.gpuscore`); nothing here scores.
What this module adds is where and how that runs:

  - the decoders (avmdec, aomdec, vpxdec) and `vmaf` are this PC's own x86_64 builds
    (LOCAL_DECODERS), not the run's `stage/bin`, which holds the timing machine's encoders
    (aarch64 for a Graviton series); ffmpeg is the pinned 8.0.1 build (procedure section 7),
    put first on PATH;
  - the GPU's guards (`nceval.gpuscore`): a place in the one scoring service, free GPU memory
    to start, the service's CPU and memory slice; a guard that fails refuses the pass, saying how to fix it,
    never a fallback to the cloud;
  - at most LOCAL_WORKERS_MAX scoring workers (the PC is not to be loaded);
  - the clips a pass needs are decompressed into the pass's own directory, deleted after it;
  - `--follow MINUTES`: passes repeat while streams still arrive (scoring beside encoding),
    and stop once a pass finds nothing new for that long;
  - Scoring is not verification (D18, D19): a benchmark pass only refuses a VMAFx build that
    is not in PROVEN_SCORERS. Verifying a build (V27: a sample scored again by libvmaf 3.2.1
    on a cloud CPU box, compared per frame) is the separate procedure
    tools/bench/verify_scorer.py, run once per build.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
from nceval import gpuscore  # noqa: E402

#: The pinned decoder build (procedure section 7: ffmpeg 8.0.1, with dav1d for AV1).
PINNED_FFMPEG_DIR = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data")) / \
    "anchors" / "x86_64" / "build" / "ffmpeg-n8.0.1"
#: The PC's decoders for every run, whatever the timing machine's architecture (decoding is
#: deterministic: a Graviton series' streams decode the same here).
LOCAL_DECODERS = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data")) / \
    "anchors" / "x86_64" / "bin"
#: The PC's scoring workers at most (owner, 2026-11-21: the PC runs agents, not load).
LOCAL_WORKERS_MAX = 8
#: The scoring scope's memory cap: the workers' decodes and the service's client side.
LOCAL_MEMORY_MAX = os.environ.get("NC_LOCAL_SCORE_MEMORY_MAX", "12G")
#: V27's sample: this share of a pass's scored streams, at least V27_MIN, every codec in it.
V27_SHARE = 0.01
#: Scorer builds (the VMAFx tool's SHA-256) already shown to agree with libvmaf 3.2.1 on CPU
#: (V27, D19): {sha256: the proof's file}. Kept durable beside the bench runs.
PROVEN_SCORERS = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data")) / \
    "bench" / "proven-scorers.json"
V27_MIN = 20
#: V27's tolerances: VMAF v1 per frame at libvmaf's printed precision (6 decimals); MS-SSIM
#: per frame within 1e-5; PSNR per plane per frame within 1e-4 dB (exact sums against
#: libvmaf's).
V27_VMAF_DECIMALS = 6
V27_MS_SSIM_TOLERANCE = 1e-5
V27_PSNR_TOLERANCE_DB = 1e-4
#: The V27 cloud box's limits (one small Graviton scoring box per pass).
V27_BUDGET_USD = 0.5
V27_LIFETIME_MIN = 40
#: Seconds between `--follow` polls for newly settled streams.
FOLLOW_POLL_S = 60


class ParityFailure(SystemExit):
    """V27 refused a pass: its scores are not merged."""


def stream_codec(rec: dict, jobs: dict[str, dict]) -> str:
    return jobs[rec["id"]]["codec"]


def unscored(base: Path, scored: set[str], ids: set[str] | None, rescore) -> list[dict]:
    """The series' encode records to score now: one per id, with a stream, not yet scored
    (or exactly `ids`)."""
    records = []
    path = base / "encodes.jsonl"
    if path.exists():
        records += whole_records(path.read_text())
    # A running series keeps its records per machine until `merge` at the run's end; reading
    # them here lets `--follow` score beside encoding (`stream_src` names the stream in its box).
    for box in sorted(b for b in base.glob("box*") if (b / "encodes.jsonl").exists()):
        for r in whole_records((box / "encodes.jsonl").read_text()):
            if "stream" in r and (box / r["stream"]).exists():
                records.append({**r, "stream_src": f"{box.name}/{r['stream']}"})
    encodes = list({r["id"]: r for r in records
                    if "stream" in r and "canary" not in r and "reference" not in r
                    and "check" not in r}.values())  # BP-4 check sets are timing checks only
    if ids is not None:
        return [r for r in encodes if r["id"] in ids]
    return [r for r in encodes if rescore(r, scored)]


#: A series-long cache of decompressed source clips (owner, 2026-11-26: decompressing every
#: pass's clips again, 6.5 GB for a VC-1K pass, held each pass for minutes). Unset: per pass.
CLIP_CACHE_ENV = "NC_BENCH_CLIP_CACHE"
#: The scoring workers (LOCAL_WORKERS_MAX by default), raised while the benchmark scores.
WORKERS_ENV = "NC_LOCAL_SCORE_WORKERS"


TWIN_CHIP = {"c8i": "c8g", "c8g": "c8i"}


def twin_streams(run: Path, base: Path) -> dict[str, Path]:
    """sha256 -> stream file in the same run of the other chip (bp4-x-c8i <-> bp4-x-c8g), for
    the streams that exist there."""
    chip = run.name.rsplit("-", 1)[-1]
    if chip not in TWIN_CHIP:
        return {}
    twin_run = run.with_name(run.name[: -len(chip)] + TWIN_CHIP[chip])
    twin_base = twin_run / "series" / TWIN_CHIP[chip]
    out: dict[str, Path] = {}
    files = [twin_base / "encodes.jsonl", *twin_base.glob("box*/encodes.jsonl")]
    for f in files:
        if not f.exists():
            continue
        root = f.parent
        for r in whole_records(f.read_text()):
            if "stream" in r and "stream_sha256" in r:
                for path in (root / r["stream"], twin_base / r["stream"]):
                    if path.exists():
                        out.setdefault(r["stream_sha256"], path)
                        break
    return out


def twin_shas(run: Path) -> set[str]:
    """The stream SHA-256s the other chip's run recorded (whether or not its files are here)."""
    chip = run.name.rsplit("-", 1)[-1]
    if chip not in TWIN_CHIP:
        return set()
    twin_base = run.with_name(run.name[: -len(chip)] + TWIN_CHIP[chip]) / "series" / TWIN_CHIP[chip]
    return {r["stream_sha256"] for f in [twin_base / "encodes.jsonl",
                                         *twin_base.glob("box*/encodes.jsonl")] if f.exists()
            for r in whole_records(f.read_text()) if "stream_sha256" in r}


def whole_records(text: str) -> list[dict]:
    """The JSON lines of `text` that parse; a line still being copied from a machine, or cut
    short by a killed writer, is left out (it is read again on the next pass)."""
    out = []
    for line in text.splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def stage_clips(run: Path, clips: set[str], target: Path) -> None:
    """The pass's clips, decompressed from the run's stage: into the series' clip cache
    (CLIP_CACHE_ENV, linked from `target`), else into `target` itself."""
    target.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ[CLIP_CACHE_ENV]) if os.environ.get(CLIP_CACHE_ENV) else None
    for name in sorted(clips):
        out = target / f"{name}.yuv"
        if out.exists():
            continue
        source = (run / "stage" / "clips" / f"{name}.yuv.zst").resolve()
        if cache is None:
            subprocess.run(["zstd", "-d", "-q", "-f", str(source), "-o", str(out)], check=True)
            continue
        # keyed by the compressed source, so two runs' different cuts of one name never mix
        kept = cache / f"{name}-{source.stat().st_size:x}-{source.stat().st_mtime_ns:x}.yuv"
        if not kept.exists():
            cache.mkdir(parents=True, exist_ok=True)
            partial = kept.with_suffix(f".part{os.getpid()}")
            subprocess.run(["zstd", "-d", "-q", "-f", str(source), "-o", str(partial)], check=True)
            partial.rename(kept)
        out.symlink_to(kept)


def link_stream(source: Path, target: Path) -> None:
    """`target` names the stream file `source`: a hard link, so the pass keeps its streams when
    a running series' merge moves them out of their box directory (2026-11-27: bp4-fasthist-c8i
    merged mid-pass; every later stream's symlink dangled, was refused and was listed as a
    decoder bug); a symbolic link across file systems."""
    try:
        os.link(source.resolve(), target)
    except OSError:
        target.symlink_to(source.resolve())


def one_per_stream(todo: list[dict]) -> list[dict]:
    """One record per stream SHA-256 (the first): many history days' builds code byte-identical
    streams for unchanged presets (bp4-fasthist-c8i: 38,949 distinct streams of 60,059), and a
    pass scores them all at once, before `copy_known` can see the first one's score. The others
    are copied by SHA-256 on the next round, after this pass merges."""
    seen: set[str] = set()
    out = []
    for r in todo:
        sha = r.get("stream_sha256")
        if sha in seen:
            continue
        if sha:
            seen.add(sha)
        out.append(r)
    if len(out) < len(todo):
        print(f"{len(todo) - len(out)} streams are copies of another in this pass (same SHA-256): "
              f"copied once it is scored", flush=True)
    return out


#: Streams per pass at most (unset: all): a pass merges its scores only at its end, so a long
#: backlog in one pass held every score (and the twin chip's copies) back for hours, and a
#: killed pass lost them.
PASS_STREAMS_ENV = "NC_LOCAL_SCORE_PASS_STREAMS"


def pass_share(todo: list[dict]) -> list[dict]:
    """The first PASS_STREAMS_ENV streams of `todo`, grouped by clip (a clip is then read from
    disk in one pass, not in several)."""
    most = int(os.environ.get(PASS_STREAMS_ENV, 0))
    if not most or len(todo) <= most:
        return todo
    return sorted(todo, key=lambda r: r["id"].split("|")[1])[:most]


def gpu_refusal() -> str | None:
    """Why the GPU may not score a pass now (gpuscore's start guard), or None."""
    return gpuscore.wait_for_gpu(gpuscore.START_FREE_MIB)


def score_command(run: Path, out: Path, clips: Path, workers: int) -> list[str]:
    """score.py with the vmafx backend, in the GPU scoring service's slice."""
    return ["systemd-run", "--user", "--scope", "--quiet", f"--slice={gpuscore.SERVICE_SLICE}",
            "-p", f"MemoryMax={LOCAL_MEMORY_MAX}",
            sys.executable, str(REPO / "tools" / "bench" / "score.py"),
            "--encodes", str(out / "encodes.jsonl"), "--jobs", str(run / "jobs.json"),
            "--clips", str(clips), "--out", str(out), "--workers", str(workers),
            "--bin", str(LOCAL_DECODERS), "--metrics-backend", "vmafx"]


def local_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{PINNED_FFMPEG_DIR}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(REPO / "eval")
    return env


def run_pass(run: Path, base: Path, encodes: list[dict], work: Path, workers: int) -> Path:
    """One local scoring pass over `encodes`; returns its scores.jsonl."""
    out = work / "out"
    (out / "streams").mkdir(parents=True, exist_ok=True)
    encodes = list({r["id"]: dict(r) for r in encodes}.values())  # a merged and a box record
    # grouped by clip, the largest clips first and each clip's longest streams first: a clip is
    # then read from disk once while its streams score (2026-11-26: the PC's disk, ~40 MB/s
    # under load, held scoring at ~300 frames/s when streams of many clips interleaved); the
    # shards take every n-th stream, so they work on the same clip at the same time
    sizes = {j["id"]: j["frames"] * j["w"] * j["h"]
             for j in json.loads((run / "jobs.json").read_text())}
    clip_size: dict[str, int] = {}
    for r in encodes:
        clip = r["id"].split("|")[1]
        clip_size[clip] = max(clip_size.get(clip, 0), sizes.get(r["id"], 0))
    encodes.sort(key=lambda r: (-clip_size[r["id"].split("|")[1]], r["id"].split("|")[1],
                                -sizes.get(r["id"], 0)))
    twins: dict[str, Path] | None = None
    kept, lost = [], []
    for r in encodes:
        source = base / r.pop("stream_src", r["stream"])
        if not source.exists():  # a box's stream merged into the series since it was listed
            source = base / r["stream"]
        if source.exists() and r.get("bytes") and source.stat().st_size != r["bytes"]:
            # not this record's stream: a failed attempt's partial file under the same name
            # (2026-11-26: a full box disk left 86 such files; V23 refused them every pass)
            source = base / "streams" / "nonexistent"
        if not source.exists():
            # the stream never reached the PC (2026-11-26: a full box disk at 16:25 lost 5,526
            # of ours-ours-c8i's streams): the other chip's run holds the byte-identical
            # stream when the SHA-256 matches (ours is bit-exact across chips)
            twins = twin_streams(run, base) if twins is None else twins
            twin = twins.get(r.get("stream_sha256", ""))
            if twin is not None and r.get("bytes") and twin.stat().st_size != r["bytes"]:
                twin = None
            if twin is None:
                lost.append(r["id"])
                continue
            source = twin
            r["stream_from_twin"] = str(twin)
        target = out / r["stream"]
        if target.is_symlink() or (target.exists() and not target.samefile(source)):
            target.unlink()  # an earlier pass's link (reused dir), or one to another file
        if not target.exists():
            link_stream(source, target)
        kept.append(r)
    encodes = kept
    if lost:  # never a failed pass: listed, and the rest scored
        (work / "missing-streams.json").write_text(json.dumps(lost))
        print(f"{len(lost)} streams missing on the PC and on the twin chip, not scored "
              f"(listed in {work / 'missing-streams.json'})", flush=True)
    if not encodes:
        (out / "scores.jsonl").write_text("")
        return out / "scores.jsonl"
    (out / "encodes.jsonl").write_text("\n".join(map(json.dumps, encodes)) + "\n")
    if not (PINNED_FFMPEG_DIR / "ffmpeg").exists():
        raise SystemExit(f"refused: the pinned ffmpeg 8.0.1 is missing ({PINNED_FFMPEG_DIR})")
    reason = gpu_refusal()
    if reason:
        raise SystemExit(f"refused: {reason}; free the GPU (or start the scoring service) "
                         f"and run again; nothing falls back to the cloud (an explicit "
                         f"`fleet.py score --cloud` is D16's labelled cloud CPU path)")
    queue = gpuscore.GpuQueue()
    why = queue.acquire()
    if why:
        raise GpuBusy(f"the GPU scoring service is busy: {why}")
    clips = work / "clips"
    started = time.time()
    try:
        stage_clips(run, {r["id"].split("|")[1] for r in encodes}, clips)
        gpuscore.ensure_slice()
        codes = run_shards(run, out, clips, workers, encodes)
    finally:
        queue.release()
        shutil.rmtree(clips, ignore_errors=True)  # decoded clips: never kept
    scores = out / "scores.jsonl"
    from_twin = {r["id"]: r["stream_from_twin"] for r in encodes if r.get("stream_from_twin")}
    if from_twin and scores.exists():  # recorded: the scored stream file came from the twin
        recs = whole_records(scores.read_text())
        for rec in recs:
            if rec.get("id") in from_twin:
                rec["stream_from_twin"] = from_twin[rec["id"]]
        scores.write_text("".join(json.dumps(rec) + "\n" for rec in recs))
    print(throughput_line(scores, time.time() - started), flush=True)
    if any(code not in (0, 1) for code in codes):  # 1: some streams refused, in scores.jsonl
        raise SystemExit(f"refused: local score.py exited {codes}")
    return scores


#: score.py processes per pass, each over its share of the streams (measured 2026-11-26: one
#: process with 6 workers scored ~150 frames/s against ~80 with one, its frame pumping and PSNR
#: sharing one Python interpreter; several processes scale with the cores).
PROCESSES_ENV = "NC_LOCAL_SCORE_PROCESSES"


def run_shards(run: Path, out: Path, clips: Path, workers: int, encodes: list[dict]) -> list[int]:
    """score.py over `encodes` in PROCESSES_ENV processes at once (each `workers` threads),
    each in its own directory beside `out` (sharing its streams); their scores are appended to
    out/scores.jsonl as each process ends. Returns the exit codes."""
    processes = max(1, min(int(os.environ.get(PROCESSES_ENV, 1)), len(encodes)))
    if processes == 1:
        return [subprocess.run(score_command(run, out, clips, workers), env=local_env()).returncode]
    running = []
    for old in sorted(out.parent.glob("shard*")):  # a killed pass's shares: keep their scores
        if (old / "scores.jsonl").exists():
            with open(out / "scores.jsonl", "a") as merged:
                merged.writelines(json.dumps(r) + "\n"
                                  for r in whole_records((old / "scores.jsonl").read_text()))
        shutil.rmtree(old)
    for k in range(processes):
        shard = out.parent / f"shard{k}"
        shard.mkdir()
        (shard / "streams").symlink_to((out / "streams").resolve())
        (shard / "encodes.jsonl").write_text(
            "\n".join(json.dumps(r) for r in encodes[k::processes]) + "\n")
        running.append((shard, subprocess.Popen(score_command(run, shard, clips, workers),
                                                env=local_env())))
    codes = []
    with open(out / "scores.jsonl", "a") as merged:
        for shard, proc in running:
            codes.append(proc.wait())
            if (shard / "scores.jsonl").exists():
                merged.writelines(json.dumps(r) + "\n"
                                  for r in whole_records((shard / "scores.jsonl").read_text()))
            shutil.rmtree(shard)
    return codes


def throughput_line(scores: Path, seconds: float) -> str:
    recs = [json.loads(line) for line in scores.read_text().splitlines()] if scores.exists() else []
    frames = sum(len(r["per_frame"]["vmaf_v1"]) for r in recs if "per_frame" in r)
    return (f"local pass: {len(recs)} streams, {frames} frames in {seconds:.0f} s "
            f"({frames / max(seconds, 1e-9):.0f} frames/s, workers and VMAFx servers as set)")


def v27_sample(records: list[dict], jobs: dict[str, dict], seed: str) -> list[str]:
    """V27's ids: V27_SHARE of the pass's scored streams, at least V27_MIN, every codec."""
    scored = [r for r in records if "per_frame" in r]
    if not scored:
        return []
    rng = random.Random(seed)
    by_codec: dict[str, list[str]] = defaultdict(list)
    for r in scored:
        by_codec[stream_codec(r, jobs)].append(r["id"])
    chosen = {rng.choice(sorted(ids)) for ids in by_codec.values()}
    want = min(len(scored), max(V27_MIN, math.ceil(V27_SHARE * len(scored))))
    rest = sorted({r["id"] for r in scored} - chosen)
    chosen |= set(rng.sample(rest, max(0, want - len(chosen))))
    return sorted(chosen)


def parity_mismatches(local: dict, cloud: dict) -> list[str]:
    """V27 for one stream: why `local` (vmafx) and `cloud` (libvmaf) differ, if they do."""
    why = []
    if local["stream_sha256"] != cloud["stream_sha256"]:
        why.append("stream SHA-256 differs")
    lf, cf = local["per_frame"], cloud["per_frame"]
    if len(lf["vmaf_v1"]) != len(cf["vmaf_v1"]):
        return why + [f"frames {len(lf['vmaf_v1'])} vs {len(cf['vmaf_v1'])}"]
    for i, (a, b) in enumerate(zip(lf["vmaf_v1"], cf["vmaf_v1"])):
        if round(a, V27_VMAF_DECIMALS) != round(b, V27_VMAF_DECIMALS):
            why.append(f"VMAF v1 frame {i}: {a} vs {b}")
            break
    for i, (a, b) in enumerate(zip(lf["ms_ssim"], cf["ms_ssim"])):
        if abs(a - b) > V27_MS_SSIM_TOLERANCE:
            why.append(f"MS-SSIM frame {i}: {a} vs {b}")
            break
    for plane in ("psnr_y", "psnr_u", "psnr_v"):
        for i, (a, b) in enumerate(zip(lf[plane], cf[plane])):
            if abs(a - b) > V27_PSNR_TOLERANCE_DB:
                why.append(f"{plane} frame {i}: {a} vs {b}")
                break
    return why


def read_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def check_parity(local: dict[str, dict], cloud: dict[str, dict], ids: list[str]) -> dict:
    """V27's verdict over `ids`: a stream missing from either side counts as a mismatch."""
    bad = {}
    for i in ids:
        if i not in local or i not in cloud or "per_frame" not in cloud[i]:
            bad[i] = ["no cloud score" if i not in cloud or "per_frame" not in cloud[i]
                      else "no local score"]
            continue
        why = parity_mismatches(local[i], cloud[i])
        if why:
            bad[i] = why
    return {"checked": len(ids), "mismatched": bad}


def run_parity(a, base: Path, scores: Path, jobs: dict[str, dict], index: int,
               cloud_score: Callable[[argparse.Namespace], None]) -> None:
    """V27 for one pass: the sample scored on a cloud CPU box, compared per frame; refuses on
    any mismatch."""
    local = read_records(scores)
    ids = v27_sample(list(local.values()), jobs, f"v27-{a.series}-{a.tag}-{index}")
    if not ids:
        return
    tag = f"{a.tag}v27-{index}"
    ids_file = base / f"v27-{tag}-ids.json"
    ids_file.write_text(json.dumps(ids))
    cloud = copy.copy(a)
    cloud.cloud, cloud.ids, cloud.tag, cloud.no_merge, cloud.boxes = False, ids_file, tag, True, 1
    cloud.budget, cloud.lifetime = V27_BUDGET_USD, V27_LIFETIME_MIN
    cloud_score(cloud)
    verdict = check_parity(local, read_records(base / f"score{tag}0" / "scores.jsonl"), ids)
    (base / f"v27-{tag}.json").write_text(json.dumps(verdict, indent=1))
    if verdict["mismatched"]:
        raise ParityFailure(f"refused: V27: {len(verdict['mismatched'])} of {len(ids)} sampled "
                            f"streams differ between the GPU and libvmaf (v27-{tag}.json); the "
                            f"pass is not merged")
    print(f"V27: {len(ids)} sampled streams agree with libvmaf 3.2.1 on CPU", flush=True)


def scorer_identity() -> str:
    """The SHA-256 of the VMAFx tool the GPU service runs (nceval.metrics)."""
    import hashlib
    from nceval import config, metrics
    tool = metrics.vmafx_tool(config.load_paths())
    return hashlib.sha256(tool.read_bytes()).hexdigest()


def proven(identity: str) -> bool:
    return PROVEN_SCORERS.exists() and identity in json.loads(PROVEN_SCORERS.read_text())


def record_proven(identity: str, proof: str) -> None:
    table = json.loads(PROVEN_SCORERS.read_text()) if PROVEN_SCORERS.exists() else {}
    table[identity] = proof
    PROVEN_SCORERS.write_text(json.dumps(table, indent=1) + "\n")


def merge_pass(base: Path, scores: Path) -> int:
    """A passed pass's records appended to the series' (a later record for an id wins)."""
    lines = [line for line in scores.read_text().splitlines() if line.strip()]
    with open(base / "scores.jsonl", "a") as out:
        out.write("\n".join(lines) + "\n")
    return len(lines)


#: Cross-series dedupe (owner, 2026-11-26): a stream already scored in any BP-4 series (same
#: SHA-256: ours is bit-exact across chips, history c8g = c8i, references where their builds
#: agree) is not scored again; its record is copied into the series that has it, marked
#: `copied_from`. The index (sha -> file, offset) lives in this directory, refreshed from the
#: new bytes of every score file under the bench root before each pass.
SCORE_INDEX_ENV = "NC_BENCH_SCORE_INDEX"


def score_files(bench: Path) -> list[Path]:
    # not the shards' files: they are merged into out/scores.jsonl and removed at a pass's end
    pats = ("bp4-*/series/*/scores.jsonl", "bp4-*/series/*/score*local-*/out/scores.jsonl",
            "bp4-*/runs/*/series/*/scores.jsonl",
            "bp4-*/runs/*/series/*/score*local-*/out/scores.jsonl")
    return sorted({p for pat in pats for p in bench.glob(pat)})


def refresh_index(index_dir: Path, bench: Path) -> dict[str, list]:
    """sha256 -> [file, offset, length] of a full score record, updated from the bytes each
    score file gained since the last refresh (under a lock: passes run side by side)."""
    import fcntl
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = index_dir / "index.json"
        state = json.loads(path.read_text()) if path.exists() else {"seen": {}, "sha": {}}
        for f in score_files(bench):
            try:
                size = f.stat().st_size
            except FileNotFoundError:
                continue  # removed since the glob
            start = state["seen"].get(str(f), 0)
            if size < start:
                # rewritten (a pass directory reused by a later pass): its entries are stale
                start = 0
                state["sha"] = {k: v for k, v in state["sha"].items() if v[0] != str(f)}
            # a series' own scores.jsonl is append-only, so an entry there is preferred over one
            # in a pass directory (2026-11-26: pass directories are reused, and stale entries
            # sent 8,372 already-scored streams back to the GPU)
            stable = f.name == "scores.jsonl" and f.parent.parent.name == "series"
            if size == start:
                continue
            with open(f, "rb") as fh:
                fh.seek(start)
                offset = start
                for raw in fh:
                    if not raw.endswith(b"\n"):
                        break  # a line still being written: next refresh
                    if b'"per_frame"' in raw and b'"refused"' not in raw:
                        try:
                            sha = json.loads(raw)["stream_sha256"]
                        except (ValueError, KeyError):
                            sha = None
                        if sha and (sha not in state["sha"] or stable):
                            state["sha"][sha] = [str(f), offset, len(raw)]
                    offset += len(raw)
            state["seen"][str(f)] = offset
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.rename(path)
        return state["sha"]


def copy_known(base: Path, series: str, todo: list[dict]) -> list[dict]:
    """Copy the records of `todo`'s already-scored streams (by SHA-256) into the series, marked
    `copied_from`; returns the streams still to score."""
    index_dir = os.environ.get(SCORE_INDEX_ENV)
    if not index_dir or not todo:
        return todo
    bench = next(p for p in base.parents if p.name == "bench")
    known = refresh_index(Path(index_dir), bench)
    copies, left = [], []
    for r in todo:
        hit = known.get(r.get("stream_sha256", ""))
        if not hit:
            left.append(r)
            continue
        try:
            with open(hit[0], "rb") as fh:
                fh.seek(hit[1])
                rec = json.loads(fh.read(hit[2]))
            if rec.get("stream_sha256") != r.get("stream_sha256"):
                raise ValueError("the indexed line moved")
        except (OSError, ValueError):
            left.append(r)  # the indexed record is gone or moved: score it
            continue
        source = Path(hit[0])
        origin = next((p for p in source.parents if p.parent.name == "series"), source.parent)
        rec.update(id=r["id"], family=r.get("family", rec.get("family")), series=series,
                   copied_from=f"{origin.parent.parent.name}/{origin.name}")
        copies.append(rec)
    if copies:
        path = base / f"scores-copied-{int(time.time())}.jsonl"
        path.write_text("".join(json.dumps(c) + "\n" for c in copies))
        print(f"copied {merge_pass(base, path)} scores by stream SHA-256 (dedupe)", flush=True)
        path.unlink()
    return left


def defer_to_twin(run: Path, series: str, todo: list[dict]) -> list[dict]:
    """A c8g stream whose identical copy (same SHA-256) is coded in the c8i twin run is left for
    the c8i pass: it is scored there once and copied here by `copy_known` on a later pass, so
    the two chips' passes never score the same stream twice (c8i is queued first)."""
    if series != "c8g" or not todo or not os.environ.get(SCORE_INDEX_ENV):
        return todo
    twin = twin_shas(run)
    keep = [r for r in todo if r.get("stream_sha256") not in twin]
    if len(keep) < len(todo):
        print(f"{len(todo) - len(keep)} streams left for the c8i twin (scored there once, then "
              f"copied)", flush=True)
    return keep


class GpuBusy(RuntimeError):
    """The shared GPU scoring service did not take this pass within its queue wait."""


class UnverifiedScorer(RuntimeError):
    """The GPU scorer build has not been verified against libvmaf (D19)."""


def score_local(a, scored: Callable[[Path], set[str]], needs_score,
                cloud_score: Callable[[argparse.Namespace], None]) -> None:
    """`fleet.py score` (the default, on the PC): passes until nothing is left (or, with --follow, until nothing
    new has arrived for that long); each pass is checked by V27, then merged."""
    if not a.verify_scorer and not proven(scorer_identity()):
        raise UnverifiedScorer("refused: this VMAFx build is not in "
                               f"{PROVEN_SCORERS}; verify it once with tools/bench/verify_scorer.py "
                               "(a separate procedure, D19)")
    run = a.run
    base = run / "series" / a.series
    jobs = {j["id"]: j for j in json.loads((run / "jobs.json").read_text())}
    ids = set(json.loads(a.ids.read_text())) if a.ids else None
    most = int(os.environ.get(WORKERS_ENV, LOCAL_WORKERS_MAX))
    workers = min(most, a.score_workers or most)
    # a fresh pass directory each pass, never one an earlier process used: a reused directory
    # re-merged its old scores into the series every pass (2026-11-26: ours-ours-c8i's
    # scores.jsonl grew to 4.8 GB, 297k lines for 13k ids)
    used = [int(p.name.rsplit("-", 1)[1]) for p in base.glob(f"score{a.tag}local-*")
            if p.name.rsplit("-", 1)[1].isdigit()]
    quiet_since, index = time.monotonic(), max(used, default=-1) + 1
    while True:
        todo = unscored(base, scored(base), ids,
                        lambda r, done: needs_score(r, done, a.rescore))
        if not a.verify_scorer:
            todo = copy_known(base, a.series, todo)
            todo = defer_to_twin(run, a.series, todo)
            if ids is None:  # the rest is scored by the next rounds of this loop
                todo = pass_share(one_per_stream(todo))
        if todo:
            work = base / f"score{a.tag}local-{index}"
            try:
                scores = run_pass(run, base, todo, work, workers)
            except GpuBusy as why:
                # The GPU is shared (other projects too): a follower waits its turn instead
                # of ending, which left a run's streams unscored until someone restarted it.
                if not a.follow:
                    raise SystemExit(f"refused: {why}") from None
                print(f"{why}: trying again in {FOLLOW_POLL_S} s", flush=True)
                time.sleep(FOLLOW_POLL_S)
                continue
            if a.verify_scorer:  # tools/bench/verify_scorer.py, not a benchmark run
                run_parity(a, base, scores, jobs, index, cloud_score)
                record_proven(scorer_identity(), str(base / f"v27-{a.tag}v27-{index}.json"))
                return
            if not a.no_merge:
                print(f"merged {merge_pass(base, scores)} records", flush=True)
            index += 1
            quiet_since = time.monotonic()
            if ids is not None:
                return
            continue
        if not a.follow or time.monotonic() - quiet_since >= a.follow * 60:
            return
        time.sleep(FOLLOW_POLL_S)
