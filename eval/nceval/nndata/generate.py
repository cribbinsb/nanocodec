"""Make shards: plan a version's jobs here, run them on a rented box, quarantine failures.

One job codes one clip excerpt at one codec, preset and rate, twice: plainly and with
`--label-dump`. The shard is kept only when

- the two streams are byte-identical (the dump changed nothing),
- the stream's long-run bitrate is at most MAX_SOURCE_FRACTION of the source's (CLAUDE.md:
  never code a clip above half its source bitrate): the rate a stream of these frames
  would have at the encoder's default key period (KEY_PERIOD_SECONDS), from its frame
  sizes. A 30-frame excerpt holds one key frame, which alone took 4-14x the source's half
  rate on the first v0 run's low-QP jobs; a deployed stream carries one per period, and
  that is the rate the rule is about. The excerpt's own rate is kept in meta.json too.
- the validator (nceval.nndata.validate) finds nothing.

Otherwise the shard goes to `quarantine/` with its reasons in `reason.json`, never into
the version. Kept shards are packed (nceval.nndata.shard.pack) and carry their provenance
in meta.json: commit, binary hash, the job (codec, preset, rate, params), the clip and its
source (dataset, class, grade, source kbps, file hash), start frame and frame count.

Frames reach the encoder through a pipe from ffmpeg: no YUV file is written. Raw (.yuv)
clips are sent to the box as lossless FFV1 excerpts of the job's frames (`excerpt`), a
few percent of the raw file.
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import random
import shutil
import subprocess
import traceback
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..clips import Clip
from ..config import find_tool
from .shard import META_FILE, PACKED_FILE, ShardError, pack, load as load_shard
from .validate import validate

MAX_SOURCE_FRACTION = 0.5
#: A job's parameter variant longer than this is named by its hash (file names stop at
#: 255 bytes).
MAX_VARIANT_NAME = 160
VARIANT_HASH_CHARS = 12
#: The encoder's default key period (nc-cli DEFAULT_INTRA_PERIOD_SECONDS), over which a
#: key frame's bytes are spread in the long-run rate.
KEY_PERIOD_SECONDS = 10
#: Frames per shard: one key frame and the P frames after it.
FRAMES = 30
#: Fixed quantizers per codec (two points, low and high rate), in each codec's scale.
QUANTIZERS = {"h264": (26, 36), "hevc": (27, 37), "vvc": (27, 37),
              "av1": (120, 200), "vp9": (120, 200), "av2": (120, 200)}
#: The fallback low-rate point where `qp-low` codes over half the source's rate (the
#: first v0 run: HEVC QP 27 broke the rule on 35 of 50 sources, AV1 120 on 14): a
#: `qp-mid` job at this quantizer replaces it.
QP_MID = {"h264": 31, "hevc": 32, "vvc": 32, "av1": 160, "vp9": 160, "av2": 160}
#: CBR target: this share of the source bitrate, capped at CBR_BITS_PER_PIXEL.
CBR_SOURCE_SHARE = 0.35
CBR_BITS_PER_PIXEL = 0.06
#: Rungs per codec, fastest first: the faster ones, medium and the slowest, all on the
#: same clips and start frames, so distillation (slowest into medium and the fast rungs,
#: nn-fast) pairs them on identical superblock keys.
RUNGS = {"h264": ("faster", "fast", "medium", "slower"),
         "hevc": ("faster", "fast", "medium", "veryslow"),
         "vvc": ("faster", "fast", "medium", "veryslow"),
         "av1": ("veryfast", "faster", "fast", "medium", "slower"),
         "vp9": ("faster", "fast", "medium", "slower"),
         "av2": ("faster", "fast", "medium", "slower")}
#: The faster rungs code two rates (one fixed QP, CBR), medium and the slowest all three.
#: (AV1 also codes veryfast: nn-fast distils into it.)
FAST_RATES = ("qp-high", "cbr")
#: Medium and the faster rungs code no inter RD partition search; a twin of each job on
#: them does, ungated, on the same clips and start frames, so the search's labels
#: (node/) cover their inter frames too: nn-lead's gate targets medium as well as the slow
#: rungs, and nn-fast distils the fast rungs' partitions from them (its oracle: the
#: partition holds the largest prize at every fast rung). Without tools.
SEARCH_TWIN = ("planner.rd_split=1", "planner.rd_split_context=0", "planner.rd_split_margin=0")
#: Rungs whose superblocks get the tool labels (sb/tool_cost): the tools' gains are
#: what the policy learns, and medium and the slowest are where they run.
TOOL_RUNGS = 2
QUARANTINE = "quarantine"
#: Tools labelled per superblock (`--label-tool`): the parameter and its off and on
#: values. A job flips each from the preset's own value (on where the preset has it off,
#: off where on), so tool_cost[t] - tool_cost[0] is what running (or dropping) the tool
#: on that superblock alone changes. Only parameters a superblock can change: engine
#: decision./planner. ones and the codecs' SUPERBLOCK_PARAMS (quant.trellis in HEVC and
#: AV1). Frame-level only, so not here: the second reference (refs.second), the loop
#: filters' searches (SAO, CDEF), quant.trellis_states and quant.trellis_key.
TOOL_CATALOGUE = {
    "rd_split": ("planner.rd_split", 0.0, 1.0),
    "key_rd_split": ("planner.key_rd_split", 0.0, 1.0),
    "trellis": ("quant.trellis", 0.0, 1.0),
    "winner_walk": ("decision.winner_walk_steps", 0.0, 2.0),
    "final_stage": ("decision.final_stage", 0.0, 1.0),
    "intra_rd_modes": ("decision.intra_rd_modes", 0.0, 4.0),
    "intra_rd_transforms": ("decision.intra_rd_transforms", 0.0, 2.0),
    # The second reference's trial on a superblock of a frame that carries one
    # (the per-superblock switch; the frame-level `refs.second` stays as the job sets
    # it), and the two-hypothesis pair trial in frames coded as generalised B slices.
    "second_reference": ("decision.second_reference", 0.0, 1.0),
    "bi_pair": ("decision.bi_pair", 0.0, 1.0),
}
#: Tools a codec cannot label (H.264 has no node trials; a backend parameter the codec
#: does not read per superblock). A tool whose trial leaves
#: state behind shows as a dump stream that differs from the plain encode's, and the
#: validator quarantines it (AV1's key-frame leaf searches until d75dfebe0, 2026-11-22).
TOOLS_UNSUPPORTED = {"h264": tuple(TOOL_CATALOGUE),
                     # quant.trellis is not among their SUPERBLOCK_PARAMS.
                     "vvc": ("trellis",), "vp9": ("trellis",), "av2": ("trellis",)}


def tools_for(codec: str, preset: str, rungs: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """The tools labelled at `preset`: every catalogue tool the codec supports, on the
    TOOL_RUNGS slowest of `rungs` (default the codec's RUNGS)."""
    if preset not in (rungs or RUNGS[codec])[-TOOL_RUNGS:]:
        return ()
    return tuple(t for t in TOOL_CATALOGUE if t not in TOOLS_UNSUPPORTED.get(codec, ()))


@dataclass(frozen=True)
class Job:
    codec: str
    preset: str
    rate: str  # qp-low, qp-high or cbr
    quantizer: int | None
    kbps: int | None
    clip: str
    #: The file the frames are read from, relative to the home directory (the same on
    #: the PC and the box), and the frame of it the job starts at: the pool's encoded
    #: clip at `start_frame`, or a lossless excerpt of a raw clip (`excerpt`) at 0.
    input_file: str
    input_seek: int
    width: int
    height: int
    fps: float
    start_frame: int
    frames: int
    clip_meta: dict
    #: Parameter overrides over the preset (`--param NAME=VALUE`): the label-stability
    #: set toggles one landed lever.
    params: tuple[str, ...] = ()
    #: Names in TOOL_CATALOGUE labelled per superblock.
    tools: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        variant = "".join(f"+{p}" for p in self.params)
        if len(variant) > MAX_VARIANT_NAME:
            # A whole tuned point does not fit a file name: its hash names it (the
            # parameters stay in meta.json's job).
            digest = hashlib.sha256(variant.encode()).hexdigest()[:VARIANT_HASH_CHARS]
            variant = f"+params-{digest}"
        return f"{self.codec}-{self.preset}-{self.rate}{variant}/{self.clip}@{self.start_frame}"


def cbr_kbps(clip: Clip) -> int:
    cap = CBR_BITS_PER_PIXEL * clip.width * clip.height * clip.fps / 1000
    return max(1, round(min(CBR_SOURCE_SHARE * clip.source_kbps, cap)))


def start_frame(clip: Clip, seed: str) -> int:
    """A start on a whole second (the pool's stored key frames), drawn by `seed`."""
    second = max(1, round(clip.fps))
    starts = range(0, max(1, clip.frames - FRAMES), second)
    return random.Random(f"{seed}:{clip.name}").choice(list(starts))


def clip_meta(clip: Clip, data_root: Path) -> dict:
    return {"name": clip.name, "dataset": clip.dataset, "family": clip.family,
            "content_class": clip.content_class, "grade": clip.grade,
            "source_kbps": clip.source_kbps, "downscale": clip.downscale,
            "file": str(clip.yuv.relative_to(data_root)), "sha256": file_sha256(clip.yuv)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def excerpt(clip: Clip, start: int, frames: int, directory: Path, ffmpeg: str) -> Path:
    """A lossless FFV1 copy of `frames` frames of raw `clip` from `start` (made once)."""
    path = directory / f"{clip.name}@{start}.mkv"
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        size = clip.frame_bytes
        with open(clip.yuv, "rb") as source:
            source.seek(start * size)
            raw = source.read(frames * size)
        partial = path.with_suffix(".partial.mkv")
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                        "yuv420p", "-s", f"{clip.width}x{clip.height}", "-r", f"{clip.fps:g}",
                        "-i", "pipe:0", "-c:v", "ffv1", "-level", "3", str(partial)],
                       input=raw, check=True)
        partial.rename(path)
    return path


#: Lossless excerpts of raw clips for the boxes: under the clip root, so nc-remote sends
#: them with --clip and the box deletes them with the clips (private frames).
EXCERPTS = Path.home() / "nanocodec-data/clips/nn-excerpts"


def plan(clips: list[Clip], codecs: list[str], data_root: Path, seed: str,
         excerpts: Path, ffmpeg: str, rungs: tuple[str, ...] | None = None) -> list[Job]:
    """Every job of a version: each clip at each codec's rungs (or `rungs`) and rates."""
    jobs, home = [], Path.home()
    for clip in clips:
        meta = clip_meta(clip, data_root)
        start = start_frame(clip, seed)
        frames = min(FRAMES, clip.frames - start)
        if clip.encoded:
            source, seek = clip.yuv, start
        else:
            source, seek = excerpt(clip, start, frames, excerpts, ffmpeg), 0
        input_file = str(source.resolve().relative_to(home))
        for codec in codecs:
            low, high = QUANTIZERS[codec]
            rates = {"qp-low": (low, None), "qp-high": (high, None), "cbr": (None, cbr_kbps(clip))}
            ladder = rungs or RUNGS[codec]
            for preset in ladder:
                chosen = tuple(rates) if preset in ladder[-TOOL_RUNGS:] else FAST_RATES
                for rate in chosen:
                    quantizer, kbps = rates[rate]
                    job = Job(codec, preset, rate, quantizer, kbps, clip.name, input_file, seek,
                              clip.width, clip.height, clip.fps, start, frames, meta,
                              tools=tools_for(codec, preset, ladder))
                    jobs.append(job)
                    if preset != ladder[-1]:
                        jobs.append(replace(job, params=SEARCH_TWIN, tools=()))
    return jobs


def frames_command(job: Job, ffmpeg: str) -> list[str]:
    seek = ["-ss", f"{job.input_seek / job.fps:.6f}"] if job.input_seek else []
    return [ffmpeg, "-nostdin", "-v", "error", "-threads", "1", *seek,
            "-i", str(Path.home() / job.input_file), "-frames:v", str(job.frames),
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]


def preset_values(binary: Path, codec: str, preset: str) -> dict[str, float]:
    """Every parameter's effective value at `preset` (`nanocodec params`)."""
    listing = subprocess.run([str(binary), "params", "--codec", codec, "--preset", preset],
                             capture_output=True, text=True, check=True).stdout
    values = {}
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                values[fields[0]] = float(fields[1])
            except ValueError:
                continue
    return values


def tool_flags(job: Job, values: dict[str, float]) -> list[str]:
    """`--label-tool` for each of the job's tools, flipped from the preset's value."""
    flags = []
    for name in job.tools:
        param, off, on = TOOL_CATALOGUE[name]
        flipped = on if values.get(param, off) == off else off
        flags += ["--label-tool", f"{name}:{param}={flipped:g}"]
    return flags


def encode_command(job: Job, binary: Path, output: Path, dump: Path | None) -> list[str]:
    rate = ["--qp", str(job.quantizer)] if job.quantizer is not None else \
        ["--bitrate", str(job.kbps)]
    command = [str(binary), "encode", "--codec", job.codec, "--preset", job.preset,
               "--input", "-", "--width", str(job.width), "--height", str(job.height),
               "--fps", f"{job.fps:g}", "--frames", str(job.frames), *rate,
               "--output", str(output)]
    command += [arg for param in job.params for arg in ("--param", param)]
    if dump is None:
        return command
    return command + ["--label-dump", str(dump)] + \
        tool_flags(job, preset_values(binary, job.codec, job.preset))


def long_run_kbps(frame_log: Path, fps: float) -> float:
    """The long-run rate of a stream whose frames `--layer-md5` logged: its mean key and
    mean inter frame, one key per KEY_PERIOD_SECONDS."""
    sizes: dict[bool, list[int]] = {True: [], False: []}
    for row in csv.DictReader(frame_log.open()):
        sizes[row["frame_type"] == "Key"].append(int(row["bytes"]))
    period = max(1.0, KEY_PERIOD_SECONDS * fps)
    key = sum(sizes[True]) / len(sizes[True]) if sizes[True] else 0.0
    inter = sum(sizes[False]) / len(sizes[False]) if sizes[False] else key
    per_frame = key / period + inter * (1 - 1 / period)
    return per_frame * 8 * fps / 1000


def _encode(job: Job, ffmpeg: str, command: list[str]) -> None:
    frames = subprocess.Popen(frames_command(job, ffmpeg), stdout=subprocess.PIPE)
    encoded = subprocess.run(command, stdin=frames.stdout, capture_output=True, text=True)
    frames.stdout.close()
    frames.wait()
    if encoded.returncode != 0:
        raise RuntimeError(f"encode failed: {encoded.stderr.strip()[-500:]}")


def run_job(job: Job, binary: Path, provenance: dict, out: Path,
            ffmpeg: str) -> tuple[str, list[str]]:
    """Code `job`, check it and keep or quarantine its shard; returns (where, problems)."""
    work = out / "work" / job.name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    shard_dir = work / "shard"
    problems: list[str] = []
    try:
        plain, dumped = work / "plain.bin", work / "dump.bin"
        frame_log = work / "frames.csv"
        _encode(job, ffmpeg, encode_command(job, binary, plain, None) + ["--layer-md5", str(frame_log)])
        _encode(job, ffmpeg, encode_command(job, binary, dumped, shard_dir))
        identical = plain.read_bytes() == dumped.read_bytes()
        excerpt_kbps = plain.stat().st_size * 8 * job.fps / job.frames / 1000
        kbps = long_run_kbps(frame_log, job.fps)
        meta = {"schema": 1, **provenance, "job": asdict(job), "stream_identical": identical,
                "stream_kbps": excerpt_kbps, "long_run_kbps": kbps,
                "stream_bytes": plain.stat().st_size}
        (shard_dir / META_FILE).write_text(json.dumps(meta, indent=1))
        if not identical:
            problems.append("stream: the dump's stream differs from the plain encode's")
        if kbps > MAX_SOURCE_FRACTION * job.clip_meta["source_kbps"]:
            problems.append(f"rate: {kbps:.0f} kbps (long run) is over half the source's "
                            f"{job.clip_meta['source_kbps']:.0f}")
        problems += validate(load_shard(shard_dir))
    except Exception as error:  # noqa: BLE001 - every failure quarantines the job, with its cause
        problems.append(f"error: {error!r}\n{traceback.format_exc(limit=3)}")
    target = out / (QUARANTINE if problems else "shards") / job.name
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if shard_dir.exists():
        shard_dir.rename(target)
    else:
        target.mkdir(parents=True)
    if problems:
        (target / "reason.json").write_text(json.dumps({"job": asdict(job), "problems": problems}, indent=1))
    else:
        pack(target)
    shutil.rmtree(work, ignore_errors=True)
    return ("quarantine" if problems else "kept"), problems


def finished(job: Job, out: Path) -> bool:
    """Whether an earlier run of `out` already kept or quarantined `job` (a run resumes).
    A kept shard that no longer loads (a run killed while packing it) is removed and coded
    again."""
    if (out / QUARANTINE / job.name / "reason.json").exists():
        return True
    kept = out / "shards" / job.name
    if not kept.exists():
        return False
    try:
        load_shard(kept)
        return True
    except (ShardError, OSError, ValueError, zipfile.BadZipFile):
        shutil.rmtree(kept)
        return False


def tally(out: Path) -> dict:
    """Kept and quarantined shards of `out` and each quarantine's first problem."""
    kept = len(list((out / "shards").glob("*/*/" + PACKED_FILE)))
    problems = {}
    for reason in sorted((out / QUARANTINE).glob("*/*/reason.json")):
        name = str(reason.parent.relative_to(out / QUARANTINE))
        problems[name] = json.loads(reason.read_text())["problems"][0][:200]
    return {"kept": kept, "quarantine": len(problems), "problems": problems}


def run_jobs(jobs: list[Job], binary: Path, provenance: dict, out: Path,
             ffmpeg: str, workers: int) -> dict:
    """Run every job not finished by an earlier run of `out` (`workers` at once); returns
    the tally of `out`, also written to summary.json."""
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        done_before = list(pool.map(lambda job: finished(job, out), jobs))
    todo = [job for job, done in zip(jobs, done_before) if not done]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} jobs finished earlier", flush=True)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(run_job, job, binary, provenance, out, ffmpeg) for job in todo]
        counts = {"kept": 0, "quarantine": 0}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            where, _ = future.result()
            counts[where] += 1
            if done % 50 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} kept {counts['kept']} quarantined {counts['quarantine']}",
                      flush=True)
    shutil.rmtree(out / "work", ignore_errors=True)
    summary = tally(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def binary_provenance(binary: Path) -> dict:
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()
    return {"binary_sha256": file_sha256(binary), "binary_version": version}


def write_jobs(jobs: list[Job], path: Path) -> None:
    path.write_text(json.dumps([asdict(j) for j in jobs]))


def read_jobs(path: Path) -> list[Job]:
    return [Job(**(entry | {"params": tuple(entry.get("params", ())),
                            "tools": tuple(entry.get("tools", ()))}))
            for entry in json.loads(path.read_text())]


def find_ffmpeg() -> Path:
    return find_tool("ffmpeg")


def rate_retries(jobs: list[Job], summaries: list[dict]) -> list[Job]:
    """`qp-mid` jobs for the `qp-low` jobs quarantined for their rate in `summaries`."""
    over = {name for summary in summaries for name, problem in summary["problems"].items()
            if problem.startswith("rate:")}
    return [replace(job, rate="qp-mid", quantizer=QP_MID[job.codec])
            for job in jobs if job.rate == "qp-low" and job.name in over]


#: Codecs whose quantizer scale is another's for the rate rule's history (VVC's QP is HEVC's).
RATE_PROXY = {"vvc": "hevc", "av2": "av1", "vp9": "av1"}


def without_rate_failures(jobs: list[Job], summary: dict) -> list[Job]:
    """`jobs` without the qp-low jobs whose clip broke the half-source-rate rule at qp-low
    in an earlier version (`summary`, its merged validator summary), by codec or its
    RATE_PROXY: they would only be quarantined again."""
    failed = set()
    for name, problem in summary["problems"].items():
        config, clip = name.split("/", 1)
        if problem.startswith("rate:") and config.split("+")[0].endswith("qp-low"):
            failed.add((config.split("-")[0], clip))
    def failed_before(job: Job) -> bool:
        key = f"{job.clip}@{job.start_frame}"
        codecs = {job.codec, RATE_PROXY.get(job.codec, job.codec)}
        return job.rate == "qp-low" and any((codec, key) in failed for codec in codecs)
    return [job for job in jobs if not failed_before(job)]
