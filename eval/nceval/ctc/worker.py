#!/usr/bin/env python3
"""Run the standards lens's jobs on a rented machine (standard library only).

    python3 worker.py prepare            # download, check and unpack the sources
    python3 worker.py run --workers N    # every job of jobs.json not yet done

The worker side of `nceval.ctc` (plan.py writes jobs.json and sources.json).
Run from the run directory; outputs go under out/.

`prepare` downloads each sequence from its public URL (a generic User-Agent),
checks the MD5 the CTC publishes (derf publishes none: the digests are
recorded instead, out/sources.json) and writes the first N frames as raw
I420 to the source tmpfs. No video ever leaves the machine.

`run` codes every job: fixed-QP jobs first, then the CBR jobs, whose target
is the rate their fixed-QP twin reached (here, or `target_kbps` when the twin
ran in an earlier launch). Per job, without any decoded file:

- the encode's user and system CPU time (`os.wait4`, the CTC's "user time");
- the bitstream's payload bytes (IVF file and frame headers not counted: the
  CTC's `.obu` bytes; Annex-B streams counted whole, as HM's files);
- a standard decoder's output, hashed per frame (MD5) and streamed into
  `vmaf --aom_ctc v6.0` plus the project's VMAF v1 model, against the source;
- for our streams: both standard decoders' per-frame hashes must equal the
  encoder's coded pictures (`--recon`); the metrics score the shown pictures
  (`--display-recon`, as all our evaluation does), which for fixed QP must
  equal the coded ones.

Each job writes out/results/<id>.json (or out/results/<id>.failed) and prints
one line, `job k/n ok|FAILED <id>`; a rerun skips finished jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUT = Path("out")
RESULTS = OUT / "results"
HASHES = OUT / "hashes"
USER_AGENT = "ctc-eval/1.0"
DOWNLOAD_CHUNK = 1 << 22
PIPE_CHUNK = 1 << 20
STDERR_TAIL_LINES = 12
IVF_FILE_HEADER = 32
IVF_FRAME_HEADER = 12
IVF_SIZE_FIELD = 4
VMAF_V1_MODEL = "vmaf_v1.0.16_3d0h.json"
#: Per-frame metrics kept in the results (the rest are kept as sequence means).
PER_FRAME = ("psnr_y", "psnr_cb", "psnr_cr")
#: The lean metric set's features: `--aom_ctc v6.0`'s own PSNR and MS-SSIM settings
#: (libvmaf v3.2.1 tools/cli_parse.c, aom_ctc_v6_0).
LEAN_PSNR = "psnr=reduced_hbd_peak=true:enable_apsnr=true:min_sse=0.5"
LEAN_MS_SSIM = "float_ms_ssim=enable_db=true:clip_db=true"
MEAN_METRICS = ("psnr_y", "psnr_cb", "psnr_cr", "float_ssim", "float_ms_ssim", "ciede2000",
                "psnr_hvs", "cambi", "vmaf", "vmaf_neg", "vmaf_v1")


def say(line: str) -> None:
    print(line, flush=True)


# --- prepare --------------------------------------------------------------------------

def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, open(path, "wb") as out:
        while chunk := response.read(DOWNLOAD_CHUNK):
            out.write(chunk)


def y4m_to_raw(y4m: Path, raw: Path, frames: int, frame_bytes: int) -> int:
    """The first `frames` frames of an 8-bit 4:2:0 Y4M file as raw I420; returns the count."""
    with open(y4m, "rb") as source, open(raw, "wb") as out:
        header = source.readline().split()
        colour = next((t for t in header if t.startswith(b"C")), b"C420")
        if not colour.startswith(b"C420") or b"p1" in colour:
            raise ValueError(f"{y4m.name}: not 8-bit 4:2:0 ({colour.decode()})")
        written = 0
        while written < frames and (marker := source.readline()):
            if not marker.startswith(b"FRAME"):
                raise ValueError(f"{y4m.name}: bad frame marker at frame {written}")
            frame = source.read(frame_bytes)
            if len(frame) != frame_bytes:
                break
            out.write(frame)
            written += 1
    return written


def prepare_one(source: dict, scratch: Path) -> dict:
    y4m = scratch / f"{source['name']}.y4m"
    download(source["url"], y4m)
    digest_md5, digest_sha = hashlib.md5(), hashlib.sha256()
    with open(y4m, "rb") as f:
        while chunk := f.read(DOWNLOAD_CHUNK):
            digest_md5.update(chunk)
            digest_sha.update(chunk)
    md5 = digest_md5.hexdigest()
    if source["md5"] and md5 != source["md5"]:
        raise ValueError(f"{source['name']}: MD5 {md5} differs from the CTC's {source['md5']}")
    Path(source["raw"]).parent.mkdir(parents=True, exist_ok=True)
    frames = y4m_to_raw(y4m, Path(source["raw"]), source["frames"], source["frame_bytes"])
    y4m.unlink()
    if frames != source["frames"]:
        raise ValueError(f"{source['name']}: {frames} frames, {source['frames']} needed")
    return {"name": source["name"], "url": source["url"], "md5": md5,
            "sha256": digest_sha.hexdigest(), "frames": frames}


def prepare(args: argparse.Namespace) -> None:
    sources = json.loads(Path("sources.json").read_text())
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare_one, s, scratch): s["name"] for s in sources}
        for future in as_completed(futures):
            records.append(future.result())
            say(f"source ok {futures[future]}")
    (OUT / "sources.json").write_text(json.dumps(sorted(records, key=lambda r: r["name"]),
                                                 indent=1) + "\n")
    say(f"prepared {len(records)} of {len(sources)} sources")


# --- one job ----------------------------------------------------------------------------

class FrameHasher:
    """Reads raw frames from a pipe, keeps their MD5s and optionally forwards the bytes."""

    def __init__(self, fd: int, frame_bytes: int, forward=None):
        self.fd, self.frame_bytes, self.forward = fd, frame_bytes, forward
        self.digests: list[str] = []
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        pending = bytearray()
        try:
            with os.fdopen(self.fd, "rb", buffering=0) as pipe:
                while chunk := pipe.read(PIPE_CHUNK):
                    pending += chunk
                    while len(pending) >= self.frame_bytes:
                        frame = bytes(pending[:self.frame_bytes])
                        del pending[:self.frame_bytes]
                        self.digests.append(hashlib.md5(frame).hexdigest())
                        if self.forward is not None:
                            self.forward.write(frame)
            if pending:
                self.error = f"partial frame of {len(pending)} bytes"
        except (OSError, ValueError) as error:
            self.error = str(error)

    def join(self) -> list[str]:
        self.thread.join()
        if self.error:
            raise RuntimeError(self.error)
        return self.digests


def fill(argv: list[str], values: dict[str, str]) -> list[str]:
    return [values.get(a, a) for a in argv]


def spawn_with_pipes(argv: list[str], names: list[str]) -> tuple[subprocess.Popen, dict[str, int]]:
    """Start `argv` with each placeholder in `names` replaced by a pipe's write end;
    returns the process and the read ends."""
    ends = {name: os.pipe() for name in names}
    command = fill(argv, {name: f"/dev/fd/{w}" for name, (_, w) in ends.items()})
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, pass_fds=[w for _, w in ends.values()])
    for _, w in ends.values():
        os.close(w)
    return process, {name: r for name, (r, _) in ends.items()}


def finish(process: subprocess.Popen, label: str) -> tuple[float, float]:
    """Wait for `process` (CPU times from wait4); raise with its stderr tail on failure."""
    stderr = process.stderr.read()
    _, status, usage = os.wait4(process.pid, 0)
    process.returncode = os.waitstatus_to_exitcode(status)
    if process.returncode != 0:
        tail = stderr.decode(errors="replace").strip().splitlines()[-STDERR_TAIL_LINES:]
        raise RuntimeError(f"{label} exit {process.returncode}: {' | '.join(tail)}")
    return usage.ru_utime, usage.ru_stime


def vmaf_process(job: dict, source: str, report: Path) -> subprocess.Popen:
    """libvmaf scoring the pictures written to its stdin against `source`.

    The job's metric set (`metrics`): "ctc" (default) is `--aom_ctc v6.0` plus
    VMAF v1; "lean" keeps only what the Weighted figure and our J read (the
    overall PSNR exactly as `--aom_ctc v6.0` computes it, MS-SSIM in dB, VMAF
    v1), a quarter of the CPU: the local iteration loop (ctc/local.py)."""
    width, height = job["width"], job["height"]
    model = f"path={VMAF_V1_MODEL}:name=vmaf_v1"
    if job.get("metrics", "ctc") == "lean":
        metrics = ["--feature", LEAN_PSNR, "--feature", LEAN_MS_SSIM]
    else:
        metrics = ["--aom_ctc", "v6.0"]
        model += f":cambi.enc_width={width}:cambi.enc_height={height}:cambi.enc_bitdepth=8"
    return subprocess.Popen(["bin/vmaf", "-r", source, "-d", "/dev/stdin", "-w", str(width),
                             "-h", str(height), "-p", "420", "-b", "8", *metrics,
                             "--model", model, "--threads", "1", "--json", "-o", str(report),
                             "-q"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def payload_bytes(bitstream: Path, container: str) -> tuple[int, int]:
    """(payload bytes, frames in the container; 0 when Annex-B)."""
    data = bitstream.read_bytes()
    if container != "ivf":
        return len(data), 0
    position, total, frames = IVF_FILE_HEADER, 0, 0
    while position < len(data):
        size = int.from_bytes(data[position:position + IVF_SIZE_FIELD], "little")
        total += size
        frames += 1
        position += IVF_FRAME_HEADER + size
    return total, frames


def decode_hashes(argv: list[str], bitstream: str, frame_bytes: int, forward=None) -> list[str]:
    process, reads = spawn_with_pipes(fill(argv, {"{BITSTREAM}": bitstream}), ["{OUT}"])
    hasher = FrameHasher(reads["{OUT}"], frame_bytes, forward)
    finish(process, f"decoder {Path(argv[0]).name}")
    return hasher.join()


def scored(report: Path) -> dict:
    data = json.loads(report.read_text())
    frames = sorted(data["frames"], key=lambda f: f["frameNum"])
    pooled = data["pooled_metrics"]
    return {"means": {m: pooled[m]["mean"] for m in MEAN_METRICS if m in pooled},
            "overall_psnr": data["aggregate_metrics"],
            "per_frame": {m: [f["metrics"][m] for f in frames] for m in PER_FRAME},
            "scored_frames": len(frames)}


def run_job(job: dict, kbps: float | None) -> dict:
    """One job end to end; returns its result record (raises on any failure)."""
    frame_bytes, frames = job["frame_bytes"], job["frames"]
    source = job["source"]
    Path(job["bitstream"]).parent.mkdir(parents=True, exist_ok=True)
    report = RESULTS / f"{job['id']}.vmaf.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    argv = fill(job["encode"], {"{KBPS}": str(round(kbps))} if kbps else {})
    vmaf = vmaf_process(job, source, report)
    started = time.monotonic()
    record = {"id": job["id"], "target_kbps": round(kbps) if kbps else None}
    if job["own"]:
        process, reads = spawn_with_pipes(argv, ["{RECON}", "{DISPLAY}"])
        coded = FrameHasher(reads["{RECON}"], frame_bytes)
        shown = FrameHasher(reads["{DISPLAY}"], frame_bytes, vmaf.stdin)
        user, system = finish(process, "encoder")
        record["wall_s"] = time.monotonic() - started
        coded_hashes, shown_hashes = coded.join(), shown.join()
        vmaf.stdin.close()
        decodes = {Path(d[0] if d[0] != "python3" else d[3]).name:
                   decode_hashes(d, job["bitstream"], frame_bytes) for d in job["decoders"]}
        record["decoders_match"] = {name: hashes == coded_hashes for name, hashes in decodes.items()}
        record["coded_frames"] = len(coded_hashes)
        record["shown_equals_coded"] = shown_hashes == coded_hashes
        hashes = coded_hashes
        if not all(record["decoders_match"].values()):
            raise RuntimeError(f"decoder mismatch {record['decoders_match']}")
        if job["kind"] == "qp" and not record["shown_equals_coded"]:
            raise RuntimeError("fixed-QP stream with held or dropped frames")
    else:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE)
        user, system = finish(process, "encoder")
        record["wall_s"] = time.monotonic() - started
        hashes = decode_hashes(job["decoders"][0], job["bitstream"], frame_bytes, vmaf.stdin)
        vmaf.stdin.close()
        if len(hashes) != frames:
            raise RuntimeError(f"decoded {len(hashes)} frames of {frames}")
    if vmaf.wait() != 0:
        raise RuntimeError(f"vmaf exit {vmaf.returncode}")
    payload, container_frames = payload_bytes(Path(job["bitstream"]), job["container"])
    fps = job["fps"][0] / job["fps"][1]
    record.update({"user_s": user, "sys_s": system, "bytes": payload,
                   "container_frames": container_frames,
                   "kbps": round(payload * 8 * fps / frames / 1000, 6), **scored(report)})
    if record["scored_frames"] != frames:
        raise RuntimeError(f"scored {record['scored_frames']} frames of {frames}")
    HASHES.joinpath(f"{job['id']}.md5").parent.mkdir(parents=True, exist_ok=True)
    HASHES.joinpath(f"{job['id']}.md5").write_text("\n".join(hashes) + "\n")
    report.unlink()
    Path(job["bitstream"]).unlink()
    return record


def guarded(job: dict, kbps: float | None) -> tuple[str, str | None]:
    """run_job, writing the result or the failure; returns (id, error or None)."""
    base = RESULTS / job["id"]
    try:
        record = run_job(job, kbps)
    except Exception as error:  # noqa: BLE001 - every failure is recorded and reported
        base.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{base}.failed").write_text(f"{type(error).__name__}: {error}\n")
        Path(job["bitstream"]).unlink(missing_ok=True)
        return job["id"], str(error)[:300]
    Path(f"{base}.json").write_text(json.dumps(record) + "\n")
    Path(f"{base}.failed").unlink(missing_ok=True)
    return job["id"], None


def done(job: dict) -> bool:
    return (RESULTS / f"{job['id']}.json").exists()


def run_batch(jobs: list[dict], workers: int, rates: dict[str, float]) -> int:
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(guarded, j, rates.get(j.get("rate_of", ""))) for j in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            key, error = future.result()
            failures += error is not None
            say(f"job {index}/{len(jobs)} " + (f"FAILED {key}: {error}" if error else f"ok {key}"))
    return failures


def run(args: argparse.Namespace) -> None:
    jobs = [j for j in json.loads(Path("jobs.json").read_text()) if not done(j)]
    fixed = [j for j in jobs if j["kind"] == "qp"]
    failures = run_batch(fixed, args.workers, {})
    rates = {}
    for job in jobs:
        if job["kind"] == "cbr":
            twin = RESULTS / f"{job['rate_of']}.json"
            if twin.exists():
                rates[job["rate_of"]] = json.loads(twin.read_text())["kbps"]
    rates.update({j["rate_of"]: j["target_kbps"] for j in jobs if "target_kbps" in j})
    cbr = [j for j in jobs if j["kind"] == "cbr" and j["rate_of"] in rates]
    failures += run_batch(cbr, args.workers, rates)
    say(f"finished: {len(fixed) + len(cbr)} jobs, {failures} failed")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--scratch", default="scratch")
    p.set_defaults(handler=prepare)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, required=True)
    r.set_defaults(handler=run)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main(sys.argv[1:])
