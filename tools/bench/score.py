#!/usr/bin/env python3
"""The box side of a BP-3 score stage: decode each stored stream, check it, and measure it.

    python3 -I score.py --encodes encodes.jsonl --jobs jobs.json --clips DIR --out out/ \
        --workers N [--cross-sample 0.02] [--bin DIR] [--metrics-backend libvmaf|vmafx]

Runs on a scoring box (docs/benchmark-procedure.md sections 7 and 10) beside `bin/` (the
decoders, `vmaf` and its v1 model) and a copy of `nceval/` (display, yuv, codecs). It reads
the encode stage's records and streams; nothing is re-encoded. Per stream:

  - V23 the score record carries the stream's SHA-256, recomputed from the file, and the
        encode record's machine series, so the report can refuse a borrowed score;
  - V9  the decoded frame count equals the job's (a '+drop' row's dropped frames show the
        previous picture, placed by their IVF timestamps, and are counted); a stream whose frame sizes change is
        decoded at each frame's size and upscaled by the pinned player model
        (`nceval.display.scale_frame`);
  - V10, only when asked (`--cross-sample`, default 0; D18): a seeded share of streams is
        decoded by every decoder of CROSS_DECODERS for their codec, and the per-frame MD5s must
        agree. The benchmark does not prove conformance: the landing check does;
  - the metrics of `nceval.jreal.HEADLINE_METRICS`: libvmaf's PSNR per plane, VMAF v1 and
        MS-SSIM, per frame and pooled (the report weights PSNR-YUV 6:1:1);
  - the frame sizes in bytes, for the report's bucket replay (section 7);
  - V29/V30 the stream's own structure (`streamcheck`), read from its bytes with no extra
        decode: AV1/VP9 reordering (hidden frames) and key frames, every codec's first-frame key.

Two metric backends give the same numbers (procedure section 7, V27); every score record names
the one it used:
  - `libvmaf` (a scoring box): libvmaf 3.2.1 on the CPU, PSNR per plane, VMAF v1 and MS-SSIM;
  - `vmafx` (the PC, `fleet.py score`, the default): the existing GPU scoring service of the dev
    tools (`nceval.metrics`: VMAF v1 and MS-SSIM in one VMAFx CUDA run on a shared server,
    VMAF v1 rounded to libvmaf's printed precision; PSNR per plane from exact squared-error
    sums with libvmaf's 60 dB cap). It imports `nceval` from the repository.

A failed check refuses the stream: its record says why, and the box exits non-zero after
the queue, so no point is ever scored silently wrong.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
from nceval.codecs import Codec  # noqa: E402  (shipped beside this file)
from nceval.display import scale_frame  # noqa: E402
from nceval.yuv import YuvFormat  # noqa: E402
import streamcheck  # noqa: E402  (shipped beside this file)

BIN = HERE / "bin"
VMAF_MODEL_NAME = "vmaf_v1.0.16_3d0h.json"
#: The metric backends (module docstring); each score record names its own.
LIBVMAF_BACKEND, VMAFX_BACKEND = "libvmaf", "vmafx"
#: The record keys of the headline metrics, as both backends report them.
METRIC_KEYS = ("psnr_y", "psnr_u", "psnr_v", "vmaf_v1", "ms_ssim")
IVF_HEADER, IVF_FRAME_HEADER = 32, 12
#: Decoders per codec. The first, the codec's standard decoder, decodes every stream for
#: scoring, ours included (section 6); V10 needs every one of them to agree, frame by frame.
CROSS_DECODERS = {
    Codec.H264: (["ffmpeg"], ["{BIN}/nanocodec", "decode", "--codec", "h264"]),
    Codec.HEVC: (["ffmpeg"], ["{BIN}/nanocodec", "decode", "--codec", "hevc"]),
    Codec.VVC: (["ffmpeg"], ["{BIN}/nanocodec", "decode", "--codec", "vvc"]),
    Codec.AV1: (["ffmpeg"], ["{BIN}/aomdec"], ["{BIN}/nanocodec", "decode", "--codec", "av1"]),
    Codec.VP9: (["ffmpeg"], ["{BIN}/vpxdec"], ["{BIN}/nanocodec", "decode", "--codec", "vp9"]),
    Codec.AV2: (["{BIN}/avmdec"], ["{BIN}/nanocodec", "decode", "--codec", "av2"]),
}
#: Per-frame-size decoders for streams whose frame size changes (they write each frame at
#: its own size; ffmpeg would rescale).
SIZED_DECODERS = {Codec.AV1: "aomdec", Codec.VP9: "vpxdec"}
PROGRESS_EVERY = 10
#: "1": decode into memory (memfd) instead of a file under the clips directory (the PC).
DECODE_IN_MEMORY_ENV = "NC_SCORE_DECODE_IN_MEMORY"
#: Bytes the pass's decodes may hold at once (workers are capped to fit it).
MEMORY_BUDGET_ENV = "NC_SCORE_MEMORY_BUDGET"
DROP_SUFFIX = "+drop"


def memory_file(name: str) -> int:
    """An anonymous in-memory file (memfd_create; this Python's os module may lack it)."""
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name)
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.memfd_create(name.encode(), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create failed")
    return fd


class StreamRefusal(RuntimeError):
    """A check failed for one stream (section 10)."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_argv(template: list[str], stream: Path, raw: Path) -> list[str]:
    if template == ["ffmpeg"]:
        # passthrough: one output frame per coded frame, as every other decoder gives; by
        # default ffmpeg repeats frames to fill timestamp gaps (a '+drop' row's dropped frames),
        # which V10 then sees as other frames, and fill_dropped would place twice
        return ["ffmpeg", "-v", "error", "-y", "-i", str(stream), "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", "yuv420p", str(raw)]
    argv = [a.replace("{BIN}", str(BIN)) for a in template]
    if argv[0].endswith(("aomdec", "vpxdec", "avmdec")):
        return [*argv, "--rawvideo", "-o", str(raw), str(stream)]
    return [*argv, "--input", str(stream), "--output", str(raw)]


def run(argv: list[str]) -> None:
    proc = subprocess.run(argv, capture_output=True)
    if proc.returncode:
        raise StreamRefusal(f"{Path(argv[0]).name} exit {proc.returncode}: "
                            f"{proc.stderr[-300:].decode(errors='replace')}")


def frame_bytes(stream: Path, codec: Codec, data: bytes) -> list[int]:
    """Bytes per coded frame: IVF frame headers, or ffprobe's access-unit packets."""
    if codec.bitstream_suffix == ".ivf":
        sizes, pos = [], IVF_HEADER
        while pos + IVF_FRAME_HEADER <= len(data):
            size = int.from_bytes(data[pos:pos + 4], "little")
            sizes.append(size)
            pos += IVF_FRAME_HEADER + size
        return sizes
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "packet=size", "-of",
                          "csv=p=0", str(stream)], capture_output=True, text=True, check=True)
    return [int(x.split(",")[0]) for x in out.stdout.split()]


def frame_sizes(stream: Path) -> list[tuple[int, int]]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "frame=width,height",
                          "-of", "csv=p=0", str(stream)], capture_output=True, text=True,
                         check=True)
    return [tuple(int(v) for v in line.split(",")[:2]) for line in out.stdout.split()]


def decode(stream: Path, codec: Codec, shown: YuvFormat, raw: Path) -> list[tuple[int, int]]:
    """Decode for scoring at the shown size; returns the coded frame sizes."""
    sizes = frame_sizes(stream) if codec in SIZED_DECODERS else []
    if any(size != (shown.width, shown.height) for size in sizes):
        run(decode_argv([f"{{BIN}}/{SIZED_DECODERS[codec]}"], stream, raw))
        data, pos, frames = raw.read_bytes(), 0, []
        for width, height in sizes:
            coded = YuvFormat(width, height)
            n = coded.luma_size * 3 // 2
            frames.append(scale_frame(data[pos:pos + n], coded, shown))
            pos += n
        raw.write_bytes(b"".join(frames))
    elif os.environ.get(OURS_DECODER_ENV):
        try:
            run([os.environ[OURS_DECODER_ENV], "decode", "--codec", codec.value,
                 "--input", str(stream), "--output", str(raw)])
        except StreamRefusal as why:
            # a stream ours cannot decode is a decoder bug: recorded, and the stream is scored
            # with the codec's standard decoder (labelled in its record)
            record_decoder_bug(stream, codec, str(why))
            run(decode_argv(CROSS_DECODERS[codec][0], stream, raw))
            DECODED_BY[stream.name] = "standard (ours failed: decoder bug)"
    else:
        run(decode_argv(CROSS_DECODERS[codec][0], stream, raw))
    return sizes


#: Streams decoded by the standard decoder after ours failed, by stream file name.
DECODED_BY: dict[str, str] = {}
#: Where decoder bugs found while scoring are listed (owner, 2026-11-26).
DECODER_BUGS_ENV = "NC_SCORE_DECODER_BUGS"


def record_decoder_bug(stream: Path, codec: Codec, error: str) -> None:
    path = os.environ.get(DECODER_BUGS_ENV)
    if path:
        with open(path, "a") as f:
            f.write(f"| {stream.name} | {codec.value} | {stream.name.split('_')[0]} | "
                    f"ours failed: {error[-200:].replace('|', '/')} |\n")


#: The PC's scoring decoder (owner, 2026-11-26): our own frozen build decodes every stream of
#: every codec, references included (OURS_DECODER_ENV: its path); a stream it cannot decode is a
#: decoder bug, listed, and decoded by the standard decoder instead. Its identity with the
#: standard decoders is checked once on a sample (bench/decode-identity-2026-11-26.md). NVDEC
#: was measured and not used: with frames copied back for VMAFx it cost as much CPU or more.
OURS_DECODER_ENV = "NC_SCORE_OURS_DECODER"


def ivf_slots(data: bytes, fps: float) -> list[int]:
    """Each coded frame's display slot, from its IVF timestamp and the header's timebase."""
    rate, scale = (int.from_bytes(data[16 + 4 * i:20 + 4 * i], "little") for i in range(2))
    slots, pos = [], IVF_HEADER
    while pos + IVF_FRAME_HEADER <= len(data):
        size = int.from_bytes(data[pos:pos + 4], "little")
        pts = int.from_bytes(data[pos + 4:pos + 12], "little")
        slots.append(round(pts * scale / rate * fps))
        pos += IVF_FRAME_HEADER + size
    return slots


def drop_slots(encode: dict, job: dict, codec: Codec, data: bytes) -> list[int]:
    """A '+drop' row's display slot per coded picture: from IVF timestamps, or, for a stream
    without timestamps (openh264's Annex B), from the input frame list its encoder recorded
    (`coded_frames`, box.py; task T16)."""
    if codec.bitstream_suffix == ".ivf":
        num, den = (int(x) for x in job["fps"].split("/"))
        return ivf_slots(data, num / den)
    if "coded_frames" not in encode:
        raise StreamRefusal("a '+drop' row needs frame timestamps (IVF) or the encoder's "
                            "coded-frame list")
    return encode["coded_frames"]


def bytes_per_slot(coded_bytes: list[int], slots: list[int], frames: int) -> list[int]:
    """Bytes per display slot, 0 for a dropped frame, so the rate and the buffer replay
    (report.py) count a '+drop' stream over its whole duration, not over its coded pictures."""
    if len(coded_bytes) != len(slots):
        raise StreamRefusal(f"V9: {len(coded_bytes)} coded pictures, {len(slots)} slots")
    out = [0] * frames
    for size, slot in zip(coded_bytes, slots):
        out[slot] = size
    return out


def fill_dropped(raw: Path, slots: list[int], frame: int, frames: int) -> int:
    """A '+drop' row: every dropped frame shows the previous picture (review 3 C1). Returns how
    many frames were dropped."""
    data = raw.read_bytes()
    coded = [data[i * frame:(i + 1) * frame] for i in range(len(slots))]
    shown, k = [], 0
    for slot in range(frames):
        while k + 1 < len(slots) and slots[k + 1] <= slot:
            k += 1
        shown.append(coded[k])
    raw.write_bytes(b"".join(shown))
    return frames - len(slots)


def frame_md5s(raw: Path, frame: int) -> list[str]:
    data = raw.read_bytes()
    return [hashlib.md5(data[i:i + frame]).hexdigest() for i in range(0, len(data), frame)]


def cross_decode(stream: Path, codec: Codec, shown: YuvFormat, work: Path) -> None:
    """V10: every decoder of the codec gives the same frames."""
    frame, first = shown.luma_size * 3 // 2, None
    for template in CROSS_DECODERS[codec]:
        raw = work / "cross.yuv"
        run(decode_argv(template, stream, raw))
        md5s = frame_md5s(raw, frame)
        raw.unlink()
        if first is None:
            first = md5s
        elif md5s != first:
            raise StreamRefusal(f"V10: {Path(template[0]).name} decodes other frames")


def vmaf(source: Path, raw: Path, shown: YuvFormat, work: Path) -> dict:
    """libvmaf on one thread: PSNR per plane, VMAF v1, MS-SSIM, per frame and pooled."""
    out = work / "vmaf.json"
    run([str(BIN / "vmaf"), "-r", str(source), "-d", str(raw), "-w", str(shown.width),
         "-h", str(shown.height), "-p", "420", "-b", "8", "-m",
         f"path={BIN / VMAF_MODEL_NAME}:name=vmaf_v1", "--feature", "psnr", "--feature", "float_ms_ssim",
         "--threads", "1", "-q", "--json", "-o", str(out)])
    result = json.loads(out.read_text())
    keys = {"psnr_y": "psnr_y", "psnr_u": "psnr_cb", "psnr_v": "psnr_cr", "vmaf_v1": "vmaf_v1",
            "ms_ssim": "float_ms_ssim"}
    pooled = {k: result["pooled_metrics"][v]["mean"] for k, v in keys.items()}
    per_frame = {k: [f["metrics"][v] for f in result["frames"]] for k, v in keys.items()}
    return {"pooled": pooled, "per_frame": per_frame, "metrics_backend": LIBVMAF_BACKEND}


def gpu_metrics(source: Path, raw: Path, shown: YuvFormat, fps: float) -> dict:
    """The `vmafx` backend: the dev tools' GPU scoring service (module docstring), in the
    libvmaf backend's record layout."""
    from nceval import config, metrics  # the repository's nceval: the PC only
    measured = metrics.score_files(
        source, raw, shown, fps, config.load_paths(),
        lambda spec: [metrics.PsnrMetric(spec), *metrics.vmaf_v1_and_ms_ssim(spec, ms_ssim=True)])
    pooled = {k: measured.summary[k] for k in METRIC_KEYS}
    per_frame = {k: [row[k] for row in measured.per_frame] for k in METRIC_KEYS}
    return {"pooled": pooled, "per_frame": per_frame, "metrics_backend": VMAFX_BACKEND}


#: "1": the vmafx backend runs VMAFx once per stream on the decoded file itself, PSNR included
#: (libvmaf's psnr: per-plane SSE, capped at 60 dB for 8 bits, the same values as ours to the
#: last digit), so no frame passes through Python (2026-11-26: the frame pumping and the Python
#: PSNR held local scoring to ~260 frames/s on 16 cores).
DIRECT_ENV = "NC_SCORE_DIRECT"
#: libvmaf's psnr feature names, as the records name the planes.
VMAFX_PSNR_FEATURES = {"psnr_y": "psnr_y", "psnr_cb": "psnr_u", "psnr_cr": "psnr_v"}


def gpu_metrics_direct(source: Path, raw: Path, shown: YuvFormat, frames: int,
                       work: Path) -> dict:
    """The `vmafx` backend without the frame pipe: one VMAFx run reads the clip's first
    `frames` frames and the decoded stream, and scores VMAF v1, MS-SSIM and PSNR on CUDA."""
    from nceval import config, metrics  # the repository's nceval: the PC only
    paths = config.load_paths()
    spec = metrics.StreamSpec(source, shown, frames, 0.0, paths)
    report_path = work / "vmafx.json"
    argv = [*metrics.vmafx_command(paths), "-r", str(source), "-d", str(raw),
            "-w", str(shown.width), "-h", str(shown.height), "-p", "420", "-b", "8",
            "--frame_cnt", str(frames), "--model", metrics._vmaf_v1_model_argument(spec),
            "--feature", "float_ms_ssim", "--feature", "psnr", "--backend", "cuda",
            "--precision", "max", "--json", "-o", str(report_path), "-q"]
    run(argv)
    report = json.loads(report_path.read_text())
    metrics.check_vmafx_report(report, frames, ms_ssim=True)
    rows = [f["metrics"] for f in sorted(report["frames"], key=lambda f: f["frameNum"])]
    per_frame = {"vmaf_v1": [metrics._libvmaf_printed(r["vmaf_v1"]) for r in rows],
                 "ms_ssim": [float(r["float_ms_ssim"]) for r in rows]}
    for theirs, ours in VMAFX_PSNR_FEATURES.items():
        per_frame[ours] = [min(metrics.PSNR_CAP_DB, float(r[theirs])) for r in rows]
    pooled = {k: sum(v) / len(v) for k, v in per_frame.items()}
    pooled["vmaf_v1"] = metrics._libvmaf_printed(report["pooled_metrics"]["vmaf_v1"]["mean"])
    return {"pooled": {k: pooled[k] for k in METRIC_KEYS},
            "per_frame": {k: per_frame[k] for k in METRIC_KEYS}, "metrics_backend": VMAFX_BACKEND}


def score(encode: dict, job: dict, clips: Path, out: Path, cross: bool,
          backend: str = LIBVMAF_BACKEND) -> dict:
    stream = out / encode["stream"]
    data = stream.read_bytes()
    codec = Codec(job["codec"])
    shown = YuvFormat(job["w"], job["h"])
    rec = {"id": encode["id"], "family": encode["family"], "series": encode["series"],
           "start_unix": round(time.time(), 1),
           "stream_sha256": sha256(data), "frame_bytes": frame_bytes(stream, codec, data)}
    if rec["stream_sha256"] != encode["stream_sha256"]:
        raise StreamRefusal("V23: the stream is not the one the encode stage recorded")
    try:  # a probe failure is recorded, and the report refuses the point on it (V29)
        rec["structure"] = streamcheck.structure(stream, job["codec"])
    except (ValueError, IndexError) as why:
        rec["structure"] = {"probe": "failed", "error": str(why)[-200:]}
    # the decoded files live on the clips' tmpfs, in memory (2026-11-24: written to a 40 GB
    # root disk, 90 workers' 1080p decodes filled it and the scoring boxes stalled)
    work = clips / "work" / encode["id"].replace("/", "_").replace("|", "_")
    work.mkdir(parents=True, exist_ok=True)
    raw, memfd, held = work / "dec.yuv", None, 0
    if DECODE_BUDGET is not None:
        held = shown.luma_size * 3 // 2 * job["frames"] + PC_WORKER_OVERHEAD_BYTES
        DECODE_BUDGET.acquire(held)
    if os.environ.get(DECODE_IN_MEMORY_ENV) == "1":
        # the PC has no tmpfs for this, and its disk measured 38 MB/s at 91 % busy under 16
        # decodes (2026-11-26): the decode goes to an anonymous memory file instead, freed
        # when its stream is scored (bounded by the workers and the scope's MemoryMax)
        memfd = memory_file("nanocodec-dec")
        raw = Path(f"/proc/{os.getpid()}/fd/{memfd}")
    try:
        sizes = decode(stream, codec, shown, raw)
        frame = shown.luma_size * 3 // 2
        if job["family"].endswith(DROP_SUFFIX):
            slots = drop_slots(encode, job, codec, data)
            rec["dropped_frames"] = fill_dropped(raw, slots, frame, job["frames"])
            rec["frame_bytes"] = bytes_per_slot(rec["frame_bytes"], slots, job["frames"])
        decoded = raw.stat().st_size // frame
        if decoded != job["frames"]:
            raise StreamRefusal(f"V9: decoded {decoded} frames of {job['frames']}")
        rec["coded_small_frames"] = sum(size != (shown.width, shown.height) for size in sizes)
        rescaled = any(size != (shown.width, shown.height) for size in sizes)
        rec["decoder"] = DECODED_BY.get(stream.name, "ours" if os.environ.get(OURS_DECODER_ENV)
                                        and not rescaled else "standard")
        if cross and not rec["coded_small_frames"]:
            cross_decode(stream, codec, shown, work)
            rec["cross_decoded"] = True
        source = clips / f"{job['input']}.yuv"
        if backend == VMAFX_BACKEND:
            num, den = (int(x) for x in job["fps"].split("/"))
            if os.environ.get(DIRECT_ENV) == "1":
                rec.update(gpu_metrics_direct(source, raw, shown, job["frames"], work))
            else:
                rec.update(gpu_metrics(source, raw, shown, num / den))
        else:
            rec.update(vmaf(source, raw, shown, work))
        rec["end_unix"] = round(time.time(), 1)  # for the driver's measured rates
    finally:
        if memfd is not None:
            os.close(memfd)
        if held:
            DECODE_BUDGET.release(held)
        for f in work.iterdir():
            f.unlink()
        work.rmdir()
    return rec


#: Seconds between heartbeat lines: the driver's stall watch stops a box silent for 5 min, and
#: on 2026-11-24 five c7a.24xlarge boxes scoring 1080p AOM-CTC streams at 90 workers went
#: over 5 min between completions. Each line carries the box's load, so a slow box shows why.
HEARTBEAT_S = 60


def box_load(out: Path) -> str:
    """Free memory, free space under `out` (the work files' disk), the load average, and the
    MB written to disks since boot."""
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    free_gb = int(mem["MemAvailable"].split()[0]) / 2**20
    disk = os.statvfs(out)
    written = sum(int((dev / "stat").read_text().split()[6])  # sectors written, whole disks
                  for dev in Path("/sys/block").iterdir()
                  if not dev.name.startswith(("loop", "ram"))) * 512 / 1e6
    return (f"mem {free_gb:.0f} GB free, disk {disk.f_bavail * disk.f_frsize / 1e9:.1f} GB free, "
            f"load {Path('/proc/loadavg').read_text().split()[0]}, written {written:.0f} MB")


def heartbeat(out: Path) -> None:
    """Print a line with the box's load every HEARTBEAT_S seconds while scoring runs."""
    def beat() -> None:
        while True:
            time.sleep(HEARTBEAT_S)
            print(f"alive: {box_load(out)}", flush=True)
    threading.Thread(target=beat, daemon=True).start()


#: Memory per scoring worker beyond its decoded files (libvmaf's frame buffers and features,
#: the decoder): a generous bound, so the cap errs towards fewer workers.
WORKER_OVERHEAD_BYTES = 1 << 30
#: A PC worker beside its decode: the thread, its ffmpeg and the VMAFx client.
PC_WORKER_OVERHEAD_BYTES = 256 << 20
#: Memory kept free for the system and the clips still to be read.
MEMORY_MARGIN_BYTES = 8 << 30


def worker_cap(requested: int, jobs: list[dict], meminfo: str | None = None,
               cpus: int | None = None) -> int:
    """The workers that fit in memory: each holds up to two decodes of its stream (the decode
    and a cross-decode) on tmpfs, plus WORKER_OVERHEAD_BYTES."""
    if not jobs:
        return 1
    decoded = max(j["w"] * j["h"] * 3 // 2 * j["frames"] for j in jobs)
    text = meminfo if meminfo is not None else Path("/proc/meminfo").read_text()
    mem = dict(line.split(":", 1) for line in text.splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024 - MEMORY_MARGIN_BYTES
    if os.environ.get(MEMORY_BUDGET_ENV):
        # a pass running inside a memory-capped unit (the PC): its decodes fit that cap, not
        # the machine's free memory (2026-11-26: an uncapped pass took the terminal to an OOM)
        available = min(available, int(os.environ[MEMORY_BUDGET_ENV]))
    cpus = cpus or os.cpu_count() or 1
    requested = min(requested, cpus) if requested else cpus  # 0: one worker per CPU
    if os.environ.get(MEMORY_BUDGET_ENV) and os.environ.get(DECODE_IN_MEMORY_ENV) == "1":
        # the PC: each stream takes its own decode's bytes from DECODE_BUDGET while it scores,
        # so the workers are not all sized for the pass's largest stream (2026-11-26: four
        # 600-frame clips held a whole VC-1K pass to one worker per process)
        return max(1, requested)
    per_worker = 2 * decoded + WORKER_OVERHEAD_BYTES
    return max(1, min(requested, available // per_worker))


class ByteBudget:
    """Bytes of in-memory decodes held at once (MEMORY_BUDGET_ENV); a stream larger than the
    whole budget still runs, alone."""

    def __init__(self, total: int) -> None:
        self.total, self.used = total, 0
        self.cond = threading.Condition()

    def acquire(self, n: int) -> None:
        with self.cond:
            while self.used and self.used + n > self.total:
                self.cond.wait()
            self.used += n

    def release(self, n: int) -> None:
        with self.cond:
            self.used -= n
            self.cond.notify_all()


DECODE_BUDGET: ByteBudget | None = None


def main() -> int:
    global BIN  # --bin: the scoring tools' directory, read by decode_argv and vmaf
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encodes", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="the encode stage's output dir")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--cross-sample", type=float, default=0.0,
                    help="share of streams cross-decoded (V10); 0 in benchmark runs (D18: "
                         "conformance is proved by the landing check, not the benchmark)")
    ap.add_argument("--bin", type=Path, default=BIN,
                    help="the decoders, `vmaf` and its model (default: bin/ beside this file)")
    ap.add_argument("--metrics-backend", choices=(LIBVMAF_BACKEND, VMAFX_BACKEND),
                    default=LIBVMAF_BACKEND)
    a = ap.parse_args()
    BIN = a.bin
    if a.metrics_backend == VMAFX_BACKEND:
        os.environ["NC_METRICS_BACKEND"] = "vmafx-cuda"  # nceval.metrics: never the CPU
    heartbeat(a.out)
    jobs = {j["id"]: j for j in json.loads(a.jobs.read_text())}
    # one per id (a re-coded job's two records name the same stream; scoring both at once would
    # share a work dir)
    encodes = list({r["id"]: r for r in map(json.loads, a.encodes.read_text().splitlines())
                    if "stream" in r and "canary" not in r and "reference" not in r
                    and "check" not in r}.values())  # BP-4 check sets: timing checks only
    rng = random.Random("cross-sample")
    cross = {r["id"]: rng.random() < a.cross_sample
             for r in encodes}
    workers = worker_cap(a.workers, [jobs[r["id"]] for r in encodes])
    global DECODE_BUDGET
    if os.environ.get(MEMORY_BUDGET_ENV) and os.environ.get(DECODE_IN_MEMORY_ENV) == "1":
        DECODE_BUDGET = ByteBudget(int(os.environ[MEMORY_BUDGET_ENV]))
    print(f"scoring {len(encodes)} streams with {workers} workers", flush=True)
    refused = scored = 0
    with open(a.out / "scores.jsonl", "a") as log, \
            concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(score, r, jobs[r["id"]], a.clips, a.out, cross[r["id"]],
                               a.metrics_backend): r
                   for r in encodes}
        for done in concurrent.futures.as_completed(futures):
            try:
                rec = done.result()
            except (StreamRefusal, subprocess.CalledProcessError) as why:
                refused += 1
                rec = {"id": futures[done]["id"], "refused": str(why)}
            except Exception:
                # a scorer failure (2026-11-26: the MPS daemon gone, every VMAFx job failing)
                # stops the pass at once; leaving the pool to drain hid it for many minutes
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            log.write(json.dumps(rec) + "\n")
            log.flush()
            scored += 1
            if scored % PROGRESS_EVERY == 0 or scored == len(encodes):
                print(f"scored {scored}/{len(encodes)}", flush=True)  # the driver's stall watch
    if refused:
        print(f"refused: {refused} streams (scores.jsonl says why)", file=sys.stderr)
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
