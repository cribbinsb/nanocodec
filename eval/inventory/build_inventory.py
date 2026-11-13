#!/usr/bin/env python3
"""Build the nanocodec test-clip inventory (eval/README.md "Methodology: Clips").

Walks the (read-only) source trees, probes every video file with ffprobe,
measures cheap content statistics on a short downscaled sample with ffmpeg
filters, proposes a content class for human review, flags duplicates, and
derives which rate–distortion ranges each clip may serve under the
"test points at ≤ 50% of the source bitrate" rule.

Per-file results are cached, so a rerun only processes new or changed files.
Uses only the Python 3 standard library plus the ffprobe/ffmpeg CLI.

Usage:
    build_inventory.py [--out DIR] [--workers N] [--limit N] [--force]

See README.md in this directory for the field reference and the heuristics.
"""

import argparse
import csv
import hashlib
import math
import json
import operator
import os
import re
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_ROOTS = ("/mldata/video", "/mldata/tracking_original")
DEFAULT_OUT = str(Path.home() / "nanocodec-data" / "inventory")
EARLIER_ATTEMPT_INPUTS = "/mldata/benchmarks/preview-codecs-nov05/iteration/inputs.json"

# Bump when the measurement or record layout changes; stale cache entries
# are then recomputed.
CACHE_VERSION = 1

CONTAINER_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".ts", ".ivf"}
ELEMENTARY_EXTS = {".264", ".h264", ".265", ".h265", ".hevc", ".obu"}
VIDEO_EXTS = CONTAINER_EXTS | ELEMENTARY_EXTS

DEFAULT_WORKERS = 12          # ≤ 16 concurrent ffmpeg processes (32-thread box)
FFMPEG_DECODE_THREADS = 2
NICE_INCREMENT = 10           # stay out of the way of interactive work
PROBE_TIMEOUT_S = 120
MEASURE_TIMEOUT_S = 600

HASH_CHUNK_BYTES = 1 << 20    # content hash = size + first and last 1 MiB

# Content sample: up to SAMPLE_SECONDS, starting a little way in (titles and
# camera start-up are not representative), at a fixed rate so motion
# statistics are comparable across clips with different native fps.
SAMPLE_SECONDS = 20.0
SAMPLE_SKIP_FRACTION = 0.10
SAMPLE_SKIP_MAX_S = 60.0
SAMPLE_FPS = 5.0
STATS_WIDTH = 320             # motion/luma statistics run at this width
NOISE_CROP_W, NOISE_CROP_H = 640, 360   # native-resolution crop for noise
NOISE_BLOCK = 8               # flat-block noise estimate block size
NOISE_GAIN = 8                # residual is amplified before 8-bit averaging
CHANGE_THRESHOLD = 10         # |Δluma| that counts a pixel as "changed"
SCDET_THRESHOLD = 10.0

# Global (camera) motion: exhaustive translation search between consecutive
# sample frames on tiny thumbnails. One thumbnail pixel = 5 px at 320 wide.
GM_WIDTH, GM_HEIGHT = 64, 36
GM_RANGE = 3                  # ± search range in thumbnail pixels
GM_MIN_RESIDUAL = 1.5         # mean |Δ| below this: nothing moved at all
GM_MIN_GAIN = 0.8             # best shift must cut the residual by ≥ 20%

# Bitrate rule (CLAUDE.md "Testing"): never test above half the source rate.
TEST_BITRATE_FRACTION = 0.5

# RD ranges (eval/README.md "Methodology: Rate–distortion and metrics"):
# ~150–800 kbps at 720p, scaled by pixel count for other resolutions. Primary operating point is 720p30.
REF_PIXELS = 1280 * 720
REF_RD_KBPS = (150.0, 800.0)
PRIMARY_MID_KBPS = 500.0
OPERATING_POINTS = {          # name: (width, height)
    "360p": (640, 360),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}
PRIMARY_OP = "720p"
PRIMARY_MIN_FPS = 24.0
RES_TOLERANCE = 0.95          # 1280x704 etc. still counts as 720p-capable

# Classification thresholds (on the 5 fps, 320-px-wide sample).
LOW_LIGHT_LUMA = 50.0         # mean Y (0–255) below this reads as low light
NIGHT_HINT_LUMA = 90.0        # night scenes with lights: trust a 'night' hint below this
IR_MONO_SATURATION = 0.5      # no colour at all: camera in IR night mode
STATIC_CHANGED_FRAC = 0.03    # median changed-pixel fraction: nothing moves
BUSY_CHANGED_FRAC = 0.10      # static camera but a lot of the frame moving
CAMERA_MOTION_FRAC = 0.2      # share of frame pairs with global translation
PAN_CONSISTENCY = 0.6         # resultant length of motion directions: steady pan
EDITED_CUTS_PER_MIN = 3.0     # many hard cuts: edited, not raw camera output

# Near-duplicate detection.
DUP_DURATION_TOL_S = 0.5
DUP_DURATION_TOL_REL = 0.01
DUP_ASPECT_TOL = 0.03
DUP_SIGNATURE_MAD = 1.0       # mean |ΔYAVG| between per-second signatures
DUP_LUMA_P10_TOL = 3.0        # re-encodes keep the dark end of the histogram
DUP_SATURATION_TOL = 1.0      # ... and the colour saturation
DUP_CHANGED_FRAC_TOL = 0.05

# Path/filename hints: (regex on the lower-cased path, class or tag, reason).
CLASS_HINTS = [
    (r"bwc|bodycam|body_cam|/antare", "handheld-bodycam", "path suggests body-worn camera"),
    (r"drone|aerial|\buav\b|dji|roundabouthd", "drone-aerial", "path suggests drone/aerial footage"),
    (r"night|llight|dark", "low-light", "path suggests night/low light"),
    (r"mot17|mot20|crowd|personpath|pedestrian|waiting_area|london|nyc|_hd_indoor",
     "crowd-busy", "path suggests crowd/busy scene"),
    (r"bdd100k|bdd_|jaad|drive|dashcam", "pan", "path suggests vehicle-mounted moving camera"),
    (r"cctv|ohcam|ffcam|meva|chirla|static|still|parking", "static",
     "path suggests fixed surveillance camera"),
]
TAG_HINTS = [
    (r"/long_composite/", "derived", "concatenated/re-encoded composite"),
    (r"/video/test/|/video/converted/", "derived", "re-encoded test copy"),
    (r"raw_movies|peep-show|john_wick|good_fortune|food_van|the_?studio|disclosureday"
     r"|theodyssey|theendofoakstreet|steadicam|long_take|tracking_shot|one_shot",
     "edited", "film/TV footage"),
    (r"/youtube/", "web-upload", "re-encoded web upload"),
]
FPS_IN_NAME = re.compile(r"_(\d+(?:\.\d+)?)fps", re.IGNORECASE)
CLASSES = ("static", "pan", "handheld-bodycam", "drone-aerial", "crowd-busy", "low-light")

# Shortlist of curated-set candidates for human review.
SHORTLIST_PER_CLASS = 10
SHORTLIST_PER_GROUP = 3       # per class and source folder, for diversity
SHORTLIST_MIN_DURATION_S = 12.0
SHORTLIST_MIN_CONFIDENCE = 0.6
SHORTLIST_EXCLUDED_TAGS = {"derived", "edited"}
SHORTLIST_FIELDS = [
    "auto_class", "path", "description", "width", "height", "fps", "duration_s",
    "source_kbps", "max_test_kbps", "ok_720p30_800", "ok_720p30_500",
    "class_confidence", "earlier_attempt",
]

CSV_FIELDS = [
    "path", "source_group", "size_bytes", "container", "codec", "profile", "width", "height",
    "fps", "fps_source", "duration_s", "frame_count", "pix_fmt",
    "source_kbps", "bitrate_source", "max_test_kbps",
    "ok_720p30_800", "ok_720p30_500", "ok_1080p_full", "ok_360p_full",
    "auto_class", "class_confidence", "class_reasons", "tags",
    "luma_mean", "changed_frac_median", "ydif_median", "ydif_cv",
    "detail", "noise_est", "scene_cuts",
    "content_hash", "duplicate_of", "duplicate_kind", "earlier_attempt", "error",
]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def run(cmd, timeout):
    """Run a command at reduced priority; return (returncode, stdout bytes, stderr)."""
    proc = subprocess.run(
        cmd, capture_output=True, timeout=timeout,
        preexec_fn=lambda: os.nice(NICE_INCREMENT), stdin=subprocess.DEVNULL)
    return proc.returncode, proc.stdout, proc.stderr.decode(errors="replace")


def parse_rate(text):
    """Parse an ffprobe rational like '30000/1001'; None if absent or invalid."""
    if not text or text in ("0/0", "N/A"):
        return None
    num, _, den = text.partition("/")
    try:
        value = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def to_float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def content_hash(path, size):
    """Cheap duplicate key: sha256 of the size plus the first and last 1 MiB."""
    digest = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        digest.update(f.read(HASH_CHUNK_BYTES))
        if size > 2 * HASH_CHUNK_BYTES:
            f.seek(-HASH_CHUNK_BYTES, os.SEEK_END)
            digest.update(f.read(HASH_CHUNK_BYTES))
    return digest.hexdigest()


def cache_path(cache_dir, path):
    return cache_dir / (hashlib.sha1(path.encode()).hexdigest() + ".json")


# --------------------------------------------------------------------------
# Discovery and probing
# --------------------------------------------------------------------------

def discover(roots):
    """Return (sorted video paths, {extension: count} of skipped files)."""
    videos, skipped = [], {}
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in VIDEO_EXTS:
                    videos.append(os.path.join(dirpath, name))
                else:
                    key = ext or "(none)"
                    skipped[key] = skipped.get(key, 0) + 1
    return sorted(videos), skipped


def probe(path):
    """Probe the first video stream. Elementary streams have no container
    timing, so their packets are counted and fps comes from the filename
    (e.g. '_7.5fps') when present, else ffprobe's assumed rate."""
    is_elementary = Path(path).suffix.lower() in ELEMENTARY_EXTS
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries",
           "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,"
           "avg_frame_rate,bit_rate,nb_frames,duration"
           ":format=format_name,duration,bit_rate",
           "-of", "json", path]
    if is_elementary:
        cmd[1:1] = ["-count_packets"]
        cmd[cmd.index("-show_entries") + 1] = cmd[cmd.index("-show_entries") + 1].replace(
            "nb_frames", "nb_frames,nb_read_packets")
    code, out, err = run(cmd, PROBE_TIMEOUT_S)
    if code != 0:
        raise RuntimeError("ffprobe failed: " + err.strip()[-300:])
    info = json.loads(out or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError("no video stream")
    s, fmt = streams[0], info.get("format", {})

    name_fps = FPS_IN_NAME.search(Path(path).name)
    avg, real = parse_rate(s.get("avg_frame_rate")), parse_rate(s.get("r_frame_rate"))
    if is_elementary and name_fps:
        fps, fps_source = float(name_fps.group(1)), "filename"
    elif is_elementary:
        fps, fps_source = avg or real, "assumed"
    elif avg:
        fps, fps_source = avg, "avg_frame_rate"
    else:
        fps, fps_source = real, "r_frame_rate"

    frames = to_float(s.get("nb_frames")) or to_float(s.get("nb_read_packets"))
    duration = to_float(s.get("duration")) or to_float(fmt.get("duration"))
    if is_elementary and frames and fps:
        duration = frames / fps
    frame_count_source = "stream"
    if not frames and duration and fps:
        frames, frame_count_source = round(duration * fps), "estimated"

    size = os.path.getsize(path)
    stream_bps = to_float(s.get("bit_rate"))
    if stream_bps and not is_elementary:
        source_kbps, bitrate_source = stream_bps / 1000, "stream"
    elif duration:
        source_kbps = size * 8 / duration / 1000
        bitrate_source = "file_size" if is_elementary else "file_size_incl_audio"
    else:
        source_kbps, bitrate_source = None, None

    return {
        "container": "elementary" if is_elementary else fmt.get("format_name"),
        "codec": s.get("codec_name"),
        "profile": s.get("profile"),
        "width": s.get("width"),
        "height": s.get("height"),
        "pix_fmt": s.get("pix_fmt"),
        "fps": round(fps, 3) if fps else None,
        "fps_source": fps_source,
        "r_frame_rate": s.get("r_frame_rate"),
        "avg_frame_rate": s.get("avg_frame_rate"),
        "duration_s": round(duration, 3) if duration else None,
        "frame_count": int(frames) if frames else None,
        "frame_count_source": frame_count_source,
        "source_kbps": round(source_kbps, 1) if source_kbps else None,
        "bitrate_source": bitrate_source,
    }


# --------------------------------------------------------------------------
# Content statistics
# --------------------------------------------------------------------------

def sample_window(duration, is_elementary):
    """(start, length) of the content sample. Elementary streams cannot seek
    cheaply, so they are sampled from the start."""
    if not duration:
        return 0.0, SAMPLE_SECONDS
    start = 0.0
    if not is_elementary and duration > SAMPLE_SECONDS * 1.5:
        start = min(duration * SAMPLE_SKIP_FRACTION, SAMPLE_SKIP_MAX_S)
    return start, min(SAMPLE_SECONDS, duration - start)


def build_filtergraph(motion_file, change_file, noise_file):
    """Four branches over the same 5 fps sample:
      motion: 320 px wide → signalstats (YAVG, YDIF, SATAVG) + scdet cuts;
      change: |frame − previous| thresholded → YAVG/255 = changed-pixel fraction;
      noise:  native-res centre crop, |x − blur(x)| (amplified) averaged per
              8×8 block → YAVG = texture/detail, YLOW (10th percentile block)
              = flat-block noise estimate;
      thumb:  64×36 grey thumbnails for the global-motion search ([o4]).
    """
    def meta(file):
        return "metadata=mode=print:file=" + file
    return ";".join([
        f"[0:v]fps={SAMPLE_FPS},split=2[full][nat]",
        f"[full]scale={STATS_WIDTH}:-2:flags=area,format=yuv420p,split=3[m][c][t]",
        f"[m]signalstats,scdet=threshold={SCDET_THRESHOLD},{meta(motion_file)}[o1]",
        f"[c]tblend=all_mode=difference,"
        f"lutyuv=y='if(gt(val,{CHANGE_THRESHOLD}),255,0)',signalstats,{meta(change_file)}[o2]",
        f"[nat]crop='min(iw,{NOISE_CROP_W})':'min(ih,{NOISE_CROP_H})',format=yuv420p,"
        f"split=2[n1][n2]",
        "[n2]gblur=sigma=1[nb]",
        f"[n1][nb]blend=all_mode=difference,lutyuv=y='min(val*{NOISE_GAIN},255)',"
        f"scale='trunc(iw/{NOISE_BLOCK})':'trunc(ih/{NOISE_BLOCK})':flags=area,"
        f"signalstats,{meta(noise_file)}[o3]",
        f"[t]scale={GM_WIDTH}:{GM_HEIGHT}:flags=area,format=gray[o4]",
    ])


def _gm_candidates():
    """Index maps for every candidate shift: thumbnail pixel indices of the
    previous frame that line up with the fixed interior of the current one."""
    interior = [y * GM_WIDTH + x for y in range(GM_RANGE, GM_HEIGHT - GM_RANGE)
                for x in range(GM_RANGE, GM_WIDTH - GM_RANGE)]
    shifts = [(dx, dy) for dy in range(-GM_RANGE, GM_RANGE + 1)
              for dx in range(-GM_RANGE, GM_RANGE + 1)]
    return interior, [(s, [i + s[1] * GM_WIDTH + s[0] for i in interior]) for s in shifts]


GM_INTERIOR, GM_CANDIDATES = _gm_candidates()


def global_motion(thumbs):
    """Per frame pair, find the translation that best explains the change.
    Returns (share of pairs where the camera visibly moved, median shift in
    320-px pixels over those pairs, direction consistency 0..1)."""
    n = len(GM_INTERIOR)
    moves = []
    for prev, cur in zip(thumbs, thumbs[1:]):
        target = [cur[i] for i in GM_INTERIOR]
        costs = {s: sum(map(abs, map(operator.sub, target, map(prev.__getitem__, idx))))
                 for s, idx in GM_CANDIDATES}
        best = min(costs, key=costs.get)
        zero = costs[(0, 0)] / n
        if best != (0, 0) and zero >= GM_MIN_RESIDUAL and costs[best] / n <= GM_MIN_GAIN * zero:
            moves.append(best)
    pairs = max(1, len(thumbs) - 1)
    if not moves:
        return 0.0, 0.0, 0.0
    px_per_thumb = STATS_WIDTH / GM_WIDTH
    magnitudes = [math.hypot(dx, dy) * px_per_thumb for dx, dy in moves]
    unit = [(dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)) for dx, dy in moves]
    resultant = math.hypot(sum(u for u, _ in unit), sum(v for _, v in unit)) / len(unit)
    return len(moves) / pairs, statistics.median(magnitudes), resultant


def read_metadata(file):
    """Parse a metadata=print file into a list of per-frame {key: value}."""
    frames = []
    with open(file) as f:
        for line in f:
            if line.startswith("frame:"):
                frames.append({})
            elif "=" in line and frames:
                key, _, value = line.strip().partition("=")
                frames[-1][key.rsplit(".", 1)[-1]] = value
    return frames


def series(frames, key):
    return [float(fr[key]) for fr in frames if key in fr]


def measure_content(path, duration, is_elementary, work_dir):
    start, length = sample_window(duration, is_elementary)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        files = [os.path.join(tmp, n) for n in ("motion.txt", "change.txt", "noise.txt")]
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-threads", str(FFMPEG_DECODE_THREADS)]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-t", f"{length:.3f}", "-i", path, "-an", "-sn", "-dn",
                "-filter_complex_threads", "1",
                "-filter_complex", build_filtergraph(*files)]
        for out in ("[o1]", "[o2]", "[o3]"):
            cmd += ["-map", out, "-f", "null", "-"]
        cmd += ["-map", "[o4]", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
        code, thumb_bytes, err = run(cmd, MEASURE_TIMEOUT_S)
        if code != 0 or not os.path.exists(files[0]):
            raise RuntimeError("ffmpeg stats failed: " + err.strip()[-300:])
        motion, change, noise = (read_metadata(f) for f in files)

    luma = series(motion, "YAVG")
    if len(luma) < 2:
        raise RuntimeError("too few decodable frames in sample")
    ydif = series(motion, "YDIF")[1:]          # first frame has no predecessor
    changed = [v / 255 for v in series(change, "YAVG")]
    cuts = sum(1 for fr in motion if "time" in fr)
    frame_bytes = GM_WIDTH * GM_HEIGHT
    thumbs = [thumb_bytes[i:i + frame_bytes]
              for i in range(0, len(thumb_bytes) - frame_bytes + 1, frame_bytes)]
    motion_frac, motion_px, motion_consistency = global_motion(thumbs)
    ydif_mean = statistics.fmean(ydif) if ydif else 0.0

    # Per-second luma signature, used for near-duplicate detection.
    per_sec = max(1, int(round(SAMPLE_FPS)))
    signature = [round(statistics.fmean(luma[i:i + per_sec]), 1)
                 for i in range(0, len(luma) - per_sec + 1, per_sec)]

    return {
        "sample_start_s": round(start, 2),
        "sample_frames": len(luma),
        "luma_mean": round(statistics.fmean(luma), 2),
        "luma_p10": round(statistics.fmean(series(motion, "YLOW")), 2),
        "sat_mean": round(statistics.fmean(series(motion, "SATAVG")), 2),
        "ydif_median": round(statistics.median(ydif), 3) if ydif else 0.0,
        "ydif_p90": round(percentile(ydif, 0.9), 3) if ydif else 0.0,
        "ydif_cv": round(statistics.pstdev(ydif) / ydif_mean, 3) if ydif_mean > 0 else 0.0,
        "changed_frac_median": round(statistics.median(changed), 4) if changed else 0.0,
        "changed_frac_p90": round(percentile(changed, 0.9), 4) if changed else 0.0,
        "camera_motion_frac": round(motion_frac, 3),
        "camera_motion_px": round(motion_px, 1),
        "camera_motion_consistency": round(motion_consistency, 3),
        "detail": round(statistics.fmean(series(noise, "YAVG")) / NOISE_GAIN, 3),
        "noise_est": round(statistics.median(series(noise, "YLOW")) / NOISE_GAIN, 3),
        "scene_cuts": cuts,
        "cuts_per_min": round(cuts * 60 / max(length, 1e-3), 2),
        "luma_signature": signature,
    }


# --------------------------------------------------------------------------
# Classification and derived fields
# --------------------------------------------------------------------------

def path_hints(path):
    """Return ([(class, reason)], [(tag, reason)]) from path/filename hints."""
    lowered = path.lower()
    classes = [(c, r) for pat, c, r in CLASS_HINTS if re.search(pat, lowered)]
    tags = [(t, r) for pat, t, r in TAG_HINTS if re.search(pat, lowered)]
    return classes, tags


def classify_from_stats(stats, check_light=True):
    """(class, reason) from content statistics alone.

    Low light dominates encoder behaviour, so it wins when the luma says so.
    Otherwise a camera that visibly translates in enough frame pairs is a
    moving camera: a steady direction reads as a pan, an erratic one as
    hand-held. A fixed camera is busy or static by how much of the frame
    changes. Forward motion (dashcam) has no single translation and reads as
    busy; drone footage is only recognisable from path hints.
    """
    luma, saturation = stats["luma_mean"], stats["sat_mean"]
    changed = stats["changed_frac_median"]
    moving, consistency = stats["camera_motion_frac"], stats["camera_motion_consistency"]
    if check_light and luma < LOW_LIGHT_LUMA:
        return "low-light", f"mean luma {luma:.0f} < {LOW_LIGHT_LUMA:.0f}"
    if check_light and saturation < IR_MONO_SATURATION:
        return "low-light", "monochrome picture: camera in IR night mode"
    if moving >= CAMERA_MOTION_FRAC:
        detail = f"camera translates in {moving:.0%} of frame pairs"
        if consistency >= PAN_CONSISTENCY:
            return "pan", f"{detail}, steady direction ({consistency:.2f})"
        return "handheld-bodycam", f"{detail}, erratic direction ({consistency:.2f})"
    if changed >= BUSY_CHANGED_FRAC:
        return "crowd-busy", f"fixed camera, {changed:.0%} of pixels change per frame"
    return "static", f"fixed camera, only {changed:.1%} of pixels change per frame"


def classify(rec):
    """Propose one content class with a confidence, the reasons and tags.

    Stats and path hints that agree give high confidence. Where they
    disagree, a hint wins only if the stats are compatible with it (e.g. a
    'bodycam' path with a moving camera); otherwise the stats win at low
    confidence so the clip stands out for review.
    """
    hints, tags = path_hints(rec["path"])
    hint_classes = [c for c, _ in hints]
    reasons = [r for _, r in hints]
    tag_names = sorted({t for t, _ in tags})
    reasons += [r for _, r in tags]

    stats = rec.get("stats")
    if not stats:
        if hints:
            return hint_classes[0], 0.4, reasons + ["no content stats; hint only"], tag_names
        return None, 0.0, ["no content stats"], tag_names

    if stats["cuts_per_min"] >= EDITED_CUTS_PER_MIN and "edited" not in tag_names:
        tag_names.append("edited")
        reasons.append(f"{stats['scene_cuts']} hard cuts in sample: edited content")

    stat_class, stat_reason = classify_from_stats(stats)
    reasons.append(stat_reason)
    if stat_class == "low-light":
        motion_class, motion_reason = classify_from_stats(stats, check_light=False)
        reasons.append(f"motion: {motion_class} ({motion_reason})")
    luma, changed = stats["luma_mean"], stats["changed_frac_median"]
    camera_moves = stats["camera_motion_frac"] >= CAMERA_MOTION_FRAC
    active = camera_moves or changed >= BUSY_CHANGED_FRAC

    if stat_class in hint_classes:
        return stat_class, 0.9, reasons, tag_names
    if stat_class == "low-light":
        return "low-light", 0.75, reasons, tag_names
    if "low-light" in hint_classes and luma < NIGHT_HINT_LUMA:
        return "low-light", 0.7, reasons + ["dim and hinted as night"], tag_names
    if "drone-aerial" in hint_classes:
        return "drone-aerial", 0.7, reasons, tag_names
    if "handheld-bodycam" in hint_classes and active:
        return "handheld-bodycam", 0.75, reasons, tag_names
    if "pan" in hint_classes and active:
        return "pan", 0.75, reasons + ["hinted moving vehicle"], tag_names
    if "crowd-busy" in hint_classes and not camera_moves and changed >= STATIC_CHANGED_FRAC:
        return "crowd-busy", 0.6, reasons + ["hinted crowd, moderate activity"], tag_names
    if hints:
        reasons.append(f"stats disagree with path hint ({hint_classes[0]})")
        return stat_class, 0.45, reasons, tag_names
    borderline = (abs(changed - BUSY_CHANGED_FRAC) < 0.3 * BUSY_CHANGED_FRAC
                  or abs(stats["camera_motion_frac"] - CAMERA_MOTION_FRAC) < 0.1
                  or abs(luma - LOW_LIGHT_LUMA) < 8)
    return stat_class, (0.45 if borderline else 0.6), reasons, tag_names


def rd_fields(rec):
    """Derived bitrate-rule fields: the 50% ceiling and, per operating point,
    the pixel-scaled RD range and how much of it the clip may serve."""
    kbps = rec.get("source_kbps")
    width, height = rec.get("width") or 0, rec.get("height") or 0
    short, long_ = min(width, height), max(width, height)
    max_test = round(TEST_BITRATE_FRACTION * kbps, 1) if kbps else None
    ops = {}
    for name, (op_w, op_h) in OPERATING_POINTS.items():
        scale = op_w * op_h / REF_PIXELS
        lo, hi = REF_RD_KBPS[0] * scale, REF_RD_KBPS[1] * scale
        res_ok = short >= RES_TOLERANCE * op_h and long_ >= RES_TOLERANCE * op_w
        ceiling = min(hi, max_test) if max_test else None
        ops[name] = {
            "rd_kbps": [round(lo, 1), round(hi, 1)],
            "resolution_ok": res_ok,
            "ceiling_kbps": round(ceiling, 1) if ceiling else None,
            "full_range_ok": bool(res_ok and max_test and max_test >= hi),
            "usable": bool(res_ok and max_test and max_test >= lo),
        }
    fps = rec.get("fps") or 0
    primary = ops[PRIMARY_OP]
    return {
        "max_test_kbps": max_test,
        "operating_points": ops,
        "ok_720p30_800": bool(primary["resolution_ok"] and fps >= PRIMARY_MIN_FPS
                              and max_test and max_test >= REF_RD_KBPS[1]),
        "ok_720p30_500": bool(primary["resolution_ok"] and fps >= PRIMARY_MIN_FPS
                              and max_test and max_test >= PRIMARY_MID_KBPS),
    }


# --------------------------------------------------------------------------
# Per-file processing (cached)
# --------------------------------------------------------------------------

def process_file(path, cache_dir, work_dir, force):
    st = os.stat(path)
    cache_file = cache_path(cache_dir, path)
    if not force and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if (cached.get("cache_version") == CACHE_VERSION
                    and cached["size_bytes"] == st.st_size
                    and cached["mtime"] == st.st_mtime):
                return cached, True
        except (ValueError, KeyError):
            pass

    rec = {"cache_version": CACHE_VERSION, "path": path, "size_bytes": st.st_size,
           "mtime": st.st_mtime, "error": None}
    try:
        rec["content_hash"] = content_hash(path, st.st_size)
        rec.update(probe(path))
        rec["stats"] = measure_content(path, rec["duration_s"],
                                       rec["container"] == "elementary", work_dir)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"

    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(cache_file)
    return rec, False


# --------------------------------------------------------------------------
# Corpus-level passes
# --------------------------------------------------------------------------

def find_duplicates(records):
    """Group exact duplicates (same content hash) and near duplicates (same
    duration and aspect, matching luma signature and activity — typically
    the same footage re-encoded at another size or rate). Within each group
    the representative is an original rather than a derived copy, then the
    one with the most pixels per second, then the highest bitrate."""
    parent = {r["path"]: r["path"] for r in records}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    kinds = {}

    def union(a, b, kind):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
        kinds[a] = kinds.get(a) or kind
        kinds[b] = kinds.get(b) or kind

    by_hash = {}
    for r in records:
        if r.get("content_hash"):
            by_hash.setdefault(r["content_hash"], []).append(r["path"])
    for paths in by_hash.values():
        for other in paths[1:]:
            union(paths[0], other, "exact")

    measured = sorted((r for r in records if r.get("stats") and r.get("duration_s")),
                      key=lambda r: r["duration_s"])
    for i, a in enumerate(measured):
        for b in measured[i + 1:]:
            tol = max(DUP_DURATION_TOL_S, DUP_DURATION_TOL_REL * a["duration_s"])
            if b["duration_s"] - a["duration_s"] > tol:
                break
            if find(a["path"]) == find(b["path"]):
                continue
            if is_near_duplicate(a, b):
                union(a["path"], b["path"], "near")

    groups = {}
    for r in records:
        groups.setdefault(find(r["path"]), []).append(r)
    for members in groups.values():
        if len(members) < 2:
            continue
        best = max(members, key=representative_key)
        for r in members:
            if r is not best:
                r["duplicate_of"] = best["path"]
                r["duplicate_kind"] = kinds.get(r["path"], "near")


def is_near_duplicate(a, b):
    sa, sb = a["stats"], b["stats"]
    aspect_a = a["width"] / a["height"] if a.get("height") else 0
    aspect_b = b["width"] / b["height"] if b.get("height") else 0
    if abs(aspect_a - aspect_b) > DUP_ASPECT_TOL * max(aspect_a, aspect_b, 1e-9):
        return False
    sig_a, sig_b = sa["luma_signature"], sb["luma_signature"]
    n = min(len(sig_a), len(sig_b))
    if n < 3:
        return False
    mad = sum(abs(x - y) for x, y in zip(sig_a[:n], sig_b[:n])) / n
    return (mad < DUP_SIGNATURE_MAD
            and abs(sa["luma_p10"] - sb["luma_p10"]) < DUP_LUMA_P10_TOL
            and abs(sa["sat_mean"] - sb["sat_mean"]) < DUP_SATURATION_TOL
            and abs(sa["changed_frac_median"] - sb["changed_frac_median"]) < DUP_CHANGED_FRAC_TOL)


def representative_key(rec):
    pixel_rate = (rec.get("width") or 0) * (rec.get("height") or 0) * (rec.get("fps") or 0)
    return ("derived" not in rec.get("tags", []), pixel_rate, rec.get("source_kbps") or 0)


def bits_per_pixel(rec):
    pixels_per_s = (rec.get("width") or 0) * (rec.get("height") or 0) * (rec.get("fps") or 0)
    return (rec.get("source_kbps") or 0) * 1000 / pixels_per_s if pixels_per_s else 0


def mark_earlier_attempt(records, inputs_file):
    """Tag the source clips of the earlier attempt (and exact copies of them)."""
    try:
        entries = json.loads(Path(inputs_file).read_text())
    except (OSError, ValueError):
        return
    by_path = {r["path"]: r for r in records}
    hash_to_names = {}
    for e in entries:
        rec = by_path.get(e.get("source"))
        if rec is None:
            continue
        rec.setdefault("earlier_attempt", []).append(e["name"])
        if rec.get("content_hash"):
            hash_to_names.setdefault(rec["content_hash"], []).append(e["name"])
    for r in records:
        names = hash_to_names.get(r.get("content_hash"))
        if names and "earlier_attempt" not in r:
            r["earlier_attempt"] = [n + " (copy)" for n in names]


def source_group(path, roots):
    """Dataset folder a file belongs to, e.g. 'tracking_original/jaad'."""
    for root in roots:
        rel = os.path.relpath(path, root)
        if not rel.startswith(".."):
            parts = Path(rel).parts
            top = os.path.basename(os.path.normpath(root))
            return top if len(parts) == 1 else f"{top}/{parts[0]}"
    return os.path.dirname(path)


def finalise(rec, roots):
    """Add classification and derived fields to a raw per-file record."""
    rec["source_group"] = source_group(rec["path"], roots)
    rec["auto_class"], rec["class_confidence"], rec["class_reasons"], rec["tags"] = classify(rec)
    rec.update(rd_fields(rec))
    rec.setdefault("duplicate_of", None)
    rec.setdefault("duplicate_kind", None)
    rec.setdefault("earlier_attempt", [])
    rec.pop("mtime", None)
    rec.pop("cache_version", None)
    return rec


def shortlist_key(rec):
    """Rank candidates: primary-operating-point usable first, then frame
    rate, confidence and source quality (bits per pixel)."""
    return (rec["ok_720p30_800"], rec["ok_720p30_500"],
            rec["operating_points"][PRIMARY_OP]["resolution_ok"],
            (rec.get("fps") or 0) >= PRIMARY_MIN_FPS,
            rec["class_confidence"], bits_per_pixel(rec))


def describe(rec):
    stats = rec["stats"]
    return (f"[{rec['source_group']}] {rec['width']}x{rec['height']} "
            f"{rec['fps']:.0f}fps {rec['duration_s']:.0f}s, src {rec['source_kbps']:.0f} kbps; "
            f"{rec['class_reasons'][-1]}; luma {stats['luma_mean']:.0f}")


def build_shortlist(records):
    """Up to SHORTLIST_PER_CLASS candidates per class (preferring at most
    SHORTLIST_PER_GROUP from any one source folder), plus the earlier
    attempt's sources."""
    eligible = [r for r in records
                if not r.get("error") and r.get("stats") and not r["duplicate_of"]
                and r["auto_class"] and r["class_confidence"] >= SHORTLIST_MIN_CONFIDENCE
                and (r.get("duration_s") or 0) >= SHORTLIST_MIN_DURATION_S
                and r.get("max_test_kbps")
                and not SHORTLIST_EXCLUDED_TAGS & set(r["tags"])]
    chosen = []
    for cls in CLASSES:
        ranked = sorted((r for r in eligible if r["auto_class"] == cls),
                        key=shortlist_key, reverse=True)
        picked, per_group = [], {}
        for rec in ranked:
            if len(picked) >= SHORTLIST_PER_CLASS:
                break
            if per_group.get(rec["source_group"], 0) < SHORTLIST_PER_GROUP:
                per_group[rec["source_group"]] = per_group.get(rec["source_group"], 0) + 1
                picked.append(rec)
        # Too few source folders for this class: fill up without the cap.
        picked += [r for r in ranked if r not in picked][:SHORTLIST_PER_CLASS - len(picked)]
        chosen += picked
    chosen += [r for r in records if r["earlier_attempt"] and r not in chosen
               and not r.get("error") and r.get("stats")]
    return chosen


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def csv_row(rec):
    stats = rec.get("stats") or {}
    row = {k: rec.get(k, stats.get(k)) for k in CSV_FIELDS}
    ops = rec.get("operating_points", {})
    row["ok_1080p_full"] = ops.get("1080p", {}).get("full_range_ok")
    row["ok_360p_full"] = ops.get("360p", {}).get("full_range_ok")
    row["class_reasons"] = "; ".join(rec.get("class_reasons") or [])
    row["tags"] = ";".join(rec.get("tags") or [])
    row["earlier_attempt"] = ";".join(rec.get("earlier_attempt") or [])
    return row


def write_outputs(out_dir, records, skipped):
    summary = {
        "video_files": len(records),
        "errors": sum(1 for r in records if r["error"]),
        "skipped_non_video": dict(sorted(skipped.items(), key=lambda kv: -kv[1])),
        "rules": {
            "test_bitrate_fraction": TEST_BITRATE_FRACTION,
            "reference_rd_kbps_at_720p": list(REF_RD_KBPS),
            "operating_points": {k: list(v) for k, v in OPERATING_POINTS.items()},
            "primary_min_fps": PRIMARY_MIN_FPS,
        },
    }
    payload = {"summary": summary, "clips": records}
    tmp = out_dir / "inventory.json.tmp"
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out_dir / "inventory.json")
    with open(out_dir / "inventory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(csv_row(rec))
    with open(out_dir / "shortlist.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHORTLIST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in build_shortlist(records):
            writer.writerow({**rec, "description": describe(rec),
                             "earlier_attempt": ";".join(rec["earlier_attempt"])})


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS))
    parser.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="concurrent ffmpeg/ffprobe processes (max 16)")
    parser.add_argument("--limit", type=int, help="process only the first N files (testing)")
    parser.add_argument("--force", action="store_true", help="ignore the per-file cache")
    args = parser.parse_args()
    workers = max(1, min(args.workers, 16))

    out_dir = Path(args.out)
    cache_dir, work_dir = out_dir / "cache", out_dir / "work"
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    paths, skipped = discover(args.roots)
    if args.limit:
        paths = paths[:args.limit]
    print(f"{len(paths)} video files, {sum(skipped.values())} non-video skipped",
          file=sys.stderr)

    records, reused = [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_file, p, cache_dir, work_dir, args.force): p
                   for p in paths}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                rec, from_cache = future.result()
            except Exception as exc:  # keep going: one bad file must not stop the run
                rec = {"path": futures[future], "error": f"{type(exc).__name__}: {exc}"}
                from_cache = False
            records.append(rec)
            reused += from_cache
            if done % 50 == 0 or done == len(paths):
                print(f"  {done}/{len(paths)} ({reused} cached)", file=sys.stderr)

    records.sort(key=lambda r: r["path"])
    for rec in records:
        finalise(rec, args.roots)
    find_duplicates(records)
    mark_earlier_attempt(records, EARLIER_ATTEMPT_INPUTS)
    write_outputs(out_dir, records, skipped)
    print(f"wrote {out_dir / 'inventory.json'} and inventory.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
