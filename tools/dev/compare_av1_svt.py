"""Matched-bitrate stream comparison: nanocodec-av1 vs svtav1-m12.

For each clip, encode SVT M12 rtc at a given QP, then find the nanocodec
qindex whose stream has about the same size (bisection, within 3%), and run
the standard stats map (nceval.streamstats) on both. Per-frame stats are
written as JSON next to a summary, for docs/analysis/av1-vs-svt.md.

    python3 -I tools/dev/compare_av1_svt.py NANOCODEC_BINARY OUT_DIR \
        clip=svt_qp [clip=svt_qp ...]

The clips come from the quick set; set NC_COMPARE_CLIPSET=main for the main
(TRAIN/VAL) set. The same clip may be listed at several QPs: outputs are named
<clip>-q<qp>.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval.clips import load_clipset  # noqa: E402
from nceval.config import find_tool, load_paths  # noqa: E402
from nceval.streamstats import analyse, summarise_clip  # noqa: E402

INTRA_PERIOD_SECONDS = 10
SIZE_TOLERANCE = 0.03
QINDEX_RANGE = (0, 255)


def encode_svt(clip, qp: int, out: Path) -> None:
    fps = round(clip.fps * 1000)
    subprocess.run([str(find_tool("SvtAv1EncApp")), "--rtc", "1", "--preset", "12",
                    "--pred-struct", "1", "--hierarchical-levels", "0", "--rc", "0",
                    "--aq-mode", "0", "--qp", str(qp), "--scd", "0", "--lp", "1",
                    "--keyint", str(round(clip.fps * INTRA_PERIOD_SECONDS)),
                    "--fps-num", str(fps), "--fps-denom", "1000",
                    "-w", str(clip.width), "-h", str(clip.height), "-n", str(clip.frames),
                    "-i", str(clip.yuv), "-b", str(out)],
                   check=True, capture_output=True, stdin=subprocess.DEVNULL)


def encode_ours(binary: str, clip, qindex: int, out: Path) -> None:
    subprocess.run([binary, "encode", "--codec", "av1", "--input", str(clip.yuv),
                    "--width", str(clip.width), "--height", str(clip.height),
                    "--fps", f"{round(clip.fps * 1000)}/1000", "--qp", str(qindex),
                    "--frames", str(clip.frames),
                    "--intra-period", str(round(clip.fps * INTRA_PERIOD_SECONDS)),
                    "--output", str(out)], check=True, capture_output=True,
                   stdin=subprocess.DEVNULL)


def match_size(binary: str, clip, target_bytes: int, out: Path) -> int:
    """Bisect our qindex until the stream is within SIZE_TOLERANCE of target."""
    lo, hi = QINDEX_RANGE
    best = None
    while lo <= hi:
        q = (lo + hi) // 2
        encode_ours(binary, clip, q, out)
        size = out.stat().st_size
        if best is None or abs(size - target_bytes) < abs(best[1] - target_bytes):
            best = (q, size)
        if abs(size - target_bytes) <= SIZE_TOLERANCE * target_bytes:
            return q
        if size > target_bytes:
            lo = q + 1
        else:
            hi = q - 1
    encode_ours(binary, clip, best[0], out)
    return best[0]


def main() -> int:
    binary, out_dir, *specs = sys.argv[1:]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = load_paths()
    clipset = os.environ.get("NC_COMPARE_CLIPSET", "quick")
    clips = {c.name: c for c in load_clipset(paths.clips / f"{clipset}.json", paths.data_root).clips}
    result = {}
    for spec in specs:
        name, qp = spec.split("=")
        clip = clips[name]
        tag = f"{name}-q{qp}"
        svt, ours = out_dir / f"{tag}.svt.ivf", out_dir / f"{tag}.ours.ivf"
        encode_svt(clip, int(qp), svt)
        qindex = match_size(binary, clip, svt.stat().st_size, ours)
        entry = {"svt_qp": int(qp), "ours_qindex": qindex}
        for label, stream in (("svt", svt), ("ours", ours)):
            frames = analyse(stream, clip.yuv, clip.width, clip.height)
            (out_dir / f"{tag}.{label}.frames.json").write_text(json.dumps(frames))
            entry[label] = {"bytes": stream.stat().st_size,
                            "I": summarise_clip(frames, "I"), "P": summarise_clip(frames, "P")}
        result[tag] = entry
        print(f"{tag}: svt qp {qp} {entry['svt']['bytes']} B, ours qindex {qindex} "
              f"{entry['ours']['bytes']} B", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
