"""Key-frame study: BD-rate of one rc_oracle arm against another over frame ranges.

    python3 eval/keystudy/ranges.py REPORT.json BASE_ARM TEST_ARM [--ranges 0-1,1-31,31-150,0-150]

The whole stream's rate is the axis; the quality of each range is the mean over
its frames of per-frame PSNR-YUV and VMAF v1 (the points the run kept in the
cloud point cache). MS-SSIM has no per-frame value in a point, so a range's J
here is the mean of the two BD-rates (full overlap), class-balanced.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval.bdrate import bd_rate  # noqa: E402

#: Ladder rungs a clip needs (a clip whose source rate caps the ladder has three).
MIN_POINTS = 3
CACHE = Path.home() / "nanocodec-data/work/cloud-measure/realuse"


def load(report: dict, arm: str) -> dict:
    """{clip: [(kbps, frame_psnr_yuv, frame_vmaf_v1)]} for one arm, from the cache: per
    (clip, target) the longest point no longer than the report's frames (a clip shorter
    than the run is measured whole)."""
    family = f"nanocodec-{report['codec']}~{arm}"
    targets = {(p["clip"], round(p["target_kbps"], 1)) for p in report["points"]
               if p["variant"] == arm}
    best: dict = {}
    for path in CACHE.glob(f"f*-ip*/*/{family}/*.json"):
        d = json.loads(path.read_text())
        key = (d["clip"], round(d.get("target_kbps", -1), 1))
        frames = len(d.get("frame_psnr_yuv") or [])
        if key not in targets or not frames or frames > report["frames"]:
            continue
        if key not in best or frames > len(best[key][1]):
            best[key] = (d["kbps"], d["frame_psnr_yuv"], d["frame_vmaf_v1"])
    out: dict = {}
    for (clip, _), point in best.items():
        out.setdefault(clip, []).append(point)
    return {c: sorted(v) for c, v in out.items()}


def main() -> None:
    report = json.loads(Path(sys.argv[1]).read_text())
    base, test = sys.argv[2], sys.argv[3]
    ranges = [tuple(map(int, r.split("-"))) for r in
              (sys.argv[5] if len(sys.argv) > 5 else "0-1,1-31,31-150,0-150").split(",")]
    classes = {p["clip"]: p["class"] for p in report["points"]}
    a, b = load(report, base), load(report, test)
    for lo, hi in ranges:
        by_class: dict = {}
        for clip in sorted(set(a) & set(b)):
            terms = []
            for m in (1, 2):
                qa = [statistics.fmean(p[m][lo:hi]) for p in a[clip]]
                qb = [statistics.fmean(p[m][lo:hi]) for p in b[clip]]
                if min(len(qa), len(qb)) < MIN_POINTS:
                    continue
                v = bd_rate([p[0] for p in a[clip]], qa, [p[0] for p in b[clip]], qb,
                            min_points=MIN_POINTS).bd_rate_percent
                if v is not None:
                    terms.append(v)
            if len(terms) == 2:
                by_class.setdefault(classes[clip], []).append(statistics.fmean(terms))
        means = {c: statistics.fmean(v) for c, v in by_class.items()}
        n = sum(len(v) for v in by_class.values())
        j = statistics.fmean(means.values()) if means else float("nan")
        print(f"{test} vs {base} frames {lo}-{hi}: J(PSNR-YUV,VMAF) {j:+.2f} % n={n}  "
              + " ".join(f"{c} {v:+.1f}" for c, v in sorted(means.items())))


if __name__ == "__main__":
    main()
