"""Key BD-J against key CPU for each candidate of one-frame `nceval.fixedqp ab` results.

    python3 eval/intracheap/front.py RESULT.json [...]

Key CPU is the candidate's summed whole-process CPU seconds over every clip and
quantizer against those of the candidate labelled `null` (a setting that changes
nothing, timed among the others; the anchor's own points may be cached from another
hour's load on the box), else against the anchor's. The ratio includes start-up and
file reading. A candidate is on the front (`*`) when no
other candidate of the file is both cheaper in CPU and better in J.
"""
import json
import sys


def rows(result: dict) -> list[tuple]:
    cpu = result["cpu_s"]
    base = {key.split("|", 1)[1]: sum(v) for key, v in cpu.items() if "+" not in key.split("|")[0]}
    out = []
    for index, candidate in enumerate(result["candidates"]):
        seconds = {key.split("|", 1)[1]: sum(v) for key, v in cpu.items()
                   if key.split("|")[0].endswith(f"+{index}")}
        clips = [clip for clip in seconds if clip in base]
        ratio = sum(seconds[c] for c in clips) / sum(base[c] for c in clips) if clips else float("nan")
        s = candidate["summary"]
        out.append((candidate["label"], s["j"], s["lo"], s["hi"], s["psnr_yuv"], s["vmaf_v1"],
                    s["ms_ssim"], ratio))
    return out


def main(paths: list[str]) -> None:
    for path in paths:
        result = json.load(open(path))
        table = rows(result)
        null = next((r[7] for r in table if r[0] == "null"), 1.0)
        table = sorted((r[:7] + (r[7] / null,) for r in table), key=lambda r: r[7])
        base = sum(sum(v) for k, v in result["cpu_s"].items() if "+" not in k.split("|")[0])
        points = sum(len(v) for k, v in result["cpu_s"].items() if "+" not in k.split("|")[0])
        print(f"\n{result['codec']}  (anchor {1000 * base / points:.0f} ms per key process)")
        print(f"  {'candidate':<22} {'J %':>6} {'[lo, hi]':>15} {'PSNR/VMAF/MS-SSIM':>20} {'CPU x':>6}")
        best = 0.0
        for label, j, lo, hi, psnr, vmaf, ms, ratio in table:
            front = "*" if j < best else " "
            best = min(best, j)
            print(f" {front}{label:<22} {j:>6.1f} [{lo:>5.1f}, {hi:>5.1f}] "
                  f"{psnr:>6.1f}/{vmaf:>5.1f}/{ms:>5.1f} {ratio:>6.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
