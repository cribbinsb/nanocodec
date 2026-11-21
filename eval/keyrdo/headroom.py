"""Key-frame headroom tables from `nceval.fixedqp gap --frames 1 --key-share` results.

    python3 eval/keyrdo/headroom.py RESULT.json [RESULT.json ...]

Per pair (test against anchor): BD-J with its interval, each metric's BD-rate, and
the test's encoder CPU against the anchor's (the sum of the rate probes' CPU seconds
at the curve quantizers, over the clips both coded). The key-frame RDO study,
docs/analysis/key-frame-rdo-2026-11-21.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def cpu_ratio(result: dict, anchor: str, test: str, clips: list[str]) -> float | None:
    """Test CPU over anchor CPU on the clips of the pair (None: a side untimed)."""
    cpu = result.get("cpu_s", {})
    sums = []
    for family in (test, anchor):
        times = [cpu.get(f"{family}|{clip}") for clip in clips]
        if any(t is None or None in t for t in times):
            return None
        sums.append(sum(sum(t) / len(t) for t in times))
    return sums[0] / sums[1] if sums[1] else None


def rows(result: dict) -> list[str]:
    out = []
    for label, pair in result["pairs"].items():
        test, anchor = label.split(" vs ")
        s = pair["summary"]
        if s["j"] is None:
            out.append(f"{test:52s} vs {anchor:16s} no curve")
            continue
        ratio = cpu_ratio(result, anchor, test, list(pair["per_clip"]))
        cpu = "    n/a" if ratio is None else f"{ratio:7.1f}"
        out.append(f"{test:52s} vs {anchor:16s} J {s['j']:+6.1f} [{s['lo']:+6.1f},{s['hi']:+6.1f}] "
                   f"PSNR {s['psnr_yuv']:+6.1f} VMAF {s['vmaf_v1']:+6.1f} "
                   f"MS-SSIM {s['ms_ssim']:+6.1f} CPU x{cpu} n={s['n']}")
    return out


def main(paths: list[str]) -> None:
    for path in paths:
        print(Path(path).name)
        print("\n".join(rows(json.loads(Path(path).read_text()))))


if __name__ == "__main__":
    main(sys.argv[1:])
