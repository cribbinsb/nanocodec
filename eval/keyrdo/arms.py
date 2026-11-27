"""Per-arm summary of `nceval.rc_oracle --base-arm` results (the key-frame RDO study).

    python3 eval/keyrdo/arms.py RESULT.json [RESULT.json ...]

Per arm against the base arm: BD-J, the key's share of the stream's bits, the largest
buffer fullness, buffer overflows, key frame time and whole-encode time ratio (wall
times from the rate logs of a shared box: ratios within one run, not timings).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def mean(stats: dict, field: str) -> float:
    """Mean over content classes of one log statistic."""
    values = [cls[field] for cls in stats.values() if cls.get(field) is not None]
    return sum(values) / len(values) if values else float("nan")


def rows(result: dict) -> list[str]:
    stats = result["arm_log_stats"]
    base = "arm-" + result["base_arm"].removeprefix("arm-")
    out = [f"{result['codec']} {result['frames']} f, base {result['base_arm']}: key share "
           f"{mean(stats[base], 'key_share'):.3f}, key {mean(stats[base], 'key_encode_ms'):.0f} ms"]
    for arm, versus in result["arms_vs_base"].items():
        s = stats[arm]
        out.append(
            f"  {arm.removeprefix('arm-'):14s} J {versus['bd_j']:+6.2f}  key share "
            f"{mean(s, 'key_share'):.3f}  fullness max "
            f"{max(cls['fullness_max'] for cls in s.values()):.2f}  overflows "
            f"{sum(cls['buffer_overflows'] for cls in s.values()):.0f}  key "
            f"{mean(s, 'key_encode_ms'):.0f} ms  encode x"
            f"{mean(s, 'encode_ms_total') / mean(stats[base], 'encode_ms_total'):.2f}")
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print("\n".join(rows(json.loads(Path(path).read_text()))))
