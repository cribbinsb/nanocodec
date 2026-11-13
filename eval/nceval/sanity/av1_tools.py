"""AV1 coding-tool activity of a stream, decoder-side (ours and SVT's alike).

Block-level data comes from libaom's inspection decoder (`aom-inspect`, via
nceval.streamstats); frame-level loop filter and CDEF parameters from the
frame headers (headers.py). The result uses the codec-neutral activity
schema of activity.py, averaged per frame type.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Paths
from ..streamstats import inspect_frames
from .activity import Activity, mean_activity, subpel_phase_shares
from .headers import trace_headers

# Block size, tx size, mode, skip, motion vectors, CDEF.
AV1_INSPECT_FLAGS = ("-bs", "-ts", "-m", "-s", "-mv", "-c")
EIGHTH_PEL = 8
AV1_KEY_FRAME = 0


def av1_activity(stream: Path, paths: Paths) -> dict[str, Activity]:
    """Activity per frame type ("I", "P") of one AV1 stream."""
    headers = trace_headers(stream, paths).frames
    frames = []
    for index, frame in enumerate(inspect_frames(stream, AV1_INSPECT_FLAGS)):
        header = headers[index] if index < len(headers) else {}
        frames.append(frame_activity(frame, header))
    return mean_activity(frames)


def frame_activity(frame: dict, header: dict[str, int]) -> Activity:
    # Inter modes are the *MV names (NEARESTMV, NEWMV, GLOBALMV, ...).
    inter = np.isin(np.array(frame["mode"]),
                    [code for name, code in frame["modeMap"].items() if name.endswith("MV")])
    skipped = np.array(frame["skip"]) == frame["skipMap"]["SKIP"]
    coded = ~skipped
    size_names = {v: k.removeprefix("BLOCK_").lower() for k, v in frame["blockSizeMap"].items()}
    tx_names = {v: k.removeprefix("TX_").lower() for k, v in frame["transformSizeMap"].items()}
    mvs = np.array(frame["motionVectors"])[..., :2]  # (row, col) of reference 0, 1/8 pel
    selected = mvs[inter].reshape(-1, 2)
    phases = subpel_phase_shares(selected[:, 0], selected[:, 1], EIGHTH_PEL)
    cdef_primary = np.array(frame["cdef_level"])
    cdef_secondary = np.array(frame["cdef_strength"])
    cdef_on = (cdef_primary > 0) | (cdef_secondary > 0)
    return Activity(
        frames=1,
        frame_type="I" if frame["frameType"] == AV1_KEY_FRAME else "P",
        skip_share=float(skipped.mean()),
        intra_share=float((~inter).mean()),
        histograms={
            "block_sizes": _shares(np.array(frame["blockSize"]), size_names),
            "tx_sizes": _shares(np.array(frame["transformSize"])[coded], tx_names),
            "subpel_phases": phases,
            "cdef_y_primary": _shares(cdef_primary[coded], {}),
        },
        filters={
            "loop_filter_level_y": float(np.mean([header.get("loop_filter_level[0]", 0),
                                                  header.get("loop_filter_level[1]", 0)])),
            "cdef_active_share": float(cdef_on[coded].mean()) if coded.any() else 0.0,
            "cdef_y_primary_mean": float(cdef_primary[coded].mean()) if coded.any() else 0.0,
            "cdef_header_max_y_primary": float(_header_max(header, "cdef_y_pri_strength")),
            "cdef_header_max_uv_primary": float(_header_max(header, "cdef_uv_pri_strength")),
            "high_precision_mv": float(header.get("allow_high_precision_mv", 0)),
        },
    )


def _shares(values: np.ndarray, names: dict[int, str]) -> dict[str, float]:
    if values.size == 0:
        return {}
    keys, counts = np.unique(values, return_counts=True)
    return {names.get(int(k), str(int(k))): float(c / values.size) for k, c in zip(keys, counts)}


def _header_max(header: dict[str, int], prefix: str) -> int:
    return max((v for k, v in header.items() if k.startswith(prefix + "[")), default=0)
