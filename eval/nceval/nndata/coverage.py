"""Coverage report and dataset card of a set of shards (a run or a released version).

The report counts superblocks (the unit a policy decides) and nodes along every axis a
model's behaviour may depend on: content class, dataset, resolution, codec, preset, rate
mode and quantizer, frame role, motion, noise, and the easy share (nceval.nndata.easy_rows);
plus the train / held-out split by whole sources. It then names the gaps against the
product's content (PRODUCT_CONTENT: live camera to browser) and proposes pool clips that
would fill them.

`card` renders the report as the version's dataset card (markdown), with the validator's
summary of the run (kept and quarantined shards by reason) and the known limitations.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .balance import easy_rows
from .shard import load as load_shard
from .splits import split
from .schema import SLOTS

#: Motion bins: mean 8x8 vector length, luma samples per frame.
MOTION_BINS = (0.0, 0.5, 2.0, 8.0, np.inf)
#: Noise bins: pre-analysis σ, luma levels.
NOISE_BINS = (0.0, 1.0, 2.0, 4.0, np.inf)
#: A content area counts as covered when it holds at least this share of superblocks.
MIN_SHARE = 0.05
#: The product's content (live camera to cloud and browser), each with the pool's
#: content classes or the measured condition that stands for it.
PRODUCT_CONTENT = {
    "live moving camera": {"classes": ("pan", "bodycam", "drone")},
    "still cameras": {"classes": ("static",)},
    "low light": {"classes": ("lowlight",)},
    "sensor noise (sigma >= 2)": {"noise_at_least": 2.0},
    "crowds": {"classes": ("crowd",)},
    "talking heads": {"classes": ("videocall",)},
    "screen content": {"classes": ("screen",)},
}


def _bin(values: np.ndarray, edges: tuple[float, ...]) -> Counter:
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    index = np.clip(np.digitize(values, edges[1:-1]), 0, len(labels) - 1)
    return Counter({labels[i]: int(n) for i, n in zip(*np.unique(index, return_counts=True))})


def report(paths: list[Path], seed: str = "coverage") -> dict:
    """The report over the shards at `paths`, read one at a time (a version holds
    ~10^8 node rows: only one shard's columns are in memory at once)."""
    axes: dict[str, Counter] = defaultdict(Counter)
    nodes: Counter = Counter()
    noise_sb, easy_total, sb_total = 0, 0, 0
    metas = [json.loads((p / "meta.json").read_text()) for p in paths]
    sides = split([m["job"]["clip_meta"] for m in metas], seed) if metas else {}
    for path in paths:
        shard = load_shard(path, REPORT_COLUMNS)
        job = shard.meta["job"]
        clip = job["clip_meta"]
        n_sb = shard.rows("sb")
        sb_total += n_sb
        frames = shard["frame"]
        key = dict(zip(frames["index"].tolist(), frames["key"].tolist()))
        noise = dict(zip(frames["index"].tolist(), frames["noise_sigma"].tolist()))
        sb_frames = shard["sb"]["frame"].tolist()
        role = Counter("key" if key[f] else "P" for f in sb_frames)
        for name, value in (("content_class", clip["content_class"]), ("dataset", clip["dataset"]),
                            ("resolution", f"{job['width']}x{job['height']}"),
                            ("codec", job["codec"]), ("preset", f"{job['codec']} {job['preset']}"),
                            ("rate", job["rate"]), ("split", sides.get(clip["name"], "?"))):
            axes[name][value] += n_sb
        axes["frame_role"].update(role)
        motion = np.hypot(*shard["sb"]["motion"][:, :2].T)
        axes["motion"].update(_bin(motion, MOTION_BINS))
        sigma = np.array([noise[f] for f in sb_frames])
        axes["noise_sigma"].update(_bin(sigma, NOISE_BINS))
        noise_sb += int((sigma >= PRODUCT_CONTENT["sensor noise (sigma >= 2)"]["noise_at_least"]).sum())
        easy_total += int(easy_rows(shard).sum())
        node = shard["node"]
        nodes["rows"] += shard.rows("node")
        for slot, name in enumerate(SLOTS):
            nodes[f"coded {name}"] += int(np.isfinite(node["cost"][:, slot]).sum())
        nodes["extra candidates"] += int((np.isfinite(node["cost"]) &
                                          ~((node["eligible"][:, None] >> np.arange(len(SLOTS))) & 1 == 1)).sum())
    gaps = []
    for area, rule in PRODUCT_CONTENT.items():
        count = noise_sb if "noise_at_least" in rule else \
            sum(axes["content_class"][c] for c in rule["classes"])
        share = count / sb_total if sb_total else 0.0
        if share < MIN_SHARE:
            gaps.append({"area": area, "share": share})
    return {"shards": len(paths), "superblocks": sb_total, "easy_share": easy_total / max(sb_total, 1),
            "axes": {k: dict(v.most_common()) for k, v in axes.items()}, "nodes": dict(nodes),
            "gaps": gaps}


def _table(title: str, counts: dict, total: int) -> list[str]:
    lines = [f"**{title}**", "", "| value | superblocks | share |", "|---|---|---|"]
    lines += [f"| {k} | {v} | {100 * v / max(total, 1):.1f} % |" for k, v in counts.items()]
    return lines + [""]


def card(version: str, rep: dict, summary: dict | None, manifest_sha: str | None,
         limitations: list[str], proposals: dict[str, list[str]]) -> str:
    total = rep["superblocks"]
    out = [f"# nn dataset {version}: card", "",
           f"Manifest sha256: `{manifest_sha or 'not released'}`. Shards: {rep['shards']}; "
           f"superblocks: {total}; nodes: {rep['nodes'].get('rows', 0)}.",
           f"Easy (near-static inter) superblocks: {100 * rep['easy_share']:.1f} % "
           "(nceval.nndata.easy_rows; `balance_weights` evens them out).", ""]
    if summary:
        out += ["## Validator", "", f"Kept {summary['kept']}, quarantined {summary['quarantine']}.", ""]
        reasons = Counter(p.split(":")[0] for p in summary.get("problems", {}).values())
        out += [f"- {reason}: {n}" for reason, n in reasons.most_common()] + [""]
    out += ["## Coverage", ""]
    for axis, counts in rep["axes"].items():
        out += _table(axis, counts, total)
    out += ["**nodes**", ""] + [f"- {k}: {v}" for k, v in rep["nodes"].items()] + [""]
    out += ["## Gaps against product content", ""]
    for gap in rep["gaps"]:
        fill = proposals.get(gap["area"], [])
        out.append(f"- **{gap['area']}**: {100 * gap['share']:.1f} % of superblocks. "
                   + (f"Fill: {', '.join(fill)}." if fill else "No pool clip fills it."))
    if not rep["gaps"]:
        out.append("- none under the threshold")
    out += ["", "## Known limitations", ""] + [f"- {item}" for item in limitations]
    return "\n".join(out) + "\n"


#: The columns the report reads (shards loaded with only these: a version has thousands).
REPORT_COLUMNS = {"frame.index", "frame.key", "frame.noise_sigma", "sb.frame", "sb.motion",
                  "sb.static_share", "sb.zero_sad", "node.cost", "node.eligible"}


def write(directory: Path, version: str, shards: list[Path], summary: dict | None,
          manifest_sha: str | None, limitations: list[str],
          proposals: dict[str, list[str]]) -> dict:
    rep = report(shards)
    (directory / "coverage.json").write_text(json.dumps(rep, indent=1))
    (directory / "CARD.md").write_text(card(version, rep, summary, manifest_sha, limitations,
                                            proposals))
    return rep
