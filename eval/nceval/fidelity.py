"""Fidelity study: how cheaply can a tuning trial be measured and still rank right?

A Pareto search (nceval.tune) spends nearly all its time measuring trials.
Each measurement has a *fidelity*: which clips, how many frames, how many
quantizers, at what resolution, how many frames under callgrind. This module
measures a sample of configurations once at full fidelity, keeping every
per-frame number, and then derives every cheaper fidelity from that one
measurement, so their rank agreement with full fidelity costs nothing extra:

  - frame prefixes: our encoder is causal at fixed QP (no look-ahead; the
    first N frames of a 150-frame encode are byte-identical to an N-frame
    encode and reconstruct identically; checked on AV1 and HEVC,
    2026-11-08), so a 30- or 60-frame curve is the prefix of the 150-frame
    point's per-frame bytes and per-frame PSNR/VMAF;
  - clip subsets and quantizer subsets: subsets of the same points;
  - VMAF subsampling: every k-th per-frame VMAF score;
  - CPU: one callgrind run per cost clip dumps the instructions of every
    frame (`--dump-after` on `Encoder::encode_frame`), so the key frame and
    every P-frame prefix (6, 12, 48 frames) come from one run.

Only resolution (360p downscaled sources) needs its own encodes.

    python3 -m nceval.fidelity measure --codec av1 --binary BIN --configs CONFIGS.json \\
        [--clips train|a,b,...] [--height 360] [--part 1/3] [--workers 8]
    python3 -m nceval.fidelity cost --codec av1 --binary BIN --configs CONFIGS.json
    python3 -m nceval.fidelity analyse --codec av1 --binary BIN --configs CONFIGS.json [--json OUT]

CONFIGS.json is a list of {"name": str, "overrides": {param: value}}; the
first entry must be the default configuration (empty overrides): every BD
is against it, as in a tune study. Points are cached under
`$NC_DATA_ROOT/work/search-infra/fidelity/<codec>/` by binary, codec, clip,
resolution, quantizer and overrides; reruns only measure what is missing.
Like every evaluation, points are not decoded (nceval.gatepolicy).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from . import tune
from .bdrate import bd_rate
from .budget import check as check_budget, point_cpu_seconds
from .cachefile import entry_lock, identity_cached, publish, sibling_temp
from .codecs import Codec
from .config import find_tool, load_paths
from .fingerprint import clip_fingerprint, measurement_fingerprint
from .framestats import probe_frames
from .gatepolicy import file_sha256
from .metrics import weighted_yuv
from .slots import cpu_slots, low_priority
from .stages import PROXY_4, PROXY_6

FULL_FRAMES = 150
FRAME_PREFIXES = (30, 60, 150)
#: Callgrind frames per cost clip: the key frame and 48 P frames, dumped per frame.
COST_FRAMES = 49
COST_P_PREFIXES = (6, 12, 48)
VMAF_SUBSAMPLES = (1, 2, 4)
LOW_HEIGHT = 360
#: The symbol callgrind dumps after: one dump per coded frame.
FRAME_SYMBOL = "*Encoder*encode_frame"
METRICS = ("psnr", "vmaf_v1")
MAX_WORKERS = 8
#: Rank agreement a cheap fidelity must keep to stand in for full fidelity.
MIN_RANK_CORRELATION = 0.9


def study_root(codec: Codec) -> Path:
    return load_paths().data_root / "work" / "search-infra" / "fidelity" / codec.value


@dataclass(frozen=True)
class Config:
    name: str
    overrides: dict


def load_configs(path: Path) -> list[Config]:
    raw = json.loads(path.read_text())
    configs = [Config(c["name"], dict(c["overrides"])) for c in raw]
    if not configs or configs[0].overrides:
        raise SystemExit("the first configuration must be the default (empty overrides)")
    if len({c.name for c in configs}) != len(configs):
        raise SystemExit("configuration names must be unique")
    return configs


# ------------------------------------------------------------------ clips


def train_clips(names: str) -> list[tune.ClipRef]:
    if names == "train":
        return tune.proxy_clips()
    by_name = {c.name: c for c in tune.load_clips("train")}
    return [by_name[n] for n in names.split(",")]


def downscaled(clip: tune.ClipRef, height: int, frames: int) -> tune.ClipRef:
    """The first `frames` frames of `clip` at `height` lines (Lanczos), cached.

    The source-bitrate cap scales with the pixel count, so the 50 % rule
    keeps the same meaning at the lower resolution.
    """
    width = 2 * round(clip.width * height / clip.height / 2)
    directory = load_paths().data_root / "work" / "search-infra" / f"clips{height}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{clip.name}_{height}p_{frames}f.yuv"
    complete = lambda: path.is_file() and path.stat().st_size == width * height * 3 // 2 * frames
    if not complete():
        with entry_lock(path):
            if not complete():
                with sibling_temp(path, ".yuv") as pending:
                    subprocess.run([str(find_tool("ffmpeg", load_paths())), "-nostdin", "-v",
                                    "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s",
                                    f"{clip.width}x{clip.height}", "-r", f"{clip.fps:g}",
                                    "-i", str(clip.yuv), "-frames:v", str(frames),
                                    "-vf", f"scale={width}:{height}:flags=lanczos", "-f",
                                    "rawvideo", "-pix_fmt", "yuv420p", "-y", str(pending)],
                                   check=True)
                    publish(pending, path)
    scale = width * height / (clip.width * clip.height)
    return replace(clip, yuv=path, width=width, height=height, frames=frames,
                   source_kbps=clip.source_kbps * scale)


# ------------------------------------------------------------------ points


def _key(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]


def measure_point(binary: Path, codec: Codec, clip: tune.ClipRef, q: int, frames: int,
                  overrides: dict) -> dict:
    """Per-frame bytes, PSNR per plane and VMAF v1 of one point (cached)."""
    identity = {"binary": file_sha256(binary), "codec": codec.value,
                "clip": clip_fingerprint(clip), "frames": frames, "q": q,
                "overrides": sorted(overrides.items()), "measurement": measurement_fingerprint()}
    path = study_root(codec) / "points" / f"{clip.name}-{_key(identity)}.json"

    def compute():
        work = study_root(codec) / "tmp"
        work.mkdir(parents=True, exist_ok=True)
        stream = work / f"{_key(identity)}{codec.bitstream_suffix}"
        try:
            point = tune.encode_point(binary, codec, clip, q, frames, overrides, stream,
                                      tune.point_metrics(tune.V1_METRIC_PROFILE),
                                      standard=False)
            sizes = [f.bytes for f in probe_frames(stream, load_paths())]
            if len(sizes) != frames:
                raise RuntimeError(f"{clip.name}: {len(sizes)} packets for {frames} frames")
            rows = point.measurement.per_frame
            return {"q": q, "bytes": sizes, "gate": point.gate.level,
                    "stream_sha256": file_sha256(stream),
                    **{k: [r[k] for r in rows] for k in ("psnr_y", "psnr_u", "psnr_v", "vmaf_v1")}}
        finally:
            stream.unlink(missing_ok=True)
    return identity_cached(path, identity, compute)


def anchor_grid(binary: Path, codec: Codec, clip: tune.ClipRef, frames: int,
                calibrate: bool = True) -> list[int] | None:
    """Four legal quantizers for the default configuration (cached per clip).

    `calibrate` False only reads the cache (None when absent): analysis never encodes.
    """
    identity = {"binary": file_sha256(binary), "codec": codec.value,
                "clip": clip_fingerprint(clip), "frames": frames}
    path = study_root(codec) / "grids" / f"{clip.name}-{_key(identity)}.json"
    if not calibrate:
        return json.loads(path.read_text())["value"] if path.is_file() else None
    # Keyed like the entry: two binaries calibrating one clip never share it.
    work = study_root(codec) / "tmp" / f"grid-{clip.name}-{_key(identity)}"
    return identity_cached(path, identity, lambda: tune.calibrate_quantizers(
        binary, codec, clip, frames, {}, work))


def calibrated_grids(binary: Path, codec: Codec, clips: list[tune.ClipRef],
                     workers: int) -> dict[str, list[int]]:
    with ThreadPoolExecutor(min(workers, MAX_WORKERS)) as pool:
        grids = pool.map(lambda c: anchor_grid(binary, codec, c, FULL_FRAMES), clips)
        return {c.name: g for c, g in zip(clips, grids)}


def frame_instructions(binary: Path, codec: Codec, clip: tune.ClipRef, q: int, frames: int,
                       overrides: dict) -> list[int]:
    """Instructions of each coded frame, from one callgrind run (cached).

    The first entry includes the encoder's start-up, as tune.instructions'
    one-frame key cost does.
    """
    identity = {"binary": file_sha256(binary), "codec": codec.value,
                "clip": clip_fingerprint(clip), "frames": frames, "q": q,
                "overrides": sorted(overrides.items()), "symbol": FRAME_SYMBOL}
    path = study_root(codec) / "costs" / f"{clip.name}-{_key(identity)}.json"

    def compute():
        work = study_root(codec) / "tmp" / f"cg-{_key(identity)}"
        work.mkdir(parents=True, exist_ok=True)
        stream = work / f"cost{codec.bitstream_suffix}"
        out = work / "cg.out"
        try:
            with cpu_slots(1):
                tune.run(low_priority([
                    "valgrind", "--tool=callgrind", f"--callgrind-out-file={out}",
                    f"--dump-after={FRAME_SYMBOL}",
                    *tune.encode_command(binary, codec, clip, q, frames, overrides, stream, None)]))
            counts = []
            for index in range(1, frames + 1):
                text = (work / f"cg.out.{index}").read_text()
                counts.append(next(int(line.split()[1]) for line in text.splitlines()
                                   if line.startswith("summary:")))
            return counts
        finally:
            for f in work.iterdir():
                f.unlink()
            work.rmdir()
    return identity_cached(path, identity, compute)


# ------------------------------------------------------------------ measure


def measure(args: argparse.Namespace) -> None:
    codec, binary = Codec(args.codec), Path(args.binary).resolve()
    configs = load_configs(Path(args.configs))
    clips = train_clips(args.clips)
    if args.height:
        clips = [downscaled(c, args.height, FULL_FRAMES) for c in clips]
    grids = calibrated_grids(binary, codec, clips, args.workers)
    jobs = [(config, clip, index, q) for config in configs for clip in clips
            for index, q in enumerate(grids[clip.name])]
    part, parts = (int(x) for x in args.part.split("/"))
    jobs = jobs[part - 1::parts]
    pending = [j for j in jobs if not _point_cached(binary, codec, *j)]
    cpu = sum(point_cpu_seconds(clip.width, clip.height, FULL_FRAMES) for _, clip, _, _ in pending)
    check_budget(f"fidelity {codec.value} part {args.part}: {len(pending)} of {len(jobs)} points",
                 cpu, "use more --part slices", args.over_budget)

    def one(job):
        config, clip, _, q = job
        measure_point(binary, codec, clip, q, FULL_FRAMES, config.overrides)
        return config.name, clip.name, q
    with ThreadPoolExecutor(min(args.workers, MAX_WORKERS)) as pool:
        for done, (name, clip, q) in enumerate(pool.map(one, pending), 1):
            print(f"[{done}/{len(pending)}] {name} {clip} q{q}", flush=True)


def _point_cached(binary, codec, config, clip, index, q) -> bool:
    identity = {"binary": file_sha256(binary), "codec": codec.value,
                "clip": clip_fingerprint(clip), "frames": FULL_FRAMES, "q": q,
                "overrides": sorted(config.overrides.items()),
                "measurement": measurement_fingerprint()}
    return (study_root(codec) / "points" / f"{clip.name}-{_key(identity)}.json").is_file()


def cost(args: argparse.Namespace) -> None:
    """Per-frame callgrind costs of every configuration on the cost clips."""
    codec, binary = Codec(args.codec), Path(args.binary).resolve()
    configs = load_configs(Path(args.configs))
    clips = train_clips(args.clips)
    if args.height:
        clips = [downscaled(c, args.height, FULL_FRAMES) for c in clips]
    grids = calibrated_grids(binary, codec, clips, args.workers)

    def one(job):
        config, clip = job
        frame_instructions(binary, codec, clip, grids[clip.name][2], COST_FRAMES,
                           config.overrides)
        return config.name, clip.name
    jobs = [(config, clip) for config in configs for clip in clips]
    with ThreadPoolExecutor(min(args.workers, MAX_WORKERS)) as pool:
        for done, (name, clip) in enumerate(pool.map(one, jobs), 1):
            print(f"[{done}/{len(jobs)}] cost {name} {clip}", flush=True)


# ------------------------------------------------------------------ fidelities


@dataclass(frozen=True)
class Fidelity:
    """One way of measuring a trial more cheaply than in full."""
    clips: tuple[str, ...]
    frames: int = FULL_FRAMES
    qp_indices: tuple[int, ...] = (0, 1, 2, 3)
    vmaf_subsample: int = 1
    height: int | None = None  # None: native resolution

    def label(self) -> str:
        res = f"{self.height}p" if self.height else "native"
        qps = "4qp" if len(self.qp_indices) == 4 else f"{len(self.qp_indices)}qp"
        sub = f" vmaf/{self.vmaf_subsample}" if self.vmaf_subsample > 1 else ""
        return f"{len(self.clips)}clips {self.frames}f {qps} {res}{sub}"

    def relative_cost(self, full_clips: int) -> float:
        """Encode+metric cost relative to full fidelity (pixels x frames x points)."""
        pixels = (self.height / 720) ** 2 if self.height else 1.0
        return (len(self.clips) / full_clips * self.frames / FULL_FRAMES
                * len(self.qp_indices) / 4 * pixels)


def curve_at(point_list: list[dict], fidelity: Fidelity, fps: float) -> dict[str, list[float]]:
    """(kbps, PSNR-YUV, VMAF v1) per quantizer at `fidelity`'s prefix and subsample."""
    n = fidelity.frames
    out = {"kbps": [], "psnr": [], "vmaf_v1": []}
    for index in fidelity.qp_indices:
        p = point_list[index]
        out["kbps"].append(sum(p["bytes"][:n]) * 8 / 1000 / (n / fps))
        planes = [float(np.mean(p[k][:n])) for k in ("psnr_y", "psnr_u", "psnr_v")]
        out["psnr"].append(weighted_yuv(*planes))
        out["vmaf_v1"].append(float(np.mean(p["vmaf_v1"][:n:fidelity.vmaf_subsample])))
    return out


def objectives(table: dict, clip_refs: dict[str, tune.ClipRef], config: str,
               fidelity: Fidelity) -> dict[str, float] | None:
    """Mean per-clip BD (PSNR-YUV, VMAF v1) of `config` vs the default, and class means."""
    per_metric = {m: {} for m in METRICS}
    for clip in fidelity.clips:
        fps = clip_refs[clip].fps
        anchor = curve_at(table["default"][clip], fidelity, fps)
        test = curve_at(table[config][clip], fidelity, fps)
        for m in METRICS:
            r = bd_rate(anchor["kbps"], anchor[m], test["kbps"], test[m],
                        min_points=len(fidelity.qp_indices))
            if r.bd_rate_percent is None or not r.reliable:
                return None
            per_metric[m][clip] = r.bd_rate_percent
    out = {m: float(np.mean(list(v.values()))) for m, v in per_metric.items()}
    for m in METRICS:
        classes: dict[str, list[float]] = {}
        for clip, value in per_metric[m].items():
            classes.setdefault(clip_refs[clip].content_class, []).append(value)
        out[f"worst_class_{m}"] = max(float(np.mean(v)) for v in classes.values())
    return out


def load_table(binary: Path, codec: Codec, configs: list[Config], clips: list[tune.ClipRef],
               ) -> dict[str, dict[str, list[dict]]]:
    """config name -> clip name -> the four cached points (ascending QP); missing skipped."""
    table: dict[str, dict[str, list[dict]]] = {}
    for config in configs:
        name = "default" if not config.overrides else config.name
        for clip in clips:
            grid = anchor_grid(binary, codec, clip, FULL_FRAMES, calibrate=False)
            if grid is None:
                continue
            points = []
            for q in grid:
                identity = {"binary": file_sha256(binary), "codec": codec.value,
                            "clip": clip_fingerprint(clip), "frames": FULL_FRAMES, "q": q,
                            "overrides": sorted(config.overrides.items()),
                            "measurement": measurement_fingerprint()}
                path = study_root(codec) / "points" / f"{clip.name}-{_key(identity)}.json"
                if path.is_file():
                    points.append(json.loads(path.read_text())["value"])
            if len(points) == len(grid):
                table.setdefault(name, {})[clip.name] = points
    return table


def p_cost(per_frame: list[int], p_frames: int) -> float:
    """Mean instructions per P frame over the first `p_frames` P frames."""
    return float(np.mean(per_frame[1:1 + p_frames]))


# ------------------------------------------------------------------ agreement


def agreement(cheap: list[float], full: list[float]) -> dict[str, float]:
    """Rank and decision agreement of a cheap objective with the full one."""
    cheap_a, full_a = np.array(cheap), np.array(full)
    return {"spearman": float(spearmanr(cheap_a, full_a)[0]),
            "kendall": float(kendalltau(cheap_a, full_a)[0]),
            "sign_agreement": float(np.mean(np.sign(cheap_a) == np.sign(full_a))),
            "rmse": float(np.sqrt(np.mean((cheap_a - full_a) ** 2))),
            "bias": float(np.mean(cheap_a - full_a)),
            "pair_sigma": pair_sigma(cheap, full)}


def pair_sigma(cheap: list[float], full: list[float]) -> float:
    """RMS error of a cheap fidelity's *difference* between two configurations.

    This is the error a ladder's dominance test makes when it compares a
    trial with a finished one at the same rung (nceval.ladder sigma): any
    bias the two share cancels.
    """
    errors = [(cheap[i] - cheap[j]) - (full[i] - full[j])
              for i, j in combinations(range(len(full)), 2)]
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")


def front_of(points: dict[str, tuple[float, ...]]) -> set[str]:
    """Names of the nondominated points (every coordinate minimised)."""
    return {a for a, va in points.items()
            if not any(all(x <= y for x, y in zip(vb, va)) and vb != va
                       for b, vb in points.items() if b != a)}


def pair_agreement(cheap: list[float], full: list[float], tolerance: float) -> float:
    """Share of config pairs whose order the cheap fidelity gets right.

    Pairs closer than `tolerance` at full fidelity are ties nobody needs to
    order and are left out.
    """
    right = total = 0
    for i, j in combinations(range(len(full)), 2):
        if abs(full[i] - full[j]) < tolerance:
            continue
        total += 1
        right += (cheap[i] - cheap[j]) * (full[i] - full[j]) > 0
    return right / total if total else float("nan")


#: The preset search's exploration set (work/presets2/run_study.sh, 2026-11-08).
PRESET_6 = ("bodycam_bc1", "crowd_mot17_09", "crowd_pp22_027", "lowlight_city_night",
            "pan_jaad_0223", "static_meva_hospital")


def quality_fidelities(full_clips: tuple[str, ...], low_clips: tuple[str, ...]) -> list[Fidelity]:
    """The cheap fidelities compared with full fidelity (all from cached points)."""
    sets = {"train": full_clips, "proxy-6": PROXY_6, "preset-6": PRESET_6, "proxy-4": PROXY_4}
    out = []
    for clips in sets.values():
        if not set(clips) <= set(full_clips):
            continue
        for frames in FRAME_PREFIXES:
            out.append(Fidelity(tuple(clips), frames))
            out.append(Fidelity(tuple(clips), frames, (0, 1, 3)))
        out.append(Fidelity(tuple(clips), 60, vmaf_subsample=4))
    for frames in FRAME_PREFIXES:
        if low_clips:
            out.append(Fidelity(low_clips, frames, height=LOW_HEIGHT))
    return out


def analyse(args: argparse.Namespace) -> dict:
    codec, binary = Codec(args.codec), Path(args.binary).resolve()
    configs = load_configs(Path(args.configs))
    native = train_clips(args.clips)
    low = [downscaled(c, LOW_HEIGHT, FULL_FRAMES) for c in native]
    table = load_table(binary, codec, configs, native)
    low_table = load_table(binary, codec, configs, low)
    refs = {c.name: c for c in native}
    complete = [c.name for c in configs[1:] if c.name in table
                and len(table[c.name]) == len(native)]
    full_clips = tuple(c.name for c in native)
    if "default" not in table or len(table["default"]) != len(native):
        raise SystemExit("the default configuration is not measured on every clip")
    low_clips = tuple(name for name in full_clips if name in low_table.get("default", {})
                      and all(name in low_table.get(n, {}) for n in complete))
    full = Fidelity(full_clips, args.reference_frames)
    reference = {n: objectives(table, refs, n, full) for n in complete}
    names = [n for n in complete if reference[n] is not None]
    print(f"{codec.value}: {len(names)} configurations; reference fidelity {full.label()}")
    costs = cost_table(binary, codec, configs)
    report = {"codec": codec.value, "configs": len(names), "quality": [], "cpu": []}
    terms = list(METRICS) + [f"worst_class_{m}" for m in METRICS]
    header = "  ".join(f"{t[:14]:>14}" for t in terms)
    print("cells: Spearman / sign agreement / pairwise sigma (%)")
    print(f"{'fidelity':<34} {'cost':>6}  {header}   front recall")
    for fidelity in quality_fidelities(full_clips, low_clips):
        source = low_table if fidelity.height else table
        source_refs = ({c.name: c for c in low} if fidelity.height else refs)
        cheap = {n: objectives(source, source_refs, n, fidelity) for n in names}
        usable = [n for n in names if cheap[n] is not None]
        row = {"fidelity": fidelity.label(),
               "relative_cost": fidelity.relative_cost(len(full_clips)),
               "configs": len(usable), "dropped": sorted(set(names) - set(usable))}
        for t in terms:
            row[t] = agreement([cheap[n][t] for n in usable], [reference[n][t] for n in usable])
            row[t]["pairs"] = pair_agreement([cheap[n][t] for n in usable],
                                             [reference[n][t] for n in usable], 0.5)
        if costs:
            full_points = {n: (reference[n]["psnr"], reference[n]["vmaf_v1"], costs[n][48])
                           for n in usable if n in costs}
            cheap_points = {n: (cheap[n]["psnr"], cheap[n]["vmaf_v1"], costs[n][48])
                            for n in usable if n in costs}
            front_full, front_cheap = front_of(full_points), front_of(cheap_points)
            row["front_recall"] = len(front_full & front_cheap) / max(1, len(front_full))
        report["quality"].append(row)
        cells = "  ".join(f"{row[t]['spearman']:>5.2f}/{row[t]['sign_agreement']:.2f}"
                          f"/{row[t]['pair_sigma']:.2f}" for t in terms)
        print(f"{row['fidelity']:<34} {row['relative_cost']:>6.3f}  {cells} "
              f"{row.get('front_recall', float('nan')):.2f}")
    if costs:
        print("CPU (callgrind P-frame instructions, ratio to default):")
        reference_cpu = [costs[n][48] for n in names if n in costs]
        for p in COST_P_PREFIXES:
            cheap = [costs[n][p] for n in names if n in costs]
            row = {"p_frames": p, **agreement(cheap, reference_cpu)}
            report["cpu"].append(row)
            print(f"  {p:>2} P frames: spearman {row['spearman']:.3f}  kendall {row['kendall']:.3f}"
                  f"  rmse {row['rmse']:.4f}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    return report


def cost_table(binary: Path, codec: Codec, configs: list[Config]) -> dict[str, dict[int, float]]:
    """config -> P prefix -> P instructions relative to the default, over the cost clips."""
    clips = {c.name: c for c in tune.load_clips("train")}
    out: dict[str, dict[int, float]] = {}
    base = None
    for config in configs:
        per_prefix = {p: [] for p in COST_P_PREFIXES}
        for name in tune.COST_CLIPS:
            clip = clips[name]
            grid = anchor_grid(binary, codec, clip, FULL_FRAMES, calibrate=False)
            if grid is None:
                break
            identity = {"binary": file_sha256(binary), "codec": codec.value,
                        "clip": clip_fingerprint(clip), "frames": COST_FRAMES, "q": grid[2],
                        "overrides": sorted(config.overrides.items()), "symbol": FRAME_SYMBOL}
            path = study_root(codec) / "costs" / f"{name}-{_key(identity)}.json"
            if not path.is_file():
                break
            frames = json.loads(path.read_text())["value"]
            for p in COST_P_PREFIXES:
                per_prefix[p].append(p_cost(frames, p))
        else:
            values = {p: float(np.mean(v)) for p, v in per_prefix.items()}
            if base is None:
                base = values
            out["default" if not config.overrides else config.name] = {
                p: values[p] / base[p] for p in COST_P_PREFIXES}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="nceval.fidelity", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("measure", measure), ("cost", cost), ("analyse", analyse)):
        p = sub.add_parser(name)
        p.add_argument("--codec", required=True, choices=[c.value for c in Codec])
        p.add_argument("--binary", required=True)
        p.add_argument("--configs", required=True)
        p.add_argument("--clips", default="train")
        p.add_argument("--workers", type=int, default=MAX_WORKERS)
        p.set_defaults(func=func)
        if name != "analyse":
            p.add_argument("--height", type=int, help="measure on downscaled sources")
        else:
            p.add_argument("--json")
            p.add_argument("--reference-frames", type=int, default=FULL_FRAMES,
                           help="frames of the reference fidelity (60: the preset study's own)")
        if name == "measure":
            p.add_argument("--part", default="1/1", help="slice K/N of the points")
            p.add_argument("--over-budget", action="store_true")
    args = parser.parse_args()
    if args.workers > MAX_WORKERS:
        raise SystemExit(f"at most {MAX_WORKERS} workers (project rule)")
    args.func(args)


if __name__ == "__main__":
    main()
