"""Clip subsets that predict the whole pool, chosen from measured clip responses.

A small clip set is only as good as its agreement with the pool. This
module measures it (owner, 2026-11-11):

  measure   the response matrix R (pool clips x perturbations): each
            perturbation is a random point of a preset's trust region
            (nceval.tune_pool space file), coded on one clip of every pool
            family at one ladder rung (150 frames, CBR, rungs cycled as in
            the tuner), its per-clip single-point J_real quality vs the
            preset's full curve recorded (jreal.point_quality). Points
            are cached (nceval.realuse), and the matrix is rewritten
            after each perturbation, so a stopped run resumes. Sized for an overnight queue.
  select    centres R, reports the SVD (variance per component and which
            classes / datasets load on it), then picks k clips by greedy
            forward selection (minimum error of the subset's class-balanced
            mean against the pool's) and by CUR (pivoted QR on the top
            components), with every stratum covered. Each is validated on
            held-out perturbations (repeated K-fold): correlation and sign
            agreement with the pool mean, against random stratified draws
            of the same size. Writes the subsets (several near-equivalent
            ones from the folds, to rotate between studies) as JSON.

On today's recorded screens (36 candidates x the 29 TRAIN+VAL clips, |J| <=
3 %), held-out greedy subsets of 8 / 12 clips reached r 0.945 / 0.954
against 0.824 / 0.908 for random stratified draws; CUR did no better than
random (docs/handoff/tuning.md).

    python3 -m nceval.clipselect measure --codec h264 --preset h264-fast \\
        --binary BIN --space tune_spaces/h264-fast.txt --count 24 --out R.json
    python3 -m nceval.clipselect select --matrix R.json --k 8,12,16 --out subsets.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np

from . import levels, pool, realuse
from .codecs import Codec
from .config import load_paths
from .encoders.nanocodec import read_preset
from .gatepolicy import EVALUATION_GATE, file_sha256
from .singlepoint import assign_rungs, single_points
from .tune import REPO_ROOT, specs_of
from .tune_pool import family, read_space, settings_label, sparse_moves

FOLDS = 6
REPEATS = 10
RANDOM_DRAWS = 20
#: L1's chosen sets: their size and how many near-equivalent ones rotate.
L1_SIZE = 12
L1_SETS_KEPT = 4
#: Sign agreement counts perturbations whose pool J is at least this (percent).
SIGN_FLOOR = 0.3


# --- measure ------------------------------------------------------------------------

def measure(args: argparse.Namespace) -> None:
    codec = Codec(args.codec)
    binary, _ = levels.build(args.binary, load_paths(), "tune")
    presets = binary.parent / "presets"
    presets = presets if presets.is_dir() else REPO_ROOT / "presets"
    preset = read_preset(presets / f"{args.preset}.toml") if args.preset != "default" else {}
    domains = read_space(Path(args.space).read_text(), specs_of(binary, codec), preset)
    by_family = {}
    for clip in pool.load_pool(load_paths(), levels.usable):
        by_family.setdefault(clip.family, clip)
    clips = sorted(by_family.values(), key=lambda c: (pool.stratum(c), c.name))
    rungs = assign_rungs(clips)
    out = Path(args.out)
    matrix = json.loads(out.read_text()) if out.exists() else {
        "codec": codec.value, "preset": args.preset, "binary_sha256": file_sha256(binary),
        "clips": {c.name: {"class": c.content_class, "stratum": pool.stratum(c),
                           "dataset": c.dataset, "family": c.family, "grade": c.grade,
                           "height": c.height, "source_bpp": c.source_bpp}
                  for c in clips},
        "perturbations": []}
    if matrix["binary_sha256"] != file_sha256(binary):
        raise SystemExit(f"{out} was measured with another binary")
    anchor = family(codec, binary, preset, f"nanocodec-{codec.value}")
    runner = realuse.Runner(load_paths(), EVALUATION_GATE, args.workers, over_budget=False)
    done = len(matrix["perturbations"])
    for index, moves in enumerate(sparse_moves(domains, args.count, args.seed)):
        if index < done:
            continue
        values = {**preset, **moves}
        test = family(codec, binary, values, f"nanocodec-{codec.value}@{settings_label(values)}")
        rows = single_points(runner, anchor, test, clips, rungs)
        matrix["perturbations"].append({"moves": moves, "per_clip": {
            name: {"quality": row["quality"], "quality_low": row["quality_low"],
                   "rung": row["rung"]} for name, row in rows.items()}})
        out.write_text(json.dumps(matrix) + "\n")
        print(f"perturbation {index + 1}/{args.count} done", flush=True)


# --- select -------------------------------------------------------------------------

class Responses:
    """R (clips x perturbations) of per-clip J, with each clip's stratum."""

    def __init__(self, matrix: dict):
        perts = matrix["perturbations"]
        j = lambda row: None if row is None or row["quality"] is None or \
            row["quality_low"] is None else (row["quality"] + row["quality_low"]) / 2
        names = [n for n in matrix["clips"]
                 if all(j(p["per_clip"].get(n)) is not None for p in perts)]
        self.names = names
        self.meta = [matrix["clips"][n] for n in names]
        self.strata = [m["stratum"] for m in self.meta]
        self.R = np.array([[j(p["per_clip"][n]) for p in perts] for n in names])

    def balanced(self, columns: np.ndarray, rows) -> np.ndarray:
        groups = collections.defaultdict(list)
        for i in rows:
            groups[self.strata[i]].append(i)
        return np.mean([columns[g].mean(axis=0) for g in groups.values()], axis=0)

    def covered(self, chosen: list[int], k: int) -> bool:
        missing = len(set(self.strata)) - len({self.strata[i] for i in chosen})
        return missing <= k - len(chosen)


def greedy(resp: Responses, columns: np.ndarray, k: int) -> list[int]:
    target = resp.balanced(columns, range(len(resp.names)))
    chosen: list[int] = []
    while len(chosen) < k:
        best = min((i for i in range(len(resp.names))
                    if i not in chosen and resp.covered(chosen + [i], k)),
                   key=lambda i: np.mean((resp.balanced(columns, chosen + [i]) - target) ** 2))
        chosen.append(best)
    return chosen


def cur(resp: Responses, columns: np.ndarray, k: int) -> list[int]:
    """Pivoted QR on the clips' top-k component scores, strata covered."""
    centred = columns - columns.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    residual = (u[:, :k] * s[:k]).T.copy()
    chosen: list[int] = []
    while len(chosen) < k:
        order = np.argsort(-(residual ** 2).sum(axis=0))
        pick = next(int(i) for i in order if i not in chosen and resp.covered(chosen + [i], k))
        chosen.append(pick)
        q = residual[:, pick] / (np.linalg.norm(residual[:, pick]) + 1e-12)
        residual -= np.outer(q, q @ residual)
    return chosen


def random_draw(resp: Responses, k: int, rng: random.Random) -> list[int]:
    by_stratum = collections.defaultdict(list)
    for i, s in enumerate(resp.strata):
        by_stratum[s].append(i)
    queues = [rng.sample(v, len(v)) for v in by_stratum.values()]
    rng.shuffle(queues)
    chosen: list[int] = []
    while len(chosen) < k:
        for queue in queues:
            if queue and len(chosen) < k:
                chosen.append(queue.pop())
    return chosen


def agreement(predicted: list[float], actual: list[float]) -> dict[str, float]:
    p, a = np.array(predicted), np.array(actual)
    clear = np.abs(a) >= SIGN_FLOOR
    return {"r": float(np.corrcoef(p, a)[0, 1]),
            "sign": float(np.mean(np.sign(p[clear]) == np.sign(a[clear]))) if clear.any() else 1.0,
            "rmse": float(np.sqrt(np.mean((p - a) ** 2)))}


def validate(resp: Responses, k: int, seed: int) -> dict:
    """Held-out agreement of greedy, CUR and random draws; greedy subsets per fold."""
    rng = random.Random(seed)
    count = resp.R.shape[1]
    target = resp.balanced(resp.R, range(len(resp.names)))
    pairs = collections.defaultdict(lambda: ([], []))
    subsets = collections.Counter()
    for _ in range(REPEATS):
        order = list(range(count))
        rng.shuffle(order)
        for fold in range(FOLDS):
            test = order[fold::FOLDS]
            train = [c for c in order if c not in test]
            methods = {"greedy": greedy(resp, resp.R[:, train], k),
                       "cur": cur(resp, resp.R[:, train], k)}
            subsets[tuple(sorted(methods["greedy"]))] += 1
            methods |= {f"random{d}": random_draw(resp, k, rng) for d in range(RANDOM_DRAWS)}
            for name, chosen in methods.items():
                pairs[name][0].extend(resp.balanced(resp.R[:, test], chosen))
                pairs[name][1].extend(target[test])
    scores = {name: agreement(*pairs[name]) for name in ("greedy", "cur")}
    randoms = [agreement(*pairs[f"random{d}"]) for d in range(RANDOM_DRAWS)]
    scores["random"] = {m: {"median": float(np.median([r[m] for r in randoms])),
                            "p90": float(np.percentile([r[m] for r in randoms], 90))}
                        for m in ("r", "sign", "rmse")}
    rotation = [[resp.names[i] for i in subset] for subset, _ in subsets.most_common(5)]
    return {"k": k, "scores": scores, "subsets": rotation}


def components(resp: Responses, top: int = 4) -> list[dict]:
    centred = resp.R - resp.R.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    share = s ** 2 / (s ** 2).sum()
    out = []
    for i in range(min(top, len(s))):
        score = u[:, i] * s[i]
        by = lambda key: {v: round(float(np.mean([score[j] for j, m in enumerate(resp.meta)
                                                    if m[key] == v])), 2)
                          for v in sorted({str(m[key]) for m in resp.meta})
                          if sum(str(m[key]) == v for m in resp.meta) >= 2}
        order = np.argsort(score)
        out.append({"share": float(share[i]), "by_stratum": by("stratum"),
                    "by_grade": by("grade"), "by_dataset": by("dataset"),
                    "low": [resp.names[j] for j in order[:4]],
                    "high": [resp.names[j] for j in order[-4:]]})
    return out


def select(args: argparse.Namespace) -> None:
    resp = Responses(json.loads(Path(args.matrix).read_text()))
    print(f"{len(resp.names)} clips x {resp.R.shape[1]} perturbations")
    report = {"components": components(resp), "subsets": []}
    for c, comp in enumerate(report["components"], 1):
        print(f"PC{c} {comp['share']:.0%}: strata {comp['by_stratum']}; low {comp['low']}; "
              f"high {comp['high']}")
    for k in (int(x) for x in args.k.split(",")):
        result = validate(resp, k, args.seed)
        report["subsets"].append(result)
        s = result["scores"]
        print(f"k={k}: greedy r {s['greedy']['r']:.3f} sign {s['greedy']['sign']:.0%}; "
              f"cur r {s['cur']['r']:.3f}; random r {s['random']['r']['median']:.3f} "
              f"(p90 {s['random']['r']['p90']:.3f}) sign {s['random']['sign']['median']:.0%}")
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.out}")
    if args.l1_out:
        write_l1_sets(report, Path(args.l1_out), args.matrix)


def write_l1_sets(report: dict, path: Path, matrix: str) -> None:
    """L1's chosen sets (nceval.pool.L1_SETS): the most frequent greedy subsets of
    size L1_SIZE, only when greedy beats the random draws' p90 on held-out r."""
    result = next((r for r in report["subsets"] if r["k"] == L1_SIZE), None)
    if result is None:
        raise SystemExit(f"--l1-out needs k {L1_SIZE} in --k")
    scores = result["scores"]
    better = scores["greedy"]["r"] > scores["random"]["r"]["p90"]
    sets = result["subsets"][:L1_SETS_KEPT] if better else []
    path.write_text(json.dumps({
        "_comment": f"written by nceval.clipselect from {Path(matrix).name}: greedy k {L1_SIZE} "
                    f"held-out r {scores['greedy']['r']:.3f} vs random median "
                    f"{scores['random']['r']['median']:.3f} (p90 "
                    f"{scores['random']['r']['p90']:.3f})", "sets": sets}, indent=1) + "\n")
    print(f"wrote {path}: {len(sets)} sets" + ("" if better else " (greedy did not beat random)"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python3 -m nceval.clipselect",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="the response matrix (resumable)")
    m.add_argument("--codec", required=True, choices=[c.value for c in Codec])
    m.add_argument("--preset", default="default")
    m.add_argument("--binary", required=True)
    m.add_argument("--space", required=True, help="nceval.tune_pool space file")
    m.add_argument("--count", type=int, default=24)
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--workers", type=int, default=realuse.DEFAULT_WORKERS)
    m.add_argument("--out", required=True)
    m.set_defaults(func=measure)
    s = sub.add_parser("select", help="SVD, subset selection and held-out validation")
    s.add_argument("--matrix", required=True)
    s.add_argument("--k", default="8,12,16")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", required=True)
    s.add_argument("--l1-out", help="also write L1's chosen sets here (eval/clipset/l1_sets.json)")
    s.set_defaults(func=select)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
