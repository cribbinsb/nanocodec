#!/usr/bin/env python3
"""Benchmark job plan: every encode's command, built from the fairness record only.

    python3 tools/bench/plan.py --clips CLIPS.json --out JOBS.json [--families a,b|all]
        [--mode cbr|qp] [--rungs 300,500,...] [--frames 150] [--long-frames 600]
        [--keyint 150] [--sealed --frozen-commit HASH]

The first piece of BP-3's `tools/bench` (docs/benchmark-procedure.md section 13): it turns a clip
list and a family list into one job per encode, each command taken from
`nceval.fairness.command` (docs/benchmark-fairness.toml) and nothing else, so a run cannot use a
command the record does not hold.

CLIPS.json: a clip set file or a list of {"name", "width", "height", "fps", "source_kbps",
"content_class"}. A CBR rung is named by its 720p30 rate; the clip's target is rung x
(pixels / 1280x720)^0.75 x (fps / 30)^0.5 (section 5). A rung above half the clip's source
bitrate is not planned, and a clip without `source_kbps` is refused. Fixed QP plans the six
base quantizers of the encoder's scale (`nceval.ctc.conditions.POINTS`). A family the record
does not chart in a mode (`fairness.NotCharted`) is left out and named on stderr.

Windows (BP-4.0, D7/D8): every clip is coded once. A clip whose entry gives
`frames_available` of at least --long-frames is coded at that length, every other clip at
--frames; a key comes every --keyint frames. The job lists the windows it serves (`windows`):
the short one always, read from the first --frames frames of the encode (report.py), and the
long one when coded at it. No clip is coded twice.

Each job: id, family, variant ("record"), input, w, h, fps, frames, windows, keyint, kbps,
rung, qp, class, codec, container, decoder (the codec's standard decoder, ours included:
section 7), argv (placeholders {BIN} {CFG} {INPUT} {OUTPUT} for the box).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import fairness, rate_ladder  # noqa: E402
from nceval.rate_ladder import SOURCE_RATE_SHARE, frame_rate  # noqa: E402
from nceval.codecs import Codec  # noqa: E402
from nceval.ctc.conditions import POINTS, Scale, ours_points  # noqa: E402

#: BP-3 ladder and its per-clip scaling: one definition, shared with the dev screens.
LADDER_720P30 = rate_ladder.LADDER_720P30_KBPS
REPO = Path(__file__).resolve().parents[2]
#: Clips that left a sealed set because they were seen (docs/results/sealed-log.md).
SEALED_EXCLUSIONS = REPO / "eval" / "clipset" / "sealed_exclusions.json"
#: (codec, container) per tool; ours takes the codec from the family name.
TOOL_STREAM = {"x264": ("h264", "annexb264"), "openh264-harness": ("h264", "annexb264"),
               "x265": ("hevc", "annexb265"), "kvazaar": ("hevc", "annexb265"),
               "vvencFFapp": ("vvc", "annexb266"), "SvtAv1EncApp": ("av1", "ivf"),
               "aomenc": ("av1", "ivf"), "svc_encoder_rtc": ("av1", "ivf"),
               "vpxenc": ("vp9", "ivf")}
OURS_CONTAINER = {"h264": "annexb264", "hevc": "annexb265", "vvc": "annexb266", "av1": "ivf",
                  "vp9": "ivf", "av2": "ivf"}
#: One base quantizer per scale for a fixed-QP probe: H.264/HEVC/VVC QP, AV1/VP9/AV2 qindex
#: (ours), and the 0-63 quantizer of SVT-AV1, libaom and libvpx.
#: Fixed QP: the six base quantizers of each scale (section 4, `nceval.ctc.conditions.POINTS`).
Q63_TOOLS = ("SvtAv1EncApp", "svc_encoder_rtc", "aomenc", "vpxenc")
#: Every stream is decoded for scoring by its codec's standard decoder, ours included (section 6).
STANDARD_DECODER = {"h264": "ffmpeg", "hevc": "ffmpeg", "vvc": "ffmpeg", "av1": "ffmpeg",
                    "vp9": "ffmpeg", "av2": "avmdec"}
SEALED_LOG = REPO / "docs" / "results" / "sealed-log.md"
#: BP-3.4 (owner): the anchor, x264 medium and its per-metric twins, also codes points past
#: each end of the ladder, so every other family's curve lies inside its quality range and BD
#: needs no extrapolation. These points serve the anchor's BD curve only (report.py).
ANCHOR_FAMILIES = ("x264-medium", "x264-medium~psnr", "x264-medium~ssim")
ANCHOR_LOW = 0.5   # x the bottom CBR rung
#: x the top CBR rung; 4x added by the owner on 2026-11-25 (D11, T30) so that encoders whose
#: curves lie above x264's quality range keep their clips.
ANCHOR_HIGH = (2.0, 4.0)
#: BP-4.0 windows (D7, D8): the short window every clip gets, the long one where the source
#: allows it, and the key interval in both (dev's L3 and final check: nceval.key_shares).
SHORT_FRAMES, LONG_FRAMES, KEY_INTERVAL = 150, 600, 150
#: Fixed QP: one CTC step past each end of the QP table, within H.264's QP range.
ANCHOR_QP_STEP, H264_MAX_QP = 5, 51


def quantizers(tool: str, codec: str) -> tuple[int, ...]:
    """The six base quantizers of the encoder's scale."""
    if tool in Q63_TOOLS:
        return POINTS[Scale.Q63]
    if tool == "nanocodec":  # one definition with the dev screens (nceval.ctc.conditions)
        return ours_points(Codec(codec))
    return POINTS[Scale.QP]


def scaled_kbps(rung: int, clip: dict) -> int:
    """A manifest clip's target for a 720p30-named rung (nceval.rate_ladder)."""
    return rate_ladder.scaled_kbps(rung, clip["width"], clip["height"], clip["fps"])


def stream_kind(family: str, tool: str) -> tuple[str, str]:
    if tool == "nanocodec":
        codec = family.removeprefix("nanocodec-").split("@")[0]
        return codec, OURS_CONTAINER[codec]
    return TOOL_STREAM[tool]


def anchor_points(clip: dict, rungs: list[int], mode: str,
                  qps: tuple[int, ...]) -> list[tuple[int, int, bool]]:
    """The anchor's extra points past each end of the ladder, as (point, kbps, True).

    CBR: 0.5x the bottom rung, and 2x and 4x the top (D11), scaled like every rung. A top point
    over half the source rate becomes the cap itself if the cap lies above the top rung's target
    and no lower top point is already at it; otherwise it is not coded (the source is then too
    compressed to go higher)."""
    if mode == "qp":
        return [(qps[0] - ANCHOR_QP_STEP, 0, True),
                (min(qps[-1] + ANCHOR_QP_STEP, H264_MAX_QP), 0, True)]
    cap = SOURCE_RATE_SHARE * clip["source_kbps"]
    low = scaled_kbps(rungs[0], clip) * ANCHOR_LOW
    top = scaled_kbps(rungs[-1], clip)
    out = [(round(rungs[0] * ANCHOR_LOW), max(1, round(low)), True)] if low <= cap else []
    coded = {round(top)}
    for factor in ANCHOR_HIGH:
        high = round(min(top * factor, cap))
        if high > top and high not in coded:
            coded.add(high)
            out.append((round(rungs[-1] * factor), high, True))
    return out


def clip_frames(clip: dict, short: int, long: int | None) -> tuple[int, list[int]]:
    """(frames to code, windows served): the long window when the clip's source allows it
    (`frames_available`), else the short one; the short window is always served."""
    if long and clip.get("frames_available", 0) >= long:
        return long, [short, long]
    return short, [short]


def plan(clips: list[dict], families: list[str], mode: str, rungs: list[int], frames: int,
         keyint: int, anchor_only: bool = False, long_frames: int | None = None) -> list[dict]:
    record = fairness.load()
    known = fairness.families()
    jobs, not_charted = [], set()
    for clip in clips:
        num, den = frame_rate(clip["fps"])
        coded, windows = clip_frames(clip, frames, long_frames)
        if mode == "cbr" and not clip.get("source_kbps"):
            raise SystemExit(f"refused: {clip['name']} has no source_kbps (the 50 % rule)")
        targets = [(r, scaled_kbps(r, clip)) for r in rungs]
        targets = [(r, k) for r, k in targets if rate_ladder.allowed(k, clip["source_kbps"])] \
            if mode == "cbr" else []
        for family in families:
            tool, _ = known[family]
            codec, container = stream_kind(family, tool)
            points = targets if mode == "cbr" else [(q, 0) for q in quantizers(tool, codec)]
            points = [(p, k, False) for p, k in points]
            if family in ANCHOR_FAMILIES:
                extra = anchor_points(clip, rungs, mode, quantizers(tool, codec))
                points = [x for x in points if not anchor_only] + extra
            for point, kbps, anchor_extra in points:
                values = {"BIN": "{BIN}", "CFG": "{CFG}", "INPUT": "{INPUT}", "OUTPUT": "{OUTPUT}",
                          "W": clip["width"], "H": clip["height"], "FPS_NUM": num, "FPS_DEN": den,
                          "N": coded, "KEYINT": keyint}
                values.update({"KBPS": kbps} if mode == "cbr" else {"QP": point})
                try:
                    argv = fairness.command(family, mode, values, record)
                except fairness.NotCharted:
                    not_charted.add(family)
                    break
                jobs.append({"id": f"{family}|{clip['name']}|{mode}{point}"
                                   + ("-anchor" if anchor_extra else ""),
                             "anchor_extra": anchor_extra,
                             "family": family, "variant": "record", "input": clip["name"],
                             "w": clip["width"], "h": clip["height"], "fps": f"{num}/{den}",
                             "frames": coded, "windows": windows, "keyint": keyint,
                             "kbps": kbps, "rung": point if mode == "cbr" else 0,
                             "qp": point if mode == "qp" else 0,
                             "class": clip.get("content_class", clip.get("class", "")),
                             "codec": codec, "container": container,
                             "decoder": STANDARD_DECODER[codec], "mode": mode, "argv": argv})
    if not_charted:
        print(f"not charted in {mode} (the record says why): {', '.join(sorted(not_charted))}",
              file=sys.stderr)
    return jobs


def check_sealed(data: dict | list, clips: list[dict], frozen: str | None) -> None:
    """V22: a sealed run uses no excluded clip, runs the frozen procedure commit, and its set
    has a sealed-log entry."""
    seen = {e["name"] for e in json.loads(SEALED_EXCLUSIONS.read_text())["clips"]}
    bad = sorted(seen & {c["name"] for c in clips})
    if bad:
        raise SystemExit(f"refused: {len(bad)} clips were seen before sealing, e.g. {bad[0]}")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "docs",
                            "eval", "tools/bench"], capture_output=True, text=True,
                           check=True).stdout.strip()
    if not frozen or not head.startswith(frozen) or dirty:
        raise SystemExit("refused: a sealed run needs --frozen-commit equal to a clean HEAD")
    name = data.get("name") if isinstance(data, dict) else None
    if not name or f"`{name}`" not in SEALED_LOG.read_text():
        raise SystemExit(f"refused: the set {name!r} has no entry in docs/results/sealed-log.md")


def select(patterns: str) -> list[str]:
    names = sorted(fairness.families())
    if patterns == "all":
        return names
    chosen = [n for p in patterns.split(",") for n in names if fnmatch.fnmatchcase(n, p)]
    if not chosen:
        raise SystemExit(f"no family matches {patterns}")
    return list(dict.fromkeys(chosen))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--families", default="all")
    ap.add_argument("--mode", choices=("cbr", "qp"), default="cbr")
    ap.add_argument("--rungs", default=",".join(map(str, LADDER_720P30)))
    ap.add_argument("--frames", type=int, default=SHORT_FRAMES,
                    help="the short window, coded for every clip")
    ap.add_argument("--long-frames", type=int, default=LONG_FRAMES,
                    help="the long window, coded instead for clips whose `frames_available` "
                         "reaches it (0: none)")
    ap.add_argument("--keyint", type=int, default=KEY_INTERVAL)
    ap.add_argument("--sealed", action="store_true",
                    help="a sealed (publishable) run: refuse excluded clips, an unfrozen tree "
                         "and a set without a sealed-log entry (V22)")
    ap.add_argument("--frozen-commit", help="the frozen procedure commit (a sealed run)")
    ap.add_argument("--anchor-extra-only", action="store_true",
                    help="only the anchor's extra points (BP-3.4), for a run that lacks them")
    a = ap.parse_args()
    data = json.loads(a.clips.read_text())
    clips = data["clips"] if isinstance(data, dict) else data  # a clip set file or a plain list
    if a.sealed:
        check_sealed(data, clips, a.frozen_commit)
    jobs = plan(clips, select(a.families), a.mode,
                [int(r) for r in a.rungs.split(",")], a.frames, a.keyint, a.anchor_extra_only,
                a.long_frames or None)
    if a.anchor_extra_only:
        jobs = [j for j in jobs if j["anchor_extra"]]
    a.out.write_text(json.dumps(jobs, indent=0) + "\n")
    print(f"{len(jobs)} jobs -> {a.out}")


if __name__ == "__main__":
    main()
