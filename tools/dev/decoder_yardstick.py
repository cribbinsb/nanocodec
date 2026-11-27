#!/usr/bin/env python3
"""Committed, reproducible single-thread decoder yardstick across rented AWS CPUs.

Builds its own self-contained stream set (our `medium` and `slower` 720p
encodes plus a few conformance streams per codec), runs single-thread
interleaved decode comparisons (`tools/dev/decoder_speed.py`: in process, against
FFmpeg's decoders and dav1d through libavcodec's `libavcodec-bench`) on rented Graviton3, Graviton4 and Zen 5 cores via
`tools/dev/aws_session.py`, and writes a paired JSON record and Markdown report
into `docs/results/decoder-yardstick-<date>.{json,md}` with machine, build,
stream set and method metadata.

Example:
    python3 tools/dev/decoder_yardstick.py \\
        --x86-bin target/release/nanocodec \\
        --arm64-bin target/aarch64-unknown-linux-gnu/release/nanocodec \\
        --targets graviton3,graviton4,zen5 \\
        --rounds 5 \\
        --out-prefix docs/results/decoder-yardstick-2026-11-22
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aws_session import Session, SessionSpec, say  # noqa: E402

DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
DEFAULT_CLIP = DATA / "release/pgo-clips/old_town_cross_420_720p50.yuv"
DEFAULT_CLIP_W, DEFAULT_CLIP_H, DEFAULT_CLIP_FPS = 1280, 720, 50

#: Codec -> (fixed QP for 720p medium/slower encodes, filename suffix for decoder_speed.py).
CODECS: dict[str, tuple[int, str]] = {
    "av1": (110, ".av1.ivf"),
    "av2": (110, ".av2.ivf"),
    "vp9": (110, ".vp9.ivf"),
    "h264": (26, ".264"),
    "hevc": (26, ".hevc"),
    "vvc": (26, ".vvc"),
}

#: Preferred conformance candidates per codec (first two present & decodable win).
CONFORMANCE_CANDIDATES: dict[str, list[str]] = {
    "av1": [
        "av1/streams/av1-1-b8-02-allintra/av1-1-b8-02-allintra.ivf",
        "av1/streams/av1-1-b8-16-intra_only-intrabc-extreme-dv/av1-1-b8-16-intra_only-intrabc-extreme-dv.ivf",
        "av1/streams/av1-1-b8-05-mv/av1-1-b8-05-mv.ivf",
    ],
    "av2": [],
    "vp9": [
        "vp9/streams/vp90-2-02-size-lf-1920x1080/vp90-2-02-size-lf-1920x1080.ivf",
        "vp9/streams/vp90-2-09-subpixel-00/vp90-2-09-subpixel-00.ivf",
        "vp9/streams/vp90-2-22-svc_1280x720_1/vp90-2-22-svc_1280x720_1.ivf",
    ],
    "h264": [
        "h264/streams/HCHP3_HHI_A/HCHP3_HHI_A.264",
        "h264/streams/MV1_BRCM_D/src19td.IBP.264",
        "h264/streams/CABACI3_Sony_B/CABACI3_Sony_B.264",
    ],
    "hevc": [
        "hevc/streams/DELTAQP_B_SONY_3/DELTAQP_B_SONY_3.bit",
        "hevc/streams/ENTP_A_QUALCOMM_1/ENTP_A_Qualcomm_1.bit",
        "hevc/streams/OPFLAG_B_Qualcomm_1/OPFLAG_B_Qualcomm_1.bit",
    ],
    "vvc": [
        "vvc/streams/8b420_B_2/8b420_B_2.bit",
        "vvc/streams/8b420_A_2/8b420_A_2.bit",
        "vvc/8b420_A_2.bit",
        "vvc/streams/BUMP_A_3/BUMP_A_3.bit",
    ],
}

#: Target machines: (instance_type, region, ami_arch, on_demand, description).
TARGETS: dict[str, tuple[str, str, str, bool, str]] = {
    "graviton3": ("c7g.large", "eu-west-1", "arm64", False, "AWS Graviton3 (Neoverse V1, NEON)"),
    "graviton4": ("c8g.large", "eu-west-1", "arm64", False, "AWS Graviton4 (Neoverse V2, NEON)"),
    "zen5": ("c8a.large", "eu-west-1", "amd64", True, "AMD EPYC 9R45 (Zen 5, AVX-512)"),
}

ANCHOR_BINS = ("dav1d", "ffmpeg-decode", "vpxdec", "vvdecapp", "dav2d", "avmdec",
               "libavcodec-bench")  # decoder_speed.py times FFmpeg and dav1d in process
NODE_VERSION = "v22.21.1"
NODE_URL = f"https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-linux-x64.tar.xz"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def find_arm64_anchors(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for cand in (
        DATA / "anchors/aarch64/bin",
        DATA / "targets/gemini-decode-speed/anchors-arm64",
    ):
        if all((cand / b).exists() for b in ANCHOR_BINS):
            return cand
    raise SystemExit(
        "could not find arm64 decoder anchors (dav1d, ffmpeg-decode, vpxdec, vvdecapp); "
        "pass --arm64-anchors DIR"
    )


def build_stream_set(
    x86_bin: Path,
    streams_dir: Path,
    clip: Path,
    frames: int,
    conformance_root: Path,
    force: bool,
) -> list[dict]:
    """Build our medium/slower 720p streams and stage verified conformance streams."""
    streams_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = streams_dir / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if all((streams_dir / item["file"]).exists() for item in manifest):
            say(f"reusing {len(manifest)} streams in {streams_dir}")
            return manifest

    manifest: list[dict] = []

    # 1. Encode our own medium + slower 720p streams for each codec.
    for codec, (qp, suffix) in CODECS.items():
        for preset in ("medium", "slower"):
            fname = f"otc720p_{preset}_{codec}{suffix}"
            dst = streams_dir / fname
            say(f"encoding {fname} ({frames} frames, qp={qp})")
            subprocess.run(
                [
                    str(x86_bin), "encode",
                    "--codec", codec,
                    "--width", str(DEFAULT_CLIP_W),
                    "--height", str(DEFAULT_CLIP_H),
                    "--fps", str(DEFAULT_CLIP_FPS),
                    "--frames", str(frames),
                    "--preset", preset,
                    "--qp", str(qp),
                    "--input", str(clip),
                    "--output", str(dst),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            # Smoke-verify our decoder accepts it.
            subprocess.run(
                [str(x86_bin), "decode", "--codec", codec, "--input", str(dst), "--output", "/dev/null"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            manifest.append({
                "file": fname,
                "codec": codec,
                "group": f"ours-{preset}",
                "source": f"{clip.name} ({DEFAULT_CLIP_W}x{DEFAULT_CLIP_H}, {frames}f, {preset}, qp={qp})",
                "bytes": dst.stat().st_size,
            })

    # 2. Gather up to 2 verified conformance streams per codec.
    for codec, (_, suffix) in CODECS.items():
        added = 0
        for rel in CONFORMANCE_CANDIDATES[codec]:
            if added >= 2:
                break
            src = conformance_root / rel
            if not src.exists():
                continue
            r = subprocess.run(
                [str(x86_bin), "decode", "--codec", codec, "--input", str(src), "--output", "/dev/null"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if r.returncode != 0:
                continue
            stem = src.stem.replace(".", "_")
            fname = f"conf_{codec}_{stem}{suffix}"
            dst = streams_dir / fname
            shutil.copy2(src, dst)
            manifest.append({
                "file": fname,
                "codec": codec,
                "group": "conformance",
                "source": rel,
                "bytes": dst.stat().st_size,
            })
            added += 1

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    say(f"built stream set: {len(manifest)} streams in {streams_dir}")
    return manifest


def run_target(
    target: str,
    nc_bin: Path,
    anchors_dir: Path,
    streams_dir: Path,
    rounds: int,
    web_pkg: Path | None = None,
) -> dict:
    """Launch one rented AWS instance, run decoder_speed.py pinned to CPU 1, return record."""
    itype, region, arch, on_demand, desc = TARGETS[target]
    run_wasm = web_pkg is not None and arch == "amd64"
    with tempfile.TemporaryDirectory(prefix="nanocodec-yardstick-") as tmp:
        stage = Path(tmp) / "stage"
        (stage / "anchors").mkdir(parents=True)
        (stage / "streams").mkdir(parents=True)
        shutil.copy2(nc_bin, stage / "nanocodec")
        shutil.copy2(REPO / "tools/dev/decoder_speed.py", stage / "decoder_speed.py")
        for b in ANCHOR_BINS:
            src = anchors_dir / b
            if src.exists():
                shutil.copy2(src, stage / "anchors" / b)
        for f in streams_dir.iterdir():
            if f.is_file() and f.name != "manifest.json":
                shutil.copy2(f, stage / "streams" / f.name)
        if run_wasm:
            (stage / "tools/web").mkdir(parents=True)
            (stage / "web").mkdir(parents=True)
            shutil.copy2(REPO / "tools/web/bench-runner.mjs", stage / "tools/web/bench-runner.mjs")
            shutil.copy2(REPO / "tools/web/packets.mjs", stage / "tools/web/packets.mjs")
            shutil.copy2(REPO / "web/codecs.js", stage / "web/codecs.js")
            shutil.copytree(web_pkg, stage / "pkg")

        spec = SessionSpec(
            region=region,
            instance_type=itype,
            purpose=f"decoder-yardstick-{target}",
            lifetime_minutes=25,
            budget_usd=0.25,
            arch=arch,
            on_demand=on_demand,
        )
        with Session(spec) as s:
            s.upload(stage, "yard")
            wasm_block = ""
            if run_wasm:
                wasm_block = f"""
curl -fsSL {NODE_URL} | tar -xJ -C yard
export PATH="$PWD/yard/node-{NODE_VERSION}-linux-x64/bin:$PATH"
python3 - <<'PYWASM'
import json, subprocess
from pathlib import Path

EXT = {{"264": "h264", "hevc": "hevc", "vvc": "vvc"}}
def codec_of(name):
    parts = name.split(".")
    return parts[-2] if parts[-1] == "ivf" else EXT[parts[-1]]

res = json.load(open("yard/results.json"))
pkgs = [("wasm-simd128", "yard/pkg")]
if Path("yard/pkg/relaxed/nc_web_bg.wasm").exists():
    pkgs.append(("wasm-relaxed", "yard/pkg/relaxed"))
streams = sorted(p for p in Path("yard/streams").iterdir() if p.is_file())
for label, pkg in pkgs:
    for st in streams:
        codec = codec_of(st.name)
        r = subprocess.run(
            ["taskset", "-c", "1", "node", "--no-warnings",
             "yard/tools/web/bench-runner.mjs", "decode",
             "--pkg", pkg, "--codec", codec, "--stream", str(st),
             "--repeat", str({min(rounds, 3)})],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            continue
        j = json.loads(r.stdout.strip().splitlines()[-1])
        dec_ms = j.get("decode_ms_per_frame", j["ms_per_frame"])
        frames = j["frames"]
        res[f"{{st.name}}|{{codec}}|{{label}}"] = {{
            "t": dec_ms * frames / 1000.0,
            "frames": frames,
            "decode_fps": round(1000.0 / dec_ms, 2) if dec_ms > 0 else 0.0,
            "total_fps": j["fps"],
        }}
Path("yard/results.json").write_text(json.dumps(res, indent=1) + "\\n")
PYWASM
"""
            cmd = f"""
set -eu
chmod +x yard/nanocodec yard/anchors/*
taskset -c 1 python3 yard/decoder_speed.py {rounds} yard/results.json \\
    --nanocodec yard/nanocodec --anchors yard/anchors --streams yard/streams
{wasm_block}
python3 - <<'PY'
import json, subprocess
meta = {{
    "target": {target!r},
    "description": {desc!r},
    "instance_type": {itype!r},
    "region": {region!r},
    "arch": {arch!r},
}}
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
meta["cpu_model"] = sh("lscpu | grep -m1 'Model name:' | sed 's/.*:[[:space:]]*//'") or {desc!r}
meta["kernel"] = sh("uname -r")
meta["nanocodec_version"] = sh("yard/nanocodec --version | head -n 1")
meta["anchors"] = {{
    "dav1d": sh("yard/anchors/dav1d --version 2>&1 | head -n 1"),
    "ffmpeg": sh("yard/anchors/ffmpeg-decode -version 2>&1 | head -n 1"),
    "vvdecapp": sh("yard/anchors/vvdecapp --version 2>&1 | head -n 1"),
}}
data = json.load(open("yard/results.json"))
print(json.dumps({{"meta": meta, "results": data}}))
PY
"""
            r = s.ssh(cmd, quiet=True)
            payload = json.loads(r.stdout.strip().splitlines()[-1])
            return payload


def geomean(xs: list[float]) -> float | None:
    pos = [x for x in xs if x > 0]
    if not pos:
        return None
    return math.exp(sum(math.log(x) for x in pos) / len(pos))


def summarize(manifest: list[dict], runs: dict[str, dict]) -> dict:
    """Compute geometric mean ours/ref time ratios by codec, reference, group and target."""
    file_group = {m["file"]: ("ours-720p" if m["group"].startswith("ours-") else "conformance") for m in manifest}
    summary: dict[str, dict] = {}
    for target, run in runs.items():
        res = run["results"]
        # Collect (codec, ref, group) -> [ratio]
        buckets: dict[tuple[str, str, str], list[float]] = {}
        for key, val in res.items():
            stream, codec, dec = key.split("|")
            if dec == "ours":
                continue
            ours = res.get(f"{stream}|{codec}|ours")
            if not ours or val["t"] <= 0:
                continue
            grp = file_group.get(stream, "other")
            # For WASM entries, report wasm / ours-native (>1 means WASM is slower than native).
            if dec.startswith("wasm-"):
                ratio = val["t"] / ours["t"] if ours["t"] > 0 else 0.0
                label = f"{codec} {dec} / native"
            else:
                ratio = ours["t"] / val["t"]
                label = f"{codec} vs {dec}"
            if ratio > 0:
                buckets.setdefault((label, grp), []).append(ratio)
                buckets.setdefault((label, "all"), []).append(ratio)
        t_sum: dict[str, dict[str, float]] = {}
        for (label, grp), ratios in sorted(buckets.items()):
            gm = geomean(ratios)
            if gm is not None:
                t_sum.setdefault(label, {})[grp] = round(gm, 3)
        summary[target] = t_sum
    return summary


def render_markdown(record: dict) -> str:
    lines: list[str] = []
    targets = list(record["machines"].keys())
    lines.append(f"# Decoder Yardstick ({record['date']})")
    lines.append("")
    lines.append(f"- **Commit:** `{record['commit']}`")
    lines.append(f"- **Method:** {record['method']}")
    lines.append(f"- **Stream set:** {len(record['streams'])} streams (our `medium` + `slower` 720p encodes + conformance streams per codec)")
    lines.append("")
    lines.append("## Machines")
    lines.append("")
    lines.append("| Target | Instance | CPU | nanocodec |")
    lines.append("|---|---|---|---|")
    for t in targets:
        m = record["machines"][t]
        lines.append(f"| {t} | `{m['instance_type']}` ({m['region']}) | {m['cpu_model']} | `{m['nanocodec_version']}` |")
    lines.append("")
    lines.append("## Summary: geometric mean decode time ratio (`ours / ref`, lower is faster)")
    lines.append("")
    header = ["Codec vs Reference", "Group"] + [f"{t}" for t in targets]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    pairs = sorted({p for t in targets for p in record["summary"].get(t, {})})
    for pair in pairs:
        for grp in ("ours-720p", "conformance", "all"):
            row = [pair, grp]
            any_val = False
            for t in targets:
                v = record["summary"].get(t, {}).get(pair, {}).get(grp)
                if v is not None:
                    any_val = True
                    row.append(f"{v:.2f}x")
                else:
                    row.append("—")
            if any_val:
                lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Per-stream single-thread wall time (ms) and ratio (`ours / ref`)")
    lines.append("")
    s_header = ["Stream", "Codec", "Group", "Ref"] + [f"{t} (ours / ref)" for t in targets]
    lines.append("| " + " | ".join(s_header) + " |")
    lines.append("|" + "|".join(["---"] * len(s_header)) + "|")
    for item in record["streams"]:
        fname, codec, grp = item["file"], item["codec"], item["group"]
        refs = sorted({
            k.split("|")[2]
            for t in targets
            for k in record["results"][t]
            if k.startswith(f"{fname}|{codec}|") and not k.endswith("|ours")
        })
        for ref in refs:
            row = [f"`{fname}`", codec, grp, ref]
            for t in targets:
                res = record["results"][t]
                o = res.get(f"{fname}|{codec}|ours")
                r = res.get(f"{fname}|{codec}|{ref}")
                if o and r and r["t"] > 0 and o["t"] > 0:
                    if ref.startswith("wasm-"):
                        ratio = r["t"] / o["t"]
                        fps = r.get("decode_fps", 0.0)
                        row.append(f"{r['t']*1e3:.1f} / {o['t']*1e3:.1f} ms (**{ratio:.2f}x**, {fps:.0f} fps)")
                    else:
                        ratio = o["t"] / r["t"]
                        row.append(f"{o['t']*1e3:.1f} / {r['t']*1e3:.1f} ms (**{ratio:.2f}x**)")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--x86-bin", type=Path, required=True, help="x86_64 release nanocodec binary")
    p.add_argument("--arm64-bin", type=Path, required=True, help="aarch64 release nanocodec binary")
    p.add_argument("--web-pkg", type=Path, default=None, help="built web/pkg directory (simd128 + optional relaxed/) for node WASM decode on x86_64")
    p.add_argument("--x86-anchors", type=Path, default=DATA / "anchors/x86_64/bin")
    p.add_argument("--arm64-anchors", type=Path, default=None)
    p.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--conformance-root", type=Path, default=DATA / "conformance")
    p.add_argument("--streams-dir", type=Path, default=DATA / "work/decoder-yardstick/streams")
    p.add_argument("--rebuild-streams", action="store_true")
    p.add_argument("--targets", default="graviton3,graviton4,zen5")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--out-prefix", type=Path, required=True, help="Output prefix (writes .json and .md)")
    args = p.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        if t not in TARGETS:
            raise SystemExit(f"unknown target {t!r}; choose from {sorted(TARGETS)}")

    arm64_anchors = find_arm64_anchors(args.arm64_anchors) if any(TARGETS[t][2] == "arm64" for t in targets) else None

    manifest = build_stream_set(
        args.x86_bin, args.streams_dir, args.clip, args.frames, args.conformance_root, args.rebuild_streams
    )

    runs: dict[str, dict] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(targets)) as pool:
        futs = {}
        for t in targets:
            is_arm = TARGETS[t][2] == "arm64"
            nc_bin = args.arm64_bin if is_arm else args.x86_bin
            anc = arm64_anchors if is_arm else args.x86_anchors
            futs[pool.submit(run_target, t, nc_bin, anc, args.streams_dir, args.rounds, args.web_pkg)] = t
        for fut in concurrent.futures.as_completed(futs):
            t = futs[fut]
            runs[t] = fut.result()
            say(f"finished {t}")

    ordered_runs = {t: runs[t] for t in targets}
    record = {
        "date": datetime.date.today().isoformat(),
        "commit": git_commit(),
        "method": (
            f"Single-thread decode (`tools/dev/decoder_speed.py`, pinned to CPU 1 via `taskset -c 1`, "
            f"{args.rounds} interleaved forward/reverse rounds, best process wall time, null output / `--noblit`)."
        ),
        "rounds": args.rounds,
        "streams": manifest,
        "machines": {t: ordered_runs[t]["meta"] for t in targets},
        "results": {t: ordered_runs[t]["results"] for t in targets},
        "summary": summarize(manifest, ordered_runs),
    }

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out_prefix.with_suffix(".json")
    md_path = args.out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(record, indent=2) + "\n")
    md_path.write_text(render_markdown(record))
    say(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
