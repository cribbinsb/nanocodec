#!/usr/bin/env python3
"""Our AV1 decoder against dav1d, single-threaded, per stream and per stage.

One command prints, for each stream, ours ÷ dav1d in instructions per frame
(total and by decode stage), after checking that both decoders produce the
same output (MD5 of the whole I420 output, streamed; no YUV is written).

* `--mode callgrind` (x86, `perf_event_paranoid` 4 there): callgrind self
  cost per function, each function put in a stage by name (`STAGES`). dav1d's
  hand-written assembly has no callgrind symbol, only an address: those are
  resolved against the binary's symbol table (`nm`). Self cost means inlined
  code counts where it was inlined (our symbol decode mostly lands in the
  syntax functions that call it, dav1d's msac is out of line), so read the
  syntax stages together.
* `--mode perf` (the Jetson, paranoid 2): user-space instructions and cycles
  from the hardware counters (`jperf.count`), minimum over `--rounds`,
  interleaved. No stages: use `tools/dev/pcsample.py` there.

    python3 tools/dev/av1_dec_yardstick.py --ours $CARGO_TARGET_DIR/release/nanocodec \\
        [--dav1d PATH] [--mode callgrind|perf] [--workers 4] [--stages] STREAM.ivf...

Streams are IVF; `--frames N` per stream is read from the IVF header.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from bisect import bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
DEFAULT_DAV1D = {
    "x86_64": DATA / "anchors/x86_64/bin/dav1d",
    "aarch64": DATA / "anchors/aarch64/bin/dav1d",
}
IVF_HEADER_FRAMES = slice(24, 28)
OUTPUT_CHUNK = 1 << 20

# Stages read together: our symbol decode is inlined into its callers.
SYNTAX_STAGES = ("symbols + block syntax", "coefficients", "mv stack + motion field")
# Rows below this many instructions per frame on both sides are not shown.
NEGLIGIBLE_PER_FRAME = 10_000

# Stage of a function by name, first match wins. Ours are Rust paths
# (nc_av1 / nc_dsp), dav1d's are its C functions and asm entry points.
STAGES: list[tuple[str, str]] = [
    ("film grain", r"grain|fgy_|fguv_|generate_grain"),
    ("loop restoration", r"restoration|wiener|sgr|lr_sbrow|lr_stripe|looprestoration|box_ab|box_sums"),
    ("cdef", r"cdef"),
    ("deblock", r"deblock|lpf_|loopfilter|lf_mask|mask_edges|EdgeMap"),
    ("warp", r"warp"),
    ("mc blend/mask/obmc", r"blend|w_mask|mask|obmc|w_avg|_avg_|::avg"),
    ("mc filter + setup", r"av1_mc|mc::|put_|prep_|\bmc\b|mc\.isra|emu_edge|predict_inter|predict_compound"
                          r"|predict_plane|Dsp>::predict|av1_predict|scaled|PlaneView|recon_b_inter"),
    ("inverse transform", r"inverse|txfm|inv_txfm|itx|transform_block|idct|iadst|iflipadst|iidentity"
                          r"|_internal_8bpc"),
    ("intra pred", r"intra_pred|intra_prediction|av1_intra|ipred|BlockEdges|intra_edges|pal_pred|cfl|::intra::"
                   r"|recon_b_intra"),
    ("coefficients", r"coefficients::|decode_coefs|read_coef|get_dc_sign_ctx|get_skip_ctx"),
    ("mv stack + motion field", r"mv_stack|refmvs|temporal|spatial_cand|scan_col|scan_row|load_tmvs|save_tmvs"
                                r"|motion_field|record_motion|MotionCell|add_ref_mv|refmv|splat_mv"
                                r"|extended_candidate"),
    ("symbols + block syntax", r"SymbolReader|msac|decode_b|decode_sb|FrameDecoder|tile::|read_|mvcoding|Cdfs|cdf"),
    ("memcpy/memset", r"memcpy|memmove|memset|mempcpy"),
]
OTHER = "other"


def ivf_frames(path: Path) -> int:
    with path.open("rb") as f:
        return struct.unpack("<I", f.read(32)[IVF_HEADER_FRAMES])[0]


def ours_cmd(ours: Path, stream: Path, out: str) -> list[str]:
    return [str(ours), "decode", "--codec", "av1", "--input", str(stream), "--output", out]


def dav1d_cmd(dav1d: Path, stream: Path, out: str) -> list[str]:
    muxer = ["--muxer", "null"] if out == "/dev/null" else ["--muxer", "yuv"]
    return [str(dav1d), "--threads", "1", "--framedelay", "1", *muxer, "-o", out, "-q", "-i", str(stream)]


def output_md5(argv_for: callable) -> str:
    """MD5 of a decoder's I420 output, read through a FIFO (nothing on disk)."""
    tmp = Path(tempfile.mkdtemp(prefix="av1yard-"))
    fifo = tmp / "out.yuv"
    os.mkfifo(fifo)
    try:
        proc = subprocess.Popen(argv_for(str(fifo)), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        digest = hashlib.md5()
        with fifo.open("rb") as f:
            while chunk := f.read(OUTPUT_CHUNK):
                digest.update(chunk)
        if proc.wait():
            raise SystemExit(f"decode failed: {proc.stderr.read()[-400:]!r}")
        return digest.hexdigest()
    finally:
        fifo.unlink()
        tmp.rmdir()


def symbol_table(binary: Path) -> tuple[list[int], list[str]]:
    """Sorted text-symbol addresses of `binary`, for asm entry points
    callgrind shows only as addresses."""
    out = subprocess.run(["nm", "-n", str(binary)], capture_output=True, text=True).stdout
    addrs, names = [], []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in "tT":
            addrs.append(int(parts[0], 16))
            names.append(parts[2])
    return addrs, names


def callgrind_self(argv: list[str], out_file: Path, reuse: bool) -> tuple[int, dict[str, int]]:
    if not (reuse and out_file.exists()):
        subprocess.run(["valgrind", "--tool=callgrind", f"--callgrind-out-file={out_file}", *argv],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    text = subprocess.run(["callgrind_annotate", "--threshold=100", str(out_file)],
                          capture_output=True, text=True).stdout
    total, functions = 0, {}
    for line in text.splitlines():
        m = re.match(r"\s*([\d,]+)\s+\(\s*[\d.]+%\)\s+(.*)", line)
        if not m:
            continue
        n = int(m.group(1).replace(",", ""))
        name = m.group(2)
        if "PROGRAM TOTALS" in name:
            total = n
            continue
        name = re.sub(r"\s*\[[^\]]*\]$", "", name).split(":", 1)[-1]
        functions[name] = functions.get(name, 0) + n
    return total, functions


def stage_of(name: str) -> str:
    for stage, pattern in STAGES:
        if re.search(pattern, name):
            return stage
    return OTHER


def by_stage(functions: dict[str, int], symbols: tuple[list[int], list[str]] | None) -> dict[str, int]:
    stages: dict[str, int] = {}
    for name, n in functions.items():
        if symbols and re.fullmatch(r"0x[0-9a-f]+", name):
            i = bisect_right(symbols[0], int(name, 16)) - 1
            name = symbols[1][i] if i >= 0 else name
        stage = stage_of(name)
        stages[stage] = stages.get(stage, 0) + n
    return stages


def measure_callgrind(opts, stream: Path, scratch: Path) -> dict:
    result = {}
    for who, argv, binary in (("ours", ours_cmd(opts.ours, stream, "/dev/null"), opts.ours),
                              ("dav1d", dav1d_cmd(opts.dav1d, stream, "/dev/null"), opts.dav1d)):
        out_file = scratch / f"{who}.{stream.stem}.cg"
        total, functions = callgrind_self(argv, out_file, opts.reuse)
        symbols = symbol_table(binary) if who == "dav1d" else None
        result[who] = {"total": total, "stages": by_stage(functions, symbols)}
        if opts.keep:
            result[who]["file"] = str(out_file)
        else:
            out_file.unlink()
    return result


def measure_perf(opts, stream: Path) -> dict:
    from jperf import count

    runs = {"ours": [], "dav1d": []}
    for _ in range(opts.rounds):
        runs["ours"].append(count(ours_cmd(opts.ours, stream, "/dev/null"), opts.core))
        runs["dav1d"].append(count(dav1d_cmd(opts.dav1d, stream, "/dev/null"), opts.core))
    return {who: {"total": min(i for i, _ in r), "cycles": min(c for _, c in r)} for who, r in runs.items()}


def report(stream: Path, frames: int, same: str, result: dict, show_stages: bool) -> str:
    ours, ref = result["ours"], result["dav1d"]
    lines = [f"== {stream.name} ({frames} frames, output {same})"]
    ratio = ours["total"] / ref["total"]
    line = (f"  instructions/frame  ours {ours['total'] / frames / 1e6:8.2f}M"
            f"  dav1d {ref['total'] / frames / 1e6:8.2f}M  ratio {ratio:.3f}")
    if "cycles" in ours:
        line += (f"\n  cycles/frame        ours {ours['cycles'] / frames / 1e6:8.2f}M"
                 f"  dav1d {ref['cycles'] / frames / 1e6:8.2f}M  ratio {ours['cycles'] / ref['cycles']:.3f}")
    lines.append(line)
    if show_stages and "stages" in ours:
        names = [s for s, _ in STAGES] + [OTHER]
        gaps = sorted(names, key=lambda s: ours["stages"].get(s, 0) - ref["stages"].get(s, 0), reverse=True)
        lines.append(f"  {'stage':26s} {'ours M/f':>9s} {'dav1d M/f':>9s} {'ratio':>6s} {'gap M/f':>8s}")
        for s in gaps:
            a, b = ours["stages"].get(s, 0), ref["stages"].get(s, 0)
            if max(a, b) < NEGLIGIBLE_PER_FRAME * frames:
                continue
            r = f"{a / b:6.2f}" if b else "     -"
            lines.append(f"  {s:26s} {a / frames / 1e6:9.2f} {b / frames / 1e6:9.2f} {r} {(a - b) / frames / 1e6:8.2f}")
        a = sum(ours["stages"].get(s, 0) for s in SYNTAX_STAGES)
        b = sum(ref["stages"].get(s, 0) for s in SYNTAX_STAGES)
        lines.append(f"  {'(syntax stages together)':26s} {a / frames / 1e6:9.2f} {b / frames / 1e6:9.2f}"
                     f" {a / b:6.2f} {(a - b) / frames / 1e6:8.2f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--dav1d", type=Path, default=DEFAULT_DAV1D.get(os.uname().machine))
    parser.add_argument("--mode", choices=("callgrind", "perf"), default="callgrind")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3, help="perf mode: interleaved rounds")
    parser.add_argument("--core", type=int, default=3, help="perf mode: core to pin to")
    parser.add_argument("--stages", action="store_true", help="callgrind mode: per-stage table")
    parser.add_argument("--no-hash", action="store_true", help="skip the output identity check")
    parser.add_argument("--keep", type=Path, help="callgrind mode: keep the profiles in this directory")
    parser.add_argument("--reuse", action="store_true",
                        help="with --keep: re-read profiles already there instead of measuring again")
    parser.add_argument("streams", nargs="+", type=Path)
    opts = parser.parse_args()
    opts.ours = opts.ours.resolve()
    if opts.mode == "callgrind" and not shutil.which("valgrind"):
        sys.exit("callgrind mode needs valgrind; use --mode perf")

    def one(stream: Path) -> str:
        same = "not checked"
        if not opts.no_hash:
            a = output_md5(lambda out: ours_cmd(opts.ours, stream, out))
            b = output_md5(lambda out: dav1d_cmd(opts.dav1d, stream, out))
            same = "identical to dav1d" if a == b else f"DIFFERS from dav1d ({a} vs {b})"
        if opts.mode == "perf":
            result = measure_perf(opts, stream)
        else:
            scratch = opts.keep or Path(tempfile.mkdtemp(prefix="av1yard-cg-"))
            scratch.mkdir(parents=True, exist_ok=True)
            result = measure_callgrind(opts, stream, scratch)
            if not opts.keep:
                scratch.rmdir()
        return report(stream, ivf_frames(stream), same, result, opts.stages)

    # perf mode times; it must not run streams side by side.
    workers = opts.workers if opts.mode == "callgrind" else 1
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for text in pool.map(one, opts.streams):
            print(text, flush=True)


if __name__ == "__main__":
    main()
