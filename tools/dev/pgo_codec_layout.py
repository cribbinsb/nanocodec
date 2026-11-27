#!/usr/bin/env python3
"""Link the PGO CLI with cache-aligned codec text groups on Linux."""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

CODECS = ("av2", "h264", "av1", "vp9", "hevc", "vvc")
CRATE_HASH = re.compile(r"\[[0-9a-f]{8,16}\]::")
OWNER = re.compile(r"\bnc_(" + "|".join(CODECS) + r")::")
BOUNDED_DCT = re.compile(r"\bnc_dsp::txfm::.*::forward_(16|32)_bounded\b")


def section_script(binary: Path) -> str:
    symbols = []
    table = subprocess.check_output(["nm", "-nS", "--defined-only", binary], text=True)
    for row in table.splitlines():
        fields = row.split(maxsplit=3)
        if len(fields) == 4 and fields[2] in ("t", "T") and int(fields[1], 16):
            if re.fullmatch(r"[\w.$]+", fields[3]):
                symbols.append(fields[3])
    names = subprocess.check_output(
        ["c++filt", "-s", "rust"], input="\n".join(symbols) + "\n", text=True
    ).splitlines()
    groups = {codec: [] for codec in CODECS}
    for symbol, name in zip(symbols, names, strict=True):
        name = CRATE_HASH.sub("::", name)
        if BOUNDED_DCT.search(name):
            # VP9's encoder is the only caller of the bounded raster DCT.
            groups["vp9"].append(symbol)
        elif owner := OWNER.search(name):
            groups[owner[1]].append(symbol)

    layout = ["SECTIONS {"]
    for codec in CODECS:
        if not groups[codec]:
            raise SystemExit(f"PGO codec layout: no text symbols for {codec}")
        # Leave space for small size changes without moving the next codec.
        layout.append(f"  .text.nc_{codec} ALIGN(0x10000) : {{")
        for symbol in dict.fromkeys(groups[codec]):
            layout.append(
                f"    *(.text.{symbol} .text.hot.{symbol} "
                f".text.unlikely.{symbol} .text.startup.{symbol})"
            )
        layout.append("  }")
    layout.append("} INSERT BEFORE .text;")
    return "\n".join(layout) + "\n"


def main() -> None:
    args = []
    for arg in sys.argv[1:]:
        if arg.startswith("@") and Path(arg[1:]).is_file():
            args.extend(shlex.split(Path(arg[1:]).read_text()))
        else:
            args.append(arg)
    output = args.index("-o") + 1 if "-o" in args else None
    output_name = Path(args[output]).name if output is not None else ""
    if re.fullmatch(r"nanocodec(?:-.+)?", output_name) is None:
        subprocess.run(["cc", *sys.argv[1:]], check=True)
        return

    args = [arg for arg in args if not arg.startswith("-fuse-ld=")]
    output = args.index("-o") + 1
    gcc_ld = Path(os.environ["NC_PGO_GCC_LD"])
    if (gcc_ld / "ld.lld").is_file():
        args.append(f"-B{gcc_ld}")
    elif not shutil.which("ld.lld"):
        raise SystemExit("PGO codec layout needs ld.lld in PATH or Rust's llvm-tools component")
    args.append("-fuse-ld=lld")

    with tempfile.TemporaryDirectory(prefix="nc-pgo-link-", dir=Path(args[output]).parent) as tmp:
        staging = Path(tmp)
        baseline = staging / "baseline"
        first = args.copy()
        first[output] = str(baseline)
        subprocess.run(["cc", *first], check=True)
        script = staging / "layout.ld"
        script.write_text(section_script(baseline))
        subprocess.run(["cc", *args, f"-Wl,-T,{script}"], check=True)


if __name__ == "__main__":
    main()
