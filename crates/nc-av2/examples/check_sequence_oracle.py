#!/usr/bin/env python3
"""Compare native sequence vectors with the original pinned AVM C writer.

Uses configured AVM headers but does not require a full AVM library build.
Linker section GC retains the original sequence writer and its dependencies;
no reference functions are replaced with stubs. Scratch contains C/binaries only.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

PIN = "966a7d7cd6fcf60360caf5dc413b2aeeb65e144d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise SystemExit(f"Expected AVM {PIN}, found {revision}")
    driver = Path(__file__).with_name("sequence_oracle.c").resolve()
    with tempfile.TemporaryDirectory(prefix="nc-av2-sequence-") as directory:
        root = Path(directory)
        # Include the original translation unit so its internal sequence helpers
        # remain intact; section GC discards unrelated picture-writing routines.
        entry = root / "oracle.c"
        entry.write_text(f'#include "{source}/av2/encoder/bitstream.c"\n#include "{driver}"\n')
        binary = root / "oracle"
        command = ["nice", "-n", "19", "cc", "-O1", "-ffunction-sections", "-fdata-sections",
                   "-Wl,--gc-sections", f"-I{source}", f"-I{args.build.resolve()}", str(entry),
                   str(source / "avm_dsp/bitwriter_buffer.c"),
                   str(source / "avm/src/avm_integer.c"),
                   str(source / "av2/common/seg_common.c"), "-lm", "-o", str(binary)]
        subprocess.run(command, check=True, timeout=120)
        expected = subprocess.check_output([str(binary)], text=True, timeout=30)
        actual = subprocess.check_output([str(args.vectors.resolve())], text=True, timeout=30)
        if actual != expected:
            raise SystemExit(f"AVM/native sequence mismatch\nAVM:\n{expected}Native:\n{actual}")
        print(f"{len(actual.splitlines())} sequence payloads match original AVM {PIN} byte-for-byte")


if __name__ == "__main__":
    main()
