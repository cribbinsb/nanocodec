#!/usr/bin/env python3
"""Differential-check native AV2 framing with pinned AVM's actual OBU parser.

Compiles only three small upstream C files in temporary storage. The synthetic
payloads exercise framing, not picture decoding or full stream conformance.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

COMMIT = "966a7d7cd6fcf60360caf5dc413b2aeeb65e144d"
ORACLE = r'''
#include <stdio.h>
#include <stdlib.h>
#include "av2/common/obu_util.h"
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *file = fopen(argv[1], "rb");
    if (!file) return 3;
    fseek(file, 0, SEEK_END); long length = ftell(file); rewind(file);
    if (length < 0 || length > 1048576) return 4;
    uint8_t *buffer = malloc((size_t)length);
    if (!buffer || fread(buffer, 1, (size_t)length, file) != (size_t)length) return 5;
    fclose(file);
    size_t offset = 0;
    while (offset < (size_t)length) {
        ObuHeader h; size_t payload = 0, consumed = 0;
        if (avm_read_obu_header_and_size(buffer + offset, (size_t)length - offset,
                                        &h, &payload, &consumed)) return 6;
        if (consumed + payload > (size_t)length - offset) return 7;
        printf("%d %d %d %d %zu\n", h.type, h.obu_tlayer_id, h.obu_mlayer_id,
               h.obu_xlayer_id, payload);
        offset += consumed + payload;
    }
    free(buffer); return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError(f"AVM oracle requires {COMMIT}, got {actual}")
    with tempfile.TemporaryDirectory(prefix="nc-av2-obu-") as directory:
        root = Path(directory)
        (root / "config").mkdir()
        # These scalar syntax files need only the C inline spelling from config.
        # No codec tools, feature gates, reference functions or headers are stubbed.
        (root / "config/avm_config.h").write_text("#define INLINE inline\n")
        (root / "oracle.c").write_text(ORACLE)
        subprocess.run(["cc", "-std=c99", "-O2", "-I", str(root), "-I", str(source),
            str(root / "oracle.c"), str(source / "av2/common/obu_util.c"),
            str(source / "avm_dsp/bitreader_buffer.c"), str(source / "avm/src/avm_integer.c"),
            "-o", str(root / "oracle")], check=True)
        native = subprocess.check_output([str(args.vectors.resolve()), str(root / "vectors.obu")])
        reference = subprocess.check_output([str(root / "oracle"), str(root / "vectors.obu")])
        assert native == reference, (native, reference)
        print(f"PASS {len(native.splitlines())} native AV2 framing vectors match final AVM OBU parser")


if __name__ == "__main__":
    main()
