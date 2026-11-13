#!/usr/bin/env python3
"""Generate a VP9 entropy golden vector with pinned, unmodified libvpx C routines.

Only the small upstream writer and its normalization table are compiled, in a
TemporaryDirectory. This is an external test oracle, not linked into nanocodec.
Requires a C compiler and network access; normal Rust tests use the saved vector.
"""
from pathlib import Path
import hashlib
import re
import subprocess
import tempfile
import urllib.request

COMMIT = "1024874c5919305883187e2953de8fcb4c3d7fa6"
BASE = f"https://raw.githubusercontent.com/webmproject/libvpx/{COMMIT}/"
ROOT = Path(__file__).resolve().parents[2]


def fetch(path):
    return urllib.request.urlopen(BASE + path, timeout=30).read().decode()


def main():
    header = fetch("vpx_dsp/bitwriter.h")
    source = fetch("vpx_dsp/bitwriter.c")
    probabilities = fetch("vpx_dsp/prob.c")
    struct = header[header.index("typedef struct vpx_writer"):header.index("} vpx_writer;") + len("} vpx_writer;")]
    routines = header[header.index("static INLINE VPX_NO_UNSIGNED_SHIFT_CHECK void vpx_write("):header.index("#define vpx_write_prob")]
    termination = source[source.index("void vpx_start_encode"):]
    norm = re.search(r"const uint8_t vpx_norm\[256\] = \{.*?\};", probabilities, re.S).group()
    code = """
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <limits.h>
#define INLINE inline
#define VPX_NO_UNSIGNED_SHIFT_CHECK
#define CONFIG_BITSTREAM_DEBUG 0
""" + struct + norm + routines + termination + """
int main(void) {
    uint8_t output[100000];
    vpx_writer writer;
    vpx_start_encode(&writer, output, sizeof(output));
    uint32_t state = 0x91735aef;
    for (uint32_t i = 0; i < 20000; ++i) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        vpx_write(&writer, state & 1, (uint8_t)i);
    }
    if (vpx_stop_encode(&writer)) return 1;
    return fwrite(output, 1, writer.pos, stdout) != writer.pos;
}
"""
    with tempfile.TemporaryDirectory(prefix="nc-vpx-oracle-") as tmp:
        tmp = Path(tmp)
        (tmp / "oracle.c").write_text(code)
        subprocess.run(["cc", "-std=c99", "-O2", str(tmp / "oracle.c"), "-o", str(tmp / "oracle")], check=True)
        vector = subprocess.check_output([str(tmp / "oracle")])
    target = ROOT / "crates/nc-vp9/tests/data/libvpx-bool.bin"
    target.write_bytes(vector)
    print(f"{target.name}: {len(vector)} bytes, sha256={hashlib.sha256(vector).hexdigest()}")


if __name__ == "__main__":
    main()
