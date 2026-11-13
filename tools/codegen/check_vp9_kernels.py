#!/usr/bin/env python3
"""Verify VP9 Rust test goldens against original pinned libvpx C routines.

Usage: python3 tools/codegen/check_vp9_kernels.py --source /path/to/libvpx-v1.16.0

Requires git and a C99 compiler (CC or --cc). No network, full anchor build,
Rust build, retained video or generated repository files. Reference functions
are extracted unchanged from the exact git commit; small C harnesses supply
explicit type/configuration declarations and input fixtures. The transform
oracle uses libvpx's high-bit-depth-build coefficient types with its 8-bit
transform functions, as the installed reference does. Expected outputs are
read directly from Rust tests, so there is no second copy of golden vectors to
drift. Larger transforms, prediction and the loop filter are checked end to end:
every evaluated stream decodes bit-exactly in vpxdec and FFmpeg.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

COMMIT = "1024874c5919305883187e2953de8fcb4c3d7fa6"
ROOT = Path(__file__).resolve().parents[2]
COMMON = """#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#define INLINE inline
"""


def section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


def function(source: str, name: str) -> str:
    """Extract one unchanged function, with its original declaration."""
    match = re.search(r"(?:static )?(?:INLINE )?(?:void|int8_t|tran_high_t) "
                      + re.escape(name) + r"\(", source)
    if match is None:
        raise ValueError(f"reference function missing: {name}")
    begin = match.start()
    end = source.index("{", begin) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end] + "\n"


def declaration(source: str, pattern: str) -> str:
    match = re.search(pattern, source, re.S)
    if match is None:
        raise ValueError(f"reference declaration missing: {pattern}")
    return match.group() + "\n"


class Reference:
    def __init__(self, root: Path):
        self.root = root
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if head != COMMIT:
            raise ValueError(f"requires libvpx {COMMIT}, found {head}")
        self.cache: dict[str, str] = {}

    def read(self, path: str) -> str:
        # Read committed bytes even if a caller's checkout has local changes.
        if path not in self.cache:
            self.cache[path] = subprocess.check_output(
                ["git", "-C", str(self.root), "show", f"{COMMIT}:{path}"], text=True)
        return self.cache[path]


def numbers(text: str) -> list[int]:
    return [int(value.replace("_", ""), 0) if "x" in value else int(value.replace("_", ""))
            for value in re.findall(r"-?(?:0x[0-9a-fA-F_]+|[0-9][0-9_]*)", text)]


def rust_array(module: str, name: str) -> list[int]:
    source = (ROOT / "crates/nc-vp9/src" / module).read_text()
    assignment = re.search(r"(?:let|const) " + name + r"(?:\s*:[^=]+)?\s*=\s*\[", source)
    if assignment is None:
        raise ValueError(f"Rust golden missing: {module}:{name}")
    begin = assignment.end()
    end, depth = begin, 1
    while depth:
        depth += (source[end] == "[") - (source[end] == "]")
        end += 1
    return numbers(source[begin:end-1])


def run(code: str, cc: str, directory: Path, name: str) -> bytes:
    source, binary = directory / f"{name}.c", directory / name
    source.write_text(COMMON + code)
    subprocess.run(["nice", "-n", "19", cc, "-std=c99", "-O2", str(source), "-o", str(binary)], check=True)
    return subprocess.check_output([str(binary)])


def require_equal(name: str, actual: list[int], expected: list[int]) -> None:
    if actual != expected:
        mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                        min(len(actual), len(expected)))
        raise ValueError(f"{name}: reference differs at item {mismatch}; "
                         f"lengths reference={len(actual)} Rust={len(expected)}")
    print(f"PASS {name}: {len(expected)} golden values")

MOTION_PREFIX = r"""#define VPX_NO_UNSIGNED_SHIFT_CHECK
#define CONFIG_BITSTREAM_DEBUG 0
typedef uint8_t vpx_prob;
typedef int8_t vpx_tree_index;
#define TREE_SIZE(x) (2*((x)-1))
"""

MOTION_CASES = r"""
int main(void) {
 uint8_t output[10000]; vpx_writer w; vpx_start_encode(&w,output,sizeof(output)); vp9_entropy_mv_init();
 const int values[]={0,2,-2,6,-8,14,16,18,-32,64,128,256,512,1024,2048,4096,8192,16382};
 for (unsigned i=0;i<sizeof(values)/sizeof(values[0]);++i) {
  const int row=values[i], col=(i%3)?-values[i]:0;
  const int joint=(row?2:0)+(col?1:0);
  vp9_write_token(&w,vp9_mv_joint_tree,default_nmv_context.joints,&mv_joint_encodings[joint]);
  if(row) encode_mv_component(&w,row,&default_nmv_context.comps[0],0);
  if(col) encode_mv_component(&w,col,&default_nmv_context.comps[1],0);
 }
 if(vpx_stop_encode(&w))return 1;
 return fwrite(output,1,w.pos,stdout)!=w.pos;
}
"""

HYBRID_PREFIX = r"""typedef int32_t tran_low_t;
typedef int64_t tran_high_t;
typedef int16_t tran_coef_t;
typedef void (*transform_1d)(const tran_low_t *,tran_low_t *);
typedef struct {transform_1d cols,rows;} transform_2d;
#define DCT_DCT 0
#define ROUND_POWER_OF_TWO(value,n) (((value)+(1<<((n)-1)))>>(n))
#define WRAPLOW(x) ((int16_t)(x))
#define fdct_round_shift(x) ROUND_POWER_OF_TWO(x,14)
#define dct_const_round_shift(x) ROUND_POWER_OF_TWO(x,14)
static uint8_t clip_pixel_add(uint8_t a,int32_t b){int v=a+b;return v<0?0:v>255?255:v;}
"""

HYBRID_CASES = r"""int main(void){
 const int16_t input[16]={17,-23,55,-101,37,0,-7,85,-92,41,5,-64,3,72,-18,29};
 const tran_low_t coeff[16]={45,-31,17,-9,83,-22,6,33,-66,14,-20,7,11,-3,24,-40};
 for(int type=0;type<4;type++){
  tran_low_t out[16];vp9_fht4x4_c(input,out,4,type);
  for(int i=0;i<16;i++) printf("%d,",out[i]);puts("");
  uint8_t pixels[16];memset(pixels,128,16);vp9_iht4x4_16_add_c(coeff,pixels,4,type);
  for(int i=0;i<16;i++) printf("%d,",(int)pixels[i]-128);puts("");
 }
}"""

def motion_source(ref: Reference) -> str:
    header = ref.read('vpx_dsp/bitwriter.h')
    entropy = ref.read('vp9/common/vp9_entropymv.h')
    tree_header = ref.read('vp9/encoder/vp9_treewriter.h')
    code = MOTION_PREFIX
    code += section(header, 'typedef struct vpx_writer', '} vpx_writer;') + '} vpx_writer;\n'
    code += declaration(ref.read('vpx_dsp/prob.c'), r'const uint8_t vpx_norm\[256\] = \{.*?\};')
    code += section(header, 'static INLINE VPX_NO_UNSIGNED_SHIFT_CHECK void vpx_write(', '#define vpx_write_prob')
    source = ref.read('vpx_dsp/bitwriter.c')
    code += source[source.index('void vpx_start_encode'):]
    code += section(entropy, '#define MV_JOINTS', 'static INLINE int mv_joint_vertical')
    code += section(entropy, '#define MV_CLASSES', 'extern const vpx_tree_index')
    code += section(entropy, 'typedef struct {', '} nmv_context;') + '} nmv_context;\n'
    code += section(ref.read('vp9/common/vp9_entropymv.c'),
                    'const vpx_tree_index vp9_mv_joint_tree', 'static void inc_mv_component')
    code += section(tree_header, 'struct vp9_token', '#ifdef __cplusplus')
    code += section(ref.read('vp9/encoder/vp9_treewriter.c'),
                    'static void tree2tok', 'static unsigned int convert_distribution')
    code += section(ref.read('vp9/encoder/vp9_encodemv.c'),
                    'static struct vp9_token', 'static void build_nmv_component_cost_table')
    return code + MOTION_CASES


def hybrid_source(ref: Reference) -> str:
    forward = ref.read('vp9/encoder/vp9_dct.c')
    inverse = ref.read('vpx_dsp/inv_txfm.c')
    code = HYBRID_PREFIX
    for name in ['cospi_8_64', 'cospi_16_64', 'cospi_24_64',
                 'sinpi_1_9', 'sinpi_2_9', 'sinpi_3_9', 'sinpi_4_9']:
        code += declaration(ref.read('vpx_dsp/txfm_common.h'),
                            r'static const tran_coef_t ' + name + r' = .*?;')
    code += function(forward, 'fdct4') + function(forward, 'fadst4')
    code += function(ref.read('vpx_dsp/fwd_txfm.c'), 'vpx_fdct4x4_c')
    code += declaration(forward, r'static const transform_2d FHT_4\[\] = \{.*?\};')
    code += function(forward, 'vp9_fht4x4_c')
    code += function(inverse, 'iadst4_c') + function(inverse, 'idct4_c')
    code += function(ref.read('vp9/common/vp9_idct.c'), 'vp9_iht4x4_16_add_c')
    return code + HYBRID_CASES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'), help='C99 compiler executable')
    args = parser.parse_args()
    ref = Reference(args.source.resolve())
    with tempfile.TemporaryDirectory(prefix='nc-vp9-kernel-oracles-') as temporary:
        directory = Path(temporary)
        actual = run(motion_source(ref), args.cc, directory, 'motion')
        require_equal('motion', list(actual), rust_array('motion.rs', 'expected'))

        rows = run(hybrid_source(ref), args.cc, directory, 'hybrid').decode().splitlines()
        require_equal('hybrid forward', numbers('\n'.join(rows[::2])),
                      rust_array('hybrid.rs', 'forward_expected'))
        require_equal('hybrid inverse', numbers('\n'.join(rows[1::2])),
                      rust_array('hybrid.rs', 'inverse_expected'))

    print('All original-C goldens match. Run cargo test -p nc-vp9 to verify Rust against them.')


if __name__ == '__main__':
    main()
