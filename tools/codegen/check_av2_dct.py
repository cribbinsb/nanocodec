#!/usr/bin/env python3
"""Original pinned AVM oracle for the AV2 square DCTs (4x4..64x64) and dequant.

Compiles AVM v1.0.0's own 2-D drivers `fwd_txfm_c` / `inv_txfm_c`, their 1-D
dispatchers and DCT kernels, the quantizer lookup and the decoder's dequant
statement (with its transform-size shift) in a TemporaryDirectory, without a
full AVM build, network access or video files. Two substitutions keep the
oracle to what Rust implements: the non-DCT 1-D kernels and the lossless WHT
entry points become aborting stubs (DCT_DCT never reaches them), and the
pixel add does not clip, so the raw residual is observable.

Checks:
1. the integer cosine table of `nc-dsp`'s `av2_txfm` reproduces all five AVM
   `tx_kernel_dct2_size*` tables (forward and inverse);
2. for every size and every lossy qindex, 8 residual patterns: hashes of the
   forward coefficients, our nearest-step levels, AVM dequantization and
   AVM inverse; plus clipped extreme levels through dequant and inverse;
3. dequantization of extreme levels at every qindex 0..255 for every size.

--emit prints the hashes; default compares them with the Rust fixtures in
crates/nc-av2/src/transform.rs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

import check_av2_transform as tx

SIZES = [4, 8, 16, 32, 64]
DSP_TABLE = tx.ROOT / 'crates/nc-dsp/src/av2_txfm/mod.rs'

CASES = r'''
static uint64_t hash_values(uint64_t hash,const int *v,int n) {
 for(int i=0;i<n;i++)hash=(hash^(uint32_t)v[i])*UINT64_C(1099511628211);
 return hash;
}
static const int tx_sizes[5]={TX_4X4,TX_8X8,TX_16X16,TX_32X32,TX_64X64};
static void forward(int s,const int16_t *residual,int *coeff) {
 TxfmParam p={0};p.tx_size=tx_sizes[s];p.tx_type=DCT_DCT;p.bd=8;
 tran_low_t out[4096];
 fwd_txfm_c(residual,out,4<<s,&p);
 for(int i=0;i<4096;i++)coeff[i]=out[i];
}
static void inverse(int s,const int *coeff,int *residual) {
 TxfmParam p={0};p.tx_size=tx_sizes[s];p.tx_type=DCT_DCT;p.bd=8;
 tran_low_t in[4096];uint16_t dest[4096];
 for(int i=0;i<4096;i++){in[i]=coeff[i];dest[i]=RESIDUAL_BASE;}
 inv_txfm_c(in,dest,4<<s,&p);
 for(int i=0;i<4096;i++)residual[i]=(int)dest[i]-RESIDUAL_BASE;
}
int main(void) {
 for(int s=0;s<5;s++) {
  const int n=4<<s,coded=n<32?n:32,area=coded*coded,scale=av2_get_tx_scale(tx_sizes[s]);
  uint64_t h[5];for(int i=0;i<5;i++)h[i]=UINT64_C(14695981039346656037);
  for(int q=1;q<256;q++) {
   const int step=av2_ac_quant_QTX(q,0,0,8);
   for(int kind=0;kind<8;kind++) {
    int16_t source[4096];int coeff[4096]={0},levels[4096]={0},dq[4096]={0},recon[4096];
    uint32_t state=q*7919+kind*104729;
    for(int i=0;i<n*n;i++) {
     state=state*1664525U+1013904223U;
     source[i]=kind==0?0:kind==1?255:kind==2?-255:kind==3?(i&1?255:-255):
       kind==4?(i==0?255:0):kind==5?(i==n*n-1?-255:0):(int)(state%511)-255;
    }
    forward(s,source,coeff);h[0]=hash_values(h[0],coeff,area);
    for(int i=0;i<area;i++) {
     const int v=coeff[i];
     levels[i]=(v<0?-1:1)*(int)(((((int64_t)abs(v))<<(3+scale))+step/2)/step);
     dq[i]=dequantize_scaled(abs(levels[i]),levels[i]<0,step,scale);
    }
    h[1]=hash_values(h[1],levels,area);h[2]=hash_values(h[2],dq,area);
    inverse(s,dq,recon);h[3]=hash_values(h[3],recon,n*n);
   }
   int dq[4096]={0},recon[4096];
   for(int i=0;i<area;i++)dq[i]=dequantize_scaled(i%3==0?1048575:i%3==1?32767:1,i%2,step,scale);
   inverse(s,dq,recon);h[4]=hash_values(h[4],recon,n*n);
  }
  uint64_t all=UINT64_C(14695981039346656037);
  const int extremes[8]={0,1,2,17,1023,32767,65535,1048575};
  for(int q=0;q<256;q++) for(int i=0;i<16;i++) {
   const int v=dequantize_scaled(extremes[i/2],i%2,av2_ac_quant_QTX(q,0,0,8),scale);
   all=hash_values(all,&v,1);
  }
  printf("%llu,%llu,%llu,%llu,%llu,%llu\n",(unsigned long long)h[0],(unsigned long long)h[1],
   (unsigned long long)h[2],(unsigned long long)h[3],(unsigned long long)h[4],(unsigned long long)all);
 }
}
'''

PREFIX = r'''
#include <stdbool.h>
#include <stdlib.h>
#define AVMMIN(x, y) (((x) < (y)) ? (x) : (y))
#define UENUM1BYTE(name) ; typedef uint8_t name
#define UENUM2BYTE(name) ; typedef uint16_t name
#define RESIDUAL_BASE 1024
'''

# DCT_DCT never reaches these AVM entry points; they abort if it ever does.
STUBS = r'''
void av2_highbd_iwht4x4_add(const tran_low_t *input,uint16_t *dest,int stride,int eob,int bd){abort();}
void av2_highbd_fwht4x4(const int16_t *input,tran_low_t *output,int stride){abort();}
typedef struct { TX_SIZE tx_size; TX_TYPE tx_type; int lossless; int bd; int use_ddt; int eob; } TxfmParam;
'''


def read(root: Path, path: str) -> str:
    return subprocess.check_output(['git', '-C', str(root), 'show', f'{tx.COMMIT}:{path}'], text=True)


def avm_matrices(root: Path) -> dict[int, list[list[int]]]:
    tables = read(root, 'av2/common/txb_common.c')
    matrices = {}
    for n in SIZES:
        body = tx.declaration(tables, rf'const int tx_kernel_dct2_size{n}\[TXFM_DIRECTIONS\]\[{n}\]\[{n}\] = \{{.*?\}};')
        values = tx.numbers(re.sub(r'//.*', '', body.split('=', 1)[1]))
        assert len(values) == 2 * n * n, n
        forward, inverse = values[:n * n], values[n * n:]
        assert forward == inverse, f'size {n}: forward and inverse tables differ'
        matrices[n] = [forward[k * n:(k + 1) * n] for k in range(n)]
    return matrices


def check_cosine_rule(root: Path) -> None:
    match = re.search(r'const DCT_COSINE: \[i8; MAX_POINTS\] = \[(.*?)\];', DSP_TABLE.read_text(), re.S)
    cosine = tx.numbers(match[1])
    for n, matrix in avm_matrices(root).items():
        for k in range(n):
            for j in range(n):
                phase = (2 * j + 1) * k * (64 // n) % 256
                unsigned = 256 - phase if phase > 128 else phase
                value = -cosine[128 - unsigned] if unsigned > 64 else cosine[unsigned]
                assert matrix[k][j] == value, (n, k, j, matrix[k][j], value)
    print('PASS cosine rule: nc-dsp DCT_COSINE reproduces AVM tx_kernel_dct2_size{4,8,16,32,64}')


def enum_block(source: str, first: str, name: str) -> str:
    return tx.declaration(source, r'enum \{\s*' + first + r',.*?\} UENUM\dBYTE\(' + name + r'\);')


def oracle_source(root: Path, families: tuple[str, ...] = ('dct2',), cases: str = CASES) -> str:
    """The oracle C program: AVM's 2-D drivers with the 1-D kernels of
    `families` (AVM kernel name infixes such as `dct2`, `adst`) compiled from
    the original source and every other kernel an aborting stub, then `cases`."""
    code = tx.source(root).removesuffix(tx.CASES) + PREFIX
    enums = read(root, 'av2/common/enums.h')
    for line in ['MAX_TX_SIZE_LOG2', 'MAX_TX_SIZE', 'MAX_TX_SQUARE']:
        code += re.search(r'#define ' + line + r' .*', enums)[0] + '\n'
    code += enum_block(enums, 'TX_4X4', 'TX_SIZE') + enum_block(enums, 'DCT_DCT', 'TX_TYPE')
    code += enum_block(enums, 'DCT2', 'TX1D_TYPE') + STUBS
    common = read(root, 'avm_dsp/avm_dsp_common.h')
    txfm = read(root, 'av2/common/av2_txfm.h')
    code += tx.declaration(common, r'static INLINE int64_t clamp64\(.*?\n\}')
    code += tx.function(txfm, 'clamp_value')
    code += tx.function(txfm, 'round_shift')
    for constant in ['NewSqrt2Bits', 'NewSqrt2', 'NewInvSqrt2']:
        code += re.search(r'#define ' + constant + r' .*|static const int32_t ' + constant + r' = .*', txfm)[0] + '\n'
    for constant in ['REPLACE_ADST4', 'REPLACE_ADST8', 'REPLACE_ADST16']:
        code += re.search(r'#define ' + constant + r' .*', txfm)[0] + '\n'
    data = read(root, 'av2/common/common_data.h')
    for table in ['tx_size_wide', 'tx_size_high', 'tx_size_wide_log2', 'tx_size_high_log2',
                  'tx_size_2d', 'g_hor_tx_type', 'g_ver_tx_type', 'fwd_tx_shift', 'inv_tx_shift']:
        code += tx.declaration(data, r'static const int ' + table + r'\[[^]]*\](?:\[2\])? = \{.*?\};')
    tables = read(root, 'av2/common/txb_common.c')
    for n in SIZES[1:]:
        code += tx.declaration(tables, rf'const int tx_kernel_dct2_size{n}\[TXFM_DIRECTIONS\]\[{n}\]\[{n}\] = \{{.*?\}};')
    for family in families:
        for n in SIZES:
            pattern = rf'const int tx_kernel_{family}_size{n}\[TXFM_DIRECTIONS\]\[{n}\]\[{n}\] = \{{.*?\}};'
            if family != 'dct2' and re.search(pattern, tables, re.S):
                code += tx.declaration(tables, pattern)
    inverse_source = read(root, 'av2/common/idct.c')
    forward_source = read(root, 'av2/encoder/hybrid_fwd_txfm.c')
    inverse_1d = tx.function(inverse_source, 'inv_transform_1d_c')
    forward_1d = tx.function(forward_source, 'fwd_transform_1d_c')
    for name in sorted(set(re.findall(r'inv_txfm_\w+?_size\d+_c', inverse_1d))):
        if any(name.startswith(f'inv_txfm_{family}_') for family in families):
            code += tx.function(inverse_source, name) if name != 'inv_txfm_dct2_size4_c' else ''
        else:
            code += f'void {name}(const int *s,int *d,int sh,int l,int sk,int z,const int lo,const int hi){{abort();}}\n'
    for name in sorted(set(re.findall(r'fwd_txfm_\w+?_size\d+_c', forward_1d))):
        if any(name.startswith(f'fwd_txfm_{family}_') for family in families):
            code += tx.function(forward_source, name) if name != 'fwd_txfm_dct2_size4_c' else ''
        else:
            code += f'void {name}(const int *s,int *d,int sh,int l,int sk,int z){{abort();}}\n'
    code += tx.function(inverse_source, 'clamp_buf') + inverse_1d + forward_1d
    # The raw residual is observed through a pixel add that does not clip.
    code += '#define highbd_clip_pixel_add(dest, trans, bd) ((uint16_t)((dest) + (trans)))\n'
    code += tx.function(inverse_source, 'inv_txfm_c').replace('void inv_txfm_c', 'static void inv_txfm_c')
    code += tx.function(inverse_source, 'av2_get_tx_scale')
    code += tx.function(forward_source, 'fwd_txfm_c').replace('void fwd_txfm_c', 'static void fwd_txfm_c')
    decoder = read(root, 'av2/decoder/decodetxb.c')
    begin = decoder.index('tran_low_t dq_coeff;', decoder.index('Invalid coeff level beyond 20 bits'))
    end_marker = 'tcoeffs[pos] = clamp(dq_coeff, min_value, max_value);'
    end = decoder.index(end_marker, begin) + len(end_marker)
    code += '''static int dequantize_scaled(int level,int sign,int step,int shift){
      const int32_t dequant[2]={step,step};const qm_val_t *iqmatrix=NULL;
      const int scan[1]={0};const int c=0,pos=0,min_value=-32768,max_value=32767;
      tran_low_t tcoeffs[1];\n'''
    code += decoder[begin:end] + '\nreturn tcoeffs[0];}\n'
    return code + cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    parser.add_argument('--emit', action='store_true')
    args = parser.parse_args()
    root = args.source.resolve()
    check_cosine_rule(root)
    with tempfile.TemporaryDirectory(prefix='nc-av2-dct-oracle-') as tmp:
        cfile, binary = Path(tmp) / 'oracle.c', Path(tmp) / 'oracle'
        cfile.write_text(oracle_source(root))
        subprocess.run(['nice', '-n', '19', args.cc, '-std=c99', '-O2', str(cfile), '-o', str(binary)], check=True)
        rows = subprocess.check_output([str(binary)], text=True).split()
    actual = [list(map(int, row.split(','))) for row in rows]
    if args.emit:
        print(actual)
        return
    expected = tx.expected('DCT_HASHES')
    assert expected == [value for row in actual for value in row], (expected, actual)
    print('PASS AVM fwd_txfm_c/inv_txfm_c/dequant: 5 sizes x 255 qindices x 8 residuals, '
          'clipped extreme levels, all-qindex dequant; 30 hashes')


if __name__ == '__main__':
    main()
