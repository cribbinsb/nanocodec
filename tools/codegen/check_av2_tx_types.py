#!/usr/bin/env python3
"""Original pinned AVM oracle for the AV2 non-DCT transform types.

Compiles AVM v1.0.0's `fwd_txfm_c` / `inv_txfm_c` with the DCT, identity,
ADST (tuned DST-VII) and flipped-ADST 1-D kernels (the DDT kernels, unused
without inter DDT, stay aborting stubs; see `check_av2_dct.py`), in a
TemporaryDirectory, without a full AVM build, network access or video files.

1. `--write` regenerates `crates/nc-dsp/src/av2_txfm/adst_tables.rs` from the
   compiled `tx_kernel_adst_size{4,8,16}` / `tx_kernel_fdst_size*` tables (after the `USE_TUNED_ADST*`
   selection) and the identity scales of `inv_txfm_idtx_size*_c`; the default
   run checks that file is current.
2. Checks that each forward table equals its inverse table, so one table
   serves both directions. (The flipped tables are row-reversed ADSTs, but
   the sign of odd rows differs between sizes, so both are generated.)
3. For every transform type whose 1-D kernels exist at a size (4..32), at 15
   qindices x 8 residual patterns: hashes of the forward coefficients, our
   nearest-step levels, AVM dequantization and AVM inverse, plus clipped
   extreme levels through dequant and inverse. `--emit` prints them; the
   default compares them with `TX_TYPE_HASHES` in crates/nc-av2/src/transform.rs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

import check_av2_dct as dct
import check_av2_transform as tx

TABLES = tx.ROOT / 'crates/nc-dsp/src/av2_txfm/adst_tables.rs'
ADST_SIZES = [4, 8, 16]
IDTX_SIZES = [4, 8, 16, 32]
FAMILIES = ('dct2', 'idtx', 'adst', 'fdst')

CASES = r'''
static uint64_t hash_values(uint64_t hash,const int *v,int n) {
 for(int i=0;i<n;i++)hash=(hash^(uint32_t)v[i])*UINT64_C(1099511628211);
 return hash;
}
static const int tx_sizes[4]={TX_4X4,TX_8X8,TX_16X16,TX_32X32};
static void forward(int s,int type,const int16_t *residual,int *coeff) {
 TxfmParam p={0};p.tx_size=tx_sizes[s];p.tx_type=type;p.bd=8;
 tran_low_t out[4096];
 fwd_txfm_c(residual,out,4<<s,&p);
 for(int i=0;i<4096;i++)coeff[i]=out[i];
}
static void inverse(int s,int type,const int *coeff,int *residual) {
 TxfmParam p={0};p.tx_size=tx_sizes[s];p.tx_type=type;p.bd=8;
 tran_low_t in[4096];uint16_t dest[4096];
 for(int i=0;i<4096;i++){in[i]=coeff[i];dest[i]=RESIDUAL_BASE;}
 inv_txfm_c(in,dest,4<<s,&p);
 for(int i=0;i<4096;i++)residual[i]=(int)dest[i]-RESIDUAL_BASE;
}
static void print_table(const char *name,const int *m,int n) {
 printf("%s",name);for(int i=0;i<n*n;i++)printf(" %d",m[i]);puts("");
}
int main(void) {
 print_table("ADST4",&tx_kernel_adst_size4[INV_TXFM][0][0],4);
 print_table("ADST8",&tx_kernel_adst_size8[INV_TXFM][0][0],8);
 print_table("ADST16",&tx_kernel_adst_size16[INV_TXFM][0][0],16);
 print_table("ADST4F",&tx_kernel_adst_size4[FWD_TXFM][0][0],4);
 print_table("ADST8F",&tx_kernel_adst_size8[FWD_TXFM][0][0],8);
 print_table("ADST16F",&tx_kernel_adst_size16[FWD_TXFM][0][0],16);
 print_table("FDST4",&tx_kernel_fdst_size4[INV_TXFM][0][0],4);
 print_table("FDST8",&tx_kernel_fdst_size8[INV_TXFM][0][0],8);
 print_table("FDST16",&tx_kernel_fdst_size16[INV_TXFM][0][0],16);
 print_table("FDST4F",&tx_kernel_fdst_size4[FWD_TXFM][0][0],4);
 print_table("FDST8F",&tx_kernel_fdst_size8[FWD_TXFM][0][0],8);
 print_table("FDST16F",&tx_kernel_fdst_size16[FWD_TXFM][0][0],16);
 for(int s=0;s<4;s++) for(int type=0;type<TX_TYPES;type++) {
  const int h=g_hor_tx_type[type],v=g_ver_tx_type[type];
  if(s==3&&((h!=DCT2&&h!=IDT)||(v!=DCT2&&v!=IDT)))continue;
  const int n=4<<s,area=n*n,scale=av2_get_tx_scale(tx_sizes[s]);
  uint64_t hs[5];for(int i=0;i<5;i++)hs[i]=UINT64_C(14695981039346656037);
  for(int q=1;q<256;q+=17) {
   const int step=av2_ac_quant_QTX(q,0,0,8);
   for(int kind=0;kind<8;kind++) {
    int16_t source[4096];int coeff[4096]={0},levels[4096]={0},dq[4096]={0},recon[4096];
    uint32_t state=q*7919+kind*104729;
    for(int i=0;i<area;i++) {
     state=state*1664525U+1013904223U;
     source[i]=kind==0?0:kind==1?255:kind==2?-255:kind==3?(i&1?255:-255):
       kind==4?(i==0?255:0):kind==5?(i==area-1?-255:0):(int)(state%511)-255;
    }
    forward(s,type,source,coeff);hs[0]=hash_values(hs[0],coeff,area);
    for(int i=0;i<area;i++) {
     const int c=coeff[i];
     levels[i]=(c<0?-1:1)*(int)(((((int64_t)abs(c))<<(3+scale))+step/2)/step);
     dq[i]=dequantize_scaled(abs(levels[i]),levels[i]<0,step,scale);
    }
    hs[1]=hash_values(hs[1],levels,area);hs[2]=hash_values(hs[2],dq,area);
    inverse(s,type,dq,recon);hs[3]=hash_values(hs[3],recon,area);
   }
   int dq[4096]={0},recon[4096];
   for(int i=0;i<area;i++)dq[i]=dequantize_scaled(i%3==0?1048575:i%3==1?32767:1,i%2,step,scale);
   inverse(s,type,dq,recon);hs[4]=hash_values(hs[4],recon,area);
  }
  printf("HASH %d %d %llu,%llu,%llu,%llu,%llu\n",s,type,(unsigned long long)hs[0],
   (unsigned long long)hs[1],(unsigned long long)hs[2],(unsigned long long)hs[3],
   (unsigned long long)hs[4]);
 }
}
'''


def oracle_source(root: Path) -> str:
    enums = dct.read(root, 'av2/common/enums.h')
    tuned = ''.join(re.search(rf'#define USE_TUNED_ADST{n} .*', enums)[0] + '\n' for n in ADST_SIZES)
    # The kernel tables select their tuned variants with these switches, so
    # they must precede the tables (an undefined switch selects the old ones).
    return tuned + dct.oracle_source(root, FAMILIES, CASES)


def identity_scales(root: Path) -> list[int]:
    source = dct.read(root, 'av2/common/idct.c')
    scales = []
    for n in IDTX_SIZES:
        body = tx.function(source, f'inv_txfm_idtx_size{n}_c')
        scales.append(int(re.search(r'const int scale = (\d+);', body)[1]))
    return scales


def rust_tables(tables: dict[str, list[int]], scales: list[int]) -> str:
    lines = ['//! Generated by `tools/codegen/check_av2_tx_types.py --write`; do not edit.',
             '//! AVM v1.0.0 `tx_kernel_adst_size{4,8,16}` (tuned DST-VII, `USE_TUNED_ADST*`),',
             '//! its flipped `tx_kernel_fdst_size*`,',
             '//! and the identity scales of `inv_txfm_idtx_size*_c`. Copyright Alliance',
             '//! for Open Media; BSD-3-Clause-Clear + AOM patent terms, see `THIRD_PARTY.md`.',
             '']
    for family in ['ADST', 'FDST']:
        for n in ADST_SIZES:
            values = tables[f'{family}{n}']
            rows = ',\n'.join('    [' + ', '.join(str(v) for v in values[k * n:(k + 1) * n]) + ']'
                              for k in range(n))
            lines.append(f'/// `tx_kernel_{family.lower()}_size{n}`: row `k` is basis vector `k`.')
            lines.append('#[rustfmt::skip]')
            lines.append(f'pub(super) const {family}{n}: [[i32; {n}]; {n}] = [\n{rows},\n];')
    lines.append('/// Identity gains of 4, 8, 16 and 32 points (Q7: 128 is unity).')
    lines.append(f'pub(super) const IDENTITY_SCALES: [i32; {len(scales)}] = {scales};')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    parser.add_argument('--emit', action='store_true')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    root = args.source.resolve()
    with tempfile.TemporaryDirectory(prefix='nc-av2-txtype-oracle-') as tmp:
        cfile, binary = Path(tmp) / 'oracle.c', Path(tmp) / 'oracle'
        cfile.write_text(oracle_source(root))
        subprocess.run(['nice', '-n', '19', args.cc, '-std=c99', '-O2', str(cfile), '-o', str(binary)], check=True)
        lines = subprocess.check_output([str(binary)], text=True).splitlines()
    tables = {}
    hashes = []
    for line in lines:
        name, rest = line.split(' ', 1)
        if name == 'HASH':
            size, kind, values = rest.split()
            hashes.append([int(size), int(kind), *map(int, values.split(','))])
        else:
            tables[name] = list(map(int, rest.split()))
    for n in ADST_SIZES:
        adst, fdst = tables[f'ADST{n}'], tables[f'FDST{n}']
        assert tables[f'ADST{n}F'] == adst and tables[f'FDST{n}F'] == fdst, n
    print('PASS tables: forward == inverse')
    generated = rust_tables(tables, identity_scales(root))
    if args.write:
        TABLES.write_text(generated)
        print(f'wrote {TABLES}')
    elif TABLES.read_text() != generated:
        raise SystemExit(f'{TABLES} is stale; rerun with --write')
    else:
        print(f'PASS {TABLES.name} is current')
    if args.emit:
        print(hashes)
        return
    expected = tx.expected('TX_TYPE_HASHES')
    flat = [value for row in hashes for value in row]
    assert expected == flat, 'transform-type hashes differ from the Rust fixture'
    print(f'PASS AVM fwd_txfm_c/inv_txfm_c for {len(hashes)} (size, type) pairs: '
          '15 qindices x 8 residuals, clipped extreme levels')


if __name__ == '__main__':
    main()
