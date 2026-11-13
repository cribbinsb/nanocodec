#!/usr/bin/env python3
"""Reproduce AV2 4x4 transform goldens using original pinned AVM C kernels.

Requires --source pointing to AVM v1.0.0 and a C99 compiler. --emit prints the
reference arrays; default compares them with transform.rs test fixtures. Only
small original functions are compiled in TemporaryDirectory, without full AVM
builds, network access or video files. DCT pass shifts are extracted from the
reference configuration tables. The quantization policy is our nearest-step
encoder choice; dequantization arithmetic is extracted from AVM's decoder.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

COMMIT = '966a7d7cd6fcf60360caf5dc413b2aeeb65e144d'
ROOT = Path(__file__).resolve().parents[2]


def function(source: str, name: str) -> str:
    pattern = r'(?:static )?(?:INLINE )?(?:void|int|int32_t|uint16_t) ' + name + r'\('
    match = re.search(pattern, source)
    if match is None:
        raise ValueError(f'missing reference function {name}')
    begin = match.start()
    end = source.index('{', begin) + 1
    depth = 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end] + '\n'


def declaration(source: str, pattern: str) -> str:
    match = re.search(pattern, source, re.S)
    if match is None:
        raise ValueError(f'missing reference declaration {pattern}')
    return match.group() + '\n'


def numbers(text: str) -> list[int]:
    return [int(value.replace('_', ''), 0) for value in re.findall(r'-?(?:0x[0-9a-fA-F_]+|\d[\d_]*)', text)]


def expected(name: str) -> list[int]:
    source = (ROOT / 'crates/nc-av2/src/transform.rs').read_text()
    match = re.search(r'const ' + name + r'\s*:[^=]+?=\s*\[', source)
    if match is None:
        raise ValueError(f'missing Rust fixture {name}')
    begin, depth = match.end(), 1
    end = begin
    while depth:
        depth += (source[end] == '[') - (source[end] == ']')
        end += 1
    return numbers(source[begin:end-1])


PREFIX = r'''
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#define INLINE inline
#define CONFIG_COEFFICIENT_RANGE_CHECKING 0
#define DO_RANGE_CHECK_CLAMP 0
#define CONFIG_INSPECTION 0
#define UNIT_QUANT_SHIFT 3
#define UNIT_QUANT_FACTOR 8
#define QUANT_TABLE_BITS 3
#define AVM_QM_BITS 5
#define FWD_TXFM 0
#define INV_TXFM 1
#define TXFM_DIRECTIONS 2
#define AVM_BITS_8 8
#define AVM_BITS_10 10
#define MAXQ_8_BITS 255
#define MAXQ_10_BITS 303
#define MAXQ 351
#define ROUND_POWER_OF_TWO_64(v,n) (((v)+(INT64_C(1)<<((n)-1)))>>(n))
typedef int avm_bit_depth_t;
typedef int32_t tran_low_t;
typedef int64_t tran_high_t;
typedef uint8_t qm_val_t;
'''

CASES = r'''
static void emit(const char *name,const int32_t *v,int n){
 printf("%s: ",name);for(int i=0;i<n;i++)printf("%d,",v[i]);puts("");
}
int main(void){
 const int16_t residual[16]={17,-23,55,-101,37,0,-7,85,-92,41,5,-64,3,72,-18,29};
 int32_t source[16],tmp[16],dct[16],wht[16];
 for(int i=0;i<16;i++)source[i]=residual[i];
 av2_fwht4x4_c(residual,wht,4);emit("WHT_FORWARD",wht,16);
 fwd_txfm_dct2_size4_c(source,tmp,fwd_shift4[0],4,0,0);
 fwd_txfm_dct2_size4_c(tmp,dct,fwd_shift4[1],4,0,0);emit("DCT_FORWARD",dct,16);
 const int qindices[7]={0,1,2,24,25,100,255};
 for(int q=0;q<7;q++){
  int qindex=qindices[q],step=av2_ac_quant_QTX(qindex,0,0,8);
  int32_t levels[16],dq[16],inverse[16];
  const int32_t *raw=qindex?dct:wht;
  for(int i=0;i<16;i++){
   int sign=raw[i]<0;int mag=sign?-raw[i]:raw[i];
   levels[i]=(mag*8+step/2)/step;if(sign)levels[i]=-levels[i];
   dq[i]=dequantize_level(levels[i]<0?-levels[i]:levels[i],levels[i]<0,step);
  }
  if(qindex){
   inv_txfm_dct2_size4_c(dq,tmp,inv_shift4[0],4,0,0,-32768,32767);
   inv_txfm_dct2_size4_c(tmp,inverse,inv_shift4[1],4,0,0,-256,255);
  }else{
   uint16_t output[16];for(int i=0;i<16;i++)output[i]=128;
   av2_highbd_iwht4x4_16_add_c(dq,output,4,8);
   for(int i=0;i<16;i++)inverse[i]=(int)output[i]-128;
  }
  emit("LEVELS",levels,16);emit("DEQUANT",dq,16);emit("RECON",inverse,16);
 }
 const int extreme[16]={0,1,-1,2,-2,17,-17,1023,-1023,32767,-32767,65535,-65535,1048575,-1048575,123456};
 for(int q=0;q<3;q++){
  int qindex=q==0?0:q==1?2:255, step=av2_ac_quant_QTX(qindex,0,0,8),dq[16];
  for(int i=0;i<16;i++)dq[i]=dequantize_level(extreme[i]<0?-extreme[i]:extreme[i],extreme[i]<0,step);
  emit("EXTREME_DEQUANT",dq,16);
 }
 uint64_t hash=UINT64_C(14695981039346656037);
 for(int q=0;q<256;q++){
  uint32_t step=av2_ac_quant_QTX(q,0,0,8);assert(step==(uint32_t)av2_dc_quant_QTX(q,0,0,8));
  for(int byte=0;byte<4;byte++)hash=(hash^((step>>(8*byte))&255))*UINT64_C(1099511628211);
 }
 printf("QUANT_HASH: 0x%016llx\n",(unsigned long long)hash);
}
'''


def source(root: Path) -> str:
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        raise ValueError(f'requires AVM {COMMIT}, found {head}')
    def read(path):
        return subprocess.check_output(['git', '-C', str(root), 'show', f'{COMMIT}:{path}'], text=True)
    common = read('avm_dsp/avm_dsp_common.h')
    txfm = read('av2/common/av2_txfm.h')
    fwd = read('av2/encoder/hybrid_fwd_txfm.c')
    quant = read('av2/common/quant_common.c')
    configuration = read('av2/common/common_data.h')
    code = PREFIX
    code += function(common, 'clamp') + function(common, 'clip_pixel_highbd')
    code += function(txfm, 'range_check_value') + function(txfm, 'highbd_clip_pixel_add')
    code += declaration(read('av2/common/txb_common.c'),
                        r'const int tx_kernel_dct2_size4\[TXFM_DIRECTIONS\]\[4\]\[4\] = \{.*?\};')
    for direction in ['fwd', 'inv']:
        shifts = re.search(direction + r'_tx_shift\[TX_SIZES_ALL\]\[2\] = \{\s*\{([^}]+)\}', configuration)
        if shifts is None:
            raise ValueError('reference transform shifts missing')
        code += f'const int {direction}_shift4[2] = {{{shifts[1]}}};\n'
    code += function(fwd, 'fwd_txfm_dct2_size4_c') + function(fwd, 'av2_fwht4x4_c')
    code += function(read('av2/common/idct.c'), 'inv_txfm_dct2_size4_c')
    code += function(read('av2/common/av2_inv_txfm2d.c'), 'av2_highbd_iwht4x4_16_add_c')
    code += declaration(quant, r'static const uint16_t ac_qlookup_QTX\[25\] = \{.*?\};')
    for name in ['qlookup', 'get_q', 'av2_dc_quant_QTX', 'av2_ac_quant_QTX']:
        code += function(quant, name)
    decoder = read('av2/decoder/decodetxb.c')
    code += function(decoder, 'get_dqv')
    begin = decoder.index('tran_low_t dq_coeff;', decoder.index('Invalid coeff level beyond 20 bits'))
    end = decoder.index('tcoeffs[pos] = clamp(dq_coeff, min_value, max_value);', begin)
    end += len('tcoeffs[pos] = clamp(dq_coeff, min_value, max_value);')
    code += '''static int dequantize_level(int level,int sign,int step){
      const int32_t dequant[2]={step,step};const qm_val_t *iqmatrix=NULL;
      const int scan[1]={0};const int c=0,pos=0,shift=0,min_value=-32768,max_value=32767;
      tran_low_t tcoeffs[1];\n'''
    code += decoder[begin:end] + '\nreturn tcoeffs[0];}\n'
    return code + CASES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    parser.add_argument('--emit', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='nc-av2-transform-oracle-') as tmp:
        root = Path(tmp)
        cfile, binary = root / 'oracle.c', root / 'oracle'
        cfile.write_text(source(args.source.resolve()))
        subprocess.run(['nice', '-n', '19', args.cc, '-std=c99', '-O2', str(cfile), '-o', str(binary)], check=True)
        rows = subprocess.check_output([str(binary)], text=True)
    if args.emit:
        print(rows, end='')
        return
    results: dict[str, list[int]] = {}
    for row in rows.splitlines():
        key, values = row.split(':')
        results.setdefault(key, []).extend(numbers(values))
    for name, actual in results.items():
        if actual != expected(name):
            raise ValueError(f'{name}: original AVM differs from Rust golden')
        print(f'PASS {name}: {len(actual)} values')


if __name__ == '__main__':
    main()
