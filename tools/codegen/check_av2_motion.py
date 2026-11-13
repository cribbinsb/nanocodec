#!/usr/bin/env python3
"""Verify quarter-pel AV2 MVD bytes against pinned AVM av2_encode_mv.

Requires configured AVM headers and libavm.a. Tiny C executable and buffers live
only in TemporaryDirectory. Magnitudes use original C, signs follow the original
bitstream.c second-pass order with sign derivation disabled.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

PIN = '966a7d7cd6fcf60360caf5dc413b2aeeb65e144d'
ROOT = Path(__file__).resolve().parents[2]
C = r'''
#include <stdio.h>
#include "av2/encoder/encodemv.h"
#include "av2/common/entropy_inits_mv.h"
#include "avm_dsp/bitwriter.h"
int main(void) {
 const MV vectors[] = { VECTORS };
 nmv_context context=default_nmv_context;
 uint8_t output[4096];avm_writer writer={0};
 avm_start_encode(&writer,output);writer.allow_update_cdf=0;
 for(unsigned i=0;i<sizeof(vectors)/sizeof(vectors[0]);i++) {
   const MV diff=vectors[i], dummy={0,0};
   av2_encode_mv(NULL,dummy,&writer,&context,diff,MV_PRECISION_QTR_PEL,0);
   if(diff.row)avm_write_literal(&writer,diff.row<0,1);
   if(diff.col)avm_write_literal(&writer,diff.col<0,1);
 }
 avm_stop_encode(&writer);
 for(unsigned i=0;i<writer.pos;i++)printf("%u,",output[i]);puts("");
}
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--build',required=True,type=Path)
    parser.add_argument('--emit',action='store_true')
    parser.add_argument('--adaptive',action='store_true')
    args=parser.parse_args()
    revision=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if revision!=PIN:raise SystemExit('AVM revision must be '+PIN)
    rust=(ROOT/'crates/nc-av2/src/motion.rs').read_text()
    match=re.search(r'const VECTORS:.*?=\s*\[(.*?)\];',rust,re.S)
    if not match:raise SystemExit('Missing VECTORS Rust fixture')
    vectors=re.sub(r'\((-?\d[\d_]*),\s*(-?\d[\d_]*)\)',r'{\1,\2}',match.group(1)).replace('_','')
    with tempfile.TemporaryDirectory(prefix='nc-av2-motion-') as tmp:
        path=Path(tmp)
        (path/'oracle.c').write_text(C.replace('VECTORS',vectors).replace('allow_update_cdf=0', 'allow_update_cdf='+str(int(args.adaptive))))
        subprocess.run(['cc','-O2','-std=c11','-I'+str(args.source),'-I'+str(args.build),
            str(path/'oracle.c'),str(args.build/'libavm.a'),'-lm','-lpthread','-o',str(path/'oracle')],check=True)
        output=subprocess.check_output([str(path/'oracle')],text=True)
    values=[int(v) for v in re.findall(r'\d+',output)]
    if args.emit:print(values);return
    name='ADAPTIVE_REFERENCE_BYTES' if args.adaptive else 'REFERENCE_BYTES'
    match=re.search(r'const '+name+r':.*?=\s*&\[(.*?)\];',rust,re.S)
    if not match:raise SystemExit('Missing REFERENCE_BYTES Rust fixture')
    expected=[int(v) for v in re.findall(r'\d+',match.group(1))]
    if values!=expected:raise SystemExit(f'Motion oracle differs: {values} != {expected}')
    print(f'AV2 original motion writer matches {len(values)} bytes.')

if __name__=='__main__':main()
