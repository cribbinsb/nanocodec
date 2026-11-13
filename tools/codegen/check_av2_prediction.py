#!/usr/bin/env python3
"""Verify the shared predictions of every subset filter against pinned original
AVM convolution (BILINEAR, EIGHTTAP_REGULAR, EIGHTTAP_SMOOTH, MULTITAP_SHARP).

Compiles only original convolve.c with configured AVM headers, in a temporary
directory. Twelve hashes cover 4/8 blocks, luma/chroma, extreme/random content,
all quarter/eighth-pel phases, positive/negative vectors and picture edges.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

PIN='966a7d7cd6fcf60360caf5dc413b2aeeb65e144d'
ROOT=Path(__file__).resolve().parents[2]
C=r'''
#include <stdio.h>
#include <stdint.h>
#include "config/av2_rtcd.h"
#include "av2/common/convolve.h"
#include "av2/common/filter.h"
static int edge(int v,int hi){return v<0?0:v>hi?hi:v;}
int main(void) {
 for(int side=4;side<=8;side+=4)for(int sub=0;sub<2;sub++)for(int pattern=0;pattern<3;pattern++) {
  uint16_t ref[24*24];uint32_t state=17;
  for(int y=0;y<24;y++)for(int x=0;x<24;x++) {
   state^=state<<13;state^=state>>17;state^=state<<5;
   ref[y*24+x]=pattern==0?(x*37+y*53+x*y*7+13)%256:pattern==1?((x+y)&1)*255:state&255;
  }
  unsigned long long hash=14695981039346656037ULL;
  const InterpFilterParams *filter=av2_get_interp_filter_params_with_block_size(FILTER,side);
  for(int pos=0;pos<3;pos++)for(int row=-18;row<=18;row+=2)for(int col=-18;col<=18;col+=2) {
   int x=pos==0?0:pos==1?8:24-side,y=x;
   int den=8<<sub,iy=row/den,ix=col/den,fy=row%den,fx=col%den;
   if(fy<0){fy+=den;iy--;}if(fx<0){fx+=den;ix--;}
   fy=fy*16/den;fx=fx*16/den;
   uint16_t window[32*32]={0},out[64]={0};
   for(int wy=0;wy<side+7;wy++)for(int wx=0;wx<side+7;wx++)
    window[wy*32+wx]=ref[edge(y+iy+wy-3,23)*24+edge(x+ix+wx-3,23)];
   const uint16_t *src=window+3*32+3;
   ConvolveParams params=get_conv_params_no_round(0,sub?1:0,NULL,0,0,8);
   if(fx&&fy)av2_highbd_convolve_2d_sr_c(src,32,out,side,side,side,filter,filter,fx,fy,&params,8);
   else if(fx)av2_highbd_convolve_x_sr_c(src,32,out,side,side,side,filter,fx,&params,8);
   else if(fy)av2_highbd_convolve_y_sr_c(src,32,out,side,side,side,filter,fy,8);
   else for(int yy=0;yy<side;yy++)for(int xx=0;xx<side;xx++)out[yy*side+xx]=src[yy*32+xx];
   for(int i=0;i<side*side;i++)hash=(hash^out[i])*1099511628211ULL;
  }
  printf("%llu,\n",hash);
 }
}
'''

# AVM filter name and the Rust fixture holding its hashes.
FILTERS=[('BILINEAR','BILINEAR_HASHES'),('EIGHTTAP_REGULAR','REGULAR_HASHES'),
         ('EIGHTTAP_SMOOTH','SMOOTH_HASHES'),('MULTITAP_SHARP','SHARP_HASHES')]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--emit',action='store_true')
    args=parser.parse_args()
    revision=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if revision!=PIN:raise SystemExit('AVM revision must be '+PIN)
    rust=(ROOT/'crates/nc-av2/src/prediction.rs').read_text()
    for avm_filter,fixture in FILTERS:
        with tempfile.TemporaryDirectory(prefix='nc-av2-prediction-') as tmp:
            path=Path(tmp);(path/'oracle.c').write_text(C)
            (path/'convolve.c').write_text(subprocess.check_output(['git','-C',str(args.source),'show',PIN+':av2/common/convolve.c'],text=True))
            subprocess.run(['cc','-O2','-std=c11','-ffunction-sections','-fdata-sections','-DFILTER='+avm_filter,
                '-I'+str(args.source),'-I'+str(args.build),str(path/'oracle.c'),str(path/'convolve.c'),
                '-Wl,--gc-sections','-o',str(path/'oracle')],check=True)
            output=subprocess.check_output([str(path/'oracle')],text=True)
        values=[int(v) for v in re.findall(r'\d+',output)]
        if args.emit:print(fixture,values);continue
        match=re.search(r'const '+fixture+r':.*?=\s*\[(.*?)\];',rust,re.S)
        if not match:raise SystemExit(f'Missing {fixture} Rust fixtures')
        expected=[int(v.replace('_','')) for v in re.findall(r"\d[\d_]*",match.group(1))]
        if values!=expected:raise SystemExit(f'{avm_filter} prediction differs: {values} != {expected}')
        print(f'AV2 original {avm_filter} convolution matches all 12 phase/content traces.')

if __name__=='__main__':main()
