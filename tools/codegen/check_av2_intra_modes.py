#!/usr/bin/env python3
"""Verify basic AV2 intra mode traces using unchanged pinned AVM functions."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

PIN = '966a7d7cd6fcf60360caf5dc413b2aeeb65e144d'
ROOT = Path(__file__).resolve().parents[2]
PREFIX = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "av2/common/av2_common_int.h"
#include "av2/common/entropy_inits_modes.h"
#include "av2/common/reconintra.h"
#include "av2/common/txb_common.h"
typedef struct { unsigned long long hash; int count; } probe_writer;
#define avm_writer probe_writer
static void avm_write_symbol(probe_writer *w,int symbol,const avm_cdf_prob *cdf,int n) {
 w->hash=(w->hash ^ symbol)*1099511628211ULL;
 for(int i=0;i<n;i++)w->hash=(w->hash ^ cdf[i])*1099511628211ULL;
 w->count++;
}
static void avm_write_literal(probe_writer *w,int value,int bits) {
 (void)w;(void)value;(void)bits;abort(); /* no offsets for supported modes */
}
'''
PIXEL_CASES = r'''
static uint16_t sample(int x,int y,int seed){return (x*19+y*37+x*y*3+seed*13)&255;}
static unsigned long long pixel_hash(void){
 unsigned long long hash=14695981039346656037ULL;
 const int positions[][2]={{0,0},{4,0},{0,4},{4,4},{4,64}};
 for(int seed=0;seed<32;seed++)for(int pos=0;pos<5;pos++)for(int mode=0;mode<4;mode++){
  const int x=positions[pos][0],y=positions[pos][1];
  uint16_t above_data[5],left[4],dst[16];uint16_t *above=above_data+1;
  for(int i=0;i<4;i++){above[i]=y?sample(x+i,y-1,seed):127;left[i]=x?sample(x-1,y+i,seed):129;}
  if(!y && x)for(int i=0;i<4;i++)above[i]=left[0];
  if(!x && y)for(int i=0;i<4;i++)left[i]=above[0];
  above[-1]=x&&y?sample(x-1,y-1,seed):x?left[0]:y?above[0]:128;
  if(mode==0){
   if(x&&y)highbd_dc_predictor(dst,4,4,4,above,left,8);
   else if(x)highbd_dc_left_predictor(dst,4,4,4,above,left,8);
   else if(y)highbd_dc_top_predictor(dst,4,4,4,above,left,8);
   else highbd_dc_128_predictor(dst,4,4,4,above,left,8);
  }else if(mode==1)highbd_v_predictor(dst,4,4,4,above,left,8);
  else if(mode==2)highbd_h_predictor(dst,4,4,4,above,left,8);
  else highbd_paeth_predictor(dst,4,4,4,above,left,8);
  for(int i=0;i<16;i++)hash=(hash^dst[i])*1099511628211ULL;
 }
 return hash;
}
'''

SUFFIX = r'''
int main(void) {
 FRAME_CONTEXT *fc=calloc(1,sizeof(*fc));MACROBLOCKD *xd=calloc(1,sizeof(*xd));
 MB_MODE_INFO mb={0},above={0},left={0}; MB_MODE_INFO *mi=&mb; xd->mi=&mi;xd->tile_ctx=fc;
 mb.sb_type[0]=above.sb_type[0]=left.sb_type[0]=BLOCK_8X8;
 mb.ref_frame[0]=above.ref_frame[0]=left.ref_frame[0]=INTRA_FRAME;
 memcpy(fc->y_mode_set_cdf,default_y_mode_set_cdf,sizeof(fc->y_mode_set_cdf));
 memcpy(fc->y_mode_idx_cdf,default_y_mode_idx_cdf,sizeof(fc->y_mode_idx_cdf));
 memcpy(fc->y_mode_idx_offset_cdf,default_y_mode_idx_offset_cdf,sizeof(fc->y_mode_idx_offset_cdf));
 memcpy(fc->uv_mode_cdf,default_uv_mode_cdf,sizeof(fc->uv_mode_cdf));
 memcpy(fc->dpcm_cdf,default_dpcm_cdf,sizeof(fc->dpcm_cdf));
 memcpy(fc->dpcm_uv_cdf,default_dpcm_uv_cdf,sizeof(fc->dpcm_uv_cdf));
 const int modes[]={DC_PRED,V_PRED,H_PRED,PAETH_PRED},joint[]={0,22,50,4};
 probe_writer w={14695981039346656037ULL,0};
 for(int lossless=0;lossless<2;lossless++)for(int a=-1;a<4;a++)for(int l=-1;l<4;l++)
 for(int y=0;y<4;y++)for(int uv=0;uv<4;uv++) {
  xd->above_right_mbmi=a<0?NULL:&above;xd->bottom_left_mbmi=l<0?NULL:&left;
  if(a>=0)above.joint_y_mode_delta_angle=joint[a];
  if(l>=0)left.joint_y_mode_delta_angle=joint[l];
  mb.mode=modes[y];mb.joint_y_mode_delta_angle=joint[y];mb.uv_mode=modes[uv];
  get_y_intra_mode_set(&mb,xd);get_uv_intra_mode_set(&mb);
  mb.y_mode_idx=mb.uv_mode_idx=255;
  for(int i=0;i<LUMA_MODE_COUNT;i++)if(mb.y_intra_mode_list[i]==joint[y])mb.y_mode_idx=i;
  for(int i=0;i<UV_INTRA_MODES-1;i++)if(mb.uv_intra_mode_list[i]==modes[uv])mb.uv_mode_idx=i;
  assert(mb.y_mode_idx<7 && mb.uv_mode_idx<7);
  if(lossless)write_dpcm_index(fc,0,&w);
  write_intra_luma_mode(xd,&w);
  if(lossless)write_dpcm_uv_index(fc,0,&w);
  write_intra_uv_mode(xd,0,&w);
 }
 AV2_COMMON *cm=calloc(1,sizeof(*cm));cm->features.reduced_tx_set_used=2;
 probe_writer tx={14695981039346656037ULL,0};
 for(int lossless=0;lossless<2;lossless++)for(int mode=0;mode<4;mode++)
 for(int plane=0;plane<3;plane++)for(int eob=1;eob<=16;eob++)for(int skip=0;skip<2;skip++){
  xd->lossless[0]=lossless;mb.mode=modes[mode];mb.segment_id=0;
  av2_write_tx_type(cm,xd,DCT_DCT,TX_4X4,&tx,plane,eob,skip);
 }
 assert(tx.count==0); /* reduced set2 infers DCT for every intra mode */
 printf("%llu %d\n%llu\n",w.hash,w.count,pixel_hash());free(cm);free(xd);free(fc);
}
'''

def extract(source, name):
    match=re.search(r'(?:static (?:AVM_INLINE|INLINE) )?(?:void\s*\*?|int|uint16_t|PREDICTION_MODE)\s*'+name+r'\(',source)
    if not match: raise ValueError(name)
    end=source.index('{',match.start())+1
    depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[match.start():end]+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--emit',action='store_true')
    args=parser.parse_args()
    def read(path):
        return subprocess.check_output(['git','-C',str(args.source),'show',PIN+':'+path],text=True)
    if subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()!=PIN:
        raise SystemExit('Wrong AVM revision')
    bits=read('av2/encoder/bitstream.c');recon=read('av2/common/reconintra.c')
    code=PREFIX+extract(read('av2/common/blockd.c'),'av2_get_joint_mode')
    for name in ['get_y_mode_idx_ctx','get_y_intra_mode_set','get_uv_intra_mode_set','get_cfl_ctx']:
        code+=extract(recon,name)
    for name in ['write_dpcm_index','write_dpcm_uv_index','write_intra_luma_mode','write_intra_uv_mode','av2_write_tx_type']:
        code+=extract(bits,name)
    prediction=read('avm_dsp/intrapred.c')
    code += '\n#include "avm_mem/avm_mem.h"\n'
    code += extract(read('avm_mem/avm_mem.c'),'avm_memset16')
    for name in ['abs_diff','paeth_predictor_single','highbd_v_predictor','highbd_h_predictor',
                 'highbd_paeth_predictor','highbd_dc_predictor','highbd_dc_left_predictor',
                 'highbd_dc_top_predictor','highbd_dc_128_predictor']:
        code += extract(prediction,name)
    code += PIXEL_CASES
    with tempfile.TemporaryDirectory(prefix='nc-av2-intra-modes-') as tmp:
        path=Path(tmp);(path/'oracle.c').write_text(code+SUFFIX)
        subprocess.run(['nice','-n','19','cc','-O2','-std=c11','-I'+str(args.source),'-I'+str(args.build),str(path/'oracle.c'),'-o',str(path/'oracle')],check=True)
        output=subprocess.check_output([str(path/'oracle')],text=True)
    if args.emit: print(output,end='');return
    rust=(ROOT/'crates/nc-av2/src/intra_modes.rs').read_text()
    match=re.search(r'const REFERENCE_TRACE:.*?=\s*\((.*?)\);',rust,re.S)
    expected=[int(v.replace('_','')) for v in re.findall(r'\d[\d_]*',match[1])]
    pixel=re.search(r'const REFERENCE_PIXELS: u64 = ([\d_]+);',rust)
    expected.append(int(pixel[1].replace('_','')))
    if [int(v) for v in output.split()]!=expected:raise SystemExit('Original AVM mode trace differs: '+output)
    print('Original AVM mode traces match all 800 neighbor/mode/lossless combinations and 640 prediction fixtures; intra TX4 emits no mode-dependent symbols.')

if __name__=='__main__':main()
