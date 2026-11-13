#!/usr/bin/env python3
"""Check uniform8x8 MV references against original AVM find_mv_refs/mode_ctx.

Links the existing pinned AVM library, visits actual binary partition order,
and records compact candidate/context hashes. No raw pixels or retained files.
Two list configurations: the basic two-entry list (first two entries, as
REFERENCE_HASHES) and six entries with the reference MV bank (the whole
list, as FULL_LIST_HASHES; the bank is reset and loaded per superblock as
in AVM decode_tile).
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
#include <stdlib.h>
#include <string.h>
#include "av2/common/av2_common_int.h"
#include "av2/common/mvref_common.h"
static AV2_COMMON *cm;
static MACROBLOCKD *xd;
static MB_MODE_INFO **grid;
static SUBMB_INFO **subgrid;
static unsigned long long hash;
static int step,pattern,full;
static void record(int value) {hash=(hash^(unsigned int)value)*1099511628211ULL;}
static void leaf(int x,int y) {
 const int stride=cm->mi_params.mi_cols;
 xd->mi_row=y/4;xd->mi_col=x/4;xd->mi=&grid[(y/4)*stride+x/4];
 xd->submi=&subgrid[(y/4)*stride+x/4];
 xd->left_available=x>0;xd->up_available=y>0;
 xd->mb_to_left_edge=-x*8;xd->mb_to_top_edge=-y*8;
 xd->mb_to_right_edge=(cm->width-x-8)*8;xd->mb_to_bottom_edge=(cm->height-y-8)*8;
 MB_MODE_INFO *mi=xd->mi[0];mi->sb_type[0]=BLOCK_8X8;mi->sb_type[1]=BLOCK_8X8;
 mi->ref_frame[0]=0;mi->ref_frame[1]=NONE_FRAME;mi->mode=NEWMV;
 mi->motion_mode=SIMPLE_TRANSLATION;
 uint8_t count[MODE_CTX_REF_FRAMES]={0};
 CANDIDATE_MV stack[MODE_CTX_REF_FRAMES][MAX_REF_MV_STACK_SIZE]={0};
 uint16_t weights[MODE_CTX_REF_FRAMES][MAX_REF_MV_STACK_SIZE]={0};
 int16_t context[MODE_CTX_REF_FRAMES]={0};
 av2_find_mode_ctx(cm,xd,context,0);
 av2_find_mv_refs(cm,xd,mi,0,count,stack,weights,NULL,NULL,NULL,0,NULL);
 if(full){
  record(count[0]);
  for(int i=0;i<MAX_REF_MV_STACK_SIZE;i++){record(stack[0][i].this_mv.as_mv.row);record(stack[0][i].this_mv.as_mv.col);}
 }else{
  record(count[0]<2?count[0]:2);
  for(int i=0;i<2;i++){record(i<count[0]?stack[0][i].this_mv.as_mv.row:0);record(i<count[0]?stack[0][i].this_mv.as_mv.col:0);}
 }
 record(context[0]);
 int row=0,col=0,is_new=0;
 if(pattern==1){is_new=step%3!=0;row=is_new?((step/3)%3-1)*16:0;col=is_new?((step/5)%3-1)*16:0;}
 if(pattern==2){is_new=1;row=((step*37)%129-64)*16;col=((step*53)%129-64)*16;}
 if(pattern==3){is_new=step%2;row=is_new?((step*17)%5-2)*2:0;col=is_new?((step*11)%5-2)*2:0;}
 mi->mode=is_new?NEWMV:GLOBALMV;mi->mv[0].as_mv.row=row;mi->mv[0].as_mv.col=col;
 mi->mi_row_start=y/4;mi->mi_col_start=x/4;
 if(full)av2_update_ref_mv_bank(cm,xd,1,mi);
 for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)
  xd->is_mi_coded[0][((y/4+dy)&15)*16+((x/4+dx)&15)]=1;
 step++;
}
static void walk(int x,int y,int w,int h) {
 if(x>=cm->width||y>=cm->height)return;
 if(w==8&&h==8){leaf(x,y);return;}
 BLOCK_SIZE b=BLOCK_INVALID;
 for(int i=0;i<BLOCK_SIZES;i++)if(block_size_wide[i]==w&&block_size_high[i]==h){b=i;break;}
 assert(b!=BLOCK_INVALID);
 PARTITION_TYPE forced=av2_get_normative_forced_partition_type(&cm->mi_params,SHARED_PART,1,1,y/4,x/4,b,NULL);
 int vertical=forced==PARTITION_INVALID?w>h:forced==PARTITION_VERT;
 if(vertical){walk(x,y,w/2,h);walk(x+w/2,y,w/2,h);}
 else{walk(x,y,w,h/2);walk(x,y+h/2,w,h/2);}
}
int main(void) {
 const int dims[][2]={{8,8},{24,16},{64,64},{72,40},{128,136}};
 for(full=0;full<2;full++)for(int d=0;d<5;d++)for(pattern=0;pattern<4;pattern++) {
  cm=calloc(1,sizeof(*cm));xd=calloc(1,sizeof(*xd));
  cm->width=dims[d][0];cm->height=dims[d][1];cm->sb_size=BLOCK_64X64;cm->mib_size=16;
  cm->mi_params.mi_cols=cm->width/4;cm->mi_params.mi_rows=cm->height/4;
  cm->features.max_drl_bits=full?5:1;cm->features.fr_mv_precision=MV_PRECISION_QTR_PEL;
  cm->seq_params.enable_refmvbank=full;cm->seq_params.mib_size=16;cm->mib_size_log2=4;
  cm->current_frame.frame_type=INTER_FRAME;cm->mi_params.mi_stride=cm->mi_params.mi_cols;
  cm->seq_params.enable_drl_reorder=DRL_REORDER_DISABLED;
  xd->width=xd->height=2;xd->mi_stride=cm->mi_params.mi_cols;
  xd->tree_type=SHARED_PART;xd->is_mi_coded_stride=16;
  xd->tile.mi_col_end=cm->mi_params.mi_cols;xd->tile.mi_row_end=cm->mi_params.mi_rows;
  xd->global_motion=cm->global_motion;
  const int cells=cm->mi_params.mi_cols*cm->mi_params.mi_rows;
  MB_MODE_INFO *blocks=calloc(cells/4,sizeof(*blocks));
  grid=calloc(cells,sizeof(*grid));subgrid=calloc(cells,sizeof(*subgrid));
  for(int y=0;y<cm->height;y+=4)for(int x=0;x<cm->width;x+=4)
   grid[y/4*cm->mi_params.mi_cols+x/4]=&blocks[y/8*(cm->width/8)+x/8];
  cm->mi_params.mi_grid_base=grid;
  xd->plane[0].dst.width=cm->width;xd->plane[0].dst.height=cm->height;
  hash=14695981039346656037ULL;step=0;
  for(int y=0;y<cm->height;y+=64)for(int x=0;x<cm->width;x+=64){
   memset(xd->is_mi_coded,0,sizeof(xd->is_mi_coded));
   if(full){if(x==0)av2_zero(xd->ref_mv_bank);av2_reset_refmv_bank(cm,xd,&xd->tile,y/4,x/4);}
   walk(x,y,64,64);
  }
  printf("%llu,\n",hash);
  free(subgrid);free(grid);free(blocks);free(xd);free(cm);
 }
}
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--emit',action='store_true')
    args=parser.parse_args()
    revision=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if revision!=PIN:raise SystemExit('AVM revision must be '+PIN)
    with tempfile.TemporaryDirectory(prefix='nc-av2-mvrefs-') as tmp:
        path=Path(tmp);(path/'oracle.c').write_text(C)
        (path/'mvref.c').write_text(subprocess.check_output(['git','-C',str(args.source),'show',PIN+':av2/common/mvref_common.c'],text=True))
        subprocess.run(['cc','-O2','-std=c11','-ffunction-sections','-fdata-sections','-I'+str(args.source),'-I'+str(args.build),
            str(path/'oracle.c'),str(path/'mvref.c'),str(args.build/'libavm.a'),'-Wl,--gc-sections','-lm','-lpthread','-o',str(path/'oracle')],check=True)
        output=subprocess.check_output([str(path/'oracle')],text=True)
    values=[int(v) for v in re.findall(r'\d+',output)]
    if args.emit:print(values);return
    rust=(ROOT/'crates/nc-av2/src/mv_refs.rs').read_text()
    expected=[]
    for name in ('REFERENCE_HASHES','FULL_LIST_HASHES'):
        match=re.search(name+r':.*?=\s*\[(.*?)\];',rust,re.S)
        if not match:raise SystemExit(f'Missing {name} Rust fixtures')
        expected+=[int(v.replace('_','')) for v in re.findall(r'\d[\d_]*',match.group(1))]
    if values!=expected:raise SystemExit(f'MV references differ: {values} != {expected}')
    print('AV2 original MV references/context match 40 picture traces (basic and full lists).')

if __name__=='__main__':main()
