#!/usr/bin/env python3
"""Check AV2 partition-symbol goldens against the original pinned AVM writer.

Uses a configured AVM build's generated headers, but compiles only a tiny C
oracle in TemporaryDirectory. No video files or full codec build is produced.
"""
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
#include "av2/common/bru.h"
typedef struct { unsigned long long hash; int count; } probe_writer;
#define avm_writer probe_writer
static void avm_write_symbol(probe_writer *w,int symbol,const avm_cdf_prob *cdf,int n) {
  w->hash=(w->hash ^ symbol)*1099511628211ULL;
  for(int i=0;i<n;i++) w->hash=(w->hash ^ cdf[i])*1099511628211ULL;
  w->count++;
}
static void avm_write_bit(probe_writer *w,int symbol) { const avm_cdf_prob c[2]={16384,0}; avm_write_symbol(w,symbol,c,2); }
'''
SUFFIX = r'''
static void walk(AV2_COMMON *cm,MACROBLOCKD *xd,int r,int c,BLOCK_SIZE b,PARTITION_TREE *parent,int index,probe_writer *w) {
 if(r>=cm->mi_params.mi_rows || c>=cm->mi_params.mi_cols) return;
 PARTITION_TREE p={0}; p.parent=parent;p.region_type=INTRA_REGION;
 set_chroma_ref_info(SHARED_PART,r,c,index,b,&p.chroma_ref_info,parent?&parent->chroma_ref_info:NULL,parent?parent->bsize:BLOCK_INVALID,parent?parent->partition:PARTITION_NONE,1,1);
 p.bsize=b;
 PARTITION_TYPE split=av2_get_normative_forced_partition_type(&cm->mi_params,SHARED_PART,1,1,r,c,b,NULL);
 if(split==PARTITION_INVALID) split=(b==BLOCK_8X8)?PARTITION_NONE:(block_size_wide[b]>block_size_high[b]?PARTITION_VERT:PARTITION_HORZ);
 p.partition=split;
 PARTITION_TYPE actual=write_partition(cm,xd,r,c,split,b,&p,NULL,w);
 assert(actual==split);
 BLOCK_SIZE sub=get_partition_subsize(b,split);
 if(split!=PARTITION_NONE) {
   walk(cm,xd,r,c,sub,&p,0,w);
   walk(cm,xd,r+(split==PARTITION_HORZ?mi_size_high[sub]:0),c+(split==PARTITION_VERT?mi_size_wide[sub]:0),sub,&p,1,w);
 }
 update_ext_partition_context(xd,r,c,sub,b,split);
}
int main(void) {
 const int dims[][2]={{8,8},{16,8},{8,24},{64,64},{72,40},{128,136}};
 AV2_COMMON *cm=calloc(1,sizeof(*cm)); MACROBLOCKD *xd=calloc(1,sizeof(*xd)); FRAME_CONTEXT *fc=calloc(1,sizeof(*fc));
 cm->seq_params.subsampling_x=cm->seq_params.subsampling_y=1;
 cm->seq_params.max_pb_aspect_ratio_log2_m1=0;
 cm->sb_size=BLOCK_64X64; xd->tree_type=SHARED_PART;xd->tile_ctx=fc;
 memcpy(fc->do_split_cdf,default_do_split_cdf,sizeof(fc->do_split_cdf));
 memcpy(fc->rect_type_cdf,default_rect_type_cdf,sizeof(fc->rect_type_cdf));
 for(unsigned i=0;i<sizeof(dims)/sizeof(dims[0]);i++) {
  cm->mi_params.mi_cols=dims[i][0]/4;cm->mi_params.mi_rows=dims[i][1]/4;
  xd->above_partition_context[0]=calloc(cm->mi_params.mi_cols,1);
  probe_writer w={14695981039346656037ULL,0};
  for(int r=0;r<cm->mi_params.mi_rows;r+=16) {
   memset(xd->left_partition_context,0,sizeof(xd->left_partition_context));
   for(int c=0;c<cm->mi_params.mi_cols;c+=16)walk(cm,xd,r,c,BLOCK_64X64,NULL,0,&w);
  }
  printf("%llu %d\n",w.hash,w.count);free(xd->above_partition_context[0]);
 }
 MB_MODE_INFO mb={0}; MB_MODE_INFO *mi=&mb; xd->mi=&mi;
 mb.sb_type[0]=BLOCK_8X8;mb.ref_frame[0]=INTRA_FRAME;
 mb.tx_partition_type[0]=TX_PARTITION_SPLIT;
 xd->reduced_tx_part_set=1;
 memcpy(fc->txfm_do_partition_cdf,default_txfm_do_partition_cdf,sizeof(fc->txfm_do_partition_cdf));
 memcpy(fc->txfm_4way_partition_type_cdf,default_txfm_4way_partition_type_reduced_cdf,sizeof(fc->txfm_4way_partition_type_cdf));
 probe_writer tw={14695981039346656037ULL,0};
 write_tx_partition(xd,&mb,TX_8X8,0,0,&tw);
 printf("%llu %d\n",tw.hash,tw.count);
 mb.ref_frame[0]=0;mb.ref_frame[1]=NONE_FRAME;
 mb.mode=GLOBALMV;
 probe_writer iw={14695981039346656037ULL,0};
 write_tx_partition(xd,&mb,TX_8X8,0,0,&iw);
 printf("%llu %d\n",iw.hash,iw.count);
 free(fc);free(xd);free(cm);
}
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--build',required=True,type=Path)
    parser.add_argument('--emit',action='store_true')
    args=parser.parse_args()
    revision=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if revision!=PIN: raise SystemExit('AVM revision must be '+PIN)
    source=subprocess.check_output(['git','-C',str(args.source),'show',PIN+':av2/encoder/bitstream.c'],text=True)
    begin=source.index('static AVM_INLINE PARTITION_TYPE write_partition(')
    end=source.index('\nstatic AVM_INLINE void write_modes_sb(',begin)
    with tempfile.TemporaryDirectory(prefix='nc-av2-partition-') as tmp:
        path=Path(tmp)
        tx_begin=source.index('static void write_tx_partition(')
        tx_end=source.index('\nstatic int write_skip(',tx_begin)
        (path/'oracle.c').write_text(PREFIX+source[begin:end]+source[tx_begin:tx_end]+SUFFIX)
        subprocess.run(['cc','-O2','-std=c11','-I'+str(args.source),'-I'+str(args.build),str(path/'oracle.c'),'-o',str(path/'oracle')],check=True)
        output=subprocess.check_output([str(path/'oracle')],text=True)
    values=[int(v) for v in output.split()]
    if args.emit: print(output,end='');return
    rust=(ROOT/'crates/nc-av2/src/partition.rs').read_text()
    match=re.search(r'const REFERENCE_TRACE:.*?=\s*\[(.*?)\];',rust,re.S)
    if not match:raise SystemExit('Missing REFERENCE_TRACE in partition.rs')
    expected=[int(v.replace('_','')) for v in re.findall(r'\d[\d_]*',match.group(1))]
    txmatch=re.search(r'const REFERENCE_TX_TRACE:.*?=\s*\((.*?)\);',rust,re.S)
    if not txmatch:raise SystemExit('Missing REFERENCE_TX_TRACE')
    expected += [int(v.replace('_','')) for v in re.findall(r'\d[\d_]*',txmatch.group(1))]
    intermatch=re.search(r'const REFERENCE_INTER_TX_TRACE:.*?=\s*\((.*?)\);',rust,re.S)
    if not intermatch:raise SystemExit('Missing REFERENCE_INTER_TX_TRACE')
    expected += [int(v.replace('_','')) for v in re.findall(r'\d[\d_]*',intermatch.group(1))]
    if values!=expected:raise SystemExit(f'Partition oracle differs: {values} != {expected}')
    print('AV2 original-C traces verified (6 partition dimensions + TX4 split).')

if __name__=='__main__':main()
