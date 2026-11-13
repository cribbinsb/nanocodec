#!/usr/bin/env python3
"""Verify native inter-header prefixes using pinned AVM's original full writer.

Requires a configured/built AVM tree for headers and libavm.a. Compiles the
original bitstream.c in TemporaryDirectory, emits only eight tiny headers and
compares them with Rust fixtures. No codec encode, video files or retained build.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

PIN = '966a7d7cd6fcf60360caf5dc413b2aeeb65e144d'
ROOT = Path(__file__).resolve().parents[2]
HARNESS = r'''
int main(void) {
 AV2_COMP *cpi=calloc(1,sizeof(*cpi)); AV2_COMMON *cm=&cpi->common;
 RefCntBuffer *cur=calloc(1,sizeof(*cur)), *ref=calloc(1,sizeof(*ref));
 cm->cur_frame=cur;cur->num_ref_frames=1;
 cm->seq_params.bit_depth=AVM_BITS_8;
 cm->seq_params.max_frame_width=cm->width=8;
 cm->seq_params.max_frame_height=cm->height=8;
 cm->seq_params.ref_frames=8;cm->seq_params.ref_frames_log2=3;
 cm->seq_params.subsampling_x=cm->seq_params.subsampling_y=1;
 cm->seq_params.order_hint_info.order_hint_bits_minus_1=7;
 cm->seq_params.equal_ac_dc_q=1;
 cm->seq_params.enable_cdef=1;cm->cdef_info.cdef_frame_enable=0;
 cm->current_frame.frame_type=INTER_FRAME;
 cm->current_frame.order_hint=1;cm->current_frame.display_order_hint=1;
 cm->current_frame.refresh_frame_flags=255;
 cm->current_frame.reference_mode=SINGLE_REFERENCE;
 cm->current_frame.skip_mode_info.skip_mode_allowed=1;
 cm->immediate_output_picture=1;
 cm->features.primary_ref_frame=PRIMARY_REF_NONE;
 cm->features.cross_frame_context=CROSS_FRAME_CONTEXT_DISABLED;
 cm->features.fr_mv_precision=cm->features.most_probable_fr_mv_precision=MV_PRECISION_QTR_PEL;
 cm->features.interp_filter=BILINEAR;
 cm->features.enabled_motion_modes=1<<SIMPLE_TRANSLATION;
 cm->seq_params.seq_enabled_motion_modes=1<<SIMPLE_TRANSLATION;
 cm->features.disable_cdf_update=1;cm->features.reduced_tx_set_used=2;
 cm->ref_frames_info.num_total_refs=1;
 cm->remapped_ref_idx[0]=0;cm->ref_frame_map[0]=ref;
 cm->tiles.uniform_spacing=1;cm->tiles.cols=cm->tiles.rows=1;
 cm->tiles.mi_cols=cm->tiles.mi_rows=2;cm->tiles.mib_size_log2=4;
 cpi->signal_primary_ref_frame=1;
 int qs[]={0,1,100,255};
 for(int d=0;d<2;d++) for(int i=0;i<4;i++) {
  cm->seq_params.max_frame_width=cm->width=d?72:8;
  cm->seq_params.max_frame_height=cm->height=d?40:8;
  cm->tiles.mi_cols=cm->width/4;cm->tiles.mi_rows=cm->height/4;
  cm->tiles.max_log2_cols=d;cm->tiles.max_log2_rows=0;
  cm->features.coded_lossless=cm->features.all_lossless=qs[i]==0;
  cm->features.tx_mode=qs[i]?TX_MODE_SELECT:ONLY_4X4;
  cm->quant_params.base_qindex=qs[i];
  uint8_t data[128]={0};struct avm_write_bit_buffer wb={data,0},saved;
  avm_wb_write_bit(&wb,1);
  write_uncompressed_header(cpi,OBU_REGULAR_TILE_GROUP,&saved,&wb);
  while(wb.bit_offset%8)avm_wb_write_bit(&wb,0);
  printf("%u ",wb.bit_offset/8);
  for(unsigned j=0;j<wb.bit_offset/8;j++)printf("%u,",data[j]);puts("");
 }
 free(ref);free(cur);free(cpi);
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
    source=subprocess.check_output(['git','-C',str(args.source),'show',PIN+':av2/encoder/bitstream.c'],text=True)
    with tempfile.TemporaryDirectory(prefix='nc-av2-inter-header-') as tmp:
        path=Path(tmp)
        (path/'bitstream.c').write_text(source)
        (path/'oracle.c').write_text('#include "bitstream.c"\n'+HARNESS.replace('disable_cdf_update=1', 'disable_cdf_update='+str(int(not args.adaptive))))
        subprocess.run(['cc','-O1','-std=c11','-ffunction-sections','-fdata-sections',
            '-I'+str(args.source),'-I'+str(args.build),str(path/'oracle.c'),
            str(args.build/'libavm.a'),'-Wl,--gc-sections','-lm','-lpthread','-o',str(path/'oracle')],check=True)
        output=subprocess.check_output([str(path/'oracle')],text=True)
    arrays=[]
    for line in output.splitlines():
        values=[int(v) for v in re.findall(r'\d+',line)]
        assert values[0]==len(values)-1
        arrays.append(values[1:])
    if args.emit:
        for arr in arrays: print(arr)
        return
    rust=(ROOT/'crates/nc-av2/src/frame_header.rs').read_text()
    name='ADAPTIVE_INTER_REFERENCE' if args.adaptive else 'INTER_REFERENCE'
    match=re.search(r'const '+name+r':.*?=\s*\[(.*?)\];',rust,re.S)
    if not match:raise SystemExit('Missing INTER_REFERENCE Rust fixtures')
    expected=[[int(v) for v in re.findall(r'\d+',arr)] for arr in re.findall(r'&\[(.*?)\]',match.group(1),re.S)]
    if arrays!=expected:raise SystemExit(f'Header oracle differs: {arrays} != {expected}')
    print('AV2 original full inter-header writer matches 8 prefixes.')

if __name__=='__main__':main()
