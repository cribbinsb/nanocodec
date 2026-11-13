#!/usr/bin/env python3
"""Original-AVM oracle for nc-av2 coefficient contexts and coded bytes.

Runs the `coeff_vectors` example (tiles of Y/U/V blocks, TX4..TX64, intra and
inter, every quantizer category, fixed and adaptive CDFs), then a temporary C
program linked against the pinned AVM v1.0.0 library that, for every block:
  * derives the contexts with AVM `get_txb_ctx` (plus the V `eob_u_flag`
    offset) from the same neighbour bytes and compares them with ours;
  * decodes our bytes with AVM `av2_read_sig_txtype` and
    `av2_read_coeffs_txb` and compares the coefficients and the returned
    entropy context (`cul_level` with DC sign) with ours.
Nothing is written outside a temporary directory; no YUV is involved.

Usage: check_av2_coeff.py --source AVM_SRC --build AVM_BUILD --vectors BIN
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

PIN = '966a7d7cd6fcf60360caf5dc413b2aeeb65e144d'
ORACLE = r'''
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "av2/common/av2_common_int.h"
#include "av2/common/idct.h"
#include "av2/common/txb_common.h"
#include "av2/decoder/decoder.h"
#include "av2/decoder/decodetxb.h"
#include "avm_dsp/bitreader.h"
#include "config/av2_rtcd.h"
#include "config/avm_dsp_rtcd.h"

/* Dequantizer that AVM's get_dqv turns into level << 2 before the
 * transform-size shift (at most 2), so levels below 2^16 are recovered. */
#define DEQUANT (1 << (QUANT_TABLE_BITS + 2))
#define UNITS 16

static const TX_SIZE tx_sizes[7] = { 0, 0, TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64 };
static const BLOCK_SIZE block_sizes[8] = { 0, 0, BLOCK_4X4, BLOCK_8X8, BLOCK_16X16,
                                           BLOCK_32X32, BLOCK_64X64, BLOCK_128X128 };

static AV2_COMMON cm;
static FRAME_CONTEXT fc, default_fc;
static DecoderCodingBlock dcb;
static MB_MODE_INFO mbmi;
static MB_MODE_INFO *mi_pointer = &mbmi;
static TX_TYPE tx_type_map[256];
static tran_low_t dqcoeff[MAX_MB_PLANE][4096];
static eob_info eob_data[MAX_MB_PLANE], bob_data[MAX_MB_PLANE];
static struct avm_internal_error_info error_info;

static void fail(const char *what, long tile, long block) {
  printf("FAIL %s tile=%ld block=%ld\n", what, tile, block);
  exit(1);
}

static void start_tile(int qindex) {
  memset(&cm, 0, sizeof(cm));
  memset(&fc, 0, sizeof(fc));
  cm.fc = &fc;
  cm.default_frame_context = &default_fc;
  cm.quant_params.base_qindex = qindex;
  cm.features.reduced_tx_set_used = 2;
  av2_set_default_frame_contexts(&cm);
  memset(&dcb, 0, sizeof(dcb));
  MACROBLOCKD *xd = &dcb.xd;
  xd->mi = &mi_pointer;
  xd->tile_ctx = &fc;
  xd->tree_type = SHARED_PART;
  xd->lossless[0] = qindex == 0;
  xd->bd = 12;
  xd->tx_type_map = tx_type_map;
  xd->tx_type_map_stride = 16;
  xd->error_info = &error_info;
  for (int p = 0; p < MAX_MB_PLANE; ++p) {
    xd->plane[p].seg_dequant_QTX[0][0] = DEQUANT;
    xd->plane[p].seg_dequant_QTX[0][1] = DEQUANT;
    dcb.dqcoeff_block[p] = dqcoeff[p];
    dcb.eob_data[p] = &eob_data[p];
    dcb.bob_data[p] = &bob_data[p];
  }
}

int main(void) {
  static uint8_t packet[1 << 22];
  static int expected[1024];
  char kind[2];
  long tiles = 0, blocks = 0, coded = 0;
  avm_dsp_rtcd();
  av2_rtcd();
  while (scanf("%1s", kind) == 1) {
    int qindex, adaptive, size;
    if (kind[0] != 'T' || scanf("%d %d %d", &qindex, &adaptive, &size) != 3) fail("format", tiles, 0);
    for (int i = 0; i < size; ++i) {
      int byte;
      if (scanf("%d", &byte) != 1) fail("packet", tiles, 0);
      packet[i] = (uint8_t)byte;
    }
    start_tile(qindex);
    avm_reader reader;
    if (avm_reader_init(&reader, packet, size)) fail("reader", tiles, 0);
    reader.allow_update_cdf = (uint8_t)adaptive;
    if (setjmp(error_info.jmp)) fail("AVM decode error", tiles, blocks);
    error_info.setjmp = 1;
    long block = 0;
    while (scanf("%1s", kind) == 1 && kind[0] == 'B') {
      int tx_log2, plane, is_inter, block_log2, skip, dc_sign, cul;
      ENTROPY_CONTEXT above[UNITS], left[UNITS];
      if (scanf("%d %d %d %d %d %d", &tx_log2, &plane, &is_inter, &block_log2, &skip, &dc_sign) != 6)
        fail("block header", tiles, block);
      for (int i = 0; i < 2 * UNITS; ++i) {
        int value;
        if (scanf("%d", &value) != 1) fail("neighbours", tiles, block);
        if (i < UNITS) above[i] = (ENTROPY_CONTEXT)value; else left[i - UNITS] = (ENTROPY_CONTEXT)value;
      }
      const TX_SIZE tx_size = tx_sizes[tx_log2];
      const int coded_side = tx_log2 > 5 ? 32 : 1 << tx_log2;
      const int len = coded_side * coded_side;
      if (scanf("%d", &cul) != 1) fail("entropy context", tiles, block);
      for (int i = 0; i < len; ++i)
        if (scanf("%d", &expected[i]) != 1) fail("coefficients", tiles, block);

      MACROBLOCKD *xd = &dcb.xd;
      memset(&mbmi, 0, sizeof(mbmi));
      mbmi.ref_frame[0] = is_inter ? 0 : INTRA_FRAME; /* reference index 0 */
      mbmi.ref_frame[1] = NONE_FRAME;
      mbmi.mode = is_inter ? GLOBALMV : DC_PRED;
      mbmi.uv_mode = UV_DC_PRED;
      mbmi.tx_size = plane == 0 ? tx_size : TX_4X4;
      memset(tx_type_map, 0, sizeof(tx_type_map));
      memset(dqcoeff[plane], 0, sizeof(dqcoeff[plane]));

      TXB_CTX txb_ctx;
      get_txb_ctx(block_sizes[block_log2], tx_size, plane, above, left, &txb_ctx, 0);
      const int v_offset = plane == AVM_PLANE_V && xd->eob_u_flag ? V_TXB_SKIP_CONTEXT_OFFSET : 0;
      if (txb_ctx.txb_skip_ctx + v_offset != skip) fail("skip context", tiles, block);
      if (txb_ctx.dc_sign_ctx != dc_sign) fail("dc sign context", tiles, block);

      int decoded_cul = 0;
      if (av2_read_sig_txtype(&cm, &dcb, &reader, 0, 0, plane, &txb_ctx, tx_size)) {
        decoded_cul = av2_read_coeffs_txb(&cm, &dcb, &reader, 0, 0, plane, &txb_ctx, tx_size);
        ++coded;
      }
      if (decoded_cul != cul) fail("entropy context value", tiles, block);
      const int shift = 2 - av2_get_tx_scale(tx_size);
      for (int i = 0; i < len; ++i) {
        const int value = dqcoeff[plane][i];
        const int level = value < 0 ? -(-value >> shift) : value >> shift;
        if (level != expected[i]) {
          printf("pos %d: AVM %d ours %d\n", i, level, expected[i]);
          fail("coefficient", tiles, block);
        }
      }
      ++block;
      ++blocks;
    }
    if (kind[0] != 'E') fail("tile end", tiles, block);
    if (avm_reader_has_overflowed(&reader)) fail("reader overflow", tiles, block);
    error_info.setjmp = 0;
    ++tiles;
  }
  printf("%ld tiles, %ld blocks (%ld coded)\n", tiles, blocks, coded);
  return 0;
}
'''


def avm_libraries(build):
    """Libraries and flags of the pinned `avmdec` link (libavm needs TFLite)."""
    tokens = (build / 'CMakeFiles/avmdec.dir/link.txt').read_text().split()
    tail = tokens[tokens.index('avmdec', tokens.index('-o')) + 1:]
    return [str(build / t) if t.endswith('.a') and not t.startswith('/') else t
            for t in tail if not t.endswith('.o')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True, help='built coeff_vectors example')
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == PIN, revision
    vectors = subprocess.check_output([str(args.vectors)])
    with tempfile.TemporaryDirectory(prefix='nc-av2-coeff-') as directory:
        source = Path(directory) / 'oracle.c'
        binary = source.with_suffix('')
        source.write_text(ORACLE)
        objects = source.with_suffix('.o')
        subprocess.run(['cc', '-O1', '-DNDEBUG', '-I' + str(args.source), '-I' + str(args.build),
                        '-c', str(source), '-o', str(objects)], check=True)
        subprocess.run(['c++', str(objects), '-o', str(binary)] + avm_libraries(args.build), check=True)
        result = subprocess.run([str(binary)], input=vectors, capture_output=True, check=False)
    output = result.stdout.decode().strip()
    if result.returncode != 0:
        raise SystemExit(f'{output}\n{result.stderr.decode()}')
    print(f'PASS: original AVM contexts and decode match ours: {output}')


if __name__ == '__main__':
    main()
