/* Test-only driver for the ORIGINAL pinned AVM sequence writer.
 * This file has no codec implementation: it fills AVM's public configuration
 * structure and invokes the reference writer. Never linked into nanocodec.
 * Build against the pinned AVM source/build headers and libavm.a; see subset doc.
 */
#include <stdio.h>
#include <string.h>
#include "av2/encoder/bitstream.h"

static int width_bits(unsigned size) {
  unsigned top = size - 1;
  int bits = 1;
  while ((top >>= 1) != 0) ++bits;
  return bits;
}

int main(void) {
  const unsigned dimensions[][2] = {{1, 1}, {64, 48}, {1920, 1080}, {65536, 65536}};
  for (int single = 0; single <= 1; ++single) {
    for (unsigned i = 0; i < sizeof(dimensions) / sizeof(dimensions[0]); ++i) {
      SequenceHeader s;
      memset(&s, 0, sizeof(s));
      s.seq_header_id = 15;
      s.seq_profile_idc = MAIN_420_10_IP0;
      s.single_picture_header_flag = single;
      s.seq_max_level_idx = SEQ_LEVEL_MAX;
      s.bit_depth = AVM_BITS_8;
      s.subsampling_x = s.subsampling_y = 1;
      s.still_picture = single;
      s.seq_max_mlayer_cnt = 1;
      s.monotonic_output_order_flag = 1;
      s.max_frame_width = dimensions[i][0];
      s.max_frame_height = dimensions[i][1];
      s.num_bits_width = width_bits(s.max_frame_width);
      s.num_bits_height = width_bits(s.max_frame_height);
      s.sb_size = BLOCK_64X64;
      s.mib_size = 16;
      s.mib_size_log2 = 4;
      s.seq_enabled_motion_modes = 1 << SIMPLE_TRANSLATION;
      s.order_hint_info.order_hint_bits_minus_1 = single ? -1 : 7;
      s.enable_drl_reorder = DRL_REORDER_DISABLED;
      s.ref_frames = single ? 2 : 8;
      s.def_max_drl_bits = MIN_MAX_DRL_BITS;
      s.def_max_bvp_drl_bits = MIN_MAX_IBC_DRL_BITS;
      s.force_integer_mv = 2;
      s.enable_chroma_dctonly = 1;
      s.reduced_tx_part_set = 1;
      s.enable_avg_cdf = single;
      s.avg_cdf_type = single;
      s.equal_ac_dc_q = 1;
      s.enable_cdef = 1;
      s.enable_cdef_on_skip_txfm = single ? CDEF_ON_SKIP_TXFM_ADAPTIVE : CDEF_ON_SKIP_TXFM_DISABLED;
      unsigned char bytes[1024] = {0};
      const unsigned size = av2_write_sequence_header_obu(&s, bytes);
      for (unsigned j = 0; j < size; ++j) printf("%02x", bytes[j]);
      puts("");
    }
  }
  return 0;
}
