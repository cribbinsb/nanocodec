/* Test-only AVM v1.0.0 entropy oracle. All coding/adaptation comes from the
 * original entenc.c, entdec.c and prob.h. No reference functions are replaced. */
#include <assert.h>
#include <stdio.h>
#include "avm_dsp/entenc.h"
#include "avm_dsp/entdec.h"
#include "avm_dsp/prob.h"
#include "config/avm_dsp_rtcd.h"

int main(void) {
  od_ec_enc encoder;
  avm_od_ec_enc_init(&encoder, 1024);
  uint16_t cdf[8] = {30000, 16000, 5000, 0, 0, 2, 3, 4};
  uint32_t state = 12345;
  for (int i = 0; i < 20000; ++i) {
    state = state * 1664525u + 1013904223u;
    const int symbol = state >> 30;
    const int bits = state % 17;
    const uint32_t literal = (state >> 8) & ((1u << bits) - 1);
    avm_od_ec_encode_cdf_q15(&encoder, symbol, cdf, 4);
    update_cdf(cdf, symbol, 4);
    for (int bit = bits - 1; bit >= 0; --bit)
      od_ec_encode_bool_bypass(&encoder, (literal >> bit) & 1);
  }
  uint32_t size;
  const unsigned char *packet = avm_od_ec_enc_done(&encoder, &size);
  assert(packet != NULL);
  od_ec_dec decoder;
  avm_od_ec_dec_init(&decoder, packet, size);
  uint16_t decoded[8] = {30000, 16000, 5000, 0, 0, 2, 3, 4};
  state = 12345;
  for (int i = 0; i < 20000; ++i) {
    state = state * 1664525u + 1013904223u;
    const int symbol = state >> 30;
    const int bits = state % 17;
    const uint32_t literal = (state >> 8) & ((1u << bits) - 1);
    const int actual = avm_od_ec_decode_cdf_q15_c(&decoder, decoded, 4);
    assert(actual == symbol);
    update_cdf(decoded, actual, 4);
    uint32_t value = 0;
    for (int bit = 0; bit < bits; ++bit)
      value = (value << 1) | od_ec_decode_bool_bypass(&decoder);
    assert(value == literal);
  }
  for (int i = 0; i < 8; ++i) assert(cdf[i] == decoded[i]);
  assert(fwrite(packet, 1, size, stdout) == size);
  avm_od_ec_enc_clear(&encoder);
  return 0;
}
