//! ICL forward32 uses guarded word lanes (including the low16 square).
//! Forward32 table entries point directly to ISA kernels, avoiding an extra
//! tail-jump wrapper. The 16- and 64-point forwards keep the AVX2 tier's
//! sixteen word lanes (`narrow_forward`): on Zen 4 (c7a, `nc-checkasm
//! --bench`, 2026-11-20) they took 61 / 1365 ns per 16x16 / 64x64 against
//! 85 / 1715 ns on the sixteen 32-bit lanes installed here before; at
//! 32x32 this tier's thirty-two word lanes stay ahead (284 against 368 ns).
//!
//! Every inverse keeps the AVX2 tier's guarded 16-bit graphs
//! (`avx2_inverse`): on Zen 5 (c8a, `nc-checkasm --bench`, 2026-11-17) the
//! sixteen 32-bit lanes installed here before were up to 1.65x slower than
//! them (sparse low-8 32x32: 235 vs 142 ns) and never faster on coefficients
//! inside the 16-bit guard. Only blocks above the guard, which run the
//! guard's 32-bit fallback, are slower (dense synthetic 32x32: 0.71 vs
//! 0.26 us); no 1080p AV1 encode or decode we profiled reached it.
use super::TxfmFns;

pub(crate) fn install(table: &mut TxfmFns) {
    super::avx512_quant::install(table);
    table.forward_dct[3] = super::avx512_forward::forward32::<false>;
    table.forward_dct_raster[3] = super::avx512_forward::forward32::<true>;
    table.forward_dct_low[1] = Some(super::avx512_forward::low32::<false>);
    table.forward_dct_raster_low[1] = Some(super::avx512_forward::low32::<true>);
    table.forward_dct_raster_bounded[1] = super::avx512_forward::bounded32;
}
