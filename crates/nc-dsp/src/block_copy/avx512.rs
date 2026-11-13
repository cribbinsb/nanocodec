//! Cached plane copies in 64-byte spans. Masked tails avoid source padding
//! reads; unlike border extension these pixels will be consumed immediately.
//! SVT-AV1's BSD-2 `copy_wxh_avx512.c` (Meta 2024) also keeps short fixed
//! copies narrow; those existing paths remain unchanged here.
use core::arch::x86_64::{
    _mm512_loadu_si512, _mm512_mask_storeu_epi8, _mm512_maskz_loadu_epi8, _mm512_storeu_si512,
};
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
pub(crate) unsafe fn copy(src: &[u8], ss: usize, dst: &mut [u8], ds: usize, w: usize, h: usize) {
    const LINE: usize = 64;
    let full = w / LINE * LINE;
    let tail = w - full;
    for y in 0..h {
        for x in (0..full).step_by(LINE) {
            // SAFETY: complete visible cache-line spans in both validated rows.
            unsafe {
                let v = _mm512_loadu_si512(src.as_ptr().add(y * ss + x).cast());
                _mm512_storeu_si512(dst.as_mut_ptr().add(y * ds + x).cast(), v);
            }
        }
        if tail != 0 {
            let mask = u64::MAX >> (LINE - tail);
            // SAFETY: the mask covers only the remaining visible row pixels.
            unsafe {
                let v = _mm512_maskz_loadu_epi8(mask, src.as_ptr().add(y * ss + full).cast());
                _mm512_mask_storeu_epi8(dst.as_mut_ptr().add(y * ds + full).cast(), mask, v);
            }
        }
    }
}
