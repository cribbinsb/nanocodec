//! AVX2 border stores: aligned 32-byte non-temporal stores (`vmovntdq`)
//! for the body of each span, ordinary stores for its unaligned ends.

use core::arch::x86_64::{_mm_sfence, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_stream_si256};

/// Bytes per non-temporal store.
const LANES: usize = 32;

/// The aligned body of `dst`: its offset and length (a multiple of
/// [`LANES`]).
fn body(dst: &[u8]) -> (usize, usize) {
    let head = dst.as_ptr().align_offset(LANES).min(dst.len());
    (head, (dst.len() - head) / LANES * LANES)
}

/// Set every byte of `dst` to `value`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn fill(dst: &mut [u8], value: u8) {
    let (head, len) = body(dst);
    dst[..head].fill(value);
    dst[head + len..].fill(value);
    let splat = _mm256_set1_epi8(value as i8);
    for chunk in dst[head..head + len].as_chunks_mut::<LANES>().0 {
        // SAFETY: `chunk` is 32 writable bytes starting at a 32-byte
        // aligned address (`body`), as `vmovntdq` requires.
        unsafe { _mm256_stream_si256(chunk.as_mut_ptr().cast(), splat) };
    }
}

/// Copy `src` into `dst` (same length).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn copy(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "border copy lengths differ");
    let (head, len) = body(dst);
    dst[..head].copy_from_slice(&src[..head]);
    dst[head + len..].copy_from_slice(&src[head + len..]);
    let spans = dst[head..head + len].as_chunks_mut::<LANES>().0;
    for (out, input) in spans
        .iter_mut()
        .zip(src[head..head + len].as_chunks::<LANES>().0)
    {
        // SAFETY: `input` is 32 readable bytes (an unaligned load) and `out`
        // 32 writable bytes at a 32-byte aligned address (`body`).
        unsafe {
            let samples = _mm256_loadu_si256(input.as_ptr().cast());
            _mm256_stream_si256(out.as_mut_ptr().cast(), samples);
        }
    }
}

/// Order the non-temporal stores before any later store, so that another
/// thread given the plane afterwards sees them.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn fence() {
    _mm_sfence();
}
