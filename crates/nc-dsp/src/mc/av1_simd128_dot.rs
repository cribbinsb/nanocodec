//! WebAssembly Relaxed-SIMD (`i16x8.relaxed_dot_i8x16_i7x16_s`) horizontal and
//! vertical-only passes for [`super::av1_simd128`], shared across AV1
//! (`put` and `prep`), HEVC and VVC (`SeparableRounding`).
//!
//! Every kernel tap `t` (AV1 halved, or HEVC/VVC full) lies in `-64..=64`, so
//! `|t|` is a 7-bit unsigned integer (`0..=64 <= 127`) whose top bit is clear
//! — x86 `vpmaddubsw` and Arm `smull` + `addp` agree bit-for-bit. Each pixel
//! byte `p` (`0..=255`) is flipped by XOR with `0x80` when `t >= 0` (`p - 128`,
//! signed `i8`) or `0x7f` when `t < 0` (`127 - p`, signed `i8`):
//! - `t >= 0`: `(p - 128) * t = p * t - 128 * t`
//! - `t < 0`:  `(127 - p) * (-t) = p * t - 127 * t`
//!
//! Each adjacent pair sum stays within `±16384` (never saturating `i16`), and
//! starting the `i16x8` accumulator at `Σ (t >= 0 ? 128·t : 127·t)` (`~8200`)
//! cancels the pixel biases so the exact unrounded `Σ p_k · t_k` emerges in
//! `P` dot products instead of `2P` widening multiplies and loads.

use super::av1_simd128::{Raw, Sink, round};
use super::simd128_byte_dot::{self as bytes, interleave as interleave_bytes};
use core::arch::wasm32::{i16x8_add, i16x8_splat, v128, v128_load64_zero, v128_xor};

/// Prepared 8-lane byte-pair filter for `P` tap pairs (`T = 2P` taps).
pub(super) struct DotFilter<const P: usize> {
    sign_xor: [v128; P],
    mags: [v128; P],
    indices: [v128; P],
    bias: v128,
}

impl<const P: usize> DotFilter<P> {
    /// Prepare `taps` (`T = 2P`, each in `-127..=127`) for 8-wide strips.
    #[inline]
    pub(super) fn new<const T: usize>(taps: &[i16; T]) -> Self {
        const { assert!(T == 2 * P && (T == 4 || T == 6 || T == 8)) };
        let flip = |t: i16| -> u16 { if t < 0 { 0x7f } else { 0x80 } };
        let zero = i16x8_splat(0);
        let mut sign_xor = [zero; P];
        let mut mags = [zero; P];
        let mut indices = [zero; P];
        for j in 0..P {
            let (t0, t1) = (taps[2 * j], taps[2 * j + 1]);
            sign_xor[j] = i16x8_splat((flip(t0) | (flip(t1) << 8)) as i16);
            mags[j] = bytes::magnitudes(t0, t1);
            indices[j] = bytes::mask(bytes::indices::<T>(2 * j, 2 * j + 1));
        }
        // Keep swizzle vectors opaque so V8 holds them in registers across the
        // row loop rather than rematerializing byte constants per iteration.
        let indices = core::hint::black_box(indices);
        let bias_sum: i32 = taps
            .iter()
            .map(|&t| {
                if t < 0 {
                    127 * i32::from(t)
                } else {
                    128 * i32::from(t)
                }
            })
            .sum();
        Self {
            sign_xor,
            mags,
            indices,
            bias: i16x8_splat(bias_sum as i16),
        }
    }

    /// Unrounded horizontal sum of 8 outputs starting at `src`.
    #[inline]
    pub(super) unsafe fn horizontal8<const T: usize>(&self, src: *const u8) -> v128 {
        // SAFETY: caller guarantees `8 + T - 1` readable bytes at `src`.
        let loaded = unsafe { bytes::load_row::<8, T>(src) };
        let mut sum = self.bias;
        for j in 0..P {
            let products =
                bytes::horizontal(loaded, self.indices[j], self.sign_xor[j], self.mags[j]);
            sum = i16x8_add(sum, products);
        }
        sum
    }

    #[inline]
    fn dot_pair(&self, j: usize, pair: v128) -> v128 {
        bytes::dot(v128_xor(pair, self.sign_xor[j]), self.mags[j])
    }

    #[inline]
    fn sum2(&self, p0: v128, p1: v128) -> v128 {
        i16x8_add(
            self.bias,
            i16x8_add(self.dot_pair(0, p0), self.dot_pair(1, p1)),
        )
    }

    #[inline]
    fn sum3(&self, p0: v128, p1: v128, p2: v128) -> v128 {
        i16x8_add(self.sum2(p0, p1), self.dot_pair(2, p2))
    }

    #[inline]
    fn sum4(&self, p0: v128, p1: v128, p2: v128, p3: v128) -> v128 {
        i16x8_add(self.sum3(p0, p1, p2), self.dot_pair(3, p3))
    }
}

/// Vertical-only `(0, frac_y)` pass for 8-wide strips, keeping sliding pairs
/// in scalar registers (not rotated arrays) so V8 holds them in XMM registers.
#[inline]
pub(super) unsafe fn vertical_only8<
    const W: usize,
    const P: usize,
    const T: usize,
    S: Sink,
    const C: usize,
>(
    window: Raw<*const u8>,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16; T],
    v_round: i16,
) {
    let f = DotFilter::<P>::new::<T>(taps);
    // SAFETY: caller checked `h` (even, > 0), `W` a multiple of 8, the window
    // `lead..lead + h + T - 1` at `C..C + W`, and destination `0..h` x `0..W`.
    unsafe {
        for c0 in (0..W).step_by(8) {
            let load = |k: usize| v128_load64_zero(window.at(lead + k, c0 + C).cast());
            if P == 2 {
                let (r0, r1, mut prev) = (load(0), load(1), load(2));
                let (mut e0, mut o0) = (interleave_bytes(r0, r1), interleave_bytes(r1, prev));
                for r in (0..h).step_by(2) {
                    let (a, b) = (load(r + 3), load(r + 4));
                    let (e1, o1) = (interleave_bytes(prev, a), interleave_bytes(a, b));
                    prev = b;
                    out.store::<8>(r, c0, round(f.sum2(e0, e1), v_round));
                    out.store::<8>(r + 1, c0, round(f.sum2(o0, o1), v_round));
                    (e0, o0) = (e1, o1);
                }
            } else if P == 3 {
                let (r0, r1, r2, r3, mut prev) = (load(0), load(1), load(2), load(3), load(4));
                let (mut e0, mut e1) = (interleave_bytes(r0, r1), interleave_bytes(r2, r3));
                let (mut o0, mut o1) = (interleave_bytes(r1, r2), interleave_bytes(r3, prev));
                for r in (0..h).step_by(2) {
                    let (a, b) = (load(r + 5), load(r + 6));
                    let (e2, o2) = (interleave_bytes(prev, a), interleave_bytes(a, b));
                    prev = b;
                    out.store::<8>(r, c0, round(f.sum3(e0, e1, e2), v_round));
                    out.store::<8>(r + 1, c0, round(f.sum3(o0, o1, o2), v_round));
                    (e0, e1) = (e1, e2);
                    (o0, o1) = (o1, o2);
                }
            } else {
                let (r0, r1, r2, r3) = (load(0), load(1), load(2), load(3));
                let (r4, r5, mut prev) = (load(4), load(5), load(6));
                let (mut e0, mut e1, mut e2) = (
                    interleave_bytes(r0, r1),
                    interleave_bytes(r2, r3),
                    interleave_bytes(r4, r5),
                );
                let (mut o0, mut o1, mut o2) = (
                    interleave_bytes(r1, r2),
                    interleave_bytes(r3, r4),
                    interleave_bytes(r5, prev),
                );
                for r in (0..h).step_by(2) {
                    let (a, b) = (load(r + 7), load(r + 8));
                    let (e3, o3) = (interleave_bytes(prev, a), interleave_bytes(a, b));
                    prev = b;
                    out.store::<8>(r, c0, round(f.sum4(e0, e1, e2, e3), v_round));
                    out.store::<8>(r + 1, c0, round(f.sum4(o0, o1, o2, o3), v_round));
                    (e0, e1, e2) = (e1, e2, e3);
                    (o0, o1, o2) = (o1, o2, o3);
                }
            }
        }
    }
}
