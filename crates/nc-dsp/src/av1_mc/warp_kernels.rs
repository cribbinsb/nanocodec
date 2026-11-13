//! The 8x8 warp filter of warped motion (AV1 §7.11.3.5) as one kernel per
//! [`crate::Dsp`]: a scalar reference and a SIMD version written once over
//! [`I16x8Shuffles`] (SIMD128); AVX2 and NEON use dedicated byte
//! kernels (`warp_avx2`, `warp_neon`).
//!
//! The kernel takes the block's source window (15 rows of 15 samples, the
//! 8x8 block with the 8-tap filters' reach, edges already clamped by the
//! caller) and its filter phases, and returns the 64 vertical sums rounded
//! by `round1`. As in dav1d 1.5.4 `warp_affine_8x8_c` (`src/mc_tmpl.c`,
//! BSD-2-Clause, see `THIRD_PARTY.md`), the horizontal pass filters 15
//! rows of 8 outputs into a 16-bit intermediate and the vertical pass 8
//! rows from it; each output has its own filter phase. The SIMD version
//! computes every output as one 8-tap dot product (`pmaddwd` and a
//! horizontal sum of four), storing the horizontal pass transposed so the
//! vertical pass is the same dot product along a column.
//!
//! Every value stays exact: horizontal sums are at most `255 · Σ|taps|`
//! (the taps sum to 128, with at most 47 of negative weight), so the
//! rounded intermediate fits 16 bits with room to spare, and the vertical
//! products accumulate in 32 bits.

#![allow(unsafe_code, clippy::inline_always)]

use super::warp::warp_taps_i16;
use super::{INTER_ROUND0, round2};
use crate::Isa;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::simd::{I16x8Shuffles, I32Lanes};

/// Side of the warped blocks.
pub(crate) const WARP_BLOCK: usize = 8;
/// Filter taps.
const TAPS: usize = 8;
/// Rows (and columns) of the source window: the block plus the filter reach.
pub(crate) const WINDOW_SIDE: usize = WARP_BLOCK + TAPS - 1;
/// Row stride of the source window: one spare column so that rows load as
/// two 8-sample vectors.
pub(crate) const WINDOW_STRIDE: usize = 16;
/// Bytes of a source window.
pub(crate) const WINDOW_LEN: usize = WINDOW_SIDE * WINDOW_STRIDE;

/// The filter phases of one 8x8 block (AV1 §7.11.3.5, `sx4`/`sy4` and the
/// shear steps): output `(r, c)` of the horizontal pass over window row
/// `i` uses phase `x.0 + x.1 · c + x.2 · (i − 3)`, of the vertical pass
/// `y.0 + y.1 · c + y.2 · r`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WarpPhases {
    /// Horizontal start, step per column (`alpha`) and per row (`beta`).
    pub(crate) x: (i32, i32, i32),
    /// Vertical start, step per column (`gamma`) and per row (`delta`).
    pub(crate) y: (i32, i32, i32),
}

impl WarpPhases {
    /// Phase of horizontal output column `c` on window row `i`.
    fn horizontal(&self, i: usize, c: usize) -> i32 {
        let (start, alpha, beta) = self.x;
        start + alpha * c as i32 + beta * (i as i32 - 3)
    }

    /// Phase of vertical output `(r, c)`.
    fn vertical(&self, r: usize, c: usize) -> i32 {
        let (start, gamma, delta) = self.y;
        start + gamma * c as i32 + delta * r as i32
    }
}

/// The 64 outputs of one block, row by row.
pub(crate) type WarpOut = [i32; WARP_BLOCK * WARP_BLOCK];

/// Warp one 8x8 block from its source window, rows of
/// [`WINDOW_STRIDE`] readable samples `stride` apart (the first
/// [`WINDOW_SIDE`] of each used): `out[r * 8 + c]` is the vertical sum of
/// output `(r, c)` rounded by `round1` bits (`1 ..= 11`).
pub(crate) type WarpFn =
    unsafe fn(window: (&[u8], usize), phases: &WarpPhases, round1: u32, out: &mut WarpOut);

/// Samples a window of rows `stride` apart spans.
pub(crate) fn window_span(stride: usize) -> usize {
    (WINDOW_SIDE - 1) * stride + WINDOW_STRIDE
}

/// The kernel of `isa` (AVX-512 tables inherit AVX2's).
pub(crate) fn kernel(isa: Isa) -> WarpFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => super::warp_avx2::warp8x8,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => super::warp_neon::warp8x8,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => warp8x8_simd128,
        _ => warp8x8_scalar,
    }
}

/// Scalar reference (AV1 §7.11.3.5, the spec's two loops).
pub(crate) fn warp8x8_scalar(
    (window, stride): (&[u8], usize),
    phases: &WarpPhases,
    round1: u32,
    out: &mut WarpOut,
) {
    let mut intermediate = [[0i32; WARP_BLOCK]; WINDOW_SIDE];
    for (i, row) in intermediate.iter_mut().enumerate() {
        let source = &window[i * stride..][..WINDOW_SIDE];
        for (c, value) in row.iter_mut().enumerate() {
            let taps = warp_taps_i16(phases.horizontal(i, c));
            let sum: i32 = taps
                .iter()
                .zip(&source[c..c + TAPS])
                .map(|(&t, &s)| i32::from(t) * i32::from(s))
                .sum();
            *value = round2(sum, INTER_ROUND0);
        }
    }
    for r in 0..WARP_BLOCK {
        for c in 0..WARP_BLOCK {
            let taps = warp_taps_i16(phases.vertical(r, c));
            let sum: i32 = taps
                .iter()
                .zip(&intermediate[r..r + TAPS])
                .map(|(&t, row)| i32::from(t) * row[c])
                .sum();
            out[r * WARP_BLOCK + c] = round2(sum, round1);
        }
    }
}

/// Four 8-tap dot products: `samples(q)` against the taps of `phase(q)`,
/// each rounded by `bits`.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn dot4<'a, V: I16x8Shuffles>(
    samples: impl Fn(usize) -> &'a [i16],
    phase: impl Fn(usize) -> i32,
    bits: u32,
) -> [i32; 4] {
    let products: [V::Wide; 4] =
        core::array::from_fn(|q| V::load(samples(q)).madd(V::load(warp_taps_i16(phase(q)))));
    let mut sums = [0; 4];
    V::sum_lanes4(products).round_shift(bits).store(&mut sums);
    sums
}

/// The SIMD kernel over 8-lane `V`.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn warp8x8_lanes<V: I16x8Shuffles>(
    (window, stride): (&[u8], usize),
    phases: &WarpPhases,
    round1: u32,
    out: &mut WarpOut,
) {
    // The horizontal pass, stored by column: `columns[c][i]` is output
    // column `c` of window row `i` (row 15 is padding).
    let mut columns = [[0i16; WINDOW_STRIDE]; WARP_BLOCK];
    let mut row = [0i16; WINDOW_STRIDE];
    for i in 0..WINDOW_SIDE {
        let source = &window[i * stride..][..WINDOW_STRIDE];
        V::load_u8(&source[..8]).store(&mut row[..8]);
        V::load_u8(&source[8..]).store(&mut row[8..]);
        for half in [0, 4] {
            // SAFETY: the caller's ISA contract.
            let sums = unsafe {
                dot4::<V>(
                    |q| &row[half + q..],
                    |q| phases.horizontal(i, half + q),
                    INTER_ROUND0,
                )
            };
            for (q, &sum) in sums.iter().enumerate() {
                // In 16 bits by the module docs' bound.
                columns[half + q][i] = sum as i16;
            }
        }
    }
    for (c, column) in columns.iter().enumerate() {
        for half in [0, 4] {
            // SAFETY: the caller's ISA contract.
            let sums = unsafe {
                dot4::<V>(
                    |q| &column[half + q..],
                    |q| phases.vertical(half + q, c),
                    round1,
                )
            };
            for (q, &sum) in sums.iter().enumerate() {
                out[(half + q) * WARP_BLOCK + c] = sum;
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn warp8x8_simd128(
    window: (&[u8], usize),
    phases: &WarpPhases,
    round1: u32,
    out: &mut WarpOut,
) {
    // SAFETY: SIMD128 is enabled for the whole module in this build.
    unsafe { warp8x8_lanes::<crate::simd::simd128::I16x8>(window, phases, round1, out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise the private output contract, not just the public single and
    /// compound paths' usual shifts. All phases are valid table indices,
    /// including endpoints 0/192: invalid internal phases are not promised
    /// scalar equivalence (the scalar accessor panics, AVX2 clamps).
    #[test]
    fn direct_kernel_rounding_and_exact_window() {
        let phases = [
            WarpPhases {
                x: (-65536, 0, 0),
                y: (-65536, 0, 0),
            },
            WarpPhases {
                x: (131_072, 0, 0),
                y: (131_072, 0, 0),
            },
            WarpPhases {
                x: (0, 0, 0),
                y: (0, 0, 0),
            },
            WarpPhases {
                x: (16384, 0, 0),
                y: (0, 0, 0),
            },
            WarpPhases {
                x: (-3000, 900, -700),
                y: (2000, -600, 500),
            },
            WarpPhases {
                x: (120_000, 600, -400),
                y: (-60000, 500, 300),
            },
        ];
        let mut saw_negative = false;
        for (stride, offset) in [(16, 0), (19, 1), (1280, 7)] {
            let mut storage = vec![0xA5; offset + window_span(stride)];
            for pattern in 0..4 {
                for i in 0..WINDOW_SIDE {
                    for c in 0..WINDOW_SIDE {
                        storage[offset + i * stride + c] = match pattern {
                            0 => 0,
                            1 => 255,
                            2 => ((i * 73 + c * 137 + i * c * 11) % 256) as u8,
                            _ => {
                                if c < 8 && warp_taps_i16(16384)[c] < 0 {
                                    255
                                } else {
                                    0
                                }
                            }
                        };
                    }
                }
                let window = &storage[offset..];
                for phase in &phases {
                    for round1 in 1..=11 {
                        let mut want = [0; 64];
                        warp8x8_scalar((window, stride), phase, round1, &mut want);
                        saw_negative |= want.iter().any(|&x| x < 0);
                        for isa in Isa::ALL.into_iter().filter(|isa| isa.is_supported()) {
                            let mut guarded = [i32::MIN; 66];
                            let out: &mut WarpOut = (&mut guarded[1..65]).try_into().unwrap();
                            // SAFETY: available ISA, exact documented window,
                            // valid table phases, full output and valid shift.
                            unsafe { kernel(isa)((window, stride), phase, round1, out) };
                            assert_eq!(
                                *out, want,
                                "{isa:?} stride{stride} pattern{pattern} phase{phase:?} shift{round1}"
                            );
                            assert_eq!(guarded[0], i32::MIN);
                            assert_eq!(guarded[65], i32::MIN);
                        }
                    }
                }
            }
        }
        assert!(saw_negative, "must exercise signed final rounding");
    }

    /// Centering makes every partial horizontal reduction fit a word, and
    /// compensation after the first round is exact even for negative sums.
    #[test]
    fn centered_horizontal_bound_and_compensation() {
        for index in 0..193 {
            let taps = warp_taps_i16((index - 64) << 10);
            assert_eq!(taps.iter().map(|&x| i32::from(x)).sum::<i32>(), 128);
            let l1 = taps.iter().map(|&x| i32::from(x).abs()).sum::<i32>();
            assert!(l1 <= 222);
            assert!(128 * l1 + 4 <= i32::from(i16::MAX));
            for positive in [false, true] {
                let mut unsigned = 0;
                let mut centered = 0;
                for &t in taps {
                    let sample = if (t >= 0) == positive { 255 } else { 0 };
                    unsigned += sample * i32::from(t);
                    centered += (sample - 128) * i32::from(t);
                }
                assert_eq!(round2(centered, 3) + 2048, round2(unsigned, 3));
            }
        }
    }

    /// The module docs' bound: every horizontal sum, rounded, fits 16 bits
    /// with the vertical sums of those in 32 bits.
    #[test]
    fn intermediates_fit_their_lanes() {
        let filters = 193;
        let worst = (0..filters)
            .map(|index| {
                let taps = warp_taps_i16((index - 64) << 10);
                taps.iter().map(|&t| i32::from(t).abs()).sum::<i32>()
            })
            .max()
            .unwrap();
        let horizontal = (255 * worst + 4) >> INTER_ROUND0;
        assert!(horizontal < i32::from(i16::MAX));
        assert!(i64::from(horizontal) * i64::from(worst) < i64::from(i32::MAX));
    }

    /// `warp_avx2`'s bound: in [`super::super::warp::WARPED_FILTERS_PAIRED`]
    /// order, every byte pair (`pmaddubsw`) and every group of four
    /// (`phaddw`) of taps times 8-bit samples stays within 16 bits.
    #[test]
    fn paired_taps_fit_16_bits() {
        let reach = |taps: &[i8]| {
            let (mut up, mut down) = (0, 0);
            for &t in taps {
                if t > 0 {
                    up += 255 * i32::from(t);
                } else {
                    down -= 255 * i32::from(t);
                }
            }
            (up, down)
        };
        for filter in &super::super::warp::WARPED_FILTERS_PAIRED {
            for group in filter.chunks(2).chain(filter.chunks(4)) {
                let (up, down) = reach(group);
                assert!(
                    up <= i32::from(i16::MAX) && down <= -i32::from(i16::MIN),
                    "{filter:?}"
                );
            }
        }
    }

    /// The scalar kernel equals the spec's loops on a flat window: every
    /// output is the sample at full precision (taps sum to 128 twice).
    #[test]
    fn flat_window_keeps_its_value() {
        let window = [77u8; WINDOW_SIDE * WINDOW_STRIDE];
        let phases = WarpPhases {
            x: (-3000, 900, -400),
            y: (2000, -700, 600),
        };
        let mut out = [0; WARP_BLOCK * WARP_BLOCK];
        warp8x8_scalar((&window, WINDOW_STRIDE), &phases, 11, &mut out);
        assert!(out.iter().all(|&v| v == 77));
    }
}
