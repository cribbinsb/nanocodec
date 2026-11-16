//! SIMD angular prediction: each row's interpolation is one call of the
//! shared directional runs (`crate::intra_runs`: `run2` for chroma, as
//! HEVC, AV1 and AV2 use it; `run4` for luma's cubic and Gaussian filters),
//! then PDPC on the generic i16 lanes over the columns it reaches.
//! Reference preparation is shared with scalar. PDPC uses `VVenC`'s
//! `IntraPredX86.h::pdpcBlend` rounding multiply-high formulation
//! (BSD-3-Clause-Clear; see `THIRD_PARTY.md`); a PDPC term is at most
//! 32 x 255 in magnitude, so its 16-bit arithmetic is exact.
#![allow(unsafe_code, clippy::inline_always)]
use super::angular::{Angular, MAIN_LEN, MAX_SIDE};
use crate::intra_runs::Runs;
use crate::simd::I16Lanes;

/// PDPC reaches at most `3 << 2` columns (H.266 §8.4.5.2.15, nScale <= 2);
/// rounded up to whole 16-lane vectors.
const PDPC_SPAN: usize = 16;
/// Convert the normative denominator 64 to the Q15 multiply denominator.
const PDPC_Q15_SHIFT: u32 = 15 - 6;

/// Returns whether `out` holds the transposed block: horizontal directions
/// compute columns as rows, and with `transposed_ok` they are stored so.
///
/// # Safety
/// The ISA of `V` and `R` is enabled; side is divisible by `V`'s lane count
/// and buffers validated.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn predict<V: I16Lanes, R: Runs>(
    n: usize,
    direction: u8,
    luma: bool,
    top: &[i32],
    left: &[i32],
    corner: i32,
    transposed_ok: bool,
    out: &mut [u8],
) -> bool {
    let angular = Angular::new(n, direction, luma, top, left, corner);
    // References are 8-bit samples (`check_square`): narrowing is exact.
    let mut main = [0u8; MAIN_LEN];
    for (narrow, &wide) in main.iter_mut().zip(angular.main()) {
        *narrow = wide as u8;
    }
    // Only the first `pdpc_width` columns carry a non-zero PDPC weight, so
    // only the vectors covering them gather side samples and blend: the
    // rest of each row is interpolation alone. The side sample offsets are
    // the same on every row.
    let pdpc_width = angular
        .pdpc_scale
        .map_or(0, |scale| (3usize << scale).min(n));
    let pdpc_span = pdpc_width.div_ceil(V::LANES) * V::LANES;
    let mut weights = [0i16; PDPC_SPAN];
    let mut offsets = [0usize; PDPC_SPAN];
    if let Some(scale) = angular.pdpc_scale {
        for x in 0..pdpc_width {
            weights[x] = (32 >> ((2 * x) >> scale)) << PDPC_Q15_SHIFT;
            offsets[x] = angular.side_offset(x);
        }
    }
    let exact = angular.angle.abs() == 32;
    let in_place = angular.vertical || transposed_ok;
    let mut predicted = [0u8; MAX_SIDE];
    for y in 0..n {
        let start = angular.row_start(y);
        let phase = angular.phase(y);
        let side_row = angular.side_row(y);
        let mut side = [0i16; PDPC_SPAN];
        for x in 0..pdpc_width {
            side[x] = side_row[offsets[x]] as i16;
        }
        let line = if in_place {
            &mut out[y * n..(y + 1) * n]
        } else {
            &mut predicted[..n]
        };
        // SAFETY: the runs' ISA is enabled (caller contract); the windows
        // lie inside `main` (the extended reference has the last taps).
        unsafe {
            if exact {
                // A whole-sample angle copies `row[x + 1]`: the 2-tap run
                // at weight 32 from `row[x]`.
                R::run2(&main[start..=start + n], 32, line);
            } else if luma {
                let taps = angular.coefficients(phase).map(|c| c as i8);
                R::run4(&main[start..start + n + 3], taps, line);
            } else {
                R::run2(&main[start + 1..=start + 1 + n], phase as usize, line);
            }
        }
        for x in (0..pdpc_span).step_by(V::LANES) {
            // For d in -255..=255 and w in 0..=32, Q15 rounding is
            // exactly (d*w + 32) >> 6, including negative half ties:
            // (d*(w<<9) + 16384) >> 15. Neither operand can be MIN.
            let value = V::load_u8(&line[x..]);
            value
                .add(
                    V::load(&side[x..])
                        .sub(value)
                        .mul_q15_round(V::load(&weights[x..])),
                )
                .store_u8_sat(&mut line[x..x + V::LANES]);
        }
        if !in_place {
            for (x, &value) in predicted[..n].iter().enumerate() {
                out[x * n + y] = value;
            }
        }
    }
    !angular.vertical && transposed_ok
}
