//! The CDEF direction search written once over [`I16x8Shuffles`]: one
//! vector per block row (or column). Same results as the scalar reference,
//! libaom's `cdef_find_dir_c`; the vector formulation follows the idea of
//! libaom's `cdef_find_dir` SIMD version (BSD-2-Clause, Alliance for Open
//! Media).
//!
//! Each direction's partial sums along its lines are built in two vectors
//! (`lo` holds line indices 0..8, `hi` 8..16) by adding every block row
//! shifted by its line offset. The rows give directions 4..7 (mostly
//! vertical lines); the same code on the block rotated a quarter turn gives
//! 0..3. Costs are `sum(partial^2 * 840 / line length)` in 32 bits, which
//! cannot overflow for 8-bit samples (each is below `2^30`).

use crate::simd::{I16x8Shuffles, I32Lanes, ZERO_BYTE};

/// Block side.
const N: usize = 8;

/// Lane `i` from lane `6 - i` for `i < 7`, lane 7 zero: lines up the `hi`
/// half of a partial-sum pair with the `lo` lines of equal length.
const MIRROR_HI: [u8; 16] = {
    let mut pattern = [ZERO_BYTE; 16];
    let mut lane = 0;
    while lane < 7 {
        pattern[2 * lane] = (2 * (6 - lane)) as u8;
        pattern[2 * lane + 1] = (2 * (6 - lane) + 1) as u8;
        lane += 1;
    }
    pattern
};

/// `840 / line length` per mirrored lane pair of the diagonal directions
/// (0 and 4: lines of 1..=8 samples, then 7..=1).
const DIAGONAL_WEIGHTS: [[i32; 4]; 2] = [[840, 420, 280, 210], [168, 140, 120, 105]];
/// The same for the other non-axis directions (1, 3, 5, 7), whose partial
/// sums start at lane 2: lines of 2, 4, 6 samples, then five of 8.
const SKEW_WEIGHTS: [[i32; 4]; 2] = [[0, 0, 420, 210], [140, 105, 105, 105]];
/// `840 / 8` for the axis directions (2 and 6).
const AXIS_WEIGHT: i32 = 105;
/// Partial sums of the skewed directions start at this lane, so their
/// mirrored pairs line up with [`SKEW_WEIGHTS`].
const SKEW_START: usize = 2;

/// A partial-sum pair: line indices `0..8` and `8..16`.
#[derive(Clone, Copy)]
struct Partial<V> {
    lo: V,
    hi: V,
}

impl<V: I16x8Shuffles> Partial<V> {
    /// Add `row` to lines `offset..offset + 8` (`offset <= 8`).
    #[allow(clippy::inline_always)] // must inline into `#[target_feature]` kernels
    #[inline(always)]
    fn add(&mut self, row: V, offset: usize) {
        self.lo = self.lo.add(row.lanes_up(offset));
        self.hi = self.hi.add(row.lanes_down(N - offset));
    }

    /// `sum(partial^2 * weight)`, with each line paired with its mirror.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn cost(self, weights: &[[i32; 4]; 2]) -> V::Wide {
        let mirrored = self.hi.shuffle_bytes(&MIRROR_HI);
        let first = self.lo.zip_lo(mirrored);
        let second = self.lo.zip_hi(mirrored);
        first
            .madd(first)
            .mul(V::Wide::load(&weights[0]))
            .add(second.madd(second).mul(V::Wide::load(&weights[1])))
    }
}

/// Costs of directions 4, 5, 6 and 7 of the block whose rows are `rows`
/// (samples minus 128).
#[allow(clippy::inline_always)]
#[inline(always)]
fn vertical_costs<V: I16x8Shuffles>(rows: &[V; N]) -> V::Wide {
    let zero = V::splat(0);
    let empty = Partial { lo: zero, hi: zero };
    // libaom's partial[4][7 + i - j], mirrored (same cost): line 7 - i + j.
    let mut diagonal = empty;
    let mut axis = zero;
    for (i, &row) in rows.iter().enumerate() {
        diagonal.add(row, N - 1 - i);
        axis = axis.add(row);
    }
    // partial[5][3 - i / 2 + j] and partial[7][i / 2 + j], from row pairs.
    let (mut skew5, mut skew7) = (empty, empty);
    for k in 0..N / 2 {
        let pair = rows[2 * k].add(rows[2 * k + 1]);
        skew5.add(pair, SKEW_START + 3 - k);
        skew7.add(pair, SKEW_START + k);
    }
    V::sum_lanes4([
        diagonal.cost(&DIAGONAL_WEIGHTS),
        skew5.cost(&SKEW_WEIGHTS),
        axis.madd(axis).mul(V::Wide::splat(AXIS_WEIGHT)),
        skew7.cost(&SKEW_WEIGHTS),
    ])
}

/// The 8x8 transpose of `rows`: three rounds of interleaving row `k` with
/// row `k + 4`, each rotating the (row, lane) bit index left by one.
#[allow(clippy::inline_always)]
#[inline(always)]
fn transpose<V: I16x8Shuffles>(rows: [V; N]) -> [V; N] {
    let mut m = rows;
    for _ in 0..3 {
        let mut next = m;
        for k in 0..N / 2 {
            next[2 * k] = m[k].zip_lo(m[k + 4]);
            next[2 * k + 1] = m[k].zip_hi(m[k + 4]);
        }
        m = next;
    }
    m
}

/// libaom `cdef_find_dir` on the 8x8 block of 8-bit samples at the start of
/// `src` (row stride `stride`): the direction and the directional contrast.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn find_dir<V: I16x8Shuffles>(src: &[u8], stride: usize) -> (u8, i32) {
    let block = &src[..(N - 1) * stride + N];
    let mut rows = [V::splat(0); N];
    for (i, row) in rows.iter_mut().enumerate() {
        // SAFETY: the whole 8x8 footprint was checked above; these eight
        // lanes read exactly one row, avoiding repeated suffix-slice checks.
        *row = unsafe { V::load_u8_at(block.as_ptr().add(i * stride)) }.sub(V::splat(128));
    }
    // A quarter turn counter-clockwise: rotated row i is column 7 - i.
    let mut rotated = transpose(rows);
    rotated.reverse();
    let mut cost = [0i32; N];
    vertical_costs(&rotated).store(&mut cost[..4]);
    vertical_costs(&rows).store(&mut cost[4..]);
    let (mut best_dir, mut best_cost) = (0, 0);
    for (d, &c) in cost.iter().enumerate() {
        if c > best_cost {
            best_cost = c;
            best_dir = d;
        }
    }
    let var = (best_cost - cost[(best_dir + 4) & 7]) >> 10;
    (best_dir as u8, var)
}
