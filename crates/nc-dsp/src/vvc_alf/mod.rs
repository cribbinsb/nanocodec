//! H.266 §8.8.5 adaptive loop filter (ALF) kernels for 8-bit luma: the 7x7
//! diamond filter of one block row (§8.8.5.2), the subsampled Laplacians that
//! classify 4x4 blocks (§8.8.5.3), and the Gram matrix of the encoder's
//! filter statistics.
//!
//! Control stays in the codec (`nc-vvc` `alf`): it pads the picture, picks
//! the rows a filter or Laplacian reads (virtual boundaries), classifies
//! blocks from the Laplacian sums and chooses coefficients. These kernels
//! see only rows of samples, so the encoder's reconstruction and our
//! decoder share them.
//!
//! Clipping (§8.8.5.2 `Clip3(−c, c, ·)` of each neighbour difference) is
//! the identity for clipping index 0, the only one a fixed filter set
//! uses (`c = 2^BitDepth`, and 8-bit differences are below it), so the
//! filter kernel takes the unclipped symmetric sums. Signalled filters
//! with other clipping indices need a clipped variant.
//!
//! Arithmetic follows `VVenC` `AdaptiveLoopFilter` (BSD-3-Clause-Clear;
//! `THIRD_PARTY.md`).
//!
//! # Kernel contract
//!
//! Kernels are `unsafe fn`s called only from [`crate::Dsp`], which checks
//! the CPU and the slice lengths documented on each type below. Loads and
//! stores index their slices, so a broken contract panics rather than reads
//! out of bounds.

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon_dotprod;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;

mod clipped;
mod cross;

use crate::Isa;

/// Coefficients of the 7x7 luma diamond besides the centre; each weights
/// a symmetric pair of samples.
pub const VVC_ALF_TAPS: usize = 12;

/// `(dx, dy)` of the first sample of each tap's pair; the second is at
/// `(−dx, −dy)`. Coefficient `k` of a filter weights tap `k`.
pub const VVC_ALF_TAP_OFFSETS: [(isize, isize); VVC_ALF_TAPS] = [
    (0, 3),
    (1, 2),
    (0, 2),
    (-1, 2),
    (2, 1),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-2, 1),
    (3, 0),
    (2, 0),
    (1, 0),
];

/// Coefficient order of the transposed filters (§8.8.5.2 `transposeIdx`
/// 0..3): transpose `t` applies coefficient `VVC_ALF_TRANSPOSES[t][k]` to
/// tap `k`, so a statistics sample's tap sum `k` belongs to that
/// coefficient.
pub const VVC_ALF_TRANSPOSES: [[usize; VVC_ALF_TAPS]; 4] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [9, 4, 10, 8, 1, 5, 11, 7, 3, 0, 2, 6],
    [0, 3, 2, 1, 8, 7, 6, 5, 4, 9, 10, 11],
    [9, 8, 10, 4, 3, 7, 11, 5, 1, 0, 2, 6],
];

/// Columns the filter reads on each side of an output sample (and rows
/// above and below it).
pub const VVC_ALF_REACH: usize = 3;

/// Filter and classification block side: one coefficient set and one
/// class per 4x4 block.
pub const VVC_ALF_BLOCK: usize = 4;

/// Statistics samples of one [`TapSumsFn`] call: every
/// [`VVC_ALF_SAMPLE_STEP`]th column of a 64-sample CTB row.
pub const VVC_ALF_SAMPLES: usize = 16;

/// Columns between the encoder's statistics samples on a row.
pub const VVC_ALF_SAMPLE_STEP: usize = 4;

/// Lanes of a statistics sample: the twelve tap sums in the order of the
/// untransposed coefficients, the reconstruction error (lane
/// [`VVC_ALF_TAPS`]), then zeros (one 256-bit vector).
pub const VVC_ALF_SAMPLE_LANES: usize = 16;

/// Values of a statistics sample that the Gram matrix covers: the tap
/// sums and the error.
pub const VVC_ALF_GRAM_COLUMNS: usize = VVC_ALF_TAPS + 1;

/// Gram kernels take samples in pairs; callers pad with a zero sample,
/// which adds nothing to the sums.
pub const VVC_ALF_GRAM_ALIGN: usize = 2;

/// Most samples one Gram call may take: with every value within
/// [`VVC_ALF_GRAM_MAX_MAGNITUDE`], every entry (at most 2^12 products of
/// at most 2^18) fits 32 bits.
pub const VVC_ALF_GRAM_MAX_SAMPLES: usize = 4096;

/// Largest magnitude of a statistics value: an 8-bit tap sum
/// (`a + b − 2c`, at most 510) or error (at most 255).
pub const VVC_ALF_GRAM_MAX_MAGNITUDE: i16 = 512;

/// One block row (four rows) of the luma filter: `rows[r]` are the seven
/// rows the taps of output row `r` read (`dy` −3..=3, already mirrored at
/// any virtual boundary), each starting [`VVC_ALF_REACH`] columns left of
/// `out[r][0]` and at least `width + 2 * VVC_ALF_REACH` long, where every
/// `out[r]` is `width` long; `coefficients[i]` (each within
/// −128..=127, the range of §7.4.3.18 `AlfCoeffL`) filters columns
/// `4i..4i + 4` of every row; row `r`'s sum is rounded by `shifts[r]`
/// (1..=16). A 4x4 block keeps its coefficients for all four rows, so
/// kernels build their coefficient vectors once per block row.
pub(crate) type FilterBlockRowFn = unsafe fn(
    rows: &[[&[u8]; 7]; VVC_ALF_BLOCK],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    shifts: &[u32; VVC_ALF_BLOCK],
    out: &mut [&mut [u8]; VVC_ALF_BLOCK],
);

/// Laplacian sums of one classification grid row (see [`laplacians`]):
/// `rows` are the rows above, at, below and two below the grid row, each
/// starting 3 columns left of column 0 (group 0 covers columns −2..=1) and
/// at least `4 * out.len() + 2` long.
pub(crate) type LaplacianFn = unsafe fn(rows: &[&[u8]; 4], out: &mut [[u16; 4]]);

/// Classes of a row of 4x4 blocks from the Laplacian group sums summed over
/// the block row's grid rows (see [`classify`]): `out[b] = [class index,
/// transpose index]` of block `b` from groups `b` and `b + 1`
/// (`groups.len() == out.len() + 1`); `boundary`: three grid rows next to
/// a virtual boundary.
pub(crate) type ClassifyFn = unsafe fn(groups: &[[u16; 4]], boundary: bool, out: &mut [[u8; 2]]);

/// The encoder's statistics samples at columns `VVC_ALF_SAMPLE_STEP · i`,
/// `i < count ≤ VVC_ALF_SAMPLES` (see [`samples`]): `rows` as for
/// [`FilterRowFn`], each at least
/// `VVC_ALF_SAMPLE_STEP * count + 2 * VVC_ALF_REACH` long; `source` the
/// original row from column 0, reaching column
/// `VVC_ALF_SAMPLE_STEP * (count − 1)` (SIMD kernels read 64 bytes only
/// when it is that long); `transposes[i]` sample `i`'s transpose index
/// (0..4).
pub(crate) type SamplesFn = unsafe fn(
    rows: &[&[u8]; 7],
    source: &[u8],
    transposes: &[u8; VVC_ALF_SAMPLES],
    count: usize,
    out: &mut [[i16; VVC_ALF_SAMPLE_LANES]; VVC_ALF_SAMPLES],
);

/// The upper triangle of the Gram matrix `Σ_n samples[n][i] ·
/// samples[n][j]` over the first [`VVC_ALF_GRAM_COLUMNS`] lanes of the
/// samples (a multiple of [`VVC_ALF_GRAM_ALIGN`], at most
/// [`VVC_ALF_GRAM_MAX_SAMPLES`], values within
/// [`VVC_ALF_GRAM_MAX_MAGNITUDE`]), packed row by row (`out[gram_index(i,
/// j)]` for `i ≤ j`), the rest of `out` zero.
pub(crate) type GramFn =
    unsafe fn(samples: &[[i16; VVC_ALF_SAMPLE_LANES]], out: &mut [i32; VVC_ALF_DOT_LANES]);

/// Entries of the packed upper triangle of the Gram matrix.
pub const VVC_ALF_GRAM_TERMS: usize = VVC_ALF_GRAM_COLUMNS * (VVC_ALF_GRAM_COLUMNS + 1) / 2;

/// Where Gram entry `(i, j)`, `i ≤ j`, lies in the packed upper triangle.
#[must_use]
pub const fn vvc_alf_gram_index(i: usize, j: usize) -> usize {
    i * VVC_ALF_GRAM_COLUMNS - i * (i + 1) / 2 + j
}

/// Length of the encoder decision's dot products: the packed upper
/// triangle of a class's Gram matrix ([`VVC_ALF_GRAM_TERMS`] entries),
/// padded with zeros to whole vectors.
pub const VVC_ALF_DOT_LANES: usize = 96;

/// `Σ a[k] · b[k]` in 64 bits (wrapping, which no ALF decision reaches).
pub(crate) type DotFn =
    unsafe fn(a: &[i32; VVC_ALF_DOT_LANES], b: &[i32; VVC_ALF_DOT_LANES]) -> i64;

/// The ALF kernels of one ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Kernels {
    pub filter_block_row: FilterBlockRowFn,
    pub laplacians: LaplacianFn,
    pub classify: ClassifyFn,
    pub samples: SamplesFn,
    pub gram: GramFn,
    pub dot: DotFn,
}

pub(crate) fn kernels(isa: Isa) -> Kernels {
    match isa {
        Isa::Scalar => Kernels {
            filter_block_row,
            laplacians,
            classify,
            samples,
            gram,
            dot,
        },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => Kernels {
            filter_block_row: avx2::filter_block_row,
            laplacians: avx2::laplacians,
            classify: avx2::classify,
            samples: avx2::samples,
            gram: avx2::gram,
            dot: avx2::dot,
        },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => Kernels {
            // The dot-product extension (Armv8.2, e.g. the Cortex-A78) is
            // optional in the architecture, so it is detected at run time.
            filter_block_row: if std::arch::is_aarch64_feature_detected!("dotprod") {
                neon_dotprod::filter_block_row
            } else {
                neon::filter_block_row
            },
            laplacians: neon::laplacians,
            classify: neon::classify,
            samples: neon::samples,
            gram: neon::gram,
            dot: neon::dot,
        },
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => Kernels {
            filter_block_row: simd128::filter_block_row,
            laplacians: simd128::laplacians,
            classify: simd128::classify,
            samples: simd128::samples,
            gram: simd128::gram,
            dot: simd128::dot,
        },
        #[allow(unreachable_patterns)]
        _ => unreachable!("unsupported ISA"),
    }
}

/// Scalar [`FilterBlockRowFn`]: each row by [`filter_row`].
fn filter_block_row(
    rows: &[[&[u8]; 7]; VVC_ALF_BLOCK],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    shifts: &[u32; VVC_ALF_BLOCK],
    out: &mut [&mut [u8]; VVC_ALF_BLOCK],
) {
    for ((rows, &shift), out) in rows.iter().zip(shifts).zip(out.iter_mut()) {
        filter_row(rows, coefficients, shift, out);
    }
}

/// One row of the filter: `centre + round(Σ c_k (a_k + b_k − 2·centre))`,
/// clamped to 8 bits (§8.8.5.2, equation 1427 with clipping index 0).
/// `rows` and `coefficients` as for one row of [`FilterBlockRowFn`].
fn filter_row(rows: &[&[u8]; 7], coefficients: &[[i16; VVC_ALF_TAPS]], shift: u32, out: &mut [u8]) {
    let sample = |dy: isize, x: usize, dx: isize| {
        i32::from(rows[(3 + dy) as usize][(x as isize + VVC_ALF_REACH as isize + dx) as usize])
    };
    for (x, out) in out.iter_mut().enumerate() {
        let centre = sample(0, x, 0);
        let sum: i32 = VVC_ALF_TAP_OFFSETS
            .iter()
            .zip(&coefficients[x / VVC_ALF_BLOCK])
            .map(|(&(dx, dy), &c)| {
                i32::from(c) * (sample(dy, x, dx) + sample(-dy, x, -dx) - 2 * centre)
            })
            .sum();
        *out = (centre + ((sum + (1 << (shift - 1))) >> shift)).clamp(0, 255) as u8;
    }
}

/// Scalar [`LaplacianFn`]. §8.8.5.3 evaluates the vertical, horizontal and
/// two diagonal Laplacians `|2c − a − b|` at the 2x2-subsampled positions
/// (x, y) and (x + 1, y + 1) of each grid point, with x and y even. Along
/// grid row y this is a checkerboard: even columns centred on row y, odd
/// columns on row y + 1. Group `q` sums the columns `4q − 2 .. 4q + 1` (two
/// grid points) into `[vertical, horizontal, diagonal 0, diagonal 1]`; a
/// 4x4 block at column 4b reads groups b and b + 1.
fn laplacians(rows: &[&[u8]; 4], out: &mut [[u16; 4]]) {
    for (q, out) in out.iter_mut().enumerate() {
        let mut sums = [0; 4];
        for k in 0..4 {
            // Column 4q − 2 + k, at index 4q + 1 + k of rows starting at −3.
            let i = 4 * q + 1 + k;
            let [up, centre, down] = if k % 2 == 0 {
                [rows[0], rows[1], rows[2]]
            } else {
                [rows[1], rows[2], rows[3]]
            };
            let laplacian = |a: u8, b: u8| {
                (2 * i32::from(centre[i]) - i32::from(a) - i32::from(b)).unsigned_abs()
            };
            let terms = [
                laplacian(up[i], down[i]),
                laplacian(centre[i - 1], centre[i + 1]),
                laplacian(up[i - 1], down[i + 1]),
                laplacian(down[i - 1], up[i + 1]),
            ];
            for (sum, term) in sums.iter_mut().zip(terms) {
                *sum += term as u16;
            }
        }
        *out = sums;
    }
}

/// Activity quantisation (§8.8.5.3: `ac`, 8-bit `BitDepth + 4` shift),
/// scaled by 64, or 96 for the three Laplacian rows next to a virtual
/// boundary.
const ACTIVITY_SHIFT: u32 = 12;
const ACTIVITY_SCALE: u32 = 64;
const BOUNDARY_ACTIVITY_SCALE: u32 = 96;
const MAX_ACTIVITY: u32 = 15;
const ACTIVITY_CLASS: [u8; 16] = [0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4];
/// The transpose index of a block by its main and secondary directions.
const TRANSPOSE: [u8; 8] = [0, 1, 0, 2, 2, 3, 1, 3];
/// Directional classes add this many activity classes per step.
const ACTIVITY_CLASSES: u8 = 5;

/// One block's `[class index, transpose index]` (§8.8.5.3) from its
/// Laplacian sums `[vertical, horizontal, diagonal 0, diagonal 1]`.
fn class(sums: [u32; 4], boundary: bool) -> [u8; 2] {
    let [vertical, horizontal, diagonal0, diagonal1] = sums;
    let scale = if boundary {
        BOUNDARY_ACTIVITY_SCALE
    } else {
        ACTIVITY_SCALE
    };
    let activity = (((vertical + horizontal) * scale) >> ACTIVITY_SHIFT).min(MAX_ACTIVITY);
    let mut index = ACTIVITY_CLASS[activity as usize];
    let (hv1, hv0, hv_direction) = if vertical > horizontal {
        (vertical, horizontal, 1)
    } else {
        (horizontal, vertical, 3)
    };
    let (d1, d0, d_direction) = if diagonal0 > diagonal1 {
        (diagonal0, diagonal1, 0)
    } else {
        (diagonal1, diagonal0, 2)
    };
    let diagonal_wins = d1 * hv0 > hv1 * d0;
    let (strong, weak, main, secondary) = if diagonal_wins {
        (d1, d0, d_direction, hv_direction)
    } else {
        (hv1, hv0, hv_direction, d_direction)
    };
    let strength = if strong * 2 > 9 * weak {
        2
    } else {
        u8::from(strong > 2 * weak)
    };
    if strength > 0 {
        index += (((main & 1) << 1) + strength) * ACTIVITY_CLASSES;
    }
    [index, TRANSPOSE[usize::from(main * 2 + (secondary >> 1))]]
}

/// Largest Laplacian group sum over a block row's grid rows: four grid
/// rows of a group's four Laplacians (two points, two subsampled positions
/// each, each at most 510). Block sums (two groups) stay below 2^15, so
/// every product in [`class`] fits 32 bits.
pub const VVC_ALF_MAX_GROUP_SUM: u16 = 4 * 4 * 510;

/// Scalar [`ClassifyFn`].
fn classify(groups: &[[u16; 4]], boundary: bool, out: &mut [[u8; 2]]) {
    for (out, pair) in out.iter_mut().zip(groups.windows(2)) {
        *out = class(
            std::array::from_fn(|d| u32::from(pair[0][d]) + u32::from(pair[1][d])),
            boundary,
        );
    }
}

/// The [`ClassifyFn`] arguments from block `start` on.
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
fn classify_from(groups: &[[u16; 4]], boundary: bool, out: &mut [[u8; 2]], start: usize) {
    if start < out.len() {
        classify(&groups[start..], boundary, &mut out[start..]);
    }
}

/// Scalar [`SamplesFn`]: sample `i` holds, at lane
/// `VVC_ALF_TRANSPOSES[transposes[i]][k]`, the tap sum `a + b − 2c` of tap
/// `k` (unweighted, at most 510 in magnitude), at lane [`VVC_ALF_TAPS`] the
/// error `c − source`, zeros after.
fn samples(
    rows: &[&[u8]; 7],
    source: &[u8],
    transposes: &[u8; VVC_ALF_SAMPLES],
    count: usize,
    out: &mut [[i16; VVC_ALF_SAMPLE_LANES]; VVC_ALF_SAMPLES],
) {
    for (i, out) in out[..count].iter_mut().enumerate() {
        let x = VVC_ALF_SAMPLE_STEP * i + VVC_ALF_REACH;
        let sample =
            |dy: isize, dx: isize| i16::from(rows[(3 + dy) as usize][(x as isize + dx) as usize]);
        let centre = sample(0, 0);
        *out = [0; VVC_ALF_SAMPLE_LANES];
        let order = &VVC_ALF_TRANSPOSES[usize::from(transposes[i])];
        for (&lane, &(dx, dy)) in order.iter().zip(&VVC_ALF_TAP_OFFSETS) {
            out[lane] = sample(dy, dx) + sample(-dy, -dx) - 2 * centre;
        }
        out[VVC_ALF_TAPS] = centre - i16::from(source[VVC_ALF_SAMPLE_STEP * i]);
    }
}

/// For each transpose index, the byte of a natural-order sample (tap sums
/// in tap order, then the error; 32 bytes) that each byte of the sample in
/// coefficient order takes, or 255 for the zero lanes: what the SIMD
/// sample kernels' byte shuffles use.
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
const SAMPLE_BYTE_SOURCES: [[u8; 2 * VVC_ALF_SAMPLE_LANES]; 4] = sample_byte_sources();

const fn sample_byte_sources() -> [[u8; 2 * VVC_ALF_SAMPLE_LANES]; 4] {
    let mut sources = [[255; 2 * VVC_ALF_SAMPLE_LANES]; 4];
    let mut t = 0;
    while t < 4 {
        let mut k = 0;
        while k <= VVC_ALF_TAPS {
            // Tap k moves to its coefficient's lane; the error stays.
            let lane = if k < VVC_ALF_TAPS {
                VVC_ALF_TRANSPOSES[t][k]
            } else {
                k
            };
            sources[t][2 * lane] = 2 * k as u8;
            sources[t][2 * lane + 1] = 2 * k as u8 + 1;
            k += 1;
        }
        t += 1;
    }
    sources
}

/// Scalar [`DotFn`].
fn dot(a: &[i32; VVC_ALF_DOT_LANES], b: &[i32; VVC_ALF_DOT_LANES]) -> i64 {
    a.iter().zip(b).fold(0i64, |sum, (&a, &b)| {
        sum.wrapping_add(i64::from(a) * i64::from(b))
    })
}

/// Scalar [`GramFn`].
fn gram(samples: &[[i16; VVC_ALF_SAMPLE_LANES]], out: &mut [i32; VVC_ALF_DOT_LANES]) {
    *out = [0; VVC_ALF_DOT_LANES];
    for i in 0..VVC_ALF_GRAM_COLUMNS {
        for j in i..VVC_ALF_GRAM_COLUMNS {
            out[vvc_alf_gram_index(i, j)] = samples
                .iter()
                .map(|v| i32::from(v[i]) * i32::from(v[j]))
                .sum();
        }
    }
}

/// The [`FilterBlockRowFn`] arguments from column `start` on (a multiple
/// of [`VVC_ALF_BLOCK`]): what a SIMD kernel hands the scalar code for the
/// columns after its last whole vector.
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
fn filter_block_row_from(
    rows: &[[&[u8]; 7]; VVC_ALF_BLOCK],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    shifts: &[u32; VVC_ALF_BLOCK],
    out: &mut [&mut [u8]; VVC_ALF_BLOCK],
    start: usize,
) {
    for ((rows, &shift), out) in rows.iter().zip(shifts).zip(out.iter_mut()) {
        if start < out.len() {
            filter_row(
                &rows.map(|row| &row[start..]),
                &coefficients[start / VVC_ALF_BLOCK..],
                shift,
                &mut out[start..],
            );
        }
    }
}

/// The [`LaplacianFn`] arguments from group `start` on.
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
fn laplacians_from(rows: &[&[u8]; 4], out: &mut [[u16; 4]], start: usize) {
    if start < out.len() {
        laplacians(&rows.map(|row| &row[4 * start..]), &mut out[start..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dsp;

    fn noise(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    #[test]
    fn filter_of_a_flat_area_is_the_identity() {
        let row = vec![77u8; 70];
        let rows = [row.as_slice(); 7];
        let coefficients = vec![[5i16; VVC_ALF_TAPS]; 16];
        let mut out = vec![0; 64];
        filter_row(&rows, &coefficients, 7, &mut out);
        assert!(out.iter().all(|&v| v == 77));
    }

    #[test]
    fn laplacians_of_an_impulse() {
        // One bright sample at column 0 of the centre row (index 3).
        let flat = vec![0u8; 16];
        let mut centre = flat.clone();
        centre[3] = 10;
        let rows = [
            flat.as_slice(),
            centre.as_slice(),
            flat.as_slice(),
            flat.as_slice(),
        ];
        let mut out = [[0; 4]; 2];
        laplacians(&rows, &mut out);
        // Column 0 is even (centred on the row): every Laplacian sees 2·10
        // there; its odd neighbours (centred one row down) see the impulse
        // only diagonally, and group 0 holds columns −2..1.
        assert_eq!(out[0], [20, 20, 30, 30]);
        assert_eq!(out[1], [0, 0, 0, 0]);
    }

    #[test]
    fn every_isa_matches_the_scalar_kernels() {
        let scalar = Dsp::with_isa(Isa::Scalar).expect("scalar");
        let best = Dsp::new();
        for width in [4, 8, 12, 16, 20, 28, 64, 68] {
            let data: Vec<Vec<u8>> = (0..7).map(|r| noise(r + width as u32, width + 6)).collect();
            let rows: [&[u8]; 7] = std::array::from_fn(|r| data[r].as_slice());
            let coefficients: Vec<[i16; VVC_ALF_TAPS]> = (0..width.div_ceil(4))
                .map(|b| std::array::from_fn(|k| (b * 7 + k * 13) as i16 % 41 - 20))
                .collect();
            // Output row r reads the rows rotated by r, with its own shift.
            let block_rows: [[&[u8]; 7]; VVC_ALF_BLOCK] =
                std::array::from_fn(|r| std::array::from_fn(|i| rows[(i + r) % 7]));
            let shifts = [7, 10, 7, 9];
            let run = |dsp: &Dsp| {
                let mut out: [Vec<u8>; VVC_ALF_BLOCK] = std::array::from_fn(|_| vec![0; width]);
                dsp.vvc_alf_filter_block_row(
                    &block_rows,
                    &coefficients,
                    &shifts,
                    &mut out.each_mut().map(Vec::as_mut_slice),
                );
                out
            };
            assert_eq!(run(&best), run(&scalar), "filter width {width}");
            let count = (width / VVC_ALF_SAMPLE_STEP).min(VVC_ALF_SAMPLES);
            let transposes = std::array::from_fn(|i| (i * 7 % 4) as u8);
            let run = |dsp: &Dsp| {
                let mut out = [[0; VVC_ALF_SAMPLE_LANES]; VVC_ALF_SAMPLES];
                dsp.vvc_alf_samples(&rows, &data[6], &transposes, count, &mut out);
                out
            };
            assert_eq!(run(&best), run(&scalar), "samples width {width}");
            // Rows of width + 6 samples hold width / 4 + 1 groups.
            let groups = width / 4 + 1;
            let lrows: [&[u8]; 4] = std::array::from_fn(|r| data[r].as_slice());
            let run = |dsp: &Dsp| {
                let mut out = vec![[0; 4]; groups];
                dsp.vvc_alf_laplacians(&lrows, &mut out);
                out
            };
            assert_eq!(run(&best), run(&scalar), "laplacians width {width}");
            let groups: Vec<[u16; 4]> = data[0]
                .chunks(4)
                .map(|c| std::array::from_fn(|d| u16::from(c[d % c.len()]) * 16))
                .collect();
            for boundary in [false, true] {
                let run = |dsp: &Dsp| {
                    let mut out = vec![[0; 2]; groups.len() - 1];
                    dsp.vvc_alf_classify(&groups, boundary, &mut out);
                    out
                };
                assert_eq!(run(&best), run(&scalar), "classify width {width}");
            }
        }
        let values = noise(9, VVC_ALF_SAMPLE_LANES * 64);
        for len in [0, 2, 4, 32, 64] {
            let samples: Vec<[i16; VVC_ALF_SAMPLE_LANES]> = (0..len)
                .map(|n| {
                    std::array::from_fn(|c| {
                        (i16::from(values[n * VVC_ALF_SAMPLE_LANES + c]) - 128) * 4
                    })
                })
                .collect();
            let run = |dsp: &Dsp| {
                let mut out = [7; VVC_ALF_DOT_LANES];
                dsp.vvc_alf_gram(&samples, &mut out);
                out
            };
            assert_eq!(run(&best), run(&scalar), "gram length {len}");
        }
        let a: [i32; VVC_ALF_DOT_LANES] = std::array::from_fn(|k| (k as i32 * 7919 % 4051) - 2025);
        let b: [i32; VVC_ALF_DOT_LANES] =
            std::array::from_fn(|k| (k as i32 * 104_729 % 1_000_003) * 1000 - 500_000_000);
        assert_eq!(best.vvc_alf_dot(&a, &b), scalar.vvc_alf_dot(&a, &b));
    }
}
