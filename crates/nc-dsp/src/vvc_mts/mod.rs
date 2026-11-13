//! H.266 primary transforms of any kind pair (§8.7.4.2 DCT-II, DST-VII and
//! DCT-VIII, multiple transform selection) for 4- to 32-point sides: the
//! normative inverse after scaling, and the encoder's forward.
//!
//! Each pass is a plain integer matrix product with the normative bases
//! (`codec_txfm::vvc_primary`, from `VVdeC`'s `RomTr.cpp`, BSD-3-Clause-Clear;
//! `THIRD_PARTY.md`). The 32-point DST-VII and DCT-VIII keep only their
//! first 16 frequencies (§7.3.11.11 zero-out): the inverse ignores the rest,
//! the forward writes zeros there. The inverse matches the scalar
//! `vvc_primary::inverse` exactly (tests below); the forward follows the
//! HEVC / `VVenC` `TrQuant::xT` scaling, so a DCT-II pair equals
//! `codec_txfm::vvc_forward`.
//!
//! Ranges: inverse inputs are dequantised coefficients within 16 bits and
//! the first pass clips to 16 bits; forward inputs are 8-bit residuals
//! (`|r| <= 255`), whose row pass stays within ±32640 + rounding (the
//! largest absolute row sum of any basis, 2^(log2 N + 6), times 255, over
//! the 2^(log2 N − 1) shift; tests below). So every product operand fits
//! 16 bits and every sum 32, which the lane kernels (`generic`) rely on.
//!
//! Kernels are `unsafe fn`s called only from [`crate::Dsp`], which checks
//! the CPU, the block shape and the slice lengths.

#[allow(unsafe_code)]
mod generic;

use crate::codec_txfm::vvc_primary::tables::{DST7_8, DST7_16, DST7_32};
use crate::codec_txfm::{DCT32, DST4};
use crate::{Dsp, Isa, VvcTransformKind};

/// Largest transform side of these kernels.
const MAX_SIDE: usize = 32;
/// Every side's basis is stored in a `MAX_SIDE` square, row `k` holding
/// frequency `k` (sample `n` at column `n`).
type Basis = [[i16; MAX_SIDE]; MAX_SIDE];

/// Basis entry `(frequency, sample)` of an `n`-point transform `kind`
/// (§8.7.4.5 `transMatrix`): the DCT-II nests in the 32-point one, the
/// DCT-VIII is the DST-VII reflected, its odd frequencies negated.
const fn entry(kind: VvcTransformKind, n: usize, frequency: usize, sample: usize) -> i16 {
    match kind {
        VvcTransformKind::Dct2 => DCT32[frequency * MAX_SIDE / n][sample] as i16,
        VvcTransformKind::Dst7 => match n {
            4 => DST4[frequency][sample] as i16,
            8 => DST7_8[frequency][sample],
            16 => DST7_16[frequency][sample],
            _ => DST7_32[frequency][sample],
        },
        VvcTransformKind::Dct8 => {
            let reflected = entry(VvcTransformKind::Dst7, n, frequency, n - 1 - sample);
            if frequency & 1 == 0 {
                reflected
            } else {
                -reflected
            }
        }
    }
}

/// The `n`-point basis of `kind`, or its transpose (row `n` holding every
/// frequency's sample `n`).
const fn build(kind: VvcTransformKind, n: usize, transpose: bool) -> Basis {
    let mut out = [[0; MAX_SIDE]; MAX_SIDE];
    let mut k = 0;
    while k < n {
        let mut s = 0;
        while s < n {
            let value = entry(kind, n, k, s);
            if transpose {
                out[s][k] = value;
            } else {
                out[k][s] = value;
            }
            s += 1;
        }
        k += 1;
    }
    out
}

/// Bases by kind (DCT-II, DST-VII, DCT-VIII) and log2 side − 2.
#[allow(
    clippy::large_stack_arrays,
    reason = "evaluated at compile time into a static"
)]
const fn bases(transpose: bool) -> [[Basis; 4]; 3] {
    use VvcTransformKind::{Dct2, Dct8, Dst7};
    let mut out = [[[[0; MAX_SIDE]; MAX_SIDE]; 4]; 3];
    let kinds = [Dct2, Dst7, Dct8];
    let mut i = 0;
    while i < kinds.len() {
        let mut log2 = 0;
        while log2 < 4 {
            out[i][log2] = build(kinds[i], 4 << log2, transpose);
            log2 += 1;
        }
        i += 1;
    }
    out
}

/// Two consecutive rows of a [`Basis`] interleaved: `pairs[p][2 * s + j]`
/// is row `2p + j`, column `s` (the lane kernels' right-hand sides).
type PairTable = [[i16; 2 * MAX_SIDE]; MAX_SIDE / 2];

const fn interleave(basis: &Basis) -> PairTable {
    let mut out = [[0; 2 * MAX_SIDE]; MAX_SIDE / 2];
    let mut row = 0;
    while row < MAX_SIDE {
        let mut column = 0;
        while column < MAX_SIDE {
            out[row / 2][2 * column + row % 2] = basis[row][column];
            column += 1;
        }
        row += 1;
    }
    out
}

/// [`bases`] interleaved.
#[allow(
    clippy::large_stack_arrays,
    reason = "evaluated at compile time into a static"
)]
const fn pair_tables(transpose: bool) -> [[PairTable; 4]; 3] {
    let bases = bases(transpose);
    let mut out = [[[[0; 2 * MAX_SIDE]; MAX_SIDE / 2]; 4]; 3];
    let mut i = 0;
    while i < 3 {
        let mut log2 = 0;
        while log2 < 4 {
            out[i][log2] = interleave(&bases[i][log2]);
            log2 += 1;
        }
        i += 1;
    }
    out
}

static BASES: [[Basis; 4]; 3] = bases(false);
static FREQUENCY_PAIRS: [[PairTable; 4]; 3] = pair_tables(false);
static SAMPLE_PAIRS: [[PairTable; 4]; 3] = pair_tables(true);

fn table_index(kind: VvcTransformKind, n: usize) -> (usize, usize) {
    let kind = match kind {
        VvcTransformKind::Dct2 => 0,
        VvcTransformKind::Dst7 => 1,
        VvcTransformKind::Dct8 => 2,
    };
    (kind, n.ilog2() as usize - 2)
}

/// The `n`-point basis of `kind`: `basis(kind, n)[k][s]` weights sample `s`
/// in frequency `k`.
fn basis(kind: VvcTransformKind, n: usize) -> &'static Basis {
    let (kind, log2) = table_index(kind, n);
    &BASES[kind][log2]
}

/// [`basis`] with frequencies interleaved in pairs.
fn frequency_pairs(kind: VvcTransformKind, n: usize) -> &'static PairTable {
    let (kind, log2) = table_index(kind, n);
    &FREQUENCY_PAIRS[kind][log2]
}

/// [`basis`] transposed (rows by sample), with samples interleaved in pairs.
fn sample_pairs(kind: VvcTransformKind, n: usize) -> &'static PairTable {
    let (kind, log2) = table_index(kind, n);
    &SAMPLE_PAIRS[kind][log2]
}

/// §8.7.4.2: the first inverse stage rounds by 7 bits and clips to 16.
const INVERSE_FIRST_SHIFT: u32 = 7;
/// §8.7.4.1 `bdShift` at 8-bit: 20 − `BitDepth`.
const INVERSE_SECOND_SHIFT: u32 = 12;
/// The forward column pass shift beyond log2(height) (`VVenC` `TrQuant::xT`:
/// `TRANSFORM_MATRIX_SHIFT`).
const FORWARD_MATRIX_SHIFT: u32 = 6;

/// Rows and columns, from the first, holding a nonzero coefficient among the
/// first `rows` x `cols` of a `width`-wide block (0 when none): the inverse
/// passes skip the rest, which contributes nothing.
fn nonzero_extent(coefficients: &[i32], width: usize, rows: usize, cols: usize) -> (usize, usize) {
    let mut extent: (usize, usize) = (0, 0);
    for (r, line) in coefficients.chunks_exact(width).take(rows).enumerate() {
        if let Some(c) = line[..cols].iter().rposition(|&v| v != 0) {
            extent = (r + 1, Ord::max(extent.1, c + 1));
        }
    }
    extent
}

/// `(size, kinds, coefficients, out)`: inverse of dequantised coefficients.
pub(crate) type InverseFn = unsafe fn((usize, usize), [VvcTransformKind; 2], &[i32], &mut [i32]);
/// `(size, kinds, residual, out)`: forward of a residual.
pub(crate) type ForwardFn = unsafe fn((usize, usize), [VvcTransformKind; 2], &[i16], &mut [i32]);

/// One ISA's primary-transform kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Kernels {
    pub inverse: InverseFn,
    pub forward: ForwardFn,
}

/// Run `f` on zeroed working memory of at least `area` values, on the
/// stack: zeroing the largest block's (or a 64x64's) for every small one
/// cost more than its transform (x86 callgrind, crowd key frame with
/// implicit MTS: memset 43% of the instructions).
pub(crate) fn with_scratch<R>(area: usize, f: impl FnOnce(&mut [i32]) -> R) -> R {
    const SMALL: usize = 64;
    const MEDIUM: usize = 256;
    const LARGE: usize = MAX_SIDE * MAX_SIDE;
    const LARGEST: usize = 4 * LARGE;
    if area <= SMALL {
        f(&mut [0; SMALL])
    } else if area <= MEDIUM {
        f(&mut [0; MEDIUM])
    } else if area <= LARGE {
        f(&mut [0; LARGE])
    } else {
        f(&mut [0; LARGEST])
    }
}

/// The scalar reference: the passes as plain sums.
mod scalar {
    use super::{FORWARD_MATRIX_SHIFT, INVERSE_FIRST_SHIFT, INVERSE_SECOND_SHIFT, basis};
    use crate::VvcTransformKind;

    fn round(sum: i32, shift: u32) -> i32 {
        (sum + (1 << (shift - 1))) >> shift
    }

    fn clip16(value: i32) -> i32 {
        value.clamp(i32::from(i16::MIN), i32::from(i16::MAX))
    }

    /// The inverse: columns (rounded by 7 bits, clipped to 16), then rows
    /// (rounded by 12, clipped). Coefficients beyond each kind's
    /// frequencies are ignored.
    pub(super) fn inverse(
        (width, height): (usize, usize),
        [horizontal, vertical]: [VvcTransformKind; 2],
        coefficients: &[i32],
        out: &mut [i32],
    ) {
        let (bh, bv) = (basis(horizontal, width), basis(vertical, height));
        let (fh, fv) = (horizontal.frequencies(width), vertical.frequencies(height));
        super::with_scratch(width * height, |columns| {
            for c in 0..fh {
                for y in 0..height {
                    let sum = (0..fv)
                        .map(|k| coefficients[k * width + c] * i32::from(bv[k][y]))
                        .sum();
                    columns[y * width + c] = clip16(round(sum, INVERSE_FIRST_SHIFT));
                }
            }
            for y in 0..height {
                for x in 0..width {
                    let sum = (0..fh)
                        .map(|c| columns[y * width + c] * i32::from(bh[c][x]))
                        .sum();
                    out[y * width + x] = clip16(round(sum, INVERSE_SECOND_SHIFT));
                }
            }
        });
    }

    /// The forward: rows (shift log2(width) − 1), then columns (shift
    /// log2(height) + 6); frequencies beyond each kind's are zero.
    pub(super) fn forward(
        (width, height): (usize, usize),
        [horizontal, vertical]: [VvcTransformKind; 2],
        residual: &[i16],
        out: &mut [i32],
    ) {
        let (bh, bv) = (basis(horizontal, width), basis(vertical, height));
        let (fh, fv) = (horizontal.frequencies(width), vertical.frequencies(height));
        super::with_scratch(width * height, |rows| {
            for y in 0..height {
                for k in 0..fh {
                    let sum = (0..width)
                        .map(|x| i32::from(residual[y * width + x]) * i32::from(bh[k][x]))
                        .sum();
                    rows[y * width + k] = round(sum, width.ilog2() - 1);
                }
            }
            out[..width * height].fill(0);
            for k in 0..fv {
                for x in 0..fh {
                    let sum = (0..height)
                        .map(|y| rows[y * width + x] * i32::from(bv[k][y]))
                        .sum();
                    out[k * width + x] = round(sum, height.ilog2() + FORWARD_MATRIX_SHIFT);
                }
            }
        });
    }

    pub(super) fn kernels() -> super::Kernels {
        super::Kernels { inverse, forward }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2 {
    use crate::simd::avx2::{I16x8, I16x16};
    super::generic::kernels!(#[target_feature(enable = "avx2")]; I16x8, I16x16);
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use crate::simd::neon::I16x8;
    super::generic::kernels!(#[target_feature(enable = "neon")]; I16x8, I16x8);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128 {
    use crate::simd::simd128::I16x8;
    super::generic::kernels!(#[target_feature(enable = "simd128")]; I16x8, I16x8);
}

pub(crate) fn kernels(isa: Isa) -> Kernels {
    match isa {
        Isa::Scalar => scalar::kernels(),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::kernels(),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::kernels(),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::kernels(),
        #[allow(unreachable_patterns)]
        _ => unreachable!("unsupported ISA"),
    }
}

/// Whether `kinds` apply to a `width` x `height` block of these kernels.
fn supported((width, height): (usize, usize), kinds: [VvcTransformKind; 2]) -> bool {
    let side = |n: usize| n.is_power_of_two() && (4..=MAX_SIDE).contains(&n);
    side(width) && side(height) && kinds[0].valid(width) && kinds[1].valid(height)
}

#[allow(unsafe_code)]
impl Dsp {
    /// The inverse primary transform of dequantised `coefficients` (within
    /// 16 bits) of a `width` x `height` block (row-major; `kinds`
    /// horizontal then vertical) into the residual `out`, as §8.7.4.2.
    /// # Panics
    /// If a side is not 4 to 32, a kind does not apply to its side, or a
    /// slice is shorter than the block.
    pub(crate) fn vvc_inverse_kinds(
        &self,
        size: (usize, usize),
        kinds: [VvcTransformKind; 2],
        coefficients: &[i32],
        out: &mut [i32],
    ) {
        assert!(supported(size, kinds));
        let area = size.0 * size.1;
        assert!(coefficients.len() >= area && out.len() >= area);
        debug_assert!(
            coefficients[..area]
                .iter()
                .all(|&c| i16::try_from(c).is_ok())
        );
        // SAFETY: the table matches this CPU; shape and lengths checked.
        unsafe { (self.vvc_mts.inverse)(size, kinds, coefficients, out) }
    }

    /// VVC 8-bit forward primary transform of a `width` x `height` residual
    /// (within ±255, row-major, `kinds` horizontal then vertical) into row-major
    /// coefficients scaled for [`crate::VvcQuantizer::rect`]. A DCT-II pair
    /// equals [`Self::vvc_forward_dct`]; the 32-point DST-VII and DCT-VIII
    /// leave frequencies 16 and above zero.
    /// # Panics
    /// If a side is not 4 to 32, a kind does not apply to its side, or a
    /// slice is shorter than the block.
    pub fn vvc_forward_primary(
        &self,
        size: (usize, usize),
        kinds: [VvcTransformKind; 2],
        residual: &[i16],
        out: &mut [i32],
    ) {
        assert!(supported(size, kinds));
        let area = size.0 * size.1;
        assert!(residual.len() >= area && out.len() >= area);
        debug_assert!(residual[..area].iter().all(|&r| r.abs() <= 255));
        // SAFETY: the table matches this CPU; shape and lengths checked.
        unsafe { (self.vvc_mts.forward)(size, kinds, residual, out) }
    }
}

#[cfg(test)]
mod tests;
