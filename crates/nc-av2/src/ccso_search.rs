//! The encoder's CCSO choice ([`crate::ccso`]): per plane, a handful of
//! classifiers, the offsets of each class, and each unit's flag, all from
//! one pass of statistics over the CDEF-filtered reconstruction (SAO-style,
//! as SVT-AV1 and libaom derive CCSO's offsets from class sums at their
//! fast speeds; AVM's full search (`av2/encoder/pickccso.c`) refilters the
//! picture for every candidate).
//!
//! For each candidate classifier the pass adds up, per unit and class, the
//! count of samples and the sum of their errors (source minus
//! reconstruction). Adding offset `v` to a class of `n` samples with error
//! sum `S` changes its squared error by `n·v² − 2·v·S`, so the best code of
//! each class, the gain of each unit and the frame's rate-distortion cost
//! follow from the sums alone. Offsets are derived from every unit, units
//! whose gain does not pay for itself are switched off, and the offsets are
//! derived again from the units left on.

use nc_common::{Picture, PlaneIndex};

use crate::ccso::{CcsoParams, CcsoSource, PlaneFilter, UNIT, UnitFlags, neighbour_pair};

/// The classifiers tried per plane: (`scale`, `quant`, `support`).
/// Scale 0 keeps offsets small (±1, ±3, ±7, −10) for 8-bit video;
/// thresholds 8 and 16 (`quant` 1 and 0); vertical and horizontal pairs.
const SEARCH_QUANTS: [u8; 2] = [1, 0];
const CANDIDATES: [(u8, u8, u8); 4] = [
    (0, SEARCH_QUANTS[0], 0),
    (0, SEARCH_QUANTS[0], 1),
    (0, SEARCH_QUANTS[1], 0),
    (0, SEARCH_QUANTS[1], 1),
];
/// Edge classes per sample: 3×3 (three levels per neighbour, one band).
const CLASSES: usize = 9;
/// Offset codes by value order, so the search tries each value once.
const CODES: u8 = 8;
/// Rounds of offset derivation and unit decisions.
const ROUNDS: usize = 2;
/// Approximate bits of a unit flag, either value.
const FLAG_BITS: f64 = 1.0;
/// Bits of a plane's filter syntax before its offsets: enable, reuse
/// flags, `ccso_bo_only`, `scale_idx`, `quant_idx`, `ext_filter_support`,
/// `edge_clf`, `max_band_log2`.
const FILTER_HEADER_BITS: f64 = 14.0;

/// A class's sample count and error sum in one unit.
#[derive(Debug, Clone, Copy, Default)]
struct ClassSum {
    count: i64,
    error: i64,
}

impl ClassSum {
    /// Change of squared error when `value` is added to every sample.
    fn delta(self, value: i64) -> i64 {
        self.count * value * value - 2 * value * self.error
    }
}

/// Per unit, the class sums of one classifier.
type UnitSums = Vec<[ClassSum; CLASSES]>;

/// Choose the frame's CCSO from the CDEF-filtered `recon` (classified on
/// `deblocked`) and `source`, setting `flags`: each plane keeps the
/// candidate of least rate-distortion cost (`lambda` per bit, squared error
/// in luma units: chroma's weighed by `chroma_weight`), or none when no
/// candidate pays for its syntax.
pub(crate) fn choose(
    source: &Picture,
    recon: &Picture,
    deblocked: &CcsoSource,
    flags: &mut UnitFlags,
    lambda: f64,
    chroma_weight: f64,
) -> CcsoParams {
    let mut planes = [None; 3];
    for plane in PlaneIndex::ALL {
        let weight = if plane == PlaneIndex::Y {
            1.0
        } else {
            chroma_weight
        };
        let sums = class_sums(source, recon, deblocked, plane, flags);
        let best = CANDIDATES
            .iter()
            .zip(&sums)
            .map(|(&candidate, sums)| fit(candidate, sums, lambda / weight))
            .min_by(|a, b| a.cost.total_cmp(&b.cost));
        if let Some(fit) = best.filter(|fit| fit.cost < 0.0) {
            for (unit, &(x, y)) in flags.positions().iter().enumerate() {
                flags.set(plane, x, y, fit.units[unit]);
            }
            planes[plane as usize] = Some(fit.filter);
        }
    }
    CcsoParams::new(planes).expect("searched filters within syntax range")
}

/// One candidate's best filter for a plane, its unit flags and its cost
/// against leaving the plane unfiltered (squared error plus λ·bits, in the
/// plane's own error units).
struct Fit {
    filter: PlaneFilter,
    units: Vec<bool>,
    cost: f64,
}

/// Fit offsets and unit flags of one candidate to its sums.
fn fit((scale, quant, support): (u8, u8, u8), sums: &UnitSums, lambda: f64) -> Fit {
    let mut filter = PlaneFilter::zero(false, scale, quant, support, false, 0);
    let mut units = vec![true; sums.len()];
    for _ in 0..ROUNDS {
        for class in 0..CLASSES {
            let total = sums.iter().zip(&units).filter(|&(_, &on)| on).fold(
                ClassSum::default(),
                |acc, (unit, _)| ClassSum {
                    count: acc.count + unit[class].count,
                    error: acc.error + unit[class].error,
                },
            );
            let code = (0..CODES)
                .min_by(|&a, &b| {
                    let cost = |code: u8| {
                        total.delta(i64::from(filter.offset_value(code))) as f64
                            + lambda * f64::from(code_bits(code))
                    };
                    cost(a).total_cmp(&cost(b))
                })
                .expect("codes exist");
            filter.set_offset(class, code);
        }
        for (on, unit) in units.iter_mut().zip(sums) {
            *on = unit_delta(&filter, unit) < 0;
        }
    }
    let distortion: i64 = sums
        .iter()
        .zip(&units)
        .filter(|&(_, &on)| on)
        .map(|(unit, _)| unit_delta(&filter, unit))
        .sum();
    let offset_bits: u32 = (0..CLASSES)
        .map(|class| code_bits(filter.offset(class)))
        .sum();
    let bits = FILTER_HEADER_BITS + f64::from(offset_bits) + FLAG_BITS * sums.len() as f64;
    Fit {
        filter,
        units,
        cost: distortion as f64 + lambda * bits,
    }
}

/// Squared-error change of one unit under `filter`.
fn unit_delta(filter: &PlaneFilter, unit: &[ClassSum; CLASSES]) -> i64 {
    unit.iter()
        .enumerate()
        .map(|(class, sum)| sum.delta(i64::from(filter.offset_value(filter.offset(class)))))
        .sum()
}

/// Bits of an offset code's unary syntax.
fn code_bits(code: u8) -> u32 {
    u32::from(code) + u32::from(code < CODES - 1)
}

/// Accumulate all four classifiers while their shared luma neighbours
/// and the reconstruction error are in registers.
#[inline(never)]
fn class_sums(
    source: &Picture,
    recon: &Picture,
    deblocked: &CcsoSource,
    plane: PlaneIndex,
    flags: &UnitFlags,
) -> Vec<UnitSums> {
    let positions = flags.positions();
    let mut sums = vec![vec![[ClassSum::default(); CLASSES]; positions.len()]; CANDIDATES.len()];
    let [small, large] = SEARCH_QUANTS
        .map(|quant| PlaneFilter::zero(false, 0, quant, 0, false, 0).threshold() as i16);
    // Put the two threshold classes in separate nibbles. A pair's class
    // is 3 * first + second, so the two results never carry into each other.
    let classes_by_difference: [u8; 512] = std::array::from_fn(|index| {
        let difference = index as i16 - 256;
        let class =
            |threshold: i16| u8::from(difference >= -threshold) + u8::from(difference > threshold);
        class(small) | class(large) << 4
    });
    let shift = usize::from(plane != PlaneIndex::Y);
    let unit_width = UNIT >> shift;
    let source = source.plane(plane).view();
    let recon = recon.plane(plane).view();
    let width = source.width().min(recon.width());
    for (unit, &(luma_x, luma_y)) in positions.iter().enumerate() {
        let x = luma_x >> shift;
        let y = luma_y >> shift;
        let right = (x + unit_width).min(width);
        let bottom = (y + unit_width).min(recon.height());
        if x >= right {
            continue;
        }
        let mut unit_sums = [[0_u64; 16]; CANDIDATES.len()];
        let luma_len = (right - x) << shift;
        let lx = luma_x as isize;
        for row in y..bottom {
            let ly = (row << shift) as isize;
            let span = |(dx, dy)| deblocked.span(lx + dx, ly + dy, luma_len);
            let center = span((0, 0));
            let [above, below] = neighbour_pair(0).map(span);
            let [left, rightward] = neighbour_pair(1).map(span);
            let original = &source.row(row)[x..right];
            let rebuilt = &recon.row(row)[x..right];
            for (sample, (&source, &recon)) in original.iter().zip(rebuilt).enumerate() {
                let at = sample << shift;
                let center = i16::from(center[at]);
                let classes = |neighbour| {
                    let difference = i16::from(neighbour) - center;
                    classes_by_difference[((difference + 256) as usize) & 511]
                };
                let vertical = 3 * classes(above[at]) + classes(below[at]);
                let horizontal = 3 * classes(left[at]) + classes(rightward[at]);
                // A unit has at most 65,536 samples with biased errors
                // in 0..=510. Their sum cannot carry into the count.
                let biased_error = u16::from(source) + 255 - u16::from(recon);
                let update = (1_u64 << 32) | u64::from(biased_error);
                let classes = [
                    vertical & 15,
                    horizontal & 15,
                    vertical >> 4,
                    horizontal >> 4,
                ]
                .map(usize::from);
                for (bins, class) in unit_sums.iter_mut().zip(classes) {
                    bins[class] += update;
                }
            }
        }
        for (candidate, bins) in sums.iter_mut().zip(unit_sums) {
            for (sum, packed) in candidate[unit].iter_mut().zip(bins) {
                let count = (packed >> 32) as i64;
                *sum = ClassSum {
                    count,
                    error: i64::from(packed as u32) - 255 * count,
                };
            }
        }
    }
    sums
}
