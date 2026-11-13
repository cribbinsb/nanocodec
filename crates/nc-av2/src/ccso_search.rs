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

use crate::ccso::{CcsoParams, CcsoSource, PlaneFilter, UNIT, UnitFlags};

/// The classifiers tried per plane: (`scale`, `quant`, `support`).
/// Scale 0 keeps offsets small (±1, ±3, ±7, −10) for 8-bit video;
/// thresholds 8 and 16 (`quant` 1 and 0); vertical and horizontal pairs.
const CANDIDATES: [(u8, u8, u8); 4] = [(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1)];
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

/// The class sums of every candidate for `plane`, per unit of `flags`, in
/// one pass over the plane.
fn class_sums(
    source: &Picture,
    recon: &Picture,
    deblocked: &CcsoSource,
    plane: PlaneIndex,
    flags: &UnitFlags,
) -> Vec<UnitSums> {
    let units = flags.positions().len();
    let columns = deblocked.width().div_ceil(UNIT);
    let mut sums = vec![vec![[ClassSum::default(); CLASSES]; units]; CANDIDATES.len()];
    let classifiers: Vec<_> = CANDIDATES
        .iter()
        .map(|&(scale, quant, support)| {
            PlaneFilter::zero(false, scale, quant, support, false, 0).classifier()
        })
        .collect();
    let shift = usize::from(plane != PlaneIndex::Y);
    let source = source.plane(plane).view();
    let recon = recon.plane(plane).view();
    for y in 0..recon.height() {
        let (source_row, recon_row) = (source.row(y), recon.row(y));
        let ly = y << shift;
        for (x, (&s, &r)) in source_row.iter().zip(recon_row).enumerate() {
            let lx = x << shift;
            let unit = ly / UNIT * columns + lx / UNIT;
            let error = i64::from(s) - i64::from(r);
            for (candidate, classifier) in classifiers.iter().enumerate() {
                // One band: the look-up index is class0 << 2 | class1.
                let index = deblocked.lut_index(classifier, lx, ly);
                let class = (index >> 2) * 3 + (index & 3);
                let sum = &mut sums[candidate][unit][class];
                sum.count += 1;
                sum.error += error;
            }
        }
    }
    sums
}
