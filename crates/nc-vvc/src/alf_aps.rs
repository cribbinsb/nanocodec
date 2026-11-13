//! H.266 §7.3.2.18 signalled luma ALF filters: the filter bank of an
//! adaptation parameter set (APS) and the encoder's derivation of one from a
//! picture's statistics.
//!
//! A bank holds up to 25 filters of 12 coefficients and maps each of the 25
//! luma classes to one of them (`alf_luma_coeff_delta_idx`). We code no
//! clipping (`alf_luma_clip_flag` 0: for 8-bit samples clipping index 0
//! clips nothing) and no chroma (`aps_chroma_present_flag` 0).
//!
//! The encoder solves, per class, the Wiener filter of the picture's
//! tap-sum statistics, then merges classes greedily (the pair whose merge
//! costs least, in squared error plus λ times the coefficient bits) and
//! keeps the bank size with the lowest cost; the merged filters are
//! quantised to integers and refined coefficient by coefficient. The approach
//! follows `VVenC` `EncAdaptiveLoopFilter` (BSD-3-Clause-Clear,
//! `THIRD_PARTY.md`), reduced to what a real-time encoder can afford: the
//! statistics are the ones the fixed-set decision already gathers.
use nc_common::BitWriter;
use nc_dsp::{VVC_ALF_DOT_LANES, VVC_ALF_TAPS as TAPS, vvc_alf_gram_index};

/// Luma classes (5 activities times 5 directions).
pub(crate) const CLASSES: usize = 25;
/// `aps_params_type` of ALF parameters.
const ALF_APS: u32 = 0;
/// Bits of `aps_params_type` and `aps_adaptation_parameter_set_id`.
const APS_TYPE_BITS: u32 = 3;
const APS_ID_BITS: u32 = 5;
/// Coefficient range (§7.4.3.18: −2^7 ..= 2^7 − 1).
const COEFFICIENT_MIN: i32 = -128;
const COEFFICIENT_MAX: i32 = 127;
/// The filter's coefficient scale: a coefficient of 2^7 weighs a tap sum by
/// one (`alf.rs` `SHIFT`).
const COEFFICIENT_ONE: f64 = 128.;
/// Regularisation of the normal equations, relative to their diagonal, so
/// that classes with few or flat samples still solve.
const RIDGE: f64 = 1e-4;
/// Rounds of coefficient-by-coefficient refinement after rounding.
const REFINE_ROUNDS: usize = 2;
/// Bits a coded APS NAL unit spends besides its filters: start code (4
/// bytes), NAL header (2), the APS header fields and flags, rbsp trailing.
const NAL_OVERHEAD_BITS: u32 = 32 + 16 + APS_TYPE_BITS + APS_ID_BITS + 1 + 1 + 1 + 1 + 8;

/// A class's (or merged group's) statistics: the packed upper triangle of
/// the Gram matrix of its samples' tap sums and error (`nc_dsp`
/// `vvc_alf_gram`), summed over CTBs.
pub(crate) type Gram = [i64; VVC_ALF_DOT_LANES];

/// A luma ALF filter bank as an APS codes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SignalledFilters {
    /// Each class's filter (`alf_luma_coeff_delta_idx`).
    pub(crate) classes: [u8; CLASSES],
    /// The filters, coefficients in the untransposed tap order.
    pub(crate) filters: Vec<[i8; TAPS]>,
}

impl SignalledFilters {
    /// The filter class `class` uses.
    pub(crate) fn filter(&self, class: usize) -> &[i8; TAPS] {
        &self.filters[usize::from(self.classes[class])]
    }

    /// The bank from a parsed APS (`decoder_headers::Adaptation`), when it is
    /// within what we code: luma filters only, without clipping.
    pub(crate) fn from_adaptation(aps: &crate::decoder_headers::Adaptation) -> Option<Self> {
        let crate::decoder_headers::AdaptationData::Alf(alf) = &aps.data else {
            return None;
        };
        let canonical = !aps.chroma
            && !alf.luma.is_empty()
            && alf.chroma.is_empty()
            && alf.cross.iter().all(Vec::is_empty)
            && alf.luma.iter().all(|f| f.clip.iter().all(|&c| c == 0));
        canonical.then(|| Self {
            classes: alf.classes,
            filters: alf
                .luma
                .iter()
                .map(|f| f.coefficients.map(|c| c as i8))
                .collect(),
        })
    }

    /// The APS RBSP (§7.3.2.6, §7.3.2.18) carrying this bank as `id`.
    pub(crate) fn rbsp(&self, id: u8) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put_bits(ALF_APS, APS_TYPE_BITS);
        w.put_bits(u32::from(id), APS_ID_BITS);
        w.put_bit(false); // aps_chroma_present_flag
        w.put_bit(true); // alf_luma_filter_signal_flag
        w.put_bit(false); // alf_luma_clip_flag
        let count = self.filters.len() as u32;
        w.put_ue(count - 1); // alf_luma_num_filters_signalled_minus1
        if count > 1 {
            let bits = index_bits(self.filters.len());
            for &class in &self.classes {
                w.put_bits(u32::from(class), bits); // alf_luma_coeff_delta_idx
            }
        }
        for filter in &self.filters {
            for &c in filter {
                w.put_ue(u32::from(c.unsigned_abs())); // alf_luma_coeff_abs
                if c != 0 {
                    w.put_bit(c < 0); // alf_luma_coeff_sign
                }
            }
        }
        w.put_bit(false); // aps_extension_flag
        w.trailing_bits();
        w.into_bytes()
    }

    /// Bits the bank's APS NAL unit costs (approximately: emulation
    /// prevention aside).
    pub(crate) fn bits(&self) -> u32 {
        bank_bits(
            self.filters.iter().map(filter_bits).sum(),
            self.filters.len(),
        )
    }
}

/// Bits of `alf_luma_coeff_delta_idx` in a bank of `count` filters.
fn index_bits(count: usize) -> u32 {
    usize::BITS - (count - 1).leading_zeros()
}

/// Bits of an APS whose `count` filters cost `filter_bits`.
fn bank_bits(filter_bits: u32, count: usize) -> u32 {
    let mapping = if count > 1 {
        CLASSES as u32 * index_bits(count)
    } else {
        0
    };
    NAL_OVERHEAD_BITS + ue_bits(count as u32 - 1) + mapping + filter_bits
}

/// Bits of `ue(v)` coding `value`.
fn ue_bits(value: u32) -> u32 {
    2 * (u32::BITS - (value + 1).leading_zeros() - 1) + 1
}

/// Bits of one filter's coefficients (magnitude `ue(v)`, sign when nonzero).
fn filter_bits(filter: &[i8; TAPS]) -> u32 {
    filter
        .iter()
        .map(|&c| ue_bits(u32::from(c.unsigned_abs())) + u32::from(c != 0))
        .sum()
}

/// The squared-error change of filtering a group's samples with integer
/// filter `c`, scaled by 2^14: `cᵀ A c + 2^8 c·b` (`alf.rs`
/// `filter_weights`).
fn integer_change(gram: &Gram, c: &[i32; TAPS]) -> i64 {
    let mut change = 0;
    for i in 0..TAPS {
        let ci = i64::from(c[i]);
        if ci == 0 {
            continue;
        }
        change += ci * ci * gram[vvc_alf_gram_index(i, i)];
        for j in i + 1..TAPS {
            change += 2 * ci * i64::from(c[j]) * gram[vvc_alf_gram_index(i, j)];
        }
        change += (ci << 8) * gram[vvc_alf_gram_index(i, TAPS)];
    }
    change
}

/// The real-valued filter minimising [`integer_change`] for `gram`: the
/// normal equations `A c = −2^7 b`, solved by Cholesky decomposition with
/// a small ridge. `None` when the group has no usable samples.
fn wiener(gram: &Gram) -> Option<[f64; TAPS]> {
    let a = |i: usize, j: usize| gram[vvc_alf_gram_index(i.min(j), i.max(j))] as f64;
    let mut l = [[0f64; TAPS]; TAPS];
    for i in 0..TAPS {
        for j in 0..=i {
            let mut sum = a(i, j) - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                sum += RIDGE * a(i, i) + RIDGE;
            }
            if i == j {
                if sum <= 0. {
                    return None;
                }
                l[i][i] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // Forward then back substitution of L Lᵀ c = −2^7 b.
    let mut y = [0f64; TAPS];
    for i in 0..TAPS {
        let known: f64 = l[i][..i].iter().zip(&y[..i]).map(|(l, y)| l * y).sum();
        y[i] = (-COEFFICIENT_ONE * gram[vvc_alf_gram_index(i, TAPS)] as f64 - known) / l[i][i];
    }
    let mut c = [0f64; TAPS];
    for i in (0..TAPS).rev() {
        let known: f64 = (i + 1..TAPS).map(|k| l[k][i] * c[k]).sum();
        c[i] = (y[i] - known) / l[i][i];
    }
    Some(c)
}

/// `c` rounded to coefficients within the coded range.
fn rounded(c: &[f64; TAPS]) -> [i32; TAPS] {
    c.map(|v| (v.round() as i32).clamp(COEFFICIENT_MIN, COEFFICIENT_MAX))
}

/// The search's units: squared-error changes as [`integer_change`] gives
/// them (per sampled position, scaled by 2^14), converted to picture
/// squared error, and λ in picture squared error per bit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Units {
    /// Picture squared error per unit of [`integer_change`].
    pub(crate) error_scale: f64,
    pub(crate) lambda: f64,
}

impl Units {
    /// The rate-distortion cost of filter `c` on `gram`.
    fn cost(self, gram: &Gram, c: &[i32; TAPS]) -> f64 {
        let filter = c.map(|v| v as i8);
        integer_change(gram, c) as f64 * self.error_scale
            + self.lambda * f64::from(filter_bits(&filter))
    }
}

/// A group of classes sharing one filter while merging.
#[derive(Debug, Clone)]
struct Group {
    /// The classes in it, as a bit mask.
    classes: u32,
    gram: Gram,
    /// The rate-distortion cost of its rounded Wiener filter.
    cost: f64,
}

impl Group {
    fn new(classes: u32, gram: &Gram, units: Units) -> Self {
        // At the real optimum c* (A c* = −2^7 b) the change cᵀAc + 2^8 c·b
        // is 2^7 c*·b; the bits are those of c* rounded.
        let cost = wiener(gram).map_or(0., |c| {
            let change: f64 = (0..TAPS)
                .map(|i| c[i] * gram[vvc_alf_gram_index(i, TAPS)] as f64)
                .sum::<f64>()
                * COEFFICIENT_ONE;
            let bits = filter_bits(&rounded(&c).map(|v| v as i8));
            (change * units.error_scale + units.lambda * f64::from(bits)).min(0.)
        });
        Self {
            classes,
            gram: *gram,
            cost,
        }
    }

    fn merged(&self, other: &Self, units: Units) -> Self {
        let mut gram = self.gram;
        for (sum, &g) in gram.iter_mut().zip(&other.gram) {
            *sum += g;
        }
        Self::new(self.classes | other.classes, &gram, units)
    }
}

/// Classes per direction class (`activity + 5 · direction`, §8.8.5.3).
const ACTIVITIES: usize = 5;

/// Whether some class of mask `a` neighbours some class of mask `b` on the
/// grid of direction and activity.
fn neighbours(a: u32, b: u32) -> bool {
    (0..CLASSES).filter(|&c| a & (1 << c) != 0).any(|c| {
        let (activity, direction) = (c % ACTIVITIES, c / ACTIVITIES);
        let near = |da: isize, dd: isize| {
            let (x, y) = (activity as isize + da, direction as isize + dd);
            (0..ACTIVITIES as isize).contains(&x)
                && (0..(CLASSES / ACTIVITIES) as isize).contains(&y)
                && b & (1 << (y as usize * ACTIVITIES + x as usize)) != 0
        };
        near(-1, 0) || near(1, 0) || near(0, -1) || near(0, 1)
    })
}

/// The filter bank for classes with statistics `grams` (`None` for classes
/// without samples) that minimises squared error plus λ times its APS bits,
/// or `None` when no bank lowers the cost (estimated on these statistics;
/// the per-CTB decision then decides where it is used).
pub(crate) fn derive(grams: &[Option<Gram>], units: Units) -> Option<SignalledFilters> {
    let mut groups: Vec<Group> = grams
        .iter()
        .enumerate()
        .filter_map(|(class, gram)| gram.as_ref().map(|g| Group::new(1 << class, g, units)))
        .collect();
    if groups.is_empty() {
        return None;
    }
    // Greedy merging of neighbours: each step merges the pair of groups
    // with neighbouring classes (on the 5x5 grid of direction and activity)
    // whose merged filter costs least more than the two apart; the bank of
    // the best size is kept. Only neighbours are tried, so that a picture
    // needs about 150 Wiener solutions rather than about 600 (any-pair
    // merging: 9 M instructions per 720p picture; neighbours in class order
    // only: -0.02..-0.04 dB luma on bodycam/lowlight at 150 kbps).
    let total = |groups: &[Group]| {
        groups.iter().map(|g| g.cost).sum::<f64>()
            + units.lambda * f64::from(bank_bits(0, groups.len()))
    };
    let mut best = (total(&groups), groups.clone());
    let pair = |a: &Group, b: &Group| neighbours(a.classes, b.classes).then(|| a.merged(b, units));
    let mut pairs: Vec<Vec<Option<Group>>> = (0..groups.len())
        .map(|i| (0..i).map(|j| pair(&groups[i], &groups[j])).collect())
        .collect();
    while groups.len() > 1 {
        let delta = |i: usize, j: usize| {
            pairs[i][j]
                .as_ref()
                .map_or(f64::INFINITY, |m| m.cost - groups[i].cost - groups[j].cost)
        };
        let (i, j) = (0..groups.len())
            .flat_map(|i| (0..i).map(move |j| (i, j)))
            .min_by(|&(a, b), &(c, d)| delta(a, b).total_cmp(&delta(c, d)))
            .expect("two groups");
        // Groups without neighbours (classes without samples between them)
        // merge with whichever group costs least.
        let merged = pairs[i][j]
            .take()
            .unwrap_or_else(|| groups[i].merged(&groups[j], units));
        // i > j: remove i first so j keeps its index.
        groups.remove(i);
        pairs.remove(i);
        for row in &mut pairs[i..] {
            row.remove(i);
        }
        groups[j] = merged;
        for k in 0..groups.len() {
            if k != j {
                let (a, b) = (k.max(j), k.min(j));
                pairs[a][b] = pair(&groups[a], &groups[b]);
            }
        }
        let cost = total(&groups);
        if cost < best.0 {
            best = (cost, groups.clone());
        }
    }
    let (_, groups) = best;
    let filters: Vec<[i8; TAPS]> = groups
        .iter()
        .map(|g| refined(&g.gram, units).map(|c| c as i8))
        .collect();
    let mut classes = [0; CLASSES];
    for (index, group) in groups.iter().enumerate() {
        for (class, slot) in classes.iter_mut().enumerate() {
            if group.classes & (1 << class) != 0 {
                *slot = index as u8;
            }
        }
    }
    let bank = SignalledFilters { classes, filters };
    let gain: f64 = groups
        .iter()
        .zip(&bank.filters)
        .map(|(g, f)| integer_change(&g.gram, &f.map(i32::from)) as f64 * units.error_scale)
        .sum();
    (gain + units.lambda * f64::from(bank.bits()) < 0.).then_some(bank)
}

/// The group's Wiener filter rounded, then each coefficient moved by ±1
/// while that lowers the rate-distortion cost.
fn refined(gram: &Gram, units: Units) -> [i32; TAPS] {
    let Some(real) = wiener(gram) else {
        return [0; TAPS];
    };
    let mut c = rounded(&real);
    let mut cost = units.cost(gram, &c);
    for _ in 0..REFINE_ROUNDS {
        let mut changed = false;
        for i in 0..TAPS {
            for step in [-1, 1] {
                let mut trial = c;
                trial[i] = (trial[i] + step).clamp(COEFFICIENT_MIN, COEFFICIENT_MAX);
                let trial_cost = units.cost(gram, &trial);
                if trial_cost < cost {
                    (c, cost, changed) = (trial, trial_cost, true);
                }
            }
        }
        if !changed {
            break;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Statistics of samples whose error is exactly `filter` applied to
    /// their tap sums (scaled by 2^-7, negated: the filter undoes it).
    fn exact_gram(filter: &[i32; TAPS]) -> Gram {
        let mut gram = [0; VVC_ALF_DOT_LANES];
        let mut seed = 12345u32;
        for _ in 0..2000 {
            let mut sample = [0i64; TAPS + 1];
            for s in &mut sample[..TAPS] {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                *s = i64::from(seed >> 24) - 128;
            }
            let dot: i64 = (0..TAPS).map(|i| sample[i] * i64::from(filter[i])).sum();
            sample[TAPS] = -dot / 128;
            for i in 0..=TAPS {
                for j in i..=TAPS {
                    gram[vvc_alf_gram_index(i, j)] += sample[i] * sample[j];
                }
            }
        }
        gram
    }

    #[test]
    fn wiener_recovers_the_generating_filter() {
        let filter = [3, -1, -2, -8, -10, -1, 19, 8, -1, 2, 5, 24];
        let solved = rounded(&wiener(&exact_gram(&filter)).unwrap());
        for (s, f) in solved.iter().zip(&filter) {
            assert!((s - f).abs() <= 1, "{solved:?} vs {filter:?}");
        }
    }

    #[test]
    fn derived_bank_round_trips_through_the_aps_parser() {
        let filter = [3, -1, -2, -8, -10, -1, 19, 8, -1, 2, 5, 24];
        let other = [0, 1, 0, -5, -8, -1, 12, 14, -2, 8, 6, 14];
        let mut grams = vec![None; CLASSES];
        for (class, gram) in grams.iter_mut().enumerate() {
            *gram = Some(exact_gram(if class < 12 { &filter } else { &other }));
        }
        let units = Units {
            error_scale: 1. / 16384.,
            lambda: 10.,
        };
        let bank = derive(&grams, units).expect("a bank pays");
        let parsed = crate::decoder_headers::Adaptation::parse(&bank.rbsp(3)).unwrap();
        assert_eq!(parsed.id, 3);
        assert_eq!(SignalledFilters::from_adaptation(&parsed), Some(bank));
    }

    #[test]
    fn no_bank_without_statistics() {
        let units = Units {
            error_scale: 1.,
            lambda: 1.,
        };
        assert_eq!(derive(&vec![None; CLASSES], units), None);
    }
}
