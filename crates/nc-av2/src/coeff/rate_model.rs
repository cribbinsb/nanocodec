//! Modelled coefficient rate for mode-decision trials (`av2.coeff_rate_model`).
//!
//! A frozen rate estimate ([`crate::entropy::Writer::estimates_frozen_now`])
//! normally prices every level of a transform with the full context walk
//! ([`super::code_levels`] under [`FrozenPricer`]): each coded position
//! derives its base and low-range contexts from its template neighbours.
//! That walk is AV2's largest pricing cost (c8a q135: 10-19 % of AV2 Faster's
//! cycles with its context helper, 2-10 % at q210).
//!
//! The model keeps the cheap, block-level symbols exact (all-zero flag, end
//! of block, transform type: a few symbols under the live models) and
//! replaces the level walk by a linear estimate over counts gathered in one
//! pass along the scan up to the end of block: per frequency region the
//! zero, one, two and larger levels, and the low-range increments, plus a
//! constant. Signs (one bypass bit each; the luma DC sign is close to one
//! bit) and high-range remainders (Rice bits at a zero running average) are
//! counted, not fitted. The weights, in 1/512 bit, are least-squares fits
//! of the exact walk's rate per transform size, plane type and coefficient
//! quantizer category on TRAIN clips; the fit and its VAL error are in
//! docs/handoff/vp9-av2.md "AV2 modelled coefficient rate". The model
//! ignores the neighbour contexts and the tile's within-frame adaptation;
//! the final write always walks.
//!
//! TCQ levels keep the exact walk: their parity-preserving state machine
//! is not in the features.

use super::context::EntropyContext;
use super::levels::{self, MAX_LOW_RANGE, Region};
use super::symbols::high_range_bits;
use super::{Block, PLANE_Y, TxSize};
use nc_encoder::symbol_cost::COST_UNITS_PER_BIT;

/// Fitted features of one transform's levels, in [`Features::counts`] order.
pub(super) const FEATURES: usize = 10;
/// Feature slots: per region (low frequency, default), the zero, one, two
/// and larger levels; then the low-range increments and the constant.
const LOW_REGION: usize = 0;
const DEFAULT_REGION: usize = 4;
const LOW_RANGE: usize = 8;
const CONSTANT: usize = 9;
/// Levels counted by their own slot before the "larger" slot.
const COUNTED_LEVELS: u32 = 3;
/// Transform sizes, plane types and quantizer categories with own weights.
const SIZES: usize = 5;
const PLANE_TYPES: usize = 2;
const QUANTIZER_CATEGORIES: usize = 4;

/// What one pass along a transform's scan gathers for the model.
pub(super) struct Features {
    /// Fitted feature counts.
    pub(super) counts: [u32; FEATURES],
    /// Rate counted exactly, in 1/512 bit: sign bits and high-range
    /// remainders at a zero running average.
    pub(super) counted_q9: u32,
}

/// The features of `levels` (raster, coded region of `block.tx`) up to
/// scan index `eob` (at least 1), adding each nonzero level to `context`.
pub(super) fn features(
    block: Block,
    levels: &[i32],
    eob: usize,
    context: &mut EntropyContext,
) -> Features {
    // Four u16 counts fit in one register, with a shift to select the
    // field. The largest coded transform has only 1024 coefficients.
    const FIELD_BITS: u32 = u16::BITS;
    const FIELD_MASK: u64 = u16::MAX as u64;
    let mut counts = [0; FEATURES];
    let mut counted_q9 = 0;
    let mut low_range = 0;
    let (low_end, low, default) = Region::frequency_regions(block.ctx.plane);
    let scan = &levels::scan(block.tx)[..eob];
    let (low_scan, default_scan) = scan.split_at(low_end.min(eob));
    for (region, scan, slot) in [
        (low, low_scan, LOW_REGION),
        (default, default_scan, DEFAULT_REGION),
    ] {
        let mut histogram = 0u64;
        let base_limit = region.base_levels() + 1;
        let saturation = region.saturation();
        for &pos in scan {
            let pos = usize::from(pos);
            let value = levels[pos];
            let level = value.unsigned_abs();
            histogram += 1 << (level.min(COUNTED_LEVELS) * FIELD_BITS);
            if level == 0 {
                continue;
            }
            context.add(pos, value);
            if region.has_low_range() && level > base_limit {
                low_range += (level - base_limit).min(MAX_LOW_RANGE);
            }
            if level >= saturation {
                counted_q9 += high_range_bits(level - saturation, 0) * COST_UNITS_PER_BIT;
            }
        }
        for (i, count) in counts[slot..slot + 4].iter_mut().enumerate() {
            *count = ((histogram >> (i as u32 * FIELD_BITS)) & FIELD_MASK) as u32;
        }
    }
    counts[LOW_RANGE] = low_range;
    let nonzero = eob as u32 - counts[LOW_REGION] - counts[DEFAULT_REGION];
    counted_q9 += nonzero * COST_UNITS_PER_BIT;
    counts[CONSTANT] = 1;
    Features { counts, counted_q9 }
}

/// The modelled rate of a transform's levels with `features`, in 1/512 bit.
pub(super) fn levels_q9(block: Block, features: &Features) -> u32 {
    let plane_type = usize::from(block.ctx.plane != PLANE_Y);
    let weights = &WEIGHTS[size_index(block.tx)][plane_type][block.q];
    let fitted: f32 = weights
        .iter()
        .zip(features.counts)
        .map(|(&w, n)| w * n as f32)
        .sum();
    fitted.max(0.0) as u32 + features.counted_q9
}

fn size_index(tx: TxSize) -> usize {
    match tx {
        TxSize::Tx4 => 0,
        TxSize::Tx8 => 1,
        TxSize::Tx16 => 2,
        TxSize::Tx32 => 3,
        TxSize::Tx64 => 4,
    }
}

/// Fitted weights in 1/512 bit, by transform size, plane type (luma,
/// chroma) and coefficient quantizer category, in [`Features::counts`] order.
#[rustfmt::skip]
const WEIGHTS: [[[[f32; FEATURES]; QUANTIZER_CATEGORIES]; PLANE_TYPES]; SIZES] = [
    [
        [
            [655.3, 516.0, 1394.6, 1815.0, 815.3, 1031.0, 1894.5, 2808.4, -240.0, 206.7],
            [439.8, 740.1, 1364.5, 1937.2, 503.8, 958.8, 1934.6, 2496.7, -78.0, -304.0],
            [735.5, 174.1, 1186.7, 1488.3, 0.0, 0.0, 0.0, 0.0, 0.0, 178.6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [-138.7, -453.0, 569.1, 949.9, 135.8, 975.9, 1724.2, 2374.4, 87.2, 927.2],
            [-547.7, -565.6, 517.5, 1298.2, 211.2, 768.7, 2269.6, 2835.5, 136.0, 702.5],
            [-781.9, -392.1, 1625.3, -0.0, -322.6, 1018.4, 0.0, 0.0, 0.0, 451.2],
            [-546.2, -564.2, 520.2, 1291.1, 209.8, 775.4, 2254.4, 2856.3, 35.8, 700.9],
        ],
    ],
    [
        [
            [169.0, 839.0, 1292.1, 2007.7, 320.4, 1173.6, 1887.2, 2819.7, -731.2, -32.5],
            [224.4, 878.5, 1475.5, 1935.1, 211.1, 1130.7, 1752.2, 2228.5, -162.9, -394.3],
            [223.6, 822.2, 1674.6, 2350.4, 107.3, 1239.9, 2157.8, 2531.8, -61.0, -490.5],
            [514.6, 213.6, 1268.3, 2349.6, 66.9, 526.2, 7601.3, 0.0, 0.0, 235.2],
        ],
        [
            [-36.4, -235.4, 131.6, 402.4, 222.1, 1124.9, 1891.5, 2267.8, -310.1, 262.2],
            [-759.4, -430.3, 521.2, 1172.9, 158.0, 1058.8, 2065.4, 2355.2, -142.6, 504.5],
            [-852.2, -678.8, 655.1, 1596.1, 169.9, 860.5, 2408.9, 2426.2, 660.5, 720.1],
            [-508.5, -577.8, 1394.9, 316.7, 63.5, 748.9, 0.0, 0.0, 0.0, 625.4],
        ],
    ],
    [
        [
            [119.0, 779.8, 1462.5, 2036.6, 159.1, 1260.3, 1845.7, 3053.2, -924.9, 666.4],
            [214.6, 1023.6, 1521.9, 1717.9, 120.6, 1386.2, 1671.9, 2225.8, -288.2, -579.3],
            [252.0, 895.8, 1510.4, 2024.4, 71.6, 1527.5, 1806.0, 2331.6, -27.6, -653.8],
            [208.9, 796.0, 1777.0, 2725.4, 59.9, 1611.5, 3650.1, -0.0, 820.3, -666.9],
        ],
        [
            [267.5, -340.9, -193.5, 561.3, 143.2, 1311.6, 2049.5, 2494.3, -494.3, 294.4],
            [-750.4, -423.6, 441.1, 1034.8, 106.7, 1379.9, 1983.5, 1915.1, -45.7, 302.0],
            [-667.2, -578.2, 620.3, 1207.2, 115.4, 1105.5, 2351.1, 3000.6, -247.3, 582.0],
            [1357.0, -1633.6, 77.2, 1827.3, 63.3, 972.7, 1143.3, 3651.4, 0.0, 1627.9],
        ],
    ],
    [
        [
            [-451.2, 1710.9, 2752.5, 1775.4, 98.2, 1297.3, 1561.2, 3401.5, -989.5, 674.4],
            [172.1, 947.2, 1557.6, 1158.9, 92.5, 1602.9, 1311.1, 2650.8, -520.7, -254.1],
            [330.2, 828.3, 1106.9, 1400.6, 67.4, 1722.1, 1287.8, 2313.7, -32.5, -536.3],
            [230.6, 796.5, 1819.9, 2103.5, 58.6, 1624.7, 2597.6, 2651.2, 587.4, -552.7],
        ],
        [
            [0.0, 514.3, -2389.5, 1061.4, 92.6, 1481.1, 2253.7, 3589.3, -177.9, -813.8],
            [-962.1, -437.9, 318.9, 1134.3, 74.8, 1692.0, 1811.8, 1277.6, 205.2, 53.2],
            [-883.2, -662.2, 628.3, 1561.0, 113.5, 1172.2, 1202.7, 4642.0, -918.3, 643.9],
            [-380.1, -488.4, -1897.9, 3273.1, 43.1, 1437.6, 5762.7, 479.4, 479.4, 506.7],
        ],
    ],
    [
        [
            [-1923.4, 4653.9, 3615.6, 3743.9, 107.8, 1340.5, 1041.0, 4512.7, -1373.2, 28.9],
            [347.7, 1647.7, 3060.1, -203.3, 83.6, 1549.5, 1479.6, 2752.1, -508.3, -1180.2],
            [191.4, 722.3, 1431.4, 1148.4, 54.7, 1824.4, 988.5, 2473.9, -193.8, -287.8],
            [244.4, 793.5, 2053.4, 2740.8, 37.6, 1706.5, 6096.3, 0.0, 278.4, -597.1],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    ],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_counts_match_the_scan_with_a_full_histogram_field() {
        let tx = TxSize::Tx64;
        let kind = super::super::Kind::DCT_ONLY;
        for plane in [0, 1, 2] {
            let ctx = super::super::Context {
                plane,
                ..Default::default()
            };
            let block = Block::new(tx, ctx, 120, (kind.set, kind.levels));
            let (low_end, low, default) = Region::frequency_regions(plane);
            let positions = levels::scan_positions(tx);
            assert_eq!(positions[..low_end].last().unwrap().region(plane), low);
            assert_eq!(positions[low_end].region(plane), default);
            for value in [1, -2, 3, -6, 15, -117] {
                let levels = vec![value; tx.coded_len()];
                let mut context = EntropyContext::default();
                let all = features(block, &levels, levels.len(), &mut context);
                let slot = value.unsigned_abs().min(COUNTED_LEVELS) as usize;
                assert_eq!(all.counts[LOW_REGION + slot], low_end as u32);
                assert_eq!(
                    all.counts[DEFAULT_REGION + slot],
                    (levels.len() - low_end) as u32
                );
                assert_eq!(all.counts[..8].iter().sum::<u32>(), levels.len() as u32);
                assert_eq!(context.value(), super::super::entropy_context(&levels));
            }
        }
    }
}
