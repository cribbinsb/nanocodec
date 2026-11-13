//! H.266 §7.3.11.9 implicit transform tree: a coding unit wider or taller
//! than the largest transform block (`MaxTbSizeY`, `headers::Geometry::
//! max_tb_log2`) is split into equal transform units, without syntax.
//! Without `sps_max_luma_transform_size_64_flag` (`MaxTbSizeY` 32) a 64x64
//! CU codes four 32x32 luma / 16x16 chroma transform units in z-order, a
//! 64x32 or 32x64 CU (or a ternary 64x16 / 16x64 one) two halves; every
//! smaller CU is one TU. With it, a 64x64 CU is one TU of a 64-point luma
//! transform and 32x32 chroma.
//!
//! A 64-point side codes only its 32 lowest frequencies (§7.3.11.11
//! zero-out), so a transform block's levels cover its coded region
//! ([`coded_samples`]): a 64x64 block's are 32x32. A CU's levels of one
//! plane (`syntax::Coding::levels`) are empty when no TU codes any, else
//! every TU's back to back in [`units`] order, each in raster order
//! ([`unit_levels`], [`place`]).
use nc_encoder::Block;

/// log2 of `MaxTbSizeY` (§7.4.3.4) without
/// `sps_max_luma_transform_size_64_flag`, and of the largest coded region's
/// side with it.
pub(crate) const MAX_TB_LOG2: u8 = 5;

/// Largest side of a transform block's coded region (§7.3.11.11
/// `log2ZoTbWidth` / `log2ZoTbHeight` of a DCT-II).
pub(crate) const MAX_CODED_SIDE: usize = 1 << MAX_TB_LOG2;

/// Samples of the largest coded region.
pub(crate) const MAX_TB_SAMPLES: usize = MAX_CODED_SIDE * MAX_CODED_SIDE;

/// The most transform units of one CU (a 64x64 CU's four).
pub(crate) const MAX_UNITS: usize = 4;

/// The samples of a `width` x `height` transform block's coded region
/// (its levels).
pub(crate) fn coded_samples(width: usize, height: usize) -> usize {
    width.min(MAX_CODED_SIDE) * height.min(MAX_CODED_SIDE)
}

/// The luma transform-unit size of coding unit `cu` (at `cu`'s position)
/// with `MaxTbSizeY` `1 << max_log2`.
pub(crate) fn unit_of(cu: &Block, max_log2: u8) -> Block {
    Block::rect(
        cu.x,
        cu.y,
        cu.log2_width.min(max_log2),
        cu.log2_height.min(max_log2),
    )
}

/// The number of transform units of coding unit `cu`.
pub(crate) fn count(cu: &Block, max_log2: u8) -> usize {
    cu.area() / unit_of(cu, max_log2).area()
}

/// The luma transform units of coding unit `cu`, in decoding order
/// (§7.3.11.9: halves while wider or taller than `MaxTbSizeY`, the vertical
/// cut first only when wider than tall; for our CUs that is raster order).
/// A region is at most twice the limit in each dimension; the general
/// decoder visits larger coding units' regions recursively first.
pub(crate) fn units(cu: &Block, max_log2: u8) -> impl Iterator<Item = Block> {
    let unit = unit_of(cu, max_log2);
    let columns = cu.width() / unit.width();
    let count = cu.area() / unit.area();
    debug_assert!(count <= MAX_UNITS);
    (0..count).map(move |i| {
        Block::rect(
            cu.x + i % columns * unit.width(),
            cu.y + i / columns * unit.height(),
            unit.log2_width,
            unit.log2_height,
        )
    })
}

/// Whether a TU of `cu` signals `tu_y_coded_flag` even when the CU is
/// inter and both chroma flags are zero (§7.3.11.10: the luma flag is
/// inferred only for a CU no wider or taller than `MaxTbSizeY`).
pub(crate) fn splits(cu: &Block, max_log2: u8) -> bool {
    cu.log2_width.max(cu.log2_height) > max_log2
}

/// Whether luma position `(x, y)` inside `cu` lies in a transform unit
/// decoded before `tu` (§6.4.4: a neighbouring sample is available once
/// decoded; within a CU, the TUs before it in [`units`] order).
pub(crate) fn decoded_before(cu: &Block, tu: &Block, x: usize, y: usize) -> bool {
    let inside = (cu.x..cu.x + cu.width()).contains(&x) && (cu.y..cu.y + cu.height()).contains(&y);
    let order = |x: usize, y: usize| {
        let columns = cu.width() / tu.width();
        (y - cu.y) / tu.height() * columns + (x - cu.x) / tu.width()
    };
    inside && order(x, y) < order(tu.x, tu.y)
}

/// The levels of transform unit `index` in a CU plane's `levels`, each
/// unit `samples` long; empty when the plane codes none.
pub(crate) fn unit_levels(levels: &[i32], index: usize, samples: usize) -> &[i32] {
    levels
        .get(index * samples..(index + 1) * samples)
        .unwrap_or(&[])
}

/// Put transform unit `index`'s `unit` levels (empty when all zero) into
/// the plane's `levels` of a CU of `count` units, `samples` long each.
pub(crate) fn place(levels: &mut Vec<i32>, count: usize, index: usize, unit: Vec<i32>) {
    if unit.is_empty() {
        return;
    }
    if count == 1 {
        *levels = unit;
        return;
    }
    let samples = unit.len();
    levels.resize(count * samples, 0);
    levels[index * samples..(index + 1) * samples].copy_from_slice(&unit);
}

/// The transform units of a CU and which planes of each code coefficients.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CodedUnits {
    units: [(Block, [bool; 3]); MAX_UNITS],
    count: usize,
    /// The luma units are intra sub-partitions (`isp.rs`) of this split:
    /// the CU's chroma is one block, coded with the last.
    isp: Option<crate::isp::Split>,
}

impl CodedUnits {
    /// Per transform unit in decoding order: its luma block and whether
    /// luma, Cb and Cr code coefficients.
    pub(crate) fn as_slice(&self) -> &[(Block, [bool; 3])] {
        &self.units[..self.count]
    }

    pub(crate) fn any(&self) -> bool {
        self.as_slice()
            .iter()
            .flat_map(|(_, f)| f)
            .any(|&coded| coded)
    }

    pub(crate) fn isp(&self) -> Option<crate::isp::Split> {
        self.isp
    }
}

/// The transform units of CU `cu` (its sub-partitions `isp`, if split) and
/// their coded planes from its `levels`, with `MaxTbSizeY` `1 << max_log2`.
pub(crate) fn coded_units(
    cu: &Block,
    isp: Option<crate::isp::Split>,
    levels: &[Vec<i32>; 3],
    max_log2: u8,
) -> CodedUnits {
    let mut units = [(*cu, [false; 3]); MAX_UNITS];
    let coded = |levels: &[i32]| levels.iter().any(|&v| v != 0);
    let count = if let Some(split) = isp {
        let parts = crate::isp::parts(cu, split);
        for (index, (unit, part)) in units.iter_mut().zip(parts).enumerate() {
            unit.0 = part;
            unit.1[0] = coded(unit_levels(&levels[0], index, part.area()));
        }
        let last = &mut units[crate::isp::PARTS - 1].1;
        last[1] = coded(&levels[1]);
        last[2] = coded(&levels[2]);
        crate::isp::PARTS
    } else {
        for (index, (unit, tu)) in units.iter_mut().zip(self::units(cu, max_log2)).enumerate() {
            unit.0 = tu;
            for (plane, flag) in unit.1.iter_mut().enumerate() {
                let samples = plane_samples(&tu, plane);
                *flag = coded(unit_levels(&levels[plane], index, samples));
            }
        }
        count(cu, max_log2)
    };
    CodedUnits { units, count, isp }
}

/// The levels of plane `plane` (0 luma) of luma transform unit `tu`.
pub(crate) fn plane_samples(tu: &Block, plane: usize) -> usize {
    let shift = usize::from(plane != 0);
    coded_samples(tu.width() >> shift, tu.height() >> shift)
}

/// Copy transform unit `tu` of a prediction of `cu` (`cu`'s width per row,
/// `shift` 1 for chroma) into `out`, `tu`'s area of samples.
pub(crate) fn extract(prediction: &[u8], cu: &Block, tu: &Block, shift: usize, out: &mut Vec<u8>) {
    let stride = cu.width() >> shift;
    let (w, h) = (tu.width() >> shift, tu.height() >> shift);
    let (dx, dy) = ((tu.x - cu.x) >> shift, (tu.y - cu.y) >> shift);
    out.clear();
    for row in prediction[dy * stride..].chunks_exact(stride).take(h) {
        out.extend_from_slice(&row[dx..dx + w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_64_cu_codes_four_32_tus_in_z_order() {
        let cu = Block::square(64, 128, 6);
        let tus: Vec<(usize, usize, u8)> = units(&cu, MAX_TB_LOG2)
            .map(|b| (b.x, b.y, b.log2_size()))
            .collect();
        assert_eq!(
            tus,
            [(64, 128, 5), (96, 128, 5), (64, 160, 5), (96, 160, 5)]
        );
        assert!(splits(&cu, MAX_TB_LOG2));
        assert_eq!(count(&cu, MAX_TB_LOG2), 4);
        let small = Block::square(8, 8, 5);
        assert_eq!(units(&small, MAX_TB_LOG2).count(), 1);
        assert!(!splits(&small, MAX_TB_LOG2));
    }

    #[test]
    fn with_64_point_transforms_a_64_cu_is_one_unit_of_32x32_levels() {
        let cu = Block::square(0, 64, 6);
        let large = MAX_TB_LOG2 + 1;
        assert_eq!(units(&cu, large).collect::<Vec<_>>(), [cu]);
        assert!(!splits(&cu, large));
        assert_eq!(plane_samples(&cu, 0), 1024);
        assert_eq!(plane_samples(&cu, 1), 1024);
        let levels = [vec![0; 1024], vec![1; 1024], Vec::new()];
        let coded = coded_units(&cu, None, &levels, large);
        assert_eq!(coded.as_slice(), &[(cu, [false, true, false])]);
    }

    #[test]
    fn long_rectangles_split_into_halves() {
        let wide = Block::rect(0, 32, 6, 5);
        let tus: Vec<Block> = units(&wide, MAX_TB_LOG2).collect();
        assert_eq!(tus, [Block::square(0, 32, 5), Block::square(32, 32, 5)]);
        let tall = Block::rect(16, 0, 4, 6);
        let tus: Vec<Block> = units(&tall, MAX_TB_LOG2).collect();
        assert_eq!(tus, [Block::rect(16, 0, 4, 5), Block::rect(16, 32, 4, 5)]);
        assert!(splits(&tall, MAX_TB_LOG2));
        let short = Block::rect(0, 0, 5, 3);
        assert_eq!(units(&short, MAX_TB_LOG2).collect::<Vec<_>>(), [short]);
        assert!(!splits(&short, MAX_TB_LOG2));
    }

    #[test]
    fn earlier_units_of_the_cu_are_decoded() {
        let cu = Block::square(0, 0, 6);
        let [first, second, third, _] =
            <[Block; 4]>::try_from(units(&cu, MAX_TB_LOG2).collect::<Vec<_>>())
                .expect("four units");
        // Above-right of the third unit is the second: decoded.
        assert!(decoded_before(&cu, &third, 32, 31));
        // Below-left of the second unit is the third: not yet.
        assert!(!decoded_before(&cu, &second, 31, 32));
        assert!(!decoded_before(&cu, &first, 0, 0));
        // Outside the CU this rule says nothing.
        assert!(!decoded_before(&cu, &third, 64, 31));
    }

    #[test]
    fn unit_levels_round_trip() {
        let mut levels = Vec::new();
        place(&mut levels, 4, 2, vec![3; 16]);
        place(&mut levels, 4, 0, Vec::new());
        assert_eq!(levels.len(), 64);
        assert_eq!(unit_levels(&levels, 2, 16), &[3; 16]);
        assert!(unit_levels(&levels, 1, 16).iter().all(|&v| v == 0));
        assert_eq!(unit_levels(&[], 3, 16), [0; 0]);
    }

    #[test]
    fn extracts_a_quadrant() {
        let cu = Block::square(0, 0, 6);
        let chroma: Vec<u8> = (0..32 * 32).map(|i| (i % 251) as u8).collect();
        let tu = units(&cu, MAX_TB_LOG2).nth(3).unwrap();
        let mut out = Vec::new();
        extract(&chroma, &cu, &tu, 1, &mut out);
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], chroma[16 * 32 + 16]);
        assert_eq!(out[255], chroma[31 * 32 + 31]);
    }
}
