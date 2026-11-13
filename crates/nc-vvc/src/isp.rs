//! H.266 intra sub-partitions (ISP, §7.3.11.5, §8.4.5.1): an intra CU's
//! luma coded as four transform blocks, horizontal strips or vertical
//! columns, each predicted in the CU's direction from the reconstruction of
//! the one before it. Chroma stays one transform block per CU, coded with
//! the last sub-partition (§7.3.11.10).
//!
//! References (§8.4.5.2.8): the first sub-partition's are the CU's own
//! (unfiltered); each later one reads the previous sub-partition's last row
//! (strips) or column (columns), repeated beyond the CU, and the CU's left
//! (strips) or top (columns) references from its own offset. Syntax, cbf
//! contexts and reference setup follow `VVenC` `CABACWriter` and `VVdeC`
//! `IntraPrediction` (BSD-3-Clause-Clear; `THIRD_PARTY.md`).
use crate::{
    coeff::{Bins, BitCost},
    contexts::Contexts,
    intra::References,
    recon,
    transform_tree::MAX_TB_LOG2,
};
use nc_common::{Picture, PlaneIndex, PlaneViewMut, decode::Result};
use nc_dsp::Dsp;
use nc_encoder::Block;

/// `intra_subpartitions_split_flag` (§7.4.12.5): 0 horizontal, 1 vertical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Split {
    /// Strips as wide as the CU, top to bottom.
    Horizontal,
    /// Columns as tall as the CU, left to right.
    Vertical,
}

impl Split {
    pub(crate) const ALL: [Self; 2] = [Self::Horizontal, Self::Vertical];
}

/// Sub-partitions of every CU we split (`NumIntraSubPartitions`: two only
/// for 4x8 and 8x4 CUs, which our quadtree never makes).
pub(crate) const PARTS: usize = 4;

/// The narrowest prediction region (§8.4.5.1 `predRegDiffFromTB`): columns
/// narrower than this (an 8x8 CU's 2-wide ones) are predicted in pairs, as
/// one region this wide, from the reconstruction before the pair.
const MIN_REGION_WIDTH: usize = 4;

/// The luma deblocking grid (§8.8.3.3): edges between sub-partitions off it
/// (inside an 8x8 CU's 2-sample strips or columns) are not filtered.
pub(crate) const DEBLOCK_GRID: usize = 4;

/// Whether CU `cu` signals `intra_subpartitions_mode_flag` (§7.3.11.5:
/// no side above `MaxTbSizeY` = `1 << max_tb_log2`, more than 16 samples).
pub(crate) fn signalled(cu: &Block, max_tb_log2: u8) -> bool {
    cu.log2_width.max(cu.log2_height) <= max_tb_log2 && cu.log2_width + cu.log2_height > 4
}

/// Whether our encoder and decoder split CU `cu`: every square CU up to
/// 32x32 (no 64-point sub-partitions) that signals the flag, except that
/// with MTS enabled (`mts`) an 8x8 CU's 2-sample sub-partitions are not
/// made (their implicit DST-VII side has no forward kernel here; MTS is off
/// in every preset).
pub(crate) fn supported(cu: &Block, mts: bool) -> bool {
    signalled(cu, MAX_TB_LOG2) && cu.is_square() && (!mts || cu.log2_size() >= MIN_MTS_LOG2)
}

/// The smallest CU whose sub-partitions are at least 4 a side.
const MIN_MTS_LOG2: u8 = 4;

/// The sub-partitions of `cu` in decoding order.
pub(crate) fn parts(cu: &Block, split: Split) -> [Block; PARTS] {
    debug_assert!(supported(cu, false));
    std::array::from_fn(|i| match split {
        Split::Horizontal => Block::rect(
            cu.x,
            cu.y + i * (cu.height() / PARTS),
            cu.log2_width,
            cu.log2_height - 2,
        ),
        Split::Vertical => Block::rect(
            cu.x + i * (cu.width() / PARTS),
            cu.y,
            cu.log2_width - 2,
            cu.log2_height,
        ),
    })
}

/// General ISP geometry: each partition contains at least sixteen samples.
/// Narrow 4x8/8x4 CUs use two partitions; others use four.
pub(crate) fn general_parts(cu: &Block, split: Split) -> impl Iterator<Item = Block> {
    let (along, across) = match split {
        Split::Horizontal => (cu.height(), cu.width()),
        Split::Vertical => (cu.width(), cu.height()),
    };
    let length = (along / PARTS).max(16 / across).max(1);
    let cu = *cu;
    (0..along / length).map(move |i| match split {
        Split::Horizontal => {
            Block::rect(cu.x, cu.y + i * length, cu.log2_width, length.ilog2() as u8)
        }
        Split::Vertical => Block::rect(
            cu.x + i * length,
            cu.y,
            length.ilog2() as u8,
            cu.log2_height,
        ),
    })
}

/// §7.3.11.5 `intra_subpartitions_mode_flag` and split flag of CU `cu`
/// (with the SPS's `tools`: `sps_isp_enabled_flag`, `MaxTbSizeY`); returns
/// the split (the selection when writing or pricing).
pub(crate) fn code_split(
    io: &mut impl Bins,
    c: &mut Contexts,
    tools: crate::headers::Tools,
    cu: &Block,
    split: Option<Split>,
) -> Result<Option<Split>> {
    if !tools.isp
        || !signalled(cu, tools.max_tb_log2())
        || !io.bin(&mut c.isp_mode, split.is_some())?
    {
        return Ok(None);
    }
    Ok(Some(
        if io.bin(&mut c.isp_split, split == Some(Split::Vertical))? {
            Split::Vertical
        } else {
            Split::Horizontal
        },
    ))
}

/// §7.3.11.10 `tu_y_coded_flag` of sub-partition `index` (`coded`); the
/// last one's is inferred 1 when no earlier sub-partition coded any
/// (`InferTuCbfLuma`). Its context is 2 + the previous flag (§9.3.4.2.5).
pub(crate) fn code_luma_flag(
    io: &mut impl Bins,
    c: &mut Contexts,
    index: usize,
    earlier: &[bool],
    coded: bool,
) -> Result<bool> {
    code_luma_flag_count(io, c, index, PARTS, earlier, coded)
}

/// The same coded-flag inference for the sequence's two/four-part ISP geometry.
pub(crate) fn code_luma_flag_count(
    io: &mut impl Bins,
    c: &mut Contexts,
    index: usize,
    count: usize,
    earlier: &[bool],
    coded: bool,
) -> Result<bool> {
    if index == count - 1 && !earlier.iter().any(|&f| f) {
        return Ok(true);
    }
    let previous = index > 0 && earlier[index - 1];
    io.bin(&mut c.cbf_y_isp[usize::from(previous)], coded)
}

/// Reference continuation shared by subset and general ISP reconstruction.
pub(crate) fn edges_for_part(
    (top, left, corner): (&[i32], &[i32], i32),
    cu: &Block,
    part: &Block,
    split: Split,
    previous: &[u8],
    other_available: bool,
) -> (Vec<i32>, Vec<i32>, i32) {
    let (ref_w, ref_h) = (cu.width() + part.width(), cu.height() + part.height());
    if part.x == cu.x && part.y == cu.y {
        return (top[..ref_w].to_vec(), left[..ref_h].to_vec(), corner);
    }
    let (line_length, other_length) = match split {
        Split::Horizontal => (ref_w, ref_h),
        Split::Vertical => (ref_h, ref_w),
    };
    // The previous sub-partition's line, then (beyond the CU, not yet
    // decoded) its last sample repeated (§8.4.5.2.8 substitution).
    let line: Vec<i32> = previous
        .iter()
        .map(|&v| i32::from(v))
        .chain(std::iter::repeat(i32::from(previous[previous.len() - 1])))
        .take(line_length)
        .collect();
    // The CU's other edge from this sub-partition's offset, its corner the
    // sample before; when that edge lies outside the picture every one of
    // its samples, and the corner, take the line's first (the first
    // available sample of §8.4.5.2.8's chain).
    let (offset, other) = match split {
        Split::Horizontal => (part.y - cu.y, left),
        Split::Vertical => (part.x - cu.x, top),
    };
    let (other, corner) = if other_available {
        (
            other[offset..offset + other_length].to_vec(),
            other[offset - 1],
        )
    } else {
        (vec![line[0]; other_length], line[0])
    };
    match split {
        Split::Horizontal => (line, other, corner),
        Split::Vertical => (other, line, corner),
    }
}

/// The prediction region of sub-partition `part` (§8.4.5.1
/// `predRegDiffFromTB`): the part itself, or the [`MIN_REGION_WIDTH`]-wide
/// column holding it.
fn region(part: &Block) -> Block {
    if part.width() >= MIN_REGION_WIDTH {
        return *part;
    }
    let log2 = MIN_REGION_WIDTH.ilog2() as u8;
    Block::rect(
        part.x & !(MIN_REGION_WIDTH - 1),
        part.y,
        log2,
        part.log2_height,
    )
}

/// The luma predictions of one ISP CU's sub-partitions, in decoding order:
/// each region is predicted from the CU's own references and the
/// reconstructed line before it (§8.4.5.2.8), once, however many
/// sub-partitions share it.
pub(crate) struct Predictor<'a> {
    references: &'a References,
    cu: Block,
    split: Split,
    mode: u8,
    /// The last region predicted and its samples.
    region: Option<(Block, Vec<u8>)>,
}

impl<'a> Predictor<'a> {
    /// The predictor of `cu`'s sub-partitions `split` in direction `mode`;
    /// `references` are the CU's (from the reconstruction before it).
    pub(crate) fn new(references: &'a References, cu: &Block, split: Split, mode: u8) -> Self {
        Self {
            references,
            cu: *cu,
            split,
            mode,
            region: None,
        }
    }

    /// The prediction of sub-partition `index`. `previous` gives the
    /// reconstructed line a region reads (see [`previous_line`]); it is
    /// called only when a new region starts after the first.
    pub(crate) fn predict(
        &mut self,
        dsp: &Dsp,
        index: usize,
        previous: impl FnOnce(&Block) -> Vec<u8>,
    ) -> Vec<u8> {
        let (cu, split) = (self.cu, self.split);
        let part = parts(&cu, split)[index];
        let region = region(&part);
        if self.region.as_ref().is_none_or(|(r, _)| *r != region) {
            let first = region.x == cu.x && region.y == cu.y;
            let line = if first { Vec::new() } else { previous(&region) };
            let other_available = match split {
                Split::Horizontal => cu.x != 0,
                Split::Vertical => cu.y != 0,
            };
            let (top, left, corner) = edges_for_part(
                self.references.raw_edges(),
                &cu,
                &region,
                split,
                &line,
                other_available,
            );
            let mut samples = vec![0; region.area()];
            dsp.vvc_intra_predict_isp(
                (cu.width(), cu.height()),
                (region.width(), region.height()),
                self.mode,
                &top,
                &left,
                corner,
                &mut samples,
            );
            self.region = Some((region, samples));
        }
        let (region, samples) = self.region.as_ref().expect("predicted above");
        if *region == part {
            return samples.clone();
        }
        let offset = part.x - region.x;
        samples
            .chunks_exact(region.width())
            .flat_map(|row| &row[offset..offset + part.width()])
            .copied()
            .collect()
    }
}

/// The reconstructed line before prediction region `region` of `cu`: row
/// `region.y - 1` (strips) or column `region.x - 1` (columns) of `cu`'s
/// reconstruction `samples` (`width` per row).
fn previous_line(
    samples: &[u8],
    width: usize,
    cu: &Block,
    split: Split,
    region: &Block,
) -> Vec<u8> {
    match split {
        Split::Horizontal => {
            let row = region.y - cu.y - 1;
            samples[row * width..(row + 1) * width].to_vec()
        }
        Split::Vertical => {
            let column = region.x - cu.x - 1;
            samples.chunks_exact(width).map(|row| row[column]).collect()
        }
    }
}

/// [`previous_line`] read from the picture being reconstructed.
pub(crate) fn previous_in(picture: &Picture, cu: &Block, split: Split, region: &Block) -> Vec<u8> {
    let view = picture.plane(PlaneIndex::Y).view();
    match split {
        Split::Horizontal => view.row(region.y - 1)[cu.x..cu.x + cu.width()].to_vec(),
        Split::Vertical => (cu.y..cu.y + cu.height())
            .map(|y| view.row(y)[region.x - 1])
            .collect(),
    }
}

/// The RD cost of coding `cu`'s luma with sub-partitions `split` in
/// direction `mode`: reconstruction SSE plus λ times the ISP flags, the
/// direction (MPM list `mpm`), the coded flags and the coefficients, each
/// sub-partition quantised and reconstructed before the next is predicted.
/// Infinite once the sub-partitions so far reach `bound` (the best cost
/// found): both terms only grow, so the rest cannot bring it below (exact;
/// most candidates stop after one or two sub-partitions).
#[allow(clippy::too_many_arguments, reason = "one CU's complete search inputs")]
pub(crate) fn cost(
    dsp: &Dsp,
    source: &Picture,
    cu_refs: &References,
    cu: &Block,
    (split, mode, mpm): (Split, u8, [u8; 6]),
    contexts: &Contexts,
    (quant, lambda, bound): (recon::Quant, f32, f32),
    scratch: &mut recon::QuantScratch,
) -> f32 {
    let mut c = contexts.clone();
    let mut bits = BitCost::default();
    let tools = crate::headers::Tools {
        isp: true,
        ..crate::headers::Tools::default()
    };
    code_split(&mut bits, &mut c, tools, cu, Some(split)).expect("valid ISP flags");
    crate::intra::mode(&mut bits, &mut c, mpm, mode, true).expect("valid intra mode");
    let width = cu.width();
    let mut samples = vec![0u8; cu.area()];
    let mut coded = [false; PARTS];
    let mut distortion = 0u64;
    let mut predictor = Predictor::new(cu_refs, cu, split, mode);
    for (index, part) in parts(cu, split).into_iter().enumerate() {
        let mut pred = predictor.predict(dsp, index, |region| {
            previous_line(&samples, width, cu, split, region)
        });
        let levels = recon::quantize_rect(
            dsp,
            source,
            PlaneIndex::Y,
            (part.x, part.y, part.width()),
            &pred,
            quant,
            None,
            scratch,
        );
        coded[index] = levels.iter().any(|&v| v != 0);
        if index == PARTS - 1 && !coded.iter().any(|&f| f) {
            // The last flag would be inferred 1 with nothing to code: a
            // CU without luma residual cannot use ISP (as `VVenC`).
            return f32::INFINITY;
        }
        code_luma_flag(&mut bits, &mut c, index, &coded[..index], coded[index])
            .expect("valid coded flag");
        let (w, h) = (part.width(), part.height());
        {
            let mut view = PlaneViewMut::new(&mut pred, w, w, h).unwrap();
            recon::add(dsp, &mut view, levels, quant.scaling());
        }
        if coded[index] {
            crate::coeff::emit_rect(&mut bits, &mut c.coeff, levels, (w, h), 0);
        }
        let source_view = source
            .plane(PlaneIndex::Y)
            .region(part.x as isize, part.y as isize, w, h)
            .unwrap();
        for (row, line) in pred.chunks_exact(w).enumerate() {
            let source_row = source_view.row(row);
            distortion += line
                .iter()
                .zip(source_row)
                .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
                .sum::<u64>();
            let (dx, dy) = (part.x - cu.x, part.y - cu.y + row);
            samples[dy * width + dx..dy * width + dx + w].copy_from_slice(line);
        }
        if distortion as f32 + lambda * bits.0 >= bound {
            return f32::INFINITY;
        }
    }
    distortion as f32 + lambda * bits.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_32_cu_splits_into_four_strips_or_columns() {
        let cu = Block::square(32, 64, 5);
        let strips = parts(&cu, Split::Horizontal);
        assert_eq!(strips[1], Block::rect(32, 72, 5, 3));
        let columns = parts(&cu, Split::Vertical);
        assert_eq!(columns[3], Block::rect(56, 64, 3, 5));
        assert!(supported(&cu, false));
        assert!(supported(&Block::square(0, 0, 3), false));
        assert!(
            supported(&Block::square(0, 0, 4), true) && !supported(&Block::square(0, 0, 3), true)
        );
        assert!(signalled(&Block::square(0, 0, 3), MAX_TB_LOG2));
        assert!(!signalled(&Block::square(0, 0, 2), MAX_TB_LOG2));
        assert!(!signalled(&Block::square(0, 0, 6), MAX_TB_LOG2));
        // With 64-point transforms a 64x64 CU signals the flag, which we
        // never set.
        assert!(signalled(&Block::square(0, 0, 6), MAX_TB_LOG2 + 1));
        assert!(!supported(&Block::square(0, 0, 6), false));
        let columns = parts(&Block::square(8, 0, 3), Split::Vertical);
        assert_eq!(columns[3], Block::rect(14, 0, 1, 3));
        assert_eq!(region(&columns[3]), Block::rect(12, 0, 2, 3));
    }

    #[test]
    fn the_last_flag_is_inferred_after_empty_sub_partitions() {
        let mut c = Contexts::new(32, false);
        let mut bits = BitCost::default();
        assert!(code_luma_flag(&mut bits, &mut c, 3, &[false; 3], true).unwrap());
        assert_eq!(bits.0, 0.);
        code_luma_flag(&mut bits, &mut c, 3, &[false, true, false], false).unwrap();
        assert!(bits.0 > 0.);
    }
}
