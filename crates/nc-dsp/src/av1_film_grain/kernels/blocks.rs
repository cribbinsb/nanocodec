//! Shared block offsets for applying luma grain directly from the template.

use super::{Av1FilmGrainParams, GrainBlocks, PlaneView, PlaneViewMut, Template};
use crate::av1_film_grain::{AR_PADDING, BLOCK, Random};
use nc_common::Sample;

pub(in crate::av1_film_grain) struct LumaBlock<T = u8, G = i8> {
    pub source: (*const T, usize),
    pub output: (*mut T, usize),
    pub noise: (*const G, usize),
    pub left: Option<*const G>,
    pub above: Option<(*const G, Option<*const G>)>,
    pub rows: usize,
}

pub(in crate::av1_film_grain) fn whole_blocks(
    blocks: GrainBlocks,
    (width, height): (usize, usize),
) -> bool {
    blocks.luma_side() == BLOCK && width.is_multiple_of(BLOCK) && height.is_multiple_of(2)
}

/// Call `apply` for each complete 32-column block, with the template locations
/// of its two-sample overlap already advanced to the edge.
#[allow(
    clippy::inline_always,
    reason = "keep each ISA's scaling table in registers"
)]
#[inline(always)]
pub(in crate::av1_film_grain) fn visit<T: Sample, G>(
    (params, blocks): (&Av1FilmGrainParams, GrainBlocks),
    template: &Template<G>,
    (source, output): (&PlaneView<'_, T>, &mut PlaneViewMut<'_, T>),
    (width, height): (usize, usize),
    mut apply: impl FnMut(LumaBlock<T, G>),
) {
    let ss = source.stride();
    let ds = output.stride();
    let gs = template.columns;
    let src = source.data().as_ptr();
    let dst = output.data_mut().as_mut_ptr();
    let grain = template.values.as_ptr();
    let next_origin = |rng: &mut Random| {
        let (x, y) = blocks.offsets(rng);
        let origin = |offset| AR_PADDING + 2 * (AR_PADDING + offset);
        origin(y) * gs + origin(x)
    };
    for (index, stripe) in (0..height).step_by(BLOCK).enumerate() {
        let rows = BLOCK.min(height - stripe);
        let vertical = params.overlap && index != 0;
        let mut current = Random(blocks.stripe_seed(params.grain_seed, index));
        let mut previous = Random(blocks.stripe_seed(params.grain_seed, index.saturating_sub(1)));
        let mut left = None;
        let mut top_left = None;
        for x in (0..width).step_by(BLOCK) {
            let origin = grain.wrapping_add(next_origin(&mut current));
            let above = if vertical {
                let top = grain.wrapping_add(next_origin(&mut previous) + BLOCK * gs);
                let above = Some((top, top_left));
                top_left = params.overlap.then(|| top.wrapping_add(BLOCK));
                above
            } else {
                None
            };
            apply(LumaBlock {
                source: (src.wrapping_add(stripe * ss + x), ss),
                output: (dst.wrapping_add(stripe * ds + x), ds),
                noise: (origin, gs),
                left,
                above,
                rows,
            });
            left = params.overlap.then(|| origin.wrapping_add(BLOCK));
        }
    }
}
