//! Film-grain stripe scheduling over shared row kernels, without full-frame noise buffers:
//! 8-bit with the ISA's row kernels ([`Dsp`]), deeper samples with the scalar ones
//! ([`DeepRows`]), one driver for both ([`GrainRows`]).
use super::kernels::{blend_at, chroma_at, luma_at};
use super::{
    AR_PADDING, Av1FilmGrainParams, Av1GrainScaling, BLOCK, Grain, OVERLAP, RESTRICTED_MAX_CHROMA,
    RESTRICTED_MAX_LUMA, RESTRICTED_MIN, Random, Template, blend_overlap, grain_range, scaling_lut,
    stripe_seed, templates, templates_at,
};
use crate::Dsp;
use nc_common::{BitDepth, PlaneView, PlaneViewMut, Sample};

/// The row kernels the synthesis driver calls, on samples `T` with grain
/// stored as `G`: [`Dsp`]'s 8-bit table, or [`DeepRows`] at a depth.
trait GrainRows<T, G> {
    /// The samples' depth.
    fn depth(&self) -> BitDepth;
    /// Blend the overlap row `old` into `new`.
    fn blend_row(&self, old: &[G], new: &mut [G], weights: [i16; 2]);
    /// Luma grain of one row, clipped to `bounds`.
    fn luma_row(
        &self,
        source: &[T],
        grain: &[G],
        scaling: &Av1GrainScaling,
        bounds: [i32; 2],
        out: &mut [T],
    );
    /// Chroma grain of one row from the co-located luma row.
    #[allow(clippy::too_many_arguments)]
    fn chroma_row(
        &self,
        source: &[T],
        luma: &[T],
        grain: &[G],
        scaling: &Av1GrainScaling,
        bounds: [i32; 2],
        coefficients: [i16; 3],
        out: &mut [T],
    );
}

impl GrainRows<u8, i8> for Dsp {
    fn depth(&self) -> BitDepth {
        BitDepth::EIGHT
    }
    fn blend_row(&self, old: &[i8], new: &mut [i8], weights: [i16; 2]) {
        self.av1_grain_blend_row(old, new, weights);
    }
    fn luma_row(
        &self,
        source: &[u8],
        grain: &[i8],
        scaling: &Av1GrainScaling,
        bounds: [i32; 2],
        out: &mut [u8],
    ) {
        self.av1_grain_luma_row(source, grain, scaling, bounds.map(|b| b as u8), out);
    }
    fn chroma_row(
        &self,
        source: &[u8],
        luma: &[u8],
        grain: &[i8],
        scaling: &Av1GrainScaling,
        bounds: [i32; 2],
        coefficients: [i16; 3],
        out: &mut [u8],
    ) {
        let bounds = bounds.map(|b| b as u8);
        self.av1_grain_chroma_row(source, luma, grain, scaling, bounds, coefficients, out);
    }
}

/// The scalar row kernels at a depth above 8 bits.
struct DeepRows(BitDepth);

impl<T: Sample> GrainRows<T, i16> for DeepRows {
    fn depth(&self) -> BitDepth {
        self.0
    }
    fn blend_row(&self, old: &[i16], new: &mut [i16], weights: [i16; 2]) {
        blend_at(old, new, weights, grain_range(self.0));
    }
    fn luma_row(
        &self,
        source: &[T],
        grain: &[i16],
        scaling: &Av1GrainScaling,
        bounds: [i32; 2],
        out: &mut [T],
    ) {
        luma_at(source, grain, scaling, (bounds, self.0.above_eight()), out);
    }
    fn chroma_row(
        &self,
        source: &[T],
        luma: &[T],
        grain: &[i16],
        scaling: &Av1GrainScaling,
        bounds: [i32; 2],
        coefficients: [i16; 3],
        out: &mut [T],
    ) {
        let bits = self.0.above_eight();
        chroma_at(
            source,
            luma,
            grain,
            scaling,
            (bounds, bits),
            coefficients,
            out,
        );
    }
}

impl Dsp {
    /// Apply normative AV1 8-bit 4:2:0 grain to shown output, leaving references untouched.
    /// Templates and intensity tables are prepared once for the frame's parameters/seed.
    /// # Panics
    /// If any plane is smaller than the requested frame dimensions.
    pub fn av1_add_film_grain(
        &self,
        params: &Av1FilmGrainParams,
        input: [&PlaneView<'_>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_>; 3],
    ) {
        add_film_grain(self, params, &templates(params), (input, size), output);
    }

    /// [`Self::av1_add_film_grain`] of samples of any depth, with the
    /// scalar row kernels (AV1 §7.18.3 at `BitDepth`: grain range, shifts,
    /// clips and the interpolated scaling function).
    ///
    /// # Panics
    ///
    /// As [`Self::av1_add_film_grain`].
    pub fn av1_add_film_grain_at<T: Sample>(
        &self,
        params: &Av1FilmGrainParams,
        input: [&PlaneView<'_, T>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_, T>; 3],
        depth: BitDepth,
    ) {
        let templates = templates_at::<i16>(params, depth);
        add_film_grain(&DeepRows(depth), params, &templates, (input, size), output);
    }
}

/// Film grain of every plane with `rows`.
fn add_film_grain<T: Sample, G: Grain>(
    rows: &impl GrainRows<T, G>,
    params: &Av1FilmGrainParams,
    templates: &[Template<G>; 3],
    (input, size): ([&PlaneView<'_, T>; 3], (usize, usize)),
    output: [&mut PlaneViewMut<'_, T>; 3],
) {
    let shift = params.grain_scaling_minus_8 + 8;
    let luma_lut = scaling_lut(&params.y_points);
    let luma = Av1GrainScaling::new(&luma_lut, shift);
    let chroma = if params.chroma_scaling_from_luma {
        [luma.clone(), luma.clone()]
    } else {
        [
            Av1GrainScaling::new(&scaling_lut(&params.cb_points), shift),
            Av1GrainScaling::new(&scaling_lut(&params.cr_points), shift),
        ]
    };
    for (plane, destination) in output.into_iter().enumerate() {
        let table = if plane == 0 {
            &luma
        } else {
            &chroma[plane - 1]
        };
        synthesize_plane(
            rows,
            params,
            &templates[plane],
            plane,
            (input, destination),
            table,
            size,
        );
    }
}

fn synthesize_plane<T: Sample, G: Grain>(
    rows: &impl GrainRows<T, G>,
    params: &Av1FilmGrainParams,
    template: &Template<G>,
    plane: usize,
    (input, output): ([&PlaneView<'_, T>; 3], &mut PlaneViewMut<'_, T>),
    scaling: &Av1GrainScaling,
    (width, height): (usize, usize),
) {
    let sub = usize::from(plane != 0);
    let (plane_w, plane_h) = ((width + sub) >> sub, (height + sub) >> sub);
    assert!(input[plane].width() >= plane_w && input[plane].height() >= plane_h);
    assert!(output.width() >= plane_w && output.height() >= plane_h);
    let points = [
        params.y_points.count,
        params.cb_points.count,
        params.cr_points.count,
    ][plane];
    if points == 0 && !(plane != 0 && params.chroma_scaling_from_luma) {
        for y in 0..plane_h {
            output.row_mut(y)[..plane_w].copy_from_slice(&input[plane].row(y)[..plane_w]);
        }
        return;
    }
    let block = BLOCK >> sub;
    let reach = OVERLAP >> sub;
    let stride = plane_w.div_ceil(block) * block + reach;
    let mut current = vec![G::default(); (block + reach) * stride];
    let mut previous = vec![G::default(); current.len()];
    let depth = rows.depth();
    let bits = depth.above_eight();
    let bounds = if params.clip_to_restricted_range {
        [
            RESTRICTED_MIN << bits,
            if plane == 0 {
                RESTRICTED_MAX_LUMA
            } else {
                RESTRICTED_MAX_CHROMA
            } << bits,
        ]
    } else {
        [0, depth.max()]
    };
    let coefficients = if params.chroma_scaling_from_luma {
        [0, 64, 0]
    } else {
        let (mult, luma_mult, offset) = if plane == 1 {
            params.cb_mult
        } else {
            params.cr_mult
        };
        [
            i16::from(mult) - 128,
            i16::from(luma_mult) - 128,
            offset as i16 - 256,
        ]
    };
    for (stripe_index, first_row) in (0..plane_h).step_by(block).enumerate() {
        fill_stripe(
            params,
            template,
            (sub, depth),
            stripe_index,
            stride,
            &mut current,
        );
        for y in 0..block.min(plane_h - first_row) {
            let noise = &mut current[y * stride..y * stride + plane_w];
            if params.overlap && stripe_index != 0 && y < reach {
                let old = &previous[(block + y) * stride..(block + y) * stride + plane_w];
                rows.blend_row(old, noise, weights(sub, y));
            }
            let source = &input[plane].row(first_row + y)[..plane_w];
            let destination = &mut output.row_mut(first_row + y)[..plane_w];
            if plane == 0 {
                rows.luma_row(source, noise, scaling, bounds, destination);
            } else {
                let luma = &input[0].row((first_row + y) * 2)[..width];
                rows.chroma_row(
                    source,
                    luma,
                    noise,
                    scaling,
                    bounds,
                    coefficients,
                    destination,
                );
            }
        }
        // Only the final overlap rows survive into the next stripe. Reusing both
        // buffers avoids frame-sized scratch and allocation per block row.
        std::mem::swap(&mut current, &mut previous);
    }
}

fn weights(sub: usize, index: usize) -> [i16; 2] {
    match (sub, index) {
        (1, _) => [23, 22],
        (_, 0) => [27, 17],
        _ => [17, 27],
    }
}

/// Lay stripe `index`'s grain blocks (each `block + reach` square, from a
/// random offset into the template) side by side in `stripe`, blending each
/// block's first `reach` columns over the previous block's overhang.
fn fill_stripe<G: Grain>(
    params: &Av1FilmGrainParams,
    template: &Template<G>,
    (sub, depth): (usize, BitDepth),
    index: usize,
    stride: usize,
    stripe: &mut [G],
) {
    // The block side is a constant per plane type, so each row's copy is a
    // fixed-size move rather than a `memcpy` call.
    const LUMA: usize = BLOCK + OVERLAP;
    const CHROMA: usize = (BLOCK >> 1) + (OVERLAP >> 1);
    if sub == 0 {
        fill_stripe_blocks::<G, LUMA>(params, template, (sub, depth), index, stride, stripe);
    } else {
        fill_stripe_blocks::<G, CHROMA>(params, template, (sub, depth), index, stride, stripe);
    }
}

/// [`fill_stripe`] for blocks of side `SIDE` (`block + reach`).
fn fill_stripe_blocks<G: Grain, const SIDE: usize>(
    params: &Av1FilmGrainParams,
    template: &Template<G>,
    (sub, depth): (usize, BitDepth),
    index: usize,
    stride: usize,
    stripe: &mut [G],
) {
    let range = grain_range(depth);
    let block = BLOCK >> sub;
    let reach = OVERLAP >> sub;
    debug_assert_eq!(block + reach, SIDE);
    let mut random = Random(stripe_seed(params.grain_seed, index));
    for x in (0..stride - reach).step_by(block) {
        let offset = random.next(8);
        let base_x = AR_PADDING + (AR_PADDING << (1 - sub)) + ((offset >> 4) << (1 - sub));
        let base_y = AR_PADDING + (AR_PADDING << (1 - sub)) + ((offset & 15) << (1 - sub));
        let overlap = if params.overlap && x != 0 { reach } else { 0 };
        for y in 0..SIDE {
            let start = (base_y + y) * template.columns + base_x;
            let source: &[G; SIDE] = template.values[start..start + SIDE]
                .try_into()
                .expect("a template row holds a block");
            let destination: &mut [G; SIDE] = (&mut stripe[y * stride + x..][..SIDE])
                .try_into()
                .expect("a stripe row holds a block");
            // The previous block's overhang, then the whole row at once (a
            // fixed-size move), then the overhang blended back in.
            let old = [destination[0], destination[1]];
            *destination = *source;
            for (j, &previous) in old.iter().enumerate().take(overlap) {
                destination[j] = G::from_i32(blend_overlap(
                    previous.into(),
                    source[j].into(),
                    (j, sub),
                    range,
                ));
            }
        }
    }
}
