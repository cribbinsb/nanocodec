//! Film-grain stripe scheduling over shared row kernels, without full-frame noise buffers:
//! 8-bit with the ISA's row kernels ([`Dsp`]), deeper samples with the scalar ones
//! ([`DeepRows`]), one driver for both ([`GrainRows`]), AV1's and AV2's block layouts
//! ([`GrainBlocks`]).
use super::high_simd::HighGrainScaling;
use super::kernels::{blend_at, chroma_at, luma_at};
use super::{
    AR_PADDING, AV2_SMALL_BLOCK, Av1FilmGrainParams, Av1GrainScaling, BLOCK, Grain, GrainBlocks,
    OVERLAP, RESTRICTED_MAX_CHROMA, RESTRICTED_MAX_LUMA, RESTRICTED_MIN, Random, Template,
    blend_overlap, grain_range, scaling_lut, templates, templates_at,
};
use crate::Dsp;
use nc_common::{BitDepth, ChromaFormat, PlaneIndex, PlaneView, PlaneViewMut, Sample};

/// The row kernels the synthesis driver calls, on samples `T` with grain
/// stored as `G`: [`Dsp`]'s 8-bit table, [`HighRows`] (`u16` SIMD), or [`DeepRows`].
trait GrainRows<T: Sample, G> {
    /// Per-plane scaling lookup table prepared once per frame.
    type Scaling;
    /// The samples' depth.
    fn depth(&self) -> BitDepth;
    /// Prepare the plane's scaling lookup table once per frame.
    fn prepare_scaling(&self, lut: &[u8; 256], shift: u8) -> Self::Scaling;
    /// Blend the overlap row `old` into `new`.
    fn blend_row(&self, old: &[G], new: &mut [G], weights: [i16; 2]);
    /// Apply luma directly from the template when the ISA has a block kernel.
    fn try_luma_plane(
        &self,
        _grain: (&Av1FilmGrainParams, GrainBlocks),
        _template: &Template<G>,
        _scaling: &Self::Scaling,
        _bounds: [i32; 2],
        _views: (&PlaneView<'_, T>, &mut PlaneViewMut<'_, T>),
        _size: (usize, usize),
    ) -> bool {
        false
    }
    /// Luma grain of one row, clipped to `bounds`.
    fn luma_row(
        &self,
        source: &[T],
        grain: &[G],
        scaling: &Self::Scaling,
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
        scaling: &Self::Scaling,
        bounds: [i32; 2],
        coefficients: [i16; 3],
        out: &mut [T],
    );
}

impl GrainRows<u8, i8> for Dsp {
    type Scaling = Av1GrainScaling;
    fn depth(&self) -> BitDepth {
        BitDepth::EIGHT
    }
    fn prepare_scaling(&self, lut: &[u8; 256], shift: u8) -> Self::Scaling {
        Av1GrainScaling::new(lut, shift)
    }
    fn blend_row(&self, old: &[i8], new: &mut [i8], weights: [i16; 2]) {
        self.av1_grain_blend_row(old, new, weights);
    }
    #[allow(
        unsafe_code,
        reason = "plane kernel selected for this CPU and checked geometry"
    )]
    fn try_luma_plane(
        &self,
        grain: (&Av1FilmGrainParams, GrainBlocks),
        template: &Template<i8>,
        scaling: &Av1GrainScaling,
        bounds: [i32; 2],
        views: (&PlaneView<'_>, &mut PlaneViewMut<'_>),
        size: (usize, usize),
    ) -> bool {
        // SAFETY: the kernel table matches this CPU, and synthesis checked
        // the input and output geometry.
        self.av1_grain().luma_plane.is_some_and(|plane| unsafe {
            plane(
                grain,
                template,
                scaling,
                bounds.map(|v| v as u8),
                views,
                size,
            )
        })
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

/// High-bit-depth (`u16`) SIMD row kernels at `depth`.
struct HighRows<'a>(&'a Dsp, BitDepth);

impl GrainRows<u16, i16> for HighRows<'_> {
    type Scaling = Box<HighGrainScaling>;
    fn depth(&self) -> BitDepth {
        self.1
    }
    fn prepare_scaling(&self, lut: &[u8; 256], shift: u8) -> Self::Scaling {
        HighGrainScaling::new(lut, shift, self.1)
    }
    fn blend_row(&self, old: &[i16], new: &mut [i16], weights: [i16; 2]) {
        self.0
            .av1_grain_high_blend_row(old, new, weights, grain_range(self.1));
    }
    fn try_luma_plane(
        &self,
        grain: (&Av1FilmGrainParams, GrainBlocks),
        template: &Template<i16>,
        scaling: &Self::Scaling,
        bounds: [i32; 2],
        views: (&PlaneView<'_, u16>, &mut PlaneViewMut<'_, u16>),
        size: (usize, usize),
    ) -> bool {
        self.0
            .av1_grain_high_luma_plane(grain, template, scaling, bounds, views, size)
    }
    fn luma_row(
        &self,
        source: &[u16],
        grain: &[i16],
        scaling: &Self::Scaling,
        bounds: [i32; 2],
        out: &mut [u16],
    ) {
        self.0
            .av1_grain_high_luma_row(source, grain, scaling, bounds, out);
    }
    fn chroma_row(
        &self,
        source: &[u16],
        luma: &[u16],
        grain: &[i16],
        scaling: &Self::Scaling,
        bounds: [i32; 2],
        coefficients: [i16; 3],
        out: &mut [u16],
    ) {
        self.0
            .av1_grain_high_chroma_row(source, luma, grain, scaling, bounds, coefficients, out);
    }
}

/// The scalar row kernels at a depth above 8 bits.
struct DeepRows(BitDepth);

impl<T: Sample> GrainRows<T, i16> for DeepRows {
    type Scaling = Av1GrainScaling;
    fn depth(&self) -> BitDepth {
        self.0
    }
    fn prepare_scaling(&self, lut: &[u8; 256], shift: u8) -> Self::Scaling {
        Av1GrainScaling::new(lut, shift)
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
        self.av1_add_film_grain_in(params, ChromaFormat::Yuv420, input, size, output);
    }

    /// [`Self::av1_add_film_grain`] of a frame in chroma `format`
    /// (§7.18.3: chroma templates, blocks, overlaps and luma averages
    /// follow `subsampling_x` and `subsampling_y`).
    ///
    /// # Panics
    ///
    /// As [`Self::av1_add_film_grain`].
    pub fn av1_add_film_grain_in(
        &self,
        params: &Av1FilmGrainParams,
        format: ChromaFormat,
        input: [&PlaneView<'_>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_>; 3],
    ) {
        let grain = (params, GrainBlocks::Av1);
        let templates = templates(params, format);
        add_film_grain(self, grain, format, &templates, (input, size), output);
    }

    /// AV2 8-bit 4:2:0 film grain (dav2d `dav2d_apply_grain`): AV1's
    /// templates, scaling and blending with AV2's block layout, 16x16 or
    /// 32x32 (`large_blocks`, FGM `block_size`).
    ///
    /// # Panics
    ///
    /// As [`Self::av1_add_film_grain`].
    pub fn av2_add_film_grain(
        &self,
        params: &Av1FilmGrainParams,
        large_blocks: bool,
        input: [&PlaneView<'_>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_>; 3],
    ) {
        let grain = (
            params,
            GrainBlocks::Av2 {
                large: large_blocks,
            },
        );
        let format = ChromaFormat::Yuv420;
        let templates = templates(params, format);
        add_film_grain(self, grain, format, &templates, (input, size), output);
    }

    /// [`Self::av2_add_film_grain`] of `u16` samples of `depth` (the general
    /// AV2 decoder's 10- and 12-bit pictures), with this ISA's high-depth
    /// row kernels (dav2d's 16bpc `fg_apply_tmpl.c`).
    ///
    /// # Panics
    ///
    /// As [`Self::av1_add_film_grain`].
    pub fn av2_add_film_grain_u16(
        &self,
        (params, large_blocks): (&Av1FilmGrainParams, bool),
        input: [&PlaneView<'_, u16>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_, u16>; 3],
        depth: BitDepth,
    ) {
        let format = ChromaFormat::Yuv420;
        let templates = templates_at::<i16>(params, depth, format);
        add_film_grain(
            &HighRows(self, depth),
            (
                params,
                GrainBlocks::Av2 {
                    large: large_blocks,
                },
            ),
            format,
            &templates,
            (input, size),
            output,
        );
    }

    /// [`Self::av1_add_film_grain_in`] of samples of any depth, with the
    /// scalar row kernels (AV1 §7.18.3 at `BitDepth`: grain range, shifts,
    /// clips and the interpolated scaling function).
    ///
    /// # Panics
    ///
    /// As [`Self::av1_add_film_grain`].
    pub fn av1_add_film_grain_at<T: Sample>(
        &self,
        (params, format): (&Av1FilmGrainParams, ChromaFormat),
        input: [&PlaneView<'_, T>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_, T>; 3],
        depth: BitDepth,
    ) {
        let templates = templates_at::<i16>(params, depth, format);
        let grain = (params, GrainBlocks::Av1);
        add_film_grain(
            &DeepRows(depth),
            grain,
            format,
            &templates,
            (input, size),
            output,
        );
    }

    /// High-bit-depth (`u16`) film grain synthesis of a frame in chroma
    /// `format` with this ISA's row kernels.
    pub(crate) fn av1_add_film_grain_u16(
        &self,
        (params, format): (&Av1FilmGrainParams, ChromaFormat),
        input: [&PlaneView<'_, u16>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_, u16>; 3],
        depth: BitDepth,
    ) {
        let templates = templates_at::<i16>(params, depth, format);
        add_film_grain(
            &HighRows(self, depth),
            (params, GrainBlocks::Av1),
            format,
            &templates,
            (input, size),
            output,
        );
    }
}

/// Film grain of every plane of a frame in chroma `format` with `rows`,
/// laid out as `blocks`.
fn add_film_grain<T: Sample, G: Grain, R: GrainRows<T, G>>(
    rows: &R,
    (params, blocks): (&Av1FilmGrainParams, GrainBlocks),
    format: ChromaFormat,
    templates: &[Template<G>; 3],
    (input, size): ([&PlaneView<'_, T>; 3], (usize, usize)),
    output: [&mut PlaneViewMut<'_, T>; 3],
) {
    let shift = params.grain_scaling_minus_8 + 8;
    let luma = (params.y_points.count != 0 || params.chroma_scaling_from_luma)
        .then(|| rows.prepare_scaling(&scaling_lut(&params.y_points), shift));
    let chroma_points = [&params.cb_points, &params.cr_points];
    for (plane, destination) in output.into_iter().enumerate() {
        let chroma;
        let table = if plane == 0 {
            luma.as_ref().filter(|_| params.y_points.count != 0)
        } else if params.chroma_scaling_from_luma {
            luma.as_ref()
        } else {
            let points = chroma_points[plane - 1];
            chroma = (points.count != 0).then(|| rows.prepare_scaling(&scaling_lut(points), shift));
            chroma.as_ref()
        };
        synthesize_plane(
            rows,
            (params, blocks),
            &templates[plane],
            (plane, format.plane_subsampling(PlaneIndex::ALL[plane])),
            (input, destination),
            table,
            size,
        );
    }
}

/// Film grain of plane `plane`, subsampled by `(sub_x, sub_y)`.
fn synthesize_plane<T: Sample, G: Grain, R: GrainRows<T, G>>(
    rows: &R,
    (params, blocks): (&Av1FilmGrainParams, GrainBlocks),
    template: &Template<G>,
    (plane, (sub_x, sub_y)): (usize, (usize, usize)),
    (input, output): ([&PlaneView<'_, T>; 3], &mut PlaneViewMut<'_, T>),
    scaling: Option<&R::Scaling>,
    (width, height): (usize, usize),
) {
    let (plane_w, plane_h) = ((width + sub_x) >> sub_x, (height + sub_y) >> sub_y);
    assert!(input[plane].width() >= plane_w && input[plane].height() >= plane_h);
    assert!(output.width() >= plane_w && output.height() >= plane_h);
    let Some(scaling) = scaling else {
        for y in 0..plane_h {
            output.row_mut(y)[..plane_w].copy_from_slice(&input[plane].row(y)[..plane_w]);
        }
        return;
    };
    let depth = rows.depth();
    let bits = depth.above_eight();
    let bounds = if params.clip_to_restricted_range {
        [
            RESTRICTED_MIN << bits,
            if plane == 0 || params.mc_identity {
                RESTRICTED_MAX_LUMA
            } else {
                RESTRICTED_MAX_CHROMA
            } << bits,
        ]
    } else {
        [0, depth.max()]
    };
    if plane == 0
        && rows.try_luma_plane(
            (params, blocks),
            template,
            scaling,
            bounds,
            (input[0], output),
            (plane_w, plane_h),
        )
    {
        return;
    }
    let (block_w, block_h) = (blocks.luma_side() >> sub_x, blocks.luma_side() >> sub_y);
    let (reach_x, reach_y) = (OVERLAP >> sub_x, OVERLAP >> sub_y);
    let stride = plane_w.div_ceil(block_w) * block_w + reach_x;
    let mut current = vec![G::default(); (block_h + reach_y) * stride];
    let mut previous = vec![G::default(); current.len()];
    // The row kernels average horizontal pairs of luma samples (4:2:0 and
    // 4:2:2); full-resolution chroma reads each luma sample twice, whose
    // pair average is the sample itself (§7.18.3.5 `averageLuma`).
    let mut doubled_luma = Vec::new();
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
    for (stripe_index, first_row) in (0..plane_h).step_by(block_h).enumerate() {
        fill_stripe(
            (params, blocks),
            template,
            ((sub_x, sub_y), depth),
            stripe_index,
            stride,
            &mut current,
        );
        for y in 0..block_h.min(plane_h - first_row) {
            let noise = &mut current[y * stride..y * stride + plane_w];
            if params.overlap && stripe_index != 0 && y < reach_y {
                let old = &previous[(block_h + y) * stride..(block_h + y) * stride + plane_w];
                rows.blend_row(old, noise, weights(sub_y, y));
            }
            let source = &input[plane].row(first_row + y)[..plane_w];
            let destination = &mut output.row_mut(first_row + y)[..plane_w];
            if plane == 0 {
                rows.luma_row(source, noise, scaling, bounds, destination);
            } else {
                let luma_row = &input[0].row((first_row + y) << sub_y)[..width];
                let luma = if sub_x == 1 {
                    luma_row
                } else {
                    doubled_luma.clear();
                    doubled_luma.extend(luma_row.iter().flat_map(|&s| [s, s]));
                    &doubled_luma[..]
                };
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

/// Lay stripe `index`'s grain blocks (each `block + reach` per axis, from
/// a random offset into the template) side by side in `stripe`, blending
/// each block's first `reach` columns over the previous block's overhang.
/// `sub` is the plane's subsampling (log2 per axis).
fn fill_stripe<G: Grain>(
    grain: (&Av1FilmGrainParams, GrainBlocks),
    template: &Template<G>,
    (sub, depth): ((usize, usize), BitDepth),
    index: usize,
    stride: usize,
    stripe: &mut [G],
) {
    // The block sides are constants per plane type and block size, so each
    // row's copy is a fixed-size move rather than a `memcpy` call.
    const FULL: usize = BLOCK + OVERLAP;
    const HALF: usize = (BLOCK >> 1) + (OVERLAP >> 1);
    const SMALL_FULL: usize = AV2_SMALL_BLOCK + OVERLAP;
    const SMALL_HALF: usize = (AV2_SMALL_BLOCK >> 1) + (OVERLAP >> 1);
    let fill = match (grain.1.luma_side(), sub) {
        (BLOCK, (0, 0)) => fill_stripe_blocks::<G, FULL, FULL>,
        (BLOCK, (1, 1)) => fill_stripe_blocks::<G, HALF, HALF>,
        (BLOCK, (1, 0)) => fill_stripe_blocks::<G, HALF, FULL>,
        (BLOCK, _) => fill_stripe_blocks::<G, FULL, HALF>,
        (_, (0, 0)) => fill_stripe_blocks::<G, SMALL_FULL, SMALL_FULL>,
        (_, (1, 1)) => fill_stripe_blocks::<G, SMALL_HALF, SMALL_HALF>,
        _ => unreachable!("AV2 film grain is 4:2:0"),
    };
    fill(grain, template, (sub, depth), index, stride, stripe);
}

/// [`fill_stripe`] for blocks of `COLS` x `ROWS` (`block + reach` per
/// axis).
fn fill_stripe_blocks<G: Grain, const COLS: usize, const ROWS: usize>(
    (params, blocks): (&Av1FilmGrainParams, GrainBlocks),
    template: &Template<G>,
    ((sub_x, sub_y), depth): ((usize, usize), BitDepth),
    index: usize,
    stride: usize,
    stripe: &mut [G],
) {
    let range = grain_range(depth);
    let block = blocks.luma_side() >> sub_x;
    let reach = OVERLAP >> sub_x;
    debug_assert_eq!(block + reach, COLS);
    debug_assert_eq!((blocks.luma_side() >> sub_y) + (OVERLAP >> sub_y), ROWS);
    let mut random = Random(blocks.stripe_seed(params.grain_seed, index));
    // Offsets count pairs of luma samples: whole samples along a
    // subsampled axis (§7.18.3.5 `planeOffsetX`, `planeOffsetY`).
    let base = |offset: usize, sub: usize| AR_PADDING + ((AR_PADDING + offset) << (1 - sub));
    for x in (0..stride - reach).step_by(block) {
        let (offset_x, offset_y) = blocks.offsets(&mut random);
        let (base_x, base_y) = (base(offset_x, sub_x), base(offset_y, sub_y));
        let overlap = if params.overlap && x != 0 { reach } else { 0 };
        for y in 0..ROWS {
            let start = (base_y + y) * template.columns + base_x;
            let source: &[G; COLS] = template.values[start..start + COLS]
                .try_into()
                .expect("a template row holds a block");
            let destination: &mut [G; COLS] = (&mut stripe[y * stride + x..][..COLS])
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
                    (j, sub_x),
                    range,
                ));
            }
        }
    }
}
