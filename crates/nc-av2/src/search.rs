//! Small candidate-first motion search reusing shared SIMD distortion kernels.
//! Shared SIMD prediction (the native coder's filter) supplies exact AV2 subpixel samples.
use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::{BlockSize, Dsp};

nc_encoder::param_group! {
    pub(crate) struct SearchParams ("av2") {
        search_lambda_scale: 0.135, [0.001,4.0], "normalized-step squared multiplier for motion candidate screening",
        early_sse_per_lambda: 0.025, [0.0,8.0], "per-pixel SSE threshold relative to motion lambda",
        diamond_passes: 2.0, [0.0,8.0], "integer diamond refinements after zero and spatial candidates",
        cdf_adaptation: 1.0, [0.0,1.0], "enable tile-local trained CDF updates",
        tx8: 1.0, [0.0,1.0], "use lossy eight-by-eight luma transforms",
        pre_tx_skip_energy: 30.0, [0.0,100.0], "lossy prediction SSE threshold as area times qindex times percent",
        residual_skip_rd: 1.0, [0.0,1.0], "price a lossy no-residual candidate using cached prediction",
        rd_lambda_scale: 0.135, [0.01,4.0], "pixel-step squared multiplier for finalist RD bits",
        chroma_weight: 1.0, [0.0,4.0], "chroma SSE weight for motion finalist reconstruction",
        subpel_steps: 1.0, [0.0,2.0], "half/quarter-pel refinement levels after integer search",
    }
}

pub(crate) struct Search {
    dsp: Dsp,
    lambda: i64,
    base_lambda: i64,
    early_sse: u32,
    passes: usize,
    subpel_steps: usize,
}
impl Search {
    pub(crate) fn new(qindex: u8) -> Self {
        let step = i64::from(crate::transform::quantizer_step_q3(qindex)) / 8;
        let lambda = (step * step / 64).max(1);
        let mut result = Self {
            dsp: Dsp::new(),
            lambda,
            base_lambda: lambda,
            early_sse: 0,
            passes: 0,
            subpel_steps: 0,
        };
        result.set_params(SearchParams::default());
        result
    }
    pub(crate) fn set_params(&mut self, params: SearchParams) {
        self.lambda =
            (self.base_lambda as f64 * f64::from(params.search_lambda_scale)).max(1.0) as i64;
        self.early_sse =
            (self.lambda as f64 * 64.0 * f64::from(params.early_sse_per_lambda)) as u32;
        self.passes = params.diamond_passes.round() as usize;
        self.subpel_steps = params.subpel_steps.round() as usize;
    }
    fn distortion(
        &self,
        source: &PlaneView<'_>,
        reference: &nc_common::Plane,
        x: usize,
        y: usize,
        vector: (i32, i32),
    ) -> Option<u32> {
        let size = BlockSize::new(8, 8).expect("supported block");
        let rx = x as i32 + vector.1.div_euclid(8);
        let ry = y as i32 + vector.0.div_euclid(8);
        let sse = if vector.0 % 8 == 0
            && vector.1 % 8 == 0
            && rx >= 0
            && ry >= 0
            && rx as usize + 8 <= reference.width()
            && ry as usize + 8 <= reference.height()
        {
            let view = reference
                .region(rx as isize, ry as isize, 8, 8)
                .expect("checked motion bounds");
            self.dsp.variance(size, source, &view).sse
        } else {
            let mut pixels = [0; 64];
            crate::prediction::predict(
                &self.dsp,
                crate::prediction::NATIVE_FILTER,
                reference,
                x,
                y,
                vector,
                0,
                &mut pixels,
            )
            .ok()?;
            let view = PlaneView::new(&pixels, 8, 8, 8).expect("fixed prediction");
            self.dsp.variance(size, source, &view).sse
        };
        Some(sse)
    }
    pub(crate) fn find(
        &self,
        source: &Picture,
        reference: &Picture,
        x: usize,
        y: usize,
        predictors: &[(i32, i32)],
    ) -> [(i32, i32); 2] {
        let source = source
            .plane(PlaneIndex::Y)
            .region(x as isize, y as isize, 8, 8)
            .expect("complete8x8block");
        let reference = reference.plane(PlaneIndex::Y);
        let cost = |vector: (i32, i32)| {
            if vector.0.abs() > 256 || vector.1.abs() > 256 {
                return None;
            }
            let sse = self.distortion(&source, reference, x, y, vector)?;
            let rate = if vector == (0, 0) {
                1
            } else {
                6 + i64::from(
                    (vector.0 - predictors[0].0)
                        .unsigned_abs()
                        .checked_ilog2()
                        .unwrap_or(0),
                ) + i64::from(
                    (vector.1 - predictors[0].1)
                        .unsigned_abs()
                        .checked_ilog2()
                        .unwrap_or(0),
                )
            };
            Some((i64::from(sse) + self.lambda * rate, sse))
        };
        let mut best = (0, 0);
        let (mut best_cost, mut best_sse) = cost(best).expect("zeroMVwithinpicture");
        if best_sse <= self.early_sse {
            return [best; 2];
        }
        for candidate in predictors
            .iter()
            .map(|&(row, col)| (row.div_euclid(2) * 2, col.div_euclid(2) * 2))
        {
            if candidate == best {
                continue;
            }
            if let Some((value, sse)) = cost(candidate)
                && value < best_cost
            {
                best = candidate;
                best_cost = value;
                best_sse = sse;
            }
        }
        if best_sse <= self.early_sse {
            return [best; 2];
        }
        for _ in 0..self.passes {
            let center = best;
            for (row, col) in [(-8, 0), (0, -8), (0, 8), (8, 0)] {
                let candidate = (center.0 + row, center.1 + col);
                if let Some((value, sse)) = cost(candidate)
                    && value < best_cost
                {
                    best = candidate;
                    best_cost = value;
                    best_sse = sse;
                }
            }
            if best == center || best_sse <= self.early_sse {
                break;
            }
        }
        let integer = best;
        if best_sse > self.early_sse {
            for step in [4, 2].into_iter().take(self.subpel_steps) {
                let center = best;
                for (row, col) in [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ] {
                    let candidate = (center.0 + row * step, center.1 + col * step);
                    if let Some((value, sse)) = cost(candidate)
                        && value < best_cost
                    {
                        best = candidate;
                        best_cost = value;
                        best_sse = sse;
                    }
                }
                if best_sse <= self.early_sse {
                    break;
                }
            }
        }
        [integer, best]
    }
}
