//! Forward AV2 prediction: one previous picture, GLOBALMV or quarter-pel NEWMV.
//!
//! Reuses the intra path's transform reconstruction and coefficient contexts.
//! Every block is predicted from the previous reconstructed picture; a block
//! whose quantized residuals vanish carries skip syntax and no coefficients.
use crate::prediction::NATIVE_FILTER;
use crate::transform::QuantDeltas;
use crate::{
    coeff, entropy,
    intra::{self, Tile},
    motion, mv_refs, partition,
    search::{Search, SearchParams},
    transform, tx_partition,
};
use coeff::TxSize;
use nc_common::decode::{DecodeError, Result};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;

const IS_INTER: [u16; 2] = [31246, 0];
/// Every block of this picture coder is 8×8.
const BLOCK_LOG2: u32 = 3;
const PERCENT_DENOMINATOR: f64 = 100.0;
const SKIP: [[u16; 2]; 3] = [[6903, 0], [18452, 0], [28170, 0]];

// AV2 keeps two spatial samples even when both point to the same neighbour:
// bottom-left, above-right, left, above (first two available entries).
fn skip_context(above: &[bool], left: &[bool], x: usize, y: usize) -> usize {
    match (x > 0, y > 0) {
        (false, false) => 0,
        (true, false) => 2 * usize::from(left[y / 8]),
        (false, true) => 2 * usize::from(above[x / 8]),
        (true, true) => usize::from(left[y / 8]) + usize::from(above[x / 8]),
    }
}

fn check_reference(source: &Picture, reference: &Picture) -> Result<()> {
    if source.width() != reference.width() || source.height() != reference.height() {
        return Err(DecodeError::Unsupported("AV2 reference dimension mismatch"));
    }
    Ok(())
}
fn prediction<const N: usize>(
    dsp: &Dsp,
    reference: &Picture,
    plane: usize,
    x: usize,
    y: usize,
    vector: (i32, i32),
) -> Result<[i32; N]> {
    let mut pixels = [0; N];
    crate::prediction::predict(
        dsp,
        NATIVE_FILTER,
        reference.plane(PlaneIndex::ALL[plane]),
        x,
        y,
        vector,
        u8::from(plane != 0),
        &mut pixels,
    )?;
    Ok(pixels.map(i32::from))
}

/// Prediction of an 8×8 block's 4×4 transform blocks in [`intra::blocks`]
/// order. Luma is predicted as one 8×8 block, as AVM predicts whole
/// prediction blocks (`av2_init_inter_params`): the filter taps depend on the
/// block size, so 4×4 luma pieces would select the 4-tap variant.
fn block_prediction(
    dsp: &Dsp,
    reference: &Picture,
    (x, y): (usize, usize),
    vector: (i32, i32),
) -> Result<[[i32; 16]; 6]> {
    let luma = prediction::<64>(dsp, reference, 0, x, y, vector)?;
    let quadrant = |left: usize, top: usize| -> [i32; 16] {
        std::array::from_fn(|i| luma[(top + i / 4) * 8 + left + i % 4])
    };
    Ok([
        quadrant(0, 0),
        quadrant(4, 0),
        quadrant(0, 4),
        quadrant(4, 4),
        prediction::<16>(dsp, reference, 1, x / 2, y / 2, vector)?,
        prediction::<16>(dsp, reference, 2, x / 2, y / 2, vector)?,
    ])
}

/// Write a single-reference zero-motion tile, returning its reconstruction.
/// The enclosing inter header must select one reference and independent CDFs.
/// # Errors
/// Rejects unsupported dimensions, reference geometry or arithmetic syntax.
pub fn encode_tile(
    source: &Picture,
    reference: &Picture,
    qindex: u8,
) -> Result<(Vec<u8>, Picture)> {
    encode_tile_impl(source, reference, qindex, None, false, false)
}

struct PredictionCache {
    pixels: [[u8; 16]; 6],
    residuals: [[i32; 16]; 6],
    distortion: [u64; 2],
}
impl PredictionCache {
    fn new(
        source: &Picture,
        reference: &Picture,
        position: (usize, usize),
        vector: (i32, i32),
        dsp: &Dsp,
    ) -> Result<Self> {
        let mut result = Self {
            pixels: [[0; 16]; 6],
            residuals: [[0; 16]; 6],
            distortion: [0; 2],
        };
        let predictions = block_prediction(dsp, reference, position, vector)?;
        for (index, (plane, x, y)) in intra::blocks(position.0, position.1)
            .into_iter()
            .enumerate()
        {
            let predicted = predictions[index];
            let view = source.plane(PlaneIndex::ALL[plane]).view();
            result.pixels[index] = predicted.map(|v| v as u8);
            result.residuals[index] =
                std::array::from_fn(|i| i32::from(view.row(y + i / 4)[x + i % 4]) - predicted[i]);
            result.distortion[usize::from(plane != 0)] += result.residuals[index]
                .iter()
                .map(|&v| (i64::from(v) * i64::from(v)) as u64)
                .sum::<u64>();
        }
        Ok(result)
    }
    fn luma(&self) -> ([i32; 64], [i32; 64]) {
        let index = |i: usize| ((i / 8 / 4) * 2 + (i % 8 / 4), (i / 8 % 4) * 4 + i % 4);
        (
            std::array::from_fn(|i| {
                let (b, p) = index(i);
                i32::from(self.pixels[b][p])
            }),
            std::array::from_fn(|i| {
                let (b, p) = index(i);
                self.residuals[b][p]
            }),
        )
    }
}
#[derive(Clone, Copy)]
struct Preparation {
    distortion: bool,
    skip: bool,
    tx8: bool,
    early_energy: f32,
}

struct Prepared {
    vector: (i32, i32),
    coefficients: [[i32; 16]; 6],
    coefficients8: [i32; 64],
    tx8: bool,
    pixels: [[u8; 16]; 6],
    distortion: [u64; 2],
    prediction_pixels: [[u8; 16]; 6],
    prediction_distortion: [u64; 2],
    skip: bool,
    early_skip: bool,
}
impl Prepared {
    fn new(
        source: &Picture,
        reference: &Picture,
        qindex: u8,
        position: (usize, usize),
        vector: (i32, i32),
        dsp: &Dsp,
        options: Preparation,
    ) -> Result<Self> {
        let (measure_distortion, measure_skip, tx8) =
            (options.distortion, options.skip, options.tx8);
        let mut result = Self {
            vector,
            coefficients: [[0; 16]; 6],
            coefficients8: [0; 64],
            tx8,
            pixels: [[0; 16]; 6],
            distortion: [0; 2],
            prediction_pixels: [[0; 16]; 6],
            prediction_distortion: [0; 2],
            skip: true,
            early_skip: false,
        };
        let cached = if measure_skip && options.early_energy > 0.0 {
            Some(PredictionCache::new(
                source, reference, position, vector, dsp,
            )?)
        } else {
            None
        };
        if let Some(cache) = &cached {
            result.prediction_pixels = cache.pixels;
            result.prediction_distortion = cache.distortion;
            // The cache contains 64 Y + 16 U + 16 V samples. The tunable
            // energy is a percentage of qindex, following AV1's early-skip screen.
            // Gated three-clip/heldout delta is below 0.00005 dB YUV; the
            // static q144 motion recipe costs 55.9M P instructions with this
            // screen. Exact settings/limits: docs/analysis/av2-motion-rd.md.
            let samples = (cache.pixels.len() * cache.pixels[0].len()) as f64;
            if (cache.distortion[0] + cache.distortion[1]) as f64 * PERCENT_DENOMINATOR
                <= samples * f64::from(qindex) * f64::from(options.early_energy)
            {
                result.pixels = cache.pixels;
                result.distortion = cache.distortion;
                result.early_skip = true;
                return Ok(result);
            }
        }
        if tx8 {
            result.prepare_luma8(
                source,
                reference,
                qindex,
                position,
                dsp,
                (
                    measure_distortion,
                    measure_skip && cached.is_none(),
                    cached.as_ref(),
                ),
            )?;
        }
        let predictions = match &cached {
            Some(cache) => cache.pixels.map(|pixels| pixels.map(i32::from)),
            None => block_prediction(dsp, reference, position, vector)?,
        };
        for (index, (plane, x, y)) in intra::blocks(position.0, position.1)
            .into_iter()
            .enumerate()
            .skip(if tx8 { 4 } else { 0 })
        {
            let predicted = predictions[index];
            let view = source.plane(PlaneIndex::ALL[plane]).view();
            let residual = cached.as_ref().map_or_else(
                || {
                    std::array::from_fn(|i| {
                        i32::from(view.row(y + i / 4)[x + i % 4]) - predicted[i]
                    })
                },
                |cache| cache.residuals[index],
            );
            let (coefficients, inverse) = code_residual(dsp, TxSize::Tx4, &residual, qindex)?;
            let zero = coefficients.iter().all(|&v| v == 0);
            result.skip &= zero;
            for i in 0..16 {
                let value = (predicted[i] + inverse[i]).clamp(0, 255) as u8;
                result.pixels[index][i] = value;
            }
            if measure_distortion {
                for i in 0..16 {
                    let error = i64::from(view.row(y + i / 4)[x + i % 4])
                        - i64::from(result.pixels[index][i]);
                    result.distortion[usize::from(plane != 0)] += (error * error) as u64;
                }
            }
            if measure_skip && cached.is_none() {
                result.prediction_pixels[index] = predicted.map(|value| value as u8);
                result.prediction_distortion[usize::from(plane != 0)] += residual
                    .iter()
                    .map(|&value| (i64::from(value) * i64::from(value)) as u64)
                    .sum::<u64>();
            }
            result.coefficients[index] = coefficients;
        }
        Ok(result)
    }
    fn prepare_luma8(
        &mut self,
        source: &Picture,
        reference: &Picture,
        qindex: u8,
        position: (usize, usize),
        dsp: &Dsp,
        measure: (bool, bool, Option<&PredictionCache>),
    ) -> Result<()> {
        let (x, y) = position;
        let view = source.plane(PlaneIndex::Y).view();
        let (predicted, residual) = if let Some(cache) = measure.2 {
            cache.luma()
        } else {
            let predicted = prediction::<64>(dsp, reference, 0, x, y, self.vector)?;
            let residual =
                std::array::from_fn(|i| i32::from(view.row(y + i / 8)[x + i % 8]) - predicted[i]);
            (predicted, residual)
        };
        let (levels, inverse) = code_residual(dsp, TxSize::Tx8, &residual, qindex)?;
        self.coefficients8 = levels;
        self.skip = levels.iter().all(|&v| v == 0);
        for i in 0..64 {
            let (row, col) = (i / 8, i % 8);
            let block = (row / 4) * 2 + col / 4;
            let offset = (row % 4) * 4 + col % 4;
            let value = (predicted[i] + inverse[i]).clamp(0, 255) as u8;
            self.pixels[block][offset] = value;
            if measure.0 {
                let error = i64::from(view.row(y + row)[x + col]) - i64::from(value);
                self.distortion[0] += (error * error) as u64;
            }
            if measure.1 {
                self.prediction_pixels[block][offset] = predicted[i] as u8;
                self.prediction_distortion[0] +=
                    (i64::from(residual[i]) * i64::from(residual[i])) as u64;
            }
        }
        Ok(())
    }
    fn without_residual(&self) -> Self {
        Self {
            vector: self.vector,
            coefficients: [[0; 16]; 6],
            coefficients8: [0; 64],
            tx8: self.tx8,
            pixels: self.prediction_pixels,
            distortion: self.prediction_distortion,
            prediction_pixels: self.prediction_pixels,
            prediction_distortion: self.prediction_distortion,
            skip: true,
            early_skip: self.early_skip,
        }
    }
    fn cost(&self, bits: f64, qindex: u8, params: SearchParams) -> f64 {
        if qindex == 0 {
            return bits;
        }
        let pixel_step = f64::from(transform::quantizer_step_q3(qindex)) / 64.0;
        self.distortion[0] as f64
            + f64::from(params.chroma_weight) * self.distortion[1] as f64
            + f64::from(params.rd_lambda_scale) * pixel_step * pixel_step * bits
    }
    fn store_pixels(&self, tile: &mut Tile, x: usize, y: usize) {
        for (index, (plane, bx, by)) in intra::blocks(x, y).into_iter().enumerate() {
            let mut view = tile
                .reconstruction
                .plane_mut(PlaneIndex::ALL[plane])
                .view_mut();
            for row in 0..4 {
                view.row_mut(by + row)[bx..bx + 4]
                    .copy_from_slice(&self.pixels[index][row * 4..row * 4 + 4]);
            }
        }
    }
}

fn write_block(
    writer: &mut entropy::Writer,
    tile: &mut Tile,
    prepared: &Prepared,
    candidates: mv_refs::Candidates,
    position: (usize, usize),
    qindex: u8,
    skip_context: usize,
) -> Result<()> {
    let is_new = prepared.vector != (0, 0);
    writer.model(1, &IS_INTER, |m| &mut m.is_inter[0])?;
    writer.model(usize::from(prepared.skip), &SKIP[skip_context], |m| {
        &mut m.block_skip[skip_context]
    })?;
    mv_refs::write_mode(writer, candidates.mode_context, is_new)?;
    if is_new {
        mv_refs::write_drl(writer, candidates, 0)?;
        motion::write_mv(
            writer,
            prepared.vector.0 - candidates.vectors[0].0,
            prepared.vector.1 - candidates.vectors[0].1,
        )?;
    }
    if !prepared.skip {
        if qindex == 0 {
            writer.model(0, &[16384, 0], |m| &mut m.lossless_size[1][1])?;
        } else {
            tx_partition::write(writer, BLOCK_LOG2, true, !prepared.tx8)?;
        }
    }
    if prepared.tx8 {
        if !prepared.skip {
            let mut context = tile.context8(position.0, position.1)?;
            context.is_inter = true;
            coeff::write(
                writer,
                &prepared.coefficients8,
                TxSize::Tx8,
                context,
                qindex,
            )?;
        }
        tile.update_context8(position.0, position.1, &prepared.coefficients8);
    }
    let mut u_nonzero = false;
    for (index, (plane, x, y)) in intra::blocks(position.0, position.1)
        .into_iter()
        .enumerate()
        .skip(if prepared.tx8 { 4 } else { 0 })
    {
        let coefficients = &prepared.coefficients[index];
        if !prepared.skip {
            let mut context = tile.context(plane, x, y, u_nonzero)?;
            context.is_inter = true;
            coeff::write(writer, coefficients, TxSize::Tx4, context, qindex)?;
        }
        if plane == 1 {
            u_nonzero = coefficients.iter().any(|&c| c != 0);
        }
        tile.update_context(plane, x, y, coefficients);
    }
    Ok(())
}

fn encode_tile_impl(
    source: &Picture,
    reference: &Picture,
    qindex: u8,
    params: Option<SearchParams>,
    adaptive_cdf: bool,
    motion: bool,
) -> Result<(Vec<u8>, Picture)> {
    check_reference(source, reference)?;
    let (width, height) = (source.width(), source.height());
    let mut tile = Tile::new(width, height, qindex)?;
    let dsp = Dsp::new();
    let mut writer = entropy::Writer::new();
    if adaptive_cdf {
        writer.enable_adaptation(qindex);
    }
    let mut references = mv_refs::ReferenceGrid::new(width, height, mv_refs::ListConfig::BASIC)?;
    let search = params.filter(|_| motion).map(|params| {
        let mut search = Search::new(qindex);
        search.set_params(params);
        search
    });
    let mut above = vec![false; width / 8];
    let mut left = vec![false; height / 8];
    partition::write_8x8(&mut writer, width, height, |writer, x, y| {
        let candidates = references.candidates(x, y, 8)?;
        let skip_context = skip_context(&above, &left, x, y);
        let force_skip = qindex != 0 && params.is_some_and(|p| p.residual_skip_rd >= 0.5);
        let tx8 = qindex != 0 && params.is_some_and(|p| p.tx8 >= 0.5);
        let measure = Preparation {
            distortion: (motion || force_skip) && qindex != 0,
            skip: force_skip,
            tx8,
            early_energy: params.map_or(0.0, |p| p.pre_tx_skip_energy),
        };
        let mut best = Prepared::new(source, reference, qindex, (x, y), (0, 0), &dsp, measure)?;
        if let Some(params) = params {
            let finalists = if best.early_skip {
                [(0, 0); 2]
            } else {
                search.as_ref().map_or([(0, 0); 2], |search| {
                    search.find(source, reference, x, y, &candidates.vectors[..2])
                })
            };
            if finalists != [(0, 0); 2] || (force_skip && !best.skip) {
                let contexts = tile.save_contexts(x, y);
                // Candidates are priced from the current probabilities
                // without coding (`entropy::Writer::start_estimate`).
                let mut price = |candidate: &Prepared| -> Result<f64> {
                    writer.start_estimate();
                    let written = write_block(
                        writer,
                        &mut tile,
                        candidate,
                        candidates,
                        (x, y),
                        qindex,
                        skip_context,
                    );
                    let bits = writer.finish_estimate();
                    tile.restore_contexts(x, y, contexts);
                    written?;
                    Ok(candidate.cost(bits, qindex, params))
                };
                let mut select_residual = |candidate: Prepared| -> Result<(Prepared, f64)> {
                    let cost = price(&candidate)?;
                    if force_skip && !candidate.skip {
                        let skipped = candidate.without_residual();
                        // Entropy rate is nonnegative: avoid pricing a skip
                        // whose distortion alone already loses to coded RD.
                        if skipped.cost(0.0, qindex, params) < cost {
                            let skip_cost = price(&skipped)?;
                            if skip_cost < cost {
                                return Ok((skipped, skip_cost));
                            }
                        }
                    }
                    Ok((candidate, cost))
                };
                let (initial, mut best_cost) = select_residual(best)?;
                best = initial;
                for (index, vector) in finalists.into_iter().enumerate() {
                    if vector == (0, 0) || (index == 1 && vector == finalists[0]) {
                        continue;
                    }
                    let candidate =
                        Prepared::new(source, reference, qindex, (x, y), vector, &dsp, measure)?;
                    let (candidate, cost) = select_residual(candidate)?;
                    if cost < best_cost {
                        best_cost = cost;
                        best = candidate;
                    }
                }
            }
        }
        write_block(
            writer,
            &mut tile,
            &best,
            candidates,
            (x, y),
            qindex,
            skip_context,
        )?;
        best.store_pixels(&mut tile, x, y);
        references.commit(x, y, 8, best.vector, best.vector != (0, 0))?;
        above[x / 8] = best.skip;
        left[y / 8] = best.skip;
        Ok(())
    })?;
    Ok((writer.finish(), tile.reconstruction))
}

/// Read a single-reference GLOBALMV/NEWMV tile against its previous picture.
/// # Errors
/// Rejects unsupported modes, malformed coefficients and invalid dimensions.
pub fn decode_tile(data: &[u8], reference: &Picture, qindex: u8) -> Result<Picture> {
    decode_tile_with_adaptation(data, reference, qindex, false)
}
pub(crate) fn decode_tile_with_adaptation(
    data: &[u8],
    reference: &Picture,
    qindex: u8,
    adaptive_cdf: bool,
) -> Result<Picture> {
    let (width, height) = (reference.width(), reference.height());
    let mut tile = Tile::new(width, height, qindex)?;
    let dsp = Dsp::new();
    let mut reader = entropy::Reader::new(data)?;
    if adaptive_cdf {
        reader.enable_adaptation(qindex);
    }
    let mut references = mv_refs::ReferenceGrid::new(width, height, mv_refs::ListConfig::BASIC)?;
    let mut above = vec![false; width / 8];
    let mut left = vec![false; height / 8];
    partition::read_8x8(&mut reader, width, height, |reader, x, y| {
        let candidates = references.candidates(x, y, 8)?;
        let skip_context = skip_context(&above, &left, x, y);
        if reader.model(&IS_INTER, |m| &mut m.is_inter[0])? != 1 {
            return Err(DecodeError::Unsupported(
                "AV2 intra block inside forward tile",
            ));
        }
        let skip = reader.model(&SKIP[skip_context], |m| &mut m.block_skip[skip_context])? != 0;
        let is_new = mv_refs::read_mode(reader, candidates.mode_context)?;
        let vector = if is_new {
            let index = mv_refs::read_drl(reader, candidates)?;
            let delta = motion::read_mv(reader)?;
            (
                candidates.vectors[index].0 + delta.0,
                candidates.vectors[index].1 + delta.1,
            )
        } else {
            (0, 0)
        };
        let mut tx8 = false;
        if !skip {
            if qindex == 0 {
                if reader.model(&[16384, 0], |m| &mut m.lossless_size[1][1])? != 0 {
                    return Err(DecodeError::Unsupported(
                        "AV2 large lossless inter transform",
                    ));
                }
            } else {
                tx8 = !tx_partition::read(reader, BLOCK_LOG2, true)?;
            }
        }
        if tx8 {
            let mut context = tile.context8(x, y)?;
            context.is_inter = true;
            let mut coefficients = [0; 64];
            coeff::read(reader, &mut coefficients, TxSize::Tx8, context, qindex)?;
            tile.reconstruct8(
                x,
                y,
                &prediction::<64>(&dsp, reference, 0, x, y, vector)?,
                &coefficients,
            )?;
        }
        let mut u_nonzero = false;
        let predictions = block_prediction(&dsp, reference, (x, y), vector)?;
        for (index, (plane, bx, by)) in
            intra::blocks(x, y)
                .into_iter()
                .enumerate()
                .skip(if tx8 { 4 } else { 0 })
        {
            let mut context = tile.context(plane, bx, by, u_nonzero)?;
            context.is_inter = true;
            let mut coefficients = [0; 16];
            if !skip {
                coeff::read(reader, &mut coefficients, TxSize::Tx4, context, qindex)?;
            }
            if plane == 1 {
                u_nonzero = coefficients.iter().any(|&c| c != 0);
            }
            tile.reconstruct(plane, bx, by, &predictions[index], &coefficients)?;
        }
        references.commit(x, y, 8, vector, is_new)?;
        above[x / 8] = skip;
        left[y / 8] = skip;
        Ok(())
    })?;
    Ok(tile.reconstruction)
}

/// Encode one forward packet, refreshing every slot to keep one distinct reference.
/// # Errors
/// Rejects incompatible source/reference geometry or unsupported frame syntax.
pub fn encode_inter(
    source: &Picture,
    reference: &Picture,
    qindex: u8,
    order_hint: u16,
) -> Result<intra::EncodedFrame> {
    encode_inter_impl(source, reference, qindex, order_hint, None, false, false)
}

/// Encode a forward packet using bounded quarter-pel search and finalist rate–distortion selection.
/// # Errors
/// Rejects incompatible pictures or unsupported syntax constraints.
pub fn encode_inter_motion(
    source: &Picture,
    reference: &Picture,
    qindex: u8,
    order_hint: u16,
) -> Result<intra::EncodedFrame> {
    encode_inter_with_params(
        source,
        reference,
        qindex,
        order_hint,
        SearchParams {
            cdf_adaptation: 0.0,
            ..Default::default()
        },
    )
}
pub(crate) fn encode_inter_with_params(
    source: &Picture,
    reference: &Picture,
    qindex: u8,
    order_hint: u16,
    params: SearchParams,
) -> Result<intra::EncodedFrame> {
    encode_inter_with_options(source, reference, qindex, order_hint, true, params)
}
pub(crate) fn encode_inter_with_options(
    source: &Picture,
    reference: &Picture,
    qindex: u8,
    order_hint: u16,
    motion: bool,
    params: SearchParams,
) -> Result<intra::EncodedFrame> {
    encode_inter_impl(
        source,
        reference,
        qindex,
        order_hint,
        (motion || params.residual_skip_rd >= 0.5 || params.tx8 >= 0.5).then_some(params),
        params.cdf_adaptation >= 0.5,
        motion,
    )
}
fn encode_inter_impl(
    source: &Picture,
    reference: &Picture,
    qindex: u8,
    order_hint: u16,
    params: Option<SearchParams>,
    adaptive_cdf: bool,
    motion: bool,
) -> Result<intra::EncodedFrame> {
    let (tile, reconstruction) =
        encode_tile_impl(source, reference, qindex, params, adaptive_cdf, motion)?;

    let sequence = crate::sequence::SequenceHeader {
        id: 0,
        max_width: source.width() as u32,
        max_height: source.height() as u32,
        level: 31,
        single_picture: false,
        order_hint_bits: 8,
        crop: crate::sequence::CropWindow::NONE,
        layering: crate::sequence::Layering::FLAT,
        chroma_delta_q: false,
        dc_delta_q: false,
        parity_hiding: false,
        tcq: crate::sequence::TcqFrames::Off,
        ccso: false,
        reference_lists: crate::mv_refs::ListConfig::BASIC,
    };
    let header = crate::frame_header::InterHeader {
        order_hint,
        quantizer: qindex,
        quant_deltas: QuantDeltas::NONE,
        transform_select: qindex != 0,
        tx_set: crate::tx_type::TxSetMode::IntraDct,
        levels: crate::coeff::LevelCoding::Plain,
        refresh_flags: 255,
        reference: None,
        adaptive_cdf,
        cross_frame_context: false,
        deblock: crate::deblock::DeblockParams::OFF,
        cdef: crate::cdef::CdefParams::OFF,
        ccso: crate::ccso::CcsoParams::OFF,
        filter: crate::interp_filter::FrameFilter::Fixed(NATIVE_FILTER),
        skip_mode: false,
        delta_q: None,
    };
    let mut payload = crate::frame_header::write_inter(&sequence, header)?;
    payload.extend(tile);
    let mut packet = Vec::new();
    crate::obu::write(
        crate::obu::Header {
            obu_type: 2,
            temporal_id: 0,
            embedded_id: 0,
            extended_id: crate::obu::GLOBAL_XLAYER_ID,
        },
        &[],
        &mut packet,
    )?;
    crate::obu::write(
        crate::obu::Header {
            obu_type: 7,
            temporal_id: 0,
            embedded_id: 0,
            extended_id: 0,
        },
        &payload,
        &mut packet,
    )?;
    Ok(intra::EncodedFrame {
        packet,
        reconstruction,
    })
}

/// Levels and reconstructed residual of an `N`-sample residual block coded
/// with `size`; zero residuals and all-zero levels skip the arithmetic.
fn code_residual<const N: usize>(
    dsp: &Dsp,
    size: TxSize,
    residual: &[i32; N],
    qindex: u8,
) -> Result<([i32; N], [i32; N])> {
    let mut levels = [0; N];
    let mut inverse = [0; N];
    if residual.iter().any(|&v| v != 0) {
        transform::quantize_residual(
            dsp,
            size,
            residual,
            (qindex, transform::NEAREST_ROUNDING_Q7),
            &mut levels,
        )?;
        if levels.iter().any(|&v| v != 0) {
            transform::reconstruct(
                dsp,
                (size, transform::TxType::DctDct),
                &levels,
                qindex,
                &mut inverse,
            )?;
        }
    }
    Ok((levels, inverse))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lossy_skip_rd_drops_unprofitable_noise_but_never_lossless_residuals() {
        let mut reference = Picture::new(8, 8, 0).unwrap();
        for p in PlaneIndex::ALL {
            let mut view = reference.plane_mut(p).view_mut();
            for y in 0..view.height() {
                view.row_mut(y).fill(128);
            }
        }
        let mut source = reference.clone();
        for p in PlaneIndex::ALL {
            let mut view = source.plane_mut(p).view_mut();
            for y in 0..view.height() {
                for (x, pixel) in view.row_mut(y).iter_mut().enumerate() {
                    *pixel = if (x + y) % 2 == 0 { 127 } else { 129 };
                }
            }
        }
        for pre_tx_skip_energy in [0.0, 30.0] {
            let params = SearchParams {
                pre_tx_skip_energy,
                residual_skip_rd: 1.0,
                ..Default::default()
            };
            for qindex in [0, 60] {
                let (baseline, _) =
                    encode_tile_impl(&source, &reference, qindex, None, false, false).unwrap();
                let (selected, reconstruction) =
                    encode_tile_impl(&source, &reference, qindex, Some(params), false, false)
                        .unwrap();
                if qindex == 0 {
                    assert_eq!(selected, baseline);
                } else {
                    assert!(
                        selected.len() < baseline.len(),
                        "{} >= {}",
                        selected.len(),
                        baseline.len()
                    );
                }
                let decoded = decode_tile(&selected, &reference, qindex).unwrap();
                for p in PlaneIndex::ALL {
                    let actual = decoded.plane(p).view();
                    let expected = reconstruction.plane(p).view();
                    for y in 0..actual.height() {
                        assert_eq!(actual.row(y), expected.row(y));
                        if qindex == 0 {
                            assert_eq!(actual.row(y), source.plane(p).view().row(y));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lossless_finalists_reduce_texture_bytes_and_preserve_pixels() {
        let mut reference = Picture::new(64, 64, 0).unwrap();
        let mut source = reference.clone();
        for p in PlaneIndex::ALL {
            let mut reference = reference.plane_mut(p).view_mut();
            let mut source = source.plane_mut(p).view_mut();
            for y in 0..reference.height() {
                for x in 0..reference.width() {
                    let i = y * reference.width() + x;
                    reference.row_mut(y)[x] = ((i * 37 + i / 64 * 11 + 19) % 256) as u8;
                    source.row_mut(y)[x] =
                        ((i * 37 + i / 64 * 11 + 19 + 23 * ((i / 8) % 2)) % 256) as u8;
                }
            }
        }
        let (baseline, _) = encode_tile_impl(&source, &reference, 0, None, false, false).unwrap();
        let (selected, reconstruction) = encode_tile_impl(
            &source,
            &reference,
            0,
            Some(SearchParams::default()),
            false,
            true,
        )
        .unwrap();
        assert!(
            selected.len() < baseline.len(),
            "{} >= {}",
            selected.len(),
            baseline.len()
        );
        let decoded = decode_tile(&selected, &reference, 0).unwrap();
        for p in PlaneIndex::ALL {
            let source = source.plane(p).view();
            let reconstruction = reconstruction.plane(p).view();
            let decoded = decoded.plane(p).view();
            for y in 0..source.height() {
                assert_eq!(source.row(y), reconstruction.row(y));
                assert_eq!(source.row(y), decoded.row(y));
            }
        }
    }

    #[test]
    fn forward_tiles_reconstruct_changed_and_skipped_blocks() {
        for (width, height) in [(8, 8), (64, 64), (72, 40)] {
            let mut reference = Picture::new(width, height, 0).unwrap();
            for p in PlaneIndex::ALL {
                let mut v = reference.plane_mut(p).view_mut();
                for y in 0..v.height() {
                    for (x, sample) in v.row_mut(y).iter_mut().enumerate() {
                        *sample = ((x * 37 + y * 11 + 19) % 256) as u8;
                    }
                }
            }
            let mut source = reference.clone();
            for p in PlaneIndex::ALL {
                let mut v = source.plane_mut(p).view_mut();
                for y in 0..v.height() {
                    for (x, sample) in v.row_mut(y).iter_mut().enumerate() {
                        if (x / 8 + y / 8) % 2 == 0 {
                            *sample = sample.wrapping_add(23);
                        }
                    }
                }
            }
            for qindex in [0, 1, 90, 140, 190, 255] {
                let (packet, reconstruction) = encode_tile(&source, &reference, qindex).unwrap();
                let decoded = decode_tile(&packet, &reference, qindex).unwrap();
                for p in PlaneIndex::ALL {
                    let expected = reconstruction.plane(p).view();
                    let actual = decoded.plane(p).view();
                    let original = source.plane(p).view();
                    for row in 0..expected.height() {
                        assert_eq!(expected.row(row), actual.row(row));
                        if qindex == 0 {
                            assert_eq!(expected.row(row), original.row(row));
                        }
                    }
                }
                let (skipped, reconstruction) =
                    encode_tile(&reference, &reference, qindex).unwrap();
                assert!(skipped.len() <= packet.len());
                for p in PlaneIndex::ALL {
                    let expected = reference.plane(p).view();
                    let actual = reconstruction.plane(p).view();
                    for row in 0..expected.height() {
                        assert_eq!(expected.row(row), actual.row(row));
                    }
                }
            }
        }
    }
}
