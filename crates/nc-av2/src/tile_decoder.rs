//! Decoder for the tile subset written by [`crate::block_coder`]: square
//! leaves from 8×8 to 64×64 reached through binary partitions, intra blocks
//! (DC/V/H/Paeth) in key and inter pictures, and inter blocks coded as
//! NEARMV, GLOBALMV or NEWMV with the frame's or a per-block switchable
//! interpolation filter ([`crate::interp_filter`]) or in skip mode (the
//! same-reference compound `NEAR_NEARMV` of the skip-mode list, in frames
//! with one fixed filter), and optional residual.
//! Luma uses one transform per block or the 4-way split
//! ([`crate::tx_partition`]). Mirrors the encoder's syntax order exactly.
use crate::block_coder::{
    IS_INTER, LOSSLESS_TX_SIZE, LOSSLESS_TX4_SYMBOL, SKIP, SKIP_MODE_FILTER,
    SKIP_MODE_SKIP_CONTEXTS, block_edges, lossless_size_group, plane_origin, skipped_layout, store,
};
use crate::block_state::{BlockInfo, BlockState};
use crate::ccso::{self, CcsoParams, CcsoSource, UnitFlags};
use crate::cdef::{self, CdefParams, StrengthIndices};
use crate::cdf_models::TileCdfs;
use crate::coeff::LevelCoding;
use crate::deblock::{self, DeblockParams};
use crate::delta_q::SuperblockQindex;
use crate::entropy::Reader;
use crate::interp_filter::{self, FilterSymbols, FrameFilter};
use crate::intra_edges::{self, IntraTarget};
use crate::intra_modes::{self, Mode, ModeContext};
use crate::mv_refs::{self, BlockMotion, InterMode, ListConfig, SKIP_MODE};
use crate::partition::{PartitionReader, SUPERBLOCK};
use crate::transform::{Dequantizer, PlaneQindex, QuantDeltas};
use crate::tx_partition::{self, Layout};
use crate::tx_size::TxSize;
use crate::tx_type::TxSetMode;
use crate::{coeff, motion, prediction, transform};
use nc_common::decode::{DecodeError, Result};
use nc_common::{Picture, PictureRecycler, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::intra_prediction::av2::Av2IntraEdges;
use nc_dsp::{BlockSize, Dsp, Residual32};

/// Most samples of one plane block (64×64 luma).
const MAX_PLANE_AREA: usize = 64 * 64;

/// Frame-level parameters of one tile.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TileParams {
    pub qindex: u8,
    /// The frame's quantizer offsets.
    pub quant_deltas: QuantDeltas,
    pub adaptive_cdf: bool,
    /// Applied to the picture after its last block.
    pub deblock: DeblockParams,
    /// Applied after deblocking.
    pub cdef: CdefParams,
    /// Applied after CDEF, classifying on the deblocked luma.
    pub ccso: CcsoParams,
    /// The frame's reduced transform set.
    pub tx_set: TxSetMode,
    /// The sequence's motion-vector reference lists.
    pub reference_lists: ListConfig,
    /// How lossy luma levels are coded (parity hiding, TCQ).
    pub levels: LevelCoding,
    /// Superblock quantizer deltas: `log2(delta_q_res)` when coded.
    pub delta_q: Option<u8>,
}

/// Where a block's predictions come from.
#[derive(Clone, Copy)]
enum Source {
    /// Intra prediction of each transform from the reconstruction so far.
    Intra { luma: Mode, chroma: Mode },
    /// The block's inter prediction, already in the reconstruction.
    Inter,
}

/// An inter picture's reference, the frame's interpolation filter mode and
/// whether blocks may code skip mode.
#[derive(Clone, Copy)]
pub(crate) struct InterReference<'r> {
    pub picture: &'r Picture,
    pub filter: FrameFilter,
    pub skip_mode: bool,
}

struct TileDecoder<'r> {
    /// The frame's base qindex (lossless test, coefficient CDF category).
    qindex: u8,
    /// The current block's qindex per plane (dequantization and
    /// deblocking): the frame's, or its superblock's with deltas.
    plane_qindex: PlaneQindex,
    /// The frame's quantizer offsets, applied to every block qindex.
    quant_deltas: QuantDeltas,
    /// The superblock quantizer deltas' state ([`crate::delta_q`]).
    superblock_qindex: SuperblockQindex,
    tx_set: TxSetMode,
    level_coding: LevelCoding,
    reference: Option<InterReference<'r>>,
    /// Fixed probabilities of the per-block filter symbol.
    filter_symbols: FilterSymbols,
    state: BlockState,
    recon: Picture,
    dsp: Dsp,
    /// Compound prediction buffers (skip mode).
    compound: prediction::CompoundScratch,
    /// Scratch of one transform: levels, residual, prediction, reconstruction.
    levels: Vec<i32>,
    residual: Vec<i32>,
    tx_prediction: Vec<u8>,
    /// Intra edge arrays, reused across predictions.
    edges: Av2IntraEdges,
    cdef: CdefParams,
    /// Each filter block's CDEF strength set, as read.
    cdef_indices: StrengthIndices,
    /// Default index models, for fixed probabilities when the tile does not
    /// adapt (present only when blocks code indices).
    cdef_defaults: Option<Box<TileCdfs>>,
    ccso: CcsoParams,
    /// Each CCSO unit's flags, as read.
    ccso_flags: UnitFlags,
    /// Default flag models, for fixed probabilities when the tile does not
    /// adapt (present only when the frame filters with CCSO).
    ccso_defaults: Option<Box<TileCdfs>>,
}

/// What the decoder keeps between pictures instead of allocating them per
/// picture: released reconstructions (a fresh zeroed 720p picture per frame
/// was ~4% of decoding), the kernel table and CDEF's source copy.
/// Decoding writes every sample of the reconstruction before reading it
/// (blocks tile the picture; prediction reads only decoded neighbours), so
/// a reused picture's stale samples never reach the output.
#[derive(Debug)]
pub(crate) struct Workspace {
    dsp: Dsp,
    pictures: PictureRecycler,
    cdef_source: cdef::PictureFilterScratch,
    ccso_source: CcsoSource,
    deblock_map: deblock::EdgeMap,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            dsp: Dsp::new(),
            pictures: PictureRecycler::default(),
            cdef_source: cdef::PictureFilterScratch::default(),
            ccso_source: CcsoSource::new(),
            deblock_map: deblock::EdgeMap::default(),
        }
    }
}

impl Workspace {
    /// Keep a picture nothing else holds for a later reconstruction.
    pub(crate) fn release(&mut self, picture: std::rc::Rc<Picture>) {
        if let Ok(picture) = std::rc::Rc::try_unwrap(picture) {
            self.pictures.give(picture);
        }
    }
}

/// Decode one tile of a key (`reference` is `None`) or inter picture and
/// apply its in-loop filters (deblocking, then CDEF). An adapting tile
/// starts from `inherited` models (cross-frame context) or the defaults,
/// and its final models are returned with the picture.
/// # Errors
/// Rejects syntax outside the subset and malformed or truncated data.
pub(crate) fn decode(
    data: &[u8],
    (width, height): (usize, usize),
    params: TileParams,
    reference: Option<InterReference<'_>>,
    inherited: Option<&TileCdfs>,
    work: &mut Workspace,
) -> Result<(Picture, Option<Box<TileCdfs>>)> {
    if reference.is_some_and(|r| r.skip_mode && r.filter == FrameFilter::Switchable) {
        // Neighbouring skip-mode blocks' filter contexts are outside the subset.
        return Err(DecodeError::Unsupported(
            "AV2 skip mode with switchable filters",
        ));
    }
    let mut reader = Reader::new(data)?;
    match (params.adaptive_cdf, inherited) {
        (true, Some(models)) => reader.adapt_from(models.clone()),
        (true, None) => reader.enable_adaptation(params.qindex),
        (false, Some(_)) => {
            return Err(DecodeError::Unsupported(
                "AV2 cross-frame context without CDF updates",
            ));
        }
        (false, None) => {}
    }
    let mut decoder = TileDecoder {
        qindex: params.qindex,
        plane_qindex: PlaneQindex::new(params.qindex, params.quant_deltas),
        quant_deltas: params.quant_deltas,
        superblock_qindex: SuperblockQindex::new(params.qindex, params.delta_q),
        tx_set: params.tx_set,
        level_coding: params.levels,
        reference,
        filter_symbols: FilterSymbols::new(),
        state: BlockState::new(width, height, params.reference_lists)?,
        recon: work
            .pictures
            .take(width, height, 0)
            .map_err(|_| DecodeError::Malformed("AV2 picture allocation failed"))?,
        dsp: work.dsp,
        compound: prediction::CompoundScratch::default(),
        levels: vec![0; TxSize::Tx64.coded_len()],
        residual: vec![0; MAX_PLANE_AREA],
        tx_prediction: vec![0; MAX_PLANE_AREA],
        edges: Av2IntraEdges::new(),
        cdef: params.cdef,
        cdef_indices: StrengthIndices::new(width, height),
        cdef_defaults: params
            .cdef
            .codes_indices()
            .then(|| Box::new(TileCdfs::new(params.qindex))),
        ccso: params.ccso,
        ccso_flags: UnitFlags::new(width, height),
        ccso_defaults: params
            .ccso
            .is_on()
            .then(|| Box::new(TileCdfs::new(params.qindex))),
    };
    let mut partitions = PartitionReader::new(width, height)?;
    for y in (0..height).step_by(SUPERBLOCK) {
        for x in (0..width).step_by(SUPERBLOCK) {
            decoder.state.motion.begin_superblock(x, y)?;
            decoder.superblock_qindex.start_superblock();
            partitions.read_superblock(&mut reader, x, y, &mut |reader, x, y, size| {
                if x + size > width || y + size > height {
                    return Err(DecodeError::Unsupported(
                        "AV2 block crossing the picture edge",
                    ));
                }
                decoder.block(reader, (x, y), size)
            })?;
        }
    }
    if params.deblock.is_on() {
        deblock::filter_picture(
            &mut decoder.recon,
            &decoder.state.edges,
            params.deblock,
            &decoder.dsp,
            &mut work.deblock_map,
        );
    }
    if params.ccso.is_on() {
        work.ccso_source
            .fill_from(&decoder.recon.plane(PlaneIndex::Y).view());
    }
    cdef::filter_picture(
        &mut decoder.recon,
        &decoder.state,
        &decoder.cdef_indices,
        params.cdef,
        &decoder.dsp,
        &mut work.cdef_source,
    );
    if params.ccso.is_on() {
        ccso::filter_picture(
            &mut decoder.recon,
            &work.ccso_source,
            &params.ccso,
            &decoder.ccso_flags,
        );
    }
    Ok((decoder.recon, reader.into_models()))
}

impl TileDecoder<'_> {
    fn lossless(&self) -> bool {
        self.qindex == 0
    }

    fn block(
        &mut self,
        reader: &mut Reader<'_>,
        (x, y): (usize, usize),
        size: usize,
    ) -> Result<()> {
        self.state.check_order(x, y)?;
        if self.reference.is_some_and(|r| r.skip_mode) {
            let context = self.state.skip_mode_context(x, y, size);
            if reader.model(&SKIP_MODE[context], |m| &mut m.skip_mode[context])? == 1 {
                return self.skip_mode_block(reader, (x, y), size);
            }
        }
        let is_inter = if self.reference.is_some() {
            let context = self.state.intra_inter_context(x, y, size);
            reader.model(&IS_INTER[context], |m| &mut m.is_inter[context])? == 1
        } else {
            false
        };
        if is_inter {
            self.inter_block(reader, (x, y), size)
        } else {
            self.read_cdef_index(reader, ((x, y), size), false)?;
            self.intra_block(reader, (x, y), size)
        }
    }

    /// `read_cdef`, after the block's skip flag (intra blocks never skip),
    /// then `read_ccso`, then the superblock quantizer delta
    /// (`read_delta_q_params`), which sets the qindex the block dequantizes
    /// and deblocks with.
    fn read_cdef_index(
        &mut self,
        reader: &mut Reader<'_>,
        (position, size): ((usize, usize), usize),
        skip: bool,
    ) -> Result<()> {
        cdef::read_index(
            reader,
            self.cdef,
            &mut self.cdef_indices,
            position,
            skip,
            self.cdef_defaults.as_deref_mut(),
        )?;
        if let Some(defaults) = self.ccso_defaults.as_deref_mut() {
            ccso::read_flags(reader, &self.ccso, &mut self.ccso_flags, position, defaults)?;
        }
        let qindex = self
            .superblock_qindex
            .read(reader, size == SUPERBLOCK, skip)?;
        self.plane_qindex = PlaneQindex::new(qindex, self.quant_deltas);
        Ok(())
    }

    fn intra_block(
        &mut self,
        reader: &mut Reader<'_>,
        (x, y): (usize, usize),
        size: usize,
    ) -> Result<()> {
        let block_log2 = size.ilog2();
        let (above, left) = self.state.mode_neighbours(x, y, size);
        let (luma, chroma) = intra_modes::read_block(
            reader,
            self.lossless(),
            ModeContext {
                above,
                left,
                block_log2,
            },
        )?;
        let layout = if self.lossless() {
            Layout::lossless(block_log2)
        } else {
            Layout::lossy(block_log2, tx_partition::read(reader, block_log2, false)?)
        };
        self.residual_planes(
            reader,
            layout,
            (x, y),
            Source::Intra { luma, chroma },
            false,
        )?;
        self.state
            .edges
            .record(block_edges((x, y), self.plane_qindex, layout, false));
        self.state.record(
            x,
            y,
            size,
            BlockInfo {
                is_inter: false,
                skip: false,
                luma_mode: luma,
                filter: None,
                skip_mode: false,
            },
        );
        self.state.motion.commit_intra(x, y, size)
    }

    fn inter_block(
        &mut self,
        reader: &mut Reader<'_>,
        (x, y): (usize, usize),
        size: usize,
    ) -> Result<()> {
        let reference = self.reference.expect("inter blocks only in inter pictures");
        let skip_context = self.state.skip_context(x, y, size);
        let skip = reader.model(&SKIP[skip_context], |m| &mut m.block_skip[skip_context])? != 0;
        self.read_cdef_index(reader, ((x, y), size), skip)?;
        let (mode, mv) = self.read_vector(reader, (x, y), size)?;
        let filter = match reference.filter {
            FrameFilter::Fixed(filter) => filter,
            FrameFilter::Switchable if interp_filter::is_coded(mode) => {
                let context = self.state.filter_context(x, y, size);
                self.filter_symbols.read(reader, context)?
            }
            FrameFilter::Switchable => interp_filter::UNCODED,
        };
        self.predict(reference.picture, filter, (x, y), size, mv)?;
        let motion = BlockMotion::Single {
            vector: mv,
            is_new: mode == InterMode::New,
        };
        self.finish_inter(reader, (x, y), size, (skip, filter), motion)
    }

    /// A skip-mode block (AVM `read_inter_block_mode_info` with
    /// `skip_mode`): `skip_txfm` in the skip-mode contexts, the CDEF index,
    /// the skip-mode DRL index, then any residual; the prediction averages
    /// the two vectors of the selected pair.
    fn skip_mode_block(
        &mut self,
        reader: &mut Reader<'_>,
        (x, y): (usize, usize),
        size: usize,
    ) -> Result<()> {
        let reference = self.reference.expect("skip mode only in inter pictures");
        let skip_context = self.state.skip_context(x, y, size) + SKIP_MODE_SKIP_CONTEXTS;
        let skip = reader.model(&SKIP[skip_context], |m| &mut m.block_skip[skip_context])? != 0;
        self.read_cdef_index(reader, ((x, y), size), skip)?;
        let candidates = self.state.motion.skip_candidates(x, y, size)?;
        let index = mv_refs::read_skip_drl(reader, candidates.max_drl_bits)?;
        let pair = candidates.pair(index);
        self.predict_compound(reference.picture, (x, y), size, pair)?;
        self.finish_inter(
            reader,
            (x, y),
            size,
            (skip, SKIP_MODE_FILTER),
            BlockMotion::Compound(pair),
        )
    }

    /// The residual and state updates of an inter block whose prediction is
    /// in the reconstruction.
    fn finish_inter(
        &mut self,
        reader: &mut Reader<'_>,
        (x, y): (usize, usize),
        size: usize,
        (skip, filter): (bool, prediction::InterpFilter),
        motion: BlockMotion,
    ) -> Result<()> {
        let block_log2 = size.ilog2();
        if skip {
            self.state.clear_coeff_contexts(x, y, size);
            self.state.edges.record(block_edges(
                (x, y),
                self.plane_qindex,
                skipped_layout(block_log2, self.lossless()),
                true,
            ));
        } else {
            let layout = if self.lossless() {
                let group = lossless_size_group(block_log2);
                let symbol = reader.model(&LOSSLESS_TX_SIZE, |m| &mut m.lossless_size[group][1])?;
                if symbol != LOSSLESS_TX4_SYMBOL {
                    return Err(DecodeError::Unsupported(
                        "AV2 large lossless inter transform",
                    ));
                }
                Layout::lossless(block_log2)
            } else {
                Layout::lossy(block_log2, tx_partition::read(reader, block_log2, true)?)
            };
            self.residual_planes(reader, layout, (x, y), Source::Inter, true)?;
            self.state
                .edges
                .record(block_edges((x, y), self.plane_qindex, layout, false));
        }
        self.state.record(
            x,
            y,
            size,
            BlockInfo {
                is_inter: true,
                skip,
                luma_mode: Mode::Dc,
                filter: Some(filter),
                skip_mode: matches!(motion, BlockMotion::Compound(_)),
            },
        );
        self.state.motion.commit_motion(x, y, size, motion)
    }

    /// Read, reconstruct and store every transform of the block in coding
    /// order, updating the entropy contexts.
    fn residual_planes(
        &mut self,
        reader: &mut Reader<'_>,
        layout: Layout,
        (x, y): (usize, usize),
        source: Source,
        is_inter: bool,
    ) -> Result<()> {
        let mut u_coded = false;
        for (plane, index) in PlaneIndex::ALL.into_iter().enumerate() {
            let tx = layout.tx(plane);
            let area = tx.width() * tx.width();
            for position in layout.positions(plane, (x, y)) {
                let predicted = &mut self.tx_prediction[..area];
                match source {
                    Source::Intra { luma, chroma } => {
                        let target = IntraTarget {
                            plane,
                            block: (x, y),
                            block_size: 1 << layout.block_log2,
                            tx: position,
                            side: tx.width(),
                        };
                        intra_edges::predict(
                            self.recon.plane(index),
                            &self.state,
                            target,
                            if plane == 0 { luma } else { chroma },
                            &self.dsp,
                            &mut self.edges,
                            predicted,
                        );
                    }
                    // Loaded below, only when a residual is added to it.
                    Source::Inter => {}
                }
                let levels = &mut self.levels[..tx.coded_len()];
                let context = self.state.coeff_context(
                    plane,
                    position,
                    tx,
                    layout.plane_log2(plane),
                    is_inter,
                    u_coded,
                )?;
                let tx_type = coeff::read_typed(
                    reader,
                    levels,
                    tx,
                    context,
                    self.qindex,
                    (self.tx_set, self.level_coding),
                )?;
                let coded = levels.iter().any(|&c| c != 0);
                if coded {
                    if let Source::Inter = source {
                        load(&self.recon, index, position, tx.width(), predicted);
                    }
                    let residual = &mut self.residual[..area];
                    let dequantizer = if self.level_coding == LevelCoding::Tcq && plane == 0 {
                        Dequantizer::Tcq
                    } else {
                        Dequantizer::Plain
                    };
                    let qindex = self.plane_qindex.of(index);
                    let eob = coeff::end_of_block(&levels[..tx.coded_len()], tx);
                    let output = if let Some(constant) = transform::dc_only_residual(
                        (tx, tx_type),
                        (levels, eob, dequantizer),
                        qindex,
                    ) {
                        Residual32::Constant(constant)
                    } else {
                        transform::reconstruct_with(
                            &self.dsp,
                            (tx, tx_type),
                            (levels, eob, dequantizer),
                            qindex,
                            residual,
                        )?;
                        Residual32::Samples(residual)
                    };
                    let side = tx.width();
                    let prediction = PlaneView::new(predicted, side, side, side).unwrap();
                    let mut plane = self.recon.plane_mut(index).view_mut();
                    let mut destination = plane
                        .sub_view_mut(position.0, position.1, side, side)
                        .unwrap();
                    self.dsp.reconstruct_i32(
                        BlockSize::new(side, side).unwrap(),
                        &prediction,
                        output,
                        &mut destination,
                    );
                } else if let Source::Intra { .. } = source {
                    // No coefficients: the reconstruction is the prediction
                    // (a zero residual added and clipped changes nothing).
                    store(&mut self.recon, index, position, tx.width(), predicted);
                }
                self.state.update_coeff_context(plane, position, tx, levels);
                if plane == 1 {
                    u_coded = coded;
                }
            }
        }
        Ok(())
    }

    /// Read the inter mode, DRL index and vector difference.
    fn read_vector(
        &self,
        reader: &mut Reader<'_>,
        (x, y): (usize, usize),
        size: usize,
    ) -> Result<(InterMode, (i32, i32))> {
        let candidates = self.state.motion.candidates(x, y, size)?;
        let mode = mv_refs::read_inter_mode(reader, candidates.mode_context)?;
        let index = if mode.has_drl() {
            mv_refs::read_drl(reader, candidates)?
        } else {
            0
        };
        let mv = match mode {
            InterMode::Global => (0, 0),
            InterMode::Near => candidates.vector(index)?,
            InterMode::New => {
                let delta = motion::read_mv(reader)?;
                let base = candidates.vector(index)?;
                (base.0 + delta.0, base.1 + delta.1)
            }
        };
        Ok((mode, mv))
    }

    /// Compound prediction of every plane of the block from the vector
    /// `pair` into the reconstruction.
    fn predict_compound(
        &mut self,
        reference: &Picture,
        (x, y): (usize, usize),
        size: usize,
        pair: mv_refs::Pair,
    ) -> Result<()> {
        for (plane, index) in PlaneIndex::ALL.into_iter().enumerate() {
            let subsample = u8::from(plane != 0);
            let side = size >> subsample;
            let (px, py) = plane_origin(plane, (x, y));
            prediction::predict_compound_into(
                &self.dsp,
                SKIP_MODE_FILTER,
                reference.plane(index),
                (px, py),
                [pair.0, pair.1],
                subsample,
                &mut self.compound,
                &mut block_view(&mut self.recon, index, (px, py), side)?,
            )?;
        }
        Ok(())
    }

    /// Prediction of every plane of the block into the reconstruction.
    fn predict(
        &mut self,
        reference: &Picture,
        filter: prediction::InterpFilter,
        (x, y): (usize, usize),
        size: usize,
        mv: (i32, i32),
    ) -> Result<()> {
        for (plane, index) in PlaneIndex::ALL.into_iter().enumerate() {
            let subsample = u8::from(plane != 0);
            let side = size >> subsample;
            let (px, py) = plane_origin(plane, (x, y));
            prediction::predict_into(
                &self.dsp,
                filter,
                reference.plane(index),
                (px, py),
                mv,
                subsample,
                &mut block_view(&mut self.recon, index, (px, py), side)?,
            )?;
        }
        Ok(())
    }
}

/// The `side`×`side` block of `plane` of `picture` at (`x`, `y`).
fn block_view(
    picture: &mut Picture,
    plane: PlaneIndex,
    (x, y): (usize, usize),
    side: usize,
) -> Result<PlaneViewMut<'_>> {
    picture
        .plane_mut(plane)
        .region_mut(x as isize, y as isize, side, side)
        .ok_or(DecodeError::Malformed("AV2 block outside the picture"))
}

/// Copy the `side`×`side` block of `plane` of `picture` at (`x`, `y`) into
/// the raster `out` (the inverse of [`store`]).
fn load(picture: &Picture, plane: PlaneIndex, (x, y): (usize, usize), side: usize, out: &mut [u8]) {
    let view = picture.plane(plane).view();
    for (row, values) in out.chunks_exact_mut(side).enumerate() {
        nc_common::copy_row(values, &view.row(y + row)[x..x + side]);
    }
}
