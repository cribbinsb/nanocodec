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
    IS_INTER, LOSSLESS_TX_SIZE, LOSSLESS_TX4_SYMBOL, SINGLE_REF, SKIP, SKIP_MODE_FILTER,
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
use nc_dsp::{BlockSize, Dsp, Residual16};

/// Default probability that a DC block has no palette (AVM
/// `default_palette_y_mode_cdf`), for tiles that do not adapt.
const PALETTE_Y: [u16; 2] = [2723, 0];

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
    /// DC blocks code a palette flag (`allow_screen_content_tools`).
    pub screen_content: bool,
    /// Lossy blocks code their luma transform partition (`TX_MODE_SELECT`);
    /// otherwise every block uses its largest transform
    /// (`TX_MODE_LARGEST`, which our encoder never writes).
    pub tx_select: bool,
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

/// An inter picture's references in rank order (`num_total_refs` of
/// them, one to seven) with their distances from the current picture in
/// frames (AVM `ref_frame_relative_dist`), the frame's interpolation
/// filter mode and whether blocks may code skip mode.
#[derive(Clone, Copy)]
pub(crate) struct InterReference<'r> {
    pub pictures: &'r [Option<&'r Picture>],
    pub distances: &'r [i32],
    pub filter: FrameFilter,
    pub skip_mode: bool,
}

impl<'r> InterReference<'r> {
    /// The picture of reference rank `rank`.
    fn picture_of(&self, rank: u8) -> Result<&'r Picture> {
        self.pictures
            .get(usize::from(rank))
            .copied()
            .flatten()
            .ok_or(DecodeError::Malformed("AV2 reference rank"))
    }
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
    tx_select: bool,
    screen_content: bool,
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
    residual: Vec<i16>,
    tx_prediction: Vec<u8>,
    /// The whole reconstruction of a transform that crosses the picture
    /// edge (empty until one does: our encoder's blocks never cross).
    crossing: Vec<u8>,
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
    /// The frame's analysis records, when they are collected.
    #[cfg(feature = "analysis")]
    analysis: crate::tile_analysis::Recorder<'r>,
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
    pub(crate) pictures: PictureRecycler,
    cdef_source: cdef::PictureFilterScratch,
    ccso_source: CcsoSource,
    deblock_map: deblock::EdgeMap,
    /// The decoder's analysis, collected only when enabled.
    #[cfg(feature = "analysis")]
    pub(crate) stats: Option<nc_common::stream_stats::StreamStats>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            dsp: Dsp::new(),
            pictures: PictureRecycler::default(),
            cdef_source: cdef::PictureFilterScratch::default(),
            ccso_source: CcsoSource::new(),
            deblock_map: deblock::EdgeMap::default(),
            #[cfg(feature = "analysis")]
            stats: None,
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

/// Check an inter frame's references against the subset and give the
/// motion lists any second reference.
/// # Errors
/// Rejects skip mode with switchable filters (neighbouring skip-mode
/// blocks' filter contexts are outside the subset) or beside a second
/// reference (skip mode would pair the two references, AVM
/// `av2_setup_skip_mode_allowed`).
fn prepare_references(state: &mut BlockState, reference: Option<InterReference<'_>>) -> Result<()> {
    let Some(reference) = reference else {
        return Ok(());
    };
    if reference.skip_mode && reference.filter == FrameFilter::Switchable {
        return Err(DecodeError::Unsupported(
            "AV2 skip mode with switchable filters",
        ));
    }
    if reference.pictures.len() < 2 {
        return Ok(());
    }
    if reference.skip_mode {
        return Err(DecodeError::Unsupported(
            "AV2 skip mode with two references",
        ));
    }
    state.motion.set_reference_distances(reference.distances)
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
        tx_select: params.tx_select,
        screen_content: params.screen_content,
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
        crossing: if width % SUPERBLOCK == 0 && height % SUPERBLOCK == 0 {
            Vec::new()
        } else {
            vec![0; MAX_PLANE_AREA]
        },
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
        #[cfg(feature = "analysis")]
        analysis: crate::tile_analysis::Recorder::new(
            work.stats.as_mut(),
            &params,
            (reference.is_some(), data.len()),
        ),
    };
    prepare_references(&mut decoder.state, reference)?;
    let mut partitions = PartitionReader::new(width, height)?;
    for y in (0..height).step_by(SUPERBLOCK) {
        for x in (0..width).step_by(SUPERBLOCK) {
            decoder.state.motion.begin_superblock(x, y)?;
            decoder.superblock_qindex.start_superblock();
            partitions.read_superblock(&mut reader, x, y, &mut |reader, x, y, size| {
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
        #[cfg(feature = "analysis")]
        {
            self.analysis.start = reader.bits_consumed();
        }
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

    /// Whether a lossy block of side `1 << block_log2` splits its luma
    /// transform four ways: read in frames that select transform sizes,
    /// never in frames of largest transforms.
    fn read_tx_split(
        &self,
        reader: &mut Reader<'_>,
        block_log2: u32,
        is_inter: bool,
    ) -> Result<bool> {
        if self.tx_select {
            tx_partition::read(reader, block_log2, is_inter)
        } else {
            Ok(false)
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
        // `read_palette_mode_info`: every block of the subset (8×8 to
        // 64×64) may use a palette when its luma mode is DC.
        if self.screen_content
            && luma == Mode::Dc
            && reader.model(&PALETTE_Y, |m| &mut m.palette_y)? != 0
        {
            return Err(DecodeError::Unsupported("AV2 palette"));
        }
        let layout = if self.lossless() {
            Layout::lossless(block_log2)
        } else {
            Layout::lossy(block_log2, self.read_tx_split(reader, block_log2, false)?)
        };
        self.residual(
            reader,
            layout,
            (x, y),
            Source::Intra { luma, chroma },
            false,
        )?;
        self.state
            .edges
            .record(block_edges((x, y), self.plane_qindex, layout, false));
        #[cfg(feature = "analysis")]
        if self.analysis.on() {
            let block = self.placed_block(reader, (x, y), size);
            self.analysis.intra_block(block, (luma, chroma), layout);
        }
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
                reference: 0,
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
        let rank = self.read_reference(reader, (x, y), size)?;
        let (mode, mv) = self.read_vector(reader, (x, y), size, rank)?;
        let filter = match reference.filter {
            FrameFilter::Fixed(filter) => filter,
            FrameFilter::Switchable if interp_filter::is_coded(mode) => {
                let context = self.state.filter_context(x, y, size, rank);
                self.filter_symbols.read(reader, context)?
            }
            FrameFilter::Switchable => interp_filter::UNCODED,
        };
        self.predict(reference.picture_of(rank)?, filter, (x, y), size, mv)?;
        let motion = BlockMotion::Single {
            vector: mv,
            is_new: mode == InterMode::New,
            reference: rank,
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
        self.predict_compound(reference.picture_of(0)?, (x, y), size, pair)?;
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
        #[cfg(feature = "analysis")]
        let mut analysis_layout = None;
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
                Layout::lossy(block_log2, self.read_tx_split(reader, block_log2, true)?)
            };
            self.residual(reader, layout, (x, y), Source::Inter, true)?;
            #[cfg(feature = "analysis")]
            {
                analysis_layout = Some(layout);
            }
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
                reference: match motion {
                    BlockMotion::Single { reference, .. } => reference,
                    BlockMotion::Compound(_) => 0,
                },
            },
        );
        #[cfg(feature = "analysis")]
        if self.analysis.on() {
            let block = self.placed_block(reader, (x, y), size);
            self.analysis.inter_block(block, &motion, analysis_layout);
        }
        self.state.motion.commit_motion(x, y, size, motion)
    }

    /// The current block as the analysis records place it.
    #[cfg(feature = "analysis")]
    fn placed_block(
        &self,
        reader: &Reader<'_>,
        origin: (usize, usize),
        size: usize,
    ) -> crate::tile_analysis::PlacedBlock {
        crate::tile_analysis::PlacedBlock {
            origin,
            size,
            qindex: self.plane_qindex.luma.ac,
            bits: (reader.bits_consumed() - self.analysis.start) as u32,
        }
    }

    /// Read, reconstruct and store every transform of the block in coding
    /// order, updating the entropy contexts.
    fn residual(
        &mut self,
        reader: &mut Reader<'_>,
        layout: Layout,
        (x, y): (usize, usize),
        source: Source,
        is_inter: bool,
    ) -> Result<()> {
        // Our encoder's blocks lie inside the picture; other encoders'
        // may cross its right or bottom edge.
        let side = 1 << layout.block_log2;
        if x + side > self.recon.width() || y + side > self.recon.height() {
            self.residual_planes::<true>(reader, layout, (x, y), source, is_inter)
        } else {
            self.residual_planes::<false>(reader, layout, (x, y), source, is_inter)
        }
    }

    /// [`Self::residual`] for a block that crosses the picture edge
    /// (`CROSSING`) or lies inside it (compiled without the edge handling:
    /// that cost our own streams 0.9% of their decode instructions).
    fn residual_planes<const CROSSING: bool>(
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
            let whole = (tx.width(), tx.width());
            let edge = CROSSING.then(|| PictureEdge::of(&self.recon, plane, layout));
            for position in layout.positions(plane, (x, y)) {
                let inside = edge.as_ref().map_or(Some(whole), |e| e.visible(position));
                let Some(visible) = inside else { continue };
                let predicted = &mut self.tx_prediction[..area];
                if let Source::Intra { luma, chroma } = source {
                    // Inter prediction is loaded below, only when a
                    // residual is added to it.
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
                        load(
                            &self.recon,
                            index,
                            position,
                            (tx.width(), visible),
                            predicted,
                        );
                    }
                    let output = inverse(
                        &self.dsp,
                        (tx, tx_type),
                        (levels, self.level_coding == LevelCoding::Tcq && plane == 0),
                        self.plane_qindex.of(index),
                        &mut self.residual[..area],
                    )?;
                    // Into the picture, or for a transform crossing the
                    // picture edge aside, of which the part inside is kept.
                    let side = tx.width();
                    let crossing = CROSSING && visible != whole;
                    let prediction = PlaneView::new(predicted, side, side, side).unwrap();
                    let mut plane = self.recon.plane_mut(index).view_mut();
                    let mut destination = if crossing {
                        PlaneViewMut::new(&mut self.crossing[..area], side, side, side)
                    } else {
                        plane.sub_view_mut(position.0, position.1, side, side)
                    }
                    .unwrap();
                    let size = BlockSize::new(side, side).unwrap();
                    self.dsp
                        .reconstruct_i16(size, &prediction, output, &mut destination);
                    if crossing {
                        let whole = &self.crossing[..area];
                        store_part(&mut self.recon, index, position, (side, visible), whole);
                    }
                } else if let Source::Intra { .. } = source {
                    // No coefficients: the reconstruction is the prediction
                    // (a zero residual added and clipped changes nothing).
                    store_part(
                        &mut self.recon,
                        index,
                        position,
                        (tx.width(), visible),
                        predicted,
                    );
                }
                self.state.update_coeff_context(plane, position, tx, levels);
                if plane == 1 {
                    u_coded = coded;
                }
            }
        }
        Ok(())
    }

    /// Read the block's reference rank (AVM `read_single_ref`): for each
    /// rank but the last, in order, one symbol says whether it is the one.
    fn read_reference(
        &self,
        reader: &mut Reader<'_>,
        (x, y): (usize, usize),
        size: usize,
    ) -> Result<u8> {
        let count = self.state.motion.reference_count();
        if count < 2 {
            return Ok(0);
        }
        let contexts = self.state.reference_contexts(x, y, size, count);
        for (rank, &context) in contexts[..count - 1].iter().enumerate() {
            let fixed = &SINGLE_REF[usize::from(context)][rank];
            let symbol = reader.model(fixed, |m| &mut m.single_ref[usize::from(context)][rank])?;
            if symbol == 1 {
                return Ok(rank as u8);
            }
        }
        Ok(count as u8 - 1)
    }

    /// Read the inter mode, DRL index and vector difference of a block on
    /// reference rank `rank`.
    fn read_vector(
        &self,
        reader: &mut Reader<'_>,
        (x, y): (usize, usize),
        size: usize,
        rank: u8,
    ) -> Result<(InterMode, (i32, i32))> {
        let candidates = self.state.motion.candidates_for(x, y, size, rank)?;
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
        if x + size > self.recon.width() || y + size > self.recon.height() {
            return self.predict_crossing(reference, filter, (x, y), size, mv);
        }
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

    /// [`Self::predict`] for a block crossing the picture's right or bottom
    /// edge (other encoders' streams): each plane is predicted whole aside
    /// and the part inside the picture kept.
    #[cold]
    fn predict_crossing(
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
            let origin = plane_origin(plane, (x, y));
            let view = self.recon.plane(index).view();
            let edge = PictureEdge {
                plane_size: (view.width(), view.height()),
                side,
            };
            let visible = edge
                .visible(origin)
                .ok_or(DecodeError::Malformed("AV2 block outside the picture"))?;
            let whole = self
                .crossing
                .get_mut(..side * side)
                .ok_or(DecodeError::Malformed("AV2 block outside the picture"))?;
            let mut block = PlaneViewMut::new(whole, side, side, side)
                .ok_or(DecodeError::Malformed("AV2 block outside the picture"))?;
            let source = reference.plane(index);
            prediction::predict_into(&self.dsp, filter, source, origin, mv, subsample, &mut block)?;
            store_part(&mut self.recon, index, origin, (side, visible), whole);
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

/// The residual of a coded transform from its `levels` (trellis-coded
/// when `tcq`): a constant for a DC-only one, else samples in `residual`.
#[allow(
    clippy::inline_always,
    reason = "one call per coded transform from each instance of `residual_planes`; out of line it measured 0.3% of our own streams' decode instructions (callgrind)"
)]
#[inline(always)]
fn inverse<'a>(
    dsp: &Dsp,
    (tx, tx_type): (TxSize, crate::tx_type::TxType),
    (levels, tcq): (&mut [i32], bool),
    qindex: crate::transform::Qindices,
    residual: &'a mut [i16],
) -> Result<Residual16<'a>> {
    let dequantizer = if tcq {
        Dequantizer::Tcq
    } else {
        Dequantizer::Plain
    };
    let eob = coeff::end_of_block(&levels[..tx.coded_len()], tx);
    if let Some(constant) =
        transform::dc_only_residual((tx, tx_type), (levels, eob, dequantizer), qindex)
    {
        return Ok(Residual16::Constant(constant));
    }
    transform::reconstruct_with(
        dsp,
        (tx, tx_type),
        (levels, eob, dequantizer),
        qindex,
        residual,
    )?;
    Ok(Residual16::Samples(residual))
}

/// How the transforms of one plane of a block that crosses the picture's
/// right or bottom edge meet it: those that start outside are not coded,
/// and of one that crosses only the part inside is kept (AVM
/// `av2_foreach_transformed_block_in_plane`'s `max_blocks_*`).
struct PictureEdge {
    /// The plane's width and height.
    plane_size: (usize, usize),
    /// Side of the plane's transforms.
    side: usize,
}

impl PictureEdge {
    fn of(picture: &Picture, plane: usize, layout: Layout) -> Self {
        let view = picture.plane(PlaneIndex::ALL[plane]).view();
        Self {
            plane_size: (view.width(), view.height()),
            side: layout.tx(plane).width(),
        }
    }

    /// The columns and rows of the transform at `position` that lie inside
    /// the picture; `None` when it starts outside.
    fn visible(&self, position: (usize, usize)) -> Option<(usize, usize)> {
        let (width, height) = self.plane_size;
        (position.0 < width && position.1 < height).then(|| {
            (
                self.side.min(width - position.0),
                self.side.min(height - position.1),
            )
        })
    }
}

/// Store the top-left `visible` columns and rows of the `side`-wide raster
/// `samples` at (`x`, `y`) of `plane` (all of it, or the part inside the
/// picture of a transform crossing its edge).
fn store_part(
    picture: &mut Picture,
    plane: PlaneIndex,
    (x, y): (usize, usize),
    (side, visible): (usize, (usize, usize)),
    samples: &[u8],
) {
    if visible == (side, side) {
        return store(picture, plane, (x, y), side, samples);
    }
    let mut view = picture.plane_mut(plane).view_mut();
    for (row, values) in samples.chunks_exact(side).take(visible.1).enumerate() {
        nc_common::copy_row(
            &mut view.row_mut(y + row)[x..x + visible.0],
            &values[..visible.0],
        );
    }
}

/// Copy the top-left `visible` columns and rows of the block of `plane` of
/// `picture` at (`x`, `y`) into the `side`-wide raster `out` (the inverse
/// of [`store_part`]; samples outside the picture keep their values).
#[inline]
fn load(
    picture: &Picture,
    plane: PlaneIndex,
    (x, y): (usize, usize),
    (side, visible): (usize, (usize, usize)),
    out: &mut [u8],
) {
    let view = picture.plane(plane).view();
    for (row, values) in out.chunks_exact_mut(side).take(visible.1).enumerate() {
        nc_common::copy_row(
            &mut values[..visible.0],
            &view.row(y + row)[x..x + visible.0],
        );
    }
}
