//! Block syntax and reconstruction of one AV2 tile on the shared engine path.
//!
//! The engine chooses partitions (square leaves from 8×8 to 64×64) and each
//! leaf's prediction (intra, or inter with a quarter-sample vector); this
//! coder writes the AVM v1.0.0 syntax for that choice (`write_modes_b`,
//! `write_inter_mode`, `write_tx_partition`, `pack_txb_tokens`
//! counterparts), picks the luma transform layout and decides between coding
//! and dropping an inter residual by rate and distortion, and writes the
//! reconstruction into the engine's picture. [`crate::tile_decoder`] reads
//! exactly this subset.
use crate::block_state::{BlockInfo, BlockState, SavedContexts};
use crate::ccso::{self, CcsoParams, CcsoSource, UnitFlags};
use crate::ccso_search;
use crate::cdef::{self, CdefParams, StrengthIndices};
use crate::cdf_models::TileCdfs;
use crate::coeff::tcq::{TcqQuantizer, TcqSearch};
use crate::coeff::trellis::{CodedBlock, TrellisQuantizer};
use crate::coeff::{LevelCoding, LevelStats};
use crate::coeff_trim::{Trim, TrimOptions};
use crate::deblock::{BlockEdges, DeblockParams};
use crate::delta_q::SuperblockQindex;
use crate::entropy::Writer;
use crate::interp_filter::{self, FilterSymbols, FrameFilter};
use crate::intra_edges::{self, IntraTarget};
use crate::intra_modes::{self, Direction, LumaList, Mode, ModeContext};
use crate::mv_refs::{self, BlockMotion, Candidates, InterMode, ListConfig, SKIP_MODE};
use crate::partition::PartitionWriter;
use crate::pre_tx_skip::{LumaBlock, PreTxSkip};
use crate::prediction::InterpFilter;
use crate::transform::{Dequantizer, PlaneQindex, Qindices, QuantDeltas};
use crate::tx_partition::{self, Layout};
use crate::tx_size::TxSize;
use crate::tx_type::{TxSetMode, TxType};
use crate::{coeff, motion, prediction, transform};
use nc_common::decode::Result;
use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::intra_prediction::av2::Av2IntraEdges;
use nc_dsp::{BlockSize, Dsp};
use nc_encoder::symbol_cost::symbol_bits;
use nc_encoder::{DebugRng, IntraModeSet, residual_quantises_to_zero, with_neighbours};

/// `intra_inter` CDFs by context (AVM `default_intra_inter_cdf`).
pub(crate) const IS_INTER: [[u16; 2]; 4] = [[31246, 0], [18387, 0], [22313, 0], [4972, 0]];
/// `skip_txfm` CDFs by context (AVM `default_skip_txfm_cdfs`): the
/// skipped neighbours, plus [`SKIP_MODE_SKIP_CONTEXTS`] in a skip-mode block.
pub(crate) const SKIP: [[u16; 2]; 6] = [
    [6903, 0],
    [18452, 0],
    [28170, 0],
    [7156, 0],
    [20402, 0],
    [29448, 0],
];
/// `skip_txfm` context offset of a skip-mode block (AVM
/// `av2_get_skip_txfm_context`: `SKIP_CONTEXTS >> 1`).
pub(crate) const SKIP_MODE_SKIP_CONTEXTS: usize = 3;
/// Skip mode predicts with `MULTITAP_SHARP` whatever the frame's filter
/// (dav2d `decode_b`; AVM `av2_is_interp_needed` is false and
/// `read_mb_interp_filter` keeps the skip-mode default).
pub(crate) const SKIP_MODE_FILTER: InterpFilter = InterpFilter::Sharp;
/// Lossless inter blocks code one `lossless_tx_size` symbol (always 4×4);
/// AVM's defaults are one half for every size group.
pub(crate) const LOSSLESS_TX_SIZE: [u16; 2] = [16384, 0];
/// The `lossless_tx_size` symbol selecting 4×4 transforms.
pub(crate) const LOSSLESS_TX4_SYMBOL: usize = 0;
/// Largest `size_group_lookup` value (AVM: `min(3, log2(side / 4))`).
const MAX_SIZE_GROUP: u32 = 3;
/// The luma plane's index in per-plane arrays.
const LUMA: usize = 0;
/// The chroma planes' indices in per-plane arrays.
const CHROMA: [usize; 2] = [1, 2];
/// Side (log2) of one 4×4 mode-info unit.
const MI_LOG2: u32 = 2;
/// Most samples of one plane block (64×64 luma).
const MAX_PLANE_AREA: usize = 64 * 64;
/// Luma samples per unit of the skip estimate threshold (a 16×16 block).
const SKIP_ESTIMATE_AREA: f64 = 256.0;
/// Smallest transform side the texture-aware zero test runs on: below it
/// the exact SSE test settles most zero transforms and a forward transform
/// costs little more than the test (counted per P frame, `crowd_mot17_09`
/// 720p q181: the test catches 161 of 354 all-zero 16x16, 84 of 211 32x32
/// and 54 of 102 64x64 transforms at allowance 1, skipping no coded one).
const TEXTURE_ZERO_MIN_SIDE: usize = 16;
/// Coefficient contexts of a transform with no coded neighbours.
const EMPTY_CONTEXT: [u8; 16] = [0; 16];
/// Chance (of 256) that a random-decision block drops its residual.
const RANDOM_SKIP_CHANCE: u64 = 64;
/// Chance (of 256) that a random-decision block codes skip mode, where the
/// frame allows it.
const RANDOM_SKIP_MODE_CHANCE: u64 = 80;
/// Eighth-sample vector components that are whole luma samples.
const LUMA_WHOLE_SAMPLE: i32 = 8;
/// Eighth-sample vector components that are whole samples in 4:2:0 chroma
/// (and so in luma).
const CHROMA_WHOLE_SAMPLE: i32 = 16;
/// The chroma planes.
const CHROMA_PLANES: [PlaneIndex; 2] = [PlaneIndex::U, PlaneIndex::V];
/// Chance (of 256) that a random-decision block splits its luma transform.
const RANDOM_SPLIT_CHANCE: u64 = 128;

/// `lossless_tx_size` model group of a block (AVM `size_group_lookup`).
pub(crate) fn lossless_size_group(block_log2: u32) -> usize {
    (block_log2 - MI_LOG2).min(MAX_SIZE_GROUP) as usize
}

/// How the luma transform layout of a lossy block is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SplitPolicy {
    /// One transform of the block size.
    Never,
    /// The cheaper of one transform and the 4-way split, by rate and distortion.
    RateDistortion,
    /// Always the 4-way split.
    Always,
}

impl SplitPolicy {
    /// The luma splits to try for a lossy block.
    fn splits(self) -> [Option<bool>; 2] {
        match self {
            Self::Never => [Some(false), None],
            Self::RateDistortion => [Some(false), Some(true)],
            Self::Always => [Some(true), None],
        }
    }
}

/// The key of a cached luma prediction ([`BlockCoder::cache_luma`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CachedLuma {
    position: (usize, usize),
    size: usize,
    mv: (i32, i32),
    filter: InterpFilter,
}

/// How RD-trial rate estimates use the tile's adaptive models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EstimateModels {
    /// Adapt a scratch copy symbol by symbol, as coding would.
    Adapting,
    /// Price with the live models as they stand (cheaper).
    Frozen,
}

/// Frame-level choices of the block coder.
#[derive(Debug, Clone, Copy)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent frame-level switches"
)]
pub(crate) struct CoderOptions {
    /// Update CDFs within the tile.
    pub adaptive_cdf: bool,
    /// Luma transform layout choice.
    pub split: SplitPolicy,
    /// How rate estimates in RD trials use the tile's models.
    pub estimates: EstimateModels,
    /// Luma transform layout choice of intra blocks.
    pub intra_split: SplitPolicy,
    /// Whether an intra block's winning coding is retried with `ADST_ADST`
    /// where the frame's transform set allows it.
    pub intra_type_retry: bool,
    /// Lagrange multiplier per squared quantizer pixel step for residual RD.
    pub lambda_per_step_squared: f64,
    /// Quantizer rounding offsets in 1/128 step ([`transform::quantize`]),
    /// by plane kind and transform size.
    pub rounding: PlaneRounding,
    /// The frame's quantizer offsets (header `quant_deltas`).
    pub quant_deltas: QuantDeltas,
    /// The frame codes superblock quantizer deltas (`delta_q_present`,
    /// [`crate::delta_q`]): blocks follow [`BlockCoder::set_block_offset`].
    pub delta_q: bool,
    /// The frame's interpolation filter mode (inter pictures).
    pub filter: FrameFilter,
    /// Inter-picture blocks may code skip mode (`skip_mode_flag`), each
    /// ranking this many leading entries of its skip-mode list; `None`
    /// when the frame does not enable it.
    pub skip_mode: Option<usize>,
    /// How a switchable frame's blocks choose their filter.
    pub filter_search: FilterSearch,
    /// How intra modes and layouts are searched.
    pub intra: IntraSearch,
    /// Key-frame planning costs chroma as well as luma.
    pub plan_chroma: bool,
    /// An inter block tries the split luma layout only when its unsplit
    /// luma coding has at least this many nonzero levels (0: always).
    pub inter_split_min_nonzero: usize,
    /// Inter blocks whose prediction error (all planes) is below this many
    /// λ per [`SKIP_ESTIMATE_AREA`] luma samples are coded as skipped
    /// without residual trials, or `None` to always run them
    /// (`av2.skip_estimate`).
    pub skip_estimate: Option<f32>,
    /// Chroma squared error in luma units in the inter block decisions,
    /// chroma trimming and the chroma trellis ([`crate::chroma_weight`]).
    pub chroma_weight: f64,
    /// How CDEF strengths are chosen.
    pub cdef: CdefMode,
    /// The sequence enables CCSO: lossy frames code unit flags, chosen with
    /// the frame's filters after CDEF.
    pub ccso: bool,
    /// The frame's reduced transform set.
    pub tx_set: TxSetMode,
    /// How the frame codes lossy luma levels: hidden DC parities
    /// (`allow_parity_hiding`, [`coeff::parity`]) or trellis-coded
    /// quantisation (`tcq_mode`, [`coeff::tcq`]).
    pub levels: LevelCoding,
    /// The TCQ search's path pruning margin in bits (`av2.tcq_prune_bits`),
    /// or `None` for the exact search.
    pub tcq_prune_bits: Option<f32>,
    /// The inter luma transform-type search.
    pub inter_tx: InterTxSearch,
    /// Trailing-level trimming ([`Trim`]), or `None` for none. Transforms
    /// the trellis quantises are not trimmed.
    pub trim: Option<TrimOptions>,
    /// Trellis quantisation of the coded transforms (`quant.trellis`), or
    /// `None` for none.
    pub trellis: Option<TrellisOptions>,
    /// Inter luma pre-transform skip, or `None` for none.
    pub pre_tx_skip: Option<PreTxSkip>,
    /// P-frame texture-aware zero test of large `DCT_DCT` transforms, at
    /// this AC energy allowance, or `None` for none.
    pub zero_test_texture_scale: Option<f32>,
    /// How inter RD trials measure each coded plane's squared error.
    pub trial_distortion: TrialDistortion,
    /// The sequence's motion-vector reference lists.
    pub reference_lists: ListConfig,
}

/// How inter RD trials measure a coded plane's squared error
/// (`av2.trial_distortion`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrialDistortion {
    /// On the plane's reconstruction.
    Reconstructed,
    /// From the coefficients for transforms up to 32x32; 64x64 transforms
    /// are reconstructed.
    EstimatedBelow64,
    /// From the transform coefficients ([`Output::Estimate`]); only the
    /// coding that is kept is reconstructed.
    Estimated,
}

/// The frame's trellis quantisation (`quant.*` engine params,
/// `docs/handoff/trellis.md`): the coded transforms' levels are chosen
/// together by distortion plus λ times their bits under the tile's current
/// probabilities ([`TrellisQuantizer`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrellisOptions {
    /// Multiple of the mode-decision λ.
    pub lambda_scale: f32,
    /// Further multiple for chroma, or `None` to leave chroma to the plain
    /// quantiser (and trimming).
    pub chroma_scale: Option<f32>,
    /// Survivors per scan position.
    pub max_states: usize,
}

/// The trellis of one frame: λ (sample SSE per bit) per plane kind.
#[derive(Debug, Clone, Copy)]
struct TrellisLambdas {
    luma: f32,
    /// `None` leaves chroma to the plain quantiser.
    chroma: Option<f32>,
    max_states: usize,
}

/// Quantizer rounding offsets by transform size, for luma and for chroma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneRounding {
    pub luma: [transform::Rounding; TxSize::ALL.len()],
    pub chroma: [transform::Rounding; TxSize::ALL.len()],
}

impl PlaneRounding {
    /// The offsets of `plane`'s transforms of size `tx`.
    fn of(&self, plane: PlaneIndex, tx: TxSize) -> transform::Rounding {
        match plane {
            PlaneIndex::Y => self.luma[tx.index()],
            PlaneIndex::U | PlaneIndex::V => self.chroma[tx.index()],
        }
    }
}

impl TrellisLambdas {
    /// The λ of `plane`, or `None` when it is not trellised.
    fn of(self, plane: PlaneIndex) -> Option<f32> {
        match plane {
            PlaneIndex::Y => Some(self.luma),
            PlaneIndex::U | PlaneIndex::V => self.chroma,
        }
    }
}

/// How a plane coding quantises its transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Quantisation {
    /// The dead-zone quantiser and trimming: the rate-distortion trials.
    Plain,
    /// The trellis where the frame has one: the block mode decision chose,
    /// coded again before it is written (as HEVC's winner-only trellis).
    /// Trials stay cheap; most P blocks end up skipped and never pay for a
    /// trellis.
    Final,
}

/// Whether [`BlockCoder::write_inter`] writes the block's motion signalling.
///
/// Speed heuristic: the residual trials of one inter block (skipped, each
/// luma layout, the retyped winner) price it `Omitted`. Every one of them
/// codes the same inter mode, DRL index, vector and filter after the skip
/// flag, so those bits cancel in their comparison; only the rate of the
/// motion symbols themselves is skipped (their models are distinct from the
/// skip flag's and the coefficients', so even adaptive estimates price the
/// rest alike). Trials comparing different signallings (skip mode against a
/// single reference) write it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MotionSyntax {
    Written,
    Omitted,
}

/// What [`Quantizer::code`] does with a transform besides choosing its
/// levels.
enum Output<'a, 'b> {
    /// Reconstruct its samples (prediction plus dequantized residual).
    Reconstruct(&'a mut PlaneViewMut<'b>),
    /// Leave the samples and add the change in squared error the levels buy
    /// over the prediction, estimated from the coefficients (Parseval,
    /// [`transform::coefficient_error_change`]; `av2.trial_distortion`).
    Estimate(&'a mut f64),
}

/// What the trellis prices one transform's levels with: the tile's writer
/// (its current probabilities) and the contexts and transform set the
/// transform will be coded with.
struct RateSource<'a> {
    writer: &'a mut Writer,
    context: coeff::Context,
    set: TxSetMode,
}

/// What an inter coding knows of one transform's forward transform. The
/// final coding of the winning block ([`Quantisation::Final`]) requantizes
/// what its trial kept instead of forming the residual and transforming
/// again: the same source, prediction and type give the same coefficients.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Forward {
    /// Not transformed yet.
    #[default]
    Missing,
    /// Settled as all-zero by the zero test, for this type.
    Zero(TxType),
    /// Transformed with this type; the coefficients are kept.
    Transformed(TxType),
}

impl Forward {
    /// The type this transform was settled or transformed with.
    fn tx_type(self) -> Option<TxType> {
        match self {
            Self::Missing => None,
            Self::Zero(tx_type) | Self::Transformed(tx_type) => Some(tx_type),
        }
    }
}

/// One transform's place in a coding's forward cache: its [`Forward`]
/// state and its coefficient storage ([`TxSize::coded_len`] values).
struct ForwardSlot<'a> {
    state: &'a mut Forward,
    coefficients: &'a mut [i32],
}

/// The per-block filter choice of a switchable frame: each inter block's
/// luma is predicted with `EIGHTTAP_REGULAR` and then with each candidate,
/// and the filter with the least prediction error plus the psychovisual
/// term (`decision.psy_strength`, [`nc_encoder::psy_penalty`]) plus
/// λ·symbol bits wins; the residual trials run once, on the winner.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterSearch {
    /// The filters tried besides `EIGHTTAP_REGULAR`, in order.
    pub candidates: &'static [InterpFilter],
    /// The largest fraction of the regular prediction's luma error a
    /// candidate is assumed to remove. A candidate whose extra symbol cost
    /// (λ·bits) exceeds that gain is not predicted. At 1 this only skips
    /// candidates that cannot win; lower values skip blocks whose
    /// prediction error is small next to the rate difference.
    pub max_gain: f64,
    /// Strength of the psychovisual term (0: prediction error alone).
    pub psy_strength: f32,
}

/// The luma transform types an inter block's winning layout retries per
/// transform after its `DCT_DCT` coding, where the frame's set allows them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct InterTxSearch {
    /// The types tried besides `DCT_DCT` (empty: no search).
    pub candidates: &'static [TxType],
    /// Transforms whose `DCT_DCT` coding has fewer nonzero levels keep it.
    pub min_nonzero: usize,
}

/// The intra mode and transform-layout search of a block.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IntraSearch {
    /// The luma modes ranked.
    pub modes: LumaCandidates,
    /// Intra mode pairs, best by prediction SATD, given full transform RD.
    pub rd_candidates: usize,
    /// How the best direction's angle deltas are searched.
    pub delta_search: DeltaSearch,
    /// Leading entries of each chroma list ranked (13: all).
    pub chroma_candidates: usize,
    /// Every RD candidate tries every transform layout (else candidates
    /// after the first reuse the first one's winning layout).
    pub layouts_per_candidate: bool,
    /// Stop the RD trials at the first coding without residual.
    pub zero_residual_exit: bool,
}

/// The luma modes an intra block ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LumaCandidates {
    /// The engine's shared set ([`IntraModeSet`]) plus the above and left
    /// neighbours' modes; DC only means no search.
    Shared(IntraModeSet),
    /// Every non-directional mode and nominal direction, then the best
    /// direction's angle deltas ([`IntraSearch::delta_search`]).
    AllDirections,
}

/// How the coder chooses the frame's CDEF strengths.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CdefMode {
    /// The quantiser model's single strength set, each plane kept where it
    /// lowers the error sampled on every this many rows of 8x8 units
    /// ([`cdef::choose_and_filter`]).
    Model(usize),
    /// The strength search with per-filter-block indices
    /// ([`cdef::search_and_filter`]).
    Search(cdef::SearchOptions),
}

/// How the angle deltas of the best-ranked direction are searched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeltaSearch {
    /// Zero deltas only.
    Off,
    /// ±2 first, then ±1 and ±3 beside the better of −2, 0, +2.
    CoarseToFine,
    /// All six nonzero deltas.
    Exhaustive,
}

/// One coding of a block, boxed: trials pass codings around by value
/// (pool, best so far, chosen), and moving the ~400-byte struct itself was
/// a memcpy each time (about 1% of Jetson time in `encode_inter`).
type Coded = Box<BlockCoding>;

/// One coding of a block: per plane, the levels of each transform
/// (concatenated in coding order, [`TxSize::coded_len`] each), the type of
/// each luma transform (chroma is DCT-only) and the reconstructed samples
/// (raster at the plane block side).
#[derive(Debug, Default)]
struct BlockCoding {
    levels: [Vec<i32>; 3],
    /// Per plane, each transform's [`LevelStats`], kept in step with
    /// `levels` so that rate estimates and the write skip their passes.
    stats: [Vec<LevelStats>; 3],
    luma_types: Vec<TxType>,
    pixels: [Vec<u8>; 3],
    /// Per plane, each transform's forward transform as the inter trials
    /// left it, and the coefficients (laid out as `levels`).
    forward: [Vec<Forward>; 3],
    coefficients: [Vec<i32>; 3],
    /// Per plane, while `pixels` does not hold the plane's reconstruction:
    /// the change in squared error its levels buy over the prediction,
    /// estimated from the coefficients (`av2.trial_distortion`,
    /// [`Quantizer::estimate`]). Only kept codings are reconstructed
    /// ([`BlockCoder::reconstruct_estimated`]).
    estimated: [Option<f64>; 3],
    distortion: u64,
    /// The layout the buffers are sized for.
    layout: Option<Layout>,
}

impl BlockCoding {
    /// Size the buffers for `layout`. Old contents are kept: coding a layout
    /// overwrites every level and sample, so clearing them would be wasted.
    fn reset(&mut self, layout: Layout) {
        for plane in 0..3 {
            let tx = layout.tx(plane);
            let count = 1 << (2 * (layout.plane_log2(plane) - tx.log2()));
            self.levels[plane].resize(count * tx.coded_len(), 0);
            self.stats[plane].clear();
            self.stats[plane].resize(count, LevelStats::EMPTY);
            self.pixels[plane].resize(1 << (2 * layout.plane_log2(plane)), 0);
            self.coefficients[plane].resize(count * tx.coded_len(), 0);
            self.forward[plane].clear();
            self.forward[plane].resize(count, Forward::Missing);
        }
        let luma_count = 1 << (2 * (layout.plane_log2(LUMA) - layout.tx(LUMA).log2()));
        self.luma_types.clear();
        self.luma_types.resize(luma_count, TxType::DctDct);
        self.estimated = [None; 3];
        self.distortion = 0;
        self.layout = Some(layout);
    }

    fn is_zero(&self) -> bool {
        (0..self.stats.len()).all(|plane| self.plane_is_zero(plane))
    }

    fn plane_is_zero(&self, plane: usize) -> bool {
        self.stats[plane].iter().all(|stats| stats.eob() == 0)
    }

    fn tx_levels(&self, layout: Layout, plane: usize, index: usize) -> &[i32] {
        let len = layout.tx(plane).coded_len();
        &self.levels[plane][index * len..(index + 1) * len]
    }
}

/// The quantizer-dependent tools of one qindex: its per-plane quantizers
/// and the Lagrange multipliers of trimming, the trellis, parity hiding and
/// TCQ, which scale with its step ([`quantizer_settings`]).
#[derive(Debug, Clone, Copy)]
struct QuantizerSettings {
    qindex: PlaneQindex,
    trim: Option<Trim>,
    trellis: Option<TrellisLambdas>,
    parity_hiding: Option<f64>,
    tcq: Option<TcqSearch>,
}

/// The [`QuantizerSettings`] of a block coded at `qindex` in a frame with
/// `options`.
fn quantizer_settings(options: &CoderOptions, qindex: u8) -> QuantizerSettings {
    let lambda = options.lambda_per_step_squared * pixel_step(qindex) * pixel_step(qindex);
    let chroma_weight = options.chroma_weight;
    QuantizerSettings {
        qindex: PlaneQindex::new(qindex, options.quant_deltas),
        trim: options.trim.map(|trim| Trim {
            lambda,
            chroma_weight,
            options: trim,
        }),
        trellis: options
            .trellis
            .map(|trellis| trellis_lambdas(trellis, lambda, chroma_weight)),
        parity_hiding: (options.levels == LevelCoding::ParityHiding).then_some(lambda),
        tcq: (options.levels == LevelCoding::Tcq).then(|| TcqSearch {
            lambda: lambda as f32 * options.trellis.map_or(1.0, |t| t.lambda_scale),
            prune_bits: options.tcq_prune_bits,
        }),
    }
}

/// Forward and inverse transform of one transform block, with its scratch.
#[derive(Debug)]
struct Quantizer {
    dsp: Dsp,
    /// The frame's base qindex, which selects the coefficient models.
    frame_qindex: u8,
    /// The quantizer of the block being coded, per plane.
    qindex: PlaneQindex,
    rounding: PlaneRounding,
    trim: Option<Trim>,
    trellis: Option<TrellisLambdas>,
    /// The trellis's search buffers.
    search: TrellisQuantizer,
    /// Settle provably all-zero `DCT_DCT` transforms from the residual's
    /// SSE before transforming ([`transform::dct_quantizes_to_zero`]). On
    /// for P frames, where most transforms quantize to zero: `crowd_mot17_09`
    /// 720p q180 P −5.1% instructions; on key frames the extra SSE pass
    /// costs more than it saves (+0.4%).
    zero_test: bool,
    /// The frame hides luma DC parities: luma levels are made to conform
    /// ([`coeff::parity::conform`]) at this λ (sample SSE per bit).
    parity_hiding: Option<f64>,
    /// The frame trellis-codes lossy luma: every luma transform's levels
    /// are chosen by the TCQ search ([`TcqQuantizer`]) with these
    /// parameters.
    tcq: Option<TcqSearch>,
    /// The TCQ search's buffers.
    tcq_search: TcqQuantizer,
    /// The texture-aware zero test ([`nc_encoder::residual_quantises_to_zero`]
    /// at this AC energy allowance) on `DCT_DCT` transforms of at least
    /// [`TEXTURE_ZERO_MIN_SIDE`] that the exact test could not settle, or
    /// `None` for none (`av2.zero_test_texture_scale`).
    texture_zero_scale: Option<f32>,
    residual: Vec<i16>,
    /// Unquantized coefficients of a transform coded without a
    /// [`ForwardSlot`].
    coefficients: Vec<i32>,
    /// Dequantized coefficients for the reconstruction.
    dequantized: Vec<i32>,
    inverse: Vec<i32>,
}

impl Quantizer {
    fn new(
        frame_qindex: u8,
        rounding: PlaneRounding,
        settings: QuantizerSettings,
        zero_test: bool,
        texture_zero_scale: Option<f32>,
    ) -> Self {
        let mut quantizer = Self {
            dsp: Dsp::new(),
            frame_qindex,
            qindex: settings.qindex,
            rounding,
            trim: None,
            trellis: None,
            search: TrellisQuantizer::default(),
            zero_test,
            parity_hiding: None,
            tcq: None,
            tcq_search: TcqQuantizer::default(),
            texture_zero_scale,
            residual: vec![0; MAX_PLANE_AREA],
            coefficients: vec![0; TxSize::Tx64.coded_len()],
            dequantized: vec![0; TxSize::Tx64.coded_len()],
            inverse: vec![0; MAX_PLANE_AREA],
        };
        quantizer.configure(settings);
        quantizer
    }

    /// Quantize at `settings` from now on (a block's own qindex).
    fn configure(&mut self, settings: QuantizerSettings) {
        let QuantizerSettings {
            qindex,
            trim,
            trellis,
            parity_hiding,
            tcq,
        } = settings;
        let lossy = qindex.luma.ac != 0;
        self.qindex = qindex;
        self.trim = trim.filter(|_| lossy);
        self.trellis = trellis.filter(|_| lossy);
        self.parity_hiding = parity_hiding;
        self.tcq = tcq.filter(|_| lossy);
    }

    /// Quantize the difference between `source` (`plane` at `position`) and
    /// `predicted` for one transform of type `tx_type` into `levels`, and
    /// reconstruct it or estimate its distortion as `output` says. With a
    /// `rate` source the frame's trellis chooses the levels of a trellised
    /// plane; otherwise the plain quantiser does, followed by trimming
    /// where it is on. A `forward` slot keeps the forward transform, or
    /// supplies it when it already holds this type's.
    #[allow(
        clippy::too_many_lines,
        reason = "one transform's quantization pipeline, in order"
    )]
    fn code(
        &mut self,
        source: (&Picture, PlaneIndex, (usize, usize)),
        (tx, tx_type): (TxSize, TxType),
        prediction: &PlaneView<'_>,
        (levels, output): (&mut [i32], Output<'_, '_>),
        rate: Option<RateSource<'_>>,
        forward: Option<ForwardSlot<'_>>,
    ) -> Result<LevelStats> {
        let (picture, plane, (x, y)) = source;
        let qindex = self.qindex.of(plane);
        let rounding = self.rounding.of(plane, tx);
        // Every coding of TCQ luma, trials included, is a TCQ search: mode
        // decision then weighs the levels that will be coded (trials of
        // plain levels cost 0.2-0.9 dB luma at equal rate).
        let tcq = self.tcq_search_of(plane);
        let dequantizer = self.dequantizer(plane);
        let side = tx.width();
        let block = BlockSize::new(side, side).expect("square transform sizes are block sizes");
        let source = picture
            .plane(plane)
            .view()
            .sub_view(x, y, side, side)
            .expect("transforms lie inside the picture");
        let known = forward
            .as_ref()
            .map(|slot| *slot.state)
            .filter(|state| state.tx_type() == Some(tx_type));
        let zero = match known {
            Some(state) => state == Forward::Zero(tx_type),
            None => {
                self.zero_test
                    && tx_type == TxType::DctDct
                    && qindex.ac != 0
                    && self.settles_zero(tx, (qindex, rounding), &source, prediction)
            }
        };
        let (state, coefficients) = match forward {
            Some(slot) => (Some(slot.state), slot.coefficients),
            None => (None, &mut self.coefficients[..tx.coded_len()]),
        };
        if known.is_none() && !zero {
            let residual = &mut self.residual[..side * side];
            self.dsp
                .residual(block, &source, prediction, residual, side);
            transform::forward_samples(
                &self.dsp,
                (tx, tx_type),
                residual,
                qindex.ac,
                coefficients,
            )?;
        }
        if let Some(state) = state {
            *state = if zero {
                Forward::Zero(tx_type)
            } else {
                Forward::Transformed(tx_type)
            };
        }
        if zero {
            levels.fill(0);
            if let Output::Reconstruct(reconstructed) = output {
                reconstructed.copy_from(prediction);
            }
            return Ok(LevelStats::EMPTY);
        }
        let coded = transform::quantize(&self.dsp, tx, coefficients, (qindex, rounding), levels);
        // One raster pass finds the plain end of block; the trellis and
        // trimming start from it and return theirs.
        let plain_eob = if coded {
            coeff::end_of_block(levels, tx)
        } else {
            0
        };
        // TCQ luma takes the rate source; the trellis, trimming and parity
        // hiding see plain levels only.
        let (tcq_rate, rate) = match tcq {
            Some(_) => (rate, None),
            None => (None, rate),
        };
        let trellis = self
            .trellis
            .and_then(|t| Some((t.of(plane)?, t.max_states)))
            .zip(rate);
        let eob = if let Some((search, rate)) = tcq.zip(tcq_rate) {
            let block = CodedBlock {
                tx,
                context: rate.context,
                qindex: self.frame_qindex,
                steps: qindex,
                kind: coeff::Kind {
                    set: rate.set,
                    tx_type,
                    levels: LevelCoding::Tcq,
                },
            };
            self.tcq_search.quantize(
                rate.writer,
                block,
                search,
                coefficients,
                (levels, plain_eob),
            )
        } else if tcq.is_some() {
            // Every TCQ luma coding has a rate source (`prices_levels`);
            // plain levels would still decode under TCQ.
            plain_eob
        } else if let Some((lambda_states, rate)) = trellis {
            let block = CodedBlock {
                tx,
                context: rate.context,
                qindex: self.frame_qindex,
                steps: qindex,
                kind: coeff::Kind {
                    set: rate.set,
                    tx_type,
                    levels: if self.parity_hiding.is_some() {
                        LevelCoding::ParityHiding
                    } else {
                        LevelCoding::Plain
                    },
                },
            };
            let eob = self.search.quantize(
                rate.writer,
                block,
                lambda_states,
                coefficients,
                (levels, plain_eob),
            );
            conform_parity(
                self.parity_hiding,
                (plane, qindex),
                (tx, tx_type),
                (levels, coefficients),
                eob,
            )
        } else {
            let eob = match self.trim {
                Some(trim) => {
                    let luma = plane == PlaneIndex::Y;
                    trim.apply(luma, tx, qindex, coefficients, (levels, plain_eob))
                }
                None => plain_eob,
            };
            conform_parity(
                self.parity_hiding,
                (plane, qindex),
                (tx, tx_type),
                (levels, coefficients),
                eob,
            )
        };
        let stats = LevelStats::quantized(levels, tx, eob);
        match output {
            Output::Reconstruct(reconstructed) => {
                self.reconstruct(
                    (plane, dequantizer),
                    (tx, tx_type),
                    (levels, stats),
                    prediction,
                    reconstructed,
                )?;
            }
            Output::Estimate(change) if stats.eob() != 0 => {
                *change += match dequantizer {
                    Dequantizer::Plain => {
                        transform::estimated_error_change(tx, levels, qindex, coefficients)
                    }
                    Dequantizer::Tcq => transform::estimated_error_from_change(
                        tx,
                        coeff::tcq::error_change(tx, (levels, stats.eob()), qindex, coefficients),
                    ),
                };
            }
            Output::Estimate(_) => {}
        }
        Ok(stats)
    }

    /// The TCQ search's parameters when `plane`'s transforms are
    /// trellis-coded.
    fn tcq_search_of(&self, plane: PlaneIndex) -> Option<TcqSearch> {
        self.tcq.filter(|_| plane == PlaneIndex::Y)
    }

    /// How `plane`'s levels dequantize: every luma level of a TCQ frame is
    /// a TCQ level.
    fn dequantizer(&self, plane: PlaneIndex) -> Dequantizer {
        if self.tcq_search_of(plane).is_some() {
            Dequantizer::Tcq
        } else {
            Dequantizer::Plain
        }
    }

    /// Reconstruct one transform of `plane` coded as `levels` (with their
    /// `stats`) over `prediction` into `reconstructed`.
    fn reconstruct(
        &mut self,
        (plane, dequantizer): (PlaneIndex, Dequantizer),
        (tx, tx_type): (TxSize, TxType),
        (levels, stats): (&[i32], LevelStats),
        prediction: &PlaneView<'_>,
        reconstructed: &mut PlaneViewMut<'_>,
    ) -> Result<()> {
        if stats.eob() == 0 {
            reconstructed.copy_from(prediction);
            return Ok(());
        }
        let side = tx.width();
        let qindex = self.qindex.of(plane);
        if let Some(residual) =
            transform::dc_only_residual((tx, tx_type), (levels, stats.eob(), dequantizer), qindex)
        {
            for row in 0..side {
                add_constant_residual(prediction.row(row), residual, reconstructed.row_mut(row));
            }
            return Ok(());
        }
        let inverse = &mut self.inverse[..side * side];
        transform::reconstruct_into(
            &self.dsp,
            (tx, tx_type),
            (levels, stats.eob(), dequantizer),
            qindex,
            &mut self.dequantized,
            inverse,
        )?;
        for (row, residual) in inverse.chunks_exact(side).enumerate() {
            add_residual(prediction.row(row), residual, reconstructed.row_mut(row));
        }
        Ok(())
    }

    /// Whether the residual between `source` and `prediction` provably
    /// quantizes to zero under a `DCT_DCT` transform of size `tx`: one SSE
    /// pass, then the texture-aware test on large transforms. Either settles
    /// the transform without the residual, forward transform and quantizer.
    fn settles_zero(
        &self,
        tx: TxSize,
        (qindex, rounding): (Qindices, transform::Rounding),
        source: &PlaneView<'_>,
        prediction: &PlaneView<'_>,
    ) -> bool {
        let side = tx.width();
        let block = BlockSize::new(side, side).expect("square transform sizes are block sizes");
        let sse = self.dsp.variance(block, source, prediction).sse;
        if transform::dct_quantizes_to_zero(tx, qindex, rounding, u64::from(sse)) {
            return true;
        }
        self.texture_zero_scale
            .filter(|_| side >= TEXTURE_ZERO_MIN_SIDE)
            .is_some_and(|scale| {
                residual_quantises_to_zero(
                    &self.dsp,
                    side,
                    source,
                    prediction,
                    crate::pre_tx_skip::zero_limits(qindex, rounding),
                    scale,
                )
            })
    }
}

/// Encoder state of one tile.
#[derive(Debug)]
pub(crate) struct BlockCoder {
    /// The frame's base qindex (lossless test, coefficient models, CDEF).
    qindex: u8,
    /// The qindex of the block being coded (its quantizers and λ): the
    /// frame's, or its superblock's with [`CoderOptions::delta_q`].
    block_qindex: u8,
    /// The superblock quantizer deltas' state ([`crate::delta_q`]).
    superblock_qindex: SuperblockQindex,
    key: bool,
    options: CoderOptions,
    /// Chroma squared error in luma units for the block decisions
    /// ([`CoderOptions::chroma_weight`]).
    chroma_weight: f64,
    writer: Writer,
    /// The chroma coefficient rate of the current inter block's trials, in
    /// 1/512 bit, once one trial has priced it ([`Self::write_coefficients`]).
    inter_chroma_q9: Option<u32>,
    partitions: PartitionWriter,
    state: BlockState,
    quantizer: Quantizer,
    /// Inter prediction of the current block, per plane.
    prediction: [Vec<u8>; 3],
    /// A skip-mode candidate's prediction, per plane (swapped into
    /// `prediction` when it wins).
    skip_prediction: [Vec<u8>; 3],
    /// The second vector's prediction of a ranked skip-mode candidate
    /// ([`Self::predict_skip_mode`]).
    skip_second: Vec<u8>,
    /// Exact compound prediction buffers.
    compound: crate::prediction::CompoundScratch,
    /// The engine's exact luma prediction of the block it is about to code
    /// ([`Self::cache_luma`]), reused instead of predicting it again.
    cached_luma: Option<CachedLuma>,
    cached_luma_samples: Vec<u8>,
    /// One filter candidate's luma prediction ([`Self::choose_filter`]).
    filter_trial: Vec<u8>,
    /// Fixed probabilities of the per-block filter symbol.
    filter_symbols: FilterSymbols,
    /// Prediction and reconstruction of the current transform.
    tx_prediction: Vec<u8>,
    /// One intra mode candidate's prediction, reused across candidates.
    mode_trial: Vec<u8>,
    /// One inter transform-type candidate's levels, and the best type's
    /// reconstruction ([`Self::refine_inter_types`]).
    type_trial_levels: Vec<i32>,
    type_best_pixels: Vec<u8>,
    /// Intra edge arrays, reused across predictions.
    edges: Av2IntraEdges,
    tx_reconstruction: Vec<u8>,
    /// Levels of one ranked intra candidate's transform.
    rank_levels: Vec<i32>,
    /// Nonzero luma levels of the most recent inter coding (split gate).
    last_luma_nonzero: usize,
    /// Reusable codings, so trials do not allocate ([`Self::take_coded`]).
    pool: Vec<Coded>,
    /// Test-only random syntax choices (`debug.random_decisions`).
    random: Option<DebugRng>,
    /// Each filter block's CDEF strength set; coded filter blocks hold a
    /// placeholder until [`Self::cdef`] chooses.
    cdef_indices: StrengthIndices,
    /// The frame's CDEF parameters, once [`Self::cdef`] has chosen them.
    cdef_params: CdefParams,
    /// Each CCSO unit's flags, once [`Self::ccso`] has chosen them.
    ccso_flags: UnitFlags,
    /// The frame's CCSO parameters, once [`Self::ccso`] has chosen them.
    ccso_params: CcsoParams,
    /// The symbols deferred until the filters are chosen, in coding order.
    deferred: Vec<DeferredSymbol>,
}

/// A symbol the block coder defers until the frame's filters are chosen.
#[derive(Debug, Clone, Copy)]
enum DeferredSymbol {
    /// The CDEF strength index of the filter block at this luma sample.
    Cdef((usize, usize)),
    /// The CCSO flags of the unit at this luma sample.
    Ccso((usize, usize)),
}

impl BlockCoder {
    /// Start a tile of `width` x `height` at `qindex`. An adapting tile
    /// starts from `inherited` models (cross-frame context) or the defaults.
    /// # Errors
    /// Rejects unsupported dimensions.
    pub fn new(
        (width, height): (usize, usize),
        qindex: u8,
        key: bool,
        options: &CoderOptions,
        inherited: Option<&TileCdfs>,
        random: Option<DebugRng>,
    ) -> Result<Self> {
        let options = *options;
        let mut writer = Writer::new();
        if options.adaptive_cdf {
            match inherited {
                Some(models) => writer.adapt_from(models.clone()),
                None => writer.enable_adaptation(qindex),
            }
            // Rate estimates price TCQ's models from a shared view, so a TCQ
            // frame builds them before its first block.
            if options.levels == LevelCoding::Tcq
                && let Some(models) = writer.live_models()
            {
                models.tcq();
            }
            if options.estimates == EstimateModels::Frozen {
                writer.freeze_estimates();
            }
        }
        // Strength indices and CCSO flags are chosen after reconstruction:
        // defer them.
        if (matches!(options.cdef, CdefMode::Search(_)) || options.ccso) && qindex != 0 {
            writer.record();
        }
        let chroma_weight = options.chroma_weight;
        Ok(Self {
            qindex,
            block_qindex: qindex,
            superblock_qindex: SuperblockQindex::new(qindex, options.delta_q.then_some(0)),
            key,
            options,
            chroma_weight,
            writer,
            inter_chroma_q9: None,
            partitions: PartitionWriter::new(width, height)?,
            state: BlockState::new(width, height, options.reference_lists)?,
            quantizer: Quantizer::new(
                qindex,
                options.rounding,
                quantizer_settings(&options, qindex),
                !key,
                options.zero_test_texture_scale.filter(|_| !key),
            ),
            prediction: Default::default(),
            skip_prediction: Default::default(),
            skip_second: Vec::new(),
            compound: crate::prediction::CompoundScratch::default(),
            cached_luma: None,
            cached_luma_samples: Vec::with_capacity(MAX_PLANE_AREA),
            filter_trial: vec![0; MAX_PLANE_AREA],
            filter_symbols: FilterSymbols::new(),
            tx_prediction: vec![0; MAX_PLANE_AREA],
            mode_trial: vec![0; MAX_PLANE_AREA],
            type_trial_levels: vec![0; TxSize::Tx64.coded_len()],
            type_best_pixels: vec![0; MAX_PLANE_AREA],
            edges: Av2IntraEdges::new(),
            tx_reconstruction: vec![0; MAX_PLANE_AREA],
            rank_levels: Vec::new(),
            last_luma_nonzero: 0,
            pool: Vec::new(),
            random,
            cdef_indices: StrengthIndices::new(width, height),
            cdef_params: CdefParams::OFF,
            ccso_flags: UnitFlags::new(width, height),
            ccso_params: CcsoParams::OFF,
            deferred: Vec::new(),
        })
    }

    /// The motion candidates of the block of side `size` at `(x, y)`.
    pub fn candidates(&self, x: usize, y: usize, size: usize) -> Result<Candidates> {
        self.state.motion.candidates(x, y, size)
    }

    /// Approximate bits of the prediction-type syntax of a block:
    /// `intra_inter`, plus DC modes for an intra block.
    pub fn prediction_type_bits(&self, x: usize, y: usize, size: usize, is_inter: bool) -> f32 {
        let context = self.state.intra_inter_context(x, y, size);
        let mut bits = symbol_bits(&IS_INTER[context], usize::from(is_inter));
        if !is_inter {
            let (above, left) = self.state.mode_neighbours(x, y, size);
            bits += intra_modes::dc_bits(above, left);
        }
        bits
    }

    /// Bits of the cheapest inter leaf under the current probabilities,
    /// each symbol at its cheapest context: `is_inter` 1, `skip_txfm` 1
    /// (outside skip mode) and the cheapest reuse of a candidate vector
    /// ([`mv_refs::cheapest_reuse_bits`]). Every leaf pays at least this
    /// however it is predicted, so four children pay it four times.
    pub fn cheapest_leaf_bits(&self) -> f32 {
        fn cheapest(bits: impl Iterator<Item = f32>) -> f32 {
            bits.fold(f32::INFINITY, f32::min)
        }
        let writer = &self.writer;
        let is_inter = cheapest((0..IS_INTER.len()).map(|context| {
            symbol_bits(
                &writer.current(&IS_INTER[context], |m| &m.is_inter[context]),
                1,
            )
        }));
        let skip = cheapest((0..SKIP_MODE_SKIP_CONTEXTS).map(|context| {
            symbol_bits(
                &writer.current(&SKIP[context], |m| &m.block_skip[context]),
                1,
            )
        }));
        is_inter + skip + mv_refs::cheapest_reuse_bits(writer)
    }

    /// Approximate bits of the partition decision for a square node.
    pub fn partition_bits(&self, x: usize, y: usize, size: usize, split: bool) -> f32 {
        self.partitions.bits(x, y, size, split)
    }

    /// Start the superblock whose top-left sample is `(x, y)`.
    /// # Errors
    /// Rejects a position off the superblock grid.
    pub fn begin_superblock(&mut self, x: usize, y: usize) -> Result<()> {
        self.superblock_qindex.start_superblock();
        self.partitions.begin_superblock(x);
        self.state.motion.begin_superblock(x, y)
    }

    /// Code the next block `offset` native qindex steps from the frame's:
    /// the first block of a superblock sets the superblock's qindex (with
    /// [`CoderOptions::delta_q`]), later ones inherit it.
    pub fn set_block_offset(&mut self, offset: i16) {
        let qindex = self.superblock_qindex.block_qindex(self.qindex, offset);
        self.quantize_at(qindex);
    }

    /// Point the quantizers and λ at `qindex`.
    fn quantize_at(&mut self, qindex: u8) {
        if qindex != self.block_qindex {
            self.block_qindex = qindex;
            self.quantizer
                .configure(quantizer_settings(&self.options, qindex));
        }
    }

    /// The superblock quantizer delta of a block of side `size` whose
    /// `skip_txfm` is `skip`, after its CDEF index (AVM
    /// `read_delta_q_params`). Trials price it; [`Self::commit_delta_q`]
    /// records it once the block is final.
    fn write_delta_q(&mut self, size: usize, skip: bool) -> Result<()> {
        if self
            .superblock_qindex
            .codes_delta(size == crate::partition::SUPERBLOCK, skip)
        {
            self.superblock_qindex
                .write(&mut self.writer, self.block_qindex)?;
        }
        Ok(())
    }

    /// The block of side `size` (`skip`: skipped) is final: a superblock's
    /// first block that coded no delta keeps the inherited qindex, which
    /// AV2 deblocking then reads as the block's own.
    fn commit_delta_q(&mut self, size: usize, skip: bool) {
        let coded = self
            .superblock_qindex
            .codes_delta(size == crate::partition::SUPERBLOCK, skip);
        self.superblock_qindex.commit(self.block_qindex, coded);
        self.quantize_at(self.superblock_qindex.current());
    }

    /// Code the engine's partition decision for a square node.
    /// # Errors
    /// Rejects decisions the syntax cannot express.
    pub fn write_partition(&mut self, x: usize, y: usize, size: usize, split: bool) -> Result<()> {
        self.partitions.write(&mut self.writer, x, y, size, split)
    }

    /// Squared-error cost of one bit.
    /// How this frame codes a transform of type `tx_type`.
    fn coding_kind(&self, tx_type: TxType) -> coeff::Kind {
        coeff::Kind {
            set: self.options.tx_set,
            tx_type,
            levels: self.options.levels,
        }
    }

    /// λ of the block being coded (at its own qindex).
    pub fn lambda(&self) -> f64 {
        self.lambda_at(self.block_qindex)
    }

    /// λ of a block at `qindex`.
    fn lambda_at(&self, qindex: u8) -> f64 {
        let step = pixel_step(qindex);
        self.options.lambda_per_step_squared * step * step
    }

    fn lossless(&self) -> bool {
        self.qindex == 0
    }

    /// The transform layouts to try for a block of side `1 << block_log2`.
    fn layouts(&mut self, block_log2: u32) -> [Option<Layout>; 2] {
        self.layouts_by(block_log2, self.options.split)
    }

    /// The luma layouts an intra block tries ([`CoderOptions::intra_split`]).
    fn intra_layouts(&mut self, block_log2: u32) -> [Option<Layout>; 2] {
        self.layouts_by(block_log2, self.options.intra_split)
    }

    fn layouts_by(&mut self, block_log2: u32, policy: SplitPolicy) -> [Option<Layout>; 2] {
        if self.lossless() {
            return [Some(Layout::lossless(block_log2)), None];
        }
        let splits = match self.random.as_mut() {
            Some(random) => [Some(random.chance(RANDOM_SPLIT_CHANCE)), None],
            None => policy.splits(),
        };
        splits.map(|split| split.map(|split| Layout::lossy(block_log2, split)))
    }

    /// A recycled coding already sized for `layout`, or a new one. Taking a
    /// coding of another layout instead meant resizing (zero-filling) its
    /// buffers, and the next block of that layout then missed in turn
    /// (~1% of x86 instructions in `memset`); the pool instead keeps a few
    /// codings of each layout, a bounded set.
    fn take_coded(&mut self, layout: Layout) -> Coded {
        let same = self.pool.iter().rposition(|c| c.layout == Some(layout));
        let mut coded = match same {
            Some(index) => self.pool.swap_remove(index),
            None => Coded::default(),
        };
        coded.reset(layout);
        coded
    }

    fn recycle(&mut self, coded: Coded) {
        self.pool.push(coded);
    }

    /// Code an intra block of side `size` into `recon`, choosing its luma
    /// and chroma modes and transform layout by rate and distortion.
    /// # Errors
    /// Propagates syntax errors.
    pub fn encode_intra(
        &mut self,
        (x, y): (usize, usize),
        size: usize,
        source: &Picture,
        recon: &mut Picture,
    ) -> Result<()> {
        self.state.check_order(x, y)?;
        let block_log2 = size.ilog2();
        let (above, left) = self.state.mode_neighbours(x, y, size);
        let modes = ModeContext {
            above,
            left,
            block_log2,
        };
        let candidates = self.intra_candidates((x, y), size, modes, (source, recon));
        let mut best: Option<(f64, Coded, Layout, (Mode, Mode))> = None;
        let mut layouts = self.intra_layouts(block_log2);
        let chroma_layout = layouts[0].expect("at least one layout");
        // Chroma codings by chroma mode, shared by every luma candidate and layout.
        let mut chroma_codings: Vec<(Mode, Coded)> = Vec::new();
        'trials: for pair in candidates {
            let chroma_index =
                if let Some(i) = chroma_codings.iter().position(|(m, _)| *m == pair.1) {
                    i
                } else {
                    let coding =
                        self.code_intra_chroma(chroma_layout, (x, y), pair.1, (source, recon))?;
                    chroma_codings.push((pair.1, coding));
                    chroma_codings.len() - 1
                };
            if best.is_some() && !self.options.intra.layouts_per_candidate {
                // Later candidates reuse the first one's winning layout.
                layouts = [best.as_ref().map(|(_, _, layout, _)| *layout), None];
            }
            for layout in layouts.into_iter().flatten() {
                let mut coded = self.take_coded(layout);
                self.code_intra(
                    &mut coded,
                    layout,
                    (x, y),
                    (pair.0, TxType::DctDct),
                    (source, recon),
                    &chroma_codings[chroma_index].1,
                )?;
                let cost = self.trial_bits(x, y, size, |coder| {
                    coder.write_intra(&coded, layout, (x, y), modes, pair)
                })? * self.lambda()
                    + coded.distortion as f64;
                // The best-ranked prediction needs no residual: a split
                // layout or a worse-ranked mode would only add syntax.
                let settled = self.options.intra.zero_residual_exit && coded.is_zero();
                match &best {
                    Some((best_cost, ..)) if *best_cost <= cost => self.recycle(coded),
                    _ => {
                        if let Some((_, previous, ..)) = best.replace((cost, coded, layout, pair)) {
                            self.recycle(previous);
                        }
                    }
                }
                if settled {
                    break 'trials;
                }
            }
        }
        let (cost, coded, layout, (luma, chroma)) =
            best.expect("at least one candidate and layout");
        let chroma_coding = chroma_codings
            .iter()
            .find(|(mode, _)| *mode == chroma)
            .map(|(_, coding)| coding)
            .expect("the winner's chroma was coded");
        let coded = self.refine_intra_type(
            (x, y),
            modes,
            (luma, chroma),
            (cost, coded, layout),
            (source, recon),
            chroma_coding,
        )?;
        for (_, coding) in chroma_codings {
            self.recycle(coding);
        }
        let coded =
            self.finish_intra_coding(coded, layout, (x, y), (luma, chroma), (source, recon))?;
        self.write_intra(&coded, layout, (x, y), modes, (luma, chroma))?;
        self.commit_delta_q(size, false);
        store_block(recon, (x, y), size, &coded);
        self.recycle(coded);
        self.state
            .edges
            .record(block_edges((x, y), self.quantizer.qindex, layout, false));
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

    /// Estimated rate-distortion cost of coding the key-frame block at
    /// `(x, y)` of side `size` as one DC leaf with one transform per plane,
    /// measured on `source` alone before its superblock is coded (the
    /// engine's transform-aware key-frame planning, `planner.intra_plan`):
    /// DC predicted from the source neighbours, transformed, quantized and
    /// reconstructed exactly, with the estimated mode, transform-partition
    /// and coefficient bits (neighbour contexts taken as empty) weighted by λ. Leaves the coded state unchanged.
    /// # Errors
    /// Propagates transform and syntax errors.
    pub fn intra_leaf_cost(
        &mut self,
        source: &Picture,
        (x, y): (usize, usize),
        size: usize,
    ) -> Result<f64> {
        let block_log2 = size.ilog2();
        let layout = if self.lossless() {
            Layout::lossless(block_log2)
        } else {
            Layout::lossy(block_log2, false)
        };
        let lossless = self.lossless();
        let mut coded = self.take_coded(layout);
        self.writer.start_estimate();
        let no_neighbours = ModeContext {
            above: None,
            left: None,
            block_log2,
        };
        let mut result = intra_modes::write_block(
            &mut self.writer,
            lossless,
            no_neighbours,
            Mode::Dc,
            Mode::Dc,
        );
        if result.is_ok() && !lossless {
            result = tx_partition::write(&mut self.writer, block_log2, false, false);
        }
        let mut distortion = 0;
        let planes = if self.options.plan_chroma { 3 } else { 1 };
        for (plane, index) in PlaneIndex::ALL.into_iter().enumerate().take(planes) {
            if result.is_err() {
                break;
            }
            let tx = layout.tx(plane);
            let side = tx.width();
            let area = side * side;
            for (i, position) in layout.positions(plane, (x, y)).enumerate() {
                let source_plane = source.plane(index);
                let dc = nc_dsp::intra_prediction::dc_value(
                    source_plane,
                    position.0,
                    position.1,
                    side,
                    position.1 > 0,
                    position.0 > 0,
                );
                let kind = self.coding_kind(TxType::DctDct);
                let predicted = &mut self.tx_prediction[..area];
                predicted.fill(dc);
                let levels = &mut coded.levels[plane][i * tx.coded_len()..(i + 1) * tx.coded_len()];
                let reconstructed = &mut self.tx_reconstruction[..area];
                let mut view = square_view_mut(reconstructed);
                result = self
                    .quantizer
                    .code(
                        (source, index, position),
                        (tx, TxType::DctDct),
                        &square_view(predicted),
                        (levels, Output::Reconstruct(&mut view)),
                        None,
                        None,
                    )
                    .and_then(|stats| {
                        let context = coeff::Context::derive(
                            plane as u8,
                            layout.plane_log2(plane),
                            tx,
                            false,
                            &EMPTY_CONTEXT,
                            &EMPTY_CONTEXT,
                            false,
                        )?;
                        coeff::write_typed_with_stats(
                            &mut self.writer,
                            levels,
                            stats,
                            tx,
                            context,
                            self.qindex,
                            kind,
                        )
                        .map(|_| ())
                    });
                if result.is_err() {
                    break;
                }
                distortion += sse(
                    &self.quantizer.dsp,
                    source,
                    index,
                    position,
                    side,
                    reconstructed,
                );
            }
        }
        let bits = self.writer.finish_estimate();
        self.recycle(coded);
        result?;
        Ok(distortion as f64 + self.lambda() * bits)
    }

    /// The block's luma candidates ranked by prediction SATD plus
    /// λ-weighted mode bits, best first: the engine's shared set plus the
    /// neighbours' modes, or every nominal direction followed by the best
    /// one's angle deltas.
    fn rank_luma(
        &mut self,
        (x, y): (usize, usize),
        size: usize,
        context: ModeContext,
        pictures: (&Picture, &Picture),
    ) -> Vec<(f64, Mode)> {
        let list = LumaList::new(context);
        let bit_cost = self.lambda().sqrt();
        let luma_cost = |coder: &mut Self, mode: Mode| {
            let satd = coder.prediction_satd(pictures, 0, (x, y), size, mode);
            (
                f64::from(satd)
                    + bit_cost * f64::from(intra_modes::luma_bits(&list, context, mode)),
                mode,
            )
        };
        let set = match self.options.intra.modes {
            LumaCandidates::Shared(set) => Some(set),
            LumaCandidates::AllDirections => None,
        };
        let modes: Vec<Mode> = match set {
            Some(set) => with_neighbours(
                set.modes().iter().map(|&mode| Mode::from(mode)),
                [context.above, context.left].into_iter().flatten(),
            ),
            None => Mode::NON_DIRECTIONAL
                .into_iter()
                .chain(Direction::ALL.map(Mode::nominal))
                .collect(),
        };
        let mut ranked: Vec<(f64, Mode)> = if modes.len() == 1 {
            // Nothing to rank.
            vec![(0.0, modes[0])]
        } else if set.is_some() && !self.lossless() {
            // The shared set is ranked as AV1's key search ranks it: by the
            // rate and distortion of each prediction's coded residual.
            let mut ranked = Vec::with_capacity(modes.len());
            for mode in modes {
                let mode_bits = f64::from(intra_modes::luma_bits(&list, context, mode));
                let cost = self.luma_transform_cost(pictures, (x, y), size, mode);
                ranked.push((cost + self.lambda() * mode_bits, mode));
            }
            ranked
        } else {
            modes
                .into_iter()
                .map(|mode| luma_cost(self, mode))
                .collect()
        };
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        if let (None, Some(direction)) = (set, ranked[0].1.direction()) {
            let nominal = ranked[0].0;
            let mut try_delta = |coder: &mut Self, delta: i8| {
                let cost = luma_cost(coder, Mode::Directional { direction, delta });
                ranked.push(cost);
                cost.0
            };
            match self.options.intra.delta_search {
                DeltaSearch::Off => {}
                DeltaSearch::CoarseToFine => {
                    // Delta costs are roughly convex: refine beside the
                    // better of −2, 0, +2 only.
                    let (lower, upper) = (try_delta(self, -2), try_delta(self, 2));
                    if lower < nominal && lower <= upper {
                        try_delta(self, -1);
                        try_delta(self, -3);
                    } else if upper < nominal {
                        try_delta(self, 1);
                        try_delta(self, 3);
                    } else {
                        try_delta(self, -1);
                        try_delta(self, 1);
                    }
                }
                DeltaSearch::Exhaustive => {
                    for delta in (-intra_modes::MAX_ANGLE_DELTA..=intra_modes::MAX_ANGLE_DELTA)
                        .filter(|&d| d != 0)
                    {
                        try_delta(self, delta);
                    }
                }
            }
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        ranked
    }

    /// The luma and chroma mode pairs worth transform RD, best first: the
    /// non-directional modes and nominal directions ranked by luma
    /// prediction SATD plus λ-weighted mode bits, the best direction's angle
    /// deltas added, and for each of the best few lumas the chroma mode with
    /// the least SATD-plus-bits. Random-decision blocks take one random pair.
    fn intra_candidates(
        &mut self,
        (x, y): (usize, usize),
        size: usize,
        context: ModeContext,
        pictures: (&Picture, &Picture),
    ) -> Vec<(Mode, Mode)> {
        if let Some(random) = self.random.as_mut() {
            let luma = Mode::from_joint(random.below(intra_modes::LUMA_MODES as u64) as usize);
            let chroma_list = intra_modes::chroma_modes(luma);
            let chroma = chroma_list[random.below(chroma_list.len() as u64) as usize];
            return vec![(luma, chroma)];
        }
        let ranked = self.rank_luma((x, y), size, context, pictures);
        let bit_cost = self.lambda().sqrt();
        let mut chroma_satd: Vec<(Mode, u32)> = Vec::new();
        ranked
            .iter()
            .take(self.options.intra.rd_candidates.max(1))
            .map(|&(_, luma)| {
                let chroma = intra_modes::chroma_modes(luma)
                    .into_iter()
                    .take(self.options.intra.chroma_candidates)
                    .map(|mode| {
                        let satd = if let Some(&(_, satd)) =
                            chroma_satd.iter().find(|(m, _)| *m == mode)
                        {
                            satd
                        } else {
                            let satd = (1..3)
                                .map(|plane| {
                                    self.prediction_satd(pictures, plane, (x, y), size, mode)
                                })
                                .sum();
                            chroma_satd.push((mode, satd));
                            satd
                        };
                        (
                            f64::from(satd)
                                + bit_cost * f64::from(intra_modes::chroma_bits(luma, mode)),
                            mode,
                        )
                    })
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .map_or(Mode::Dc, |(_, mode)| mode);
                (luma, chroma)
            })
            .collect()
    }

    /// Rate and distortion of coding `mode`'s luma prediction of the block
    /// as one transform (the unsplit layout, DCT): the reconstruction's
    /// squared error plus λ times the estimated coefficient bits under the
    /// block's real coefficient contexts. Codes nothing.
    fn luma_transform_cost(
        &mut self,
        (source, recon): (&Picture, &Picture),
        (x, y): (usize, usize),
        size: usize,
        mode: Mode,
    ) -> f64 {
        let block_log2 = size.ilog2();
        let tx = Layout::lossy(block_log2, false).tx(LUMA);
        let side = tx.width();
        let area = side * side;
        let target = IntraTarget {
            plane: LUMA,
            block: (x, y),
            block_size: size,
            tx: (x, y),
            side,
        };
        let predicted = &mut self.tx_prediction[..area];
        intra_edges::predict(
            recon.plane(PlaneIndex::Y),
            &self.state,
            target,
            mode,
            &self.quantizer.dsp,
            &mut self.edges,
            predicted,
        );
        self.rank_levels.resize(tx.coded_len(), 0);
        let reconstructed = &mut self.tx_reconstruction[..area];
        let coded = self.quantizer.code(
            (source, PlaneIndex::Y, (x, y)),
            (tx, TxType::DctDct),
            &square_view(predicted),
            (
                &mut self.rank_levels,
                Output::Reconstruct(&mut square_view_mut(reconstructed)),
            ),
            None,
            None,
        );
        let context = self
            .state
            .coeff_context(LUMA, (x, y), tx, block_log2, false, false);
        let (Ok(stats), Ok(context)) = (coded, context) else {
            return f64::INFINITY;
        };
        let distortion = sse(
            &self.quantizer.dsp,
            source,
            PlaneIndex::Y,
            (x, y),
            side,
            reconstructed,
        );
        let kind = self.coding_kind(TxType::DctDct);
        self.writer.start_estimate();
        let written = coeff::write_typed_with_stats(
            &mut self.writer,
            &self.rank_levels,
            stats,
            tx,
            context,
            self.qindex,
            kind,
        );
        let bits = self.writer.finish_estimate();
        if written.is_err() {
            return f64::INFINITY;
        }
        distortion as f64 + self.lambda() * bits
    }

    /// SATD of `mode`'s whole-block prediction of `plane` (one transform of
    /// the plane block's size) against the source.
    fn prediction_satd(
        &mut self,
        (source, recon): (&Picture, &Picture),
        plane: usize,
        (x, y): (usize, usize),
        size: usize,
        mode: Mode,
    ) -> u32 {
        let index = PlaneIndex::ALL[plane];
        let side = size >> usize::from(plane != 0);
        let origin = plane_origin(plane, (x, y));
        let predicted = &mut self.mode_trial[..side * side];
        let target = IntraTarget {
            plane,
            block: (x, y),
            block_size: size,
            tx: origin,
            side,
        };
        intra_edges::predict(
            recon.plane(index),
            &self.state,
            target,
            mode,
            &self.quantizer.dsp,
            &mut self.edges,
            predicted,
        );
        let predicted = PlaneView::new(predicted, side, side, side).expect("square prediction");
        let source = source
            .plane(index)
            .view()
            .sub_view(origin.0, origin.1, side, side)
            .expect("block inside the picture");
        let block = BlockSize::new(side, side).expect("intra blocks are 4..64 square");
        self.quantizer.dsp.satd(block, &source, &predicted)
    }

    /// Recode the winning intra coding with `ADST_ADST` luma transforms when
    /// the frame's set allows them at its luma transform size, keeping the
    /// cheaper by rate and distortion. Only the winner is retried: the mode
    /// and layout searches stay DCT-only, so this costs one luma coding per
    /// block (all-intra 640x360 q80..200: PSNR BD -0.45% at +9% CPU;
    /// searching ADST in every trial: -1.11% at +80%).
    fn refine_intra_type(
        &mut self,
        (x, y): (usize, usize),
        modes: ModeContext,
        pair: (Mode, Mode),
        (cost, coded, layout): (f64, Coded, Layout),
        pictures: (&Picture, &mut Picture),
        chroma: &Coded,
    ) -> Result<Coded> {
        let allowed = self.options.tx_set.luma_types(layout.tx(LUMA), false);
        if self.random.is_some()
            || !self.options.intra_type_retry
            || !allowed.contains(&TxType::AdstAdst)
        {
            return Ok(coded);
        }
        let mut trial = self.take_coded(layout);
        self.code_intra(
            &mut trial,
            layout,
            (x, y),
            (pair.0, TxType::AdstAdst),
            pictures,
            chroma,
        )?;
        if trial.luma_types.iter().all(|&t| t == TxType::DctDct) {
            self.recycle(trial);
            return Ok(coded);
        }
        let size = 1 << layout.block_log2;
        let trial_cost = self.trial_bits(x, y, size, |coder| {
            coder.write_intra(&trial, layout, (x, y), modes, pair)
        })? * self.lambda()
            + trial.distortion as f64;
        Ok(if trial_cost < cost {
            self.recycle(coded);
            trial
        } else {
            self.recycle(trial);
            coded
        })
    }

    /// The type of the next luma transform of an intra block coded with
    /// `preferred`: `preferred` where the frame's set allows it at `tx`
    /// (else `DCT_DCT`); random-decision blocks draw any allowed type.
    fn intra_luma_type(&mut self, tx: TxSize, preferred: TxType) -> TxType {
        let allowed = self.options.tx_set.luma_types(tx, false);
        match self.random.as_mut() {
            Some(random) if !allowed.is_empty() => {
                allowed[random.below(allowed.len() as u64) as usize]
            }
            _ if allowed.contains(&preferred) => preferred,
            _ => TxType::DctDct,
        }
    }

    /// Predict, quantize and reconstruct every luma transform of an intra
    /// block in coding order with [`Self::intra_luma_type`] of `luma_type`,
    /// writing each into `recon` before the next is predicted. The chroma
    /// planes are copied from `chroma`, the block's chroma coding with its
    /// chroma mode ([`Self::code_intra_chroma`]): chroma takes one transform
    /// layout under every luma layout and is predicted from chroma neighbours
    /// only, so it is coded once per mode.
    fn code_intra(
        &mut self,
        coded: &mut Coded,
        layout: Layout,
        (x, y): (usize, usize),
        (luma, luma_type): (Mode, TxType),
        (source, recon): (&Picture, &mut Picture),
        chroma: &Coded,
    ) -> Result<()> {
        let luma = self.code_intra_plane(
            coded,
            layout,
            LUMA,
            (luma, luma_type),
            (x, y),
            (source, recon),
            Quantisation::Plain,
        )?;
        for plane in CHROMA {
            coded.levels[plane].clone_from(&chroma.levels[plane]);
            coded.stats[plane].clone_from(&chroma.stats[plane]);
            coded.pixels[plane].clone_from(&chroma.pixels[plane]);
        }
        coded.distortion = luma + chroma.distortion;
        Ok(())
    }

    /// The chroma planes of an intra block coded with `mode` and `layout`'s
    /// chroma transforms, and their distortion.
    fn code_intra_chroma(
        &mut self,
        layout: Layout,
        (x, y): (usize, usize),
        mode: Mode,
        (source, recon): (&Picture, &mut Picture),
    ) -> Result<Coded> {
        let mut chroma = self.take_coded(layout);
        for plane in CHROMA {
            chroma.distortion += self.code_intra_plane(
                &mut chroma,
                layout,
                plane,
                (mode, TxType::DctDct),
                (x, y),
                (source, recon),
                Quantisation::Plain,
            )?;
        }
        Ok(chroma)
    }

    /// Predict, quantize and reconstruct every transform of one plane of an
    /// intra block into `coded` and `recon`, returning that plane's squared
    /// error. Luma transforms take [`Self::intra_luma_type`] of `luma_type`,
    /// except that a DC-only result is recoded as `DCT_DCT`, which such
    /// blocks infer; chroma is DCT-only.
    #[allow(clippy::too_many_arguments)]
    fn code_intra_plane(
        &mut self,
        coded: &mut Coded,
        layout: Layout,
        plane: usize,
        (mode, luma_type): (Mode, TxType),
        (x, y): (usize, usize),
        (source, recon): (&Picture, &mut Picture),
        quantisation: Quantisation,
    ) -> Result<u64> {
        let index = PlaneIndex::ALL[plane];
        let tx = layout.tx(plane);
        let side = 1 << layout.plane_log2(plane);
        let origin = plane_origin(plane, (x, y));
        let len = tx.coded_len();
        let area = tx.width() * tx.width();
        let set = self.options.tx_set;
        let saved = self.begin_trellis_contexts((x, y), layout, plane, quantisation);
        for (i, position) in layout.positions(plane, (x, y)).enumerate() {
            let mut tx_type = match (plane, quantisation) {
                (LUMA, Quantisation::Plain) => self.intra_luma_type(tx, luma_type),
                // The final coding keeps the type each transform won with.
                (LUMA, Quantisation::Final) => coded.luma_types[i],
                _ => TxType::DctDct,
            };
            let context =
                self.trellis_context(coded, layout, plane, position, (false, quantisation))?;
            let predicted = &mut self.tx_prediction[..area];
            let target = IntraTarget {
                plane,
                block: (x, y),
                block_size: 1 << layout.block_log2,
                tx: position,
                side: tx.width(),
            };
            intra_edges::predict(
                recon.plane(index),
                &self.state,
                target,
                mode,
                &self.quantizer.dsp,
                &mut self.edges,
                predicted,
            );
            let levels = &mut coded.levels[plane][i * len..(i + 1) * len];
            let reconstructed = &mut self.tx_reconstruction[..area];
            let mut stats = self.quantizer.code(
                (source, index, position),
                (tx, tx_type),
                &square_view(predicted),
                (
                    levels,
                    Output::Reconstruct(&mut square_view_mut(reconstructed)),
                ),
                context.map(|context| RateSource {
                    writer: &mut self.writer,
                    context,
                    set,
                }),
                None,
            )?;
            // Only the DC (scan index 0) left: the type cannot be signalled.
            if tx_type != TxType::DctDct && stats.eob() <= 1 {
                tx_type = TxType::DctDct;
                stats = self.quantizer.code(
                    (source, index, position),
                    (tx, tx_type),
                    &square_view(predicted),
                    (
                        levels,
                        Output::Reconstruct(&mut square_view_mut(reconstructed)),
                    ),
                    context.map(|context| RateSource {
                        writer: &mut self.writer,
                        context,
                        set,
                    }),
                    None,
                )?;
            }
            if saved.is_some() {
                self.state.update_coeff_context(plane, position, tx, levels);
            }
            if plane == LUMA {
                coded.luma_types[i] = tx_type;
            }
            coded.stats[plane][i] = stats;
            store(recon, index, position, tx.width(), reconstructed);
            copy_into_block(
                &mut coded.pixels[plane],
                side,
                (position.0 - origin.0, position.1 - origin.1),
                tx.width(),
                reconstructed,
            );
        }
        self.end_trellis_contexts(saved);
        Ok(sse(
            &self.quantizer.dsp,
            source,
            index,
            origin,
            side,
            &coded.pixels[plane],
        ))
    }

    /// Write an intra block's syntax and coefficients and update the contexts.
    fn write_intra(
        &mut self,
        coded: &Coded,
        layout: Layout,
        (x, y): (usize, usize),
        modes: ModeContext,
        (luma, chroma): (Mode, Mode),
    ) -> Result<()> {
        let size = 1 << layout.block_log2;
        if !self.key {
            self.write_skip_mode((x, y), size, false)?;
            let context = self.state.intra_inter_context(x, y, size);
            self.writer
                .model(0, &IS_INTER[context], |m| &mut m.is_inter[context])?;
        }
        self.cdef_index((x, y), false);
        self.ccso_flags((x, y));
        self.write_delta_q(size, false)?;
        let lossless = self.lossless();
        intra_modes::write_block(&mut self.writer, lossless, modes, luma, chroma)?;
        if !lossless {
            tx_partition::write(&mut self.writer, layout.block_log2, false, layout.split())?;
        }
        self.write_coefficients(coded, layout, (x, y), false)
    }

    /// Code every transform's coefficients in coding order and update the
    /// entropy contexts.
    fn write_coefficients(
        &mut self,
        coded: &Coded,
        layout: Layout,
        (x, y): (usize, usize),
        is_inter: bool,
    ) -> Result<()> {
        let mut u_coded = false;
        // An inter block's trials share one chroma coding (see
        // `code_inter_chroma`) and price under the same frozen
        // probabilities, so its chroma rate is priced once per block. Chroma
        // is written last: skipping it leaves out only coefficient contexts
        // the trial restores anyway.
        let reuse_chroma = is_inter && self.writer.estimates_frozen_now();
        let mut chroma_start = None;
        for plane in 0..3 {
            if plane == 1 && reuse_chroma {
                if let Some(q9) = self.inter_chroma_q9 {
                    self.writer.add_estimate_q9(q9);
                    return Ok(());
                }
                chroma_start = Some(self.writer.estimate_q9());
            }
            let tx = layout.tx(plane);
            for (i, position) in layout.positions(plane, (x, y)).enumerate() {
                let levels = coded.tx_levels(layout, plane, i);
                let context = self.state.coeff_context(
                    plane,
                    position,
                    tx,
                    layout.plane_log2(plane),
                    is_inter,
                    u_coded,
                )?;
                let kind = coeff::Kind {
                    set: self.options.tx_set,
                    levels: self.options.levels,
                    tx_type: if plane == LUMA {
                        coded.luma_types[i]
                    } else {
                        TxType::DctDct
                    },
                };
                let entropy = coeff::write_typed_with_stats(
                    &mut self.writer,
                    levels,
                    coded.stats[plane][i],
                    tx,
                    context,
                    self.qindex,
                    kind,
                )?;
                self.state.set_coeff_context(plane, position, tx, entropy);
                if plane == 1 {
                    // Only an all-zero block has the zero context byte.
                    u_coded = entropy != 0;
                }
            }
        }
        if let Some(start) = chroma_start {
            self.inter_chroma_q9 = Some(self.writer.estimate_q9() - start);
        }
        Ok(())
    }

    /// Code an inter block of side `size` predicted with `mv` (row, column;
    /// eighth-pel luma) into `recon`, choosing the transform layout and
    /// dropping the residual when that is cheaper.
    /// # Errors
    /// Propagates prediction and syntax errors.
    pub fn encode_inter(
        &mut self,
        (x, y): (usize, usize),
        size: usize,
        mv: (i32, i32),
        source: &Picture,
        reference: &Picture,
        recon: &mut Picture,
    ) -> Result<()> {
        self.state.check_order(x, y)?;
        self.inter_chroma_q9 = None;
        let (block, plane_sse) = self.predict_block((x, y), size, mv, source, reference)?;
        let block_log2 = size.ilog2();
        let skipped = plane_sse[LUMA] + self.weighted_chroma(plane_sse[1] + plane_sse[2]);
        let random_skip = self.random.as_mut().map(|r| r.chance(RANDOM_SKIP_CHANCE));
        // Speed heuristic (`av2.skip_estimate`): a prediction error this
        // small decides skip without residual trials or rate estimates. Most
        // P blocks end up skipped (69% / 72% / 85% of 16 / 32 / 64 blocks
        // over four clips) after paying for full coding; this settles the
        // clearest of them. Not exact: a few blocks that would have coded a
        // residual are skipped. At the default 4: P instructions -5.2%,
        // J_real proxy-6 -2.4% (quality +0.5% full / +0.8% low).
        let estimated_skip = random_skip.is_none()
            && !self.lossless()
            && self.options.skip_estimate.is_some_and(|per_unit| {
                (skipped as f64)
                    < f64::from(per_unit) * self.lambda() * (size * size) as f64
                        / SKIP_ESTIMATE_AREA
            });
        let best = if random_skip == Some(true) || estimated_skip {
            None
        } else {
            // Only the residual trials use the pre-transform skip test (its
            // zero test runs 8x8 variances over the whole block): blocks
            // settled above never pay for it.
            let luma_skipped = random_skip.is_none()
                && self.skips_luma_before_transform((x, y), size, source, plane_sse[LUMA]);
            self.best_inter_coding(
                &block,
                source,
                random_skip.is_some(),
                (plane_sse, luma_skipped),
            )?
        };
        let best = match best {
            Some((cost, coded, layout))
                if self.beats_skip(&block, cost, skipped, random_skip)? =>
            {
                Some((coded, layout))
            }
            Some((_, coded, _)) => {
                self.recycle(coded);
                None
            }
            None => None,
        };
        let keep_zero = random_skip.is_some();
        let chosen = self.finish_inter_coding(best, (x, y), source, keep_zero)?;
        self.write_inter(
            &block,
            chosen.as_ref().map(|(coded, layout)| (coded, *layout)),
            MotionSyntax::Written,
        )?;
        let skip = chosen.is_none();
        self.commit_delta_q(size, skip);
        let layout = chosen.as_ref().map_or_else(
            || skipped_layout(block_log2, self.lossless()),
            |&(_, layout)| layout,
        );
        self.state
            .edges
            .record(block_edges((x, y), self.quantizer.qindex, layout, skip));
        self.state.record(
            x,
            y,
            size,
            BlockInfo {
                is_inter: true,
                skip,
                luma_mode: Mode::Dc,
                filter: Some(block.filter),
                skip_mode: matches!(block.signal, Signal::Skip { .. }),
            },
        );
        match chosen {
            Some((coded, _)) => {
                store_block(recon, (x, y), size, &coded);
                self.recycle(coded);
            }
            None => {
                for (plane, index) in PlaneIndex::ALL.into_iter().enumerate() {
                    let side = size >> usize::from(plane != 0);
                    store(
                        recon,
                        index,
                        plane_origin(plane, (x, y)),
                        side,
                        &self.prediction[plane],
                    );
                }
            }
        }
        self.state.motion.commit_motion(x, y, size, block.motion())
    }

    /// The signalling of the inter block of side `size` at `(x, y)` that
    /// the engine gave the vector `mv`, its prediction in `self.prediction`
    /// and each plane's prediction error: the vector's single-reference
    /// signalling, or skip mode where that predicts more cheaply
    /// ([`Self::choose_skip_mode`]).
    fn predict_block(
        &mut self,
        (x, y): (usize, usize),
        size: usize,
        mv: (i32, i32),
        source: &Picture,
        reference: &Picture,
    ) -> Result<(InterBlock, [u64; 3])> {
        let candidates = self.state.motion.candidates(x, y, size)?;
        let (mode, index) = choose_signalling(candidates, mv);
        let filter_context = (self.options.filter == FrameFilter::Switchable
            && interp_filter::is_coded(mode))
        .then(|| self.state.filter_context(x, y, size));
        let (filter, plane_sse) =
            self.predict_inter(reference, source, (x, y), size, mv, filter_context)?;
        let single = InterBlock {
            x,
            y,
            size,
            filter,
            signal: Signal::Single {
                mv,
                mode,
                index,
                candidates,
                filter_context,
            },
        };
        Ok(
            match self.choose_skip_mode(&single, plane_sse, source, reference)? {
                Some(chosen) => {
                    std::mem::swap(&mut self.prediction, &mut self.skip_prediction);
                    chosen
                }
                None => (single, plane_sse),
            },
        )
    }

    /// Skip mode for the inter block `single` (whose prediction, in
    /// `self.prediction`, has errors `single_sse`) if one of its skip-mode
    /// entries predicts more cheaply by error plus signalling (λ · bits,
    /// neither with a residual): that block and its errors, its prediction
    /// in `self.skip_prediction`. Random-decision blocks take skip mode by
    /// chance, with any DRL index.
    fn choose_skip_mode(
        &mut self,
        single: &InterBlock,
        single_sse: [u64; 3],
        source: &Picture,
        reference: &Picture,
    ) -> Result<Option<(InterBlock, [u64; 3])>> {
        let Some(ranked) = self.options.skip_mode.filter(|_| !self.lossless()) else {
            return Ok(None);
        };
        let (x, y, size) = (single.x, single.y, single.size);
        let candidates = self.state.motion.skip_candidates(x, y, size)?;
        let skip_block = |index: usize| InterBlock {
            filter: SKIP_MODE_FILTER,
            signal: Signal::Skip {
                index,
                pair: candidates.pair(index),
                max_drl_bits: candidates.max_drl_bits,
            },
            ..*single
        };
        if let Some(random) = self.random.as_mut() {
            if !random.chance(RANDOM_SKIP_MODE_CHANCE) {
                return Ok(None);
            }
            let index = random.below(u64::from(candidates.max_drl_bits) + 1) as usize;
            let block = skip_block(index);
            let sse = self.predict_skip_mode(&block, source, reference)?;
            return Ok(Some((block, sse)));
        }
        let Signal::Single { mv, .. } = single.signal else {
            unreachable!("a single-reference block");
        };
        let single_cost = self.prediction_cost(single, single_sse)?;
        let mut best = (single_cost, None);
        let ranked = usize::from(candidates.count).min(ranked);
        for index in 0..ranked {
            let block = skip_block(index);
            // Every plane's error and the bits only add to the luma error:
            // a luma error already at the best cost cannot win.
            let Some(sse) =
                self.rank_skip_mode(&block, (mv, single_sse), source, reference, best.0)?
            else {
                continue;
            };
            let cost = self.prediction_cost(&block, sse)?;
            if cost < best.0 {
                best = (cost, Some(index));
            }
        }
        let Some(index) = best.1 else {
            return Ok(None);
        };
        let block = skip_block(index);
        let sse = self.predict_skip_mode(&block, source, reference)?;
        // The ranking was approximate: the exact prediction must still win.
        Ok((self.prediction_cost(&block, sse)? < single_cost).then_some((block, sse)))
    }

    /// Each plane's error of the skip-mode block `block` for ranking, its
    /// prediction in `self.skip_prediction`; `None` (chroma not predicted)
    /// once the luma error reaches `bound`. An approximation of the exact
    /// prediction ([`Self::predict_skip_mode`]), for ranking only:
    /// - each plane averages the two vectors' final 8-bit predictions (the
    ///   shared SIMD kernels) instead of their compound-precision ones
    ///   (within one sample value);
    /// - a vector equal to the single-reference choice's (`single_mv`)
    ///   reuses that prediction (`self.prediction`, the frame filter
    ///   instead of skip mode's sharp one), and a pair of equal vectors is
    ///   predicted once. When both vectors are the single choice's, the
    ///   errors are its errors `single_sse` and nothing is predicted.
    ///   Skipping these predictions removes about half of the ranking's
    ///   (`crowd_mot17_09` 720p CBR 600: 22% of first entries are the single
    ///   vector twice, 36% hold it once, 11% another vector twice).
    fn rank_skip_mode(
        &mut self,
        block: &InterBlock,
        (single_mv, single_sse): ((i32, i32), [u64; 3]),
        source: &Picture,
        reference: &Picture,
        bound: f64,
    ) -> Result<Option<[u64; 3]>> {
        let Signal::Skip { pair, .. } = block.signal else {
            unreachable!("a skip-mode block");
        };
        if pair == (single_mv, single_mv) {
            return Ok(((single_sse[LUMA] as f64) < bound).then_some(single_sse));
        }
        // The vector to predict besides the other one, which is the single
        // choice's (`Some(true)`), the same vector again (`Some(false)`),
        // or predicted too (`None`).
        let (first, second, shared) = if pair.1 == single_mv {
            (pair.0, pair.1, Some(true))
        } else if pair.0 == single_mv {
            (pair.1, pair.0, Some(true))
        } else if pair.0 == pair.1 {
            (pair.0, pair.1, Some(false))
        } else {
            (pair.0, pair.1, None)
        };
        let (x, y, size) = (block.x, block.y, block.size);
        let dsp = &self.quantizer.dsp;
        let mut errors = [0; 3];
        for (plane, index) in PlaneIndex::ALL.into_iter().enumerate() {
            let subsample = u8::from(plane != 0);
            let side = size >> subsample;
            let (px, py) = plane_origin(plane, (x, y));
            let samples = &mut self.skip_prediction[plane];
            samples.resize(side * side, 0);
            let reference = reference.plane(index);
            crate::prediction::predict(
                dsp,
                SKIP_MODE_FILTER,
                reference,
                px,
                py,
                first,
                subsample,
                samples,
            )?;
            let other: &[u8] = match shared {
                Some(true) => &self.prediction[plane],
                Some(false) => &[],
                None => {
                    let other = &mut self.skip_second;
                    other.resize(side * side, 0);
                    crate::prediction::predict(
                        dsp,
                        SKIP_MODE_FILTER,
                        reference,
                        px,
                        py,
                        second,
                        subsample,
                        other,
                    )?;
                    other
                }
            };
            for (a, &b) in samples.iter_mut().zip(other) {
                *a = ((u16::from(*a) + u16::from(b) + 1) >> 1) as u8;
            }
            errors[plane] = sse(dsp, source, index, (px, py), side, samples);
            if plane == LUMA && errors[LUMA] as f64 >= bound {
                return Ok(None);
            }
        }
        Ok(Some(errors))
    }

    /// Predict the skip-mode block `block` exactly (compound precision)
    /// into `self.skip_prediction`, returning each plane's prediction error.
    fn predict_skip_mode(
        &mut self,
        block: &InterBlock,
        source: &Picture,
        reference: &Picture,
    ) -> Result<[u64; 3]> {
        let Signal::Skip { pair, .. } = block.signal else {
            unreachable!("a skip-mode block");
        };
        let (x, y, size) = (block.x, block.y, block.size);
        let dsp = &self.quantizer.dsp;
        for (plane, index) in PlaneIndex::ALL.into_iter().enumerate() {
            let side = size >> usize::from(plane != 0);
            let samples = &mut self.skip_prediction[plane];
            samples.resize(side * side, 0);
            crate::prediction::predict_compound(
                dsp,
                SKIP_MODE_FILTER,
                reference.plane(index),
                plane_origin(plane, (x, y)),
                [pair.0, pair.1],
                u8::from(plane != 0),
                &mut self.compound,
                samples,
            )?;
        }
        Ok(prediction_distortion(
            dsp,
            source,
            (x, y),
            size,
            &self.skip_prediction,
        ))
    }

    /// Whether the residual coding of `block` that costs `cost` (rate and
    /// distortion) is kept over skipping the block, whose prediction error
    /// (chroma weighted) is `skipped`. Lossless blocks and random decisions
    /// that keep their residual always code it.
    ///
    /// Speed (exact): the skip syntax is priced only here, once a coding
    /// competes with it, not for every block that runs residual trials:
    /// about half of those quantize to nothing and are skipped unpriced
    /// (`crowd_mot17_09` 720p CBR 600, AV2 Fast: 516 of 970 per P frame).
    fn beats_skip(
        &mut self,
        block: &InterBlock,
        cost: f64,
        skipped: u64,
        random_skip: Option<bool>,
    ) -> Result<bool> {
        if self.lossless() || random_skip == Some(false) {
            return Ok(true);
        }
        let skip_cost = self.trial_bits(block.x, block.y, block.size, |coder| {
            coder.write_inter(block, None, MotionSyntax::Omitted)
        })? * self.lambda()
            + skipped as f64;
        Ok(cost <= skip_cost)
    }

    /// Rate-distortion cost of coding `block` without residual: its
    /// prediction errors `sse` (chroma weighted) plus λ · its bits.
    fn prediction_cost(&mut self, block: &InterBlock, sse: [u64; 3]) -> Result<f64> {
        let distortion = sse[LUMA] + self.weighted_chroma(sse[1] + sse[2]);
        let bits = self.trial_bits(block.x, block.y, block.size, |coder| {
            coder.write_inter(block, None, MotionSyntax::Written)
        })?;
        Ok(bits * self.lambda() + distortion as f64)
    }

    /// The cheapest residual coding of an inter block over its luma layouts,
    /// with its rate-distortion cost. An all-zero residual is dropped (it
    /// reconstructs as the skipped block, more cheaply) unless `keep_zero`.
    /// `plane_sse` are the prediction's errors; `luma_skipped` codes no luma
    /// residual and tries only the unsplit layout
    /// ([`Self::skips_luma_before_transform`]).
    fn best_inter_coding(
        &mut self,
        block: &InterBlock,
        source: &Picture,
        keep_zero: bool,
        (plane_sse, luma_skipped): ([u64; 3], bool),
    ) -> Result<Option<(f64, Coded, Layout)>> {
        let (x, y, size) = (block.x, block.y, block.size);
        let luma_sse = plane_sse[LUMA];
        let layouts = self.layouts(size.ilog2());
        let first = layouts[0].expect("at least one layout");
        let chroma = self.code_inter_chroma(first, (x, y), (source, plane_sse))?;
        let mut best: Option<(f64, Coded, Layout)> = None;
        for (i, layout) in layouts.into_iter().enumerate() {
            let Some(layout) = layout else { continue };
            if i > 0 && (luma_skipped || self.split_pointless()) {
                break;
            }
            let mut coded = self.take_coded(layout);
            if luma_skipped {
                self.prediction_luma(&mut coded, luma_sse, &chroma);
            } else {
                self.code_inter(&mut coded, layout, (x, y), (source, luma_sse), &chroma)?;
            }
            // Counted in u32 lanes: a usize count vectorizes two lanes wide
            // on aarch64, which perf showed hot on the Jetson.
            let nonzero: u32 = coded.levels[LUMA].iter().map(|&l| u32::from(l != 0)).sum();
            self.last_luma_nonzero = nonzero as usize;
            if coded.is_zero() && !keep_zero {
                self.recycle(coded);
                continue;
            }
            self.keep_cheaper(block, coded, layout, &mut best)?;
        }
        // The trials priced chroma from its levels alone; only the kept
        // coding needs the chroma samples and forward transforms.
        let mut chroma = chroma;
        if let Some((_, coded, _)) = &mut best {
            for plane in CHROMA {
                self.reconstruct_estimated(&mut chroma, first, plane, (x, y))?;
            }
            copy_chroma_reconstruction(coded, &chroma);
        }
        self.recycle(chroma);
        // A skipped luma has no transform whose type could change.
        if let Some((cost, coded, layout)) = &mut best
            && !luma_skipped
            && self.refine_inter_types(coded, *layout, (x, y), source)?
        {
            *cost = self.trial_bits(x, y, size, |coder| {
                coder.write_inter(block, Some((coded, *layout)), MotionSyntax::Omitted)
            })? * self.lambda()
                + coded.distortion as f64;
        }
        Ok(best)
    }

    /// Chroma squared error `sse` in luma units for the block decisions.
    fn weighted_chroma(&self, sse: u64) -> u64 {
        (sse as f64 * self.chroma_weight).round() as u64
    }

    /// Price the inter coding `coded` with `layout` (bits × λ plus its
    /// distortion) and keep it in `best` if it is cheaper; the loser's
    /// buffers are recycled.
    fn keep_cheaper(
        &mut self,
        block: &InterBlock,
        coded: Coded,
        layout: Layout,
        best: &mut Option<(f64, Coded, Layout)>,
    ) -> Result<()> {
        let cost = self.trial_bits(block.x, block.y, block.size, |coder| {
            coder.write_inter(block, Some((&coded, layout)), MotionSyntax::Omitted)
        })? * self.lambda()
            + coded.distortion as f64;
        match best {
            Some((best_cost, ..)) if *best_cost <= cost => self.recycle(coded),
            _ => {
                if let Some((_, previous, _)) = best.replace((cost, coded, layout)) {
                    self.recycle(previous);
                }
            }
        }
        Ok(())
    }

    /// The residual coding of an inter block at `(x, y)`: the `chosen` one
    /// (mode decision kept it over skipping), each trellised plane that has
    /// levels coded again with [`Quantisation::Final`]. `None` codes the
    /// block as skipped, also when the trellis leaves no level and
    /// `keep_zero` is false (skipping reconstructs the same prediction with
    /// less syntax). Mode decision is over, so `coded.distortion` is left as
    /// it was.
    fn finish_inter_coding(
        &mut self,
        chosen: Option<(Coded, Layout)>,
        (x, y): (usize, usize),
        source: &Picture,
        keep_zero: bool,
    ) -> Result<Option<(Coded, Layout)>> {
        let Some((mut coded, layout)) = chosen else {
            return Ok(None);
        };
        for plane in 0..coded.levels.len() {
            if self.final_coding_skips(&coded, plane) {
                self.reconstruct_estimated(&mut coded, layout, plane, (x, y))?;
                continue;
            }
            self.code_inter_plane(
                &mut coded,
                layout,
                plane,
                (x, y),
                (source, Quantisation::Final),
            )?;
        }
        if coded.is_zero() && !keep_zero {
            self.recycle(coded);
            return Ok(None);
        }
        Ok(Some((coded, layout)))
    }

    /// The chosen intra coding `coded` with `layout` and the `modes` (luma,
    /// chroma), each trellised plane that has levels predicted and coded
    /// again with [`Quantisation::Final`] into `recon` (unchanged without a
    /// trellis). Mode decision is over, so `coded.distortion` is left as it
    /// was.
    fn finish_intra_coding(
        &mut self,
        mut coded: Coded,
        layout: Layout,
        (x, y): (usize, usize),
        (luma, chroma): (Mode, Mode),
        (source, recon): (&Picture, &mut Picture),
    ) -> Result<Coded> {
        if self.quantizer.trellis.is_none() {
            return Ok(coded);
        }
        for plane in 0..coded.levels.len() {
            if self.final_coding_skips(&coded, plane) {
                continue;
            }
            let mode = if plane == LUMA { luma } else { chroma };
            self.code_intra_plane(
                &mut coded,
                layout,
                plane,
                (mode, TxType::DctDct),
                (x, y),
                (source, recon),
                Quantisation::Final,
            )?;
        }
        Ok(coded)
    }

    /// Whether the final coding leaves `plane` of `coded` as the trials
    /// coded it: the trellis does not run on it, it has no level, or it is
    /// TCQ luma (which every coding searches). The trellis searches only the
    /// plain quantiser's span, so an empty plane stays empty (and a luma
    /// skipped before its transform stays skipped).
    fn final_coding_skips(&self, coded: &Coded, plane: usize) -> bool {
        !self.trellises(plane, Quantisation::Final)
            || coded.plane_is_zero(plane)
            || self.searches_tcq(plane)
    }

    /// Whether the split luma layout is not worth a trial after the unsplit
    /// one: its luma residual was (nearly) empty, so smaller transforms would
    /// mostly add syntax. Skips a quantization, reconstruction and trial
    /// write per inter block; see `av2.inter_split_min_nonzero`.
    fn split_pointless(&self) -> bool {
        self.random.is_none() && self.last_luma_nonzero < self.options.inter_split_min_nonzero
    }

    /// Whether the current inter prediction's luma at `(x, y)` (prediction
    /// error `luma_sse`) is coded without residual, skipping the luma
    /// transform trials (`av2.pre_tx_skip`).
    fn skips_luma_before_transform(
        &self,
        (x, y): (usize, usize),
        size: usize,
        source: &Picture,
        luma_sse: u64,
    ) -> bool {
        let Some(policy) = self.options.pre_tx_skip else {
            return false;
        };
        if self.lossless() {
            return false;
        }
        let block = LumaBlock {
            side: size,
            source: source
                .plane(PlaneIndex::Y)
                .view()
                .sub_view(x, y, size, size)
                .expect("blocks lie inside the picture"),
            prediction: PlaneView::new(&self.prediction[LUMA], size, size, size)
                .expect("square prediction"),
            sse: luma_sse,
            both_neighbours_skip: self.state.above_and_left_skip(x, y),
        };
        let tx = Layout::lossy(size.ilog2(), false).luma;
        policy.skips_luma(
            &self.quantizer.dsp,
            &block,
            self.quantizer.qindex.luma,
            self.options.rounding.luma[tx.index()],
        )
    }

    /// The current inter prediction as the luma of `coded` (no residual,
    /// squared error `luma_sse`), with the chroma levels of `chroma`.
    fn prediction_luma(&self, coded: &mut Coded, luma_sse: u64, chroma: &Coded) {
        coded.levels[LUMA].fill(0);
        coded.stats[LUMA].fill(LevelStats::EMPTY);
        coded.pixels[LUMA].copy_from_slice(&self.prediction[LUMA]);
        copy_chroma_levels(coded, chroma);
        coded.distortion = luma_sse + chroma.distortion;
    }

    /// Quantize and reconstruct the residual of the current inter prediction
    /// with `layout`, whose luma prediction error is `luma_sse`. The chroma
    /// levels are copied from `chroma`, the block's chroma coding
    /// ([`Self::code_inter_chroma`]): chroma takes one transform at the
    /// chroma block size under every luma layout, so it is coded once (the
    /// winner gets the rest of it, [`Self::best_inter_coding`]).
    fn code_inter(
        &mut self,
        coded: &mut Coded,
        layout: Layout,
        (x, y): (usize, usize),
        (source, luma_sse): (&Picture, u64),
        chroma: &Coded,
    ) -> Result<()> {
        self.code_inter_plane(coded, layout, LUMA, (x, y), (source, Quantisation::Plain))?;
        let luma = self.inter_plane_sse(coded, layout, LUMA, (x, y), (source, luma_sse));
        copy_chroma_levels(coded, chroma);
        coded.distortion = luma + chroma.distortion;
        Ok(())
    }

    /// The chroma planes of the current inter prediction (with errors
    /// `plane_sse`, per plane) coded with `layout`'s chroma transforms, and
    /// their distortion.
    fn code_inter_chroma(
        &mut self,
        layout: Layout,
        (x, y): (usize, usize),
        (source, plane_sse): (&Picture, [u64; 3]),
    ) -> Result<Coded> {
        let mut chroma = self.take_coded(layout);
        let mut distortion = 0;
        for plane in CHROMA {
            self.code_inter_plane(
                &mut chroma,
                layout,
                plane,
                (x, y),
                (source, Quantisation::Plain),
            )?;
            distortion +=
                self.inter_plane_sse(&chroma, layout, plane, (x, y), (source, plane_sse[plane]));
        }
        chroma.distortion = self.weighted_chroma(distortion);
        Ok(chroma)
    }

    /// The type of the next luma transform of an inter block: `DCT_DCT`, or
    /// for random-decision blocks any type the frame's set allows at `tx`
    /// (the search, [`Self::refine_inter_types`], runs afterwards).
    fn random_inter_luma_type(&mut self, tx: TxSize) -> TxType {
        let allowed = self.options.tx_set.luma_types(tx, true);
        let lossless = self.lossless();
        match self.random.as_mut() {
            Some(random) if !allowed.is_empty() && !lossless => {
                allowed[random.below(allowed.len() as u64) as usize]
            }
            _ => TxType::DctDct,
        }
    }

    /// Retry each luma transform of an inter coding with the search's
    /// candidate types ([`InterTxSearch`]), keeping per transform the
    /// cheapest by its coefficient bits and squared error. Transforms are
    /// visited in coding order with the entropy contexts of the earlier
    /// choices, then the contexts are restored. Returns whether any type
    /// changed. Only the winning layout is searched, as for intra
    /// ([`Self::refine_intra_type`]).
    fn refine_inter_types(
        &mut self,
        coded: &mut Coded,
        layout: Layout,
        (x, y): (usize, usize),
        source: &Picture,
    ) -> Result<bool> {
        let tx = layout.tx(LUMA);
        let allowed = self.options.tx_set.luma_types(tx, true);
        let search = self.options.inter_tx;
        if self.random.is_some()
            || self.lossless()
            || !search.candidates.iter().any(|t| allowed.contains(t))
        {
            return Ok(false);
        }
        // The search compares each type with the DCT coding's samples.
        self.reconstruct_estimated(coded, layout, LUMA, (x, y))?;
        let size = 1 << layout.block_log2;
        let side = 1 << layout.plane_log2(LUMA);
        let (len, width) = (tx.coded_len(), tx.width());
        let area = width * width;
        let saved = self.state.save_coeff_contexts(x, y, size);
        let mut changed = false;
        // Taken so the trials can borrow the coder; errors end the encode.
        let mut level_buffer = std::mem::take(&mut self.type_trial_levels);
        let trial_levels = &mut level_buffer[..len];
        let mut pixel_buffer = std::mem::take(&mut self.type_best_pixels);
        let best_pixels = &mut pixel_buffer[..area];
        for (i, position) in layout.positions(LUMA, (x, y)).enumerate() {
            let offset = (position.0 - x, position.1 - y);
            let context = self.state.coeff_context(
                LUMA,
                position,
                tx,
                layout.plane_log2(LUMA),
                true,
                false,
            )?;
            let levels = &coded.levels[LUMA][i * len..(i + 1) * len];
            let nonzero = levels.iter().filter(|&&l| l != 0).count();
            if nonzero < search.min_nonzero.max(1) {
                self.state.update_coeff_context(LUMA, position, tx, levels);
                continue;
            }
            copy_from_block(&coded.pixels[LUMA], side, offset, width, best_pixels);
            let dct_distortion = sse(
                &self.quantizer.dsp,
                source,
                PlaneIndex::Y,
                position,
                width,
                best_pixels,
            );
            let mut best = (
                self.tx_rd_cost(levels, tx, context, TxType::DctDct, dct_distortion)?,
                TxType::DctDct,
                dct_distortion,
            );
            for &candidate in search.candidates.iter().filter(|t| allowed.contains(t)) {
                let predicted = &mut self.tx_prediction[..area];
                copy_from_block(&self.prediction[LUMA], side, offset, width, predicted);
                let reconstructed = &mut self.tx_reconstruction[..area];
                let trial_stats = self.quantizer.code(
                    (source, PlaneIndex::Y, position),
                    (tx, candidate),
                    &square_view(predicted),
                    (
                        trial_levels,
                        Output::Reconstruct(&mut square_view_mut(reconstructed)),
                    ),
                    None,
                    None,
                )?;
                if trial_stats.eob() == 0 {
                    continue;
                }
                let distortion = sse(
                    &self.quantizer.dsp,
                    source,
                    PlaneIndex::Y,
                    position,
                    width,
                    &self.tx_reconstruction[..area],
                );
                let cost = self.tx_rd_cost(trial_levels, tx, context, candidate, distortion)?;
                if cost < best.0 {
                    best = (cost, candidate, distortion);
                    coded.levels[LUMA][i * len..(i + 1) * len].copy_from_slice(trial_levels);
                    coded.stats[LUMA][i] = trial_stats;
                    best_pixels.copy_from_slice(&self.tx_reconstruction[..area]);
                }
            }
            if best.1 != coded.luma_types[i] {
                changed = true;
                coded.luma_types[i] = best.1;
                coded.distortion = coded.distortion - dct_distortion + best.2;
                copy_into_block(&mut coded.pixels[LUMA], side, offset, width, best_pixels);
            }
            let levels = &coded.levels[LUMA][i * len..(i + 1) * len];
            self.state.update_coeff_context(LUMA, position, tx, levels);
        }
        self.state.restore_coeff_contexts(&saved);
        self.type_trial_levels = level_buffer;
        self.type_best_pixels = pixel_buffer;
        Ok(changed)
    }

    /// Rate-distortion cost of one luma transform's coefficients coded as
    /// `tx_type` in `context`, with squared error `distortion`.
    fn tx_rd_cost(
        &mut self,
        levels: &[i32],
        tx: TxSize,
        context: coeff::Context,
        tx_type: TxType,
        distortion: u64,
    ) -> Result<f64> {
        let kind = self.coding_kind(tx_type);
        self.writer.start_estimate();
        let written = coeff::write_typed(&mut self.writer, levels, tx, context, self.qindex, kind);
        let bits = self.writer.finish_estimate();
        written.map(|_| bits * self.lambda() + distortion as f64)
    }

    /// A snapshot of the coefficient contexts of the block at `(x, y)` with
    /// `layout` when the trellis codes `plane` in several transforms: the
    /// plane's coding then updates them transform by transform, so that
    /// each transform is priced with the contexts its coding will see, and
    /// [`Self::end_trellis_contexts`] restores them (the block is only
    /// written once mode decision has chosen it). A single transform reads
    /// only its neighbours' contexts, which need no update.
    fn begin_trellis_contexts(
        &self,
        (x, y): (usize, usize),
        layout: Layout,
        plane: usize,
        quantisation: Quantisation,
    ) -> Option<SavedContexts> {
        let several = layout.plane_log2(plane) > layout.tx(plane).log2();
        let priced = self.prices_levels(plane, quantisation);
        (several && priced).then(|| self.state.save_coeff_contexts(x, y, 1 << layout.block_log2))
    }

    /// Whether a `quantisation` coding of `plane` runs the trellis.
    fn trellises(&self, plane: usize, quantisation: Quantisation) -> bool {
        quantisation == Quantisation::Final
            && self
                .quantizer
                .trellis
                .is_some_and(|t| t.of(PlaneIndex::ALL[plane]).is_some())
    }

    /// Whether a `quantisation` coding of `plane` prices its levels with
    /// the tile's models: the trellis's final coding, and every coding of
    /// TCQ luma (whose trials then weigh the levels that will be coded).
    fn prices_levels(&self, plane: usize, quantisation: Quantisation) -> bool {
        self.trellises(plane, quantisation) || self.searches_tcq(plane)
    }

    /// Whether `plane`'s levels are chosen by the TCQ search.
    fn searches_tcq(&self, plane: usize) -> bool {
        self.quantizer
            .tcq_search_of(PlaneIndex::ALL[plane])
            .is_some()
    }

    /// Restore the contexts [`Self::begin_trellis_contexts`] saved.
    fn end_trellis_contexts(&mut self, saved: Option<SavedContexts>) {
        if let Some(saved) = saved {
            self.state.restore_coeff_contexts(&saved);
        }
    }

    /// The coefficient contexts the trellis prices `plane`'s transform at
    /// `position` with, or `None` when that plane is not trellised. A V
    /// transform's contexts depend on whether the block's last U transform
    /// in `coded` has levels, as in [`Self::write_coefficients`].
    fn trellis_context(
        &self,
        coded: &Coded,
        layout: Layout,
        plane: usize,
        position: (usize, usize),
        (is_inter, quantisation): (bool, Quantisation),
    ) -> Result<Option<coeff::Context>> {
        if !self.prices_levels(plane, quantisation) {
            return Ok(None);
        }
        let u_coded = coded.stats[1].last().is_some_and(|stats| stats.eob() != 0);
        self.state
            .coeff_context(
                plane,
                position,
                layout.tx(plane),
                layout.plane_log2(plane),
                is_inter,
                u_coded,
            )
            .map(Some)
    }

    /// Quantize and reconstruct one plane of the current inter prediction
    /// into `coded`.
    fn code_inter_plane(
        &mut self,
        coded: &mut Coded,
        layout: Layout,
        plane: usize,
        (x, y): (usize, usize),
        (source, quantisation): (&Picture, Quantisation),
    ) -> Result<()> {
        let index = PlaneIndex::ALL[plane];
        let tx = layout.tx(plane);
        let side = 1 << layout.plane_log2(plane);
        let origin = plane_origin(plane, (x, y));
        let len = tx.coded_len();
        let set = self.options.tx_set;
        let estimate = self.estimates_distortion(quantisation, layout.tx(plane));
        let mut estimated_change = 0.0;
        let saved = self.begin_trellis_contexts((x, y), layout, plane, quantisation);
        for (i, position) in layout.positions(plane, (x, y)).enumerate() {
            let tx_type = match (plane, quantisation) {
                (LUMA, Quantisation::Plain) => self.random_inter_luma_type(tx),
                // The final coding keeps the type each transform won with.
                (LUMA, Quantisation::Final) => coded.luma_types[i],
                _ => TxType::DctDct,
            };
            let context =
                self.trellis_context(coded, layout, plane, position, (true, quantisation))?;
            // The transform reads its prediction from, and reconstructs into,
            // the block-sized rasters in place (no per-transform copies).
            let offset = (position.0 - origin.0, position.1 - origin.1);
            let prediction = block_window(&self.prediction[plane], side, offset, tx.width());
            let mut reconstructed =
                block_window_mut(&mut coded.pixels[plane], side, offset, tx.width());
            let levels = &mut coded.levels[plane][i * len..(i + 1) * len];
            let forward = ForwardSlot {
                state: &mut coded.forward[plane][i],
                coefficients: &mut coded.coefficients[plane][i * len..(i + 1) * len],
            };
            let output = if estimate {
                Output::Estimate(&mut estimated_change)
            } else {
                Output::Reconstruct(&mut reconstructed)
            };
            let stats = self.quantizer.code(
                (source, index, position),
                (tx, tx_type),
                &prediction,
                (levels, output),
                context.map(|context| RateSource {
                    writer: &mut self.writer,
                    context,
                    set,
                }),
                Some(forward),
            )?;
            if saved.is_some() {
                self.state.update_coeff_context(plane, position, tx, levels);
            }
            coded.stats[plane][i] = stats;
            if plane == LUMA {
                // All-zero transforms code no type and infer `DCT_DCT`.
                let zero = stats.eob() == 0;
                coded.luma_types[i] = if zero { TxType::DctDct } else { tx_type };
            }
        }
        self.end_trellis_contexts(saved);
        coded.estimated[plane] = estimate.then_some(estimated_change);
        Ok(())
    }

    /// Whether a `quantisation` coding of inter planes with transforms `tx`
    /// estimates its distortion instead of reconstructing
    /// (`av2.trial_distortion`): the plain trials only, with `DCT_DCT`
    /// transforms (random decisions pick other types) and not lossless.
    fn estimates_distortion(&self, quantisation: Quantisation, tx: TxSize) -> bool {
        let sizes = match self.options.trial_distortion {
            TrialDistortion::Reconstructed => false,
            // A 64-point transform codes only its low 32x32 frequencies and
            // is not an orthogonal projection: on noise the estimate is up to
            // ~10% low, which favours coding noise over skipping.
            TrialDistortion::EstimatedBelow64 => tx != TxSize::Tx64,
            TrialDistortion::Estimated => true,
        };
        sizes && quantisation == Quantisation::Plain && self.random.is_none() && !self.lossless()
    }

    /// Reconstruct `plane` of the inter coding `coded` (with `layout`, of
    /// the block at `(x, y)`) from its levels if its trial only estimated
    /// the distortion ([`BlockCoding::estimated`]).
    fn reconstruct_estimated(
        &mut self,
        coded: &mut Coded,
        layout: Layout,
        plane: usize,
        (x, y): (usize, usize),
    ) -> Result<()> {
        if coded.estimated[plane].take().is_none() {
            return Ok(());
        }
        let tx = layout.tx(plane);
        let side = 1 << layout.plane_log2(plane);
        let origin = plane_origin(plane, (x, y));
        let len = tx.coded_len();
        for (i, position) in layout.positions(plane, (x, y)).enumerate() {
            let offset = (position.0 - origin.0, position.1 - origin.1);
            let prediction = block_window(&self.prediction[plane], side, offset, tx.width());
            let mut reconstructed =
                block_window_mut(&mut coded.pixels[plane], side, offset, tx.width());
            let tx_type = if plane == LUMA {
                coded.luma_types[i]
            } else {
                TxType::DctDct
            };
            let index = PlaneIndex::ALL[plane];
            self.quantizer.reconstruct(
                (index, self.quantizer.dequantizer(index)),
                (tx, tx_type),
                (
                    &coded.levels[plane][i * len..(i + 1) * len],
                    coded.stats[plane][i],
                ),
                &prediction,
                &mut reconstructed,
            )?;
        }
        Ok(())
    }

    /// The squared error of `plane` of the inter coding `coded` (with
    /// `layout`) of the block at (`x`, `y`). Only the trials need it: the
    /// final coding comes after the mode decision. A plane without levels
    /// reconstructs as the prediction, whose error `prediction_sse` the
    /// coder already has (most trial planes code nothing).
    fn inter_plane_sse(
        &self,
        coded: &Coded,
        layout: Layout,
        plane: usize,
        (x, y): (usize, usize),
        (source, prediction_sse): (&Picture, u64),
    ) -> u64 {
        let measure = || {
            sse(
                &self.quantizer.dsp,
                source,
                PlaneIndex::ALL[plane],
                plane_origin(plane, (x, y)),
                1 << layout.plane_log2(plane),
                &coded.pixels[plane],
            )
        };
        if let Some(change) = coded.estimated[plane] {
            (prediction_sse as f64 + change).max(0.0).round() as u64
        } else if coded.stats[plane].iter().all(|stats| stats.eob() == 0) {
            debug_assert_eq!(prediction_sse, measure());
            prediction_sse
        } else {
            measure()
        }
    }

    /// Estimated bits of what `write` would code for the block at (`x`,
    /// `y`) of side `size`, priced from the current probabilities without
    /// coding anything ([`Writer::start_estimate`]). The coefficient
    /// contexts and CDEF index that `write` updates are restored.
    fn trial_bits(
        &mut self,
        x: usize,
        y: usize,
        size: usize,
        write: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<f64> {
        let saved = self.state.save_coeff_contexts(x, y, size);
        let cdef_index = self.cdef_indices.get(x, y);
        self.writer.start_estimate();
        let written = write(self);
        let bits = self.writer.finish_estimate();
        self.state.restore_coeff_contexts(&saved);
        self.cdef_indices.set(x, y, cdef_index);
        written.map(|()| bits)
    }

    /// `write_cdef` for the block at (`x`, `y`), after its skip flag: the
    /// first non-skipped block of a filter block holds its CDEF strength
    /// index, deferred until [`Self::cdef`] has chosen it. Lossless frames
    /// have no CDEF syntax.
    fn cdef_index(&mut self, (x, y): (usize, usize), skip: bool) {
        if self.lossless() || !self.cdef_indices.codes_index(x, y, skip) {
            return;
        }
        self.cdef_indices.set(x, y, Some(0));
        if matches!(self.options.cdef, CdefMode::Search(_)) && self.writer.defer() {
            self.deferred.push(DeferredSymbol::Cdef((x, y)));
        }
    }

    /// `write_ccso`, after the CDEF index: the unit's first block codes its
    /// flags, deferred until [`Self::ccso`] has chosen them.
    fn ccso_flags(&mut self, (x, y): (usize, usize)) {
        if self.options.ccso
            && !self.lossless()
            && UnitFlags::codes_flags(x, y)
            && self.writer.defer()
        {
            self.deferred.push(DeferredSymbol::Ccso((x, y)));
        }
    }

    /// The `skip_mode` flag of an inter-picture block, where the frame
    /// allows skip mode (AVM `read_skip_mode`, before `is_inter`).
    fn write_skip_mode(
        &mut self,
        (x, y): (usize, usize),
        size: usize,
        skip_mode: bool,
    ) -> Result<()> {
        if self.options.skip_mode.is_none() {
            return Ok(());
        }
        let context = self.state.skip_mode_context(x, y, size);
        self.writer
            .model(usize::from(skip_mode), &SKIP_MODE[context], |m| {
                &mut m.skip_mode[context]
            })
    }

    /// Write an inter block's syntax, with its residual or skipped; its
    /// motion signalling (inter mode, DRL index, vector, filter) only with
    /// [`MotionSyntax::Written`].
    fn write_inter(
        &mut self,
        block: &InterBlock,
        residual: Option<(&Coded, Layout)>,
        motion: MotionSyntax,
    ) -> Result<()> {
        let InterBlock {
            x,
            y,
            size,
            filter,
            signal,
        } = *block;
        let skip_mode = matches!(signal, Signal::Skip { .. });
        self.write_skip_mode((x, y), size, skip_mode)?;
        if !skip_mode {
            let context = self.state.intra_inter_context(x, y, size);
            self.writer
                .model(1, &IS_INTER[context], |m| &mut m.is_inter[context])?;
        }
        let skip_context = self.state.skip_context(x, y, size)
            + if skip_mode {
                SKIP_MODE_SKIP_CONTEXTS
            } else {
                0
            };
        self.writer
            .model(usize::from(residual.is_none()), &SKIP[skip_context], |m| {
                &mut m.block_skip[skip_context]
            })?;
        self.cdef_index((x, y), residual.is_none());
        self.ccso_flags((x, y));
        self.write_delta_q(size, residual.is_none())?;
        match signal {
            _ if motion == MotionSyntax::Omitted => {}
            Signal::Single {
                mv,
                mode,
                index,
                candidates,
                filter_context,
            } => {
                mv_refs::write_inter_mode(&mut self.writer, candidates.mode_context, mode)?;
                if mode.has_drl() {
                    mv_refs::write_drl(&mut self.writer, candidates, index)?;
                }
                if mode == InterMode::New {
                    let base = candidates.vectors[index];
                    motion::write_mv(&mut self.writer, mv.0 - base.0, mv.1 - base.1)?;
                }
                if let Some(context) = filter_context {
                    self.filter_symbols
                        .write(&mut self.writer, context, filter)?;
                }
            }
            Signal::Skip {
                index,
                max_drl_bits,
                ..
            } => mv_refs::write_skip_drl(&mut self.writer, max_drl_bits, index)?,
        }
        let Some((coded, layout)) = residual else {
            // A skipped block leaves all-zero entropy contexts (AVM
            // `av2_reset_entropy_context`).
            self.state.clear_coeff_contexts(x, y, size);
            return Ok(());
        };
        if self.lossless() {
            let group = lossless_size_group(layout.block_log2);
            self.writer
                .model(LOSSLESS_TX4_SYMBOL, &LOSSLESS_TX_SIZE, |m| {
                    &mut m.lossless_size[group][1]
                })?;
        } else {
            tx_partition::write(&mut self.writer, layout.block_log2, true, layout.split())?;
        }
        self.write_coefficients(coded, layout, (x, y), true)
    }

    /// Predict every plane of an inter block into `self.prediction` with
    /// the filter it will code, returning that filter and each plane's
    /// prediction error: the frame's filter, or in a switchable frame the
    /// searched one ([`Self::choose_filter`]; random-decision blocks code a
    /// random one) when the block codes its filter in `filter_context`.
    fn predict_inter(
        &mut self,
        reference: &Picture,
        source: &Picture,
        (x, y): (usize, usize),
        size: usize,
        mv: (i32, i32),
        filter_context: Option<usize>,
    ) -> Result<(InterpFilter, [u64; 3])> {
        let filter = match (self.options.filter, filter_context, self.random.as_mut()) {
            (FrameFilter::Fixed(filter), ..) => filter,
            (FrameFilter::Switchable, None, _) => interp_filter::UNCODED,
            (FrameFilter::Switchable, Some(_), Some(random)) => {
                let filters = &InterpFilter::SWITCHABLE;
                filters[random.below(filters.len() as u64) as usize]
            }
            (FrameFilter::Switchable, Some(context), None) => {
                return self.choose_filter(reference, source, (x, y), size, mv, context);
            }
        };
        let key = CachedLuma {
            position: (x, y),
            size,
            mv,
            filter,
        };
        if self.cached_luma.take() == Some(key) {
            // The cache becomes the luma prediction (no copy); the old
            // prediction buffer takes the next block's cache.
            std::mem::swap(&mut self.prediction[LUMA], &mut self.cached_luma_samples);
            self.prediction[LUMA].truncate(size * size);
            self.predict(
                reference,
                filter,
                (x, y),
                size,
                mv,
                &[PlaneIndex::U, PlaneIndex::V],
            )?;
        } else {
            self.predict(reference, filter, (x, y), size, mv, &PlaneIndex::ALL)?;
        }
        let plane_sse =
            prediction_distortion(&self.quantizer.dsp, source, (x, y), size, &self.prediction);
        Ok((filter, plane_sse))
    }

    /// Keep the engine's exact luma prediction of the block at `(x, y)` with
    /// vector `mv` and `filter` (tightly packed at block width) for the
    /// block's coding. Exact: the engine predicts with the same kernel
    /// ([`InterpFilter::kernel`]) from the same reference, and a key
    /// mismatch predicts afresh. The coder is rebuilt every frame, so the
    /// cache never outlives its reference.
    pub fn cache_luma(
        &mut self,
        (x, y): (usize, usize),
        size: usize,
        mv: (i32, i32),
        filter: InterpFilter,
        samples: &[u8],
    ) {
        self.cached_luma_samples.clear();
        self.cached_luma_samples.extend_from_slice(samples);
        self.cached_luma = Some(CachedLuma {
            position: (x, y),
            size,
            mv,
            filter,
        });
    }

    /// The switchable filter of an inter block coding its filter in
    /// `context` ([`FilterSearch`]), with every plane predicted by it into
    /// `self.prediction` and their errors. A vector that is whole-sample in
    /// every plane predicts alike with every filter, so it takes the
    /// cheapest symbol; one that is whole-sample in luma only keeps regular
    /// (only its chroma could differ: not worth a search).
    fn choose_filter(
        &mut self,
        reference: &Picture,
        source: &Picture,
        (x, y): (usize, usize),
        size: usize,
        mv: (i32, i32),
        context: usize,
    ) -> Result<(InterpFilter, [u64; 3])> {
        self.predict(
            reference,
            InterpFilter::Regular,
            (x, y),
            size,
            mv,
            &PlaneIndex::ALL,
        )?;
        let mut plane_sse =
            prediction_distortion(&self.quantizer.dsp, source, (x, y), size, &self.prediction);
        let bits = self.filter_symbols.bits(&self.writer, context);
        let symbol_cost = |filter: InterpFilter| {
            let symbol = InterpFilter::SWITCHABLE
                .iter()
                .position(|&f| f == filter)
                .expect("switchable candidates");
            f64::from(bits[symbol])
        };
        if mv.0 % CHROMA_WHOLE_SAMPLE == 0 && mv.1 % CHROMA_WHOLE_SAMPLE == 0 {
            let cheapest = InterpFilter::SWITCHABLE
                .into_iter()
                .min_by(|&a, &b| symbol_cost(a).total_cmp(&symbol_cost(b)))
                .expect("switchable filters");
            return Ok((cheapest, plane_sse));
        }
        if mv.0 % LUMA_WHOLE_SAMPLE == 0 && mv.1 % LUMA_WHOLE_SAMPLE == 0 {
            return Ok((InterpFilter::Regular, plane_sse));
        }
        let lambda = self.lambda();
        let search = self.options.filter_search;
        let dsp = &self.quantizer.dsp;
        let block = BlockSize::new(size, size).expect("square block sizes are kernel sizes");
        // The psychovisual term needs the source's energy once per block.
        let source_energy = (search.psy_strength > 0.0).then(|| {
            let view = source
                .plane(PlaneIndex::Y)
                .view()
                .sub_view(x, y, size, size)
                .expect("blocks lie inside the picture");
            nc_encoder::ac_energy(dsp, block, &view)
        });
        let distortion = |sse: u64, prediction: &[u8]| {
            let psy = source_energy.map_or(0.0, |energy| {
                let view = PlaneView::new(prediction, size, size, size).expect("square raster");
                let predicted = nc_encoder::ac_energy(dsp, block, &view);
                f64::from(nc_encoder::psy_penalty(
                    search.psy_strength,
                    energy,
                    predicted,
                ))
            });
            sse as f64 + psy
        };
        let regular_distortion = distortion(plane_sse[LUMA], &self.prediction[LUMA]);
        let mut best = (
            InterpFilter::Regular,
            plane_sse[LUMA],
            regular_distortion + lambda * symbol_cost(InterpFilter::Regular),
        );
        for &candidate in search.candidates {
            let extra_rate = lambda * (symbol_cost(candidate) - symbol_cost(InterpFilter::Regular));
            if regular_distortion * search.max_gain <= extra_rate {
                continue;
            }
            let trial = &mut self.filter_trial[..size * size];
            prediction::predict(
                &self.quantizer.dsp,
                candidate,
                reference.plane(PlaneIndex::Y),
                x,
                y,
                mv,
                0,
                trial,
            )?;
            let sse = sse(
                &self.quantizer.dsp,
                source,
                PlaneIndex::Y,
                (x, y),
                size,
                trial,
            );
            let cost = distortion(sse, trial) + lambda * symbol_cost(candidate);
            if cost < best.2 {
                best = (candidate, sse, cost);
                self.prediction[LUMA].copy_from_slice(trial);
            }
        }
        let (filter, luma_sse, _) = best;
        if filter != InterpFilter::Regular {
            self.predict(reference, filter, (x, y), size, mv, &CHROMA_PLANES)?;
            plane_sse =
                prediction_distortion(&self.quantizer.dsp, source, (x, y), size, &self.prediction);
            debug_assert_eq!(plane_sse[LUMA], luma_sse);
        }
        Ok((filter, plane_sse))
    }

    /// Prediction of the block's `planes` into `self.prediction` with
    /// `filter`.
    fn predict(
        &mut self,
        reference: &Picture,
        filter: InterpFilter,
        (x, y): (usize, usize),
        size: usize,
        mv: (i32, i32),
        planes: &[PlaneIndex],
    ) -> Result<()> {
        for &index in planes {
            let plane = index as usize;
            let subsample = u8::from(plane != 0);
            let side = size >> subsample;
            let (px, py) = plane_origin(plane, (x, y));
            let samples = &mut self.prediction[plane];
            samples.resize(side * side, 0);
            prediction::predict(
                &self.quantizer.dsp,
                filter,
                reference.plane(index),
                px,
                py,
                mv,
                subsample,
                samples,
            )?;
        }
        Ok(())
    }

    /// Deblock the finished reconstruction `recon` of this tile with the
    /// caller's (clear) edge map `map`, which it leaves clear.
    pub fn deblock(
        &self,
        recon: &mut Picture,
        params: DeblockParams,
        map: &mut crate::deblock::EdgeMap,
    ) {
        crate::deblock::filter_picture(recon, &self.state.edges, params, &self.quantizer.dsp, map);
    }

    /// Choose this tile's CDEF parameters (and, in search mode, each filter
    /// block's strength index) and filter the deblocked reconstruction
    /// `recon` with them. Random-decision tiles take random choices instead.
    pub fn cdef(
        &mut self,
        source: &Picture,
        recon: &mut Picture,
        intra: bool,
        scratch: &mut cdef::CdefScratch,
    ) -> CdefParams {
        if self.lossless() {
            return CdefParams::OFF;
        }
        // Frame-level filter: the frame's λ, whatever the last block's was.
        let lambda = self.lambda_at(self.qindex);
        let dsp = &self.quantizer.dsp;
        let step = cdef_model_step(self.qindex);
        self.cdef_params = match (self.random.as_mut(), self.options.cdef) {
            (Some(random), mode) => {
                let search = matches!(mode, CdefMode::Search(_));
                let params = cdef::random_params(random, search);
                if search {
                    cdef::random_indices(random, params, &mut self.cdef_indices);
                }
                cdef::filter_picture(
                    recon,
                    &self.state,
                    &self.cdef_indices,
                    params,
                    dsp,
                    scratch.unsearched_source(),
                );
                params
            }
            (None, CdefMode::Model(check_row_period)) => cdef::choose_and_filter(
                source,
                recon,
                (&self.state, &self.cdef_indices),
                (step, intra, check_row_period),
                dsp,
                scratch,
            ),
            (None, CdefMode::Search(options)) => cdef::search_and_filter(
                source,
                recon,
                &self.state,
                &mut self.cdef_indices,
                step,
                intra,
                options,
                lambda,
                dsp,
                scratch,
            ),
        };
        self.cdef_params
    }

    /// Choose this tile's CCSO filters ([`ccso_search::choose`], or random
    /// ones for random-decision tiles) from the CDEF-filtered
    /// reconstruction `recon` and `source`, and filter `recon`, classifying
    /// on `deblocked`, the luma before CDEF.
    pub fn ccso(
        &mut self,
        source: &Picture,
        recon: &mut Picture,
        deblocked: &CcsoSource,
    ) -> CcsoParams {
        if !self.options.ccso || self.lossless() {
            return CcsoParams::OFF;
        }
        self.ccso_params = if let Some(random) = self.random.as_mut() {
            let params = ccso::random_params(random);
            ccso::random_flags(random, &params, &mut self.ccso_flags);
            params
        } else {
            let lambda = self.lambda_at(self.qindex);
            ccso_search::choose(
                source,
                recon,
                deblocked,
                &mut self.ccso_flags,
                lambda,
                self.chroma_weight,
            )
        };
        ccso::filter_picture(recon, deblocked, &self.ccso_params, &self.ccso_flags);
        self.ccso_params
    }

    /// Finish the tile's entropy coding, with the CDEF strength indices
    /// and CCSO flags coded where they were deferred.
    pub fn finish(self) -> FinishedTile {
        let cdef_params = self.cdef_params;
        let ccso_params = self.ccso_params;
        let mut defaults = (cdef_params.codes_indices() || ccso_params.is_on())
            .then(|| Box::new(TileCdfs::new(self.qindex)));
        let mut deferred = self.deferred.into_iter();
        let (indices, flags) = (&self.cdef_indices, &self.ccso_flags);
        let (bytes, models) = self
            .writer
            .replay(|writer| {
                let symbol = deferred.next().expect("every deferred symbol was listed");
                let Some(defaults) = defaults.as_deref_mut() else {
                    return Ok(());
                };
                match symbol {
                    DeferredSymbol::Cdef(position) => {
                        cdef::write_index(writer, cdef_params, indices, position, defaults)
                    }
                    DeferredSymbol::Ccso(position) if ccso_params.is_on() => {
                        ccso::write_flags(writer, &ccso_params, flags, position, defaults)
                    }
                    DeferredSymbol::Ccso(_) => Ok(()),
                }
            })
            .expect("chosen filter symbols are codable");
        FinishedTile {
            bytes,
            models,
            random: self.random,
        }
    }
}

/// A coded tile.
pub struct FinishedTile {
    /// The tile's entropy-coded bytes.
    pub bytes: Vec<u8>,
    /// Its final CDFs, when it adapts (for later frames' cross-frame
    /// context).
    pub models: Option<Box<TileCdfs>>,
    /// The random decision source, lent back for the next frame.
    pub random: Option<DebugRng>,
}

/// An inter block's signalling, fixed before residual trials.
#[derive(Debug, Clone, Copy)]
struct InterBlock {
    x: usize,
    y: usize,
    size: usize,
    /// The block's interpolation filter.
    filter: InterpFilter,
    signal: Signal,
}

/// How an inter block's prediction is signalled.
#[derive(Debug, Clone, Copy)]
enum Signal {
    /// A single-reference vector.
    Single {
        mv: (i32, i32),
        mode: InterMode,
        index: usize,
        candidates: Candidates,
        /// The context its filter symbol is coded in; `None` when it codes
        /// none (a fixed-filter frame, or a mode that codes no filter).
        filter_context: Option<usize>,
    },
    /// Skip mode: entry `index` of the skip-mode list.
    Skip {
        index: usize,
        pair: mv_refs::Pair,
        max_drl_bits: u8,
    },
}

impl InterBlock {
    /// The motion the block leaves for later reference lists.
    fn motion(&self) -> BlockMotion {
        match self.signal {
            Signal::Single { mv, mode, .. } => BlockMotion::Single {
                vector: mv,
                is_new: mode == InterMode::New,
            },
            Signal::Skip { pair, .. } => BlockMotion::Compound(pair),
        }
    }
}

/// The transforms a skipped inter block implies: the largest of the block
/// (AVM `tx_size_from_tx_mode`), or 4×4 when lossless (`read_tx_size`).
pub(crate) fn skipped_layout(block_log2: u32, lossless: bool) -> Layout {
    if lossless {
        Layout::lossless(block_log2)
    } else {
        Layout::lossy(block_log2, false)
    }
}

/// Deblocking record of the block at `position` coded with `layout`.
pub(crate) fn block_edges(
    (x, y): (usize, usize),
    qindex: PlaneQindex,
    layout: Layout,
    skipped_inter: bool,
) -> BlockEdges {
    let narrow = |v: usize| u16::try_from(v).expect("picture coordinates fit u16");
    BlockEdges {
        x: narrow(x),
        y: narrow(y),
        size: 1 << layout.block_log2,
        luma_tx: layout.luma,
        chroma_tx: layout.chroma,
        skipped_inter,
        luma_qindex: qindex.luma.ac,
        chroma_qindex: qindex.chroma.ac,
    }
}

/// The cheapest way to signal `mv`: NEARMV when it is a DRL candidate,
/// GLOBALMV for zero, otherwise NEWMV from the nearest candidate.
pub(crate) fn choose_signalling(candidates: Candidates, mv: (i32, i32)) -> (InterMode, usize) {
    let mut best = if mv == (0, 0) {
        Some((InterMode::Global, 0))
    } else {
        None
    };
    for (index, &vector) in candidates.derived().iter().enumerate() {
        if vector == mv {
            let near = (InterMode::Near, index);
            best = Some(match best {
                Some(current)
                    if mv_refs::mode_bits(candidates, current.0, current.1)
                        <= mv_refs::mode_bits(candidates, near.0, near.1) =>
                {
                    current
                }
                _ => near,
            });
        }
    }
    best.unwrap_or_else(|| {
        let index = (0..candidates.derived().len())
            .min_by_key(|&i| {
                let v = candidates.vectors[i];
                (mv.0 - v.0).abs() + (mv.1 - v.1).abs()
            })
            .unwrap_or(0);
        (InterMode::New, index)
    })
}

/// The trellis λ per plane kind of a frame whose mode-decision λ is
/// `lambda`, with chroma squared error weighted by `chroma_weight`
/// ([`crate::chroma_weight`]). With the step-derived weight, the chroma λ
/// scales with the chroma step squared, as x264 and HM scale chroma λ by
/// the chroma QP offset (nc-av1's trellis, where the luma λ crushed chroma:
/// −2.9 dB U on crowd).
fn trellis_lambdas(options: TrellisOptions, lambda: f64, chroma_weight: f64) -> TrellisLambdas {
    let luma = lambda as f32 * options.lambda_scale;
    TrellisLambdas {
        luma,
        chroma: options
            .chroma_scale
            .map(|scale| luma * scale / chroma_weight as f32),
        max_states: options.max_states,
    }
}

/// The end of block of `plane`'s quantized `levels` of a `(tx, tx_type)`
/// transform at `qindex` after making them decodable under the frame's
/// parity hiding at λ `hiding` ([`coeff::parity::conform`]; luma of lossy
/// frames only, unchanged otherwise).
fn conform_parity(
    hiding: Option<f64>,
    (plane, qindex): (PlaneIndex, transform::Qindices),
    (tx, tx_type): (TxSize, TxType),
    (levels, coefficients): (&mut [i32], &[i32]),
    eob: usize,
) -> usize {
    match hiding {
        Some(lambda) if plane == PlaneIndex::Y && qindex.ac != 0 => coeff::parity::conform(
            (levels, coefficients),
            tx,
            (tx_type, eob),
            lambda,
            (qindex.dc, qindex.ac),
        ),
        _ => eob,
    }
}

/// Quantizer step in reconstructed-sample units (AV2 transforms carry an
/// extra factor of eight over the Q3 table).
pub(crate) fn pixel_step(qindex: u8) -> f64 {
    const Q3_PER_SAMPLE: f64 = 64.0;
    f64::from(transform::quantizer_step_q3(qindex)) / Q3_PER_SAMPLE
}

/// The AC step the CDEF strength model reads, on AV1's `ac_qlookup` scale:
/// the Q3 step over eight, the scale rate control also uses.
fn cdef_model_step(qindex: u8) -> f32 {
    const Q3_PER_AV1_STEP: f32 = 8.0;
    transform::quantizer_step_q3(qindex) as f32 / Q3_PER_AV1_STEP
}

/// Plane coordinates of a luma position.
pub(crate) fn plane_origin(plane: usize, (x, y): (usize, usize)) -> (usize, usize) {
    let shift = usize::from(plane != 0);
    (x >> shift, y >> shift)
}

/// Clamped sum of a prediction and a reconstructed residual.
pub(crate) fn add_residual(predicted: &[u8], residual: &[i32], out: &mut [u8]) {
    // Equal-length slices indexed together vectorize; a three-way zip did
    // not (about 1% of the encoder's instructions). The sum runs in 16-bit
    // lanes, because baseline x86-64 has no 32-bit minimum or maximum but
    // has saturating 16-bit packs and adds: the residual saturates to
    // `i16`, the sum saturates, and the clamp to a sample is then the same
    // as clamping the exact 32-bit sum.
    let n = out.len();
    let (predicted, residual) = (&predicted[..n], &residual[..n]);
    for i in 0..n {
        let residual = residual[i].clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        out[i] = i16::from(predicted[i])
            .saturating_add(residual)
            .clamp(0, i16::from(u8::MAX)) as u8;
    }
}

/// [`add_residual`] of a residual that is `residual` at every sample (a DC
/// only transform, [`transform::dc_only_residual`]), `-256..=255`.
pub(crate) fn add_constant_residual(predicted: &[u8], residual: i32, out: &mut [u8]) {
    let n = out.len();
    let predicted = &predicted[..n];
    let residual = residual.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    for i in 0..n {
        out[i] = i16::from(predicted[i])
            .saturating_add(residual)
            .clamp(0, i16::from(u8::MAX)) as u8;
    }
}

/// Copy a `side`×`side` raster `samples` into `plane` of `picture` at `(x, y)`.
pub(crate) fn store(
    picture: &mut Picture,
    plane: PlaneIndex,
    (x, y): (usize, usize),
    side: usize,
    samples: &[u8],
) {
    let mut view = picture.plane_mut(plane).view_mut();
    assert!(
        x + side <= view.width() && y + side <= view.height(),
        "stored blocks lie inside the plane"
    );
    // Rows by strided chunks: one check for the block, not one per row
    // (most stored blocks are a few samples wide).
    let stride = view.stride();
    let rows = view.data_mut()[y * stride + x..].chunks_mut(stride);
    for (row, values) in rows.zip(samples.chunks_exact(side)) {
        nc_common::copy_row(&mut row[..side], values);
    }
}

/// Store every plane of a coded block.
fn store_block(recon: &mut Picture, (x, y): (usize, usize), size: usize, coded: &Coded) {
    for (plane, index) in PlaneIndex::ALL.into_iter().enumerate() {
        let side = size >> usize::from(plane != 0);
        store(
            recon,
            index,
            plane_origin(plane, (x, y)),
            side,
            &coded.pixels[plane],
        );
    }
}

/// Copy a `tx`×`tx` raster into a block raster of side `side` at `offset`.
pub(crate) fn copy_into_block(
    block: &mut [u8],
    side: usize,
    (ox, oy): (usize, usize),
    tx: usize,
    samples: &[u8],
) {
    for (row, values) in samples.chunks_exact(tx).enumerate() {
        let start = (oy + row) * side + ox;
        nc_common::copy_row(&mut block[start..start + tx], values);
    }
}

/// A square raster (`samples.len()` a square) as a view.
fn square_view(samples: &[u8]) -> PlaneView<'_> {
    let side = samples.len().isqrt();
    PlaneView::new(samples, side, side, side).expect("a square raster")
}

/// [`square_view`], writable.
fn square_view_mut(samples: &mut [u8]) -> PlaneViewMut<'_> {
    let side = samples.len().isqrt();
    PlaneViewMut::new(samples, side, side, side).expect("a square raster")
}

/// The `tx`×`tx` window at `offset` of a block raster of side `side`.
fn block_window(block: &[u8], side: usize, (ox, oy): (usize, usize), tx: usize) -> PlaneView<'_> {
    PlaneView::new(&block[oy * side + ox..], side, tx, tx).expect("windows lie inside the block")
}

/// [`block_window`], writable.
fn block_window_mut(
    block: &mut [u8],
    side: usize,
    (ox, oy): (usize, usize),
    tx: usize,
) -> PlaneViewMut<'_> {
    PlaneViewMut::new(&mut block[oy * side + ox..], side, tx, tx)
        .expect("windows lie inside the block")
}

/// Copy a `tx`×`tx` window at `offset` out of a block raster of side `side`.
fn copy_from_block(block: &[u8], side: usize, (ox, oy): (usize, usize), tx: usize, out: &mut [u8]) {
    for (row, values) in out.chunks_exact_mut(tx).enumerate() {
        let start = (oy + row) * side + ox;
        nc_common::copy_row(values, &block[start..start + tx]);
    }
}

/// Squared error of a `side`×`side` raster against `plane` of `source` at `(x, y)`.
fn sse(
    dsp: &Dsp,
    source: &Picture,
    plane: PlaneIndex,
    (x, y): (usize, usize),
    side: usize,
    samples: &[u8],
) -> u64 {
    let size = BlockSize::new(side, side).expect("square block sizes are kernel sizes");
    let source = source
        .plane(plane)
        .view()
        .sub_view(x, y, side, side)
        .expect("blocks lie inside the picture");
    let samples = PlaneView::new(samples, side, side, side).expect("square raster");
    u64::from(dsp.variance(size, &source, &samples).sse)
}

/// Squared error of the prediction alone (the skipped block), per plane.
fn prediction_distortion(
    dsp: &Dsp,
    source: &Picture,
    (x, y): (usize, usize),
    size: usize,
    prediction: &[Vec<u8>; 3],
) -> [u64; 3] {
    std::array::from_fn(|plane| {
        sse(
            dsp,
            source,
            PlaneIndex::ALL[plane],
            plane_origin(plane, (x, y)),
            size >> usize::from(plane != 0),
            &prediction[plane],
        )
    })
}

/// Copy the chroma levels of `chroma` into `coded`: all that pricing a
/// trial coding reads.
fn copy_chroma_levels(coded: &mut Coded, chroma: &Coded) {
    for plane in CHROMA {
        coded.levels[plane].clone_from(&chroma.levels[plane]);
        coded.stats[plane].clone_from(&chroma.stats[plane]);
    }
}

/// Copy the chroma samples and forward transforms of `chroma` into
/// `coded`, which [`copy_chroma_levels`] gave its levels: the rest of a
/// coding that is kept.
fn copy_chroma_reconstruction(coded: &mut Coded, chroma: &Coded) {
    for plane in CHROMA {
        coded.pixels[plane].clone_from(&chroma.pixels[plane]);
        coded.forward[plane].clone_from(&chroma.forward[plane]);
        coded.coefficients[plane].clone_from(&chroma.coefficients[plane]);
    }
}
