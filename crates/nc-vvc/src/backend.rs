//! VVC backend callbacks for the ordinary shared-engine partition walk.
use crate::coding_tree::{Allowed, TreeSplit};
use crate::{
    cabac::CabacWriter,
    chroma_qp::ChromaQpTable,
    contexts::Contexts,
    headers, inter,
    inter_intra::InterIntraSearch,
    motion::{Motion, MotionGrid, Mv16},
    recon, syntax, transform_tree,
};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::{Dsp, Filter};
use nc_encoder::{
    Block, BlockDecision, Codec, CodecBackend, ConfigError, EncoderConfig, FramePlan, FrameType,
    Mv, PartitionShape, PredictionMode, ShapeSet, TimingSignal,
};

mod alf_search;
mod block_qp;
mod motion_choice;

use alf_search::{AlfSearch, AlfSearchParams, AlfSkip};

nc_encoder::param_group! {
    /// Directional intra search budget for the VVC backend.
    pub struct VvcParams("vvc") {
        chroma_search: 1.0, [0.0, 1.0], "Enable full RD selection among derived and four explicit chroma directions",
        chroma_search_inter: 1.0, [0.0, 1.0],
            "with vvc.chroma_search, 1 also searches the chroma direction of P-picture intra CUs (0: key pictures only): more search",
        chroma_rd_candidates: 0.0, [0.0, 5.0],
            "with vvc.chroma_search, full RD only for this many chroma directions, the cheapest by the SATD of both planes' predictions plus √λ times their bits (VVenC's chroma pre-selection; 0: all five): lower is faster, coarser chroma",
        // 3 from a step-4 sweep vs 8 from step 2 on key frames: key frames
        // -32% instructions; staged TRAIN (k 0.1) J full +0.1%, low +0.0%,
        // J_real -0.2%, no guard, periodic-key pulse -0.03 dB.
        angular_candidates: 3.0, [0.0, 63.0], "Additional angular modes retained by shared SATD shortlist for full RD; 0 keeps four core modes, 63 is exhaustive",
        angular_step: 4.0, [1.0, 8.0],
            "the SATD shortlist's first pass visits every Nth angular direction (a power of two), then refines around the kept ones at half the distance down to one: higher is faster",
        // P-frame intra CUs mostly repair occlusions and brightness changes,
        // where few directions matter: 3 candidates from a step-4 sweep vs
        // 8 from step 2, staged TRAIN (k 0.1): P frames -7.6% instructions,
        // quality +0.1% (proxy-6 J full +0.1%, low +0.2%), J_real -0.6%, no
        // guard, periodic-key pulse 0.00 dB.
        inter_angular_candidates: 3.0, [0.0, 63.0],
            "vvc.angular_candidates for intra CUs in P frames",
        inter_angular_step: 4.0, [1.0, 8.0],
            "vvc.angular_step for intra CUs in P frames",
        rd_candidates: 0.0, [0.0, 16.0],
            "full-RD directions per key-frame intra leaf: the cheapest this many by the SATD proxy among the core modes and the angular shortlist (0: all of them): lower is faster",
        rd_proxy_margin: 0.0, [0.0, 2000.0],
            "full-RD directions per key-frame intra leaf are only those whose SATD proxy is within this many ‰ of the cheapest (0: no margin, every shortlisted direction): lower is faster",
        inter_rd_candidates: 0.0, [0.0, 16.0],
            "vvc.rd_candidates for intra CUs in P frames",
        intra_rounding: recon::INTRA_ROUNDING_512THS, [0.0, 256.0],
            "intra quantiser rounding offset (1/512 step, HM 171; 256 rounds to nearest): lower widens the dead zone",
        inter_rounding: recon::INTER_ROUNDING_512THS, [0.0, 256.0],
            "inter luma quantiser rounding offset (1/512 step; HEVC's 128, HM 85): lower widens the dead zone",
        chroma_inter_rounding: recon::INTER_ROUNDING_512THS, [0.0, 256.0],
            "inter chroma quantiser rounding offset (1/512 step): higher keeps more colour detail",
        // HEVC Default's 4: there, +0.5% / +0.75% PSNR-YUV / VMAF BD vs 2
        // for -2.7% instructions; SAO statistics are 18% of a VVC P frame
        // at every row (pan_jaad_0101 300 kbps, callgrind).
        // Default off until screened (docs/handoff/vvc.md "vvc-quality3").
        dep_quant: 0.0, [0.0, 1.0],
            "1 codes every slice with dependent quantisation (sps_dep_quant_enabled_flag): two interleaved quantisers chosen by a four-state search, instead of the trellis and sign hiding: finer reconstruction for the bits, more search",
        dep_quant_lambda_scale: 1.0, [0.25, 2.0],
            "with vvc.dep_quant, multiple of the mode-decision λ its search trades bits at: higher drops more levels",
        dep_quant_chroma_scale: 0.25, [0.0, 2.0],
            "with vvc.dep_quant, a further multiple of that λ for chroma (the plain chroma quantiser keeps a fine dead zone): lower keeps more colour",
        dep_quant_key_scale: 0.5, [0.0, 2.0],
            "with vvc.dep_quant, a further multiple of that λ on key pictures (as vvc.trellis_key_lambda_scale)",
        // Default off until screened (docs/handoff/vvc.md "ISP").
        isp: 0.0, [0.0, 1.0],
            "1 enables intra sub-partitions (sps_isp_enabled_flag): intra CUs from 8x8 may code their luma as four strips or columns, each predicted from the one before: sharper intra, a second RD pass per such CU",
        isp_inter: 0.0, [0.0, 1.0],
            "with vvc.isp, 1 also tries sub-partitions on intra CUs of P pictures (they still code the flag): more search",
        isp_modes: 1.0, [1.0, 2.0],
            "with vvc.isp, the whole-CU directions tried as sub-partitions: 1 the best, 2 also the runner-up (more search)",
        isp_residual_gate: 0.0, [0.0, 1.0],
            "with vvc.isp, 1 tries sub-partitions only on CUs whose best whole-CU direction codes a luma residual (VVenC's fast ISP: sub-partitions must code one): faster key pictures, a few CUs lose ISP",
        // Default off until screened (docs/handoff/vvc.md "vvc-quality4").
        affine: 0.0, [0.0, 2.0],
            "affine motion (sps_affine_enabled_flag): 0 off, 1 4-parameter models (zoom and rotation), 2 also 6-parameter; CUs from 8x8 may merge an affine model from their neighbours and CUs from 16x16 may code one, predicted per 4x4 sub-block: sharper motion, more search",
        affine_merge: 1.0, [0.0, 1.0],
            "with vvc.affine, 1 tries the sub-block merge candidates of every inter CU from 8x8: more search",
        affine_amvp: 1.0, [0.0, 1.0],
            "with vvc.affine, 1 fits an affine model to inter CUs from 16x16 (Gauss-Newton steps on the prediction's gradients): more search",
        affine_iterations: 4.0, [1.0, 8.0],
            "with vvc.affine_amvp, Gauss-Newton steps per start model: higher fits better, more search",
        // 0.4 keeps 68% of the fitted models' SATD gain (weighted by their
        // CUs' cost) at 11% of the inter area (bodycam_bc1, antare,
        // airport, crowd_mot17_09, 30 f at 300 kbps; docs/handoff/vvc.md).
        affine_amvp_gate: 0.4, [0.0, 4.0],
            "with vvc.affine_amvp, fit affine models only to CUs whose vector predicts them worse than this luma SATD per sample over √λ: lower fits more CUs, more search",
        // 0.1 keeps 90% of the merged models' SATD gain (weighted as for
        // the AMVP gate) at 83% of the candidate area.
        affine_merge_gate: 0.1, [0.0, 4.0],
            "with vvc.affine_merge, try sub-block merge candidates only on CUs whose vector predicts them worse than this luma SATD per sample over √λ: lower tries more CUs, more search",
        alf: 0.0, [0.0, 1.0],
            "1 enables the luma adaptive loop filter: per CTB off, one of the 16 fixed filter sets or (vvc.alf_aps) the picture's signalled filters, chosen by squared error and rate: sharper, more filter work",
        sao_luma_lambda_scale: 1.0, [0.25, 16.0],
            "λ multiple for luma SAO decisions (HEVC's hevc.sao_luma_lambda_scale): higher applies fewer luma offsets (less smoothing)",
        sao_stats_row_step: 4.0, [1.0, 4.0],
            "sample every Nth row for SAO statistics; higher trades decision precision for speed",
        sao_search_interval: 1.0, [1.0, 8.0],
            "SAO classes are searched on key pictures and every Nth P picture (by POC) and reused in between, offsets re-derived (HEVC's search interval 4; with vvc.sao_skip_unchanged, all-skip CTUs keep their last classes): higher gathers a quarter of the statistics more often, coarser SAO",
        sao_skip_unchanged: 0.0, [0.0, 1.0],
            "1 switches SAO off without statistics in P-frame CTUs whose CUs are all skipped (they copy already filtered samples; HEVC's hevc.sao_skip_unchanged): faster",
        // On everywhere it tripped the worst-1% NEG guard (proxy-6 +3.6,
        // bodycam_antare_0815 +14.7%: its fast-motion P frames are coded
        // mostly intra). With the intra-dense gate (`inter_intra`), on the
        // tuned Default (7acc741f), staged TRAIN J_real at k 0.1: BETTER,
        // -0.5% (quality full +0.2% / low +0.1%, worst-1% NEG +0.4%, antare
        // +2.3%), no guard, periodic-key pulse +0.02 dB; VAL BETTER, -0.6%
        // (full +0.2% / low +0.1%); long GOP no drift. CPU x86 -3.8%,
        // Jetson -8.1% / -9.0% (crowd_mot17_09 / bodycam_bc1, CBR 300).
        coarse_inter_intra: 1.0, [0.0, 1.0],
            "1 gives P-frame intra CUs full RD over planar, DC, pure horizontal/vertical and the MPMs only (no angular sweep) outside intra-dense pictures: faster, coarser intra",
        // Skips the full RD (quantise, reconstruct, price) of the coarse
        // set's directions the SATD proxy ranks below the N cheapest: in
        // pictures that are not intra-dense, P intra CUs mostly repair
        // occlusions and brightness changes, where the proxy's order
        // rarely loses the RD winner by much (HEVC Default keeps one).
        // Jetson cycles (CBR 600, 150 f, crowd / bodycam / static): 1
        // -3.9 / -4.0 / -1.4%, 2 -3.4 / -3.2 / -1.3%. Default 2 (`DEFAULT_PARAMS`).
        coarse_rd_candidates: 0.0, [0.0, 10.0],
            "with vvc.coarse_inter_intra, full RD only for this many of the core modes and MPMs, the cheapest by the SATD proxy (HEVC Default codes its single cheapest; 0: all of them): lower is faster, coarser intra",
        inter_intra_dc_only: 0.0, [0.0, 1.0],
            "1 codes P-frame intra CUs outside intra-dense pictures with DC and the derived chroma direction, without a search (HEVC's hevc.inter_intra_dc_only; overrides vvc.coarse_inter_intra): faster, coarser intra",
        // 0.3: see `inter_intra` (worst frames 56-99% intra, crowd and
        // bodycam_bc1 P pictures below 30%).
        coarse_intra_dense_share: 0.3, [0.0, 1.0],
            "with vvc.coarse_inter_intra or vvc.inter_intra_dc_only, P-frame intra CUs keep the ranked search once intra CUs cover more than this share of the picture so far (or of the previous P picture): lower protects more fast-motion pictures and saves less; 1 never",
        pre_transform_zero_scale: recon::PRE_TRANSFORM_ZERO_SCALE, [1.0, 4.0],
            "inter residual SAD zero-bound multiplier before transform: 1 exact, higher skips near-zero detail for speed; intra unchanged",
        fast_forward: 0.0, [0.0, 1.0],
            "1 computes 16- and 32-point forward DCT rows with staged butterflies and skips proven-zero column groups (HEVC Default's hevc.fast_forward16/32): faster, approximate coefficients",
        shave_gap: 8.0, [0.0, 16.0],
            "drop trailing ±1 levels that are at least this many scan positions past the previous level (SVT shave_coeff, HEVC's hevc.shave_gap; 0 off): lower drops more",
        trellis_round_up: 1.0, [0.0, 1.0],
            "with quant.trellis, 1 lets a level rise to the nearest above the dead-zone quantiser's (HEVC's trellis); 0 only lowers levels: cheaper, keeps the dead zones",
        trellis_key_lambda_scale: 1.0, [0.0, 2.0],
            "with quant.trellis and quant.trellis_key, a further multiple of the trellis λ on key pictures: lower keeps more of the key's levels (the copies after a key inherit its quality)",
        key_qp_offset_delta: 0.0, [-6.0, 6.0],
            "added to the intra-period-derived fixed-QP key-frame QP offset (HEVC's hevc.key_qp_offset_delta): higher spends more on key frames",
        chroma_qp_offset: CHROMA_QP_OFFSET, [-12.0, 6.0],
            "chroma QP offset folded into the SPS chroma QP table (HEVC's 4:2:0 map of QP + offset, never above the luma QP; 6 is the identity): higher spends fewer bits on chroma",
        // Default off until screened (docs/handoff/vvc.md "Multi-type tree").
        mtt: 0.0, [0.0, 2.0],
            "inter-slice multi-type tree (sps_max_mtt_hierarchy_depth_inter_slice 1): 0 quadtree only, 1 adds binary splits of quadtree leaves, 2 binary and ternary; the engine plans the rectangular CUs (planner.rect_*): more partitioning work, finer motion boundaries",
        // On in Default (`DEFAULT_PARAMS`).
        temporal_mvp: 0.0, [0.0, 1.0],
            "1 enables temporal motion vector prediction (sps_temporal_mvp_enabled_flag): merge and AMVP lists also hold the vector of the previous picture's block below-right of (or at the centre of) the CU: cheaper motion where it continues, a few more candidates to evaluate",
        merge_candidates: headers::DEFAULT_MERGE_CANDIDATES as f32, [1.0, 6.0],
            "regular merge list size (MaxNumMergeCand): spatial, temporal, history, pairwise average and zero candidates; each further index costs a bypass bin: more merge choices, more candidates to evaluate",
        // 1 keeps the quality of every reach (probes, TMVP + 6 candidates,
        // 60 f CBR 300 on 4 clips) at less than half the search: x86 P
        // instructions +4.0% over 0, against +12.2% unlimited.
        pairwise_merge_reach: 1.0, [0.0, 64.0],
            "with vvc.merge_candidates 3 or more (no affine), eighth-sample pairwise merge candidates within this many quarter samples of the engine's vector are ranked by SATD and checked by luma RD (0: never): higher tries more, more search",
        // Speed heuristic: without the check, the SATD ranking alone decides
        // (two luma rate-distortion codings per CU it runs at, about 4% of
        // a P picture, 8% on low-light content). The SATD winner beats the
        // vector by RD in 71-85% of checks; accepting every one keeps 73-93%
        // of the checked gain (720p CBR 600, 60 f: bodycam, pan, crowd,
        // lowlight). Screen evidence: docs/handoff/vvc.md "vvc-default-speed".
        pairwise_merge_check: 1.0, [0.0, 1.0],
            "with vvc.pairwise_merge_reach, 1 confirms an off-grid merge candidate that beats the engine's vector by SATD with luma rate-distortion costs of both; 0 takes it on SATD alone: faster, a few worse choices",
        // Default off until screened (docs/handoff/vvc.md "vvc-mts").
        mts: 0.0, [0.0, 1.0],
            "1 enables multiple transform selection (sps_mts_enabled_flag): luma may use DST-VII and DCT-VIII as well as the DCT-II; intra sub-partitions then use them implicitly by size: sharper residuals, more transform work",
        mts_intra: 1.0, [0.0, 1.0],
            "with vvc.mts, 1 lets intra CUs up to 32x32 choose their luma transforms (mts_idx, by RD); 0 transforms every intra luma block implicitly by size (DST-VII on sides of 4 to 16)",
        mts_intra_search: 1.0, [0.0, 1.0],
            "with vvc.mts and vvc.mts_intra, 1 chooses intra CUs' mts_idx by RD; 0 codes them all with the DCT-II (mts_idx 0): less search",
        mts_inter: 1.0, [0.0, 1.0],
            "with vvc.mts, 1 lets inter CUs up to 32x32 choose their luma transforms (mts_idx, by RD); 0 keeps their DCT-II",
        mts_min_levels: 2.0, [1.0, 64.0],
            "with vvc.mts_intra or vvc.mts_inter, other transforms are tried only on luma blocks whose DCT-II codes at least this many nonzero levels: higher tries fewer blocks, less search",
        mts_lambda_scale: 1.0, [0.25, 8.0],
            "with vvc.mts_intra or vvc.mts_inter, multiple of the mode-decision λ the mts_idx choice trades bits at: higher picks the DST-VII / DCT-VIII only where they save more",
        mts_predicted: 0.0, [0.0, 1.0],
            "with vvc.mts_intra or vvc.mts_inter, 1 tries only the DST-VII / DCT-VIII pair the residual's energy profile suggests (DST-VII where the residual is stronger in the right / lower half), instead of vvc.mts_candidates of them: less search",
        mts_max_side: 16.0, [4.0, 32.0],
            "with vvc.mts_intra or vvc.mts_inter, other transforms are tried only on CUs no wider or taller than this: 32x32 CUs hold 3-9% of their RD gain for about a third of the search (720p CBR 600 probes)",
        mts_candidates: 4.0, [1.0, 4.0],
            "with vvc.mts_intra or vvc.mts_inter, the mts_idx values tried: 1 DST-VII both ways only, up to 4 every DST-VII / DCT-VIII pair: more search",
        ctu_size: DEFAULT_CTU_SIZE, [32.0, 64.0],
            "CTU size, 32 or 64 (64x64 CUs code four 32x32 transform units, or one 64-point unit with vvc.tx64): larger lets static and smooth areas code as one CU with fewer split flags and SAO parameter sets",
        /// Off: TRAIN gained (real-use k 0.1 `J_real` −0.9% with the
        /// planner's leaf cost), but VAL `bodycam_airport` lost +14.5% with
        /// three guards. That clip runs at QP 25–32 (PSNR-Y 47–51 dB), where
        /// the approximate 64-point forward (2x2 mean, 32-point DCT) costs
        /// 0.4–0.6 dB; it needs an exact forward at fine quantisers.
        tx64: 0.0, [0.0, 1.0],
            "with vvc.ctu_size 64 and vvc.mtt 0, 1 codes 64x64 CUs with one 64-point DCT-II transform unit of their 32x32 lowest frequencies (sps_max_luma_transform_size_64_flag) instead of four 32x32 ones: cheaper residual for large smooth blocks, an approximate forward transform",
    }
}

/// Default `vvc.ctu_size`: 64, with depth removal at coarse quantizers only
/// (`DEFAULT_PARAMS`). Staged TRAIN `J_real` (18 clips, 150 f, k 0.1) vs 32:
/// quality -3.9% (full -5.0%, low -2.7%; PSNR-YUV -6.7%, NEG -2.1%,
/// MS-SSIM -6.2%, VMAF v1 -3.7%, worst-1% Y -6.7%, worst-1% NEG -0.5%), no
/// guard, periodic-key pulse -0.29 dB; CPU -4.2% instructions (x86, P and
/// key), Jetson in docs/handoff/vvc.md. 64 alone (depth removal everywhere)
/// was -2.2% but tripped the worst-1% NEG guard (+2.3).
const DEFAULT_CTU_SIZE: f32 = 64.;

/// Default `vvc.chroma_qp_offset`: HEVC's 4:2:0 map of the QP itself
/// (chroma up to 6 QP finer than luma above QP 43, as HEVC with no
/// offset). Staged TRAIN (18 clips, 150 f, k 0.1) vs +3, after SAO merges:
/// BETTER, `J_real` -0.9% (full -0.8 / low -0.9; PSNR-YUV -5.6%, NEG +1.6%,
/// MS-SSIM +1.5%, worst-1% Y +1.3%, worst-1% NEG +1.5%), no guard, periodic
/// pulse -0.05 dB: chroma bits come from luma, and PSNR-YUV gains more
/// than the luma metrics lose. -3 (HEVC Default's) trips the worst-1% luma
/// guards (proxy-6: Y +4.1%, NEG +4.3%). Earlier, before SAO merges and sign
/// hiding, +3 had won over the identity (6) and over 0.
const CHROMA_QP_OFFSET: f32 = 0.;

/// Conventional QP-domain Lagrangian (the HM/VTM model also used by the HEVC
/// backend): λ = 0.57 · 2^((QP − 12) / 3), doubling every 3 QP.
const LAMBDA_AT_QP12: f32 = 0.57;

/// VVC's rate model: the shared QP calibration (VVC keeps HEVC's step,
/// doubling every 6 QP, H.266 §8.7.3) over the whole 8-bit range 0-63
/// (§7.4.3.4: `SliceQpY` up to 63). The shared default stops at HEVC's 51,
/// which left the lowest CBR rungs a quarter of the step short (at QP 51
/// every pan frame, 150 kbps: 2.8x the target). The start-up lookahead's
/// first-key cap is HEVC's 0.5 buffers.
const RATE_MODEL: nc_encoder::RateModel = nc_encoder::RateModel {
    max_quantizer: headers::MAX_QP,
    first_key_lookahead: 0.5,
    ..nc_encoder::RateModel::QP
};
/// Fixed-QP key-frame QP offset at the 60-frame reference period
/// (`nc_encoder::fixed_qp_key_offset`; HEVC uses 12). Fixed QP 42/47/52,
/// 150 f, PSNR-YUV BD vs offset 0 on `bodycam_bc1`, `crowd_mot17_09` and
/// `static_cam22`: intra period 60: 9 -20.9%, 12 -19.5%, 15 -15.3%; period
/// 300: 9 -19.8%, 12 -20.4%, 15 -19.5% (docs/handoff/vvc.md, qp-cascade).
const KEY_QP_OFFSET_AT_REFERENCE: f64 = 9.0;

/// How SAO parameters are searched (`vvc.sao_*`), and the classes reused
/// between search pictures.
#[derive(Debug)]
struct SaoSearch {
    /// `vvc.sao_stats_row_step`: statistics on every Nth row.
    row_step: usize,
    /// `vvc.sao_skip_unchanged`: all-skip P-frame CTUs off, no statistics.
    skip_unchanged: bool,
    /// `vvc.sao_luma_lambda_scale`.
    luma_lambda_scale: f32,
    /// `vvc.sao_search_interval`: classes searched every Nth P picture.
    interval: u8,
    /// The classes of the last searched picture, and those of the picture
    /// being coded once it is a search picture (committed in
    /// `commit_frame`, so a recode never sees them).
    classes: Vec<crate::sao::Classes>,
    pending: Option<Vec<crate::sao::Classes>>,
}
impl SaoSearch {
    fn new(params: &VvcParams) -> Self {
        Self {
            row_step: params.sao_stats_row_step.round() as usize,
            skip_unchanged: params.sao_skip_unchanged >= 0.5,
            luma_lambda_scale: params.sao_luma_lambda_scale,
            interval: params.sao_search_interval.round() as u8,
            classes: Vec::new(),
            pending: None,
        }
    }
}

/// The 16- and 32-point forward DCT: `vvc.fast_forward`'s staged rows, or
/// exact (fully so under the trellis, which may raise a level the
/// zero-column skip would have left zero).
fn forward_mode(params: &VvcParams, trellis: bool) -> nc_dsp::HevcForwardMode {
    if params.fast_forward >= 0.5 {
        recon::FAST_FORWARD
    } else if trellis {
        nc_dsp::HevcForwardMode::EXACT
    } else {
        recon::EXACT_FORWARD
    }
}

/// How intra sub-partitions are searched (`vvc.isp_*`).
#[derive(Debug, Clone, Copy)]
struct IspSearch {
    /// `vvc.isp_modes`: whole-CU directions tried as sub-partitions.
    modes: usize,
    /// `vvc.isp_inter`: P-picture intra CUs are searched too.
    inter: bool,
    /// `vvc.isp_residual_gate`: CUs whose best whole-CU direction codes no
    /// luma residual try no sub-partitions.
    residual_gate: bool,
}

/// How affine motion is searched (`vvc.affine_*`).
#[derive(Debug, Clone, Copy)]
struct AffineSearch {
    /// `vvc.affine_merge`: sub-block merge candidates are tried.
    merge: bool,
    /// `vvc.affine_amvp`: affine AMVP models are fitted.
    amvp: bool,
    /// `vvc.affine_iterations`: refinement steps per start model.
    iterations: usize,
    /// `vvc.affine_amvp_gate`: the vector's luma SATD per sample over √λ
    /// above which models are fitted.
    amvp_gate: f32,
    /// `vvc.affine_merge_gate`: the same, above which sub-block merge
    /// candidates are tried.
    merge_gate: f32,
}

/// The `aps_adaptation_parameter_set_id` of every picture's luma ALF bank.
const ALF_APS_ID: u8 = 0;

/// VVC backend: shared-engine quadtree CUs inside 32x32 or
/// 64x64 CTUs (`vvc.ctu_size`).
///
/// The shared engine owns traversal, frame planning, rate control and padding.
/// Prediction arithmetic uses shared DSP; syntax and reference state are VVC-specific.
#[derive(Debug)]
pub struct VvcBackend {
    geometry: headers::Geometry,
    qp: u32,
    /// Forward rounding offsets (dead zones) by prediction type and plane.
    rounding: Rounding,
    /// `vvc.pre_transform_zero_scale` in percent (`recon::Quant`).
    inter_zero_bound_percent: u32,
    shave_gap: usize,
    forward: nc_dsp::HevcForwardMode,
    /// The SPS's luma-to-chroma QP map.
    chroma_qp: ChromaQpTable,
    colour: nc_common::ColourInfo,
    key_qp_offset: i32,
    rate_control: bool,
    /// The intra-refresh window of the current frame (`set_refresh`), if
    /// any: then it uses no temporal candidates.
    refresh: Option<nc_encoder::RefreshWindow>,
    inter: bool,
    poc: u8,
    next_poc: u8,
    temporal_layers: u8,
    references: crate::references::References,
    lambda: f32,
    dsp: Dsp,
    motion: MotionGrid,
    items: Vec<syntax::Item>,
    sao: Vec<crate::sao::CtuSao>,
    /// The SPS's optional tools (`vvc.alf`).
    tools: headers::Tools,
    /// This picture's luma ALF decision.
    alf: crate::alf::Decision,
    /// The luma ALF search (`vvc.alf_*`) and its kept bank.
    alf_search: AlfSearch,
    /// The predictors of the block the engine is deciding (set by
    /// `mv_candidates`/`skip_vector`, cleared when any block is coded or a
    /// CTU starts): the engine prices many modes of one block on one grid.
    current: Option<(Block, crate::motion::Predictors)>,
    /// The CU (x, y, log2 size) whose intra the engine chose only through
    /// its directional estimate, and the inter vector it beat
    /// (`CodecBackend::intra_fallback`), consumed by its `encode_block`.
    intra_fallback: Option<((usize, usize, u8), Mv)>,
    /// The engine's exact luma prediction of its best candidate
    /// (`cache_exact_luma`), with its block and vector; reused by
    /// `encode_block` when the decision keeps that vector.
    cached_luma: Option<(Block, Mv)>,
    /// Samples of `cached_luma`, block-width rows (kept allocated).
    cached_luma_samples: Vec<u8>,
    /// Each plane's inter prediction of the CU being coded (kept allocated).
    inter_predictions: [Vec<u8>; 3],
    /// Luma predictions of the motion choice's candidates, which it
    /// prices through a shared reference (kept allocated): the one being
    /// ranked, and the cheapest off-grid merge candidate so far, which its
    /// rate-distortion confirmation reuses.
    rd_prediction: std::cell::RefCell<Vec<u8>>,
    rd_best_prediction: std::cell::RefCell<Vec<u8>>,
    contexts: Contexts,
    /// `vvc.chroma_search` on key pictures, and on P pictures.
    chroma_search: [bool; 2],
    /// `vvc.chroma_rd_candidates`: chroma directions given full RD (0: all).
    chroma_rd_candidates: usize,
    sao_search: SaoSearch,
    /// Luma direction search of intra CUs in key frames and in P frames.
    key_intra_search: crate::intra::DirectionSearch,
    inter_intra_search: InterIntraSearch,
    /// How intra sub-partitions are searched (`vvc.isp_*`).
    isp_search: IspSearch,
    /// How affine motion is searched (`vvc.affine_*`).
    affine_search: AffineSearch,
    /// How explicit MTS is searched (`vvc.mts_*`).
    mts_search: crate::mts::Search,
    /// `vvc.pairwise_merge_reach`, quarter samples.
    pairwise_merge_reach: i32,
    /// `vvc.pairwise_merge_check`: off-grid merge candidates are confirmed
    /// by luma RD.
    pairwise_merge_check: bool,
    /// `vvc.dep_quant_*` λ multiples: luma, chroma's further, keys' further.
    dep_quant_lambda: [f32; 3],
    /// `quant.trellis`: trellis quantisation of every coded transform block.
    trellis: Option<crate::trellis::TrellisQuant>,
    /// Quantiser working memory, one per plane of a chroma pair (luma
    /// uses the first).
    quant_scratch: Box<[recon::QuantScratch; 2]>,
    /// Quantiser working memory of [`CodecBackend::inter_leaf_cost`], which
    /// the planner calls through a shared reference.
    planning_scratch: std::cell::RefCell<Box<recon::QuantScratch>>,
    /// Block-level QP (`cu_qp_delta`), once the engine enables it.
    block_qp: Option<block_qp::BlockQp>,
}
impl VvcParams {
    /// How intra sub-partitions are searched.
    fn isp_search(&self) -> IspSearch {
        IspSearch {
            modes: self.isp_modes.round() as usize,
            inter: self.isp_inter >= 0.5,
            residual_gate: self.isp_residual_gate >= 0.5,
        }
    }

    /// The SPS's optional tools.
    fn tools(&self) -> headers::Tools {
        headers::Tools {
            alf: self.alf >= 0.5,
            isp: self.isp >= 0.5,
            dep_quant: self.dep_quant >= 0.5,
            affine: self.affine_tool(),
            temporal_mvp: self.temporal_mvp >= 0.5,
            merge_candidates: self.merge_candidates.round() as usize,
            mts: (self.mts >= 0.5).then_some(crate::mts::Mts {
                intra: self.mts_intra >= 0.5,
                inter: self.mts_inter >= 0.5,
            }),
            large_transforms: self.large_transforms(),
        }
    }

    /// `vvc.tx64` where it applies: 64x64 CTUs without multi-type trees
    /// (whose 64x32 and 64x16 CUs would need rectangular 64-point
    /// transforms, which we do not implement).
    fn large_transforms(&self) -> bool {
        self.tx64 >= 0.5
            && self.ctu_size.round() as usize > 1 << headers::MIN_CTU_LOG2
            && self.mtt.round() == 0.
    }

    /// How explicit MTS is searched.
    fn mts_search(&self) -> crate::mts::Search {
        crate::mts::Search {
            intra: self.mts_intra_search >= 0.5,
            min_levels: self.mts_min_levels.round() as usize,
            max_side: self.mts_max_side.round() as usize,
            lambda_scale: self.mts_lambda_scale,
            predicted: self.mts_predicted >= 0.5,
            candidates: self.mts_candidates.round() as u8,
        }
    }

    /// How affine motion is searched.
    fn affine_search(&self) -> AffineSearch {
        AffineSearch {
            merge: self.affine_merge >= 0.5,
            amvp: self.affine_amvp >= 0.5,
            iterations: self.affine_iterations.round() as usize,
            amvp_gate: self.affine_amvp_gate,
            merge_gate: self.affine_merge_gate,
        }
    }

    /// The affine motion the SPS enables (`vvc.affine`).
    fn affine_tool(&self) -> headers::AffineTool {
        match self.affine.round() as u8 {
            0 => headers::AffineTool::Off,
            1 => headers::AffineTool::FourParameter,
            _ => headers::AffineTool::SixParameter,
        }
    }

    /// The direction search of key-frame intra leaves.
    fn key_intra_search(&self) -> crate::intra::DirectionSearch {
        crate::intra::DirectionSearch::Ranked {
            angular_candidates: self.angular_candidates.round() as usize,
            coarse_step: self.angular_step.round() as u8,
            rd_candidates: self.rd_candidates.round() as usize,
            rd_margin_permille: self.rd_proxy_margin.round() as u32,
        }
    }

    /// The direction search of P-picture intra CUs: the ranked search, and
    /// outside intra-dense pictures the coarse one, if enabled.
    fn inter_intra_search(&self) -> InterIntraSearch {
        InterIntraSearch::new(
            crate::intra::DirectionSearch::Ranked {
                angular_candidates: self.inter_angular_candidates.round() as usize,
                coarse_step: self.inter_angular_step.round() as u8,
                rd_candidates: self.inter_rd_candidates.round() as usize,
                rd_margin_permille: 0,
            },
            if self.inter_intra_dc_only >= 0.5 {
                Some(crate::intra::DirectionSearch::DcOnly)
            } else if self.coarse_inter_intra >= 0.5 {
                Some(crate::intra::DirectionSearch::CoreAndMpm {
                    rd_candidates: self.coarse_rd_candidates.round() as usize,
                })
            } else {
                None
            },
            self.coarse_intra_dense_share,
        )
    }
}

impl CodecBackend for VvcBackend {
    fn intra_fallback(&mut self, block: &Block, mv: Mv) {
        self.intra_fallback = Some(((block.x, block.y, block.log2_size()), mv));
    }

    const RECODE: bool = true;
    // Multi-type tree leaves (`vvc.mtt`) are rectangular.
    const RECTANGULAR_LEAVES: bool = true;
    // Depth removal (dropping blocks below 32x32 where the motion analysis
    // says they barely help) only at coarse quantizers, as HEVC Default:
    // the analysis statistics it reads are per 64x64 superblock, so it acts
    // from `vvc.ctu_size` 64 on, where everywhere it cost detail (worst-1%
    // VMAF NEG +2.3, crowd/pan/lowlight NEG +2.5%). See `DEFAULT_CTU_SIZE`.
    //
    // The rest is one jointly tuned point (real-use TPE study
    // `vvc-quality-rt3`, trial 27 rounded; docs/handoff/vvc.md): a lower
    // mode-decision λ, a planner that splits less, finer intra and chroma
    // inter dead zones, a coarser luma inter one, less luma SAO smoothing,
    // a slightly wider shave, and depth removal from a little finer
    // quantizers. Staged J_real TRAIN (18 clips, 150 f, k 0.1) vs the
    // previous defaults: BETTER, J_real -3.6% (full -3.4%, low -4.0%;
    // PSNR-YUV -4.8%, NEG -3.9%, MS-SSIM -1.4%, worst-1% Y -1.8%, NEG
    // -3.7%), 1/18 clips worse (crowd_mot17_09 +0.8%), no guard, periodic
    // pulse -0.02 dB, dCPU +0.4%. Long GOP (900 f, one key, 300 kbps, Y by
    // thirds): pan_bdd +0.07 / 0.00 / +0.08 dB, chirla -0.20 / -0.09 /
    // -0.04 dB (YUV -0.06): no drift.
    const DEFAULT_PARAMS: &'static [(&'static str, f32)] = &[
        // Start-up QP cadence (period 8, base frames 6 QP finer, for the
        // first 100 frames after a key, then the engine's period 4 / -3 / +1)
        // and the cadence motion gate (`qp_cadence.rs`; docs/handoff/vvc.md,
        // qp-cascade). J_real (k 0.1, dCPU 0): proxy-6 BETTER -4.2% (static
        // -7.2%, crowd -6.1%, pan -3.5%, bodycam +0.1%); VAL BETTER -11.3%
        // (static_otw_homes390 -34%, crowd -15.7%; pulse +0.08 dB);
        // conference (adv12 Johnny / FourPeople / MeridianTalk) BETTER -19.3%,
        // worst-1% Y -6.8%, window flicker +0.09 dB, within the headroom below
        // SVT-AV1 rtc. Long GOP (900 f, one key, CBR 300): chirla Y 37.94 ->
        // 38.08, pan_bdd 31.09 -> 31.11.
        ("rate.cadence_startup_period", 8.0),
        ("rate.cadence_startup_base_boost", 6.0),
        ("rate.cadence_motion_gate", 1.0),
        ("planner.depth_removal_min_quantizer", 0.8),
        // Re-tuned 2026-11-13 as a CPU cut at equal quality (CPU-capped study
        // vvc-default-speed-2026-11-13, trial #2): mode-decision lambda 0.8 ->
        // 0.9125 and a coarser luma inter dead zone, 133 -> 95. Disjoint L3 at
        // head 0f2a708b (79 clips, single-point) J -0.27% [-0.57, +0.03], no
        // guard; x86 instructions -3.2%, Jetson cycles -1.3%. vvc-fast pins
        // the old values.
        ("decision.lambda_scale", 0.9125),
        ("planner.lambda_scale", 4.0),
        ("vvc.inter_rounding", 95.0),
        ("vvc.intra_rounding", 196.0),
        ("vvc.chroma_inter_rounding", 170.0),
        // Jointly re-tuned (equal-CPU study vvc-default-eq2-2026-11-11,
        // trial #3, with `vvc.alf_detail_weight` 0.125 and
        // `vvc.chroma_qp_offset` -1 below): less luma SAO smoothing, 1.4 ->
        // 2.1. Evidence in the commit and docs/handoff/tuning.md.
        ("vvc.sao_luma_lambda_scale", 2.1),
        ("vvc.chroma_qp_offset", -1.0),
        ("vvc.shave_gap", 9.0),
        // Trellis quantisation of the coded luma blocks (`trellis.rs`; HEVC
        // Default's arm), at half its λ on key pictures: the full λ saved
        // 5-15% of a key's bits at 0.2-0.4 dB less luma that rate control
        // left unspent, so the copies after each key were the clip's worst
        // frames (VAL worst-1% Y +2.6, a guard); no key trellis lost static
        // (static_cam22 +6% J low at proxy-4). Staged J_real (k 0.1, dCPU
        // +12.5%: host tune cost +23.0%, Jetson user time +7.2% at CBR 300,
        // weighted 1:2): proxy-6 BETTER, J_real -3.0% (full -4.3%, low
        // -4.3%; NEG -5.2%, worst-1% Y -0.9%), 0/6 clips worse.
        // VAL (11 clips): BETTER, J_real -2.4% (full -3.7%, low -3.6%; NEG
        // -5.0%, static -6.4%, worst-1% Y -1.1%), no guard. Long GOP: no
        // drift (chirla +0.3..+0.55 dB). docs/handoff/vvc.md.
        ("quant.trellis", 1.0),
        ("quant.trellis_chroma", 0.0),
        ("vvc.trellis_key_lambda_scale", 0.5),
        // Intra sub-partitions of 16x16 and 32x32 intra CUs, searched on key
        // pictures (`isp.rs`). Staged J_real (k 0.1, dCPU +0.5%: x86 tune
        // cost +0.64%, key +26.6%; Jetson user time +0.3%): TRAIN BETTER,
        // J_real -0.5% (full -0.4%, low -0.7%; static -1.7% / -2.6%,
        // static_otw_homes432 -2.9%), no guard. VAL (11 clips): BETTER,
        // J_real -0.3% (full -0.3%, low -0.4%; static -1.5%, pan +0.9%), no
        // guard, periodic-key pulse +0.03 dB. Long GOP (900 f, one key): no
        // drift (chirla -0.17 / -0.04 / -0.04 dB by thirds, pan_bdd -0.01).
        // docs/handoff/vvc.md "vvc-quality3".
        // Extended to 8x8 CUs (2-sample strips and columns; vvc-intra):
        // all-intra fixed QP vs 16/32 only -2.0% PSNR-YUV, -2.2% Y (static
        // -4.4%). Staged J_real (k 0.1, dCPU +1.1%: x86 tune cost +2.3%,
        // key +29%; Jetson cycles +0.9% / +0.1% crowd / bodycam): TRAIN
        // BETTER (full -0.3%, low -0.3%; static -0.7%), VAL BETTER, J_real
        // -0.2% (full -0.4%, low -0.3%; worst-1% Y -0.9%), no guard, pulse
        // +0.04 dB; long GOP no drift (chirla +0.05 / +0.15 / +0.11 dB).
        ("vvc.isp", 1.0),
        // Luma ALF with each picture's own APS filters (`alf_aps.rs`): the
        // bank's CTBs decided on their exact error change plus a quarter of
        // the detail (AC energy) filtering removes (an eighth since the
        // 2026-11-13 joint re-tune, trial #3 above); no fixed sets, none in
        // all-skip P CTUs, statistics every 4th row. Staged J_real (k 0.1,
        // dCPU +16.5%: x86 +12.1% P instructions, Jetson +15% / +22% cycles
        // crowd / bodycam, weighted 2:1): proxy-6 BETTER, J_real -2.9%
        // (full -4.7%, low -4.4%; PSNR-YUV -4.2%, NEG -3.5%, MS-SSIM -6.4%,
        // worst-1% NEG +0.1%), 0/6 clips worse, pulse -0.11 dB. VAL proxy-5:
        // BETTER, J_real -4.3% (full -6.2%, low -5.6%), no guard. Long GOP
        // (one key, full gate): every third better (pan_bdd +0.2..+0.3 dB,
        // chirla +0.1..+0.3). Without the detail term the worst-1% NEG
        // guard tripped (+3.3%). docs/handoff/vvc.md "vvc-depquant-alf".
        ("vvc.alf", 1.0),
        ("vvc.alf_fixed_sets", 0.0),
        ("vvc.alf_skip_unchanged", 1.0),
        ("vvc.alf_stats_row_step", 4.0),
        ("vvc.alf_detail_weight", 0.125),
        // Temporal bank reuse (VVenC's APS reuse): P pictures re-signal the
        // bank derived up to 4 pictures earlier, without statistics or a
        // derivation, deciding on the exact error change only the CTBs the
        // previous picture filtered. CPU: x86 tune cost P -5.6% (183.2 ->
        // 172.9 M), Jetson cycles (CBR 600, 150 f) -3.1 / -4.6 / -3.7%
        // crowd / bodycam / static. J_real (k 0.1, dCPU -4.4%): L2 quality
        // full -0.0% / low -0.0%, no guard; L3 (2026-11-13 draw, pooled
        // n 127) BETTER, J -0.40% [-0.48, -0.31]. Long horizon (720 f,
        // keys every 10 s, full ladder): J +0.39% (z 0.85), no guard.
        // Interval 8 measured the same at L2 (J -0.56%, x86 -6.0%).
        ("vvc.alf_bank_interval", 4.0),
        ("vvc.alf_reuse_on_only", 1.0),
        // HEVC Default's staged 16/32-point forward rows (approximate
        // coefficients; proven-zero column groups skipped, under the
        // trellis too). Staged J_real (k 0.1, dCPU -3.9%: x86 tune cost
        // -2.2%, Jetson -5.4 / -4.8 / -3.9% cycles crowd / bodycam / static
        // at CBR 600, weighted 2:1): TRAIN BETTER, J_real -0.7% (full -0.3%,
        // low -0.3%; static -1.7% / -2.2%, worst-1% Y +0.4%), no guard,
        // periodic-key pulse -0.03 dB. VAL (11 clips): BETTER, J_real -0.9%
        // (full -0.4%, low -0.6%), no guard, pulse -0.02 dB. Long GOP (one
        // key, full gate, Y by thirds): pan_bdd -0.02 / -0.01 / 0, chirla
        // -0.07 / 0 / -0.04, antare -0.05 / -0.07 / -0.02 dB: no drift.
        // docs/handoff/vvc.md "vvc-cpu4".
        ("vvc.fast_forward", 1.0),
        // Temporal MVP and six merge candidates (`temporal.rs`,
        // `motion::MergeList`; eighth-sample pairwise averages within one
        // quarter sample of the vector tried by RD). Staged J_real (k 0.1,
        // dCPU +6.9%: x86 P instructions +6.9%, Jetson cycles +6.8% / +7.1%
        // crowd / bodycam): proxy-6 BETTER, J_real -1.9% (full -2.5%, low
        // -2.6%; PSNR-YUV -2.6%, NEG -2.9%, worst-1% Y -3.7%, NEG -3.3%),
        // 0/6 clips worse (pan_jaad_0223 -5.6%), pulse -0.02 dB. VAL (11
        // clips): BETTER, J_real -1.3% (full -2.0%, low -1.9%), 1/11 worse,
        // no guard, pulse -0.13 dB. Long GOP: no drift. TMVP alone: TRAIN
        // BETTER, -0.4%; four candidates alone: proxy-6 +0.1% (stopped).
        // docs/handoff/vvc.md "vvc-tmvp".
        ("vvc.temporal_mvp", 1.0),
        ("vvc.merge_candidates", 6.0),
        // Seven VVC Fast speed levers, screened as one bundle: SAO classes
        // searched every 4th P picture and none in all-skip P CTUs; full RD
        // for the 2 cheapest P-intra core/MPM modes by the SATD proxy (HEVC
        // Default codes 1); key pictures RD only the 4 cheapest directions
        // and derive chroma from luma; the inter SAD zero bound at 2 (HEVC
        // Default's); HEVC's sub-pel seed gate. Staged J_real (k 0.1, dCPU
        // -11%: Jetson cycles -13.4 / -10.5 / -10.1% crowd / bodycam /
        // static at CBR 600, x86 tune cost -9.5%, weighted 2:1): proxy-4
        // WORSE on static_cam22 at 60 f alone (+3.7%; the same clip at
        // 150 f: +0.8%), the proxy-6 clips at 150 f BETTER, J_real -0.9%
        // (full +0.3%, low +0.1%; NEG -0.5%, PSNR-YUV +1.5%, worst-1% NEG
        // -1.0%), no guard, periodic-key pulse +0.03 dB. VAL (11 clips):
        // BETTER, J_real -0.2% (full +0.8%, low +1.0%; static_london_bus
        // +4.6% / +5.5%, class static +1.6% / +2.4%), no guard. A CPU
        // trade, not a quality one: docs/handoff/vvc.md "vvc-cpu5". Long
        // GOP (one key, full gate, Y by thirds): pan_bdd -0.01 / -0.01 /
        // -0.01, chirla +0.08 / -0.09 / +0.02, antare +0.01 / +0.01 / -0.02
        // dB: no drift. Dropped from the bundle: `vvc.alf_skip_unchanged` 2
        // with `vvc.alf_stats_row_step` 8 (quality +1.0%, J_real +0.6% on
        // static_cam22 / crowd_pp22_027). Screened before TMVP (above).
        ("vvc.sao_search_interval", 4.0),
        ("vvc.sao_skip_unchanged", 1.0),
        ("vvc.coarse_rd_candidates", 2.0),
        ("vvc.rd_candidates", 4.0),
        // The chroma direction searched on key pictures only (five
        // candidates, full RD; P pictures keep the derived direction).
        // Staged J_real (k 0.1, dCPU +0.4%: x86 tune cost +0.42%, key
        // +6.3%; Jetson cycles +0.3% / +0.4% crowd / bodycam): TRAIN BETTER, J_real -0.2% (full -0.2%, low -0.3%;
        // PSNR-YUV -0.6%, NEG 0.0%), VAL BETTER, J_real -0.5% (full -0.5%,
        // low -0.6%; static -1.6%), no guard, pulse +0.02 dB. All-intra
        // fixed QP: PSNR-YUV -1.7%, Y 0.0%. Long GOP neutral (chirla
        // -0.02 / +0.06 / +0.01 dB). docs/handoff/vvc.md "vvc-intra".
        ("vvc.chroma_search", 1.0),
        ("vvc.chroma_search_inter", 0.0),
        ("vvc.pre_transform_zero_scale", 2.0),
        ("decision.codec_subpel_seed_lambdas", 0.05),
        // Directional P-frame intra estimate (`decision.intra_directional`,
        // engine `intra_estimate.rs`), estimated only where DC intra costs
        // under 2x inter, with this backend's RD fallback
        // (`inter_fallback`). J_real (k 0.1, dCPU 0.8): TRAIN (18 clips)
        // BETTER -0.3%, quality -0.3% full / -0.3% low (PSNR -0.5%, NEG
        // -0.2%, VMAF v1 +0.1%), 5/18 clips worse (static_meva_hospital
        // +0.9% / +1.9%), pulse +0.01 dB; VAL proxy-5 BETTER -1.0% (every
        // class better), pulse +0.05 dB. Cost-clip P frames +0.7%, key 0.
        // vvc-fast keeps it off (not screened at k 0.6).
        ("decision.intra_directional", 1.0),
        ("decision.intra_directional_dc_ratio", 2.0),
    ];
    const PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] =
        &[VvcParams::SPECS, AlfSearchParams::SPECS];
    /// The ALF search settings are read when a picture is filtered, so each
    /// may take a non-base picture value (`nonbase.vvc.alf_*`).
    const FRAME_PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] =
        &[AlfSearchParams::SPECS];
    const SUPPORTS_BLOCK_QP: bool = true;
    // One `cu_qp_delta` per CTB quantization group; a 32x32 CTU repeats its
    // 64x64 square's offset (an exact, slightly redundant delta).
    const BLOCK_QP_LOG2: u8 = 6;
    /// Single-layer streams only: our decoder takes `cu_qp_delta` in the
    /// subset's own parameter sets, while a layered SPS goes to its general
    /// path, which has no `cu_qp_delta` yet (`decoder_blocks::validate_tools`).
    /// Layered streams keep one QP per picture until it does.
    fn enable_block_qp(&mut self) {
        if self.temporal_layers == 1 {
            self.block_qp = Some(block_qp::BlockQp::new());
        }
    }

    /// The engine's offset where block QP is coded, else the picture's.
    fn block_qp_offset(&mut self, requested: i16) -> i16 {
        if self.block_qp.is_some() {
            requested
        } else {
            0
        }
    }
    fn new(config: &EncoderConfig) -> Result<Self, ConfigError> {
        check_config(config)?;
        let params = VvcParams::from_overrides(&config.params);
        let ctu_size = params.ctu_size.round() as usize;
        let ctu_log2 = ctu_size.ilog2() as u8;
        if !ctu_size.is_power_of_two() || !headers::CTU_LOG2_RANGE.contains(&ctu_log2) {
            return Err(ConfigError::Unsupported("vvc.ctu_size must be 32 or 64"));
        }
        let partitioning =
            crate::coding_tree::Partitioning::of_mode(params.mtt.round() as u8, ctu_log2);
        let geometry = headers::Geometry::new(config.width, config.height, ctu_log2)
            .with_partitioning(partitioning);
        let trellis = crate::trellis::TrellisQuant::from_params(
            &nc_encoder::QuantParams::from_overrides(&config.params),
            params.trellis_round_up >= 0.5,
            params.trellis_key_lambda_scale,
        );
        Ok(Self {
            geometry,
            qp: config.quantizer,
            inter_zero_bound_percent: (params.pre_transform_zero_scale * 100.).round() as u32,
            shave_gap: params.shave_gap.round() as usize,
            forward: forward_mode(&params, trellis.is_some() || params.dep_quant >= 0.5),
            rounding: Rounding {
                intra: params.intra_rounding.round() as i32,
                inter: [
                    params.inter_rounding.round() as i32,
                    params.chroma_inter_rounding.round() as i32,
                ],
            },
            chroma_qp: match nc_encoder::chroma_policy::ChromaPolicy::from_params(&config.params) {
                Some(policy) => ChromaQpTable::policy(policy),
                None => ChromaQpTable::hevc(params.chroma_qp_offset.round() as i32),
            },
            colour: config.signal.colour,
            key_qp_offset: config.key_qp_offset.unwrap_or_else(|| {
                nc_encoder::fixed_qp_key_offset(config.intra_period, KEY_QP_OFFSET_AT_REFERENCE)
                    + params.key_qp_offset_delta.round() as i32
            }),
            rate_control: false,
            refresh: None,
            chroma_search: [
                params.chroma_search >= 0.5,
                params.chroma_search >= 0.5 && params.chroma_search_inter >= 0.5,
            ],
            chroma_rd_candidates: params.chroma_rd_candidates.round() as usize,
            sao_search: SaoSearch::new(&params),
            key_intra_search: params.key_intra_search(),
            inter_intra_search: params.inter_intra_search(),
            trellis,
            dep_quant_lambda: [
                params.dep_quant_lambda_scale,
                params.dep_quant_chroma_scale,
                params.dep_quant_key_scale,
            ],
            isp_search: params.isp_search(),
            affine_search: params.affine_search(),
            mts_search: params.mts_search(),
            pairwise_merge_reach: params.pairwise_merge_reach.round() as i32,
            pairwise_merge_check: params.pairwise_merge_check >= 0.5,
            quant_scratch: Box::default(),
            planning_scratch: std::cell::RefCell::default(),
            items: Vec::new(),
            sao: Vec::new(),
            tools: params.tools(),
            alf: crate::alf::Decision::default(),
            alf_search: AlfSearch::new(&AlfSearchParams::from_overrides(&config.params)),
            current: None,
            intra_fallback: None,
            cached_luma: None,
            cached_luma_samples: Vec::new(),
            inter_predictions: Default::default(),
            rd_prediction: std::cell::RefCell::default(),
            rd_best_prediction: std::cell::RefCell::default(),
            contexts: Contexts::new(config.quantizer, false),
            inter: false,
            poc: 0,
            next_poc: 0,
            temporal_layers: config.temporal_layers,
            references: crate::references::References::new(config.temporal_layers),
            lambda: 1.,
            dsp: Dsp::new(),
            motion: MotionGrid::new(
                geometry.coded_width,
                geometry.coded_height,
                ctu_log2,
                params.tools().merge_candidates,
            )
            .with_slots(nc_encoder::reference_slots(config.temporal_layers)),
            block_qp: None,
        })
    }
    fn superblock_log2(&self) -> u8 {
        self.geometry.ctu_log2
    }
    fn min_block_log2(&self) -> u8 {
        headers::MIN_QT_LOG2
    }
    fn coded_size(&self, width: usize, height: usize) -> (usize, usize) {
        let geometry = headers::Geometry::new(width, height, self.geometry.ctu_log2);
        (geometry.coded_width, geometry.coded_height)
    }
    fn loop_filter_reach(&self) -> usize {
        nc_encoder::DEFAULT_LOOP_FILTER_REACH
    }
    fn supports_inter(&self) -> bool {
        true
    }
    fn luma_filter(&self) -> Option<Filter> {
        Some(Filter::HevcLuma)
    }
    // Exact: VVC's quarter-sample luma filter is HEVC's (`inter.rs`), the
    // filter the engine ranks with, and the reference does not change during
    // the picture walk, so the cached samples equal a fresh prediction.
    const CACHE_EXACT_LUMA: bool = true;

    fn cache_exact_luma(
        &mut self,
        block: &Block,
        mv: Mv,
        filter: Filter,
        prediction: &[u8],
        _stats: nc_dsp::Variance,
    ) {
        if filter != Filter::HevcLuma {
            return;
        }
        self.cached_luma_samples.clear();
        self.cached_luma_samples.extend_from_slice(prediction);
        self.cached_luma = Some((*block, mv));
    }

    fn lambda(&self) -> f32 {
        self.lambda
    }
    fn supports_intra_leaf_cost(&self) -> bool {
        true
    }
    fn intra_leaf_cost(&mut self, source: &Picture, block: &Block) -> Option<f32> {
        let quant = self.quantizers().plane(PlaneIndex::Y, true);
        Some(crate::intra::leaf_cost(
            &self.dsp,
            source,
            (block, self.tools.max_tb_log2()),
            &self.contexts,
            quant,
            self.lambda,
            &mut self.quant_scratch[0],
        ))
    }

    fn inter_leaf_cost(
        &self,
        source: &Picture,
        reference: &Picture,
        block: &Block,
        mv: Mv,
    ) -> Option<f32> {
        let quant = self.quantizers().plane(PlaneIndex::Y, false);
        let mut contexts = self.contexts.coeff.clone();
        let (sse, coefficient_bits) = crate::inter_cost::luma_cost(
            &self.dsp,
            source,
            reference,
            (block, mv),
            &mut contexts,
            (quant, self.tools.max_tb_log2()),
            &mut self.planning_scratch.borrow_mut(),
        );
        let mode_bits = self.mode_bits(block, &PredictionMode::Inter(mv));
        Some(sse as f32 + self.lambda * (coefficient_bits + mode_bits))
    }

    // Merge candidates off the quarter-sample grid (merged from affine
    // neighbours, pairwise averages) are not offered: the engine prices
    // quarter samples only. Each vector is offered once (the engine measures
    // every entry; longer merge lists repeat the zero vector).
    fn mv_candidates(&mut self, block: &Block, out: &mut Vec<Mv>) {
        let p = self.cache_predictors(block);
        let start = out.len();
        let amvp = &p.amvp[crate::motion::Amvr::Quarter as usize];
        for mv in p
            .merge
            .as_slice()
            .iter()
            .chain(amvp)
            .filter_map(|mv| mv.quarter())
        {
            if !out[start..].contains(&mv) {
                out.push(mv);
            }
        }
    }
    fn skip_vector(&mut self, block: &Block) -> Option<Mv> {
        self.cache_predictors(block).merge[0].quarter()
    }
    fn mode_bits(&self, block: &Block, mode: &PredictionMode) -> f32 {
        let computed;
        let p = match &self.current {
            Some((cached, p)) if cached == block => p,
            _ => {
                computed = self.motion.predictors(block);
                &computed
            }
        };
        let c = &self.contexts;
        match *mode {
            PredictionMode::IntraDc => {
                // Price the shared engine's DC candidate with the actual MPM list.
                // Direction refinement happens after the engine selects intra.
                let direction = crate::intra::mode_bits(c, p.mpm, crate::intra::DC);
                let isp =
                    if self.tools.isp && crate::isp::signalled(block, self.tools.max_tb_log2()) {
                        c.isp_mode.bits(false)
                    } else {
                        0.
                    };
                c.skip[p.skip_ctx].bits(false)
                    + c.intra[p.intra_ctx].bits(true)
                    + direction
                    + isp
                    + c.chroma_mode.bits(false)
            }
            PredictionMode::Inter(mv) => self.vector_bits(block, p, Mv16::from_quarter(mv)),
        }
    }
    fn rate_model(&self) -> nc_encoder::RateModel {
        RATE_MODEL
    }
    fn relative_quantizer(&self) -> f32 {
        self.qp as f32 / headers::MAX_QP as f32
    }
    fn supports_temporal_layers(&self) -> bool {
        true
    }
    fn set_rate_control_enabled(&mut self, enabled: bool) {
        self.rate_control = enabled;
    }
    fn set_frame_params(&mut self, params: &nc_encoder::ParamOverrides) {
        self.alf_search
            .configure(&AlfSearchParams::from_overrides(params));
    }
    fn set_refresh(&mut self, window: Option<&nc_encoder::RefreshWindow>) {
        self.refresh = window.cloned();
    }
    fn begin_frame(&mut self, plan: &FramePlan) {
        self.qp = if plan.frame_type == FrameType::Key && !self.rate_control {
            (i64::from(plan.quantizer) - i64::from(self.key_qp_offset))
                .clamp(0, i64::from(headers::MAX_QP)) as u32
        } else {
            plan.quantizer
        };
        self.items.clear();
        self.sao.clear();
        self.sao_search.pending = None;
        self.alf_search.begin_picture();
        self.alf = crate::alf::Decision::default();
        self.inter = plan.frame_type == FrameType::Inter;
        self.inter_intra_search.begin_picture();
        self.poc = if self.inter { self.next_poc } else { 0 };
        self.lambda = LAMBDA_AT_QP12 * 2f32.powf((self.qp as f32 - 12.) / 3.) * plan.lambda_scale;
        self.contexts =
            Contexts::new(self.qp, self.inter).with_dependent_quantization(self.tools.dep_quant);
        self.begin_block_qp_frame();
        self.references.begin(plan.references, self.poc, self.inter);
        self.motion.select_collocated(
            plan.references.reference.slot,
            self.references.syntax.distances[0],
            self.references.collocated_distance(),
        );
        self.motion.reset(self.temporal_mvp());
        self.current = None;
        self.cached_luma = None;
    }
    fn begin_superblock(&mut self, x: usize, y: usize) {
        self.items.push(syntax::Item::Ctu { x, y });
        self.begin_quant_group(x, y);
        self.motion.begin_ctu(x);
        self.current = None;
    }
    fn write_partition(&mut self, node: &Block, split: bool) {
        let split = if split {
            TreeSplit::Quad
        } else {
            TreeSplit::None
        };
        self.push_split(node, split);
    }
    /// The cheapest inter leaf at the current probabilities, for the
    /// shared planner: a skipped CU on merge candidate 0 (`cu_skip_flag`
    /// at its cheapest context, `merge_idx` bin 0). Without it a split paid
    /// only its split bins, and whole-sample variance gains worth under a
    /// bit split 64x64 CUs, which the 64-point transform (`vvc.tx64`) codes
    /// more cheaply than four 32x32 ones (see the AV1 backend's estimate).
    fn planner_leaf_bits(&self) -> f32 {
        let skip = self
            .contexts
            .skip
            .iter()
            .map(|&c| c.bits(true))
            .fold(f32::INFINITY, f32::min);
        skip + self.contexts.merge_idx.bits(false)
    }
    fn partition_bits(&self, node: &Block, split: bool) -> f32 {
        let split = if split {
            TreeSplit::Quad
        } else {
            TreeSplit::None
        };
        let (allowed, contexts) = self.split_choices(node);
        crate::coding_tree::bits(&self.contexts.tree, allowed, contexts, split)
    }
    fn partition_shapes(&self, node: &Block) -> ShapeSet {
        if !self.inter || !node.is_inside(self.geometry.coded_width, self.geometry.coded_height) {
            return ShapeSet::EMPTY;
        }
        let (allowed, _) = self.split_choices(node);
        PartitionShape::ALL
            .into_iter()
            .filter(|&shape| allowed.permits(TreeSplit::Shape(shape)))
            .fold(ShapeSet::EMPTY, ShapeSet::with)
    }
    fn shape_bits(&self, node: &Block, shape: PartitionShape) -> f32 {
        let (allowed, contexts) = self.split_choices(node);
        crate::coding_tree::bits(
            &self.contexts.tree,
            allowed,
            contexts,
            TreeSplit::Shape(shape),
        )
    }
    fn write_shape(&mut self, node: &Block, shape: PartitionShape) {
        self.push_split(node, TreeSplit::Shape(shape));
    }
    fn encode_block(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        source: &Picture,
        reference: Option<&Picture>,
        reconstruction: &mut Picture,
    ) {
        self.begin_block_qp(decision.qp_offset);
        let (mode, (intra_mode, chroma_mode, isp)) =
            self.block_mode(block, decision.mode, source, reference, reconstruction);
        let motion = self.block_motion(block, mode, source, reference);
        // A sub-partitioned luma quantised to nothing is coded whole instead.
        let mut isp_luma = isp.and_then(|split| {
            self.reconstruct_isp(block, split, intra_mode, source, reconstruction)
                .map(|luma| (split, luma))
        });
        let isp = isp_luma.as_ref().map(|&(split, _)| split);
        let mut levels: [Vec<i32>; 3] = Default::default();
        let max_tb_log2 = self.tools.max_tb_log2();
        let count = transform_tree::count(block, max_tb_log2);
        let quantizers = self.quantizers();
        let mut unit_prediction = Vec::new();
        let lambdas = [self.lambda, quantizers.chroma_lambda(self.lambda)];
        let mut mts = 0;
        let search_mts = self
            .mts_search
            .tries(self.tools.mts, motion.is_none(), block);
        for plane in PlaneIndex::ALL {
            if plane == PlaneIndex::Y
                && let Some((_, luma)) = isp_luma.take()
            {
                levels[0] = luma;
                continue;
            }
            let shift = usize::from(plane != PlaneIndex::Y);
            let mut buffer = std::mem::take(&mut self.inter_predictions[plane as usize]);
            let inter_prediction = motion.map(|motion| {
                self.inter_prediction(
                    block,
                    &motion,
                    plane,
                    reference.expect("inter frame reference"),
                    &mut buffer,
                );
                &buffer
            });
            let quant = quantizers.plane(plane, motion.is_none());
            let searched = plane == PlaneIndex::Y && search_mts;
            let direction = if plane == PlaneIndex::Y {
                intra_mode
            } else {
                chroma_mode
            };
            for (index, tu) in transform_tree::units(block, max_tb_log2).enumerate() {
                let prediction = match &inter_prediction {
                    Some(prediction) if tu == *block => prediction,
                    Some(prediction) => {
                        transform_tree::extract(
                            prediction,
                            block,
                            &tu,
                            shift,
                            &mut unit_prediction,
                        );
                        &unit_prediction
                    }
                    None => {
                        unit_prediction = crate::intra::prediction(
                            &self.dsp,
                            reconstruction,
                            Some(&self.motion),
                            plane,
                            (&tu, block),
                            direction,
                        );
                        &unit_prediction
                    }
                };
                let unit = (tu.x >> shift, tu.y >> shift, tu.width() >> shift);
                let (level, unit_mts) = self.code_unit(
                    (source, reconstruction),
                    (block, plane, motion.is_none(), searched),
                    (unit, prediction),
                    (quant, lambdas),
                );
                mts = mts.max(unit_mts);
                transform_tree::place(&mut levels[plane as usize], count, index, level);
            }
            self.inter_predictions[plane as usize] = buffer;
        }
        let residual =
            transform_tree::coded_units(block, isp, &levels, self.tools.max_tb_log2()).any();
        let qp_delta = self.end_block_qp(block, residual);
        self.push_leaf(
            block,
            motion,
            levels,
            (intra_mode, chroma_mode, isp),
            qp_delta,
            mts,
        );
        self.current = None;
    }
    fn loop_filter_frame(&mut self, recon: &mut Picture) {
        let qp = crate::filter::DeblockQp {
            frame: self.quantizers().qp,
            blocks: self.block_qp.as_ref().map(|b| (&b.map, &self.chroma_qp)),
        };
        crate::filter::apply(&self.dsp, recon, &self.motion, &qp);
    }
    fn filter_frame(&mut self, source: &Picture, recon: &mut Picture) {
        self.loop_filter_frame(recon);
        self.decide_sao(source, recon);
        crate::sao::apply(&self.dsp, recon, &self.sao, self.geometry.ctu_size());
        self.alf = crate::alf::Decision::default();
        if self.tools.alf {
            let ctu = self.geometry.ctu_size();
            // All-skip P-frame CTBs copy samples their reference already
            // filtered (`vvc.alf_skip_unchanged`).
            let skipped = match self.alf_search.skip_unchanged {
                _ if !self.inter => None,
                AlfSkip::None => None,
                AlfSkip::Skipped => Some(self.motion.skipped_ctus()),
                AlfSkip::LumaPredicted => Some(self.motion.luma_predicted_ctus()),
            };
            self.alf = self.alf_search.filter(
                &self.dsp,
                (source, recon),
                ctu,
                self.inter,
                skipped.as_deref(),
                self.lambda,
                &self.contexts,
            );
        }
    }
    fn commit_frame(&mut self) {
        // Recode attempts and dropped frames must not advance the reference
        // POC or replace the collocated motion.
        self.next_poc = self.poc.wrapping_add(1);
        if self.tools.temporal_mvp {
            for slot in 0..nc_encoder::reference_slots(self.temporal_layers) {
                if self.references.plan.refresh.contains(slot) {
                    self.motion.store_motion_field(slot);
                }
            }
        }
        self.references.commit(self.poc);
        let samples = self.geometry.coded_width * self.geometry.coded_height;
        self.inter_intra_search.commit(self.inter, samples);
        if let Some(classes) = self.sao_search.pending.take() {
            self.sao_search.classes = classes;
        }
        if self.tools.alf {
            self.alf_search.commit();
        }
    }
    fn end_frame(&mut self) -> Vec<u8> {
        // A picture whose CTBs all leave ALF off disables it in its slice.
        let alf_enabled = self.alf.ctbs.iter().any(Option::is_some);
        let signalled = self.alf.signalled.is_some();
        let mut cabac = CabacWriter::new();
        let mut contexts =
            Contexts::new(self.qp, self.inter).with_dependent_quantization(self.tools.dep_quant);
        for item in &self.items {
            match item {
                syntax::Item::Ctu { x, y } => {
                    let ctu = self.geometry.ctu_size();
                    let columns = self.geometry.coded_width.div_ceil(ctu);
                    let index = y / ctu * columns + x / ctu;
                    let params = |i: usize| self.sao.get(i).copied().unwrap_or_default();
                    crate::sao::code(
                        &mut cabac,
                        &mut contexts,
                        params(index),
                        (*x > 0).then(|| params(index - 1)),
                        (*y > 0).then(|| params(index - columns)),
                    )
                    .expect("valid SAO");
                    if alf_enabled {
                        let decision = |i: usize| self.alf.ctbs[i];
                        crate::alf::code(
                            &mut cabac,
                            &mut contexts,
                            decision(index),
                            (
                                (*x > 0).then(|| decision(index - 1)),
                                (*y > 0).then(|| decision(index - columns)),
                            ),
                            signalled,
                        )
                        .expect("valid ALF");
                    }
                }
                &syntax::Item::Split {
                    allowed,
                    contexts: split_contexts,
                    split,
                } => {
                    crate::coding_tree::code(
                        &mut cabac,
                        &mut contexts.tree,
                        allowed,
                        split_contexts,
                        split,
                    )
                    .expect("an allowed split");
                }
                syntax::Item::Leaf {
                    contexts: neighbours,
                    coding,
                } => {
                    syntax::block(
                        &mut cabac,
                        &mut contexts,
                        (self.inter, self.tools),
                        neighbours,
                        coding,
                    );
                }
            }
        }
        cabac.encode_terminate(true);
        let mut coded = cabac.finish();
        coded.trailing_bits();
        let mut slice = self.slice_header();
        slice.extend(coded.into_bytes());
        let mut out = Vec::new();
        if !self.inter {
            self.write_parameter_sets(&mut out);
        }
        // Each picture carries the bank it uses, so no picture depends on
        // another's APS (a lost picture never breaks a later one).
        if let Some(bank) = self.alf.signalled.as_ref().filter(|_| alf_enabled) {
            headers::temporal_nal(
                headers::PREFIX_APS_NUT,
                self.references.plan.position.layer,
                &bank.rbsp(ALF_APS_ID),
                &mut out,
            );
        }
        headers::temporal_nal(
            if self.inter {
                headers::TRAIL_NUT
            } else {
                headers::IDR_N_LP
            },
            self.references.plan.position.layer,
            &slice,
            &mut out,
        );
        out
    }
}

/// Reject what this backend cannot code of `config` (before its parameters).
fn check_config(config: &EncoderConfig) -> Result<(), ConfigError> {
    if config.codec != Codec::Vvc {
        return Err(ConfigError::Unsupported("VvcBackend requires Codec::Vvc"));
    }
    if config.width > headers::MAX_DIMENSION || config.height > headers::MAX_DIMENSION {
        return Err(ConfigError::Unsupported(
            "VVC dimensions are limited to 4096",
        ));
    }
    if config.quantizer > headers::MAX_QP {
        return Err(ConfigError::Quantizer {
            value: config.quantizer,
            min: 0,
            max: headers::MAX_QP,
        });
    }
    if config.signal.timing != TimingSignal::Nominal {
        return Err(ConfigError::Unsupported("VVC has no HRD timing yet"));
    }
    Ok(())
}

/// The `merge_idx` of the first merge candidate equal to `mv`.
fn merge_position(p: &crate::motion::Predictors, mv: Mv16) -> Option<usize> {
    p.merge.position(mv)
}

/// The encoder's forward rounding offsets, in 1/512 of a step.
#[derive(Debug, Clone, Copy)]
struct Rounding {
    intra: i32,
    /// Luma, chroma.
    inter: [i32; 2],
}

impl VvcBackend {
    /// Record coded CU `block` (its `levels` and directions) for the slice
    /// data and in the neighbour grid.
    fn push_leaf(
        &mut self,
        block: &Block,
        motion: Option<Motion>,
        levels: [Vec<i32>; 3],
        (intra_mode, chroma_mode, isp): (u8, u8, Option<crate::isp::Split>),
        qp_delta: Option<i32>,
        mts: u8,
    ) {
        let p = self.predictors(block);
        let inter = motion.map(|motion| syntax::Inter {
            motion,
            syntax: self.inter_syntax(block, &p, motion),
        });
        let coded = transform_tree::coded_units(block, isp, &levels, self.tools.max_tb_log2());
        let merged = inter.is_some_and(|i| {
            matches!(
                i.syntax,
                syntax::InterSyntax::Merge(_) | syntax::InterSyntax::SubblockMerge(_)
            )
        });
        let skip = merged && !coded.any();
        self.items.push(syntax::Item::Leaf {
            contexts: p.contexts(),
            coding: syntax::Coding {
                inter,
                cu: *block,
                levels,
                skip,
                intra_mode,
                chroma_mode,
                isp,
                qp_delta,
                mts,
            },
        });
        self.motion
            .record(block, motion.as_ref(), skip, intra_mode, &coded);
    }

    /// The cheapest signalling of inter CU motion `motion` given the CU's
    /// predictors `p`: a merge candidate equal to it, else AMVP.
    fn inter_syntax(
        &self,
        block: &Block,
        p: &crate::motion::Predictors,
        motion: Motion,
    ) -> syntax::InterSyntax {
        match motion {
            Motion::Vector(mv) => {
                if let Some(index) = merge_position(p, mv) {
                    return syntax::InterSyntax::Merge(index);
                }
                let (amvr, mvp, _) = syntax::amvp_coding(&self.contexts, p, mv);
                syntax::InterSyntax::Amvp {
                    mvp,
                    amvr,
                    mvd: syntax::mvd_of(mv, p.amvp[amvr as usize][mvp], amvr),
                }
            }
            Motion::Affine(model) => {
                if let Some(index) = self.subblock_merge_position(block, &model) {
                    return syntax::InterSyntax::SubblockMerge(index);
                }
                let preds = crate::affine_candidates::amvp_list(&self.motion, block, model.six);
                let (mvp, mvd, _) = syntax::affine_amvp_coding(&self.contexts, &preds, &model);
                syntax::InterSyntax::AffineAmvp { mvp, mvd }
            }
        }
    }

    /// Quantise one transform unit of `plane` at `(x, y)` against its
    /// `prediction` (trellis at the plane's λ when on) and store its
    /// reconstruction; returns its levels (empty, allocating nothing, when
    /// all are zero).
    #[allow(
        clippy::inline_always,
        reason = "the coded-block path: outlined (three callers) it cost 0.3% of a VVC Fast encode (x86 callgrind)"
    )]
    #[inline(always)]
    fn reconstruct_unit(
        &mut self,
        source: &Picture,
        reconstruction: &mut Picture,
        plane: PlaneIndex,
        unit: (usize, usize, usize),
        prediction: &[u8],
        (quant, lambdas): (recon::Quant, [f32; 2]),
    ) -> Vec<i32> {
        let search = if self.tools.dep_quant {
            let ch = usize::from(plane != PlaneIndex::Y);
            let [scale, chroma, key] = self.dep_quant_lambda;
            let lambda = lambdas[ch]
                * scale
                * if ch == 0 { 1. } else { chroma }
                * if self.inter { 1. } else { key };
            Some(recon::LevelSearch::Dependent(crate::dep_quant::DepQuant {
                contexts: &self.contexts.coeff,
                ch,
                lambda,
            }))
        } else {
            self.trellis
                .as_mut()
                .and_then(|t| t.for_block(&self.contexts.coeff, plane, self.inter, lambdas))
                .map(recon::LevelSearch::Trellis)
        };
        let scratch = &mut self.quant_scratch[0];
        let level = recon::quantize_rect(
            &self.dsp, source, plane, unit, prediction, quant, search, scratch,
        );
        recon::store(
            &self.dsp,
            reconstruction,
            plane,
            unit,
            prediction,
            level,
            recon::Scaling {
                qp: quant.qp,
                dependent: self.tools.dep_quant,
                transform: quant.transform,
            },
        );
        level.to_vec()
    }

    /// The motion of CU `block` coded in `mode` (none for intra), refined
    /// by the backend's motion choice; notes P-picture intra CUs for the
    /// intra-dense gate.
    fn block_motion(
        &mut self,
        block: &Block,
        mode: PredictionMode,
        source: &Picture,
        reference: Option<&Picture>,
    ) -> Option<Motion> {
        match mode {
            PredictionMode::IntraDc => {
                if self.inter {
                    self.inter_intra_search.add(block);
                }
                None
            }
            PredictionMode::Inter(mv) => Some(self.choose_motion(
                block,
                Mv16::from_quarter(mv),
                source,
                reference.expect("inter frame reference"),
            )),
        }
    }

    /// Code transform unit `unit` of `plane` of CU `block` (`intra` or
    /// inter): with the MTS search when `searched`, else the DCT-II.
    /// Returns its levels and the `mts_idx` chosen (0 unless searched).
    fn code_unit(
        &mut self,
        (source, reconstruction): (&Picture, &mut Picture),
        (block, plane, intra, searched): (&Block, PlaneIndex, bool, bool),
        (unit, prediction): ((usize, usize, usize), &[u8]),
        inputs: (recon::Quant, [f32; 2]),
    ) -> (Vec<i32>, u8) {
        if searched {
            let pictures = (source, reconstruction);
            self.reconstruct_luma(pictures, (block, intra), unit, prediction, inputs)
        } else {
            let level =
                self.reconstruct_unit(source, reconstruction, plane, unit, prediction, inputs);
            (level, 0)
        }
    }

    /// [`Self::reconstruct_unit`] of luma transform unit `unit` of CU
    /// `block` (`intra` or inter, which the MTS search tries) with the
    /// `mts_idx` [`Self::choose_mts`] picks; returns the levels and that index. The coded block's level
    /// search (trellis) may leave levels that cannot signal it (only a DC
    /// level): the unit is then coded with its default transform.
    // Kept out of `encode_block`, whose loop codes every other unit.
    #[inline(never)]
    fn reconstruct_luma(
        &mut self,
        (source, reconstruction): (&Picture, &mut Picture),
        (block, intra): (&Block, bool),
        unit: (usize, usize, usize),
        prediction: &[u8],
        (quant, lambdas): (recon::Quant, [f32; 2]),
    ) -> (Vec<i32>, u8) {
        let mut code = |backend: &mut Self, quant| {
            backend.reconstruct_unit(
                source,
                reconstruction,
                PlaneIndex::Y,
                unit,
                prediction,
                (quant, lambdas),
            )
        };
        // The default transform first: most blocks keep it, and its coded
        // levels decide whether the others are tried at all.
        let level = code(self, quant);
        let nonzero = level.iter().filter(|&&v| v != 0).count();
        if nonzero < self.mts_search.min_levels {
            return (level, 0);
        }
        let index = self.choose_mts(source, block, intra, prediction, quant);
        if index == 0 {
            return (level, 0);
        }
        let chosen = recon::Quant {
            transform: crate::mts::luma(self.tools.mts, intra, false, index),
            ..quant
        };
        let level = code(self, chosen);
        if crate::mts::signalled(self.tools.mts, intra, block, &level) {
            return (level, index);
        }
        (code(self, quant), 0)
    }

    /// The allowed splits of quadtree node `node` in this slice and their
    /// contexts (§6.4, §9.3.4.2.2).
    fn split_choices(&self, node: &Block) -> (Allowed, crate::coding_tree::SplitContexts) {
        let node = crate::coding_tree::Node {
            block: *node,
            mtt_depth: 0,
        };
        let limits = if self.inter {
            self.geometry.partitioning
        } else {
            self.geometry.partitioning.intra()
        };
        let picture = (self.geometry.coded_width, self.geometry.coded_height);
        let allowed = Allowed::of(&node, limits, picture);
        (allowed, self.motion.split_contexts(&node, allowed))
    }

    /// Record quadtree node `node`'s `split` for the slice data. The leaves
    /// of a shape code no split (one multi-type tree level at most).
    fn push_split(&mut self, node: &Block, split: TreeSplit) {
        let (allowed, contexts) = self.split_choices(node);
        assert!(allowed.permits(split), "{split:?} at {node:?}");
        if allowed.signals() {
            self.items.push(syntax::Item::Split {
                allowed,
                contexts,
                split,
            });
        }
    }

    /// Copy the engine's cached luma prediction of `block` at `mv` into
    /// `out` if there is one; the cache is consumed either way.
    fn take_cached_luma(&mut self, block: &Block, mv: Mv, out: &mut [u8]) -> bool {
        let hit = self.cached_luma.take() == Some((*block, mv));
        if hit {
            out.copy_from_slice(&self.cached_luma_samples);
        }
        hit
    }

    /// The inter prediction of `plane` of CU `block` at `motion`: the
    /// engine's cached luma when it matches.
    fn inter_prediction(
        &mut self,
        block: &Block,
        motion: &Motion,
        plane: PlaneIndex,
        reference: &Picture,
        prediction: &mut Vec<u8>,
    ) {
        if plane == PlaneIndex::Y
            && let Motion::Vector(mv) = motion
            && let Some(mv) = mv.quarter()
        {
            prediction.resize(block.area(), 0);
            if self.take_cached_luma(block, mv, prediction) {
                // The engine's exact ranking already interpolated it.
                return;
            }
        }
        inter::predict_cu_into(&self.dsp, reference, plane, block, motion, prediction);
    }

    /// `block`'s predictors: the cached ones when the engine is deciding
    /// it (the grid changes only when a CU is coded, which clears them).
    fn predictors(&self, block: &Block) -> crate::motion::Predictors {
        self.current
            .filter(|(cached, _)| cached == block)
            .map_or_else(|| self.motion.predictors(block), |(_, p)| p)
    }

    /// Parameter sets repeated at every key for independent extraction and recovery.
    fn write_parameter_sets(&self, out: &mut Vec<u8>) {
        headers::nal(
            headers::SPS_NUT,
            &headers::sps_layered(
                &self.geometry,
                self.colour,
                &self.chroma_qp,
                self.tools,
                self.temporal_layers,
            ),
            out,
        );
        headers::nal(
            headers::PPS_NUT,
            &headers::pps(&self.geometry, self.block_qp.is_some()),
            out,
        );
    }

    /// The current picture's slice header (with its picture header).
    fn slice_header(&self) -> Vec<u8> {
        headers::slice_layered(
            self.qp,
            self.inter,
            self.poc,
            self.tools.alf.then(|| self.alf.slice_syntax(ALF_APS_ID)),
            self.block_qp.is_some(),
            self.tools.temporal_mvp.then(|| self.temporal_mvp()),
            &self.references.syntax,
        )
    }

    /// Whether the current picture uses temporal candidates
    /// (`ph_temporal_mvp_enabled_flag`). Not under intra refresh: after a
    /// loss the decoder's collocated motion differs from the encoder's, so
    /// a merge index would select a different vector and the clean region
    /// would not recover (as HEVC's backend).
    fn temporal_mvp(&self) -> bool {
        self.tools.temporal_mvp && self.inter && self.refresh.is_none()
    }

    /// `block`'s predictors, kept for the engine's following calls about
    /// it (`current` is cleared whenever a CU is coded).
    fn cache_predictors(&mut self, block: &Block) -> &crate::motion::Predictors {
        if self
            .current
            .as_ref()
            .is_none_or(|(cached, _)| cached != block)
        {
            self.current = Some((*block, self.motion.predictors(block)));
        }
        &self.current.as_ref().expect("cached above").1
    }

    /// The current picture's forward quantisers.
    fn quantizers(&self) -> recon::Quantizers {
        recon::Quantizers {
            qp: [self.qp, self.chroma_qp.map(self.qp)],
            intra_rounding: self.rounding.intra,
            inter_rounding: self.rounding.inter,
            inter_zero_bound_percent: self.inter_zero_bound_percent,
            shave_gap: self.shave_gap,
            forward: self.forward,
            intra_luma: crate::mts::luma(self.tools.mts, true, false, 0),
        }
    }

    /// This picture's SAO parameters (`self.sao`): classes searched on key
    /// pictures and every `vvc.sao_search_interval`-th P picture, reused in
    /// between.
    fn decide_sao(&mut self, source: &Picture, recon: &Picture) {
        let searched = !self.inter
            || self.sao_search.interval <= 1
            || self.poc.is_multiple_of(self.sao_search.interval)
            || self.sao_search.classes.is_empty();
        let skipped =
            (self.sao_search.skip_unchanged && self.inter).then(|| self.motion.skipped_ctus());
        let search = crate::sao::Search {
            row_step: self.sao_search.row_step,
            skipped: skipped.as_deref(),
            classes: (!searched).then_some(self.sao_search.classes.as_slice()),
        };
        self.sao = crate::sao::decide(
            &self.dsp,
            source,
            recon,
            [
                self.lambda * self.sao_search.luma_lambda_scale,
                self.quantizers().chroma_lambda(self.lambda),
            ],
            self.geometry.ctu_size(),
            &search,
        );
        if searched && self.sao_search.interval > 1 {
            // A skipped CTU's "off" says nothing about its classes: it keeps
            // the last search's until it is coded again.
            let mut classes = crate::sao::classes(&self.sao);
            if let Some(skipped) = &skipped
                && classes.len() == self.sao_search.classes.len()
            {
                for ((class, old), &skip) in classes
                    .iter_mut()
                    .zip(&self.sao_search.classes)
                    .zip(skipped)
                {
                    if skip {
                        *class = *old;
                    }
                }
            }
            self.sao_search.pending = Some(classes);
        }
    }

    /// The prediction of leaf `block` the engine decided as `decided`
    /// (intra may fall back to inter, [`Self::inter_fallback`]) and its
    /// luma direction, chroma direction and intra sub-partitions (DC and
    /// none for inter).
    fn block_mode(
        &mut self,
        block: &Block,
        decided: PredictionMode,
        source: &Picture,
        reference: Option<&Picture>,
        reconstruction: &Picture,
    ) -> (PredictionMode, (u8, u8, Option<crate::isp::Split>)) {
        let luma = (decided == PredictionMode::IntraDc)
            .then(|| self.luma_direction(block, source, reconstruction))
            .flatten();
        let mode = match self.inter_fallback(block, luma, source, reference) {
            Some(mv) => PredictionMode::Inter(mv),
            None => decided,
        };
        let directions = match (mode, luma) {
            (PredictionMode::IntraDc, Some(luma)) => {
                self.intra_directions(block, luma, source, reconstruction)
            }
            // DC, chroma derived from it: no search.
            _ => (crate::intra::DC, crate::intra::DC, None),
        };
        (mode, directions)
    }

    /// The best whole-CU luma direction of intra leaf `block`, its cost and
    /// the runner-up (`intra::choose`), or `None` where the search is DC
    /// only (DC, chroma derived from it).
    fn luma_direction(
        &mut self,
        block: &Block,
        source: &Picture,
        reconstruction: &Picture,
    ) -> Option<crate::intra::LumaChoice> {
        let search = if self.inter {
            self.inter_intra_search.search(
                block,
                self.geometry.coded_width,
                self.geometry.ctu_size(),
            )
        } else {
            self.key_intra_search
        };
        (search != crate::intra::DirectionSearch::DcOnly).then(|| {
            crate::intra::choose(
                &self.dsp,
                source,
                reconstruction,
                Some(&self.motion),
                block,
                &self.contexts,
                &self.quantizers(),
                self.lambda,
                search,
                &mut self.quant_scratch[0],
            )
        })
    }

    /// The inter vector to code intra leaf `block` with after all: the one
    /// the engine's directional estimate beat (`CodecBackend::intra_fallback`)
    /// where its luma rate-distortion cost (`inter_leaf_cost`) is below the
    /// intra direction's (`luma`, priced on the first transform unit and
    /// scaled to the CU's area). A CU whose search is DC only falls back
    /// unconditionally: the estimate promised a direction it cannot use.
    fn inter_fallback(
        &mut self,
        block: &Block,
        luma: Option<crate::intra::LumaChoice>,
        source: &Picture,
        reference: Option<&Picture>,
    ) -> Option<Mv> {
        let (at, mv) = self.intra_fallback.take()?;
        let reference = reference.filter(|_| at == (block.x, block.y, block.log2_size()))?;
        let Some(crate::intra::LumaChoice {
            cost: intra_cost, ..
        }) = luma
        else {
            return Some(mv);
        };
        // `intra::choose` prices a 64x64 CU on its 32x32 quadrant, with
        // or without 64-point transforms.
        let unit = transform_tree::units(block, transform_tree::MAX_TB_LOG2)
            .next()
            .expect("a CU has a transform unit");
        let intra_cost = intra_cost * (block.area() / unit.area()) as f32;
        let inter_cost = self.inter_leaf_cost(source, reference, block, mv)?;
        (inter_cost < intra_cost).then_some(mv)
    }

    /// Refine the directions of intra leaf `block` from its best whole-CU
    /// luma direction `luma` ([`Self::luma_direction`]): intra
    /// sub-partitions and the chroma direction.
    fn intra_directions(
        &mut self,
        block: &Block,
        luma: crate::intra::LumaChoice,
        source: &Picture,
        reconstruction: &Picture,
    ) -> (u8, u8, Option<crate::isp::Split>) {
        let (intra_mode, isp) = self.choose_isp(block, luma, source, reconstruction);
        let chroma_mode = if self.chroma_search[usize::from(self.inter)] {
            crate::chroma::choose(
                &self.dsp,
                source,
                reconstruction,
                Some(&self.motion),
                block,
                &self.contexts,
                &self.quantizers(),
                self.quantizers().chroma_lambda(self.lambda),
                intra_mode,
                self.chroma_rd_candidates,
                &mut self.quant_scratch,
            )
            .0
        } else {
            intra_mode
        };
        (intra_mode, chroma_mode, isp)
    }

    /// The luma direction and intra sub-partitions of `block`, whose best
    /// whole-CU direction `best` costs `cost` (`intra::choose`, without the
    /// ISP flag): sub-partitions in that direction, and in the runner-up
    /// `second` with `vvc.isp_modes` 2, are kept where they cost less.
    fn choose_isp(
        &mut self,
        block: &Block,
        luma: crate::intra::LumaChoice,
        source: &Picture,
        reconstruction: &Picture,
    ) -> (u8, Option<crate::isp::Split>) {
        let crate::intra::LumaChoice {
            mode: best,
            cost,
            runner_up: second,
            residual,
        } = luma;
        // Speed heuristic (`vvc.isp_residual_gate`, `VVenC`'s fast ISP): a CU
        // whose best whole-CU direction quantises its luma residual away
        // tries no sub-partitions. ISP must code a residual in some part,
        // and the parts' smaller transforms rarely keep levels the whole
        // CU's transform drops; it skips both ISP splits' prediction,
        // quantisation and pricing for such CUs.
        if !self.tools.isp
            || !crate::isp::supported(block, self.tools.mts.is_some())
            || (self.inter && !self.isp_search.inter)
            || (self.isp_search.residual_gate && !residual)
        {
            return (best, None);
        }
        let whole = cost + self.lambda * self.contexts.isp_mode.bits(false);
        let references = crate::intra::References::new(
            reconstruction,
            Some(&self.motion),
            PlaneIndex::Y,
            block,
            block,
        );
        let quant = self.isp_quant(&self.quantizers());
        let mpm = self.motion.intra_mpm(block);
        let modes = std::iter::once(best).chain(second.filter(|_| self.isp_search.modes > 1));
        let mut choice = (best, None, whole);
        for mode in modes {
            for split in crate::isp::Split::ALL {
                let cost = crate::isp::cost(
                    &self.dsp,
                    source,
                    &references,
                    block,
                    (split, mode, mpm),
                    &self.contexts,
                    (quant, self.lambda, choice.2),
                    &mut self.quant_scratch[0],
                );
                if cost < choice.2 {
                    choice = (mode, Some(split), cost);
                }
            }
        }
        (choice.0, choice.1)
    }

    /// The luma quantiser of intra sub-partitions: implicit MTS whenever
    /// the SPS enables MTS (§8.7.4.2).
    fn isp_quant(&self, quantizers: &recon::Quantizers) -> recon::Quant {
        recon::Quant {
            transform: crate::mts::luma(self.tools.mts, true, true, 0),
            ..quantizers.plane(PlaneIndex::Y, true)
        }
    }

    /// The `mts_idx` of CU `block` (`intra` or inter, one transform unit)
    /// whose luma predicts as `prediction` at quantiser `quant` (0 when
    /// it signals none).
    fn choose_mts(
        &mut self,
        source: &Picture,
        block: &Block,
        intra: bool,
        prediction: &[u8],
        quant: recon::Quant,
    ) -> u8 {
        let candidate = crate::mts::Candidate {
            source,
            cu: block,
            intra,
            prediction,
            quant,
            lambda: self.lambda * self.mts_search.lambda_scale,
        };
        crate::mts::choose(
            &self.dsp,
            &candidate,
            self.tools.mts,
            &self.contexts,
            self.mts_search,
            &mut self.quant_scratch[0],
        )
    }

    /// Reconstruct the luma of `block` as intra sub-partitions `split` in
    /// direction `mode`, one after the other; returns its levels, or `None`
    /// when they are all zero (the trellis may zero what the search's
    /// quantiser kept): the last sub-partition's coded flag would then be
    /// inferred 1 with nothing to code, so the CU cannot use ISP.
    fn reconstruct_isp(
        &mut self,
        block: &Block,
        split: crate::isp::Split,
        mode: u8,
        source: &Picture,
        reconstruction: &mut Picture,
    ) -> Option<Vec<i32>> {
        let references = crate::intra::References::new(
            reconstruction,
            Some(&self.motion),
            PlaneIndex::Y,
            block,
            block,
        );
        let quantizers = self.quantizers();
        let quant = self.isp_quant(&quantizers);
        let lambdas = [self.lambda, quantizers.chroma_lambda(self.lambda)];
        let mut levels = Vec::new();
        let mut predictor = crate::isp::Predictor::new(&references, block, split, mode);
        for (index, part) in crate::isp::parts(block, split).into_iter().enumerate() {
            let prediction = predictor.predict(&self.dsp, index, |region| {
                crate::isp::previous_in(reconstruction, block, split, region)
            });
            let level = self.reconstruct_unit(
                source,
                reconstruction,
                PlaneIndex::Y,
                (part.x, part.y, part.width()),
                &prediction,
                (quant, lambdas),
            );
            transform_tree::place(&mut levels, crate::isp::PARTS, index, level);
        }
        levels.iter().any(|&v| v != 0).then_some(levels)
    }
}
