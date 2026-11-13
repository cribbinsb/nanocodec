//! AV2 on the shared engine: the engine plans frames, partitions and block
//! modes and owns rate control and references; this backend writes AVM v1.0.0
//! syntax through [`BlockCoder`] and packages each picture.
use crate::block_coder::{
    self, BlockCoder, CdefMode, CoderOptions, DeltaSearch, EstimateModels, FilterSearch,
    FinishedTile, InterTxSearch, IntraSearch, LumaCandidates, PlaneRounding, SplitPolicy,
    TrellisOptions, TrialDistortion,
};
use crate::ccso::{CcsoParams, CcsoSource};
use crate::cdef::{self, CdefParams};
use crate::cdf_models::TileCdfs;
use crate::chroma_weight::ChromaWeight;
use crate::coeff::LevelCoding;
use crate::coeff_trim::TrimOptions;
use crate::deblock::DeblockParams;
use crate::frame_header::{InterHeader, KeyframeHeader, REFERENCE_SLOTS, REFRESH_ALL};
use crate::interp_filter::FrameFilter;
use crate::mv_refs::{self, InterMode, ListConfig};
use crate::pre_tx_skip::PreTxSkip;
use crate::prediction::InterpFilter;
use crate::sequence::{Layering, SequenceHeader, TcqFrames};
use crate::transform::{self, PlaneQindex, QuantDeltas};
use crate::tx_size::TxSize;
use crate::tx_type::{TxSetMode, TxType};
use crate::{motion, packet};
use nc_common::Picture;
use nc_encoder::{
    Block, BlockDecision, Codec, CodecBackend, ConfigError, EncoderConfig, FramePlan,
    FrameReferences, FrameType, IntraModeSet, LtrParams, Mv, PredictionMode, QuantParams,
};
use std::rc::Rc;

nc_encoder::param_group! {
    struct ToolParams ("av2") {
        cdf_adaptation: 1.0, [0.0, 1.0], "update CDFs within each tile (1) or code with fixed defaults (0)",
        tx_split: 1.0, [0.0, 2.0], "lossy luma transforms: one per block (0), one or a 4-way split by rate-distortion (1), always split (2)",
        p_tx_split: 1.0, [0.0, 2.0], "as av2.tx_split, for the blocks of P frames. AV1 Default codes one luma layout per inter block, but AV2 needs the split: 0 (with av2.p_intra_chroma_candidates=1) saves 8.2% of P instructions at CBR 600 yet J_real proxy-6 is WORSE (quality full +3.7% / low +2.1%; crowd, pan and static class guards)",
        p_intra_tx_split: 0.0, [0.0, 2.0], "as av2.p_tx_split, for the intra blocks of P frames only; default 0 (one layout, as AV1's P-frame intra) with p_intra_type_retry 0 and p_intra_chroma_candidates 1: P instructions -3.8% at CBR 600, J_real proxy-6 vs (1, 1, 7) -2.3% with the x86 dCPU (quality full +0.0% / low -0.1%, no guards); Jetson -5.5%",
        p_intra_type_retry: 0.0, [0.0, 1.0], "intra blocks of P frames retry their winning coding with ADST_ADST where the transform set allows it (1), or keep DCT (0, as AV1's P-frame intra)",
        estimate_adaptation: 0.0, [0.0, 1.0], "rate estimates in RD trials adapt a scratch copy of the tile's CDFs symbol by symbol (1) or price with the live CDFs frozen for the trial (0, default): P instructions -3.4% x86 / Jetson -1.4% at CBR 600; J_real proxy-6 vs 1 -1.0% (quality full +0.2% / low +0.3%, weighted dCPU -2.1%), no guards",
        lambda_per_step_squared: 0.135, [0.01, 4.0], "Lagrange multiplier per squared quantizer step (SSE per bit)",
        deblock: 1.0, [0.0, 1.0], "apply the deblocking filter with AVM's qindex-derived strength (1) or leave it off (0)",
        deblock_chroma: 1.0, [0.0, 1.0], "with av2.deblock, also deblock the chroma planes (1) or luma only (0; apply_deblocking_filter_u/v off). AV2 Fast takes 0: the chroma channels' edge walk and filters are 2.8% of its x86 instructions (bodycam / lowlight / crowd 720p CBR 600) and 2.7% of its Jetson time, for PSNR-YUV +2.0% but VMAF v1 -1.4%, MS-SSIM 0.0% (pool L2); J_real k 0.6 L2+L3 -1.14% BETTER, no guard",
        ccso: 0.0, [0.0, 1.0], "the sequence enables CCSO, the cross-component sample offset filter after CDEF (enable_ccso): 1 lets lossy frames filter each plane by offsets looked up from the deblocked luma around each sample, chosen from class statistics (crate::ccso_search); 0 off. Default 0: proxy-4 (60 f) J_real +0.5% (PSNR-YUV -0.2%, VMAF v1 +0.5%, MS-SSIM +1.0%); the AVM oracle puts CCSO at -0.6% PSNR-Y BD (docs/handoff/av2.md, \"In-loop luma filters\")",
        cdef: 2.0, [0.0, 2.0], "CDEF: off (0), the quantiser model's single strength set, each plane kept where a sampled error check favours it (1), or a strength search with up to cdef_max_sets sets chosen per 64x64 block (2)",
        cdef_check_row_period: 4.0, [1.0, 16.0], "the quantiser model's CDEF strengths (cdef=1) are kept per plane type where they lower the error measured on every n-th row of 8x8 units: higher checks faster on a smaller sample; rounded. AV2 Fast takes 16: x86 instructions -1.0% at 8 and -1.4% at 16 (bodycam / lowlight / crowd 720p CBR 600; crowd and bodycam streams identical, the check reaches the same verdicts), Jetson time -1.6%; pool L2 quality 0.0%, J_real k 0.6 L2+L3 -0.95% BETTER, no guard",
        cdef_max_sets: 4.0, [1.0, 8.0], "most CDEF strength sets a searched frame may code (cdef=2); more sets cost index bits per 64x64 block; rounded",
        cdef_search_row_period: 8.0, [1.0, 8.0], "CDEF search measures candidates on every n-th row of 8x8 units (cdef=2): higher searches faster on a smaller sample; rounded. Proxy-4 vs single-set: 4 -4.00%/-1.48%, 8 -4.36%/-1.75% PSNR-YUV/VMAF BD at +5.5%/+3.7% instructions",
        key_rounding_q7: 48.0, [16.0, 64.0], "key-frame coefficient rounding offset in 1/128 step (64 rounds to nearest; lower is an encoder-side deadzone)",
        inter_rounding_q7: 32.0, [16.0, 64.0], "P-frame coefficient rounding offset in 1/128 step (64 rounds to nearest; lower is an encoder-side deadzone)",
        inter_chroma_rounding_delta_q7: 0.0, [-32.0, 32.0], "added to av2.inter_rounding_q7 for the chroma transforms of P frames only (1/128 step; positive narrows chroma's deadzone without moving luma's)",
        dc_rounding_delta_q7: 0.0, [-32.0, 32.0], "added to the frame's rounding offset for each transform's DC coefficient (1/128 step; positive narrows the DC deadzone)",
        coeff_trim: 3.0, [0.0, 3.0], "drop trailing +-1 coefficient levels whose estimated rate outweighs their distortion: never (0), in every frame (1), in key frames only (2), in P frames only (3, default). J_real proxy-6 vs 0: P frames -2.7% (BETTER; VMAF v1 -0.9%, worst clip +1.7%); every frame -3.4% but VMAF v1 +9.3% on crowd_pp22 (trimming the key frame costs VMAF across the GOP); key frames plus P chroma, proxy-4: +1.1%; rounded",
        coeff_trim_rate_scale: 1.0, [0.25, 4.0], "scale of the static rate estimate that luma coefficient trimming weighs against distortion (higher trims more)",
        coeff_trim_chroma_rate_scale: 1.0, [0.0, 4.0], "as coeff_trim_rate_scale for chroma transforms (0 never trims chroma)",
        large_tx_rounding_delta_q7: 0.0, [-32.0, 32.0], "added to the frame's AC rounding offset for 16x16 and larger transforms (1/128 step; negative widens their deadzone)",
        interp_filter: 1.0, [0.0, 3.0], "frame inter prediction filter when av2.interp_switchable is 0: BILINEAR (0), EIGHTTAP_REGULAR (1, default), EIGHTTAP_SMOOTH (2) or MULTITAP_SHARP (3); 720p proxy-4: regular -2.30% PSNR / -6.96% VMAF BD vs bilinear (360p is mixed, +-0.15 dB); rounded",
        interp_switchable: 0.0, [0.0, 3.0], "per-block switchable interpolation filters in P frames: off (0, the frame filter av2.interp_filter), each block picks regular or smooth (1), regular, smooth or sharp (2), or regular or sharp (3) by luma prediction error plus filter bits; rounded. J_real proxy-4 vs off (max_gain 0.05): 1 +0.8% (PSNR -2.8% but VMAF NEG +3.6%, worst-NEG guard +5.9: smooth blurs), 2 +0.9% (guard), 3 +0.4% (quality -0.1%, P-frame CPU +0.9%); with decision.psy_strength 0.25 (which the search shares) vs switchable off: 1 +0.8%, 2 +0.3% (quality -0.75%, CPU +2.1%), no guard",
        interp_search_max_gain: 0.25, [0.0, 1.0], "switchable filter search: a candidate filter is predicted only if this fraction of the regular prediction's luma error exceeds its extra symbol cost (lambda * bits); 1 skips only candidates that cannot win, lower skips more; P-frame CPU with switchable 1 (crowd 720p q180): 0.25 +2.4%, 0.10 +1.6%, 0.05 +1.4%, with most of the PSNR gain kept at 0.05",
        intra_delta_search: 2.0, [0.0, 2.0], "angle deltas of the best-ranked intra direction: none (0), +-2 then the better side's +-1/+-3 (1), all six (2)",
        intra_chroma_candidates: 7.0, [1.0, 13.0], "leading entries of each block's chroma mode list ranked by SATD (13: all; 7: the single-symbol indices); key-frame guard (key/P <= 10x), with av2.key_plan_chroma=0 and av2.intra_layouts_per_candidate=0: crowd_mot17_09 720p q180 key 2310M -> 1917M, P 202.8M -> 198.0M (key/P 11.4x -> 9.7x); J_real proxy-4 vs the old defaults -2.1% (quality full -0.0%, low -0.1%, dCPU -3.4%, no guard)",
        intra_layouts_per_candidate: 0.0, [0.0, 1.0], "every intra RD candidate tries both luma transform layouts (1), or later candidates reuse the first one's winner (0); default 0 since the key-frame guard (see av2.intra_chroma_candidates)",
        intra_zero_exit: 0.0, [0.0, 1.0], "stop the intra RD trials at the first coding without residual (1) or try every candidate and layout (0)",
        key_plan_chroma: 0.0, [0.0, 1.0], "key-frame partition planning costs chroma too (1) or luma only (0); default 0 since the key-frame guard (see av2.intra_chroma_candidates)",
        intra_modes: 3.0, [0.0, 3.0], "key-frame intra luma candidates: the engine's shared set (0 DC only, 1 smooth/Paeth, 2 adds vertical/horizontal) plus the neighbours' modes, or (3) every nominal direction and the best one's angle deltas",
        p_intra_modes: 0.0, [0.0, 3.0], "as av2.intra_modes, for intra blocks of P frames; default 0 (DC only, as AV1 P frames): over the one-candidate, 7-chroma P search, P frames 167.6M -> 160.1M (-4.4%, crowd_mot17_09 720p q180), J_real proxy-4 vs 3: -2.7% (quality full -0.1%, low -0.2%, dCPU -4.2%, no guard)",
        p_intra_rd_candidates: 1.0, [1.0, 4.0], "as av2.intra_rd_candidates, for intra blocks of P frames; default 1 with p_intra_chroma_candidates=7 (vs 2 and 13): crowd_mot17_09 720p q180 P 177.8M -> 168.8M (-5.0%), J_real proxy-6 clips -2.5% (quality +0.3%, dCPU -4.7%, no guard); also dropping p_intra_modes to 1 (-8.4%) trips the worst-frame guard (bodycam worst-1% VMAF NEG +19.8%)",
        p_intra_delta_search: 2.0, [0.0, 2.0], "as av2.intra_delta_search, for intra blocks of P frames",
        p_intra_chroma_candidates: 1.0, [1.0, 13.0], "as av2.intra_chroma_candidates, for intra blocks of P frames; default 1 since the cheap P-intra path (see p_intra_tx_split); 7 before",
        inter_split_min_nonzero: 4.0, [0.0, 16.0], "inter blocks try the split luma layout only when the unsplit luma coding has at least this many nonzero levels (0: always); J_real proxy-4 vs 0: -7.4% (quality +0.3%, dCPU -12.8%)",
        intra_rd_candidates: 2.0, [1.0, 4.0], "key-frame intra luma modes (best by prediction SATD plus mode bits, each with its best chroma mode) given full transform rate-distortion",
        keyframe_step_permille: 240.0, [100.0, 1000.0], "fixed-QP key-frame quantizer step relative to the inter step, in 1/1000 (as av1.keyframe_step_permille); rate control picks key quantizers itself",
        intra_tx_types: 1.0, [0.0, 1.0], "intra luma transform types: DCT only (0), or DCT or ADST_ADST for TX4..TX16 (1: reduced_tx_set_used 1; the winning mode and layout is recoded with ADST_ADST and kept if cheaper by rate and distortion)",
        chroma_delta_q: -16.0, [-64.0, 63.0], "chroma qindex offset (u_ac_delta_q, shared by V and DC): negative codes chroma finer. AV2's step doubles every 24 qindex, so AV1's chroma_delta_q -32 is about -16 here at qindex 100 and -21 at 255. Default -16 (as AV1's -32), noise-attenuated by chroma_noise_scale: J_real TRAIN vs 0 -0.4% (PSNR-YUV -2.7%, NEG +1.0%, MS-SSIM +0.9%; full -0.2% / low -0.5%), no guards, periodic-key pulse -0.09 dB",
        chroma_delta_q_coarse: 0.0, [-32.0, 0.0], "extra chroma qindex offset reached at av2.chroma_delta_q_coarse_end, ramped linearly from zero at av2.chroma_delta_q_coarse_start (0 off): coarse frames (crowd, pan, static at low rates, qindex 140-205) take finer chroma, fine-q frames (bodycam_airport at qindex 85-120) keep av2.chroma_delta_q, where a uniformly finer offset (-28) cost MS-SSIM +7..+16%. The AV2 Default takes -12 (J_real k 0.1 vs 0: TRAIN -1.2%, w1% Y guard +2.08; VAL -1.2% BETTER; x86 instructions -0.2%)",
        chroma_delta_q_coarse_start: 120.0, [0.0, 255.0], "frame qindex at which av2.chroma_delta_q_coarse starts to apply",
        chroma_delta_q_coarse_end: 180.0, [1.0, 255.0], "frame qindex from which the whole av2.chroma_delta_q_coarse applies",
        chroma_delta_gap_slope: 2.0, [0.0, 4.0], "chroma qindex offset change per dB that the previous frame's chroma was cleaner than its luma beyond av2.chroma_delta_gap_pivot (finer chroma above it, coarser below; at most 16 either way; 0 off). J_real TRAIN (k 0.1): 1 vs 0 -0.3% BETTER (PSNR-YUV -0.7%, NEG -0.1%, MS-SSIM -0.2%; pan -1.2%), 2 vs 1 -0.4% BETTER (full -0.4% / low -0.3%; PSNR-YUV -0.8%, NEG -0.2%, MS-SSIM -0.2%; pan -1.2%, static -1.0%, texture +0.4%), no guard, periodic-key pulse +0.05 / -0.07 dB; no extra work (the ratio is already measured for chroma_weight_adapt)",
        chroma_delta_gap_pivot: 10.0, [0.0, 30.0], "luma-chroma PSNR gap (dB) at which av2.chroma_delta_gap_slope leaves the chroma offset unchanged",
        chroma_noise_scale: 2.0, [0.0, 32.0], "attenuate chroma_delta_q linearly to zero at this analyzed noise sigma (as av1.chroma_noise_scale): 0 disables adaptation; fractional values allowed",
        dc_delta_q: 0.0, [-64.0, 63.0], "DC quantizer offset of every plane below its AC qindex (y_dc_delta_q; u_dc_delta_q is the chroma AC offset plus this): negative codes the DC coefficient finer, as AV1's DC table does (its DC step is 0.83x the AC step at qindex 100 and 0.61x at 200, about -6 and -17 in AV2 units). Default 0: -16 on every frame is J_real proxy-6 +0.5% (pan -0.5%, static -0.6%, but bodycam +3.0% / +3.2%, a class guard)",
        key_dc_delta_q: -16.0, [-64.0, 63.0], "av2.dc_delta_q for key frames. Default -16 (vs 0): J_real TRAIN -0.6% BETTER (full -0.5% / low -0.7%; PSNR-YUV -0.8%, NEG -0.5%, MS-SSIM -0.2%, worst-1% NEG -1.1%; every class better), periodic-key pulse +0.05 dB, first key +0.04 dB",
        inter_tx_types: 0.0, [0.0, 2.0], "P-frame inter luma transform types: DCT only (0, default); DCT or IDTX (1: reduced_tx_set_used as for intra_tx_types, each transform of the winning layout retried with IDTX); or the full sets (2: reduced_tx_set_used 0, retried with ADST_ADST, ADST_DCT and DCT_ADST; P-frame intra blocks are then DCT only); kept per transform when cheaper by rate and distortion. J_real proxy-6, 2 vs 0: -0.7% (every clip better on J; VMAF v1 worst +0.8%) at +7.5% P-frame instructions, a net loss once CPU is weighed (J_real with dCPU about +3.8%), so off until cheaper; 1 vs 0, proxy-4: +0.3% (static VMAF v1 +4.9%); rounded",
        inter_tx_min_nonzero: 2.0, [1.0, 16.0], "inter transforms whose DCT coding has fewer nonzero levels skip the type search (inter_tx_types 1 and 2); rounded",
        pre_tx_skip: 0.0, [0.0, 1.0], "inter luma pre-transform skip (SVT light-PD1, as av1.pre_tx_skip with av1.pre_tx_skip_model=1): a block whose score reaches pre_tx_skip_score codes no luma residual and skips its luma transform trials",
        pre_tx_skip_energy: 15.0, [0.0, 100.0], "pre-transform low-energy gate: 100*SSE < energy*area*qindex (AV2 qindex); higher skips more",
        pre_tx_skip_score: 70.0, [0.0, 250.0], "pre-transform skip score threshold (low energy 50, both neighbours skipped +20, residual quantizes to zero +150, minus the low-quantizer protection); lower skips more",
        pre_tx_texture_scale: 8.0, [1.0, 16.0], "pre-transform zero-residual test: AC energy allowance in squared zero limits; higher skips more texture",
        pre_tx_luma_dominant: 1.0, [0.0, 1.0], "pre-transform low-quantizer protection: 1 treats every input as luma-dominant (stronger), 0 uses SVT's nondominant penalties",
        skip_estimate: 4.0, [0.0, 80.0], "inter blocks whose prediction error (all planes) is below this many lambda per 256 luma samples are coded as skipped without any residual trial or rate estimate; 0 off. Jetson (crowd q181 60 f) -2.3% at 4; J_real vs 0 (proxy-6, dCPU = x86 P instructions): 4 (default) -2.4% (-1.3% with the 2:1 Jetson:x86 dCPU of -3.3%) (quality full +0.5% / low +0.8%, bodycam +1.7% / +2.5%; P instructions -5.2%), no guards, periodic-key pulse +0.07 dB; 10 -2.6% but WORSE (bodycam quality +9.7% / +11.9% trips the class guards; P instructions -10.6%); 2: P instructions -2.4%",
        chroma_distortion_weight: 1.0, [0.25, 16.0], "weight of chroma squared error against luma in inter block decisions (skip, layouts) and the chroma trellis λ: higher keeps more chroma residual",
        chroma_rd_by_step: 1.0, [0.0, 1.0], "chroma squared error in inter block decisions and chroma trimming is weighted by the squared luma/chroma step ratio, as the chroma trellis λ is (1, default), or by chroma_distortion_weight alone (0). J_real proxy-6 vs 0: -1.6% (PSNR-YUV -5.5%, NEG +0.6%, MS-SSIM +0.1%; every clip better on J), no guard, periodic-key pulse +0.01 dB; callgrind (crowd_mot17_09 720p CBR 600, 11 f) -0.5%",
        chroma_weight_adapt: 0.67, [0.0, 4.0], "chroma error weight from the previous frame's MSE_Y / MSE_C times this gain (clamped 0.25..16) in place of the step-derived one (0 off); 0.67 is PSNR-YUV's marginal chroma value. Default 0.67 vs 0: J_real TRAIN -0.4% BETTER (full -0.4% / low -0.5%; PSNR-YUV -2.2%, NEG +0.7%, MS-SSIM +0.4%), no guard, periodic-key pulse -0.07 dB; callgrind +0.2%",
        zero_test_texture_scale: 4.0, [0.0, 8.0], "P-frame zero test of DCT_DCT transforms of 16x16 and larger that the exact SSE test cannot settle: nc_encoder::residual_quantises_to_zero (8x8 sub-block texture and the sub-block means) with this AC energy allowance in squared zero limits skips the forward transform when it predicts all-zero levels; 0 off. At 1 a near-proof (no coded transform skipped on crowd_mot17_09 / bodycam_airport q181): Jetson -1.2%, J_real proxy-6 -0.7% vs 0. At 4 (default): Jetson -4.8% and PC P instructions -2.7% vs 1; J_real proxy-6 vs 1 -2.2% with the Jetson dCPU (-1.0% with the PC one), quality full +0.7% / low +0.5%, no guards, periodic-key pulse 0.00 dB",
        trial_distortion: 0.0, [0.0, 2.0], "inter RD trials measure each coded plane's squared error on its reconstruction (0) or estimate it from the transform coefficients (Parseval over the levels' quantization error plus the kernels' measured rounding) and reconstruct only the coding that is kept, skipping the trials' inverse transforms and error passes: 1 for planes with transforms up to 32x32, 2 for every size; rounded. Jetson instructions (crowd/bodycam CBR 600) 1: -2.9%/-4.1%, 2: -2.8%/-4.2%. Off by default: 2 (without the rounding term) passed TRAIN (J -0.2) and VAL (J -0.1, no guard) but lost on bodycam_airport (~50 dB: J +5.6/+6.4, NEG +12.8%), where the estimate's bias is a large share of the coding error; 1 with the rounding term moved airport to over-estimation (val-proxy +5.5/+3.8). Needs a quantizer gate (estimate only where the step dwarfs the kernels' rounding)",
        max_drl_bits: 1.0, [1.0, 5.0], "DRL size: P-frame blocks may select among max_drl_bits + 1 predicted vectors (def_max_drl_bits; spatial neighbours, the reference MV bank, global motion and row/column combinations, AVM setup_ref_mv_list); rounded",
        cross_frame_cdf: 1.0, [0.0, 1.0], "inter frames start entropy coding from their reference picture's final CDFs (cross_frame_context, primary_ref_frame 0; only inter pictures pass theirs on) instead of the defaults; needs av2.cdf_adaptation. Default 1 (vs 0, no CPU): J_real TRAIN proxy-6 -3.4% BETTER (full -3.1% / low -3.7%, every clip better), VAL -2.6% BETTER (full -2.3% / low -3.0%; PSNR-YUV -2.6%, NEG -1.8%, MS-SSIM -2.4%; worst-1% Y -2.7%, NEG -3.1%; static_otw_homes390 +3.0% alone), no guard, periodic-key pulse -0.14 dB; long GOP (one key, 600 f) YUV +0.07..+0.19 dB per 200 frames, no drift",
        ref_mv_bank: 0.0, [0.0, 1.0], "reference MV bank (enable_refmvbank): open DRL entries take the superblock row's recent vectors",
        parity_hiding: 0.0, [0.0, 1.0], "the sequence enables parity hiding and lossy frames use it (allow_parity_hiding): luma transforms other than IDTX with more than four positions and at least four nonzero levels after DC leave the DC level's parity to the sum of those levels; the encoder fixes a disagreeing parity by the one-level move of least squared error plus lambda times approximate rate. Default 0: J_real TRAIN -0.3% BETTER (quality -0.4%, dCPU +1.1%: key frames +5-6%, P +1%) but VAL +0.1% WORSE (static +0.8%)",
        tcq: 0.0, [0.0, 2.0], "trellis-coded quantisation of lossy luma (enable_tcq per frame, tcq_mode): 0 off, 1 every lossy frame, 2 inter frames only; the luma levels of each transform are chosen by an 8-state Viterbi search over the two interleaved quantizers (distortion plus lambda times rate), and frames using it never hide parities. Default 0: with 2, J_real TRAIN proxy-6 quality full -2.9% / low -2.3%; P-frame instructions (crowd 720p CBR 600, 11 f) +44% (+74% before the search was reworked), Jetson 60 f crowd/bodycam instructions +33%/+42%, cycles +37%/+50%; J +1.8% at k 0.1 (see av2.tcq_prune_bits); key frames cost +239% with 1",
        tcq_prune_bits: 0.0, [0.0, 32.0], "TCQ search (av2.tcq) path pruning: at each position the paths whose cost trails the cheapest by more than this many bits (times lambda) are dropped; 0 keeps every state (exact Viterbi). Higher keeps more of TCQ's gain for more CPU. Luma at equal rate, crowd/pan/lowlight 60 f CBR vs TCQ off: exact +0.15/+0.23/+0.18 dB, 8 +0.13/+0.23/+0.17, 5 +0.12/+0.20/+0.16, 3 +0.12/+0.13/+0.10. P-frame instructions vs TCQ off (crowd 720p CBR 600, 11 f): exact +44%, 8 +39%, 5 +32%, 3 +25%. At 5, J_real proxy-6 quality full -2.8% / low -2.2%, J +1.0% at k 0.1 with dCPU 35%; Jetson 60 f crowd/bodycam instructions +23%/+28% but cycles +37%/+47% (pruning saves instructions, not branchy cycles)",
        skip_mode: 0.0, [0.0, 1.0], "inter frames with one fixed filter enable skip mode (skip_mode_flag): a block may code the same-reference compound NEAR_NEARMV of its skip-mode list (two neighbouring vectors averaged, MULTITAP_SHARP) with one flag and a DRL index, chosen where it predicts more cheaply than the single-reference choice by error plus signalling",
        skip_mode_candidates: 2.0, [1.0, 6.0], "leading skip-mode list entries each block ranks (av2.skip_mode); rounded",
        drl_ranked_entries: 6.0, [1.0, 6.0], "leading DRL entries mode decision ranks with the exact filter as candidate vectors; later entries only seed sub-sample refinement (bilinear-measured, nc_encoder mv_seeds) and are coded when refinement keeps them; 6: all ranked; rounded",
        max_leaf_log2: 6.0, [3.0, 6.0], "largest coding block the engine may choose (log2 of its side); 720p proxy-4: 32x32 vs 16x16 -11.78% PSNR / -9.51% VMAF BD, then 64x64 vs 32x32 -4.60% / -1.32% (every clip better)",
    }
}

/// `intra_modes` values from which every direction is ranked rather than
/// the engine's shared set.
const ALL_DIRECTIONS: f32 = 2.5;
/// Largest dimension the backend accepts.
const MAX_DIMENSION: usize = 4096;
/// Largest picture area, in 64x64 superblocks (uniform one-tile limit).
const MAX_SUPERBLOCKS: usize = 2304;
/// Largest native qindex.
const MAX_QINDEX: u32 = 255;
/// Rounding offsets stay inside one step (1/128 units): above zero, below
/// [`transform::ROUNDING_Q7_ONE`].
const MIN_ROUNDING_Q7: i32 = 1;
const MAX_ROUNDING_Q7: i32 = transform::ROUNDING_Q7_ONE - 1;
/// Superblock and smallest-block sizes (log2).
const SUPERBLOCK_LOG2: u8 = 6;
const MIN_BLOCK_LOG2: u8 = 3;
/// Quarter-sample engine vectors are eighth-sample AV2 vectors times two.
const EIGHTHS_PER_QUARTER: i32 = 2;
/// Order hints wrap at the sequence's eight order-hint bits.
const ORDER_HINT_MASK: u16 = 255;

/// The filters a switchable search tries besides regular
/// (`av2.interp_switchable` 1, 2 and 3).
const SMOOTH_CANDIDATE: [InterpFilter; 1] = [InterpFilter::Smooth];
const SMOOTH_SHARP_CANDIDATES: [InterpFilter; 2] = [InterpFilter::Smooth, InterpFilter::Sharp];
const SHARP_CANDIDATE: [InterpFilter; 1] = [InterpFilter::Sharp];
/// Chance (in 1/256) that a random-decision frame codes switchable filters
/// (otherwise a random fixed filter).
const RANDOM_SWITCHABLE_CHANCE: u64 = 128;

/// Parts per thousand.
const PERMILLE: f64 = 1000.0;
/// Chance (in 1/256) that a random-decision frame codes each quantizer delta at random.
const RANDOM_DELTA_Q_CHANCE: u64 = 128;

/// The finest lossy qindex whose step is at least `scale` times `qindex`'s
/// (lossless qindex 0 stays lossless).
fn key_qindex(qindex: u8, scale: f64) -> u8 {
    if qindex == 0 {
        return 0;
    }
    let target = f64::from(crate::transform::quantizer_step_q3(qindex)) * scale;
    (1..=qindex)
        .find(|&q| f64::from(crate::transform::quantizer_step_q3(q)) >= target)
        .unwrap_or(qindex)
}

/// The registered trellis controls (`nc_encoder::QuantParams`,
/// `docs/handoff/trellis.md`).
#[derive(Debug, Clone, Copy)]
struct TrellisSettings {
    options: TrellisOptions,
    /// Whether key frames are trellised too.
    key_frames: bool,
}

impl TrellisSettings {
    fn from_params(quant: &QuantParams) -> Option<Self> {
        (quant.trellis >= 0.5).then(|| Self {
            options: TrellisOptions {
                lambda_scale: quant.trellis_lambda_scale,
                chroma_scale: (quant.trellis_chroma > 0.0).then_some(quant.trellis_chroma),
                max_states: quant.trellis_states.round() as usize,
            },
            key_frames: quant.trellis_key >= 0.5,
        })
    }
}

/// Which frames and planes coefficient trimming touches (`av2.coeff_trim`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrimScope {
    Never,
    AllFrames,
    KeyFrames,
    PFrames,
}

impl TrimScope {
    fn from_param(value: f32) -> Self {
        match value.round() as u8 {
            0 => Self::Never,
            1 => Self::AllFrames,
            2 => Self::KeyFrames,
            _ => Self::PFrames,
        }
    }
}

/// The P-frame inter luma transform types (`av2.inter_tx_types`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InterTxTypes {
    /// `DCT_DCT` only.
    DctOnly,
    /// `DCT_DCT` or `IDTX`, the reduced sets' inter choice.
    DctIdtx,
    /// The full sets (`reduced_tx_set_used = 0`), searched over the
    /// [`FULL_SET_CANDIDATES`].
    Full,
}

/// The types inter transforms of a [`InterTxTypes::Full`] frame retry
/// besides `DCT_DCT`: the 2-D ADST combinations, which suit residuals
/// growing away from a predicted edge. The FLIPADST types, IDTX and the
/// 1-D types are left out to bound the search.
const FULL_SET_CANDIDATES: [TxType; 3] = [TxType::AdstAdst, TxType::AdstDct, TxType::DctAdst];
/// The type inter transforms of a [`InterTxTypes::DctIdtx`] frame retry.
const IDTX_CANDIDATE: [TxType; 1] = [TxType::Idtx];

impl InterTxTypes {
    fn from_param(value: f32) -> Self {
        match value.round() as u8 {
            0 => Self::DctOnly,
            1 => Self::DctIdtx,
            _ => Self::Full,
        }
    }

    fn candidates(self) -> &'static [TxType] {
        match self {
            Self::DctOnly => &[],
            Self::DctIdtx => &IDTX_CANDIDATE,
            Self::Full => &FULL_SET_CANDIDATES,
        }
    }
}

/// Native AV2 backend on the shared engine.
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent per-stream and per-frame switches"
)]
pub struct Av2Backend {
    /// Coded picture size: the visible size padded to whole 8x8 blocks.
    coded_width: usize,
    coded_height: usize,
    params: ToolParams,
    /// `quant.trellis`: trellis quantisation of the coded transforms.
    trellis: Option<TrellisSettings>,
    order_hint: u16,
    qindex: u8,
    /// The current frame's quantizer offsets.
    quant_deltas: QuantDeltas,
    /// The analyzer's latest sensor-noise estimate.
    noise_sigma: f32,
    /// The engine chroma policy (`chroma.policy` 1): replaces
    /// `av2.chroma_delta_q`, its noise and gap adaptations, and the
    /// adaptive chroma error weight.
    chroma_policy: Option<nc_encoder::chroma_policy::ChromaPolicy>,
    key: bool,
    /// Whether rate control chooses frame quantizers (no fixed-QP key boost).
    rate_control: bool,
    coder: Option<BlockCoder>,
    /// The current frame's deblocking, chosen in `filter_frame`.
    deblock: DeblockParams,
    /// The current frame's CDEF, chosen in `filter_frame`.
    cdef: CdefParams,
    /// The current frame's CCSO, chosen in `filter_frame`.
    ccso: CcsoParams,
    /// The deblocked luma CCSO classifies on, reused across frames.
    ccso_source: CcsoSource,
    /// The current frame's reduced transform set.
    tx_set: TxSetMode,
    /// The current frame's interpolation filter mode.
    frame_filter: FrameFilter,
    /// `decision.psy_strength`, which the filter search shares with the
    /// engine's ranking.
    psy_strength: f32,
    /// CDEF's copy of the unfiltered plane and the search's buffers, reused
    /// across frames.
    cdef_scratch: crate::cdef::CdefScratch,
    /// The deblocking edge map, reused across frames: it is kept clear
    /// between uses, so a frame neither allocates nor zeroes it.
    deblock_map: crate::deblock::EdgeMap,
    /// `MSE_Y / MSE_C` of the last committed frame, measured while
    /// `av2.chroma_weight_adapt` is on.
    chroma_error_ratio: Option<f64>,
    /// The current frame's ratio, committed with the frame (a recoded
    /// frame must not see its own first coding).
    frame_chroma_error_ratio: Option<f64>,
    /// Kernels of the frame-level measurements.
    dsp: nc_dsp::Dsp,
    /// Test-only random syntax source, lent to each frame's coder.
    random: Option<nc_encoder::DebugRng>,
    /// Each reference slot's picture's final CDFs, adaptation restarted,
    /// as a decoder keeps them (`av2.cross_frame_cdf`).
    slot_contexts: [Option<Rc<TileCdfs>>; REFERENCE_SLOTS as usize],
    /// The current frame starts from its reference's CDFs.
    inherits_context: bool,
    /// The coded frame's final CDFs, stored into its slots on commit.
    frame_context: Option<Rc<TileCdfs>>,
    /// The sequence header every key picture repeats.
    sequence: SequenceHeader,
    /// The current frame's temporal layer, reference slot and refreshes.
    references: FrameReferences,
    /// Lossy frames code superblock quantizer deltas
    /// ([`CodecBackend::enable_block_qp`]).
    block_qp: bool,
}

/// The sequence's layering for `config`: layered and long-term-reference
/// structures keep several pictures in the slots and name their reference
/// explicitly.
fn layering(config: &EncoderConfig) -> Layering {
    let long_term = LtrParams::from_overrides(&config.params).interval > 0.0;
    Layering {
        temporal_layers: config.temporal_layers,
        explicit_references: config.temporal_layers > 1 || long_term,
    }
}

impl Av2Backend {
    /// Whether inter frames start from their reference's final CDFs.
    /// How this frame codes its lossy luma levels: trellis-coded where
    /// `av2.tcq` covers the frame (1 every lossy frame, 2 inter frames
    /// only), else with hidden DC parities where `av2.parity_hiding` is on;
    /// lossless frames are plain.
    fn level_coding(&self) -> LevelCoding {
        let tcq_frames = match self.params.tcq.round() as u8 {
            0 => false,
            1 => true,
            _ => !self.key,
        };
        if self.qindex == 0 {
            LevelCoding::Plain
        } else if tcq_frames && self.sequence.tcq == TcqFrames::Signalled {
            LevelCoding::Tcq
        } else if self.sequence.parity_hiding && self.params.parity_hiding >= 0.5 {
            LevelCoding::ParityHiding
        } else {
            LevelCoding::Plain
        }
    }

    /// Whether this frame enables skip mode (`av2.skip_mode`): inter frames
    /// with one fixed filter (the subset does not track skip-mode blocks'
    /// switchable filter contexts), lossy.
    fn skip_mode_enabled(&self) -> bool {
        self.params.skip_mode >= 0.5
            && !self.key
            && self.qindex != 0
            && matches!(self.frame_filter, FrameFilter::Fixed(_))
    }

    /// DRL entries offered as ranked candidates (`av2.drl_ranked_entries`).
    fn ranked_drl_entries(&self) -> usize {
        self.params.drl_ranked_entries.round() as usize
    }

    fn cross_frame_cdf(&self) -> bool {
        self.params.cross_frame_cdf >= 0.5 && self.params.cdf_adaptation >= 0.5
    }

    /// The frame header's `delta_q` (`log2(delta_q_res)`, coded 0): lossy
    /// frames code superblock deltas once the engine enabled block QP;
    /// lossless frames have no delta syntax.
    fn frame_delta_q(&self) -> Option<u8> {
        (self.block_qp && self.qindex != 0).then_some(0)
    }

    /// Start the frame's block coder with `options`, from the inherited
    /// CDFs where the frame has them.
    fn start_coder(&mut self, options: &CoderOptions) {
        let random = self.random.take();
        let inherited = self.inherited_context();
        let inherits = inherited.is_some();
        let coder = BlockCoder::new(
            (self.coded_width, self.coded_height),
            self.qindex,
            self.key,
            options,
            inherited,
            random,
        );
        self.inherits_context = inherits;
        self.coder = Some(coder.expect("validated AV2 dimensions"));
    }

    /// The CDFs the current frame starts from: its reference slot's, for an
    /// inter frame with `av2.cross_frame_cdf`; `None` for the defaults.
    fn inherited_context(&self) -> Option<&TileCdfs> {
        if self.key || !self.cross_frame_cdf() {
            return None;
        }
        let (reference, _) = self.slot_signalling();
        self.slot_contexts[usize::from(reference.unwrap_or(0))].as_deref()
    }

    /// The current inter frame's signalled reference slot (`None`:
    /// implicit, every slot holds the previous picture, as slot 0) and the
    /// slots it refreshes. A flat sequence keeps one picture in every
    /// slot; explicit references follow the engine's slots.
    fn slot_signalling(&self) -> (Option<u8>, u8) {
        if self.sequence.layering.explicit_references {
            (
                Some(self.references.reference.slot as u8),
                self.references.refresh.0,
            )
        } else {
            (None, REFRESH_ALL)
        }
    }
}

/// The sequence's motion-vector reference lists from `av2.max_drl_bits`
/// and `av2.ref_mv_bank`.
fn reference_lists(params: &ToolParams) -> ListConfig {
    ListConfig {
        max_drl_bits: params.max_drl_bits.round() as u8,
        ref_mv_bank: params.ref_mv_bank >= 0.5,
    }
}

fn to_av2(mv: Mv) -> (i32, i32) {
    (
        i32::from(mv.y) * EIGHTHS_PER_QUARTER,
        i32::from(mv.x) * EIGHTHS_PER_QUARTER,
    )
}

fn to_engine((row, col): (i32, i32)) -> Mv {
    Mv {
        x: (col / EIGHTHS_PER_QUARTER) as i16,
        y: (row / EIGHTHS_PER_QUARTER) as i16,
    }
}

/// AV2's rate-model calibration. Its qindex scale is not AV1's: the step
/// doubles every 24 qindex from a different base (`transform.rs`), so at
/// the shared seed qindex 180 its frames cost less. With AV1's bootstrap
/// 0.03 bits/pixel its first key landed at 0.4-0.7x the key budget where
/// AV1 and VP9 land at 0.7-1.3x (CBR 150 and 800 kbps, 720p, four clips);
/// 0.016 matches them (CBR staged proxy-4, single key 240 f: PSNR-YUV /
/// VMAF BD -1.4% / -3.5%, worst-1% -6.0% / -9.5%).
///
/// AV2's bits and quality react far more to a step change between
/// consecutive frames than AV1's (`pan_jaad_0101` at 600 kbps: a frame
/// coarser than its finer reference collapses to near-skip, about -1 dB),
/// so its QP cadence was made gentler (scale 0.7) to pass the old
/// frame-to-frame flicker guard. Under the window flicker guard and MS-SSIM
/// on every frame the engine's full cadence (scale 1) wins: `J_real` (k 0.1)
/// vs 0.7, TRAIN 18 clips -1.6%, VAL 11 clips -1.0%, held-out
/// (`broad_independent_v1`, 13 clips, every class) -2.3%, no guard;
/// 900-frame single-key runs (chirla, crowd, pan) -0.04..+0.12 dB Y per
/// third. `av2-fast` keeps 0.7 (its VAL lost on static, +4.5%).
///
/// Start-up lookahead first-key cap (`rate.first_key_lookahead`) 0.5
/// buffers under camera motion, looser than AV1's 0.3 as for HEVC. `J_real`
/// TRAIN, 150 f, vs off: 0.5 -0.2% (pan -0.6%, worst-1% Y -0.6%, no clip
/// worse); 0.3 -0.1% with 5/18 clips worse at low rate (texture +0.5%).
const RATE_MODEL: nc_encoder::RateModel = nc_encoder::RateModel {
    bootstrap_bits_per_pixel: 0.016,
    first_key_lookahead: 0.5,
    // `rate_control_quantizer_step` divides out the table's Q3 precision
    // (`QUANT_TABLE_BITS`), leaving AV1's scale: eighths of a sample step,
    // AV1's `sample_step_scale`. At 1 the reuse term (`nc_encoder`
    // `reuse.rs`) took every step for 8x its size and judged two in five
    // still cells of a starved key not refinable (`av2-fast` stayed 0.9 dB
    // short for good: `nc-cli/tests/static_recovery.rs`).
    ..nc_encoder::RateModel::AC_LOOKUP_QINDEX
};

impl Av2Backend {
    /// The registered fixed interpolation filter (`av2.interp_filter`).
    fn fixed_filter(&self) -> InterpFilter {
        match self.params.interp_filter.round() as u8 {
            0 => InterpFilter::Bilinear,
            1 => InterpFilter::Regular,
            2 => InterpFilter::Smooth,
            _ => InterpFilter::Sharp,
        }
    }

    /// The filters a switchable frame's block search tries besides regular;
    /// `None` when frames use the fixed filter (`av2.interp_switchable`).
    fn switchable_candidates(&self) -> Option<&'static [InterpFilter]> {
        match self.params.interp_switchable.round() as u8 {
            0 => None,
            1 => Some(&SMOOTH_CANDIDATE),
            2 => Some(&SMOOTH_SHARP_CANDIDATES),
            _ => Some(&SHARP_CANDIDATE),
        }
    }

    /// The current frame's filter mode: the fixed filter, or per-block
    /// filters. Random-decision frames pick either at random.
    fn choose_frame_filter(&mut self) -> FrameFilter {
        if let Some(random) = self.random.as_mut() {
            if random.chance(RANDOM_SWITCHABLE_CHANCE) {
                return FrameFilter::Switchable;
            }
            let filters = [
                InterpFilter::Regular,
                InterpFilter::Smooth,
                InterpFilter::Sharp,
                InterpFilter::Bilinear,
            ];
            return FrameFilter::Fixed(filters[random.below(filters.len() as u64) as usize]);
        }
        match self.switchable_candidates() {
            Some(_) => FrameFilter::Switchable,
            None => FrameFilter::Fixed(self.fixed_filter()),
        }
    }

    /// The registered CDEF strength choice (unused when CDEF is off).
    fn cdef_mode(&self) -> CdefMode {
        /// `cdef` values from which the search runs.
        const SEARCH: f32 = 1.5;
        if self.params.cdef >= SEARCH {
            CdefMode::Search(cdef::SearchOptions {
                max_sets: self.params.cdef_max_sets.round() as usize,
                row_period: self.params.cdef_search_row_period.round() as usize,
            })
        } else {
            CdefMode::Model(self.params.cdef_check_row_period.round() as usize)
        }
    }

    /// The frame's quantizer rounding by plane kind and transform size: the
    /// key or P-frame offset (P-frame chroma moved by
    /// `av2.inter_chroma_rounding_delta_q7`), with the DC and
    /// large-transform deltas, kept within a step.
    fn rounding(&self) -> PlaneRounding {
        let (luma, chroma) = if self.key {
            (self.params.key_rounding_q7, self.params.key_rounding_q7)
        } else {
            let base = self.params.inter_rounding_q7;
            (base, base + self.params.inter_chroma_rounding_delta_q7)
        };
        let by_size = |base: f32| {
            let offset = |delta: f32| {
                ((base + delta).round() as i32).clamp(MIN_ROUNDING_Q7, MAX_ROUNDING_Q7)
            };
            TxSize::ALL.map(|tx| {
                let large = if tx >= TxSize::Tx16 {
                    self.params.large_tx_rounding_delta_q7
                } else {
                    0.0
                };
                transform::Rounding {
                    dc_q7: offset(self.params.dc_rounding_delta_q7),
                    ac_q7: offset(large),
                }
            })
        };
        PlaneRounding {
            luma: by_size(luma),
            chroma: by_size(chroma),
        }
    }

    /// The frame's coefficient trimming (`av2.coeff_trim` scope).
    fn trim_options(&self) -> Option<TrimOptions> {
        let applies = match TrimScope::from_param(self.params.coeff_trim) {
            TrimScope::Never => false,
            TrimScope::AllFrames => true,
            TrimScope::KeyFrames => self.key,
            TrimScope::PFrames => !self.key,
        };
        applies.then(|| TrimOptions {
            luma_rate_scale: f64::from(self.params.coeff_trim_rate_scale),
            chroma_rate_scale: f64::from(self.params.coeff_trim_chroma_rate_scale),
        })
    }

    /// The frame's quantizer offsets. Chroma AC: the noise-attenuated
    /// `av2.chroma_delta_q` on every frame type, moved by the previous
    /// frame's luma/chroma error gap ([`nc_encoder::error_gap_offset`]). Unlike AV1
    /// (`av1.keyframe_chroma_scaled`), key frames need no rescaling: AV2's
    /// step doubles every 24 qindex, so the same offset keeps the inter
    /// chroma/luma step ratio at the finer key qindex. DC: `av2.dc_delta_q`
    /// (`av2.key_dc_delta_q` on key frames) below each plane's AC qindex, so
    /// a plane's DC/AC step ratio does not depend on its AC offset. Lossless frames code none. Random-decision
    /// frames code random deltas.
    fn frame_quant_deltas(&mut self) -> QuantDeltas {
        if self.qindex == 0 {
            return QuantDeltas::NONE;
        }
        let range = i32::from(transform::MIN_DELTA_Q)..=i32::from(transform::MAX_DELTA_Q);
        let clamp = |delta: f32| (delta.round() as i32).clamp(*range.start(), *range.end()) as i8;
        if let Some(random) = self.random.as_mut() {
            let span = (range.end() - range.start() + 1) as u64;
            let mut draw = || {
                if random.chance(RANDOM_DELTA_Q_CHANCE) {
                    (range.start() + random.below(span) as i32) as i8
                } else {
                    0
                }
            };
            return QuantDeltas {
                luma_dc: draw(),
                chroma_ac: draw(),
                chroma_dc: draw(),
            };
        }
        let chroma_ac = match self.chroma_policy {
            Some(policy) => {
                policy_chroma_delta(policy, u32::from(self.qindex), range.clone()) as f32
            }
            None => self.legacy_chroma_delta(),
        };
        let dc = if self.key {
            self.params.key_dc_delta_q
        } else {
            self.params.dc_delta_q
        };
        QuantDeltas {
            luma_dc: clamp(dc),
            chroma_ac: clamp(chroma_ac),
            chroma_dc: clamp(chroma_ac + dc),
        }
    }

    /// The chroma AC offset without the engine policy: the noise-attenuated
    /// `av2.chroma_delta_q` moved by the previous frame's error gap.
    fn legacy_chroma_delta(&self) -> f32 {
        nc_encoder::noise_attenuated_chroma_offset(
            self.params.chroma_delta_q + self.coarse_chroma_extra(),
            self.noise_sigma,
            self.params.chroma_noise_scale,
        ) + nc_encoder::error_gap_offset(
            self.chroma_error_ratio,
            self.params.chroma_delta_gap_slope,
            self.params.chroma_delta_gap_pivot,
        )
    }

    /// `av2.chroma_delta_q_coarse`, ramped in by the frame's qindex. The
    /// tuned -16 is the best single offset over all rates, but it is a
    /// compromise: at coarse quantizers chroma is cheap to keep and AV1's
    /// matched step is about -28, while bodycam clips coding at fine
    /// quantizers lose MS-SSIM to finer chroma (handoff/av2.md, "Chroma
    /// rounding and speed arms").
    fn coarse_chroma_extra(&self) -> f32 {
        let p = &self.params;
        nc_encoder::coarse_chroma_extra(
            f32::from(self.qindex),
            p.chroma_delta_q_coarse,
            p.chroma_delta_q_coarse_start,
            p.chroma_delta_q_coarse_end,
        )
    }

    /// The frame's chroma error weight ([`chroma_weight`]).
    fn chroma_weight(&self) -> f64 {
        let weight = match self.chroma_policy {
            Some(_) => ChromaWeight::BY_STEP,
            None => ChromaWeight {
                scale: self.params.chroma_distortion_weight,
                by_step: self.params.chroma_rd_by_step >= 0.5,
                adapt_gain: self.params.chroma_weight_adapt,
            },
        };
        weight.of(
            PlaneQindex::new(self.qindex, self.quant_deltas),
            self.chroma_error_ratio,
        )
    }

    fn coder(&mut self) -> &mut BlockCoder {
        self.coder
            .as_mut()
            .expect("block callbacks run inside a frame")
    }

    /// The intra search of the current frame: P frames have their own
    /// candidate counts and delta search (their intra blocks are few, and
    /// cheaper to get slightly wrong than key-frame blocks).
    fn intra_search(&self) -> IntraSearch {
        let p = &self.params;
        let (modes, rd_candidates, delta_search, chroma_candidates) = if self.key {
            (
                p.intra_modes,
                p.intra_rd_candidates,
                p.intra_delta_search,
                p.intra_chroma_candidates,
            )
        } else {
            (
                p.p_intra_modes,
                p.p_intra_rd_candidates,
                p.p_intra_delta_search,
                p.p_intra_chroma_candidates,
            )
        };
        IntraSearch {
            modes: if modes >= ALL_DIRECTIONS {
                LumaCandidates::AllDirections
            } else {
                LumaCandidates::Shared(IntraModeSet::from_param(modes))
            },
            rd_candidates: rd_candidates.round() as usize,
            delta_search: match delta_search.round() as u8 {
                0 => DeltaSearch::Off,
                1 => DeltaSearch::CoarseToFine,
                _ => DeltaSearch::Exhaustive,
            },
            chroma_candidates: chroma_candidates.round() as usize,
            layouts_per_candidate: p.intra_layouts_per_candidate >= 0.5,
            zero_residual_exit: p.intra_zero_exit >= 0.5,
        }
    }
}

impl CodecBackend for Av2Backend {
    const PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] = &[ToolParams::SPECS];
    /// Every `av2.*` setting is read when a frame begins (the block coder,
    /// filters and quantizer offsets are built from `self.params` there),
    /// so each may take a non-base frame value (`nonbase.av2.<name>`).
    const FRAME_PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] = &[ToolParams::SPECS];
    /// `begin_frame` builds a fresh block coder from the plan (no CDF
    /// state crosses frames); `commit_frame` advances the order hint.
    const RECODE: bool = true;

    /// Superblock quantizer deltas (`delta_q`, [`crate::delta_q`]): one
    /// qindex per 64x64 superblock, set by its first block.
    const SUPPORTS_BLOCK_QP: bool = true;
    const BLOCK_QP_LOG2: u8 = SUPERBLOCK_LOG2;

    /// Speed arms taken from AV2 Fast, with chroma's P-frame rounding at
    /// AV1's 48/128 (`av2.inter_chroma_rounding_delta_q7` 16; luma keeps
    /// 32) paying for their quality: exact-rank pruning at margin 1.1 and
    /// no cheap-vector refinement (x86 instructions -1.3% and -3.8% alone;
    /// proxy-4 `J_real` +0.1% and +0.4%). The bundle against the Default:
    /// x86 instructions -4.1% (crowd / bodycam / static -4.9 / -5.7 /
    /// -1.7%), `J_real` TRAIN -1.8% and VAL -1.8% BETTER; see
    /// docs/handoff/av2.md "Chroma rounding and speed arms". Coarse frames
    /// then code chroma finer still (`av2.chroma_delta_q_coarse` -12,
    /// ramped in from qindex 120 to 180): `J_real` TRAIN and VAL -1.2%.
    /// Luma then takes some of it back: P-frame luma rounding 36/128
    /// (chroma stays at 48: delta 12) and λ 0.125 per squared step (was
    /// 0.135): TRAIN -0.7%, VAL -0.2% (MS-SSIM -1.0% / -0.6%), x86
    /// instructions about +0.1% (docs/handoff/av2.md "Luma give-back").
    ///
    /// Trellis quantisation of P frames' coded blocks (`quant.*`,
    /// `docs/handoff/trellis.md`): λ×1, one survivor, chroma at 0.25, key
    /// frames at the plain quantiser; it replaces trimming on the coded
    /// block (trials still trim). `J_real` staged TRAIN (18 clips) BETTER:
    /// quality −3.3% full / −3.0% low, every clip better (PSNR-YUV −2.7%,
    /// NEG −4.7%, MS-SSIM −2.4%, VMAF v1 −5.9%), no guard, for +14.9% host
    /// P-frame instructions (CBR 600) and +10.8% Jetson time: J −1.9 at
    /// k 0.1. λ×0.6 (AV1's) spent the bits trimming saved and lost PSNR on
    /// pans (proxy-4 J −0.1, worst-1% Y guard).
    ///
    /// First-key boost `rate.first_key_boost_scale` 3 (engine 2.5): at the
    /// engine value AV2's first key landed ~15% smaller than AV1's (static
    /// and lowlight clips, same loan) and 0.4–0.6 dB lower, and static
    /// scenes inherit it. `J_real` TRAIN 150 f vs 2.5: −1.4% BETTER (full
    /// −1.2% / low −1.6%; static −3.8%, lowlight −1.6%, bodycam +0.5%;
    /// worst-1% Y static −6.5%), no guard, periodic pulse −0.02 dB; buffer
    /// overrun frames 49 vs 13, no non-compliant point. 3.5 was −2.7% but
    /// bodycam +1.1% and 138 overrun frames. Long GOP (600 f, one key):
    /// within ±0.05 dB, no drift.
    ///
    /// Inter re-code ceiling `rate.recode_overflow_inter` 1.1 buffers (keys
    /// keep the engine's 1.4): AV2's static refinement bursts (15-17x a
    /// frame's budget, 0.5-0.6 buffers) overran a strict 1 s CBR buffer on
    /// 107 board frames against our AV1's 40, nearly all on
    /// `static_london_bus`, peaking at 1.34-1.37 buffers, under the 1.4
    /// re-code. `J_real` VAL (11 clips, full gate) vs 1.4: overrun frames 66
    /// -> 12, `J_real` +0.4% (`static_london_bus` +5.1%, `homes390` +0.2%,
    /// every other clip byte-identical), no guard; TRAIN identical (no TRAIN
    /// clip bursts past 1.1). The refinement given up is the price of
    /// compliance, as for the 1.4 ceiling itself. Long GOP (one key):
    /// `long_static_chirla_s024c2` 150 kbps YUV per 200 f -0.39/-0.35/-0.15
    /// dB at equal rate (overrun frames 43 -> 2), 300 kbps -0.04..-0.08 dB;
    /// `static_london_bus` 300 f the first 100 f -0.2..-0.4 dB, then level.
    const DEFAULT_PARAMS: &'static [(&'static str, f32)] = &[
        ("decision.exact_rank_margin", 1.1),
        ("decision.refine_cheap_vector", 0.0),
        ("av2.inter_rounding_q7", 36.0),
        // Tuned jointly (eval/nceval/tune_pool.py, study
        // av2-default-eq2-2026-11-11, equal x86 CPU): chroma rounding delta
        // 12 -> 22 and planner lambda 2 -> 2.58. Study clips J -1.45%, L3 of
        // the study's draw -0.62% [-0.96, -0.30] (overfit gap +0.83);
        // re-checked at head 91ad0e87 on the study's rotation: disjoint L3
        // (79 clips, single-point) J -0.71% [-1.05, -0.40], L2 (ladder)
        // -1.00% [-1.41, -0.61], no guard. CPU x86 +0.6%, Jetson cycles
        // +1.1%. av2-fast pins the previous planner lambda.
        ("av2.inter_chroma_rounding_delta_q7", 22.0),
        ("planner.lambda_scale", 2.581_25),
        ("av2.lambda_per_step_squared", 0.125),
        ("av2.chroma_delta_q_coarse", -12.0),
        ("quant.trellis", 1.0),
        ("quant.trellis_key", 0.0),
        ("quant.trellis_states", 1.0),
        ("quant.trellis_lambda_scale", 1.0),
        ("rate.first_key_boost_scale", 3.0),
        ("rate.recode_overflow_inter", 1.1),
        // Start-up QP cadence (period 8, base frames 5 QP finer before AV2's
        // 0.7 cadence scale, for the first 100 frames after a key) and the
        // cadence motion gate (`qp_cadence.rs`; docs/handoff/vvc.md,
        // qp-cascade). J_real (k 0.1, dCPU 0) TRAIN (18 clips): boost 4
        // BETTER -1.5%, boost 5 BETTER -2.5% (lowlight -5.4%, static -4.0%,
        // crowd -3.6%, bodycam +0.3%), worst-1% Y -1.2%, pulse -0.22 dB;
        // VAL (val-proxy-5, full gate) BETTER -5.3% (static -18.2%, crowd
        // -6.3%), pulse -0.11 dB. Long GOP (900 f, one key, CBR 300): chirla
        // Y 37.20 -> 37.27, pan_bdd 31.25 -> 31.26. AV2 Fast (k 0.6,
        // proxy-6) BETTER -2.2%, pulse -0.16 dB.
        ("rate.cadence_startup_period", 8.0),
        ("rate.cadence_startup_base_boost", 5.0),
        ("rate.cadence_motion_gate", 1.0),
        // Depth removal at half the engine's threshold scale (SVT
        // `qp_scale_factor`): the scale grows with the headroom below the
        // top quantizer, and AV2's coarser qindex scale (step doubling every
        // 24) leaves more headroom than AV1 at the same step, so at 2 the
        // planner dropped 16x16 children that pay on moving and noisy
        // content. J_real (k 0.1) vs 2: TRAIN (18 clips) BETTER -0.7% (full
        // -0.9 / low -0.5; crowd -2.0, lowlight -1.9, worst-1% Y -1.3%),
        // no guard; VAL (11 clips, full gate) -1.2% (VMAF v1 -2.5%), no
        // guard. x86 instructions -0.9..+0.6%; long GOP without drift
        // (docs/handoff/av2.md "Depth removal scale"). Switching removal
        // off below a quantizer gate instead (0.5..1) lost the static class
        // guard on `static_otw_homes390` at low rates.
        ("planner.depth_removal_qp_scale", 1.0),
    ];

    // AV1's motion-decision quality point is the engine default (see
    // `nc_av1`'s `DEFAULT_PARAMS` for each value's AV1 measurement); AV2
    // keeps the engine's DC-aware intra test. Before AV2 took these values
    // it under-used sub-sample vectors on camera pans: `pan_jaad_0101` q174, 45% of
    // NEAR/NEW area had integer vectors in both components (AV1 at equal
    // rate: 37%); with them 33%. CBR ladder 150..1200 kbps, 4 pan clips,
    // 150 f: luma +0.10..+0.30 dB at equal rate on every point.

    fn new(config: &EncoderConfig) -> std::result::Result<Self, ConfigError> {
        if config.codec != Codec::Av2 {
            return Err(ConfigError::Unsupported("AV2 backend requires AV2 codec"));
        }
        if config.signal.colour != nc_common::ColourInfo::default() {
            return Err(ConfigError::Unsupported(
                "AV2 colour metadata overrides are not yet supported",
            ));
        }
        let (coded_width, coded_height) = packet::coded_size(config.width, config.height);
        let superblock = 1 << SUPERBLOCK_LOG2;
        if config.width == 0
            || config.height == 0
            || coded_width > MAX_DIMENSION
            || coded_height > MAX_DIMENSION
            || coded_width.div_ceil(superblock) * coded_height.div_ceil(superblock)
                > MAX_SUPERBLOCKS
        {
            return Err(ConfigError::Dimensions {
                width: config.width,
                height: config.height,
                multiple: 1 << MIN_BLOCK_LOG2,
            });
        }
        if config.quantizer > MAX_QINDEX {
            return Err(ConfigError::Quantizer {
                value: config.quantizer,
                min: 0,
                max: MAX_QINDEX,
            });
        }
        // Non-base settings (`nonbase.<name>`) are the engine's to check.
        let (_, ordinary) = config.params.split_prefix(nc_encoder::NONBASE_PREFIX);
        ordinary
            .validate(&[nc_encoder::engine_specs(), Self::PARAM_SPECS.to_vec()].concat())
            .map_err(ConfigError::Param)?;
        if config.key_qp_offset.is_some() {
            return Err(ConfigError::Unsupported(
                "AV2 does not yet support key-QP overrides",
            ));
        }
        Ok(Self {
            coded_width,
            coded_height,
            params: ToolParams::from_overrides(&config.params),
            trellis: TrellisSettings::from_params(&QuantParams::from_overrides(&config.params)),
            order_hint: 0,
            qindex: 0,
            quant_deltas: QuantDeltas::NONE,
            noise_sigma: 0.0,
            chroma_policy: nc_encoder::chroma_policy::ChromaPolicy::from_params(&config.params),
            key: true,
            rate_control: false,
            coder: None,
            deblock: DeblockParams::OFF,
            cdef: CdefParams::OFF,
            ccso: CcsoParams::OFF,
            ccso_source: CcsoSource::new(),
            tx_set: TxSetMode::IntraDct,
            frame_filter: FrameFilter::Fixed(InterpFilter::Regular),
            psy_strength: nc_encoder::DecisionParams::from_overrides(&config.params).psy_strength,
            cdef_scratch: crate::cdef::CdefScratch::default(),
            deblock_map: crate::deblock::EdgeMap::default(),
            chroma_error_ratio: None,
            frame_chroma_error_ratio: None,
            dsp: nc_dsp::Dsp::new(),
            random: None,
            slot_contexts: Default::default(),
            inherits_context: false,
            frame_context: None,
            sequence: packet::sequence_header(
                config.width,
                config.height,
                layering(config),
                reference_lists(&ToolParams::from_overrides(&config.params)),
                {
                    let tools = ToolParams::from_overrides(&config.params);
                    (
                        tools.parity_hiding >= 0.5,
                        tools.tcq >= 0.5,
                        tools.ccso >= 0.5,
                    )
                },
            ),
            references: FrameReferences::FLAT,
            block_qp: false,
        })
    }

    fn superblock_log2(&self) -> u8 {
        SUPERBLOCK_LOG2
    }

    fn min_block_log2(&self) -> u8 {
        MIN_BLOCK_LOG2
    }

    fn max_leaf_log2(&self) -> u8 {
        self.params.max_leaf_log2.round() as u8
    }

    fn supports_temporal_layers(&self) -> bool {
        true
    }

    fn supports_long_term_reference(&self) -> bool {
        true
    }

    fn supports_inter(&self) -> bool {
        true
    }

    fn coded_size(&self, width: usize, height: usize) -> (usize, usize) {
        packet::coded_size(width, height)
    }

    fn enable_random_decisions(&mut self, rng: nc_encoder::DebugRng) {
        self.random = Some(rng);
        // Random decisions exercise every filter's syntax.
        self.sequence.ccso = true;
    }

    fn enable_block_qp(&mut self) {
        self.block_qp = true;
    }

    /// The shared kernel that computes the frame's filter exactly; motion
    /// search in switchable frames ranks with regular, the filter every
    /// block starts its filter search from.
    fn luma_filter(&self) -> Option<nc_dsp::Filter> {
        let filter = match self.frame_filter {
            FrameFilter::Fixed(filter) => filter,
            FrameFilter::Switchable => InterpFilter::Regular,
        };
        Some(filter.kernel())
    }

    // The engine's exact luma candidates are reused by the block encode
    // (`BlockCoder::cache_luma`) instead of predicted again.
    const CACHE_EXACT_LUMA: bool = true;

    fn cache_exact_luma(
        &mut self,
        block: &Block,
        mv: Mv,
        filter: nc_dsp::Filter,
        prediction: &[u8],
        _stats: nc_dsp::Variance,
    ) {
        // Only a fixed frame filter predicts every block with this kernel;
        // switchable blocks choose theirs while coding.
        let FrameFilter::Fixed(frame_filter) = self.frame_filter else {
            return;
        };
        if frame_filter.kernel() != filter {
            return;
        }
        self.coder().cache_luma(
            (block.x, block.y),
            block.size(),
            to_av2(mv),
            frame_filter,
            prediction,
        );
    }

    fn lambda(&self) -> f32 {
        let step = block_coder::pixel_step(self.qindex);
        (f64::from(self.params.lambda_per_step_squared) * step * step) as f32
    }

    fn relative_quantizer(&self) -> f32 {
        f32::from(self.qindex) / MAX_QINDEX as f32
    }

    fn mv_candidates(&mut self, block: &Block, out: &mut Vec<Mv>) {
        let ranked = self.ranked_drl_entries();
        if let Ok(candidates) = self.coder().candidates(block.x, block.y, block.size()) {
            let derived = candidates.derived();
            out.extend(
                derived[..ranked.min(derived.len())]
                    .iter()
                    .map(|&v| to_engine(v)),
            );
        }
    }

    fn mv_seeds(&mut self, block: &Block, out: &mut Vec<Mv>) {
        let ranked = self.ranked_drl_entries();
        if let Ok(candidates) = self.coder().candidates(block.x, block.y, block.size()) {
            let derived = candidates.derived();
            out.extend(
                derived[ranked.min(derived.len())..]
                    .iter()
                    .map(|&v| to_engine(v)),
            );
        }
    }

    fn mode_bits(&self, block: &Block, mode: &PredictionMode) -> f32 {
        let Some(coder) = self.coder.as_ref() else {
            return 0.0;
        };
        let is_inter = matches!(mode, PredictionMode::Inter(_));
        let type_bits = if self.key {
            0.0
        } else {
            coder.prediction_type_bits(block.x, block.y, block.size(), is_inter)
        };
        let PredictionMode::Inter(mv) = *mode else {
            return type_bits;
        };
        let Ok(candidates) = coder.candidates(block.x, block.y, block.size()) else {
            return type_bits;
        };
        let target = to_av2(mv);
        let (signal, index) = block_coder::choose_signalling(candidates, target);
        let mut bits = type_bits + mv_refs::mode_bits(candidates, signal, index);
        if signal == InterMode::New {
            bits += motion::difference_bits(target, candidates.vectors[index]);
        }
        bits
    }

    fn supports_intra_leaf_cost(&self) -> bool {
        true
    }

    fn intra_leaf_cost(&mut self, source: &Picture, block: &Block) -> Option<f32> {
        let cost = self
            .coder()
            .intra_leaf_cost(source, (block.x, block.y), block.size())
            .expect("key-frame leaves are codable");
        Some(cost as f32)
    }

    /// The cheapest inter leaf at the current probabilities
    /// ([`BlockCoder::cheapest_leaf_bits`]), as AV1 and VP9 report it.
    /// Without it a split paid only its partition symbols, so still and
    /// easy content split for prediction gains worth under a bit. `J_real`
    /// (k 0.1, dCPU -0.3%: x86 instructions -0.5% at CBR 600) TRAIN BETTER
    /// -0.5% (full -0.3% / low -0.7%; bodycam -1.3% / -1.6%), no guard,
    /// periodic-key pulse -0.08 dB; VAL (11 clips, full gate) +0.0%
    /// (pan -0.7%, static +1.5%), no guard.
    fn planner_leaf_bits(&self) -> f32 {
        self.coder
            .as_ref()
            .map_or(0.0, BlockCoder::cheapest_leaf_bits)
    }

    fn partition_bits(&self, node: &Block, split: bool) -> f32 {
        self.coder.as_ref().map_or(0.0, |coder| {
            coder.partition_bits(node.x, node.y, node.size(), split)
        })
    }

    fn rate_model(&self) -> nc_encoder::RateModel {
        RATE_MODEL
    }

    fn rate_control_quantizer_step(&self, quantizer: u32) -> f64 {
        f64::from(crate::transform::quantizer_step_q3(
            quantizer.min(MAX_QINDEX) as u8,
        )) / 8.0
    }

    fn set_rate_control_enabled(&mut self, enabled: bool) {
        self.rate_control = enabled;
    }

    fn set_frame_params(&mut self, params: &nc_encoder::ParamOverrides) {
        self.params = ToolParams::from_overrides(params);
    }

    fn set_noise_sigma(&mut self, sigma: f32) {
        self.noise_sigma = sigma;
    }

    fn begin_frame(&mut self, plan: &FramePlan) {
        self.key = plan.frame_type == FrameType::Key;
        self.references = plan.references;
        let planned = plan.quantizer.min(MAX_QINDEX) as u8;
        // Fixed-QP key frames code at a finer step, as AV1's do: a static
        // scene keeps the key frame's detail for the whole GOP.
        self.qindex = if self.key && !self.rate_control {
            key_qindex(
                planned,
                f64::from(self.params.keyframe_step_permille) / PERMILLE,
            )
        } else {
            planned
        };
        self.quant_deltas = self.frame_quant_deltas();
        self.frame_filter = self.choose_frame_filter();
        let inter_tx = InterTxTypes::from_param(self.params.inter_tx_types);
        // Lossless intra blocks infer DCT (WHT) whatever the set.
        self.tx_set = if self.qindex == 0 {
            TxSetMode::IntraDct
        } else if !self.key && inter_tx == InterTxTypes::Full {
            TxSetMode::Full
        } else if self.params.intra_tx_types >= 0.5 {
            TxSetMode::IntraDctAdst
        } else {
            TxSetMode::IntraDct
        };
        let policy = |value: f32| match value.round() as u8 {
            0 => SplitPolicy::Never,
            1 => SplitPolicy::RateDistortion,
            _ => SplitPolicy::Always,
        };
        let (split, intra_split, intra_type_retry) = if self.key {
            let split = policy(self.params.tx_split);
            (split, split, true)
        } else {
            (
                policy(self.params.p_tx_split),
                policy(self.params.p_intra_tx_split),
                self.params.p_intra_type_retry >= 0.5,
            )
        };
        let options = CoderOptions {
            adaptive_cdf: self.params.cdf_adaptation >= 0.5,
            split,
            intra_split,
            intra_type_retry,
            estimates: if self.params.estimate_adaptation >= 0.5 {
                EstimateModels::Adapting
            } else {
                EstimateModels::Frozen
            },
            lambda_per_step_squared: f64::from(self.params.lambda_per_step_squared),
            rounding: self.rounding(),
            quant_deltas: self.quant_deltas,
            delta_q: self.frame_delta_q().is_some(),
            filter: self.frame_filter,
            skip_mode: self
                .skip_mode_enabled()
                .then(|| self.params.skip_mode_candidates.round() as usize),
            filter_search: FilterSearch {
                candidates: self.switchable_candidates().unwrap_or(&SMOOTH_CANDIDATE),
                max_gain: f64::from(self.params.interp_search_max_gain),
                psy_strength: self.psy_strength,
            },
            intra: self.intra_search(),
            plan_chroma: self.params.key_plan_chroma >= 0.5,
            inter_split_min_nonzero: self.params.inter_split_min_nonzero.round() as usize,
            cdef: self.cdef_mode(),
            ccso: self.sequence.ccso,
            tx_set: self.tx_set,
            levels: self.level_coding(),
            tcq_prune_bits: (self.params.tcq_prune_bits > 0.0)
                .then_some(self.params.tcq_prune_bits),
            inter_tx: InterTxSearch {
                candidates: inter_tx.candidates(),
                min_nonzero: self.params.inter_tx_min_nonzero.round() as usize,
            },
            trim: self.trim_options(),
            trellis: self
                .trellis
                .filter(|t| t.key_frames || !self.key)
                .map(|t| t.options),
            pre_tx_skip: (self.params.pre_tx_skip >= 0.5).then_some(PreTxSkip {
                energy: self.params.pre_tx_skip_energy,
                score: self.params.pre_tx_skip_score,
                texture_scale: self.params.pre_tx_texture_scale,
                luma_dominant: self.params.pre_tx_luma_dominant >= 0.5,
            }),
            skip_estimate: (self.params.skip_estimate > 0.0).then_some(self.params.skip_estimate),
            chroma_weight: self.chroma_weight(),
            zero_test_texture_scale: (self.params.zero_test_texture_scale > 0.0)
                .then_some(self.params.zero_test_texture_scale),
            trial_distortion: match self.params.trial_distortion.round() as u32 {
                0 => TrialDistortion::Reconstructed,
                1 => TrialDistortion::EstimatedBelow64,
                _ => TrialDistortion::Estimated,
            },
            reference_lists: self.sequence.reference_lists,
        };
        self.start_coder(&options);
    }

    fn begin_superblock(&mut self, x: usize, y: usize) {
        self.coder()
            .begin_superblock(x, y)
            .expect("the engine walks the superblock grid");
    }

    fn write_partition(&mut self, node: &Block, split: bool) {
        self.coder()
            .write_partition(node.x, node.y, node.size(), split)
            .expect("engine quadtrees are expressible in AV2 partitions");
    }

    fn encode_block(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        source: &Picture,
        reference: Option<&Picture>,
        recon: &mut Picture,
    ) {
        let coder = self.coder();
        coder.set_block_offset(decision.qp_offset);
        let position = (block.x, block.y);
        match (decision.mode, reference) {
            (PredictionMode::Inter(mv), Some(reference)) => {
                coder.encode_inter(position, block.size(), to_av2(mv), source, reference, recon)
            }
            _ => coder.encode_intra(position, block.size(), source, recon),
        }
        .expect("engine decisions are codable");
    }

    /// Deblock, then CDEF, then CCSO, the finished reconstruction (it
    /// becomes the reference); CDEF strengths are measured against `source`.
    fn filter_frame(&mut self, source: &Picture, recon: &mut Picture) {
        self.deblock = if self.params.deblock < 0.5 {
            DeblockParams::OFF
        } else if self.params.deblock_chroma < 0.5 {
            DeblockParams::from_qindex(self.qindex).luma_only()
        } else {
            DeblockParams::from_qindex(self.qindex)
        };
        if self.deblock.is_on() {
            let params = self.deblock;
            let coder = self.coder.as_mut().expect("filtering runs inside a frame");
            coder.deblock(recon, params, &mut self.deblock_map);
        }
        let ccso = self.sequence.ccso && self.qindex != 0;
        if ccso {
            self.ccso_source
                .fill_from(&recon.plane(nc_common::PlaneIndex::Y).view());
        }
        self.cdef = if self.params.cdef >= 0.5 {
            let coder = self.coder.as_mut().expect("filtering runs inside a frame");
            coder.cdef(source, recon, self.key, &mut self.cdef_scratch)
        } else {
            CdefParams::OFF
        };
        self.ccso = if ccso {
            let coder = self.coder.as_mut().expect("filtering runs inside a frame");
            coder.ccso(source, recon, &self.ccso_source)
        } else {
            CcsoParams::OFF
        };
        let measures_ratio = self.chroma_policy.is_none()
            && (self.params.chroma_weight_adapt > 0.0 || self.params.chroma_delta_gap_slope > 0.0);
        if measures_ratio && self.qindex != 0 {
            self.frame_chroma_error_ratio = nc_encoder::plane_error_ratio(&self.dsp, source, recon);
        }
    }

    fn end_frame(&mut self) -> Vec<u8> {
        let FinishedTile {
            bytes: tile,
            models,
            random,
        } = self.coder.take().expect("frame in progress").finish();
        self.random = random;
        // Only inter pictures pass their CDFs on. AVM would also inherit a
        // key picture's, but dav2d (237229e) derives no primary reference
        // from a list holding only key pictures and decodes such a frame
        // from the defaults; inheriting from inter pictures alone keeps the
        // decoders in agreement.
        let passes_context = self.cross_frame_cdf() && !self.key;
        self.frame_context = models.filter(|_| passes_context).map(|mut models| {
            models.restart_adaptation();
            Rc::from(models)
        });
        let adaptive_cdf = self.params.cdf_adaptation >= 0.5;
        let packet = if self.key {
            packet::key(
                &self.sequence,
                KeyframeHeader {
                    order_hint: 0,
                    quantizer: self.qindex,
                    quant_deltas: self.quant_deltas,
                    transform_select: self.qindex != 0,
                    tx_set: self.tx_set,
                    levels: self.level_coding(),
                    adaptive_cdf,
                    deblock: self.deblock,
                    cdef: self.cdef,
                    ccso: self.ccso,
                    delta_q: self.frame_delta_q(),
                },
                &tile,
            )
        } else {
            let (reference, refresh_flags) = self.slot_signalling();
            let header = InterHeader {
                order_hint: self.order_hint,
                quantizer: self.qindex,
                quant_deltas: self.quant_deltas,
                transform_select: self.qindex != 0,
                tx_set: self.tx_set,
                levels: self.level_coding(),
                refresh_flags,
                reference,
                adaptive_cdf,
                cross_frame_context: self.inherits_context,
                deblock: self.deblock,
                cdef: self.cdef,
                ccso: self.ccso,
                filter: self.frame_filter,
                skip_mode: self.skip_mode_enabled(),
                delta_q: self.frame_delta_q(),
            };
            packet::inter(
                &self.sequence,
                self.references.position.layer,
                header,
                &tile,
            )
        };
        packet.expect("validated AV2 headers")
    }

    fn commit_frame(&mut self) {
        if let Some(ratio) = self.frame_chroma_error_ratio.take() {
            self.chroma_error_ratio = Some(ratio);
        }
        let refresh = if self.key {
            REFRESH_ALL
        } else {
            self.slot_signalling().1
        };
        let context = self.frame_context.take();
        for (slot, slot_context) in self.slot_contexts.iter_mut().enumerate() {
            if refresh & (1 << slot) != 0 {
                slot_context.clone_from(&context);
            }
        }
        // A key frame has order hint 0, so the next inter frame's is 1.
        self.order_hint = if self.key {
            1
        } else {
            (self.order_hint + 1) & ORDER_HINT_MASK
        };
    }
}

/// The chroma AC `delta_q` (within `range`) that puts chroma's step at frame
/// qindex `qindex` nearest the engine chroma policy's step
/// (`nc_encoder::chroma_policy`; [`pixel_step`] is in its sample units).
fn policy_chroma_delta(
    policy: nc_encoder::chroma_policy::ChromaPolicy,
    qindex: u32,
    range: std::ops::RangeInclusive<i32>,
) -> i32 {
    let step = |delta: i32| {
        crate::block_coder::pixel_step((qindex as i32 + delta).clamp(0, MAX_QINDEX as i32) as u8)
    };
    let target = policy.chroma_step(step(0));
    nc_encoder::chroma_policy::closest_index(range, step, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::PlaneIndex;
    use nc_encoder::{Encoder, FrameRate, ParamOverrides, RateControlConfig};

    fn config() -> EncoderConfig {
        EncoderConfig {
            codec: Codec::Av2,
            width: 16,
            height: 16,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 0,
            intra_period: 30,
            temporal_layers: 1,
            block_log2: 3,
            key_qp_offset: None,
            signal: nc_encoder::StreamSignal::default(),
            // The tests code repeated pictures.
            params: ParamOverrides::default().coding_every_frame(),
        }
    }
    fn bytes(picture: &Picture) -> Vec<u8> {
        let mut bytes = Vec::new();
        nc_common::yuv::write_frame(&mut bytes, picture).unwrap();
        bytes
    }
    #[test]
    fn shared_engine_keyframes_and_written_input() {
        let mut encoder = Encoder::<Av2Backend>::new(config()).unwrap();
        let mut decoder = crate::Decoder::default();
        for frame in 0..4 {
            let picture = encoder.source_mut();
            for plane in PlaneIndex::ALL {
                let mut view = picture.plane_mut(plane).view_mut();
                for y in 0..view.height() {
                    view.row_mut(y).fill(60 + frame * 4);
                }
            }
            let source = bytes(encoder.source_mut());
            if frame == 2 {
                encoder.request_keyframe();
            }
            let encoded = encoder.encode_written();
            assert_eq!(
                encoded.metadata.frame_type == FrameType::Key,
                frame == 0 || frame == 2
            );
            let decoded = decoder.decode(&encoded.data).unwrap();
            assert_eq!(bytes(&decoded), bytes(encoder.reconstruction()));
            assert_eq!(bytes(&decoded), source);
        }
    }
    #[test]
    fn order_hint_wrap_and_forced_key_keep_reference_chain() {
        let mut config = config();
        config.width = 8;
        config.height = 8;
        config.intra_period = 300;
        let mut encoder = Encoder::<Av2Backend>::new(config).unwrap();
        let mut decoder = crate::Decoder::default();
        let mut source = Picture::new(8, 8, 0).unwrap();
        for index in 0..261 {
            source
                .plane_mut(PlaneIndex::Y)
                .view_mut()
                .row_mut(0)
                .fill(index as u8);
            if index == 258 {
                encoder.request_keyframe();
            }
            let packet = encoder.encode(&source);
            assert_eq!(
                packet.metadata.frame_type == FrameType::Key,
                index == 0 || index == 258
            );
            assert_eq!(
                bytes(decoder.decode_ref(&packet.data).unwrap()),
                bytes(&source)
            );
        }
    }

    #[test]
    fn shared_rate_feedback_reconfigures_and_fixed_quantizer_returns() {
        let mut encoder = Encoder::<Av2Backend>::new(config()).unwrap();
        let source = Picture::new(16, 16, 0).unwrap();
        let mut rc = RateControlConfig::new(80_000);
        rc.min_quantizer = Some(40);
        rc.max_quantizer = Some(180);
        encoder.set_rate_control(rc).unwrap();
        let first = encoder.encode(&source);
        assert!((40..=180).contains(&first.metadata.quantizer));
        assert!(encoder.rate_control_stats().is_some());
        rc.target_bps = 40_000;
        encoder.set_rate_control(rc).unwrap();
        let second = encoder.encode(&source);
        assert!((40..=180).contains(&second.metadata.quantizer));
        encoder.disable_rate_control();
        assert_eq!(encoder.encode(&source).metadata.quantizer, 0);
        assert!(encoder.rate_control_stats().is_none());
    }
    #[test]
    fn entropy_and_transform_parameters_work_through_shared_engine() {
        for (adaptation, split) in [(0.0, 0.0), (1.0, 1.0), (1.0, 2.0)] {
            let mut config = config();
            config.quantizer = 0;
            config.params.set("av2.cdf_adaptation", adaptation);
            config.params.set("av2.tx_split", split);
            let mut encoder = Encoder::<Av2Backend>::new(config).unwrap();
            let mut decoder = crate::Decoder::default();
            let mut source = Picture::new(16, 16, 0).unwrap();
            for frame in 0..3 {
                for y in 0..16 {
                    for (x, pixel) in source
                        .plane_mut(PlaneIndex::Y)
                        .view_mut()
                        .row_mut(y)
                        .iter_mut()
                        .enumerate()
                    {
                        *pixel = ((x + frame) * 17 + y * 11) as u8;
                    }
                }
                let packet = encoder.encode(&source);
                assert_eq!(
                    bytes(decoder.decode_ref(&packet.data).unwrap()),
                    bytes(&source)
                );
            }
        }
    }

    #[test]
    fn shared_scene_controls_insert_keys_without_changing_disabled_packets() {
        let mut baseline = Encoder::<Av2Backend>::new(config()).unwrap();
        let mut default_packets = Vec::new();
        for enabled in [false, true] {
            let mut config = config();
            config.width = 64;
            config.height = 64;
            config.params.set("scene.detect", f32::from(enabled));
            config.params.set("rate.cadence_fixed_qp", 0.0);
            assert!(Av2Backend::new(&config).is_ok());
            if !enabled {
                let mut plain = config.clone();
                plain.params = ParamOverrides::default().coding_every_frame();
                plain.params.set("rate.cadence_fixed_qp", 0.0);
                baseline = Encoder::<Av2Backend>::new(plain).unwrap();
            }
            let mut encoder = Encoder::<Av2Backend>::new(config).unwrap();
            let mut decoder = crate::Decoder::default();
            for frame in 0..12 {
                for plane in PlaneIndex::ALL {
                    let mut view = encoder.source_mut().plane_mut(plane).view_mut();
                    for y in 0..view.height() {
                        view.row_mut(y).fill(if frame < 8 { 40 } else { 210 });
                    }
                }
                let source = encoder.source_mut().clone();
                if frame == 11 {
                    encoder.request_keyframe();
                    baseline.request_keyframe();
                }
                let encoded = encoder.encode_written();
                assert_eq!(
                    encoded.metadata.frame_type == FrameType::Key,
                    frame == 0 || frame == 11 || (enabled && frame == 8)
                );
                assert_eq!(
                    bytes(decoder.decode_ref(&encoded.data).unwrap()),
                    bytes(&source)
                );
                if !enabled {
                    assert_eq!(encoded.data, baseline.encode(&source).data);
                    default_packets.push(encoded.data);
                } else if frame < 8 {
                    assert_eq!(encoded.data, default_packets[frame]);
                }
            }
        }
        let mut invalid = config();
        invalid.params.set("scene.detect", 2.0);
        assert!(Av2Backend::new(&invalid).is_err());
    }

    #[test]
    fn scene_analysis_preserves_references_across_drops_and_forced_keys() {
        let mut config = config();
        config.width = 64;
        config.height = 64;
        config.params.set("scene.detect", 1.0);
        config.params.set("scene.min_key_interval", 300.0);
        let mut encoder = Encoder::<Av2Backend>::new(config).unwrap();
        let mut decoder = crate::Decoder::default();
        let mut rc = RateControlConfig::new(1_000);
        rc.frame_drop = true;
        rc.buffer_ms = 1000;
        rc.min_quantizer = Some(1);
        rc.max_quantizer = Some(2);
        encoder.set_rate_control(rc).unwrap();
        let mut dropped = 0;
        let mut previous = Vec::new();
        for frame in 0..10 {
            for plane in PlaneIndex::ALL {
                let mut view = encoder.source_mut().plane_mut(plane).view_mut();
                for y in 0..view.height() {
                    for (x, pixel) in view.row_mut(y).iter_mut().enumerate() {
                        *pixel = (x * 17 + y * 31 + (x * y + 1) * (frame + 1) * 53) as u8;
                    }
                }
            }
            if frame == 6 {
                encoder.request_keyframe();
            }
            let encoded = encoder.encode_written();
            if frame == 6 {
                assert!(!encoded.metadata.dropped);
                assert_eq!(encoded.metadata.frame_type, FrameType::Key);
            }
            if encoded.metadata.dropped {
                dropped += 1;
                assert_eq!(encoded.data, [] as [u8; 0]);
                assert_eq!(bytes(encoder.reconstruction()), previous);
            } else {
                let decoded = decoder.decode_ref(&encoded.data).unwrap();
                assert_eq!(bytes(decoded), bytes(encoder.reconstruction()));
                previous = bytes(decoded);
            }
        }
        assert!(
            dropped > 0,
            "fixture must drop frames before forced refresh"
        );
    }

    #[test]
    fn deblocked_reconstruction_is_the_decoded_picture_and_the_reference() {
        let mut recons = Vec::new();
        for deblock in [0.0, 1.0] {
            let mut config = config();
            config.width = 64;
            config.height = 48;
            config.quantizer = 210;
            config.params.set("av2.deblock", deblock);
            let mut encoder = Encoder::<Av2Backend>::new(config).unwrap();
            let mut decoder = crate::Decoder::default();
            let mut source = Picture::new(64, 48, 0).unwrap();
            for frame in 0..4 {
                for plane in PlaneIndex::ALL {
                    let mut view = source.plane_mut(plane).view_mut();
                    for y in 0..view.height() {
                        for (x, pixel) in view.row_mut(y).iter_mut().enumerate() {
                            *pixel = ((x + 2 * frame) * 9 + y * 5 + (x * y) % 23) as u8;
                        }
                    }
                }
                let packet = encoder.encode(&source);
                let decoded = decoder.decode_ref(&packet.data).unwrap();
                assert_eq!(bytes(decoded), bytes(encoder.reconstruction()));
            }
            recons.push(bytes(encoder.reconstruction()));
        }
        assert_ne!(recons[0], recons[1], "the filter must change the picture");
    }

    /// Block AQ codes superblock quantizer deltas (`delta_q`) that our
    /// decoder follows exactly (dequantization and per-block deblocking),
    /// and changes the stream; without AQ the header codes none.
    #[test]
    fn block_qp_streams_reconstruct_exactly_and_differ() {
        let packets = |aq_strength: f32| {
            let mut config = config();
            (config.width, config.height, config.quantizer) = (192, 128, 150);
            config.params.set("rate.aq_strength", aq_strength);
            config.params.set("rate.aq_max_offset", 12.0);
            let mut encoder = Encoder::<Av2Backend>::new(config).unwrap();
            let mut decoder = crate::Decoder::default();
            (0..4)
                .map(|frame| {
                    // A flat left half beside the moving pattern.
                    let mut source = moving(192, 128, frame);
                    for plane in PlaneIndex::ALL {
                        let mut view = source.plane_mut(plane).view_mut();
                        let half = view.width() / 2;
                        for y in 0..view.height() {
                            view.row_mut(y)[..half].fill(90 + frame as u8);
                        }
                    }
                    let packet = encoder.encode(&source);
                    let decoded = decoder.decode_ref(&packet.data).unwrap();
                    assert_eq!(bytes(decoded), bytes(encoder.reconstruction()));
                    packet.data
                })
                .collect::<Vec<_>>()
        };
        assert_ne!(packets(0.0), packets(3.0));
    }

    /// A pattern scrolling by whole samples, so inter frames use vectors.
    fn moving(width: usize, height: usize, frame: usize) -> Picture {
        let mut picture = Picture::new(width, height, 0).unwrap();
        for plane in PlaneIndex::ALL {
            let mut view = picture.plane_mut(plane).view_mut();
            let (plane_width, plane_height) = (view.width(), view.height());
            for row in 0..plane_height {
                for (col, sample) in view.row_mut(row).iter_mut().enumerate() {
                    let x = (col + 3 * frame) % plane_width;
                    let y = (row + frame) % plane_height;
                    *sample = ((x * 13 + y * 7 + (x * y) % 5) % 256) as u8;
                }
            }
        }
        picture
    }

    /// Packets and reconstructions of 16 moving 64x48 frames at qindex
    /// 160 (key every 7, a requested key at frame 9) with `params`, under
    /// rate control at `bitrate` (bits/s, 200 ms buffer) if given.
    fn coded_sequence(
        params: &[(&str, f32)],
        bitrate: Option<u64>,
    ) -> (
        Vec<Vec<u8>>,
        Vec<Vec<u8>>,
        Option<nc_encoder::RateControlStats>,
    ) {
        let mut config = config();
        (
            config.width,
            config.height,
            config.quantizer,
            config.intra_period,
        ) = (64, 48, 160, 7);
        for &(name, value) in params {
            config.params.set(name, value);
        }
        let mut encoder = Encoder::<Av2Backend>::new(config).unwrap();
        if let Some(bps) = bitrate {
            let mut rate = RateControlConfig::new(bps);
            rate.buffer_ms = 200;
            encoder.set_rate_control(rate).unwrap();
        }
        let (mut packets, mut recons) = (Vec::new(), Vec::new());
        for frame in 0..16 {
            if frame == 9 {
                encoder.request_keyframe();
            }
            packets.push(encoder.encode(&moving(64, 48, frame)).data);
            recons.push(bytes(encoder.reconstruction()));
        }
        (packets, recons, encoder.rate_control_stats())
    }

    /// The recode contract (`CodecBackend::RECODE`): coding every frame
    /// twice (`debug.recode_identity`) leaves the stream and the
    /// reconstruction unchanged, with keys, a requested key and rate control.
    #[test]
    fn recoding_every_frame_changes_nothing() {
        for bitrate in [None, Some(150_000)] {
            let once = coded_sequence(&[("debug.recode_identity", 0.)], bitrate);
            let twice = coded_sequence(&[("debug.recode_identity", 1.)], bitrate);
            assert_eq!(once.0, twice.0, "rate {bitrate:?}");
            assert_eq!(once.1, twice.1, "rate {bitrate:?}");
        }
    }

    /// `rate.recode_overflow`: frames that would overflow a small buffer
    /// are coded again, coarser, and still decode to the reconstruction.
    #[test]
    fn overflowing_frames_are_recoded_and_decode_exactly() {
        let (packets, recons, stats) =
            coded_sequence(&[("rate.recode_overflow", 1.)], Some(20_000));
        assert!(stats.unwrap().recoded_frames > 0);
        let mut decoder = crate::Decoder::default();
        for (packet, recon) in packets.iter().zip(&recons) {
            assert_eq!(&bytes(decoder.decode_ref(packet).unwrap()), recon);
        }
    }

    /// Switchable frames (searched filters, and random per-block filters in
    /// random-decision frames, which also pick random fixed filters) decode
    /// to the reconstruction, and the search changes the stream.
    #[test]
    fn switchable_filters_decode_to_the_reconstruction() {
        let fixed = coded_sequence(&[], None).0;
        for params in [
            &[("av2.interp_switchable", 1.0)][..],
            &[
                ("av2.interp_switchable", 2.0),
                ("av2.interp_search_max_gain", 1.0),
            ],
            &[("debug.random_decisions", 5.0)],
            &[("av2.interp_filter", 2.0)],
            &[("av2.interp_filter", 3.0)],
        ] {
            let (packets, recons, _) = coded_sequence(params, None);
            assert_ne!(packets, fixed, "{params:?}");
            let mut decoder = crate::Decoder::default();
            for (packet, recon) in packets.iter().zip(&recons) {
                assert_eq!(
                    &bytes(decoder.decode_ref(packet).unwrap()),
                    recon,
                    "{params:?}"
                );
            }
        }
    }

    /// Skip mode (chosen, and random with random skip-mode DRL indices,
    /// with a deeper list and the bank) decodes to the reconstruction and
    /// changes the stream.
    #[test]
    fn skip_mode_decodes_to_the_reconstruction() {
        let plain = coded_sequence(&[], None).0;
        for params in [
            &[("av2.skip_mode", 1.0)][..],
            &[
                ("av2.skip_mode", 1.0),
                ("av2.max_drl_bits", 5.0),
                ("av2.ref_mv_bank", 1.0),
                ("av2.skip_mode_candidates", 6.0),
            ],
            &[("av2.skip_mode", 1.0), ("debug.random_decisions", 3.0)],
        ] {
            let (packets, recons, _) = coded_sequence(params, None);
            assert_ne!(packets, plain, "{params:?}");
            let mut decoder = crate::Decoder::default();
            for (packet, recon) in packets.iter().zip(&recons) {
                assert_eq!(
                    &bytes(decoder.decode_ref(packet).unwrap()),
                    recon,
                    "{params:?}"
                );
            }
        }
    }

    /// Sizes that are not whole 8x8 blocks are coded padded and cropped back
    /// by the conformance window; lossless pictures survive exactly.
    #[test]
    fn any_even_size_decodes_at_the_visible_size() {
        for (width, height) in [(66, 34), (18, 10), (2, 2)] {
            for quantizer in [0, 160] {
                let mut config = config();
                config.width = width;
                config.height = height;
                config.quantizer = quantizer;
                let mut encoder = Encoder::<Av2Backend>::new(config).unwrap();
                let mut decoder = crate::Decoder::default();
                let mut source = Picture::new(width, height, 0).unwrap();
                for frame in 0..3 {
                    for plane in PlaneIndex::ALL {
                        let mut view = source.plane_mut(plane).view_mut();
                        for y in 0..view.height() {
                            for (x, pixel) in view.row_mut(y).iter_mut().enumerate() {
                                *pixel = ((x + 3 * frame) * 13 + y * 7) as u8;
                            }
                        }
                    }
                    let packet = encoder.encode(&source);
                    let decoded = decoder.decode_ref(&packet.data).unwrap();
                    assert_eq!((decoded.width(), decoded.height()), (width, height));
                    assert_eq!(bytes(decoded), bytes(encoder.reconstruction()));
                    if quantizer == 0 {
                        assert_eq!(bytes(decoded), bytes(&source));
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_unknown_params() {
        let mut config = config();
        config.params.set("av2.unknown_tool", 1.0);
        assert!(Encoder::<Av2Backend>::new(config).is_err());
    }
}
