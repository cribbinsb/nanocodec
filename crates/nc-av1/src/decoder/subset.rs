//! The frame-level step in front of the block layer: maps a fully parsed
//! frame header onto the parameters [`super::tile`] decodes with (§5.9.2
//! and the per-segment derivations of §5.9.12–14, §7.12.2). Every Main
//! profile frame tool is decoded, so it rejects nothing; what the decoder
//! does not support (other profiles, large scale tile) is refused where it
//! is parsed, and those names are what `tools/ci/av1_conformance.sh`
//! reports per vector.
use super::frame_header::quant::{
    MAX_SEGMENTS, SEG_LVL_ALT_Q, SEG_LVL_GLOBALMV, SEG_LVL_REF_FRAME, SEG_LVL_SKIP,
    SegmentationFeatures,
};
use super::frame_header::{
    CdefParams, FrameHeader, FrameSize, FrameType, GlobalMotion, LoopFilter, Restoration,
    TOTAL_REFS, TxMode,
};
use crate::obu::FrameKind;

/// `NUM_QM_LEVELS - 1`: the flat (unweighted) matrix level.
const FLAT_QM_LEVEL: u8 = 15;

/// What the block layer decodes a frame with.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // the specification's flags
pub(super) struct FrameParameters {
    pub kind: FrameKind,
    /// Luma size: the frame's, or the coded (8-aligned) size the block
    /// layer works in.
    pub width: usize,
    pub height: usize,
    pub base_q_idx: u32,
    /// (DC, AC) quantiser offsets of Y, U and V (§5.9.12).
    pub quant_deltas: [(i32, i32); 3],
    /// `reduced_tx_set`: the smaller transform-type sets.
    pub reduced_tx_set: bool,
    /// `TX_MODE_SELECT`: blocks code their transform sizes.
    pub tx_mode_select: bool,
    /// `LosslessArray[segment_id]` (§5.9.2): the segment's blocks use 4x4
    /// Walsh-Hadamard transforms only. Without segmentation only entry 0
    /// is read (`CodedLossless`).
    pub lossless: [bool; MAX_SEGMENTS],
    /// Whether luma transform types are coded in a segment (§5.11.47:
    /// `get_qindex(1, segment_id) > 0`, or `base_q_idx > 0` without
    /// segmentation).
    pub tx_types_coded: [bool; MAX_SEGMENTS],
    /// `qm_y`, `qm_u`, `qm_v` when `using_qmatrix`: the matrix level of
    /// each plane's non-lossless blocks (`SegQMLevel`; lossless segments
    /// are flat).
    pub qm: Option<[u8; 3]>,
    /// Segmentation (`None`: off).
    pub segments: Option<Segments>,
    pub loop_filter: LoopFilter,
    /// `delta_lf_res` (a shift) and `delta_lf_multi` when superblocks code
    /// loop filter deltas (`delta_lf_present`, §5.9.18).
    pub delta_lf: Option<DeltaLf>,
    pub cdef: CdefParams,
    /// The frame's size, including its superres upscaling.
    pub size: FrameSize,
    /// `lr_params()`: each plane's restoration type and unit size.
    pub restoration: Restoration,
    pub allow_high_precision_mv: bool,
    pub force_integer_mv: bool,
    /// `gm_params` per reference name.
    pub global_motion: [GlobalMotion; TOTAL_REFS],
    /// `RefFrameSignBias` per reference name.
    pub sign_bias: [bool; TOTAL_REFS],
    /// `reference_select`: blocks may be compound.
    pub reference_select: bool,
    /// `SkipModeFrame[]` when `skip_mode_present`.
    pub skip_mode_frames: Option<[i8; 2]>,
    /// `delta_q_res` (a shift) when superblocks code `delta_q_abs`
    /// (`delta_q_present`).
    pub delta_q_res: Option<u8>,
    /// `allow_screen_content_tools`: blocks may use palettes.
    pub screen_content: bool,
    /// `allow_intrabc`: blocks of this intra frame may copy from it.
    pub allow_intrabc: bool,
}

/// `delta_lf_params()` (§5.9.18) of a frame that codes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DeltaLf {
    /// `delta_lf_res`, a shift.
    pub res: u8,
    /// `delta_lf_multi`: one delta per `loop_filter_level[i]`, not one for
    /// all.
    pub multi: bool,
}

/// The segmentation the block layer decodes (§5.9.14): ids coded with
/// spatial prediction (§5.11.9), kept from or temporally predicted by the
/// primary reference's map (§5.11.19), before the skip flag when a segment
/// uses a reference, skip or global motion feature (`SegIdPreSkip`), and
/// each segment's features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Segments {
    /// `FeatureEnabled[][]` and `FeatureData[][]`.
    pub features: SegmentationFeatures,
    /// `LastActiveSegId`.
    pub last_active: u8,
    /// `SegIdPreSkip`.
    pub pre_skip: bool,
    /// `segmentation_update_map` and `segmentation_temporal_update`.
    pub update_map: bool,
    pub temporal_update: bool,
}

impl Segments {
    /// `FeatureData[segment][feature]` when the feature is enabled
    /// (`seg_feature_active_idx`).
    pub(super) fn feature(&self, segment: u8, feature: usize) -> Option<i16> {
        let segment = usize::from(segment);
        self.features.enabled[segment][feature].then_some(self.features.data[segment][feature])
    }
    /// The quantizer offset of `segment` (`SEG_LVL_ALT_Q`; 0 when off).
    pub(super) fn alt_q(&self, segment: u8) -> i32 {
        self.feature(segment, SEG_LVL_ALT_Q).map_or(0, i32::from)
    }
    /// What `segment`'s reference, skip and global motion features force
    /// on a block's syntax (§5.11.10, §5.11.11, §5.11.20, §5.11.25,
    /// §5.11.23).
    pub(super) fn forced(&self, segment: u8) -> ForcedSyntax {
        ForcedSyntax {
            reference: self
                .feature(segment, SEG_LVL_REF_FRAME)
                .map(|reference| reference as i8),
            skip: self.feature(segment, SEG_LVL_SKIP).is_some(),
            global_motion: self.feature(segment, SEG_LVL_GLOBALMV).is_some(),
        }
    }
}

/// The block syntax a segment's features infer rather than code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct ForcedSyntax {
    /// `SEG_LVL_REF_FRAME`: the block's only reference (`INTRA_FRAME` for
    /// an intra block); `is_inter` follows from it.
    pub reference: Option<i8>,
    /// `SEG_LVL_SKIP`: the block is skipped, predicted from `LAST_FRAME`
    /// with `GLOBALMV` when it is inter.
    pub skip: bool,
    /// `SEG_LVL_GLOBALMV`: the block is inter, `GLOBALMV` from
    /// `LAST_FRAME` unless the reference is forced.
    pub global_motion: bool,
}

impl ForcedSyntax {
    /// Whether any feature is forced: `skip_mode` is then not coded
    /// (§5.11.10).
    pub(super) fn any(self) -> bool {
        self.reference.is_some() || self.skip || self.global_motion
    }
    /// Whether the inter mode is `GLOBALMV` from `LAST_FRAME` (unless the
    /// reference is forced): skip or global motion features (§5.11.23,
    /// §5.11.25).
    pub(super) fn global_mode(self) -> bool {
        self.skip || self.global_motion
    }
}

/// The block layer's frame parameters for `h`.
pub(super) fn frame_parameters(h: &FrameHeader) -> FrameParameters {
    let kind = match h.frame_type {
        // Switch frames are inter frames (§6.8.2) whose header infers
        // error resilience, a frame size override and a full refresh.
        FrameType::Inter | FrameType::Switch => FrameKind::Inter,
        // The block layer decodes intra-only frames as key frames
        // (`FrameIsIntra`); their references and refresh are the frame
        // header's.
        FrameType::Key | FrameType::IntraOnly => FrameKind::Key,
    };
    let q = &h.quant;
    let segment_qindex = |segment: usize| h.segmentation.qindex(q.base_q_idx, segment);
    // `LosslessArray[]` (§5.9.2).
    let lossless: [bool; MAX_SEGMENTS] =
        std::array::from_fn(|segment| segment_qindex(segment) == 0 && q.deltas_zero());
    let segments = h.segmentation.enabled.then_some(Segments {
        features: h.segmentation.features,
        last_active: h.segmentation.last_active,
        pre_skip: h.segmentation.pre_skip,
        update_map: h.segmentation.update_map,
        temporal_update: h.segmentation.temporal_update,
    });
    FrameParameters {
        kind,
        width: h.size.width,
        height: h.size.height,
        base_q_idx: q.base_q_idx,
        quant_deltas: [
            (q.delta_y_dc, 0),
            (q.delta_u_dc, q.delta_u_ac),
            (q.delta_v_dc, q.delta_v_ac),
        ],
        reduced_tx_set: h.reduced_tx_set,
        tx_mode_select: h.tx_mode == TxMode::Select,
        lossless,
        tx_types_coded: std::array::from_fn(|segment| segment_qindex(segment) > 0),
        qm: q
            .qm
            .filter(|levels| levels.iter().any(|&l| l < FLAT_QM_LEVEL)),
        segments,
        loop_filter: h.loop_filter,
        delta_lf: h.delta.lf_present.then_some(DeltaLf {
            res: h.delta.lf_res,
            multi: h.delta.lf_multi,
        }),
        cdef: h.cdef,
        size: h.size,
        restoration: h.restoration,
        allow_high_precision_mv: h.allow_high_precision_mv,
        force_integer_mv: h.force_integer_mv,
        global_motion: h.global_motion,
        sign_bias: h.sign_bias,
        reference_select: h.reference_select,
        skip_mode_frames: h
            .skip_mode_frames
            .map(|frames| frames.map(|name| name as i8)),
        delta_q_res: h.delta.q_present.then_some(h.delta.q_res),
        screen_content: h.allow_screen_content_tools,
        allow_intrabc: h.allow_intrabc,
    }
}
