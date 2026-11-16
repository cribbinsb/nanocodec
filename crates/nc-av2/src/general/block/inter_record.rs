//! What an inter block's syntax says (dav2d `Av2Block`'s inter part, read
//! by `decode_b`'s inter branch): its references, mode, the vector
//! residuals and DRL indices, the motion mode with its warp syntax, and
//! the compound, inter-intra, block-adaptive weight and filter choices.
//! The motion group resolves it into final vectors and warp models; the
//! prediction reads both.
use super::motion::Mv;

/// A reference rank of an inter block (0..7 for the frame's references),
/// or the TIP frame (dav2d `TIP_FRAME`).
pub type Reference = i8;
/// The TIP frame as a reference (dav2d `TIP_FRAME`).
pub const TIP_FRAME: Reference = 7;
/// No second reference (dav2d `-1`).
pub const NO_REFERENCE: Reference = -1;

/// An inter prediction mode, with dav2d's numbering (single-reference
/// modes from 13, compound from 18), which the syntax compares by order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum InterMode {
    /// A listed vector.
    Near = 13,
    /// The global motion vector.
    Global,
    /// A listed vector plus a residual.
    New,
    /// A listed warp model.
    Warp,
    /// A warp with a vector residual.
    WarpNew,
    /// Compound: listed vectors for both.
    NearNear,
    /// Compound: listed, then with a residual.
    NearNew,
    /// Compound: with a residual, then listed.
    NewNear,
    /// Compound: both global.
    GlobalGlobal,
    /// Compound: residuals for both.
    NewNew,
    /// Compound: one residual, projected to the other reference.
    JointNew,
    /// The optical-flow refined compound modes, in the order above.
    OpflNearNear,
    /// See `OpflNearNear`.
    OpflNearNew,
    /// See `OpflNearNear`.
    OpflNewNear,
    /// See `OpflNearNear`.
    OpflNewNew,
    /// See `OpflNearNear`.
    OpflJointNew,
}

impl InterMode {
    /// The mode `offset` places after `self` (dav2d's arithmetic on mode
    /// numbers).
    #[must_use]
    pub fn offset(self, offset: i32) -> Option<Self> {
        Self::from_u8(u8::try_from(self as i32 + offset).ok()?)
    }

    /// The mode numbered `value`.
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        const MODES: [InterMode; 16] = [
            InterMode::Near,
            InterMode::Global,
            InterMode::New,
            InterMode::Warp,
            InterMode::WarpNew,
            InterMode::NearNear,
            InterMode::NearNew,
            InterMode::NewNear,
            InterMode::GlobalGlobal,
            InterMode::NewNew,
            InterMode::JointNew,
            InterMode::OpflNearNear,
            InterMode::OpflNearNew,
            InterMode::OpflNewNear,
            InterMode::OpflNewNew,
            InterMode::OpflJointNew,
        ];
        MODES
            .get(usize::from(value.checked_sub(Self::Near as u8)?))
            .copied()
    }

    /// A compound mode.
    #[must_use]
    pub fn is_compound(self) -> bool {
        self >= Self::NearNear
    }

    /// The single-reference mode each reference of a compound mode uses
    /// (dav2d `dav2d_comp_inter_pred_modes`); a single mode is its own.
    #[must_use]
    pub fn per_reference(self) -> [Self; 2] {
        use InterMode::{Global, Near, New};
        match self {
            Self::NearNear | Self::OpflNearNear => [Near, Near],
            Self::NearNew | Self::OpflNearNew => [Near, New],
            Self::NewNear | Self::OpflNewNear => [New, Near],
            Self::GlobalGlobal => [Global, Global],
            Self::NewNew | Self::JointNew | Self::OpflNewNew | Self::OpflJointNew => [New, New],
            single => [single, single],
        }
    }
}

/// How the block's motion is modelled (dav2d `MotionMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum MotionMode {
    /// One vector.
    #[default]
    Translation,
    /// Blended with an intra prediction.
    InterIntra,
    /// A warp fitted to the neighbours' motion.
    WarpCausal,
    /// A listed warp model plus coded deltas.
    WarpDelta,
    /// A neighbour's warp model extended to the block.
    WarpExtend,
}

/// How a compound block blends its two predictions (dav2d
/// `CompInterType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CompoundType {
    /// A single reference.
    #[default]
    None,
    /// The (possibly weighted) average.
    Average,
    /// A wedge mask.
    Wedge,
    /// A mask from the predictions' difference.
    Difference,
}

/// A warp delta's parameter not coded (dav2d `-0x80`): two-parameter
/// deltas leave the last two out.
pub const WARP_DELTA_ABSENT: i8 = -0x80;
/// The compound weight of an unweighted average (dav2d `cwp_idx` 8, of
/// 16).
pub const EQUAL_WEIGHT: i8 = 8;

/// An inter block's syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools, reason = "the block's flags")]
pub struct InterRecord {
    /// The references (the second [`NO_REFERENCE`] for a single one).
    pub references: [Reference; 2],
    /// The mode.
    pub mode: InterMode,
    /// The DRL index per reference.
    pub drl: [u8; 2],
    /// The vector residuals as coded (eighth samples), before the
    /// predictors are added; a joint mode's second is already projected.
    pub residual: [Mv; 2],
    /// The residual precision per reference (dav2d `mv_prec`, 0..6 for 8,
    /// 4, 2, 1, 1/2, 1/4, 1/8 samples).
    pub precision: [u8; 2],
    /// Adaptive MVD.
    pub amvd: bool,
    /// The motion model.
    pub motion_mode: MotionMode,
    /// Which candidate warp model a warp block starts from.
    pub warp_reference: u8,
    /// A WARPMV block codes a vector residual.
    pub warp_with_residual: bool,
    /// The coded warp delta parameters (dav2d `matrix`), scaled by their
    /// step; [`WARP_DELTA_ABSENT`] where not coded.
    pub warp_delta: [i8; 4],
    /// Inter-intra with a warp (`warp_ii`).
    pub warp_inter_intra: bool,
    /// The inter-intra mode (dav2d `InterIntraPredMode`).
    pub inter_intra_mode: u8,
    /// The wedge index, -1 for none (inter-intra and compound).
    pub wedge_index: i8,
    /// Which prediction the wedge mask favours.
    pub wedge_sign: bool,
    /// Which prediction the difference mask favours.
    pub mask_sign: bool,
    /// The compound blend.
    pub compound: CompoundType,
    /// The compound weight of the first prediction in sixteenths
    /// ([`EQUAL_WEIGHT`] for none); TIP blocks' global weight.
    pub weight: i8,
    /// MV refinement: 0 off, 1 coded on, 2 implied on.
    pub refine_mv: u8,
    /// Block-adaptive weighted prediction: luma (low two bits the kind,
    /// bits 2-3 its context) and chroma.
    pub bawp: [u8; 2],
    /// The interpolation filter (dav2d `Dav2dFilterMode`).
    pub filter: u8,
}

impl Default for InterRecord {
    /// A single-reference NEAR block of reference 0 with nothing else
    /// coded.
    fn default() -> Self {
        Self {
            references: [0, NO_REFERENCE],
            mode: InterMode::Near,
            drl: [0; 2],
            residual: [Mv::default(); 2],
            precision: [0; 2],
            amvd: false,
            motion_mode: MotionMode::Translation,
            warp_reference: 0,
            warp_with_residual: false,
            warp_delta: [0; 4],
            warp_inter_intra: false,
            inter_intra_mode: 0,
            wedge_index: -1,
            wedge_sign: false,
            mask_sign: false,
            compound: CompoundType::None,
            weight: EQUAL_WEIGHT,
            refine_mv: 0,
            bawp: [0; 2],
            filter: 0,
        }
    }
}
