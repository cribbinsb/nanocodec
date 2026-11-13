//! AV2 intra mode syntax: every luma mode (five non-directional modes and
//! eight directions with seven angle deltas each) and every chroma mode
//! except chroma-from-luma.
//!
//! Follows pinned AVM v1.0.0 (BSD-3-Clause-Clear): `get_y_intra_mode_set`,
//! `get_uv_intra_mode_set`, `av2_set_y_mode_and_delta_angle` and
//! `get_y_mode_idx_ctx` (`av2/common/reconintra.c`), and
//! `read_intra_luma_mode` / `read_intra_uv_mode` (`av2/decoder/decodemv.c`).
//!
//! A luma mode is coded as an index into a 61-entry list that depends on
//! the modes of the blocks above the rightmost and left of the bottommost
//! sample: the non-directional modes first (their index is their mode),
//! then the distinct directional neighbour modes, for blocks above 64
//! samples their ±1..±4 angle-delta neighbours, then a default order (the
//! eight nominal directions, then deltas ±2, ±1, ±3). Indices 0..6 are one
//! symbol, 7..12 add an offset symbol, 13..60 are a mode-set symbol plus a
//! four-bit literal. Chroma codes an index into a 13-entry list led by the
//! luma direction (when luma is directional), which then also inherits the
//! luma angle delta. DPCM (lossless), chroma-from-luma, MRL and the other optional intra
//! tools are disabled by the sequence header.
//!
//! Prediction itself is [`crate::intra_edges`].
use crate::entropy::{Reader, Writer};
use nc_common::decode::{DecodeError, Result};
use nc_encoder::symbol_cost::symbol_bits;

/// The eight nominal prediction directions, in AVM's reordered luma order
/// (`reordered_y_mode`), which is also their joint-mode order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// 45 degrees (towards the above-right).
    D45,
    /// 67 degrees.
    D67,
    /// 90 degrees: vertical.
    Vertical,
    /// 113 degrees.
    D113,
    /// 135 degrees.
    D135,
    /// 157 degrees.
    D157,
    /// 180 degrees: horizontal.
    Horizontal,
    /// 203 degrees (towards the below-left).
    D203,
}

impl Direction {
    /// Every direction, in joint-mode order.
    pub const ALL: [Self; 8] = [
        Self::D45,
        Self::D67,
        Self::Vertical,
        Self::D113,
        Self::D135,
        Self::D157,
        Self::Horizontal,
        Self::D203,
    ];

    /// Nominal angle in degrees (AVM `mode_to_angle_map`).
    #[must_use]
    pub const fn angle(self) -> u16 {
        match self {
            Self::D45 => 45,
            Self::D67 => 67,
            Self::Vertical => 90,
            Self::D113 => 113,
            Self::D135 => 135,
            Self::D157 => 157,
            Self::Horizontal => 180,
            Self::D203 => 203,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// Largest angle delta magnitude (`MAX_ANGLE_DELTA`).
pub const MAX_ANGLE_DELTA: i8 = 3;
/// Degrees per angle-delta step (`ANGLE_STEP`).
pub const ANGLE_STEP: i16 = 3;
/// Angle deltas per direction (`TOTAL_ANGLE_DELTA_COUNT`).
const ANGLE_DELTAS: usize = 7;
/// Non-directional luma modes (`NON_DIRECTIONAL_MODES_COUNT`).
const NON_DIRECTIONAL: usize = 5;
/// Joint luma modes: non-directional plus every direction and delta
/// (`LUMA_MODE_COUNT`).
pub const LUMA_MODES: usize = NON_DIRECTIONAL + Direction::ALL.len() * ANGLE_DELTAS;
/// Joint directional modes, the cycle the derived-mode offsets wrap around.
const DIRECTIONAL_JOINT_MODES: usize = LUMA_MODES - NON_DIRECTIONAL;
/// Largest offset of the derived angular modes inserted for large blocks.
const DERIVED_MODE_OFFSETS: usize = 4;
/// Blocks larger than this many samples insert derived angular modes into
/// the luma list (AVM `is_large_block`).
const DERIVED_MODES_MIN_AREA: usize = 64;
/// Luma list indices coded by the first `y_mode_idx` symbol alone
/// (`LUMA_INTRA_MODE_INDEX_COUNT - 1`); the last symbol value escapes.
const LUMA_FIRST_SYMBOLS: usize = 7;
/// Indices reachable with mode set zero (`FIRST_MODE_COUNT`).
const FIRST_MODE_COUNT: usize = 13;
/// Indices per later mode set (`SECOND_MODE_COUNT`), a four-bit literal.
const SECOND_MODE_COUNT: usize = 16;
const SECOND_MODE_BITS: u32 = 4;
/// Chroma modes in the list (`UV_INTRA_MODES - 1`: chroma-from-luma excluded).
const CHROMA_MODES: usize = 13;
/// Chroma list indices coded by the symbol alone; the last value escapes to
/// a three-bit literal (`CHROMA_INTRA_MODE_INDEX_COUNT - 1`).
const CHROMA_FIRST_SYMBOLS: usize = 7;
const CHROMA_ESCAPE_BITS: u32 = 3;

/// AVM `default_mode_list_y`: joint modes minus five, in default list order
/// (nominal directions, then deltas ±2, ±1, ±3).
const DEFAULT_LUMA_ORDER: [u8; DIRECTIONAL_JOINT_MODES] = [
    17, 45, 3, 10, 24, 31, 38, 52, 15, 19, 43, 47, 1, 5, 8, 12, 22, 26, 29, 33, 36, 40, 50, 54, 16,
    18, 44, 46, 2, 4, 9, 11, 23, 25, 30, 32, 37, 39, 51, 53, 14, 20, 42, 48, 0, 6, 7, 13, 21, 27,
    28, 34, 35, 41, 49, 55,
];
/// AVM `default_mode_list_uv`: the chroma list's directional tail.
const DEFAULT_CHROMA_DIRECTIONS: [Direction; 8] = [
    Direction::Vertical,
    Direction::Horizontal,
    Direction::D45,
    Direction::D135,
    Direction::D67,
    Direction::D113,
    Direction::D157,
    Direction::D203,
];

/// One intra prediction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Mode {
    /// Mean of available reconstructed edges.
    #[default]
    Dc,
    /// AV2 smooth: both edge ramps blended.
    Smooth,
    /// AV2 smooth along columns.
    SmoothVertical,
    /// AV2 smooth along rows.
    SmoothHorizontal,
    /// Nearest edge to the planar extrapolation.
    Paeth,
    /// Directional prediction at `direction`'s angle plus
    /// `delta` × [`ANGLE_STEP`] degrees.
    Directional {
        /// Nominal direction.
        direction: Direction,
        /// Angle delta in −3..=3.
        delta: i8,
    },
}

impl Mode {
    /// Zero-delta vertical prediction.
    pub const VERTICAL: Self = Self::nominal(Direction::Vertical);
    /// Zero-delta horizontal prediction.
    pub const HORIZONTAL: Self = Self::nominal(Direction::Horizontal);
    /// The four modes of the standalone picture coders.
    pub const BASIC: [Self; 4] = [Self::Dc, Self::VERTICAL, Self::HORIZONTAL, Self::Paeth];
    /// Non-directional modes, in joint-mode order.
    pub const NON_DIRECTIONAL: [Self; NON_DIRECTIONAL] = [
        Self::Dc,
        Self::Smooth,
        Self::SmoothVertical,
        Self::SmoothHorizontal,
        Self::Paeth,
    ];

    /// The zero-delta mode of `direction`.
    #[must_use]
    pub const fn nominal(direction: Direction) -> Self {
        Self::Directional {
            direction,
            delta: 0,
        }
    }

    /// Whether this mode is directional (enters the luma context count and
    /// leads the chroma list).
    #[must_use]
    pub const fn directional(self) -> bool {
        matches!(self, Self::Directional { .. })
    }

    /// The direction, for directional modes.
    #[must_use]
    pub const fn direction(self) -> Option<Direction> {
        match self {
            Self::Directional { direction, .. } => Some(direction),
            _ => None,
        }
    }

    /// Prediction angle in degrees, for directional modes.
    #[must_use]
    pub fn angle(self) -> Option<u16> {
        match self {
            Self::Directional { direction, delta } => {
                Some((direction.angle() as i16 + i16::from(delta) * ANGLE_STEP) as u16)
            }
            _ => None,
        }
    }

    /// AVM `joint_y_mode_delta_angle`.
    #[must_use]
    pub fn joint(self) -> usize {
        match self {
            Self::Dc => 0,
            Self::Smooth => 1,
            Self::SmoothVertical => 2,
            Self::SmoothHorizontal => 3,
            Self::Paeth => 4,
            Self::Directional { direction, delta } => {
                debug_assert!(delta.abs() <= MAX_ANGLE_DELTA);
                NON_DIRECTIONAL
                    + direction.index() * ANGLE_DELTAS
                    + (delta + MAX_ANGLE_DELTA) as usize
            }
        }
    }

    /// Inverse of [`Mode::joint`] (AVM `av2_set_y_mode_and_delta_angle`).
    /// # Panics
    /// If `joint` is not below [`LUMA_MODES`].
    #[must_use]
    pub fn from_joint(joint: usize) -> Self {
        if joint < NON_DIRECTIONAL {
            return Self::NON_DIRECTIONAL[joint];
        }
        let directional = joint - NON_DIRECTIONAL;
        Self::Directional {
            direction: Direction::ALL[directional / ANGLE_DELTAS],
            delta: (directional % ANGLE_DELTAS) as i8 - MAX_ANGLE_DELTA,
        }
    }

    /// Every luma mode, in joint order.
    pub fn all_luma() -> impl Iterator<Item = Self> {
        (0..LUMA_MODES).map(Self::from_joint)
    }
}

impl From<nc_dsp::intra_prediction::IntraMode> for Mode {
    /// The engine's shared intra modes; vertical and horizontal are the
    /// zero-delta directional modes.
    /// The oblique AV1 directions map to AV2's zero-delta ones likewise.
    fn from(mode: nc_dsp::intra_prediction::IntraMode) -> Self {
        use nc_dsp::intra_prediction::IntraMode as Shared;
        match mode {
            Shared::Dc => Self::Dc,
            Shared::Vertical => Self::VERTICAL,
            Shared::Horizontal => Self::HORIZONTAL,
            Shared::Smooth => Self::Smooth,
            Shared::SmoothVertical => Self::SmoothVertical,
            Shared::SmoothHorizontal => Self::SmoothHorizontal,
            Shared::Paeth => Self::Paeth,
            Shared::D45 => Self::nominal(Direction::D45),
            Shared::D135 => Self::nominal(Direction::D135),
            Shared::D113 => Self::nominal(Direction::D113),
            Shared::D157 => Self::nominal(Direction::D157),
            Shared::D203 => Self::nominal(Direction::D203),
            Shared::D67 => Self::nominal(Direction::D67),
        }
    }
}

/// Neighbour modes and size of the block whose modes are coded.
#[derive(Debug, Clone, Copy)]
pub struct ModeContext {
    /// Luma mode of the block above the rightmost sample (inter: DC).
    pub above: Option<Mode>,
    /// Luma mode of the block left of the bottommost sample (inter: DC).
    pub left: Option<Mode>,
    /// Luma block side, log2.
    pub block_log2: u32,
}

impl ModeContext {
    /// `y_mode_idx` context: directional neighbours counted (AVM `get_y_mode_idx_ctx`).
    fn y_context(self) -> usize {
        usize::from(self.above.is_some_and(Mode::directional))
            + usize::from(self.left.is_some_and(Mode::directional))
    }
}

/// The luma mode list of one block (AVM `y_intra_mode_list`): joint modes
/// by index, and the inverse.
#[derive(Debug, Clone)]
pub struct LumaList {
    modes: [u8; LUMA_MODES],
    indices: [u8; LUMA_MODES],
}

impl LumaList {
    /// Build the list for `context` (AVM `get_y_intra_mode_set`).
    #[must_use]
    pub fn new(context: ModeContext) -> Self {
        let mut list = Self {
            modes: [0; LUMA_MODES],
            indices: [u8::MAX; LUMA_MODES],
        };
        let mut length = 0;
        let mut push = |list: &mut Self, joint: usize| {
            if list.indices[joint] == u8::MAX {
                list.modes[length] = joint as u8;
                list.indices[joint] = length as u8;
                length += 1;
            }
        };
        for joint in 0..NON_DIRECTIONAL {
            push(&mut list, joint);
        }
        let directional = |mode: Option<Mode>| mode.filter(|m| m.directional()).map(Mode::joint);
        // Left first; a lone directional above neighbour takes its place.
        let neighbours: Vec<usize> = match (directional(context.left), directional(context.above)) {
            (Some(left), Some(above)) if left != above => vec![left, above],
            (Some(mode), _) | (None, Some(mode)) => vec![mode],
            (None, None) => Vec::new(),
        };
        for &joint in &neighbours {
            push(&mut list, joint);
        }
        if (1_usize << (2 * context.block_log2)) > DERIVED_MODES_MIN_AREA {
            for offset in 1..=DERIVED_MODE_OFFSETS {
                for &joint in &neighbours {
                    let cyclic = joint - NON_DIRECTIONAL;
                    let lower =
                        (cyclic + DIRECTIONAL_JOINT_MODES - offset) % DIRECTIONAL_JOINT_MODES;
                    let upper = (cyclic + offset) % DIRECTIONAL_JOINT_MODES;
                    push(&mut list, lower + NON_DIRECTIONAL);
                    push(&mut list, upper + NON_DIRECTIONAL);
                }
            }
        }
        for &joint in &DEFAULT_LUMA_ORDER {
            push(&mut list, usize::from(joint) + NON_DIRECTIONAL);
        }
        debug_assert_eq!(length, LUMA_MODES);
        list
    }

    /// The list index of `mode`.
    #[must_use]
    pub fn index(&self, mode: Mode) -> usize {
        usize::from(self.indices[mode.joint()])
    }

    /// The mode at list `index`.
    #[must_use]
    pub fn mode(&self, index: usize) -> Mode {
        Mode::from_joint(usize::from(self.modes[index]))
    }
}

/// The chroma mode list for luma mode `luma` (AVM `get_uv_intra_mode_set`):
/// a directional chroma mode carries the luma angle delta when it shares
/// the luma direction, and zero otherwise.
fn chroma_list(luma: Mode) -> [Mode; CHROMA_MODES] {
    let mut list = [Mode::Dc; CHROMA_MODES];
    let mut length = 0;
    let luma_direction = luma.direction();
    if luma_direction.is_some() {
        list[length] = luma;
        length += 1;
    }
    for mode in Mode::NON_DIRECTIONAL {
        list[length] = mode;
        length += 1;
    }
    for direction in DEFAULT_CHROMA_DIRECTIONS {
        if Some(direction) != luma_direction {
            list[length] = Mode::nominal(direction);
            length += 1;
        }
    }
    debug_assert_eq!(length, CHROMA_MODES);
    list
}

/// The chroma modes codable with luma mode `luma`, in list order.
#[must_use]
pub fn chroma_modes(luma: Mode) -> [Mode; CHROMA_MODES] {
    chroma_list(luma)
}

fn chroma_index(chroma: Mode, luma: Mode) -> Option<usize> {
    chroma_list(luma).iter().position(|&mode| mode == chroma)
}

const MODE_SET: [u16; 4] = [3905, 1746, 1044, 0];
const Y_MODE: [[u16; 8]; 3] = [
    [17_593, 12_693, 11_040, 8670, 6363, 5113, 3908, 0],
    [22_654, 17_811, 15_953, 13_641, 12_621, 7185, 5599, 0],
    [27_132, 23_764, 22_312, 20_646, 20_024, 12_443, 7161, 0],
];
const Y_MODE_OFFSET: [[u16; 6]; 3] = [
    [20_025, 14_596, 12_574, 9120, 6349, 0],
    [23_792, 16_684, 11_941, 8173, 4272, 0],
    [23_984, 18_212, 13_058, 7865, 4044, 0],
];
const UV_MODE: [[u16; 8]; 2] = [
    [23_405, 11_811, 9903, 8015, 6357, 4785, 2340, 0],
    [11_486, 9158, 4560, 3457, 2420, 1610, 1277, 0],
];
const DPCM: [u16; 2] = [16_384, 0];

/// How one luma list index is coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LumaIndexCode {
    /// Mode set zero, then `y_mode_idx`, then `y_mode_idx_offset` when the
    /// first symbol escapes.
    First {
        symbol: usize,
        offset: Option<usize>,
    },
    /// Mode set `set` (1..=3), then a four-bit literal.
    Later { set: usize, literal: u32 },
}

impl LumaIndexCode {
    fn new(index: usize) -> Self {
        if index < LUMA_FIRST_SYMBOLS {
            Self::First {
                symbol: index,
                offset: None,
            }
        } else if index < FIRST_MODE_COUNT {
            Self::First {
                symbol: LUMA_FIRST_SYMBOLS,
                offset: Some(index - LUMA_FIRST_SYMBOLS),
            }
        } else {
            let later = index - FIRST_MODE_COUNT;
            Self::Later {
                set: 1 + later / SECOND_MODE_COUNT,
                literal: (later % SECOND_MODE_COUNT) as u32,
            }
        }
    }
}

/// Approximate bits of coding `luma` and `chroma` under the default CDFs,
/// for mode ranking.
#[must_use]
pub fn mode_bits(list: &LumaList, context: ModeContext, luma: Mode, chroma: Mode) -> f32 {
    luma_bits(list, context, luma) + chroma_bits(luma, chroma)
}

/// Approximate bits of the luma mode alone (see [`mode_bits`]).
#[must_use]
pub fn luma_bits(list: &LumaList, context: ModeContext, luma: Mode) -> f32 {
    let y_context = context.y_context();
    match LumaIndexCode::new(list.index(luma)) {
        LumaIndexCode::First { symbol, offset } => {
            symbol_bits(&MODE_SET, 0)
                + symbol_bits(&Y_MODE[y_context], symbol)
                + offset.map_or(0.0, |o| symbol_bits(&Y_MODE_OFFSET[y_context], o))
        }
        LumaIndexCode::Later { set, .. } => symbol_bits(&MODE_SET, set) + SECOND_MODE_BITS as f32,
    }
}

/// Approximate bits of the chroma mode alone (see [`mode_bits`]).
/// # Panics
/// If `chroma` is not codable with `luma` ([`chroma_modes`]).
#[must_use]
pub fn chroma_bits(luma: Mode, chroma: Mode) -> f32 {
    let index = chroma_index(chroma, luma).expect("chroma mode from the luma's list");
    let cdf = &UV_MODE[usize::from(luma.directional())];
    if index < CHROMA_FIRST_SYMBOLS {
        symbol_bits(cdf, index)
    } else {
        symbol_bits(cdf, CHROMA_FIRST_SYMBOLS) + CHROMA_ESCAPE_BITS as f32
    }
}

/// Approximate bits of DC luma and chroma modes, for the engine's
/// intra/inter ranking.
#[must_use]
pub fn dc_bits(above: Option<Mode>, left: Option<Mode>) -> f32 {
    let context = ModeContext {
        above,
        left,
        block_log2: 0,
    };
    symbol_bits(&MODE_SET, 0)
        + symbol_bits(&Y_MODE[context.y_context()], 0)
        + symbol_bits(&UV_MODE[0], 0)
}

/// Write the luma and chroma modes of a block (DPCM, chroma-from-luma and MRL disabled).
/// # Errors
/// Rejects a chroma mode outside the luma's list and propagates entropy
/// coding errors.
pub fn write_block(
    writer: &mut Writer,
    lossless: bool,
    context: ModeContext,
    luma: Mode,
    chroma: Mode,
) -> Result<()> {
    let chroma_index = chroma_index(chroma, luma).ok_or(DecodeError::Unsupported(
        "AV2 chroma mode outside the luma's list",
    ))?;
    if lossless {
        writer.model(0, &DPCM, |m| &mut m.dpcm_y)?;
    }
    let y_context = context.y_context();
    match LumaIndexCode::new(LumaList::new(context).index(luma)) {
        LumaIndexCode::First { symbol, offset } => {
            writer.model(0, &MODE_SET, |m| &mut m.mode_set)?;
            writer.model(symbol, &Y_MODE[y_context], |m| &mut m.y_mode[y_context])?;
            if let Some(offset) = offset {
                writer.model(offset, &Y_MODE_OFFSET[y_context], |m| {
                    &mut m.y_mode_offset[y_context]
                })?;
            }
        }
        LumaIndexCode::Later { set, literal } => {
            writer.model(set, &MODE_SET, |m| &mut m.mode_set)?;
            writer.literal(literal, SECOND_MODE_BITS)?;
        }
    }
    if lossless {
        writer.model(0, &DPCM, |m| &mut m.dpcm_uv)?;
    }
    let uv_context = usize::from(luma.directional());
    let symbol = chroma_index.min(CHROMA_FIRST_SYMBOLS);
    writer.model(symbol, &UV_MODE[uv_context], |m| &mut m.uv_mode[uv_context])?;
    if symbol == CHROMA_FIRST_SYMBOLS {
        writer.literal(
            (chroma_index - CHROMA_FIRST_SYMBOLS) as u32,
            CHROMA_ESCAPE_BITS,
        )?;
    }
    Ok(())
}

/// Read the luma and chroma modes of a block.
/// # Errors
/// Rejects DPCM, an out-of-range chroma index and truncated syntax.
pub fn read_block(
    reader: &mut Reader<'_>,
    lossless: bool,
    context: ModeContext,
) -> Result<(Mode, Mode)> {
    if lossless && reader.model(&DPCM, |m| &mut m.dpcm_y)? != 0 {
        return Err(DecodeError::Unsupported("AV2 intra DPCM"));
    }
    let y_context = context.y_context();
    let set = reader.model(&MODE_SET, |m| &mut m.mode_set)?;
    let index = if set == 0 {
        let symbol = reader.model(&Y_MODE[y_context], |m| &mut m.y_mode[y_context])?;
        if symbol == LUMA_FIRST_SYMBOLS {
            symbol
                + reader.model(&Y_MODE_OFFSET[y_context], |m| {
                    &mut m.y_mode_offset[y_context]
                })?
        } else {
            symbol
        }
    } else {
        FIRST_MODE_COUNT
            + (set - 1) * SECOND_MODE_COUNT
            + reader.literal(SECOND_MODE_BITS)? as usize
    };
    let luma = LumaList::new(context).mode(index);
    if lossless && reader.model(&DPCM, |m| &mut m.dpcm_uv)? != 0 {
        return Err(DecodeError::Unsupported("AV2 chroma DPCM"));
    }
    let uv_context = usize::from(luma.directional());
    let mut chroma_index = reader.model(&UV_MODE[uv_context], |m| &mut m.uv_mode[uv_context])?;
    if chroma_index == CHROMA_FIRST_SYMBOLS {
        chroma_index += reader.literal(CHROMA_ESCAPE_BITS)? as usize;
    }
    let chroma = *chroma_list(luma)
        .get(chroma_index)
        .ok_or(DecodeError::Malformed("AV2 chroma intra mode index"))?;
    Ok((luma, chroma))
}

/// Side (log2) of the uniform blocks of the standalone picture coders.
const SMALL_BLOCK_LOG2: u32 = 3;

/// [`write_block`] for the standalone picture coders' 8×8 blocks, whose
/// modes are [`Mode::BASIC`].
/// # Errors
/// As [`write_block`].
pub fn write(
    writer: &mut Writer,
    lossless: bool,
    above: Option<Mode>,
    left: Option<Mode>,
    luma: Mode,
    chroma: Mode,
) -> Result<()> {
    let context = ModeContext {
        above,
        left,
        block_log2: SMALL_BLOCK_LOG2,
    };
    write_block(writer, lossless, context, luma, chroma)
}

/// [`read_block`] for the standalone picture coders' 8×8 blocks.
/// # Errors
/// As [`read_block`], and rejects modes outside [`Mode::BASIC`].
pub fn read(
    reader: &mut Reader<'_>,
    lossless: bool,
    above: Option<Mode>,
    left: Option<Mode>,
) -> Result<(Mode, Mode)> {
    let context = ModeContext {
        above,
        left,
        block_log2: SMALL_BLOCK_LOG2,
    };
    let (luma, chroma) = read_block(reader, lossless, context)?;
    if !Mode::BASIC.contains(&luma) || !Mode::BASIC.contains(&chroma) {
        return Err(DecodeError::Unsupported(
            "AV2 intra mode outside the standalone DC/V/H/Paeth subset",
        ));
    }
    Ok((luma, chroma))
}

/// Predict one 4×4 transform of a standalone picture from already
/// reconstructed samples (frame-edge availability only).
/// Call after reconstructing each preceding transform, including inside an 8×8 block.
/// # Panics
/// Panics if the complete 4×4 block lies outside the plane or `mode` is not
/// in [`Mode::BASIC`].
#[must_use]
pub fn predict4(plane: &nc_common::Plane, x: usize, y: usize, mode: Mode) -> [i32; 16] {
    predict_basic::<16>(plane, x, y, mode, 4)
}

/// Predict an unsplit 8×8 transform of a standalone picture (see [`predict4`]).
/// # Panics
/// As [`predict4`].
#[must_use]
pub fn predict8(plane: &nc_common::Plane, x: usize, y: usize, mode: Mode) -> [i32; 64] {
    predict_basic::<64>(plane, x, y, mode, 8)
}

fn predict_basic<const N: usize>(
    plane: &nc_common::Plane,
    x: usize,
    y: usize,
    mode: Mode,
    size: usize,
) -> [i32; N] {
    use nc_dsp::intra_prediction::{IntraEdges, IntraMode};
    debug_assert_eq!(size * size, N);
    let kernel = match mode {
        Mode::Dc => IntraMode::Dc,
        Mode::Paeth => IntraMode::Paeth,
        Mode::VERTICAL => IntraMode::Vertical,
        Mode::HORIZONTAL => IntraMode::Horizontal,
        _ => panic!("standalone coders predict DC/V/H/Paeth only"),
    };
    let mut samples = [0; N];
    let mut view =
        nc_common::PlaneViewMut::new(&mut samples, size, size, size).expect("square output block");
    IntraEdges::new(plane, x, y, size).predict(kernel, &mut view);
    samples.map(i32::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf_models::TileCdfs;

    #[test]
    fn fixed_cdfs_equal_the_generated_defaults() {
        let models = TileCdfs::new(0);
        assert_eq!(models.mode_set.inverse(), &MODE_SET);
        assert_eq!(models.dpcm_y.inverse(), &DPCM);
        assert_eq!(models.dpcm_uv.inverse(), &DPCM);
        for (model, fixed) in models.y_mode.iter().zip(&Y_MODE) {
            assert_eq!(model.inverse(), fixed);
        }
        for (model, fixed) in models.y_mode_offset.iter().zip(&Y_MODE_OFFSET) {
            assert_eq!(model.inverse(), fixed);
        }
        for (model, fixed) in models.uv_mode.iter().zip(&UV_MODE) {
            assert_eq!(model.inverse(), fixed);
        }
    }

    #[test]
    fn joint_modes_round_trip_and_match_avm_values() {
        for joint in 0..LUMA_MODES {
            assert_eq!(Mode::from_joint(joint).joint(), joint);
        }
        // AVM's DPCM modes name these joint values for V_PRED and H_PRED.
        assert_eq!(Mode::VERTICAL.joint(), 22);
        assert_eq!(Mode::HORIZONTAL.joint(), 50);
    }

    #[test]
    fn luma_lists_are_permutations_led_by_neighbours() {
        let neighbours = [
            None,
            Some(Mode::Dc),
            Some(Mode::VERTICAL),
            Some(Mode::Directional {
                direction: Direction::D45,
                delta: -3,
            }),
            Some(Mode::Directional {
                direction: Direction::D203,
                delta: 3,
            }),
        ];
        for above in neighbours {
            for left in neighbours {
                for block_log2 in [3, 4, 6] {
                    let context = ModeContext {
                        above,
                        left,
                        block_log2,
                    };
                    let list = LumaList::new(context);
                    let mut seen = [false; LUMA_MODES];
                    for index in 0..LUMA_MODES {
                        let mode = list.mode(index);
                        assert_eq!(list.index(mode), index);
                        assert!(!std::mem::replace(&mut seen[mode.joint()], true));
                    }
                    for mode in Mode::NON_DIRECTIONAL {
                        assert_eq!(list.index(mode), mode.joint());
                    }
                    let lead = left
                        .filter(|m| m.directional())
                        .or(above.filter(|m| m.directional()));
                    let expected = lead.unwrap_or(Mode::VERTICAL);
                    assert_eq!(list.mode(NON_DIRECTIONAL), expected);
                }
            }
        }
    }

    #[test]
    fn large_blocks_insert_cyclic_derived_modes() {
        let lowest = Mode::Directional {
            direction: Direction::D45,
            delta: -3,
        };
        let list = LumaList::new(ModeContext {
            above: None,
            left: Some(lowest),
            block_log2: 4,
        });
        // Joint 5 wraps down to the last directional joint mode (60).
        assert_eq!(list.mode(5), lowest);
        assert_eq!(list.mode(6).joint(), LUMA_MODES - 1);
        assert_eq!(list.mode(7).joint(), 6);
    }

    #[test]
    fn every_mode_pair_round_trips() {
        let contexts = [
            (None, None, 3),
            (Some(Mode::VERTICAL), Some(Mode::Smooth), 4),
            (
                Some(Mode::Directional {
                    direction: Direction::D157,
                    delta: 2,
                }),
                Some(Mode::HORIZONTAL),
                6,
            ),
        ];
        let mut pairs = Vec::new();
        for (above, left, block_log2) in contexts {
            let context = ModeContext {
                above,
                left,
                block_log2,
            };
            for luma in Mode::all_luma() {
                for chroma in chroma_modes(luma) {
                    for lossless in [false, true] {
                        pairs.push((context, lossless, luma, chroma));
                    }
                }
            }
        }
        for adaptive in [false, true] {
            let mut writer = Writer::new();
            if adaptive {
                writer.enable_adaptation(100);
            }
            for &(context, lossless, luma, chroma) in &pairs {
                write_block(&mut writer, lossless, context, luma, chroma).unwrap();
            }
            let packet = writer.finish();
            let mut reader = Reader::new(&packet).unwrap();
            if adaptive {
                reader.enable_adaptation(100);
            }
            for &(context, lossless, luma, chroma) in &pairs {
                assert_eq!(
                    read_block(&mut reader, lossless, context).unwrap(),
                    (luma, chroma)
                );
            }
        }
    }

    #[test]
    fn chroma_inherits_the_luma_delta_only_for_its_direction() {
        let luma = Mode::Directional {
            direction: Direction::D113,
            delta: -2,
        };
        let list = chroma_modes(luma);
        assert_eq!(list[0], luma);
        assert_eq!(list[1], Mode::Dc);
        assert_eq!(list[6], Mode::VERTICAL);
        assert!(
            !list[1..]
                .iter()
                .any(|m| m.direction() == Some(Direction::D113))
        );
    }

    /// Symbols and CDFs of the standalone 8×8 coders' basic modes, hashed
    /// against a trace of AVM's original writer.
    const REFERENCE_TRACE: (u64, usize) = (12_253_148_367_770_241_149, 3200);
    #[test]
    fn basic_modes_match_original_avm_mode_trace() {
        let neighbours = [
            None,
            Some(Mode::Dc),
            Some(Mode::VERTICAL),
            Some(Mode::HORIZONTAL),
            Some(Mode::Paeth),
        ];
        let mut hash = 14_695_981_039_346_656_037u64;
        let mut count = 0;
        let mut symbol = |value: usize, cdf: &[u16]| {
            hash = (hash ^ value as u64).wrapping_mul(1_099_511_628_211);
            for &entry in cdf {
                hash = (hash ^ u64::from(entry)).wrapping_mul(1_099_511_628_211);
            }
            count += 1;
        };
        for lossless in [false, true] {
            for above in neighbours {
                for left in neighbours {
                    let context = ModeContext {
                        above,
                        left,
                        block_log2: SMALL_BLOCK_LOG2,
                    };
                    let list = LumaList::new(context);
                    for luma in Mode::BASIC {
                        for chroma in Mode::BASIC {
                            if lossless {
                                symbol(0, &DPCM);
                            }
                            symbol(0, &MODE_SET);
                            symbol(list.index(luma), &Y_MODE[context.y_context()]);
                            if lossless {
                                symbol(0, &DPCM);
                            }
                            let index = chroma_index(chroma, luma).unwrap();
                            symbol(index, &UV_MODE[usize::from(luma.directional())]);
                        }
                    }
                }
            }
        }
        assert_eq!((hash, count), REFERENCE_TRACE);
    }

    const REFERENCE_PIXELS: u64 = 18_203_240_803_125_398_559;
    #[test]
    fn standalone_prediction_matches_original_avm_pixels() {
        let mut plane = nc_common::Plane::new(16, 72, 0).unwrap();
        let mut hash = 14_695_981_039_346_656_037u64;
        for seed in 0..32 {
            for y in 0..72 {
                for x in 0..16 {
                    plane.view_mut().row_mut(y)[x] =
                        (x * 19 + y * 37 + x * y * 3 + seed * 13) as u8;
                }
            }
            for (x, y) in [(0, 0), (4, 0), (0, 4), (4, 4), (4, 64)] {
                for mode in Mode::BASIC {
                    for pixel in predict4(&plane, x, y, mode) {
                        hash = (hash ^ pixel as u64).wrapping_mul(1_099_511_628_211);
                    }
                }
            }
        }
        assert_eq!(hash, REFERENCE_PIXELS);
    }
}
