//! Scan orders, coefficient regions and level-dependent contexts inside one
//! transform block, for the 2D (DCT/WHT) transform class.
//!
//! Reimplements AVM v1.0.0 `av2/common/scan.c` default scans and the
//! `txb_common.h` helpers `get_lf_limits`, `get_lower_levels_ctx_eob`,
//! `get_lower_levels_ctx_2d`, `get_lower_levels_ctx_lf_2d` (and chroma
//! variants), `get_br_ctx_2d`, `get_br_lf_ctx_2d`, `get_br_ctx_2d_chroma`
//! and `get_lp2tx_ctx`. Neighbours outside the block read as zero, like
//! AVM's padded level buffers.
use super::{PLANE_V, PLANE_Y, TxSize};

/// Largest low-range increment: one four-symbol model (AVM `COEFF_BASE_RANGE`).
pub(super) const MAX_LOW_RANGE: u32 = 3;
/// Base levels before the low range starts (AVM `NUM_BASE_LEVELS`,
/// `LF_NUM_BASE_LEVELS`).
const BASE_LEVELS: u32 = 2;
const LOW_FREQUENCY_BASE_LEVELS: u32 = 4;
/// Diagonal (row + column) limits of the low-frequency region
/// (AVM `LF_2D_LIM`, `LF_2D_LIM_UV`).
const LUMA_LOW_FREQUENCY_DIAGONALS: usize = 4;
const CHROMA_LOW_FREQUENCY_DIAGONALS: usize = 1;
/// Neighbour levels are clipped before summing: to five in the
/// low-frequency region, three elsewhere, five for low-range contexts
/// (AVM `clip_max5`, `clip_max3`, `MAX_VAL_BR_CTX`).
const LOW_FREQUENCY_CLIP: u32 = 5;
const DEFAULT_CLIP: u32 = 3;
const LOW_RANGE_CLIP: u32 = 5;
/// The largest of the clips: no context tells levels above it apart, so a
/// context state may saturate neighbour levels here.
pub(super) const CONTEXT_LEVEL_CAP: u32 =
    max(max(LOW_FREQUENCY_CLIP, DEFAULT_CLIP), LOW_RANGE_CLIP);
/// Right, below, below-right, two right, two below, as (row, column) steps.
pub(super) const NEIGHBOURS: [(usize, usize); 5] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)];
/// The saturated levels of a coefficient's template neighbours, in the
/// order of [`NEIGHBOURS`] (zero outside the block or not yet coded).
pub(super) type Neighbours = [u8; NEIGHBOURS.len()];
/// Template sizes: luma base contexts use all five neighbours, chroma base
/// and every low-range context the first three.
const LUMA_TEMPLATE: usize = 5;
const SMALL_TEMPLATE: usize = 3;
/// Luma low-frequency base contexts: DC 0..=8, the two first-diagonal
/// positions 9..=15, the rest 16..=20.
const LF_DC_MAX: usize = 8;
const LF_NEAR_DIAGONALS: usize = 2;
const LF_NEAR_MAX: usize = 6;
const LF_NEAR_OFFSET: usize = 9;
const LF_FAR_MAX: usize = 4;
const LF_FAR_OFFSET: usize = 16;
/// Luma default base contexts: magnitude 0..=4 plus a band offset of 0, 5
/// or 10 at diagonals below 6, below 8, and beyond.
const DEFAULT_MAX: usize = 4;
const DEFAULT_BANDS: [(usize, usize); 2] = [(6, 0), (8, 5)];
const DEFAULT_LAST_BAND_OFFSET: usize = 10;
/// Hidden-parity DC contexts: magnitude 0..=4 (AVM
/// `COEFF_BASE_PH_CONTEXTS - 1`).
const HIDDEN_DC_MAX: usize = 4;
/// Chroma base contexts: magnitude 0..=3, V offset by four.
const CHROMA_MAX: usize = 3;
const CHROMA_V_OFFSET: usize = 4;
/// Low-range contexts: magnitude 0..=6 (0..=3 for chroma); luma
/// low-frequency AC positions use the second group of seven.
const LOW_RANGE_MAX: usize = 6;
const CHROMA_LOW_RANGE_MAX: usize = 3;
const LOW_FREQUENCY_AC_LOW_RANGE_OFFSET: usize = 7;
/// Base contexts this derivation produces: the luma low-frequency family's
/// (the largest) run to its far group's last context.
pub(super) const BASE_CONTEXTS: usize = LF_FAR_OFFSET + LF_FAR_MAX + 1;
/// Low-range contexts this derivation produces (luma low-frequency AC's
/// group is the last).
pub(super) const LOW_RANGE_CONTEXTS: usize = LOW_FREQUENCY_AC_LOW_RANGE_OFFSET + LOW_RANGE_MAX + 1;
/// The largest base alphabet: zero to one past the low-frequency base
/// levels.
pub(super) const MAX_BASE_SYMBOLS: usize = LOW_FREQUENCY_BASE_LEVELS as usize + 2;
/// Transform-type EOB contexts: default, end near DC, end near the far corner
/// (AVM `get_lp2tx_ctx` with `lim = 2`).
const TX_TYPE_EOB_DEFAULT: usize = 0;
const TX_TYPE_EOB_NEAR_DC: usize = 1;
const TX_TYPE_EOB_FAR: usize = 2;
const TX_TYPE_EOB_LIMIT: usize = 2;

const fn max(a: u32, b: u32) -> u32 {
    if a > b { a } else { b }
}

const SCAN4: [u16; 16] = diagonal_scan(TxSize::Tx4);
const SCAN8: [u16; 64] = diagonal_scan(TxSize::Tx8);
const SCAN16: [u16; 256] = diagonal_scan(TxSize::Tx16);
const SCAN32: [u16; 1024] = diagonal_scan(TxSize::Tx32);

/// AVM `default_scan_NxN`: anti-diagonals from the DC outwards, each walked
/// from its bottom-left end to its top-right end. Raster positions.
const fn diagonal_scan<const N: usize>(tx: TxSize) -> [u16; N] {
    let width = tx.coded_width();
    let mut scan = [0; N];
    let mut index = 0;
    let mut diagonal = 0;
    while diagonal < 2 * width - 1 {
        let mut row = if diagonal < width {
            diagonal
        } else {
            width - 1
        };
        loop {
            let col = diagonal - row;
            if col >= width {
                break;
            }
            scan[index] = (row * width + col) as u16;
            index += 1;
            if row == 0 {
                break;
            }
            row -= 1;
        }
        diagonal += 1;
    }
    scan
}

/// Scan index of each raster position (the inverse of the scan order).
const fn inverse_scan<const N: usize>(scan: &[u16; N]) -> [u16; N] {
    let mut inverse = [0; N];
    let mut index = 0;
    while index < N {
        inverse[scan[index] as usize] = index as u16;
        index += 1;
    }
    inverse
}

const INVERSE_SCAN4: [u16; 16] = inverse_scan(&SCAN4);
const INVERSE_SCAN8: [u16; 64] = inverse_scan(&SCAN8);
const INVERSE_SCAN16: [u16; 256] = inverse_scan(&SCAN16);
const INVERSE_SCAN32: [u16; 1024] = inverse_scan(&SCAN32);

/// End of block of raster `coefficients` (the coded region of `tx`): one
/// past the scan index of the last nonzero coefficient, zero if there is
/// none. One branch-free pass in raster order, which vectorizes, instead
/// of a backward walk through the scan that reads every position of a
/// sparse block in scattered order.
pub(super) fn end_of_block(coefficients: &[i32], tx: TxSize) -> usize {
    let inverse = inverse_scan_of(tx);
    // Signed 16-bit lanes (scan positions are below 1024): the baseline
    // x86-64 SIMD has a signed 16-bit maximum, so the loop vectorizes.
    let end = coefficients
        .iter()
        .zip(inverse)
        .fold(0i16, |end, (&c, &index)| {
            end.max(if c == 0 { 0 } else { index as i16 + 1 })
        });
    end as usize
}

/// [`end_of_block`] and the OR of all magnitudes (which exceeds a limit of
/// all-ones bits exactly when one magnitude does), in one raster pass.
pub(super) fn end_and_magnitude_bits(coefficients: &[i32], tx: TxSize) -> (usize, u32) {
    let inverse = inverse_scan_of(tx);
    let (end, bits) =
        coefficients
            .iter()
            .zip(inverse)
            .fold((0i16, 0u32), |(end, bits), (&c, &index)| {
                (
                    end.max(if c == 0 { 0 } else { index as i16 + 1 }),
                    bits | c.unsigned_abs(),
                )
            });
    (end as usize, bits)
}

fn inverse_scan_of(tx: TxSize) -> &'static [u16] {
    match tx {
        TxSize::Tx4 => &INVERSE_SCAN4,
        TxSize::Tx8 => &INVERSE_SCAN8,
        TxSize::Tx16 => &INVERSE_SCAN16,
        TxSize::Tx32 | TxSize::Tx64 => &INVERSE_SCAN32,
    }
}

/// Scan order of a transform's coded region.
pub(super) fn scan(tx: TxSize) -> &'static [u16] {
    match tx {
        TxSize::Tx4 => &SCAN4,
        TxSize::Tx8 => &SCAN8,
        TxSize::Tx16 => &SCAN16,
        TxSize::Tx32 | TxSize::Tx64 => &SCAN32,
    }
}

/// One position of a scan with what the context derivations need of it,
/// precomputed per transform size: the coefficient walk reads it instead of
/// re-deriving the row, column, diagonal and map index of every
/// coefficient for every context.
#[derive(Debug, Clone, Copy)]
pub(super) struct ScanPosition {
    /// Raster position in the coded region.
    pos: u16,
    /// Index in [`LevelMap`].
    map: u16,
    /// Anti-diagonal (row + column).
    diagonal: u8,
    /// Row in the coded region.
    row: u8,
}

impl ScanPosition {
    /// Raster position in the coded region.
    pub(super) fn pos(self) -> usize {
        usize::from(self.pos)
    }

    /// The position's coding region in `plane`.
    pub(super) fn region(self, plane: u8) -> Region {
        Region::of_diagonal(usize::from(self.diagonal), plane)
    }

    /// Anti-diagonal (row + column).
    pub(super) fn diagonal(self) -> usize {
        usize::from(self.diagonal)
    }

    /// Row in the coded region.
    pub(super) fn row(self) -> usize {
        usize::from(self.row)
    }
}

/// [`ScanPosition`]s of `scan` (the scan of `tx`).
const fn scan_positions_of<const N: usize>(scan: &[u16; N], tx: TxSize) -> [ScanPosition; N] {
    let log2 = tx.coded_log2();
    let mut positions = [ScanPosition {
        pos: 0,
        map: 0,
        diagonal: 0,
        row: 0,
    }; N];
    let mut i = 0;
    while i < N {
        let pos = scan[i] as usize;
        let (row, col) = (pos >> log2, pos & ((1 << log2) - 1));
        positions[i] = ScanPosition {
            pos: pos as u16,
            map: (row * MAP_STRIDE + col) as u16,
            diagonal: (row + col) as u8,
            row: row as u8,
        };
        i += 1;
    }
    positions
}

const POSITIONS4: [ScanPosition; 16] = scan_positions_of(&SCAN4, TxSize::Tx4);
const POSITIONS8: [ScanPosition; 64] = scan_positions_of(&SCAN8, TxSize::Tx8);
const POSITIONS16: [ScanPosition; 256] = scan_positions_of(&SCAN16, TxSize::Tx16);
const POSITIONS32: [ScanPosition; 1024] = scan_positions_of(&SCAN32, TxSize::Tx32);

/// [`ScanPosition`]s of a transform's scan, in scan order.
pub(super) fn scan_positions(tx: TxSize) -> &'static [ScanPosition] {
    match tx {
        TxSize::Tx4 => &POSITIONS4,
        TxSize::Tx8 => &POSITIONS8,
        TxSize::Tx16 => &POSITIONS16,
        TxSize::Tx32 | TxSize::Tx64 => &POSITIONS32,
    }
}

/// The coding region of a coefficient, which picks its models and limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Region {
    /// Luma, row + column below four.
    LumaLow,
    /// Luma elsewhere.
    LumaDefault,
    /// Chroma DC.
    ChromaLow,
    /// Chroma elsewhere.
    ChromaDefault,
}

impl Region {
    /// Number of regions ([`Self::index`] is below it).
    pub(super) const COUNT: usize = 4;

    /// The region's index in per-region tables.
    pub(super) fn index(self) -> usize {
        self as usize
    }

    /// Region of raster position `pos` (AVM `get_lf_limits`, 2D class).
    #[cfg(test)]
    fn of(pos: usize, tx: TxSize, plane: u8) -> Self {
        Self::of_diagonal(diagonal(pos, tx), plane)
    }

    /// Region of a position on anti-diagonal `diagonal` (row + column).
    fn of_diagonal(diagonal: usize, plane: u8) -> Self {
        match (plane == PLANE_Y, diagonal) {
            (true, d) if d < LUMA_LOW_FREQUENCY_DIAGONALS => Self::LumaLow,
            (true, _) => Self::LumaDefault,
            (false, d) if d < CHROMA_LOW_FREQUENCY_DIAGONALS => Self::ChromaLow,
            (false, _) => Self::ChromaDefault,
        }
    }

    /// Levels coded by the base symbol before the low range takes over.
    pub(super) fn base_levels(self) -> u32 {
        match self {
            Self::LumaLow | Self::ChromaLow => LOW_FREQUENCY_BASE_LEVELS,
            Self::LumaDefault | Self::ChromaDefault => BASE_LEVELS,
        }
    }

    /// Chroma DC has no low-range symbol: its base symbol saturates directly.
    pub(super) fn has_low_range(self) -> bool {
        self != Self::ChromaLow
    }

    /// Largest level carried by base and low-range symbols; a coefficient at
    /// this level continues with a high-range remainder (AVM `read_high_range`).
    pub(super) fn saturation(self) -> u32 {
        self.base_levels()
            + 1
            + if self.has_low_range() {
                MAX_LOW_RANGE
            } else {
                0
            }
    }
}

fn diagonal(pos: usize, tx: TxSize) -> usize {
    let log2 = tx.coded_log2();
    (pos >> log2) + (pos & ((1 << log2) - 1))
}

/// Base context of the last coefficient (AVM `get_lower_levels_ctx_eob`):
/// by its scan index relative to an eighth and a quarter of the block.
pub(super) fn last_base_context(index: usize, tx: TxSize) -> usize {
    let area = tx.coded_len();
    if index == 0 {
        0
    } else if index <= area / 8 {
        1
    } else if index <= area / 4 {
        2
    } else {
        3
    }
}

/// Inter transform-type context from the last position's diagonal.
pub(super) fn tx_type_eob_context(eob: usize, tx: TxSize) -> usize {
    let diagonal = diagonal(eob - 1, tx);
    let max_diagonal = 2 * tx.coded_width() - 2;
    if diagonal < TX_TYPE_EOB_LIMIT {
        TX_TYPE_EOB_NEAR_DC
    } else if diagonal > max_diagonal - TX_TYPE_EOB_LIMIT {
        TX_TYPE_EOB_FAR
    } else {
        TX_TYPE_EOB_DEFAULT
    }
}

/// Zero columns right of and rows below the coded region in [`LevelMap`]:
/// the context templates reach two positions right and down.
const MAP_PADDING: usize = 2;
/// Row stride of [`LevelMap`]: the widest coded region plus the padding.
pub(super) const MAP_STRIDE: usize = TxSize::Tx64.coded_width() + MAP_PADDING;
/// The template neighbours as index offsets in [`LevelMap`], in the order
/// of [`NEIGHBOURS`].
const NEIGHBOUR_OFFSETS: [usize; 5] = {
    let mut offsets = [0; 5];
    let mut i = 0;
    while i < offsets.len() {
        offsets[i] = NEIGHBOURS[i].0 * MAP_STRIDE + NEIGHBOURS[i].1;
        i += 1;
    }
    offsets
};

/// Base plus low-range levels of the coefficients coded so far (later in scan
/// order), saturated as in the decoder; zero elsewhere. Stored with zero
/// padding right and below (AVM's padded level buffer), so the context
/// templates read neighbours without bounds tests.
pub(super) struct LevelMap {
    levels: [u8; MAP_STRIDE * MAP_STRIDE],
}

impl LevelMap {
    /// An all-zero map (positions come with their map index, so the map
    /// does not depend on the transform size).
    pub(super) fn new() -> Self {
        Self {
            levels: [0; MAP_STRIDE * MAP_STRIDE],
        }
    }

    /// Record a coded level (at most [`Region::saturation`], so it fits a byte).
    pub(super) fn set(&mut self, at: ScanPosition, level: u32) {
        self.levels[usize::from(at.map)] = level.min(u32::from(u8::MAX)) as u8;
    }

    pub(super) fn get(&self, at: ScanPosition) -> u32 {
        u32::from(self.levels[usize::from(at.map)])
    }

    /// The template neighbours of `at`.
    pub(super) fn neighbours(&self, at: ScanPosition) -> Neighbours {
        let index = usize::from(at.map);
        NEIGHBOUR_OFFSETS.map(|offset| self.levels[index + offset])
    }
}

/// Half the sum of the first `template` neighbours clipped at `clip`,
/// rounded up.
fn magnitude(neighbours: Neighbours, template: usize, clip: u32) -> usize {
    let sum: u32 = neighbours[..template]
        .iter()
        .map(|&level| u32::from(level).min(clip))
        .sum();
    sum.div_ceil(2) as usize
}

/// Context of a hidden-parity DC level (AVM `get_base_ctx_ph`): half the
/// sum of its five template neighbours clipped at three, rounded up, at
/// most four.
pub(super) fn hidden_dc_context(neighbours: Neighbours) -> usize {
    magnitude(neighbours, LUMA_TEMPLATE, DEFAULT_CLIP).min(HIDDEN_DC_MAX)
}

/// Base-symbol context of a coefficient other than the last one at `at`, in
/// its `region` of `plane`, from its template `neighbours`.
pub(super) fn base_context(
    neighbours: Neighbours,
    at: ScanPosition,
    region: Region,
    plane: u8,
) -> usize {
    match region {
        Region::ChromaLow | Region::ChromaDefault => {
            let clip = if region == Region::ChromaLow {
                LOW_FREQUENCY_CLIP
            } else {
                DEFAULT_CLIP
            };
            magnitude(neighbours, SMALL_TEMPLATE, clip).min(CHROMA_MAX)
                + usize::from(plane == PLANE_V) * CHROMA_V_OFFSET
        }
        Region::LumaLow | Region::LumaDefault => {
            luma_base_magnitude(neighbours, at, region) + luma_base_offset(at, region)
        }
    }
}

/// The part of a luma base context (other than the last coefficient's)
/// read from the template `neighbours` of `at` in `region`: their clipped
/// magnitude, capped for the position. [`base_context`] adds
/// [`luma_base_offset`], so equal magnitudes give equal contexts at a
/// position.
pub(super) fn luma_base_magnitude(
    neighbours: Neighbours,
    at: ScanPosition,
    region: Region,
) -> usize {
    let diagonal = usize::from(at.diagonal);
    match region {
        Region::LumaLow => {
            let magnitude = magnitude(neighbours, LUMA_TEMPLATE, LOW_FREQUENCY_CLIP);
            if at.pos == 0 {
                magnitude.min(LF_DC_MAX)
            } else if diagonal < LF_NEAR_DIAGONALS {
                magnitude.min(LF_NEAR_MAX)
            } else {
                magnitude.min(LF_FAR_MAX)
            }
        }
        _ => magnitude(neighbours, LUMA_TEMPLATE, DEFAULT_CLIP).min(DEFAULT_MAX),
    }
}

/// Largest [`luma_base_magnitude`] (the DC's cap).
pub(super) const MAX_LUMA_BASE_MAGNITUDE: usize = LF_DC_MAX;

/// The luma base context of `at` in `region` at magnitude zero.
pub(super) fn luma_base_offset(at: ScanPosition, region: Region) -> usize {
    let diagonal = usize::from(at.diagonal);
    match region {
        Region::LumaLow if at.pos == 0 => 0,
        Region::LumaLow if diagonal < LF_NEAR_DIAGONALS => LF_NEAR_OFFSET,
        Region::LumaLow => LF_FAR_OFFSET,
        _ => DEFAULT_BANDS
            .iter()
            .find(|&&(limit, _)| diagonal < limit)
            .map_or(DEFAULT_LAST_BAND_OFFSET, |&(_, offset)| offset),
    }
}

/// Low-range context of the coefficient at `at` in its `region`, from its
/// template `neighbours`.
pub(super) fn low_range_context(neighbours: Neighbours, at: ScanPosition, region: Region) -> usize {
    let magnitude = magnitude(neighbours, SMALL_TEMPLATE, LOW_RANGE_CLIP);
    match region {
        Region::LumaLow if at.pos != 0 => {
            magnitude.min(LOW_RANGE_MAX) + LOW_FREQUENCY_AC_LOW_RANGE_OFFSET
        }
        Region::LumaLow | Region::LumaDefault => magnitude.min(LOW_RANGE_MAX),
        Region::ChromaLow | Region::ChromaDefault => magnitude.min(CHROMA_LOW_RANGE_MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_visit_every_position_once_in_diagonal_order() {
        assert_eq!(
            SCAN4,
            [0, 4, 1, 8, 5, 2, 12, 9, 6, 3, 13, 10, 7, 14, 11, 15]
        );
        for tx in TxSize::ALL {
            let scan = scan(tx);
            let mut seen = vec![false; tx.coded_len()];
            for pair in scan.windows(2) {
                let (a, b) = (usize::from(pair[0]), usize::from(pair[1]));
                assert!(diagonal(a, tx) <= diagonal(b, tx));
            }
            for &pos in scan {
                assert!(!std::mem::replace(&mut seen[usize::from(pos)], true));
            }
            assert!(seen.into_iter().all(|s| s));
        }
    }

    #[test]
    fn end_of_block_follows_the_scan() {
        for tx in TxSize::ALL {
            let scan = scan(tx);
            let mut coefficients = vec![0; tx.coded_len()];
            assert_eq!(end_of_block(&coefficients, tx), 0);
            for (index, &pos) in scan.iter().enumerate().step_by(7) {
                coefficients[usize::from(pos)] = if index % 2 == 0 { 3 } else { -1 };
                assert_eq!(end_of_block(&coefficients, tx), index + 1);
            }
        }
    }

    #[test]
    fn region_saturation_matches_avm_high_range_starts() {
        assert_eq!(Region::LumaLow.saturation(), 8);
        assert_eq!(Region::ChromaLow.saturation(), 5);
        assert_eq!(Region::LumaDefault.saturation(), 6);
        assert_eq!(Region::ChromaDefault.saturation(), 6);
        assert_eq!(Region::of(3, TxSize::Tx16, 0), Region::LumaLow);
        assert_eq!(Region::of(16 * 2 + 2, TxSize::Tx16, 0), Region::LumaDefault);
        assert_eq!(Region::of(1, TxSize::Tx8, 2), Region::ChromaDefault);
    }

    #[test]
    fn last_and_tx_type_contexts_scale_with_size() {
        assert_eq!(last_base_context(128, TxSize::Tx32), 1);
        assert_eq!(last_base_context(129, TxSize::Tx32), 2);
        assert_eq!(last_base_context(257, TxSize::Tx64), 3);
        assert_eq!(tx_type_eob_context(1, TxSize::Tx16), TX_TYPE_EOB_NEAR_DC);
        assert_eq!(tx_type_eob_context(256, TxSize::Tx16), TX_TYPE_EOB_FAR);
        assert_eq!(
            tx_type_eob_context(16 * 5 + 1, TxSize::Tx16),
            TX_TYPE_EOB_DEFAULT
        );
    }
}
