//! H.266 §9.3 rectangular residuals and dependent-quantization state transitions.
//! Shares the existing CABAC engine and ordinary coefficient context banks.
//! Additional state contexts follow BSD-Clear `VVdeC` (`THIRD_PARTY.md`).
use std::sync::OnceLock;

use super::initialization::bank;
use crate::{coeff_contexts::CoeffContexts, contexts::Context, decoder_cabac::Reader};
use nc_common::decode::{DecodeError, Result};

const GROUP_START: [usize; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];
const LAST_GROUP_LIMIT: [usize; 6] = [0, 1, 3, 5, 7, 9];
const STATE_TRANSITIONS: u32 = 32_040;
const MAX_CODED_SIDE: usize = 32;
const PAD: usize = 2;
const BUDGET_NUM: usize = 7;
const BUDGET_DEN: usize = 4;
const MIN_REGULAR_BUDGET: usize = 4;
const MAX_GROUP_SIZE: usize = 16;
const MAX_CG_COUNT: usize = 64;
const FIRST_RICE_BASE: u32 = 4;
const MTS_REGION_LIMIT: usize = 16;
const ZERO_OUT_GROUP_LIMIT: usize = 4;
/// `log2TransformRange` without extended precision (H.266 §7.4.3.22): 15.
pub(super) const BASE_LOG2_RANGE: u32 = 15;
/// Bits of the prefix-limited escape code's whole span (§9.3.3.11:
/// `maxPrefixExtLen = 32 − log2TransformRange` prefix bins).
const ESCAPE_SPAN: u32 = 32;
/// Template sum thresholds and shifts of the extended Rice derivation
/// (§9.3.3.2 with `sps_rrc_rice_extension_flag`, VTM `g_riceThreshold`).
const RICE_SHIFT_THRESHOLDS: [u32; 4] = [32, 128, 512, 2048];
/// The shift of each threshold interval (2 per interval).
const RICE_SHIFT_STEP: u32 = 2;
/// The largest clipped template sum (§9.3.3.2: `Clip3(0, 31, ...)`).
const MAX_TEMPLATE_SUM: u32 = 31;

pub(super) const PADDED_AREA: usize = (MAX_CODED_SIDE + PAD) * (MAX_CODED_SIDE + PAD);

/// What a slice's regular residual coding takes from the range extension
/// (H.266 §7.3.11.11, §9.3.3.2, §9.3.3.11), with the persistent Rice
/// statistics (`StatCoeff`) it carries from block to block.
#[derive(Debug, Clone, Copy)]
pub(super) struct RiceContext {
    /// `log2TransformRange` (15, or `Max(15, BitDepth + 6)` with extended
    /// precision): the escape code's length and the coefficient range.
    pub log2_range: u32,
    /// `baseLevel` of the remainders after the context-coded flags: 4, or
    /// by depth and slice type with the extended Rice derivation.
    pub base_level: u32,
    /// `sps_rrc_rice_extension_flag`.
    pub extension: bool,
    /// `StatCoeff` per component with persistent Rice adaptation.
    pub statistics: Option<[u32; 3]>,
    /// `sh_reverse_last_sig_coeff_flag`.
    pub reverse_last: bool,
}

impl RiceContext {
    /// Version 1 residual coding: none of the extension's tools.
    pub(super) const PLAIN: Self = Self {
        log2_range: BASE_LOG2_RANGE,
        base_level: FIRST_RICE_BASE,
        extension: false,
        statistics: None,
        reverse_last: false,
    };

    /// `StatCoeff` at a slice, tile or entropy-synchronised CTU row start
    /// (§9.3.2.2: `2 · Floor(Log2(BitDepth − 10))`).
    pub(super) fn reset_statistics(&mut self, bits: u32) {
        if let Some(statistics) = &mut self.statistics {
            *statistics = [2 * (bits.saturating_sub(10)).max(1).ilog2(); 3];
        }
    }

    /// The smallest and largest coefficient (`CoeffMinY`... §7.4.11.11).
    fn coefficient_range(&self) -> std::ops::RangeInclusive<i32> {
        -(1 << self.log2_range)..=(1 << self.log2_range) - 1
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CoeffStats {
    pub last_scan: Option<u16>,
    pub outside_16: bool,
}

#[derive(Clone)]
pub(super) struct Models {
    pub common: CoeffContexts,
    extra_significance: [[[Context; 12]; 2]; 2],
}
impl Models {
    pub fn new(qp: u32, kind: u8) -> Self {
        Self {
            common: super::initialization::coefficients(qp, kind),
            extra_significance: [
                [
                    bank(
                        [
                            [26, 45, 53, 46, 49, 54, 61, 39, 35, 39, 39, 39],
                            [19, 38, 38, 46, 34, 54, 54, 39, 6, 39, 39, 39],
                            [11, 38, 46, 54, 27, 39, 39, 39, 44, 39, 39, 39],
                        ],
                        [9, 13, 8, 8, 8, 8, 8, 5, 8, 0, 0, 0],
                        qp,
                        kind,
                    ),
                    bank(
                        [
                            [19, 54, 39, 39, 50, 39, 39, 39, 0, 39, 39, 39],
                            [19, 39, 54, 39, 19, 39, 39, 39, 56, 39, 39, 39],
                            [18, 39, 39, 39, 27, 39, 39, 39, 0, 39, 39, 39],
                        ],
                        [8, 8, 8, 8, 8, 0, 4, 4, 0, 0, 0, 0],
                        qp,
                        kind,
                    ),
                ],
                [
                    bank(
                        [
                            [34, 45, 38, 31, 58, 39, 39, 39],
                            [35, 45, 53, 54, 44, 39, 39, 39],
                            [19, 46, 38, 39, 52, 39, 39, 39],
                        ],
                        [8, 12, 12, 8, 4, 0, 0, 0],
                        qp,
                        kind,
                    ),
                    bank(
                        [
                            [34, 38, 54, 39, 41, 39, 39, 39],
                            [34, 38, 62, 39, 26, 39, 39, 39],
                            [11, 39, 39, 39, 19, 39, 39, 39],
                        ],
                        [8, 8, 8, 8, 4, 0, 0, 0],
                        qp,
                        kind,
                    ),
                ],
            ],
        }
    }
    fn significance(&mut self, channel: usize, state: u32, index: usize) -> &mut Context {
        if state < 2 {
            &mut self.common.sig[channel][index]
        } else {
            &mut self.extra_significance[channel][state as usize - 2][index]
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ScanEntry {
    pub pos: u16,
    pub pad_pos: u16,
    pub sig_base: [u8; 2],
    pub level_base: [u8; 2],
    pub outside_16: bool,
}

#[derive(Debug)]
pub(super) struct Scan {
    pub(super) positions: Vec<usize>,
    pub(super) entries: Vec<ScanEntry>,
    pub(super) inverse: Vec<u16>,
    pub(super) groups: Vec<usize>,
    pub(super) group_size: usize,
    pub(super) columns: usize,
    pub(super) rows: usize,
    pub(super) coded_width: usize,
}
impl Scan {
    pub(super) fn new(width: usize, height: usize) -> Self {
        // Narrow chroma transforms use 2x2 or 2x8 groups; the ordinary
        // square path keeps the encoder's precomputed 4x4-group scans.
        let (gw, gh) = match (width, height) {
            (1, _) => (1, height.min(16)),
            (_, 1) => (width.min(16), 1),
            (2, 2 | 4) | (4, 2) => (2, 2),
            (2, _) => (2, 8),
            (_, 2) => (8, 2),
            _ => (4, 4),
        };
        let coded_width = width.min(MAX_CODED_SIDE);
        let coded_height = height.min(MAX_CODED_SIDE);
        let pad_stride = coded_width + PAD;
        let columns = coded_width / gw;
        let rows = coded_height / gh;
        let groups = diagonal(columns, rows);
        let local = diagonal(gw, gh);
        let positions: Vec<usize> = groups
            .iter()
            .flat_map(|&group| {
                local.iter().map(move |&position| {
                    (group / columns * gh + position / gw) * width
                        + group % columns * gw
                        + position % gw
                })
            })
            .collect();
        let mut entries = Vec::with_capacity(positions.len());
        let mut inverse = vec![0u16; coded_width * coded_height];
        for (scan_idx, &pos) in positions.iter().enumerate() {
            let (x, y) = (pos % width, pos / width);
            let diag = (x + y).min(crate::coeff::DIAGONAL_CLASSES - 1);
            let (s0, l0) = crate::coeff::DIAGONAL_OFFSETS[0][diag];
            let (s1, l1) = crate::coeff::DIAGONAL_OFFSETS[1][diag];
            entries.push(ScanEntry {
                pos: pos as u16,
                pad_pos: (y * pad_stride + x) as u16,
                sig_base: [s0, s1],
                level_base: [l0 + 1, l1 + 1],
                outside_16: x >= MTS_REGION_LIMIT || y >= MTS_REGION_LIMIT,
            });
            inverse[y * coded_width + x] = scan_idx as u16;
        }
        Self {
            positions,
            entries,
            inverse,
            groups,
            group_size: gw * gh,
            columns,
            rows,
            coded_width,
        }
    }
    pub(super) fn of(width: usize, height: usize) -> &'static Self {
        static SCANS: OnceLock<[[Scan; 7]; 7]> = OnceLock::new();
        let table = SCANS.get_or_init(|| {
            std::array::from_fn(|w_log2| {
                std::array::from_fn(|h_log2| Self::new(1 << w_log2, 1 << h_log2))
            })
        });
        &table[width.ilog2() as usize][height.ilog2() as usize]
    }
}
fn diagonal(width: usize, height: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(width * height);
    for sum in 0..width + height - 1 {
        for x in sum.saturating_sub(height - 1)..=sum.min(width - 1) {
            order.push((sum - x) * width + x);
        }
    }
    order
}
fn last_prefix(
    r: &mut Reader<'_>,
    models: &mut [Context],
    side: usize,
    channel: usize,
    limit_side: usize,
) -> Result<usize> {
    let log2 = side.ilog2() as usize;
    let offset = if channel == 0 {
        [0, 0, 0, 3, 6, 10, 15][log2]
    } else {
        0
    };
    let shift = if channel == 0 {
        (log2 + 1) >> 2
    } else {
        (side >> 3).min(2)
    };
    let limit = LAST_GROUP_LIMIT[limit_side.min(MAX_CODED_SIDE).ilog2() as usize];
    let mut group = 0;
    while group < limit && r.bin(&mut models[offset + (group >> shift)])? {
        group += 1;
    }
    Ok(group)
}
fn last_suffix(r: &mut Reader<'_>, group: usize) -> Result<usize> {
    Ok(GROUP_START[group]
        + if group > 3 {
            r.bypass_bits(((group - 2) >> 1) as u32)? as usize
        } else {
            0
        })
}
/// A side of the coefficients a block may code: at most 32, and 16 of 32
/// when the MTS or SBT zero-out applies (H.266 §7.3.11.11).
fn zero_out_side(side: usize, zero_out: bool) -> usize {
    if zero_out {
        side.min(MTS_REGION_LIMIT)
    } else {
        side.min(MAX_CODED_SIDE)
    }
}

fn last_position(
    r: &mut Reader<'_>,
    models: &mut Models,
    width: usize,
    height: usize,
    channel: usize,
    scan: &Scan,
    (zero_out, reverse): (bool, bool),
) -> Result<usize> {
    let limit_w = zero_out_side(width, zero_out);
    let limit_h = zero_out_side(height, zero_out);
    let gx = last_prefix(
        r,
        &mut models.common.last_x[channel],
        width,
        channel,
        limit_w,
    )?;
    let gy = last_prefix(
        r,
        &mut models.common.last_y[channel],
        height,
        channel,
        limit_h,
    )?;
    let mut x = last_suffix(r, gx)?;
    let mut y = last_suffix(r, gy)?;
    if x >= limit_w || y >= limit_h {
        return Err(DecodeError::Malformed("VVC last coefficient position"));
    }
    if reverse {
        // Coded from the far corner of the zero-out region (§7.4.11.11).
        (x, y) = (limit_w - 1 - x, limit_h - 1 - y);
    }
    Ok(usize::from(scan.inverse[y * scan.coded_width + x]))
}

#[inline(always)]
#[allow(clippy::inline_always, reason = "per-coefficient CABAC context lookup")]
fn grid_context(pass1: &[u8], pad_stride: usize, entry: ScanEntry, ch: usize) -> (usize, usize) {
    let p = usize::from(entry.pad_pos);
    let win = &pass1[p..=p + 2 * pad_stride];
    let packed = u32::from(win[1])
        + u32::from(win[2])
        + u32::from(win[pad_stride])
        + u32::from(win[pad_stride + 1])
        + u32::from(win[2 * pad_stride]);
    let sum = packed & 0x1f;
    let count = packed >> 5;
    let sig = (((sum + 1) >> 1).min(3) as usize) + usize::from(entry.sig_base[ch]);
    let level = ((sum - count).min(4) as usize) + usize::from(entry.level_base[ch]);
    (sig, level)
}

#[inline(always)]
#[allow(clippy::inline_always, reason = "per-coefficient Golomb-Rice lookup")]
fn grid_rice(grid: &[u32], pad_stride: usize, pad_pos: u16, base: u32) -> u32 {
    crate::coeff::rice_parameter(grid_sum(grid, pad_stride, pad_pos), base)
}

/// The template sum `locSumAbs` of the padded grid at `pad_pos` (§9.3.3.2).
#[inline(always)]
#[allow(clippy::inline_always, reason = "per-coefficient Golomb-Rice lookup")]
fn grid_sum(grid: &[u32], pad_stride: usize, pad_pos: u16) -> u32 {
    let p = usize::from(pad_pos);
    let win = &grid[p..=p + 2 * pad_stride];
    win[1] + win[2] + win[pad_stride] + win[pad_stride + 1] + win[2 * pad_stride]
}

/// One `abs_remainder` / `dec_abs_level` (§9.3.3.11): a Rice code of
/// parameter `rice`, escaping to Exp-Golomb after 5 prefix bins, the
/// prefix limited to `32 − log2_range` bins, after which `log2_range`
/// bits follow.
#[inline]
/// `EXT` with extended precision; without it `log2_range` is 15 and the
/// escape's bounds are constants.
pub(super) fn remainder<const EXT: bool>(
    r: &mut Reader<'_>,
    rice: u32,
    log2_range: u32,
) -> Result<u32> {
    const CUTOFF: u32 = 5;
    let log2_range = if EXT { log2_range } else { BASE_LOG2_RANGE };
    let max_prefix = ESCAPE_SPAN - log2_range;
    let prefix = r.bypass_unary(max_prefix);
    if prefix < CUTOFF {
        return Ok(if rice == 0 {
            prefix
        } else {
            (prefix << rice) + r.bypass_bits(rice)?
        });
    }
    let escape = prefix - CUTOFF;
    let base = (CUTOFF + (1 << escape) - 1) << rice;
    Ok(base
        + r.bypass_bits(if prefix == max_prefix {
            log2_range
        } else {
            escape + rice
        })?)
}

/// Parse one transform block's regular residual coding (§7.3.11.11) of
/// `plane` into `arena`; `rice` carries the range extension's state.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn read_into(
    r: &mut Reader<'_>,
    models: &mut Models,
    arena: &mut super::CoeffArena,
    width: usize,
    height: usize,
    (plane, rice): (usize, &mut RiceContext),
    dependent: bool,
    hiding: bool,
    zero_out: bool,
) -> Result<((u32, u16), CoeffStats)> {
    let channel = usize::from(plane != 0);
    if ![1, 2, 4, 8, 16, 32, 64].contains(&width)
        || ![1, 2, 4, 8, 16, 32, 64].contains(&height)
        || channel > 1
    {
        return Err(DecodeError::Unsupported("VVC residual geometry"));
    }
    // Levels beyond 16 bits only with extended precision.
    let start = arena.span_start(rice.log2_range > BASE_LOG2_RANGE)?;
    let scan = Scan::of(width, height);
    let last = last_position(
        r,
        models,
        width,
        height,
        channel,
        scan,
        (zero_out, rice.reverse_last),
    )?;
    let statistic = rice.statistics.map(|statistics| statistics[plane]);
    let extended = rice.extension
        || statistic.is_some()
        || rice.log2_range != BASE_LOG2_RANGE
        || rice.base_level != FIRST_RICE_BASE;
    let block = Block {
        size: (width, height),
        plane,
        scan,
        last,
        flags: (dependent, hiding, zero_out),
    };
    let stats = if extended {
        walk_groups::<true>(r, models, arena, block, rice)?
    } else {
        walk_groups::<false>(r, models, arena, block, rice)?
    };
    Ok((arena.span_from(start)?, stats))
}

/// The transform block a residual walk covers.
#[derive(Clone, Copy)]
struct Block<'s> {
    size: (usize, usize),
    plane: usize,
    scan: &'s Scan,
    last: usize,
    /// Dependent quantisation, sign hiding and the MTS/SBT zero-out.
    flags: (bool, bool, bool),
}

/// The coefficient groups of `block` from its last one back (§7.3.11.11).
#[inline]
fn walk_groups<const EXT: bool>(
    r: &mut Reader<'_>,
    models: &mut Models,
    arena: &mut super::CoeffArena,
    block: Block<'_>,
    rice: &mut RiceContext,
) -> Result<CoeffStats> {
    let Block {
        size: (width, height),
        plane,
        scan,
        last,
        flags: (dependent, hiding, zero_out),
    } = block;
    let channel = usize::from(plane != 0);
    let statistic = rice.statistics.map(|statistics| statistics[plane]);
    let mut walk = Walk::<EXT> {
        narrow: &mut arena.entries,
        wide: (rice.log2_range > BASE_LOG2_RANGE).then_some(&mut arena.wide),
        grid: &mut arena.grid[..],
        pass1: &mut arena.pass1[..],
        max_pad: 0,
        pad_stride: scan.coded_width + PAD,
        state: 0,
        // Context-coded bins: 7/4 per coefficient of the area left after
        // the zero-out (H.266 §7.3.11.11 `remCcbs` from `log2ZoTbWidth`,
        // `log2ZoTbHeight`; SBT and MTS keep 16 of 32).
        budget: zero_out_side(width, zero_out) * zero_out_side(height, zero_out) * BUDGET_NUM
            / BUDGET_DEN,
        channel,
        dependent,
        hiding,
        stats: CoeffStats::default(),
        rice: Rice {
            context: *rice,
            size: (width, height),
            history: statistic.map_or(0, |stat| 1 << stat),
            statistic,
            pending: statistic.is_some(),
            general: rice.extension || statistic.is_some(),
        },
    };
    let _ = MAX_CG_COUNT;
    let mut significant = 0u64;
    let last_group = last / scan.group_size;
    for group in (0..=last_group).rev() {
        let raster = scan.groups[group];
        let (gx, gy) = (raster % scan.columns, raster / scan.columns);
        if zero_out
            && ((width == MAX_CODED_SIDE && gx >= ZERO_OUT_GROUP_LIMIT)
                || (height == MAX_CODED_SIDE && gy >= ZERO_OUT_GROUP_LIMIT))
        {
            continue;
        }
        let adjacent = (gx + 1 < scan.columns && (significant >> (raster + 1)) & 1 != 0)
            || (gy + 1 < scan.rows && (significant >> (raster + scan.columns)) & 1 != 0);
        if group != 0
            && group != last_group
            && !r.bin(&mut models.common.group[channel][usize::from(adjacent)])?
        {
            continue;
        }
        significant |= 1u64 << raster;
        walk.group(r, models, scan, group, last)?;
    }
    if let (Some(statistics), Some(statistic)) = (&mut rice.statistics, walk.rice.statistic) {
        statistics[plane] = statistic;
    }
    Ok(walk.stats)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn read_general(
    r: &mut Reader<'_>,
    models: &mut Models,
    width: usize,
    height: usize,
    channel: usize,
    dependent: bool,
    hiding: bool,
    zero_out: bool,
) -> Result<(Vec<i32>, CoeffStats)> {
    let mut arena = super::CoeffArena::default();
    let mut rice = RiceContext::PLAIN;
    let (_, stats) = read_into(
        r,
        models,
        &mut arena,
        width,
        height,
        (channel, &mut rice),
        dependent,
        hiding,
        zero_out,
    )?;
    let mut out = vec![0i32; width * height];
    arena
        .slice((0, arena.entries.len() as u16))
        .scatter(&mut out);
    Ok((out, stats))
}

#[derive(Clone, Copy, Default)]
struct ActiveCoeff {
    scan_idx: u16,
    pos: u16,
    pad_pos: u16,
    dep_adj: u8,
    outside_16: bool,
}

/// One block's regular residual walk; `EXT` when a range-extension tool
/// (extended precision, Rice extension, persistent Rice) is on, so version
/// 1 streams run the version 1 derivations with constant bounds.
struct Walk<'a, const EXT: bool> {
    /// Where 16-bit levels go, and 32-bit ones with extended precision.
    narrow: &'a mut Vec<(u16, i16)>,
    wide: Option<&'a mut Vec<(u16, i32)>>,
    grid: &'a mut [u32],
    pass1: &'a mut [u8],
    max_pad: usize,
    pad_stride: usize,
    state: u32,
    budget: usize,
    channel: usize,
    dependent: bool,
    hiding: bool,
    stats: CoeffStats,
    rice: Rice,
}

/// The Rice parameter derivation of one transform block (§9.3.3.2).
struct Rice {
    context: RiceContext,
    /// The block's width and height (template positions beyond them take
    /// `history`).
    size: (usize, usize),
    /// `HistValue`: `1 << StatCoeff` at the block's start, or 0.
    history: u32,
    /// The component's `StatCoeff` (with persistent Rice adaptation).
    statistic: Option<u32>,
    /// Whether `statistic` still awaits the block's first nonzero
    /// remainder, which alone updates it.
    pending: bool,
    /// Whether the extension's derivation applies (else version 1's).
    general: bool,
}

impl Rice {
    /// The Rice parameter of the coefficient at raster `pos` whose padded
    /// grid template sums to `sum`, above `base` (`baseLevel`).
    fn parameter(&self, sum: u32, pos: u16, base: u32) -> u32 {
        let (width, height) = self.size;
        let (x, y) = (usize::from(pos) % width, usize::from(pos) / width);
        // Template positions beyond the block count `HistValue` (VTM
        // `templateAbsSum`: a last column counts two).
        let outside = if x + 1 < width {
            u32::from(x + 2 >= width) + u32::from(y + 1 >= height)
        } else {
            2
        } + if y + 1 < height {
            u32::from(y + 2 >= height)
        } else {
            1
        };
        let sum = sum + outside * self.history;
        if !self.context.extension {
            return crate::coeff::rice_parameter(sum, base);
        }
        let shift = RICE_SHIFT_STEP
            * RICE_SHIFT_THRESHOLDS
                .iter()
                .take_while(|&&threshold| sum >= threshold)
                .count() as u32;
        let sum = sum >> shift;
        let clipped = if base == 0 {
            sum.min(MAX_TEMPLATE_SUM)
        } else {
            sum.saturating_sub(base).min(MAX_TEMPLATE_SUM)
        };
        // `clipped` is below 5 · 7, the version 1 table's first step.
        crate::coeff::rice_parameter(clipped, 0) + shift
    }

    /// `StatCoeff` after the block's first nonzero remainder `rem`
    /// (§9.3.3.11: with `abs_remainder`, `regular`, two more).
    fn update(&mut self, rem: u32, regular: bool) {
        if rem == 0 || !self.pending {
            return;
        }
        if let Some(stat) = &mut self.statistic {
            *stat = (*stat + rem.ilog2() + if regular { 2 } else { 0 }) >> 1;
            self.pending = false;
        }
    }
}

impl<const EXT: bool> Drop for Walk<'_, EXT> {
    fn drop(&mut self) {
        self.grid[..self.max_pad].fill(0);
        self.pass1[..self.max_pad].fill(0);
    }
}
impl<const EXT: bool> Walk<'_, EXT> {
    #[inline(always)]
    #[allow(clippy::inline_always, reason = "per-coefficient state transition")]
    fn advance(&mut self, parity: u32) {
        self.state = if self.dependent {
            (STATE_TRANSITIONS >> (4 * self.state + 2 * parity)) & 3
        } else {
            0
        };
    }

    /// The Rice parameter at padded position `pad_pos` (raster `pos`) above
    /// `base`: version 1's from the padded grid alone, else the extension's.
    #[inline(always)]
    #[allow(clippy::inline_always, reason = "per-coefficient Golomb-Rice lookup")]
    fn rice_parameter(&self, pad_pos: u16, pos: u16, base: u32) -> u32 {
        if EXT && self.rice.general {
            self.rice
                .parameter(grid_sum(self.grid, self.pad_stride, pad_pos), pos, base)
        } else {
            grid_rice(self.grid, self.pad_stride, pad_pos, base)
        }
    }

    fn group(
        &mut self,
        r: &mut Reader<'_>,
        models: &mut Models,
        scan: &Scan,
        group: usize,
        last: usize,
    ) -> Result<()> {
        let first = group * scan.group_size;
        let end = (first + scan.group_size - 1).min(last);
        let mut nonzero = [ActiveCoeff::default(); MAX_GROUP_SIZE];
        let mut nonzero_len = 0usize;
        let mut regular = [(0u16, 0u16); MAX_GROUP_SIZE];
        let mut regular_len = 0usize;
        let mut bypass = [0u16; MAX_GROUP_SIZE];
        let mut bypass_len = 0usize;
        for index in (first..=end).rev() {
            let entry = scan.entries[index];
            if self.budget < MIN_REGULAR_BUDGET {
                bypass[bypass_len] = index as u16;
                bypass_len += 1;
                continue;
            }
            let (sig, mut level) = grid_context(self.pass1, self.pad_stride, entry, self.channel);
            let inferred = index == last || (index == first && group != 0 && nonzero_len == 0);
            let present = if inferred {
                true
            } else {
                self.budget -= 1;
                r.bin(models.significance(self.channel, self.state, sig))?
            };
            let mut val = 0u32;
            if present {
                if index == last {
                    level = 0;
                }
                let dep_adj = (self.state >> 1) as u8;
                self.budget -= 1;
                val = if r.bin(&mut models.common.gt1[self.channel][level])? {
                    self.budget -= 2;
                    let parity = r.bin(&mut models.common.parity[self.channel][level])?;
                    let greater = r.bin(&mut models.common.gt2[self.channel][level])?;
                    2 + u32::from(parity) + 2 * u32::from(greater)
                } else {
                    1
                };
                let p = usize::from(entry.pad_pos);
                self.grid[p] = val;
                self.pass1[p] = (val.min(4 + (val & 1)) as u8) | 0x20;
                self.max_pad = self.max_pad.max(p + 1);
                if val >= FIRST_RICE_BASE {
                    regular[regular_len] = (entry.pad_pos, entry.pos);
                    regular_len += 1;
                }
                nonzero[nonzero_len] = ActiveCoeff {
                    scan_idx: index as u16,
                    pos: entry.pos,
                    pad_pos: entry.pad_pos,
                    dep_adj,
                    outside_16: entry.outside_16,
                };
                nonzero_len += 1;
            }
            self.advance(val & 1);
        }
        let (log2_range, base_level) = if EXT {
            (self.rice.context.log2_range, self.rice.context.base_level)
        } else {
            (BASE_LOG2_RANGE, FIRST_RICE_BASE)
        };
        for &(pad_pos, pos) in &regular[..regular_len] {
            let parameter = self.rice_parameter(pad_pos, pos, base_level);
            let rem = remainder::<EXT>(r, parameter, log2_range)?;
            if EXT {
                self.rice.update(rem, true);
            }
            self.grid[usize::from(pad_pos)] += 2 * rem;
        }
        for &scan_idx in &bypass[..bypass_len] {
            let entry = scan.entries[usize::from(scan_idx)];
            let parameter = self.rice_parameter(entry.pad_pos, entry.pos, 0);
            let zero = (if self.state < 2 { 1 } else { 2 }) << parameter;
            let rem = remainder::<EXT>(r, parameter, log2_range)?;
            if EXT {
                self.rice.update(rem, false);
            }
            let val = match rem.cmp(&zero) {
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Less => rem + 1,
                std::cmp::Ordering::Greater => rem,
            };
            if val != 0 {
                let p = usize::from(entry.pad_pos);
                self.grid[p] = val;
                self.pass1[p] = (val.min(4 + (val & 1)) as u8) | 0x20;
                self.max_pad = self.max_pad.max(p + 1);
                nonzero[nonzero_len] = ActiveCoeff {
                    scan_idx,
                    pos: entry.pos,
                    pad_pos: entry.pad_pos,
                    dep_adj: (self.state >> 1) as u8,
                    outside_16: entry.outside_16,
                };
                nonzero_len += 1;
            }
            self.advance(val & 1);
        }
        self.finish_group_signs(r, &nonzero[..nonzero_len])
    }

    fn finish_group_signs(&mut self, r: &mut Reader<'_>, active: &[ActiveCoeff]) -> Result<()> {
        if active.is_empty() {
            return Ok(());
        }
        let hidden = self.hiding
            && !self.dependent
            && usize::from(active[0].scan_idx - active[active.len() - 1].scan_idx)
                > crate::coeff::SIGN_HIDING_THRESHOLD;
        let sign_count = active.len() - usize::from(hidden);
        let signs = r.bypass_bits(sign_count as u32)?;
        let mut sum_parity = 0u32;
        if hidden {
            for item in active {
                sum_parity ^= self.grid[usize::from(item.pad_pos)];
            }
        }
        for (i, item) in active.iter().enumerate() {
            let v = self.grid[usize::from(item.pad_pos)] as i32;
            let neg = if hidden && i + 1 == active.len() {
                sum_parity & 1 != 0
            } else {
                (signs >> (sign_count - 1 - i)) & 1 != 0
            };
            let mut out = if self.dependent {
                2 * v - i32::from(item.dep_adj)
            } else {
                v
            };
            if neg {
                out = -out;
            }
            let range = if EXT {
                self.rice.context.coefficient_range()
            } else {
                RiceContext::PLAIN.coefficient_range()
            };
            if !range.contains(&out) {
                return Err(DecodeError::Malformed("VVC coefficient dynamic range"));
            }
            match (EXT, self.wide.as_deref_mut()) {
                (true, Some(wide)) => wide.push((item.pos, out)),
                // The range check above kept `out` within 16 bits.
                _ => self.narrow.push((item.pos, out as i16)),
            }
            self.stats.outside_16 |= item.outside_16;
            self.stats.last_scan = Some(match self.stats.last_scan {
                Some(prev) => prev.max(item.scan_idx),
                None => item.scan_idx,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacWriter;

    #[test]
    fn luma_scans_match_the_encoder_tables() {
        let narrow = [(1, 16), (1, 32), (2, 8), (2, 16), (2, 32)];
        for (width, height) in narrow.into_iter().chain(narrow.map(|(w, h)| (h, w))) {
            let scan = Scan::new(width, height);
            assert_eq!(scan.positions, crate::scan::rect_positions(width, height));
            assert_eq!(scan.groups, crate::scan::rect_groups(width, height));
        }
    }

    #[test]
    fn rectangular_scans_cover_the_effective_transform() {
        for width in [2, 4, 8, 16, 32, 64] {
            for height in [2, 4, 8, 16, 32, 64] {
                let scan = Scan::new(width, height);
                let mut positions = scan.positions.clone();
                positions.sort_unstable();
                let expected: Vec<_> = (0..height.min(32))
                    .flat_map(|y| (0..width.min(32)).map(move |x| y * width + x))
                    .collect();
                assert_eq!(positions, expected, "{width}x{height}");
                assert_eq!(scan.group_size * scan.groups.len(), positions.len());
            }
        }
    }
    #[test]
    fn general_reader_matches_shared_square_syntax() {
        for side in [4, 8, 16, 32] {
            for inter in [false, true] {
                for channel in 0..2 {
                    for pattern in 0..4 {
                        let mut source: Vec<i32> = (0..side * side)
                            .map(|i| match pattern {
                                0 if i == side * side - 1 => -32767,
                                1 if i % 17 == 0 => 32767,
                                2 => i as i32 % 11 - 5,
                                3 => {
                                    if i % 2 == 0 {
                                        -32767
                                    } else {
                                        32767
                                    }
                                }
                                _ => 0,
                            })
                            .collect();
                        // Give the shared writer legal hidden signs, without
                        // consulting the new decoder's scan or state machinery.
                        for group in crate::scan::positions(side).chunks(16) {
                            let first = group.iter().position(|&p| source[p] != 0);
                            let last = group.iter().rposition(|&p| source[p] != 0);
                            if let (Some(first), Some(last)) = (first, last) {
                                let sum: i32 = group.iter().map(|&p| source[p].abs()).sum();
                                if last - first > crate::coeff::SIGN_HIDING_THRESHOLD {
                                    source[group[first]] = source[group[first]].abs()
                                        * if sum % 2 == 1 { -1 } else { 1 };
                                }
                            }
                        }
                        let mut writer = CabacWriter::new();
                        crate::coeff::emit(
                            &mut writer,
                            &mut CoeffContexts::new(37, inter),
                            &nc_dsp::Dsp::new(),
                            &source,
                            side,
                            channel,
                        );
                        writer.encode_terminate(true);
                        let mut bits = writer.finish();
                        bits.trailing_bits();
                        let bytes = bits.into_bytes();
                        let mut reader = Reader::new(&bytes).unwrap();
                        // Deliberately test `read_general` directly.
                        let (actual, _) = read_general(
                            &mut reader,
                            &mut Models::new(37, if inter { 1 } else { 2 }),
                            side,
                            side,
                            channel,
                            false,
                            true,
                            false,
                        )
                        .unwrap();
                        assert_eq!(
                            actual, source,
                            "side={side} inter={inter} channel={channel} pattern={pattern}"
                        );
                        assert!(reader.terminate().unwrap());
                    }
                }
            }
        }
    }
}
