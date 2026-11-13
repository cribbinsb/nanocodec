//! AV2 CDEF, the constrained directional enhancement filter (AVM v1.0.0
//! `av2/common/cdef.c`, `av2/common/cdef_block.c`): the frame-header
//! parameters and their syntax (`setup_cdef` in `av2/decoder/decodeframe.c`,
//! `encode_cdef` in `av2/encoder/bitstream.c`), the picture filter shared by
//! the encoder reconstruction and our decoder, and the encoder's choice of
//! strengths.
//!
//! The filter runs after deblocking, on the whole picture, before it becomes
//! the reference. For 8-bit video AVM's block filter equals AV1's (same
//! directions, taps, `constrain`, damping, rounding and `adjust_strength`;
//! AVM clamps to the taps' range only when both strengths are on, which
//! cannot change a result), so the kernels are nc-dsp's `cdef_find_dir` and
//! `cdef_filter`, reading a padded copy of the unfiltered plane. AVM also
//! reads only unfiltered samples (its line and column buffers save them
//! before each 64×64 filter block changes them) and marks samples outside
//! the picture `CDEF_VERY_LARGE`, which the padding reproduces.
//!
//! A frame codes one to eight strength sets (`nb_cdef_strengths`); with two
//! or more, each 64×64 filter block (AVM's CDEF unit) codes which set it
//! uses, at its first non-skipped block, after that block's skip flag
//! (`read_cdef` in `av2/decoder/decodemv.c`, [`StrengthIndices`]). The sequence disables CDEF on
//! skipped transforms: an 8×8 block whose four 4×4 units all skipped their
//! luma transforms is left unfiltered (`av2_cdef_compute_sb_list`,
//! `is_8x8_block_skip`), and a filter block with no non-skipped block codes
//! no index and is not filtered (`cdef_strength = -1`). Lossless frames have no CDEF
//! syntax.
//!
//! The encoder chooses the strengths after the frame's reconstruction, but
//! the indices sit inside the block syntax: the block coder defers them in a
//! recording entropy writer and codes them when it finishes the tile
//! ([`crate::entropy::Writer::replay`]).
//!
//! Strength choice ([`search_and_filter`]): every candidate luma and chroma
//! strength around SVT-AV1's quantiser model
//! ([`nc_encoder::cdef_strengths_from_q`]) is measured per filter block on
//! sampled rows of 8×8 units, and [`nc_encoder::select_strength_sets`]
//! picks the frame's sets and each filter block's index by distortion and
//! index bits. The
//! fallback ([`choose_and_filter`]) codes the model's single set, each plane
//! kept only if it lowers the sampled error.

use nc_common::decode::{BitReader, DecodeError, Result};
use nc_common::{BitWriter, Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, CdefBlock, CdefLines, CdefSets, CdefSource, CdefStrengths, CdefUnit, Dsp};

use crate::block_state::BlockState;
use crate::cdf_models::TileCdfs;
use crate::entropy::{Cdf, Reader, Writer};
use crate::partition::SUPERBLOCK;

/// Luma side of the unit CDEF filters (`CDEF_BLOCKSIZE` / `CDEF_NBLOCKS`).
const UNIT: usize = 8;
/// Smallest damping (`cdef_damping_minus_3` codes the excess).
const MIN_DAMPING: u8 = 3;
/// Largest damping.
const MAX_DAMPING: u8 = MIN_DAMPING + (1 << DAMPING_BITS) - 1;
/// Width of `cdef_damping_minus_3`.
const DAMPING_BITS: u32 = 2;
/// Width of `nb_cdef_strengths - 1`.
const STRENGTH_COUNT_BITS: u32 = 3;
/// Secondary strengths per primary strength (`CDEF_SEC_STRENGTHS`): a coded
/// strength is `primary * 4 + secondary`.
const SECONDARY_STRENGTHS: u8 = 4;
/// Largest primary strength (`CDEF_PRI_STRENGTHS - 1`).
const MAX_PRIMARY: u8 = 15;
/// Largest coded secondary strength (3 means 4).
const MAX_CODED_SECONDARY: u8 = SECONDARY_STRENGTHS - 1;
/// Width of a full coded strength (`CDEF_STRENGTH_BITS`).
const STRENGTH_BITS: u32 = 6;
/// Coded strengths below this value use the short form (`less_4`).
const SHORT_STRENGTH_LIMIT: u32 = 4;
/// Width of a short-form coded strength.
const SHORT_STRENGTH_BITS: u32 = 2;
/// Most strength sets a frame codes (`nb_cdef_strengths`, 3 bits plus one).
pub(crate) const MAX_SETS: usize = 1 << STRENGTH_COUNT_BITS;
/// Side of a filter block (AVM's CDEF unit), the area one strength index
/// covers: 64×64 regardless of the superblock size
/// (`MI_IN_CDEF_LINEAR_LOG2`).
const FILTER_BLOCK: usize = 64;
// Our superblocks are filter blocks, which `StrengthIndices::context`
// relies on.
const _: () = assert!(SUPERBLOCK == FILTER_BLOCK);

/// One plane type's strength: primary 0..=15, secondary as coded 0..=3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Strength {
    /// Primary (directional) strength.
    pub primary: u8,
    /// Secondary strength as coded; 3 means 4.
    pub secondary: u8,
}

impl Strength {
    /// No filtering.
    pub const OFF: Self = Self {
        primary: 0,
        secondary: 0,
    };

    fn is_off(self) -> bool {
        self == Self::OFF
    }

    fn is_valid(self) -> bool {
        self.primary <= MAX_PRIMARY && self.secondary <= MAX_CODED_SECONDARY
    }

    /// The secondary strength the filter applies (`sec_strength += sec == 3`).
    fn secondary_value(self) -> u8 {
        self.secondary + u8::from(self.secondary == MAX_CODED_SECONDARY)
    }

    fn coded(self) -> u32 {
        u32::from(self.primary * SECONDARY_STRENGTHS + self.secondary)
    }

    fn from_coded(value: u32) -> Self {
        let value = value as u8;
        Self {
            primary: value / SECONDARY_STRENGTHS,
            secondary: value % SECONDARY_STRENGTHS,
        }
    }

    /// `encode_cdef`: a `less_4` flag, then two or six bits.
    fn write(self, bits: &mut BitWriter) {
        let coded = self.coded();
        let short = coded < SHORT_STRENGTH_LIMIT;
        bits.put_bit(short);
        bits.put_bits(
            coded,
            if short {
                SHORT_STRENGTH_BITS
            } else {
                STRENGTH_BITS
            },
        );
    }

    fn read(bits: &mut BitReader<'_>) -> Result<Self> {
        let width = if bits.read_bit()? {
            SHORT_STRENGTH_BITS
        } else {
            STRENGTH_BITS
        };
        Ok(Self::from_coded(bits.read_bits(width)?))
    }
}

/// One strength set: a luma and a chroma strength (`cdef_strengths[i]`,
/// `cdef_uv_strengths[i]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StrengthSet {
    /// Luma strength.
    pub luma: Strength,
    /// Strength of both chroma planes.
    pub chroma: Strength,
}

impl StrengthSet {
    /// No filtering in either plane type.
    pub const OFF: Self = Self {
        luma: Strength::OFF,
        chroma: Strength::OFF,
    };

    fn is_off(self) -> bool {
        self.luma.is_off() && self.chroma.is_off()
    }

    fn plane(self, plane: PlaneIndex) -> Strength {
        if plane == PlaneIndex::Y {
            self.luma
        } else {
            self.chroma
        }
    }
}

/// The frame's CDEF parameters: off (`cdef_frame_enable = 0`) or one to
/// [`MAX_SETS`] strength sets with their damping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CdefParams {
    enabled: bool,
    damping: u8,
    count: u8,
    sets: [StrengthSet; MAX_SETS],
}

impl CdefParams {
    /// No CDEF in this frame.
    pub const OFF: Self = Self {
        enabled: false,
        damping: MIN_DAMPING,
        count: 1,
        sets: [StrengthSet::OFF; MAX_SETS],
    };

    /// One strength set with luma damping `damping` (3..=6; chroma uses one
    /// less).
    ///
    /// # Errors
    ///
    /// Rejects a damping or strength outside its syntax range.
    pub fn new(damping: u8, luma: Strength, chroma: Strength) -> Result<Self> {
        Self::with_sets(damping, &[StrengthSet { luma, chroma }])
    }

    /// One to [`MAX_SETS`] strength sets with luma damping `damping`.
    ///
    /// # Errors
    ///
    /// Rejects a damping, set count or strength outside its syntax range.
    pub fn with_sets(damping: u8, sets: &[StrengthSet]) -> Result<Self> {
        if !(MIN_DAMPING..=MAX_DAMPING).contains(&damping)
            || !(1..=MAX_SETS).contains(&sets.len())
            || sets
                .iter()
                .any(|set| !set.luma.is_valid() || !set.chroma.is_valid())
        {
            return Err(DecodeError::Malformed("AV2 CDEF parameter out of range"));
        }
        let mut params = Self {
            enabled: true,
            damping,
            count: sets.len() as u8,
            ..Self::OFF
        };
        params.sets[..sets.len()].copy_from_slice(sets);
        Ok(params)
    }

    /// Whether the frame codes CDEF parameters (`cdef_frame_enable`).
    #[must_use]
    pub fn is_on(self) -> bool {
        self.enabled
    }

    /// The strength sets the frame codes (none when off).
    #[must_use]
    pub fn sets(&self) -> &[StrengthSet] {
        if self.enabled {
            &self.sets[..usize::from(self.count)]
        } else {
            &[]
        }
    }

    /// Whether blocks code a strength index (two or more sets).
    #[must_use]
    pub fn codes_indices(self) -> bool {
        self.sets().len() > 1
    }

    /// Whether filtering can change any sample.
    fn filters(self) -> bool {
        self.sets().iter().any(|set| !set.is_off())
    }

    /// Whether any set filters `plane`.
    fn filters_plane(self, plane: PlaneIndex) -> bool {
        self.sets().iter().any(|set| !set.plane(plane).is_off())
    }

    /// Write `setup_cdef`'s syntax. A single-picture sequence infers
    /// `cdef_frame_enable = 1`, so an off frame codes zero strengths there;
    /// it also infers adaptive CDEF on skipped transforms, coded off.
    pub fn write(self, bits: &mut BitWriter, single_picture: bool) {
        let coded = if single_picture {
            Self {
                enabled: true,
                ..self
            }
        } else {
            bits.put_bit(self.enabled);
            self
        };
        if !coded.enabled {
            return;
        }
        bits.put_bits(u32::from(coded.damping - MIN_DAMPING), DAMPING_BITS);
        bits.put_bits(u32::from(coded.count - 1), STRENGTH_COUNT_BITS);
        if single_picture {
            bits.put_bit(false); // cdef_on_skip_txfm_frame_enable
        }
        for set in coded.sets() {
            set.luma.write(bits);
            set.chroma.write(bits);
        }
    }

    /// Read `setup_cdef`'s syntax (see [`Self::write`]).
    ///
    /// # Errors
    ///
    /// Rejects truncation and syntax outside the subset (CDEF on skipped
    /// transforms).
    pub fn read(bits: &mut BitReader<'_>, single_picture: bool) -> Result<Self> {
        if !single_picture && !bits.read_bit()? {
            return Ok(Self::OFF);
        }
        let damping = bits.read_bits(DAMPING_BITS)? as u8 + MIN_DAMPING;
        let count = bits.read_bits(STRENGTH_COUNT_BITS)? as usize + 1;
        if single_picture && bits.read_bit()? {
            return Err(DecodeError::Unsupported("AV2 CDEF on skipped transforms"));
        }
        let mut sets = [StrengthSet::OFF; MAX_SETS];
        for set in &mut sets[..count] {
            set.luma = Strength::read(bits)?;
            set.chroma = Strength::read(bits)?;
        }
        let params = Self::with_sets(damping, &sets[..count])?;
        // A single picture's one zero set is how it codes "off".
        Ok(
            if single_picture && !params.filters() && damping == MIN_DAMPING && count == 1 {
                Self::OFF
            } else {
                params
            },
        )
    }
}

/// `av2_get_cdef_context` when the left filter block uses set 0.
const LEFT_INDEX0_CONTEXT: usize = 2;
/// `av2_get_cdef_context` otherwise (no left block, or another set or none).
const OTHER_CONTEXT: usize = 0;

/// The strength set index of every 64×64 filter block (`cdef_strength` of
/// the block's first coding block): `None` until a non-skipped coding block
/// codes it, and for good when none does (AVM's -1: not filtered).
#[derive(Debug, Clone)]
pub(crate) struct StrengthIndices {
    columns: usize,
    indices: Vec<Option<u8>>,
}

impl StrengthIndices {
    /// No index coded yet in a `width` x `height` picture.
    pub(crate) fn new(width: usize, height: usize) -> Self {
        let columns = width.div_ceil(FILTER_BLOCK);
        Self {
            columns,
            indices: vec![None; columns * height.div_ceil(FILTER_BLOCK)],
        }
    }

    fn block(&self, x: usize, y: usize) -> usize {
        y / FILTER_BLOCK * self.columns + x / FILTER_BLOCK
    }

    /// The index of the filter block holding luma sample (`x`, `y`).
    pub(crate) fn get(&self, x: usize, y: usize) -> Option<u8> {
        self.indices[self.block(x, y)]
    }

    /// Set the index of the filter block holding luma sample (`x`, `y`).
    pub(crate) fn set(&mut self, x: usize, y: usize, index: Option<u8>) {
        let block = self.block(x, y);
        self.indices[block] = index;
    }

    /// Whether the coding block at (`x`, `y`) codes its filter block's
    /// index: it is the first non-skipped one (`read_cdef`, with CDEF on
    /// skipped transforms disabled).
    pub(crate) fn codes_index(&self, x: usize, y: usize, skip: bool) -> bool {
        !skip && self.get(x, y).is_none()
    }

    /// Top-left luma samples of the filter blocks that have an index, in
    /// coding (raster) order.
    #[cfg(test)]
    fn coded_positions(&self) -> Vec<(usize, usize)> {
        self.indices
            .iter()
            .enumerate()
            .filter(|(_, index)| index.is_some())
            .map(|(block, _)| {
                (
                    block % self.columns * FILTER_BLOCK,
                    block / self.columns * FILTER_BLOCK,
                )
            })
            .collect()
    }

    /// `av2_get_cdef_context` for the filter block at (`x`, `y`). The above
    /// neighbour counts only inside the same superblock; ours are filter
    /// blocks, so it never does, and the left block decides alone.
    fn context(&self, x: usize, y: usize) -> usize {
        if x >= FILTER_BLOCK && self.get(x - FILTER_BLOCK, y) == Some(0) {
            LEFT_INDEX0_CONTEXT
        } else {
            OTHER_CONTEXT
        }
    }
}

/// Code `symbol` with the model `select` picks (fixed at its default from
/// `defaults` when the tile does not adapt).
fn write_symbol<const N: usize>(
    writer: &mut Writer,
    symbol: usize,
    defaults: &mut TileCdfs,
    select: impl Fn(&mut TileCdfs) -> &mut Cdf<N> + Copy,
) -> Result<()> {
    let fixed = *select(defaults).inverse();
    writer.model(symbol, &fixed, select)
}

/// Read a symbol with the model `select` picks (see [`write_symbol`]).
fn read_symbol<const N: usize>(
    reader: &mut Reader<'_>,
    defaults: &mut TileCdfs,
    select: impl Fn(&mut TileCdfs) -> &mut Cdf<N> + Copy,
) -> Result<usize> {
    let fixed = *select(defaults).inverse();
    reader.model(&fixed, select)
}

/// Write the strength index of the filter block at (`x`, `y`) (AVM
/// `write_cdef`): a "set 0" flag, then for three or more sets the index
/// less one from the model of that set count. `defaults` supplies fixed
/// probabilities when the tile does not adapt.
/// # Errors
/// Propagates entropy-coder errors.
pub(crate) fn write_index(
    writer: &mut Writer,
    params: CdefParams,
    indices: &StrengthIndices,
    (x, y): (usize, usize),
    defaults: &mut TileCdfs,
) -> Result<()> {
    let count = params.sets().len();
    if count < 2 {
        return Ok(());
    }
    let index = usize::from(
        indices
            .get(x, y)
            .expect("an index for every coded position"),
    );
    let context = indices.context(x, y);
    write_symbol(writer, usize::from(index == 0), defaults, |m| {
        &mut m.cdef_index0[context]
    })?;
    if index == 0 || count == 2 {
        return Ok(());
    }
    let symbol = index - 1;
    match count {
        3 => write_symbol(writer, symbol, defaults, |m| &mut m.cdef_index3),
        4 => write_symbol(writer, symbol, defaults, |m| &mut m.cdef_index4),
        5 => write_symbol(writer, symbol, defaults, |m| &mut m.cdef_index5),
        6 => write_symbol(writer, symbol, defaults, |m| &mut m.cdef_index6),
        7 => write_symbol(writer, symbol, defaults, |m| &mut m.cdef_index7),
        _ => write_symbol(writer, symbol, defaults, |m| &mut m.cdef_index8),
    }
}

/// `read_cdef` for the coding block at (`x`, `y`): when it is its filter
/// block's first non-skipped block, read the index (0 without syntax for a
/// single set) into `indices`. `defaults` (needed only when the frame codes
/// indices) supplies fixed probabilities when the tile does not adapt.
/// # Errors
/// Rejects truncated or malformed data.
pub(crate) fn read_index(
    reader: &mut Reader<'_>,
    params: CdefParams,
    indices: &mut StrengthIndices,
    (x, y): (usize, usize),
    skip: bool,
    defaults: Option<&mut TileCdfs>,
) -> Result<()> {
    if !params.is_on() || !indices.codes_index(x, y, skip) {
        return Ok(());
    }
    let count = params.sets().len();
    let index = match defaults {
        Some(defaults) if count > 1 => {
            let context = indices.context(x, y);
            let index0 = read_symbol(reader, defaults, |m| &mut m.cdef_index0[context])? == 1;
            if index0 {
                0
            } else if count == 2 {
                1
            } else {
                1 + match count {
                    3 => read_symbol(reader, defaults, |m| &mut m.cdef_index3)?,
                    4 => read_symbol(reader, defaults, |m| &mut m.cdef_index4)?,
                    5 => read_symbol(reader, defaults, |m| &mut m.cdef_index5)?,
                    6 => read_symbol(reader, defaults, |m| &mut m.cdef_index6)?,
                    7 => read_symbol(reader, defaults, |m| &mut m.cdef_index7)?,
                    _ => read_symbol(reader, defaults, |m| &mut m.cdef_index8)?,
                }
            }
        }
        _ if count > 1 => panic!("index models are needed when the frame codes indices"),
        _ => 0,
    };
    indices.set(x, y, Some(index as u8));
    Ok(())
}

/// An 8×8 luma unit CDEF filters, at luma sample (`x`, `y`), with its
/// direction, directional contrast (`var`) and its filter block's strength
/// set.
#[derive(Debug, Clone, Copy)]
struct Unit {
    x: usize,
    y: usize,
    dir: u8,
    var: i32,
    set: u8,
}

/// The units of a picture CDEF filters, in raster order: every 8×8 block
/// that is not all skipped (`av2_cdef_compute_sb_list`) in a filter block
/// with an index. Directions come from the unfiltered luma when
/// `with_directions`, otherwise they are 0, which is what a zero primary
/// strength uses.
fn filtered_units(
    dsp: &Dsp,
    luma: &PlaneView<'_>,
    state: &BlockState,
    indices: &StrengthIndices,
    with_directions: bool,
) -> Vec<Unit> {
    let mut units = Vec::new();
    for y in (0..luma.height()).step_by(UNIT) {
        for x in (0..luma.width()).step_by(UNIT) {
            let Some(set) = indices.get(x, y) else {
                continue;
            };
            if state.skips_transform(x, y) {
                continue;
            }
            let (dir, var) = if with_directions {
                dsp.cdef_find_dir(luma, (x, y))
            } else {
                (0, 0)
            };
            units.push(Unit {
                x,
                y,
                dir,
                var,
                set,
            });
        }
    }
    units
}

/// The reusable buffers of [`filter_picture`]: nc-dsp's staged band state
/// and one band's unit list.
#[derive(Debug, Default)]
pub(crate) struct PictureFilterScratch {
    lines: CdefLines,
    units: Vec<CdefUnit>,
}

/// Filter the finished (deblocked) picture in place with `params`, using
/// the block skip flags recorded in `state` and the filter blocks' strength
/// sets in `indices`: nc-dsp's shared band driver (`Dsp::cdef_filter_band`,
/// the AV1 decoder's), one row of filter blocks at a time, over the units
/// of [`filtered_units`] (the same selection, directions found by the
/// driver).
pub(crate) fn filter_picture(
    picture: &mut Picture,
    state: &BlockState,
    indices: &StrengthIndices,
    params: CdefParams,
    dsp: &Dsp,
    scratch: &mut PictureFilterScratch,
) {
    if !params.filters() {
        return;
    }
    let strengths: Vec<CdefStrengths> = params
        .sets()
        .iter()
        .map(|set| CdefStrengths {
            luma_primary: set.luma.primary,
            luma_secondary: set.luma.secondary_value(),
            chroma_primary: set.chroma.primary,
            chroma_secondary: set.chroma.secondary_value(),
        })
        .collect();
    let sets = CdefSets::new(&strengths, params.damping);
    let PictureFilterScratch { lines, units } = scratch;
    lines.start(picture);
    let (width, height) = (picture.width(), picture.height());
    for top in (0..height).step_by(FILTER_BLOCK) {
        let rows = top..(top + FILTER_BLOCK).min(height);
        units.clear();
        for y in rows.clone().step_by(UNIT) {
            for x in (0..width).step_by(UNIT) {
                if let Some(set) = indices.get(x, y)
                    && !state.skips_transform(x, y)
                {
                    units.push(CdefUnit { x, y, set });
                }
            }
        }
        dsp.cdef_filter_band(lines, picture, rows, units, &sets);
    }
}

/// [`apply`] after a search, from the unfiltered planes it copied.
fn apply_searched(
    picture: &mut Picture,
    units: &[Unit],
    params: CdefParams,
    dsp: &Dsp,
    unfiltered: &mut UnfilteredPlanes,
) {
    for plane in PlaneIndex::ALL {
        if params.filters_plane(plane) {
            let input = unfiltered.plane(picture, plane);
            filter_plane(picture, plane, input, units, params, dsp);
        }
    }
}

/// Filter `plane` of `picture` in place from its unfiltered copy `input`.
fn filter_plane(
    picture: &mut Picture,
    plane: PlaneIndex,
    input: &CdefSource,
    units: &[Unit],
    params: CdefParams,
    dsp: &Dsp,
) {
    filter_units(
        dsp,
        input,
        &mut picture.plane_mut(plane).view_mut(),
        units,
        params.damping,
        plane,
        |unit| params.sets[usize::from(unit.set)].plane(plane),
    );
}

/// `av2_cdef_filter_fb` for one plane: filter every listed unit, with the
/// strength `strength` gives it, from the padded unfiltered `input` into
/// `output`.
///
/// The units go to the kernels in one [`Dsp::cdef_filter_blocks`] call per
/// secondary strength (the call shares it between its blocks), not one
/// call per unit: the per-call checks and strength set-up were twice the
/// work of AV1's per-plane calls (x86 `cdef_filter_blocks` self cost 142M
/// against 46M instructions, crowd 720p CBR 600, 31 frames). Every unit
/// reads only the unfiltered `input` and writes its own block of `output`,
/// so the order of the calls cannot change the result.
fn filter_units(
    dsp: &Dsp,
    input: &CdefSource,
    output: &mut PlaneViewMut<'_>,
    units: &[Unit],
    damping: u8,
    plane: PlaneIndex,
    strength: impl Fn(&Unit) -> Strength,
) {
    let chroma = plane != PlaneIndex::Y;
    let shift = usize::from(chroma);
    let size = UNIT >> shift;
    let damping = damping - u8::from(chroma);
    // The blocks of each coded secondary strength.
    let mut by_secondary: [Vec<CdefBlock>; SECONDARY_STRENGTHS as usize] = Default::default();
    for unit in units {
        let strength = strength(unit);
        let primary = if chroma {
            strength.primary
        } else {
            adjust_strength(strength.primary, unit.var)
        };
        if primary == 0 && strength.secondary == 0 {
            continue; // the output already holds the unfiltered samples
        }
        by_secondary[usize::from(strength.secondary)].push(CdefBlock {
            x: unit.x >> shift,
            y: unit.y >> shift,
            pri: primary,
            dir: if strength.primary == 0 { 0 } else { unit.dir },
        });
    }
    for (secondary, blocks) in (0..SECONDARY_STRENGTHS).zip(&by_secondary) {
        if !blocks.is_empty() {
            let secondary = Strength {
                primary: 0,
                secondary,
            }
            .secondary_value();
            dsp.cdef_filter_blocks(size, input, blocks, secondary, damping, output);
        }
    }
}

/// AVM `adjust_strength`: scale the luma primary strength by the block's
/// directional contrast.
fn adjust_strength(strength: u8, var: i32) -> u8 {
    /// Contrast below `1 << VAR_SHIFT` counts as the lowest class.
    const VAR_SHIFT: u32 = 6;
    /// Highest contrast class (`AVMMIN(get_msb(var >> 6), 12)`).
    const MAX_CLASS: i32 = 12;
    /// Strength scale of the lowest class, in sixteenths.
    const BASE_SCALE: i32 = 4;
    if var == 0 {
        return 0;
    }
    let scaled = var >> VAR_SHIFT;
    let class = if scaled == 0 {
        0
    } else {
        (scaled.ilog2() as i32).min(MAX_CLASS)
    };
    ((i32::from(strength) * (BASE_SCALE + class) + 8) >> 4) as u8
}

/// AC steps (AV1 `ac_qlookup` scale) from which the damping grows by one:
/// AV1's `3 + (base_q_idx >> 6)` (SVT-AV1 real time) at qindex 64, 128, 192.
const DAMPING_STEPS: [f32; 3] = [72.0, 177.0, 552.0];

/// The quantiser model's strength set and damping for AC step `step` (AV1
/// `ac_qlookup` scale).
fn model_strengths(step: f32, intra: bool) -> (StrengthSet, u8) {
    let q = step.min(nc_encoder::CDEF_MODEL_MAX_Q);
    let (luma, chroma) = nc_encoder::cdef_strengths_from_q(q, intra);
    let set = StrengthSet {
        luma: Strength {
            primary: luma.0,
            secondary: luma.1,
        },
        chroma: Strength {
            primary: chroma.0,
            secondary: chroma.1,
        },
    };
    let damping = MIN_DAMPING + DAMPING_STEPS.iter().filter(|&&s| q >= s).count() as u8;
    (set, damping)
}

/// The units on every `period`-th row of units.
fn sampled_units(units: &[Unit], period: usize) -> Vec<Unit> {
    units
        .iter()
        .filter(|unit| (unit.y / UNIT).is_multiple_of(period))
        .copied()
        .collect()
}

/// The single-set fallback: the quantiser model's luma and chroma
/// strengths, each kept only where it lowers the squared error against
/// `source` on every `check_row_period`-th row of units; then filter
/// `picture` (deblocked, it becomes the reference) with them.
///
/// `step` is the frame's AC quantiser step on AV1's `ac_qlookup` scale.
/// Skipping the check on the other rows saves most of the trial filtering
/// (three quarters at the default period 4); whole rows of units keep the
/// sample representative of the picture's content.
pub(crate) fn choose_and_filter(
    source: &Picture,
    picture: &mut Picture,
    (state, indices): (&BlockState, &StrengthIndices),
    (step, intra, check_row_period): (f32, bool, usize),
    dsp: &Dsp,
    scratch: &mut CdefScratch,
) -> CdefParams {
    let (model, damping) = model_strengths(step, intra);
    scratch.unfiltered.start();
    if model.is_off() {
        return CdefParams::OFF;
    }
    let units = filtered_units(
        dsp,
        &picture.plane(PlaneIndex::Y).view(),
        state,
        indices,
        model.luma.primary != 0 || model.chroma.primary != 0,
    );
    let sampled = sampled_units(&units, check_row_period);
    let mut keep = |planes: &[PlaneIndex], strength: Strength| {
        if strength.is_off() {
            return false;
        }
        let errors = unit_errors(
            (source, &*picture),
            &sampled,
            damping,
            &[Strength::OFF, strength],
            planes,
            (1, |_: &Unit| 0),
            dsp,
            scratch,
        );
        errors[1] < errors[0]
    };
    let set = StrengthSet {
        luma: if keep(&[PlaneIndex::Y], model.luma) {
            model.luma
        } else {
            Strength::OFF
        },
        chroma: if keep(&[PlaneIndex::U, PlaneIndex::V], model.chroma) {
            model.chroma
        } else {
            Strength::OFF
        },
    };
    if set.is_off() {
        return CdefParams::OFF;
    }
    let params = CdefParams::with_sets(damping, &[set]).expect("strengths within syntax range");
    apply_searched(picture, &units, params, dsp, &mut scratch.unfiltered);
    params
}

/// Luma primary strengths every search tries, with each of
/// [`LUMA_SECONDARIES`], besides the quantiser model's set. A coarse grid
/// over the whole 0..=15 range: the model alone is nearly off at high
/// quality, where noise then builds up over P frames (see nc-av1's
/// `cdef_local_strong_luma`).
const LUMA_PRIMARIES: [u8; 5] = [0, 3, 6, 10, 15];
/// Coded luma secondary strengths every search tries.
const LUMA_SECONDARIES: [u8; 2] = [0, 2];
/// Chroma primary strengths every search tries (secondary off), besides the
/// model's.
const CHROMA_PRIMARIES: [u8; 4] = [0, 2, 5, 9];
/// Coded chroma secondary strengths every search tries.
const CHROMA_SECONDARIES: [u8; 1] = [0];
/// Header bits of one strength set, at most (two long-form strengths).
const SET_HEADER_BITS: f64 = 2.0 * (1 + STRENGTH_BITS) as f64;

/// The candidate strengths of one plane type: the `primaries` x
/// `secondaries` grid (off first, as both lists start at 0), then `model`
/// if the grid lacks it.
fn candidates(model: Strength, primaries: &[u8], secondaries: &[u8]) -> Vec<Strength> {
    let mut list: Vec<Strength> = primaries
        .iter()
        .flat_map(|&primary| {
            secondaries
                .iter()
                .map(move |&secondary| Strength { primary, secondary })
        })
        .collect();
    if !list.contains(&model) {
        list.push(model);
    }
    list
}

/// Approximate bits of one filter block's index with `count` sets: the
/// "set 0" flag, plus the index among the other sets for three or more.
fn index_bits(count: usize) -> f64 {
    match count {
        0 | 1 => 0.0,
        2 => 1.0,
        _ => 1.0 + ((count - 1) as f64).log2(),
    }
}

/// How [`search_and_filter`] searches.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchOptions {
    /// Most strength sets the frame may code (1..=[`MAX_SETS`]).
    pub max_sets: usize,
    /// Every `row_period`-th row of 8×8 units measures the candidates.
    pub row_period: usize,
}

/// The unfiltered planes of the picture being searched, each copied once
/// on first use: the search's trials and the final filtering read the same
/// deblocked samples (the search filters only into its trial buffer), so
/// the filtering need not copy them again (a 720p frame's planes are
/// ~1.4 MB; the second copy was ~0.3% of the encoder's Jetson cycles).
#[derive(Debug, Default)]
struct UnfilteredPlanes {
    planes: [CdefSource; PlaneIndex::ALL.len()],
    copied: [bool; PlaneIndex::ALL.len()],
}

impl UnfilteredPlanes {
    /// Forget the copies: a new picture is searched.
    fn start(&mut self) {
        self.copied = [false; PlaneIndex::ALL.len()];
    }

    /// The unfiltered `plane` of `picture`, which must not have been
    /// filtered since [`Self::start`].
    fn plane(&mut self, picture: &Picture, plane: PlaneIndex) -> &CdefSource {
        let index = plane as usize;
        if !self.copied[index] {
            self.planes[index].fill_from(&picture.plane(plane).view());
            self.copied[index] = true;
        }
        &self.planes[index]
    }
}

/// The CDEF search's reusable buffers, kept by the backend across frames.
#[derive(Debug, Default)]
pub(crate) struct CdefScratch {
    /// The unfiltered planes the trials and the filter read.
    unfiltered: UnfilteredPlanes,
    /// One candidate's trial filtering of a plane ([`unit_errors`]). Kept,
    /// and only grown, so a frame does not zero-fill a new plane-sized
    /// buffer per plane (~0.6% of x86 instructions in `memset`).
    trial: Vec<u8>,
    /// [`filter_picture`]'s buffers.
    picture: PictureFilterScratch,
}

impl CdefScratch {
    /// Buffers for [`filter_picture`] without a search; the search's
    /// copies are forgotten.
    pub(crate) fn unsearched_source(&mut self) -> &mut PictureFilterScratch {
        self.unfiltered.start();
        &mut self.picture
    }
}

/// The encoder's strength search: measure every luma and chroma candidate
/// (around the quantiser model, `step` on AV1's `ac_qlookup` scale) per
/// filter block on every `row_period`-th row of units, choose up to
/// `max_sets` sets and each filter block's index (stored in `indices`, whose
/// coded filter blocks hold placeholders) by distortion plus `lambda`
/// (squared error per bit) times the index and header bits, and filter `picture` with them.
///
/// Sampling rows of units scales the trial filtering down by `row_period`;
/// whole rows keep each filter block's sample spread over its content.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search_and_filter(
    source: &Picture,
    picture: &mut Picture,
    state: &BlockState,
    indices: &mut StrengthIndices,
    step: f32,
    intra: bool,
    options: SearchOptions,
    lambda: f64,
    dsp: &Dsp,
    scratch: &mut CdefScratch,
) -> CdefParams {
    let (model, damping) = model_strengths(step, intra);
    scratch.unfiltered.start();
    let luma = candidates(model.luma, &LUMA_PRIMARIES, &LUMA_SECONDARIES);
    let chroma = candidates(model.chroma, &CHROMA_PRIMARIES, &CHROMA_SECONDARIES);
    let mut units = filtered_units(
        dsp,
        &picture.plane(PlaneIndex::Y).view(),
        state,
        indices,
        true,
    );
    if units.is_empty() {
        return CdefParams::OFF;
    }
    // One error-table row per filter block with an index.
    let mut rows = vec![usize::MAX; indices.indices.len()];
    let mut coded = Vec::new();
    for (block, index) in indices.indices.iter().enumerate() {
        if index.is_some() {
            rows[block] = coded.len();
            coded.push(block);
        }
    }
    let row = |unit: &Unit| rows[indices.block(unit.x, unit.y)];
    let sampled = sampled_units(&units, options.row_period);
    let tables = [
        (&luma, &[PlaneIndex::Y][..]),
        (&chroma, &[PlaneIndex::U, PlaneIndex::V][..]),
    ]
    .map(|(candidates, planes)| {
        unit_errors(
            (source, &*picture),
            &sampled,
            damping,
            candidates,
            planes,
            (coded.len(), row),
            dsp,
            scratch,
        )
    });
    let [luma_errors, chroma_errors] = &tables;
    // Joint candidates: luma `l` with chroma `c` is `l * chroma.len() + c`.
    let pairs = luma.len() * chroma.len();
    let scale = options.row_period as u64;
    let mut errors = Vec::with_capacity(coded.len() * pairs);
    for block in 0..coded.len() {
        let luma_row = &luma_errors[block * luma.len()..(block + 1) * luma.len()];
        let chroma_row = &chroma_errors[block * chroma.len()..(block + 1) * chroma.len()];
        for &l in luma_row {
            errors.extend(chroma_row.iter().map(|&c| (l + c) * scale));
        }
    }
    let selection =
        nc_encoder::select_strength_sets(&errors, pairs, options.max_sets, lambda, |n| {
            n as f64 * SET_HEADER_BITS + index_bits(n) * coded.len() as f64
        });
    let sets: Vec<StrengthSet> = selection
        .sets
        .iter()
        .map(|&pair| StrengthSet {
            luma: luma[pair / chroma.len()],
            chroma: chroma[pair % chroma.len()],
        })
        .collect();
    if sets.iter().all(|set| set.is_off()) {
        return CdefParams::OFF;
    }
    for (&block, &choice) in coded.iter().zip(&selection.choices) {
        indices.indices[block] = Some(choice as u8);
    }
    for unit in &mut units {
        unit.set = indices
            .get(unit.x, unit.y)
            .expect("filtered units have an index");
    }
    let params = CdefParams::with_sets(damping, &sets).expect("strengths within syntax range");
    apply_searched(picture, &units, params, dsp, &mut scratch.unfiltered);
    params
}

/// Squared error against the source of `units` of `planes` filtered with
/// each candidate strength, summed per row of the error table: row
/// `row(unit)` of `rows`, `candidates.len()` values per row.
#[allow(clippy::too_many_arguments)]
fn unit_errors(
    (source, picture): (&Picture, &Picture),
    units: &[Unit],
    damping: u8,
    candidates: &[Strength],
    planes: &[PlaneIndex],
    (rows, row): (usize, impl Fn(&Unit) -> usize),
    dsp: &Dsp,
    scratch: &mut CdefScratch,
) -> Vec<u64> {
    let mut errors = vec![0; rows * candidates.len()];
    let CdefScratch {
        unfiltered: unfiltered_planes,
        trial,
        ..
    } = scratch;
    for &plane in planes {
        let unfiltered = picture.plane(plane).view();
        let original = source.plane(plane).view();
        let (width, height) = (unfiltered.width(), unfiltered.height());
        let storage = unfiltered_planes.plane(picture, plane);
        let side = UNIT >> usize::from(plane != PlaneIndex::Y);
        // Only refreshed rows are read, so earlier contents may stay.
        if trial.len() < width * height {
            trial.resize(width * height, 0);
        }
        let filtered = &mut trial[..width * height];
        let measured_rows = unit_rows(units, plane, side);
        for (index, &candidate) in candidates.iter().enumerate() {
            let measured = if candidate.is_off() {
                unfiltered
            } else {
                // Units the filter leaves alone must read unfiltered. Only
                // the units' rows are measured, so only they are refreshed
                // (a whole-plane copy per candidate was ~0.6% of the
                // encoder's instructions at the default row period 8).
                for &y in &measured_rows {
                    filtered[y * width..(y + 1) * width].copy_from_slice(unfiltered.row(y));
                }
                let mut out = PlaneViewMut::new(filtered, width, width, height)
                    .expect("a plane-sized buffer");
                filter_units(dsp, storage, &mut out, units, damping, plane, |_| candidate);
                PlaneView::new(filtered, width, width, height).expect("a plane-sized buffer")
            };
            for unit in units {
                errors[row(unit) * candidates.len() + index] +=
                    unit_sse(dsp, (&original, &measured), unit, side, plane);
            }
        }
    }
    errors
}

/// The rows of `plane` that the blocks of `units` (raster order, `side`
/// samples square in that plane) cover, ascending.
fn unit_rows(units: &[Unit], plane: PlaneIndex, side: usize) -> Vec<usize> {
    let shift = usize::from(plane != PlaneIndex::Y);
    let mut rows: Vec<usize> = Vec::new();
    for unit in units {
        let top = unit.y >> shift;
        if rows.last().is_none_or(|&last| last < top) {
            rows.extend(top..top + side);
        }
    }
    rows
}

/// Squared error of one unit of `plane` (`side` samples square) between
/// `original` and `measured`.
fn unit_sse(
    dsp: &Dsp,
    (original, measured): (&PlaneView<'_>, &PlaneView<'_>),
    unit: &Unit,
    side: usize,
    plane: PlaneIndex,
) -> u64 {
    let shift = usize::from(plane != PlaneIndex::Y);
    let (x, y) = (unit.x >> shift, unit.y >> shift);
    let size = BlockSize::new(side, side).expect("CDEF units are 8x8 or 4x4");
    let inside = "units lie inside the plane";
    let original = original.sub_view(x, y, side, side).expect(inside);
    let measured = measured.sub_view(x, y, side, side).expect(inside);
    u64::from(dsp.variance(size, &original, &measured).sse)
}

/// Uniformly random parameters (or off), so `debug.random_decisions`
/// streams exercise every strength and damping; with `multiple_sets`, also
/// every set count (pair with [`random_indices`]).
pub(crate) fn random_params(random: &mut nc_encoder::DebugRng, multiple_sets: bool) -> CdefParams {
    /// Chance (of 256) that a random frame leaves CDEF off.
    const OFF_CHANCE: u64 = 32;
    if random.chance(OFF_CHANCE) {
        return CdefParams::OFF;
    }
    let count = if multiple_sets {
        1 + random.below(MAX_SETS as u64) as usize
    } else {
        1
    };
    let mut strength = || Strength {
        primary: random.below(u64::from(MAX_PRIMARY) + 1) as u8,
        secondary: random.below(u64::from(SECONDARY_STRENGTHS)) as u8,
    };
    let mut sets = [StrengthSet::OFF; MAX_SETS];
    for set in &mut sets[..count] {
        set.luma = strength();
        set.chroma = strength();
    }
    let damping = MIN_DAMPING + random.below(u64::from(MAX_DAMPING - MIN_DAMPING) + 1) as u8;
    CdefParams::with_sets(damping, &sets[..count]).expect("strengths within syntax range")
}

/// Give every coded filter block a uniformly random index among `params`'
/// sets.
pub(crate) fn random_indices(
    random: &mut nc_encoder::DebugRng,
    params: CdefParams,
    indices: &mut StrengthIndices,
) {
    let count = params.sets().len().max(1) as u64;
    for index in indices.indices.iter_mut().flatten() {
        *index = random.below(count) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_syntax_roundtrips() {
        let sets = [
            CdefParams::OFF,
            CdefParams::new(3, Strength::OFF, Strength::OFF).unwrap(),
            CdefParams::new(
                5,
                Strength {
                    primary: 0,
                    secondary: 3,
                },
                Strength {
                    primary: 15,
                    secondary: 3,
                },
            )
            .unwrap(),
            CdefParams::new(
                6,
                Strength {
                    primary: 4,
                    secondary: 1,
                },
                Strength::OFF,
            )
            .unwrap(),
        ];
        for single_picture in [false, true] {
            for params in sets {
                let mut bits = BitWriter::new();
                params.write(&mut bits, single_picture);
                bits.put_bits(0x5, 3);
                bits.byte_align_zero();
                let bytes = bits.into_bytes();
                let mut reader = BitReader::new(&bytes);
                let read = CdefParams::read(&mut reader, single_picture).unwrap();
                let expected = if single_picture && !params.filters() {
                    CdefParams::OFF
                } else {
                    params
                };
                assert_eq!(read, expected);
                assert_eq!(reader.read_bits(3).unwrap(), 0x5);
            }
        }
    }

    fn sets(count: usize) -> Vec<StrengthSet> {
        (0..count)
            .map(|i| StrengthSet {
                luma: Strength {
                    primary: (i * 2) as u8,
                    secondary: (i % 4) as u8,
                },
                chroma: Strength {
                    primary: (15 - i) as u8,
                    secondary: ((i + 1) % 4) as u8,
                },
            })
            .collect()
    }

    #[test]
    fn several_strength_sets_roundtrip() {
        for count in [2, 3, MAX_SETS] {
            let params = CdefParams::with_sets(4, &sets(count)).unwrap();
            let mut bits = BitWriter::new();
            params.write(&mut bits, false);
            bits.byte_align_zero();
            let bytes = bits.into_bytes();
            let read = CdefParams::read(&mut BitReader::new(&bytes), false).unwrap();
            assert_eq!(read, params);
            assert_eq!(read.sets(), &sets(count)[..]);
        }
        assert!(CdefParams::with_sets(4, &sets(MAX_SETS + 1)).is_err());
        assert!(CdefParams::with_sets(4, &[]).is_err());
    }

    #[test]
    fn strength_indices_roundtrip_with_left_contexts() {
        const QINDEX: u8 = 120;
        // 5x3 filter blocks; every third has no index (all skipped).
        let (width, height) = (5 * FILTER_BLOCK, 3 * FILTER_BLOCK);
        for count in 1..=MAX_SETS {
            let params = CdefParams::with_sets(3, &sets(count)).unwrap();
            for adaptive in [false, true] {
                let mut written = StrengthIndices::new(width, height);
                for (i, index) in written.indices.iter_mut().enumerate() {
                    *index = (i % 3 != 2).then_some(((i * 5 + i / 4) % count) as u8);
                }
                let mut writer = Writer::new();
                let mut reader_models = None;
                if adaptive {
                    writer.enable_adaptation(QINDEX);
                    reader_models = Some(QINDEX);
                }
                let mut defaults = TileCdfs::new(QINDEX);
                for position in written.coded_positions() {
                    write_index(&mut writer, params, &written, position, &mut defaults).unwrap();
                }
                writer.literal(0x2a, 6).unwrap();
                let bytes = writer.finish();
                let mut reader = Reader::new(&bytes).unwrap();
                if let Some(qindex) = reader_models {
                    reader.enable_adaptation(qindex);
                }
                let mut read = StrengthIndices::new(width, height);
                for y in (0..height).step_by(FILTER_BLOCK) {
                    for x in (0..width).step_by(FILTER_BLOCK) {
                        // A skipped block first, then the block that codes.
                        let skip = written.get(x, y).is_none();
                        read_index(&mut reader, params, &mut read, (x, y), true, None).unwrap();
                        read_index(
                            &mut reader,
                            params,
                            &mut read,
                            (x, y),
                            skip,
                            Some(&mut defaults),
                        )
                        .unwrap();
                    }
                }
                assert_eq!(reader.literal(6).unwrap(), 0x2a);
                assert_eq!(read.indices, written.indices);
            }
        }
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let strong = Strength {
            primary: 16,
            secondary: 0,
        };
        assert!(CdefParams::new(3, strong, Strength::OFF).is_err());
        assert!(CdefParams::new(7, Strength::OFF, Strength::OFF).is_err());
        assert!(CdefParams::new(2, Strength::OFF, Strength::OFF).is_err());
    }

    #[test]
    fn adjust_strength_matches_avm_classes() {
        assert_eq!(adjust_strength(15, 0), 0);
        assert_eq!(adjust_strength(15, 63), (15 * 4 + 8) >> 4);
        assert_eq!(adjust_strength(15, 64), (15 * 4 + 8) >> 4);
        assert_eq!(adjust_strength(15, 128), (15 * 5 + 8) >> 4);
        assert_eq!(adjust_strength(15, i32::MAX), 15);
    }
}
