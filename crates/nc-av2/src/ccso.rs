//! AV2 CCSO, the cross-component sample offset filter (AVM v1.0.0
//! `av2/common/ccso.c`): the frame-header parameters and their syntax
//! (`setup_ccso` in `av2/decoder/decodeframe.c`), the per-unit enable flags
//! (`read_ccso` in `av2/decoder/decodemv.c`, context `av2_get_ccso_context`
//! in `av2/common/pred_common.c`), and the picture filter shared by the
//! encoder reconstruction and our decoder.
//!
//! CCSO runs after CDEF. Each sample of an enabled plane gets an offset
//! looked up from the *deblocked, pre-CDEF* luma around its co-located
//! luma sample: the luma sample's band (its top `band_log2` bits) and how
//! two neighbours on a chosen line (`support`) compare with it, each
//! quantised to below / near / above (or below / not below) a threshold.
//! The offsets are coded per frame and plane; each 256×256 luma unit
//! (`CCSO_BLK_SIZE`; our sequences do not match units to superblocks, and
//! code one tile) codes a flag per enabled plane at its first block, after
//! the CDEF index. Frames never reuse a reference frame's filters
//! (`reuse_ccso`, `sb_reuse_ccso` coded 0).
//!
//! The encoder chooses the filters after CDEF ([`crate::ccso_search`]), so
//! the unit flags are deferred like CDEF's indices and coded when the tile
//! finishes ([`crate::entropy::Writer::replay`]).

use nc_common::decode::{BitReader, DecodeError, Result};
use nc_common::{BitWriter, Picture, PlaneIndex, PlaneView};

use crate::cdf_models::TileCdfs;
use crate::entropy::{Reader, Writer};

/// Side of a flag unit in luma samples (`1 << CCSO_BLK_SIZE`, with
/// `ccso_unit_matches_sb_size` 0 and one tile).
pub const UNIT: usize = 256;
/// Luma border the filter's neighbours read (`CCSO_PADDING_SIZE`).
const PADDING: usize = 5;
/// Offset magnitudes by code (`ccso_offset`), before the frame's scale.
const OFFSET_STEPS: [i8; 8] = [0, 1, -1, 3, -3, 7, -7, -10];
/// Largest offset code: its unary code has no terminating zero.
const MAX_OFFSET_CODE: u8 = OFFSET_STEPS.len() as u8 - 1;
/// Offset multiplier by `scale_idx` (`ccso_scale`).
const SCALES: [i8; 4] = [1, 2, 3, 4];
/// Edge threshold by `scale_idx` and `quant_idx` (`quant_sz`).
const THRESHOLDS: [[i32; 4]; 4] = [
    [16, 8, 32, 0],
    [56, 40, 64, 128],
    [48, 24, 96, 192],
    [80, 112, 160, 256],
];
/// Width of `scale_idx` and of `quant_idx`.
const INDEX_BITS: u32 = 2;
/// Width of `ext_filter_support`; its largest value is reserved.
const SUPPORT_BITS: u32 = 3;
/// Neighbour pairs (`support`) the syntax allows: 0..=6.
pub const SUPPORTS: u8 = 7;
/// Width of `max_band_log2` with and without edge classes.
const BAND_ONLY_BITS: u32 = 3;
const EDGE_BAND_BITS: u32 = 2;
/// Largest band count (`CCSO_BAND_NUM`), as a power of two.
const MAX_BAND_LOG2: u8 = 6;
/// Largest band count with edge classes (two bits of `max_band_log2`).
const MAX_EDGE_BAND_LOG2: u8 = (1 << EDGE_BAND_BITS) - 1;
/// Coded offsets of one plane at most: 64 bands without edge classes, or
/// 8 bands of 3×3 edge classes.
pub const MAX_ENTRIES: usize = 72;
/// Edge classes per neighbour (`edge_clf_to_edge_interval`).
const THREE_LEVEL: usize = 3;
const TWO_LEVEL: usize = 2;
/// Entries of the look-up table the filter indexes as
/// `band << 4 | class0 << 2 | class1` (`filter_offset`).
pub const LUT_LEN: usize = (1 << MAX_BAND_LOG2) << 4;
/// Largest 8-bit sample.
const MAX_SAMPLE: i32 = 255;
/// Planes CCSO may filter.
const PLANES: usize = 3;
/// Context of a unit whose left unit is filtered (the left and bottom-left
/// neighbours of a unit's first block both lie in its left unit:
/// `av2_get_ccso_context`'s same-superblock case), otherwise 0.
const LEFT_FILTERED_CONTEXT: usize = 2;

/// One plane's filter: how samples are classified and each class's offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneFilter {
    /// Bands only, no edge classes (`ccso_bo_only`).
    pub band_only: bool,
    /// `scale_idx`: the offsets' multiplier and the threshold row.
    pub scale: u8,
    /// `quant_idx`: the threshold column (0 when `band_only`).
    pub quant: u8,
    /// `ext_filter_support`: which neighbour pair (0 when `band_only`).
    pub support: u8,
    /// `edge_clf` 1: neighbours classed below / not below (two levels)
    /// instead of below / near / above.
    pub two_level: bool,
    /// `max_band_log2`: 2^n bands of the luma sample.
    pub band_log2: u8,
    /// Offset codes (indices into [`OFFSET_STEPS`]) in coded order: first
    /// neighbour's class, second's, then band; two 4-bit codes per byte
    /// ([`Self::offset`]), so frame headers stay small to copy.
    offsets: [u8; MAX_ENTRIES / 2],
}

impl PlaneFilter {
    /// A filter whose offsets are all zero.
    #[must_use]
    pub fn zero(
        band_only: bool,
        scale: u8,
        quant: u8,
        support: u8,
        two_level: bool,
        band_log2: u8,
    ) -> Self {
        Self {
            band_only,
            scale,
            quant,
            support,
            two_level,
            band_log2,
            offsets: [0; MAX_ENTRIES / 2],
        }
    }

    /// The offset code of entry `index` (coded order).
    #[must_use]
    pub fn offset(&self, index: usize) -> u8 {
        self.offsets[index / 2] >> (index % 2 * 4) & 0xf
    }

    /// Set the offset code of entry `index`.
    pub fn set_offset(&mut self, index: usize, code: u8) {
        let shift = index % 2 * 4;
        let byte = &mut self.offsets[index / 2];
        *byte = *byte & !(0xf << shift) | (code & 0xf) << shift;
    }

    /// Edge classes per neighbour.
    #[must_use]
    pub fn intervals(&self) -> usize {
        if self.band_only {
            1
        } else if self.two_level {
            TWO_LEVEL
        } else {
            THREE_LEVEL
        }
    }

    /// Bands of the luma sample.
    #[must_use]
    pub fn bands(&self) -> usize {
        1 << self.band_log2
    }

    /// Offsets the plane codes.
    #[must_use]
    pub fn entries(&self) -> usize {
        self.intervals() * self.intervals() * self.bands()
    }

    /// Index into [`Self::offsets`] of a class.
    #[must_use]
    pub fn entry(&self, band: usize, class0: usize, class1: usize) -> usize {
        (class0 * self.intervals() + class1) * self.bands() + band
    }

    /// The edge threshold.
    #[must_use]
    pub fn threshold(&self) -> i32 {
        THRESHOLDS[usize::from(self.scale)][usize::from(self.quant)]
    }

    /// The offset an offset code adds at this plane's scale.
    #[must_use]
    pub fn offset_value(&self, code: u8) -> i32 {
        i32::from(OFFSET_STEPS[usize::from(code)]) * i32::from(SCALES[usize::from(self.scale)])
    }

    /// The filter's look-up table, indexed like AVM's `filter_offset`.
    #[must_use]
    pub fn lut(&self) -> [i8; LUT_LEN] {
        let mut lut = [0; LUT_LEN];
        for class0 in 0..self.intervals() {
            for class1 in 0..self.intervals() {
                for band in 0..self.bands() {
                    let code = self.offset(self.entry(band, class0, class1));
                    lut[band << 4 | class0 << 2 | class1] = self.offset_value(code) as i8;
                }
            }
        }
        lut
    }

    /// The classifier of this filter's syntax.
    #[must_use]
    pub fn classifier(&self) -> Classifier {
        Classifier {
            neighbours: neighbour_pair(self.support),
            threshold: self.threshold(),
            levels: self.intervals(),
            band_shift: if self.band_log2 == 0 {
                None
            } else {
                Some(8 - u32::from(self.band_log2))
            },
        }
    }

    fn is_valid(&self) -> bool {
        let edge_ok = if self.band_only {
            self.quant == 0
                && self.support == 0
                && !self.two_level
                && self.band_log2 <= MAX_BAND_LOG2
        } else {
            self.support < SUPPORTS
                && self.band_log2 <= MAX_EDGE_BAND_LOG2
                && (!self.two_level || self.threshold() != 0)
        };
        edge_ok
            && usize::from(self.scale) < SCALES.len()
            && usize::from(self.quant) < THRESHOLDS[0].len()
            && (0..MAX_ENTRIES).all(|i| {
                let code = self.offset(i);
                code <= MAX_OFFSET_CODE && (i < self.entries() || code == 0)
            })
    }

    /// `setup_ccso`'s per-plane syntax after the reuse flags.
    fn write(&self, bits: &mut BitWriter) {
        bits.put_bit(self.band_only);
        bits.put_bits(u32::from(self.scale), INDEX_BITS);
        if self.band_only {
            bits.put_bits(u32::from(self.band_log2), BAND_ONLY_BITS);
        } else {
            bits.put_bits(u32::from(self.quant), INDEX_BITS);
            bits.put_bits(u32::from(self.support), SUPPORT_BITS);
            if self.threshold() != 0 {
                bits.put_bit(self.two_level);
            }
            bits.put_bits(u32::from(self.band_log2), EDGE_BAND_BITS);
        }
        for code in (0..self.entries()).map(|i| self.offset(i)) {
            // `read_ccso_offset_idx`: unary, the largest code unterminated.
            for _ in 0..code {
                bits.put_bit(true);
            }
            if code < MAX_OFFSET_CODE {
                bits.put_bit(false);
            }
        }
    }

    fn read(bits: &mut BitReader<'_>) -> Result<Self> {
        let band_only = bits.read_bit()?;
        let scale = bits.read_bits(INDEX_BITS)? as u8;
        let mut filter = Self::zero(band_only, scale, 0, 0, false, 0);
        if band_only {
            filter.band_log2 = bits.read_bits(BAND_ONLY_BITS)? as u8;
            if filter.band_log2 > MAX_BAND_LOG2 {
                return Err(DecodeError::Malformed("AV2 CCSO band count"));
            }
        } else {
            filter.quant = bits.read_bits(INDEX_BITS)? as u8;
            filter.support = bits.read_bits(SUPPORT_BITS)? as u8;
            if filter.support >= SUPPORTS {
                return Err(DecodeError::Malformed("AV2 CCSO filter support"));
            }
            if filter.threshold() != 0 {
                filter.two_level = bits.read_bit()?;
            }
            filter.band_log2 = bits.read_bits(EDGE_BAND_BITS)? as u8;
        }
        for index in 0..filter.entries() {
            let mut code = 0;
            while code < MAX_OFFSET_CODE && bits.read_bit()? {
                code += 1;
            }
            filter.set_offset(index, code);
        }
        Ok(filter)
    }
}

/// The two neighbours of a support, as (dx, dy) luma offsets: the first
/// sets class 0's index, the second class 1's (`derive_ccso_sample_pos`).
#[must_use]
pub fn neighbour_pair(support: u8) -> [(isize, isize); 2] {
    match support {
        0 => [(0, -1), (0, 1)],
        1 => [(-1, 0), (1, 0)],
        2 => [(-1, -1), (1, 1)],
        3 => [(1, -1), (-1, 1)],
        4 => [(-2, -1), (2, 1)],
        5 => [(-2, 1), (2, -1)],
        _ => [(2, 0), (-2, 0)],
    }
}

/// How a luma sample and its neighbour pair select a look-up table entry
/// (`cal_filter_support` and the band shift).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classifier {
    /// The neighbour pair.
    pub neighbours: [(isize, isize); 2],
    /// Edge threshold.
    pub threshold: i32,
    /// Classes per neighbour: 1 (bands only), 2 or 3.
    pub levels: usize,
    /// Right shift of the luma sample to its band, when there are bands.
    pub band_shift: Option<u32>,
}

impl Classifier {
    /// The class of one neighbour difference `d` (neighbour minus centre).
    #[must_use]
    pub fn class(&self, d: i32) -> usize {
        match self.levels {
            THREE_LEVEL if d > self.threshold => 2,
            THREE_LEVEL | TWO_LEVEL if d < -self.threshold => 0,
            THREE_LEVEL | TWO_LEVEL => 1,
            _ => 0,
        }
    }

    /// The look-up index of a luma sample `centre` with neighbours `a`, `b`.
    #[must_use]
    pub fn lut_index(&self, centre: u8, a: u8, b: u8) -> usize {
        let band = self
            .band_shift
            .map_or(0, |shift| usize::from(centre >> shift));
        let class0 = self.class(i32::from(a) - i32::from(centre));
        let class1 = self.class(i32::from(b) - i32::from(centre));
        band << 4 | class0 << 2 | class1
    }
}

/// The frame's CCSO: a filter per plane, or none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CcsoParams {
    planes: [Option<PlaneFilter>; PLANES],
}

impl CcsoParams {
    /// No CCSO in this frame (`ccso_frame_flag` 0).
    pub const OFF: Self = Self {
        planes: [None; PLANES],
    };

    /// The given planes' filters.
    ///
    /// # Errors
    ///
    /// Rejects a filter outside its syntax range.
    pub fn new(planes: [Option<PlaneFilter>; PLANES]) -> Result<Self> {
        if planes.iter().flatten().any(|filter| !filter.is_valid()) {
            return Err(DecodeError::Malformed("AV2 CCSO parameter out of range"));
        }
        Ok(Self { planes })
    }

    /// Whether any plane is filtered.
    #[must_use]
    pub fn is_on(&self) -> bool {
        self.planes.iter().any(Option::is_some)
    }

    /// The filter of `plane`, if it is filtered.
    #[must_use]
    pub fn plane(&self, plane: PlaneIndex) -> Option<&PlaneFilter> {
        self.planes[plane as usize].as_ref()
    }

    /// Write `setup_ccso`'s syntax for a frame of a sequence that enables
    /// CCSO. A single picture infers `ccso_frame_flag` 1; an intra frame
    /// has no reuse flags.
    pub fn write(&self, bits: &mut BitWriter, single_picture: bool, intra: bool) {
        if !single_picture {
            bits.put_bit(self.is_on());
            if !self.is_on() {
                return;
            }
        }
        for filter in &self.planes {
            bits.put_bit(filter.is_some());
            let Some(filter) = filter else {
                continue;
            };
            if !intra {
                bits.put_bit(false); // reuse_ccso
                bits.put_bit(false); // sb_reuse_ccso
            }
            filter.write(bits);
        }
    }

    /// Read `setup_ccso`'s syntax (see [`Self::write`]).
    ///
    /// # Errors
    ///
    /// Rejects truncation, reserved values and filter reuse (outside the
    /// subset).
    pub fn read(bits: &mut BitReader<'_>, single_picture: bool, intra: bool) -> Result<Self> {
        if !single_picture && !bits.read_bit()? {
            return Ok(Self::OFF);
        }
        let mut planes = [None; PLANES];
        for plane in &mut planes {
            if !bits.read_bit()? {
                continue;
            }
            if !intra && (bits.read_bit()? || bits.read_bit()?) {
                return Err(DecodeError::Unsupported("AV2 CCSO filter reuse"));
            }
            *plane = Some(PlaneFilter::read(bits)?);
        }
        Self::new(planes)
    }
}

/// Each unit's enable flag per plane. Units start disabled; a unit's flag
/// is coded at its first block, the block at its top-left sample.
#[derive(Debug, Clone)]
pub struct UnitFlags {
    columns: usize,
    flags: Vec<[bool; PLANES]>,
}

impl UnitFlags {
    /// Flags for a `width` × `height` luma picture.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        let columns = width.div_ceil(UNIT);
        Self {
            columns,
            flags: vec![[false; PLANES]; columns * height.div_ceil(UNIT)],
        }
    }

    fn index(&self, x: usize, y: usize) -> usize {
        y / UNIT * self.columns + x / UNIT
    }

    /// The flag of `plane` in the unit holding luma sample (`x`, `y`).
    #[must_use]
    pub fn get(&self, plane: PlaneIndex, x: usize, y: usize) -> bool {
        self.flags[self.index(x, y)][plane as usize]
    }

    /// Set the flag of `plane` in the unit holding luma sample (`x`, `y`).
    pub fn set(&mut self, plane: PlaneIndex, x: usize, y: usize, on: bool) {
        let index = self.index(x, y);
        self.flags[index][plane as usize] = on;
    }

    /// Top-left luma samples of the units, in coding (raster) order.
    #[must_use]
    pub fn positions(&self) -> Vec<(usize, usize)> {
        (0..self.flags.len())
            .map(|unit| (unit % self.columns * UNIT, unit / self.columns * UNIT))
            .collect()
    }

    /// Whether the block at (`x`, `y`) codes its unit's flags.
    #[must_use]
    pub fn codes_flags(x: usize, y: usize) -> bool {
        x.is_multiple_of(UNIT) && y.is_multiple_of(UNIT)
    }

    fn context(&self, plane: PlaneIndex, x: usize, y: usize) -> usize {
        if x >= UNIT && self.get(plane, x - UNIT, y) {
            LEFT_FILTERED_CONTEXT
        } else {
            0
        }
    }
}

/// Write the flags of the unit at (`x`, `y`) for every plane `params`
/// filters (`write_ccso`). `defaults` supplies fixed probabilities when the
/// tile does not adapt.
/// # Errors
/// Propagates entropy-coder errors.
pub(crate) fn write_flags(
    writer: &mut Writer,
    params: &CcsoParams,
    flags: &UnitFlags,
    (x, y): (usize, usize),
    defaults: &mut TileCdfs,
) -> Result<()> {
    for plane in PlaneIndex::ALL {
        if params.plane(plane).is_none() {
            continue;
        }
        let context = flags.context(plane, x, y);
        let p = plane as usize;
        let fixed = *defaults.ccso[p][context].inverse();
        writer.model(usize::from(flags.get(plane, x, y)), &fixed, |m| {
            &mut m.ccso[p][context]
        })?;
    }
    Ok(())
}

/// `read_ccso` for the coding block at (`x`, `y`): when it is its unit's
/// first block, read a flag for every plane `params` filters.
/// # Errors
/// Rejects truncated or malformed data.
pub(crate) fn read_flags(
    reader: &mut Reader<'_>,
    params: &CcsoParams,
    flags: &mut UnitFlags,
    (x, y): (usize, usize),
    defaults: &mut TileCdfs,
) -> Result<()> {
    if !params.is_on() || !UnitFlags::codes_flags(x, y) {
        return Ok(());
    }
    for plane in PlaneIndex::ALL {
        if params.plane(plane).is_none() {
            continue;
        }
        let context = flags.context(plane, x, y);
        let p = plane as usize;
        let fixed = *defaults.ccso[p][context].inverse();
        let on = reader.model(&fixed, |m| &mut m.ccso[p][context])? == 1;
        flags.set(plane, x, y, on);
    }
    Ok(())
}

/// The deblocked luma CCSO classifies, copied before CDEF changes it and
/// padded by [`PADDING`] replicated samples (`extend_ccso_border`).
#[derive(Debug, Default)]
pub struct CcsoSource {
    stride: usize,
    width: usize,
    height: usize,
    samples: Vec<u8>,
}

impl CcsoSource {
    /// An empty source; [`Self::fill_from`] sizes it.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Copy and pad the deblocked luma plane.
    pub fn fill_from(&mut self, luma: &PlaneView<'_>) {
        let (width, height) = (luma.width(), luma.height());
        self.stride = width + 2 * PADDING;
        self.width = width;
        self.height = height;
        self.samples.resize(self.stride * (height + 2 * PADDING), 0);
        for y in 0..height {
            let row = luma.row(y);
            let start = (y + PADDING) * self.stride;
            let out = &mut self.samples[start..start + self.stride];
            out[..PADDING].fill(row[0]);
            out[PADDING..PADDING + width].copy_from_slice(&row[..width]);
            out[PADDING + width..].fill(row[width - 1]);
        }
        let (top, rest) = self.samples.split_at_mut(PADDING * self.stride);
        let first = &rest[..self.stride];
        for row in top.chunks_exact_mut(self.stride) {
            row.copy_from_slice(first);
        }
        let last_start = (height + PADDING - 1) * self.stride;
        let (body, bottom) = self.samples.split_at_mut(last_start + self.stride);
        let last = &body[last_start..];
        for row in bottom.chunks_exact_mut(self.stride) {
            row.copy_from_slice(last);
        }
    }

    /// The luma sample at (`x`, `y`), which may lie up to [`PADDING`]
    /// samples outside the picture.
    #[must_use]
    pub fn at(&self, x: isize, y: isize) -> u8 {
        let px = (x + PADDING as isize) as usize;
        let py = (y + PADDING as isize) as usize;
        self.samples[py * self.stride + px]
    }

    /// Luma width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Luma height.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The look-up index of the luma sample at (`x`, `y`).
    #[must_use]
    pub fn lut_index(&self, classifier: &Classifier, x: usize, y: usize) -> usize {
        let (x, y) = (x as isize, y as isize);
        let [(ax, ay), (bx, by)] = classifier.neighbours;
        classifier.lut_index(
            self.at(x, y),
            self.at(x + ax, y + ay),
            self.at(x + bx, y + by),
        )
    }
}

/// Filter `picture` (after CDEF) with `params`, in every unit whose flag is
/// set, classifying on `source` (`ccso_frame`).
pub fn filter_picture(
    picture: &mut Picture,
    source: &CcsoSource,
    params: &CcsoParams,
    flags: &UnitFlags,
) {
    for plane in PlaneIndex::ALL {
        let Some(filter) = params.plane(plane) else {
            continue;
        };
        let lut = filter.lut();
        let classifier = filter.classifier();
        let shift = usize::from(plane != PlaneIndex::Y);
        let mut view = picture.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            let row = view.row_mut(y);
            for (x, sample) in row.iter_mut().enumerate() {
                let (lx, ly) = (x << shift, y << shift);
                if !flags.get(plane, lx, ly) {
                    continue;
                }
                let offset = i32::from(lut[source.lut_index(&classifier, lx, ly)]);
                *sample = (i32::from(*sample) + offset).clamp(0, MAX_SAMPLE) as u8;
            }
        }
    }
}

/// A random filter set and random unit flags (`debug.random_decisions`).
pub(crate) fn random_params(random: &mut nc_encoder::DebugRng) -> CcsoParams {
    /// Chance (of 256) that a random frame leaves a plane unfiltered.
    const OFF_CHANCE: u64 = 64;
    let mut planes = [None; PLANES];
    for plane in &mut planes {
        if random.chance(OFF_CHANCE) {
            continue;
        }
        let band_only = random.chance(OFF_CHANCE);
        let scale = random.below(SCALES.len() as u64) as u8;
        let mut filter = PlaneFilter::zero(band_only, scale, 0, 0, false, 0);
        if band_only {
            filter.band_log2 = random.below(u64::from(MAX_BAND_LOG2) + 1) as u8;
        } else {
            filter.quant = random.below(THRESHOLDS[0].len() as u64) as u8;
            filter.support = random.below(u64::from(SUPPORTS)) as u8;
            filter.two_level = filter.threshold() != 0 && random.chance(128);
            filter.band_log2 = random.below(u64::from(MAX_EDGE_BAND_LOG2) + 1) as u8;
        }
        for index in 0..filter.entries() {
            filter.set_offset(index, random.below(u64::from(MAX_OFFSET_CODE) + 1) as u8);
        }
        *plane = Some(filter);
    }
    CcsoParams::new(planes).expect("random filters within syntax range")
}

/// Random unit flags for every plane `params` filters.
pub(crate) fn random_flags(
    random: &mut nc_encoder::DebugRng,
    params: &CcsoParams,
    flags: &mut UnitFlags,
) {
    for (x, y) in flags.positions() {
        for plane in PlaneIndex::ALL {
            if params.plane(plane).is_some() {
                flags.set(plane, x, y, random.chance(128));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter() -> PlaneFilter {
        let mut filter = PlaneFilter::zero(false, 1, 2, 5, true, 2);
        for i in 0..filter.entries() {
            filter.set_offset(i, (i % 8) as u8);
        }
        filter
    }

    #[test]
    fn params_round_trip_for_intra_and_inter_frames() {
        let mut band = PlaneFilter::zero(true, 3, 0, 0, false, 6);
        for i in 0..band.entries() {
            band.set_offset(i, (i % 8) as u8);
        }
        let params = CcsoParams::new([Some(filter()), None, Some(band)]).unwrap();
        for (single, intra) in [(false, true), (false, false), (true, true)] {
            for p in [params, CcsoParams::OFF] {
                let mut bits = BitWriter::new();
                p.write(&mut bits, single, intra);
                bits.put_bits(0b101, 3);
                bits.byte_align_zero();
                let bytes = bits.into_bytes();
                let mut reader = BitReader::new(&bytes);
                assert_eq!(CcsoParams::read(&mut reader, single, intra).unwrap(), p);
                assert_eq!(reader.read_bits(3).unwrap(), 0b101);
            }
        }
    }

    #[test]
    fn lut_follows_avm_indexing() {
        let f = filter();
        let lut = f.lut();
        // Two-level classes, four bands: band 3, class0 1, class1 0.
        let entry = f.entry(3, 1, 0);
        assert_eq!(
            i32::from(lut[3 << 4 | 1 << 2]),
            f.offset_value(f.offset(entry))
        );
        let c = f.classifier();
        assert_eq!(c.threshold, 64);
        assert_eq!(c.class(-65), 0);
        assert_eq!(c.class(-64), 1);
        assert_eq!(c.class(200), 1);
        assert_eq!(c.lut_index(200, 100, 200), (3 << 4) | 1);
    }

    #[test]
    fn reuse_and_reserved_values_are_rejected() {
        let mut bits = BitWriter::new();
        bits.put_bit(true); // frame flag
        bits.put_bit(true); // plane 0 enabled
        bits.put_bit(true); // reuse_ccso
        bits.put_bits(0, 8);
        bits.byte_align_zero();
        let bytes = bits.into_bytes();
        assert!(CcsoParams::read(&mut BitReader::new(&bytes), false, false).is_err());
    }
}
