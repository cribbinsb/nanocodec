//! Loop restoration parameters of a frame header: each plane's filter
//! type, the non-separable Wiener filters coded for the frame (predicted
//! from earlier filters of the frame, the references' filters, or a
//! trained dictionary) and the restoration unit sizes (dav2d
//! `parse_frame_hdr` "restoration", AVM `read_lr_params` and
//! `read_wienerns_framefilters_hdr`).
use super::filters::rank_bits;
use super::{FrameHeader, FrameType, Parser};
use crate::general::bits::subexp;
use nc_common::decode::{DecodeError, Result};

/// Restoration types (dav2d `Dav2dRestorationType`).
pub mod kind {
    /// No restoration.
    pub const NONE: u8 = 0;
    /// Pixel-classified Wiener (luma).
    pub const PC_WIENER: u8 = 1;
    /// Non-separable Wiener.
    pub const NS_WIENER: u8 = 2;
    /// Chosen per unit.
    pub const SWITCHABLE: u8 = 3;
}

/// Most filter classes of a plane, and taps of a luma and chroma filter.
pub const MAX_CLASSES: usize = 16;
/// Taps of a chroma filter (luma uses 16).
pub const MAX_TAPS: usize = 18;
const LUMA_TAPS: usize = 16;
/// Bits of a restoration type, and of the luma class-count index.
const TYPE_BITS: u32 = 2;
const CLASS_INDEX_BITS: u32 = 3;
/// log2 of the largest restoration unit, luma (chroma is one smaller in
/// 4:2:0).
const MAX_UNIT_LOG2: u8 = 9;
/// Smallest chroma unit log2 in 4:2:0.
const MIN_CHROMA_UNIT_LOG2: u8 = 5;
/// Filters a plane may predict from: with and without the 16 PC-Wiener
/// filters of the luma dictionary.
const DICTIONARY: usize = 64;
const PC_WIENER_FILTERS: usize = 16;
/// Most reference filters gathered (64 - 16 - 1 for luma).
const MAX_REFERENCE_FILTERS: usize = 47;
/// First group width of the filter-reference code.
const REFERENCE_K: u32 = 4;
/// Group widths of a tap below its range's bit width.
const TAP_K_SAVING: u32 = 3;
/// Most refinement levels of a luma and a chroma filter.
const LUMA_LEVELS: usize = 3;
/// Taps coded at each refinement level (dav2d `dav2d_subset_masks_y`,
/// `_uv`).
pub(crate) const LUMA_TAP_MASKS: [u32; 4] = [0x3f, 0xfc3, 0xfff, 0xffff];
pub(crate) const CHROMA_TAP_MASKS: [u32; 3] = [0x3f, 0x3ff, 0x3ffff];
/// First tap that the asymmetric chroma code pairs with the next.
const FIRST_PAIRED_TAP: usize = 6;
/// Bit width and minimum of each tap (dav2d `dav2d_ns_wiener_coef_range_y`,
/// `_uv`).
pub(crate) const LUMA_TAP_RANGE: [(u8, i8); LUMA_TAPS] = [
    (6, -24),
    (6, -24),
    (5, -14),
    (5, -14),
    (5, -16),
    (5, -16),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
];
pub(crate) const CHROMA_TAP_RANGE: [(u8, i8); MAX_TAPS] = [
    (6, -24),
    (6, -24),
    (5, -14),
    (5, -14),
    (5, -16),
    (5, -16),
    (5, -16),
    (5, -16),
    (5, -16),
    (5, -16),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
    (4, -8),
];
/// The order in which the trained dictionary's filters follow the
/// reference filters (dav2d `shuffled_index`).
const SHUFFLED: [u8; DICTIONARY] = [
    16, 7, 58, 21, 12, 61, 26, 38, 18, 30, 50, 45, 23, 49, 43, 62, 42, 54, 27, 36, 17, 44, 32, 34,
    4, 24, 52, 31, 37, 11, 33, 19, 35, 6, 22, 53, 63, 25, 41, 47, 1, 59, 0, 28, 40, 55, 48, 8, 5,
    51, 9, 46, 56, 60, 15, 2, 13, 14, 57, 29, 3, 20, 39, 10,
];
/// The trained luma dictionary (dav2d `dav2d_wiener_ns_filters`; the
/// last four taps are zero).
const DICTIONARY_FILTERS: [[i8; 12]; DICTIONARY] = [
    [39, 39, -14, -14, -16, -16, 7, 7, -1, -3, 1, 7],
    [-1, 3, 1, -1, -2, -1, 0, 0, 1, 0, 0, 0],
    [12, 14, -5, -6, -6, 2, 1, -1, 2, -1, 1, 1],
    [39, 23, -7, -6, -12, -7, -2, -3, 3, 2, 2, 1],
    [5, 12, -1, -2, -4, 15, 1, -5, -1, -7, 1, 2],
    [8, 8, -3, -3, -2, -4, 0, 2, 0, 0, 0, 1],
    [5, 7, -2, -3, -2, -3, 1, 1, 0, 0, 0, 1],
    [16, 16, -7, -6, 11, -6, -3, 1, -3, 2, 2, 1],
    [18, 39, -3, -14, 3, -16, -2, 5, -4, 1, 0, 4],
    [16, 4, -6, -1, -1, -6, 0, 1, 0, 2, 1, 0],
    [-7, 11, 7, -3, -2, 11, 0, -5, 0, -3, -1, 1],
    [21, 26, -6, -8, -10, 7, 1, -4, 1, -3, 2, 2],
    [5, 7, -2, -4, -3, -1, 1, 0, 1, 0, 0, 1],
    [14, 15, -6, -5, -4, -4, 1, 1, 0, 0, 1, 1],
    [9, 10, -4, -4, -6, 0, 1, 0, 2, -1, 1, 1],
    [25, 17, -6, -4, -12, -9, 1, 0, 3, 2, 1, 0],
    [13, 13, -4, -5, 3, -7, -2, 1, -1, 2, 1, 1],
    [4, 7, -1, -3, -5, -4, 1, 1, 2, 1, 0, 0],
    [8, 33, 0, -13, -10, 1, 2, -2, 2, -1, 0, 3],
    [-2, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [11, 11, -5, -5, -4, -5, 1, 1, 1, 1, 1, 1],
    [-2, 17, 4, -3, 7, -6, -3, 2, -3, 0, -1, 1],
    [30, 15, -7, 0, -5, -13, 0, 1, -1, 2, 1, 0],
    [33, 26, -8, -4, 7, -12, -4, 0, -5, 0, 2, 2],
    [8, 10, -4, -5, -4, -4, 1, 1, 1, 1, 1, 1],
    [23, 24, -8, -8, -1, -9, -1, 1, -1, 1, 2, 2],
    [18, 39, -5, -10, -11, -9, 5, 3, -2, -1, -1, 3],
    [7, 15, -3, -6, -5, -5, 2, 2, 0, 0, 0, 2],
    [12, 9, -5, -3, -8, -5, 2, 1, 2, 1, 1, 0],
    [11, 8, -4, -1, -5, -6, 1, 1, 0, 1, 1, 0],
    [3, 3, 0, 1, -2, 13, 0, -4, -1, -4, 1, 1],
    [4, 3, 2, 2, -4, 15, 0, -6, 1, -6, 0, 0],
    [13, 16, -7, -5, 15, -3, -3, 2, -6, 1, 1, 1],
    [32, 20, 1, 9, 15, 15, -7, -7, -7, -7, -2, -2],
    [16, 18, -6, -7, -9, -9, 2, 2, 2, 2, 1, 1],
    [10, 11, -3, -3, -7, 8, 1, -3, 2, -4, 1, 1],
    [36, -1, -12, 4, 15, 15, -8, -8, -5, -6, 7, 1],
    [36, 12, -12, -3, -11, 5, 1, -1, 3, -3, 2, 1],
    [1, 1, 0, -1, -2, -3, 0, 1, 1, 1, 0, 0],
    [11, 11, -5, -5, -1, -4, 0, 1, 0, 1, 1, 1],
    [5, 7, -3, -4, -3, -1, 1, 0, 1, 0, 1, 1],
    [8, 14, -3, -4, -4, -8, 1, 3, 0, 1, 0, 1],
    [28, 39, -9, -13, -13, -16, 3, 6, 1, 2, 0, 3],
    [1, 1, 0, -1, -3, -2, 1, 0, 1, 1, 0, 0],
    [17, 17, -5, -1, 2, 5, -1, -2, -3, -4, 1, 1],
    [8, 12, -1, -4, 3, -5, -2, -1, -2, 2, 1, 1],
    [4, 8, -2, -2, -5, -3, 2, 1, 1, 0, 0, 0],
    [-3, 6, 2, -2, 1, -2, 0, 1, -1, 0, -1, 1],
    [5, 6, -1, -2, -4, 4, 0, -2, 1, -2, 1, 1],
    [14, 12, -4, -5, -8, -7, 1, 0, 3, 3, 1, 0],
    [17, 18, -7, -8, -8, -3, 1, 0, 2, 0, 2, 2],
    [39, 18, -14, -3, -5, -14, 1, 1, -1, 4, 3, 0],
    [17, 0, -4, 2, 15, -5, -5, 0, -5, 2, 1, 0],
    [16, 14, -6, -5, -2, -8, 0, 2, -1, 1, 1, 2],
    [39, 2, -10, 8, 15, 15, -8, -8, -7, -8, 6, 1],
    [18, 7, -4, 1, -10, 2, 1, -2, 2, -2, 1, 0],
    [39, 39, -13, -14, -16, -16, 6, 5, 6, 4, 1, 3],
    [24, 22, -7, -5, 15, -6, -5, -2, -6, 0, 2, 2],
    [28, 0, -9, 9, -10, 14, 2, -1, -1, -8, 2, 0],
    [2, 2, -1, -1, -2, -1, 1, 0, 1, 0, 0, 0],
    [-5, 11, 6, -3, 15, -3, -6, 1, -3, 0, -1, 1],
    [39, 32, -9, 3, -6, 12, -1, -5, -6, -8, 2, 1],
    [5, 4, -2, -1, -4, -4, 1, 1, 1, 1, 0, 0],
    [25, 25, -8, -8, -7, -5, 1, 0, 0, 0, 1, 2],
];

/// A filter's taps.
pub type Taps = [i8; MAX_TAPS];

/// One plane's non-separable Wiener filters for the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NsWiener {
    /// The frame codes filters for this plane.
    pub frame_filters: bool,
    /// The class-count index and the class count.
    pub class_index: u8,
    /// Filter classes.
    pub classes: u8,
    /// The filters are a reference's (`temporal`), of rank `reference`.
    pub temporal: bool,
    /// The reference rank of a temporal plane.
    pub reference: u8,
    /// One filter per class.
    pub filters: [Taps; MAX_CLASSES],
}

/// One plane's restoration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Plane {
    /// The restoration type ([`kind`]).
    pub kind: u8,
    /// Its non-separable Wiener filters.
    pub wiener: NsWiener,
}

/// The frame's loop restoration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Restoration {
    /// Each plane's restoration.
    pub planes: [Plane; 3],
    /// log2 of the luma and chroma unit sizes.
    pub unit_log2: [u8; 2],
}

/// Parse the frame's loop restoration.
pub(super) fn parse(parser: &mut Parser<'_, '_, '_, '_>, header: &mut FrameHeader) -> Result<()> {
    let sequence = parser.context.sequence;
    if header.lossless.all || !sequence.filters.restoration {
        return Ok(());
    }
    for plane in 0..3 {
        plane_type(parser, header, plane)?;
    }
    let superblock = header.superblock_log2 - super::SMALLEST_SUPERBLOCK_LOG2;
    header.restoration.unit_log2[0] = MAX_UNIT_LOG2;
    if header.restoration.planes[0].kind != kind::NONE {
        header.restoration.unit_log2[0] -= unit_reduction(parser, superblock)?;
    }
    header.restoration.unit_log2[1] = MAX_UNIT_LOG2 - 1;
    if header.restoration.planes[1..]
        .iter()
        .any(|plane| plane.kind != kind::NONE)
    {
        header.restoration.unit_log2[1] -= unit_reduction(parser, superblock)?;
        if header.restoration.unit_log2[1] < MIN_CHROMA_UNIT_LOG2 {
            return Err(DecodeError::Malformed("AV2 chroma restoration unit"));
        }
    }
    for plane in 0..3 {
        if header.restoration.planes[plane].wiener.frame_filters {
            filters(parser, header, plane)?;
        }
    }
    parser.checkpoint("post-restoration");
    Ok(())
}

/// How much smaller than the largest a plane class's units are: one
/// halving, or (superblocks below 256) two or three.
fn unit_reduction(parser: &mut Parser<'_, '_, '_, '_>, superblock: u8) -> Result<u8> {
    let bits = &mut *parser.bits;
    Ok(if bits.read_bit()? {
        1
    } else if superblock < 2 && !bits.read_bit()? {
        2 + u8::from(superblock == 0 && !bits.read_bit()?)
    } else {
        0
    })
}

/// One plane's type and, for non-separable Wiener, how its frame filters
/// are coded.
fn plane_type(
    parser: &mut Parser<'_, '_, '_, '_>,
    header: &mut FrameHeader,
    plane: usize,
) -> Result<()> {
    let disabled = parser.context.sequence.filters.restoration_disable_mask[usize::from(plane > 0)];
    let kind = match disabled {
        0 => parser.bits.read_bits(TYPE_BITS)? as u8,
        3 => kind::NONE,
        _ => u8::from(parser.bits.read_bit()?) * (3 - disabled),
    };
    let mut wiener = NsWiener {
        classes: 1,
        ..NsWiener::default()
    };
    if kind >= kind::NS_WIENER {
        wiener.frame_filters = parser.bits.read_bit()?;
        if wiener.frame_filters {
            if header.frame_type == FrameType::Inter {
                wiener.temporal = parser.bits.read_bit()?;
            }
            if wiener.temporal {
                let width = rank_bits(header.reference_count);
                if width != 0 {
                    wiener.reference = parser.bits.read_bits(width)? as u8;
                    if wiener.reference >= header.reference_count {
                        return Err(DecodeError::Malformed("AV2 restoration reference"));
                    }
                }
                let source = temporal_source(parser, header, wiener.reference, plane)?;
                wiener.class_index = source.class_index;
                wiener.classes = source.classes;
            } else if plane == 0 {
                let index = parser.bits.read_bits(CLASS_INDEX_BITS)? as u8;
                wiener.class_index = index;
                wiener.classes = 1 + index + index.saturating_sub(3) + 2 * index.saturating_sub(5);
            }
        }
    }
    header.restoration.planes[plane] = Plane { kind, wiener };
    Ok(())
}

/// The reference filters a temporally predicted plane copies: the same
/// plane's, or for chroma the other chroma plane's.
fn temporal_source(
    parser: &Parser<'_, '_, '_, '_>,
    header: &FrameHeader,
    rank: u8,
    plane: usize,
) -> Result<NsWiener> {
    let planes = &parser.reference(header, rank)?.restoration.planes;
    let mut source = planes[plane].wiener;
    if !source.frame_filters && plane != 0 {
        source = planes[3 - plane].wiener;
    }
    if !source.frame_filters {
        return Err(DecodeError::Malformed(
            "AV2 restoration reference without filters",
        ));
    }
    Ok(source)
}

/// One plane's filters: copied from a reference, or each predicted from a
/// group (earlier classes of the frame, the references' filters, the
/// dictionary) and refined.
fn filters(
    parser: &mut Parser<'_, '_, '_, '_>,
    header: &mut FrameHeader,
    plane: usize,
) -> Result<()> {
    let wiener = header.restoration.planes[plane].wiener;
    let classes = usize::from(wiener.classes);
    let taps = if plane == 0 { LUMA_TAPS } else { MAX_TAPS };
    if wiener.temporal {
        let source = temporal_source(parser, header, wiener.reference, plane)?;
        let target = &mut header.restoration.planes[plane].wiener.filters;
        for (filter, source) in target.iter_mut().zip(&source.filters).take(classes) {
            filter[..taps].copy_from_slice(&source[..taps]);
        }
        return Ok(());
    }
    let no_pc_wiener = parser.context.sequence.filters.restoration_disable_mask[0] & 1 != 0;
    let total = if no_pc_wiener {
        PC_WIENER_FILTERS
    } else {
        DICTIONARY
    };
    let use_dictionary = plane == 0 && !no_pc_wiener;
    let room = total - classes - PC_WIENER_FILTERS * usize::from(use_dictionary);
    let references = reference_filters(parser, header, plane, room, taps)?;
    let groups = [
        classes,
        references.len(),
        usize::from(use_dictionary) * (total - classes - references.len()),
    ];
    let sources = filter_sources(parser, groups)?;
    let bits = &mut *parser.bits;
    let mut exact = 0_u32;
    for class in 0..classes {
        exact |= u32::from(bits.read_bit()?) << class;
    }
    let (masks, ranges): (&[u32], &[(u8, i8)]) = if plane == 0 {
        (&LUMA_TAP_MASKS, &LUMA_TAP_RANGE)
    } else {
        (&CHROMA_TAP_MASKS, &CHROMA_TAP_RANGE)
    };
    let mut decoded = [[0_i8; MAX_TAPS]; MAX_CLASSES];
    for class in 0..classes {
        let source = sources[class];
        let reference: Taps = if source == 0 {
            [0; MAX_TAPS]
        } else if source < classes {
            decoded[source - 1]
        } else if source < classes + groups[1] {
            references[source - classes]
        } else {
            let mut taps = [0; MAX_TAPS];
            let entry = DICTIONARY_FILTERS[usize::from(SHUFFLED[source - classes - groups[1]])];
            taps[..entry.len()].copy_from_slice(&entry);
            taps
        };
        if exact & (1 << class) != 0 {
            decoded[class][..taps].copy_from_slice(&reference[..taps]);
            continue;
        }
        let max_level = if plane == 0 {
            LUMA_LEVELS
        } else {
            LUMA_LEVELS - 1
        };
        let mut level = 0;
        while level < max_level && bits.read_bit()? {
            level += 1;
        }
        let asymmetric = plane != 0 && level != 0 && bits.read_bit()?;
        let mask = masks[level];
        let filter = &mut decoded[class];
        let mut tap = 0;
        while tap < taps {
            if mask & (1 << tap) != 0 {
                let (width, minimum) = ranges[tap];
                let value = subexp(
                    bits,
                    (i32::from(reference[tap]) - i32::from(minimum)) as u32,
                    1 << width,
                    u32::from(width) - TAP_K_SAVING,
                )?;
                filter[tap] = (value as i32 + i32::from(minimum)) as i8;
                if asymmetric && tap >= FIRST_PAIRED_TAP {
                    filter[tap + 1] = filter[tap];
                    tap += 1;
                }
            }
            tap += 1;
        }
    }
    header.restoration.planes[plane].wiener.filters[..classes].copy_from_slice(&decoded[..classes]);
    Ok(())
}

/// The references' filters a plane may predict from, at most `room`: for
/// each reference, its same plane's (and for chroma, then the other chroma
/// plane's) classes.
fn reference_filters(
    parser: &Parser<'_, '_, '_, '_>,
    header: &FrameHeader,
    plane: usize,
    room: usize,
    taps: usize,
) -> Result<Vec<Taps>> {
    let mut gathered = Vec::with_capacity(MAX_REFERENCE_FILTERS);
    for rank in 0..header.reference_count {
        let planes = &parser.reference(header, rank)?.restoration.planes;
        let order: &[usize] = match plane {
            0 => &[0],
            1 => &[1, 2],
            _ => &[2, 1],
        };
        for &source_plane in order {
            let source = &planes[source_plane].wiener;
            if source.frame_filters {
                let count = (room - gathered.len()).min(usize::from(source.classes));
                for filter in &source.filters[..count] {
                    let mut copy = [0; MAX_TAPS];
                    copy[..taps].copy_from_slice(&filter[..taps]);
                    gathered.push(copy);
                }
            }
        }
    }
    Ok(gathered)
}

/// For each class, the filter it is predicted from: an index into the
/// frame's classes (0 is the zero filter, `n` the `n`-th decoded), then the
/// reference filters, then the dictionary. Each class codes its group
/// (relative to a running prediction) and an index within it.
fn filter_sources(parser: &mut Parser<'_, '_, '_, '_>, groups: [usize; 3]) -> Result<Vec<usize>> {
    let bits = &mut *parser.bits;
    let mut predicted = if groups[1] > 2 {
        1
    } else if groups[1] <= 1 && groups[2] <= 1 {
        0
    } else if groups[1] >= groups[2] {
        1
    } else {
        2
    };
    let nonempty = 1 + usize::from(groups[1] != 0) + usize::from(groups[2] != 0);
    let mut chosen = [0_usize; 3];
    let mut sources = Vec::with_capacity(groups[0]);
    for class in 0..groups[0] {
        let group = if nonempty == 1 || !bits.read_bit()? {
            predicted
        } else if nonempty == 2 {
            2 - usize::from(groups[2] == 0) - predicted
        } else if bits.read_bit()? {
            2 - usize::from(predicted == 2)
        } else {
            usize::from(predicted == 0)
        };
        chosen[group] += 1;
        if chosen[group] + usize::from(group < predicted) > chosen[predicted] {
            predicted = group;
        }
        let base = groups[0] * usize::from(group != 0) + groups[1] * usize::from(group == 2);
        let range = if group == 0 { class + 1 } else { groups[group] };
        let index = if range == 1 {
            0
        } else {
            subexp(bits, (range >> 1) as u32, range as u32, REFERENCE_K)? as usize
        };
        sources.push(base + index);
    }
    Ok(sources)
}
