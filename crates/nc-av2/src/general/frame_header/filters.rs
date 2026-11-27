//! The in-loop filter parameters of a frame header: deblocking, the guided
//! detail filter, CDEF, (loop restoration, [`super::restoration`]) and
//! CCSO, in coding order (dav2d `parse_frame_hdr`, AVM
//! `read_uncompressed_header`).
use super::{FrameHeader, FrameType, Parser, restoration};
use nc_common::decode::{DecodeError, Result};

/// Most CDEF strength sets of a frame (`CDEF_MAX_STRENGTHS`).
pub const MAX_CDEF_STRENGTHS: usize = 8;
/// Bits of the CDEF damping (minus 3) and of the strength count (minus 1).
const CDEF_DAMPING_BITS: u32 = 2;
const CDEF_DAMPING_BASE: u8 = 3;
const CDEF_STRENGTH_COUNT_BITS: u32 = 3;
/// Bits of a CDEF strength: six, or two when the short form is flagged.
const CDEF_STRENGTH_BITS: u32 = 6;
const CDEF_SHORT_STRENGTH_SAVING: u32 = 4;
/// GDF: the quantizer index base of intra and inter frames, the offset and
/// step of the quantizer classes, the last class before the coded offset,
/// and its field widths.
const GDF_INTRA_QP_BASE: i32 = 85;
const GDF_INTER_QP_BASE: i32 = 110;
const GDF_QP_OFFSET: i32 = 37;
const GDF_QP_STEP: i32 = 25;
const GDF_LAST_QP_CLASS: i32 = 2;
const GDF_FIELD_BITS: u32 = 2;
/// GDF block side in luma samples (doubled in 256x256 superblocks).
const GDF_BLOCK: u32 = 128;
/// CCSO field widths and the reserved filter support.
const CCSO_SCALE_BITS: u32 = 2;
const CCSO_QUANT_BITS: u32 = 2;
const CCSO_SUPPORT_BITS: u32 = 3;
const CCSO_RESERVED_SUPPORT: u8 = 7;
const CCSO_BAND_ONLY_BITS: u32 = 3;
const CCSO_BAND_BITS: u32 = 2;
/// Largest band-only `max_band_log2`: its 3 bits could code 7, but AVM
/// rejects more than `CCSO_BAND_NUM` (64) bands ("Invalid CCSO Band
/// number", `read_ccso` in AVM v1.0.0 `av2/decoder/decodeframe.c`).
const CCSO_MAX_BAND_ONLY_LOG2: u8 = 6;
/// Largest CCSO offset index (unary code, 3 bits of magnitude-sign).
const CCSO_MAX_OFFSET: u8 = 7;
/// Edge classes per axis of the three-class classifier.
const CCSO_EDGE_CLASSES: usize = 3;
/// Edge classes per axis in the offset layout (dav2d `filter_off`:
/// `class0 << 5 | class1 << 3 | band`).
pub const CCSO_CLASS_STRIDE: usize = 4;
/// Bands per class in that layout: two bits of `max_band_log2`.
pub const CCSO_BAND_STRIDE: usize = 1 << ((1 << CCSO_BAND_BITS) - 1);
/// Offsets of one CCSO plane: the larger of 64 bands (band-only) and 3×3
/// edge classes of up to 8 bands in the layout above (88 entries).
pub const CCSO_OFFSETS: usize = {
    let band_only = 1 << CCSO_MAX_BAND_ONLY_LOG2;
    let edge = ((CCSO_EDGE_CLASSES - 1) * CCSO_CLASS_STRIDE + CCSO_EDGE_CLASSES) * CCSO_BAND_STRIDE;
    if band_only > edge { band_only } else { edge }
};
/// CCSO quantization step per scale and quantizer index; zero has no edge
/// classifier choice (dav2d `dav2d_ccso_quant_sz`).
const CCSO_QUANT_SIZE: [[u16; 4]; 4] = [
    [16, 8, 32, 0],
    [56, 40, 64, 128],
    [48, 24, 96, 192],
    [80, 112, 160, 256],
];
/// Planes of 4:2:0.
const PLANES: usize = 3;
/// 8-sample units per CCSO and GDF block side (`b64size` counts 64-sample
/// blocks).
const UNIT_SHIFT: u32 = 3;

/// Deblocking parameters (`deblocking_filter_params`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Deblock {
    /// Sub-prediction-unit edges are filtered too.
    pub sub_pu: bool,
    /// Luma vertical and horizontal edges, U and V filtered.
    pub level_y: [bool; 2],
    /// U edges filtered.
    pub level_u: bool,
    /// V edges filtered.
    pub level_v: bool,
    /// Quantizer offsets of the filter strength.
    pub delta_q_y: [i8; 2],
    /// U strength offset.
    pub delta_q_u: i8,
    /// V strength offset.
    pub delta_q_v: i8,
}

/// The guided detail filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Gdf {
    /// 0 off, 1 on for the whole frame, 2 per block.
    pub mode: u8,
    /// Quantizer class and scale.
    pub qp_index: u8,
    /// Strength scale, 1..=4.
    pub scale: u8,
    /// Filter block side in 64-sample units.
    pub block64: u8,
}

/// CDEF parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cdef {
    /// Damping (3..=6).
    pub damping: u8,
    /// Strength sets coded (`cdef_bits` as a count).
    pub strengths: u8,
    /// Blocks without residual are filtered too.
    pub on_skipped: bool,
    /// Luma strengths of each set.
    pub y_strength: [u8; MAX_CDEF_STRENGTHS],
    /// Chroma strengths of each set.
    pub uv_strength: [u8; MAX_CDEF_STRENGTHS],
}

/// One plane's CCSO parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools, reason = "the header's flags")]
pub struct CcsoPlane {
    /// The plane is filtered.
    pub enabled: bool,
    /// The parameters are a reference's (`reuse`), and its per-block flags
    /// too (`sb_reuse`).
    pub reuse: bool,
    /// The reference's per-block flags are reused.
    pub block_reuse: bool,
    /// The reference rank reused.
    pub reference: u8,
    /// Band offsets only, no edge classification.
    pub band_only: bool,
    /// Offset scale index.
    pub scale_index: u8,
    /// Edge quantization index.
    pub quant_index: u8,
    /// Filter tap pattern (`ext_filter_support`).
    pub filter_support: u8,
    /// The two-class edge classifier.
    pub edge_classifier: bool,
    /// log2 of the band count.
    pub max_band_log2: u8,
    /// Offset index of each class and band: `band` when band-only, else
    /// `(row * CCSO_CLASS_STRIDE + column) * CCSO_BAND_STRIDE + band`.
    pub offsets: [u8; CCSO_OFFSETS],
}

impl Default for CcsoPlane {
    fn default() -> Self {
        Self {
            enabled: false,
            reuse: false,
            block_reuse: false,
            reference: 0,
            band_only: false,
            scale_index: 0,
            quant_index: 0,
            filter_support: 0,
            edge_classifier: false,
            max_band_log2: 0,
            offsets: [0; CCSO_OFFSETS],
        }
    }
}

/// CCSO parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ccso {
    /// CCSO is on in the frame.
    pub enabled: bool,
    /// Each plane's parameters.
    pub planes: [CcsoPlane; PLANES],
    /// Filter block side in 64-sample units.
    pub block64: u8,
}

/// Bits of a reference rank among `count` references (dav2d `n_bits` of
/// restoration and CCSO reuse).
pub(super) fn rank_bits(count: u8) -> u32 {
    match count {
        0..=2 => u32::from(count.saturating_sub(1)),
        _ => 1 + u32::from(count - 1).ilog2(),
    }
}

/// Parse deblocking, GDF, CDEF, loop restoration and CCSO.
pub(super) fn parse(parser: &mut Parser<'_, '_, '_, '_>, header: &mut FrameHeader) -> Result<()> {
    deblock(parser, header)?;
    gdf(parser, header)?;
    cdef(parser, header)?;
    restoration::parse(parser, header)?;
    ccso(parser, header)?;
    Ok(())
}

fn deblock(parser: &mut Parser<'_, '_, '_, '_>, header: &mut FrameHeader) -> Result<()> {
    if !header.lossless.all {
        let bits = &mut *parser.bits;
        let d = &mut header.deblock;
        if header.frame_type == FrameType::Inter && parser.context.sequence.inter.deblock_sub_pu {
            d.sub_pu = bits.read_bit()?;
        }
        // A multi-frame header may set the switches for its frames (AVM
        // `setup_loopfilter`).
        let named = parser.multi_frame_header.and_then(|named| named.deblock);
        let mut flag = |index: usize| match named {
            Some(flags) => Ok(flags[index]),
            None => bits.read_bit(),
        };
        d.level_y = [flag(0)?, flag(1)?];
        if !parser.context.sequence.monochrome && (d.level_y[0] || d.level_y[1]) {
            d.level_u = flag(2)?;
            d.level_v = flag(3)?;
        }
        let width = u32::from(parser.context.sequence.filters.deblock_param_bits);
        let offset = 1_i32 << (width - 1);
        let value = |bits: &mut nc_common::decode::BitReader<'_>| -> Result<i8> {
            Ok((bits.read_bits(width)? as i32 - offset) as i8)
        };
        if d.level_y[0] && bits.read_bit()? {
            d.delta_q_y[0] = value(bits)?;
        }
        if d.level_y[1] {
            d.delta_q_y[1] = if bits.read_bit()? {
                value(bits)?
            } else {
                d.delta_q_y[0]
            };
        }
        if d.level_u && bits.read_bit()? {
            d.delta_q_u = value(bits)?;
        }
        if d.level_v && bits.read_bit()? {
            d.delta_q_v = value(bits)?;
        }
    }
    parser.checkpoint("post-deblock");
    Ok(())
}

fn gdf(parser: &mut Parser<'_, '_, '_, '_>, header: &mut FrameHeader) -> Result<()> {
    let sequence = parser.context.sequence;
    let large = header.superblock_log2 == super::LARGEST_SUPERBLOCK_LOG2;
    if !header.lossless.all && sequence.filters.gdf {
        let intra = header.is_intra();
        let gdf = &mut header.gdf;
        let bits = &mut *parser.bits;
        if sequence.single_picture || bits.read_bit()? {
            gdf.mode = 1;
            let block = GDF_BLOCK << u32::from(large);
            if header.width.max(header.height) > block {
                gdf.mode += u8::from(bits.read_bit()?);
            }
            let base = if intra {
                GDF_INTRA_QP_BASE
            } else {
                GDF_INTER_QP_BASE
            };
            // The class is counted on the 8-bit quantizer scale (dav2d
            // `qp_diff`: `48 * hbd` less above 8 bits).
            let difference =
                i32::from(header.quantizer.base) - base - i32::from(sequence.qindex_depth_offset());
            let offset = bits.read_bits(GDF_FIELD_BITS)? as i32;
            // Integer division truncating toward zero, as C's.
            gdf.qp_index = (((difference - GDF_QP_OFFSET) / GDF_QP_STEP)
                .clamp(0, GDF_LAST_QP_CLASS)
                + offset) as u8;
            gdf.scale = bits.read_bits(GDF_FIELD_BITS)? as u8 + 1;
        }
        parser.checkpoint("post-gdf");
    }
    let superblock64 = header.superblock_log2 - super::SMALLEST_SUPERBLOCK_LOG2;
    let unit_is_superblock = sequence.filters.gdf_unit_is_superblock;
    header.gdf.block64 = 1 << superblock64.max(u8::from(!unit_is_superblock));
    if !unit_is_superblock && superblock64 == 0 && tile_starts_mask(header) & 1 != 0 {
        header.gdf.block64 = 1;
    }
    Ok(())
}

/// The bits of every inner tile start (in superblocks) combined by OR:
/// odd starts keep 128-sample filter blocks from straddling tiles.
fn tile_starts_mask(header: &FrameHeader) -> u16 {
    let tiles = &header.tiles;
    let inner = |starts: &[u16]| {
        starts[1..starts.len() - 1]
            .iter()
            .fold(0, |mask, &start| mask | start)
    };
    inner(&tiles.column_starts) | inner(&tiles.row_starts)
}

fn cdef(parser: &mut Parser<'_, '_, '_, '_>, header: &mut FrameHeader) -> Result<()> {
    let sequence = parser.context.sequence;
    if header.lossless.all || !sequence.filters.cdef {
        return Ok(());
    }
    if sequence.single_picture || parser.bits.read_bit()? {
        let damping = parser.bits.read_bits(CDEF_DAMPING_BITS)? as u8 + CDEF_DAMPING_BASE;
        let strengths = parser.bits.read_bits(CDEF_STRENGTH_COUNT_BITS)? as u8 + 1;
        let on_skipped = parser.tri(sequence.filters.cdef_on_skipped)?;
        let mut cdef = Cdef {
            damping,
            strengths,
            on_skipped,
            ..Cdef::default()
        };
        let bits = &mut *parser.bits;
        let strength = |bits: &mut nc_common::decode::BitReader<'_>| -> Result<u8> {
            let short = bits.read_bit()?;
            let width = CDEF_STRENGTH_BITS - CDEF_SHORT_STRENGTH_SAVING * u32::from(short);
            Ok(bits.read_bits(width)? as u8)
        };
        for set in 0..usize::from(strengths) {
            cdef.y_strength[set] = strength(bits)?;
            if !sequence.monochrome {
                cdef.uv_strength[set] = strength(bits)?;
            }
        }
        header.cdef = Some(cdef);
    }
    parser.checkpoint("post-cdef");
    Ok(())
}

fn ccso(parser: &mut Parser<'_, '_, '_, '_>, header: &mut FrameHeader) -> Result<()> {
    let sequence = parser.context.sequence;
    if !header.lossless.all && sequence.filters.ccso {
        header.ccso.enabled = sequence.single_picture || parser.bits.read_bit()?;
        if header.ccso.enabled {
            for plane in 0..sequence.planes() {
                header.ccso.planes[plane] = ccso_plane(parser, header, plane)?;
            }
        }
        parser.checkpoint("post-ccso");
    }
    let superblock64 = header.superblock_log2 - super::SMALLEST_SUPERBLOCK_LOG2;
    let block64 = &mut header.ccso.block64;
    *block64 = 1
        << if sequence.filters.ccso_unit_is_superblock {
            superblock64
        } else {
            2
        };
    if !sequence.filters.ccso_unit_is_superblock && superblock64 < 2 {
        let mask = tile_starts_mask(header);
        let block64 = &mut header.ccso.block64;
        if mask & 3 != 0 && superblock64 == 0 {
            *block64 = 2 - (mask & 1) as u8;
        } else if mask & 1 != 0 {
            *block64 = 2;
        }
    }
    Ok(())
}

/// One plane's CCSO parameters: reused from a reference or coded.
fn ccso_plane(
    parser: &mut Parser<'_, '_, '_, '_>,
    header: &FrameHeader,
    plane: usize,
) -> Result<CcsoPlane> {
    let mut p = CcsoPlane {
        enabled: parser.bits.read_bit()?,
        ..CcsoPlane::default()
    };
    if !p.enabled {
        return Ok(p);
    }
    if header.frame_type == FrameType::Inter {
        p.reuse = parser.bits.read_bit()?;
        p.block_reuse = parser.bits.read_bit()?;
        if p.reuse || p.block_reuse {
            let width = rank_bits(header.reference_count);
            if width != 0 {
                p.reference = parser.bits.read_bits(width)? as u8;
                if p.reference >= header.reference_count {
                    return Err(DecodeError::Malformed("AV2 CCSO reference"));
                }
            }
            let reference = parser.reference(header, p.reference)?;
            if !reference.ccso.planes[plane].enabled {
                return Err(DecodeError::Malformed("AV2 CCSO reuse of a disabled plane"));
            }
            let units = |size: u32| (size + 7) >> UNIT_SHIFT;
            if p.block_reuse
                && (units(header.width) != units(reference.width)
                    || units(header.height) != units(reference.height))
            {
                return Err(DecodeError::Malformed("AV2 CCSO block reuse across sizes"));
            }
        }
    }
    if p.reuse {
        let reference = parser.reference(header, p.reference)?.ccso.planes[plane];
        return Ok(CcsoPlane {
            enabled: true,
            reuse: true,
            block_reuse: p.block_reuse,
            reference: p.reference,
            ..reference
        });
    }
    let bits = &mut *parser.bits;
    p.band_only = bits.read_bit()?;
    p.scale_index = bits.read_bits(CCSO_SCALE_BITS)? as u8;
    if p.band_only {
        p.max_band_log2 = bits.read_bits(CCSO_BAND_ONLY_BITS)? as u8;
        if p.max_band_log2 > CCSO_MAX_BAND_ONLY_LOG2 {
            return Err(DecodeError::Malformed("AV2 CCSO band count"));
        }
    } else {
        p.quant_index = bits.read_bits(CCSO_QUANT_BITS)? as u8;
        p.filter_support = bits.read_bits(CCSO_SUPPORT_BITS)? as u8;
        if p.filter_support == CCSO_RESERVED_SUPPORT {
            return Err(DecodeError::Malformed("AV2 CCSO filter support"));
        }
        if CCSO_QUANT_SIZE[usize::from(p.scale_index)][usize::from(p.quant_index)] != 0 {
            p.edge_classifier = bits.read_bit()?;
        }
        p.max_band_log2 = bits.read_bits(CCSO_BAND_BITS)? as u8;
    }
    let classes = if p.band_only {
        1
    } else {
        CCSO_EDGE_CLASSES - usize::from(p.edge_classifier)
    };
    let bands = 1_usize << p.max_band_log2;
    for row in 0..classes {
        for column in 0..classes {
            for band in 0..bands {
                let mut offset = 0;
                while offset < CCSO_MAX_OFFSET && bits.read_bit()? {
                    offset += 1;
                }
                p.offsets[(row * CCSO_CLASS_STRIDE + column) * CCSO_BAND_STRIDE + band] = offset;
            }
        }
    }
    Ok(p)
}
