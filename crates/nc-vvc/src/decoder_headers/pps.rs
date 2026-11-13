//! H.266 §7.3.2.5 picture parameter sets for single-subpicture pictures.
use super::{DecodeError, Reader, Result, Sequence};

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // Independent flags mandated by PPS syntax.
pub(crate) struct PictureParameters {
    pub partitioning: Partitioning,
    pub deblock: Deblocking,
    pub location: HeaderLocation,
    pub id: u8,
    pub sequence_id: u8,
    pub mixed_nal_types: bool,
    pub dimensions: [u32; 2],
    pub crop: [u32; 4],
    pub scaling_window: [i32; 4],
    pub output_flag: bool,
    pub no_partition: bool,
    pub cabac_init: bool,
    pub default_active: [u8; 2],
    pub rpl1_index: bool,
    pub weighted: [bool; 2],
    pub wraparound: Option<u32>,
    pub initial_qp: i32,
    pub cu_qp_delta: bool,
    pub chroma_offsets_present: bool,
    pub chroma_offsets: [i8; 3],
    pub joint_offset_present: bool,
    pub slice_chroma_offsets: bool,
    pub cu_chroma_offsets: Vec<[i8; 3]>,
    pub ph_extension: bool,
    pub sh_extension: bool,
}
impl PictureParameters {
    pub fn parse(bytes: &[u8], s: &Sequence) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let id = r.bits(6)? as u8;
        let sequence_id = r.bits(4)? as u8;
        if sequence_id != s.id {
            return Err(DecodeError::Malformed("VVC PPS references missing SPS"));
        }
        let mixed_nal_types = r.flag()?;
        let Geometry {
            dimensions,
            crop,
            scaling_window,
        } = geometry(&mut r, s)?;
        let output_flag = r.flag()?;
        let no_partition = r.flag()?;
        if no_partition && mixed_nal_types {
            return Err(DecodeError::Malformed(
                "VVC mixed NAL picture without partitions",
            ));
        }
        if r.flag()? {
            return Err(DecodeError::Malformed(
                "VVC PPS subpicture mapping absent from SPS",
            ));
        }
        let grid = dimensions.map(|v| v.div_ceil(1 << s.ctu_log2));
        let partitioning = partition(&mut r, s, grid, no_partition)?;
        let cabac_init = r.flag()?;
        let default_active = [
            r.ue(14, "VVC active L0 references")? as u8 + 1,
            r.ue(14, "VVC active L1 references")? as u8 + 1,
        ];
        let rpl1_index = r.flag()?;
        let weighted = [r.flag()?, r.flag()?];
        if weighted
            .iter()
            .zip(s.tools.weighted)
            .any(|(&enabled, permitted)| enabled && !permitted)
        {
            return Err(DecodeError::Malformed("VVC PPS weighted prediction"));
        }
        let wraparound = wraparound(&mut r, s, dimensions[0])?;
        let initial_qp = 26 + r.se(-26, 37, "VVC PPS initial QP")?;
        let cu_qp_delta = r.flag()?;
        let chroma_offsets_present = r.flag()?;
        if chroma_offsets_present && !s.chroma {
            return Err(DecodeError::Malformed("VVC monochrome chroma offsets"));
        }
        let ChromaOffsets {
            offsets: chroma_offsets,
            joint: joint_offset_present,
            slice: slice_chroma_offsets,
            list: cu_chroma_offsets,
        } = chroma(&mut r, s, chroma_offsets_present)?;
        let deblock = deblocking(&mut r, no_partition, chroma_offsets_present)?;
        let location = header_location(&mut r, no_partition, weighted)?;
        let ph_extension = r.flag()?;
        let sh_extension = r.flag()?;
        let extension = r.flag()?;
        r.finish(extension)?;
        Ok(Self {
            partitioning,
            deblock,
            location,
            id,
            sequence_id,
            mixed_nal_types,
            dimensions,
            crop,
            scaling_window,
            output_flag,
            no_partition,
            cabac_init,
            default_active,
            rpl1_index,
            weighted,
            wraparound,
            initial_qp,
            cu_qp_delta,
            chroma_offsets_present,
            chroma_offsets,
            joint_offset_present,
            slice_chroma_offsets,
            cu_chroma_offsets,
            ph_extension,
            sh_extension,
        })
    }
}

fn offsets(r: &mut Reader<'_>) -> Result<[i8; 2]> {
    Ok([
        r.se(-12, 12, "VVC deblock beta offset")? as i8,
        r.se(-12, 12, "VVC deblock tc offset")? as i8,
    ])
}
fn chroma(r: &mut Reader<'_>, s: &Sequence, present: bool) -> Result<ChromaOffsets> {
    let mut offsets = [0; 3];
    if !present {
        return Ok(ChromaOffsets {
            offsets,
            joint: false,
            slice: false,
            list: Vec::new(),
        });
    }
    offsets[0] = r.se(-12, 12, "VVC Cb QP offset")? as i8;
    offsets[1] = r.se(-12, 12, "VVC Cr QP offset")? as i8;
    let joint = r.flag()?;
    if joint && !s.transforms.joint_cbcr {
        return Err(DecodeError::Malformed(
            "VVC joint QP offset without joint transform",
        ));
    }
    if joint {
        offsets[2] = r.se(-12, 12, "VVC joint QP offset")? as i8;
    }
    let slice = r.flag()?;
    let mut list = Vec::new();
    if r.flag()? {
        let count = r.ue(5, "VVC chroma QP offset list")? + 1;
        for _ in 0..count {
            list.push([
                r.se(-12, 12, "VVC Cb QP offset list")? as i8,
                r.se(-12, 12, "VVC Cr QP offset list")? as i8,
                if joint {
                    r.se(-12, 12, "VVC joint QP offset list")? as i8
                } else {
                    0
                },
            ]);
        }
    }
    Ok(ChromaOffsets {
        offsets,
        joint,
        slice,
        list,
    })
}

fn partition(
    r: &mut Reader<'_>,
    s: &Sequence,
    grid: [u32; 2],
    no_partition: bool,
) -> Result<Partitioning> {
    if no_partition {
        return Ok(Partitioning {
            tiles: [vec![grid[0]], vec![grid[1]]],
            filter_across_tiles: false,
            rectangular_slices: true,
            single_slice: true,
            filter_across_slices: false,
        });
    }
    if r.bits(2)? + 5 != u32::from(s.ctu_log2) {
        return Err(DecodeError::Malformed("VVC PPS CTU size differs from SPS"));
    }
    let counts = [
        r.ue(grid[0] - 1, "VVC tile columns")? + 1,
        r.ue(grid[1] - 1, "VVC tile rows")? + 1,
    ];
    let mut tiles: [Vec<u32>; 2] = std::array::from_fn(|_| Vec::new());
    for axis in 0..2 {
        let mut remaining = grid[axis];
        for _ in 0..counts[axis] {
            let size = r.ue(grid[axis] - 1, "VVC tile extent")? + 1;
            remaining = remaining
                .checked_sub(size)
                .ok_or(DecodeError::Malformed("VVC tiles exceed picture"))?;
            tiles[axis].push(size);
        }
        let repeated = *tiles[axis].last().unwrap();
        while remaining > 0 {
            let size = remaining.min(repeated);
            tiles[axis].push(size);
            remaining -= size;
        }
    }
    let multiple = tiles.iter().map(Vec::len).product::<usize>() > 1;
    let filter_tiles = multiple && r.flag()?;
    let rect = !multiple || r.flag()?;
    let single = rect && r.flag()?;
    if rect && !single && r.ue(599, "VVC slice count")? != 0 {
        return Err(DecodeError::Unsupported("VVC rectangular multi-slice PPS"));
    }
    let filter_slices = (!rect || single) && r.flag()?;
    Ok(Partitioning {
        tiles,
        filter_across_tiles: filter_tiles,
        rectangular_slices: rect,
        single_slice: single,
        filter_across_slices: filter_slices,
    })
}

struct Geometry {
    dimensions: [u32; 2],
    crop: [u32; 4],
    scaling_window: [i32; 4],
}
fn geometry(r: &mut Reader<'_>, s: &Sequence) -> Result<Geometry> {
    let dimensions = [
        r.ue(s.dimensions[0], "VVC PPS width")?,
        r.ue(s.dimensions[1], "VVC PPS height")?,
    ];
    let align = 1 << s.min_cb_log2.max(3);
    if dimensions
        .iter()
        .any(|&size| size == 0 || !size.is_multiple_of(align))
        || (!s.resolution_change && dimensions != s.dimensions)
    {
        return Err(DecodeError::Malformed("VVC PPS geometry"));
    }
    let crop_present = r.flag()?;
    let mut crop = if dimensions == s.dimensions {
        s.crop
    } else {
        [0; 4]
    };
    if crop_present {
        if dimensions == s.dimensions {
            return Err(DecodeError::Malformed("VVC redundant PPS crop"));
        }
        for value in &mut crop {
            *value = r.ue(4096, "VVC PPS crop")?;
        }
    }
    let scale = if s.chroma { 2 } else { 1 };
    if scale * (crop[0] + crop[1]) >= dimensions[0] || scale * (crop[2] + crop[3]) >= dimensions[1]
    {
        return Err(DecodeError::Malformed("VVC PPS crop exceeds picture"));
    }
    let mut scaling_window = crop.map(|value| value as i32);
    if r.flag()? {
        if !s.resampling {
            return Err(DecodeError::Malformed(
                "VVC scaling window without resampling",
            ));
        }
        for (index, value) in scaling_window.iter_mut().enumerate() {
            let size = dimensions[index / 2] as i32;
            *value = r.se(
                -15 * size / scale as i32,
                (size - 1) / scale as i32,
                "VVC scaling window",
            )?;
        }
        for axis in 0..2 {
            let sum = i64::from(scale)
                * i64::from(scaling_window[axis * 2] + scaling_window[axis * 2 + 1]);
            if sum < -15 * i64::from(dimensions[axis]) || sum >= i64::from(dimensions[axis]) {
                return Err(DecodeError::Malformed("VVC scaling window sum"));
            }
        }
    }
    Ok(Geometry {
        dimensions,
        crop,
        scaling_window,
    })
}

#[allow(clippy::struct_excessive_bools)] // PPS tile and slice flags are independent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Partitioning {
    pub tiles: [Vec<u32>; 2],
    pub filter_across_tiles: bool,
    pub rectangular_slices: bool,
    pub single_slice: bool,
    pub filter_across_slices: bool,
}
struct ChromaOffsets {
    offsets: [i8; 3],
    joint: bool,
    slice: bool,
    list: Vec<[i8; 3]>,
}

fn wraparound(r: &mut Reader<'_>, s: &Sequence, width: u32) -> Result<Option<u32>> {
    let result = if r.flag()? {
        let limit = (width >> s.min_cb_log2)
            .checked_sub((1 << s.ctu_log2) >> s.min_cb_log2)
            .and_then(|v| v.checked_sub(2));
        if !s.tools.wraparound || limit.is_none() {
            return Err(DecodeError::Malformed("VVC PPS wraparound"));
        }
        let delta = r.ue(limit.unwrap(), "VVC wraparound offset")?;
        Some(width - (delta << s.min_cb_log2))
    } else {
        None
    };
    Ok(result)
}

#[allow(clippy::struct_excessive_bools)] // Independent PPS deblocking flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Deblocking {
    pub control: bool,
    pub override_enabled: bool,
    pub disabled: bool,
    pub in_ph: bool,
    pub offsets: [[i8; 2]; 3],
}
fn deblocking(
    r: &mut Reader<'_>,
    no_partition: bool,
    chroma_offsets_present: bool,
) -> Result<Deblocking> {
    let deblock_control = r.flag()?;
    let deblock_override = deblock_control && r.flag()?;
    let deblock_disabled = deblock_control && r.flag()?;
    let deblock_in_ph = deblock_override && !no_partition && r.flag()?;
    let mut deblock_offsets = [[0; 2]; 3];
    if deblock_control && !deblock_disabled {
        deblock_offsets[0] = offsets(r)?;
        deblock_offsets[1] = if chroma_offsets_present {
            offsets(r)?
        } else {
            deblock_offsets[0]
        };
        deblock_offsets[2] = if chroma_offsets_present {
            offsets(r)?
        } else {
            deblock_offsets[0]
        };
    }
    Ok(Deblocking {
        control: deblock_control,
        override_enabled: deblock_override,
        disabled: deblock_disabled,
        in_ph: deblock_in_ph,
        offsets: deblock_offsets,
    })
}

#[allow(clippy::struct_excessive_bools)] // Each tool independently selects PH versus SH syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HeaderLocation {
    pub rpl_in_ph: bool,
    pub sao_in_ph: bool,
    pub alf_in_ph: bool,
    pub weights_in_ph: bool,
    pub qp_in_ph: bool,
}
fn header_location(
    r: &mut Reader<'_>,
    no_partition: bool,
    weighted: [bool; 2],
) -> Result<HeaderLocation> {
    let rpl_in_ph = !no_partition && r.flag()?;
    let sao_in_ph = !no_partition && r.flag()?;
    let alf_in_ph = !no_partition && r.flag()?;
    let weights_in_ph = !no_partition && weighted.contains(&true) && rpl_in_ph && r.flag()?;
    let qp_in_ph = !no_partition && r.flag()?;
    Ok(HeaderLocation {
        rpl_in_ph,
        sao_in_ph,
        alf_in_ph,
        weights_in_ph,
        qp_in_ph,
    })
}
