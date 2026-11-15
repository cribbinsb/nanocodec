//! H.266 §7.3.2.4 subpicture layout of a sequence parameter set: each
//! subpicture's CTU rectangle, whether it is treated as a picture in inter
//! prediction, whether the loop filters cross its edges, and its ID.
//! Inferences checked against BSD-licensed VTM; see `THIRD_PARTY.md`.
use super::{DecodeError, Reader, Result};

/// Largest `sps_subpic_id_len_minus1 + 1` (§7.4.3.4).
const MAX_ID_BITS: u32 = 16;

/// One subpicture: a rectangle of CTUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Subpicture {
    /// Top-left CTU column and row, width and height in CTUs.
    pub ctus: [u32; 4],
    /// `sps_subpic_treated_as_pic_flag`: motion compensation and temporal
    /// motion stay inside the subpicture.
    pub treated_as_picture: bool,
    /// `sps_loop_filter_across_subpic_enabled_flag`.
    pub filter_across: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Subpictures {
    /// At least one; a sequence without subpicture information has one
    /// covering the picture.
    pub layout: Vec<Subpicture>,
    /// `sps_subpic_id_len_minus1 + 1`; zero without subpicture information
    /// (no `sh_subpic_id` is coded).
    pub id_bits: u32,
    /// `sps_subpic_id_mapping_explicitly_signalled_flag`.
    pub explicit_ids: bool,
    /// The SPS's IDs by subpicture index (`SubpicIdVal` unless a PPS
    /// overrides them).
    pub ids: Vec<u32>,
}
impl Subpictures {
    /// The layout of a sequence without subpicture information.
    pub fn whole(grid: [u32; 2]) -> Self {
        Self {
            layout: vec![Subpicture {
                ctus: [0, 0, grid[0], grid[1]],
                treated_as_picture: true,
                filter_across: false,
            }],
            id_bits: 0,
            explicit_ids: false,
            ids: vec![0],
        }
    }

    /// `r` after `sps_conformance_window_flag`'s syntax; `grid` is the
    /// largest picture in CTUs, `ctu_log2` and `max_size` its CTU and luma
    /// sizes (each coordinate is coded only when the picture holds more
    /// than one CTU along that axis).
    pub fn parse(
        r: &mut Reader<'_>,
        grid: [u32; 2],
        max_size: [u32; 2],
        ctu_log2: u8,
    ) -> Result<Self> {
        if !r.flag()? {
            return Ok(Self::whole(grid));
        }
        let count = r.ue(grid[0] * grid[1] - 1, "VVC subpicture count")? as usize + 1;
        let mut layout = Vec::with_capacity(count);
        if count == 1 {
            layout = Self::whole(grid).layout;
        } else {
            let independent = r.flag()?;
            let same_size = r.flag()?;
            let coded = max_size.map(|size| size > 1 << ctu_log2);
            let bits = grid.map(|cells| u32::BITS - (cells - 1).leading_zeros());
            for index in 0..count {
                let mut ctus = [0; 4];
                if same_size && index > 0 {
                    let first: Subpicture = layout[0];
                    let columns = grid[0] / first.ctus[2];
                    ctus = [
                        (index as u32 % columns) * first.ctus[2],
                        (index as u32 / columns) * first.ctus[3],
                        first.ctus[2],
                        first.ctus[3],
                    ];
                } else {
                    for axis in 0..2 {
                        if index > 0 && coded[axis] {
                            ctus[axis] = r.bits(bits[axis])?;
                        }
                    }
                    for axis in 0..2 {
                        ctus[axis + 2] = if index + 1 < count && coded[axis] {
                            r.bits(bits[axis])? + 1
                        } else {
                            grid[axis].saturating_sub(ctus[axis])
                        };
                    }
                    if same_size
                        && (!grid[0].is_multiple_of(ctus[2])
                            || !grid[1].is_multiple_of(ctus[3])
                            || (grid[0] / ctus[2]) * (grid[1] / ctus[3]) != count as u32)
                    {
                        return Err(DecodeError::Malformed("VVC same-size subpictures"));
                    }
                }
                if ctus[2] == 0
                    || ctus[3] == 0
                    || ctus[0] + ctus[2] > grid[0]
                    || ctus[1] + ctus[3] > grid[1]
                {
                    return Err(DecodeError::Malformed("VVC subpicture outside picture"));
                }
                let (treated_as_picture, filter_across) = if independent {
                    (true, false)
                } else {
                    (r.flag()?, r.flag()?)
                };
                layout.push(Subpicture {
                    ctus,
                    treated_as_picture,
                    filter_across,
                });
            }
            check_tiling(&layout, grid)?;
        }
        let id_bits = r.ue(MAX_ID_BITS - 1, "VVC subpicture ID length")? + 1;
        if (1u64 << id_bits) < count as u64 {
            return Err(DecodeError::Malformed("VVC subpicture ID length"));
        }
        let explicit_ids = r.flag()?;
        let ids = if explicit_ids && r.flag()? {
            (0..count)
                .map(|_| r.bits(id_bits))
                .collect::<Result<Vec<_>>>()?
        } else {
            (0..count as u32).collect()
        };
        Ok(Self {
            layout,
            id_bits,
            explicit_ids,
            ids,
        })
    }

    /// The index of the subpicture whose ID is `id`, with `ids` the IDs in
    /// force (the PPS's when it signals them, §7.4.3.5).
    pub fn index_of(ids: &[u32], id: u32) -> Result<usize> {
        ids.iter()
            .position(|&value| value == id)
            .ok_or(DecodeError::Malformed("VVC unknown subpicture ID"))
    }

    /// The index of the subpicture holding CTU (`x`, `y`).
    pub fn at(&self, x: u32, y: u32) -> usize {
        self.layout
            .iter()
            .position(|s| {
                (s.ctus[0]..s.ctus[0] + s.ctus[2]).contains(&x)
                    && (s.ctus[1]..s.ctus[1] + s.ctus[3]).contains(&y)
            })
            .unwrap_or(0)
    }
}

/// The subpictures must cover every CTU of the picture exactly once
/// (§7.4.3.4).
fn check_tiling(layout: &[Subpicture], grid: [u32; 2]) -> Result<()> {
    let mut covered = vec![false; (grid[0] * grid[1]) as usize];
    for subpicture in layout {
        let [x0, y0, width, height] = subpicture.ctus;
        for y in y0..y0 + height {
            for x in x0..x0 + width {
                let cell = &mut covered[(y * grid[0] + x) as usize];
                if *cell {
                    return Err(DecodeError::Malformed("VVC overlapping subpictures"));
                }
                *cell = true;
            }
        }
    }
    if covered.contains(&false) {
        return Err(DecodeError::Malformed("VVC subpictures leave a gap"));
    }
    Ok(())
}
