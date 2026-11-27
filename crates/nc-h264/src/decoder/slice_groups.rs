//! Slice groups (flexible macroblock ordering, Baseline and Extended
//! profiles): the slice group syntax of the PPS (§7.3.2.2, §7.4.2.2), the
//! size of `slice_group_change_cycle` (eq. 7-35) and the macroblock to
//! slice group map (§8.2.2) whose `NextMbAddress` (eq. 8-16) orders the
//! macroblocks of each slice.
//!
//! A picture without slice groups has no map: its macroblocks follow in
//! raster order, so the macroblock loop of the common streams is unchanged.

use super::reader::Reader;
use super::sps::{MAX_FRAME_MBS, Sps};
use super::{DecodeError, Result};

/// Largest `num_slice_groups_minus1` of any profile (§A.2.1, §A.2.3).
const MAX_SLICE_GROUPS_MINUS1: u32 = 7;
/// Largest `slice_group_map_type` (§7.4.2.2).
const MAX_MAP_TYPE: u32 = 6;
/// Largest map-unit address any PPS field may hold.
const MAX_MAP_UNIT: u32 = MAX_FRAME_MBS as u32 - 1;

/// How map types 3..=5 grow slice group 0 (Table 8-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Growth {
    /// 3: box-out from the centre (§8.2.2.4).
    BoxOut,
    /// 4: raster scan (§8.2.2.5).
    Raster,
    /// 5: wipe, column by column (§8.2.2.6).
    Wipe,
}

/// `slice_group_map_type` and its parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum MapType {
    /// 0: `run_length_minus1 + 1` map units per group, in turn (§8.2.2.1).
    Interleaved(Vec<u32>),
    /// 1: dispersed, a checkerboard-like pattern (§8.2.2.2).
    Dispersed,
    /// 2: (`top_left`, `bottom_right`) rectangles of every group but the
    /// last, which holds the rest (§8.2.2.3).
    Foreground(Vec<(u32, u32)>),
    /// 3..=5: slice group 0 grows by `slice_group_change_cycle` steps.
    Changing {
        growth: Growth,
        /// `slice_group_change_direction_flag`: the refined type's reverse
        /// direction (Table 8-1).
        reverse: bool,
        /// `SliceGroupChangeRate` (eq. 7-23).
        rate: u32,
    },
    /// 6: `slice_group_id` of every map unit (§8.2.2.7).
    Explicit(Vec<u8>),
}

/// The slice groups of a PPS with more than one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SliceGroups {
    /// `num_slice_groups_minus1 + 1`.
    count: u32,
    map_type: MapType,
}

/// The size of a picture in slice group map units (§7.4.2.2).
#[derive(Debug, Clone, Copy)]
pub(super) struct MapUnits {
    width: usize,
    /// `PicHeightInMapUnits`.
    height: usize,
    /// Map units of two vertically adjacent macroblocks (frames of a
    /// stream with `frame_mbs_only_flag` 0, eq. 8-26).
    pairs: bool,
}

impl MapUnits {
    /// The map units of a frame of `sps`'s sequence.
    pub fn of_frame(sps: &Sps) -> Self {
        let pairs = !sps.frame_mbs_only;
        Self {
            width: sps.width_mbs,
            height: sps.height_mbs >> usize::from(pairs),
            pairs,
        }
    }

    /// `PicSizeInMapUnits`.
    fn count(self) -> usize {
        self.width * self.height
    }
}

/// Parse the slice group fields of a PPS after `num_slice_groups_minus1`;
/// `None` for a single slice group.
pub(super) fn parse(r: &mut Reader<'_>) -> Result<Option<SliceGroups>> {
    let count = r.ue_max(MAX_SLICE_GROUPS_MINUS1, "num_slice_groups_minus1")? + 1;
    if count == 1 {
        return Ok(None);
    }
    let map_type = match r.ue_max(MAX_MAP_TYPE, "slice_group_map_type")? {
        0 => MapType::Interleaved(
            (0..count)
                .map(|_| Ok(r.ue_max(MAX_MAP_UNIT, "run_length_minus1")? + 1))
                .collect::<Result<_>>()?,
        ),
        1 => MapType::Dispersed,
        2 => MapType::Foreground(
            (1..count)
                .map(|_| {
                    let top_left = r.ue_max(MAX_MAP_UNIT, "top_left")?;
                    let bottom_right = r.ue_max(MAX_MAP_UNIT, "bottom_right")?;
                    Ok((top_left, bottom_right))
                })
                .collect::<Result<_>>()?,
        ),
        changing @ 3..=5 => {
            if count != 2 {
                return Err(DecodeError::Malformed(
                    "changing slice groups need two groups",
                ));
            }
            let growth = [Growth::BoxOut, Growth::Raster, Growth::Wipe][changing as usize - 3];
            let reverse = r.flag()?;
            let rate = r.ue_max(MAX_MAP_UNIT, "slice_group_change_rate_minus1")? + 1;
            MapType::Changing {
                growth,
                reverse,
                rate,
            }
        }
        _ => {
            let units = r.ue_max(MAX_MAP_UNIT, "pic_size_in_map_units_minus1")? + 1;
            // Ceil(Log2(num_slice_groups_minus1 + 1)) bits each.
            let bits = u32::BITS - (count - 1).leading_zeros();
            let ids = (0..units)
                .map(|_| {
                    let id = r.bits(bits)?;
                    if id >= count {
                        return Err(DecodeError::Malformed("slice_group_id"));
                    }
                    Ok(id as u8)
                })
                .collect::<Result<_>>()?;
            MapType::Explicit(ids)
        }
    };
    Ok(Some(SliceGroups { count, map_type }))
}

impl SliceGroups {
    /// The bits of `slice_group_change_cycle` in a slice header (eq. 7-35:
    /// `Ceil(Log2(PicSizeInMapUnits ÷ SliceGroupChangeRate + 1))`, an exact
    /// division), or `None` when the map type has no such field.
    pub fn change_cycle_bits(&self, units: MapUnits) -> Option<u32> {
        let MapType::Changing { rate, .. } = self.map_type else {
            return None;
        };
        // The least `bits` with 2^bits >= units / rate + 1.
        let (units, rate) = (units.count() as u64, u64::from(rate));
        Some(
            (0..u64::BITS)
                .find(|&bits| (rate << bits) >= units + rate)
                .unwrap_or(0),
        )
    }

    /// `MbToSliceGroupMap` (§8.2.2) of a frame (or, `field_picture`, a
    /// field) with `units` map units and `slice_group_change_cycle` `cycle`.
    pub fn mb_map(&self, units: MapUnits, cycle: u32, field_picture: bool) -> Result<Box<[u8]>> {
        let map = self.map_unit_map(units, cycle)?;
        // §8.2.2.8: eq. 8-24, 8-25 for a field (a map unit is one of its
        // macroblocks), or 8-26 for a frame whose map units are pairs.
        if !units.pairs || field_picture {
            return Ok(map.into_boxed_slice());
        }
        let width = units.width;
        Ok((0..2 * map.len())
            .map(|i| map[(i / (2 * width)) * width + i % width])
            .collect())
    }

    /// `mapUnitToSliceGroupMap` (§8.2.2.1..=8.2.2.7).
    fn map_unit_map(&self, units: MapUnits, cycle: u32) -> Result<Vec<u8>> {
        let size = units.count();
        let width = units.width;
        let last_group = (self.count - 1) as u8;
        Ok(match &self.map_type {
            MapType::Interleaved(runs) => {
                // eq. 8-17: the groups' runs in turn until the picture is full.
                runs.iter()
                    .enumerate()
                    .flat_map(|(group, &run)| std::iter::repeat_n(group as u8, run as usize))
                    .cycle()
                    .take(size)
                    .collect()
            }
            MapType::Dispersed => {
                let groups = self.count as usize;
                // eq. 8-18.
                (0..size)
                    .map(|i| ((i % width + (i / width) * groups / 2) % groups) as u8)
                    .collect()
            }
            MapType::Foreground(rects) => {
                let mut map = vec![last_group; size];
                // eq. 8-19: lower groups last, so they win where rectangles
                // overlap.
                for (group, &(top_left, bottom_right)) in rects.iter().enumerate().rev() {
                    let (top_left, bottom_right) = (top_left as usize, bottom_right as usize);
                    let (x0, y0) = (top_left % width, top_left / width);
                    let (x1, y1) = (bottom_right % width, bottom_right / width);
                    if top_left > bottom_right || bottom_right >= size || x0 > x1 {
                        return Err(DecodeError::Malformed("slice group rectangle"));
                    }
                    for y in y0..=y1 {
                        map[y * width + x0..=y * width + x1].fill(group as u8);
                    }
                }
                map
            }
            &MapType::Changing {
                growth,
                reverse,
                rate,
            } => {
                // eq. 7-34.
                let in_group0 = (u64::from(cycle) * u64::from(rate)).min(size as u64) as usize;
                match growth {
                    Growth::BoxOut => box_out(units, in_group0, reverse),
                    Growth::Raster | Growth::Wipe => {
                        // eq. 8-14.
                        let upper_left = if reverse { size - in_group0 } else { in_group0 };
                        let (first, second) = (u8::from(reverse), u8::from(!reverse));
                        let mut map = vec![second; size];
                        if growth == Growth::Raster {
                            // eq. 8-21.
                            map[..upper_left].fill(first);
                        } else {
                            // eq. 8-22: column by column.
                            for k in 0..upper_left {
                                map[(k % units.height) * width + k / units.height] = first;
                            }
                        }
                        map
                    }
                }
            }
            MapType::Explicit(ids) => {
                if ids.len() != size {
                    return Err(DecodeError::Malformed("pic_size_in_map_units_minus1"));
                }
                ids.clone()
            }
        })
    }
}

/// The box-out map (§8.2.2.4, eq. 8-20): slice group 0 spirals out from
/// the centre, clockwise (or counter-clockwise when `reverse`), over
/// `in_group0` map units.
fn box_out(units: MapUnits, in_group0: usize, reverse: bool) -> Vec<u8> {
    let (width, height) = (units.width as i64, units.height as i64);
    let direction = i64::from(reverse);
    let mut map = vec![1u8; units.count()];
    let (mut x, mut y) = ((width - direction) / 2, (height - direction) / 2);
    let (mut left, mut top, mut right, mut bottom) = (x, y, x, y);
    let (mut x_dir, mut y_dir) = (direction - 1, direction);
    let mut k = 0;
    while k < in_group0 {
        let unit = &mut map[(y * width + x) as usize];
        let vacant = *unit == 1;
        if vacant {
            *unit = 0;
        }
        if x_dir == -1 && x == left {
            left = (left - 1).max(0);
            x = left;
            (x_dir, y_dir) = (0, 2 * direction - 1);
        } else if x_dir == 1 && x == right {
            right = (right + 1).min(width - 1);
            x = right;
            (x_dir, y_dir) = (0, 1 - 2 * direction);
        } else if y_dir == -1 && y == top {
            top = (top - 1).max(0);
            y = top;
            (x_dir, y_dir) = (1 - 2 * direction, 0);
        } else if y_dir == 1 && y == bottom {
            bottom = (bottom + 1).min(height - 1);
            y = bottom;
            (x_dir, y_dir) = (2 * direction - 1, 0);
        } else {
            (x, y) = (x + x_dir, y + y_dir);
        }
        k += usize::from(vacant);
    }
    map
}

/// `NextMbAddress(addr)` (eq. 8-16): the next macroblock of `addr`'s slice
/// group, or the picture size when there is none.
pub(super) fn next_mb(map: &[u8], addr: usize) -> usize {
    let group = map[addr];
    map[addr + 1..]
        .iter()
        .position(|&g| g == group)
        .map_or(map.len(), |offset| addr + 1 + offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    const QCIF: MapUnits = MapUnits {
        width: 11,
        height: 9,
        pairs: false,
    };

    fn groups(count: u32, map_type: MapType) -> SliceGroups {
        SliceGroups { count, map_type }
    }

    #[test]
    fn box_out_starts_at_the_centre_then_turns() {
        let changing = MapType::Changing {
            growth: Growth::BoxOut,
            reverse: false,
            rate: 1,
        };
        let map = groups(2, changing).mb_map(QCIF, 3, false).unwrap();
        // Centre (5, 4), left to (4, 4), then up to (4, 3).
        let zeros: Vec<usize> = (0..map.len()).filter(|&i| map[i] == 0).collect();
        assert_eq!(zeros, [3 * 11 + 4, 4 * 11 + 4, 4 * 11 + 5]);
    }

    #[test]
    fn wipe_fills_columns_and_next_mb_follows_the_group() {
        let changing = MapType::Changing {
            growth: Growth::Wipe,
            reverse: false,
            rate: 2,
        };
        let map = groups(2, changing).mb_map(QCIF, 5, false).unwrap();
        // Ten map units: column 0 (9) and the top of column 1.
        assert_eq!(map.iter().map(|&g| usize::from(g == 0)).sum::<usize>(), 10);
        assert_eq!(map[1], 0);
        assert_eq!(map[12], 1);
        assert_eq!(next_mb(&map, 0), 1);
        assert_eq!(next_mb(&map, 1), 11);
        assert_eq!(next_mb(&map, 2), 3);
        assert_eq!(next_mb(&map, map.len() - 1), map.len());
    }

    #[test]
    fn change_cycle_bits_is_an_exact_division() {
        let changing = |rate| {
            groups(
                2,
                MapType::Changing {
                    growth: Growth::Raster,
                    reverse: false,
                    rate,
                },
            )
        };
        // 99 / 1 + 1 = 100 -> 7 bits; 99 / 2 + 1 = 50.5 -> 6 bits;
        // 99 / 99 + 1 = 2 -> 1 bit.
        assert_eq!(changing(1).change_cycle_bits(QCIF), Some(7));
        assert_eq!(changing(2).change_cycle_bits(QCIF), Some(6));
        assert_eq!(changing(99).change_cycle_bits(QCIF), Some(1));
    }

    #[test]
    fn foreground_lower_groups_win_overlaps() {
        let map = groups(3, MapType::Foreground(vec![(12, 13), (0, 24)]))
            .mb_map(QCIF, 0, false)
            .unwrap();
        assert_eq!(&map[..3], &[1, 1, 1]);
        assert_eq!(&map[11..15], &[1, 0, 0, 2]);
        assert_eq!(map[25], 2);
    }

    #[test]
    fn frame_pairs_repeat_each_map_unit_row() {
        let units = MapUnits {
            width: 2,
            height: 2,
            pairs: true,
        };
        let map = groups(2, MapType::Interleaved(vec![1, 2]))
            .mb_map(units, 0, false)
            .unwrap();
        // Map units 0 1 / 1 0, each row twice.
        assert_eq!(&*map, &[0, 1, 0, 1, 1, 0, 1, 0]);
    }
}
