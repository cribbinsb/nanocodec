//! The edges SAO and ALF may not read across (H.266 §8.8.4.2, §8.8.5.2):
//! slice, tile and subpicture edges whose loop filtering the PPS or SPS
//! turns off, and the picture header's virtual boundaries. The rules are
//! VTM's (`SampleAdaptiveOffset::deriveLoopFilterBoundaryAvailability`,
//! `AdaptiveLoopFilter::isCrossedByVirtualBoundaries`): a neighbouring CTB
//! may be read unless it is in another slice and
//! `pps_loop_filter_across_slices_enabled_flag` is 0, in another tile and
//! `pps_loop_filter_across_tiles_enabled_flag` is 0, or in another
//! subpicture and the current subpicture's
//! `sps_loop_filter_across_subpic_enabled_flag` is 0.
use super::{PictureParameters, Sequence, SliceHeader};
use crate::alf::Region;

/// What the filters after deblocking may read, CTB by CTB.
pub(crate) struct Closures {
    ctu: usize,
    /// The luma picture size.
    size: (usize, usize),
    columns: usize,
    /// For each CTB (raster order), which of the 3x3 CTBs around it
    /// (row-major, `[1][1]` itself) may be read; `None` when all may.
    neighbours: Vec<Option<[[bool; 3]; 3]>>,
    /// The virtual boundaries' luma columns, then rows.
    virtual_boundaries: [Vec<usize>; 2],
}

impl Closures {
    /// The closures of a picture of `size` luma samples whose CTUs belong to
    /// the slices `owners` names (by CTU address, slices by index).
    pub fn new(
        (sequence, parameters): (&Sequence, &PictureParameters),
        headers: &[SliceHeader],
        owners: &[Option<u16>],
        size: (usize, usize),
    ) -> Self {
        let ctu = 1usize << sequence.ctu_log2;
        let columns = size.0.div_ceil(ctu);
        let rows = size.1.div_ceil(ctu);
        let partitioning = &parameters.partitioning;
        let tiles = &partitioning.tiles;
        let slice = |address: usize| owners[address].map_or(0, usize::from);
        let tile =
            |address: usize| tiles.tile_at((address % columns) as u32, (address / columns) as u32);
        let subpicture = |address: usize| headers[slice(address)].subpicture;
        let layout = &sequence.subpictures.layout;
        let readable = |own: usize, other: usize| {
            (partitioning.filter_across_slices || slice(own) == slice(other))
                && (partitioning.filter_across_tiles || tile(own) == tile(other))
                && (layout[subpicture(own)].filter_across || subpicture(own) == subpicture(other))
        };
        let neighbours = (0..columns * rows)
            .map(|own| {
                let (x, y) = (own % columns, own / columns);
                let cells: [[bool; 3]; 3] = std::array::from_fn(|dy| {
                    std::array::from_fn(|dx| {
                        let (nx, ny) = ((x + dx).wrapping_sub(1), (y + dy).wrapping_sub(1));
                        nx >= columns || ny >= rows || readable(own, ny * columns + nx)
                    })
                });
                cells.iter().flatten().any(|&open| !open).then_some(cells)
            })
            .collect();
        let boundaries = headers[0]
            .picture
            .boundaries
            .each_ref()
            .map(|positions| positions.iter().map(|&p| p as usize).collect());
        Self {
            ctu,
            size,
            columns,
            neighbours,
            virtual_boundaries: boundaries,
        }
    }

    /// Which of the 3x3 CTBs around CTB `ctb` may be read.
    pub fn neighbours(&self, ctb: usize) -> [[bool; 3]; 3] {
        self.neighbours[ctb].unwrap_or([[true; 3]; 3])
    }

    /// The virtual boundaries in samples of a plane subsampled by
    /// `shift`: columns, then rows.
    pub fn cuts(&self, shift: usize) -> [Vec<usize>; 2] {
        self.virtual_boundaries
            .each_ref()
            .map(|positions| positions.iter().map(|&p| p >> shift).collect())
    }

    /// The regions ALF filters CTB `ctb` in when anything closes its edges
    /// or a virtual boundary crosses it (VTM's split at virtual boundaries,
    /// with the CTB's own closures on its outer edges); `None` when it
    /// filters as part of the picture.
    pub fn alf_regions(&self, ctb: usize) -> Option<Vec<Region>> {
        let (x0, y0) = (ctb % self.columns * self.ctu, ctb / self.columns * self.ctu);
        let (x1, y1) = (
            (x0 + self.ctu).min(self.size.0),
            (y0 + self.ctu).min(self.size.1),
        );
        let cells = self.neighbours(ctb);
        // Splits strictly inside the CTB; boundaries on its edges close them.
        let inside = |positions: &[usize], start: usize, end: usize| -> Vec<usize> {
            let mut cuts: Vec<usize> = positions
                .iter()
                .copied()
                .filter(|&p| start < p && p < end)
                .collect();
            cuts.sort_unstable();
            cuts.dedup();
            cuts
        };
        let on = |positions: &[usize], at: usize| positions.contains(&at);
        let [columns, rows] = &self.virtual_boundaries;
        let closed = [
            !cells[1][0] || on(columns, x0),
            !cells[0][1] || on(rows, y0),
            !cells[1][2] || on(columns, x1),
            !cells[2][1] || on(rows, y1),
        ];
        let corners = [
            !closed[0] && !closed[1] && !cells[0][0],
            !closed[2] && !closed[3] && !cells[2][2],
        ];
        let (splits_x, splits_y) = (inside(columns, x0, x1), inside(rows, y0, y1));
        if !closed.contains(&true)
            && !corners.contains(&true)
            && splits_x.is_empty()
            && splits_y.is_empty()
        {
            return None;
        }
        let bands = |start: usize, cuts: &[usize], end: usize| -> Vec<(usize, usize)> {
            let edges: Vec<usize> = std::iter::once(start)
                .chain(cuts.iter().copied())
                .chain(std::iter::once(end))
                .collect();
            edges.windows(2).map(|pair| (pair[0], pair[1])).collect()
        };
        let (horizontal, vertical) = (bands(x0, &splits_x, x1), bands(y0, &splits_y, y1));
        let mut regions = Vec::with_capacity(horizontal.len() * vertical.len());
        for (i, &(top, bottom)) in vertical.iter().enumerate() {
            for (j, &(left, right)) in horizontal.iter().enumerate() {
                let (first, last) = (
                    (i == 0, j == 0),
                    (i + 1 == vertical.len(), j + 1 == horizontal.len()),
                );
                regions.push(Region {
                    rect: (left, top, right - left, bottom - top),
                    closed: [
                        !first.1 || closed[0],
                        !first.0 || closed[1],
                        !last.1 || closed[2],
                        !last.0 || closed[3],
                    ],
                    corners: [
                        first.0 && first.1 && corners[0],
                        last.0 && last.1 && corners[1],
                    ],
                });
            }
        }
        Some(regions)
    }
}
