//! The projected motion field of a frame: per superblock row, each
//! projected reference's kept vectors moved to where they cross the
//! current frame (`rp_proj`), the trajectories through every reference
//! (`rp_traj`) with the maps that link their ends (`rp_map`), and the
//! gaps of a coarser sample step filled (dav2d `refmvs.c`:
//! `dav2d_refmvs_load_tmvs`, `load_tmvs_refs_c`, `check_traj_intersect`,
//! `fill_gap_proj_c`, `fill_gap_traj_c`; dav2d commit 237229e,
//! BSD-2-Clause, see `docs/licenses/dav2d`).
//!
//! The projected field covers the whole frame (dav2d's frame-threading
//! layout: its single-threaded layout keeps one superblock row plus two
//! rows copied from the row above, which reads the same); the
//! trajectories and maps cover one superblock row.
use super::frame::{FrameMotion, INVALID_DISTANCE, MotionParams, Projection};
use super::vector::{Mv, REFERENCES, TemporalMv, project_in_field, scale_temporal, whole};

/// Field vectors are in 1/64 of an 8x8 unit: this shift gives units.
const UNIT_SHIFT: u32 = 6;
/// Each field unit's map has one entry per neighbour column group (left,
/// same, right).
const MAP_GROUPS: usize = 3;
/// Projection blocks are 8 field units at sample step 1.
const PROJECTION_BLOCK: usize = 8;
/// The gap filling works in 16-unit blocks of 2x2 samples.
const GAP_BLOCK: usize = 16;

/// Reset the sampled positions of one contiguous row.
fn clear_samples<T: Copy>(row: &mut [T], step: usize, invalid: T) {
    if step == 1 {
        row.fill(invalid);
    } else {
        for sample in row.iter_mut().step_by(step) {
            *sample = invalid;
        }
    }
}

/// Where a trajectory continues: the offset from one end to the other in
/// field units (dav2d `refmvs_traj_map`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrajectoryLink {
    y: i8,
    x: i8,
}

impl TrajectoryLink {
    const INVALID: Self = Self { y: -128, x: -128 };
}

/// The projected field, trajectories and maps.
#[derive(Debug, Clone)]
pub struct Projected {
    stride: usize,
    /// The frame in 8x8 units.
    columns8: usize,
    rows8: usize,
    /// Superblock rows are this many 8x8 units.
    superblock8: usize,
    /// Projection reach: the block (`mfmv_sbsz8`), the margin beside it
    /// (`mfmv_edge`), and the column-group shift (`mfmv_k_shift`).
    block8: usize,
    edge: usize,
    group_shift: u32,
    step: usize,
    /// The projected vectors and their reference distances, whole frame.
    vectors: Vec<TemporalMv>,
    distances: Vec<u8>,
    /// Per rank, the trajectory vector of each unit of the row.
    trajectories: Vec<Vec<TemporalMv>>,
    /// Per rank and unit, the links to the three adjacent column groups.
    links: Vec<Vec<[TrajectoryLink; MAP_GROUPS]>>,
}

impl Projected {
    /// An empty field for a frame `rows8` units high.
    #[must_use]
    pub fn new(params: &MotionParams, stride: usize, rows8: usize) -> Self {
        let superblock8 = (params.superblock_units >> 1) as usize;
        let wide = params.superblock_units > 16 && params.sample_step > 1;
        let block8 = PROJECTION_BLOCK << usize::from(wide);
        let edge = block8 >> usize::from(params.sample_step == 1);
        // Two spare rows: gap filling writes one row past an odd end.
        let rows = rows8.next_multiple_of(superblock8) + 2;
        let row_units = superblock8 * stride;
        let (trajectories, links) = if params.trajectories && params.use_motion_fields {
            (
                vec![vec![TemporalMv::INVALID; row_units]; REFERENCES],
                vec![vec![[TrajectoryLink::INVALID; MAP_GROUPS]; row_units]; REFERENCES],
            )
        } else {
            (Vec::new(), Vec::new())
        };
        Self {
            stride,
            columns8: 0,
            rows8,
            superblock8,
            block8,
            edge,
            group_shift: 3 + u32::from(wide),
            step: params.sample_step as usize,
            vectors: vec![TemporalMv::INVALID; rows * stride],
            distances: vec![0; rows * stride],
            trajectories,
            links,
        }
    }

    /// A projected vector and its reference distance at an 8x8 unit of
    /// the frame.
    #[must_use]
    pub fn projected(&self, row8: usize, column8: usize) -> (TemporalMv, u8) {
        let at = row8 * self.stride + column8;
        (self.vectors[at], self.distances[at])
    }

    /// The vectors, row-major with [`Projected::stride`].
    pub(super) fn vectors_mut(&mut self) -> (&mut [TemporalMv], &mut [u8], usize) {
        (&mut self.vectors, &mut self.distances, self.stride)
    }

    /// A trajectory vector toward rank `rank` at a unit of the current
    /// superblock row (`row8` within it), when the frame has them.
    #[must_use]
    pub fn trajectory(&self, rank: usize, row8: usize, column8: usize) -> TemporalMv {
        self.trajectories
            .get(rank)
            .map_or(TemporalMv::INVALID, |t| t[row8 * self.stride + column8])
    }

    /// The row index within the superblock row's buffers.
    fn row(&self, y: usize) -> usize {
        (y & (self.superblock8 - 1)) * self.stride
    }

    /// Clear the trajectories and maps across a superblock row.
    fn clear_trajectories(&mut self, rows: (usize, usize)) {
        let step = self.step;
        let columns = self.columns8;
        let block = PROJECTION_BLOCK * step;
        // The previous-group links start one projection block in. The
        // next-group links end one block before the aligned right edge.
        let previous = columns.min(block);
        let next = columns.next_multiple_of(block).saturating_sub(block);
        for n in 0..REFERENCES {
            let trajectory = &mut self.trajectories[n];
            let links = &mut self.links[n];
            for y in (rows.0..rows.1).step_by(step) {
                let row = (y - rows.0) * self.stride;
                clear_samples(
                    &mut trajectory[row..row + columns],
                    step,
                    TemporalMv::INVALID,
                );
                let (left, tail) = links[row..row + columns].split_at_mut(previous);
                let (middle, right) = tail.split_at_mut(next.saturating_sub(previous));
                if next == 0 {
                    for cell in left.iter_mut().step_by(step) {
                        cell[1] = TrajectoryLink::INVALID;
                    }
                } else {
                    for cell in left.iter_mut().step_by(step) {
                        cell[1..].fill(TrajectoryLink::INVALID);
                    }
                }
                clear_samples(middle, step, [TrajectoryLink::INVALID; MAP_GROUPS]);
                for cell in right.iter_mut().step_by(step) {
                    cell[..2].fill(TrajectoryLink::INVALID);
                }
            }
        }
    }
}

impl FrameMotion {
    /// Project the references' fields onto the superblock row of 8x8
    /// rows `row_start8..row_end8` (dav2d `dav2d_refmvs_load_tmvs` over the
    /// frame's width), then the TIP projection and the gap filling.
    pub fn load(&mut self, row_start8: usize, row_end8: usize) {
        let columns8 = (self.columns >> 1) as usize;
        self.temporal.columns8 = columns8;
        let row_end8 = row_end8.min(self.temporal.rows8);
        if row_end8 <= row_start8 {
            return;
        }
        self.project_references((row_start8, row_end8), (0, columns8));
        let h8 = row_end8 - row_start8;
        if self.params.tip_mode != 0 {
            let delta = self.tip.distance;
            let step = self.temporal.step;
            let field = &mut self.temporal;
            let (vectors, distances, stride) = field.vectors_mut();
            let base = row_start8 * stride;
            super::tip::project(
                &mut vectors[base..],
                &mut distances[base..],
                stride,
                (0, columns8),
                h8,
                step,
                delta,
            );
            if self.params.tip_hole_fill {
                super::tip::fill_holes(&mut vectors[base..], stride, (0, columns8), h8, step);
                super::tip::smoothen(&mut vectors[base..], stride, (0, columns8), h8, step);
            }
        }
        if self.temporal.step > 1 {
            let field = &mut self.temporal;
            if !field.trajectories.is_empty() {
                for n in 0..self.params.reference_count {
                    fill_gap_trajectory(
                        &mut field.trajectories[n],
                        field.stride,
                        (0, columns8),
                        h8,
                    );
                }
            }
            let stride = field.stride;
            let base = row_start8 * stride;
            fill_gap_projected(
                &mut field.vectors[base..],
                &mut field.distances[base..],
                stride,
                (0, columns8),
                h8,
            );
        }
    }

    /// dav2d `load_tmvs_refs_c`.
    fn project_references(&mut self, rows: (usize, usize), columns: (usize, usize)) {
        let field = &mut self.temporal;
        let step = field.step;
        let edge = field.edge;
        let shift = field.group_shift;
        let mask = !(step - 1);
        let block8 = PROJECTION_BLOCK * step;
        let stride = field.stride;
        let wide_columns = (
            columns.0.saturating_sub(edge),
            (columns.1 + edge).min(field.columns8),
        );
        let groups = (columns.0 >> shift, (columns.1 - 1) >> shift);
        for y in (rows.0..rows.1).step_by(step) {
            let row = y * stride;
            clear_samples(
                &mut field.vectors[row + columns.0..row + columns.1],
                step,
                TemporalMv::INVALID,
            );
        }
        let trajectories = !field.trajectories.is_empty();
        if trajectories {
            field.clear_trajectories(rows);
        }
        for projection in &self.projections {
            if projection.to_current == INVALID_DISTANCE {
                continue;
            }
            let Some(source) = self.references[projection.reference].as_deref() else {
                continue;
            };
            let side = projection.side;
            for y in (rows.0..rows.1).step_by(step) {
                for x in (wide_columns.0..wide_columns.1).step_by(step) {
                    let at = y * source.stride + x;
                    let Some(far) = super::vector::RefPair::rank(source.refs[at].0[side]) else {
                        continue;
                    };
                    let mv = source.mvs[at][side].dequantize();
                    if !mv.is_valid() {
                        continue;
                    }
                    project_unit(
                        field,
                        projection,
                        (far, mv),
                        (y, x),
                        (rows, columns),
                        (groups, block8, mask),
                        trajectories,
                    );
                }
            }
        }
    }
}

/// One kept vector of a projected reference at unit (`y`, `x`): its
/// trajectories, then its crossing of the current frame.
#[allow(clippy::too_many_arguments, reason = "the loop's state")]
fn project_unit(
    field: &mut Projected,
    projection: &Projection,
    (far, mv): (usize, TemporalMv),
    (y, x): (usize, usize),
    (rows, columns): ((usize, usize), (usize, usize)),
    (groups, block8, mask): ((usize, usize), usize, usize),
    trajectories: bool,
) {
    let far_rank = projection.reference_rank[far];
    if trajectories && let Some(far_rank) = far_rank {
        intersect(field, (projection.reference, far_rank), (y, x), mv, groups);
    }
    // A reference at the projected reference's own order projects either
    // side (dav2d 54856741).
    let Some(to_reference) = projection.to_reference[far].map(i32::from) else {
        return;
    };
    if to_reference != 0 && usize::from(to_reference < 0) != projection.side {
        return;
    }
    let mv1 = scale_temporal(mv, -projection.factors[far][0]);
    let y1 = y as i32 - whole(mv1.y, UNIT_SHIFT);
    if y1 < 0 || y1 as usize >= field.rows8 {
        return;
    }
    let y1 = y1 as usize & mask;
    let x1 = x as i32 - whole(mv1.x, UNIT_SHIFT);
    if x1 < columns.0 as i32 || x1 >= columns.1 as i32 {
        return;
    }
    let x1 = x1 as usize & mask;
    let y_start = y1 & !(block8 - 1);
    let y_end = (y_start + block8).min(rows.1);
    if y < y_start || y >= y_end {
        return;
    }
    let x_align = x1 & !(block8 - 1);
    let x_start = x_align.saturating_sub(field.edge);
    let x_end = (x_align + block8 + field.edge).min(field.columns8);
    if x < x_start || x >= x_end {
        return;
    }
    let stride = field.stride;
    let at1 = y1 * stride + x1;
    let distance = to_reference.unsigned_abs() as u8;
    if field.vectors[at1].is_valid()
        && (projection.target.is_none()
            || far_rank != projection.target
            || field.distances[at1] == distance)
    {
        return;
    }
    if trajectories {
        let reference = projection.reference;
        let (y, x, y1, x1) = (y as i32, x as i32, y1 as i32, x1 as i32);
        let at1 = field.unit(y1, x1);
        field.trajectories[reference][at1] = TemporalMv::clipped(mv1);
        field.link(reference, (y1, x1), (y, x));
        if let Some(far_rank) = far_rank {
            let mv2 = scale_temporal(mv, projection.factors[far][1]);
            field.trajectories[far_rank][at1] = TemporalMv::clipped(mv2);
            let y2 = y + whole(i32::from(mv.y), UNIT_SHIFT);
            let x2 = x + whole(i32::from(mv.x), UNIT_SHIFT);
            if within(y2, (y_start as i32, y_end as i32))
                && within(x2, (x_start as i32, x_end as i32))
            {
                let mask = mask as i32;
                field.link(far_rank, (y1, x1), (y2 & mask, x2 & mask));
            }
        }
    }
    let mv = if to_reference < 0 {
        TemporalMv {
            y: mv.y.wrapping_neg(),
            x: mv.x.wrapping_neg(),
        }
    } else {
        mv
    };
    field.vectors[at1] = mv;
    field.distances[at1] = distance;
}

/// Whether `v` lies in `[start, end)`.
fn within(v: i32, (start, end): (i32, i32)) -> bool {
    v >= start && v < end
}

impl Projected {
    /// A column's group (dav2d `x >> mfmv_k_shift`).
    fn group(&self, x: i32) -> i32 {
        x >> self.group_shift
    }

    /// The columns a trajectory ending at column `x` may reach.
    fn x_reach(&self, x: i32) -> (i32, i32) {
        let block = self.block8 as i32;
        let align = x & !(block - 1);
        (
            (align - self.edge as i32).max(0),
            (align + block + self.edge as i32).min(self.columns8 as i32),
        )
    }

    /// The rows a trajectory ending at row `y` may reach.
    fn y_reach(&self, y: i32) -> (i32, i32) {
        let block = self.block8 as i32;
        let start = y & !(block - 1);
        (start, (start + block).min(self.rows8 as i32))
    }

    /// A unit's index in the row buffers.
    fn unit(&self, y: i32, x: i32) -> usize {
        self.row(y as usize) + x as usize
    }

    /// The link groups to look at from column `x` (dav2d `min_k`,
    /// `max_k`), as map indices.
    fn link_groups(&self, groups: (usize, usize), x: i32) -> std::ops::RangeInclusive<i32> {
        let k_min = (groups.0 as i32 - self.group(x)).max(-1);
        let k_max = (groups.1 as i32 - self.group(x)).min(1);
        k_min + 1..=k_max + 1
    }

    /// Link a trajectory end (`y1`, `x1`) from (`y2`, `x2`) in rank
    /// `rank`'s map.
    fn link(&mut self, rank: usize, (y1, x1): (i32, i32), (y2, x2): (i32, i32)) {
        let k = self.group(x1) - self.group(x2) + 1;
        let at = self.unit(y2, x2);
        self.links[rank][at][k as usize] = TrajectoryLink {
            y: (y1 - y2) as i8,
            x: (x1 - x2) as i8,
        };
    }
}

/// Extend trajectories that cross unit (`y`, `x`) of reference `from`'s
/// field toward reference `to`, and back (dav2d `check_traj_intersect`).
fn intersect(
    field: &mut Projected,
    (from, to): (usize, usize),
    (y, x): (usize, usize),
    mv: TemporalMv,
    groups: (usize, usize),
) {
    let (y, x) = (y as i32, x as i32);
    intersect_forward(field, (from, to), (y, x), mv, groups);
    intersect_backward(field, (from, to), (y, x), mv, groups);
}

/// Trajectories linked at (`y`, `x`) in `from`'s map continue by `mv`
/// toward `to`.
fn intersect_forward(
    field: &mut Projected,
    (from, to): (usize, usize),
    (y, x): (i32, i32),
    mv: TemporalMv,
    groups: (usize, usize),
) {
    let mask = !(field.step as i32 - 1);
    let at = field.unit(y, x);
    for k in field.link_groups(groups, x) {
        let link = field.links[from][at][k as usize];
        if link == TrajectoryLink::INVALID {
            continue;
        }
        let x1 = x + i32::from(link.x);
        if field.group(x1) - field.group(x) + 1 != k {
            continue;
        }
        let xr = field.x_reach(x1);
        let y1 = y + i32::from(link.y);
        let yr = field.y_reach(y1);
        if !within(x, xr) || !within(y, yr) {
            continue;
        }
        let at1 = field.unit(y1, x1);
        if field.trajectories[to][at1].is_valid() {
            continue;
        }
        let source = field.trajectories[from][at1];
        let extended = TemporalMv::clipped(Mv {
            y: i32::from(source.y) + i32::from(mv.y),
            x: i32::from(source.x) + i32::from(mv.x),
        });
        field.trajectories[to][at1] = extended;
        let y2 = y1 + whole(i32::from(extended.y), UNIT_SHIFT);
        let x2 = x1 + whole(i32::from(extended.x), UNIT_SHIFT);
        if within(x2, xr) && within(y2, yr) {
            field.link(to, (y1, x1), (y2 & mask, x2 & mask));
        }
    }
}

/// Trajectories linked where `mv` lands in `to`'s map continue back
/// toward `from`.
fn intersect_backward(
    field: &mut Projected,
    (from, to): (usize, usize),
    (y, x): (i32, i32),
    mv: TemporalMv,
    groups: (usize, usize),
) {
    let mask = !(field.step as i32 - 1);
    let y1 = y + whole(i32::from(mv.y), UNIT_SHIFT);
    let x1 = x + whole(i32::from(mv.x), UNIT_SHIFT);
    if y1.min(x1) < 0 || y1 >= field.rows8 as i32 || x1 >= field.columns8 as i32 {
        return;
    }
    let (y1, x1) = (y1 & mask, x1 & mask);
    let at1 = field.unit(y1, x1);
    for k in field.link_groups(groups, x1) {
        let link = field.links[to][at1][k as usize];
        if link == TrajectoryLink::INVALID {
            continue;
        }
        let x2 = x1 + i32::from(link.x);
        if field.group(x2) - field.group(x1) + 1 != k {
            continue;
        }
        let xr = field.x_reach(x2);
        let y2 = y1 + i32::from(link.y);
        let yr = field.y_reach(y2);
        if !within(x, xr) || !within(x1, xr) || !within(y, yr) || !within(y1, yr) {
            continue;
        }
        let at2 = field.unit(y2, x2);
        if field.trajectories[from][at2].is_valid() {
            continue;
        }
        let source = field.trajectories[to][at2];
        // dav2d clips to 17 bits and keeps 16.
        let clip = |a: i16, b: i16| (i32::from(a) - i32::from(b)).clamp(-0xffff, 0xffff) as i16;
        let extended = TemporalMv {
            y: clip(source.y, mv.y),
            x: clip(source.x, mv.x),
        };
        field.trajectories[from][at2] = extended;
        let y3 = y2 + whole(i32::from(extended.y), UNIT_SHIFT);
        let x3 = x2 + whole(i32::from(extended.x), UNIT_SHIFT);
        if within(x3, xr) && within(y3, yr) {
            field.link(from, (y2, x2), (y3 & mask, x3 & mask));
        }
    }
}

/// Half a sum, positive halves up (dav2d `(s + (s > 0)) >> 1`).
fn half(sum: i32) -> i16 {
    ((sum + i32::from(sum > 0)) >> 1) as i16
}

/// The mean of 1..4 summed vectors (dav2d's `switch (sum_n)`).
fn mean((y, x): (i32, i32), count: u32) -> TemporalMv {
    let m = |s: i32| match count {
        2 => half(s),
        3 => ((s * 85 + 128 - i32::from(s < 0)) >> 8) as i16,
        _ => ((s + 1 + i32::from(s > 0)) >> 2) as i16,
    };
    TemporalMv { y: m(y), x: m(x) }
}

/// Fill the odd rows and columns of a field sampled every other unit
/// (dav2d `fill_gap_proj_c`): from the right, bottom and bottom-right
/// samples of each 16-unit block, projected to the sample's distance.
fn fill_gap_projected(
    vectors: &mut [TemporalMv],
    distances: &mut [u8],
    stride: usize,
    columns: (usize, usize),
    h8: usize,
) {
    for sx in (columns.0..columns.1).step_by(GAP_BLOCK) {
        let x_end = (columns.1 - sx).min(GAP_BLOCK);
        for y in (0..h8).step_by(2) {
            let y_end = ((y & !(GAP_BLOCK - 1)) + GAP_BLOCK).min(h8);
            for x in (0..x_end).step_by(2) {
                let at = y * stride + sx + x;
                let (right, below) = (at + 1, at + stride);
                let mv = vectors[at];
                if !mv.is_valid() {
                    vectors[right] = TemporalMv::INVALID;
                    vectors[below] = TemporalMv::INVALID;
                    vectors[below + 1] = TemporalMv::INVALID;
                    continue;
                }
                let distance = distances[at];
                let (mut sum, mut count) = ((i32::from(mv.y), i32::from(mv.x)), 1);
                let have_right = x + 2 < x_end;
                let projected =
                    |at: usize| project_in_field(vectors[at], i32::from(distance), distances[at]);
                if have_right && vectors[at + 2].is_valid() {
                    let r = projected(at + 2);
                    sum = (sum.0 + i32::from(r.y), sum.1 + i32::from(r.x));
                    vectors[right] = TemporalMv {
                        y: half(sum.0),
                        x: half(sum.1),
                    };
                    count += 1;
                } else {
                    vectors[right] = mv;
                }
                distances[right] = distance;
                let have_bottom = y + 2 < y_end;
                let bottom = at + 2 * stride;
                if have_bottom && vectors[bottom].is_valid() {
                    let b =
                        project_in_field(vectors[bottom], i32::from(distance), distances[bottom]);
                    sum = (sum.0 + i32::from(b.y), sum.1 + i32::from(b.x));
                    vectors[below] = TemporalMv {
                        y: half(i32::from(mv.y) + i32::from(b.y)),
                        x: half(i32::from(mv.x) + i32::from(b.x)),
                    };
                    count += 1;
                } else {
                    vectors[below] = mv;
                }
                distances[below] = distance;
                if have_right && have_bottom && vectors[bottom + 2].is_valid() {
                    let d = project_in_field(
                        vectors[bottom + 2],
                        i32::from(distance),
                        distances[bottom + 2],
                    );
                    sum = (sum.0 + i32::from(d.y), sum.1 + i32::from(d.x));
                    count += 1;
                }
                vectors[below + 1] = if count == 1 { mv } else { mean(sum, count) };
                distances[below + 1] = distance;
            }
        }
    }
}

/// The same for one rank's trajectories (dav2d `fill_gap_traj_c`: no
/// distances; bottom before right).
fn fill_gap_trajectory(t: &mut [TemporalMv], stride: usize, columns: (usize, usize), h8: usize) {
    for sx in (columns.0..columns.1).step_by(GAP_BLOCK) {
        let x_end = columns.1.min(sx + GAP_BLOCK);
        for y in (0..h8).step_by(2) {
            let y_end = ((y & !(GAP_BLOCK - 1)) + GAP_BLOCK).min(h8);
            for x in (sx..x_end).step_by(2) {
                let at = y * stride + x;
                let mv = t[at];
                if !mv.is_valid() {
                    t[at + stride] = TemporalMv::INVALID;
                    t[at + 1] = TemporalMv::INVALID;
                    t[at + 1 + stride] = TemporalMv::INVALID;
                    continue;
                }
                let (mut sum, mut count) = ((i32::from(mv.y), i32::from(mv.x)), 1);
                let have_bottom = y + 2 < y_end;
                if have_bottom && t[at + 2 * stride].is_valid() {
                    let b = t[at + 2 * stride];
                    sum = (sum.0 + i32::from(b.y), sum.1 + i32::from(b.x));
                    t[at + stride] = TemporalMv {
                        y: half(sum.0),
                        x: half(sum.1),
                    };
                    count += 1;
                } else {
                    t[at + stride] = mv;
                }
                let have_right = x + 2 < x_end;
                if have_right && t[at + 2].is_valid() {
                    let r = t[at + 2];
                    sum = (sum.0 + i32::from(r.y), sum.1 + i32::from(r.x));
                    t[at + 1] = TemporalMv {
                        y: half(i32::from(mv.y) + i32::from(r.y)),
                        x: half(i32::from(mv.x) + i32::from(r.x)),
                    };
                    count += 1;
                } else {
                    t[at + 1] = mv;
                }
                if have_right && have_bottom && t[at + 2 * (1 + stride)].is_valid() {
                    let d = t[at + 2 * (1 + stride)];
                    sum = (sum.0 + i32::from(d.y), sum.1 + i32::from(d.x));
                    count += 1;
                }
                t[at + 1 + stride] = if count == 1 { mv } else { mean(sum, count) };
            }
        }
    }
}
