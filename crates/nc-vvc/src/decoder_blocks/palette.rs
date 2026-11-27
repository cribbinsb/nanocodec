//! H.266 palette mode: the `palette_coding()` syntax (§7.3.11.13, semantics
//! §7.4.12.13), the palette predictor kept across CUs (§7.4.12.13,
//! `PredictorPaletteEntries`, updated by §8.4.5.3 at the end of each palette
//! CU), and the reconstruction of a palette CU from its index map, entries
//! and escape values (§8.4.5.3).
//!
//! A palette CU codes one table of colours shared by the planes it covers
//! (all of them in a single tree, luma or both chroma planes in a dual tree)
//! and, for each sample of its first plane, an index into the table or an
//! escape value. Planes subsampled under a single tree take the index of
//! their collocated luma sample.
//!
//! Structure follows the BSD-licensed VTM reference decoder
//! (`CABACReader::cu_palette_info`, `CodingStructure::reorderPrevPLT`,
//! `DecCu::xReconPLT`); see `THIRD_PARTY.md`.
use super::initialization::bank;
use super::{Context, DecodeError, Parser, Reader, Result, Unit};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Pixel;

/// Largest palette of a single tree (`maxNumPaletteEntries`, §7.4.12.13).
const MAX_SIZE: usize = 31;
/// Largest palette of a dual or local dual tree CU.
const MAX_SIZE_DUAL: usize = 15;
/// Largest palette predictor of a single (or local dual) tree
/// (`maxNumPalettePredictorSize`, §7.4.12.13).
const MAX_PREDICTOR: usize = 63;
/// Largest palette predictor of each channel of a dual tree.
const MAX_PREDICTOR_DUAL: usize = 31;
/// Scan positions per run-syntax group (`palette_coding()` codes the flags
/// of 16 positions, then their indices, then their escapes).
const GROUP: usize = 16;
/// Largest CU side palette mode takes (§7.3.11.5).
const MAX_SIDE: usize = 64;
/// Smallest CU area (in luma samples, or chroma samples of a chroma tree)
/// above which palette mode may be signalled (§7.3.11.5).
const MIN_AREA: usize = 16;
/// Order of the exp-Golomb code of `palette_escape_val` (§9.3.3.12).
const ESCAPE_ORDER: u32 = 5;
/// Longest exp-Golomb prefix accepted before the stream is called malformed.
const MAX_PREFIX: u32 = 24;
/// `levelScale` of escape dequantisation (§8.4.5.3, as §8.7.3).
const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];
/// The rounding shift of escape dequantisation (§8.4.5.3).
const ESCAPE_SHIFT: u32 = 6;
/// The run-type context lookup by distance since the last coded run type,
/// after an index run and after a copy-above run (§9.3.4.2.11).
const INDEX_RUN_CONTEXTS: [usize; 5] = [0, 1, 2, 3, 4];
const COPY_RUN_CONTEXTS: [usize; 5] = [0, 1, 1, 2, 2];

#[derive(Clone)]
pub(super) struct Models {
    flag: [Context; 1],
    transpose: [Context; 1],
    run_type: [Context; 1],
    index_run: [Context; 5],
    copy_run: [Context; 3],
}
impl Models {
    pub fn new(qp: u32, kind: u8) -> Self {
        Self {
            flag: bank([[17], [0], [25]], [1], qp, kind),
            transpose: bank([[35], [42], [42]], [5], qp, kind),
            run_type: bank([[50], [59], [42]], [9], qp, kind),
            index_run: bank(
                [
                    [58, 45, 45, 30, 38],
                    [51, 30, 30, 38, 23],
                    [50, 37, 45, 30, 46],
                ],
                [9, 6, 9, 10, 5],
                qp,
                kind,
            ),
            copy_run: bank(
                [[45, 38, 46], [38, 53, 46], [45, 38, 46]],
                [0, 9, 5],
                qp,
                kind,
            ),
        }
    }
}

/// `PredictorPaletteSize` (luma or shared, chroma) and
/// `PredictorPaletteEntries` per plane: empty at the start of a slice and
/// of a tile, and restored with the contexts at a wavefront row start.
#[derive(Clone, Debug)]
pub(super) struct Predictor {
    sizes: [usize; 2],
    entries: [[u16; MAX_PREDICTOR]; 3],
}
impl Default for Predictor {
    fn default() -> Self {
        Self {
            sizes: [0; 2],
            entries: [[0; MAX_PREDICTOR]; 3],
        }
    }
}

/// The decoded palette syntax of one CU.
#[derive(Debug)]
pub(crate) struct Palette {
    /// The first plane coded: 0 (luma, or every plane of a single tree) or
    /// 1 (both chroma planes of a chroma tree).
    first: usize,
    /// How many planes are coded from `first`.
    planes: usize,
    /// `CurrentPaletteEntries` of each plane.
    entries: [Vec<u16>; 3],
    /// Width and height of the index map: the first plane's block.
    size: (usize, usize),
    /// `PaletteIndexMap` in raster order; `entries[..].len()` marks an escape.
    indices: Vec<u8>,
    /// `PaletteEscapeVal` of each plane, raster order in that plane's block.
    escapes: [Vec<u32>; 3],
}

/// How a CU's palette relates to the predictor: which planes it codes,
/// its predictor channel and its size limits.
struct Layout {
    first: usize,
    planes: usize,
    /// `PredictorPaletteSize` entry shared by these planes.
    channel: usize,
    /// A luma CU of a local dual tree, whose palette still carries chroma
    /// entries for the predictor.
    local: bool,
    max_size: usize,
    max_predictor: usize,
}
impl Parser<'_> {
    /// Whether `pred_mode_plt_flag` is present for an intra (not IBC) CU
    /// (§7.3.11.5), and its value.
    pub(super) fn palette_flag(&mut self, r: &mut Reader<'_>, unit: &Unit) -> Result<bool> {
        let n = unit.node;
        if !self.sequence.tools.palette || n.width > MAX_SIDE || n.height > MAX_SIDE {
            return Ok(false);
        }
        let area = if unit.channel == 1 {
            let (sx, sy) = self.sequence.subsampling();
            (n.width >> sx) * (n.height >> sy)
        } else {
            n.width * n.height
        };
        // A local dual tree's chroma CU is never palette coded.
        let local_chroma = unit.channel == 1 && !self.palette_dual_tree();
        if area <= MIN_AREA || local_chroma {
            return Ok(false);
        }
        r.bin(&mut self.palette_models.flag[0])
    }

    /// Whether the slice codes separate luma and chroma trees throughout.
    fn palette_dual_tree(&self) -> bool {
        self.header.kind == 2 && self.sequence.chroma && self.sequence.dual_tree
    }

    fn palette_layout(&self, unit: &Unit) -> Layout {
        let separate = self.sequence.chroma && !unit.combined;
        let local = separate && !self.palette_dual_tree();
        let all = if self.sequence.chroma { 3 } else { 1 };
        let (first, planes) = match (separate, unit.channel) {
            (false, _) => (0, all),
            (true, 0) => (0, 1),
            (true, _) => (1, 2),
        };
        let joint = !separate || local;
        Layout {
            first,
            planes,
            channel: if joint { 0 } else { first },
            local,
            max_size: if separate { MAX_SIZE_DUAL } else { MAX_SIZE },
            max_predictor: if joint {
                MAX_PREDICTOR
            } else {
                MAX_PREDICTOR_DUAL
            },
        }
    }

    /// The subsampling of `plane` against the index map of a palette whose
    /// first plane is `first`: a single tree's chroma takes the indices of
    /// the luma samples collocated with its own.
    fn map_subsampling(&self, first: usize, plane: usize) -> (usize, usize) {
        if first == 0 && plane != 0 {
            self.sequence.subsampling()
        } else {
            (0, 0)
        }
    }

    /// `palette_coding()` (§7.3.11.13) of `unit`, its QP syntax, and the
    /// predictor update that follows the CU.
    pub(super) fn palette_unit(&mut self, r: &mut Reader<'_>, unit: &mut Unit) -> Result<()> {
        let layout = self.palette_layout(unit);
        let coded = layout.first..layout.first + layout.planes;
        let predicted = self.predicted_entries(r, &layout)?;
        let mut entries = self.palette_entries(r, &layout, &predicted)?;
        let size = entries[layout.first].len();
        let escape = size == 0 || r.bypass()?;
        let symbols = size + usize::from(escape);
        let transpose = symbols > 1 && r.bin(&mut self.palette_models.transpose[0])?;
        if escape {
            self.unit_qp_syntax(r, unit, true, true)?;
        }
        let (sx, sy) = if layout.first == 0 {
            (0, 0)
        } else {
            self.sequence.subsampling()
        };
        let block = (unit.node.width >> sx, unit.node.height >> sy);
        let scan = traverse(block, transpose);
        let mut map = IndexMap {
            indices: vec![0; block.0 * block.1],
            copy: vec![false; block.0 * block.1],
            width: block.0,
            escape: size as u8,
        };
        let mut escapes: [Vec<u32>; 3] = std::array::from_fn(|plane| {
            let (gx, gy) = self.map_subsampling(layout.first, plane);
            if coded.contains(&plane) {
                vec![0; (block.0 >> gx) * (block.1 >> gy)]
            } else {
                Vec::new()
            }
        });
        let mut run = RunState::default();
        for start in (0..scan.len()).step_by(GROUP) {
            let group = start..(start + GROUP).min(scan.len());
            let starts = self.run_syntax(
                r,
                &mut map,
                (&scan, group.clone()),
                (symbols, transpose),
                &mut run,
            )?;
            index_syntax(
                r,
                &mut map,
                (&scan, group.clone()),
                &starts,
                (symbols, transpose),
            )?;
            self.escape_syntax(
                r,
                &map,
                &scan[group],
                (layout.first, coded.clone()),
                &mut escapes,
            )?;
        }
        self.palette_predictor = updated(&self.palette_predictor, &layout, &entries, &predicted);
        if !layout.local {
            for (plane, list) in entries.iter_mut().enumerate() {
                if !coded.contains(&plane) {
                    list.clear();
                }
            }
        }
        unit.palette = Some(Box::new(Palette {
            first: layout.first,
            planes: layout.planes,
            entries,
            size: block,
            indices: map.indices,
            escapes,
        }));
        Ok(())
    }

    /// `CurrentPaletteEntries` (§7.4.12.13): the reused predictor entries,
    /// then `new_palette_entries` of each coded plane.
    fn palette_entries(
        &mut self,
        r: &mut Reader<'_>,
        layout: &Layout,
        predicted: &[usize],
    ) -> Result<[Vec<u16>; 3]> {
        let depth = self.sequence.depth.bits();
        let mut entries: [Vec<u16>; 3] = std::array::from_fn(|plane| {
            predicted
                .iter()
                .map(|&index| self.palette_predictor.entries[plane][index])
                .collect()
        });
        let reused = predicted.len();
        let signalled = if reused < layout.max_size {
            exp_golomb(r, 0)? as usize
        } else {
            0
        };
        let size = reused + signalled;
        if size > layout.max_size {
            return Err(DecodeError::Malformed("VVC palette size"));
        }
        let coded = layout.first..layout.first + layout.planes;
        for (plane, list) in entries.iter_mut().enumerate() {
            let read = coded.contains(&plane);
            for _ in reused..size {
                list.push(if read {
                    r.bypass_bits(depth)? as u16
                } else {
                    // A local dual tree's luma palette has mid-grey chroma.
                    1 << (depth - 1)
                });
            }
        }
        Ok(entries)
    }

    /// `palette_escape_val` of one group's escape-coded positions, plane by
    /// plane; a single tree's subsampled chroma has escapes only at its own
    /// sample positions.
    fn escape_syntax(
        &self,
        r: &mut Reader<'_>,
        map: &IndexMap,
        positions: &[(usize, usize)],
        (first, coded): (usize, std::ops::Range<usize>),
        escapes: &mut [Vec<u32>; 3],
    ) -> Result<()> {
        let escape = map.escape;
        for plane in coded {
            let (gx, gy) = self.map_subsampling(first, plane);
            let width = map.width >> gx;
            for &(x, y) in positions {
                if map.indices[map.at(x, y)] != escape
                    || x & ((1 << gx) - 1) != 0
                    || y & ((1 << gy) - 1) != 0
                {
                    continue;
                }
                escapes[plane][(y >> gy) * width + (x >> gx)] = exp_golomb(r, ESCAPE_ORDER)?;
            }
        }
        Ok(())
    }

    /// `palette_predictor_run` (§7.3.11.13): the predictor entries reused,
    /// in order.
    fn predicted_entries(&mut self, r: &mut Reader<'_>, layout: &Layout) -> Result<Vec<usize>> {
        let available = self.palette_predictor.sizes[layout.channel];
        let mut reused = Vec::new();
        if available == 0 {
            return Ok(reused);
        }
        let mut index = 0;
        let mut run = exp_golomb(r, 0)?;
        while run != 1 && index < available && reused.len() < layout.max_size {
            if index > 0 {
                run = exp_golomb(r, 0)?;
                if run == 1 {
                    break;
                }
            }
            // A run of 0 reuses the next entry, a run of n > 1 skips n − 1.
            index += (run as usize).saturating_sub(1);
            if index >= available {
                return Err(DecodeError::Malformed("VVC palette predictor run"));
            }
            reused.push(index);
            index += 1;
        }
        Ok(reused)
    }

    /// `run_copy_flag` and `copy_above_palette_indices_flag` of one group
    /// of scan positions; returns which positions start a new run.
    fn run_syntax(
        &mut self,
        r: &mut Reader<'_>,
        map: &mut IndexMap,
        (scan, group): (&[(usize, usize)], std::ops::Range<usize>),
        (symbols, transpose): (usize, bool),
        run: &mut RunState,
    ) -> Result<[bool; GROUP]> {
        let base = group.start;
        let mut starts = [false; GROUP];
        starts[0] = base == 0;
        if symbols <= 1 {
            return Ok(starts);
        }
        for position in group {
            let (x, y) = scan[position];
            let here = map.at(x, y);
            let previous =
                (position > 0).then(|| map.at(scan[position - 1].0, scan[position - 1].1));
            let new_run = if previous.is_none() {
                true
            } else {
                // §9.3.4.2.11: the context counts the positions since the
                // last coded run type.
                let distance = (position - run.start - 1).min(INDEX_RUN_CONTEXTS.len() - 1);
                let models = &mut self.palette_models;
                let context = if run.copy {
                    &mut models.copy_run[COPY_RUN_CONTEXTS[distance]]
                } else {
                    &mut models.index_run[INDEX_RUN_CONTEXTS[distance]]
                };
                !r.bin(context)?
            };
            starts[position - base] = new_run;
            if new_run {
                let first_line = if transpose { x == 0 } else { y == 0 };
                let after_copy = previous.is_some_and(|p| map.copy[p]);
                map.copy[here] =
                    !first_line && !after_copy && r.bin(&mut self.palette_models.run_type[0])?;
                run.copy = map.copy[here];
                run.start = position;
            } else if let Some(p) = previous {
                map.copy[here] = map.copy[p];
            }
        }
        Ok(starts)
    }
}

/// `palette_idx_idc` of one group: each new index run's index, and the
/// indices that index and copy-above runs repeat.
fn index_syntax(
    r: &mut Reader<'_>,
    map: &mut IndexMap,
    (scan, group): (&[(usize, usize)], std::ops::Range<usize>),
    starts: &[bool; GROUP],
    (symbols, transpose): (usize, bool),
) -> Result<()> {
    let base = group.start;
    for position in group {
        let (x, y) = scan[position];
        let here = map.at(x, y);
        let previous = (position > 0).then(|| map.at(scan[position - 1].0, scan[position - 1].1));
        // The sample above (left, transposed): copy runs never start on the
        // first line, so it exists whenever it is needed.
        let above = || {
            if transpose {
                map.at(x - 1, y)
            } else {
                map.at(x, y - 1)
            }
        };
        let new_run = starts[position - base];
        map.indices[here] = if symbols <= 1 {
            match previous {
                Some(p) if !new_run => map.indices[p],
                _ => 0,
            }
        } else if new_run && !map.copy[here] {
            // §7.4.12.13: the coded index skips the one the previous
            // position holds (the sample above it, after a copy run).
            let reference = previous.map(|p| {
                if map.copy[p] {
                    map.indices[above()]
                } else {
                    map.indices[p]
                }
            });
            let coded = truncated_binary(r, symbols - usize::from(position > 0))?;
            let index = match reference {
                Some(reference) if coded >= usize::from(reference) => coded + 1,
                _ => coded,
            };
            if index >= symbols {
                return Err(DecodeError::Malformed("VVC palette index"));
            }
            index as u8
        } else if !map.copy[here] {
            map.indices[previous.expect("a run continues a position")]
        } else {
            map.indices[above()]
        };
    }
    Ok(())
}

/// The predictor after a palette CU (§8.4.5.3, `reorderPrevPLT`): the CU's
/// entries, then the predictor entries it did not reuse, up to the
/// predictor's size limit.
fn updated(
    old: &Predictor,
    layout: &Layout,
    entries: &[Vec<u16>; 3],
    reused: &[usize],
) -> Predictor {
    let mut predictor = old.clone();
    let size = entries[layout.first].len();
    let kept: Vec<usize> = (0..old.sizes[layout.channel])
        .filter(|index| !reused.contains(index))
        .take(layout.max_predictor.saturating_sub(size))
        .collect();
    let planes = if layout.local {
        0..3
    } else {
        layout.first..layout.first + layout.planes
    };
    for plane in planes {
        let target = &mut predictor.entries[plane];
        target[..size].copy_from_slice(&entries[plane][..size]);
        for (slot, &index) in target[size..].iter_mut().zip(&kept) {
            *slot = old.entries[plane][index];
        }
    }
    predictor.sizes[layout.channel] = size + kept.len();
    predictor
}

/// The last coded run type and where it started.
#[derive(Default)]
struct RunState {
    start: usize,
    copy: bool,
}

/// Run types and indices of a CU's index map, in raster order.
struct IndexMap {
    indices: Vec<u8>,
    /// Whether each position copies the index above (`CopyAboveIndicesFlag`).
    copy: Vec<bool>,
    width: usize,
    /// The index of escape-coded samples: the palette size.
    escape: u8,
}
impl IndexMap {
    fn at(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
}

/// The horizontal (or, transposed, vertical) traverse scan of a block
/// (§6.5.4): lines alternate direction.
fn traverse((width, height): (usize, usize), transpose: bool) -> Vec<(usize, usize)> {
    let (lines, length) = if transpose {
        (width, height)
    } else {
        (height, width)
    };
    (0..lines)
        .flat_map(|line| {
            (0..length).map(move |step| {
                let along = if line % 2 == 0 {
                    step
                } else {
                    length - 1 - step
                };
                if transpose {
                    (line, along)
                } else {
                    (along, line)
                }
            })
        })
        .collect()
}

/// A `k`-th order exp-Golomb bypass code (§9.3.3.5).
fn exp_golomb(r: &mut Reader<'_>, order: u32) -> Result<u32> {
    let mut value = 0u32;
    let mut k = order;
    while r.bypass()? {
        value += 1 << k;
        k += 1;
        if k > MAX_PREFIX + order {
            return Err(DecodeError::Malformed("VVC palette exp-Golomb prefix"));
        }
    }
    if k > 0 {
        value += r.bypass_bits(k)?;
    }
    Ok(value)
}

/// A truncated binary bypass code of `symbols` values (§9.3.3.4).
fn truncated_binary(r: &mut Reader<'_>, symbols: usize) -> Result<usize> {
    let bits = symbols.ilog2();
    let whole = 1usize << bits;
    let short = whole - (symbols - whole);
    let mut value = if bits == 0 {
        0
    } else {
        r.bypass_bits(bits)? as usize
    };
    if value >= short {
        value = 2 * value - short + usize::from(r.bypass()?);
    }
    Ok(value)
}

/// Write the samples of palette CU `unit` into `picture` (§8.4.5.3): each
/// sample its entry, or its escape value dequantised at the plane's QP
/// `qps` (already at least `QpPrimeTsMin`).
pub(super) fn reconstruct<T: Pixel>(
    picture: &mut Picture<T>,
    unit: &Unit,
    palette: &Palette,
    qps: [u8; 3],
    depth: u32,
) {
    let (sx, sy) = picture.chroma_format().subsampling();
    let size = palette.entries[palette.first].len();
    let max = (1i64 << depth) - 1;
    let planes = palette.planes();
    for (plane, &qp) in qps.iter().enumerate().take(planes.end).skip(planes.start) {
        let (px, py) = if plane == 0 { (0, 0) } else { (sx, sy) };
        // The index-map step per plane sample: a single tree's chroma reads
        // the collocated luma indices.
        let (gx, gy) = if palette.first == 0 { (px, py) } else { (0, 0) };
        let (width, height) = (unit.node.width >> px, unit.node.height >> py);
        let (x0, y0) = (unit.node.x >> px, unit.node.y >> py);
        let qp = i64::from(qp);
        let (scale, shift) = (LEVEL_SCALE[(qp % 6) as usize], qp / 6);
        let mut view = picture.plane_mut(PlaneIndex::ALL[plane]).view_mut();
        for y in 0..height {
            let row = &mut view.row_mut(y0 + y)[x0..x0 + width];
            for (x, sample) in row.iter_mut().enumerate() {
                let entry = usize::from(palette.indices[(y << gy) * palette.size.0 + (x << gx)]);
                let value = if entry == size {
                    let escape = i64::from(palette.escapes[plane][y * width + x]);
                    (((escape * scale) << shift) + (1 << (ESCAPE_SHIFT - 1))) >> ESCAPE_SHIFT
                } else {
                    i64::from(palette.entries[plane][entry])
                };
                *sample = T::from_i32(value.clamp(0, max) as i32);
            }
        }
    }
}

impl Palette {
    /// The planes a palette CU codes.
    pub(super) fn planes(&self) -> std::ops::Range<usize> {
        self.first..self.first + self.planes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traverse_scan_snakes_along_lines() {
        assert_eq!(
            traverse((3, 2), false),
            [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]
        );
        assert_eq!(
            traverse((2, 3), true),
            [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
        );
    }

    #[test]
    fn predictor_keeps_new_entries_then_unused_old_ones() {
        let layout = Layout {
            first: 0,
            planes: 1,
            channel: 0,
            local: false,
            max_size: MAX_SIZE_DUAL,
            max_predictor: 4,
        };
        let mut predictor = Predictor::default();
        predictor.sizes[0] = 4;
        predictor.entries[0][..4].copy_from_slice(&[10, 11, 12, 13]);
        let entries = [vec![11, 20, 21], Vec::new(), Vec::new()];
        let next = updated(&predictor, &layout, &entries, &[1]);
        assert_eq!(next.sizes, [4, 0]);
        assert_eq!(next.entries[0][..4], [11, 20, 21, 10]);
    }
}
