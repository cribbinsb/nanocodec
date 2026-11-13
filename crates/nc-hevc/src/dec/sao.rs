//! Sample adaptive offset: the CTB syntax (H.265 §7.3.8.3, semantics
//! §7.4.9.3) and the filter (§8.7.3), one CTB row at a time, with slice,
//! tile and picture boundary rules and the PCM / lossless exemptions.

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{Dsp, SAO_EO_NEIGHBOURS};

use super::cabac::Engine;
use super::ctx::{self, Contexts};
use super::picture::{Blocks, GRID_LOG2, PictureState, flag};
use super::stats::SaoSyntax;

/// `sao_offset_abs` maximum for 8-bit video: `(1 << (Min(bitDepth, 10) − 5)) − 1`.
const MAX_OFFSET: u32 = 7;
/// `sao_band_position` bits.
const BAND_POSITION_BITS: u32 = 5;
/// `sao_eo_class` bits.
const EO_CLASS_BITS: u32 = 2;
/// `edgeIdx` remapping of §8.7.3.2: 0, 1, 2 become 1, 2, 0.
const EDGE_REMAP: [usize; 5] = [1, 2, 0, 3, 4];

/// `SaoTypeIdx` and its parameters for one component of a CTB.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum Kind {
    #[default]
    Off,
    /// Band offset from `sao_band_position`.
    Band(u8),
    /// Edge offset of `SaoEoClass`.
    Edge(u8),
}

/// One component's SAO: kind and `SaoOffsetVal[1..=4]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Component {
    pub kind: Kind,
    pub offsets: [i8; 4],
}

/// The SAO parameters of a CTB, per component.
pub(super) type CtbSao = [Component; 3];

/// Where the CTB being parsed sits relative to its slice and tile.
#[derive(Debug, Clone, Copy)]
pub(super) struct Neighbours {
    /// Left / above CTB in the same slice and tile: merge candidates.
    pub left: Option<usize>,
    pub up: Option<usize>,
    /// `slice_sao_luma_flag`, `slice_sao_chroma_flag`.
    pub enabled: [bool; 2],
}

/// The index of a SAO type for analysis records: 0 off, 1 band, 2 edge
/// (`SaoTypeIdx`, §7.4.9.3).
pub(super) fn kind_index(kind: Kind) -> u8 {
    match kind {
        Kind::Off => 0,
        Kind::Band(_) => 1,
        Kind::Edge(_) => 2,
    }
}

/// `sao(rx, ry)` (§7.3.8.3) for the CTB at raster address `ctb`. Returns
/// whether the parameters were merged or coded.
pub(super) fn parse(
    e: &mut Engine<'_>,
    ctx: &mut Contexts,
    pic: &mut PictureState,
    ctb: usize,
    n: Neighbours,
) -> SaoSyntax {
    if let Some(left) = n.left
        && e.bin(ctx.at(ctx::SAO_MERGE, 0))
    {
        pic.sao[ctb] = pic.sao[left];
        return SaoSyntax::MergeLeft;
    }
    if let Some(up) = n.up
        && e.bin(ctx.at(ctx::SAO_MERGE, 0))
    {
        pic.sao[ctb] = pic.sao[up];
        return SaoSyntax::MergeUp;
    }
    let mut sao = CtbSao::default();
    for c in 0..3 {
        if !n.enabled[usize::from(c > 0)] {
            continue;
        }
        // Cr shares Cb's type and edge class.
        let kind = if c == 2 {
            sao[1].kind
        } else if !e.bin(ctx.at(ctx::SAO_TYPE, 0)) {
            Kind::Off
        } else if e.bypass() {
            Kind::Edge(0)
        } else {
            Kind::Band(0)
        };
        if kind == Kind::Off {
            continue;
        }
        let mut magnitudes = [0i8; 4];
        for m in &mut magnitudes {
            let mut v = 0;
            while v < MAX_OFFSET && e.bypass() {
                v += 1;
            }
            *m = v as i8;
        }
        let mut component = Component {
            kind,
            offsets: magnitudes,
        };
        match kind {
            Kind::Band(_) => {
                for o in &mut component.offsets {
                    if *o != 0 && e.bypass() {
                        *o = -*o;
                    }
                }
                component.kind = Kind::Band(e.bypass_bits(BAND_POSITION_BITS) as u8);
            }
            Kind::Edge(class) => {
                // Categories 1 and 2 add, 3 and 4 subtract (§7.4.9.3).
                component.offsets[2] = -component.offsets[2];
                component.offsets[3] = -component.offsets[3];
                let class = if c == 0 || c == 1 {
                    e.bypass_bits(EO_CLASS_BITS) as u8
                } else {
                    class
                };
                component.kind = Kind::Edge(class);
            }
            Kind::Off => {}
        }
        sao[c] = component;
    }
    pic.sao[ctb] = sao;
    SaoSyntax::Explicit
}

/// Working memory of SAO, kept between bands: the deblocked samples the
/// band's edge offsets classify from, and each plane's last line before
/// SAO for the band below.
#[derive(Debug, Default)]
pub(super) struct Work {
    /// The band's rows plus one above and one below, at band coordinates
    /// (row 0 is the line above the band).
    band: Option<Picture>,
    /// Each plane's last line of the previous band, before its SAO.
    last: [Vec<u8>; 3],
}

/// Apply SAO to CTB row `row` of the deblocked picture (§8.7.3), whose
/// row below is deblocked too (see `loop_filter`). Edge offsets classify
/// from deblocked samples, but the CTBs to the left and the row above have
/// had SAO already: the band's rows, the line below and the line above
/// (kept from before the band above changed it) are first copied into a
/// band-sized buffer, which stays in cache. Band offsets read only the
/// sample itself and run in place.
pub(super) fn apply_band(
    dsp: &Dsp,
    pic: &mut PictureState,
    tiles_cross: bool,
    work: &mut Work,
    row: usize,
) {
    let layout = pic.layout.clone();
    let usage = |r: usize| {
        let mut used = [false; 3];
        let mut edge = [false; 3];
        for ctb in &pic.sao[r * layout.ctb_w..(r + 1) * layout.ctb_w] {
            for c in 0..3 {
                used[c] |= ctb[c].kind != Kind::Off;
                edge[c] |= matches!(ctb[c].kind, Kind::Edge(_));
            }
        }
        (used, edge)
    };
    let (used, edge) = usage(row);
    if !used.contains(&true) {
        return;
    }
    // Whether the row above changed its samples: then its last line before
    // SAO is in `work.last`.
    let above_changed = if row > 0 {
        usage(row - 1).0
    } else {
        [false; 3]
    };
    let (samples, blocks) = pic.split();
    let ctb = 1usize << layout.log2_ctb;
    // The band's lines (luma), with one above and one below; chroma has
    // half, with the same one above and below.
    let band_lines = ctb + 2 * BAND_MARGIN;
    let fits = work
        .band
        .as_ref()
        .is_some_and(|p| p.width() == blocks.width && p.height() == band_lines);
    if !fits {
        work.band = Picture::new(blocks.width, band_lines, 0).ok();
    }
    let Some(buffer) = work.band.as_mut() else {
        return;
    };
    let mut windows = [Window::default(); 3];
    for (c, plane) in PlaneIndex::ALL.into_iter().enumerate() {
        let shift = usize::from(c > 0);
        let height = blocks.height >> shift;
        let width = blocks.width >> shift;
        let (y0, y1) = (
            (row * ctb) >> shift,
            (((row + 1) * ctb) >> shift).min(height),
        );
        let window = Window {
            top: y0.saturating_sub(1),
            end: (y1 + 1).min(height),
        };
        windows[c] = window;
        let from = samples.plane(plane).view();
        if edge[c] {
            let mut to = buffer.plane_mut(plane).view_mut();
            for y in window.top..window.end {
                let line = if y + 1 == y0 && above_changed[c] {
                    &work.last[c][..width]
                } else {
                    &from.row(y)[..width]
                };
                to.row_mut(y - window.top)[..width].copy_from_slice(line);
            }
        }
        if used[c] {
            // Keep this band's last line before its SAO for the band below.
            let last = &mut work.last[c];
            last.clear();
            last.extend_from_slice(&from.row(y1 - 1)[..width]);
        }
    }
    let buffer = &*buffer;
    let ctbs = row * layout.ctb_w..(row + 1) * layout.ctb_w;
    let band_blocks = blocks.stride * ((row * ctb) >> GRID_LOG2)
        ..blocks.stride * ((((row + 1) * ctb).min(blocks.height)) >> GRID_LOG2);
    let any_exempt = blocks.info[band_blocks]
        .iter()
        .any(|b| b.flags & flag::NO_FILTER != 0);
    for ctb in ctbs {
        let params = blocks.sao[ctb];
        if params.iter().all(|c| c.kind == Kind::Off) {
            continue;
        }
        let allowed = neighbour_ctbs(&blocks, ctb, tiles_cross);
        for (c, plane) in PlaneIndex::ALL.into_iter().enumerate() {
            if params[c].kind == Kind::Off {
                continue;
            }
            let ctx = CtbContext {
                blocks: &blocks,
                plane,
                shift: usize::from(c > 0),
                ctb,
                allowed: &allowed,
                any_exempt,
            };
            filter_ctb(dsp, &ctx, samples, (buffer, windows[c]), params[c]);
        }
    }
}

/// Lines of the band buffer above and below the band.
const BAND_MARGIN: usize = 2;

/// The plane lines `top..end` a band's SAO reads: the band and one line
/// above and below it inside the picture. The band buffer holds them from
/// its row 0.
#[derive(Debug, Clone, Copy, Default)]
struct Window {
    top: usize,
    end: usize,
}

/// Whether each of the 3x3 CTBs around `ctb` (row-major, `[1][1]` itself)
/// may be read by its edge offsets: not across a slice or tile boundary
/// whose loop filtering is disabled (§8.7.3.2). Positions outside the
/// picture count as allowed here; picture bounds are checked separately.
fn neighbour_ctbs(b: &Blocks<'_>, ctb: usize, tiles_cross: bool) -> [[bool; 3]; 3] {
    let layout = b.layout;
    let (cx, cy) = (ctb % layout.ctb_w, ctb / layout.ctb_w);
    let slice_of = |c: usize| {
        let x = (c % layout.ctb_w) << layout.log2_ctb;
        let y = (c / layout.ctb_w) << layout.log2_ctb;
        usize::from(b.info[b.block(x, y)].slice.saturating_sub(1))
    };
    let own = slice_of(ctb);
    let mut allowed = [[false; 3]; 3];
    for (dy, row) in allowed.iter_mut().enumerate() {
        for (dx, a) in row.iter_mut().enumerate() {
            let (nx, ny) = (cx as isize + dx as isize - 1, cy as isize + dy as isize - 1);
            if nx < 0 || ny < 0 || nx as usize >= layout.ctb_w || ny as usize >= layout.ctb_h {
                *a = true;
                continue;
            }
            let n = ny as usize * layout.ctb_w + nx as usize;
            let other = slice_of(n);
            // The later slice's flag governs a slice boundary.
            let slice_ok = other == own
                || if other < own {
                    b.slices[own].loop_filter_across_slices
                } else {
                    b.slices[other].loop_filter_across_slices
                };
            let tile_ok = tiles_cross || layout.tile_of[n] == layout.tile_of[ctb];
            *a = slice_ok && tile_ok;
        }
    }
    allowed
}

/// Which CTB of which plane is filtered, and what restricts it.
struct CtbContext<'a, 'b> {
    blocks: &'a Blocks<'b>,
    plane: PlaneIndex,
    shift: usize,
    ctb: usize,
    allowed: &'a [[bool; 3]; 3],
    /// The picture has PCM / lossless samples somewhere.
    any_exempt: bool,
}

/// SAO of one component of one CTB.
fn filter_ctb(
    dsp: &Dsp,
    cx: &CtbContext<'_, '_>,
    samples: &mut Picture,
    (buffer, window): (&Picture, Window),
    component: Component,
) {
    let b = cx.blocks;
    let shift = cx.shift;
    let size = (1usize << b.layout.log2_ctb) >> shift;
    let (pw, ph) = (b.width >> shift, b.height >> shift);
    let x0 = (cx.ctb % b.layout.ctb_w) * size;
    let y0 = (cx.ctb / b.layout.ctb_w) * size;
    let (x1, y1) = ((x0 + size).min(pw), (y0 + size).min(ph));
    let mut exempt = Vec::new();
    if cx.any_exempt {
        for y in y0..y1 {
            for x in x0..x1 {
                exempt.push(b.info[b.block(x << shift, y << shift)].flags & flag::NO_FILTER != 0);
            }
        }
        if !exempt.contains(&true) {
            exempt.clear();
        }
    }
    let offsets = component.offsets.map(i32::from);
    let area = Area {
        x0,
        y0,
        x1,
        y1,
        size,
        pw,
        ph,
    };
    match component.kind {
        Kind::Off => {}
        Kind::Band(position) => dsp.sao_band_apply_masked(
            position,
            component.offsets,
            &mut samples.plane_mut(cx.plane).view_mut(),
            (x0, y0, x1, y1),
            &exempt,
        ),
        Kind::Edge(class) => {
            // The band's window of the plane and of the buffer, in band
            // coordinates (row 0 is plane row `window.top`).
            let lines = window.end - window.top;
            let src = buffer
                .plane(cx.plane)
                .region(0, 0, pw, lines)
                .expect("band buffer covers the window");
            let mut dst = samples
                .plane_mut(cx.plane)
                .region_mut(0, window.top as isize, pw, lines)
                .expect("window inside the plane");
            let local = Area {
                y0: y0 - window.top,
                y1: y1 - window.top,
                ph: ph.min(window.end) - window.top,
                ..area
            };
            // The window ends at the picture's top and bottom rows only
            // where the picture does, so its bounds are the picture's there.
            if exempt.is_empty() && cx.allowed.iter().flatten().all(|&a| a) {
                // Nothing restricts the neighbours but the picture edge: the
                // nc-dsp kernel over the CTB less the samples whose
                // neighbours leave the picture.
                let [(ax, ay), (bx, by)] = SAO_EO_NEIGHBOURS[usize::from(class)];
                let reach =
                    |a: isize, b: isize| (usize::from(a < 0 || b < 0), usize::from(a > 0 || b > 0));
                let (left, right) = reach(ax, bx);
                let (top, bottom) = reach(ay, by);
                let rect = (
                    x0.max(left),
                    y0.max(top) - window.top,
                    x1.min(pw - right),
                    y1.min(ph - bottom) - window.top,
                );
                dsp.sao_edge_apply(class, component.offsets, &src, &mut dst, rect);
            } else {
                edge_checked(&src, &mut dst, local, &exempt, class, offsets, cx.allowed);
            }
        }
    }
}

/// One CTB's samples of a plane and the plane's size.
#[derive(Debug, Clone, Copy)]
struct Area {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    size: usize,
    pw: usize,
    ph: usize,
}

/// Edge offset of a CTB whose neighbours are restricted by slice or tile
/// boundaries or that holds exempt (PCM, lossless) samples: each sample on
/// the CTB border checks its neighbours.
fn edge_checked(
    src: &PlaneView<'_>,
    dst: &mut PlaneViewMut<'_>,
    area: Area,
    exempt: &[bool],
    class: u8,
    offsets: [i32; 4],
    allowed: &[[bool; 3]; 3],
) {
    let Area {
        x0,
        y0,
        x1,
        y1,
        size,
        pw,
        ph,
    } = area;
    let any_exempt = !exempt.is_empty();
    let [(ax, ay), (bx, by)] = SAO_EO_NEIGHBOURS[usize::from(class)];
    let region = |v: isize, lo: usize, hi: usize| -> usize {
        if v < lo as isize {
            0
        } else if v >= hi as isize {
            2
        } else {
            1
        }
    };
    let row_at = |v: isize| src.row(v.clamp(0, ph as isize - 1) as usize);
    for y in y0..y1 {
        // Only samples on the CTB's border can have a neighbour
        // outside it; interior ones skip the checks.
        let border_row = y == y0 || y + 1 == y1;
        let (row_a, row, row_b) = (row_at(y as isize + ay), src.row(y), row_at(y as isize + by));
        let out = dst.row_mut(y);
        for x in x0..x1 {
            if any_exempt && exempt[(y - y0) * (x1 - x0) + x - x0] {
                continue;
            }
            let (xa, xb) = (x as isize + ax, x as isize + bx);
            let border = border_row || x == x0 || x + 1 == x1;
            let ok = |xn: isize, yn: isize| {
                xn >= 0
                    && yn >= 0
                    && (xn as usize) < pw
                    && (yn as usize) < ph
                    && allowed[region(yn, y0, y0 + size)][region(xn, x0, x0 + size)]
            };
            if border && !(ok(xa, y as isize + ay) && ok(xb, y as isize + by)) {
                continue;
            }
            let s = i32::from(row[x]);
            let a = i32::from(row_a[xa as usize]);
            let b = i32::from(row_b[xb as usize]);
            let edge = EDGE_REMAP[(2 + (s - a).signum() + (s - b).signum()) as usize];
            if edge != 0 {
                out[x] = (s + offsets[edge - 1]).clamp(0, 255) as u8;
            }
        }
    }
}
