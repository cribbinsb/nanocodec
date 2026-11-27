//! Sample adaptive offset: the CTB syntax (H.265 §7.3.8.3, semantics
//! §7.4.9.3) and the filter (§8.7.3), one CTB row at a time, with slice,
//! tile and picture boundary rules and the PCM / lossless exemptions.
//!
//! At high bit depth offsets reach `(1 << (Min(BitDepth, 10) − 5)) − 1`
//! and scale by `2^log2OffsetScale` (`log2_sao_offset_scale_luma` /
//! `_chroma` of the range extension, 0 without it; §7.4.9.3.2); bands are
//! `2^(BitDepth − 5)` values wide (the kernels'). Chroma CTBs and lines
//! follow the picture's chroma format.

use nc_common::{BitDepth, BitDepths, Picture, PlaneIndex};
use nc_dsp::{DepthKernels, Dsp, Pixel, SAO_EO_NEIGHBOURS, SaoCtb, SaoLimits, sao_edge_restricted};

use super::cabac::Engine;
use super::ctx::{self, Contexts};
use super::picture::{Blocks, GRID_LOG2, PictureState, flag};

/// `sao_offset_abs` is coded with `Min(BitDepth, OFFSET_DEPTH_LIMIT) − 5`
/// bits of range (§7.4.9.3.2); deeper samples scale the offsets instead.
const OFFSET_DEPTH_LIMIT: u32 = 10;
/// `SaoOffsetVal` magnitude bits below the depth (`cMax = (1 << (depth −
/// 5)) − 1`).
const OFFSET_RANGE_BITS: u32 = 5;

/// `cMax` of `sao_offset_abs` at `depth`.
fn offset_limit(depth: BitDepth) -> u32 {
    let coded = depth.bits().min(OFFSET_DEPTH_LIMIT);
    (1 << (coded - OFFSET_RANGE_BITS)) - 1
}
/// `sao_band_position` bits.
const BAND_POSITION_BITS: u32 = 5;
/// `sao_eo_class` bits.
const EO_CLASS_BITS: u32 = 2;

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

/// One component's SAO: kind and `SaoOffsetVal[1..=4]` (scaled to the
/// depth).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Component {
    pub kind: Kind,
    pub offsets: [i16; 4],
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
    /// `log2_sao_offset_scale_luma`, `log2_sao_offset_scale_chroma`.
    pub log2_offset_scale: [u8; 2],
}

/// How a CTB's SAO parameters were signalled (analysis records).
#[derive(Debug, Clone, Copy)]
pub(super) enum SaoSyntax {
    MergeLeft = 0,
    MergeUp = 1,
    Explicit = 2,
}

/// What [`parse`] reports: the signalling in analysis builds, nothing
/// otherwise (the normal decoder is unchanged).
#[cfg(feature = "analysis")]
pub(super) type Signalled = SaoSyntax;
#[cfg(not(feature = "analysis"))]
pub(super) type Signalled = ();

#[inline]
fn signalled(syntax: SaoSyntax) -> Signalled {
    #[cfg(feature = "analysis")]
    {
        syntax
    }
    #[cfg(not(feature = "analysis"))]
    {
        let _ = syntax;
    }
}

/// The index of a SAO type for analysis records: 0 off, 1 band, 2 edge
/// (`SaoTypeIdx`, §7.4.9.3).
#[cfg(feature = "analysis")]
pub(super) fn kind_index(kind: Kind) -> u8 {
    match kind {
        Kind::Off => 0,
        Kind::Band(_) => 1,
        Kind::Edge(_) => 2,
    }
}

/// `sao(rx, ry)` (§7.3.8.3) for the CTB at raster address `ctb`.
#[cfg_attr(
    not(feature = "analysis"),
    allow(clippy::semicolon_if_nothing_returned)
)]
pub(super) fn parse<T: Pixel>(
    e: &mut Engine<'_>,
    ctx: &mut Contexts,
    pic: &mut PictureState<T>,
    ctb: usize,
    n: Neighbours,
) -> Signalled {
    let depths = pic.depths;
    if let Some(left) = n.left
        && e.bin(ctx.at(ctx::SAO_MERGE, 0))
    {
        pic.sao[ctb] = pic.sao[left];
        return signalled(SaoSyntax::MergeLeft);
    }
    if let Some(up) = n.up
        && e.bin(ctx.at(ctx::SAO_MERGE, 0))
    {
        pic.sao[ctb] = pic.sao[up];
        return signalled(SaoSyntax::MergeUp);
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
        let max_offset = offset_limit(depths.plane(c));
        let scale = n.log2_offset_scale[usize::from(c > 0)];
        let mut magnitudes = [0i16; 4];
        for m in &mut magnitudes {
            let mut v = 0;
            while v < max_offset && e.bypass() {
                v += 1;
            }
            *m = (v << scale) as i16;
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
    signalled(SaoSyntax::Explicit)
}

/// Working memory of SAO, kept between bands: the deblocked samples the
/// band's edge offsets classify from, and each plane's last line before
/// SAO for the band below.
#[derive(Debug)]
pub(super) struct Work<T: Pixel> {
    /// The band's rows plus one above and one below, at band coordinates
    /// (row 0 is the line above the band).
    band: Option<Picture<T>>,
    /// Each plane's last line of the previous band, before its SAO.
    last: [Vec<T>; 3],
}

impl<T: Pixel> Default for Work<T> {
    fn default() -> Self {
        Self {
            band: None,
            last: Default::default(),
        }
    }
}

/// Apply SAO to CTB row `row` of the deblocked picture (§8.7.3), whose
/// row below is deblocked too (see `loop_filter`). Edge offsets classify
/// from deblocked samples, but the CTBs to the left and the row above have
/// had SAO already: the band's rows, the line below and the line above
/// (kept from before the band above changed it) are first copied into a
/// band-sized buffer, which stays in cache. Band offsets read only the
/// sample itself and run in place.
pub(super) fn apply_band<T: Pixel>(
    dsp: &Dsp,
    pic: &mut PictureState<T>,
    tiles_cross: bool,
    work: &mut Work<T>,
    row: usize,
) {
    let depths: BitDepths = pic.depths;
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
    let filter_exempt = pic.filter_exempt;
    let (samples, blocks) = pic.split();
    let format = samples.chroma_format();
    let ctb = 1usize << layout.log2_ctb;
    // The band's lines (luma), with one above and one below; subsampled
    // chroma has fewer, with the same one above and below.
    let band_lines = ctb + 2 * BAND_MARGIN;
    let fits = work.band.as_ref().is_some_and(|p| {
        p.width() == blocks.width && p.height() == band_lines && p.chroma_format() == format
    });
    if !fits {
        work.band = Picture::with_format(blocks.width, band_lines, format, 0).ok();
    }
    let Some(buffer) = work.band.as_mut() else {
        return;
    };
    let mut windows = [Window::default(); 3];
    for (c, plane) in PlaneIndex::ALL.into_iter().enumerate() {
        if !used[c] {
            continue;
        }
        let (sx, sy) = format.plane_subsampling(plane);
        let height = blocks.height >> sy;
        let width = blocks.width >> sx;
        let (y0, y1) = ((row * ctb) >> sy, (((row + 1) * ctb) >> sy).min(height));
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
    let any_exempt = filter_exempt
        && blocks.info[band_blocks]
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
                subsampling: format.plane_subsampling(plane),
                ctb,
                allowed: &allowed,
                any_exempt,
            };
            let kernels = dsp.at(depths.plane(c));
            filter_ctb(kernels, &ctx, samples, (buffer, windows[c]), params[c]);
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
    /// The plane's `(sx, sy)` (log2 `SubWidthC`, `SubHeightC`; 0 for luma).
    subsampling: (usize, usize),
    ctb: usize,
    allowed: &'a [[bool; 3]; 3],
    /// The picture has PCM / lossless samples somewhere.
    any_exempt: bool,
}

/// SAO of one component of one CTB.
fn filter_ctb<T: Pixel>(
    kernels: DepthKernels<'_>,
    cx: &CtbContext<'_, '_>,
    samples: &mut Picture<T>,
    (buffer, window): (&Picture<T>, Window),
    component: Component,
) {
    let b = cx.blocks;
    let (sx, sy) = cx.subsampling;
    let ctb = 1usize << b.layout.log2_ctb;
    let (size_x, size_y) = (ctb >> sx, ctb >> sy);
    let (pw, ph) = (b.width >> sx, b.height >> sy);
    let x0 = (cx.ctb % b.layout.ctb_w) * size_x;
    let y0 = (cx.ctb / b.layout.ctb_w) * size_y;
    let (x1, y1) = ((x0 + size_x).min(pw), (y0 + size_y).min(ph));
    let mut exempt = Vec::new();
    if cx.any_exempt {
        for y in y0..y1 {
            for x in x0..x1 {
                exempt.push(b.info[b.block(x << sx, y << sy)].flags & flag::NO_FILTER != 0);
            }
        }
        if !exempt.contains(&true) {
            exempt.clear();
        }
    }
    let area = SaoCtb {
        x0,
        y0,
        x1,
        y1,
        size: (size_x, size_y),
        width: pw,
        height: ph,
    };
    match component.kind {
        Kind::Off => {}
        Kind::Band(position) => kernels.sao_band_apply_masked(
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
            let local = SaoCtb {
                y0: y0 - window.top,
                y1: y1 - window.top,
                height: ph.min(window.end) - window.top,
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
                kernels.sao_edge_apply(class, component.offsets, &src, &mut dst, rect);
            } else {
                let limits = SaoLimits {
                    ctbs: *cx.allowed,
                    cuts: [&[], &[]],
                    exempt: &exempt,
                };
                let rule = (class, component.offsets, kernels.depth());
                sao_edge_restricted(&src, &mut dst, local, rule, &limits);
            }
        }
    }
}
