//! Encoder search for AV1 loop restoration (AV1 §7.17): a Wiener filter
//! per restoration unit, fitted by least squares between the source and
//! the reconstruction after deblocking and CDEF, switched per unit against
//! `RESTORE_NONE` by rate-distortion cost, with the unit size chosen per
//! frame among 64, 128 and 256. Writes the frame header's `lr_params`
//! (§5.9.20) and each unit's `use_wiener` and taps (§5.11.57–58), and
//! filters the reconstruction as the decoder will (`nc_dsp::av1_restoration`).
//!
//! Ported from libaom v3.15.3 `av1/encoder/pickrst.c` (BSD-2-Clause, see
//! `THIRD_PARTY.md`): the separable symmetric decomposition
//! (`wiener_decompose_sep_sym`: alternate the vertical and the horizontal
//! taps, each a small least-squares solve with the taps summing to 128),
//! quantization (`finalize_sym_filter`), the refinement around the
//! quantized taps (`finer_tile_search_wiener`) and the per-unit RD switch
//! with taps coded against the previous Wiener unit's
//! (`count_wiener_bits`); and SVT-AV1's real-time restrictions
//! (`WnFilterCtrls`, BSD-3-Clause-Clear: fewer taps, luma only, no
//! refinement), which the parameters select.
//!
//! Two departures make the statistics cheap enough for real time; both are
//! exact for the decisions they serve:
//! - The window is folded by the filter's symmetry: a sample's features are
//!   the sums of each group of mirrored positions less as many times the
//!   centre, `(h + 1)^2 - 1` of them for reach `h` (3 taps: 3, 5: 8, 7: 15)
//!   rather than `(2h + 1)^2`, with the error source − reconstruction as
//!   the last column. Their Gram matrix ([`Dsp::gram_rows`]) has 136 terms
//!   at most rather than 1225, and the filter's taps summing to 128 is
//!   built in, so no mean needs removing.
//! - A filter's squared error is evaluated from those statistics, never by
//!   filtering. The statistics read the reconstruction clamped at the
//!   picture edges; the deblocked rows the decoder substitutes across
//!   stripe edges (§7.17.6) and the filter's intermediate rounding are
//!   ignored (a few rows per 64, a fraction of a level).

use crate::cdf::Cdf;
use crate::range_coder::SymbolSink;
use nc_common::{BitWriter, Picture, PlaneIndex, PlaneView};
use nc_dsp::av1_restoration::{
    Av1LoopRestoration, Av1LrFilter, Av1LrScratch, Av1StripeBoundaries, Av1Wiener,
    av1_lr_unit_count,
};
use nc_dsp::{Dsp, GRAM_MAX_ROWS, GRAM_ROW_ALIGN, gram_index, gram_terms};
use nc_encoder::symbol_cost::BitCounter;

/// `RESTORATION_TILESIZE_MAX`: the largest unit; the frame header codes the
/// unit size as this shifted down by 2, 1 or 0 (§5.9.20).
const RESTORATION_TILESIZE_MAX: usize = 256;
/// Unit sizes a frame may use (luma samples; chroma units are the same
/// number of chroma samples, `lr_uv_shift` = 0).
pub(crate) const UNIT_SIZES: [usize; 3] = [64, 128, 256];
/// Luma side of the superblocks that code the units (64x64).
const SUPERBLOCK_SIZE: usize = 64;
/// Luma rows unit rows are shifted up (`RESTORATION_UNIT_OFFSET`).
const UNIT_OFFSET: usize = 8;
/// Largest reach (taps each side of the centre) of a luma Wiener filter;
/// chroma's is one less (5 taps).
pub(crate) const MAX_LUMA_REACH: usize = 3;
const MAX_CHROMA_REACH: usize = 2;
/// Wiener taps coded per direction: the outer three (`[outer, middle,
/// inner]`, the centre is derived).
const WIENER_COEFFS: usize = 3;
/// `Wiener_Taps_Min`, `Wiener_Taps_Max`, `Wiener_Taps_K` and
/// `Wiener_Taps_Mid` (§3), per coded tap.
const WIENER_TAPS_MIN: [i32; WIENER_COEFFS] = [-5, -23, -17];
const WIENER_TAPS_MAX: [i32; WIENER_COEFFS] = [10, 8, 46];
const WIENER_TAPS_K: [u32; WIENER_COEFFS] = [1, 2, 3];
const WIENER_TAPS_MID: [i32; WIENER_COEFFS] = [3, -7, 15];
/// Taps are in 1/128 (`WIENER_FILT_PREC_BITS` = 7); each direction's sum
/// to 128.
const WIENER_SCALE: f64 = 128.0;
/// Alternations of the separable decomposition (`NUM_WIENER_ITERS`).
const DECOMPOSE_ITERATIONS: usize = 5;
/// `lr_type` values of `lr_params` (the inverse of `Remap_Lr_Type`).
const LR_TYPE_NONE: u32 = 0;
const LR_TYPE_WIENER: u32 = 2;
/// Bits of `lr_type`.
const LR_TYPE_BITS: u32 = 2;
/// Pivots below this fraction of the matrix's largest entry make a solve
/// singular (flat units): the previous taps stay.
const SINGULAR_PIVOT: f64 = 1e-9;

/// How hard to search; from the backend's parameters.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchSettings {
    /// Taps each side of the centre of luma filters, 1..=3 (3, 5 or 7
    /// taps); chroma uses at most 2.
    pub luma_reach: usize,
    /// Whether chroma planes are searched.
    pub chroma: bool,
    /// The unit size, or `None` to choose it per frame.
    pub unit_size: Option<usize>,
    /// Statistics take every `row_step`th row.
    pub row_step: usize,
    /// Passes of ±1 refinement of each quantized tap.
    pub refine_passes: usize,
    /// Rate weight: squared error per bit.
    pub lambda: f64,
}

/// The frame header's `lr_params` (§5.9.20) of a sequence with
/// `enable_restoration`: which planes use Wiener units, and the unit size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LrParams {
    /// Planes whose `FrameRestorationType` is `RESTORE_WIENER`.
    pub wiener: [bool; 3],
    /// `LoopRestorationSize` of every plane (`lr_uv_shift` = 0).
    pub unit_size: usize,
}

impl Default for LrParams {
    fn default() -> Self {
        Self {
            wiener: [false; 3],
            unit_size: RESTORATION_TILESIZE_MAX,
        }
    }
}

impl LrParams {
    /// Write `lr_params()` for 64x64 superblocks and 4:2:0.
    pub(crate) fn write(self, w: &mut BitWriter) {
        for wiener in self.wiener {
            w.put_bits(
                if wiener { LR_TYPE_WIENER } else { LR_TYPE_NONE },
                LR_TYPE_BITS,
            );
        }
        if !self.wiener.contains(&true) {
            return;
        }
        // LoopRestorationSize[0] = RESTORATION_TILESIZE_MAX >> (2 - shift).
        let shift = (self.unit_size / UNIT_SIZES[0]).ilog2();
        w.put_bit(shift > 0); // lr_unit_shift
        if shift > 0 {
            w.put_bit(shift > 1); // lr_unit_extra_shift
        }
        if self.wiener[1..].contains(&true) {
            w.put_bit(false); // lr_uv_shift
        }
    }
}

/// The restoration of one plane: its unit grid and each unit's filter.
#[derive(Debug, Clone, Default)]
struct PlaneUnits {
    columns: usize,
    rows: usize,
    filters: Vec<Av1LrFilter>,
}

/// One frame's restoration: the header parameters and every unit's filter.
#[derive(Debug, Clone, Default)]
pub(crate) struct FrameRestoration {
    params: LrParams,
    planes: [PlaneUnits; 3],
}

/// Whether `plane` is subsampled (4:2:0).
fn subsampled(plane: PlaneIndex) -> bool {
    plane != PlaneIndex::Y
}

/// Size of `plane` of a `width`x`height` frame.
fn plane_size(plane: PlaneIndex, (width, height): (usize, usize)) -> (usize, usize) {
    if subsampled(plane) {
        (width.div_ceil(2), height.div_ceil(2))
    } else {
        (width, height)
    }
}

/// The visible `width`x`height` frame's view of `plane` of `picture`.
fn visible(picture: &Picture, plane: PlaneIndex, size: (usize, usize)) -> PlaneView<'_> {
    let (w, h) = plane_size(plane, size);
    picture
        .plane(plane)
        .region(0, 0, w, h)
        .expect("picture covers its frame")
}

impl FrameRestoration {
    /// The frame header's parameters.
    pub(crate) fn params(&self) -> LrParams {
        self.params
    }

    /// Filter `recon` (after CDEF, the visible `size`) in place as the
    /// decoder does, reading `boundaries` (kept after deblocking) across
    /// stripe edges. `scratch` is reused across frames.
    pub(crate) fn apply(
        &self,
        (dsp, scratch): (&Dsp, &mut Av1LrScratch),
        recon: &mut Picture,
        boundaries: &[Av1StripeBoundaries; 3],
        size: (usize, usize),
    ) {
        for (index, plane) in PlaneIndex::ALL.into_iter().enumerate() {
            if !self.params.wiener[index] {
                continue;
            }
            let units = &self.planes[index];
            let (w, h) = plane_size(plane, size);
            let mut view = recon
                .plane_mut(plane)
                .region_mut(0, 0, w, h)
                .expect("picture covers its frame");
            Av1LoopRestoration {
                unit_size: self.params.unit_size,
                columns: units.columns,
                rows: units.rows,
                filters: &units.filters,
            }
            .apply(
                dsp,
                &mut view,
                &boundaries[index],
                subsampled(plane),
                scratch,
            );
        }
    }

    /// Code the units whose top-left sample lies in the 64x64 superblock
    /// at luma (`x`, `y`) (`read_lr`, §5.11.57), plane by plane in raster
    /// order, against the tile's `references`.
    pub(crate) fn write_superblock<S: SymbolSink>(
        &self,
        sink: &mut S,
        use_wiener: &mut Cdf<2>,
        references: &mut WienerReferences,
        (x, y): (usize, usize),
    ) {
        let unit_size = self.params.unit_size;
        for (index, plane) in PlaneIndex::ALL.into_iter().enumerate() {
            if !self.params.wiener[index] {
                continue;
            }
            let units = &self.planes[index];
            let shift = u32::from(subsampled(plane));
            let side = SUPERBLOCK_SIZE >> shift;
            // Units whose first sample lies in the superblock's
            // `side` x `side` plane samples from (x, y).
            let first = |p: usize| (p >> shift).div_ceil(unit_size);
            let end = |p: usize, count: usize| ((p >> shift) + side).div_ceil(unit_size).min(count);
            let (rows, columns) = (
                first(y)..end(y, units.rows),
                first(x)..end(x, units.columns),
            );
            for row in rows {
                for column in columns.clone() {
                    let filter = units.filters[row * units.columns + column];
                    references.write_unit(sink, use_wiener, index, filter);
                }
            }
        }
    }
}

/// The references unit taps are coded against (`RefLrWiener`), which a
/// tile starts at `Wiener_Taps_Mid`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WienerReferences {
    taps: [[[i32; WIENER_COEFFS]; 2]; 3],
}

impl Default for WienerReferences {
    fn default() -> Self {
        Self {
            taps: [[WIENER_TAPS_MID; 2]; 3],
        }
    }
}

impl WienerReferences {
    /// `read_lr_unit` of one unit of plane `index` of a `RESTORE_WIENER`
    /// plane: `use_wiener`, then the taps, vertical first; chroma codes no
    /// outermost tap.
    fn write_unit<S: SymbolSink>(
        &mut self,
        sink: &mut S,
        use_wiener: &mut Cdf<2>,
        index: usize,
        filter: Av1LrFilter,
    ) {
        let Av1LrFilter::Wiener(taps) = filter else {
            sink.write_bool(false, use_wiener);
            return;
        };
        sink.write_bool(true, use_wiener);
        let first = usize::from(index != 0);
        for (pass, coded) in [taps.vertical, taps.horizontal].into_iter().enumerate() {
            let reference = &mut self.taps[index][pass];
            for j in first..WIENER_COEFFS {
                let value = i32::from(coded[j]);
                write_signed_subexp_with_ref(
                    sink,
                    (WIENER_TAPS_MIN[j], WIENER_TAPS_MAX[j] + 1),
                    WIENER_TAPS_K[j],
                    reference[j],
                    value,
                );
                reference[j] = value;
            }
        }
    }
}

/// The inverse of `decode_signed_subexp_with_ref_bool(low, high, k, r)`
/// (§5.11.58).
fn write_signed_subexp_with_ref<S: SymbolSink>(
    sink: &mut S,
    (low, high): (i32, i32),
    k: u32,
    reference: i32,
    value: i32,
) {
    let mx = (high - low) as u32;
    let (reference, x) = ((reference - low) as u32, (value - low) as u32);
    // The inverse of `decode_unsigned_subexp_with_ref_bool`.
    let v = if reference << 1 <= mx {
        recenter(reference, x)
    } else {
        recenter(mx - 1 - reference, mx - 1 - x)
    };
    write_subexp(sink, mx, k, v);
}

/// The inverse of `inverse_recenter(r, v)` (§5.9.28).
fn recenter(r: u32, x: u32) -> u32 {
    if x > 2 * r {
        x
    } else if x >= r {
        (x - r) << 1
    } else {
        ((r - x) << 1) - 1
    }
}

/// The inverse of `decode_subexp_bool(numSyms, k)`: `value` below
/// `symbols`.
fn write_subexp<S: SymbolSink>(sink: &mut S, symbols: u32, k: u32, value: u32) {
    let mut i = 0;
    let mut mk = 0;
    loop {
        let b2 = if i > 0 { k + i - 1 } else { k };
        let a = 1 << b2;
        if symbols <= mk + 3 * a {
            write_non_symmetric(sink, symbols - mk, value - mk);
            return;
        }
        let more = value >= mk + a;
        sink.write_literal_bit(more);
        if !more {
            sink.write_literal(value - mk, b2);
            return;
        }
        i += 1;
        mk += a;
    }
}

/// The inverse of `NS(n)` (§4.10.10): the first `2^w - n` values take one
/// bit fewer.
fn write_non_symmetric<S: SymbolSink>(sink: &mut S, n: u32, value: u32) {
    let bits = n.ilog2() + 1;
    let short = (1 << bits) - n;
    if value < short {
        sink.write_literal(value, bits - 1);
    } else {
        let extended = value + short;
        sink.write_literal(extended >> 1, bits - 1);
        sink.write_literal_bit(extended & 1 == 1);
    }
}

/// The folded window of reach `h`: features `(dr, dc)` in raster order of
/// `0..=h` x `0..=h` without the centre, then the error.
#[derive(Debug, Clone, Copy)]
struct Window {
    reach: usize,
}

impl Window {
    /// Columns of the statistics: the features and the error.
    fn columns(self) -> usize {
        (self.reach + 1) * (self.reach + 1)
    }

    /// The column of the error.
    fn error(self) -> usize {
        self.columns() - 1
    }

    /// The column of feature (`dr`, `dc`), not both 0.
    fn feature(self, dr: usize, dc: usize) -> usize {
        dr * (self.reach + 1) + dc - 1
    }
}

/// Gram matrices of the folded features and error, summed over the
/// sampled rows of one cell or unit.
#[derive(Debug, Clone)]
struct Stats {
    terms: Vec<i64>,
}

impl Stats {
    fn new(window: Window) -> Self {
        Self {
            terms: vec![0; gram_terms(window.columns())],
        }
    }

    fn add(&mut self, other: &Self) {
        for (sum, term) in self.terms.iter_mut().zip(&other.terms) {
            *sum += term;
        }
    }
}

/// The Gram matrix of a unit as `f64`, with its error terms split out.
struct Normal {
    window: Window,
    /// Feature autocorrelation, `features` x `features`.
    gram: Vec<f64>,
    /// Feature-error cross-correlation.
    cross: Vec<f64>,
    /// Squared error without restoration.
    energy: f64,
}

impl Normal {
    fn new(window: Window, stats: &Stats) -> Self {
        let n = window.columns();
        let features = n - 1;
        let term = |i: usize, j: usize| stats.terms[gram_index(n, i.min(j), i.max(j))] as f64;
        let mut gram = vec![0.0; features * features];
        for i in 0..features {
            for j in 0..features {
                gram[i * features + j] = term(i, j);
            }
        }
        Self {
            window,
            gram,
            cross: (0..features).map(|i| term(i, window.error())).collect(),
            energy: term(window.error(), window.error()),
        }
    }

    fn features(&self) -> usize {
        self.cross.len()
    }

    /// Squared error after filtering with feature weights `w`:
    /// `E - 2 w·c + wᵀ G w`.
    fn error(&self, w: &[f64]) -> f64 {
        let n = self.features();
        let mut sse = self.energy;
        for i in 0..n {
            let row = &self.gram[i * n..(i + 1) * n];
            let gw: f64 = row.iter().zip(w).map(|(g, w)| g * w).sum();
            sse += w[i] * (gw - 2.0 * self.cross[i]);
        }
        sse
    }

    /// The feature weights of separable taps: vertical `v` and horizontal
    /// `u` by distance from the centre (index 0 the centre).
    fn weights(&self, v: &[f64], u: &[f64]) -> Vec<f64> {
        let reach = self.window.reach;
        let mut w = vec![0.0; self.features()];
        for dr in 0..=reach {
            for dc in 0..=reach {
                if dr + dc > 0 {
                    w[self.window.feature(dr, dc)] = v[dr] * u[dc];
                }
            }
        }
        w
    }

    /// The least-squares outer taps of one direction (distances
    /// `1..=reach`) given the other direction's taps `other` (centre
    /// first; `vertical`: which one is solved), with the centre tap
    /// `1 - 2 Σ` (libaom `update_a_sep_sym` / `update_b_sep_sym`).
    fn solve_direction(&self, other: &[f64], vertical: bool) -> Option<Vec<f64>> {
        let (reach, features) = (self.window.reach, self.features());
        // w = w0 + J t over the unknown taps t.
        let mut w0 = vec![0.0; features];
        let mut jacobian = vec![0.0; features * reach];
        for a in 0..=reach {
            for (b, &weight) in other.iter().enumerate().take(reach + 1) {
                if a + b == 0 {
                    continue;
                }
                let (dr, dc) = if vertical { (a, b) } else { (b, a) };
                let k = self.window.feature(dr, dc);
                let row = &mut jacobian[k * reach..(k + 1) * reach];
                if a == 0 {
                    // The centre tap 1 − 2Σt carries the other direction's weight.
                    w0[k] = weight;
                    row.fill(-2.0 * weight);
                } else {
                    row[a - 1] = weight;
                }
            }
        }
        // (Jᵀ G J) t = Jᵀ (c − G w0).
        let gw0: Vec<f64> = (0..features)
            .map(|i| {
                let row = &self.gram[i * features..(i + 1) * features];
                row.iter().zip(&w0).map(|(g, w)| g * w).sum()
            })
            .collect();
        let residual: Vec<f64> = self.cross.iter().zip(&gw0).map(|(c, g)| c - g).collect();
        let mut gj = vec![0.0; features * reach];
        for i in 0..features {
            for t in 0..reach {
                gj[i * reach + t] = (0..features)
                    .map(|k| self.gram[i * features + k] * jacobian[k * reach + t])
                    .sum();
            }
        }
        let mut a = vec![0.0; reach * reach];
        let mut rhs = vec![0.0; reach];
        for s in 0..reach {
            for t in 0..reach {
                a[s * reach + t] = (0..features)
                    .map(|k| jacobian[k * reach + s] * gj[k * reach + t])
                    .sum();
            }
            rhs[s] = (0..features)
                .map(|k| jacobian[k * reach + s] * residual[k])
                .sum();
        }
        solve(&mut a, &mut rhs, reach)
    }
}

/// Solve the `n`x`n` system `a x = b` in place by Gaussian elimination
/// with partial pivoting; `None` when it is singular.
fn solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return None;
    }
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        if a[pivot * n + col].abs() <= SINGULAR_PIVOT * scale {
            return None;
        }
        for k in 0..n {
            a.swap(col * n + k, pivot * n + k);
        }
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row * n + col] / a[col * n + col];
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|k| a[row * n + k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row * n + row];
    }
    Some(x)
}

/// Quantized taps of one direction by distance (index 0 unused: the
/// centre is derived), and their real weights with the centre.
fn real_taps(quantized: &[i32]) -> Vec<f64> {
    let outer: i32 = quantized[1..].iter().sum();
    let mut taps: Vec<f64> = quantized
        .iter()
        .map(|&q| f64::from(q) / WIENER_SCALE)
        .collect();
    taps[0] = 1.0 - 2.0 * f64::from(outer) / WIENER_SCALE;
    taps
}

/// The coded range of the tap at `distance` (1..=3) from the centre.
fn tap_range(distance: usize) -> (i32, i32) {
    let j = WIENER_COEFFS - distance;
    (WIENER_TAPS_MIN[j], WIENER_TAPS_MAX[j])
}

/// The best Wiener filter of a unit and its squared error, or `None` when
/// the statistics are degenerate.
fn fit_wiener(normal: &Normal, refine_passes: usize) -> Option<(Av1Wiener, f64)> {
    let reach = normal.window.reach;
    // libaom starts both directions from the mid taps.
    let mid = |d: usize| f64::from(WIENER_TAPS_MID[WIENER_COEFFS - d]) / WIENER_SCALE;
    let centred = |outer: &[f64]| {
        let mut taps = vec![1.0 - 2.0 * outer.iter().sum::<f64>()];
        taps.extend_from_slice(outer);
        taps
    };
    let mut v = centred(&(1..=reach).map(mid).collect::<Vec<_>>());
    let mut u = v.clone();
    for _ in 0..DECOMPOSE_ITERATIONS {
        if let Some(outer) = normal.solve_direction(&u, true) {
            v = centred(&outer);
        }
        if let Some(outer) = normal.solve_direction(&v, false) {
            u = centred(&outer);
        }
    }
    let quantize = |taps: &[f64]| -> Vec<i32> {
        let mut q = vec![0; reach + 1];
        for d in 1..=reach {
            let (min, max) = tap_range(d);
            q[d] = ((taps[d] * WIENER_SCALE).round() as i32).clamp(min, max);
        }
        q
    };
    let mut q = [quantize(&v), quantize(&u)];
    let error =
        |q: &[Vec<i32>; 2]| normal.error(&normal.weights(&real_taps(&q[0]), &real_taps(&q[1])));
    let mut best = error(&q);
    if !best.is_finite() {
        return None;
    }
    // finer_tile_search_wiener: step each tap by ±1 while the error falls.
    for _ in 0..refine_passes {
        let mut improved = false;
        for direction in 0..2 {
            for d in 1..=reach {
                let (min, max) = tap_range(d);
                for delta in [-1, 1] {
                    let tap = q[direction][d] + delta;
                    if !(min..=max).contains(&tap) {
                        continue;
                    }
                    let mut trial = q.clone();
                    trial[direction][d] = tap;
                    let e = error(&trial);
                    if e < best {
                        (best, q) = (e, trial);
                        improved = true;
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }
    let coded = |q: &[i32]| -> [i8; WIENER_COEFFS] {
        std::array::from_fn(|j| q.get(WIENER_COEFFS - j).map_or(0, |&t| t as i8))
    };
    Some((
        Av1Wiener {
            vertical: coded(&q[0]),
            horizontal: coded(&q[1]),
        },
        best,
    ))
}

/// The statistics of one plane on the unit grid of the smallest unit size
/// searched (its cells; rows shifted up like the units): a unit of any
/// larger size is whole cells, since unit edges fall at multiples of the
/// size less the offset and the last unit takes the remainder. Cells as
/// large as the units also keep each Gram call long (one reduction per
/// pair per cell row).
struct PlaneStats {
    window: Window,
    /// Cell side in plane samples.
    cell: usize,
    columns: usize,
    rows: usize,
    cells: Vec<Stats>,
    /// Plane rows per sampled row.
    row_step: usize,
}

/// Gather the statistics of one plane: `source` and `recon` its visible
/// samples.
fn plane_stats(
    dsp: &Dsp,
    window: Window,
    (source, recon): (&PlaneView<'_>, &PlaneView<'_>),
    (cell, sub): (usize, bool),
    row_step: usize,
) -> PlaneStats {
    let (width, height) = (recon.width(), recon.height());
    let offset = UNIT_OFFSET >> u32::from(sub);
    let (columns, rows) = (
        av1_lr_unit_count(cell, width),
        av1_lr_unit_count(cell, height),
    );
    let mut cells = vec![Stats::new(window); columns * rows];
    let reach = window.reach;
    let padded = width.next_multiple_of(GRAM_ROW_ALIGN);
    // Feature rows (zero past the width, which adds nothing).
    let mut features = vec![vec![0i16; padded]; window.columns()];
    // Rows `y - reach ..= y + reach` with `reach` clamped columns each side.
    let mut window_rows = vec![vec![0i16; width + 2 * reach]; 2 * reach + 1];
    // Vertical pair sums by distance.
    let mut pairs = vec![vec![0i16; width + 2 * reach]; reach + 1];
    for y in (0..height).step_by(row_step) {
        for (r, row) in window_rows.iter_mut().enumerate() {
            let source_row = &recon.row((y + r).saturating_sub(reach).min(height - 1))[..width];
            let (left, rest) = row.split_at_mut(reach);
            let (middle, right) = rest.split_at_mut(width);
            for (sample, &s) in middle.iter_mut().zip(source_row) {
                *sample = i16::from(s);
            }
            left.fill(i16::from(source_row[0]));
            right.fill(i16::from(source_row[width - 1]));
        }
        for (d, pair) in pairs.iter_mut().enumerate() {
            let (above, below) = (&window_rows[reach - d], &window_rows[reach + d]);
            if d == 0 {
                pair.copy_from_slice(above);
                continue;
            }
            for ((p, &a), &b) in pair.iter_mut().zip(above).zip(below) {
                *p = a + b;
            }
        }
        let centre = &window_rows[reach][reach..reach + width];
        for dr in 0..=reach {
            for dc in 0..=reach {
                if dr + dc == 0 {
                    continue;
                }
                let multiple = (1 + i16::from(dr > 0)) * (1 + i16::from(dc > 0));
                let pair = &pairs[dr];
                let out = &mut features[window.feature(dr, dc)][..width];
                let left = &pair[reach - dc..reach - dc + width];
                let lanes = out.iter_mut().zip(left).zip(centre);
                if dc == 0 {
                    for ((f, &l), &c) in lanes {
                        *f = l - multiple * c;
                    }
                } else {
                    let right = &pair[reach + dc..reach + dc + width];
                    for (((f, &l), &c), &r) in lanes.zip(right) {
                        *f = l + r - multiple * c;
                    }
                }
            }
        }
        let target = source.row(y);
        for ((e, &s), &c) in features[window.error()][..width]
            .iter_mut()
            .zip(target)
            .zip(centre)
        {
            *e = i16::from(s) - c;
        }
        let cell_row = ((y + offset) / cell).min(rows - 1);
        for column in 0..columns {
            let start = column * cell;
            let end = if column + 1 == columns {
                padded
            } else {
                start + cell
            };
            let mut slices: [&[i16]; GRAM_MAX_ROWS] = [&[]; GRAM_MAX_ROWS];
            for (slice, feature) in slices.iter_mut().zip(&features) {
                *slice = &feature[start..end];
            }
            dsp.gram_rows(
                &slices[..features.len()],
                &mut cells[cell_row * columns + column].terms,
            );
        }
    }
    PlaneStats {
        window,
        cell,
        columns,
        rows,
        cells,
        row_step,
    }
}

impl PlaneStats {
    /// The statistics of each unit of `unit_size` samples (raster order),
    /// and the grid's columns and rows.
    fn units(
        &self,
        unit_size: usize,
        sub: bool,
        (width, height): (usize, usize),
    ) -> (usize, usize, Vec<Stats>) {
        let cell = self.cell;
        let offset = UNIT_OFFSET >> u32::from(sub);
        let (columns, rows) = (
            av1_lr_unit_count(unit_size, width),
            av1_lr_unit_count(unit_size, height),
        );
        let mut units = vec![Stats::new(self.window); columns * rows];
        for cell_row in 0..self.rows {
            let top = (cell_row * cell).saturating_sub(offset);
            let row = ((top + offset) / unit_size).min(rows - 1);
            for cell_column in 0..self.columns {
                let column = (cell_column * cell / unit_size).min(columns - 1);
                units[row * columns + column]
                    .add(&self.cells[cell_row * self.columns + cell_column]);
            }
        }
        (columns, rows, units)
    }
}

/// Choose each unit's filter of one plane for units of `unit_size`
/// (greedy in coding order, as libaom: each unit's taps cost bits against
/// the last Wiener unit's). Returns the units and their total RD cost, or
/// `None` when no unit takes a filter.
fn choose_plane(
    stats: &PlaneStats,
    index: usize,
    unit_size: usize,
    size: (usize, usize),
    settings: &SearchSettings,
    use_wiener: &Cdf<2>,
) -> Option<(PlaneUnits, f64)> {
    let sub = index != 0;
    let (columns, rows, units) = stats.units(unit_size, sub, size);
    let scale = stats.row_step as f64;
    let mut references = WienerReferences::default();
    let mut filters = Vec::with_capacity(units.len());
    let mut cost = 0.0;
    let bits = |references: &WienerReferences, filter: Av1LrFilter| {
        let mut counter = BitCounter::default();
        let mut cdf = *use_wiener;
        references
            .clone()
            .write_unit(&mut counter, &mut cdf, index, filter);
        f64::from(counter.bits())
    };
    for unit in &units {
        let normal = Normal::new(stats.window, unit);
        let none = normal.energy * scale + settings.lambda * bits(&references, Av1LrFilter::None);
        let wiener = fit_wiener(&normal, settings.refine_passes).map(|(taps, sse)| {
            let filter = Av1LrFilter::Wiener(taps);
            (
                filter,
                sse * scale + settings.lambda * bits(&references, filter),
            )
        });
        match wiener {
            Some((filter, wiener_cost)) if wiener_cost < none => {
                let mut sink = BitCounter::default();
                let mut cdf = *use_wiener;
                references.write_unit(&mut sink, &mut cdf, index, filter);
                filters.push(filter);
                cost += wiener_cost;
            }
            _ => {
                filters.push(Av1LrFilter::None);
                cost += none;
            }
        }
    }
    filters.iter().any(|f| *f != Av1LrFilter::None).then_some((
        PlaneUnits {
            columns,
            rows,
            filters,
        },
        cost,
    ))
}

/// Search one frame: `source` and `recon` (after CDEF) at the visible
/// `size`. `None` when no plane gains from restoration.
pub(crate) fn search(
    dsp: &Dsp,
    settings: &SearchSettings,
    (source, recon): (&Picture, &Picture),
    size: (usize, usize),
    use_wiener: &Cdf<2>,
) -> Option<FrameRestoration> {
    let planes: &[PlaneIndex] = if settings.chroma {
        &PlaneIndex::ALL
    } else {
        &[PlaneIndex::Y]
    };
    let sizes: &[usize] = match &settings.unit_size {
        Some(size) => std::slice::from_ref(size),
        None => &UNIT_SIZES,
    };
    // Units are the same number of samples in every plane (`lr_uv_shift`
    // = 0), so the smallest searched size is every plane's cell.
    let cell = sizes[0];
    let stats: Vec<(usize, PlaneStats, f64)> = planes
        .iter()
        .map(|&plane| {
            let index = plane as usize;
            let reach = if index == 0 {
                settings.luma_reach.clamp(1, MAX_LUMA_REACH)
            } else {
                settings.luma_reach.clamp(1, MAX_CHROMA_REACH)
            };
            let stats = plane_stats(
                dsp,
                Window { reach },
                (&visible(source, plane, size), &visible(recon, plane, size)),
                (cell, subsampled(plane)),
                settings.row_step.max(1),
            );
            let energy: f64 = stats
                .cells
                .iter()
                .map(|cell| Normal::new(stats.window, cell).energy)
                .sum::<f64>()
                * stats.row_step as f64;
            (index, stats, energy)
        })
        .collect();
    let mut best: Option<(FrameRestoration, f64)> = None;
    for &unit_size in sizes {
        let mut frame = FrameRestoration {
            params: LrParams {
                wiener: [false; 3],
                unit_size,
            },
            planes: Default::default(),
        };
        let mut cost = 0.0;
        for (index, stats, energy) in &stats {
            let plane_size = plane_size(PlaneIndex::ALL[*index], size);
            match choose_plane(stats, *index, unit_size, plane_size, settings, use_wiener) {
                Some((units, plane_cost)) if plane_cost < *energy => {
                    frame.params.wiener[*index] = true;
                    frame.planes[*index] = units;
                    cost += plane_cost;
                }
                _ => cost += energy,
            }
        }
        if frame.params.wiener.contains(&true) && best.as_ref().is_none_or(|(_, c)| cost < *c) {
            best = Some((frame, cost));
        }
    }
    best.map(|(frame, _)| frame)
}

/// Keep the deblocked rows around each stripe edge of every plane of the
/// visible `size` (before CDEF), for [`FrameRestoration::apply`].
pub(crate) fn stripe_boundaries(
    deblocked: &Picture,
    size: (usize, usize),
) -> [Av1StripeBoundaries; 3] {
    PlaneIndex::ALL.map(|plane| {
        Av1StripeBoundaries::save(&visible(deblocked, plane, size), subsampled(plane), None)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth source and a noisy reconstruction: Wiener smoothing helps.
    fn pictures(width: usize, height: usize) -> (Picture, Picture) {
        let mut source = Picture::new(width, height, 64).unwrap();
        let mut recon = Picture::new(width, height, 64).unwrap();
        let mut state = 7u32;
        for plane in PlaneIndex::ALL {
            let mut s = source.plane_mut(plane).view_mut();
            for y in 0..s.height() {
                for (x, v) in s.row_mut(y).iter_mut().enumerate() {
                    *v =
                        (128.0 + 60.0 * ((x as f64) / 9.0).sin() * ((y as f64) / 13.0).cos()) as u8;
                }
            }
            let mut r = recon.plane_mut(plane).view_mut();
            for y in 0..r.height() {
                let src = source.plane(plane).view();
                for (x, v) in r.row_mut(y).iter_mut().enumerate() {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let noise = (state >> 28) as i32 - 8;
                    *v = (i32::from(src.row(y)[x]) + noise).clamp(0, 255) as u8;
                }
            }
        }
        (source, recon)
    }

    fn settings(luma_reach: usize, unit_size: Option<usize>) -> SearchSettings {
        SearchSettings {
            luma_reach,
            chroma: true,
            unit_size,
            row_step: 1,
            refine_passes: 1,
            lambda: 20.0,
        }
    }

    /// Statistics predict the error of the filter the decoder applies
    /// within the stripe edges and rounding the estimate ignores.
    #[test]
    fn restoration_reduces_the_error_it_predicts() {
        let size = (200, 136);
        let (source, recon) = pictures(size.0, size.1);
        for reach in 1..=MAX_LUMA_REACH {
            let dsp = Dsp::new();
            let frame = search(
                &dsp,
                &settings(reach, None),
                (&source, &recon),
                size,
                &Cdf::new([21198, 0]),
            )
            .expect("noise is worth filtering");
            let boundaries = stripe_boundaries(&recon, size);
            let mut restored = recon.clone();
            frame.apply(
                (&dsp, &mut Av1LrScratch::default()),
                &mut restored,
                &boundaries,
                size,
            );
            for plane in PlaneIndex::ALL {
                let sse = |p: &Picture| -> u64 {
                    let (a, b) = (visible(&source, plane, size), visible(p, plane, size));
                    a.rows()
                        .zip(b.rows())
                        .flat_map(|(a, b)| {
                            a.iter()
                                .zip(b)
                                .map(|(&a, &b)| (i64::from(a) - i64::from(b)).pow(2) as u64)
                        })
                        .sum()
                };
                assert!(
                    sse(&restored) < sse(&recon) * 9 / 10,
                    "reach {reach} plane {plane:?}"
                );
            }
        }
    }

    /// The units each superblock codes cover every unit exactly once, in
    /// raster order, for every unit size and odd frame sizes.
    #[test]
    fn superblocks_code_every_unit_once() {
        struct Recorder(Vec<bool>);
        impl SymbolSink for Recorder {
            fn write_symbol<const N: usize>(&mut self, s: usize, _: &mut Cdf<N>, _: usize) {
                self.0.push(s == 1);
            }
            fn write_literal_bit(&mut self, _: bool) {}
        }
        for size in [(64, 64), (200, 136), (1920, 1080), (72, 40)] {
            for unit_size in UNIT_SIZES {
                let mut frame = FrameRestoration {
                    params: LrParams {
                        wiener: [true; 3],
                        unit_size,
                    },
                    planes: Default::default(),
                };
                for plane in PlaneIndex::ALL {
                    let (w, h) = plane_size(plane, size);
                    let (columns, rows) = (
                        av1_lr_unit_count(unit_size, w),
                        av1_lr_unit_count(unit_size, h),
                    );
                    frame.planes[plane as usize] = PlaneUnits {
                        columns,
                        rows,
                        filters: vec![Av1LrFilter::None; columns * rows],
                    };
                }
                let mut recorder = Recorder(Vec::new());
                let mut references = WienerReferences::default();
                let mut cdf = Cdf::new([21198, 0]);
                for y in (0..size.1.next_multiple_of(8)).step_by(64) {
                    for x in (0..size.0.next_multiple_of(8)).step_by(64) {
                        frame.write_superblock(&mut recorder, &mut cdf, &mut references, (x, y));
                    }
                }
                let total: usize = frame.planes.iter().map(|p| p.filters.len()).sum();
                assert_eq!(recorder.0.len(), total, "{size:?} unit {unit_size}");
            }
        }
    }

    #[test]
    fn subexp_codes_round_trip_through_the_decoder_formulas() {
        // decode_unsigned_subexp_with_ref_bool on the encoder's value.
        fn inverse_recenter(r: u32, v: u32) -> u32 {
            if v > 2 * r {
                v
            } else if v & 1 == 1 {
                r - v.div_ceil(2)
            } else {
                r + v / 2
            }
        }
        for mx in [16u32, 32, 64] {
            for r in 0..mx {
                for x in 0..mx {
                    let v = if r << 1 <= mx {
                        recenter(r, x)
                    } else {
                        recenter(mx - 1 - r, mx - 1 - x)
                    };
                    let back = if r << 1 <= mx {
                        inverse_recenter(r, v)
                    } else {
                        mx - 1 - inverse_recenter(mx - 1 - r, v)
                    };
                    assert_eq!(back, x, "mx {mx} r {r}");
                }
            }
        }
    }
}
