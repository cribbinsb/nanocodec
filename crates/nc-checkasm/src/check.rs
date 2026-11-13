//! Bit-exactness checks of every kernel of an ISA against the scalar reference.

use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, GRID_BLOCKS, GridBest, GridRows, Isa, Moments};

use crate::rng::Rng;

/// Largest extra stride beyond the block width; odd so strides are often odd.
pub(crate) const MAX_EXTRA_STRIDE: usize = 67;
/// Largest offset of a block from the start of its buffer, so starts are
/// usually unaligned.
const MAX_START_OFFSET: usize = 31;

/// How a test block's samples are chosen.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Pattern {
    /// Independent random source and reference.
    Random,
    /// Reference = source plus small noise: the typical motion-search case.
    NearMatch,
    /// Both all zero.
    AllZero,
    /// Both all 255.
    AllMax,
    /// Source 255, reference 0: the largest possible differences.
    MaxDifference,
    /// Alternating 0/255 checkerboards in opposite phase: the largest
    /// high-frequency Hadamard coefficients.
    Checkerboard,
}

pub(crate) const PATTERNS: [Pattern; 6] = [
    Pattern::Random,
    Pattern::NearMatch,
    Pattern::AllZero,
    Pattern::AllMax,
    Pattern::MaxDifference,
    Pattern::Checkerboard,
];

/// Result of checking one ISA.
#[derive(Debug)]
pub struct CheckReport {
    /// The ISA checked.
    pub isa: Isa,
    /// Number of kernel calls compared against scalar.
    pub cases: usize,
    /// One line per mismatch.
    pub failures: Vec<String>,
}

impl CheckReport {
    /// True when no kernel disagreed with the reference.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Compare every kernel of `isa` with scalar. `iterations` random rounds
/// run per block size and pattern. `None` if the ISA cannot run here.
#[must_use]
pub fn check(isa: Isa, seed: u64, iterations: usize) -> Option<CheckReport> {
    let candidate = Dsp::with_isa(isa)?;
    let reference = Dsp::with_isa(Isa::Scalar).expect("scalar is always available");
    let mut rng = Rng::new(seed);
    let mut report = CheckReport {
        isa,
        cases: 0,
        failures: Vec::new(),
    };
    for size in BlockSize::all() {
        for pattern in PATTERNS {
            for _ in 0..iterations {
                check_case(&candidate, &reference, size, pattern, &mut rng, &mut report);
            }
        }
    }
    for pattern in PATTERNS {
        for _ in 0..iterations {
            check_moments(&candidate, &reference, pattern, &mut rng, &mut report);
            check_pairs(&candidate, &reference, pattern, &mut rng, &mut report);
            check_sse_8x8_row(&candidate, &reference, pattern, &mut rng, &mut report);
            check_downscale(&candidate, &reference, pattern, &mut rng, &mut report);
            check_deinterleave(&candidate, &reference, pattern, &mut rng, &mut report);
        }
    }
    crate::check_codec::check_codec_kernels(
        &candidate,
        &reference,
        iterations,
        &mut rng,
        &mut report,
    );
    crate::check_vp9_filter::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_vp9_line::check(&candidate, &reference, &mut rng.clone(), &mut report);
    crate::check_cdef_interior::check(&candidate, &reference, &mut report);
    crate::check_hevc_put32::check(&candidate, &reference, &mut report);
    crate::check_display::check(&candidate, &reference, iterations, &mut rng, &mut report);
    crate::check_border::check(&candidate, &reference, iterations, &mut rng, &mut report);
    crate::check_vp9_inverse::check(&candidate, &reference, iterations, &mut rng, &mut report);
    crate::check_vp9_inverse::check_add(&candidate, &reference, iterations, &mut rng, &mut report);
    crate::check_av1_inverse::check(&candidate, &reference, iterations, &mut rng, &mut report);
    crate::check_filters::check_filters(&candidate, &reference, iterations, &mut rng, &mut report);
    crate::check_vp9_mc::check(&candidate, &reference, &mut rng, &mut report);
    #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
    crate::check_relaxed_dot::check(&mut report);
    crate::check_vp9_scaled::check(&candidate, &mut rng.clone(), &mut report);
    crate::check_transpose::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_av2_typed::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_vp9_quant::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_vvc::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_gram::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_vvc_angular::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_av1_deblock::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_av1_grid::check(&candidate, &reference, &mut rng.clone(), &mut report);
    crate::check_filter_runs::check(&candidate, &reference, &mut rng.clone(), &mut report);
    crate::check_vp9_intra::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_h264_intra::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_hevc_intra::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_av1_grain::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_av1_scaled::check(&candidate, &reference, iterations, &mut rng, &mut report);
    crate::check_av1_restoration::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_av1_warp::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_av1_mask::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_hevc_weight::check(&candidate, &reference, &mut rng, &mut report);
    crate::check_memory::check(&candidate, &reference, &mut report);
    // Compiled-in baseline SIMD, the same whatever `isa` (cheap to repeat).
    crate::check_av1_symbol::check(iterations, &mut rng, &mut report);
    Some(report)
}

fn check_case(
    candidate: &Dsp,
    reference: &Dsp,
    size: BlockSize,
    pattern: Pattern,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let (w, h) = (size.width(), size.height());
    let src = TestBlock::new(w, h, 1, rng);
    let rf = TestBlock::new(w, h, 4, rng);
    let (src_data, ref_data) = fill(pattern, &src, &rf, rng);
    let src_view = src.view(&src_data, 0);
    let ref_views: [PlaneView<'_>; 4] = [0, 1, 2, 3].map(|i| rf.view(&ref_data, i));

    let mut compare = |kernel: &str, got: &[u32], want: &[u32]| {
        report.cases += 1;
        if got != want {
            report.failures.push(format!(
                "{} {kernel} {w}x{h} {pattern:?} src_stride={} ref_stride={}: got {got:?}, want {want:?}",
                candidate.isa(),
                src.stride,
                rf.stride
            ));
        }
    };

    compare(
        "sad",
        &[candidate.sad(size, &src_view, &ref_views[0])],
        &[reference.sad(size, &src_view, &ref_views[0])],
    );
    if h >= 8 {
        compare(
            "sad_even_rows",
            &[candidate.sad_even_rows_at(size, &src_view, &ref_views[0], (0, 0))],
            &[reference.sad_even_rows_at(size, &src_view, &ref_views[0], (0, 0))],
        );
    }
    compare(
        "satd",
        &[candidate.satd(size, &src_view, &ref_views[0])],
        &[reference.satd(size, &src_view, &ref_views[0])],
    );
    compare(
        "satd_4x4_tiles",
        &[candidate.satd_4x4_tiles(size, &src_view, &ref_views[0])],
        &[reference.satd_4x4_tiles(size, &src_view, &ref_views[0])],
    );
    compare(
        "sad_x4",
        &candidate.sad_x4(size, &src_view, &ref_views),
        &reference.sad_x4(size, &src_view, &ref_views),
    );
    if (w, h) == (16, 16) {
        let eight = BlockSize::new(8, 8).unwrap();
        let expected = core::array::from_fn::<_, 4, _>(|q| {
            let (x, y) = (8 * (q % 2), 8 * (q / 2));
            reference.satd(
                eight,
                &src_view.sub_view(x, y, 8, 8).unwrap(),
                &ref_views[0].sub_view(x, y, 8, 8).unwrap(),
            )
        });
        compare(
            "satd_8x8_quads",
            &candidate.satd_8x8_quads(&src_view, &ref_views[0]),
            &expected,
        );
    }
    if (w, h) == (16, 16) {
        // Scan across the four adjacent candidate blocks: every alignment of
        // the reference, up to the scan length the motion search uses.
        let wide = PlaneView::new(&ref_data[rf.start..], rf.stride, rf.count * w, h)
            .expect("test buffer covers every block");
        let positions = (wide.width() - 16 + 1).min(MAX_SCAN_POSITIONS);
        for count in [1, 3, 4, 7, 8, 9, 15, 16, 17, 32, positions] {
            let mut got = vec![0u32; count];
            let mut want = got.clone();
            candidate.sad_scan_16x16_even_rows_at(&src_view, &wide, (0, 0), &mut got);
            reference.sad_scan_16x16_even_rows_at(&src_view, &wide, (0, 0), &mut want);
            compare("sad_scan_16x16_even_rows", &got, &want);
        }
    }
    if (w, h) == (64, 64) {
        let wide = PlaneView::new(&ref_data[rf.start..], rf.stride, rf.count * w, h)
            .expect("test buffer covers four grids");
        check_grid(candidate, reference, &src_view, &wide, rng, &mut compare);
    }
    if h == 8 {
        // Every row length, including tails after a 2/4-block SIMD batch.
        for count in 1..=w / 8 {
            let mut got = vec![[0u32; 4]; count];
            let mut want = got.clone();
            candidate.sad_quads_8x8_row(&src_view, &ref_views[0], &mut got);
            reference.sad_quads_8x8_row(&src_view, &ref_views[0], &mut want);
            compare("sad_quads_8x8_row", got.as_flattened(), want.as_flattened());
        }
    }
    if (w, h) == (16, 16) {
        check_quads_16x16(candidate, reference, &src_view, &ref_views, &mut compare);
    }
}

/// The 16x16 quadrant SAD kernels, including the fused 4:2:0 macroblock
/// form (chroma from the top-left 8x8 of other candidate blocks, with their
/// own strides and contents).
fn check_quads_16x16(
    candidate: &Dsp,
    reference: &Dsp,
    src: &PlaneView<'_>,
    refs: &[PlaneView<'_>; 4],
    compare: &mut impl FnMut(&str, &[u32], &[u32]),
) {
    compare(
        "sad_quads_16x16",
        &candidate.sad_quads_16x16_at(src, &refs[0], (0, 0)),
        &reference.sad_quads_16x16_at(src, &refs[0], (0, 0)),
    );
    let chroma = [(src, &refs[3]), (&refs[3], &refs[0])];
    let (got, want) = (
        candidate.sad_quads_macroblock((src, &refs[0]), chroma),
        reference.sad_quads_macroblock((src, &refs[0]), chroma),
    );
    compare(
        "sad_quads_macroblock luma",
        got.0.as_flattened(),
        want.0.as_flattened(),
    );
    compare(
        "sad_quads_macroblock chroma",
        got.1.as_flattened(),
        want.1.as_flattened(),
    );
}

/// Largest downscale output width tried: covers several full SIMD iterations
/// plus every tail length.
const MAX_DOWNSCALE_WIDTH: usize = 100;
/// Largest downscale output height tried.
const MAX_DOWNSCALE_HEIGHT: usize = 6;

/// Longest scan the SAD-scan check tries (the motion search scans 32).
const MAX_SCAN_POSITIONS: usize = 33;

/// Longest row of blocks the moments check tries: two of the widest
/// (four-block) groups and a tail.
const MAX_MOMENTS_BLOCKS: usize = 9;

fn check_moments(
    candidate: &Dsp,
    reference: &Dsp,
    pattern: Pattern,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    // Rows of 1 to MAX_MOMENTS_BLOCKS blocks.
    for blocks in 1..=MAX_MOMENTS_BLOCKS {
        let block = TestBlock::new(8 * blocks, 8, 1, rng);
        let other = TestBlock::new(8 * blocks, 8, 1, rng);
        let (data, _) = fill(pattern, &block, &other, rng);
        let view = block.view(&data, 0);
        let mut got = vec![Moments::default(); blocks];
        let mut want = got.clone();
        candidate.moments_8x8_row(&view, &mut got);
        reference.moments_8x8_row(&view, &mut want);
        report.cases += 1;
        if got != want {
            report.failures.push(format!(
                "{} moments_8x8_row x{blocks} {pattern:?} stride={}: got {got:?}, want {want:?}",
                candidate.isa(),
                block.stride
            ));
        }
        candidate.moments_8x4_row(&view, &mut got);
        reference.moments_8x4_row(&view, &mut want);
        report.cases += 1;
        if got != want {
            report.failures.push(format!(
                "{} moments_8x4_row x{blocks} {pattern:?} stride={}: got {got:?}, want {want:?}",
                candidate.isa(),
                block.stride
            ));
        }
    }
}

/// Widest output row the pairs check tries: two of the widest SIMD steps
/// and a tail.
const MAX_PAIRS_WIDTH: usize = 75;

fn check_pairs(
    candidate: &Dsp,
    reference: &Dsp,
    pattern: Pattern,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for w in [1, 15, 16, 17, 31, 32, 33, MAX_PAIRS_WIDTH] {
        let block = TestBlock::new(2 * w, 3, 1, rng);
        let other = TestBlock::new(2 * w, 3, 1, rng);
        let (data, _) = fill(pattern, &block, &other, rng);
        let view = block.view(&data, 0);
        for pick in [false, true] {
            let (mut got, mut want) = (vec![0u8; w * 3], vec![0u8; w * 3]);
            let mut got_view = PlaneViewMut::new(&mut got, w, w, 3).expect("covers");
            candidate.pairs_2x(&view, &mut got_view, pick);
            let mut want_view = PlaneViewMut::new(&mut want, w, w, 3).expect("covers");
            reference.pairs_2x(&view, &mut want_view, pick);
            report.cases += 1;
            if got != want {
                report.failures.push(format!(
                    "{} pairs_2x w={w} pick={pick} {pattern:?}: got {got:?}, want {want:?}",
                    candidate.isa()
                ));
            }
        }
    }
}

fn check_sse_8x8_row(
    candidate: &Dsp,
    reference: &Dsp,
    pattern: Pattern,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    // Rows of 1 to MAX_MOMENTS_BLOCKS blocks: every SIMD group and tail.
    for blocks in 1..=MAX_MOMENTS_BLOCKS {
        let block = TestBlock::new(8 * blocks, 8, 1, rng);
        let other = TestBlock::new(8 * blocks, 8, 1, rng);
        let (src, pred) = fill(pattern, &block, &other, rng);
        let (src, pred) = (block.view(&src, 0), other.view(&pred, 0));
        let mut got = vec![0; blocks];
        let mut want = got.clone();
        candidate.sse_8x8_row(&src, &pred, &mut got);
        reference.sse_8x8_row(&src, &pred, &mut want);
        report.cases += 1;
        if got != want {
            report.failures.push(format!(
                "{} sse_8x8_row x{blocks} {pattern:?}: got {got:?}, want {want:?}",
                candidate.isa(),
            ));
        }
    }
}

fn check_downscale(
    candidate: &Dsp,
    reference: &Dsp,
    pattern: Pattern,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let w = 1 + rng.below(MAX_DOWNSCALE_WIDTH);
    let h = 1 + rng.below(MAX_DOWNSCALE_HEIGHT);
    let src = TestBlock::new(2 * w, 2 * h, 1, rng);
    let other = TestBlock::new(2 * w, 2 * h, 1, rng);
    let (src_data, _) = fill(pattern, &src, &other, rng);
    let src_view = src.view(&src_data, 0);
    let dst_stride = w + rng.below(MAX_EXTRA_STRIDE + 1);
    // Exactly long enough: a write past the last output leaves the buffer.
    let dst_len = (h - 1) * dst_stride + w;
    let run = |dsp: &Dsp| {
        let mut out = vec![0xA5u8; dst_len];
        let mut view = PlaneViewMut::new(&mut out, dst_stride, w, h).expect("buffer fits");
        dsp.downscale_2x(&src_view, &mut view);
        out
    };
    let (got, want) = (run(candidate), run(reference));
    report.cases += 1;
    if got != want {
        report.failures.push(format!(
            "{} downscale_2x {w}x{h} {pattern:?} src_stride={} dst_stride={dst_stride}: outputs differ",
            candidate.isa(),
            src.stride
        ));
    }
}

/// Largest deinterleave output width tried: several full SIMD iterations
/// (32 pairs on AVX2, 16 elsewhere) plus every tail length.
const MAX_DEINTERLEAVE_WIDTH: usize = 100;
/// Largest deinterleave height tried.
const MAX_DEINTERLEAVE_HEIGHT: usize = 4;

fn check_deinterleave(
    candidate: &Dsp,
    reference: &Dsp,
    pattern: Pattern,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let w = 1 + rng.below(MAX_DEINTERLEAVE_WIDTH);
    let h = 1 + rng.below(MAX_DEINTERLEAVE_HEIGHT);
    let src = TestBlock::new(2 * w, h, 1, rng);
    let other = TestBlock::new(2 * w, h, 1, rng);
    let (src_data, _) = fill(pattern, &src, &other, rng);
    let src_view = src.view(&src_data, 0);
    let (u_stride, v_stride) = (
        w + rng.below(MAX_EXTRA_STRIDE + 1),
        w + rng.below(MAX_EXTRA_STRIDE + 1),
    );
    // Exactly long enough: a write past the last output leaves the buffer;
    // the 0xA5 fill shows writes between rows.
    let run = |dsp: &Dsp| {
        let mut u = vec![0xA5u8; (h - 1) * u_stride + w];
        let mut v = vec![0xA5u8; (h - 1) * v_stride + w];
        let mut u_view = PlaneViewMut::new(&mut u, u_stride, w, h).expect("buffer fits");
        let mut v_view = PlaneViewMut::new(&mut v, v_stride, w, h).expect("buffer fits");
        dsp.deinterleave_uv(&src_view, &mut u_view, &mut v_view);
        (u, v)
    };
    let (got, want) = (run(candidate), run(reference));
    report.cases += 1;
    if got != want {
        report.failures.push(format!(
            "{} deinterleave_uv {w}x{h} {pattern:?} src_stride={} u_stride={u_stride} v_stride={v_stride}: outputs differ",
            candidate.isa(),
            src.stride
        ));
    }
}

/// Geometry of a test buffer holding `count` blocks side by side, laid out
/// so the buffer ends exactly at the last block's last sample (any
/// over-read leaves the allocation).
pub(crate) struct TestBlock {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) stride: usize,
    pub(crate) start: usize,
    pub(crate) count: usize,
}

impl TestBlock {
    pub(crate) fn new(width: usize, height: usize, count: usize, rng: &mut Rng) -> Self {
        let stride = count * width + rng.below(MAX_EXTRA_STRIDE + 1);
        Self {
            width,
            height,
            stride,
            start: rng.below(MAX_START_OFFSET + 1),
            count,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.start + (self.height - 1) * self.stride + self.count * self.width
    }

    pub(crate) fn view<'a>(&self, data: &'a [u8], index: usize) -> PlaneView<'a> {
        let offset = self.start + index * self.width;
        PlaneView::new(&data[offset..], self.stride, self.width, self.height)
            .expect("test buffer covers every block")
    }
}

pub(crate) fn fill(
    pattern: Pattern,
    src: &TestBlock,
    rf: &TestBlock,
    rng: &mut Rng,
) -> (Vec<u8>, Vec<u8>) {
    let mut s = vec![0u8; src.len()];
    let mut r = vec![0u8; rf.len()];
    match pattern {
        Pattern::Random => {
            rng.fill(&mut s);
            rng.fill(&mut r);
        }
        Pattern::NearMatch => {
            rng.fill(&mut s);
            for (i, v) in r.iter_mut().enumerate() {
                let base = s[i % s.len()];
                let noise = rng.below(9) as i16 - 4;
                *v = (i16::from(base) + noise).clamp(0, 255) as u8;
            }
        }
        Pattern::AllZero => {}
        Pattern::AllMax => {
            s.fill(255);
            r.fill(255);
        }
        Pattern::MaxDifference => s.fill(255),
        Pattern::Checkerboard => {
            checkerboard(&mut s, src.stride, 0);
            checkerboard(&mut r, rf.stride, 1);
        }
    }
    (s, r)
}

fn checkerboard(buf: &mut [u8], stride: usize, phase: usize) {
    for (i, v) in buf.iter_mut().enumerate() {
        let (x, y) = (i % stride, i / stride);
        *v = if (x + y + phase).is_multiple_of(2) {
            255
        } else {
            0
        };
    }
}

/// Hierarchy grid SADs, masked edges and ordered multi-candidate ties.
fn check_grid(
    candidate: &Dsp,
    reference: &Dsp,
    src: &PlaneView<'_>,
    wide: &PlaneView<'_>,
    rng: &mut Rng,
    compare: &mut impl FnMut(&str, &[u32], &[u32]),
) {
    // Distinct tags on identical nonzero SADs prove that the first
    // candidate wins, both within a batch and across successive batches.
    let flat_source = [31; 64 * 64];
    let flat_reference = [73; 67 * 64];
    let flat_src = PlaneView::new(&flat_source, 64, 64, 64).unwrap();
    let flat_ref = PlaneView::new(&flat_reference, 67, 67, 64).unwrap();
    for rows in [GridRows::All, GridRows::Even] {
        let mut best = GridBest::default();
        for tags in [[7, 2, 1, 0], [6, 5, 4, 3]] {
            candidate.sad_grid_best_x4_at(
                &flat_src,
                &flat_ref,
                [(0, 0), (1, 0), (2, 0), (3, 0)],
                &[u32::MAX; 64],
                rows,
                &mut best,
                tags,
            );
        }
        compare(
            "sad_grid_best_x4 constant ties",
            &best.tag,
            &[7; GRID_BLOCKS],
        );
    }
    // Random starting bests (so wins, losses and ties all occur) and a
    // random visibility mask.
    let mut start = GridBest::default();
    let mut bytes = [0u8; 2 * GRID_BLOCKS + 64];
    rng.fill(&mut bytes);
    for i in 0..GRID_BLOCKS {
        start.sad[i] = u32::from(bytes[i]) << (4 + i / 32);
        start.tag[i] = u32::from(bytes[GRID_BLOCKS + i]);
    }
    let visible: [u32; 64] = std::array::from_fn(|i| {
        if bytes[2 * GRID_BLOCKS + i] < 32 {
            0
        } else {
            u32::MAX
        }
    });
    for rows in [GridRows::All, GridRows::Even] {
        let (mut got, mut want) = (start, start);
        let (s, r) = (src, wide);
        candidate.sad_grid_best_64x64_at(s, r, (0, 0), &visible, rows, &mut got, 7);
        reference.sad_grid_best_64x64_at(s, r, (0, 0), &visible, rows, &mut want, 7);
        compare("sad_grid_best_64x64 sad", &got.sad, &want.sad);
        compare("sad_grid_best_64x64 tag", &got.tag, &want.tag);
        for positions in [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(3, 0), (0, 0), (3, 0), (1, 0)],
        ] {
            let tags = [17, 3, 29, 8];
            got = start;
            want = start;
            candidate.sad_grid_best_x4_at(s, wide, positions, &visible, rows, &mut got, tags);
            for (position, tag) in positions.into_iter().zip(tags) {
                reference.sad_grid_best_64x64_at(s, wide, position, &visible, rows, &mut want, tag);
            }
            compare("sad_grid_best_x4 sad", &got.sad, &want.sad);
            compare("sad_grid_best_x4 tag", &got.tag, &want.tag);
            // Repeating the candidates must retain the original tags on ties.
            candidate.sad_grid_best_x4_at(s, wide, positions, &visible, rows, &mut got, [99; 4]);
            compare("sad_grid_best_x4 ties", &got.tag, &want.tag);
        }
    }
}
