//! Bounded probes for transforms, SATD, CDEF, HEVC MC/SAO statistics, variance and in-place VVC edges.
use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{
    BlockSize, Dsp, Filter, HevcForwardMode, HevcForwardStaging, HevcTransform, VvcIntraMode,
};
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map_or("tx32", String::as_str);
    let rounds: usize = args.get(2).map_or(Ok(100_000), |s| s.parse()).unwrap();
    let dsp = Dsp::new();
    if mode == "vvc-angular" {
        let side = args.get(3).map_or(Ok(16), |s| s.parse()).unwrap();
        let direction = args.get(4).map_or(Ok(66), |s| s.parse()).unwrap();
        let oriented = args.get(5).is_some_and(|s| s == "oriented");
        vvc_angular(&dsp, rounds, side, direction, oriented);
        return;
    }
    if mode == "vvc-rect" {
        let width = args.get(3).map_or(Ok(16), |s| s.parse()).unwrap();
        let height = args.get(4).map_or(Ok(4), |s| s.parse()).unwrap();
        vvc_rect(&dsp, rounds, width, height);
        return;
    }
    if mode == "sao-stats" {
        let width = args.get(3).map_or(Ok(64), |s| s.parse()).unwrap();
        let height = args.get(4).map_or(Ok(64), |s| s.parse()).unwrap();
        let step = args.get(5).map_or(Ok(1), |s| s.parse()).unwrap();
        let class = args.get(6).map_or(Ok(0), |s| s.parse()).unwrap();
        sao_stats(&dsp, rounds, width, height, step, class);
        return;
    }
    if matches!(mode, "satd-quads" | "satd-split") {
        satd_quads(&dsp, rounds, mode == "satd-quads");
        return;
    }
    if mode.starts_with("hevc32-") {
        let height = args.get(3).map_or(Ok(32), |s| s.parse()).unwrap();
        hevc32(&dsp, rounds, mode, height);
        return;
    }
    if mode.starts_with("cdef8-") {
        let direction: u8 = args.get(3).map_or(Ok(3), |s| s.parse()).unwrap();
        cdef8(&dsp, rounds, mode, direction);
        return;
    }
    let source: Vec<u8> = (0..4096).map(|i| (i * 73 + i / 17) as u8).collect();
    let residual: Vec<i16> = source.iter().map(|&x| i16::from(x) - 128).collect();
    let mut out = [0i32; 1024];
    let edge: Vec<u8> = (0..32 * 24).map(|i| (100 + i % 32 / 8) as u8).collect();
    let mut pixels = edge.clone();
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..rounds {
        if mode == "low32" {
            dsp.av1_forward_dct_raster_low(32, black_box(&residual), 32, &mut out);
            sum = sum.wrapping_add(black_box(out[1]) as u64);
        } else if mode.starts_with("tx") {
            let tx = if mode.contains("16") {
                HevcTransform::Dct16
            } else {
                HevcTransform::Dct32
            };
            if mode.ends_with("rows") {
                dsp.hevc_forward_staged_unless_zero(
                    tx,
                    black_box(&residual),
                    tx.side(),
                    0,
                    HevcForwardMode::staged(HevcForwardStaging::Rows),
                    &mut out,
                );
            } else {
                dsp.hevc_forward_unless_zero(tx, black_box(&residual), tx.side(), 0, &mut out);
            }
            sum = sum.wrapping_add(black_box(out[1]) as u64);
        } else if mode == "variance" {
            let a = PlaneView::new(&source, 64, 64, 64).unwrap();
            let b = PlaneView::new(&source[3..], 64, 64, 63).unwrap();
            sum += u64::from(
                dsp.variance(BlockSize::new(64, 32).unwrap(), black_box(&a), &b)
                    .sse,
            );
        } else {
            pixels.copy_from_slice(black_box(&edge));
            let mut view = PlaneViewMut::new(&mut pixels, 32, 24, 24).unwrap();
            let vertical = mode.ends_with('v');
            if mode.starts_with("hevc") {
                dsp.hevc_deblock_luma_edge(&mut view, (8, 8), vertical, [16, 16], 64);
            } else if mode.starts_with("edge8") {
                dsp.vvc_deblock_luma_edge(&mut view, (8, 8), vertical, 88, 16, [7, 7]);
            } else {
                dsp.vvc_deblock_luma_segment(&mut view, (8, 8), vertical, 88, 16, [7, 7]);
            }
            sum += u64::from(black_box(pixels[8 * 32 + 8]));
        }
    }
    println!(
        "{mode} rounds={rounds} ns={:.3} sum={sum}",
        start.elapsed().as_nanos() as f64 / rounds as f64
    );
}

/// Fixed-call profile of the partition seed's actual picture/prediction layout.
/// Subtract runs with different round counts to remove startup instructions;
/// SDE elapsed time is not a hardware speed measurement.
fn satd_quads(dsp: &Dsp, rounds: usize, batch: bool) {
    const PICTURE_STRIDE: usize = 1280;
    let source: Vec<u8> = (0..15 * PICTURE_STRIDE + 16)
        .map(|i| (i * 37 + i / PICTURE_STRIDE * 13) as u8)
        .collect();
    let prediction: Vec<u8> = (0..256).map(|i| (i * 11 + i / 16 * 29) as u8).collect();
    let src = PlaneView::new(&source, PICTURE_STRIDE, 16, 16).unwrap();
    let rf = PlaneView::new(&prediction, 16, 16, 16).unwrap();
    let eight = BlockSize::new(8, 8).unwrap();
    for _ in 0..rounds {
        let costs = if batch {
            dsp.satd_8x8_quads(black_box(&src), black_box(&rf))
        } else {
            core::array::from_fn(|q| {
                let (x, y) = (8 * (q % 2), 8 * (q / 2));
                dsp.satd(
                    eight,
                    &black_box(&src).sub_view(x, y, 8, 8).unwrap(),
                    &black_box(&rf).sub_view(x, y, 8, 8).unwrap(),
                )
            })
        };
        black_box(costs);
    }
    println!("done");
}

/// A whole interior luma block at independent picture-like strides. Fixed
/// input keeps strength and clipping work identical at every iteration.
fn cdef8(dsp: &Dsp, rounds: usize, mode: &str, direction: u8) {
    const EXTENT: usize = 12;
    const SOURCE_STRIDE: usize = 1280;
    const DEST_STRIDE: usize = 1293;
    let input: Vec<u8> = (0..11 * SOURCE_STRIDE + EXTENT)
        .map(|i| 112 + ((i % SOURCE_STRIDE * 7 + i / SOURCE_STRIDE * 3) % 24) as u8)
        .collect();
    let mut output = vec![0; 11 * DEST_STRIDE + EXTENT];
    let src = PlaneView::new(&input, SOURCE_STRIDE, EXTENT, EXTENT).unwrap();
    let mut dst = PlaneViewMut::new(&mut output, DEST_STRIDE, EXTENT, EXTENT).unwrap();
    let strengths = match mode {
        "cdef8-pri" => (15, 0),
        "cdef8-sec" => (0, 4),
        "cdef8-both" => (15, 4),
        "cdef8-off" => (0, 0),
        _ => panic!("unknown CDEF probe"),
    };
    for _ in 0..rounds {
        dsp.cdef_filter_interior(
            8,
            black_box(&src),
            (2, 2),
            strengths,
            direction,
            4,
            black_box(&mut dst),
        );
    }
    black_box(dst);
    println!(
        "{mode} dir={direction} sum={}",
        output.iter().map(|&v| u64::from(v)).sum::<u64>()
    );
}

/// A fixed-call MC probe over picture-sized, unequal source/output strides.
/// Compare two round counts to remove setup; emulation time is not speed evidence.
fn hevc32(dsp: &Dsp, rounds: usize, mode: &str, height: usize) {
    const WIDTH: usize = 32;
    const HALO: usize = 7;
    const SOURCE_STRIDE: usize = 1280;
    const OUTPUT_STRIDE: usize = 1293;
    let size = BlockSize::new(WIDTH, height).expect("public MC height");
    let phase = match mode {
        "hevc32-copy" => (0, 0),
        "hevc32-h" => (2, 0),
        "hevc32-v" => (0, 2),
        "hevc32-2d" => (2, 2),
        _ => panic!("unknown HEVC32 phase"),
    };
    let rows = height + HALO;
    let source: Vec<u8> = (0..3 + (rows - 1) * SOURCE_STRIDE + WIDTH + HALO)
        .map(|i| (i * 73 + i / SOURCE_STRIDE * 137) as u8)
        .collect();
    let mut output = vec![0xA5; 7 + (height - 1) * OUTPUT_STRIDE + WIDTH];
    let src = PlaneView::new(&source[3..], SOURCE_STRIDE, WIDTH + HALO, rows).unwrap();
    let mut dst = PlaneViewMut::new(&mut output[7..], OUTPUT_STRIDE, WIDTH, height).unwrap();
    for _ in 0..rounds {
        dsp.predict(
            Filter::HevcLuma,
            size,
            black_box(&src),
            phase.0,
            phase.1,
            black_box(&mut dst),
        );
    }
    let checksum = output.iter().fold(0_u64, |sum, &x| {
        sum.wrapping_mul(257).wrapping_add(u64::from(x))
    });
    println!("{mode} height={height} rounds={rounds} checksum={checksum}");
}

/// Fixed-count encoder SAO statistics, including the public validation,
/// sampling weights and final partial row. Inputs/views stay outside the loop.
/// Usage: sao-stats ROUNDS WIDTH HEIGHT STEP CLASS. Compare 1000/2000 rounds
/// in the same binary under AVX2/AVX-512. Useful widths are31/32/63/64,
/// heights1/3/32/64 and sampling steps1/2/4; class0..3 selects neighbours.
fn sao_stats(dsp: &Dsp, rounds: usize, width: usize, height: usize, step: usize, class: u8) {
    const SOURCE_STRIDE: usize = 1280;
    const RECON_STRIDE: usize = 1293;
    const SOURCE_OFFSET: usize = 1;
    const RECON_OFFSET: usize = 7;
    assert!((1..=64).contains(&width));
    assert!((1..=129).contains(&height));
    assert!((1..=4).contains(&step));
    assert!(class < 4);
    let (w, h) = (width + 2, height + 2);
    let source: Vec<u8> = (0..SOURCE_OFFSET + (h - 1) * SOURCE_STRIDE + w)
        .map(|i| (i * 73 + i / SOURCE_STRIDE * 137) as u8)
        .collect();
    let reconstruction: Vec<u8> = (0..RECON_OFFSET + (h - 1) * RECON_STRIDE + w)
        .map(|i| (i * 41 + i / RECON_STRIDE * 37 + i / 3) as u8)
        .collect();
    let src = PlaneView::new(&source[SOURCE_OFFSET..], SOURCE_STRIDE, w, h).unwrap();
    let rec = PlaneView::new(&reconstruction[RECON_OFFSET..], RECON_STRIDE, w, h).unwrap();
    let bounds = (1, 1, width + 1, height + 1);
    let mut checksum = 0_i64;
    for _ in 0..rounds {
        let stats =
            dsp.sao_edge_stats_sampled(class, black_box(&src), black_box(&rec), bounds, step);
        let stats = black_box(stats);
        checksum = checksum
            .wrapping_add(stats.sum[0])
            .wrapping_add(stats.count[3]);
    }
    println!(
        "sao-stats width={width} height={height} step={step} class={class} rounds={rounds} checksum={checksum}"
    );
}

/// Fixed-count shared angular predictor, including parameter preparation and
/// public dispatch. Use sides4/8/16/32; directions66 (exact PDPC),58
/// (fractional PDPC) and42 (no PDPC), or their horizontal counterparts.
fn vvc_angular(dsp: &Dsp, rounds: usize, side: usize, direction: u8, oriented: bool) {
    assert!([4, 8, 16, 32].contains(&side));
    assert!((2..=66).contains(&direction));
    let top: Vec<i32> = (0..2 * side)
        .map(|i| ((i * 73 + 19) % 256) as i32)
        .collect();
    let left: Vec<i32> = (0..2 * side)
        .map(|i| ((i * 41 + 137) % 256) as i32)
        .collect();
    let mode = VvcIntraMode::Angular {
        direction,
        luma: true,
    };
    let mut out = vec![0; side * side];
    let mut checksum = 0_u64;
    for _ in 0..rounds {
        if oriented {
            black_box(dsp.vvc_intra_predict_oriented(
                side,
                mode,
                black_box(&top),
                &left,
                128,
                &mut out,
            ));
        } else {
            dsp.vvc_intra_predict(side, mode, black_box(&top), &left, 128, &mut out);
        }
        checksum = checksum.wrapping_add(u64::from(black_box(out[0])));
    }
    println!(
        "vvc-angular side={side} direction={direction} oriented={oriented} rounds={rounds} checksum={checksum}"
    );
}

/// Fixed-call rectangular transform probe, including one/two-sample sides.
/// Force the same binary's ISA with `NC_DSP_ISA`. Compare two round counts to
/// remove startup; only quiet-machine timing establishes throughput.
fn vvc_rect(dsp: &Dsp, rounds: usize, width: usize, height: usize) {
    assert!([1, 2, 4, 8, 16, 32].contains(&width));
    assert!([1, 2, 4, 8, 16, 32].contains(&height));
    let area = width * height;
    assert!(area >= 16);
    let residual: Vec<i16> = (0..=area)
        .map(|i| ((i * 37 + i / width * 101) % 511) as i16 - 255)
        .collect();
    let mut output = vec![0_i32; area + 1];
    let start = Instant::now();
    for _ in 0..rounds {
        dsp.vvc_forward_dct(
            black_box(width),
            black_box(height),
            black_box(&residual[1..]),
            black_box(&mut output[1..]),
        );
    }
    let elapsed = start.elapsed();
    let checksum = output[1..].iter().fold(0_u64, |sum, &v| {
        sum.wrapping_mul(0x100_0000_01b3).wrapping_add(v as u64)
    });
    println!(
        "vvc-rect {width}x{height} rounds={rounds} ns={:.3} checksum={checksum}",
        elapsed.as_nanos() as f64 / rounds as f64
    );
}
