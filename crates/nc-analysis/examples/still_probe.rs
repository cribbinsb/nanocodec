//! Measure the stillness detector (`nc_analysis::still`) as a classifier on
//! a raw I420 clip, against a long-horizon oracle.
//!
//! ```text
//! cargo run --release -p nc-analysis --example still_probe -- \
//!     clip.yuv WIDTH HEIGHT [--frames N]
//! ```
//!
//! The oracle calls an 8x8 block *truly static* at frame t when it is
//! unchanged (8x8 luma SAD and every 4x4 quarter within pre-analysis'
//! noise-scaled static limits) against each of the previous
//! [`HORIZON`] source frames: slow movers that pass a one-frame test drift
//! past the limit over that span, sensor noise does not accumulate. Only
//! frames with a full horizon and whole blocks are scored.
//!
//! Prints one line per detector configuration:
//! `config tp fp fn tn mean_fraction frames>=0.6 frames>=0.8 ema_min ema_max`,
//! where the EMA is the still fraction's running average (weight
//! [`TREND_WEIGHT`], as the encoder's adaptive structure keeps it) over the
//! scored frames.

use std::fs::File;
use std::io::BufReader;
use std::process::ExitCode;

use nc_analysis::still::{StillConfig, StillTracker};
use nc_analysis::{Analyzer, FrameLink};
use nc_common::{PlaneIndex, yuv};
use nc_dsp::Dsp;

/// Weight of the newest frame in the still-fraction running average.
const TREND_WEIGHT: f32 = 0.1;
/// Frames the oracle looks back.
const HORIZON: usize = 8;
const BLOCK: usize = 8;
const QUARTER: usize = 4;
/// Pre-analysis' static limits (`preanalysis::StaticLimits`), repeated here
/// so the oracle uses the same notion of "unchanged".
const MIN_PER_PIXEL: f32 = 1.0;
const MAX_PER_PIXEL: f32 = 10.0;
const NOISE_MARGIN: f32 = 1.5 * nc_analysis::noise::MEAN_ABS_DIFF_PER_SIGMA;
const SUB_BLOCK_FACTOR: f32 = 2.0;

/// Detector configurations measured side by side.
const CONFIGS: [StillConfig; 9] = [
    StillConfig {
        min_run: 1,
        guard: 0,
    },
    StillConfig {
        min_run: 1,
        guard: 1,
    },
    StillConfig {
        min_run: 4,
        guard: 0,
    },
    StillConfig {
        min_run: 4,
        guard: 1,
    },
    StillConfig {
        min_run: 8,
        guard: 0,
    },
    StillConfig {
        min_run: 8,
        guard: 1,
    },
    StillConfig {
        min_run: 8,
        guard: 2,
    },
    StillConfig {
        min_run: 16,
        guard: 1,
    },
    StillConfig {
        min_run: 16,
        guard: 2,
    },
];

#[derive(Default, Clone)]
struct Score {
    tp: u64,
    fp: u64,
    fn_: u64,
    tn: u64,
    fractions: Vec<f32>,
}

fn block_sads(a: &[u8], b: &[u8], width: usize, col: usize, row: usize) -> (u32, [u32; 4]) {
    let mut quads = [0u32; 4];
    for y in 0..BLOCK {
        let start = (row * BLOCK + y) * width + col * BLOCK;
        for x in 0..BLOCK {
            let d = u32::from(a[start + x].abs_diff(b[start + x]));
            quads[(y / QUARTER) * 2 + x / QUARTER] += d;
        }
    }
    (quads.iter().sum(), quads)
}

fn truly_static(
    history: &[Vec<u8>],
    cur: &[u8],
    width: usize,
    col: usize,
    row: usize,
    per_pixel: f32,
) -> bool {
    let block_limit = (per_pixel * (BLOCK * BLOCK) as f32) as u32;
    let quad_limit = (SUB_BLOCK_FACTOR * per_pixel * (QUARTER * QUARTER) as f32) as u32;
    history.iter().all(|old| {
        let (sad, quads) = block_sads(cur, old, width, col, row);
        sad <= block_limit && quads.iter().all(|&q| q <= quad_limit)
    })
}

fn run(path: &str, width: usize, height: usize, frames: usize) -> Result<(), String> {
    let mut reader = BufReader::new(File::open(path).map_err(|e| format!("{path}: {e}"))?);
    let mut analyzer = Analyzer::new(width, height, Dsp::new()).map_err(|e| e.to_string())?;
    let (cols, rows) = (width.div_ceil(BLOCK), height.div_ceil(BLOCK));
    let (whole_cols, whole_rows) = (width / BLOCK, height / BLOCK);
    let mut trackers: Vec<StillTracker> = CONFIGS
        .iter()
        .map(|_| StillTracker::new(cols, rows))
        .collect();
    let mut scores = vec![Score::default(); CONFIGS.len()];
    let mut history: Vec<Vec<u8>> = Vec::new();
    let mut index = 0;
    while index < frames
        && yuv::read_frame(&mut reader, analyzer.next_frame_mut()).map_err(|e| e.to_string())?
    {
        let analysis =
            analyzer.analyze_written_timed(FrameLink::PREVIOUS, &mut nc_analysis::NoTimer);
        let pre = analysis.pre.clone();
        let per_pixel = (NOISE_MARGIN * analysis.noise_sigma).clamp(MIN_PER_PIXEL, MAX_PER_PIXEL);
        let luma = analyzer.frame().plane(PlaneIndex::Y).view();
        let cur: Vec<u8> = luma
            .rows()
            .take(height)
            .flat_map(|r| r[..width].iter().copied())
            .collect();
        let maps: Vec<_> = trackers
            .iter_mut()
            .zip(CONFIGS)
            .map(|(t, c)| t.update(&pre, c))
            .collect();
        if history.len() == HORIZON {
            let oracle: Vec<bool> = (0..whole_rows)
                .flat_map(|r| (0..whole_cols).map(move |c| (c, r)))
                .map(|(c, r)| truly_static(&history, &cur, width, c, r, per_pixel))
                .collect();
            for (score, map) in scores.iter_mut().zip(&maps) {
                for (i, &truth) in oracle.iter().enumerate() {
                    let detected = *map.blocks.get(i % whole_cols, i / whole_cols);
                    match (detected, truth) {
                        (true, true) => score.tp += 1,
                        (true, false) => score.fp += 1,
                        (false, true) => score.fn_ += 1,
                        (false, false) => score.tn += 1,
                    }
                }
                score.fractions.push(map.fraction);
            }
        }
        history.push(cur);
        if history.len() > HORIZON {
            history.remove(0);
        }
        index += 1;
    }
    for (config, score) in CONFIGS.iter().zip(&scores) {
        let n = score.fractions.len().max(1) as f32;
        let mean = score.fractions.iter().sum::<f32>() / n;
        let at = |t: f32| score.fractions.iter().filter(|&&f| f >= t).count() as f32 / n;
        let trend: Vec<f32> = score
            .fractions
            .iter()
            .scan(None, |ema: &mut Option<f32>, &f| {
                let next = ema.map_or(f, |e| e + TREND_WEIGHT * (f - e));
                *ema = Some(next);
                Some(next)
            })
            .collect();
        let ema_min = trend.iter().copied().fold(f32::INFINITY, f32::min);
        let ema_max = trend.iter().copied().fold(0.0, f32::max);
        println!(
            "run{}g{} {} {} {} {} {:.4} {:.4} {:.4} {:.3} {:.3}",
            config.min_run,
            config.guard,
            score.tp,
            score.fp,
            score.fn_,
            score.tn,
            mean,
            at(0.6),
            at(0.8),
            ema_min,
            ema_max
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: still_probe clip.yuv WIDTH HEIGHT [--frames N]";
    let parsed = (|| -> Option<(String, usize, usize, usize)> {
        let frames = match args.get(3).map(String::as_str) {
            Some("--frames") => args.get(4)?.parse().ok()?,
            None => usize::MAX,
            _ => return None,
        };
        Some((
            args.first()?.clone(),
            args.get(1)?.parse().ok()?,
            args.get(2)?.parse().ok()?,
            frames,
        ))
    })();
    let Some((path, width, height, frames)) = parsed else {
        eprintln!("{usage}");
        return ExitCode::FAILURE;
    };
    match run(&path, width, height, frames) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
