//! Run the analysis pipeline over a raw I420 clip and report per-stage time.
//!
//! ```text
//! cargo run --release -p nc-analysis --example analyze -- \
//!     clip.yuv WIDTH HEIGHT [--frames N] [--verbose]
//! ```
//!
//! For meaningful timings pin to one core (`taskset -c 3 ...`). Timings are
//! wall clock per stage, summed over frames 2..N (frame 1 has no motion).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::BufReader;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use nc_analysis::effort::MotionEffort;
use nc_analysis::{Analyzer, FrameAnalysis, FrameLink, Stage, StageTimer};
use nc_common::yuv;
use nc_dsp::Dsp;

/// Accumulates wall time per stage.
#[derive(Default)]
struct WallClock {
    started: HashMap<Stage, Instant>,
    totals: HashMap<Stage, Duration>,
}

impl StageTimer for WallClock {
    fn begin(&mut self, stage: Stage) {
        self.started.insert(stage, Instant::now());
    }

    fn end(&mut self, stage: Stage) {
        let elapsed = self.started[&stage].elapsed();
        *self.totals.entry(stage).or_default() += elapsed;
    }
}

struct Args {
    path: String,
    width: usize,
    height: usize,
    frames: usize,
    verbose: bool,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: analyze clip.yuv WIDTH HEIGHT [--frames N] [--verbose]";
    if raw.len() < 3 {
        return Err(usage.into());
    }
    let number = |s: &str| {
        s.parse::<usize>()
            .map_err(|_| format!("bad number '{s}'\n{usage}"))
    };
    let mut args = Args {
        path: raw[0].clone(),
        width: number(&raw[1])?,
        height: number(&raw[2])?,
        frames: usize::MAX,
        verbose: false,
    };
    let mut rest = raw[3..].iter();
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--frames" => args.frames = number(rest.next().ok_or(usage)?)?,
            "--verbose" => args.verbose = true,
            other => return Err(format!("unknown argument '{other}'\n{usage}")),
        }
    }
    Ok(args)
}

fn summary_line(index: usize, a: &FrameAnalysis) -> String {
    let mut line = format!(
        "frame {index:4}: static {:5.1}%  noise {:4.2}",
        100.0 * a.pre.static_fraction,
        a.noise_sigma
    );
    if let Some(m) = &a.motion {
        let mean_sad = m.residual_per_sample();
        let _ = write!(
            line,
            "  sad/px {mean_sad:5.2}  compensable {:4.2}  global ({:+6.1},{:+6.1}) zoom {:+.4} inliers {:3.0}%  \
             SB paths {:?}",
            m.compensable_share(),
            m.global.tx,
            m.global.ty,
            m.global.zoom,
            100.0 * m.global.inlier_fraction,
            m.effort.sb_paths
        );
    }
    line
}

fn run(args: &Args) -> Result<(), String> {
    let file = File::open(&args.path).map_err(|e| format!("{}: {e}", args.path))?;
    let mut reader = BufReader::new(file);
    let dsp = Dsp::new();
    let mut analyzer = Analyzer::new(args.width, args.height, dsp).map_err(|e| e.to_string())?;
    let mut clock = WallClock::default();
    let mut effort = MotionEffort::default();
    let mut sad_8x8_total = 0.0;
    let mut frames = 0;
    // Frames are read straight into the analyzer's buffer (no copy).
    while frames < args.frames
        && yuv::read_frame(&mut reader, analyzer.next_frame_mut()).map_err(|e| e.to_string())?
    {
        if frames == 1 {
            // Time only frames with motion analysis.
            clock.totals.clear();
        }
        let analysis = analyzer.analyze_written_timed(FrameLink::PREVIOUS, &mut clock);
        if let Some(m) = &analysis.motion {
            effort.accumulate(&m.effort);
            sad_8x8_total += m.fine.cells().iter().map(|b| f64::from(b.sad)).sum::<f64>();
        }
        if args.verbose {
            println!("{}", summary_line(frames, analysis));
        }
        frames += 1;
    }
    let timed_frames = frames.saturating_sub(1).max(1) as f64;
    println!(
        "{frames} frames {}x{} on {} ({}); ms/frame over frames 2..{frames}:",
        args.width,
        args.height,
        dsp.isa(),
        args.path
    );
    let mut total = 0.0;
    for stage in Stage::ALL {
        let ms = clock
            .totals
            .get(&stage)
            .map_or(0.0, |d| d.as_secs_f64() * 1e3)
            / timed_frames;
        total += ms;
        println!("  {:<14} {ms:7.3}", format!("{stage:?}"));
    }
    println!("  {:<14} {total:7.3}", "total");
    println!(
        "final 8x8 SAD per frame {:.0} ({:.3}/px): the search-quality measure",
        sad_8x8_total / timed_frames,
        sad_8x8_total / timed_frames / (args.width * args.height) as f64
    );
    print_effort(&effort, &clock, timed_frames);
    Ok(())
}

/// Search effort: superblocks per search path, positions per searched SB,
/// and motion-stage time per SB.
fn print_effort(effort: &MotionEffort, clock: &WallClock, timed_frames: f64) {
    let per_frame = |n: u32| f64::from(n) / timed_frames;
    let [still, one, window, hme] = effort.sb_paths;
    println!(
        "superblocks/frame: static {:.1}, one position {:.1}, window at zero {:.1}, hierarchical {:.1}",
        per_frame(still),
        per_frame(one),
        per_frame(window),
        per_frame(hme)
    );
    println!(
        "  HME        {:6.1} SBs/frame, {:6.1} quarter-res positions each, {:.1} seed checks/frame",
        per_frame(effort.hme.blocks),
        effort.hme.mean_positions(),
        per_frame(effort.hme_seeds)
    );
    println!(
        "  full-pel   {:6.1} SBs/frame, {:6.1} 8x8-grid positions each, {:.1} mixed-motion SBs/frame",
        per_frame(effort.full_pel.blocks),
        effort.full_pel.mean_positions(),
        per_frame(effort.mixed_sbs)
    );
    let total_sbs: u32 = effort.sb_paths.iter().sum();
    if let (Some(time), true) = (clock.totals.get(&Stage::Motion), total_sbs > 0) {
        println!(
            "  motion us per SB: {:.2}",
            time.as_secs_f64() * 1e6 / f64::from(total_sbs)
        );
    }
}

fn main() -> ExitCode {
    match parse_args().and_then(|args| run(&args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
