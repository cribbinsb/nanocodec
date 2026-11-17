//! `nc-checkasm`: verify SIMD kernels against scalar, or benchmark them.
//!
//! ```text
//! nc-checkasm [--seed N] [--iterations N]   verify every supported ISA
//! nc-checkasm --bench                        ns/call per kernel and ISA
//! nc-checkasm --bench-encoder-wide           forward/intra encoder kernels
//! nc-checkasm --bench-av1-runs               only AV1 edge/run filters
//! nc-checkasm --bench-vp9-mc                 only VP9 interpolation
//! ```
//!
//! Exits non-zero if any kernel disagrees with its scalar reference.

#![forbid(unsafe_code)]

mod replay_filter_runs;
mod replay_hevc_runs;

use std::process::ExitCode;

use nc_checkasm::{
    BenchResult, bench, bench_av1_inverse, bench_av1_runs, bench_cdef, bench_copies,
    bench_deblock_batches, bench_encoder_wide, bench_vp9_mc, check,
};
use nc_dsp::Isa;

const DEFAULT_SEED: u64 = 1;
const DEFAULT_ITERATIONS: usize = 200;
/// Mismatch lines printed per ISA before summarising.
const MAX_PRINTED_FAILURES: usize = 20;

/// Which kernels `--bench*` times.
#[derive(Clone, Copy)]
enum BenchSet {
    All,
    EncoderWide,
    Av1Inverse,
    Av1Runs,
    Cdef,
    Copies,
    DeblockBatches,
    Vp9Mc,
}

struct Options {
    bench: Option<BenchSet>,
    filter_runs: Option<String>,
    run_batch: bool,
    seed: u64,
    iterations: usize,
    isa: Option<Isa>,
}

fn main() -> ExitCode {
    let options = match parse_args(std::env::args().skip(1)) {
        Ok(o) => o,
        Err(message) => {
            eprintln!("nc-checkasm: {message}");
            eprintln!(
                "usage: nc-checkasm [--bench|--bench-encoder-wide|--bench-av1-inverse|--bench-av1-runs|--bench-deblock-batches|--bench-cdef|--bench-copies|--bench-vp9-mc] [--isa ISA] [--seed N] [--iterations N]"
            );
            return ExitCode::from(2);
        }
    };
    if let Some(path) = &options.filter_runs {
        return match replay_filter_runs::replay(path, options.run_batch, options.isa) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("nc-checkasm: {message}");
                ExitCode::FAILURE
            }
        };
    }
    if let Some(set) = options.bench {
        let isas = options
            .isa
            .as_ref()
            .map_or(Isa::ALL.as_slice(), std::slice::from_ref);
        let results = match set {
            BenchSet::All => bench(isas),
            BenchSet::EncoderWide => bench_encoder_wide(isas),
            BenchSet::Av1Inverse => bench_av1_inverse(isas),
            BenchSet::Av1Runs => bench_av1_runs(isas),
            BenchSet::Cdef => bench_cdef(isas),
            BenchSet::Copies => bench_copies(isas),
            BenchSet::DeblockBatches => bench_deblock_batches(isas),
            BenchSet::Vp9Mc => bench_vp9_mc(isas),
        };
        print_bench(&results);
        return ExitCode::SUCCESS;
    }
    run_checks(&options)
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut options = Options {
        bench: None,
        filter_runs: None,
        run_batch: true,
        seed: DEFAULT_SEED,
        iterations: DEFAULT_ITERATIONS,
        isa: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" | "bench" => options.bench = Some(BenchSet::All),
            "--bench-encoder-wide" => options.bench = Some(BenchSet::EncoderWide),
            "--bench-av1-inverse" => options.bench = Some(BenchSet::Av1Inverse),
            "--bench-av1-runs" => options.bench = Some(BenchSet::Av1Runs),
            "--bench-cdef" => options.bench = Some(BenchSet::Cdef),
            "--bench-copies" => options.bench = Some(BenchSet::Copies),
            "--bench-deblock-batches" => options.bench = Some(BenchSet::DeblockBatches),
            "--bench-vp9-mc" => options.bench = Some(BenchSet::Vp9Mc),
            "--filter-runs" => {
                options.filter_runs = Some(
                    args.next()
                        .ok_or("--filter-runs needs a file or - for stdin")?,
                );
            }
            "--run-dispatch" => {
                options.run_batch = match args.next().as_deref() {
                    Some("batch") => true,
                    Some("single") => false,
                    _ => return Err("--run-dispatch needs batch or single".into()),
                }
            }
            "--isa" => {
                let value = args.next().ok_or("--isa needs an ISA name")?;
                let isa: Isa = value.parse()?;
                if !isa.is_supported() {
                    return Err(format!("{isa} is unavailable"));
                }
                options.isa = Some(isa);
            }
            "--seed" => options.seed = parse_value(&arg, args.next().as_deref())?,
            "--iterations" => options.iterations = parse_value(&arg, args.next().as_deref())?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(options)
}

fn parse_value<T: std::str::FromStr>(flag: &str, value: Option<&str>) -> Result<T, String> {
    value
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("{flag} needs a numeric value"))
}

fn run_checks(options: &Options) -> ExitCode {
    let mut all_passed = true;
    for isa in Isa::ALL.into_iter().filter(|&isa| {
        options
            .isa
            .map_or(isa != Isa::Scalar, |chosen| chosen == isa)
    }) {
        let Some(report) = check(isa, options.seed, options.iterations) else {
            println!("{isa:>8}: not available on this machine/build, skipped");
            continue;
        };
        if report.passed() {
            println!("{isa:>8}: ok ({} cases)", report.cases);
            continue;
        }
        all_passed = false;
        println!(
            "{isa:>8}: FAILED {} of {} cases",
            report.failures.len(),
            report.cases
        );
        for line in report.failures.iter().take(MAX_PRINTED_FAILURES) {
            println!("    {line}");
        }
    }
    if all_passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn print_bench(results: &[BenchResult]) {
    println!(
        "{:<20} {:>6} {:<8} {:>10} {:>9}",
        "kernel", "size", "isa", "ns/call", "vs scalar"
    );
    for r in results {
        let scalar = results
            .iter()
            .find(|s| s.isa == Isa::Scalar && s.kernel == r.kernel && s.size == r.size)
            .map_or(f64::NAN, |s| s.ns_per_call);
        println!(
            "{:<20} {:>6} {:<8} {:>10.2} {:>8.1}x",
            r.kernel,
            format!("{}x{}", r.size.0, r.size.1),
            r.isa,
            r.ns_per_call,
            scalar / r.ns_per_call
        );
    }
}
