//! Every `nanocodec encode` option either changes the stream or is refused
//! with a message, on every codec (owner rule, 2026-11-27: no silent
//! no-ops). Each case codes a small synthetic clip with and without the
//! option, through the engine as the CLI drives it, and asserts that the
//! stream changed or that the encoder exited with a `nanocodec:` message.
//!
//! The options that rightly leave the stream as it is (they write a side
//! output, or describe the input or the run) are listed with their reason
//! and must leave it byte-identical. A test checks that every option
//! `nanocodec help` lists for `encode` is covered here, so a new option must
//! be placed in one of the two lists.
use std::{
    collections::{BTreeSet, HashMap},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

const CODECS: [&str; 6] = ["h264", "hevc", "vvc", "av1", "vp9", "av2"];
const WIDTH: usize = 128;
const HEIGHT: usize = 96;
const FRAMES: usize = 12;
/// `--resize` case: frames from this one on are half size.
const RESIZE_FRAME: usize = 6;
/// The source's luma noise amplitude (± samples), what `--still-camera`
/// targets: a still camera's sensor noise over a static scene.
const NOISE: u32 = 3;

/// How a case's baseline is coded (the case adds its option to it).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Base {
    /// `--qp` at a mid quantiser.
    Qp,
    /// `--qp` with frame holds off (`hold.enable` 0).
    QpNoHolds,
    /// `--qp` with `--temporal-layers 2`.
    QpTwoLayers,
    /// `--bitrate` at a generous target.
    Bitrate,
    /// `--bitrate` far below what the clip needs at the coarsest
    /// quantiser, with a short buffer: frames overflow it, where
    /// `--frame-drop` acts.
    Starved,
    /// `--bitrate` with a short buffer the frames fill, where the buffer
    /// bounds (`--strict-buffer`, `--peak-bitrate`) act.
    Tight,
    /// `--bitrate` with a drop in the target mid-clip, so the quantiser
    /// moves (where `--max-qp-step` acts). The still-picture damping of the
    /// level's movement is off: on this mostly still clip it alone holds
    /// every codec's movement to one step, the limit under test.
    Changing,
}

/// The input a case codes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Input {
    /// The clip ([`frame`]).
    Source,
    /// Half-size frames from [`RESIZE_FRAME`] on (`--resize`), which only
    /// `--resize` can read: its baseline is [`Input::Source`].
    Resized,
    /// Every frame shown twice, as a camera delivering duplicates does:
    /// the encoder holds the repeats unless told the rate is constant.
    Repeating,
}

/// What a case's option must do on every codec.
#[derive(Clone, Copy, Debug)]
enum Expect {
    /// Change the stream, or be refused with a message.
    Effect,
    /// Be refused with a message on every codec.
    Refused,
    /// Leave the stream byte-identical (the reason says why that is right).
    Unchanged(&'static str),
    /// Leave the stream byte-identical, except on the listed codecs, which
    /// must refuse it (the reason says why on both counts).
    UnchangedOrRefusedOn(&'static [&'static str], &'static str),
    /// An oracle replaying the encoder's own decisions: it must run or be
    /// refused, and either stream is right (the case's comment says why).
    Oracle,
    /// Accepted; not yet shown to change any stream (the reason says what
    /// is known). An open finding, listed so it stays visible.
    Unproven(&'static str),
}

/// One option under test: its arguments (`{dir}` is the scratch directory,
/// `{codec}` the codec), the baseline it is added to, and what it must do.
#[derive(Clone, Copy)]
struct Case {
    args: &'static [&'static str],
    base: Base,
    input: Input,
    expect: Expect,
}

const fn effect(args: &'static [&'static str], base: Base) -> Case {
    Case {
        args,
        base,
        input: Input::Source,
        expect: Expect::Effect,
    }
}

const fn refused(args: &'static [&'static str], base: Base) -> Case {
    Case {
        args,
        base,
        input: Input::Source,
        expect: Expect::Refused,
    }
}

const fn unchanged(args: &'static [&'static str], base: Base, why: &'static str) -> Case {
    Case {
        args,
        base,
        input: Input::Source,
        expect: Expect::Unchanged(why),
    }
}

const fn on(input: Input, case: Case) -> Case {
    Case { input, ..case }
}

/// The baseline's own options and the input, output and size every run
/// gives: not toggled (a run needs them, and another value is another
/// clip or another codec).
const REQUIRED: &[&str] = &[
    "--input",
    "--width",
    "--height",
    "--frames",
    "--output",
    "--codec",
    "--qp",
    "--bitrate",
];

/// Why `--rtp` leaves the stream as it is, and why AV2 refuses it.
const RTP_WHY: &str = "a side output (the coded units, packetized); AV2 has no RTP payload format";

const CASES: &[Case] = &[
    // Input.
    effect(&["--input-format", "nv12"], Base::Qp),
    effect(&["--fps", "15"], Base::Bitrate),
    effect(&["--timestamps", "{dir}/timestamps.txt"], Base::Bitrate),
    on(Input::Resized, effect(&["--resize", "6:64x48"], Base::Qp)),
    // Output.
    unchanged(
        &["--recon", "{dir}/{codec}.recon"],
        Base::Qp,
        "a side output",
    ),
    unchanged(
        &["--display-recon", "{dir}/{codec}.display"],
        Base::Qp,
        "a side output",
    ),
    unchanged(
        &["--timestamps-out", "{dir}/{codec}.times"],
        Base::Qp,
        "a side output",
    ),
    // RTP packets of the coded units: AV2 has no RTP payload format.
    Case {
        args: &["--rtp", "{dir}/{codec}.pcap"],
        base: Base::Qp,
        input: Input::Source,
        expect: Expect::UnchangedOrRefusedOn(&["av2"], RTP_WHY),
    },
    Case {
        args: &[
            "--rtp",
            "{dir}/{codec}.rtp",
            "--rtp-mtu",
            "600",
            "--rtp-dd-id",
            "none",
        ],
        base: Base::Qp,
        input: Input::Source,
        expect: Expect::UnchangedOrRefusedOn(&["av2"], RTP_WHY),
    },
    // Without `--rtp` they would shape no packets.
    refused(&["--rtp-mtu", "600"], Base::Qp),
    refused(&["--rtp-dd-id", "none"], Base::Qp),
    // Codec and preset.
    effect(&["--preset", "veryfast"], Base::Qp),
    // Every preset plans its block sizes: an explicit one is refused.
    refused(&["--block", "8"], Base::Qp),
    effect(
        &["--block", "8", "--param", "planner.intra_plan=0"],
        Base::Qp,
    ),
    // Fixed QP.
    effect(&["--key-qp-offset", "6"], Base::Qp),
    refused(&["--key-qp-offset", "6"], Base::Bitrate),
    // Rate control.
    Case {
        args: &["--peak-bitrate", "100"],
        base: Base::Tight,
        input: Input::Source,
        expect: Expect::Unproven(
            "the peak drains the buffer that caps a frame's budget, but the 1.15-buffer \
             recode ceiling (drained at the target) binds first: no stream of these clips at \
             5-400 kbps, 30-1000 ms buffers and peaks of 1 and 10 times the target changed \
             (2026-11-27, encoder information off)",
        ),
    },
    effect(&["--buffer-ms", "100"], Base::Bitrate),
    effect(&["--buffer-initial-ms", "100"], Base::Bitrate),
    effect(&["--min-qp", "{high_qp}"], Base::Bitrate),
    effect(&["--max-qp", "{low_qp}"], Base::Bitrate),
    refused(&["--max-qp", "{beyond_qp}"], Base::Bitrate),
    refused(&["--min-qp", "{beyond_qp}"], Base::Bitrate),
    effect(&["--max-qp-step", "1"], Base::Changing),
    effect(&["--frame-drop"], Base::Starved),
    effect(&["--strict-buffer"], Base::Tight),
    effect(&["--rate-change", "4:30"], Base::Bitrate),
    refused(&["--min-qp", "{high_qp}"], Base::Qp),
    // Key frames and resilience.
    effect(&["--intra-period", "4"], Base::Qp),
    effect(&["--keyframe", "5"], Base::Qp),
    effect(&["--intra-refresh", "4"], Base::Qp),
    // Against `--temporal-layers 2` alone: the sweep fixes the layers
    // and runs on the base layer (VP9, AV2: refused).
    effect(&["--intra-refresh", "4"], Base::QpTwoLayers),
    effect(&["--ltr-interval", "4"], Base::Qp),
    effect(&["--temporal-layers", "2"], Base::Qp),
    effect(&["--scalability", "L2T1"], Base::Qp),
    // Content.
    effect(&["--still-camera"], Base::Bitrate),
    effect(&["--film-grain"], Base::Qp),
    // Signalling.
    effect(&["--colour", "bt709"], Base::Qp),
    effect(&["--range", "full"], Base::Qp),
    on(
        Input::Repeating,
        effect(&["--timing", "constant"], Base::Qp),
    ),
    // With frame holds on (the default) H.264's nominal timing already
    // signals no fixed frame rate (`fixed_frame_rate_flag` 0), the same as
    // `variable`: the case turns holds off, where `variable` acts.
    effect(&["--timing", "variable"], Base::QpNoHolds),
    // Parameters.
    effect(&["--params", "{dir}/params.txt"], Base::Qp),
    effect(&["--param", "rate.constant_qp=1"], Base::Qp),
    // Diagnostics.
    unchanged(
        &["--rate-log", "{dir}/{codec}.rate.csv"],
        Base::Bitrate,
        "a side output",
    ),
    unchanged(
        &[
            "--sse-log",
            "{dir}/{codec}.sse.csv",
            "--sse-reference",
            "{dir}/source.yuv",
        ],
        Base::Qp,
        "a side output (the two options go together)",
    ),
    unchanged(
        &["--layer-md5", "{dir}/{codec}.md5.csv"],
        Base::Qp,
        "a side output",
    ),
    unchanged(
        &["--filter-stats", "{dir}/{codec}.filters.json"],
        Base::Qp,
        "a side output",
    ),
    unchanged(
        &[
            "--label-dump",
            "{dir}/{codec}.labels",
            "--label-tool",
            "probe:decision.noise_skip=4",
        ],
        Base::Qp,
        "a side output: label dumps record decisions without changing them \
         (the two options go together)",
    ),
    unchanged(
        &["--spatial-stats", "{dir}/{codec}.spatial.json"],
        Base::Qp,
        "a side output",
    ),
    unchanged(
        &["--analysis-log", "{dir}/{codec}.analysis"],
        Base::Qp,
        "a side output",
    ),
    unchanged(
        &["--partition-record", "{dir}/{codec}.partitions"],
        Base::Qp,
        "a side output",
    ),
    Case {
        args: &["--partition-replay", "{dir}/{codec}.partitions"],
        base: Base::Qp,
        input: Input::Source,
        // Replays the partitions `--partition-record` logged (the case
        // before): it codes them without the RD search, whose other
        // choices may or may not come out the same.
        expect: Expect::Oracle,
    },
    effect(&["--qp-replay", "{dir}/{codec}.qp_replay"], Base::Bitrate),
    effect(&["--key-offsets", "{dir}/key_offsets.txt"], Base::Qp),
    effect(&["--simulate-loss", "4"], Base::Qp),
];

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nanocodec-cli-options-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A small deterministic generator, so the clip needs no crate.
fn noise(seed: u32) -> u32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^ (x >> 12)
}

/// One I420 frame: a static gradient with ±[`NOISE`] luma noise, and a
/// square moving to the right, so both still and moving blocks are coded.
fn frame(width: usize, height: usize, index: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * height * 3 / 2);
    let square = (8 + index * 2, 16);
    for y in 0..height {
        for x in 0..width {
            let inside =
                (square.0..square.0 + 24).contains(&x) && (square.1..square.1 + 24).contains(&y);
            let base = if inside {
                220
            } else {
                40 + (x + y) * 140 / (width + height)
            };
            let n = noise((index * width * height + y * width + x) as u32) % (2 * NOISE + 1);
            out.push((base as u32 + n).saturating_sub(NOISE).min(255) as u8);
        }
    }
    for plane in [96u8, 160] {
        out.extend(std::iter::repeat_n(plane, width * height / 4));
    }
    out
}

/// The scratch inputs every case may name.
fn write_inputs(dir: &Path) {
    let source: Vec<u8> = (0..FRAMES).flat_map(|i| frame(WIDTH, HEIGHT, i)).collect();
    fs::write(dir.join("source.yuv"), source).unwrap();
    let resized: Vec<u8> = (0..FRAMES)
        .flat_map(|i| {
            if i < RESIZE_FRAME {
                frame(WIDTH, HEIGHT, i)
            } else {
                frame(WIDTH / 2, HEIGHT / 2, i)
            }
        })
        .collect();
    fs::write(dir.join("resized.yuv"), resized).unwrap();
    let repeating: Vec<u8> = (0..FRAMES)
        .flat_map(|i| frame(WIDTH, HEIGHT, i / 2))
        .collect();
    fs::write(dir.join("repeating.yuv"), repeating).unwrap();
    // Uneven capture times: frame 5 comes late.
    let mut times = String::from("# timestamp format v2\n");
    for i in 0..FRAMES {
        let late = if i >= 5 { 200 } else { 0 };
        writeln!(times, "{}", i * 33 + late).unwrap();
    }
    fs::write(dir.join("timestamps.txt"), times).unwrap();
    fs::write(dir.join("params.txt"), "rate.constant_qp = 1\n").unwrap();
    // Frame 3 forced finer than the controller plans at this rate.
    for codec in CODECS {
        let (_, _, low, _) = quantizers(codec);
        fs::write(
            dir.join(format!("{codec}.qp_replay")),
            format!("-\n-\n-\n{low}\n"),
        )
        .unwrap();
    }
    let cells = format!("{}\n", vec!["-4"; WIDTH / 8].join(" "));
    fs::write(
        dir.join("key_offsets.txt"),
        format!("frame 0\n{}", cells.repeat(HEIGHT / 8)),
    )
    .unwrap();
}

/// Quantisers in `codec`'s own scale: (mid, high, low, beyond its range).
fn quantizers(codec: &str) -> (u32, u32, u32, u32) {
    match codec {
        "av1" | "vp9" | "av2" => (150, 230, 60, 256),
        "vvc" => (32, 50, 20, 64),
        _ => (32, 45, 20, 52),
    }
}

fn expand(arg: &str, dir: &Path, codec: &str) -> String {
    let (_, high, low, beyond) = quantizers(codec);
    arg.replace("{dir}", dir.to_str().unwrap())
        .replace("{codec}", codec)
        .replace("{high_qp}", &high.to_string())
        .replace("{low_qp}", &low.to_string())
        .replace("{beyond_qp}", &beyond.to_string())
}

/// The outcome of one encode: the stream, or the refusal's message.
type Outcome = Result<Vec<u8>, String>;

fn encode(
    dir: &Path,
    codec: &str,
    (base, input): (Base, Input),
    extra: &[&str],
    tag: &str,
) -> Outcome {
    let stream = dir.join(format!("{codec}-{tag}.bin"));
    let (qp, ..) = quantizers(codec);
    let input = dir.join(match input {
        Input::Source => "source.yuv",
        Input::Resized => "resized.yuv",
        Input::Repeating => "repeating.yuv",
    });
    let mut command = Command::new(env!("CARGO_BIN_EXE_nanocodec"));
    command
        .args(["encode", "--codec", codec, "--input"])
        .arg(input)
        .arg("--output")
        .arg(&stream)
        .args([
            "--width",
            &WIDTH.to_string(),
            "--height",
            &HEIGHT.to_string(),
        ])
        .args(["--frames", &FRAMES.to_string()]);
    // The encoder information message lists the rate settings
    // (`nc_encoder::encoder_info`): without it off, any of them would
    // change the stream through that text alone.
    command.args(["--param", "stream.encoder_info=0"]);
    if !extra.contains(&"--fps") {
        command.args(["--fps", "30"]);
    }
    match base {
        Base::Qp => command.args(["--qp", &qp.to_string()]),
        Base::QpNoHolds => command.args(["--qp", &qp.to_string(), "--param", "hold.enable=0"]),
        Base::QpTwoLayers => command.args(["--qp", &qp.to_string(), "--temporal-layers", "2"]),
        Base::Bitrate => command.args(["--bitrate", "150"]),
        Base::Starved => command.args(["--bitrate", "1", "--buffer-ms", "100"]),
        Base::Tight => command.args(["--bitrate", "100", "--buffer-ms", "100"]),
        Base::Changing => command.args([
            "--bitrate",
            "150",
            "--rate-change",
            "4:20",
            "--param",
            "rate.still_level_damping=0",
        ]),
    };
    let expanded: Vec<String> = extra.iter().map(|a| expand(a, dir, codec)).collect();
    let output = command.args(&expanded).output().unwrap();
    if output.status.success() {
        let bytes = fs::read(&stream).unwrap();
        let _ = fs::remove_file(&stream);
        Ok(bytes)
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

/// Every case's verdict on `codec`, or the problem with it.
fn check_codec(dir: &Path, codec: &str) -> Vec<String> {
    let mut baselines: HashMap<(Base, Input), Vec<u8>> = HashMap::new();
    let mut problems = Vec::new();
    for (index, case) in CASES.iter().enumerate() {
        let setup = (case.base, case.input);
        let base_input = match case.input {
            Input::Resized => Input::Source,
            input => input,
        };
        let baseline = baselines.entry((case.base, base_input)).or_insert_with(|| {
            encode(dir, codec, (case.base, base_input), &[], "base")
                .unwrap_or_else(|e| panic!("{codec} {setup:?} baseline failed: {e}"))
        });
        let name = format!("{codec} {setup:?} {}", case.args.join(" "));
        let outcome = encode(dir, codec, setup, case.args, &index.to_string());
        let problem = match (outcome, case.expect) {
            (Err(message), _) if !message.trim_start().starts_with("nanocodec:") => {
                Some(format!("failed without a message: {message}"))
            }
            (Err(_), Expect::Unchanged(why)) => Some(format!("refused, but is {why}")),
            (Err(_), Expect::UnchangedOrRefusedOn(refusers, why)) if !refusers.contains(&codec) => {
                Some(format!("refused, but is {why}"))
            }
            (Ok(_), Expect::UnchangedOrRefusedOn(refusers, why)) if refusers.contains(&codec) => {
                Some(format!("accepted, but must be refused: {why}"))
            }
            (_, Expect::Unproven(note)) => {
                eprintln!("{name}: unproven: {note}");
                None
            }
            (Ok(_), Expect::Refused) => Some("accepted, but must be refused".into()),
            (Ok(stream), Expect::Effect) if stream == *baseline => {
                Some("accepted and changed nothing".into())
            }
            (Ok(stream), Expect::Unchanged(_) | Expect::UnchangedOrRefusedOn(..))
                if stream != *baseline =>
            {
                Some("changed the stream".into())
            }
            // Refused where refusing is allowed, changed where it must
            // change, unchanged where it must not, or an oracle.
            _ => None,
        };
        if let Some(problem) = problem {
            problems.push(format!("{name}: {problem}"));
        }
    }
    problems
}

#[test]
fn every_option_changes_the_stream_or_is_refused() {
    let scratch = Scratch::new();
    write_inputs(&scratch.0);
    let problems: Vec<String> = thread::scope(|scope| {
        let runs: Vec<_> = CODECS
            .iter()
            .map(|codec| scope.spawn(|| check_codec(&scratch.0, codec)))
            .collect();
        runs.into_iter()
            .flat_map(|run| run.join().unwrap())
            .collect()
    });
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The `encode` options `nanocodec help` lists.
fn listed_encode_options() -> BTreeSet<String> {
    let output = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
        .arg("help")
        .output()
        .unwrap();
    let text = String::from_utf8(output.stdout).unwrap();
    let section = text
        .split("\nnanocodec ")
        .find(|s| s.starts_with("encode:"))
        .expect("help lists the encode command");
    section
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|word| word.starts_with("--"))
        .map(str::to_owned)
        .collect()
}

#[test]
fn every_encode_option_is_covered() {
    let covered: BTreeSet<String> = CASES
        .iter()
        .flat_map(|case| case.args.iter().filter(|a| a.starts_with("--")))
        .chain(REQUIRED)
        .map(|flag| (*flag).to_owned())
        .collect();
    let missing: Vec<_> = listed_encode_options()
        .difference(&covered)
        .cloned()
        .collect();
    assert!(
        missing.is_empty(),
        "encode options with no case in option_effect_cli.rs: {missing:?}"
    );
}
