//! The command line's commands and options, grouped: the one list that
//! `nanocodec help` prints and that the project site's command-line page is
//! generated from (`web/docs/data/cli.json`, written by the
//! `cli_reference_is_current` test with `NC_BLESS=1`). A test checks that
//! every option the parsers read is listed here and every listed one is read,
//! so the help and the page cannot drift from the code.

use std::fmt::Write as _;

/// One option of a command.
pub struct OptionDoc {
    /// The flag, e.g. `--bitrate`.
    pub flag: &'static str,
    /// Its value's placeholder (empty for a switch).
    pub value: &'static str,
    /// What it does, one or two sentences.
    pub help: &'static str,
}

/// A titled group of a command's options.
pub struct GroupDoc {
    /// The group's heading, e.g. "Rate control (CBR) and buffer".
    pub title: &'static str,
    /// Its options, in the order they are listed.
    pub options: &'static [OptionDoc],
}

/// A subcommand: its name, a one-line description and its option groups.
pub struct CommandDoc {
    /// The subcommand, e.g. `encode`.
    pub name: &'static str,
    /// What it does, one line.
    pub about: &'static str,
    /// Its options, grouped.
    pub groups: &'static [GroupDoc],
}

const fn opt(flag: &'static str, value: &'static str, help: &'static str) -> OptionDoc {
    OptionDoc { flag, value, help }
}

const CODECS: &str = "av1|h264|hevc|vvc|vp9|av2";

const ENCODE_GROUPS: &[GroupDoc] = &[
    GroupDoc {
        title: "Input",
        options: &[
            opt(
                "--input",
                "PATH",
                "Raw 8-bit 4:2:0 frames, back to back; `-` reads stdin.",
            ),
            opt(
                "--input-format",
                "i420|yv12|nv12",
                "Plane layout of the input (default i420).",
            ),
            opt("--width", "W", "Frame width in pixels (even)."),
            opt("--height", "H", "Frame height in pixels (even)."),
            opt(
                "--fps",
                "F",
                "Nominal frame rate: `30`, `29.97` or `30000/1001`.",
            ),
            opt("--frames", "N", "Number of input frames to code."),
            opt(
                "--timestamps",
                "FILE",
                "Capture time of each frame (mkvmerge timestamp format v2); rate control budgets \
                 the real frame intervals.",
            ),
            opt(
                "--resize",
                "FRAME:WxH",
                "The input's frames have this size from FRAME on (a key frame with new headers); \
                 repeatable.",
            ),
        ],
    },
    GroupDoc {
        title: "Output",
        options: &[
            opt(
                "--output",
                "PATH",
                "The stream: IVF for AV1, VP9 and AV2, Annex-B for H.264, HEVC and VVC; `-` \
                 writes stdout.",
            ),
            opt(
                "--recon",
                "PATH",
                "The reconstruction of every coded frame (I420).",
            ),
            opt(
                "--display-recon",
                "PATH",
                "One picture per input frame, as a decoder shows it: a frame left uncoded repeats \
                 the last one.",
            ),
            opt("--timestamps-out", "FILE", "The time of every coded frame."),
            opt(
                "--rtp",
                "PATH",
                "The coded units as RTP packets: pcap for .pcap, otherwise RFC 4571 framing. \
                 AV1 carries the Dependency Descriptor (extension id 1), VP9 its layer indices.",
            ),
            opt(
                "--rtp-mtu",
                "BYTES",
                "Largest RTP packet of --rtp (default 1200).",
            ),
            opt(
                "--rtp-dd-id",
                "ID|none",
                "Dependency Descriptor extension id of --rtp (AV1 default 1, others none).",
            ),
        ],
    },
    GroupDoc {
        title: "Codec and preset",
        options: &[
            opt(
                "--codec",
                CODECS,
                "Output format (`h266` is accepted for vvc). Required.",
            ),
            opt(
                "--preset",
                "veryfast|faster|fast|medium|slow|slower|veryslow",
                "A rung of the speed ladder; medium is the codec's Default and is used without \
                 --preset. A name is a position on that codec's ladder (in this order); not every \
                 codec has every rung (`nanocodec params --codec C`); with \
                 --scalability L2*/L3* a rung takes the codec's file for that many layers.",
            ),
            opt(
                "--block",
                "8|16|32|64",
                "Fixed key-frame leaf size (default 16; AV2 8, VVC 32), coded only with the key \
                 planner off (`planner.intra_plan` 0; H.264 is always 16): refused where the \
                 planner chooses every block size, as the presets do.",
            ),
        ],
    },
    GroupDoc {
        title: "Fixed QP",
        options: &[
            opt(
                "--qp",
                "Q",
                "Fixed quantiser in the codec's scale (H.264, HEVC: QP 0-51; VVC: QP 0-63; AV1, \
                 VP9, AV2: base_q_idx); with --bitrate, the starting quantiser only. One of --qp \
                 and --bitrate is required.",
            ),
            opt(
                "--key-qp-offset",
                "D",
                "Code fixed-QP key frames D quantiser steps finer than --qp (default: the codec \
                 chooses). Refused with --bitrate: rate control sizes key frames itself.",
            ),
        ],
    },
    GroupDoc {
        title: "Rate control (CBR) and buffer",
        options: &[
            opt(
                "--bitrate",
                "KBPS",
                "Real-time rate control at this target bitrate.",
            ),
            opt(
                "--peak-bitrate",
                "KBPS",
                "Peak drain rate (default twice the target).",
            ),
            opt(
                "--buffer-ms",
                "MS",
                "Virtual decoder buffer duration (default 1000 ms).",
            ),
            opt(
                "--buffer-initial-ms",
                "MS",
                "Decoder buffer fullness when the stream starts, at most the buffer: the \
                 first frames may send this much ahead of the drain (x264 --vbv-init, \
                 --buf-initial-sz). Default: the whole buffer.",
            ),
            opt(
                "--min-qp",
                "Q",
                "Lowest quantiser rate control may use, in the --qp scale and range.",
            ),
            opt(
                "--max-qp",
                "Q",
                "Highest quantiser rate control may use, in the --qp scale and range.",
            ),
            opt(
                "--max-qp-step",
                "N",
                "Largest quantiser change between inter frames (default 4 QP, 20 AV1 indices).",
            ),
            opt(
                "--frame-drop",
                "",
                "Skip an inter frame that would overflow the buffer even at the highest \
                 quantiser.",
            ),
            opt(
                "--strict-buffer",
                "",
                "Never pass one buffer: re-code any frame that would fill it past its \
                 capacity (a hard VBV). By default a frame may reach 1.15 buffers, as the \
                 reference encoders' worst points do.",
            ),
            opt(
                "--rate-change",
                "FRAME:KBPS[:PEAK_KBPS[:BUFFER_MS]]",
                "Change the target (and peak, buffer) before frame FRAME, as a transport's \
                 bandwidth estimate would; repeatable.",
            ),
        ],
    },
    GroupDoc {
        title: "Key frames and resilience",
        options: &[
            opt(
                "--intra-period",
                "N",
                "A key frame every N frames (default: every 10 seconds of video).",
            ),
            opt(
                "--keyframe",
                "FRAME",
                "Code input frame FRAME as a key frame, as a receiver's request would; \
                 repeatable.",
            ),
            opt(
                "--intra-refresh",
                "FRAMES",
                "Intra refresh sweep over this many frames in place of periodic key frames \
                 (`refresh.period`). With --temporal-layers 2|3 or --scalability LxT2/T3 the \
                 layers are fixed and the sweep runs on the base layer. Not with spatial layers, \
                 VP9 or AV2.",
            ),
            opt(
                "--ltr-interval",
                "FRAMES",
                "Mark a long-term reference every N frames, for loss recovery (`ltr.interval`).",
            ),
            opt(
                "--temporal-layers",
                "1|2|3",
                "Hierarchical-P temporal layers; the top layer is droppable. Not with \
                 --scalability.",
            ),
            opt(
                "--scalability",
                "LxTy[_KEY]",
                "Scalability mode L1T1..L3T3: spatial layers (L2, L3: 2:1 steps) and temporal \
                 layers in one stream; `_KEY` codes K-SVC.",
            ),
        ],
    },
    GroupDoc {
        title: "Content",
        options: &[
            opt(
                "--still-camera",
                "",
                "Noise handling for a fixed camera over the preset: lowers the bitrate on still \
                 noisy scenes; off in every preset.",
            ),
            opt(
                "--film-grain",
                "[STRENGTH]",
                "AV1 only: film grain synthesis, STRENGTH relative to the measured source noise.",
            ),
        ],
    },
    GroupDoc {
        title: "Signalling",
        options: &[
            opt(
                "--colour",
                "bt709|bt601|bt601-625|P,T,M",
                "Colour description to signal (ISO/IEC 23091-2 code points for P,T,M).",
            ),
            opt("--range", "full|limited", "Sample range to signal."),
            opt(
                "--timing",
                "nominal|constant|variable",
                "Header timing information (default nominal; variable with --timestamps); constant \
                 also codes every frame (no frame holds). VP9 and AV2 streams carry no timing, \
                 so they refuse variable.",
            ),
        ],
    },
    GroupDoc {
        title: "Parameters",
        options: &[
            opt(
                "--params",
                "FILE",
                "A parameter file (one `name = value` per line, `#` comments), over the preset.",
            ),
            opt(
                "--param",
                "NAME=VALUE",
                "One parameter, over the file; repeatable. `nanocodec params` lists them.",
            ),
        ],
    },
    GroupDoc {
        title: "Diagnostics",
        options: &[
            opt(
                "--rate-log",
                "FILE",
                "With --bitrate: one CSV row per frame of rate control's state.",
            ),
            opt(
                "--sse-log",
                "FILE",
                "Per-plane squared error of every shown picture against --sse-reference (CSV).",
            ),
            opt(
                "--sse-reference",
                "PATH",
                "The reference frames for --sse-log.",
            ),
            opt(
                "--layer-md5",
                "FILE",
                "CSV of each layer picture's size, bytes and MD5.",
            ),
            opt(
                "--filter-stats",
                "FILE",
                "JSON of the frame-level filter searches' units by outcome (off, unchanged), by tool and frame role; \"given\": [units, off, unchanged] of the searched units by their own last outcomes (off at the last 1, 2, 4 searches, on, unchanged) and content (quiet: mostly coded without residual).",
            ),
            opt(
                "--label-dump",
                "DIR",
                "Decision labels for learned models: per superblock its inputs and coded cost, per searched node every candidate's coded cost and time, as .npy columns (docs/analysis/nn-dataset-schema.md). The stream is unchanged; encoding is slower.",
            ),
            opt(
                "--label-tool",
                "NAME:PARAM=VALUE[,...]",
                "With --label-dump: also code every superblock with these parameters on top of the preset's, the rest of the frame as the preset codes it, recording its J and time (sb/tool_cost, sb/tool_ns); repeatable. Engine decision./planner. parameters, or the codec's per-superblock ones.",
            ),
            opt(
                "--spatial-stats",
                "FILE",
                "JSON summary of per-layer bytes, rates and references.",
            ),
            opt(
                "--qp-replay",
                "FILE",
                "With --bitrate: impose these quantisers on rate control's plan, by frame.",
            ),
            opt(
                "--key-offsets",
                "FILE",
                "QP offsets per 8x8 luma cell imposed on key frames, by input frame (an oracle: \
                 `frame N`, then one line of offsets per cell row).",
            ),
            opt(
                "--partition-record",
                "FILE",
                "Every inter superblock's coded partition (an oracle: the transplant log).",
            ),
            opt(
                "--partition-replay",
                "FILE",
                "Code a --partition-record log's partitions in place of the plan, without the RD \
                 partition search (an oracle: the partition transplant).",
            ),
            opt(
                "--analysis-log",
                "FILE",
                "Every frame's analysis per 8x8 luma cell (binary: variance, motion SAD and vector).",
            ),
            opt(
                "--simulate-loss",
                "FRAME[:LAYER]",
                "Report frame FRAME (or spatial layer LAYER of it) lost; every other frame is \
                 acknowledged.",
            ),
        ],
    },
];

/// Every command, in the order `nanocodec help` prints them.
pub const COMMANDS: &[CommandDoc] = &[
    CommandDoc {
        name: "encode",
        about: "Encode raw video to a stream.",
        groups: ENCODE_GROUPS,
    },
    CommandDoc {
        name: "decode",
        about: "Decode a stream, ours or another encoder's, within the supported profiles (gaps: docs/FAQ.md#conformance).",
        groups: &[GroupDoc {
            title: "Options",
            options: &[
                opt("--codec", CODECS, "Stream format. Required."),
                opt(
                    "--input",
                    "PATH",
                    "The stream (IVF or Annex-B); `-` reads stdin.",
                ),
                opt(
                    "--output",
                    "PATH",
                    "Decoded frames as raw I420 (16-bit little-endian above 8 bits); `-` writes stdout.",
                ),
                opt("--frame-md5", "FILE", "One MD5 line per shown frame."),
                opt(
                    "--discard",
                    "0|1",
                    "1: decode every frame and write nothing (for timing).",
                ),
                opt(
                    "--bench",
                    "N",
                    "Time N in-process decodes of the stream in memory (JSON lines).",
                ),
                opt(
                    "--film-grain",
                    "0|1",
                    "AV1: 0 shows frames without film grain.",
                ),
                opt(
                    "--operating-point",
                    "N",
                    "AV1 or VP9 operating point to decode.",
                ),
                opt(
                    "--all-layers",
                    "0|1",
                    "AV1 or VP9: 1 writes every spatial layer's picture.",
                ),
            ],
        }],
    },
    CommandDoc {
        name: "extract",
        about: "Keep a temporal or spatial sub-stream of a stream, without re-encoding.",
        groups: &[GroupDoc {
            title: "Options",
            options: &[
                opt("--codec", CODECS, "Stream format. Required."),
                opt("--input", "PATH", "The stream."),
                opt("--output", "PATH", "The kept sub-stream."),
                opt(
                    "--temporal-layers",
                    "N",
                    "Temporal layers the stream was coded with.",
                ),
                opt(
                    "--max-layer",
                    "M",
                    "Keep temporal layers 0..M (below --temporal-layers).",
                ),
                opt(
                    "--list",
                    "",
                    "List each access unit's temporal layer instead of extracting.",
                ),
                opt(
                    "--base-only",
                    "",
                    "AV1 or VP9: keep the spatial base layer.",
                ),
                opt(
                    "--operating-point",
                    "N",
                    "AV1 or VP9: keep this operating point.",
                ),
                opt(
                    "--layer-md5",
                    "FILE",
                    "VP9: the encoder's layer sidecar to split by.",
                ),
            ],
        }],
    },
    CommandDoc {
        name: "packetize",
        about: "A stream file (IVF or Annex B) to RTP packets (no Dependency Descriptor: use encode --rtp).",
        groups: &[GroupDoc {
            title: "Options",
            options: &[
                opt("--codec", CODECS, "Stream format (not av2). Required."),
                opt("--input", "PATH", "The stream."),
                opt(
                    "--output",
                    "PATH",
                    "pcap for .pcap, otherwise RFC 4571 framing.",
                ),
                opt("--mtu", "BYTES", "Largest RTP packet (default 1200)."),
                opt("--payload-type", "PT", "RTP payload type (default 96)."),
                opt(
                    "--fps",
                    "N",
                    "Annex-B input: frames per second for timestamps (default 30).",
                ),
                opt(
                    "--vp9-mode",
                    "flexible|non-flexible",
                    "VP9 payload descriptor mode (default flexible).",
                ),
            ],
        }],
    },
    CommandDoc {
        name: "depacketize",
        about: "RTP packets (pcap or RFC 4571) to a stream file, through the receiver's loss handling.",
        groups: &[GroupDoc {
            title: "Options",
            options: &[
                opt("--codec", CODECS, "Stream format (not av2). Required."),
                opt("--input", "PATH", "The capture."),
                opt("--output", "PATH", "IVF (AV1, VP9) or Annex B."),
                opt("--payload-type", "PT", "Keep only this payload type."),
                opt(
                    "--dd-id",
                    "ID|none",
                    "Dependency Descriptor extension id (AV1 default 1, others none).",
                ),
                opt(
                    "--layers",
                    "S:T",
                    "Forward only spatial layers 0..S and temporal layers 0..T first, by the \
                     Dependency Descriptor, as a forwarding server would.",
                ),
                opt(
                    "--deliver-after-loss",
                    "",
                    "Keep delivering after a loss instead of waiting for a key frame.",
                ),
            ],
        }],
    },
    CommandDoc {
        name: "session",
        about: "Encoder and receiver stepped by commands on stdin (encode options, no --output).",
        groups: &[GroupDoc {
            title: "Options",
            options: &[
                opt(
                    "--display",
                    "PATH",
                    "One picture per input frame: what the viewer saw.",
                ),
                opt(
                    "--receiver",
                    "transport|bitstream",
                    "Decode only frames whose listed references arrived (transport, the default), \
                     or every arrived frame.",
                ),
                opt(
                    "--operating-point",
                    "N",
                    "AV1 operating point the receiver decodes.",
                ),
            ],
        }],
    },
    CommandDoc {
        name: "params",
        about: "List every tunable parameter with its default and range, and the presets.",
        groups: &[GroupDoc {
            title: "Options",
            options: &[opt(
                "--codec",
                CODECS,
                "Show this codec's defaults and presets.",
            )],
        }],
    },
    CommandDoc {
        name: "stat",
        about: "Coding-tool statistics of a stream (analysis builds: --features analysis).",
        groups: &[GroupDoc {
            title: "Options",
            options: &[
                opt("--codec", CODECS, "Stream format."),
                opt("--input", "PATH", "The stream."),
                opt("--json", "FILE", "Write the statistics as JSON."),
                opt(
                    "--records",
                    "FILE",
                    "HEVC, H.264: also write every picture's block records (packed, little-endian).",
                ),
            ],
        }],
    },
    CommandDoc {
        name: "help",
        about: "Print this help.",
        groups: &[],
    },
];

/// `nanocodec help`: every command with its grouped options.
pub fn text() -> String {
    let mut out = String::from(
        "usage: nanocodec COMMAND [OPTIONS]   (nanocodec --version: version and commit;\n       --help: this help)\n",
    );
    for command in COMMANDS {
        let _ = write!(out, "\nnanocodec {}: {}\n", command.name, command.about);
        for group in command.groups {
            let _ = writeln!(out, "  {}", group.title);
            for option in group.options {
                let usage = format!("{} {}", option.flag, option.value);
                let _ = writeln!(out, "    {:<40} {}", usage.trim_end(), option.help);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::path::Path;

    /// The parsers' sources, whose flag literals must all be documented.
    const PARSER_SOURCES: &[&str] = &[
        include_str!("main.rs"),
        include_str!("rate_control.rs"),
        include_str!("params.rs"),
        include_str!("colour.rs"),
        include_str!("decode.rs"),
        include_str!("extract.rs"),
        include_str!("label_dump.rs"),
        include_str!("session.rs"),
        include_str!("rtp.rs"),
        include_str!("stat.rs"),
        include_str!("stdio.rs"),
    ];
    /// The flags `main` reads before any command (the usage line names them).
    const TOP_LEVEL_FLAGS: &[&str] = &["--help", "--version"];
    /// Flag literals of the parsers' own unit tests, not options.
    const TEST_ONLY_FLAGS: &[&str] = &["--unknown"];
    /// The checked-in site data this list generates.
    const SITE_JSON: &str = "../../web/docs/data/cli.json";
    /// The codecs whose presets the site lists, in its order.
    const SITE_CODECS: &[&str] = &["h264", "hevc", "vvc", "av1", "vp9", "av2"];

    /// Every `"--name"` literal in `source`, outside its unit tests.
    fn flag_literals(source: &str) -> BTreeSet<String> {
        let code = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .unwrap_or(source);
        code.split("\"--")
            .skip(1)
            .filter_map(|rest| {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
                    .collect();
                (!name.is_empty() && rest[name.len()..].starts_with('"'))
                    .then(|| format!("--{name}"))
            })
            .collect()
    }

    fn documented() -> BTreeSet<&'static str> {
        COMMANDS
            .iter()
            .flat_map(|c| c.groups)
            .flat_map(|g| g.options)
            .map(|o| o.flag)
            .chain(TOP_LEVEL_FLAGS.iter().copied())
            .collect()
    }

    #[test]
    fn every_parsed_flag_is_documented_and_every_documented_flag_parsed() {
        let parsed: BTreeSet<String> = PARSER_SOURCES
            .iter()
            .flat_map(|s| flag_literals(s))
            .filter(|f| !TEST_ONLY_FLAGS.contains(&f.as_str()))
            .collect();
        let documented = documented();
        let missing: Vec<_> = parsed
            .iter()
            .filter(|f| !documented.contains(f.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "flags read but not in help.rs: {missing:?}"
        );
        let unread: Vec<_> = documented
            .iter()
            .filter(|f| !parsed.contains(**f))
            .collect();
        assert!(
            unread.is_empty(),
            "flags in help.rs no parser reads: {unread:?}"
        );
    }

    fn json_string(text: &str) -> String {
        let mut out = String::from("\"");
        for c in text.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    /// The presets `nanocodec params --codec C` reports.
    fn presets(codec: &str) -> Vec<String> {
        let listing = crate::params::listing(Some(codec)).unwrap();
        let line = listing
            .lines()
            .find_map(|l| l.strip_prefix("# presets: "))
            .expect("the listing names its presets");
        let names = line.split(" (").next().unwrap_or(line);
        names.split(", ").map(str::to_owned).collect()
    }

    /// The site's command-line reference: every command's groups and options,
    /// and each codec's presets.
    fn site_json() -> String {
        let mut out = String::from("{\n \"commands\": [\n");
        for (ci, command) in COMMANDS.iter().enumerate() {
            let _ = write!(
                out,
                "  {{\"name\": {}, \"about\": {}, \"groups\": [",
                json_string(command.name),
                json_string(command.about)
            );
            for (gi, group) in command.groups.iter().enumerate() {
                let _ = write!(
                    out,
                    "\n   {{\"title\": {}, \"options\": [",
                    json_string(group.title)
                );
                for (oi, option) in group.options.iter().enumerate() {
                    let _ = write!(
                        out,
                        "\n    {{\"flag\": {}, \"value\": {}, \"help\": {}}}{}",
                        json_string(option.flag),
                        json_string(option.value),
                        json_string(option.help),
                        if oi + 1 < group.options.len() {
                            ","
                        } else {
                            ""
                        }
                    );
                }
                let _ = write!(
                    out,
                    "]}}{}",
                    if gi + 1 < command.groups.len() {
                        ","
                    } else {
                        ""
                    }
                );
            }
            let _ = writeln!(out, "]}}{}", if ci + 1 < COMMANDS.len() { "," } else { "" });
        }
        out.push_str(" ],\n \"presets\": {");
        for (i, codec) in SITE_CODECS.iter().enumerate() {
            let names: Vec<String> = presets(codec).iter().map(|n| json_string(n)).collect();
            let _ = write!(
                out,
                "\n  {}: [{}]{}",
                json_string(codec),
                names.join(", "),
                if i + 1 < SITE_CODECS.len() { "," } else { "" }
            );
        }
        out.push_str("\n }\n}\n");
        out
    }

    /// `web/docs/data/cli.json` is this list; `NC_BLESS=1` rewrites it.
    #[test]
    fn cli_reference_is_current() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(SITE_JSON);
        let expected = site_json();
        if std::env::var_os("NC_BLESS").is_some() {
            std::fs::write(&path, &expected).unwrap();
        }
        let actual = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            actual == expected,
            "{} is out of date: NC_BLESS=1 cargo test -p nanocodec-cli --bin nanocodec help::",
            path.display()
        );
    }
}
