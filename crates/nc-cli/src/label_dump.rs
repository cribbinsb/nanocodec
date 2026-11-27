//! `--label-dump DIR`: the encoder's decision labels (`nc_encoder::labels`)
//! as one shard of the nn dataset, a directory of numpy `.npy` columns in
//! four tables (`frame/`, `sb/`, `node/`, `leaf/`), declared in
//! `label_columns.rs` and listed with their types and shapes in
//! `labels.json`. The layout is `docs/analysis/nn-dataset-schema.md`;
//! `eval/nceval/nndata` is the only reader, and it checks [`FORMAT_VERSION`].

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use nc_encoder::labels::{CANDIDATE_SLOTS, LabelConfig, LabelTool, PATCH_SIDE};

use crate::label_columns::{self, Place, Spec, TOOLS};
use nc_encoder::{Codec, CodecBackend, Encoder, ParamOverrides};

/// The shard layout's version: the reader refuses any other.
pub const FORMAT_VERSION: u32 = 7;
/// `.npy` format 1.0 headers are padded to a multiple of this.
const NPY_ALIGN: usize = 64;

/// The label dump's clock (`--label-dump`): nanoseconds since the first
/// call. The epoch is set once and only read after, so the clock stays a
/// plain function the engine can hold.
fn clock_ns() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let epoch = EPOCH.get_or_init(std::time::Instant::now);
    u64::try_from(epoch.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Every `--label-tool` of an encode's arguments.
pub fn tools_from_args(args: &[String]) -> Result<Vec<LabelTool>, String> {
    let specs: Vec<&str> = args
        .windows(2)
        .filter(|pair| pair[0] == "--label-tool")
        .map(|pair| pair[1].as_str())
        .collect();
    parse_tools(&specs)
}

/// `--label-tool NAME:PARAM=VALUE[,PARAM=VALUE...]`, each.
pub fn parse_tools(specs: &[&str]) -> Result<Vec<LabelTool>, String> {
    specs
        .iter()
        .map(|spec| {
            let (name, settings) = spec
                .split_once(':')
                .ok_or_else(|| format!("--label-tool {spec}: NAME:PARAM=VALUE[,...]"))?;
            let mut overrides = ParamOverrides::default();
            for setting in settings.split(',') {
                let (param, value) = setting
                    .split_once('=')
                    .ok_or_else(|| format!("--label-tool {spec}: {setting} is not PARAM=VALUE"))?;
                let value: f32 = value
                    .parse()
                    .map_err(|_| format!("--label-tool {spec}: {value} is not a number"))?;
                overrides.set(param, value);
            }
            Ok(LabelTool {
                name: name.to_string(),
                overrides,
            })
        })
        .collect()
}

/// Write `rows` as `dir/<name>.npy`, one file per column of `specs`; returns
/// the columns' `labels.json` entries.
fn write_table<R>(
    dir: &Path,
    specs: &[Spec<R>],
    rows: &[(&R, Place)],
    tools: usize,
) -> Result<Vec<String>, String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut entries = Vec::with_capacity(specs.len());
    for spec in specs {
        let shape: Vec<usize> = spec
            .shape
            .iter()
            .map(|&d| if d == TOOLS { tools } else { d })
            .collect();
        let mut data = Vec::new();
        for &(record, place) in rows {
            (spec.put)(record, place, &mut data);
        }
        let item: usize = spec.dtype[2..]
            .parse()
            .map_err(|_| format!("{}: dtype {}", spec.name, spec.dtype))?;
        let per_row = item * shape.iter().product::<usize>();
        if data.len() != per_row * rows.len() {
            return Err(format!(
                "{}: {} bytes for {} rows of {per_row}",
                spec.name,
                data.len(),
                rows.len()
            ));
        }
        write_npy(
            &dir.join(format!("{}.npy", spec.name)),
            spec.dtype,
            rows.len(),
            &shape,
            &data,
        )?;
        let declared: Vec<String> = spec
            .shape
            .iter()
            .map(|&d| {
                if d == TOOLS {
                    "-1".to_string()
                } else {
                    d.to_string()
                }
            })
            .collect();
        entries.push(format!(
            "{{\"name\": \"{}\", \"dtype\": \"{}\", \"shape\": [{}]}}",
            spec.name,
            spec.dtype,
            declared.join(", ")
        ));
    }
    Ok(entries)
}

/// One `.npy` file (format 1.0) of `rows` rows of `shape`.
fn write_npy(
    path: &Path,
    dtype: &str,
    rows: usize,
    shape: &[usize],
    data: &[u8],
) -> Result<(), String> {
    let dims: Vec<String> = std::iter::once(rows)
        .chain(shape.iter().copied())
        .map(|d| d.to_string())
        .collect();
    let shape = if dims.len() == 1 {
        format!("({},)", dims[0])
    } else {
        format!("({})", dims.join(", "))
    };
    let mut header = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}");
    // Magic (6) + version (2) + length (2) + header + newline, aligned.
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat(unpadded.next_multiple_of(NPY_ALIGN) - unpadded));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    let length = u16::try_from(header.len()).map_err(|_| "npy header too long")?;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

/// A shard being gathered, written by [`LabelDump::finish`].
pub struct LabelDump {
    dir: PathBuf,
    /// The labelled tools' names, after the preset's own slot.
    tools: Vec<String>,
}

impl LabelDump {
    /// With `--label-dump DIR`: turn the encoder's labels on (with `tools`,
    /// timed by `clock`) and gather a shard for `DIR`.
    pub fn start<B: CodecBackend>(
        dir: Option<&Path>,
        tools: &[LabelTool],
        encoder: &mut Encoder<B>,
    ) -> Result<Option<Self>, String> {
        let Some(dir) = dir else {
            return Ok(None);
        };
        encoder.enable_labels(LabelConfig {
            clock: clock_ns,
            tools: tools.to_vec(),
        })?;
        let names = tools.iter().map(|t| t.name.clone()).collect();
        Ok(Some(Self::new(dir.to_path_buf(), names)))
    }

    fn new(dir: PathBuf, tools: Vec<String>) -> Self {
        Self { dir, tools }
    }

    /// Write the shard: every table's columns, then `labels.json` naming the
    /// format, the tools and every column. The encoder holds the whole shard
    /// in memory until here.
    pub fn finish<B: CodecBackend>(
        self,
        encoder: &mut Encoder<B>,
        codec: Codec,
    ) -> Result<(), String> {
        let codec = format!("{codec:?}").to_lowercase();
        let frames = encoder.take_labels();
        let tools = self.tools.len() + 1;
        let (mut contexts, mut sb, mut nodes, mut leaves) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for frame in &frames {
            let place = Place {
                frame: u32::try_from(frame.context.index).unwrap_or(u32::MAX),
                sb_base: sb.len() as u32,
                node_base: nodes.len() as i32,
            };
            contexts.push((&frame.context, place));
            sb.extend(frame.superblocks.iter().map(|s| (s, place)));
            nodes.extend(frame.nodes.iter().map(|n| (n, place)));
            leaves.extend(frame.leaves.iter().map(|l| (l, place)));
        }
        let tables = [
            (
                "frame",
                write_table(
                    &self.dir.join("frame"),
                    label_columns::FRAME,
                    &contexts,
                    tools,
                )?,
            ),
            (
                "sb",
                write_table(&self.dir.join("sb"), label_columns::SB, &sb, tools)?,
            ),
            (
                "node",
                write_table(&self.dir.join("node"), label_columns::NODE, &nodes, tools)?,
            ),
            (
                "leaf",
                write_table(&self.dir.join("leaf"), label_columns::LEAF, &leaves, tools)?,
            ),
        ];
        let tables: Vec<String> = tables
            .iter()
            .map(|(name, columns)| format!("\"{name}\": [{}]", columns.join(", ")))
            .collect();
        let tool_names: Vec<String> = std::iter::once("preset")
            .chain(self.tools.iter().map(String::as_str))
            .map(|t| format!("\"{t}\""))
            .collect();
        let path = self.dir.join("labels.json");
        let mut out = fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        writeln!(
            out,
            "{{\"format_version\": {FORMAT_VERSION}, \"codec\": \"{codec}\", \"candidate_slots\": {CANDIDATE_SLOTS}, \"patch_side\": {PATCH_SIDE}, \"tools\": [{}], \"tables\": {{{}}}}}",
            tool_names.join(", "),
            tables.join(", ")
        )
        .map_err(|e| e.to_string())
    }
}
