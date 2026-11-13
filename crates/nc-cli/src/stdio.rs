//! Opening path arguments, where `-` means standard input for an input and
//! standard output for an output (the Unix convention): every nc-cli path
//! argument goes through here, so none of them creates a file literally
//! named `-` (as agents' `--output -` once did, inside the shared checkout).
//!
//! A run may stream at most one input from stdin and one output to stdout
//! ([`check_streams`]): two of either would interleave.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Buffer size of inputs and outputs. Raw video moves megabytes per frame:
/// with the default 8 KiB, writing decoded 1080p frames took ~400 `write`
/// calls per frame, as much CPU time as the decoding itself on a loaded
/// machine.
const IO_BUFFER_BYTES: usize = 1 << 20;

/// The path argument that stands for stdin or stdout.
pub(crate) const STDIO_PATH: &str = "-";

/// An opened input: a file or stdin.
pub(crate) type Input = BufReader<Box<dyn Read>>;
/// An opened output: a file or stdout.
pub(crate) type Output = BufWriter<Box<dyn Write>>;

/// Whether `path` stands for stdin or stdout.
pub(crate) fn is_stdio(path: &Path) -> bool {
    path.as_os_str() == STDIO_PATH
}

/// Buffer size of raw picture inputs, which the encoder reads row by row:
/// small enough to stay in the core's cache. With the 1 MiB buffer of the
/// other inputs (twice the Ryzen 5950X's L2), every row passed through
/// memory: encode CPU time on the PC's timing core, 720p `crowd_mot17_09`,
/// 150 f, 64 KiB vs 1 MiB: H.264 Faster -22 %, HEVC Faster -12 %, AV1 Fast
/// -14 % (user and system time; the Jetson was not affected).
const RAW_INPUT_BUFFER_BYTES: usize = 64 << 10;

/// Open the input `path` (stdin for `-`).
pub(crate) fn open(path: &Path) -> Result<Input, String> {
    open_buffered(path, IO_BUFFER_BYTES)
}

/// Open the raw picture input `path` (stdin for `-`), read row by row.
pub(crate) fn open_raw(path: &Path) -> Result<Input, String> {
    open_buffered(path, RAW_INPUT_BUFFER_BYTES)
}

fn open_buffered(path: &Path, capacity: usize) -> Result<Input, String> {
    let reader: Box<dyn Read> = if is_stdio(path) {
        Box::new(std::io::stdin().lock())
    } else {
        Box::new(File::open(path).map_err(|e| format!("{}: {e}", path.display()))?)
    };
    Ok(BufReader::with_capacity(capacity, reader))
}

/// Create the output `path` (stdout for `-`).
pub(crate) fn create(path: &Path) -> Result<Output, String> {
    let writer: Box<dyn Write> = if is_stdio(path) {
        Box::new(std::io::stdout().lock())
    } else {
        Box::new(File::create(path).map_err(|e| format!("{}: {e}", path.display()))?)
    };
    Ok(BufWriter::with_capacity(IO_BUFFER_BYTES, writer))
}

/// Read all of the input `path` (stdin for `-`).
pub(crate) fn read_all(path: &Path) -> Result<Vec<u8>, String> {
    let mut data = Vec::new();
    open(path)?
        .read_to_end(&mut data)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(data)
}

/// Read all of the text input `path` (stdin for `-`).
pub(crate) fn read_text(path: &Path) -> Result<String, String> {
    String::from_utf8(read_all(path)?).map_err(|e| format!("{}: {e}", path.display()))
}

/// Write `data` to the output `path` (stdout for `-`).
pub(crate) fn write_all(path: &Path, data: &[u8]) -> Result<(), String> {
    let mut out = create(path)?;
    out.write_all(data)
        .and_then(|()| out.flush())
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Refuse a command line `args` that would read more than one of the
/// `inputs` flags' paths from stdin, or write more than one of the `outputs`
/// flags' paths to stdout.
pub(crate) fn check_streams(
    args: &[String],
    inputs: &[&str],
    outputs: &[&str],
) -> Result<(), String> {
    let at_most_one = |flags: &[&str], stream: &str| {
        let dashed: Vec<&str> = flags
            .iter()
            .copied()
            .filter(|flag| {
                args.windows(2)
                    .any(|pair| pair[0] == *flag && is_stdio(Path::new(&pair[1])))
            })
            .collect();
        if dashed.len() > 1 {
            return Err(format!(
                "only one of {} can be {stream} ('-')",
                dashed.join(", ")
            ));
        }
        Ok(())
    };
    at_most_one(inputs, "stdin")?;
    at_most_one(outputs, "stdout")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_lone_dash_is_stdio() {
        assert!(is_stdio(Path::new("-")));
        assert!(!is_stdio(Path::new("./-")));
        assert!(!is_stdio(Path::new("--")));
        assert!(!is_stdio(Path::new("out.ivf")));
    }

    #[test]
    fn at_most_one_stream_each_way() {
        let args = |text: &str| -> Vec<String> { text.split(' ').map(String::from).collect() };
        let inputs = ["--input", "--timestamps"];
        let outputs = ["--output", "--recon"];
        let check = |text: &str| check_streams(&args(text), &inputs, &outputs);
        assert!(check("--input - --output -").is_ok());
        assert!(check("--output - --recon a.yuv").is_ok());
        let error = check("--output - --recon -").unwrap_err();
        assert!(error.contains("--output, --recon"), "{error}");
        assert!(check("--input - --timestamps -").is_err());
    }
}
