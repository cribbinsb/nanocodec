//! AV2 streams of other encoders (avmenc v1.0.0 and the dav2d test
//! vectors) that the decoder decodes bit-exactly: each must still give
//! avmdec's frame hashes, listed in `data/av2-foreign.md5`
//! (`tools/dev/av2_foreign/compare.py --write-hashes`).
//!
//! The streams are not in the repository (`tools/dev/av2_foreign/` fetches
//! and encodes them into `$NC_AV2_FOREIGN`, by default
//! `~/nanocodec-data/work/av2-foreign`). Without the default directory the
//! test says so and checks nothing; a directory named by `NC_AV2_FOREIGN`
//! must exist (the nightly verifier sets it, so a missing corpus fails there),
//! and with a corpus a missing stream fails.
//!
//! Three small streams are in the repository (`data/av2-odd-size`, avmenc
//! v1.0.0 at `--cpu-used=9`, with and without `--lag-in-frames=0`, of a
//! synthetic pattern at 202x98 and 201x99): frames whose size is not a
//! multiple of 8, whose prediction reads the reference's coded area past
//! the frame, and odd sizes. They are always checked.
use std::path::{Path, PathBuf};
use std::process::Command;

const HASHES: &str = include_str!("data/av2-foreign.md5");

/// Where the corpus keeps its streams, relative to its root.
const STREAM_DIRS: [&str; 2] = ["streams", "vectors/dav2d-test-data/avm"];

fn corpus() -> Option<PathBuf> {
    if let Some(root) = std::env::var_os("NC_AV2_FOREIGN") {
        let root = PathBuf::from(root);
        assert!(
            root.is_dir(),
            "NC_AV2_FOREIGN={} is not a directory",
            root.display()
        );
        return Some(root);
    }
    let root = Path::new(&std::env::var_os("HOME")?).join("nanocodec-data/work/av2-foreign");
    root.is_dir().then_some(root)
}

/// Decode `stream` with our CLI and check its frame hashes against
/// `expected` (avmdec's).
fn check_stream(name: &str, stream: &Path, expected: &[&str]) {
    let output = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
        .args(["decode", "--codec", "av2", "--frame-md5", "-", "--input"])
        .arg(stream)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let hashes = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        hashes.split_whitespace().collect::<Vec<_>>(),
        expected,
        "{name}"
    );
}

#[test]
fn checked_in_odd_size_streams_decode_to_the_reference_hashes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/av2-odd-size");
    for line in include_str!("data/av2-odd-size/hashes.md5").lines() {
        let mut fields = line.split_whitespace();
        let name = fields.next().expect("a stream name");
        let expected: Vec<&str> = fields.collect();
        check_stream(name, &root.join(name), &expected);
    }
}

#[test]
fn foreign_streams_decode_to_the_reference_hashes() {
    let Some(root) = corpus() else {
        eprintln!("AV2 foreign-stream corpus not present: nothing checked");
        return;
    };
    for line in HASHES.lines() {
        let mut fields = line.split_whitespace();
        let name = fields.next().expect("a stream name");
        let expected: Vec<&str> = fields.collect();
        let stream = STREAM_DIRS
            .iter()
            .map(|dir| root.join(dir).join(name))
            .find(|path| path.is_file())
            .unwrap_or_else(|| panic!("{name} is missing from {}", root.display()));
        check_stream(name, &stream, &expected);
    }
}
