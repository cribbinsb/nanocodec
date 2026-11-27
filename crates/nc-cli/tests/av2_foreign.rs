//! AV2 streams of other encoders (avmenc v1.0.0 and the dav2d test
//! vectors) that the decoder decodes bit-exactly: each must still give
//! avmdec's frame hashes, listed in `data/av2-foreign.md5`
//! (`tools/dev/av2_foreign/compare.py --write-hashes`; it lists apart the
//! streams where the pinned dav2d disagrees with avmdec, the reference).
//!
//! The streams are not in the repository (`tools/dev/av2_foreign/` fetches
//! and encodes them into `$NC_AV2_FOREIGN`, by default
//! `~/nanocodec-data/work/av2-foreign`). Without the default directory the
//! test says so and checks nothing; a directory named by `NC_AV2_FOREIGN`
//! must exist (the nightly verifier sets it, so a missing corpus fails there),
//! and with a corpus a missing stream fails.
//!
//! Small streams are in the repository and always checked: three in
//! `data/av2-odd-size` (avmenc v1.0.0 at `--cpu-used=9`, with and without
//! `--lag-in-frames=0`, of a synthetic pattern at 202x98 and 201x99):
//! frames whose size is not a multiple of 8, whose prediction reads the
//! reference's coded area past the frame, and odd sizes; one in
//! `data/av2-sb256` (`tools/dev/av2_foreign/encode.sh bodycam_airport_360p
//! ra 210 5 --sb-size=256`): 256x256 superblocks whose large TIP and
//! compound blocks predict in 64x64 pieces.
use std::path::{Path, PathBuf};
use std::process::Command;

const HASHES: &str = include_str!("data/av2-foreign.md5");

/// Where the corpus keeps its streams, relative to its root.
const STREAM_DIRS: [&str; 3] = [
    "streams",
    "vectors/dav2d-test-data/avm",
    "vectors/dav2d-test-data/avm-10bit",
];

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

/// Check every stream of a checked-in directory against its `hashes.md5`.
fn check_checked_in(dir: &str, hashes: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(dir);
    for line in hashes.lines() {
        let mut fields = line.split_whitespace();
        let name = fields.next().expect("a stream name");
        let expected: Vec<&str> = fields.collect();
        check_stream(name, &root.join(name), &expected);
    }
}

#[test]
fn checked_in_odd_size_streams_decode_to_the_reference_hashes() {
    check_checked_in("av2-odd-size", include_str!("data/av2-odd-size/hashes.md5"));
}

#[test]
fn checked_in_256_superblock_stream_decodes_to_the_reference_hashes() {
    check_checked_in("av2-sb256", include_str!("data/av2-sb256/hashes.md5"));
}

#[test]
fn foreign_streams_decode_to_the_reference_hashes() {
    let Some(root) = corpus() else {
        eprintln!("AV2 foreign-stream corpus not present: nothing checked");
        return;
    };
    // `#` lines are notes (compare.py marks the avmdec-only streams).
    for line in HASHES.lines().filter(|line| !line.starts_with('#')) {
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
