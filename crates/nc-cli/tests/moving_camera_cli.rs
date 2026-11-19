//! `rate.fixed_qp_moving_key`: a fixed-QP stream whose first two frames show
//! camera motion codes its key coarser (`nc_encoder` `moving_camera.rs`, read
//! from the start-up lookahead the CLI hands over); a still scene's key is
//! unchanged.
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const WIDTH: usize = 320;
const HEIGHT: usize = 192;
const FRAMES: usize = 2;
/// Luma samples the moving input's second frame is shifted by.
const PAN: usize = 8;
/// Side of the texture's constant cells: coarse enough to survive the
/// motion search's quarter-resolution level.
const CELL: usize = 4;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("nc-cli-moving-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// One I420 frame of a random cell texture, panned `shift` samples to the
/// left (flat chroma).
fn frame(shift: usize) -> Vec<u8> {
    let cell = |x: usize, y: usize| {
        let hash = (x / CELL).wrapping_mul(2_654_435_761) ^ (y / CELL).wrapping_mul(40_503);
        (64 + (hash >> 7) % 128) as u8
    };
    let mut bytes = Vec::with_capacity(WIDTH * HEIGHT * 3 / 2);
    for y in 0..HEIGHT {
        bytes.extend((0..WIDTH).map(|x| cell(x + shift, y)));
    }
    bytes.resize(WIDTH * HEIGHT * 3 / 2, 128);
    bytes
}

/// The size of the two-frame fixed-QP HEVC stream of `input` with
/// `rate.fixed_qp_moving_key` at `coarser`.
fn stream_bytes(scratch: &Path, input: &Path, coarser: u32) -> u64 {
    let stream = scratch.join(format!("moving{coarser}.bin"));
    let output = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
        .args(["encode", "--codec", "hevc"])
        .args(["--input", input.to_str().unwrap()])
        .args(["--output", stream.to_str().unwrap()])
        .args(["--width", &WIDTH.to_string()])
        .args(["--height", &HEIGHT.to_string()])
        .args(["--fps", "30", "--frames", &FRAMES.to_string(), "--qp", "32"])
        .args(["--param", &format!("rate.fixed_qp_moving_key={coarser}")])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "rate.fixed_qp_moving_key={coarser}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::metadata(&stream).unwrap().len()
}

#[test]
fn a_fixed_qp_key_is_coarser_only_under_camera_motion() {
    let scratch = Scratch::new();
    let write = |name: &str, second_shift: usize| {
        let path = scratch.0.join(name);
        fs::write(&path, [frame(0), frame(second_shift)].concat()).unwrap();
        path
    };
    let still = write("still.yuv", 0);
    let moving = write("moving.yuv", PAN);
    assert_eq!(
        stream_bytes(&scratch.0, &still, 6),
        stream_bytes(&scratch.0, &still, 0),
        "a still scene keeps its key"
    );
    let (coarser, level) = (
        stream_bytes(&scratch.0, &moving, 6),
        stream_bytes(&scratch.0, &moving, 0),
    );
    assert!(
        coarser < level,
        "a panning scene's key: {coarser} bytes against {level}"
    );
}
