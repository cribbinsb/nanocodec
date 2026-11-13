//! `--sse-log`: every row is the exact per-plane squared error of the
//! picture `--display-recon` gets against `--sse-reference`, the sums the
//! evaluation harness turns into PSNR (`eval/nceval/metrics.py`).
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const WIDTH: usize = 96;
const HEIGHT: usize = 64;
const FRAMES: usize = 6;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nc-cli-sse-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
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

/// Per-plane sums of squared differences of two packed I420 frames.
fn plane_sse(a: &[u8], b: &[u8]) -> [u64; 3] {
    let luma = WIDTH * HEIGHT;
    let chroma = luma / 4;
    let sse = |range: std::ops::Range<usize>| {
        a[range.clone()]
            .iter()
            .zip(&b[range])
            .map(|(&x, &y)| u64::from(x.abs_diff(y)).pow(2))
            .sum()
    };
    [
        sse(0..luma),
        sse(luma..luma + chroma),
        sse(luma + chroma..luma + 2 * chroma),
    ]
}

#[test]
fn sse_log_matches_the_shown_pictures() {
    let scratch = Scratch::new();
    let frame_len = WIDTH * HEIGHT * 3 / 2;
    let input = scratch.0.join("source.yuv");
    let source: Vec<u8> = (0..FRAMES * frame_len)
        .map(|i| ((i * 29 + (i / WIDTH) * 11 + i / frame_len * 23) % 251) as u8)
        .collect();
    fs::write(&input, &source).unwrap();
    let (recon, log) = (scratch.0.join("shown.yuv"), scratch.0.join("sse.csv"));
    let arg = |path: &PathBuf| path.to_str().unwrap().to_owned();
    let status = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
        .args(["encode", "--codec", "h264", "--qp", "36", "--fps", "30"])
        .args([
            "--width",
            &WIDTH.to_string(),
            "--height",
            &HEIGHT.to_string(),
        ])
        .args(["--frames", &FRAMES.to_string(), "--input", &arg(&input)])
        .args(["--output", &arg(&scratch.0.join("out.264"))])
        .args(["--display-recon", &arg(&recon)])
        .args(["--sse-log", &arg(&log), "--sse-reference", &arg(&input)])
        .status()
        .unwrap();
    assert!(status.success());
    let shown = fs::read(&recon).unwrap();
    let rows = fs::read_to_string(&log).unwrap();
    let mut lines = rows.lines();
    assert_eq!(lines.next(), Some("frame,sse_y,sse_u,sse_v"));
    let logged: Vec<Vec<u64>> = lines
        .map(|line| line.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(logged.len(), FRAMES);
    for (index, row) in logged.iter().enumerate() {
        let at = index * frame_len..(index + 1) * frame_len;
        let expected = plane_sse(&source[at.clone()], &shown[at]);
        assert_eq!(row[0], index as u64);
        assert_eq!(row[1..], expected, "frame {index}");
        assert!(
            expected[0] > 0,
            "frame {index} is lossless: the test proves nothing"
        );
    }
}
