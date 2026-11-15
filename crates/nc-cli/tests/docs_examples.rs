//! The usage pages' command lines run: every `<pre class="run">` block in
//! `web/docs/*.html` is executed in page order, in a scratch directory per
//! page, against a synthetic copy of the pages' test clip (`in.yuv`, 640x360
//! I420), after writing the page's `<pre class="file" data-name=...>`
//! blocks. `nanocodec` lines run the built binary and must succeed; `cmp A B`
//! lines require identical files. To stay cheap, every `--frames N` codes
//! `TEST_FRAMES` frames instead (the examples' frame indices, e.g.
//! `--keyframe 90`, then lie past the end, which the CLI accepts).
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const DOCS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/docs");
/// The pages' test clip (Getting started: `ffmpeg ... testsrc2=size=640x360`).
const WIDTH: usize = 640;
const HEIGHT: usize = 360;
const TEST_FRAMES: usize = 6;
const INPUT: &str = "in.yuv";

struct Scratch(PathBuf);
impl Scratch {
    fn new(page: &str) -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("nc-cli-docs-{page}-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A page block the test acts on.
enum Block {
    /// Shell lines to run.
    Run(String),
    /// A file to write first: name and contents.
    File(String, String),
}

fn unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&amp;", "&")
}

/// The run and file blocks of `html`, in page order.
fn blocks(html: &str) -> Vec<Block> {
    const RUN: &str = "<pre class=\"run\"><code>";
    const FILE: &str = "<pre class=\"file\" data-name=\"";
    const END: &str = "</code></pre>";
    let mut found = Vec::new();
    let mut rest = html;
    loop {
        let run = rest.find(RUN);
        let file = rest.find(FILE);
        let Some(start) = [run, file].into_iter().flatten().min() else {
            return found;
        };
        let tail = &rest[start..];
        let end = tail.find(END).expect("an unterminated <pre>");
        if Some(start) == run {
            found.push(Block::Run(unescape(&tail[RUN.len()..end])));
        } else {
            let after = &tail[FILE.len()..end];
            let (name, body) = after
                .split_once("\"><code>")
                .expect("data-name then <code>");
            found.push(Block::File(name.to_owned(), unescape(body)));
        }
        rest = &tail[end..];
    }
}

/// `lines` with backslash continuations joined.
fn commands(lines: &str) -> Vec<String> {
    lines
        .replace("\\\n", " ")
        .lines()
        .map(str::to_owned)
        .collect()
}

/// A packed I420 clip with motion, so inter coding has work to do.
fn write_input(dir: &Path) {
    let mut clip = Vec::with_capacity(WIDTH * HEIGHT * 3 / 2 * TEST_FRAMES);
    for frame in 0..TEST_FRAMES {
        for row in 0..HEIGHT {
            clip.extend((0..WIDTH).map(|col| (16 + (col + row + 3 * frame) % 220) as u8));
        }
        clip.extend((0..WIDTH * HEIGHT / 2).map(|i| (96 + (i + frame) % 64) as u8));
    }
    fs::write(dir.join(INPUT), clip).unwrap();
}

fn run(dir: &Path, page: &str, line: &str) {
    let words: Vec<&str> = line.split_whitespace().collect();
    match words.as_slice() {
        ["nanocodec", args @ ..] => {
            let mut args: Vec<String> = args.iter().map(|a| (*a).to_owned()).collect();
            if let Some(i) = args.iter().position(|a| a == "--frames") {
                args[i + 1] = TEST_FRAMES.to_string();
            }
            let out = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
                .args(&args)
                .current_dir(dir)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{page}: `{line}` failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        ["cmp", a, b] => assert!(
            fs::read(dir.join(a)).unwrap() == fs::read(dir.join(b)).unwrap(),
            "{page}: `{line}`: the files differ"
        ),
        _ => panic!("{page}: a run block line the test cannot run: `{line}`"),
    }
}

#[test]
fn every_documented_command_runs() {
    let mut ran = 0;
    for entry in fs::read_dir(DOCS).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "html") {
            continue;
        }
        let page = path.file_stem().unwrap().to_str().unwrap().to_owned();
        let scratch = Scratch::new(&page);
        write_input(&scratch.0);
        for block in blocks(&fs::read_to_string(&path).unwrap()) {
            match block {
                Block::File(name, body) => fs::write(scratch.0.join(name), body).unwrap(),
                Block::Run(lines) => {
                    for line in commands(&lines) {
                        run(&scratch.0, &page, &line);
                        ran += 1;
                    }
                }
            }
        }
    }
    assert!(ran > 0, "no runnable examples found in {DOCS}");
}
