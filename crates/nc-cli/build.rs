//! Embeds the commit the CLI is built from, so `nanocodec --version` names
//! it and the evaluation harness can refuse a stale binary (nceval.buildinfo).
//!
//! `NC_GIT_COMMIT` is `git rev-parse HEAD` of the source tree, or the
//! `NC_BUILD_COMMIT` environment variable for a tree without git (a `git
//! archive` export), else "unknown". `NC_GIT_DIRTY` is "-dirty" when tracked
//! files under crates/ or presets/ differ from that commit.

use std::path::Path;
use std::process::Command;

/// `git` in the workspace root (a build script runs in its package directory).
fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-env-changed=NC_BUILD_COMMIT");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for watched in ["crates", "presets"] {
        println!("cargo:rerun-if-changed={}", root.join(watched).display());
    }
    let git_path = |name: &str| {
        git(
            &root,
            &["rev-parse", "--path-format=absolute", "--git-path", name],
        )
    };
    if let Some(head) = git_path("HEAD") {
        println!("cargo:rerun-if-changed={head}");
    }
    if let Some(reference) = git(&root, &["symbolic-ref", "-q", "HEAD"]).and_then(|r| git_path(&r))
    {
        println!("cargo:rerun-if-changed={reference}");
    }
    let commit = git(&root, &["rev-parse", "HEAD"])
        .or_else(|| std::env::var("NC_BUILD_COMMIT").ok())
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = git(
        &root,
        &[
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "crates",
            "presets",
        ],
    )
    .is_some_and(|status| !status.is_empty());
    println!("cargo:rustc-env=NC_GIT_COMMIT={commit}");
    println!(
        "cargo:rustc-env=NC_GIT_DIRTY={}",
        if dirty { "-dirty" } else { "" }
    );
}
