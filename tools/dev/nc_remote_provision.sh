#!/bin/bash
# nc-remote build box: one-time setup, run as the ubuntu user over SSH by
# tools/dev/nc_remote.py when the box has no current marker (a fresh Ubuntu
# image; the baked AMI already has it). Installs what builds, tests, clippy,
# callgrind, fuzzing and the conformance runs need. Conformance streams are
# public and fetched on the box by a job (docs/PROCEDURES.md "Remote build box").
set -euo pipefail

MARKER=/home/ubuntu/ncbox/provisioned-@PROVISION_VERSION@

export DEBIAN_FRONTEND=noninteractive
sudo apt-get -qq update
sudo apt-get -qq install -y --no-install-recommends \
  build-essential pkg-config clang lld cmake ninja-build nasm meson \
  valgrind python3 python3-venv python3-pip python3-xxhash python3-numpy \
  rsync git zstd curl unzip ffmpeg nodejs npm > /dev/null

if [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
fi
source "$HOME/.cargo/env"
rustup toolchain install stable --profile minimal -c clippy -c rustfmt -t wasm32-unknown-unknown
rustup toolchain install nightly --profile minimal -c miri -c rust-src -c llvm-tools-preview
cargo install --locked cargo-fuzz > /dev/null 2>&1 || cargo install --locked cargo-fuzz

touch "$MARKER"
echo "provisioned: $(rustc --version), $(valgrind --version), $(python3 --version)"
