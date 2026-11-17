#!/bin/bash
# nc-remote build box: one-time setup, run as the ubuntu user over SSH by
# tools/dev/nc_remote.py when the box has no current marker (a fresh Ubuntu
# image; the baked AMI already has it). Installs what builds, tests, clippy,
# callgrind, fuzzing and the conformance runs need. Conformance streams are
# public and fetched on the box by a job (docs/PROCEDURES.md "Remote build box").
#
# The Rust toolchains are the PC's exactly (nc_remote.py reads them there):
# stable @RUST_STABLE@ as the default, and the nightly of commit
# @NIGHTLY_HASH@ (dist date @NIGHTLY_DATE@) under the name `nightly`, so a
# clippy or rustc verdict on the box is the one landing and verify give.
set -euo pipefail

MARKER=/home/ubuntu/ncbox/provisioned-@PROVISION_VERSION@

export DEBIAN_FRONTEND=noninteractive
sudo apt-get -qq update
sudo apt-get -qq install -y --no-install-recommends \
  build-essential pkg-config clang lld cmake ninja-build nasm yasm meson \
  autoconf automake libtool xxd \
  valgrind python3 python3-venv python3-pip python3-xxhash python3-numpy \
  rsync git zstd curl unzip ffmpeg nodejs npm > /dev/null

if [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
fi
source "$HOME/.cargo/env"
rustup toolchain install @RUST_STABLE@ --profile minimal -c clippy -c rustfmt -c llvm-tools -t wasm32-unknown-unknown
rustup default @RUST_STABLE@
nightly=nightly-@NIGHTLY_DATE@
rustup toolchain install "$nightly" --profile minimal -c miri -c rust-src -c llvm-tools-preview
got=$(rustup run "$nightly" rustc -vV | sed -n 's/^commit-hash: //p')
if [ "$got" != @NIGHTLY_HASH@ ]; then
  echo "FAILED: $nightly is commit $got, the PC's nightly is @NIGHTLY_HASH@" >&2
  exit 1
fi
# `cargo +nightly` must mean the pinned one. rustup refuses `nightly` as a
# custom (linked) name, so the channel's directory is made a link to the
# dated toolchain: `nightly` is then installed and resolves to it (as on the
# PC, where the channel was installed then and never updated). A floating
# `stable` from an older image stays installed but is not the default:
# removing it would break builds running on the box.
toolchains=$HOME/.rustup/toolchains
host=x86_64-unknown-linux-gnu
if [ "$(readlink "$toolchains/nightly-$host")" != "$nightly-$host" ]; then
  rustup toolchain uninstall nightly > /dev/null 2>&1 || true
  ln -sfn "$nightly-$host" "$toolchains/nightly-$host"
fi
got=$(rustc +nightly -vV | sed -n 's/^commit-hash: //p')
[ "$got" = @NIGHTLY_HASH@ ] || { echo "FAILED: +nightly is $got" >&2; exit 1; }

# The evaluation harness's Python packages (nceval: the standards lens, eval
# tests), in a venv that nc_remote_job.py puts first on every job's PATH.
python3 -m venv "$HOME/ncbox/venv"
"$HOME/ncbox/venv/bin/pip" install -q --upgrade pip
"$HOME/ncbox/venv/bin/pip" install -q numpy scipy opencv-python-headless xxhash av \
  matplotlib scikit-learn optuna pytest pytest-xdist

touch "$MARKER"
echo "provisioned: $(rustc --version), $(cargo clippy --version), $(rustc +nightly --version), $(valgrind --version), $(python3 --version)"
