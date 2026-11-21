#!/usr/bin/env bash
# The static-recovery check (crates/nc-cli/tests/static_recovery.rs: every shipped
# preset of every codec recovers a still region after a starved key), on the build
# box by default (owner, 2026-11-21: the PC carries no material compute):
#
#   tools/dev/static_recovery.sh            # nc-remote run in this worktree
#   NC_LOCAL=1 tools/dev/static_recovery.sh # here
#
# Run it for any change to frame structure, references or rate control (CLAUDE.md
# "Testing"), once for what lands.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
test=(cargo test --release -p nanocodec-cli --test integration static_recovery)
if [[ "${NC_LOCAL:-0}" == 1 || -n "${NC_REMOTE_BOX:-}" ]]; then
    exec "${test[@]}"
fi
exec "$HOME/nanocodec-data/bin/nc-remote" run --name "static-recovery-$(basename "$PWD")" -- "${test[@]}"
