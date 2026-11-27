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
#
# The gate profile (no LTO) codes byte-identical streams to release, and shares the
# build with the agent's own `--profile gate` builds on the box; release's fat LTO
# made this a 4-8 min check (median 249 s, worst 484 s over 40 box runs to 2026-11-25)
# for no difference in what it checks (docs/analysis/time-audit-2026-11-25.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
test=(cargo test --profile gate -p nanocodec-cli --test integration static_recovery)
if [[ "${NC_LOCAL:-0}" == 1 || -n "${NC_REMOTE_BOX:-}" ]]; then
    exec "${test[@]}"
fi
exec "$HOME/nanocodec-data/bin/nc-remote" run --name "static-recovery-$(basename "$PWD")" -- "${test[@]}"
