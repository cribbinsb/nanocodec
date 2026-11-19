#!/usr/bin/env bash
# Print the directory holding a landed commit's ready CLI binary (`nanocodec`,
# the gate build; often `nc-checkasm` too), for NC_NANOCODEC_BIN:
#
#   tools/dev/binary_for.sh COMMIT
#   NC_NANOCODEC_BIN=$(tools/dev/binary_for.sh 1da6bfbfb) python3 -m nceval.screen ...
#
# From the landed-commit cache (~/nanocodec-data/charts/builds/<commit>/, filled by
# land.sh --verify), else a cached build with the same sources, else built on the
# remote build box. Never a local compile (tools/dev/build_cache.py).
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/build_cache.py" for "${1:?usage: binary_for.sh COMMIT}"
