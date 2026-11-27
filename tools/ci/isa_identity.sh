#!/usr/bin/env bash
# isa_identity.sh BINARY [preset_identity options]: every shipped preset of every codec
# codes the same stream on every DSP tier this CPU runs (docs/benchmark-procedure.md
# "Determinism"). x86_64: scalar = avx2, and avx2 = avx512 where the CPU has the AVX-512
# tier; aarch64: scalar = neon, and neon = sve2 on Linux SVE2. One binary, NC_DSP_ISA set per side
# (tools/dev/preset_identity.py --old-env/--new-env); exit status 1 on any difference.
# Kernel-level bit-exactness is nc-checkasm's; this checks whole encodes, where a tier
# difference anywhere (dispatch, a kernel checkasm does not cover) would change a stream.
# On x86 it runs on the build box like preset_identity (whose CPU has the AVX-512 tier);
# on aarch64 run it on the board or a Graviton box with --local and --clip-dir.
set -euo pipefail
if [ $# -lt 1 ]; then
  sed -n "2,11p" "$0"; exit 2
fi
binary=$1; shift
here=$(dirname "$0")/../dev
case "$(file -b "$binary")" in
  *aarch64*|*ARM*)
    pairs="scalar:neon"
    if [[ "$(uname -s)" == Linux ]] && grep -qw sve2 /proc/cpuinfo; then
      pairs+=" neon:sve2"
    fi
    ;;
  *) pairs="scalar:avx2 avx2:avx512" ;;
esac
status=0
for pair in $pairs; do
  a=${pair%%:*}; b=${pair##*:}
  echo "== NC_DSP_ISA=$a against NC_DSP_ISA=$b"
  python3 "$here/preset_identity.py" "$binary" "$binary" \
    --old-env "NC_DSP_ISA=$a" --new-env "NC_DSP_ISA=$b" "$@" || status=1
done
exit $status
