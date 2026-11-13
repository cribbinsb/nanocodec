//! Public decoder fuzz targets: input layout, seeds and bounded run commands.

# Decoder fuzzing

One cargo-fuzz target per decoder (`decode_h264`, `decode_hevc`, `decode_vvc`,
`decode_av1`, `decode_vp9`, `decode_av2`). Each drives the entry points a
stranger's bytes reach, so a finding is a finding in shipped code:

| Config byte `& 3` | Path | Input after the config byte |
| --- | --- | --- |
| 0 | browser (`nc-web` `BrowserDecoder`): `decode_shown`, then the shown picture read row by row | packets separated by `NCPK` |
| 1 | C API receiver (`nc-capi`): `decode` (AV1 `decode_optional`) | packets separated by `NCPK` |
| 2 | command-line decoder: the CLI's own framing (`crates/nc-cli/src/framing.rs`, compiled in) and its per-codec streaming calls (H.264 `decode_nal` + output queue, HEVC/VVC `push_nal` + output + flush, AV1/VP9/AV2 IVF) | a whole Annex-B or IVF file |
| 3 | as 2 with `--all-layers 1` (AV1 `decode_all_layers`, VP9 superframes) | a whole IVF file |

Bits 2..6 are the AV1 operating point (VP9: `> 0` decodes the base layer
only); bit 7 skips AV1 film grain. One decoder instance takes every packet of
an input, so references, probability contexts, reordering and layer state
carry over. `fuzz_targets/common/mod.rs` holds the shared layout code.

## Seeds

`fuzz/build_seeds.py --codec C --out DIR STREAM...` turns streams (Annex-B or
IVF; H.264/HEVC in IVF as in `web/streams-ref`) into seeds of every layout,
each holding at most 8 leading frames within `--max-len` (65536). Sources
used: retained landing-conformance streams (every shipped preset, layers,
refresh, starved keys, SVC), the official vectors under
`~/nanocodec-data/conformance`, `web/streams-ref`, and short encodes of our
own (colour-signalled, tiny and odd sizes, `--resize`, LTR, film grain,
L2T3/L3T3). Merge them into a minimal corpus with `-merge=1`.

## Build and run

Every cargo call goes through `~/nanocodec-data/bin/cargo-capped`; cargo-fuzz
has no `-j`, so set `CARGO_BUILD_JOBS=4`. Nightly with address sanitizer:

```sh
export CARGO_TARGET_DIR=$HOME/nanocodec-data/targets/<agent> CARGO_BUILD_JOBS=4
~/nanocodec-data/bin/cargo-capped +nightly fuzz build --fuzz-dir fuzz --sanitizer address
```

Copy the built binary aside before running it, so a rebuild cannot change a
session under way. One session (at most three workers machine-wide, one
memory-capped scope, at most 25 minutes):

```sh
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
  "$BINARY" "$CORPUS" -dict=fuzz/fuzz.dict -max_len=65536 -rss_limit_mb=4096 \
  -malloc_limit_mb=2048 -timeout=10 -max_total_time=1500 -fork=3 \
  -ignore_crashes=1 -ignore_ooms=1 -ignore_timeouts=1 \
  -artifact_prefix="$ARTIFACTS/" -print_final_stats=1
```

(`fuzz.dict` holds the packet separator, start codes and the IVF
signature). A clean exit proves nothing by itself: read the final
`oom/timeout/crash` counters and the artifact directory. A timeout is a
finding only when the decode time is out of proportion to the coded picture
area (a valid 49-frame 832x480 VVC stream takes ~45 s under the sanitizer).
Replay an artifact with `"$BINARY" ARTIFACT`, minimise it with
`-minimize_crash=1 -runs=N`, fix the cause, and land the minimised input as
a regression test in the decoder crate.

Campaign results: `docs/results/fuzzing.md`; working log:
`docs/handoff/astra-fuzz.md`.
