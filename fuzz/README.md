//! Public subset-decoder fuzz targets and bounded reproduction commands.

# Subset decoder fuzzing

Each target passes arbitrary bytes to the codec's public packet decoder and
lets panics, sanitizer findings and resource failures reach libFuzzer. Decoder
implementation and accepted syntax are not modified here. VP9 uses
`nc_vp9::stream::Decoder` (`decode_packet`; the input is up to four packets,
each preceded by its little-endian u16 length, so inter frames see their
references); AV2 uses `nc_av2::Decoder`.

The independent workspace avoids changing the root workspace manifest. Build
with `cargo +nightly fuzz build` from this directory after installing
`cargo-fuzz`. Run one target, for example:

```sh
CARGO_TARGET_DIR="$HOME/nanocodec-data/targets/astra-fuzz" \
  systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
  cargo +nightly fuzz run decode_h264 -- \
    -max_total_time=1800 -timeout=5 -rss_limit_mb=700 -max_len=262144
```

Repeat for `decode_hevc`, `decode_av1`, `decode_vp9`, `decode_av2`, and `decode_vvc`.
At most four fuzz workers may run together under the shared 4 GiB scope.
Use a dedicated corpus and artifact directory per target. Keep crash inputs;
minimise with `cargo +nightly fuzz tmin TARGET INPUT`, then replay with
`cargo +nightly fuzz run TARGET INPUT`. A timeout or memory-limit finding is
also evidence to report, not permission to modify the decoder.

An empty corpus mostly exercises header rejection. Seed real public-API packets
(IVF frame payloads for AV1/VP9/AV2, Annex-B access units for H.264/HEVC/VVC) to reach
valid parsing paths. Do not pass the IVF container header as an AV1/VP9/AV2 packet.
Campaign results and owner handoff belong in `docs/handoff/astra-fuzz.md`.

To continue after a discovered failure while retaining its input, use libFuzzer's
fork mode with one active worker per target:

```sh
cargo +nightly fuzz run decode_av1 CORPUS_DIR -- \
  -fork=1 -ignore_crashes=1 -ignore_ooms=1 -ignore_timeouts=1 \
  -max_total_time=1800 -timeout=5 -rss_limit_mb=700 \
  -malloc_limit_mb=512 -max_len=262144 -artifact_prefix=ARTIFACT_DIR/
```

Place all concurrent targets in **one** `MemoryMax=4G`, `MemorySwapMax=0` scope;
separate 4 GiB scopes would not enforce the shared limit. Fork supervisors do
not fuzz inputs; cap active fuzz workers at four. Shorter segments may reuse
the same corpus, but record and sum each target's actual fuzzing duration.
A successful supervisor exit is not proof of zero findings: inspect its
`oom/timeout/crash` counters and artifact directory.

The targets deliberately use the public decoder API, accept ordinary parse
errors, and let panics abort. Each input starts a fresh decoder. This campaign
covers packet parsing and valid seeded keyframes; it does not establish complete
coverage of long-lived inter-frame decoder state. No decoder fixes belong here.
