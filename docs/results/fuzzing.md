# Decoder fuzzing

What has been run against nanocodec's own subset decoders with malformed input,
what it found, and what was not tested. A bounded search finding nothing more
is not a proof that a decoder is safe.

## Method

- **Targets** (`fuzz/`, cargo-fuzz, libFuzzer with AddressSanitizer, debug
  assertions and overflow checks on): one per decoder. Each input drives one
  decoder instance through the entry point the first byte selects: the
  browser's `decode_shown` plus the shown picture (nc-web), the C API's
  `decode` (nc-capi), or the command-line decoder's own Annex-B/IVF framing
  and streaming calls, with or without `--all-layers` (`fuzz/README.md`).
  Packets share one decoder, so references, probability contexts, reordering
  and layer state carry over between them.
- **Seeds** (`fuzz/build_seeds.py`, at most 8 frames per seed): the retained
  landing-conformance streams (every shipped preset; temporal layers, intra
  refresh, starved keys, spatial SVC), the official vectors we hold (H.264
  JVT, HEVC and VVC conformance, AOM AV1, libvpx VP9), `web/streams-ref`
  (openh264, x264, x265, SVT-AV1, NVENC, libvpx) and our own short encodes
  (colour-signalled, 16x16, 66x34, `--resize`, LTR, film grain, L2T3, L3T3).
  AV2 has no official vectors here: its seeds are our own streams only.
- **Limits**: 10 s per input, 4 GiB RSS, 2 GiB per allocation, inputs of
  64 KiB (first round) or 16 KiB (later rounds), at most three workers machine-wide on
  a loaded shared 5950X.
- **Reading audit**: for every decoder, each array indexed by a value read
  from the stream and each allocation, loop or arithmetic sized by one
  (table in `docs/handoff/astra-fuzz.md`), including sums and products that
  wrap a 32-bit `usize` in the WASM build.
- **Round 2 (deeper per CPU-hour)**: seeds cut to 3-frame encodes of 16x16
  to 64x64 pictures for every preset and tool, plus header-only seeds
  (parameter sets, sequence headers, VP9 superframe indices, IVF headers,
  truncated first packets); 4 KiB inputs; a per-codec dictionary (start
  codes, NAL/OBU header bytes, VP9 frame and superframe markers); three
  rotations of 25-minute single-worker sessions per decoder.
- **WASM check**: `fuzz/replay` runs the same drivers built for
  wasm32-wasip1 with SIMD128 under node: every triaged crash input and
  every corpus input (round 1 and round 2) must replay without a trap.

## Results

### Round 1 (2026-11-13, inputs up to 64 KiB)

| Decoder | Sessions | Worker-hours | Executions | Edges covered | Corpus (inputs) | Findings |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| VVC | 3 | 2.6 | 117,345 | 16,567 | 2,851 | none |
| AV2 | 2 | 1.3 | 237,719 | 9,677 | 1,597 | none |
| H.264 | 2 | 1.3 | 167,566 | 9,680 | 2,117 | none |
| HEVC | 3 | 1.3 | 78,418 | 8,947 | 2,259 | 1 crash (one cause) |
| AV1 | 3 | 2.4 | 102,145 | 18,351 | 3,181 | 2 crashes (one cause), plus 12 on seed merge (one cause) |
| VP9 | 2 | 0.8 | 21,507 | 7,225 | 1,631 | none |

Each session ran 20-25 minutes with one to three workers; worker-hours are
session time times workers. No out-of-memory or timeout findings.

Executions are libFuzzer's own totals per session, summed;
coverage is the highest edge count reached. Round 1 throughput was low because seeds
are real multi-frame pictures and the sanitizer build decodes about 30 times
slower than release (a valid 8-frame 1080p VVC input takes ~30 s): most
time goes into realistic, deep decodes rather than many shallow ones.

### Round 2 (2026-11-13/14, 4 KiB inputs, one worker per session)

| Decoder | Sessions | Worker-hours | Executions | Exec/s | Edges covered | Corpus (inputs) | Findings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HEVC | 3 | 1.3 | 323,667 | 71 | 8,924 | 2,379 | none |
| VP9 | 3 | 1.3 | 147,644 | 33 | 7,676 | 2,437 | none |
| H.264 | 3 | 1.3 | 543,844 | 118 | 9,792 | 2,380 | none |
| VVC | 3 | 1.3 | 87,375 | 19 | 14,265 | 2,728 | none |
| AV2 | 3 | 1.3 | 833,562 | 180 | 10,418 | 1,829 | none |
| AV1 | 3 | 1.3 | 88,004 | 19 | 17,856 | 2,350 | none |

Small inputs raised throughput 1.5-5 times per worker against round 1's
averages (HEVC from 17 to 71 executions per second, VP9 from 7 to 33,
H.264 from 36 to 118, AV2 from 52 to 180, VVC from 13 to 19, AV1 from 12 to
19), not to thousands: a whole
decode of a 64x64 three-frame input costs 1-2 ms natively and 20-50 times
more under AddressSanitizer with coverage and overflow checks. The edge
counts are lower than round 1's for VVC and AV1 because 4 KiB inputs do not
reach the larger-picture paths that round 1's corpus covers. In all, the two
rounds ran about 2.75 million executions.

WASM: the 13,636 round-1 corpus inputs, the 14,103 round-2 corpus inputs
and all 15 triaged crash inputs replayed in the wasm32 build (SIMD128)
without a trap. As a control, a WASM build from before the HEVC fix traps
on that crash's input.

### Round 3 (2026-11-23, at 0.6.0: 10-bit and foreign seeds, 64 KiB inputs, eight workers per decoder)

Commit 98069a8b3 (nanocodec 0.6.0 plus later integration work), built
with the round 1 and 2 settings (AddressSanitizer, debug assertions,
overflow checks). Every decoder has gained code since round 2: HEVC Main 10,
AV1 10-bit and film grain, VP9 profile 2, H.264 High 10, the general VVC
decoder (Main 10, 226 conformance streams) and the general AV2 decoder.

- **Seeds** (`fuzz/build_seeds.py --max-len 65536 --max-frames 3 --headers`):
  the leading frames of every stream in each codec's conformance manifest that
  fit in 64 KiB, plus header-only and truncated seeds of every one of them
  (H.264 190 streams with 9 High 10 / High 10 Intra, HEVC 147 with 11 Main 10,
  VVC 226 with 217 10-bit, AV1 242 with 66 10-bit and the film-grain vectors,
  VP9 305 with the profile 2 10- and 12-bit vectors); `web/streams-ref`
  (openh264, x264, x265, SVT-AV1, aom, libvpx, NVENC); ffmpeg encodes of a
  synthetic picture at 64x64, 66x34 and 176x144 (x264 8-bit and High 10, x265
  Main and Main 10, libaom 10-bit with grain synthesis, SVT-AV1 10-bit with
  film grain, libvpx-vp9 profile 0 and profile 2); and our own encodes of the
  same pictures at every preset, plain, L1T3 and with a 2-frame key period
  (54 per codec, 45 for AV2). AV2 has no official or foreign streams here.
  Seeds: H.264 1,968, HEVC 1,200, VVC 1,812, AV1 2,667, VP9 2,813, AV2 301.
- **Run**: one 25-minute session per decoder, all six at once, each in its own
  8-vCPU slot of a 96-vCPU AWS box (`nc-remote`), `-fork=8`; 10 s per input,
  2 GiB RSS and 2 GiB per allocation per worker, `-max_len=65536`, the codec's
  dictionary; crashes, OOMs and timeouts recorded and the session continued.

| Decoder | Workers | Worker-hours | Executions | Exec/s per worker | Edges covered | Corpus (inputs) | Findings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H.264 | 8 | 3.4 | 388,339 | 32 | 16,020 | 3,113 | none |
| HEVC | 8 | 3.4 | 243,225 | 20 | 17,574 | 3,468 | none |
| VVC | 8 | 3.4 | 165,540 | 14 | 42,360 | 3,710 | none |
| AV1 | 8 | 3.4 | 125,117 | 10 | 38,477 | 4,296 | none |
| VP9 | 8 | 3.4 | 180,120 | 15 | 13,490 | 3,317 | none (5 timeouts on large declared pictures, below) |
| AV2 | 8 | 3.4 | 192,916 | 16 | 23,755 | 2,910 | 3 crashes, one cause (open) |

Executions, edges and corpus are libFuzzer's final fork-mode totals. Edge
counts are not comparable with rounds 1 and 2: the binaries hold much more
code. In all, round 3 ran about 1.3 million executions in 20 worker-hours.

- **AV2, fixed**: a CCSO band-only filter with `max_band_log2` 7 (3 bits read,
  128 bands) overran the 72-entry `PlaneFilter` offset table
  (`crates/nc-av2/src/ccso.rs:131`), an index-out-of-bounds panic (release
  builds too) reached through `general::filters::ccso_params` from
  `Decoder::filter_decoded_planes`. AVM v1.0.0 rejects more than
  `CCSO_BAND_NUM` (64) bands ("Invalid CCSO Band number", `decodeframe.c`), so
  the general header parser now rejects 7 as a clean error, and
  `ccso_params` checks each filter's shape before filling its table.
- **VP9 timeouts, fixed**: the five inputs (one, `220c4518`, also logged as a
  slow unit) replayed in 6-26 s each under the sanitizer; their key frames
  declare pictures such as 2896x65536 and 4081x53478. The decoder now rejects
  coded pictures larger than the highest level's area, 6.2 (libvpx
  `vp9_level_defs`: 35,651,584 samples), as AV1 caps at its level 6.3
  (`e0336751`). VP9 signals no level, so the side is not capped: the libvpx
  vector `vp90-2-13-largescaling` is 19200x108, beyond 6.2's 16832.
- **Regression inputs** live in `~/nanocodec-data/fuzz/regressions/{av2,vp9}`
  (retained), checked in as `crates/nc-{av2,vp9}/tests/data/fuzz-*.fuzz` and
  run by each crate's `tests/malformed.rs` (the expected error; VP9 within
  10 s). The round's corpora and artifacts are kept in
  `~/nanocodec-data/fuzz/round3-2026-11-23`.
- **WASM** (2026-11-23, after the fixes): every round-3 corpus input and
  artifact replayed in the wasm32-wasip1 build (SIMD128) under node 22.22.1
  without a trap: H.264 1,968, HEVC 1,200, VVC 1,812, AV1 2,667, VP9 2,819,
  AV2 304. Control: the same replay at the commit before the fixes traps on
  each of the three AV2 crash inputs (`ccso.rs:131`) and on nothing else in
  AV2 or VP9.

## Findings and fixes

| Decoder | Class | Cause | Fix |
| --- | --- | --- | --- |
| AV1 | unbounded allocation (by reading) | sequence header bounded each side at 16384 only: a 16384x16384 picture (~400 MB per frame buffer) was accepted | `e0336751`: level 6.3 limits (Annex A.3: 16384 wide, 8704 high, 35,651,584 samples) |
| AV1 | arithmetic overflow (12 fuzzer inputs, one cause; found on official 18-sample-wide vectors) | deblocking areas that start beyond the frame edge (MiCols rounds up to 8) gave a negative line range; release builds wrapped it to an empty run, debug builds panicked | `13c9d0e2`: such areas are skipped (§7.14.2 onScreen); output unchanged, 176/176 AOM vectors bit-exact |
| HEVC | index out of bounds (panic in release builds too) | `slice_pic_parameter_set_id` was not range-checked and indexed the 64 PPS slots: a 13-byte stream crashed the browser, C API and CLI decoders | `b5f5c850`: ids above 63 are a clean error (§7.4.7.1); 136/136 HEVC conformance streams still match HM |
| AV1 | division by zero / arithmetic overflow (panic in release builds too) | film grain scaling points were not checked to increase; the scaling lookup divides by their difference | `ca6ee2e5`: non-increasing points are a clean error (§6.8.20); 176/176 AOM vectors bit-exact |
| H.264 | arithmetic overflow on 32-bit (by reading) | crop offsets were doubled and summed before the size check; on a 32-bit `usize` (WASM) they wrap into a window outside the picture | `17778a70`: bounded as read; 65/65 conformance streams pass |
| HEVC | arithmetic overflow (by reading) | about twenty `ue(v)` values (log2 sizes, POC bits, DPB and RPS counts and deltas, tiles, merge level, entry-point length, conformance window) were used in `+ n`, shifts or sums before their range check | `fec88cbf`: each read with its §7.4 range (`read_ue_max`); 136/136 HEVC conformance streams match HM |
| VVC | arithmetic overflow (by reading) | the chroma QP table start was biased before its range check | `d0bb84bf`: checked first (§7.4.3.4); 9/9 VVC Main vectors match |
| AV2 | index out of bounds (panic in release builds too; 3 inputs, one cause) | band-only CCSO `max_band_log2` 7 (128 bands) indexed past the 64-band offset table | general header rejects more than 64 bands, as AVM does; `ccso_params` checks the filter shape first |
| VP9 | unbounded picture size (5 timeouts) | the 16-bit size fields admit pictures up to 65536x65536 | coded pictures beyond level 6.2's area (35,651,584 samples) are a clean error; sides uncapped (the libvpx vectors include 19200x108) |

Timeouts seen while merging seeds were valid multi-frame streams decoding
slowly under the sanitizer (decode time in proportion to picture area); they
are not findings.

## Not tested

- Inputs longer than 64 KiB, so long sequences, large pictures and
  long-GOP reference state are reached only through short prefixes.
- 10-bit was seeded only in round 3 (2026-11-23: 25 minutes, eight workers
  per decoder), from the official 10-bit vectors' leading frames within
  64 KiB and small foreign encodes; profiles the decoders reject (H.264
  4:2:2 / 4:4:4, interlaced H.264, VP9 profiles 1 and 3, VVC 4:2:2 / 4:4:4
  and multilayer) are exercised only up to the rejection.
- Fuzzing inside the WASM build: the WASM decoders were only replayed on
  the corpora, not fuzzed there; the JavaScript glue and WebCodecs paths
  are not covered. `nc-web`'s own conversion code (RGBA, I420 packing) is
  not called, only mirrored by reading every shown row.
- The NEON kernels; SIMD128 kernels only through the WASM corpus replay.
- Decoder speed as a denial of service: inputs are bounded by picture-size
  limits, not by a decode-time budget.
- Encoders.
