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
- **Reading audit**: every decoder's picture-size limit, and the
  syntax-sized allocations and loops in the VVC and AV2 headers.

## Results (2026-11-13)

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
coverage is the highest edge count reached. Throughput is low because seeds
are real multi-frame pictures and the sanitizer build decodes about 30 times
slower than release (a valid 8-frame 1080p VVC input takes ~30 s): most
time goes into realistic, deep decodes rather than many shallow ones.

## Findings and fixes

| Decoder | Class | Cause | Fix |
| --- | --- | --- | --- |
| AV1 | unbounded allocation (by reading) | sequence header bounded each side at 16384 only: a 16384x16384 picture (~400 MB per frame buffer) was accepted | `e0336751`: level 6.3 limits (Annex A.3: 16384 wide, 8704 high, 35,651,584 samples) |
| AV1 | arithmetic overflow (12 fuzzer inputs, one cause; found on official 18-sample-wide vectors) | deblocking areas that start beyond the frame edge (MiCols rounds up to 8) gave a negative line range; release builds wrapped it to an empty run, debug builds panicked | `13c9d0e2`: such areas are skipped (§7.14.2 onScreen); output unchanged, 176/176 AOM vectors bit-exact |
| HEVC | index out of bounds (panic in release builds too) | `slice_pic_parameter_set_id` was not range-checked and indexed the 64 PPS slots: a 13-byte stream crashed the browser, C API and CLI decoders | `b5f5c850`: ids above 63 are a clean error (§7.4.7.1); 136/136 HEVC conformance streams still match HM |
| AV1 | division by zero / arithmetic overflow (panic in release builds too) | film grain scaling points were not checked to increase; the scaling lookup divides by their difference | `ca6ee2e5`: non-increasing points are a clean error (§6.8.20); 176/176 AOM vectors bit-exact |

Timeouts seen while merging seeds were valid multi-frame streams decoding
slowly under the sanitizer (decode time in proportion to picture area); they
are not findings.

## Not tested

- Inputs longer than 64 KiB, so long sequences, large pictures and
  long-GOP reference state are reached only through short prefixes.
- 10-bit and other profiles the decoders reject early (only the rejection is
  exercised).
- The WASM build itself: the targets run the same Rust code natively; the
  JavaScript glue and WebCodecs paths are not fuzzed. `nc-web`'s own
  conversion code (RGBA, I420 packing) is not called, only mirrored by
  reading every shown row.
- The nc-dsp SIMD kernels beyond what the decoders call on the host (AVX2);
  NEON and SIMD128 kernels are not fuzzed.
- Decoder speed as a denial of service: inputs are bounded by picture-size
  limits, not by a decode-time budget.
- Encoders.
