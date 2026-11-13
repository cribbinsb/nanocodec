# Exact AV2 bypass batching investigation

2026-11-08. Follow-up: verified 16-bit encoder batching is now applied in
`entropy.rs`; decoder arithmetic remains unchanged.
Reference AVM v1.0.0 commit `966a7d7cd6fcf60360caf5dc413b2aeeb65e144d`.

**Implemented:** encoder literal batching up to 16 bits. Both 8-bit and 16-bit
chunks pass exact-byte checks against original AVM and the native
bit-at-a-time reference updates. Keep decoder batching as a separate experiment.

## Source and arithmetic

AVM `avm_dsp/bitwriter.h:avm_write_literal` splits literals into at most eight
bits. `entenc.c:od_ec_encode_literal_bypass` computes
`low = (low << n) + range * value`, then normalizes once. Since bypass leaves
`range` unchanged, this is the algebraic composition of n one-bit updates.
Native `Writer::normalize` already accepts a bypass-bit count and emits up to
two carry bytes. Its pending count is −9..−1 after normalization, so adding
at most 16 bits makes at most two bytes ready. The native `u64` accumulator
holds the intermediate value; the normalized low value needs at most 24 bits,
so shifting by 16 remains safely within 64 bits. Do not batch arbitrary
32-bit literals into a single call with this normalization implementation.

AVM `bitreader.h:avm_read_literal_` also groups by eight, but
`entdec.c:od_ec_decode_literal_bypass` still compares/subtracts one bit at a
time and refills once per group. It does **not** establish that an integer
quotient/remainder decoder would be faster. A native decoder experiment could
first batch input-byte extraction while retaining register-local bit steps;
changing both the arithmetic and refill mechanism at once is unnecessary.
The existing bound of at most 15 implicit trailing zero bits must survive.

The [paper](https://arxiv.org/html/2601.02712v1), Appendix A, motivates entropy
throughput work; it is not a benchmark of either proposed native implementation.

## Reproducible evidence

`tools/codegen/check_av2_bypass_batching.py` compiles original C and a temporary
Rust crate containing the current entropy source with only the literal loop
replaced. It retains neither raw media nor temporary binaries/source. Example:

```sh
nice -n 19 python3 tools/codegen/check_av2_bypass_batching.py \
  --source ~/nanocodec-data/anchors/vp9-av2/src/avm-v1.0.0 \
  --build ~/nanocodec-data/anchors/vp9-av2/build/avm-v1.0.0 \
  --chunk 16
```

Both chunk sizes passed the initial 11-test diagnostic, including existing adaptive-CDF oracle and
rollback tests. The new mixed stream interleaves 8,192 adaptive four-symbol
codes with literal widths cycling through 0..32. Both candidates produce the
same 18,438 bytes as original C and native scalar coding. Additional checks
cover all widths, zero/max/half values at 16 prefix offsets, rejected widths
and overflowing values without writer-state mutation, 257 truncated prefixes,
and 512 bit mutations with bounded, matching decoder outcomes. The decoder
itself is unchanged, so these are compatibility/bounds regression checks, not
proof of a new grouped decoder.

| Writer | Literal normalization calls in mixed fixture |
| --- | ---: |
| Former one-bit loop | 130,972 |
| Chunks up to 8 bits | 19,847 |
| Chunks up to 16 bits | 11,911 |

These are structural call counts, **not timings or a claimed codec speedup**.
Single-bit signs and unary calls receive no batching benefit. The mixed-width
fixture intentionally exercises long literals more heavily than typical video
may do. Chunk extraction also has a cost. An isolated optimized-build test and
one existing representative encode pair should decide between 8/16-bit chunks
and whether a dedicated one-bit fast path is useful. Output bytes must remain
identical; no quality sweep is needed for this exact implementation change.
