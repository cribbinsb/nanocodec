# AV2 reference fixtures

`avm-sequence-headers.hex` contains eight complete sequence-header payloads
produced by the ORIGINAL `av2_write_sequence_header_obu` in AVM v1.0.0,
commit `966a7d7cd6fcf60360caf5dc413b2aeeb65e144d`. Reference terms are
BSD-3-Clause-Clear and the separate AOM patent terms in `docs/licenses/avm`.
The repository's Rust sequence writer matches every byte.

Order: video then single-picture; dimensions 1×1, 64×48, 1920×1080,
65536×65536; sequence ID 15, level 31, eight-bit 4:2:0, Main IP0,
eight order-hint bits for video. Tool flags are the basic subset documented
in `docs/analysis/av2-subset.md`.

Regenerate/compare with `examples/sequence_oracle.c`,
`examples/sequence_vectors.rs` and `examples/check_sequence_oracle.py`.
These are header fixtures, not complete AV2 picture streams.

`avm-entropy.bin` is 25,099 bytes from AVM's original range encoder, checked
by AVM's original scalar decoder and the native Rust decoder. It contains
20,000 adaptive four-symbol events interleaved with 0..16 true bypass bits;
all CDF adaptation time intervals and long carry chains are exercised. The
native writer produces exactly these bytes. Reproduce by compiling
`examples/entropy_oracle.c` with original `avm_dsp/{entenc,entdec,entcode}.c`
against configured pinned headers and writing its stdout to the fixture.
