# VP9 reference vectors

`libvpx-bool.bin` (3559 bytes) is produced by
`python3 tools/codegen/gen_vp9_bool_vector.py` using libvpx v1.16.0 commit
`1024874c5919305883187e2953de8fcb4c3d7fa6`'s original C writer. It contains
20,000 symbols, cycling every eight-bit probability and taking bits from the
specified xorshift32 sequence. The Rust test compares bytes and decodes every
symbol. SHA-256:
`04d8b7bc8ef43db168a861ff6657cc761dc7da591f0ab7119721d250ffae4ee5`.

It is synthetic test data, not performance measurements. Reference software
notices are retained in `docs/licenses/libvpx/` and `THIRD_PARTY.md`.

## Fuzz regressions

`fuzz-4x4-idct-overflow.ivf` (1662 bytes) is libFuzzer artifact
`crash-55628c130652583e3faadd14889415818743cdf1` from Astra's seeded VP9
decoder campaign (`~/nanocodec-data/work/astra-fuzz-seeded/runs/vp9`): an
IVF stream whose corrupt coefficients overflowed the 4x4 inverse DCT's
32-bit products (a debug-build panic; 123 of the campaign's 125 artifacts).
`tests/malformed.rs` decodes it and requires no panic. SHA-256:
`b7c281ef0d776e8550e105e363fddd1a09f40008f866fa5e2507ac4085f21c79`.
