# nanocodec-hevc

Nanocodec HEVC encoder backend for the shared engine, and an HEVC Main, Main 10 and Main Still Picture decoder (147/147 of the official HEVC_v1 conformance suite).

What the decoder passes and every valid-stream feature it does not decode yet: [FAQ: Conformance](https://github.com/cribbinsb/nanocodec/blob/main/docs/FAQ.md#conformance).

This is one of the crates of [nanocodec](https://github.com/cribbinsb/nanocodec).
Most users want the [`nanocodec`](https://crates.io/crates/nanocodec) crate,
which re-exports the engine and the codecs behind cargo features. In Rust
code this crate's library is `nc_hevc`.

- Repository: <https://github.com/cribbinsb/nanocodec>
- Documentation and results: <https://cribbinsb.github.io/nanocodec/>

## Licence and patents

BSD-3-Clause (`LICENSE`). Code reimplemented from other permissively
licensed projects is credited in `THIRD_PARTY.md`. The licence grants no
patent rights: H.264, HEVC and VVC are covered by patents, and using these
codecs may need licences from their holders.
