# nanocodec-web

Browser WASM bindings for nanocodec encode and decode.

This is one of the crates of [nanocodec](https://github.com/cribbinsb/nanocodec).
Most users want the [`nanocodec`](https://crates.io/crates/nanocodec) crate,
which re-exports the engine and the codecs behind cargo features. In Rust
code this crate's library is `nc_web`.

- Repository: <https://github.com/cribbinsb/nanocodec>
- Documentation and results: <https://cribbinsb.github.io/nanocodec/>

## Decoding deep streams

`BrowserDecoder.decode(packet)` and `decode_frame(...)` also accept the
supported 10-bit streams of AV1, HEVC, VP9, H.264 and VVC. Packets retain
the existing decoding-order, one-picture/access-unit contract. The renderer
still receives 8-bit I420 or RGBA: a deep frame is truncated by each plane's
own `bit_depth - 8` shift into a reused display picture. This conversion
preserves the decoder's original samples and references. It does not tone
map HDR or dither.

- `bit_depth` and `chroma_bit_depth`: native sample depths; zero before output.
- `decoded_i420_u16()`: native packed Y/Cb/Cr words, with every significant
  bit retained (also available for 8-bit output).
- `decoded_i420()`, `i420_frame()` and `rgba_frame(colour)`: the existing
  8-bit display interfaces. Eight-bit output keeps its direct borrowed path.

The browser display requires an even 4:2:0 visible window, as before.
Native decoder/conformance interfaces retain their broader output support.
See the [WASM deep-output handoff](../../docs/handoff/ext-wasm10.md) for
reproducible tests and measurements.

## Licence and patents

BSD-3-Clause (`LICENSE`). Code reimplemented from other permissively
licensed projects is credited in `THIRD_PARTY.md`. The licence grants no
patent rights: H.264, HEVC and VVC are covered by patents, and using these
codecs may need licences from their holders.
