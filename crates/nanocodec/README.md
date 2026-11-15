# nanocodec

Real-time, low-latency video encoders and matching decoders for H.264, HEVC, VVC, AV1, VP9 and AV2 on one shared engine.

Once published, add it with `cargo add nanocodec`. Each codec is a cargo
feature (`h264`, `hevc`, `vvc`, `av1`, `vp9`, `av2`; all on by default):

```toml
nanocodec = { version = "0.1", default-features = false, features = ["h264", "av1"] }
```

The crate documentation has an encode and decode example. The command-line
tool is the `nanocodec-cli` crate.

- Repository: <https://github.com/cribbinsb/nanocodec>
- Documentation and results: <https://cribbinsb.github.io/nanocodec/>

## Licence and patents

BSD-3-Clause (`LICENSE`). Code reimplemented from other permissively
licensed projects is credited in `THIRD_PARTY.md`. The licence grants no
patent rights: H.264, HEVC and VVC are covered by patents, and using these
codecs may need licences from their holders.
