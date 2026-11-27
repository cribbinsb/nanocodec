# nanocodec

Low-latency (no B-frames, no lookahead) video encoders and matching decoders for H.264, HEVC, VVC, AV1, VP9 and AV2 on one shared engine.

Add it with `cargo add nanocodec`. Each codec is a cargo
feature (`h264`, `hevc`, `vvc`, `av1`, `vp9`, `av2`; all on by default):

```toml
nanocodec = { version = "0.8", default-features = false, features = ["h264", "av1"] }
```

The crate documentation has an encode and decode example. The command-line
tool is the `nanocodec-cli` crate.

The results site names the build each figure was measured on. Each GitHub
release attaches profile-guided (PGO) builds of the command-line tool and C
API; a crate built by cargo is a plain release build: on x86 it took
1.4-3.2 % more CPU than the PGO build (measured 2026-11-11 on AV1, HEVC and
H.264 Faster).

- Repository: <https://github.com/cribbinsb/nanocodec>
- Documentation and results: <https://cribbinsb.github.io/nanocodec/>

## Licence and patents

BSD-3-Clause (`LICENSE`). Code reimplemented from other permissively
licensed projects is credited in `THIRD_PARTY.md`. The licence grants no
patent rights: H.264, HEVC and VVC are covered by patents, and using these
codecs may need licences from their holders.
