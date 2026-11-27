# nanocodec-rtp

RTP payload formats for nanocodec streams: packetizers and depacketizers for H.264 (RFC 6184), HEVC (RFC 7798), VVC (RFC 9328), AV1 (the AOM RTP payload format, with the Dependency Descriptor header extension for SVC forwarding) and VP9 (RFC 9628), plus pcap and RFC 4571 capture files.

This is one of the crates of [nanocodec](https://github.com/cribbinsb/nanocodec).
Most users want the [`nanocodec`](https://crates.io/crates/nanocodec) crate,
which re-exports this one as `nanocodec::rtp`. In Rust code this crate's
library is `nc_rtp`.

- Repository: <https://github.com/cribbinsb/nanocodec>
- Documentation and results: <https://cribbinsb.github.io/nanocodec/>

## Licence and patents

BSD-3-Clause (`LICENSE`). Code reimplemented from other permissively
licensed projects is credited in `THIRD_PARTY.md`. The licence grants no
patent rights: H.264, HEVC and VVC are covered by patents, and using these
codecs may need licences from their holders.
