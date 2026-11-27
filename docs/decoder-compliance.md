# Decoder compliance register

The one list of every known gap between our decoders and the standards they decode, with what is being
done about each (owner decision, 2026-11-27: every missing feature is implemented, or recorded here as an
exception with its reason).

**How it stays true.** `tools/ci/decoder_gaps.py` (run by `tools/ci/check.sh`) reads every rejection path
in the decoder crates (`crates/nc-{h264,hevc,vvc,av1,vp9,av2}/src`: each `DecodeError::Unsupported`
name and the helpers that wrap it) and fails if any name is missing from its codec's section below. A new
rejection path therefore cannot land without a row here. The names in backticks are the exact strings the
decoder reports. Encoder-side errors (`ConfigError::Unsupported`, e.g. "VvcBackend requires Codec::Vvc")
and decoder-crate files that hold only encoder or test-oracle code are excluded in the script, each with
its reason. Gaps that have no rejection path (a feature silently ignored, or one outside the native decoder)
are rows without names; the script cannot find those, so they are listed by hand.

**Status** is one of:

- **implemented**: decoded; the row records a former gap.
- **in progress (who)**: an agent is implementing it now.
- **open**: a gap no agent is on yet; it needs one or an exception decision.
- **proposed exception (owner to confirm)**: we think it is not worth implementing; the reason is given.
  Not an exception until the owner confirms it.

Each section also lists the rejections that are **not feature gaps**: guards against streams that break a
bitstream-conformance requirement, resource bounds above the highest level a profile defines, API
contracts (the caller used the wrong entry point), and routing checks after which the decoder takes its
general path instead of failing.

Suites and pass counts per codec: [FAQ, Conformance](FAQ.md#conformance). Gap audit this register
started from: [decoder-compliance-gaps-2026-11-27](analysis/decoder-compliance-gaps-2026-11-27.md).

## Summary (2026-11-27)

Feature rows per status (`tools/ci/decoder_gaps.py --summary`; the check fails if this table is stale):

| Codec | implemented | in progress | open | proposed exception |
|---|---|---|---|---|
| H.264 | 0 | 7 | 1 | 2 |
| HEVC | 0 | 1 | 1 | 1 |
| VVC | 0 | 4 | 2 | 1 |
| AV1 | 0 | 1 | 1 | 1 |
| VP9 | 0 | 2 | 0 | 0 |
| AV2 | 0 | 7 | 0 | 0 |
| Browser (WASM) module | 0 | 1 | 0 | 0 |

## H.264 [h264]

| Feature | Spec | Profiles | Status | Rejection names |
|---|---|---|---|---|
| Field pictures (PAFF) | H.264 §7.3.3 `field_pic_flag`, §8.2.4.2.5 | Main, Extended, High and above | in progress (H.264 interlace agent) | `H264 field pictures` |
| MBAFF frames | H.264 §7.3.4, §6.4.10 | Main, Extended, High and above | in progress (H.264 interlace agent) | `H264 MBAFF frames` |
| Pictures above level 5.2 (levels 6 to 6.2, 8K) | H.264 Table A-1 | all, levels 6 to 6.2 | in progress (compliance-extras agent: VVC NAL/layer/VPS ids, H.264 level 6, VP9 profile 1 sizing, WASM monochrome) | `H264 picture size` |
| 4:2:2 and 4:4:4 chroma, separate colour planes, lossless transform bypass, more than 10 bits | H.264 §7.4.2.1.1, §8.5.15, A.2.6 to A.2.11 | High 4:2:2, High 4:4:4 Predictive, their Intra profiles | in progress (non-4:2:0 chroma agent: H.264 High 4:2:2 / 4:4:4) | `H264 chroma format above 4:2:0` `H264 separate colour planes` `H264 lossless transform bypass` `H264 bit depth above 10` |
| Slice groups (FMO) and arbitrary slice order (ASO) | H.264 §7.3.2.2, §8.2.2 | Baseline, Extended | in progress (H.264 Baseline/Extended agent) | `H264 slice groups (FMO)` |
| Redundant pictures | H.264 §7.3.2.2 `redundant_pic_cnt_present_flag` | Baseline, Extended | in progress (H.264 Baseline/Extended agent) | `H264 redundant pictures` |
| Data partitioning (NAL types 2 to 4), SP/SI slices | H.264 §7.3.2.9, §8.6 | Extended | in progress (H.264 Baseline/Extended agent) | `H264 data partitioning` `H264 SP/SI slices` |
| Monochrome (4:0:0) with an odd cropped size | H.264 §7.4.2.1.1 frame cropping | High and above | open | `odd cropped 4:0:0 size` |
| MVC (multiview, stereo) | H.264 Annex H | Multiview High, Stereo High | proposed exception (owner to confirm): stereo/multiview video, never produced by live camera encoders or browsers; the base view already decodes | |
| SVC (scalable) | H.264 Annex G | Scalable Baseline/High | proposed exception (owner to confirm): no browser or WebRTC stack decodes H.264 SVC enhancement layers and current encoders do not produce them; the base layer already decodes | |

Not feature gaps:

| Rejection names | Why |
|---|---|
| `H264 bit depth above 8` | API contract: a deeper stream through the 8-bit call; `push_nal` decodes it. |
| `H264 packet allocation limit` | Resource bound on one packet's size against hostile input. |
| `several pictures in one packet` | API contract of the one-picture-per-packet call; `push_nal` takes any NAL sequence. |

## HEVC [hevc]

| Feature | Spec | Profiles | Status | Rejection names |
|---|---|---|---|---|
| Range extensions: 4:0:0, 4:2:2, 4:4:4, 12 and 16 bits, RExt tools | H.265 §7.3.2.2.2 `sps_range_extension`, A.3.5 | Main 12, Main 4:2:2 10/12, Main 4:4:4 (8 to 16), Monochrome, their Intra/Still profiles | in progress (non-4:2:0 chroma agent: HEVC RExt) | `range extension` `chroma format other than 4:2:0` `bit depth above 10` |
| Screen content coding (palette, IBC, ACT, adaptive MV resolution) | H.265 §7.3.2.2.3 `sps_scc_extension`, §7.3.2.3.3 `pps_scc_extension`, A.3.7 | Screen-Extended Main, Main 10, Main 4:4:4 (and High Throughput 4:4:4) | open; refused cleanly since 2026-11-27 (an SCC `general_profile_idc` 9 or 11 that claims no Main profile, or `pps_scc_extension_flag`), it used to misdecode silently. Implementing it is L, about 1.5 weeks: SPS/PPS SCC extensions and slice syntax (half a day), adaptive MV resolution (half a day), IBC with the current picture as reference (2 days), palette mode with predictor and escapes (3 days), ACT (1 day, after the 4:4:4 work of the RExt row), the JCT-VC SCC conformance set fetched and run (1 day) | `HEVC screen content coding` |
| Multilayer: MV-HEVC, SHVC, 3D-HEVC | H.265 Annexes F, G, H, I | Multiview Main, Scalable Main/Main 10, 3D Main | proposed exception (owner to confirm): stereo/3D and scalable broadcast layers are not produced by live camera encoders and no browser decodes them; the base layer already decodes | |

Not feature gaps:

| Rejection names | Why |
|---|---|
| `picture size` | Bound at level 6.2 (`MaxLumaPs` 35,651,584, side ≤ √(8 · MaxLumaPs)), the highest level H.265 defines. |
| `decoding-order output of more than 8 bits (use push and next_output)` | API contract: the 8-bit decoding-order call; `push` decodes the stream. |

## VVC [vvc]

| Feature | Spec | Profiles | Status | Rejection names |
|---|---|---|---|---|
| Reserved and unspecified NAL unit types to be ignored, not refused | H.266 §7.4.2.2 | all | in progress (compliance-extras agent: VVC NAL/layer/VPS ids, H.264 level 6, VP9 profile 1 sizing, WASM monochrome) | `VVC NAL type` `VVC metadata NAL type` |
| NAL units with `nuh_layer_id` above 0 to be ignored by a single-layer decoder | H.266 §7.4.2.2 | all | in progress (compliance-extras agent: VVC NAL/layer/VPS ids, H.264 level 6, VP9 profile 1 sizing, WASM monochrome) | `VVC layer/temporal id` |
| SPS with `sps_video_parameter_set_id` above 0 | H.266 §7.4.3.4 | Main 10 and up | in progress (compliance-extras agent: VVC NAL/layer/VPS ids, H.264 level 6, VP9 profile 1 sizing, WASM monochrome) | `VVC VPS-dependent SPS` |
| 4:2:2 and 4:4:4 chroma, palette, ACT | H.266 §7.3.2.4, §7.3.10.6 (palette), §8.7.4.6 (ACT), A.3.2 | Main 10 4:4:4 and its Still Picture profile | in progress (non-4:2:0 chroma agent: VVC 4:4:4/palette/ACT) | `VVC non-Main chroma format` `VVC palette mode` `VVC profile outside Main 10/Still Picture` |
| 12- and 16-bit operating range extensions | H.266 (v2) A.3.5, A.3.6 | Main 12, Main 12 4:4:4, Main 16 4:4:4 and their Intra/Still profiles | open | `VVC bit depth above 10` |
| Own-stream fast path: a stream whose SPS is byte-identical to one our encoder writes but that then uses a PPS, slice header, APS, partitioning or CU tool outside our encoder's subset is refused instead of being sent down the general path | H.266 §7.3.2.5, §7.3.7, §7.3.11 | Main 10 | open (fix: fall back to the general path, as an unmatched SPS already does) | `VVC PPS tools` `VVC slice header` `VVC APS outside the subset` `VVC ALF APS count` `VVC RPL0 size` `VVC subset temporal layer` `VVC NAL type/order or missing PPS` `VVC multi-type split` `VVC affine second reference` `VVC half-sample AMVR` `VVC rectangular intra CU` `VVC intra sub-partitions of this size` `VVC intra direction` `VVC transform size` |
| Multilayer (several layers, several pictures per access unit) | H.266 §7.4.3.3 (VPS), A.3.4 | Multilayer Main 10, Multilayer Main 10 4:4:4 | proposed exception (owner to confirm): scalable/multiview layers are not produced by live camera encoders and no browser decodes VVC layers; the base layer decodes once its NAL units are ignored properly (rows above) | `VVC multilayer profile` `VVC multiple pictures per access unit` |

Not feature gaps:

| Rejection names | Why |
|---|---|
| `VVC SPS tools` `VVC SPS partition constraints` `VVC VUI payload` `VVC coded dimensions` | Routing: our own-SPS match failed, so the decoder takes the general path (`decoder.rs` `accept_sps`); never a decode failure. |
| `VVC CTU size` | Guard: `CtbLog2SizeY` above 7 breaks a bitstream-conformance requirement (§7.4.3.4). |
| `VVC general transform size` `VVC residual geometry` | Guards on block geometry no conforming 4:2:0 stream produces (transform sides 1 to 64, §7.4.3.4); 4:4:4 geometry comes with the 4:4:4 row. |
| `VVC MVD prefix` `VVC motion vector range` `VVC POC range` | Guards on values outside the ranges §7.4.11.10 and §8.3.1 allow. |
| `VVC DPB capacity` | Bound: no Main 10 level allows more than 16 pictures in the DPB (A.4.2). |
| `VVC APS type` | Guard: `aps_params_type` values above 2 are reserved (§7.4.3.18) and their APS NAL units are not usable. |
| `VVC pending access unit size` | Resource bound on one access unit against hostile input. |
| `decoding-order output of more than 8 bits (use push_nal and next_output)` | API contract: the 8-bit decoding-order call; `push_nal` decodes the stream. |

## AV1 [av1]

| Feature | Spec | Profiles | Status | Rejection names |
|---|---|---|---|---|
| Quantizer matrices, block loop-filter deltas, segment features other than the quantizer, lossless segments in a mixed frame, odd frame sizes, switch frames | AV1 §5.9.12, §5.9.18, §5.9.14, §7.12.2, §5.9.5, §7.20 (S-frames) | Main | in progress (AV1 Main-gaps fix agent) | `AV1 quantizer matrices` `AV1 block loop filter deltas` `AV1 segment features other than the quantizer` `AV1 lossless segments` `AV1 odd frame size` `AV1 switch frames` |
| High and Professional profiles: 4:4:4, 4:2:2, 12 bits, identity matrix (RGB) | AV1 §5.5.2 `color_config`, Annex A.2 | High, Professional | open (no agent yet; not in the non-4:2:0 brief) | `AV1 profile other than Main` `AV1 identity matrix (4:4:4)` |
| Large-scale tile decoding (tile list OBUs) | AV1 §5.12 `tile_list_obu`, Annex D | all (separate decoding mode) | proposed exception (owner to confirm): only for random access into pre-encoded tiles (light-field/VR tile viewers, `--large-scale-tile` in libaom); never produced for live or streamed video, and dav1d does not implement it either | `AV1 tile list OBUs` |

Not feature gaps:

| Rejection names | Why |
|---|---|
| `AV1 frame size limit` | Bound at level 6.3 (`MaxPicSize` 35,651,584, side ≤ 16,384), the highest level AV1 defines limits for. |
| `AV1 OBUs per temporal unit` `AV1 temporal unit size limit` | Resource bounds against hostile input. |
| `AV1 operating point > 31` | Guard: at most 32 operating points (§5.5.1, 5-bit count). |
| `AV1 operating point out of range` | API contract: the caller asked for an operating point the stream does not have. |
| `AV1 bit depth of another instance` `AV1 picture of more than 8 bits through an 8-bit call` | API contract: one decoder instance per bit depth; the high-bit-depth call decodes the stream. |

## VP9 [vp9]

| Feature | Spec | Profiles | Status | Rejection names |
|---|---|---|---|---|
| 4:2:2, 4:4:0, 4:4:4 chroma (and sRGB) | VP9 §6.2 `color_config`, §7.2 | 1, 3 | in progress (non-4:2:0 chroma agent: VP9 profiles 1/3) | `VP9 profile 1 or 3 (4:4:4, 4:2:2 or 4:4:0)` |
| Profile 1 frame sizing (chroma subsampling in reference-sized frames) | VP9 §7.2 `frame_size_with_refs` | 1, 3 | in progress (compliance-extras agent: VVC NAL/layer/VPS ids, H.264 level 6, VP9 profile 1 sizing, WASM monochrome) | |

Not feature gaps:

| Rejection names | Why |
|---|---|
| `VP9 reserved profile` | Guard: the reserved bit after profile 3 must be 0 (§7.2). |
| `VP9 sRGB in profile 0` `VP9 sRGB in profile 0 or 2` | Guard: `color_space` CS_RGB is not allowed in profiles 0 and 2 (§7.2). |
| `VP9 frame marker` `VP9 empty compressed header` | Guards: `frame_marker` must be 2 and `header_size_in_bytes` non-zero (§7.2). |
| `VP9 frame size beyond level 6.2` `VP9 frame size` | Bound at level 6.2, the highest VP9 level, and the picture allocation that follows from it. |
| `VP9 motion difference outside the coded range or precision` | Guard on motion vector differences outside what §6.4.19 can code. |
| `VP9 odd frame dimensions` `VP9 packet without a shown frame` `VP9 bit depth of another decoder instance` | API contract of the simple calls (exact-size, one shown frame per packet, one bit depth per instance); `push`/`next_output` decode these streams. |
| `VP9 operating point out of range` `VP9 spatial superframe; use --operating-point` | API contract: no or a wrong operating point for a spatial-SVC stream. |
| `VP9 profile other than 0` `VP9 temporal layer count out of range` `VP9 frame sized from a reference: its width is needed` | The SVC layer-extraction tool (`extract.rs`, `superframe.rs`) for our own SVC streams, not the decoder. |

## AV2 [av2]

AV2 is a draft (AVM v1.0.0); the decoder's general path decodes other encoders' single-tile 8-bit 4:2:0
streams. The rows below are the general decoder's gaps.

| Feature | Spec | Profiles | Status | Rejection names |
|---|---|---|---|---|
| 10-bit, monochrome (4:0:0) and other chroma formats | AVM v1.0.0 `color_config` | Main 4:2:0 10-bit and above | in progress (AV2 agent) | `AV2 bit depth other than eight` `AV2 chroma format other than 420` `AV2 transform requires an 8-bit residual` |
| Several tiles, several tile groups | AVM `tile_info`, tile group OBU | all | in progress (AV2 agent) | `AV2 multiple tiles` `AV2 several tile groups in a frame` |
| References of another size (reference scaling) | AVM motion compensation with scaled references | all | in progress (AV2 agent) | `AV2 references of another size` |
| Multi-frame headers, bridge frames, quantizer-matrix / layer / operating-point OBUs, embedded and extended layers | AVM OBU syntax | all | in progress (AV2 agent) | `AV2 multi-frame headers` `AV2 bridge frames` `AV2 OBU type outside the decoder` `AV2 embedded layers` `AV2 extended layers` |
| Wedge compound when motion fields are stored, TIP frames with an explicit quantizer, odd 4:2:0 sizes and conformance windows | AVM inter prediction, TIP, frame size | all | in progress (AV2 agent) | `AV2 wedge compound motion field` `AV2 TIP frames with an explicit quantizer` `AV2 4:2:0 frame of odd width or height` `AV2 conformance window of odd or empty size` |
| Bilinear chroma phase of odd eighth-sample vectors | AVM inter prediction | all | in progress (AV2 agent) | |
| Block tools inside a frame routed to the subset tile decoder (our encoder's syntax) that the subset lacks, and mixing subset and general CDF contexts | AVM block syntax | all | in progress (AV2 agent: the general block layer replaces the subset path) | `AV2 subset frames inheriting a general frame's CDFs` `AV2 general frames inheriting a subset frame's CDFs` `AV2 cross-frame context without CDF updates` `AV2 skip mode with switchable filters` `AV2 skip mode with two references` `AV2 palette` `AV2 intra DPCM` `AV2 chroma DPCM` `AV2 CCSO filter reuse` `AV2 large lossless inter transform` `AV2 lossless coding uses only the 4x4 WHT` `AV2 lossless inter IDTX` `AV2 lossless requires TX4` `AV2 non-DCT transform type` `AV2 1-D transform type` `AV2 intra mode-dependent transform type` `AV2 reduced transform set` `AV2 transform type is inferred as DCT here` `AV2 transform type outside the frame's set` `AV2 transform type does not exist at this size` `AV2 chroma transforms stop at TX32` `AV2 transform partition block size` `AV2 transform partition other than the 4-way split` `AV2 chroma mode outside the luma's list` `AV2 non-8x8 leaf` `AV2 rectangular partition leaf` `AV2 sub-8x8 partition` `AV2 partition implied at the boundary differs from the decision` `AV2 partition split implied at the boundary` `AV2 prediction block size` |

Not feature gaps:

| Rejection names | Why |
|---|---|
| `AV2 MV grid dimensions` `AV2 reference distances` | Guards on internal motion-field bounds (sizes above the decoder's maximum dimension, distances outside 1 to 7 references). |
| `AV2 temporal unit without a shown picture` | API contract of the one-picture call; `next_output` takes such units. |

The subset sequence parse (`sequence.rs`), the subset frame-tool check (`general/subset.rs`) and the
encoder's header writer and test oracle (`frame_header.rs`, `frame_header/read_back.rs`) are excluded in
the script: the first two only route a stream to the general path, the last two are encoder code.

## Browser (WASM) module

| Feature | Spec | Profiles | Status | Rejection names |
|---|---|---|---|---|
| Monochrome (4:0:0) output in the browser module (all codecs) | per codec, as above | H.264 High, HEVC/VVC 4:0:0, AV1 Main | in progress (compliance-extras agent: VVC NAL/layer/VPS ids, H.264 level 6, VP9 profile 1 sizing, WASM monochrome) | |
