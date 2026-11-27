# Decoder compliance gaps (audit 2026-11-27)

Read-only audit of what each of our decoders rejects or ignores among **valid**
streams of its codec's main consumer profiles, at main `02975de9e`. Method: every
`DecodeError::Unsupported` (and `unimplemented!`/`todo!`) in the decoder paths of
nc-h264, nc-hevc, nc-vvc, nc-vp9 and nc-av2, read in context; the size/level caps;
the 2026-11-23 conformance record (`docs/results/conformance-2026-11-23/summary.json`)
and `tools/ci/*_conformance.sh`. AV1 is being fixed by another agent; its suite
position is listed but its tool gaps are not re-audited here. Nothing was run:
"encoders produce it" is from the encoders' documented defaults and options, not
from new test streams.

Ranking: gaps that common encoders produce in default or common settings first.
Size: S = hours, M = a day or two, L = a week or more.

## H.264 (target: Constrained Baseline, Main, High, High 10; progressive and interlaced)

| # | Gap (rejected) | Spec | In the target profiles? | Who produces it | Fix |
|---|---|---|---|---|---|
| 1 | **Field pictures (PAFF)** — `slice.rs:153` `field_pic_flag=1` | §7.3.3, §8.2.1, §8.2.4.2.5, §8.4.2.1 (field refs), §8.7 | Main, High, High 10 (not CB) | broadcast 1080i (DVB/ATSC/Blu-ray), hardware/broadcast encoders; x264 does not (MBAFF only) | L (with #2) |
| 2 | **MBAFF frames** — `slice.rs:156` | §7.3.4 (mb_field_decoding_flag), §6.4.12.2 neighbours, §8.7.1 deblocking | Main, High, High 10 | `x264 --tff/--bff/--interlaced` (ffmpeg `-flags +ilme+ildct`), broadcast | L (shares field DPB/MC with #1) |
| 3 | Picture size above level 5.2 — `sps.rs:225/237`, `MAX_FRAME_MBS` 36 864 | Table A-1 (levels 6, 6.1, 6.2: 139 264 MBs, 8K) | yes, any profile at level 6.x | x264/ffmpeg at 8K (`--level 6.2`), rare | S |
| 4 | Several pictures in one packet (API) — `mod.rs:281` | — (API, not syntax) | n/a | only if a caller hands raw multi-AU buffers; the CLI framer splits AUs | S (doc) |
| — | FMO / ASO, redundant pictures, data partitioning, SP/SI | §7.3.2.2, Annex A.2.1/A.2.3 | Baseline / Extended only, **not** CB/Main/High | essentially none | out of scope (document) |
| — | 4:2:2, 4:4:4, >10-bit, lossless bypass, separate planes | A.2.6–A.2.11 | High 4:2:2 / 4:4:4 only | x264 `--output-csp i422/i444` | out of scope (document) |

Suite: ITU-T H.264.1 AVCv1 + FRExt + Professional (High 10) zips, 110/190; all 80
failures are #1/#2 (75 streams: 44 field, 31 MBAFF) or FMO/SP (5, out of
scope). So **75 official Main/High streams fail**. The 4:2:2/4:4:4, MVC and SVC
zips are not fetched. Browser run leaves out interlaced and 4:0:0 streams.
The 8 x264 supplement streams are I/P-only High tools (no B-pyramid, no
interlace). Not run: x264 `--interlaced`, ffmpeg PAFF, level 6 sizes.

## HEVC (target: Main, Main 10, Main Still Picture)

| # | Gap | Spec | In target? | Who produces it | Fix |
|---|---|---|---|---|---|
| — | No gap found for the target profiles. 4:0:0/4:2:2/4:4:4 and `sps_range_extension` are rejected (RExt profiles), other SPS/PPS extensions are ignored as §7.4.3.2.1 requires, reserved NAL types and `nuh_layer_id>0` units are ignored (§7.4.2.2), caps at level 6.2. | | | | |
| 5 | Range extensions (Main 12, Main 4:2:2 10, Main 4:4:4, ...) | §7.3.2.2.2, A.3.5 | no (RExt) | `x265 --output-csp i422/i444`, Main 12 | L (out of target) |

Suite: JCT-VC HEVC_v1 147/147 (whole set). Not run: the RExt, MV-HEVC, SHVC,
SCC and 3D-HEVC conformance sets (all outside the target profiles). No x265
real-world stream corpus is in the record (field-coded x265 `--interlace`
streams are progressive-decodable pictures with an SEI; not tested).

## VVC (target: Main 10, Main 10 Still Picture)

| # | Gap | Spec | In target? | Who produces it | Fix |
|---|---|---|---|---|---|
| 6 | Reserved / unspecified NAL unit types are an error, not ignored — `decoder.rs:213` and the `_ =>` arm at `decoder.rs:736` (types 4–6, 11, 26–31) | §7.4.2.2 ("decoders shall ignore") | yes | none today; future extensions and system-layer unspecified types (28–31) | S |
| 7 | `nuh_layer_id > 0` units are an error (`decoder.rs:186/678`), not dropped | §7.4.2.2, C.6 (base-layer extraction) | multilayer streams are Multilayer Main 10; a Main 10 decoder may be fed one | VVenC: no; VTM multilayer configs | S (drop them, as HEVC does) |
| 8 | SPS with `sps_video_parameter_set_id > 0` rejected (`decoder_headers/sps.rs:91`) | §7.4.3.4 | allowed in single-layer Main 10 streams | rare (VTM with VPS on; VPS_C_2 is not in the record) | S |
| 9 | General profile other than 1/65 rejected (`ptl.rs:17`) | A.3 | Multilayer Main 10 (17/81) out of target | — | out of scope |
| — | 4:2:2/4:4:4, palette, ACT, >10-bit | A.3.2–A.3.4 | Main 4:4:4 10 / RExt only | VVenC 4:4:4 option | out of scope |

Suite: 226/226 of the 283 H.266.1 first-edition streams (all Main 10 / Main 10
Still Picture 4:2:0 and 4:0:0 streams except HRD_B_2, which VTM fails, and
VPS_C_2, never probed). Not run: the second-edition (RExt/VVC v2, 12/16-bit,
4:4:4 operation-range) streams and the multilayer streams. Plus 138 own
VTM/VVenC tool streams (8-bit).

## VP9 (target: profile 0 and 2; 1 and 3 optional)

| # | Gap | Spec | In target? | Who produces it | Fix |
|---|---|---|---|---|---|
| 10 | Profiles 1 and 3 (4:4:4, 4:2:2, 4:4:0; sRGB) — `decoder/header.rs:374` | VP9 §6.2, §7.2 | optional | `libvpx --profile=1` for 4:4:4 screen share (WebRTC screen-content), rare | M |
| — | No other rejection found in the decoder: sizes to level 6.2 area, odd sizes, resizing, intra-only, superframes, 10/12-bit all decode (the "odd frame dimensions" / "packet without a shown frame" errors are only in the 8-bit `Picture` API; the CLI and browser paths use the general output API). The profile-0-only checks in `superframe.rs`/`extract.rs` are the SVC extractor, not the decoder. | | | | |

Suite: libvpx v1.16.0 vectors, 305/305 (303 profile 0, one 10-bit, one 12-bit).
VP9 has no ITU/ISO conformance suite; the libvpx vectors are the de facto one.
Not run: the profile 1/3 vectors (`vp91-*`, `vp93-*`), and the Argon VP9 streams
(a commercial coverage suite; not available to us). Only **2** high-bit-depth
vectors are in the record, so profile 2 coverage is thin.

**Profiles 1 and 3: sizing (2026-11-27), superseded the same day:** the non-4:2:0 chroma agent landed them on a shared chroma-format picture (124deac4b). The sizing as written then: They do
not fit in an hour, because the shared sample type has no chroma format:
`nc_common::Picture` is 4:2:0 only (`ChromaFormat::Yuv420`, plane sizes from
`PlaneIndex::is_subsampled`), and about 74 files build pictures through it. The
VP9 decoder itself is 4:2:0 in the loop filter masks, the chroma block and
transform sizes, and sub-8x8 chroma motion. Plan, about two days (M):

1. **Shared picture subsampling (about a day, all codecs benefit).** `Picture` and
   its recycler, `ShownPicture`/`DecodedPicture` carry `(ss_x, ss_y)`; plane
   geometry and borders come from it; 4:2:0 stays the default constructor. The
   CLI's raw/Y4M writers and the browser module's `decoded_i420*` write the planes
   at their own size (the browser display path converts 4:4:4/4:2:2 to its 4:2:0
   display, or refuses them with a clear error).
2. **VP9 header and block layer (half a day).** `read_profile` takes 1 and 3;
   `read_color_config` reads `subsampling_x/y` and the reserved bit (sRGB means
   4:4:4, VP9 §6.2); chroma block size from `ss_size_lookup` (4:2:2 and 4:4:0
   make non-square chroma blocks), `get_uv_tx_size` with both subsampling
   factors; sub-8x8 chroma MVs per 4x4 in 4:4:4, averaged by pairs in 4:2:2 and
   4:4:0 (`average_split_mvs`); intra edge availability and MC positions with
   separate horizontal and vertical subsampling.
3. **Loop filter (half a day).** 4:4:4 chroma uses the luma masks
   (`filter_block_plane_ss00`); 4:2:2 and 4:4:0 need libvpx's generic per-block
   path (`filter_block_plane_non420`), which our 4:2:0 mask builder does not
   cover.
4. **Vectors.** Add the libvpx `vp91-2-04-yuv{422,440,444}` and
   `vp93-2-20-{10,12}bit-yuv{422,440,444}` vectors to the manifest, bit-exact
   against vpxdec. Profile 3 then costs about two hours on top (the decoder is
   sample-generic already).

## AV2 (target: AVM v1.0.0 main profile as the draft defines it)

| # | Gap | Spec (AVM v1.0.0 / dav2d) | In target? | Who produces it | Fix |
|---|---|---|---|---|---|
| 11 | **10-bit** (`general/sequence.rs:465`) | seq header `bit_depth` | yes (AVM's main profile takes 8- and 10-bit 4:2:0, as AV1 Main does; to confirm against the draft text) | `avmenc --bit-depth=10`, any 10-bit input; the 13th dav2d test vector | L (sample-generic port, as done for AV1/VP9/VVC) |
| 12 | **Several tiles** (`general/block.rs:261`) and several tile groups (`general/decoder.rs:407`) | tile_info, OBU_TILE_GROUP | yes | `avmenc --tile-columns/--tile-rows`; large-resolution presets | M |
| 13 | References of another size (reference scaling) — `general/decoder.rs:515` | frame size override + scaled MC | yes | `avmenc --resize-mode`, RTC resolution switching, spatial SVC | M |
| 14 | 4:0:0 (monochrome) — `general/sequence.rs:462` | seq header chroma format | likely (AV1 Main allows it; to confirm against the draft) | `avmenc --monochrome` | S–M |
| 15 | Multi-frame headers (`general/frame_header.rs:571`) and bridge frames (`general/decoder.rs:427`) | MFH OBU, bridge frame OBU | yes (normative tools) | not in avmenc's default configs; which options emit them is not checked | M |
| 16 | Wedge compound when motion fields are stored (`general/block/motion_resolution.rs:471`) | compound wedge + TMVP store | yes | possible wherever compound wedge is on (random access); not met in the 30-stream corpus | S |
| 17 | Embedded / extended layers (`general/decoder.rs:334/337`) | layer syntax | scalable profiles | avmenc SVC configs | M |
| 18 | Bilinear chroma phase of odd eighth-sample vectors (handoff "Known gap") | MC filter | yes | not met in corpus | S |

Suite: none exists (searched 2026-11-15). Checked: 30 streams (12 dav2d test-data
streams + 18 avmenc v1.0.0 streams we made, all 640x360 or smaller, single tile,
8-bit), bit-exact against avmdec and dav2d. This is a smoke corpus, not
conformance.

## Conformance suites: run vs not run

| Codec | Run (record 2026-11-23) | Kind | Official/standard suites NOT run |
|---|---|---|---|
| H.264 | H.264.1 AVCv1 + FRExt + Professional High 10: 110/190 | official ITU-T | H.264.1 Hi422 / Hi444 / CAVLC444 / MVC / SVC zips; the 75 interlaced streams fail |
| HEVC | HEVC_v1 (H.265.1): 147/147 | official | RExt, SCC, MV-HEVC, SHVC, 3D sets (outside target) |
| VVC | H.266.1 first edition: 226/283 (all of the target profiles but 2) | official | H.266.1 v2 (RExt) streams, multilayer streams |
| AV1 | libaom v3.15.3 test vectors 242/242 | encoder test vectors | **Argon AV1 conformance streams** (the coverage suite AOM publishes for AV1 decoder conformance): not run; the AV1 rejections the owner found are what Argon would have exposed |
| VP9 | libvpx v1.16.0 vectors 305/305 | encoder test vectors (the de facto suite; no official one) | profile 1/3 vectors; Argon VP9 (commercial) |
| AV2 | 30 foreign streams | ad-hoc corpus | none exists yet |

Also not run anywhere: a real-world encoder corpus per codec (x264/ffmpeg
interlaced, x265 defaults incl. B-pyramid/WPP/tiles at 1080p+, VVenC presets,
libvpx profile 2 with `--row-mt`, avmenc 10-bit / tiles / resize). A small
generated corpus per codec from the common encoders' default and common options
would have caught the AV1 case; it is the cheapest next check.

## Claim audit

Every decoder-compliance claim found (repo at `02975de9e`, published site
cribbinsb.github.io/nanocodec, GitHub releases v0.3.0–v0.6.0; there is no v0.7.0
release although CHANGELOG has a 0.7.0 section). Verdicts: **A** accurate,
**O** overstated, **W** wrong/stale. Claims about our *encoders'* streams being
conformant (README:28, STATUS:9/84/788) are about encoding and are not judged here.

### Public (README, FAQ, site, crates.io docs, releases)

| Where | Claim (quoted) | Verdict | Corrected wording |
|---|---|---|---|
| README.md:29-30 | "and the libaom, libvpx and AV2 test vectors" | O | "...the libaom and libvpx encoder test vectors (AV1 has no official suite in our record; Argon not run), and 30 AV2 streams (12 dav2d vectors, 18 we made with avmenc)" |
| README.md:30-32 | "They do not yet decode every valid stream: some AV1 tools ... and interlaced H.264 are still missing" | O (incomplete) | add: "AV2 10-bit, multi-tile and reference scaling; VP9 4:4:4 (profiles 1/3)" |
| README.md:29 | "VVC's 4:2:0 streams (226/226)" | A (imprecise) | "VVC's Main 10 streams (226/226; 4:2:0 and 4:0:0)" |
| docs/FAQ.md:223 | "Our decoders also decode other encoders' streams, 8-bit and 10-bit" | O | "...8-bit and 10-bit (AV2: 8-bit only); see the gaps below" |
| docs/FAQ.md:229 | "AV1 (Main, 8- and 10-bit): 242/242 libaom v3.15.3 test vectors" | A as a count, O as a profile claim | add: "encoder test vectors, not a conformance suite; valid Main streams with quantizer matrices, delta LF, some segment features, switch frames, odd sizes or mixed lossless are rejected" |
| docs/FAQ.md:236-238 | "(High and High 10, frames only): 110 of the 190 ... the 80 others use field pictures, MBAFF, FMO or SP/SI" | A | add "(75 of the 80 are interlaced Main/High streams)" |
| docs/FAQ.md:240-241 | AV2 "all 30 of 30 foreign test streams" | A as stated | add "8-bit, single tile, ≤ 640x360; no AV2 conformance suite exists" |
| docs/FAQ.md:264-267 "What is not done" | lists H.264 interlace, FMO/SP, VP9 1/3, VVC 4:2:2/4:4:4, AV2 corpus | O (incomplete) | add AV1 tool gaps (as README), AV2 10-bit / tiles / reference scaling / monochrome |
| CHANGELOG.md:142-144 + GitHub v0.6.0 release | "Conformance (official streams decoded bit-exactly ...): HEVC 147/147, VP9 305/305, AV1 242/242 ..." | O | "official" is wrong for VP9 and AV1: "conformance and test-vector streams decoded bit-exactly: HEVC 147/147 (official), VVC 226/226 (official), H.264 110/190 (official), VP9 305/305 and AV1 242/242 (libvpx/libaom test vectors)" |
| CHANGELOG.md:331-338 (0.2.0) | "VVC official 224/226 ... AV2 30/30 foreign" | A for its date | leave (history) |
| web/docs/cli.html:94-98 (live on the site) | "H.264 (Constrained High), HEVC (Main), AV1 (Main, 8-bit 4:2:0) and VP9 (profile 0) pass ...; VVC ... not a full Main-profile decoder; AV2 ... 14 of 30 ... All are 8-bit 4:2:0 only." | W (stale on every codec) | replace with the FAQ conformance summary (or a link to it) |
| web/docs/cli.html:28, nc-cli/src/help.rs:373, web/docs/data/cli.json:68 | "decodes our streams" / "Decode one of our streams with our own subset decoder" | W (understates) | "Decode a stream (ours or another encoder's, within the supported profiles: FAQ#conformance)" |
| web/docs/c-api.html:29 (site) | "our decoders for the same four" | check | the C API exposes 4 of 6 decoders; state which |
| crates/nanocodec/src/lib.rs:12-14 (docs.rs front page) | "Each decoder handles the subset its encoder emits." | W (understates) | "The decoders also decode other encoders' streams of the main profiles; gaps: FAQ#conformance" |
| crates/nc-{h264,hevc,vvc,av1,av2}/README.md:3 (crates.io) | "decoder for the streams it emits" | W (understates) | as above, per codec |
| crates/nc-vp9/README.md:3 | "a VP9 Profile 0 decoder" | W | "profile 0 and 2 decoder" |
| crates/nc-cli/README.md:70-80 | "subset decoders ... accept the subset produced by our encoder"; "tightly packed 8-bit I420" | W | general decoders; >8-bit output is 16-bit LE |
| crates/nc-hevc/src/lib.rs:11-13 (docs.rs) | "complete HEVC Main / Main Still Picture profile decoder (any conforming 8-bit 4:2:0 stream ... 136 Main streams)" | A in substance, stale | "Main, Main 10 and Main Still Picture (147/147 HEVC_v1)" |
| crates/nc-vp9/src/lib.rs:1-3, decoder/mod.rs:1-2 | "any conforming profile 0 or profile 2 stream" | O (mild) | "...checked on the 305 libvpx vectors (no official VP9 suite)"; no known gap behind it |
| crates/nc-av1/src/lib.rs:10-11 | "aims at full Main profile conformance" | A (an aim) | keep, add "known gaps: subset.rs" |
| crates/nc-av2/src/general.rs:1-4 | "8-bit 4:2:0 streams of any encoder that follows AVM v1.0.0" | O | "...single-tile 8-bit 4:2:0 streams without reference scaling, MFH or bridge frames"; also drop the stale "blocks still go through the subset" |
| crates/nc-av2/src/lib.rs:6-9 | "its blocks still go through the subset our encoder writes" | W (stale, understates) | general block layer decodes the 30 foreign streams |
| tools/bench/pages/decoders.js:129 (new Decoders page, 26e34034d) | "every decoder passes the landing conformance check ... and its conformance suite" | O | "...and is checked on its codec's conformance or test-vector streams (FAQ#conformance): H.264 passes 110 of 190 (interlaced streams are not supported); AV1 and VP9 are checked on libaom/libvpx test vectors; AV2 has no suite" |
| web/index.html:34 (site) | "every coded version is decoded here by nanocodec's Rust decoder in WebAssembly" | A (demo streams are progressive 8-bit) | — |

### Internal docs

| Where | Claim | Verdict | Fix |
|---|---|---|---|
| docs/ARCHITECTURE.md:395-402 | "listed in the README's Decoders table (H.264 Constrained High, HEVC Main and Main 10, AV1 Main 8-bit 4:2:0, VP9 profile 0; VVC 9 official 8-bit streams; AV2 our own streams only)" | W (table gone, every item stale) | point at FAQ#conformance and this file |
| docs/ARCHITECTURE.md:323-328 | "8-bit High conformance suite"; VVC/AV2 "our emitted subset" | W | update |
| docs/STATUS.md:455-459 | "The HEVC, AV1 and VP9 ones are subset decoders, not general-purpose decoders" | W (contradicts FAQ) | delete / replace |
| docs/STATUS.md:183 | "9/9 official 8-bit streams pass" (VVC) | W (stale; 226/226) | update |
| docs/handoff/h264.md:511-512 | "`decode_nal` ... (any stream ...)" | O | "any progressive stream of the supported profiles" |
| docs/handoff/hevc.md:814-815 | "decodes any Main stream" | A (no gap found) | — |
| docs/handoff/av1-decoder.md:8 | "decodes any AV1 Main-profile 8-bit 4:2:0 stream" | O (goal stated as fact) | mark as goal; list gaps |
| docs/handoff/av1-decoder.md:10 | "(Constrained High, 65/65)" | W (old) | 110/190 |
| docs/handoff/vp9.md:126-129 | "decodes any VP9 profile 0 or 2 stream" | O (mild) | "...all 305 libvpx vectors" |
| crates/nc-vvc/src/decoder.rs:1 | "Bounded decoder for our H.266 intra/inter subset" | W (understates) | general Main 10 decoder with an own-stream fast path |
| crates/nc-av1/src/decoder/sequence.rs:1 | "8-bit 4:2:0 Main profile sequence header" | W (10-bit too) | update |
| web/README.md:420-426 | "all three codecs"; "independent emitted-subset decoders" | W (old) | update |

### Gaps known to readers vs not

Documented somewhere public: H.264 field/MBAFF/FMO/SP-SI; H.264 4:2:2/4:4:4
suites not fetched; VP9 profiles 1/3; VVC 4:2:2/4:4:4, multilayer, palette, ACT;
AV1 quantizer matrices, loop-filter deltas, "some segment features" (README
since today); AV2 "30 foreign streams only" and 8-bit scope (crate docs only).

**Not stated in any public claim:**
- AV1: switch frames, odd frame sizes, mixed lossless segments (the code's gap list
  in `nc-av1/src/decoder/subset.rs` has them; README names only three tools).
- AV1: no Argon run; the libaom vectors are not a conformance suite.
- AV2: 10-bit (FAQ:223 says the decoders do 10-bit), several tiles / tile groups,
  reference scaling, monochrome, multi-frame headers, bridge frames, wedge with
  motion fields, layers.
- VVC: reserved NAL types and `nuh_layer_id > 0` rejected instead of ignored;
  `sps_video_parameter_set_id > 0` rejected.
- H.264: sizes above level 5.2 rejected.
- Browser: 4:0:0 and interlaced streams are left out of every browser run, so
  monochrome decoding in WASM is unchecked.
- That the AV2 30/30 is not in the conformance record the FAQ cites beside it.
