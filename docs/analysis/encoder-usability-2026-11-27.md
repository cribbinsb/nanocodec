# Encoder usability audit: identification, signalling, levels, containers (2026-11-27)

Owner question: do our encoders write proper SEI (or the codec's equivalent)
and the basic stream information that x264 and the other reference encoders
write, so a player, a muxer or a person can tell what a stream is and play
it correctly?

**Method.** One 30-frame 320x180 30 fps clip (ffmpeg `testsrc2`), coded at
300 kbps CBR by each of our six encoders (`main` d06e3135e, preset medium)
and by each reference encoder in its real-time mode (x264 0.165 veryfast
zerolatency, x265 4.2 veryfast zerolatency, VVenC 1.14 faster, aomenc `--rt
--cpu-used=10`, SvtAv1EncApp preset 12 `--rtc 1`, vpxenc VP9 `--rt
--cpu-used=9`, avmenc `--cpu-used=9`). Headers dumped with ffmpeg's
`trace_headers` bitstream filter, `ffprobe -show_streams/-show_frames`,
`mediainfo`; remux checked with `ffmpeg -i S -c copy S.mp4`. A second run
used `--colour bt709 --range full --timing constant`, a third
`--intra-refresh 10`, a fourth a still (repeated-frame) source.

## Summary, ranked by user impact

| # | Gap (before this change) | Codecs | Impact | Status |
|---|---|---|---|---|
| 1 | No timing in the headers by default: raw Annex-B HEVC/VVC streams had no frame rate at all (`ffprobe` r_frame_rate 1200000/1; `ffmpeg -i x.265 -c copy x.mp4` gave 25 fps, 1.2 s for a 1 s clip). AV1 had no `timing_info` (IVF carried the rate). | HEVC, VVC, AV1 | High: wrong playback speed after the commonest remux | **Fixed** |
| 2 | Raw Annex-B output plus frame holds: a held (repeated) input frame is not coded, and Annex-B has no timestamps, so a stream of a repeat-frame source plays fast once remuxed at the nominal rate (still clip: 18 of 30 frames coded). IVF (AV1/VP9/AV2) carries per-frame pts and is fine. | H.264, HEVC, VVC | High for 15-in-30 fps cameras and frozen sources written by the CLI | **Fixed** (round 2): MP4/MKV/WebM output keeps each frame's time; Annex-B/IVF unchanged |
| 3 | Fixed level: VVC always signalled level 6.0 (`general_level_idc` 96) and AV1 single-layer streams `seq_level_idx` 31 ("no constraints"), whatever the size and rate. Hardware and browser decoders size (or refuse) by level. VVenC: 2.0, libaom/SVT-AV1: 2.0 for the same clip. | VVC, AV1 | Medium: device compatibility | **Fixed** (level from coded size and rate) |
| 4 | Level ignores the bit rate (MaxBR / CPB, AV1 MaxBitrate): a small picture at a high rate (e.g. 320x180 at 3 Mbps) signals a level whose bit-rate limit it exceeds. x264 raises the level for the VBV. | H.264, HEVC, VVC, AV1 | Low at our ladder (300-1200 kbps), real at high rates | **Fixed** (round 2) |
| 5 | No recovery-point SEI (H.264/HEVC) or GDR picture (VVC) with `--intra-refresh`: a decoder joining mid-stream has no marked entry point (x264 `--intra-refresh` writes recovery-point SEI). | H.264, HEVC, VVC | Medium for intra-refresh users only (intra refresh is low priority) | **Fixed**: H.264/HEVC recovery-point SEI, VVC GDR pictures (VVdeC skips the first sweep, see below) |
| 6 | Encoder identification is not shown by `mediainfo` (it decodes only the x264/x265 UUIDs) nor by `ffprobe` as text; it is present and readable (`strings`, ffprobe frame side data "User Data Unregistered SEI"). | H.264, HEVC, VVC | Low | As designed |
| 7 | AV1 metadata OBU type 6 (unregistered user private, AV1 §6.7.1) makes dav1d log "Unknown Metadata OBU type 6" once per key frame in ffmpeg-based tools. libaom and SVT-AV1 write no identification at all. | AV1, AV2 | Low (log noise) | **Fixed** (round 2): first key frame only |
| 8 | AV2 rejects `--colour`/`--range` ("not yet supported"); ffprobe does not know AV2 yet. | AV2 | Low (no AV2 players) | **Fixed** (round 2): content interpretation OBU (colour, range, timing) and Annex A level |
| 9 | No HRD (`--nal-hrd`), buffering-period/picture-timing SEI, AUD | all | Low: x264/x265 write none by default; needed for broadcast TS only | Not needed now |
| 10 | No MP4/MKV output | all | Medium (see 2) | **Fixed** (round 2), not AV2 (no mapping) and not VVC in Matroska (FFmpeg 8 cannot read it) |
| 11 | Mastering display / content light | all | None: 8-bit SDR encoders; HDR metadata would be wrong to write | Not applicable |

## Per codec, ours vs the reference encoder (default streams, after the fixes)

### H.264 (ours vs x264 0.165)

| Item | nanocodec | x264 |
|---|---|---|
| Identification | `user_data_unregistered` SEI on every IDR, own UUID 72c2229f-…, "nanocodec 0.7.0 - H.264 encoder - options: preset=medium rc=cbr bitrate=… keyint=… layers=…" (~130 B) | same SEI, x264 UUID, full option string (~600 B), first IDR only |
| Profile / level | High, level from MB count and MB rate (1.3 here) | High@1.3 (also checks VBV) |
| Cropping | `frame_cropping` for non-multiple-of-16 sizes | same |
| VUI timing | `num_units_in_tick`=1, `time_scale`=2×fps; `fixed_frame_rate_flag` 1 unless frames may be held | same, fixed 1 |
| Colour / range | absent by default; `--colour`, `--range` | absent by default; `--colorprim/--transfer/--colormatrix/--range` |
| Low-latency hints | `bitstream_restriction`: `max_num_reorder_frames` 0, `max_dec_frame_buffering` = refs | same |
| SAR | not written (square) | not written |
| HRD / AUD | none | none by default (`--nal-hrd`, `--aud`) |
| Intra refresh | no recovery-point SEI | recovery-point SEI |

### HEVC (ours vs x265 4.2)

| Item | nanocodec | x265 |
|---|---|---|
| Identification | `user_data_unregistered` SEI (prefix) on every IRAP | same, x265 UUID, first IRAP |
| Profile / tier / level | Main, Main tier, level from coded size and rate (2 here) | Main@L2@Main |
| Conformance window | yes (184 → 180) | yes |
| VUI | **now always**: timing (`poc_proportional_to_timing` when constant rate); colour when given | timing |
| Reorder | `sps_max_num_reorder_pics` 0, `max_latency_increase_plus1` 0 | same reorder 0 |
| HRD | none | none by default |

### VVC (ours vs VVenC 1.14)

| Item | nanocodec | VVenC |
|---|---|---|
| Identification | `user_data_unregistered` SEI (H.274) on every IRAP | none by default |
| Profile / level | Main 10, **level now from coded size and rate** (2.0 here; was 6.0) | Main 10, 2.0 |
| Timing | **now always**: `general_timing_hrd_parameters` (no NAL/VCL HRD), `fixed_pic_rate` when constant | same |
| Colour | VUI when given | none by default |
| Conformance window, DPB | yes; reorder 0 | yes; random access GOP reorders (has_b_frames 5) |
| Intra refresh | no GDR pictures | GDR enabled in its low-delay configs |

### AV1 (ours vs libaom 3.x `--rt`, SVT-AV1 `--rtc`)

| Item | nanocodec | libaom / SVT-AV1 |
|---|---|---|
| Identification | metadata OBU type 6 (unregistered user private) on every key | none |
| `seq_level_idx` | **now the Annex A.3 level** of the size and rate per operating point (2.0 here; was 31); 31 kept for fixed `base_q_idx` < 32, pictures under 128x128 and random-decision test streams | 2.0 |
| `timing_info` | **now always** (`equal_picture_interval` when constant) | none (IVF carries the rate) |
| `color_config` | `color_description_present_flag` when given | same |
| Container | IVF with per-frame pts | IVF |

### VP9 (ours vs libvpx)

| Item | nanocodec | libvpx |
|---|---|---|
| Identification | none (no syntax; container only) | none |
| `color_space` | BT.601 by default, BT.709 etc. with `--colour` | CS_UNKNOWN |
| Container | IVF, pts | IVF |

### AV2 (ours vs avmenc)

| Item | nanocodec | avmenc |
|---|---|---|
| Identification | metadata OBU (as AV1) | none |
| Colour | `--colour` refused | defaults |
| Players | ffprobe/mediainfo do not parse AV2 yet | same |

## What landed

- **Timing in every codec's headers** (gap 1). `TimingSignal::Nominal`, the
  default, is resolved by the engine (`nc_encoder::signalled_timing`):
  constant rate unless frames may be held, then variable. H.264 already
  behaved so (its streams are unchanged); HEVC now always writes a VUI with
  `vui_timing_info`, VVC `general_timing_hrd_parameters`, AV1 `timing_info`.
  About 10 bytes per key frame. After the change `ffmpeg -c copy` to MP4
  gives 30 fps and 1.0 s for the HEVC clip.
- **Levels from size and rate** (gap 3). One shared table
  (`nc_encoder::luma_level`, HEVC Table A.8 = VVC Tables A.1/A.2 for levels
  1-6.2) gives HEVC its `general_level_idc` (it had its own copy) and VVC
  its level in place of the fixed 6.0. AV1 single-layer streams use the
  Annex A.3 selection the spatial-SVC header already used, per operating
  point.

Header-only changes: BD numbers do not move; stream bytes do (about 10 B
per key for timing, 0 B for levels; the 320x180 probe: HEVC +12 B, VVC
+11 B, AV1 +6 B, pictures identical). Rate control charges the extra header
bytes, so CBR streams may differ by those bytes' worth.

**Verification.** New streams: ffprobe r_frame_rate 30/1 for H.264 (via
MP4), HEVC, VVC, AV1; `ffmpeg -c copy` to MP4 gives 30 fps and 1.0 s for
H.264, HEVC and VVC. Levels: 320x180@30 HEVC 2 / VVC 2.0 / AV1 2.0;
1920x1080@30 HEVC 4 / VVC 4.0 / AV1 4.0; AV1 with 3 temporal layers signals
2.0 on all three operating points. HM, VVdeC, aomdec, dav1d and our
decoders decode the new HEVC/VVC/AV1 streams bit-exactly to the encoder's
reconstruction. Unit tests of nc-encoder, H.264, HEVC, VVC, AV1 and the
CLI pass. `preset_identity.py` (387 streams): every H.264, VP9 and AV2
preset identical; every HEVC (fast faster medium slow slower veryslow),
VVC (same) and AV1 (veryfast faster fast medium slow slower, l2-fast
l2-medium l3-fast l3-medium) preset changed (headers).

VP9 has no timing or level syntax in the bitstream (container only). AV2
is not changed: its sequence header has no timing syntax in our writer,
and its level (31) also sets its tile limits, so a computed level needs its
own conformance pass against avmdec/dav2d.

## Open, with options

- **Gap 2 (held frames in Annex-B files).** The hold design (`hold.rs`)
  assumes a container carries timestamps; the CLI writes raw Annex-B. Options:
  (a) MP4 or Matroska output from the CLI with each frame's time (the
  cleanest; also fixes duration, seeking and "what codec is this" for every
  player); (b) code a held frame as a repeat picture in Annex-B codecs
  (an all-skip P picture, a few bytes, exact for H.264 with zero MV
  predictors; HEVC/VVC need a zero-vector merge choice); (c) turn holds off
  for Annex-B file output (changes benchmark streams). Workaround today:
  `--timestamps-out ts.txt` then `mkvmerge -o out.mkv --timestamps 0:ts.txt
  stream.264`.
- **Gap 4 (bit-rate-aware levels).** Needs the rate (peak) handed to the
  backends with the encoder-information message, and the level re-chosen at
  the next key on a rate change; the H.264 MV range must stay that of the
  size level (a higher level only widens what is allowed).
- **Gap 5 (intra-refresh entry points).** H.264/HEVC recovery-point SEI at
  each sweep start with `recovery_frame_cnt` = sweep length; VVC GDR NAL type
  with `sps_gdr_enabled_flag`. Conformance-checkable with JM/HM/VVdeC.
- **Gap 7 (AV1 identification metadata).** Conformant as is. Options: keep;
  write it on the first key frame only (as x264 does, one log line per
  stream); or drop it by default for AV1/AV2 as libaom and SVT-AV1 do. An
  ITU-T T.35 payload would need a registered country/provider code we do not
  have.

## Round 2 (coordinator decisions, same day)

**Container output (gap 2, 10).** `--output x.mp4|.m4v|.mkv|.webm` writes
that container (`crates/nc-cli/src/mux/`): MP4 with `mdat` then `moov`
(sample entries `avc3`, `hev1`, `vvi1`, `av01`, `vp09`; `stts` durations from
each coded frame's time, `stss` key frames, `colr` when signalled),
Matroska/WebM with one cluster per key frame, `Cues` and `Duration`. A held
frame lengthens the sample before it. Checked on a 60-frame 320x180 30 fps
clip and a 15-in-30 repeat clip (32 of 60 frames coded):

| | ffprobe (MP4 / MKV, WebM) | decode vs recon (ffmpeg) | headless Chromium |
|---|---|---|---|
| H.264 | 30 fps, 2.0 s; repeat clip 2.0 s | identical | MP4 plays, 2.0 s, seeks |
| HEVC | 30 fps, 2.0 s | identical | not supported by Chromium |
| VVC | MP4 30 fps, 2.0 s (MKV refused: FFmpeg 8 has no `V_MPEGI/ISO/VVC`) | identical | not supported |
| AV1 | 30 fps, 2.0 s | identical | MP4, MKV, WebM play, seek |
| VP9 | 30 fps, 2.0 s | identical | MP4, WebM play, seek |

Before, the repeat clip's raw H.264 had 18 frames for 1 s of input and
remuxed to 0.6 s. AV2 has no MP4/Matroska mapping yet (refused; IVF).

**Bit-rate-aware levels (gap 4).** The engine hands each backend the peak
rate and buffer of the rate control in force (`CodecBackend::set_stream_rate`,
`nc_encoder::StreamRate`), and the next key's headers signal a level that
covers them: H.264 `MaxBR`/`MaxCPB` (High, x1.25; Table A-1 now has levels
2, 4.1, 6.1, 6.2), HEVC/VVC Main-tier `MaxBR`/`MaxCPB` in the shared table,
AV1 and AV2 `MaxBitrate`. The H.264 motion-vector range stays that of the
size level, so the coded pictures do not depend on the rate. 320x180 at
3 Mbps: H.264 3.0, HEVC 3, VVC 3.0, AV1 3.0 (were 1.3, 2, 2.0, 2.0).

**Intra-refresh entry points (gap 5).** H.264 and HEVC write a
recovery-point SEI on each frame starting a sweep (`recovery_frame_cnt` /
`recovery_poc_cnt` = the sweep's length less one, in frame-index units, so
conservative under layered structures; exact_match 1). JM, HM, FFmpeg and
ours decode bit-exactly. VVC: GDR pictures were implemented and decoded
bit-exactly by FFmpeg and ours, but VVdeC (d261469, as VTM) withholds output
of the first GDR picture's recovering pictures even after an IDR
(`m_gdrRecovered`), dropping 9 of 60 frames when decoding from the start, so
they are not used; VVC has no recovery-point SEI. Left open. This is not
the spec's start-at-GDR rule: decoding began at the IDR picture (POC 0), so
the GDR picture's `NoOutputBeforeRecoveryFlag` is 0 (H.266 §8.1.1) and its
recovering pictures have `PictureOutputFlag` 1 (§8.1.2); VVdeC suppresses
them because no GDR has yet completed (`!m_gdrRecovered`), whatever the
flag. Using GDR would be conformant, but VVdeC-based players would skip a
sweep of frames after every stream start.

**VVC GDR, owner decision (later the same day): on.** With `--intra-refresh`
each VVC sweep starts with a GDR picture (`sps_gdr_enabled_flag` when the
encoder is created with intra refresh, `ph_recovery_poc_cnt` = the sweep's
length less one). The VVdeC behaviour is documented in the FAQ resilience
section and the `--intra-refresh` help; the conformance gate expects VVdeC's
frames to be the others' with one contiguous run (the first sweep) left out
(`nceval.conformance.withheld_sweep`), FFmpeg and ours bit-exact on every
frame. Intra refresh is off in every preset, so default streams do not
change.

**Identification metadata (gap 7).** AV1 and AV2 write the encoder
information metadata OBU with the first key frame only, and again after a
settings change such as a rate change (one dav1d log line per stream);
H.264/HEVC/VVC keep it on every IRAP.

**AV2 signalling (gap 8).** Every key frame carries a content
interpretation OBU (AVM `av2_write_content_interpretation_obu`): progressive
scan, the colour description and range when given, and the timing
(`--colour`/`--range` are no longer refused). The sequence header signals the
Annex A level of the coded size, rate and peak bit rate (AVM
`av2_level_defs`; 31 kept for fixed `base_q_idx` < 32, pictures under
128x128 and random-decision streams), e.g. 2.0 at 320x180@30. avmdec and our
decoder match the reconstruction for medium and fast, with and without
`--colour bt709 --range full` and `--timing constant`; dav2d matches after
its 8-row padding is cropped, as before.
