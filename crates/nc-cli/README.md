# Native encoder and decoder

Build with `cargo build --release -p nc-cli`. The executable is `target/release/nanocodec`.

Encode raw 8-bit I420 and optionally save the encoder reconstruction:

```sh
nanocodec encode --codec av1 --input input.yuv --width 640 --height 480 \
  --fps 30000/1001 --qp 190 --frames 90 --intra-period 30 \
  --output video.ivf --recon encoder.yuv
```

`--input-format yv12|nv12` reads tightly packed YV12 or NV12 frames instead
(camera layouts); they go through `Encoder::encode_raw`, the library entry
point for borrowed I420/YV12/NV12 frames with any stride (`nc_common::RawFrame`),
and code exactly the same stream as the equivalent I420 input.

`--colour bt709|bt601|bt601-625|P,T,M` and `--range full|limited` signal the
colour description and sample range (H.264/HEVC VUI, AV1 `color_config`);
without them nothing is signalled. They do not change the coded pictures.

`--timestamps in.txt` gives each input frame its capture time (mkvmerge
"timestamp format v2": a header line, then milliseconds per frame); rate
control then budgets every frame for its real interval, so variable frame
rates and dropped camera frames keep the target bitrate in time. AV1 IVF
packets carry those times (microsecond time base); `--timestamps-out out.txt`
writes the coded frames' times in the same format for muxing H.264/HEVC
(`mkvmerge --timestamps 0:out.txt`). `--timing nominal|constant|variable`
chooses the header timing info (timestamped input defaults to variable).

`--intra-refresh N` replaces periodic key frames by an intra column sweeping
across the picture every N frames (flat structures): a decoder that lost a
frame recovers within two sweeps, without a key frame burst.

`--film-grain [STRENGTH]` (AV1 only; other codecs refuse it) turns on
experimental film grain synthesis: the encoder measures the source's own noise
on flat areas (capped at what coding removed there) and signals AV1 film grain
parameters, and decoders lay matching grain over the shown picture. It is off
by default. STRENGTH is the noise level relative to the measured source noise
(default 1, up to 4); clean content stays grain-free (`grain.min_strength`).
It is shorthand for
`--param grain.enable=1` (plus `grain.strength`); the other `grain.*`
parameters, such as `grain.chroma=1` for chroma grain, still apply. Grain
lowers PSNR and VMAF by construction, and it restores noise, not lost texture
(foliage and spray are not noise); decoded output then differs from `--recon`, which stays
grain-free. `decode --film-grain 0` shows AV1 frames grain-free too (as
`aomdec --skip-film-grain`); the evaluation harness scores grain streams that
way.

Add `--preset av1-fast` for the optional AV1 speed/quality tradeoff used by
browser Fast. Held-out VMAF BD-rate costs 3.45% versus the current default;
the default remains the quality choice.

Enable single-pass realtime bitrate control instead of fixed quantization:

```sh
nanocodec encode --codec av1 --input input.yuv --width 640 --height 480 \
  --fps 15 --frames 150 --intra-period 30 --bitrate 200 \
  --buffer-ms 1000 --peak-bitrate 400 --output video.ivf --rate-log frames.csv
```

Bitrate values are decimal kbps. `--qp` is optional in bitrate mode and seeds the controller; omitted seeds are 32 for H.264/HEVC and 180 for AV1. The controller selects each frame's quantizer, budgets keyframes separately and uses completed frame sizes for feedback without future lookahead. `--buffer-ms` sets the modeled VBV duration (default 1000 ms). Key frames may overshoot it: the first key may take up to about 1.4 buffers (`--param rate.recode_overflow`, the re-code ceiling in buffers) and its extra bits are repaid over `--param rate.key_repay_seconds` (default 10 s; 5 s lands short clips on target for about 1% more bits); optional `--peak-bitrate` sets a best-effort peak target. These controls do not pad frames. By default they never drop frames either, so unreachable targets at a quantizer clamp can still miss their budget; `--frame-drop` skips inter frames that would overflow the buffer even at the maximum quantizer, in either the peak-rate or the target-rate bucket (so the long-term rate holds even with the default peak of twice the target). Dropped frames leave nothing in the stream, the reconstruction or `--timestamps-out`, and appear as `Dropped` in the rate log.

Optional `--min-qp` and `--max-qp` bound the codec's native quantizer. `--max-qp-step` smooths inter-to-inter quantizer changes. Keyframes choose their own quantizer from the reserved key budget, within the same bounds, so a keyframe does not force the following P frames to use its quality level. Bounds apply to actual key and P quantizers; bitrate control owns the keyframe budget instead of applying the fixed-QP keyframe boost. `--rate-log` writes per-frame planned/actual quantizer, payload bytes, buffer fullness, running bitrate and overflow count as CSV. Bitrate feedback counts compressed access-unit payload, excluding IVF container headers.

Omitting `--bitrate` preserves fixed-QP encoding and requires `--qp`; other rate-control options require bitrate mode.

Decode with nanocodec's independent Rust subset decoders:

```sh
nanocodec decode --codec av1 --input video.ivf --output decoded.yuv
nanocodec decode --codec h264 --input video.264 --output decoded.yuv
nanocodec decode --codec hevc --input video.hevc --output decoded.yuv
```

AV1 input uses IVF; H.264 and HEVC use Annex-B elementary streams. These commands accept the subset produced by our encoder and browser demo, including key/P pictures and repeated parameter headers. They do not substitute external decoder libraries. Unsupported tools or malformed/truncated data return an error and a failing exit status.

Output is tightly packed 8-bit I420 (Y, U, V), with dimensions from the stream and no file header. IVF timestamps and any nonzero time base are accepted; decoding preserves packet order and does not resample frames. Input is streamed with a 64 MiB limit per compressed frame/NAL and at most 16 parameter NALs before a picture, so movie length does not determine memory use.

For streams encoded with `--recon`, verify exact reconstruction:

```sh
cmp encoder.yuv decoded.yuv
```

`nanocodec params` lists encoder controls. Decoder controls are limited to codec, input and output; no encoder settings are needed to reconstruct a saved stream.

## H.266 / VVC

The native CLI accepts `--codec vvc` (alias `h266`) for encode and decode:

```sh
nanocodec encode --codec vvc --input input.yuv --width 640 --height 480 \
  --fps 30 --qp 32 --frames 30 --output output.vvc
nanocodec decode --codec vvc --input output.vvc --output /dev/null
```

This subset uses IDR/P pictures, 8–32 quadtree blocks, planar/DC and all 65
square-block angular intra directions, independent chroma prediction, forward inter (skip/merge/AMVP), full DCT-II residuals,
deblocking and edge-offset SAO. The shared engine owns motion search, partition
planning, intra/inter selection and rate control. It accepts even sizes up to
4096 per axis and QP 0–63; the default maximum block is 32. Colour VUI and fixed
QP keyframe offsets are supported. Temporal layers, LTR, intra refresh and HRD
timing are not yet supported. Output is Annex B.

VVC considers every angular direction using shared DSP SATD and signalling
cost, then uses the shared engine's shortlist helper before transform RD.
`--param vvc.angular_candidates=8` is the default: retain the four
core modes plus eight ranked angular alternatives. `0` keeps the previous four
modes; `63` evaluates all 67 modes with full RD.

Key-frame partitions are planned from a luma-only DC transform cost on the
source (as HEVC's); the final leaf direction search uses `vvc.angular_candidates`.

VVC chroma selection compares all five ordinary candidates using reconstructed
distortion and estimated bits. `--param vvc.chroma_search=0` retains derived-only
chroma for comparisons; the default is `1`. CCLM is not implemented.
