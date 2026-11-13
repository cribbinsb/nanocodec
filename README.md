# Browser codec comparison

The page is static HTML/CSS/ES modules. `nc-web` supplies Rust encoders and
independent Rust decoders compiled to `wasm32-unknown-unknown` with SIMD128.
There is no upload endpoint, analytics, server-side encoding, or runtime CDN.

Build and serve from the repository root:

```sh
tools/web/build.sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory web
```

The build requires Rust's `wasm32-unknown-unknown` target and wasm-bindgen CLI
**0.2.105**, matching the crate version in Cargo.lock. `WASM_BINDGEN` can name an
existing executable; `CARGO_TARGET_DIR` selects the build cache, and `BUILD_JOBS`
defaults to2. The build reserves8MiB of WASM stack for codec transform kernels.
Generated assets are placed in `web/pkg/` and can be hosted by any static server.
The build makes two modules: `web/pkg/` (SIMD128) and `web/pkg/relaxed/` (SIMD128
plus relaxed SIMD). Each worker loads the relaxed one when
`WebAssembly.validate` accepts a relaxed-SIMD probe; both decode bit-identically,
because nc-dsp only uses relaxed operations whose results are fully specified for
its inputs. Speed numbers and the benchmark (`tools/web/bench.py`) are in
`docs/handoff/wasm-perf.md`.
Use HTTPS for a remotely hosted camera demo. No cross-origin isolation headers
are needed: each worker has its own ordinary WASM memory.

## Controls and behavior

- **Sample clip / Test scene / Webcam / Video file:** the page opens comparing the
  default sample clip at the default settings, with no click needed (the hidden
  input video is muted and inline, which browsers allow to autoplay; if one still
  refuses, the notice asks for Start). Picking another clip starts it at once.
  Sample clips loop; local files stop at their end and must be playable by the
  browser. The input is fitted into the selected resolution with letterboxing.
- **Sample clips** are excerpts of at least 10 s (or the whole source where it is
  shorter) in `web/clips/`, grouped in the picker as
  People / Motion / Hard texture / Animation: the public-domain JCT-VC/Vidyo
  conferencing sequences (Johnny, the default, KristenAndSara, FourPeople,
  vidyo1/3/4), Netflix El Fuente and Chimera scenes (CC BY-NC-ND 4.0), SVT's 2006
  Multi Format set (its own terms; its PDF ships alongside), SVT's 2022 Natural
  Complexity suite (CC BY 4.0) and Tears of Steel / Big Buck Bunny (CC BY 3.0).
  They are loaded by relative URL, listed in `clips.js` and credited next to the
  picker; a clip's file is fetched only when it is selected. Each clip has a 360p
  file (used at 384×216 and 640×360) and a 720p file (854×480, the default, and
  1280×720); changing
  resolution reloads the matching one. Sources, segment times, changes and
  verbatim notices are in [`clips/ATTRIBUTION.md`](clips/ATTRIBUTION.md). Only add
  clips whose terms allow redistribution (never the evaluation sets in `/mldata`).
- **Two versions:** choose H.264, HEVC, AV1, VVC, VP9 or AV2 independently. Both
  versions get byte-identical RGBA source frames. Their displays advance together, on the
  same source frame, once both decoders have finished it. At most three frames are in
  flight; when the pipeline is full the page skips capture instead of queueing
  (see "Pipeline" below).
- **Target bitrate (default):** both versions start at 300 kbps, with a 600 kbps
  peak rate and a one-second buffer. Compare equal targets after startup has
  settled; measured bitrate remains visible. The shared Rust controller chooses
  each frame's actual quantizer, including keyframes, from current-frame analysis
  and packet feedback. Buffer overruns are reported rather than hidden.
- **Live rate changes:** target, peak, buffer and rate-mode changes apply at the
  next frame without restarting references or clearing the download. The rolling
  measurement window restarts so it reflects the new setting.
- **Pause / Resume:** stops submitting frames and pauses file playback or the test
  animation while preserving codec references, rate-control history and downloads.
  The frames already submitted (at most three) finish before the views settle. Webcam capture
  resumes from the live feed. Stop releases camera access.
- **Keyframe interval:** choose 1, 2, 5, 10 or 30 seconds independently per version
  (converted to frames using the target frame rate).
- **Request keyframes:** refreshes both versions on their next input frame while
  retaining rate-control history.
- **Fixed quantizer:** native fixed QP (0–51 for H.264/HEVC, 0–63 for VVC, 1–255 AV1
  qindex; 0–255 VP9/AV2 qindex, where 0 is lossless). Lower values usually preserve more detail at a larger bitrate.
  QP numbers are not comparable between codecs. Lower QP generally means more data.
- **Fast / Faster:** uses the checked-in `h264-fast`, `hevc-fast`, `av1-fast` or
  `vvc-fast` preset, and for H.264 also `h264-faster` (the same files as the
  CLI's `--preset`). These are explicit speed/quality tradeoffs. The page offers
  no Fast for VP9 and AV2.
- **Advanced** (per version, collapsed by default): settings over the chosen
  preset. Its toggles appear only where the codec supports them (the codec module
  says which, `codec_options`):
  - **Film grain** (AV1): the encoder's experimental film grain synthesis at its
    default strength (the CLI's `--film-grain`); the WASM decoder lays the grain over
    its output. The decoder's grain synthesis is still scalar code, so decode time
    rises (roughly doubling on textured clips; the tooltip says so), and PSNR,
    measured on the grained output, drops by construction.
  - **Persistence AQ** (codecs with block QP): `persist.strength=2`; the reuse term
    (`persist.reuse`) is on by default.
  - **Temporal layers** 1–3 (`layers.count`) and **Second reference**
    (`refs.second=1`, H.264; not with layers).
  - **Parameter overrides:** free-form `name=value` lines (the CLI's parameter-file
    format), applied after the toggles. The search box autocompletes the codec's
    registered parameters and shows each one's description, default and range;
    every line is checked against the engine's registry (`check_parameters`, the same
    check as encoder construction), and a bad name or value shows an inline error
    and is not applied. The registry comes from the module (`parameter_registry`), so
    the page never keeps its own copy. Opening Advanced the first time loads the
    codec module on the page as well (the same file the workers use, from the cache).
- **Settings · CLI command** (below each download): the effective preset and
  overrides, and the equivalent `nanocodec encode …` line (same codec, size, frame
  rate, key interval, quantizer seed, rate settings, preset and overrides), to run on
  an I420 file of the source; the page itself codes its limited-range BT.601
  conversion of the RGBA frames.
- **Other live changes:** codec, quantizer, preset, Advanced settings, GOP, resolution and fps changes
  restart both pipelines on a keyframe and
  clears statistics/download buffers. Video files continue from their current
  playback position; the test scene restarts its deterministic animation.
- **Hold to view original:** pointer or Space/Enter reveals the shared source
  in both views. Release returns to independently decoded pixels.
- **Magnifier:** switch it on below the views, then hover, or drag on touch, over
  either view. The same region (mapped by normalized coordinates) is shown from A,
  B and the original frame side by side at 2×/4×/8× (buttons or scroll wheel),
  nearest-neighbour so blocking and ringing stay visible. Click or tap a view to
  pin the region; Pause holds the frame. It only copies from the displayed
  canvases on pointer moves and new frames, and does nothing while off.
- **Stop:** ends both workers and releases camera tracks. Switching away from
  the tab also stops capture; press Start to resume.
- **Download:** AV1, VP9 and AV2 are IVF with captured media timestamps; H.264/HEVC/VVC are Annex-B
  elementary streams. Each view retains at most16MiB of complete packets from
  the beginning of the current run. When full, playback continues and the saved
  prefix remains downloadable. Elementary streams do not carry the capture
  timestamp gaps that can occur when processing cannot keep up.

## Measurements

The bitrate controller is one-pass and does not add filler bytes, drop frames or
re-encode a frame. Peak rate is a buffer-based burst allowance, not a limit on
each individual packet or a network packet pacer. Scene changes, restrictive QP
bounds or an unattainably low target can overrun that allowance; the displayed
counter makes this visible. Very simple content may undershoot the target even
at maximum quality. Native callers can set QP limits and smoothing through
`RateControlConfig`; the CLI exposes these controls too.


Numbers summarize the last 90 compared frames (or all available frames).
Bitrate is mean compressed bytes/frame × target fps, not network throughput.
Encode time includes RGBA→limited-range BT.601 I420 conversion; decode time is
the decoder alone (the display hand-off is not included). Both are worker wall timings, not single-core
native CPU benchmarks. PSNR uses the same 6:1:1 average of per-plane PSNR as the
harness, capped at 100 dB, against the encoder input: the decode worker converts the
RGBA source frame exactly as the encoder does and sums squared errors with a SIMD128
kernel (`BrowserDecoder.psnr`, every frame: about 2 ms per 720p frame in the decode
worker, mostly the RGBA copy and conversion; the squared-error sum itself takes 0.13 ms,
was 1.8 ms scalar). VMAF and
MS-SSIM are deliberately not fabricated or estimated in the live page.

The session line shows the delivered frame rate and the busiest of the four workers:
"limited by encode B (38 ms)" when delivery falls below the target rate (more than
5 % short), else "keeping up, slowest …". A worker's time per frame is its encode, or its decode
plus PSNR plus display hand-off.

**Pipeline (`pipeline.js`).** Each version has two workers: `encode-worker.js` runs
`BrowserEncoder.encode` and posts the packet, with the source frame, straight to its
version's `decode-worker.js` over a `MessageChannel` (not through the page). The decode
worker runs a separate `BrowserDecoder`, measures PSNR, and posts the frame to display,
the packet and all timings to the page; only that decoder's pixels are shown. So the
encoder codes frame N+1 while the decoder decodes frame N, and the page captures the next
frame while earlier ones are still in flight. Up to three frames (`MAX_FRAMES_IN_FLIGHT`)
are in flight; results are matched by frame id and released in capture order only when
both versions have decoded a frame. While the queue is full no frame is captured, so a
slow pipeline lowers the delivered rate instead of adding latency. Delivered fps is
therefore set by the slowest single worker, not by the sum of encode, decode and PSNR.
Source and display buffers circulate between page and workers, so no frame-sized
buffer is allocated per frame in steady state.

**Display (`renderer.js`).** With a hardware GPU each view draws with WebGL2
(else WebGL1): `BrowserDecoder.i420_frame` packs the decoded planes without row
padding into a reused WASM buffer, the worker copies them out (1.5 bytes per
pixel), and the page uploads them as three one-channel textures that a fragment
shader converts with the same limited-range BT.601 integer arithmetic as the
CPU path; the output is identical (checked by `tools/web/display-bench.py`).
Otherwise (no WebGL, or only software GL such as SwiftShader, which
`failIfMajorPerformanceCaveat` refuses because it is slower than the CPU path)
`BrowserDecoder.rgba_frame` converts to RGBA in WASM (SIMD128) and the page
draws it with `putImageData`. A GL context lost for more than 3 s, or a failed GL
set-up, switches that view to the CPU path on a fresh canvas. `?display=webgl2`,
`webgl` or `cpu` overrides the automatic choice (the chosen renderer is in each
viewport's `data-display`). The GL canvas keeps its drawing buffer, so the
magnifier and "hold to view original" (an RGBA texture) work on either path.
`BrowserDecoder.decoded_i420` exposes raw decoded samples for conformance tests.
No WebCodecs implementation or reference decoder is used in this path.

**Start-up warm-up (`warmup.js`).** V8 compiles each WASM function on its first call, so
a fresh decode worker's first frames are slow. Right after `ready`, the decode worker
decodes a tiny built-in stream of its codec (`warmup/<codec>.ivf`: eight 256x144 frames
of the Big Buck Bunny clip, one key frame and seven inter frames, made by
`tools/web/make-warmup.sh`). It converts each frame for both display paths and scores it,
then discards everything. It yields before every frame and stops at the first real
packet, so a real frame waits for at most one tiny frame, with unchanged output
(`tests/warmup-interrupt.py`, in the landing smoke). It finishes in 30–70 ms, well
before the encoder's first packet. Headless Chromium, pinned, 720p, cold → warm
(`tools/web/warmup-bench.py`):
- **Frame 0 decode:** H.264 28 → 12–18 ms, HEVC 37 → 19–31, AV1 42 → 28–30, VP9 30 → 15,
  AV2 78 → 37–44, VVC 77 → 36–38.
- **First 30 frames:** 8–17 % faster.
- **Later frames:** unchanged.

## Receiving over a lossy network

The demo is loss-free and uses `BrowserDecoder.decode`. A receiver that can
lose frames uses `BrowserDecoder.decode_frame(packet, index, references)`,
which reports each frame as decoded or frozen (missing reference, missing key,
corrupt) with the frame to name in a loss report. Over RTP, pass the frames the
packet depends on from the **AV1 dependency descriptor** RTP header extension
(WebRTC's practice for AV1 and VP9; any codec here) as `references`: the
transport then decides, for every codec, and costs the stream nothing.
Detecting losses from the stream alone works for H.264 and HEVC, and for AV1
only when the encoder codes frame ids (`av1.frame_ids`, on with a long-term
reference; 23-65 bits per inter frame, off by default); VP9 and AV2 cannot
(`docs/handoff/loss-resilience.md`).

## Checks

```sh
cargo test -p nc-web
cargo clippy -p nc-web --all-targets -- -D warnings
node --test web/tests/*.test.js
```

Codec decoder tests cover key/inter reference chains, relevant transforms and
prediction modes, malformed/truncated packets and reconstruction equality.
The browser demo was additionally exercised in Chromium with real WASM using
the test scene, simulated webcam, local video, all codec quantizer endpoints,
fast presets, downloads and mobile layout. Browser-produced default/fast
packets for all three codecs were decoded by both independent standard decoders
and compared byte-for-byte with browser-WASM decoded I420 output.

## Initial VP9 and AV2

Both use native Rust encoders and independent emitted-subset decoders, with
shared GOP planning and realtime bitrate controls. VP9 uses 8×8/16×16 inter
blocks and TX8; AV2 uses 8×8 prediction blocks, TX8 luma, adaptive probabilities
and residual skipping. Geometry must be divisible by eight, and one temporal
layer is supported.

## VVC

The same H.266 encoder and own-stream decoder as the CLI (`--codec vvc`, Default
or `--preset vvc-fast`, 32x32 leaves, CTU 64): WASM packets are byte-identical to
native ones for the same input and settings, and the WASM decoder's output equals
the native reconstruction, which VVdeC and FFmpeg reproduce. Any even size is
coded (padded to 8, cropped by the conformance window). Downloads are Annex-B
(`.vvc`). The decoder accepts our own streams only.

After rebuilding the actual WASM module:

```sh
node web/tests/vp9-wasm-smoke.mjs
node web/tests/av2-wasm-smoke.mjs
node web/tests/vvc-wasm-smoke.mjs
node web/tests/sustained-wasm.mjs
python3 web/tests/browser-product-smoke.py
```

The last command requires Python Playwright and its Chromium installation
(`python3 -m pip install playwright`, then `python3 -m playwright install chromium`).
It starts an isolated local server/browser with a synthetic webcam and checks
both codec workers, pause/resume, live CBR changes, key requests and downloads.
The sustained test checks 9,600 encode/decode pairs for instance isolation and
memory stability. Neither test accesses a physical webcam or modifies another
demo server.
