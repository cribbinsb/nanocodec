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
Use HTTPS for a remotely hosted camera demo. No cross-origin isolation headers
are needed: each worker has its own ordinary WASM memory.

## Controls and behavior

- **Sample clip / Test scene / Webcam / Video file:** the page opens comparing the
  default sample clip at the default settings, with no click needed (the hidden
  input video is muted and inline, which browsers allow to autoplay; if one still
  refuses, the notice asks for Start). Picking another clip starts it at once.
  Sample clips loop; local files stop at their end and must be playable by the
  browser. The input is fitted into the selected resolution with letterboxing.
- **Sample clips** are 12–14 s openly licensed excerpts (Tears of Steel, Big Buck
  Bunny; CC BY 3.0) in `web/clips/`, loaded by relative URL, listed in
  `clips.js` and credited next to the picker. Sources, segment times and changes
  are in [`clips/ATTRIBUTION.md`](clips/ATTRIBUTION.md). Only add clips whose
  licence allows redistribution (never the evaluation sets in `/mldata`).
- **Two versions:** choose H.264, HEVC, AV1 or experimental VP9 independently. Both workers get
  byte-identical RGBA source frames. Their displays advance together only after
  both decoders finish; there is never an unbounded frame queue.
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
  An already submitted frame finishes before the views settle. Webcam capture
  resumes from the live feed. Stop releases camera access.
- **Keyframe interval:** choose 1, 2, 5, 10 or 30 seconds independently per version
  (converted to frames using the target frame rate).
- **Request keyframes:** refreshes both versions on their next input frame while
  retaining rate-control history.
- **Fixed quantizer:** native fixed QP (0–51 for H.264/HEVC,1–255 AV1
  qindex; 0–255 VP9 qindex, where 0 is lossless). Lower values usually preserve more detail at a larger bitrate.
  QP numbers are not comparable between codecs. Lower QP generally means more data.
- **Fast:** uses the checked-in `candidates/h264-joint-fast`, `hevc-fast`, or
  `av1-fast` preset. These are explicit speed/quality tradeoffs. VP9 currently
  has one scalar preset, so its preset control is disabled.
- **Other live changes:** codec, quantizer, preset, GOP, resolution and fps changes
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
- **Download:** AV1 and VP9 are IVF with captured media timestamps; H.264/HEVC are Annex-B
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


Numbers summarize the last90 compared frames (or all available frames).
Bitrate is mean compressed bytes/frame × target fps, not network throughput.
Encode time includes RGBA→limited-range BT.601 I420 conversion; decode time
includes I420→RGBA conversion. Both are worker wall timings, not single-core
native CPU benchmarks. PSNR uses the same6:1:1 average of per-plane PSNR as the
harness, capped at100dB, against the converted encoder input. VMAF and MS-SSIM
are deliberately not fabricated or estimated in the live page.

The worker calls `BrowserEncoder.encode`, passes the resulting compressed
bytes to a separate `BrowserDecoder`, and displays only that decoder's pixels.
`BrowserDecoder.decoded_i420` exposes raw decoded samples for conformance tests.
No WebCodecs implementation or reference decoder is used in this path.

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
layer is supported. These are experimental implementations with measured
compression/CPU gaps; see the [handoff](../docs/handoff/vp9-av2.md).

After rebuilding the actual WASM module:

```sh
node web/tests/vp9-wasm-smoke.mjs
node web/tests/av2-wasm-smoke.mjs
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
