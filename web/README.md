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
our internal notes (not published).
Use HTTPS for a remotely hosted camera demo. No cross-origin isolation headers
are needed: each worker has its own ordinary WASM memory.

## Controls and behavior

- **Sample clip / Test scene / Webcam / Video file:** the page opens comparing the
  default sample clip at the default settings, with no click needed (the hidden
  input video is muted and inline, which browsers allow to autoplay; if one still
  refuses, the notice asks for Start). Picking another clip starts it at once.
  Sample clips loop; local files stop at their end and must be playable by the
  browser. The input is fitted into each version's resolution with letterboxing.
- **Sample clips** are excerpts of at least 10 s (or the whole source where it is
  shorter) in `web/clips/`, grouped in the picker as
  People / Motion / Hard texture / Animation: the public-domain JCT-VC/Vidyo
  conferencing sequences (Johnny, the default, KristenAndSara, FourPeople,
  vidyo1/3/4), Netflix El Fuente and Chimera scenes (CC BY-NC-ND 4.0), SVT's 2006
  Multi Format set (its own terms; its PDF ships alongside), SVT's 2022 Natural
  Complexity suite (CC BY 4.0), Tears of Steel / Big Buck Bunny (CC BY 3.0) and
  MOTChallenge MOT17-11, MOT17-04 and MOT20-05 (crowds; CC BY-NC-SA 3.0, made from the
  original JPEGs by `tools/web/make-mot-clips.py`). Every clip plays each source frame
  once on the page's 30 Hz timeline: 24 and 25 fps sources are **retimed to 30 fps**
  (1.25x / 1.2x real time, timestamps only, `tools/web/retime-clips.py`; the source line
  says so), since repeated frames would judder and be nearly free to code
  (`tests/clip-cadence.py` checks one distinct frame per tick). They are loaded by relative URL, listed in `clips.js` and credited next to the
  picker; a clip's file is fetched only when it is selected. Each clip has a 360p
  file (used at 384×216 and 640×360) and a 720p file (854×480, the default, and
  1280×720); each version codes, and is scored against, the file for its own size
  (the rendition the pre-encoded streams of that size were made from; two files are
  decoded when the versions straddle 360p, `exact-source.js` `ViewSources`), and the
  analysis statistics name it. Without WebCodecs the one playing video is the larger
  version's file. A resolution change that needs another file reloads it. Sources, segment times, changes and
  verbatim notices are in [`clips/ATTRIBUTION.md`](clips/ATTRIBUTION.md). Only add
  clips whose terms allow redistribution (never the evaluation sets in `/mldata`).
- **Two versions:** choose H.264, HEVC, AV1, VVC, VP9 or AV2 independently, or a
  pre-encoded stream (below). Each version has its own **resolution** (384×216 to
  1280×720) and **frame rate** (30, 15, 10, 6 or 5 fps); both are shown at the same
  on-screen size, so split views and the magnifier line up.
- **One source timeline (`timeline.js`, `capture.js`):** every source is sampled on
  30 Hz ticks. A version at 30/N fps codes every Nth tick and holds its picture in
  between. A sample clip's frames map onto the ticks as a 30 Hz sampler sees them
  (a 25 fps clip repeats a frame every sixth tick); the page knows which frame it
  captured from `requestVideoFrameCallback`. Each version shows the newest tick it
  has coded; versions at the same rate show the same tick, and the badge gives
  each version's frame number and source time. At most four captures are in
  flight; when the pipeline is full the page skips capture instead of queueing
  (see "Pipeline" below).
- **PSNR when the versions differ:** each version's PSNR is against the source at
  its own resolution, so with different sizes the two are not directly comparable
  (the page says so under the session line); bitrate, timings and PSNR are each
  per version at its own frame rate.
- **Pre-encoded streams (`reference.js`, `streams-ref/`):** each version's top has two
  buttons, "nanocodec (live)" and "Reference encoder" (disabled, with the reason, for
  the webcam, a file or the test scene: streams exist for the sample clips only). A
  reference stream is a clip coded offline by x264, openh264, x265,
  SVT-AV1, libaom, libvpx or NVENC in the chart configurations
  (`docs/results/bd-vs-speed.md`), fetched with progress when chosen and decoded here
  by our WASM decoder (no encode worker). The picker lists the playing clip's streams
  grouped by encoder, each named by encoder, codec, size, frame rate and bitrate, then
  other clips' (choosing one switches the clip; a clip with none says so). A reference
  version hides the codec picker and the live settings. The panel labels it "pre-encoded offline"
  and shows the encoder, its version and source commit, the exact command (Copy),
  clip, size, frame rate, target and actual bitrate; Encode shows "n/a". Choosing a
  stream switches to its clip; it loops with the clip on the shared timeline, and its
  PSNR is against the same captured source frames as a live version's. **Match nanocodec to
  this stream** sets the other version live to the stream's size, frame rate,
  target (peak 2×), 1 s buffer and key interval, starting with the same codec (the
  preset is yours). **BWC SOC**: six FoodMarket streams supplied by the owner from a
  body-worn-camera SoC's hardware encoder (EVBR, a key frame every 5 s), shown as supplied, with no
  command line, and with x264 medium / openh264 / x265 medium encodes at the same 1000 / 1500 /
  2000 kbps and key interval beside them (each panel names its rate control). The 68 streams (every clip has 3 or more, several at one size and rate
  across encoders; Xiph, Blender, SVT 2022 and MOTChallenge clips, the MOT streams under
  CC BY-NC-SA 3.0 with that notice in their panel; of Netflix only FoodMarket, by the owner's decision,
  with its CC BY-NC-ND notice) and their commands are listed in
  [`docs/results/reference-streams.md`](../docs/results/reference-streams.md);
  `tools/web/make-reference-streams.py` regenerates them all (the supplied ones are pinned and checked) and that table.
- **Target bitrate (default):** both versions start at 300 kbps, with a 600 kbps
  peak rate and a one-second buffer. Compare equal targets after startup has
  settled; measured bitrate remains visible. The shared Rust controller chooses
  each frame's actual quantizer, including keyframes, from current-frame analysis
  and packet feedback. Buffer overruns are reported rather than hidden.
- **Live rate changes:** target, peak, buffer and rate-mode changes apply at the
  next frame without restarting references or clearing the download. The rolling
  measurement window restarts so it reflects the new setting.
- **Pause / Resume / Step:** Pause keeps capturing until the next tick both versions
  code (for 30 against 15 fps, the next 15 fps instant), then stops submitting frames
  and pauses playback while preserving codec references, rate-control history and
  downloads; the frames already submitted (at most four) finish, so both versions
  show the same instant. **Step** (while paused) moves both to the next common
  instant: a video seeks to that frame (the server must answer HTTP range
  requests, as static hosts do), the scene draws it; the webcam cannot step.
  Webcam capture resumes from the live feed. Stop releases camera access.
- **Keyframe interval:** choose 1, 2, 5, 10 or 30 seconds independently per version
  (converted to frames using that version's frame rate).
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
  an I420 file of the source (for a sample clip, `ffmpeg`'s decode of it, scaled
  as the pipeline below describes when the size differs; add `--colour bt709` to
  signal its colour as the page does).
- **Other live changes:** codec, quantizer, preset, Advanced settings, GOP, resolution, frame rate and source changes
  restart both pipelines on a keyframe and
  clears statistics/download buffers. Video files continue from their current
  playback position; the test scene restarts its deterministic animation.
- **Hold to view original:** pointer or Space/Enter reveals the source in both
  views, each at its own resolution. Release returns to independently decoded pixels.
- **Magnifier:** switch it on below the views, then hover, or drag on touch, over
  either view. The same region (mapped by normalized coordinates) is shown from A,
  B and the original frame side by side at 2×/4×/8× (buttons or scroll wheel),
  nearest-neighbour so blocking and ringing stay visible. Click or tap a view to
  pin the region; Pause holds the frame. It only copies from the displayed
  canvases on pointer moves and new frames, and does nothing while off.
- **Analysis** (`analysis.js`, `analysis-view.js`; off by default, and the page is
  unchanged while off): shows which coding tools each version's encoder used, from
  the decoder's own block records (nc-common `stream_stats`; H.264 and HEVC so far,
  other codecs show a notice). Overlay toggles: blocks coloured by prediction
  (intra, inter, skip, merge, with partitions), transform blocks (coded ones
  brighter), motion vectors, intra directions (P = planar, DC), and a QP or
  bits-per-pixel heat map on fixed scales so the two views compare. Each view gets
  a statistics table (this frame and the mean since analysis started, by area).
  Layouts: side by side, stacked, or one view large; ⛶ puts a view fullscreen.
  Zoom (scroll, pinch, the +/− buttons) and pan (drag) apply to both views; past 1:1
  the pixels are unsmoothed and the overlay stays sharp. Hovering a block (or tapping
  it) shows its record. The magnifier draws the overlay too. With analysis on, the
  decode worker copies the records out of WASM memory each frame: see
  `docs/handoff/stream-analysis.md` for the decode cost. Switching analysis on or
  off does not restart: the comparison keeps its instant (paused stays paused,
  playing continues) while each view's decoder moves to the other module and
  decodes the view's stream again from its last key frame to the picture on
  screen (live encoders keep running; a live view keeps the packets since its
  last key frame for this, up to 64 MiB, past which the switch restarts).
  **PSNR map** (`error-map.js`; nc-web `error_map`, analysis build only): each
  block of a fixed grid (16×16 luma, 8×8 from 4× zoom; the same blocks for every
  codec, unlike each codec's own partition) coloured by its PSNR against the exact
  source frame the view coded, on a fixed scale (red 20 dB or less, blue 50 dB or
  more; identical blocks are 100 dB, the cap, drawn as the top), for Y, Cb or Cr.
  The statistics give the frame's PSNR of each plane and PSNR-YUV (6:1:1, the same
  formula as the PSNR-YUV number) and the source tick compared; the hover record
  gives the block's. The decode worker computes it (nc-dsp's SSE kernel per block,
  dB in JS). `web/tests/psnr-map.py` checks every value against a native decode
  of the same access units scored against the demo's own source frames.
  **MS-SSIM** (luma, the evaluation's definition: `eval/nceval/msssim.py` =
  libvmaf `float_ms_ssim`, reimplemented in nc-dsp `ms_ssim`) costs ~0.1 s per 720p
  frame in WASM, so with analysis on each view's decode worker hands it to its own
  metrics worker (`metrics-worker.js`), which scores the newest shown frame whenever
  it is free: not every frame, but always the paused one. The statistics show the
  latest score with its tick and the mean of the scored frames; `psnr-map.py`
  checks every score against `msssim.py` within 0.001.
- **Stop:** ends both workers and releases camera tracks. Switching away from
  the tab also stops capture; press Start to resume.
- **Debug report** (`debug-report.js`): "Debug report ↓" saves the page's whole
  state as one Markdown file (readable sections, then a JSON block, schema
  `nanocodec-debug-report/1`): the build commit (`pkg/build.json`, written by
  `tools/web/build.sh`), browser, source (clip id and file SHA-256, exact or sampled
  path, loop; a webcam or local file is described, never embedded), each view's
  effective configuration, every number shown in its panel and statistics, its
  recent coded frames (tick, source frame, type, quantizer, bytes, SHA-256 of the
  access unit, timings, PSNR), the bitrate and PSNR windows with their formulas,
  mid-run changes (rate, key frame requests) and the event log tail. "with
  streams" adds the access units since each view's last key frame (base64).
  `python3 tools/web/replay-debug-report.py REPORT.md [--nanocodec BIN]`
  regenerates a sample clip's source frames, re-encodes live views natively,
  checks every access unit's hash, decodes with ffmpeg and re-scores every
  frame's PSNR, and recomputes the shown bitrate, PSNR and times; any mismatch
  fails it. `web/tests/debug-report.py` runs it on reports the page saved.
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
Bitrate is mean compressed bytes/frame × the version's frame rate, not network throughput.
Encode time includes the input conversion (placing the video's planes in the view,
or RGBA→BT.601 I420 for RGBA input); decode time is
the decoder alone (the display hand-off is not included). Both are worker wall timings, not single-core
native CPU benchmarks. PSNR uses the same 6:1:1 average of per-plane PSNR as the
harness, capped at 100 dB, against the encoder input itself: the YUV planes the
encoder coded (for a video or camera source, the decoded frame's own planes, at a
size other than the file's scaled as described under the pipeline), which the
encode worker passes on with the packet; squared errors come from a SIMD128 kernel
(`BrowserDecoder.psnr`, every frame). It is therefore the same number the offline
evaluation computes on YUV, with no colour conversion in between (before
2026-11-13 the page measured against a BT.709 → RGB → BT.601 round trip of the
video, which lowered PSNR by up to about 1.5 dB and made live and pre-encoded
views differ). VMAF and
MS-SSIM are deliberately not fabricated or estimated in the live page.

The session line shows the delivered frame rate and the busiest of the four workers:
"limited by encode B (38 ms)" when delivery falls below the target rate (more than
5 % short), else "keeping up, slowest …". A worker's time per frame is its encode, or its decode
plus PSNR plus display hand-off.

**Pipeline (`pipeline.js`).** The page captures each frame as a `VideoFrame` (an
`ImageBitmap` without WebCodecs) and hands it to `capture-worker.js`. A video or camera
frame in I420 or NV12 (what browsers decode H.264 to) goes to each version's encoder
over a `MessageChannel` as its own decoded YUV planes (`VideoFrame.copyTo` in its own
format, once for both versions: no YUV → RGB → YUV round trip, 1.5 bytes per pixel,
and a read-back 4-30x cheaper than RGBA when frames live on the GPU). The encode worker
places them in the version's size in WASM (`SourceFrame`): unchanged at the file's own
size, else scaled with the player-model bilinear filter (`nc_dsp::display`, each plane on
its own grid) into the largest centred rectangle of the source's shape, black bars
around. The stream signals the frame's colour (BT.709 limited range for the sample
clips; a webcam's own BT.601/BT.709 and range; AV2, whose subset has no colour syntax,
signals nothing), and both display paths convert with it. The sample clips are coded
exactly as `ffmpeg` decodes the files (`web/tests/capture-identity.py`), which is also
what the pre-encoded reference streams code, so a live and a pre-encoded view of a
clip see identical source frames. Other pictures (the test scene, other formats or
colour matrices, engines without `VideoFrame`) take the RGBA path: letterboxed and read
back as RGBA (`VideoFrame.copyTo` at the source's own size, else an `OffscreenCanvas`)
and converted to limited-range BT.601. Reading video back was the page's
largest cost (15-90 ms per 720p frame in headless Chromium under load), and while it ran
the page could not receive or draw results. Each version has two workers: `encode-worker.js` runs
`BrowserEncoder.encode` and posts the packet, with the source frame, straight to its
version's `decode-worker.js` over a `MessageChannel` (not through the page). The decode
worker runs a separate `BrowserDecoder`, measures PSNR, and posts the frame to display,
the packet and all timings to the page; only that decoder's pixels are shown. So the
encoder codes frame N+1 while the decoder decodes frame N, and the page captures the next
frame while earlier ones are still in flight. Up to four frames (`MAX_FRAMES_IN_FLIGHT`, one per stage)
are in flight; results are matched by frame id and released in capture order only when
both versions have decoded a frame. While the queue is full no frame is captured, so a
slow pipeline lowers the delivered rate instead of adding latency. Delivered fps is
therefore set by the slowest single worker, not by the sum of encode, decode and PSNR.
Display buffers circulate between page and decode workers; the source frame the
decode worker returns becomes the page's copy of the original (magnifier, "hold
original").

**Display (`renderer.js`).** With a hardware GPU each view draws with WebGL2
(else WebGL1): `BrowserDecoder.i420_frame` packs the decoded planes without row
padding into a reused WASM buffer, the worker copies them out (1.5 bytes per
pixel), and the page uploads them as three one-channel textures that a fragment
shader converts with the same integer arithmetic as the CPU path, in the source's
colour matrix and range (`colour.js`; BT.601 or BT.709, limited or full); the output
is identical (checked by `tools/web/display-bench.py`; the page's JS conversion of the
original frame against WASM's by `web/tests/colour-wasm.mjs`).
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

The encode worker does the same (`encode-warmup.js`): after `ready`, while it has no
frame to code yet (the first captures, and frame 0 held for the start-up lookahead), a
throwaway encoder of the view's codec, preset and rate mode codes eight synthetic
256x144 frames (one key, seven inter with motion), yielding before each and stopping
once the first real frame is coded. Headless Chromium, default page, fresh profile per
load, 4 interleaved loads each (without → with): the first 10 encodes of H.264 take
509 → 234 ms in total, of AV1 903 → 509 ms; first frame on screen 1158 → 1241 ms
(within the loads' spread, 906–2162 ms).

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
(our internal notes (not published)).

## Checks

```sh
cargo test -p nc-web
cargo clippy -p nc-web --all-targets -- -D warnings
node --test web/tests/*.test.js
node web/tests/reference-streams-wasm.mjs   # after tools/web/build.sh
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
