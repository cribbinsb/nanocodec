# Public claims audit, 2026-11-27

An adversarial, read-only pass over the public wording: README.md, docs/FAQ.md, docs/LEGAL.md,
docs/ARCHITECTURE.md, THIRD_PARTY.md, CHANGELOG.md, crates/*/README.md and the crate docs, web/ (demo and
usage docs) and tools/bench/pages (results site). It looks for one failure pattern: a test or metric
passed, so the broader thing was claimed. It was prompted by the AV1 decoder "242/242 conformance"
wording. Decoder-compliance claims are out of scope here; a separate audit covers them.

Checked at main 8cfcb9d6d. Nothing in the code or the public text was changed. Each finding gives the
quote, file:line, a verdict (accurate, overstated, false or unverifiable), the evidence and corrected
wording. "Unverified" means it was not checked.

## Ranked findings (false and prominent first)

### 1. VVC long-term references: false (front page)

- **Quote:** "Temporal layers, long-term references, intra refresh and key frames on request (H.264, HEVC,
  VVC, AV1) keep live streams robust to loss." (`README.md:35-36`)
- **Evidence:**
  - VVC has no LTR. `CodecBackend::supports_long_term_reference` defaults to `false`
    (`crates/nc-encoder/src/backend.rs:519`). H.264, HEVC, AV1, VP9 and AV2 override it; nc-vvc does not
    (`grep -n supports_long_term_reference crates/*/src/backend.rs`).
  - Asking for LTR on VVC is refused with "not supported by this codec yet" (`encoder.rs:5500-5503`).
  - The list also understates two codecs: VP9 and AV2 have LTR and temporal layers. VP9 and AV2 have no
    intra refresh (`encoder.rs:1603`).
- **What is tested:**
  - Real loss recovery is simulated in `tools/ci/realuse_conformance.sh:94-111`. `--simulate-loss` drops a
    frame, and the rest must decode bit-exactly in the standard decoders, with a key or LTR inter frame
    as the recovery picture.
  - That only covers LTR loss for `h264 hevc av1 av2`, and its default codec list is the same four
    (line 269). VVC and VP9 get none of it.
  - It is a manual pre-release suite, not part of landing or verify (`docs/PROCEDURES.md:548-549`).
    No run is recorded under `docs/results/`, so when it last passed is unverified.
  - Intra refresh streams are checked bit-exact on landing (`landing_conformance.py` `refresh` case).
    Recovery after a loss with intra refresh is not simulated in any independent decoder.
- **Corrected:** "Temporal layers and key frames on request (all six formats), long-term references with
  loss feedback (H.264, HEVC, AV1, VP9, AV2) and intra refresh (H.264, HEVC, VVC, AV1). Recovery from a
  lost frame is tested with the standard decoders for H.264, HEVC, AV1 and AV2."
- **Also stale:** `crates/nc-cli/README.md:105` says VVC has no temporal layers or intra refresh; it has
  both. `docs/STATUS.md:842` lists LTR, intra refresh, the feedback API and the C API as not started; all
  four exist.

### 2. "Every stream our encoders make is checked against two independent decoders": overstated (front page)

- **Quote:** "Every stream our encoders make is checked against two independent decoders per format."
  (`README.md:28-29`). Related: FAQ:23, "conformance checks against independent decoders (below) before
  numbers are published".
- **What the mechanism actually is:**
  - `tools/ci/landing_conformance.py` codes a fixed set of short test streams for each shipped preset:
    main, fixedqp, still, starved, layers, refresh, large, farqp, timed, switch, and SVC operating
    points.
  - It decodes them with two standard decoders plus ours and compares frame by frame. It is a good
    check, but it covers a sample of streams, not every stream.
  - It runs after landing ("land first, verify beside"), when the coordinator runs `land.sh --verify`.
    It does not gate a landing.
- **Its state today:**
  - The streams half last passed at bcd8f4caa (2026-11-27 07:31), 10 commits behind main
    (`run/verified-streams-head`).
  - The next run, for 3db6eec8d, failed for an infrastructure reason ("fatal: not a git repository",
    `run/land-logs/verify-streams-3db6eec8d.log`). It was not a stream mismatch.
  - The measured BP-4 build 23c962d8b and the 0.7.0 release base 27e4c45da both passed the streams half
    (`run/verified-history:590`).
- **The tests half has not passed since c15638cab (2026-11-25 10:32)** (`grep ^tests run/verified-history`).
  - That half runs fmt, clippy `-D warnings`, the workspace tests, checkasm SIMD-vs-scalar bit-exactness
    and the browser page tests.
  - 0.7.0 was released 273 commits later, many of them new SIMD kernels, with no recorded tests-half
    pass.
  - Release 0.7.0's own checks were the PGO build and stream identity only
    (`~/nanocodec-data/release/v0.7.0/state.json`).
  - So FAQ:19-22, which lists checkasm, clippy and tests as the safeguards, describes checks that exist.
    Whether they were run on the released code is unverified: they may have run in individual agents'
    trees.
- **Not in the landing check at all:** LTR, key-on-request and loss recovery. Those are in the manual
  real-use suite of finding 1.
- **Corrected:** "Standard streams, checked. After each change lands, short test streams of every
  shipped preset of every format are decoded by two independent decoders and ours and compared frame by
  frame with the encoder's own pictures."

### 3. "The C ABI catches panics at the boundary": false for the documented build

- **Quote:** "The C ABI catches panics at the boundary." (`docs/FAQ.md:303`)
  - Also `docs/ARCHITECTURE.md:97` ("panics are caught at the boundary") and `crates/nc-capi/src/lib.rs:17-18`
    ("never unwinds across the ABI: panics are caught at the boundary").
- **Evidence:**
  - The workspace `[profile.release]` sets `panic = "abort"` (`Cargo.toml:74`), and `[profile.gate]`
    and `[profile.web]` inherit it.
  - The documented way to build the C library is `cargo build --release -p nanocodec-capi`
    (`web/docs/c-api.html:33`, `web/docs/index.html:39`).
  - With `panic = "abort"`, `catch_unwind` in `guarded()` (`crates/nc-capi/src/ffi.rs:262`) never
    returns `NcStatus::Internal`: a panic aborts the host process.
  - The guard code itself is present on every session-calling function (via `with_session`).
  - No test exercises a panic through the ABI. Cargo ignores `panic` for test builds, so a unit test
    would not see this anyway.
  - The 0.7.0 PGO `libnanocodec.{a,so}` are release builds, so presumably abort too. That inference
    was not checked on the binaries.
- **Corrected (wording only):** "The C ABI never unwinds into C: a panic inside the library aborts the
  process (release builds use `panic = "abort"`). Do not pass untrusted streams to it in production."
- **Fix in code instead:** build nc-capi with `panic = "unwind"` so the guard works, and add a test.

### 4. "Plays in any browser": overstated (front page)

- **Quote:** "Plays in any browser. All six decoders compile to WebAssembly…" (`README.md:26`)
- **Evidence:**
  - Both browser modules are built with `+simd128` (`tools/web/build.sh:55-56`), and `web/wasm.js:14-16`
    only chooses between SIMD128 and SIMD128 plus relaxed SIMD. There is no build without SIMD.
  - Every browser test launches Chromium only: `grep -rhoE '(chromium|firefox|webkit)\.launch' web/tests
    tools/web tools/ci` gives 29 × `p.chromium.launch`, and there is no CI workflow.
  - Firefox and Safari are untested.
  - `web/docs/browser.html:43-47` already states this correctly.
  - Threads are not a factor: there is no SharedArrayBuffer, and no COOP/COEP headers are needed.
- **Corrected:** "Plays in the browser. All six decoders compile to WebAssembly (SIMD128: Chrome/Edge
  91+, Firefox 89+, Safari 16.4+; tested in Chromium), so a page can play HEVC, VVC or AV2 without the
  browser's own codecs."

### 5. "including the WebRTC modes": overstated (front page)

- **Quote:** "Spatial and temporal layers for AV1 and VP9, including the WebRTC modes (L1T2, L1T3, L2T1,
  L3T1 and K-SVC)" (`README.md:33-34`)
- **What is true, and well tested:**
  - The encoder takes 15 modes, all named after WebRTC-SVC: L1T1–L3T3 and the K-SVC `_KEY` modes
    (`crates/nc-encoder/src/spatial.rs:285-340`). Only AV1 and VP9 take spatial layers.
  - Every AV1 operating point is decoded by dav1d (`--oppoint`) and aomdec, both whole and after
    `nanocodec extract`. VP9 full, stripped and K-SVC sub-streams are decoded by vpxdec and ffmpeg.
    This is on every landing (`landing_conformance.py:381-404`, `eval/nceval/conformance.py:611-882`).
- **What does not exist:**
  - No RTP packetisation, no AV1 Dependency Descriptor and no VP9 payload descriptor or scalability
    structure (grep finds these names only in comments).
  - No test in a WebRTC or browser-RTC context.
- **Corrected:** "Spatial and temporal layers for AV1 and VP9 in the WebRTC-SVC modes (L1T1–L3T3 and
  K-SVC); every layer is checked in the standard decoders. RTP packetisation is not included."

### 6. Demo reference streams "at the settings of the project's charts": false for the current charts

- **Quote:** "a stream pre-encoded offline by another encoder at the settings of the project's charts"
  (`web/index.html:34`). Also `web/panel.js:260` and `web/README.md:128`.
- **Evidence:**
  - The streams date from 2026-11-22 (`git log web/streams-ref/manifest.json`). They were made by
    `tools/web/make-reference-streams.py` with the old `eval/nceval/cbr.py` commands.
  - The current BP-4 charts use `docs/benchmark-fairness.toml`. The two differ:
    - **libvpx:** the demo lacks `--aq-mode=3`, the undershoot/overshoot settings, `--max-intra-rate`,
      `--min-q` and max-q 63. `docs/benchmark-fairness.toml:292` records that this costs vpxenc +10–11 %
      BD-J.
    - **libaom:** the demo uses aomenc, while BP-4 charts it through `svc_encoder_rtc`.
    - **Key interval:** the demo keys every 300 frames, BP-4 every 150.
  - The error makes the references look worse than in the charts.
- **Corrected:** regenerate the streams from `fairness.toml`. Until then: "at real-time settings from an
  earlier version of our benchmark; its current settings may do better."

### 7. Results site footer lists VVenC as a comparison point: false

- **Quote:** "…Kvazaar and VVenC are its comparison points here." (`tools/bench/pages/index.html:71`
  and the other pages' footers)
- **Evidence:** `docs/benchmark-procedure.md:785` says "VVenC is not charted". The published
  `results/data/bp3.json` has no VVenC point.
- **Corrected:** drop VVenC, or add "(not charted: no real-time CBR mode)".

### 8. Results method page: "these reports time one machine": false

- **Quote:** "Speed on other machines (Jetson, Zen, Graviton)… these reports time one machine", and the
  list of things "not measured by these reports" includes history and decoder speed
  (`tools/bench/pages/method.html:34-35`).
- **Evidence:**
  - The reports time c8i (Granite Rapids) and c8g (Graviton 4), and the README chart is Graviton 4.
  - The site now has History and Decoders pages.
- **Corrected:** "Speed on Jetson, Zen and older Graviton generations: these reports time c8i and c8g
  only."

### 9. FAQ benchmark section describes the 0.6.0 procedure, not the charted BP-4: stale and contradictory

- **Quotes:**
  - The ladder "150, 300, 600 and 1200 kbps … plus 40 and 75 kbps" (`docs/FAQ.md:52-55`).
  - "The headline chart uses the 138 fresh held-out clips … with the 209-clip held-out set"
    (`FAQ:149-151`).
  - Speed "at a fixed quantizer … on three 720p development clips … not in CBR mode and not on the clips
    the quality is measured on" (`FAQ:180-183`).
  - "README's `[headline](../README.md#headline-results)`" (`FAQ:86`): the anchor no longer exists.
  - A link to `results/buffer.html` (`FAQ:199`): the page is no longer on the site.
- **Evidence:**
  - The README's chart is BP-4 (`docs/images/bd-vs-speed-bp4.svg`): "held-out camera clips, common set;
    CBR", "timed with half the machine's cores busy".
  - The BP-4 ladder is 300/500/800/1200 kbps at 720p30 (`eval/nceval/rate_ladder.py:15`, used by
    `tools/bench/plan.py:47`).
  - BP-4 holds 97 clips with 92 in the common set (`~/nanocodec-data/bench/bp4-prep/reports/ours-c8g.json`).
  - BP-4 times the same CBR encodes it scores, enforced by SHA-256 (`tools/bench/report.py:868-881`).
  - A reader of the FAQ is told the chart's speed axis is fixed-QP timing on other clips, which is no
    longer true.
- **Corrected:** rewrite "What the benchmarks show" from `docs/benchmark-procedure.md` (BP-4). Keep the
  0.6.0 figures only where they are labelled as history.

### 10. 0.7.0 results and "each release's PGO builds": the measured build is not the release

- **Quotes:**
  - "Benchmark procedure BP-4 … measured development build 23c962d8b, shortly before this release"
    (`CHANGELOG.md:9-11`).
  - "The speed figures on the results site are measured on each release's profile-guided (PGO) builds
    of the command-line tool and C API, which are attached to the GitHub release"
    (`crates/nanocodec/README.md:15-17`).
- **Evidence:**
  - `git rev-list --count 23c962d8b..v0.7.0` gives 187.
  - Among those commits are stream-changing defaults the CHANGELOG lists under 0.7.0, which the
    published numbers therefore do not include:
    - HEVC variance AQ `rate.aq_strength=0.25` (d2c1b41f5);
    - `rate.base_mc_aq_qp=4.0` in all six codecs (e26b2447d);
    - AV1 `delta_q_res=2` (53620f8dd);
    - HEVC `quant.trellis_key_lambda_scale=0.5` and the AV2 Slow unpin (2ba71ce99);
    - the AV1 slow/slower `rate.in_frame` unpin (d22dc4048).
  - The CHANGELOG line is honest about the build. The README chart caption and the results pages do not
    name it.
  - The nanocodec crate README's sentence is false for 0.7.0: the timed binary was built from 23c962d8b,
    not the release's PGO artefacts. Whether that build was PGO at all is unverified;
    `docs/benchmark-procedure.md:822` says "PGO only for a release".
- **Corrected:** CHANGELOG: "measured development build 23c962d8b (187 commits before the release; the
  0.7.0 encoder changes after it, listed below, are not in these numbers)". Crate README: "The results
  site names the build each figure was measured on."

### 11. "Safe Rust outside the SIMD kernels and the C API": slightly overstated

- **Quote:** `README.md:40`. Also FAQ:19-20 and FAQ:285 ("no `unsafe` outside the SIMD kernels and the C
  ABI").
- **Evidence:**
  - At crate level the claim is accurate. Every crate root except nc-dsp and nc-capi has
    `#![forbid(unsafe_code)]`, including nanocodec, nc-web, nc-vvc and nc-checkasm.
  - The workspace lints deny `unsafe_code` for tests, examples and build.rs.
  - A grep of all `.rs` files outside nc-dsp and nc-capi, comments excluded, found no `unsafe`.
- **Where the claim is too narrow:**
  - Within nc-dsp, unsafe is not only in kernels. The dispatch layer (`dispatch.rs`, file-wide
    `allow(unsafe_code)`, 213 uses), helpers (`block_copy.rs`, `prefetch.rs`) and 41 `mod.rs` tables
    also hold it.
  - There is also inline `asm!` and two hand-written `.S` files (`sao/sve2_high.S`,
    `av1_film_grain/kernels/high_sve2.S`).
  - nc-web's `#[wasm_bindgen]` generates unsafe glue from the dependency's macro.
  - Every `unsafe {}` block has a SAFETY comment (clippy `undocumented_unsafe_blocks`).
- **Corrected:** "safe Rust except the DSP crate (SIMD kernels, assembly and their dispatch) and the C
  API; every other crate has `#![forbid(unsafe_code)]`."

### 12. GPL/LGPL "only run as separate programs": slightly overstated (tools, not shipped code)

- **Quote:** "GPL and LGPL tools (x264, x265, FFmpeg) are only run as separate programs for comparison"
  (`README.md:99`). Also `docs/LEGAL.md:43-44`: "separate, unmodified external tools".
- **Evidence:**
  - `tools/anchors/libavcodec_bench.c` is in the repo and links FFmpeg's libraries to time its decoders
    in process (`tools/dev/decoder_speed.py:14`).
  - ffmpeg.wasm is run as a browser yardstick (fetched at run time).
  - Neither is in any shipped crate, binary or the demo.
  - No source file claims code from x264, x265, FFmpeg or libde265. About 40 mentions are "idea, no
    code" notes, e.g. `intra_estimate.rs:11`. That code is not derived cannot be proven by grep
    (unverified).
- **Corrected:** "No GPL or LGPL code is in nanocodec's libraries, binaries or demo. Our benchmark tools
  run x264, x265 and FFmpeg as separate programs, or link FFmpeg's libraries to time its decoders."

### 13. "on our held-out camera clips": slightly overstated

- **Quote:** `README.md:22`
- **Evidence:**
  - `docs/benchmark-procedure.md:252` says 84 of the 97 clips were "audited once, never tuned on".
  - `docs/results/sealed-log.md` records the set measured at several builds, and rule changes made after
    results were seen (entries 17, 19, 26, 27; each logged).
  - The numbers are right. BD-J against x264 medium `--tune psnr` on c8g is VVC −36.3 %, AV2 −34.7 %,
    AV1 −30.9 % and HEVC −30.2 % (from the chart's own point titles and the report).
  - "Similar or higher speed" holds: x264-psnr runs at 44.8 / 62.0 Mpx per CPU-second
    (c8g / c8i), and VVC at 45.2 / 57.6.
- **Corrected:** "on our held-out camera clips (never tuned on; measured at several builds)".

### 14. Stale numbers and lists (low)

- `web/docs/presets.html`: the slowest rungs' "about 1.5x Default's CPU for H.264, 1.6x for HEVC…
  (held-out clips, build bad14976)". bad14976 is from 2026-11-15, before the 2026-11-21 ladder rework and
  the current held-out set. Corrected: label it with its build and date, or re-measure.
- `web/README.md:420`: "packets for all three codecs" (there are six). `web/README.md:425-430`
  describes the old 8×8 VP9/AV2 subset.
- `crates/nanocodec/README.md:5` "Once published" and `crates/nc-cli/README.md:121`: both are
  published.
- `crates/nanocodec/Cargo.toml:3` and the crate doc call it "Real-time" without saying on what. The
  project's own rule is not to make real-time claims, only measured fps on a named machine.
  Corrected: "Low-latency (no B-frames, no lookahead) video encoders…".
- `docs/ARCHITECTURE.md:34-35`: "8-bit 4:2:0 only (the decoders' high-bit-depth output is not shown in
  the browser yet)". This contradicts `crates/nc-web/README.md`, which describes 10-bit decode with 8-bit
  display conversion, and FAQ:249. The decoder audit owns the 10-bit claims; this sentence is just stale.

### 15. Licence notices (low; no false statement found)

- THIRD_PARTY.md attribution is complete as far as a scripted check could tell. Every `.rs` module doc
  naming a permissive source maps to a THIRD_PARTY entry, and no project cited in sources is missing.
- **Gaps in the notice texts:**
  - libavc and libhevc (Apache-2.0) give only the licence URL. Apache §4(a) asks for a copy; NOTICE
    files are unverified.
  - `docs/licenses/` has no VTM, HM or JM text, although VTM coefficients are used (`mc/rpr.rs`).
  - SVT-AV1 is described as BSD-3-Clause-Clear at one place (~line 314) and as BSD-2 + AOM patent
    licence at others (~1325, 1379).
  - `docs/LEGAL.md:23-25` (credits "in particular") omits libavc, libhevc, dav2d, rav1e and wasm-bindgen.
- **Media:**
  - `LICENSE` says nothing about `web/clips` and `web/streams-ref` not being BSD; only LEGAL.md does.
  - The Vidyo/JCT-VC sequences' "public domain" status rests only on Xiph's label (unverified).
  - The Netflix clips are downscaled and re-encoded. "Format change the licence permits" is a hedged
    judgement ("we believe"), which is correct to keep hedged.

## Checked and accurate

- **AI authorship** (README:38-39, FAQ:11-13):
  - 4,063 commits from 2026-11-06 to 2026-11-20 inclusive (`git log --format=%ad --date=short | uniq -c`).
  - 511,424 lines of `.rs` at the last commit before 2026-11-21 (cc7368dd7), against FAQ's "about
    510,000"; 562k today.
  - 4,260 Claude Opus, 191 Claude Fable, 68 Codex and 1 Gemini `Co-Authored-By` trailers.
  - The public repository gets periodic sync commits, as the FAQ says, so these counts can only be
    checked from the development history.
- **Headline BD figures and speed:** see finding 13. "Single-thread" holds: every reference runs one
  thread in `docs/benchmark-fairness.toml`.
- **"Over 2,000 § references":** 2,951 in `crates/` (`git grep -c '§'`).
- **Fuzzing:** one `cargo fuzz` target per decoder (`fuzz/fuzz_targets/decode_*.rs`), driving the browser,
  C API receiver (the Rust session, not the `extern "C"` layer) and CLI framing paths.
- **SVC operating points:** checked in the standard decoders on every landing (finding 5).
- **Feedback API:** LTR with receiver acknowledgement and loss reports is implemented and unit-tested
  (`nc-capi` `nc_encoder_acknowledge` / `nc_encoder_report_loss`, `session.rs:522-607`).
- **Zero line:** x264 medium `--tune psnr` is disclosed everywhere it is used, and is the harder anchor
  for us (default-tuned x264 medium is +14.2 % BD-J on the same chart).
- **Demo measurement claims** (PSNR against the exact source frame, skip handling, speed hint): each is
  backed by a named Chromium test.

## Suggested order of fixes

1. README lines 26, 28-29, 33-36 and 40 (findings 1, 2, 4, 5, 11): one edit.
2. FAQ:303, ARCHITECTURE:97 and the nc-capi crate doc (finding 3), or fix the panic strategy.
3. The demo reference-stream wording or regeneration, and the results footer and method page
   (findings 6-8).
4. The FAQ benchmark section rewritten for BP-4 (finding 9), and the CHANGELOG/crate README build
   wording (finding 10).
5. The rest when convenient.
