# Adversarial review of the public claims (2026-11-13)

Before showing the project publicly we asked a reviewer (an AI agent, briefed to act as a hostile but competent
critic) to find every criticism an expert could make of nanocodec's claims, benchmarks, use of AI, licensing and
presentation, to decide which are fair, and to fix what was wrong or overstated. This page is the result. We
publish it as it is: critics are welcome to read it, and to tell us what it missed.

Reviewed at `variance-integration` c47e12ee (public `main` f9d9ea6d). Fixes landed with this page.

**Rules applied to every public sentence (owner's instructions).**

1. A claim says only what was measured, on what (clips, rates, frames, hardware, date, build), against what
   (encoder, version, settings), by which metric. "Beats / faster than / better than X" becomes "on <set>, at
   <settings>, <metric>: A vs B", or is cut.
2. Where the held-out result differs from the chart's, both are shown side by side. No headline rests on the
   tuning-adjacent clip set alone.
3. No universal words ("always", "every", "fully", "strictly conformant", "production-ready", "real-time on one
   core") without the conditions they were measured under. Goals are labelled as goals.
4. Other projects are described respectfully and per configuration: we tested their real-time presets on one
   thread; their other modes may do better. They are mature, general-purpose and multi-threaded, and cover the
   full standards; ours is a narrow subset built after them, with their code to learn from.
5. AI authorship is stated once, factually.

Severity: **B** = blocker before publicity, **S** = should fix, **N** = note / FAQ answer.

## Findings

| # | Sev | The criticism, as a critic would put it | Fair? Evidence checked | Response |
|---|---|---|---|---|
| 1 | B | "Your 'filtered' public repo isn't filtered: the internal working docs are right there." | **Fair.** The public repository still has a branch (`variance-integration`, 2026-11-08) with the internal rules and handoff notes, and `main`'s history contains them in every commit before the filtering started. Removing files from HEAD does not remove them from history. | Owner decision: delete the branch; rewrite `main`'s history or accept that those notes are public. Not changed by the reviewer. |
| 2 | B | (privacy) "Personal details in the public tree and history." | **Fair as a risk.** A few files and history entries carried machine-specific paths and personal details. | Test path fixed (decoders from `NC_VPXDEC` / `NC_FFMPEG`). The rest is listed privately for the owner's decision. |
| 3 | B | "Your headline −51% is measured on the clips you tuned on." | **Fair.** The chart's 11 clips are the VAL set, which was used to confirm which changes were kept. The results page called them "never used for tuning"; the held-out check sat in a collapsed block. On 71 held-out clips our AV1 Default is −42.5% (chart −50.9%). | README and results now show both columns side by side with the explanation; "never used for tuning" corrected. The held-out numbers will be refreshed from the independent re-run. |
| 4 | N | "The held-out drop proves you overfit." | **Not supported by the data.** Every encoder's saving against x264 `medium` shrinks on the held-out clips, by a median of 7.6 points; ours moved less than the references relative to that common shift (median −0.4 vs +1.9 points), and ranks held within about one place. Likely, not proven, cause: many held-out sources were H.264-encoded. | Stated that way in README, results and FAQ. |
| 5 | S | "Your held-out clips are not held out either." | **Partly fair.** They share no source with the fixed tuning sets, but since 2026-11-11 the same ~190-clip pool feeds the rotating adoption screens (`eval/nceval/levels.py`, level 3 draws from every pool family), and the check ran on 2026-11-12. | Text now says "held out from most, not all, tuning". A truly sealed set (TEST, adv12) exists and should be used for the next milestone check. |
| 6 | B | "'Every stream we report decodes bit-exactly in the standard decoders' is false: you score your own reconstruction." | **Fair.** Chart streams are scored on the encoder's reconstruction (`eval/nceval/cbr.py`), and the references are scored on their decoded output. Conformance is checked on short streams of every preset at landing, and that check now runs after landing ("land first, verify beside"). | Wording now states exactly what is checked (README, ARCHITECTURE, results, chart template). Measurement requested: decode a sample of chart streams and compare with the scored reconstruction. |
| 7 | S | "Your 'rate-invalid' rule lets you overshoot by 30%." | **Partly fair.** The code (`realuse.rate_valid`) allows a bucket peak up to 1.5× the buffer and an average overshoot up to 10% plus buffer/window (30% at 150 frames), judging our points without the key-frame loan. BD-rate uses every bit, so overshoot does not inflate the BD numbers, but "break the 1 s buffer" on the page was not what the code does. | Rule now described as coded. Strict-rule compliance table for all encoders requested from the independent re-run. |
| 8 | S | "Your first key frame borrows 1.4 buffers while x264 is held to a strict VBV." | **Fair.** Documented in the README notice and CLI docs, but not where results are shown. The loan bits are counted in the rate. | Caveat added to results, chart method, FAQ. Compliance table requested. |
| 9 | S | "You don't code frames the references must code ('holds')." | **Fair to disclose.** `hold.enable` = 1 by default; it holds exact or near-exact repeats only. Held frames cost no bits or encode time and are scored as the repeated picture (`framepipe.py`). Effect on BD and fps not yet measured. | Disclosed in README scope, results caveats, FAQ. Held-frame counts per clip requested. |
| 10 | S | "fps is timed at fixed QP, not in the CBR mode you score." | **Fair to disclose.** Both sides are timed at fixed QP, and the page already said so in its caveats. | Stated in the README caption and FAQ. CBR-mode timing comparison requested. |
| 11 | S | "x264 runs with psy-rd/AQ defaults while you report PSNR and MS-SSIM; you tuned on your own metric." | **Fair.** x264/x265 are run without `--tune psnr/ssim`; our presets are tuned on this composite. VMAF partly rewards psy. | Disclosed. Suggest (not done) an extra x264 `--tune ssim` / `psnr` row to bound the effect. |
| 12 | S | "BD-J is a made-up composite." | **Partly fair.** It is our own tuning objective. Plain BD-PSNR-YUV and BD-VMAF are in every table; BD-MS-SSIM is not in the main table. | README caption now names the components and links the plain columns. Suggest adding a BD-MS-SSIM column (data exists in the per-point results). |
| 13 | N | "Which VMAF? 'v1' is non-standard." | **Fair question.** It is libvmaf 3's `vmaf_v1.0.16` model, computed by libvmaf 3.2.1 or by VMAFx (a GPU implementation with local patches), which agree at the printed precision on the points checked. | Stated in results Method and FAQ. |
| 14 | S | "Results page disagrees with the chart page (H.264 −20.4% vs −23.0%, 83 vs 74 fps, ...)." | **Fair.** The results-page summary table was stale for 9 rows. MS-SSIM was said to be "every 4th frame" on one page and "every frame" on another. | Table synced with `bd-vs-speed.md`; MS-SSIM wording fixed. |
| 15 | S | "The SVC comparison against libaom/libvpx is a strawman: L2T3 at +49% vs L2T1 at −9% says your reference is misconfigured." | **Fair.** Our harness gives libaom's three temporal layers WebRTC's cumulative 40/60/100% rate split; libaom's own SVC tests use 50/70/100 or 60/80/100. A one-clip probe (crowd_mot20_06, 1400 kbps, speed 10) gave libaom L2T3 28.9 / 29.6 / 30.0 dB PSNR-Y for the three splits, vs 30.9 dB for L2T1. | libaom L2T3 and L3T3 rows marked "configuration under review" in svc-chart.md and the README and kept out of every comparison sentence. Fix to make: `TEMPORAL_SHARES_L2T3` in `eval/nceval/svc_chart.py` (and the L3T3 equivalent) to libaom's 0.6/0.8/1.0 (or 0.5/0.7/1.0), then re-measure those rows. |
| 16 | S | "'Decode faster than the reference decoders' compares you with avmdec, a research decoder." | **Fair.** AV2's reference is avmdec (not built for speed); AV1 and VP9 decode slower than dav1d and vpxdec; all references were run single-threaded. | Decoders table now says who is faster and that the references were run with one thread. |
| 17 | S | "'Real time on one core' is false for VVC." | **Fair.** VVC Default measured 27.5 fps at 720p on the Jetson core. | Stated in README; ARCHITECTURE goals labelled as goals. |
| 18 | S | "'Standards-strict' / 'Constrained High: 65/65 conformance streams': which suite, which subset?" | **Partly fair.** Counts are of the streams tested per profile; VVC is 9 streams, not Main-profile conformance; AV2 decodes our streams only. | Wording says what was tested. Owner to add the stream lists (or links) for each count. |
| 19 | N | "Many headline numbers come from a loaded PC." | **Fair, already disclosed.** PC load ~40–60; each run normalised by same-round x264. Jetson is the primary, clock-locked platform. | Kept, with "read as noise" wording. |
| 20 | N | "Only 150 frames with one key frame." | **Fair limitation.** | FAQ. |
| 21 | N | "Low-bitrate ladder capped at half the source rate: cherry-picking the low end." | **Partly fair.** It is the target use case, and the cap avoids coding source artefacts; higher rates are not measured. | FAQ says so. |
| 22 | N | "AI slop / unreviewed code." | **Partly fair: no line-by-line human review.** The safeguards are mechanical (compiler, `forbid(unsafe_code)` outside two crates, SAFETY comments on all ~2,100 unsafe blocks, checkasm, conformance against two independent decoders). Seven sampled tests assert specific values. Round-trip tests alone prove self-consistency; the external-decoder tests are `#[ignore]` and run in the landing check. | FAQ "AI authorship". |
| 23 | N | "Plagiarised from x264/FFmpeg." | **Not found.** Spot check of ~12 distinctive items (lambda tables, AQ, psy, SATD, CABAC cost tables, rate control, identifier names): only credited ideas, implemented differently; no x264/FFmpeg names or non-normative tables. Not diffed against upstream source line by line. | FAQ. |
| 24 | S | "Your BSD attributions are incomplete." | **Fair.** Licence texts were missing for dav1d, Thor, libyuv and libvmaf (BSD notices require them); a dav1d port and two AVM-derived modules were not listed; the JM/HM/VTM practices used (λ formula, HM rounding and sign hiding, VTM chroma QP table) were not credited. | Fixed (THIRD_PARTY.md, `docs/licenses/`). |
| 25 | S | "Hallucinated spec citations." | **Mostly not.** A first sample of 30 found about 25 right and 5 wrong or doubtful; a second sample of 33 (11 each H.264, HEVC, AV1) found 29 right and 4 wrong. The errors were off-by-one clauses, mostly from mixing editions (HEVC intra planar/DC, H.264 CABAC encoder initialisation and terminate, VP9 reference clamping, VVC trTypeHor/Ver vs LFNST, HEVC encoder-side §9.3.5.x). | Fixed, with the same errors elsewhere (26 files, comment-only). Checked against the AV1 spec, H.264 2003–2007, H.265 2013 + 2023 contents, H.266 2023, VP9 v0.6 draft. Not checked: AV2 citations, HEVC equation numbers. |
| 26 | S | "Decoders exposed to the web with no fuzzing." | **Partly fair.** Six fuzz targets exist; one campaign of about 30 min per decoder on five decoders found no crashes; VVC never fuzzed; no corpus committed; a handful of `expect()`s in decode paths. The demo decodes only its own streams. | FAQ "Security". Recommend a longer campaign (VVC first) before inviting untrusted input. |
| 27 | S | "Netflix clips are CC BY-NC-ND: your excerpts are derivatives." | **Arguable.** Unmodified contiguous excerpts, format-converted (10→8 bit, downscaled with the whole frame kept, H.264) for browser playback; CC 4.0 §2(a)(4) allows technical format changes, and excerpting is commonly not adaptation, but a strict reader could disagree. Not re-encoded into comparison streams. | Owner decision: keep. LEGAL says it plainly with a removal offer; the demo's clip credit now reads "excerpt, format-converted" and its tooltip states the licence and exactly what was done, as ATTRIBUTION.md does. |
| 28 | N | "Identifiable people in MOTChallenge street scenes." | **A fair concern.** The data is published by MOTChallenge for research; we use it unchanged in content, non-commercially. | LEGAL note and removal offer. |
| 29 | N | "Pre-encoded streams from x264/x265 are GPL." | **No.** Running a GPL program does not put its output under the GPL; only output streams are distributed. | Already in LEGAL; FAQ. |
| 30 | S | "Disparaging wording ('beats', 'bar', 'dominates')." | **Fair.** | Reworded throughout (tables below). |
| 31 | S | "Broken links: your public docs point at files you stripped." | **Fair.** ARCHITECTURE, RETROSPECTIVE, results, eval and web READMEs linked `docs/handoff/*`, `PROCEDURES.md`, `STATUS.md`, `CLAUDE.md` and a private reports directory. | Replaced with "internal notes, not published" or a public file. Code comments still mention handoff notes (left; low impact). |
| 32 | S | "RETROSPECTIVE's numbers don't match the results." | **Fair.** It was written on 2026-11-10/11 under an earlier score. It also said "our encoders hold the bucket at every rung", which is not true of the first key frame. | Dated banner, qualified numbers, the bucket sentence corrected. |
| 33 | N | "Can I reproduce a headline number?" | **Not fully from the README.** Encoders and command lines are public; the anchor build script is `tools/anchors/`; most test clips are not redistributable, the harness expects a local data layout, and VMAFx needs local patches. | The independent re-run is writing `docs/results/reproducibility.md`; not yet reviewed (not available at the time of writing). |
| 34 | N | "The demo's PSNR: against what?" | **Was unclear.** | Demo text now says: each decoded frame against the source frame given to that version, at its size. |

## Measurements requested (via the coordinator, from the independent re-run)

1. Held frames per clip and preset on the chart set, and BD with holds off for one or two presets.
2. Per-point peak buffer fullness and loan-adjusted overshoot for every encoder, and a compliance count under one
   strict 1 s VBV rule applied to everyone.
3. A sample of chart streams decoded by both standard decoders and compared with the scored reconstruction.
4. CBR-mode encode fps vs the fixed-QP fps the chart uses, for ours and for the references.
5. (Suggested) x264 `medium --tune ssim` and `--tune psnr` rows; libaom SVC L2T3 settings check; a BD-MS-SSIM column.

## Wording changed for over-reach (before → after)

| Where | Before | After |
|---|---|---|
| README, top | "**Real-time video streaming from edge devices to the cloud and the browser.** nanocodec encodes live camera video on small, power-limited devices … with low delay and few bits, and decodes it anywhere" | "An experimental, low-delay video encoder for live camera footage … It aims at streaming from small edge devices … at low bitrate to the cloud or a browser." |
| README | "Real time on one core of an embedded ARM CPU (NVIDIA Jetson class) as well as x86." | Per-encoder fps table with resolution and hardware; "our VVC Default is below 30 fps at 720p on the Jetson core". |
| README | "Natively, H.264, HEVC, VVC and AV2 decode faster than the single-threaded reference decoders" | Decoders table: ratios per reference, "(dav1d faster)", "(vpxdec faster)", avmdec described as "AVM's research reference decoder, not built for speed", "all references were run with one thread; they are normally run multi-threaded". |
| README | "**Standards-strict.** No custom syntax, ever. Every stream we report decodes bit-exactly in the standard decoders." | "No custom syntax is used. Before a build's numbers are published, short streams from every shipped preset … must match … The longer chart streams themselves are scored on the encoder's reconstruction and are not each decoded." |
| README | Chart caption: "11 natural camera clips (validation set)…" with held-out in a collapsed block | "These clips were also used to rank candidate changes during development, so the chart is not independent of the tuning", followed by a side-by-side table of chart and held-out BD-J. |
| README | "Our presets other than H.264 stay within about ±2 points of that shift … SVT-AV1 M12/M13 and x265 lose 5–7 points more." | "every encoder's saving against x264 `medium` is smaller, by a median of 7.6 points, ours and the references alike … Relative to the other encoders, our presets keep their ranks within about one place; the size of any saving against x264 depends on the clips." |
| README | "H.264 Fast runs close to openh264's speed (114 vs 111 fps …) and still uses about 26% fewer bits." | "H.264 Fast: 114 fps vs openh264 (realtime, medium complexity) 111 fps; BD-J −25.6% against openh264." |
| README | "AV1 likewise has a Faster point: faster than SVT-AV1 M12 `--rtc` (81 vs 71 fps on one Jetson core) with 7% fewer bits." | "AV1 Faster: 81 fps vs SVT-AV1 M12 `--rtc` 71 fps; BD-J −7.4% against M12 (per-clip range −19.1% to +5.4%)." |
| README | "HEVC has one too: faster than SVT-AV1 M13 `--rtc` … with 4% fewer bits." | "HEVC Faster: 98 fps vs SVT-AV1 M13 `--rtc` 79 fps …; BD-J −4.4% against M13 (BD-PSNR-YUV +1.0%, i.e. no saving on PSNR alone)." |
| README | "Our HEVC WASM decoder is faster than Antare's `hevc.wasm` on all five of their test rows." | "On the five test rows published with the `hevc.wasm` decoder, our HEVC WASM decoder took less time." |
| README | "H.264 \| Constrained High profile: 65/65 conformance streams MD5-exact" (and similar) | "65/65 of the conformance streams we test in that profile"; VVC: "9/9 official 8-bit conformance streams … tested so far; not a full Main-profile conformance claim". |
| README | "Nothing leaves the browser." | "Your video is not uploaded." (the page does fetch clips and streams) |
| results | "11 natural validation clips that are never used for tuning" | "the 11 natural camera clips of our VAL set, which were also used to rank candidate changes during development" |
| results | "confirmed on 11 natural VAL clips that screens never see, and every number on this page comes from VAL or the held-out pristine set above, which is never used for tuning or screening" | "…so VAL (the charts' clips) has influenced which changes were kept … The pristine set above and the held-out check are the less tuning-exposed numbers." |
| results | "MS-SSIM every 4th frame" | "MS-SSIM on every frame (every 4th frame before 2026-11-11)" |
| results / chart template | "Points that miss their target … or break the 1 s buffer are counted as rate-invalid" | The rule as coded (1.5× buffer peak; 10% + buffer/window average; loan excluded for ours; strict VBV for references). |
| results / chart template | "the build passed the landing conformance check (standard decoders and ours bit-exact with the reconstruction on every shipped preset)" | "…(short streams of every shipped preset …); the chart streams themselves are not each decoded." |
| RETROSPECTIVE | "Every number here comes from a measurement recorded in `docs/results/` or `docs/handoff/`." | "…recorded at the time, in `docs/results/` or in our internal working notes (not published)", plus a dated banner. |
| RETROSPECTIVE | "our AV1 needs about half the bits of x264 `medium` … (−49.8%)" | "on our 11 VAL clips, our AV1 needed about half … (−49.8% at the time; −42.5% on 71 held-out clips …)" |
| RETROSPECTIVE | "11 natural VAL clips never used for tuning" | "11 natural VAL clips, which were used to confirm changes" |
| RETROSPECTIVE | "Our encoders hold the bucket at every rung." | "After the first key frame's loan is repaid, our encoders hold the bucket at every rung (the loan itself is up to about 1.4 buffers)." |
| RETROSPECTIVE | "Ours have none on the chart." | "…under that rule, which judges our points without the first key frame's outstanding loan …, so it is not a strict 1 s VBV." |
| ARCHITECTURE | "**Standards-strict.** Every stream is plain, conformant H.264, HEVC, AV1, VP9, AV2 or VVC." | "**Standard syntax only.** Every stream is meant to be plain …, within the subset each encoder uses, and is checked as described in §9." |
| ARCHITECTURE | "H.264, HEVC and VP9 pass their conformance suites; AV1 is close" | "Status: the profiles and conformance streams tested so far are listed in the README's Decoders table …" |
| ARCHITECTURE | "Every stream we report decodes bit-exactly in two standard decoders" | "Before a build's numbers are published, short streams of every shipped preset … Longer evaluation streams are scored on the reconstruction and are not each decoded." |

## Wording changed for respect to the projects we build on (before → after)

| Where | Before | After |
|---|---|---|
| README, top | Notice: "It builds on ideas from permissively licensed projects, above all the excellent SVT-AV1 …" (inside a disclaimer) | A "Built on others' work" box before anything else: SVT-AV1 first, then libaom, dav1d, libvpx, openh264, kvazaar, VVenC/VVdeC, AVM, Thor, libyuv, libvmaf, JM/HM/VTM and the standards' authors, with links; "The coding tools are the standards'; the ideas are largely those projects'." Plus an Acknowledgements section. |
| README | (none) | "The projects we compare with are general-purpose, mature, multi-threaded … We measured only their real-time settings on one thread; their other modes may do better … nanocodec had the advantage of being built after them, with their code to learn from." |
| README | (none) | "Where the references are ahead: SVT-AV1 M12 and M13 `--rtc`, x264 `veryfast` / `ultrafast` and openh264 encode faster than our AV1, HEVC and VVC Defaults …" |
| README, SVC | Reference rows listed as "libaom SVC L2T3 rt 10 +48.9%" with no context | Named as the projects' example SVC encoders, "demonstrations of the API rather than tuned products … their settings, not the library, are the likely limit." |
| results | "HEVC Faster beats SVT-AV1 M13 on both machines with −4.4% bits … so M13 is off both fronts." | "HEVC Faster against SVT-AV1 M13 `--rtc`: BD-J −4.4% (BD-PSNR-YUV +1.0%), Jetson 98 vs 79 fps …" |
| results | "Each of our encoders is compared with its bar" | "…with a reference" |
| results | "On the never-tuned set every codec compared with SVT-AV1 M12 still beats it" | "On this set our HEVC and AV1 Defaults need fewer bits than SVT-AV1 M12 `--rtc` at the settings above (… smaller than on VAL)" |
| results caveats | "The reference encoders' real-time modes are tuned for latency, not for these metrics." | Kept, plus: x264/x265 default psy settings vs our metric-tuned presets; their other modes may do better. |
| RETROSPECTIVE | "**Beat the baselines.**" | "**Targets set by the baselines.**" |
| RETROSPECTIVE | "it remains the bar for our HEVC and AV1" | "its `--rtc` presets remain the reference we measure our HEVC and AV1 against" |
| RETROSPECTIVE | "Our VP9 beats libvpx's real-time mode by 39.8%" | "Our VP9 needed 39.8% fewer bits than libvpx's real-time mode (cpu-used 7, one thread)" |
| RETROSPECTIVE | "It beats M12 by buying quality with that time" | "It needs fewer bits than M12 by spending that time on search" |
| RETROSPECTIVE | "x264 and x265 beat us" | "x264 and x265 need fewer bits than ours" |
| RETROSPECTIVE | "The H.264 → HEVC step dominates, and it matches the reference world" | "The H.264 → HEVC step is the largest, and it matches the reference encoders" |
| ARCHITECTURE | "The bars: H.264 must beat openh264 …; AV1 must match or beat SVT-AV1 M12 `--rtc` …" | "Targets we set ourselves (goals, not results …): H.264 to need fewer bits than openh264 …; AV1 to match SVT-AV1 M12 `--rtc` (one thread) …" |
| LEGAL | Credits: projects only | Adds the standards bodies (ITU-T VCEG, ISO/IEC MPEG, JVT, JCT-VC, JVET, AOMedia), VMAF / MS-SSIM / BD-rate authors, and the test-material providers. |
| Demo | "Built on ideas from permissively licensed projects, notably SVT-AV1." | "Built on the work of SVT-AV1, libaom, dav1d, libvpx, openh264, kvazaar, VVenC/VVdeC, AVM and the standards' authors, with thanks; written by AI coding agents under one person's direction." + clip credits link |
| Demo, reference panel | "Pre-encoded offline, decoded here by nanocodec's WASM decoder" | "Encoded offline by this third-party encoder at the real-time settings our charts use (one thread, low delay, CBR); its other settings may give better results. Decoded here by nanocodec's WASM decoder." |

## Not verified by this review

- AV2 citation section numbers (draft spec not checked) and HEVC equation numbers.
- A line-by-line comparison with x264/FFmpeg source (spot checks only).
- The independent re-run's numbers and `docs/results/reproducibility.md` (not yet available).
- The lists of conformance streams behind the decoder counts (65/65, 136/136, 176/176, 303/303, 9/9).
- The live demo page: the text changes here are in `web/` and go live only when the demo is republished.
