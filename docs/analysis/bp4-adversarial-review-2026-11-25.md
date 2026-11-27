# BP-4.0 adversarial review (2026-11-25)

This is a hostile external review of the benchmark that BP-4.0 defines
(`docs/benchmark-procedure.md`, `docs/benchmark-fairness.toml` at 025140fbf). It asks what an
x264, x265, SVT-AV1 or libaom developer, a standards participant, an academic reviewer or a
journalist could credibly attack. The procedure's own framing was not trusted. Every claim
was checked against code, encoder source or stored records.

**Data used.** No encode or score was made, and no AWS was used. Every number comes from
stored records:
- `bench/bp3r-ours-cbr` (ours, build 889309c99);
- `bench/bp3-ours-cbr-c8a{,-r2,-anchor}` (references);
- the report `bench/bp3r-ours-cbr-c8i-report/report.json` (BP-3.14 rules: 97 view clips, 84
  common, 373 points per family).

These are BP-3 commands. BP-4 changes some reference commands and drops `--strict-buffer` for
ours; where that matters, the finding says so.

**Re-computation.** One analysis re-ran the report code on the stored points, changing only
the zero line. It imports `tools/bench/report.py` unchanged, patches `ZERO_FAMILY`, bypasses
the V18 command check, and calls `report(parts=…, timing_series="c8i")` with the stored
`ours-cbr-parts.json`. The script was not kept; it is those four lines.

**Labels.** VERIFIED means shown from code or data. SUSPECTED means plausible but needing the
named test.

---

## 1. The zero line is x264's worst tune for the composite score. Every "vs x264" headline is inflated by 8-14 points (VERIFIED)

**Critic's claim.** "You measure J and VMAF against x264 in the one tune that x264's own PSNR
tune beats by 20 % on VMAF and 12 % on J, on your own data. Your 'x % fewer bits than x264'
is largely x264 misconfigured for your metric."

**Evidence** (`report.json`, common set of 84 clips, each BD against x264 medium as the
report computes it):

| x264 medium twin | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | BD-J |
|---|---:|---:|---:|---:|
| default zerolatency tune (the BD-J and BD-VMAF zero) | +18.1 % vs ~psnr | -0.4 % | +6.5 % vs ~ssim | -0.1 % |
| `~psnr` | +0.1 % | **-20.6 % [-23.1, -18.3]** | +5.8 % | **-12.5 % [-14.8, -10.3]** |
| `~ssim` | +17.1 % | +3.0 % | -0.1 % | -1.6 % |

- The record's reason for using the default tune as the VMAF zero is that `--tune psnr`
  "disables AQ, which is not x264's best for VMAF/MS-SSIM" (`benchmark-fairness.toml`, x264
  `--tune` entry). The data contradicts it for VMAF.
- The default tune is x264's worst twin for J.
- The project already knew this:
  - `docs/analysis/fixed-qp-fast-rungs-2026-11-24.md` item 1: "x264-medium starves frame 0
    ... and loses -13.6 % J to x264-medium~psnr";
  - `docs/analysis/standards-lens-improvement-2026-11-14.md` Part A: x264 in its own
    zerolatency CBR "needs +55 to +71 % bits on J against itself at fixed QP ... most of the
    margin is x264's rate control".

**Re-computed with `x264-medium~psnr` as the zero for BD-VMAF and BD-J**, nothing else
changed (all points stored, the same envelope and curve rules; the common set becomes 87
clips):

| Family | BD-J as reported | BD-J vs x264 ~psnr | BD-VMAF as reported | BD-VMAF vs ~psnr |
|---|---:|---:|---:|---:|
| nanocodec-h264@veryfast | -3.9 % | **+11.0 %** | -12.7 % | +11.0 % |
| nanocodec-h264@faster | -8.0 % | **+6.3 %** | -15.7 % | +7.2 % |
| nanocodec-h264@medium | -21.3 % | -9.1 % | -26.7 % | -6.8 % |
| nanocodec-av1@veryfast | -23.3 % | -12.1 % | -26.9 % | -8.4 % |
| nanocodec-av1@medium | -39.0 % | -30.4 % | -43.7 % | -29.4 % |
| nanocodec-hevc@medium | -37.8 % | -29.2 % | -42.8 % | -28.6 % |
| nanocodec-vvc@medium | -43.9 % | -36.1 % | -48.5 % | -35.4 % |
| svtav1-m12 | -13.7 % | -1.3 % | -15.9 % | +4.7 % |
| x265-medium~psnr | -29.6 % | -20.0 % | -30.9 % | -14.0 % |

- The relative order and the front do not change: the front is identical, 23 families.
- Every number quoted "against x264" moves by 8 to 14 points.
- Our fast H.264 rungs go from beating x264 medium to losing to it.

**Severity.** This biases every "saving vs x264" headline by about 8-14 BD-J points and
12-20 BD-VMAF points. Comparisons between encoders are mostly unaffected.

**Fix.**
- Make the zero, per metric, x264 medium's best twin on that metric, measured (today
  `~psnr` for PSNR and VMAF, `~ssim` for MS-SSIM).
- Compute BD-J as the mean of the three matched-twin BDs.
- At the least, print BD-J against `~psnr` beside the default-tune figure.
- Correct the record's VMAF reason.

## 2. A 150-frame window with one key at t=0 and a full starting buffer rewards a start-up burst. Ours is built to spend it; the references are not (VERIFIED)

**Critic's claim.** "Your test is mostly the first two seconds of a stream. Your encoder spends
0.5-0.8 s of the channel on frame 0, which in a real call is half a second of extra start-up
delay. The real-time encoders start small by design, so on a 5-second clip they spend half
of it climbing to steady state."

**Evidence.** First-frame size in seconds of the target rate, over 373 points per family:

| Family | median | p90 | max | key's share of the stream |
|---|---:|---:|---:|---:|
| nanocodec (all 35 rungs) | 0.50-0.78 | 0.69-0.95 | 1.00 | 9.4-14.6 % |
| x264 (all presets) | 0.12-0.21 | 0.17-0.31 | 0.55 | 2.3-3.9 % |
| x265 | 0.10-0.11 | 0.18 | 0.40 | 2.0-2.1 % |
| SVT-AV1 M8-M13 | 0.09-0.25 | 0.12-0.52 | 1.16 | 1.6-4.4 % |
| libaom cpu8 | 0.19 | 0.43 | 1.32 | 3.0 % |
| libvpx cpu7 | 0.46 | 0.83 | 1.79 | 6.0 % |

Computed from `frame_bytes` in the score records of `bp3r-ours-cbr` and `bp3-ours-cbr-c8a{,-r2}`, ladder points only.

- The references start 4-6 dB PSNR-Y below their own steady state on static, texture,
  videocall and drone content, and take 30-90 of the 150 frames to recover. Ours starts
  +3 dB above (`docs/analysis/first-key-2026-11-25.md` section 2).
- Capping our first key at 6 mean frames costs ours +5.0 to +7.8 % J, and +11.4 % on AV1
  veryfast (same document, section 1).
- Ours has first-key-only machinery tuned at this window: `rate.first_key_boost_scale`
  2.5 (`params.rs:1330`) and the key-size defaults "a key up to one buffer".
- BP-4.0 also changed every encoder's initial buffer level to full (1.0 s; the fairness
  record's `[phase]`, owner 2026-11-25). That is the setting that maximises this credit. The
  references' own defaults are lower (x264 0.9, SVT-AV1 600 ms, the libvpx and libaom
  examples 600 ms), and the 600 ms "final" phase of BP-3 was dropped.
- libaom's and libvpx's shipped real-time configurations cap the key at 300 % and 1000 % of a
  mean frame (`svc_encoder_rtc.cc:2117-2123`, `vpx_temporal_svc_encoder.c:939-943`), as
  real-time latency controls. Ours runs about 1500-2300 % (median 0.5-0.78 s at 30 fps).

**Severity.**
- Our own measurement attributes 5-8 % of ours' J to the large first key.
- The references' share of the loss from their start-up climb is not measured. Their 4-6 dB
  deficit over 20-60 % of the window suggests it is larger.
- Overall this is likely the largest single bias in ours' favour that a window choice
  produces. It biases the result; it does not invalidate it, because start-up is a real
  condition. But the headline presents start-up as steady-state efficiency.

**Fix.**
- Publish the main BD on long windows, for example frames 150-599 of a 600-frame encode, or
  the 600-frame window with keys every 10 s that the history set already plans (T31) run
  for every encoder, with the 150-frame start-up view beside it.
- Report first-key latency (key bits ÷ target rate) as a headline column beside compliance.
- Measure ours with the 600 ms initial level, which the references' own projects use, and
  state the difference.
- Test cost: the 24-clip reference-audit pool at 600 frames, every family, 4 rungs, about
  $2-4 of Graviton spot plus local GPU scoring.

## 3. Ours alone is allowed past the shared 1 s buffer by design; the references are configured to stay inside it (VERIFIED)

**Critic's claim.** "The 'shared 1 s buffer' is a constraint for x264, x265 and SVT but a
target for you: your rate control's default bound is 1.15 buffers, and every point counts in
the BD whatever it overshoots."

**Evidence.**
- BP-4.0 drops `--strict-buffer` from our command. Ours runs at its default re-code ceiling,
  `rate.recode_overflow` 1.15 (`params.rs:1501`).
- `docs/results/cbr-buffer-cost-2026-11-22.json` (medium, 24 clips, 150 frames):
  - the 1.15 default against strict 1.0 is worth -1.08 % BD-J (AV1) and -1.03 % (HEVC);
  - at the 1.15 default, 19 of 95 AV1 points and 15 of 95 HEVC points peak above one buffer
    at the target (0 of 95 when strict).
- The references, as measured (BP-3 report, the bucket at the target):
  - x264 medium peaks at p95 0.58, worst 0.68 buffers;
  - x265 medium p95 0.63;
  - SVT-AV1 M12 p95 0.96, 1.9 % of points over one buffer.
- Ours at strict already used the whole buffer (p95 1.00). BP-4 lets it use 1.15.

**Severity.**
- About 1 % BD-J in ours' favour (one medium-preset run, no interval).
- The larger, disclosure-level point: x264 and x265 use only 50-60 % of the 1 s buffer that
  ours fills. Part of the margin is buffer aggressiveness, which in x264 is tunable
  (`--vbv-init`, `--ratetol`).

**Fix.**
- Either run ours at the bound the references are configured to (1.0), or chart both of ours
  (the procedure forbids the second series until the owner decides, BP-3.13).
- Print "points over one buffer at the target" in the same table cell as the BD, not in a
  separate compliance table.

## 4. The AOM-CTC view is in-sample for ours (VERIFIED)

**Critic's claim.** "Your 'standard test set' results are on sequences you tuned on."

**Evidence.**
- The benchmark's AOM-CTC set includes the 12 low-delay A3-A5 sequences
  (`eval/clipset/aomctc_bp3.manifest.json`).
- These same 12 sequences were the "standards lens" used to develop and justify our presets.
  `docs/analysis/standards-lens-improvement-2026-11-14.md` says "Overfitting to the lens's
  sequences is allowed for this project". `docs/handoff/tuning.md:874-889` records
  `refs.second` chosen with lens gains of -5.0 % J.
- Preset files cite lens results as their justification: `presets/av1/{veryfast,faster,fast}.toml`,
  `av2/{fast,faster}.toml`, `hevc/{fast,faster}.toml`, `vvc/faster.toml` (26 lens mentions in
  8 preset files), and `nc-av1/src/backend.rs:1476`.
- KristenAndSara was in the screening pool before the lens existed (`pool.json`
  `pool_kristen_sara`; `eval/nceval/ctc/sequences.py:106-108`).
- The procedure states that datasets are "disjoint from every ... tuning set" (section 4,
  V3). V3's test (`eval/tests/test_content_id.py:66`) checks only the PGO set against the
  datasets, not the tuning pool or the lens against AOM-CTC.

**Severity.** The A3-A5 AOM-CTC numbers for ours are tuning-set results; the size of the
effect is not measured. A2 (1080p) is not affected as far as found.

**Fix.**
- Label the AOM-CTC low-delay view "tuned on (standards lens)", or drop it from claims.
- Extend V3's test to the tuning pool and to the lens sequences.

## 5. The rate ladder stops where our fast rungs start to lose (VERIFIED on an older build; SUSPECTED for the current one)

**Critic's claim.** "300-1200 kbps at 720p30 is well below what WebRTC or real-time services
use at 720p. Your own diagnosis shows your fast presets fall apart above it."

**Evidence.**
- `docs/analysis/upper-rates-diagnosis-2026-11-23.md` (4 clips at 720p, 150 frames, 600-4000
  kbps, base 3900962da), against x264 medium:
  - AV1 veryfast BD +5.4 % (on the ladder: -23 %);
  - H.264 faster +13.7 % (ladder: -8 %);
  - H.264 veryfast +41.9 % (ladder: -4 %).
  - The cause is preset tools that only pay at low rates: `av1.partial_forward = 2` "makes
    it physically impossible to represent fine detail" at 4000 kbps.
- Procedure section 16 says only that the gap "narrows" above 1200 kbps.
- The ladder was fixed on 2026-11-23, the day of the held-out audit. Sealed-log entry 1c
  says the procedure "cannot prove the ladder and rule changes made after them did not see
  them".

**Severity.** No headline number is wrong, but it is a cherry-picking target for the fast
rungs, and the disclosure understates it: it is a reversal, not a narrowing.

**Fix.**
- Add one upper rung band (2400 and 4000 kbps at 720p30) as a labelled secondary view, at
  least for the families on the front.
- Reword section 16 with the measured numbers.
- Test cost: the 24-clip pool, every family, 2 rungs, about $2.

## 6. Ours was tuned at exactly these conditions; the references run defaults, and two references run configurations measured to be worse (VERIFIED)

**Critic's claim.** "Thirty-five presets, each tuned by hundreds of screens against this
exact ladder, window, metric mix and clip distribution, against references at stock defaults,
some of them in their known-worse configuration."

**Evidence.**
- Ours:
  - "Dev conditions match the benchmark's" is a project rule, enforced by
    `eval/tests/test_dev_matches_benchmark.py`;
  - the tuning pool and the held-out set come from the same build pipeline, classes and
    collections (`build_pool.py`, `select_heldout.py`).
- SVT-AV1: the rule forces its default 2-level structure, measured +4.2 to +7.2 % BD-J
  worse than flat P, which SVT also documents for RTC (`reference-fair-config-2026-11-25.md`
  section 3).
- x264: the anchor's default tune, finding 1.
- The rule is applied consistently (ours runs its defaults too). But ours' defaults are the
  tuned result, while the references' defaults are general-purpose.

**Severity.** Biases the result; not separable from finding 2 without a test. Section 16
discloses the tuning point ("our presets are tuned at these rates and this window"), but not
that two references run configurations the project measured as worse.

**Fix.**
- Chart the references' best measured real-time configuration beside their default
  (SVT flat P; x264 `~psnr` as in finding 1) as labelled rows.
- Add generalisation views that ours was not tuned on (finding 5's rates, finding 2's long
  windows, 1080p; see finding 8).

## 7. The fast end and the slow end of the speed front are not contested equally (VERIFIED)

**Evidence** (BP-3 report, c8i, Mpx per CPU-second):
- The slowest references charted:
  - x265-fast 22.9;
  - x265-medium 26.0;
  - SVT M8 55.7;
  - libaom cpu8 73.4;
  - libvpx cpu7 105.
- Front members of ours slower than the slowest reference: vvc@veryslow 19.0 and
  av2@slower 10.6. vvc@slower (29.6) and vvc@slow (44.0) sit below SVT M8.
- No reference is charted at x265 slow, SVT M5-M7, libaom cpu6/7 or libvpx cpu5/6/8.
- `fairness.py:29-38` limits x264 and x265 to presets ultrafast-medium, SVT to M8-M13 and
  libvpx/libaom to two speeds each. Ours has 35 rungs, including hevc@veryslow and
  vvc@veryslow.

**Severity.** Disclosure: any "front" claim at the slow end is uncontested by construction.
Fill the gaps or truncate the front at the slowest reference.

## 8. The "ours" dataset is not what "real-time 720p/1080p camera video" suggests (VERIFIED)

**Evidence** (`heldout_2026-11-23b.manifest.json`, 97 clips):
- Size and rate:
  - 0 clips above 720p;
  - 3 at 1280×720;
  - 48 at 960×540, 27 at 640×360, 7 at 320×240;
  - 0 above 30 fps: `build_pool.py:MAX_FPS = 30` halves faster sources.
- Pan: 1 clip.
- Every source is an x264 medium re-encode at CRF 16 with closed 1-second GOPs and scene-cut
  off (`build_pool.py:150-166`). So every source carries a once-a-second key-frame quality
  pulse, and H.264 block structure, which every encoder then re-codes.

**Severity.**
- Possibly biased (SUSPECTED; the procedure's T7 is still open).
- 1080p and 50/60 fps are measured only in the separately reported AOM-CTC A2 view.

**Fix.**
- Finish T7: rank the families on the pool's "pristine" sources in both forms
  (near-lossless and the x264 CRF 16 camera form). About 24 clips, about $2.
- State the size and frame-rate distribution in every chart caption.

## 9. Build asymmetries in speed (SUSPECTED, each one cheap to measure)

- **ISA baseline.** Ours is compiled with `-C target-cpu=x86-64-v3`
  (`.cargo/config.toml`), so all its Rust code, control logic included, may use AVX2, BMI2
  and FMA. Every reference is compiled for generic x86-64 (no `-march` in
  `anchors/x86_64-opt-bp3/build-flags.json`), with only its hand-written kernels dispatched.
  Test: build ours at `target-cpu=x86-64`, or x264 and SVT with `-march=x86-64-v3`, and take
  callgrind or one c8i timing. Under 1 core-hour.
- **Real-time-only builds.** libaom is built without `CONFIG_REALTIME_ONLY=1` and libvpx
  without `--enable-realtime-only` (build-flags.json). These are the builds their projects
  ship for real-time use (Chrome and WebRTC), and they strip non-real-time paths. Test:
  rebuild both and time one family each.
- **Stale PGO.** Every reference's PGO profile was trained on 2026-11-23 with the BP-3
  commands (report provenance block). BP-4 changed:
  - libaom's main rows from `svc_encoder_rtc` to `aomenc` (the profile trained `aomenc`
    only in Fixed QP, `benchmark-fairness.toml` libaom `remaining`);
  - Kvazaar from `--gop 0` to `lp-g8d4t1`;
  - SVT-AV1 from flat P to 2 levels;
  - openh264 to scene-change on;
  - x264 to scene-cut on.

  The training also used 4 clips × 30 frames at a single rate (600 kbps).
  `-fprofile-partial-training` limits the harm. Test: retrain the changed families and time
  them once.
- **Pixels credited.** Ours' AV1 adaptive resolution codes quarter-size frames at low rates,
  but is credited with full-size megapixels in the speed figure (`report.speed` uses
  `w*h*frames` of the job). Disclose how often it switches (response Q3).

Timing method, checked and **not** found biased towards ours: half load slows ours' fast
presets most (×1.23-1.47 on Graviton, `whole-box-timing-2026-11-25.md`). Loaded speed
therefore costs ours relative to idle-core speed.

## 10. Sealed-set handling: reference commands, history subset, timing method (VERIFIED, disclosure)

- Reference commands were first chosen by scoring 1,986 reference encodes on 12 sealed clips
  (sealed-log entry 20, logged late). The audit was redone on 24 pool clips before the
  record changed (`reference-fair-config-2026-11-25.md` section 1). This is clean now; keep
  the entry visible in publication.
- The history subset's 11 extra "ours" clips were chosen by reading per-clip BDs of the
  sealed set (`eval/clipset/history-set.json` `ours_extension.rule` and `data`). There is no
  sealed-log entry for that reading. Three of the chosen clips are clips that entries 18 and
  20 had touched (`hob_wm_music_ululation`, `ho_vcd2_fac675bc…`,
  `ho_epic_kitchens_p09_105`).
- BP-4's timing method was decided after timing 8 sealed clips (entries 18 and 19); no
  quality was read.
- Rule changes made after seeing results on the 84 "audited once" clips:
  - the envelope (entry 10);
  - BP-3.14 curve rules (entry 17), which moved ours' figure the unflattering way;
  - the anchor extensions.

**Fix.** Add the missing sealed-log entry for the history selection. A publication should
carry the sealed-log summary.

## 11. Documents that contradict the record (VERIFIED, disclosure)

- Procedure section 2 says x265 runs `--asm avx512` on x86_64 (`{X265_ASM}`). The record
  (x265 entry) and the command dropped it.
- The ours row of the record still says "x265 --asm avx512, below" (`benchmark-fairness.toml:107`).
- The record's `version = "BP-3.10"` while the procedure is BP-4.0.
- The rendered section 17 still lists Fixed QP (`--qp`) rows for every encoder, though BP-4
  has no Fixed QP mode.
- A reviewer reading these will assume the published runs used whichever text suits the
  attack.

## Checked and found sound

- **Our start-up lookahead is off under the record's flags.**
  - `wants_look_ahead` (`encoder.rs:1989-2001`) asks for frame 1 only when
    `first_key_lookahead.cap > 0`.
  - `rate.first_key_lookahead_off=1` makes it 0 (`rate_control.rs:218-224`).
  - So no frame-1 information reaches the first key, `key_still_boost` included.
- **The encoder never sees the clip's length.** `--frames` reaches only the CLI loop and the
  IVF header.
- **Film grain synthesis is off by default** (`nc-av1/src/grain.rs`).
- **PSNR pooling is the same per-frame mean on both scoring paths** (`score.py:189`,
  `metrics.psnr_measurement`).
- **Decoding.** Every stream, ours included, is decoded by the standard decoder.
- **SEI.** SEI is stripped from all streams.
- **libaom's aomenc main rows carry the example's settings faithfully.** Checked against
  `svc_encoder_rtc.cc:1899-2123`, including the 300 % intra cap set at 2117-2123.

## Summary of the top findings

1. **The zero line (VERIFIED, 8-14 BD-J points on every "vs x264" figure).**
   - x264 medium's default zerolatency tune is beaten by its own `~psnr` tune by 20.6 % on
     VMAF and 12.5 % on J.
   - Against `~psnr`, ours' AV1 medium goes from -39.0 % to -30.4 % BD-J, and H.264
     veryfast and faster go from beating x264 medium to losing (+11.0 % and +6.3 %).
   - The front is unchanged.
2. **The window (VERIFIED, likely the largest bias).**
   - 150 frames, one key at t=0 and a full starting buffer reward ours' 0.5-0.78 s first
     keys; the references use 0.09-0.21 s.
   - The references spend 30-90 frames below steady state.
   - Our own data puts 5-8 % of our J on the large first key.
   - Publish long-window results for every encoder.
3. **The buffer (VERIFIED, about 1 % BD-J plus a disclosure issue).**
   - BP-4 lets ours run to 1.15 buffers (about 20 % of points over one buffer) while the
     references stay inside 1.0, and x264 and x265 use about 60 % of it.
4. **AOM-CTC is in-sample (VERIFIED).** Its 12 low-delay sequences were our tuning "lens".
5. **The ladder ends where our fast rungs lose (VERIFIED on an older build).** H.264
   veryfast measured +42 % and AV1 veryfast +5 % against x264 medium at 600-4000 kbps.
   Section 16 calls this "narrowing".
6. **Uneven contest.** The slow end of the front is uncontested, the dataset holds no 1080p
   and no clip above 30 fps, and the speed builds are asymmetric (x86-64-v3 for ours only,
   no real-time-only builds for libaom and libvpx, PGO trained on superseded commands).
