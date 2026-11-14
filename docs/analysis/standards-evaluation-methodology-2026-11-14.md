# A second evaluation lens: standards-body test conditions (2026-11-14)

**Question (owner).** Is there a second, standard methodology, the kind codec-standards groups use, that we can
run on our codecs, staying as close as possible to low delay and low bitrate?

**Answer in one paragraph.** Two common test conditions (CTCs) fit. The **AOM AV2 CTC (v8) low-delay
configuration** is the primary lens. Its test sequences are public with checksums on media.xiph.org, it states
an exact low-delay command line, 130 frames per sequence, six fixed quantiser points that reach low rates,
and one metric tool (libvmaf `--aom_ctc`) that we already build. The **JVET low-delay P configuration on class
E** (the video-conferencing class) is the secondary lens. It is the condition HEVC and VVC people quote, and
all three class E sequences are public domain. Both lenses use fixed QP, so they measure coding efficiency at
equal quantiser control, not rate control. Our usual charts measure the encoder as it is used (CBR). The
sections below give what each CTC prescribes, what we cannot match, and what the harness must do.

Citations are to documents we read for this note. Where a fact comes from memory of an older document that
could not be fetched, it says so.

## 1. What the standards groups prescribe

### 1.1 JVET / JCT-VC (HEVC and VVC)

| Item | HM CTC (JCTVC-L1100, Jan 2013) | VTM SDR CTC (JVET-T2010, Oct 2020, latest SDR edition we could confirm) |
|---|---|---|
| Classes | A 2560x1600, B 1920x1080, C 832x480, D 416x240, **E 1280x720 (conferencing: FourPeople, Johnny, KristenAndSara)**, F screen content | A1/A2 3840x2160 (10-bit), B 1920x1080, C, D, **E (same three)**, F |
| Frames | the whole sequence (class E: 600 frames, 10 s at 60 fps) | the whole sequence for RA/LD; AI subsampled by 8 |
| Bit depth | 8-bit sequences coded at 8 bits (Main) | Main 10: every sequence coded at 10 bits, 8-bit inputs up-shifted |
| Configurations | AI, RA, **LD-B, LD-P** (class A not in LD, class E not in RA) | AI, RA, **LD-B, LD-P** (class A not in LD) |
| Quantiser | fixed QP 22, 27, 32, 37 | fixed QP 22, 27, 32, 37 |
| Metric | PSNR-Y, -U, -V (per frame, averaged per sequence, as HM reports) | same, plus optional reporting of others |
| BD-rate | Bjøntegaard with piecewise cubic interpolation, JVET Excel template | same template; YUV figure added in later reporting |
| Report | per class and overall: BD-rate Y, U, V, EncT, DecT (time ratios vs the anchor) | same |
| Anchor | HM of the period | VTM of the period |

What JVET low delay enables, read from the configuration files themselves:

- `HM-18.0/cfg/encoder_lowdelay_P_main.cfg` (https://vcgit.hhi.fraunhofer.de/jvet/HM, the same in HM-19.0):
  `IntraPeriod -1` (one key frame), `GOPSize 8`, all frames P, **4 active references** per frame
  (`-1 -9 -17 -25`, `-1 -2 -10 -18`, ...), **hierarchical QP offsets 1/4/5** across the GOP of 8 plus a
  QP-dependent model offset (`QPOffsetModelOffset -6.5`, `QPOffsetModelScale 0.259`), `SearchRange 64` with
  TZ search, RDOQ, SAO, `FEN 1`. No reordering: output order equals coding order.
- `VVCSoftware_VTM/cfg/encoder_lowdelay_P_vtm.cfg` (master, VTM-24.0): the same GOP of 8, 4 references and
  QP offsets, CTU 128, plus the full VVC tool set (ALF, LMCS, affine, MTS, ...).
- LD-B is the same structure with B slices that predict from two *past* pictures; we code no B slices, so
  **LD-P is the comparable configuration**.

Low bitrates. QP 22-37 is high-quality-heavy. The SDR CTC never adds QP 42/47; they appear only in individual
contributions and some exploration test conditions, always as a stated extension. We would run QP 42 and 47
as a **labelled extension** and never fold them into a CTC-labelled figure. The bitrates the four CTC points
give on class E with HM LD-P are measured in Part 2 (results page) rather than quoted from memory.

### 1.2 AOM (AV1 and AV2)

Source: AOM Codec WG, *AV2 CTC v8*, CWG-F384o, 2025-10-10, https://aomedia.org/docs/CWG-F384o_AV2_CTC_v8.pdf
(read in full for this note), and the AVM CTC framework,
https://aomedia.googlesource.com/avm/+/refs/heads/main/tools/convexhull_framework/USER_GUIDE.md.

| Item | AV2 CTC v8 |
|---|---|
| Classes | A1 4K (10-bit), **A2 1080p, A3 720p, A4 360p, A5 270p** (natural), B1 synthetic/gaming, B2 screen content, G HDR, F stills, E UGC/non-pristine (optional) |
| Sequences | all public: https://media.xiph.org/video/aomctc/test_set/ with MD5 lists per class |
| Configurations | AI, RA, **LD (§4.3, "intended for videoconferencing and remote access")**, AS (adaptive streaming), resize |
| LD classes reported | A2, A3, A4, A5, B1, B2 (Table 13); B2 reported separately, not in the overall average |
| LD settings | one key frame (frame 0), **130 frames**, zero structural delay; `--cpu-used=0 --passes=1 --lag-in-frames=0 --min-gf-interval=16 --max-gf-interval=16 --gf-min-pyr-height=4 --gf-max-pyr-height=4 --limit=130 --kf-min-dist=9999 --kf-max-dist=9999 --use-fixed-qp-offsets=1 --deltaq-mode=0 --enable-tpl-model=0 --end-usage=q --qp=x --subgop-config-str=ld --enable-keyframe-filtering=0` |
| Quantiser | qindex **110, 135, 160, 185, 210, 235** for LD (Table 12; the 8-bit scale is 0-255) |
| Threads | single thread, one tile for A4/A5; tiles allowed for A2/A3 only to speed up the simulations |
| Bit depth | A2, A4, B1 coded at 10 bits regardless of source (`--bit-depth=10`, bit shift only) |
| Metrics (§2.2, implementation §2.2.11) | PSNR-Y/U/V (overall: MSE averaged over frames), frame-averaged PSNR, APSNR-YUV (MSE weights 2/3, 1/6, 1/6), PSNR-YUV (14:1:1), PSNR-HVS-M, SSIM, MS-SSIM (both in dB, −10 log10(1−x)), CIEDE2000, VMAF and VMAF-NEG (v0.6.1), CAMBI; all from `vmaf --aom_ctc v6.0` |
| Rate | `round(file bytes · 8 · fps / frames / 1000, 6)` over the `.obu` file |
| BD-rate (§5.5) | log rate, PCHIP of log-rate as a function of quality, per metric; **BD-RateWeighted = 0.92·Y + 0.04·U + 0.04·V** (23:1:1) |
| Report (§5.1) | per sequence, per-class average (equal weight), overall average, min and max; encode and decode time (user time from `/usr/bin/time`) relative to the anchor |
| Anchor | AVM `research-alt-v1-anchor_r4.0` (§5.4) |

We read the libvmaf v3.2.1 source of `--aom_ctc` (`libvmaf/tools/cli_parse.c`, `aom_ctc_v1_0` to `v7_0`):
v6.0 is `psnr=reduced_hbd_peak=true:enable_apsnr=true:min_sse=0.5`, `ciede`,
`float_ssim=enable_db=true:clip_db=true`, `float_ms_ssim=enable_db=true:clip_db=true`, `psnr_hvs`, `cambi`,
models `vmaf_v0.6.1` and `vmaf_v0.6.1neg`, and a common bit depth. **The CTC's own metric tool is a binary we
already build**, so we use it as is and need not check our metrics code against it.

### 1.3 Real-time and conferencing practice

- **AOM RTC.** AOM has no published RTC CTC beside the LD configuration. RTC encoder work (libaom `--rt`,
  libvpx `vpx_temporal_svc_encoder`, the WebRTC tooling) is reported on rate-controlled runs, usually on the
  derf/Xiph CIF and 720p conferencing clips with PSNR and SSIM. There is no fixed condition to adopt.
- **ITU-T VCEG (H.264 era).** VCEG-N81 / VCEG-M33 style conditions used CIF/QCIF sequences, fixed QP points
  (28, 32, 36, 40 for low delay; 4 points), IPPP with one key, and the original Bjøntegaard cubic fit. These
  are historical; JVET's conditions replaced them.
- **Video conferencing evaluations** in the literature mostly use JVET class E (and the derf CIF set) under
  LD-P or LD-B. They often add extra QPs at the low end, because conferencing runs at low rates.

## 2. What fits this project

| Criterion | JVET LD-P (class E) | AOM CTC v8 LD (A3-A5) |
|---|---|---|
| Low delay | yes: no reordering; GOP-8 QP hierarchy, 4 references | yes: zero structural delay; the same idea through `subgop-config-str=ld` |
| Low bitrate | QP 37 is the floor; low end needs a labelled extension | qindex 235 is in the CTC itself and is low rate |
| Recognisable | the reference for HEVC/VVC people | the reference for AV1/AV2 people |
| Sequences we may download | class E: all three public domain (Vidyo) on media.xiph.org/video/derf; classes B/C/D are distributed to members, mostly not public (only Cactus, BasketballDrive and BQTerrace appear publicly, as 1080p) | every sequence public with MD5; licences per file (table below) |
| 8-bit | the VTM CTC codes at 10 bits; HM Main CTC at 8 bits | A3 and A5 code at native depth; A2/A4/B1 at 10 bits |
| Size of run | 3 sequences x 600 frames | 13 sequences x 130 frames |

**Recommendation.**

- **Primary: the AOM CTC v8 LD configuration on the 8-bit sequences of classes A3, A4 and A5.** It is the only
  CTC whose low-delay condition reaches low rates within the CTC itself. Its sequences are public and
  checksummed, its metric tool is pinned and already built, and 720p and below is the size range we are built
  for. A2 (1080p) and B1/B2 are left out for now (cost, and outside our 720p evaluation range). They are the
  natural next step.
- **Secondary: JVET LD-P on class E (FourPeople, Johnny, KristenAndSara, 1280x720p60, 600 frames), QP 22, 27,
  32, 37, with QP 42 and 47 as a labelled low-rate extension**, HM-19.0 `encoder_lowdelay_P_main.cfg` as the
  anchor, reported as JVET reports it (BD-rate Y, U, V and EncT). This is the table HEVC/VVC people expect,
  and the 600-frame single-key run exercises long-term reference behaviour.

### 2.1 The sequences and their licences

From each file's `_Copyright.txt` on media.xiph.org (read 2026-11-14). We download on the rented instance only,
never redistribute, and record the URL and the MD5 in the results.

| Class | Sequence (frames, fps) | Licence (summary of the file) | Used |
|---|---|---|---|
| A3 720p | Johnny (60), KristenAndSara (60) | public domain (Vidyo) | yes |
| A3 | ControlledBurn (30), WestWindEasy (30) | NTIA/ITS, research use, attribution kept | yes |
| A3 | Vidyo3, Vidyo4 (60) | Vidyo: "only for developing, testing and promulgating technology standards" | **no**: our use is product evaluation, not standards work |
| A3 | DrivingPOV, RollerCoaster | 10-bit sources | no (8-bit project) |
| A4 360p | BlueSky_v2 (25), Stockholm (59.94) | TU München (Taurus Media) / SVT 2001 set: "Restrictions of use: No restrictions" | yes |
| A4 | RedKayak, SnowMountain, SpeedBag, TouchdownPass (29.97) | NTIA/ITS research use | yes |
| A5 270p | FourPeople (60) | public domain (Vidyo) | yes |
| A5 | ParkJoy (50) | SVT 2006 multi-format set: "only ... for developing, testing and presenting technology standards" | **no**, as Vidyo3/4 |
| A5 | Vertical_Bayshore 270x480 (29.97) | CC BY-SA 4.0 (James Leslie Skinner) | yes |
| A5 | SparksElevator | 10-bit source | no |
| JVET E | FourPeople, Johnny, KristenAndSara 1280x720p60, 600 frames (`media.xiph.org/video/derf/y4m/`) | public domain (Vidyo) | yes (secondary) |

That is 12 primary sequences (4 + 6 + 2) and 3 secondary ones. Every one is 8-bit 4:2:0 at its native depth.
Whether the research-use terms (NTIA, SVT, TUM) cover a company's internal evaluation is a judgement call. We
take them to cover it, because nothing is redistributed and the use is measurement. The owner may decide
otherwise, and the run can drop those sequences without changing the method.

## 3. What we cannot match, and how the results must be read

| Deviation | Why | Effect on reading |
|---|---|---|
| **No B slices, no multi-reference GOP-8 structure in the CTC's form.** Our fixed-QP mode is IPPP with one key, a key QP offset (`nc-encoder/src/key_qp.rs`) and our own QP cadence (`rate.cadence_fixed_qp`, a periodic QP sawtooth like the CTC's offsets but not its exact values). | project design (low delay, real-time subset) | the CTC anchors keep a structural advantage (4 refs, tuned QP hierarchy). This is a real gap, not noise |
| **Real-time subset vs exhaustive reference encoders.** HM, VTM and libaom `--cpu-used=0` search exhaustively by design and run 100-10,000x slower than real time. | that is what reference software is for | BD-rate against them measures distance to the format's practical ceiling at CTC settings, not competitiveness. The fair yardsticks are the real-time encoders below |
| **8-bit coding everywhere.** | project decision | A3/A5 match the CTC (native 8-bit). The A4 class differs: the CTC codes A4 at 10 bits. We code A4 at 8 bits for every encoder, ours and the anchors, and say so |
| **Quantiser scales differ by codec.** H.264/HEVC/VVC take QP; AV1/VP9/AV2 take qindex. | different standards | each codec runs its own CTC points: QP 22-37 (+42/47) for QP codecs, qindex 110-235 for the AOM family (libaom/SVT/libvpx 0-63 scales via their own `quantizer_to_qindex`: 28, 34, 40, 46, 52, 59). BD-rate integrates only the overlapping quality range (PCHIP, no extrapolation) and flags small overlaps, as our `nceval.bdrate` already does |
| **Container.** The CTC counts `.obu` bytes. | our AV1 writes IVF | we count packet bytes without IVF headers (the `.obu` equivalent), and Annex-B bytes for H.264/HEVC/VVC as HM does |
| **One run, one machine for time.** | | encode time is single-thread user time on the cloud instance, as a ratio to the anchor (CTC §5), not our Jetson figures |

Anchors, two groups, pinned by version and command line in the results:

1. **Standards yardsticks** (what a CTC reader expects): HM-19.0 LD-P, VTM-24.0 LD-P, libaom v3.15.3
   `--good --cpu-used=0` with the CTC LD flags that exist in libaom (the AV1 analogue of the CTC; the AVM anchor
   itself codes AV2, and AVM at speed 0 is out of budget).
2. **Practical yardsticks** (real-time encoders at matched fixed QP, low-delay, one thread): x264 medium
   zerolatency, x265 medium zerolatency, SVT-AV1 v4.3.0 `--rtc` M10, libaom rt cpu 8, libvpx-vp9 rt cpu 7,
   openh264 v2.7.0, kvazaar v2.3.2 (LP GOP), VVenC v1.14.0 faster (low delay). Command lines are those in
   `eval/nceval/encoders/` (unchanged) plus new adapters for HM, VTM, VVenC and libaom-good.

Our encoders: every codec at its Default and Fast presets, fixed QP (`--qp`, one key frame,
`--intra-period 9999`). As a labelled extra, our normal CBR mode at the rates our fixed-QP points reached.

## 4. What changes in our tooling

- `nanocodec encode --qp` already gives a fixed-quantiser path for every codec (`crates/nc-cli/src/main.rs`:
  native QP for H.264/HEVC/VVC, `base_q_idx` for AV1/VP9/AV2). No encoder change is needed.
- A new module `eval/nceval/ctc/` with its own sequence list (URLs, MD5s, licences), the two configurations
  (frames, QP points), anchor adapters for HM/VTM/VVenC/libaom-good, a runner that encodes, decodes with the
  standard decoder, checks our streams' per-frame hashes against our reconstruction, scores with
  `vmaf --aom_ctc v6.0` (plus the VMAF v1 model for our J), and a report writer for the CTC tables. It reuses
  `nceval.bdrate` (already PCHIP over log rate, overlap only, the AOM §5.5 method) and the existing adapters.
  It does not touch the screens or charts.
- Per-plane PSNR "the CTC way": the overall PSNR from the frame-averaged MSE (AOM §2.2.1) is the primary
  figure. The frame-averaged PSNR (what HM prints) is reported for the JVET table. Both come from libvmaf's
  per-frame output, so no metric code of ours is involved.
- The sequences are sealed: registered in `SEALED_SETS` (`eval/nceval/pool.py`) so no screen or tuning
  study draws them. They are an evaluation lens only.
- Cost: about 13 x 6 QP x ~25 configurations ≈ 2,000 short encodes plus 3 x 6 x ~10 long ones. The slowest
  single jobs (VTM and libaom speed 0 at 720p) take minutes, so one spot c7a session fits well inside the
  $25 cap.

## 5. Changes after this note was first landed (same day)

- **Licence tiers.** Each sequence carries a tier (`free`: public domain, CC BY-SA or "no restrictions";
  `research`: NTIA/ITS research-use). The owner chose to include both in the main tables, and the free-only
  figure is shown beside them. ParkJoy was dropped: its notice limits use to standards work, like Vidyo3/4.
- **Run time.** Every run must finish within 30 minutes (owner rule). VTM-24 and libaom speed 0 take about 4 s
  per 270p frame on our PC, so they run on classes A4/A5 only; A3 has HM as its standards anchor.
  **JVET class E codes the first 150 frames, not the CTC's 600.** That is a stated deviation; the full
  600-frame run remains an option.
- **Pool overlap.** KristenAndSara was already in the screening pool (an excerpt of the same Xiph file). It
  was screened on before this lens existed. Sealing the lens's families removes it from the pool from now on.
