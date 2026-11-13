# Rate-control and settings audit (2026-11-13)

This audit asks whether any of our published BD-rate gaps come from how rate is counted, from buffer behaviour,
from frame handling, or from the settings we chose for the reference encoders. Unfavourable findings come first.
Every number says what it was measured on.

- **Chart clips:** the 11 natural VAL clips behind the published chart ([bd-vs-speed.md](bd-vs-speed.md)), on
  the chart's cached points (each family on its recorded build). The audit's extra rows are our Default of each
  codec at build cba4292f (the encoder sources of the verified head e1c3d232), which reproduces the chart's BD-J
  exactly (AV1 −50.87, H.264 −23.04).
- **Held-out clips:** the 209 clips of the independent test ([independent-2026-11-13.md](independent-2026-11-13.md)):
  71 screening-pool clips and 138 never-used sources.
- **Common settings:** 150 frames from frame 0, one key frame, low-delay CBR on the real-use ladder with a 1 s
  buffer, one thread.
- **Notation:** BD values are % bits for the same quality, so negative means fewer bits. The interval is a 95%
  bootstrap over clips, resampled within each class. "J" is BD-J, the mean of PSNR-YUV, VMAF v1 and MS-SSIM.

The tables are in CSV beside this page's working data (`tools/dev/independent_report.py` writes them).
This page quotes them.

**The held-out run confirms the chart-clip findings and makes the buffer finding larger.** On the 209 held-out
clips, 24–43% of our ladder points per preset peak above one buffer (5–12 of 42 on the chart). Our strict
settings cost 6.8–9.1% bits there (4.7–6.9% on the chart). Held-out numbers are quoted beside the chart's
below. Held-out tables: [independent-2026-11-13/](independent-2026-11-13/) (`heldout-*.csv`); chart tables:
`chart-*.csv`.

## 1. Reference settings: x264 and x265 lose bits to their psycho-visual defaults (unfavourable)

The chart's anchor is x264 `medium` with `--tune zerolatency` and x264's psycho-visual defaults: psy-rd and AQ.
Those defaults trade metric scores for perceived detail. Our encoders have no psycho-visual mode; their parameters
are tuned to J's three metrics. The like-for-like comparison is therefore against metric-tuned references.

Method: every reference's own metric-oriented settings at the pinned versions, checked against its `--help` and
log, with every other chart setting unchanged. Chart clips, ladder plus the chart's low rungs. BD of the
alternative vs that encoder's chart setting:

| Encoder (version) | Chart setting | Alternative | BD-J | PSNR-YUV | VMAF v1 | MS-SSIM | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| x264 0.165 medium | `--tune zerolatency` (psy-rd, AQ on) | `--tune zerolatency,psnr` | −9.8 [−12.0, −7.6] | −8.3 | −19.4 | −1.8 | better: the chart understates x264 |
| | | `--tune zerolatency,ssim` | −1.4 [−2.8, −0.0] | −1.3 | −0.2 | −2.8 | slightly better |
| x265 4.2 medium | `--tune zerolatency` (psy-rd 2.0, AQ 2 at 1.0) | `--tune psnr` + low-delay flags | −5.9 [−7.4, −4.4] | −9.4 | −9.9 | +1.5 | better: the chart understates x265 |
| | | `--tune ssim` + low-delay flags | −1.9 [−2.8, −1.0] | −3.9 | −0.9 | −0.8 | slightly better |
| SVT-AV1 v4.3.0 | `--rtc`, tune 1 (PSNR) in effect | none in this mode: tune 0 is forced back to 1 in low delay, tunes 2–5 are CRF/random-access only | | | | | already metric-tuned |
| libaom v3.15.3 rt | `--tune=psnr` (default), aq 0 | `--tune=ssim` (cpu 8 / 10) | +0.6 / +1.2 | +2.4 / +2.5 | +7.2 / +8.0 | −7.7 / −7.0 | chart setting is the better one; vmaf tunes are not built (CONFIG_TUNE_VMAF) |
| libvpx v1.16.0 rt | `--tune=psnr` (default), aq 0 | `--tune=ssim` (cpu 7 / 9) | −0.3 / −1.0 [−1.7, −0.3] | +0.3 / −0.4 | +5.1 / −0.1 | −6.3 / −2.4 | within noise at cpu 7; 1% better at cpu 9 |
| openh264 v2.7.0 | adaptive quant off | adaptive quant on | 0.0 | 0.0 | 0.0 | 0.0 | no effect: byte-identical streams at our settings |
| NVENC (RTX 5090) p4 `-tune ll` | spatial AQ off | `-spatial-aq 1` (H.264 / HEVC / AV1) | +14.3 / +8.3 / +8.7 | +15.7 / +10.5 / +12.2 | +23.5 / +19.6 / +15.2 | +3.7 / −5.3 / −1.1 | chart setting is the better one |

With `--tune psnr` our numbers against the anchor shrink:

| Ours, Default | vs x264 medium (chart) | vs x264 medium `--tune psnr` |
|---|---:|---:|
| AV1 | −50.9 [−56.5, −45.3] | −43.6 [−49.0, −38.1] |
| HEVC | −49.3 | −41.7 [−47.3, −36.0] |
| VVC | −54.0 | −47.1 [−52.5, −41.7] |
| AV2 | −53.7 | −46.8 [−52.4, −41.3] |
| VP9 | −48.1 | −40.2 [−46.0, −34.4] |
| H.264 | −23.0 [−30.5, −15.3] | −13.3 [−20.7, −5.4] |
| H.264 Fast / Faster / Fastest | −16.2 / −8.7 / −1.0 | −5.8 [−13.5, +2.1] / +2.5 [−5.0, +9.8] / +10.7 [+3.8, +17.6] |

**Verdict:** our BD-J against the chart's anchor is about 7–10 points more favourable than against x264 tuned for
PSNR. H.264 Faster does not beat a PSNR-tuned x264 medium on these clips, and H.264 Fastest needs more bits
(+10.7%). **Disclose and fix:** the chart should show x264 and x265 at their best metric tune. SVT-AV1, libaom,
libvpx, openh264 and NVENC already run at their metric-oriented setting, or an alternative made no material
difference (libvpx cpu 9: 1%).

On the 209 held-out clips, BD-J against x264 medium (chart setting):

- x264 medium `--tune psnr` −9.8 [−10.8, −8.7]; `--tune ssim` −1.2.
- x265 medium −32.3 with `--tune psnr` and −30.8 with `--tune ssim`, against −29.2 at its chart setting.
- libaom cpu 8: −23.6 with `--tune=ssim` vs −23.8.
- libvpx cpu 7: +1.4 with `--tune=ssim` vs +2.1. Against x264 medium
`--tune psnr`, our Defaults are −37 to −43 there (H.264 −15.6; [independent page](independent-2026-11-13.md)).

**Other settings of the references** (held-out 30-clip extras subset; BD of the variant vs the chart's setting):

| Encoder | Variant | BD-J | Note |
|---|---|---:|---|
| x264 medium | lookahead and B-frames (no `--tune zerolatency`) | −29.7 [−32.6, −26.6] | not low delay: an upper bound |
| x264 medium | 4 sliced threads (`--threads 4`) | +17.8 [+14.9, +20.6] | our single-thread setting favours x264 |
| x265 medium | lookahead and B-frames | −26.5 [−29.3, −23.7] | not low delay |
| x265 medium | 4-thread pool + WPP | +7.4 [+6.3, +8.5] | single thread favours x265 |
| SVT-AV1 M8 / M12 | `--rtc` with its default low-delay hierarchy (temporal layers) | +3.5 / +4.5 | flat P (the chart's) favours SVT |
| libvpx VP9 rt 7 | WebRTC rate control (cyclic-refresh AQ, max-q 52, ±50% under/overshoot) | −12.1 [−14.6, −9.6] | but it misses its targets (median 1.64× target, up to 29×; 70 of 107 points rate-invalid): not a valid CBR comparison |
| libaom rt 8 | the same WebRTC settings | −11.8 [−14.4, −9.3] | the same rate failure (median 1.29× target, up to 25×; 61 of 107 points rate-invalid) |
| libvpx VP9 cpu 2 | `--good` (one pass, no lag) | −36.2 vs x264 medium (rt 7: −1.5) | not a real-time speed |
| openh264 | AQ on | 0.0 | byte-identical streams |

The low-delay, single-thread chart settings favour x264, x265 and SVT-AV1 over their multi-threaded or
hierarchical alternatives. The real-time-mode alternatives that score better (lookahead, `--good`) are not low
delay or not real-time speed. The WebRTC settings gain BD only by missing their rate targets.

x265's real `--tune psnr` / `--tune ssim` replace `zerolatency` (x265 keeps one tune). Explicit `--no-cutree
--bframes 0 --rc-lookahead 0 --no-scenecut --frame-threads 1` restore the rest. On a test clip its full settings
log differs from "zerolatency + psy-rd 0 + psy-rdoq 0 + aq-strength 0" only in b-adapt, which is inert with no
B-frames. The decoded pictures and stream sizes are identical; the bytes differ only in x265's options SEI.

## 2. Buffer: our first key frame borrows up to 1.4 buffers, and our points rely on the rate-valid allowance (unfavourable)

Replay of every stream through the same leaky bucket: 1 s at the target rate, starting empty, drained every frame
(`cbr_stats.replay_buffer`). Chart clips, 42 ladder points per family:

| Family | median peak (buffers) | max peak | points above 1 buffer | points above 1.5 | mean s in violation | first frame (buffers, median / max) |
|---|---:|---:|---:|---:|---:|---:|
| ours, every preset | 0.65–0.78 | 1.09–1.40 | 5–12 | 0 | 0.02–0.16 | 0.46–0.60 / 0.69–1.23 |
| x264 medium | 0.15 | 2.11 | 2 | 2 | 0.18 | 0.14 / 0.32 |
| x264 ultrafast / veryfast, x265 (all) | 0.16–0.28 | 0.45–0.65 | 0 | 0 | 0.00 | 0.14–0.24 / 0.36–0.51 |
| SVT-AV1 M8–M13 | 0.17–0.26 | 2.21–2.35 | 1–2 | 1 | 0.06–0.09 | 0.10–0.12 / 0.23–0.25 |
| libvpx VP9 rt 7 / 9 | 0.42 | 12.9 / 13.6 | 11 / 13 | 9 / 10 | 0.92 / 1.06 | 0.35 / 0.66 |
| libaom rt 8 / 10 | 0.47 | 9.3 / 9.6 | 6 | 4 | 0.5 | 0.36 / 0.74 |
| openh264 | 0.75 | 22.5 | 17 | 15 | 1.62 | 0.66 / 2.77 |

Held-out clips, 778 ladder points per family:

| Family | median peak | max peak | points above 1 buffer | above 1.5 | mean s in violation | first frame (buffers, median / max) | rate-valid | valid only by allowance | strict-valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours, every preset | 0.72–0.92 | 1.26–1.44 | 189–335 (24–43%) | 0 | 0.13–0.42 | 0.48–0.76 / 1.26–1.40 | 778 | 190–335 | 214–467 |
| x264 medium | 0.21 | 2.30 | 6 | 3 | 0.03 | 0.16 / 0.70 | 775 | 12 | 732 |
| x265 (all presets) | 0.34 | 1.15–1.37 | 7–10 | 0 | 0.02–0.03 | 0.23 / 1.08 | 778 | 17–19 | 680–689 |
| SVT-AV1 M8 / M12 / M13 | 0.28–0.35 | 5.4–9.3 | 12–16 | 5–7 | 0.05 | 0.11–0.12 / 0.42–0.66 | 771–773 | 13–31 | 680–687 |
| libvpx VP9 rt 7 / 9 | 0.45–0.46 | 25–26 | 170 / 174 | 123 / 131 | 0.90 / 0.95 | 0.36–0.38 / 1.1 | 655 / 647 | 82 / 71 | 536 / 531 |
| libaom rt 8 / 10 | 0.47–0.48 | 17.6 / 18.1 | 96 / 101 | 58 / 64 | 0.46 | 0.36–0.38 / 1.4 | 720 / 714 | 53 / 60 | 602 / 589 |
| openh264 | 0.85 | 48.1 | 336 | 244 | 1.72 | 0.77 / 6.64 | 534 | 93 | 439 |

**The chart's "rate-valid" rule** (`realuse.rate_valid`) counts a point as valid when:

- its bucket peaks at most 1.5 buffers, and
- its average overshoots by at most 10% plus buffer/window (+30% at 150 frames, 30 fps).

For ours, the overshoot is loan-adjusted.

**Our rate-valid points.** All 42 of each of our presets are rate-valid. 5–13 per preset are valid only because of
that allowance (peak above 1 buffer). Reference encoders have 0–5 such points; libvpx, libaom and openh264 have
more outright-invalid points (4–15) than any of ours (0). A rate-invalid point stays on its curve against
external anchors: the chart counts it and does not drop it.

**A strict rule for everyone.** The same rule applied to all encoders: peak at most 1.0 buffer, raw average at
most +5% over target. Only 22–31 of our 42 points per preset pass it, against 38–41 for x264/x265/SVT-AV1. Dropping
failing points shrinks the curves until BD exists on only 4–6 of 11 clips, so this rule does not give a usable BD
for us.

**Our strict settings instead** (`rate.recode_overflow=1`, `rate.recode_overflow_inter=1`,
`rate.key_cap_buffer=1`, `rate.key_repay_seconds=1`):

- 45–49 of 50 points pass the strict rule (27–33 at default settings).
- Cost vs our default, J: H.264 +4.7% [+3.3, +6.2], HEVC +5.8, AV1 +6.4 [+4.2, +8.4], VP9 +5.5, AV2 +6.1,
  VVC +6.9 [+4.6, +9.1].
- BD-J vs x264 medium moves from −23.0 to −20.7 (H.264), −49.3 to −47.6 (HEVC), −50.9 to −49.0 (AV1),
  −48.1 to −46.5 (VP9), −53.7 to −52.1 (AV2), −54.0 to −52.3 (VVC).

On the held-out clips (30-clip extras subset, class round robin), the strict settings cost more:

- vs our own default, J: H.264 +7.3% [+5.5, +9.1], HEVC +7.5, AV1 +7.2 [+5.3, +8.9], VP9 +7.4, AV2 +6.8,
  VVC +9.1 [+7.4, +10.8].
- BD-J vs x264 medium on those clips moves from −29.0 to −25.6 (H.264), −48.1 to −45.1 (HEVC), −47.2 to −44.1
  (AV1), −47.8 to −44.5 (VP9), −50.5 to −47.6 (AV2), −53.4 to −49.7 (VVC).
- Applying the strict rule to the points instead, for everyone alike, leaves ours a BD on only 40–99 of 209
  clips. On those clips BD-J changes by under 1.6 points: AV1 −46.2 → −45.9, H.264 −20.6 → −19.0.

**Verdict:** about 2 points of each published BD-J on the chart clips, and 3–4 points on the held-out clips (≈7%
of bits), come from letting the first key frame exceed a strict 1 s buffer. **Disclose:** the README states the 1.4-buffer loan; it should also state its 2–4 point BD effect, or
publish the strict numbers beside the default ones.

**Front-loading (key-frame pulse).** On chart clips our first frame takes a median 8–10% of the stream's bits,
against 2–3% for x264, x265 and SVT-AV1. Its PSNR-Y is +2.3 to +4.9 dB above our steady state; x264 medium's is
2.7 dB below its own.

At matched rate, ours vs x264 medium:

- The first-key pulse is 0.5–2.4 dB larger.
- BD on the worst-1% (4-frame window) curves is −25% to −57%: our worst frames are no worse than x264's at the same
  rate.

Held-out clips, matched rate, ours vs x264 medium:

- Worst-window BD (PSNR-Y / VMAF): −20 to −51% / −35 to −58%.
- The first-key pulse is −0.03 to +0.69 dB larger than x264's.
- Our first frame takes a median 8–11% of the bits; x264's takes 3%.

A 150-frame window with one key rewards investing in the first key; [section 3](#3-window-length-our-gaps-shrink-over-600-frames-unfavourable) tests a longer
window.

## 3. Window length: our gaps shrink over 600 frames (unfavourable)

The chart scores 150 frames (5 s) with one key frame. Our controller spends heavily on that first key and repays
the loan over 10 s, so a 5 s window may favour it.

We encoded the 5 clips of the `long` set (static, crowd, pan, two body-cam segments) at 150 frames and at 600
frames. At 600 frames there are keys at frames 0 and 300 (the chart's 10 s key period). Ours ran at cba4292f;
x264 medium at the chart setting and with `--tune psnr`. Same ladder and rules otherwise. Three of these clips are
TRAIN clips, so this set is not independent. With one or two clips per class the bootstrap intervals are not
meaningful, so none are given.

| Ours vs x264 medium (chart setting) | 150 frames | 600 frames | change |
|---|---:|---:|---:|
| AV1 Default BD-J | −54.7 | −50.6 | +4.1 |
| HEVC Default BD-J | −52.2 | −46.7 | +5.5 |
| H.264 Default BD-J | −24.5 | −15.9 | +8.6 |
| vs x264 medium `--tune psnr`: AV1 / HEVC / H.264 | −46.6 / −43.7 / −13.1 | −42.0 / −37.6 / −4.1 | +4.6 / +6.1 / +9.0 |

Over 600 frames:

- Our achieved rate is on target (median 0.998–1.000 of target); at 150 frames it is 1.047–1.062.
- Our first frame's share of the stream falls from 8–10% to 2–3%.
- x264's rate rises from 0.955 to 0.997 of target.

**Verdict:** on these 5 clips our BD-J against x264 is 4–9 points less favourable over 20 s than over 5 s; H.264
loses most. Part of the published gap is a start-up effect of a single-key 5 s window. **Disclose:** state the
window length beside every headline number. A longer-window chart would be the fairer headline for streaming
use; the 5 s figure suits short sessions and the start of a stream.

## 4. Our rate overshoots its target over 150 frames; x264 and x265 undershoot (unfavourable framing, neutral BD)

Achieved rate over target, real-use ladder points (42 per family), chart clips:

| Family | median | p90 | max | loan-adjusted median | points > +10% | points < −10% |
|---|---:|---:|---:|---:|---:|---:|
| ours, every preset | 1.029–1.049 | 1.06–1.14 | 1.11–1.19 | 0.987–1.000 | 3–6 | 0 |
| x264 medium | 0.908 | 1.038 | 1.41 | — | 3 | 19 |
| x264 ultrafast / veryfast | 0.956 / 0.897 | | | | 0 | 12 / 21 |
| x265 ultrafast / superfast / medium | 0.968 / 0.960 / 0.966 | | | | 0 | 8–11 |
| SVT-AV1 M8–M13 | 0.998–1.001 | 1.03–1.07 | 1.43–1.46 | | 1–2 | 1–2 |
| libvpx VP9 rt 7 / 9 | 1.011 / 1.013 | 1.67 / 1.84 | 3.57 / 3.72 | | 14 | 0 |
| libaom rt 8 / 10 | 1.024 / 1.019 | 1.23 / 1.25 | 2.86 / 2.91 | | 6–7 | 0 |
| openh264 | 1.002 | 2.67 | 5.49 | | 11 | 0 |

Held-out clips (778 ladder points per family) give the same picture:

| Family | median | p90 | max | loan-adjusted median | points > +10% | points < −10% |
|---|---:|---:|---:|---:|---:|---:|
| ours, every preset | 1.037–1.053 | 1.06–1.09 | 1.18–1.26 | 0.990–1.011 | 23–40 | 0–5 |
| x264 medium / `--tune psnr` | 0.916 / 0.921 | 1.02 | 1.43 / 1.44 | — | 14 / 11 | 319 / 296 |
| x265 medium / `--tune psnr` | 0.989 / 0.984 | 1.05 | 1.15 / 1.14 | | 9 / 10 | 75 / 90 |
| SVT-AV1 M8 / M12 / M13 | 1.003 / 1.007 / 1.009 | 1.05–1.06 | 2.54 / 1.88 / 1.79 | | 34 / 8 / 9 | 26 / 10 / 10 |
| libvpx VP9 rt 7 / 9 | 1.015 / 1.016 | 1.37 / 1.44 | 4.36 / 5.17 | | 187 / 182 | 0 |
| libaom rt 8 / 10 | 1.021 / 1.023 | 1.12 / 1.14 | 3.52 / 3.49 | | 93 / 103 | 0 |
| openh264 | 1.002 | 1.70 | 6.07 | | 175 | 16 |

Our controller lends the first key frame extra bits and repays them over `rate.key_repay_seconds`, 10 s by
default. A 5 s clip ends with part of the loan unpaid, so our streams land 3–5% over target. x264 and x265 land
3–10% under it.

**Every BD uses the achieved bytes**, for every encoder alike, so the overshoot moves our points right on the
rate axis and is paid for there. We confirmed this in code:

- `cbr.Bench.rate_kbps_from_sizes` sums the packet sizes ffprobe reports.
- `jreal.curve_bd` integrates over `kbps`, the achieved rate, not over targets or loan-adjusted rates.

**Containers.** IVF file and frame headers (ours for AV1/VP9/AV2, and libvpx, libaom, SVT) are not counted for
anyone. Annex-B start codes, parameter sets and SEI (ours for H.264/HEVC/VVC; x264, x265, openh264, NVENC) are
counted for everyone. x264's options SEI (~0.6 kB, once) therefore counts against x264. That is about 1 kbps at
150 frames, below 1% of the lowest rung.

**Verdict:** no BD advantage from counting. The loan-adjusted rate is used only in the chart's "rate-valid" label
(section 2), not in any BD.

## 5. Rate ladder: the low rungs favour everyone against x264; higher rates shrink every gap

**The chart's low rungs.** The chart adds 40 and 75 kbps rungs (720p30-scaled) on clips where some encoder's curve
does not overlap x264's on the ladder. Those are mostly static and body-cam clips, where x264 medium is weakest.
BD-J under that rule vs the ladder alone (the ladder alone leaves fewer clips with a BD):

| | chart clips: rule / ladder only (clips) | held-out: rule / ladder only (clips) |
|---|---:|---:|
| AV1 Default | −50.9 / −45.3 (7) | −44.8 / −42.0 (166) |
| H.264 Default | −23.0 / −17.3 (9) | −25.1 / −22.7 (200) |
| VVC Default | −54.0 / −49.0 (8) | −49.6 / −45.6 (167) |
| SVT-AV1 M12 | −34.9 / −27.8 (10) | −24.0 / −21.6 (207) |
| x265 medium | −40.7 / −35.3 (10) | −29.2 / −27.9 (206) |
| x264 medium `--tune psnr` | | −9.8 / −10.8 (209) |

The rule moves the reference encoders about as much as ours: 1–7 points for everyone except x264 itself. It is
not an advantage particular to our encoder. The headline numbers rest on it, though. **Disclose:** without the low
rungs the chart's BD-J would be 4–7 points less negative for every encoder, and some clips would have no BD.

**Higher rates.** On the 30-clip extras subset we added 2,400 and 4,000 kbps rungs (720p30-scaled, within half
the source rate; only 10–11 clips have room for them):

| BD-J vs x264 medium | chart rule (30 clips) | ladder only | ladder + high rungs | upper range: top two ladder rungs + high (10–11 clips) |
|---|---:|---:|---:|---:|
| H.264 | −29.0 | −25.1 | −23.2 | −9.3 [−13.5, −5.4] |
| HEVC | −48.1 | −44.5 | −41.0 | −26.9 [−30.4, −23.4] |
| AV1 | −47.2 | −43.5 | −39.9 | −24.8 [−27.9, −21.8] |
| VP9 | −47.8 | −42.1 | −40.6 | −22.5 [−24.2, −20.1] |
| AV2 | −50.5 | −44.1 | −41.9 | −26.9 [−31.7, −22.1] |
| VVC | −53.4 | −47.3 | −45.3 | −30.4 [−31.8, −28.6] |
| x265 medium | −30.8 | −29.1 | −27.4 | −22.6 [−27.1, −18.0] |
| SVT-AV1 M8 / M12 | −39.4 / −28.2 | −35.4 / −24.6 | −34.5 / −22.3 | −19.0 / −4.2 |
| libaom rt 8 | −27.1 | −26.7 | −25.5 | −12.4 |
| x264 medium `--tune psnr` | −10.0 | −11.2 | −11.1 | −8.7 |

**Verdict:** our gaps are largest at the low rates the chart targets. Above ~1 Mbps at 720p30 they roughly halve,
and the references' gaps shrink too. The upper-range column covers only 10–11 clips. **Disclose:** "at low
rates" belongs in every headline sentence.

## 6. Speed is timed at fixed QP; CBR costs about 2% more CPU (small, unfavourable)

The chart's fps comes from fixed-QP encodes rate-matched to the 600 kbps rung. Paired timing on one AWS
c7i.metal-24xl (Xeon Platinum 8488C) used the PGO build at cba4292f, single thread, pinned, 5 rotated rounds. It
ran 3 clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1), CBR at the rung vs the plan's fixed QP. CBR/QP
CPU time ratio per preset, median of 3 clips: 0.997–1.071, median 1.023 over our 18 presets (HEVC Default
1.071, HEVC Faster 1.036, AV1 Default 1.035).

**Verdict:** the chart's fps overstates CBR-mode speed by about 2% (up to 7% for HEVC Default). **Disclose.**

## 7. Frames: no holds, no drops, no skipped frames on the chart (neutral)

| | ours | references |
|---|---|---|
| input frames / coded frames, chart, every preset | 6,300 / 6,300 per family (0 held) | 6,300 / 6,300 (packet count is checked: a reference that dropped a frame would fail the point) |

**Frame hold.** Our frame hold (on by default since 2026-11-11) held 0 frames on all chart points. On the held-out
set: at most 0.25% of input frames per preset were held (238–301 of 116,572), in 14–17 of 778 streams, on 12 of 209
clips. Those sources repeat frames: CCTV and CEVO captures, MEVA, CHIRLA, and the 25→30 fps-retimed SVT clips.
On those 12 clips, our Default of each codec with hold off (`hold.enable=0`) vs on, J: H.264 +0.7%
[+0.2, +1.2], HEVC +0.3, AV1 +2.9% [+0.7, +6.6], VP9 −0.1, AV2 +0.2, VVC +0.1. Across all 209 clips that is below
0.2 points of BD-J (an estimate from the 12 clips' share; not measured on all clips).

**Scoring rules.** A held frame would be scored against its own source frame, with the previous picture repeated,
as a player shows it (`framepipe.HeldFrameSink`). Fps is counted per input frame. No reference drops frames:
openh264 runs with frame skip off, and every reference stream must have one packet per input frame or the point
fails.

**Verdict:** no advantage from frame handling on the chart. On the held-out clips, hold gains up to 3% (AV1) on
the 12 clips whose sources repeat frames; each held frame is scored against its own source frame.

## 8. Reconstruction vs decode (neutral, with one tooling caveat)

Chart points are scored on our encoder's reconstruction. We re-encoded 144 chart points: 6 codecs' Default × 11
clips × 2 rungs, plus low rungs (24 per codec). Each had the full gate: both standard decoders (JM/ffmpeg,
HM/ffmpeg, aomdec/dav1d, vpxdec/ffmpeg, avmdec/dav2d, VVdeC/ffmpeg) and ours had to equal the reconstruction frame
by frame.

- **All 144 passed.** The re-encoded streams have exactly the cached points' per-frame sizes, and PSNR-YUV and
  MS-SSIM are identical.
- **VMAF v1:** identical on 136. On 8 the sequence mean moved by 0.0007–0.037 points, and single frames by up
  to 5.5 points, on bit-identical pictures. That is GPU nondeterminism in VMAFx, not a decode difference.

On the held-out run every one of our points is scored from a standard decoder's output whose frames equal the
encoder's reconstruction digests. All 18,318 of our held-out points were decoded by the first standard decoder of each codec: ffmpeg's libavcodec
for H.264/HEVC/VP9/VVC, dav1d (through ffmpeg) for AV1, avmdec for AV2. Every frame equalled the encoder's
reconstruction digest; the second standard decoder was not run on the held-out set.

**Verdict:** "every stream we report decodes bit-exactly in the standard decoders" holds on this sample and on
the whole held-out run. The README should add that chart scores come from the reconstruction, proven equal by
sampling and by the landing conformance check.

### Metrics: GPU vs CPU

The held-out run was scored in the cloud by stock libvmaf v3.2.1 on the CPU (model `vmaf_v1.0.16_3d0h.json`)
and by `nceval.msssim`. Every earlier number came from the GPU path (VMAFx).

We drew 200 held-out points with a fixed hash: 31 families, ours and references. Each was scored again here,
from the same streams, by the GPU path in a scratch work root:

| Metric | max abs difference | mean abs difference |
|---|---:|---:|
| PSNR-YUV (dB) | 7.1e-15 | 8.9e-17 |
| VMAF v1 (sequence mean) | 0 | 0 |
| MS-SSIM | 3.9e-6 | 1.7e-7 |

All 200 had identical per-frame packet sizes. **Verdict:** the CPU and GPU paths agree at the precision we
store, so the held-out numbers are comparable with the chart's. (The GPU path's own run-to-run nondeterminism on
the chart sample (section 8, above), is a separate effect, seen on 8 of 144 points.)

## What this audit cannot show

- **Clips:** the chart uses 11 clips and the held-out set 209. Neither is a random sample of all camera video.
- **One mode:** every number is low-delay CBR, one thread, 150 frames, one key frame.
- **Not measured:** our encoder in other modes (VBR, multi-pass, lookahead), sessions longer than 20 s
  (section 3), packet loss, and subjective quality.
- **"Best settings":** the metric-tuned reference settings are the best we found among each encoder's
  documented metric tunes. They are not an exhaustive search of each encoder's settings.
