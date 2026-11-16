# Fixed QP charts: speed timed on the charts' own encodes (2026-11-16)

The results site's Fixed QP chart used to take its speed axis from the CBR chart's timing on 3 private chart clips.
Its quality axis came from the 12 public AOM CTC sequences. The two axes therefore described different content.

Both axes now come from the same encodes:

- **AOM CTC view.** Every encoder on the chart was timed on the fixed-QP lens's own command lines: the same 12
  sequences, 130 frames and quantizers.
- **Held-out view (new).** Quality and speed come from one fixed-QP run on 12 clips of the CBR chart's held-out,
  fresh set.

Our encoders are variance-integration b11b2f83 throughout. The x86 binary is stamped 67ed090b: b11b2f83 plus a commit
that touched only eval/. The Jetson binary is the chart build of b11b2f83.

## Method

- **Tool.** `tools/dev/fixedqp_timing.py`; the instance side is `tools/dev/fixedqp_timing_remote.py`.
- **What one timing is.** The CPU time (user + system, rusage) of one encoder process, pinned to one core, one encode
  at a time. Our `--recon` / `--display-recon` outputs are dropped, as in the CBR chart's timing.
- **x86 machine.** AWS c8a.large (AMD EPYC 9R45, Zen 5), core 1, on demand. The encodes were spread over several
  instances of the same type. Sequences were packed whole onto an instance where they fitted, so most of each
  sequence's encodes ran on one instance.
- **Rounds.** An encode estimated under 20 s was timed in 3 rotated rounds after a warm-up round, keeping the median.
  A slower encode was timed once.
- **Round-to-round spread.** The largest deviation from the median, over encodes of 0.2 s or more:
  - CTC: median 0.79 %, 90th percentile 2.7 % (1958 encodes);
  - held-out: median 0.64 %, 90th percentile 1.6 % (1265 encodes).
- **Jetson machine.** Jetson Orin Nano, Cortex-A78AE core 3 at 1.728 GHz, under the board's timing lock. Every encode
  was timed once.
- **fps.** Per encoder and sequence: CPU ms per frame over all its timed quantizers. The fps is 1000 over the
  geometric mean of those across the sequences, so each sequence counts equally, as in the BD-rate means.
- **Not a 720p figure.** The CTC sequences run from 1280×720 down to 480×270 and the held-out clips from 1280×720
  down to 424×240. The fps is therefore higher than the CBR chart's 720p fps.

### AOM CTC view

- **Quality.** The lens refreshed for all our encoders at b11b2f83 on the build box: `nceval.ctc local`, 33 runs.
  - Our faster and veryfast rungs are now on the lens (`nceval.ctc.conditions._OPTIONAL_RUNGS`).
  - The references keep their stored curves (2026-11-14 / 2026-11-15).
  - Data: [fixed-qp-lens-2026-11-16.csv](fixed-qp-lens-2026-11-16.csv), on the site as
    `web/results/data/fixed-qp-lens.json`.
- **x86 speed.** Every chart encoder at all 6 quantizers on all 12 sequences, 21 instances,
  [fixed-qp-timing-x86.json](fixed-qp-timing-x86.json).
  - HM is the exception: it ran at 2 quantizers (QP 27 and 42) on every sequence, one encode taking up to 15 min.
- **Jetson speed.** Our 24 encoders and the realtime references:
  - x264 ultrafast, veryfast and medium; x265 ultrafast, superfast and medium; SVT-AV1 M10, M12 and M13; libaom rt 8
    and 10; libvpx rt 7 and 9; openh264.
  - 12 sequences at 2 of the 6 quantizers (the 2nd and 5th: QP 27/42, qindex 135/210, q63 34/52), one round.
  - About 60 min under the lock; [fixed-qp-timing-jetson.json](fixed-qp-timing-jetson.json).
  - The slow references (SVT M8, x265 slow, x264 veryslow, libvpx good 2, VVenC, HM) are not timed on the Jetson.
    The site lists them under the chart on that machine.

### Held-out view (new)

- **Clips.** 12 clips of the CBR chart's "Held-out, fresh" set (`independent_fresh_2026-11-13.json`), chosen by a
  fixed rule (`nceval.fixedqp.held_out_clips`):
  - content classes in turn, largest class first;
  - within a class, in the order of a SHA-256 of the clip name.
  - Result: 3 pan, 3 crowd, 2 static, 2 bodycam and 2 low-light clips, at 150 frames.
- **Quality.** `nceval.fixedqp gap --clipset ... --clips 12`, the pool point engine on this PC:
  - 4 quantizers per curve, spanning the CBR ladder's rates;
  - J against x264 medium at fixed QP, with a 90 % bootstrap interval over clips;
  - our 24 encoders took 15 min and the 14 references 20 min.
  - Data: [fixed-qp-fresh-2026-11-16.json](fixed-qp-fresh-2026-11-16.json), made by `nceval.fixedqp site`.
  - libaom and libvpx have a reliable overlap on only 6-7 of the 12 clips (n in the data).
- **Speed.** The same encodes at the quantizers each curve used, timed on c8a.large: `fixedqp_timing.py plan-pool`,
  10 instances, [fixed-qp-fresh-timing-x86.json](fixed-qp-fresh-timing-x86.json). Not timed on the Jetson.
- **Not included.** VVenC, HM, x264 veryslow, x265 slow and libvpx good 2 are not in this view: too slow to measure
  on 12 clips here.

### Cost

- **AWS c8a timing:** $0.84 in 32 on-demand c8a.large sessions, all terminated, in the ledger.
- **Lens refresh:** 5 slots of the shared build box for about 21 min.
- **(b) estimate.** It was sent before running. Measured: 35 min of PC time for quality, and 8 + 13 min of wall
  time (10 instances) for speed.

## Results

BD-rates are % against x264 medium at fixed QP; fps as above. "Ahead of" means faster and fewer bits, both at once.

### AOM CTC, x86 (Zen 5)

| Point | Weighted | J | fps |
|---|---:|---:|---:|
| SVT-AV1 M13 | +4.1 | -4.8 | 984.1 |
| AV1 Faster | +2.1 | -13.5 | 900.8 |
| SVT-AV1 M12 | -1.5 | -10.4 | 899.0 |
| AV1 Fast | -11.0 | -19.6 | 523.6 |
| SVT-AV1 M10 | -14.2 | -19.5 | 518.9 |
| AV2 | -25.2 | -34.7 | 305.7 |
| AV1 | -18.9 | -26.7 | 303.7 |
| SVT-AV1 M8 | -30.3 | -30.8 | 168.8 |

- **Weighted.** No point of ours is ahead of an SVT point.
  - AV1 Faster and AV1 Fast run at the speed of M12 and M10: 900.8 against 899.0 fps, and 523.6 against 518.9.
  - They are 3.6 and 3.2 points of Weighted behind those two.
  - The combined frontier, fast to slow: M13, AV1 Faster, M12, AV1 Fast, M10, AV2, M8.
- **J.**
  - AV1 Faster is level with M12 on speed and 3.1 points ahead on J.
  - AV1 Fast is level with M10 on speed and on J (-19.6 against -19.5).
  - AV2 is ahead of M8 (-34.7 against -30.8, 305.7 against 168.8 fps).
  - Nothing of ours is ahead of M13.

### AOM CTC, Jetson (2 quantizers, one round)

- **Weighted.**
  - AV1 Faster is ahead of M13: +2.1 at 222.0 fps, against +4.1 at 196.1.
  - Nothing of ours is ahead of M12 (-1.5 at 178.1) or M10 (-14.2 at 99.0).
  - AV1 Fast and AV2 Fast (-11.0 and -11.7 at 142.8 and 129.2) are faster than M10 but 3.2 and 2.5 points behind it.
- **J.**
  - AV1 Faster is ahead of M12 and M13: -13.5 at 222.0, against -10.4 at 178.1 and -4.8 at 196.1.
  - AV1 Fast and AV2 Fast are ahead of M10: -19.6 / -20.6 at 142.8 / 129.2, against -19.5 at 99.0.

### Held-out clips, x86 (J only; no Weighted on this view)

| Point | J | fps |
|---|---:|---:|
| SVT-AV1 M13 | -2.3 | 1001.2 |
| HEVC Faster | -12.1 | 914.8 |
| SVT-AV1 M12 | -7.3 | 912.6 |
| AV1 Faster | -17.0 | 787.0 |
| HEVC Fast | -21.0 | 695.3 |
| AV1 Fast | -23.4 | 585.1 |
| SVT-AV1 M10 | -15.3 | 537.1 |
| SVT-AV1 M8 | -25.5 | 174.1 |

- HEVC Faster is ahead of M12.
- AV1 Faster, HEVC Fast and AV1 Fast are ahead of M10.
- Ten of ours are ahead of M8, among them AV1, HEVC, VP9, AV2 and every VVC point.
- Nothing of ours is ahead of M13; H.264 Fast (+4.8 at 885.1) is behind it.
- After M13, the combined frontier is ours all the way down: HEVC Faster, AV1 Faster, HEVC Fast, AV1 Fast, VVC Fast,
  HEVC, AV2, VVC and VVC Slow.

### Against the earlier axis

- **Old axis.** The CBR chart's x86 timing on its 3 chart clips (`bd-vs-speed-timing-x86.json`): AV1 Faster 340.9 fps
  against M12 408.7; AV1 157.8 against M10 228.9.
- **New axis.** On the lens's own sequences, at the same quantizers the quality comes from: AV1 Faster 900.8 against
  M12 899.0; AV1 303.7 against M10 518.9.
- **What differs.** The two axes differ in content (sequences, resolutions and quantizers), not only in machine
  time. This run does not say which of those causes the difference.
