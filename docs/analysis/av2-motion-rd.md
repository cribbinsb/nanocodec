# AV2 motion decision diagnostic

2026-11-08, final AVM v1.0.0 native subset. Production motion remains opt-in.
The sections below record successive experiments; the final policy and valid
measurements are summarized in the last section.

## Observed cause

The search minimizes **prediction luma SSE + approximate motion bits × lambda**.
It does not price quantized residual coefficients or measure reconstruction error.
Bilinear subpixel prediction can reduce prediction energy while spreading a sparse
residual into more nonzero coefficients. Lossless coding makes the mismatch
especially clear: every candidate reconstructs exactly, so prediction SSE has no
place in the final objective; encoded bits should determine the winner.

A bounded native diagnostic compares the same source and reconstructed reference
at three quantizers and three synthetic patterns. It includes the production
search module directly, avoiding a copied search algorithm. Integer means current
one-pixel diamond search without subpixel refinement, not the historical two-pixel
search used for the earliest conformance checkpoint.

| Pattern / q | Path | Luma prediction SSE | Chroma prediction SSE | Reconstruction SSE | Coefficient bytes |
|---|---|---:|---:|---:|---:|
| stress / 0 | integer | 4,153,928 | 4,789,000 | 0 | 3,178 |
| stress / 0 | subpixel | 3,265,024 | 4,663,200 | 0 | 4,126 |
| stress / 100 | integer | 4,174,377 | 4,806,958 | 50,714 | 2,031 |
| stress / 100 | subpixel | 3,287,083 | 4,721,628 | 53,111 | 2,330 |
| translated texture / 0 | integer | 3,231,350 | 17,000,220 | 0 | 4,515 |
| translated texture / 0 | subpixel | 2,924,720 | 16,887,272 | 0 | 4,953 |

Coefficient bytes are actual native arithmetic-coded coefficients, but each
8×8 block uses a fresh coder and controlled initial neighbor contexts. They
include termination overhead and are **diagnostic measurements, not complete
stream sizes**. Motion bytes are recorded separately. The actual complete-stream
125-frame gate previously demonstrated a size regression too; it is not necessary
to repeat external conformance merely to inspect these native decisions.

Chroma omission can additionally favor a vector that hurts U/V. It does not
explain the stress result above, where both prediction SSEs fall. The primary
problem is selecting on prediction energy instead of coded rate/distortion.

## Lambda units

`quantizer_step_q3(q) / 8` is the dequantized coefficient step. The current 4×4
DCT scales an orthonormal transform by approximately eight: a constant-one block
has DC coefficient 32 rather than orthonormal DC 4. Consequently dividing
coefficient-step squared by 64 is a defensible conversion to pixel-domain SSE
units, not evidence of an accidental 64× underweighting.

Current motion lambda is effectively **1 × pixel-step²**, floored to one. Existing
AV1 explicitly normalizes its quantizer step and defaults to a tunable
`lambda_per_step_squared=0.135`; HEVC distinguishes SSE lambda from square-root
lambda for SATD mode screening. Neither supports blindly multiplying AV2's
current lambda to fix missing coefficient costs. At lossless qindex zero, use
rate alone rather than extrapolating a lossy quantizer model.

## Implemented bounded remedy

1. Retain three finalists: GLOBAL zero vector, integer-search winner, subpixel
   winner. Deduplicate identical vectors. Keep spatial predictor seeding and
   early termination; do not transform every diamond candidate.
2. For each finalist, predict six 4×4 transform blocks, quantize once and compute
   reconstructed Y/U/V SSE. Price actual coefficient syntax with current block
   contexts and motion syntax, including skip and transform partition effects.
3. Use `D_y + chroma_weight * D_uv + lambda * R` for lossy coding; use `R` only
   for lossless. Expose normalized lambda scale and chroma weight as tunable
   constants. Start with fixed-CDF syntax cost rather than an arbitrary nonzero
   count or raw prediction-energy estimate.
4. Retain the selected prediction, coefficients and reconstruction scratch for
   final emission. Avoid repeating selected transforms, interpolation or inverse
   transforms. Fixed frame CDFs make candidate pricing independent of adaptive
   entropy-state cloning; coefficient spatial contexts still must match reality.
5. Compare finalist diagnostics first; adopt only after the existing combined
   two-family forward gate and meaningful byte/distortion checks. Record the
   small extra CPU cost rather than claiming a free compression gain.

## Reproduction

```sh
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=~/nanocodec-data/targets/vp9-av2-sequence \
  cargo run -p nc-av2 --example motion_rd
```

The example processes 64×64 source/reference pairs entirely in memory and prints
27 CSV rows. Patterns are stress texture, translated texture and static gradient;
quantizers are 0, 100 and 200. It is a diagnostic, not a broad quality benchmark.
Strict clippy passes for this example. The diagnostic now additionally reports complete zero-motion and finalist-RD
P-packet bytes and reconstruction SSE on stderr. Production motion remains
opt-in; the finalist implementation is used when it is enabled.

## Finalist result

The fixed-CDF arithmetic writer now supports scalar checkpoint/rewind and
fractional-bit measurement from the actual interval. Candidate pricing emits
normal block/motion/transform/coefficient syntax, then restores the writer and
12 bytes of touched coefficient-edge context. No full tile/picture clone is
needed. Selected quantized coefficients and reconstructed pixels are retained;
final emission repeats syntax coding but no transforms or prediction. Distortion
accumulation was omitted for the original zero-motion baseline and remains
omitted for lossless rate-only selection. The final default enables lossy
residual-skip RDO, which measures prediction/reconstruction distortion even
with motion search disabled. Runtime parameters are `av2.rd_lambda_scale` (default 0.135) and
`av2.chroma_weight` (default 1), alongside existing bounded search controls.

Complete single-P packets against the same reconstructed reference:

| Pattern / q | Zero bytes | RD bytes | Zero reconstruction SSE | RD reconstruction SSE |
|---|---:|---:|---:|---:|
| stress / 0 | 3,553 | 3,044 | 0 | 0 |
| stress / 100 | 2,369 | 1,986 | 51,158 | 50,625 |
| stress / 200 | 734 | 668 | 13,231,581 | 13,058,545 |
| translated texture / 0 | 8,368 | 4,524 | 0 | 0 |
| translated texture / 100 | 5,778 | 2,979 | 54,027 | 50,135 |
| translated texture / 200 | 1,743 | 1,014 | 12,971,765 | 12,441,867 |

Static gradient packets and reconstruction are unchanged at all three tested
quantizers. These synthetic checks establish a useful improvement over zero
motion on these examples, not broad clip performance or reference parity.

The combined 125-frame AVM+dav2d gate passes after finalist integration. Compared
with the preceding subpixel-only decision, total stream bytes change:
8×8 1,575→1,469; 16×16 4,078→4,006; 64×64 53,099→47,886;
72×40 52,416→46,281; 128×136 241,116→229,667. This recovers part of the initial
subpixel regression. Historical coarse-search totals have different motion
candidates and cannot establish equal-distortion superiority.

Native tests include 512 arithmetic checkpoint/carry rewinds with byte-exact
baseline recovery, exact bypass-bit accounting, and a lossless texture regression
requiring fewer bytes and exact source pixels. Forty library tests and strict
all-target clippy pass. Broader real-clip RD and CPU measurements remain pending;
three finalists can cost more CPU than a single prediction-energy winner.

## Real-clip out-of-domain diagnostic

The fixed release binary was measured on the first 16 frames of three existing
quick360 clips at 640×360, qindices 96 and 160, one key followed by 15 P pictures.
All 12 points pass native/AVM/dav2d reconstruction gates (one identical stream
reuses its verified gate). PSNR is streamed against the existing source; no raw
video was copied or written.

| Clip | q | Zero payload bytes | RD payload bytes | Byte change | YUV PSNR change | Y PSNR change |
|---|---:|---:|---:|---:|---:|---:|
| static_chirla_s000c3_360p | 96 | 232,319 | 231,765 | -0.24% | -0.0007 dB | +0.0008 dB |
| static_chirla_s000c3_360p | 160 | 84,229 | 84,229 | +0.00% | +0.0000 dB | +0.0000 dB |
| crowd_mot17_09_360p | 96 | 726,900 | 574,745 | -20.93% | +0.1487 dB | +0.1167 dB |
| crowd_mot17_09_360p | 160 | 238,165 | 215,148 | -9.66% | -0.2845 dB | -0.3474 dB |
| pan_jaad_0117_360p | 96 | 966,859 | 656,192 | -32.13% | +0.3051 dB | +0.1588 dB |
| pan_jaad_0117_360p | 160 | 348,561 | 282,362 | -18.99% | +0.1764 dB | -0.0818 dB |

**These fixed-q points exceed the repository source-rate budget and are not a
valid quality benchmark or evidence for changing defaults.** For example, static
source rate is 1,170 kbps, its 50% cap is 585 kbps, and the measured q96 zero-motion
stream is 3,485 kbps. The JSON now explicitly records source rate, cap and
`within_rate_cap` for every point. In-domain quantizer bracketing is required.

This is a short diagnostic, not BD-rate: two quantizers do not establish a
matched-quality curve. These out-of-domain moving clips show byte reductions; crowd q160
trades 0.285 dB YUV PSNR for fewer bits. Static q160 is identical, while q96 changes
very little. No speed claim is made because the machine was busy. Motion remains
opt-in pending CPU evidence and valid in-domain quality evaluation. Forced-skip RDO is a
next hypothesis for static noisy residuals, separate from this candidate change.

Reproduction: `tools/dev/av2_efficiency_probe.py --cli PATH --output DIRECTORY`;
optional repeated `--param NAME=VALUE` applies only to the RD variant. It reuses
the VP9 probe command/container helpers and evaluator streamed point/metric/gate
framework, and resumes completed points only when binary/source metadata and
bitstream hashes agree. Results and binary SHA are stored under
`~/nanocodec-data/results/av2-rd-quick360-20261108/{metadata,quality}.json`.

## In-domain adaptive-CDF screening

Fixed-CDF rate-only bracketing at q208/224/240 could not reach any source-rate
cap. An already-running q255 check confirmed a common 945.69 kbps floor across
all three clips. Adaptive CDFs removed that obstruction. Rate-only screening
selected static q144/160 and crowd/pan q208/224, then all 12 selected points were
measured with streamed PSNR and full native/AVM/dav2d gates. Both compared
variants enable `av2.cdf_adaptation=1`; only motion search differs.

| Clip | q | Payload bytes (both) | kbps | Source-rate cap | YUV PSNR (both) |
|---|---:|---:|---:|---:|---:|
| static_chirla_s000c3_360p | 144 | 32,998 | 494.970 | 585.175 | 34.113 dB |
| static_chirla_s000c3_360p | 160 | 16,336 | 245.040 | 585.175 | 31.670 dB |
| crowd_mot17_09_360p | 208 | 23,804 | 357.060 | 374.750 | 23.058 dB |
| crowd_mot17_09_360p | 224 | 13,521 | 202.815 | 374.750 | 21.010 dB |
| pan_jaad_0117_360p | 208 | 29,463 | 441.945 | 594.288 | 25.275 dB |
| pan_jaad_0117_360p | 224 | 13,553 | 203.295 | 594.288 | 23.920 dB |

**RD motion is byte-identical to zero motion at all six in-domain points.** The
current coarse-q candidate-screening thresholds prevent useful motion choices;
this does not justify enabling motion by default. Candidate screening uses a
normalized-step-squared lambda factor of 1, whereas final RD uses 0.135, and its
early-SSE threshold grows with that larger screening lambda. A bounded ablation
of early exit and screening lambda is the next test, not a broad parameter sweep.

Recorded result directory:
`~/nanocodec-data/results/av2-adaptive-rd-quick360-20261108`. The probe accepts
`--common-param av2.cdf_adaptation=1` and `--qmap` to reproduce the selected
per-clip quantizers. Six RD streams reuse byte-identical verified baseline gates.
The optimized binary used here predates the allocation-only inline checkpoint
change; pixel/bitstream behavior is the same, but CPU evidence must identify the
exact binary separately. No BD-rate or reference-parity claim follows.

### Coupled screening ablation

A single in-domain pan q208 point isolates the screening policy. Disabling
`early_sse_per_lambda` alone remains byte-identical to baseline. Adding
`av2.search_lambda_scale=0.135` produces 28,088 payload bytes instead of 29,463
(−4.67%), at YUV PSNR 25.2454 rather than 25.2749 dB (−0.0295 dB). The complete
native/AVM/dav2d gate passes, and 421.32 kbps remains below the source-rate cap.
This confirms the screening motion-rate scale prevented useful finalists at
coarse quantizers; it does not establish a universally optimal replacement.

The new screening scale is independently tunable over 0.001–4, with its old
behavior initially retained as default 1.0; the final policy below changes it
to 0.135. Motion remains disabled by default. Results:
`~/nanocodec-data/results/av2-adaptive-rd-lambda0135-20261108/quality.json`.
The low-level `encode_inter_motion` helper explicitly retains fixed CDFs, like
`encode_inter`, so native motion diagnostics compare the same entropy policy;
shared backend and stateful stream defaults use adaptive CDFs.


## Final v1 policy and bounded validation

The shared encoder now defaults to adaptive CDFs, lossy residual-skip RDO,
`pre_tx_skip_energy=30`, and luma TX8 for lossy pictures. Motion search remains
selectable. When enabled, its screening lambda scale is 0.135, early SSE scale
0.025, and subpixel refinement one step; final RD lambda remains 0.135 and
chroma weight 1. Lossless pictures retain TX4 and never drop residuals.
Low-level explicit fixed-CDF helpers retain their entropy policy.

Residual-skip RDO compares the actual syntax rate and reconstructed YUV SSE of
coded residuals against prediction-only reconstruction. The early screen tests
`100 * (Y_SSE + UV_SSE) <= 96 * qindex * pre_tx_skip_energy`: 96 is the number
of cached samples (64 Y, 16 U, 16 V), and 100 gives the parameter percentage
units. It is an empirical threshold, not an exact pruning proof. A passing
zero-motion candidate bypasses transforms, motion search and entropy trials.
Failed screens reuse cached predictions/residuals, avoiding repeated work.

The bounded study used the first 16 frames of existing 640×360 quick360 clips,
three training clips and held-out `bodycam_airport_360p`. Every reported point
is below its source-rate cap and passes native/AVM/dav2d reconstruction gates
(or reuses a gate only after matching the exact compressed-stream hash).
No raw video was copied. These short PSNR points are not BD-rate, perceptual
metric results, reference parity, or production-speed measurements.

The final motion-off, skip-on, pre-screen-30 recipe was checked separately
with TX4 to isolate the motion/skip changes:

| Clip | q | Payload bytes | kbps | YUV PSNR |
|---|---:|---:|---:|---:|
| static | 144 | 15,248 | 228.720 | 34.00461 dB |
| crowd | 208 | 13,728 | 205.920 | 22.19011 dB |
| pan | 208 | 13,015 | 195.225 | 24.17101 dB |
| held-out bodycam | 160 | 4,461 | 66.848 | 35.65343 dB |

Results: `~/nanocodec-data/results/av2-final-default-20261108/quality.json`.
This directory records the TX4 isolation run, **not the subsequent TX8 default**.
Nearby-rate baseline comparisons support residual-skip RDO: static no-skip q160
used 245.04 kbps at 31.670 dB, crowd no-skip q224 used 202.815 kbps at 21.010 dB,
and pan no-skip q224 used 203.295 kbps at 23.920 dB. This is point evidence,
not interpolation or a BD-rate claim.

With motion enabled, adding the pre-screen changed static by -1 byte and
-0.000027 dB, left crowd and pan identical, and changed held-out bodycam by
+8 bytes and -0.000040 dB versus the same tuned motion/skip recipe without
the screen. Results are in `av2-skip-motion-prescreen30-20261108` and
`av2-skip-motion-heldout-prescreen30-20261108` under the results directory.
The motion-off static comparison changed by -67 bytes and -0.00259 dB versus
skip RDO without screening; the tighter motion-on bounds do not describe every
configuration.

Same-binary callgrind measurements subtract the keyframe instruction count from
a three-frame stream and divide by two P pictures. All streams were gated:

| Clip | No skip, no motion | Skip + pre-screen | Tuned motion + skip + pre-screen |
|---|---:|---:|---:|
| static q144 | 52.958 M instructions/P | 48.456 M (-8.5%) | 55.926 M (+5.6%) |
| pan q208 | 54.234 M | 72.456 M (+33.6%) | 97.485 M (+79.8%) |

Results: `av2-skip-motion-prescreen30-instructions-20261108/instructions.json`.
These instruction counts isolate TX4 and are not wall-clock speed or the final
TX8 CPU cost. Tuned motion reduced payload versus skip-only by 2.19% static,
8.54% crowd and 18.13% pan, with YUV changes -0.015, -0.104 and -0.012 dB.
Its moving-content CPU cost therefore keeps it opt-in for this v1 release.

The separate TX8 comparison held the actual default motion/skip/pre-screen
settings fixed. Relative to TX4, bodycam q160 gained 1.5277 dB for 1.84% more
bytes; pan q208 gained 1.0175 dB with 1.00% fewer bytes; static q144 lost
0.0812 dB with 1.21% more bytes. All six points passed gates and caps. This
bounded cross-content evidence justified enabling TX8, while retaining its
runtime switch for wider quality/CPU evaluation. Results and exact binary/
parameter metadata: `~/nanocodec-data/results/av2-tx8-default-rd-20261108/`
(`comparison.json`, plus `tx4/` and `tx8/` point records). It does not imply
that every clip improves. TX8 also passed 48-keyframe and 125 mixed key/inter frame
conformance batches, including adaptive CDFs and subpixel motion.
