# Key frames under a tight buffer: quantization, allocation and size control (2026-11-21)

**Question (owner).** Our CBR keys are 3-4x the size of the references'. Held to a reference-like
buffer they cost 10-13 % J. With a fixed key budget, how should the key be coded, and where should
its bits go inside it ("unequal quant"), so that the key and the frames that inherit it are best?
And how well can rate control hit that budget in one pass?

**Answer.**

- **At equal key bits, quantization levers gain on the key alone, not in the stream.** One-frame BD-J
  at 0.15-buffer sizes: AV1 trellis on keys -5.0 % (with LR and rounding 56: -5.9 %), AV2 -3.5 %,
  H.264 deblock -2 -0.6 %. HEVC defaults are at their optimum. Variance AQ, intra bias, chroma
  offsets, rounding and filter strengths are at or past their optimum on every codec.
- **In the stream they are worth nothing.** Under the tight arm the key is 2.4-3.3 % of a 150-frame
  stream's bits. AV1 trellis on keys: +0.06 % J. At equal key bits it is -6 % on the key and -1.5 %
  on frames 1-30, but the whole stream is not better (+1.5 % / +2.1 %, one pan clip +21 %). Key
  size moves J; key coding at a fixed size does not.
- **Allocation inside the key by inheritance is worth under 1 %.** The hindsight MB-tree oracle
  (full knowledge of the next 150 frames or the key period) at equal key bits: best -0.5 % (HEVC),
  -0.3 % (AV1). At x264's strength, uncentred, it loses 0.3-4 % (section 3).
- **Size control.** A one-shot key re-code (`rate.key_recode_miss`, built, off) hits the budget:
  first-key SD 0.45-0.56 → 0.15-0.26, no 2x misses. Without it, first keys miss their target by log2 SD 0.38-0.47 (AV1 and H.264 overshoot
  by +0.25..+0.29 on average, HEVC undershoots by -0.21). 9-11 % of first keys miss by 2x or more,
  reaching up to 3x the 0.15 cap, and about 8 % are planned past the codec's coarsest quantizer.
  Periodic keys miss with SD 0.21 and never by 2x. Section 4.
- **Per bit, against the references at these sizes** (one frame, the same 24 clips; negative: the
  reference needs fewer bits). Two runs:
  - Ours (`fixedqp gap`): AV1 is ahead of SVT-AV1 M12 in its fixed-QP mode (ours -11 %, n 6 clips
    with curve overlap) and of libaom cpu8 (-6 %, n 11).
  - The key-rdo agent's runs (`~/nanocodec-data/scratch/key-rdo/ref-av1.json`,
    `ref-h26x-nosei.json`, info SEI removed):
    - x264 medium / slow need 7.9 / 10.7 % fewer bits than our H.264 key.
    - x265 medium / placebo need 9.3 / 12.6 % fewer than our HEVC.
    - SVT-AV1 still M2 needs 17.3 % fewer and libaom tune=iq 13.8 % fewer than our AV1.
  - The coding headroom is real, and it is in the search: key-rdo's RD partition search alone is
    HEVC -8.2 %, AV1 -3.6 %. In quantization it is not.

Nothing here changes a shipped stream. The diagnostics added are default-off.

## Method

- **Stage 1, the key alone.** `nceval.fixedqp ab --frames 1 --key-share 0.15` (new). One frame per
  clip, coded at quantizers walked to key sizes of 0.15 x a 1 s buffer at each ladder rate. A
  one-frame stream's rate is the ladder rate x 0.15 x fps, and the 50 %-of-source cap scales with
  it. Ours are coded with `--key-qp-offset 0`. BD-J is the project's J (PSNR-YUV, VMAF v1, MS-SSIM
  in thirds, full and low halves) at matched QP, class-balanced, with a 90 % interval.
  `fixedqp gap --key-share` codes the references' keys the same way. Their x264/x265 rows counted
  the encoders' information SEI in the rate; the key-rdo agent's rerun without it replaces them.
- **Stage 2, the stream.** `nceval.rc_oracle` live arms (new `--base-arm`: arms against an arm, and
  each arm's rate logs in the report). `eval/keystudy/ranges.py` gives BD over frame ranges on the
  whole stream's rate axis; it covers PSNR-YUV and VMAF only, since points keep no per-frame MS-SSIM.
  - Tight arm: `rate.key_cap_buffer=0.15, rate.recode_overflow=1.0, rate.window_seconds=1`.
  - Mid arm (the coordinator's 0.6.1 base, from section 3 on): `rate.key_cap_buffer=0.30,
    rate.recode_overflow=1.0`.
- **Clips.** Draw `fixedqp-2026-11-21`, quick level: 24 clips (crowd 4, lowlight 4, pan 4,
  static 3, bodycam 3, videocall 3, texture 2, drone 1). These are ranking clips, not the adoption
  draw. 600 f runs use 6 clips (2 each of bodycam, crowd, lowlight, pan, static with 600 frames),
  keys every 10 s: indicative only.
- Build 55aac741 (variance-integration), medium presets. Rented c7i.24xlarge boxes, about $2 in all.

## 1. The key alone: candidates at equal bits

BD-J % of the key frame against the preset (negative is better), n = 23-24:

| Candidate | AV1 | HEVC | H.264 |
|---|---:|---:|---:|
| variance AQ 0.5 / 1 / 2 (`rate.aq_strength`) | +3.1 / +5.2 / +13.9 | +0.8 / +3.7 / +13.8 | +12.9 / +22.6 / +37.8 |
| `planner.intra_bias` 900 / 1250 | +13.9 / +3.8 | +10.9 / +7.7 | inert |
| trellis on keys (`quant.trellis_key`, AV1 default 0) | **-5.0** [-5.4, -4.5] | off: +1.8 | off: inert (key not trellised) |
| trellis λ (keys) | 0.4 / 0.8 / 1.0: -4.1 / -3.9 / -2.3 (with trellis) | 0.6 / 1.5: -0.2 / +2.4 | inert |
| trellis on keys + `av1.restoration` | -5.5 | | |
| trellis on keys + rounding 56 / 40 | -5.9 / -3.4 | | |
| trellis on keys, `quant.trellis_states` 4 | -6.1 | | |
| intra rounding | 32 / 64: +6.5 / +0.9 | 171 / 240: -0.4 / +1.5 | |
| chroma offset | `chroma_delta_q` 0 / -16 / -48: +3.5 / +1.8 / 0.0 | `chroma_qp_offset` 0 / -6: -0.3 / +1.6 | 0 / -6: +0.3 / +0.2 |
| loop filter | level x0.7 / x1.3: +0.2 / 0.0; sharpness 3: +0.6 | deblock +2 / -2: +0.2 / 0.0; SAO λ x0.5 / x2 +0.3 / -0.1; band -0.1; off 0.0 | deblock -2 / -4 / alpha -2: **-0.6** / -0.5 / -0.5; +2: +0.8 |
| loop restoration on the key | -0.6 | | |
| λ (`decision.lambda_scale`, codec λ) | inert | `hevc.lambda_scale` 0.8 / 1.25: -0.3 / +0.5 | inert |
| `planner.lambda_scale` | inert | inert | inert |
| `quant.trellis_chroma` 1 | inert | +10.1 | inert |
| H.264 I16x16 / I4x4 trellis | | | +1.9 |
| H.264 psy 0 / 0.5 | | | inert |
| `av1.intra_rate_scale` 0.7 / 1.4 | +0.7 / +0.7 | | |

Other codecs, for the winners:

- **AV2:** trellis on keys (default off): -3.5 %; with trellis λ 0.6: -4.8 %; λ 1.5: +1.6 %.
- **VP9:** already trellises keys (off: +4.8 %). λ 0.8: -0.5 %; λ 1.5: +2.2 %.
- **VVC:** already trellises keys (off: +1.4 %). λ 0.6 / 1.5: +1.0 / -0.7 %. Intra rounding 171 / 220:
  +0.4 / +0.9 %.

"Inert" means byte-identical streams: the parameter is not read on a key at fixed QP. Two silent
no-ops were found:

- `key.av1.restoration=1` is accepted, but under CBR it codes the same stream as without it.
  Restoration is enabled in the sequence header, so a key-only LR needs code.
- `key.quant.trellis_lambda_scale` is refused (not re-read per frame). A key-only trellis λ would
  need a parameter of its own.

**Image-encoder candidates** (libaom all-intra / `tune=iq`, SVT-AV1 tune IQ, read in their source).

| Lever | What it is in their source | Our measurement |
|---|---|---|
| Variance boost | Low-variance 8x8 octile → finer qindex, up to 80; SVT disables it under CBR | Our variance AQ loses (+3..+38 %). The key-rdo agent measured libaom deltaq-mode 3 worse than off (-5.0 vs -8.9 % against ours) |
| Quantization matrices | qm 2..10 by qindex; enable-qm 0 by default | Needs code here (no QM syntax). key-rdo: libaom and SVT enable-qm 0.0 at these sizes |
| Sharpness 7 | Intra rounding 64/128 (no dead zone), filter sharpness | Rounding 64 +0.9 %, sharpness 3 +0.6 %. key-rdo: libaom sharpness 2 -1.4 % |
| Chroma delta-q | -(q/2 - 14), clamped 0..16 | Ours is already -32 qindex; 0 / -16 lose |
| Luminance QP bias (SVT) | 0 on keys (scaled by temporal layer) | Not applicable |

kvazaar and VVenC sources are not on the machine and were not read.

## 2. The stream: key coding under the tight arm

Tight against CBR (shipped key sizes): AV1 +13.4, HEVC +13.9, H.264 +12.9 % (150 f); +15.9 /
+14.5 / +18.5 % (600 f, 6 clips). The key is 2.4-3.3 % of the stream's bits under tight (150 f;
CBR 8.6-9.8 %).

| Arm, AV1 150 f, against tight | J % | key (frame 0) | frames 1-30 | 31-149 | whole (PSNR, VMAF) |
|---|---:|---:|---:|---:|---:|
| trellis on keys | +0.06 | +2.2 | +0.6 | -0.4 | -0.2 |
| trellis on keys + LR + rounding 56 | -0.41 | | | | |
| trellis on keys, cap 0.165 (equal key bits) | +2.12 (pan +21.0) | -6.1 | -1.5 | +2.5 | +1.5 |
| cap 0.165 (8 % more key bits) | -0.70 | -8.8 | -1.0 | -0.15 | -0.55 |
| cap 0.20 | -2.45 | -24.2 | -4.8 | -1.1 | -2.4 |
| cap 0.11 | +6.78 (pan +37.6) | | | | |
| (CBR) trellis on keys against CBR | -0.28 | | | | |

- **Why trellis does nothing at the controller's quantizer.** The key is coded at the same qindex
  with 8 % fewer bits (mean log2 -0.12). The size model does not know trellis saves bits, so the
  key comes out smaller and slightly worse, and the P frames spend the difference.
- **At equal key bits.** The key is 6 % better and the next second 1.5 % better, but frames 31-149
  lose 2.5 % and the stream is no better. The per-clip spread is large (pan +21 % on one arm).
- **AV2 trellis on keys** (tight): +2.0 % (crowd +3.5, static +3.4).
- **HEVC** `key.hevc.intra_rounding=171`: +1.8 % (pan +10.9). **H.264** key deblock -2: +0.1 %
  (150 f), -0.3 % (600 f).
- **Inheritance offsets on bounded periodic keys** (`persist.reuse_keys=1`, 600 f): AV1 +0.9,
  HEVC +1.3, H.264 +0.85 %, static +4.4..+6.0 %. This repeats the 2026-11-21 mode-2 result.
- **Key size is the lever.** Cap 0.15 → 0.20 is -2.4 %, 0.15 → 0.11 is +6.8 %.

CPU (x86 callgrind, `tune.instruction_cost` key frame, 3 cost clips): trellis on keys AV1
1651 → 1771 M (+7.3 %), AV2 1325 → 1467 M (+10.8 %) per key.

## 3. Allocation inside the key by inheritance: the hindsight oracle

**Mechanism (diagnostic, default off; streams are identical without it).**
- `nanocodec encode --analysis-log FILE` writes each frame's analysis per 8x8 cell: sum of
  squared deviations, 8x8 motion SAD and vector.
- `--key-offsets FILE` imposes a QP-offset map on given key frames
  (`Encoder::set_next_key_offsets`, `nc-encoder/src/key_oracle.rs`). The map adds to the block-AQ
  terms, and rate control budgets it like any block offset.
- `nceval.key_inherit` takes a first coding of the arm and propagates each period's frames back to
  its key with MB-tree's rule. A cell sends `(intra + received) * max(0, 1 - inter/intra)` along its
  vector, by area. `intra` is the Gaussian SAD estimate, as in `layer_propagation.rs`.
- Each key cell's offset is `-s * log2((intra + received) / intra)`. That is x264's closed form,
  `s = 5 (1 - qcompress)`, which is 2 at its default.
- `rc_oracle --key-map ARM:S,...` codes ARM's keys with those maps; the rest of the stream runs
  live. With full hindsight (150 frames, or the whole 10 s key period), it is the best that
  MB-tree's signal can give.

**Base: the mid arm** (`rate.key_cap_buffer=0.30, rate.recode_overflow=1.0`; against CBR it is
AV1 +5.1, HEVC +6.4, H.264 +6.6 % at 150 f, AV1 +7.5, HEVC +6.3 % at 600 f).

J % against the mid arm (negative is better), with the key's bits against mid (log2):

| Strength | AV1 150 f | HEVC 150 f | H.264 150 f | AV1 600 f (n 6) | HEVC 600 f (n 6) |
|---|---|---|---|---|---|
| 1 | +0.60 (+0.05) | +0.33 (-0.02) | +0.49 (+0.06) | +0.13 (+0.06) | +1.31 (-0.03) |
| 2 | +1.23 (+0.08) | +0.69 (0.00) | +0.92 (+0.11) | +1.86 (+0.15) | +4.07 (+0.04) |
| 3 | +2.81 (+0.11) | +1.34 (+0.07) | +1.92 (+0.16) | +3.08 (+0.16) | +2.42 (+0.14) |

By class (strength 2, 150 f): static loses the most (AV1 +5.9, HEVC +4.8 %), then crowd
(+2.8 / +1.8). Texture is the one class that gains (AV1 -2.4, HEVC -3.5, H.264 -2.1 %; 2 clips).
Static was expected near zero, since every cell is inherited. It is not, because the ratio
`(intra + received) / intra` is largest on flat cells (small intra), so the map refines flat
background and coarsens texture whatever the motion.

By frame range (PSNR-YUV and VMAF, strength 1 / 2): AV1 key +0.7 / +3.9 %, frames 1-30
+0.8 / +1.5 %, 31-149 +0.1 / +0.5 %. HEVC key +3.4 / +6.9 %, frames 1-30 +0.3 / +1.5 %. The map makes
the key worse, and the frames that inherit from it are worse too, not better.

**Centred maps** (the map less its intra-weighted mean, so rate control needs no uniform shift; the
key's bits are within 0.5-1.5 % of mid). 150 f:

| Strength | AV1 | HEVC |
|---|---|---|
| c0.5 | +0.20 | -0.17 |
| c1 | -0.27 (texture -1.6, drone -0.7, static -0.7, lowlight +0.8) | -0.51 (texture -3.2, crowd -0.8, pan -0.6) |

**Verdict: the hindsight inheritance ceiling with MB-tree's signal is under 1 %.** The best was
-0.5 % (HEVC, centred strength 1), inside the per-clip noise of a live arm (SD 1.3-2.2 %).
Uncentred maps at x264's strength lose 0.3-4 %. No causal mechanism is worth building on this
signal. Static and videocall are near zero as expected; texture (2 clips) is the one class with a
consistent gain (-1.1..-3.5 % in every centred and uncentred arm). That class gain is a lead, not
established.

Not tried: an optimised (per-cell searched) allocation, the true ceiling, which would need many
codings per key. This agrees with the earlier causal results: reuse offsets on bounded keys
(`persist.reuse_keys` 1: +0.9..+1.3 %), the hindsight reuse weighting of P frames (frame-allocation
2026-11-21: -0.3..+1.1 %) and variance AQ.

## 4. Key size control

First key, log2(actual / predicted), from the arms' rate logs (150 f, 24 clips x 4 rungs):

| | AV1 | HEVC | H.264 |
|---|---|---|---|
| CBR: bias / SD / missed by 2x | +0.16 / 0.51 / 11 % | -0.26 / 0.53 / 7 % | +0.10 / 0.50 / 8 % |
| tight: bias / SD / missed by 2x | +0.29 / 0.47 / 9 % | -0.21 / 0.38 / 1 % | +0.25 / 0.46 / 11 % |
| tight: key size median / max (buffers) | 0.16 / 0.48 | 0.13 / 0.28 | 0.16 / 0.39 |
| periodic keys (600 f): bias / SD | -0.08 / 0.21 | -0.04 / 0.21 | -0.09 / 0.22 |

- No tight key was re-coded: the 1.0 re-code ceiling is far above a 0.15 key.
- 11-13 of about 130 first keys (8-10 %) were planned past the coarsest quantizer (AV1 qindex 255,
  QP 51): at 0.15 buffer those pictures cannot be coded small enough.
- **A better one-pass predictor.** On the logged first keys (both arms, in-sample), a fit
  `log2 bits = a + 1.05..1.09 log2(spatial_hint) - 0.93..0.98 log2(step)` leaves a residual SD of
  0.33 (AV1), 0.36 (HEVC) and 0.31 (H.264), against the controller's 0.47-0.49. The same with
  today's fixed exponents (hint^1, step^-1.1) and one constant per codec gives 0.38-0.42. The
  bootstrap's guess at the seed quantizer (`bootstrap_bits_per_pixel`, `initial_key_complexity` 8)
  is what misses; the hint itself predicts about 30 % better.

**One-shot key re-code, built (`rate.key_recode_miss` = N octaves, default 0).** A key whose size
misses its budget (capped by the strict room) by more than N octaves either way is coded once more,
at the step the miss implies under the rate exponent (`RateController::key_size_recode`, after
any overflow re-code, at most once per key). Mid arm, 150 f, 24 clips x 4 rungs:

| | AV1 | HEVC | H.264 |
|---|---|---|---|
| mid: first key log2(bits / budget) mean / SD / ≥ 2x | +0.27 / 0.56 / 14 % | -0.19 / 0.45 / 2 % | +0.19 / 0.51 / 14 % |
| mid: key size median / max (buffers) | 0.33 / 0.95 | 0.25 / 0.63 | 0.31 / 0.81 |
| N = 0.5: mean / SD / ≥ 2x; keys re-coded | -0.01 / 0.26 / 0 %; 25 / 92 | -0.16 / 0.23 / 0 %; 31 / 92 | -0.05 / 0.26 / 0 %; 22 / 92 |
| N = 0.5: size median / max | 0.30 / 0.46 | 0.26 / 0.42 | 0.29 / 0.41 |
| N = 0.25: mean / SD; keys re-coded | -0.04 / 0.18; 48 / 92 | -0.11 / 0.15; 61 / 92 | -0.09 / 0.21; 49 / 92 |
| J % against mid, N = 0.5 / 0.25 | +2.80 / +3.23 | +0.14 / +0.37 | +2.59 / +2.73 |

- **It hits the budget.** The SD falls from 0.45-0.56 to 0.15-0.26, no key misses by 2x, and the
  largest key falls from 0.63-0.95 to 0.36-0.46 buffers of a 0.30 cap.
- **The J cost is the overshoot it removes, not the re-code.** AV1 and H.264 keys ran 15-21 %
  over the cap on average, and those bits were worth J: static +8.4..+11.4 %, crowd +3..+5 %.
  HEVC, whose keys undershot, is unchanged (+0.1 / +0.4 %), so correcting undershoots gains
  nothing measurable. Under a hard bound the overshoot is not allowed. The fair 0.6.1 comparison is
  at equal realised key size (the cap set where mid's keys actually land, about 0.35): not run.
- CPU: one more key coding on 24-34 % (N = 0.5) or 52-66 % (N = 0.25) of first keys.

## Recommendations (0.6.1 ready list)

No preset or default change is proposed. Held for 0.6.1:

- **`rate.key_recode_miss` 0.5 together with the 0.6.1 hard key bound.** It is the mechanism that
  makes a key cap a bound: the largest key falls from 0.63-0.95 to 0.36-0.46 buffers at cap 0.30,
  for one more coding on 24-34 % of first keys. Its J cost (+2.6..+2.8 % AV1/H.264, +0.1 % HEVC
  against an arm whose keys overshoot the cap by 15-21 %) is the price of enforcing the cap. Set the
  cap where the keys should land, not where they used to land after overshooting. Before a preset
  takes it: the cap at equal realised size (about 0.35), and `nceval.quick` / `train`.
- **`quant.trellis_key` 1 for AV1 and AV2: not on its own.** -5.0 % / -3.5 % on the key at equal
  bits; in the stream 0.0 % (AV1) and +2.0 % (AV2) under tight, -0.3 % under CBR. Its gain would show
  only through a key size model that knows trellis shrinks the key (and the key's share of a stream
  is 2-3 %). +7 % / +11 % key instructions.
- **Key quantization at a fixed budget: nothing to adopt.** Variance AQ, intra bias, rounding,
  chroma offsets, deblock/SAO/LR and λ are at or past their optimum on keys at these sizes, and the
  image-encoder levers (variance boost, sharpness, chroma delta-q) do not transfer.
- **Inheritance allocation: nothing to build** (section 3).
- **Where the J is:** key size (cap 0.15 → 0.20: -2.4 %; → 0.11: +6.8 %) and key search (key-rdo:
  RD partition search on keys HEVC -8.2, AV1 -3.6 % on the key alone).

## Open

- A one-pass first-key size model from the spatial hint (in-sample residual SD 0.31-0.36 against
  0.47-0.49). It would make the re-code fire less often. Not built: the re-code already hits the
  budget, and the predictor alone does not reach SD 0.2.
- The re-code at equal realised key size (cap about 0.35 with `rate.key_recode_miss` 0.5 against
  mid). It tells whether the re-code costs anything beyond the overshoot it removes.
- The texture class's gain from centred inheritance maps (2 clips).
- An optimised per-cell key allocation (searched, not MB-tree's closed form), the true ceiling.
- Quantization matrices on keys (AV1 qm levels, H.264/HEVC scaling lists): not implemented here,
  so not measured. key-rdo measured libaom and SVT enable-qm at 0.0 on these keys.

## Tools added (streams unchanged; `preset_identity` 387/387 identical)

- `nceval.fixedqp ab|gap --frames 1 --key-share S`: one-frame key curves at S x a 1 s buffer per
  ladder rate.
- `nceval.rc_oracle`:
  - `--base-arm NAME`: arms against an arm.
  - Each arm's rate logs in the report (`arm_logs`).
  - `--key-map ARM:S,cS,...`: the inheritance oracle.
- `eval/keystudy/ranges.py`: BD of one arm against another over frame ranges. Candidate sets are in
  `eval/keystudy/*.txt`.
- `nanocodec encode --analysis-log FILE`, `--key-offsets FILE` (`nc-cli/src/key_oracle.rs`,
  `nc-encoder/src/key_oracle.rs`), and `nceval.key_inherit`.
- `rate.key_recode_miss` (default 0).
