# Upper-Rate Regime Diagnosis & Rate-Aware Fixes (600–4000 kbps)

Date: 2026-11-23  
Branch: `feature/gemini-upper-rates` (base `3900962da`, commit `5166726b9`, unmerged, no preset edits)  
Byte-identity while default-off: **387/387 streams identical** (`tools/dev/preset_identity.py` vs `3900962da`).

---

## 1. Diagnosis: Measured Causes of Upper-Regime Loss (Largest First)

Evaluated on 720p pool/main clips (`crowd_mot17_09`, `pan_jaad_0101`, `crowd_mot20_06`, `lowlight_city_night`, 150f, 1 key) at 600–4000 kbps against `x264 medium` (zerolatency CBR) and `SVT-AV1 M12` (`--rtc 1` CBR). Rate control itself hits target accurately (`98–99%` of 4000 kbps; no undershoot — in fact `x264 medium` undershoots to `93%` and still wins).

### A. AV1 `veryfast` (why `+5.4` vs `x264 medium` in the upper regime)

1. **`av1.partial_forward = 2` (unconditional frequency truncation on 32×32 and 64×64 coded residuals) — `#1 cause, +1.0 to +3.3 dB YUV (+1.5 to +4.8 dB Y) at 4000 kbps`:**
   - `av1/veryfast.toml` sets `av1.partial_forward = 2` (truncating all 32×32 and 64×64 coded transforms to their low 16×16 frequencies — discarding 75% of 32×32 and 93.75% of 64×64 coefficients) AND `planner.depth_removal_min_quantizer = 0` + `planner.depth_removal_qp_scale = 8` (collapsing blocks to 32×32/64×64 even at low QP) AND `av1.inter_tx_split = 0`.
   - At 4000 kbps (`q_idx = 33..158`), it becomes physically impossible to represent fine detail: on `pan_jaad_0101`, raising bitrate from 2400 → 4000 kbps (+67% bits) gained only `+0.46 dB` Y-PSNR (`29.31 → 29.77 dB`), trailing SVT-AV1 M12 by `-3.53 dB` YUV.
   - Disabling `partial_forward` at high rates immediately recovers **`+3.30 dB` YUV (`pan`)**, **`+2.92 dB` (`lowlight`)**, **`+1.08 dB` (`crowd17`)**, **`+1.00 dB` (`crowd20`)** at 4000 kbps — completely closing the gap to SVT-AV1 M12. Crucially, at 600 kbps on noisy/hard clips (`crowd_mot20_06`, `q ≈ 250`), `partial_forward = 2` still helps (`+0.67 dB`), so the fix **must be QP-gated** (`gate.partial_transform_min_quantizer`, already wired in `nc-av1` but left at `0.0`).
   - Note: `av1.inter_tx_split = 1` (`+3.57 dB`) and `planner.depth_removal_min_quantizer = 0.65` (`+3.45 dB`) achieved the same jump primarily by bypassing `partial_forward = 2` (splitting to 16×16 transforms, which are not truncated). Once `partial_forward` is QP-gated, splitting 64×64 → 32×32 adds only `+0.01 dB`.

2. **Hierarchical-P (`layers.count = 3`) at high rates — `#2 cause, +0.45 to +0.63 dB YUV (+0.56 to +0.83 dB Y)`:**
   - At 2400–4000 kbps, T0 base frames (referencing 4 frames back) are near-lossless while 50% of frames (T2) are coded coarsely with `nonbase.*` shortcuts (`nonbase.av1.loop_filter = 0`, `light.*` effort). Falling back to flat distance-1 P-frames at high rates gains `+0.41 to +0.63 dB` YUV on `crowd` and `pan`.

3. **Minor contributors on AV1 `veryfast` at 4000 kbps:**
   - `effort.light_nonbase_only` / `light.planner.inter_min_log2 = 5` (32×32 min inter block): `+0.05 to +0.15 dB` YUV.
   - `decision.subpel_move_bias = 1.1`: `+0.05 dB` YUV.
   - `quant.trellis = 0`: `+0.01 to +0.07 dB` YUV.
   - `av1.chroma_delta_q = -24`: actually helps YUV PSNR slightly (`chroma_dq = 0` loses `-0.09 to -0.14 dB` YUV).

### B. H.264 `faster` & `veryfast` (why `+13.7` / `+41.9` vs `x264 medium` in the upper regime)

1. **Hierarchical-P (`layers.count = 3`) + `nonbase.*` cheap overrides on 50% of frames — `#1 cause, +0.41 to +0.47 dB YUV (+0.62 to +0.73 dB Y)`:**
   - `nonbase.*` disables quarter-pel (`nonbase.decision.subpel_half_only = 1`), deblocking (`nonbase.h264.deblock_inter = 0`), sets fixed `nonbase.decision.intra_gate_variance = 1024`, narrows search to `window_half_width = 1`, and sets `nonbase.h264.inter_zero_luma_scale = 3`.
   - Turning off `nonbase.*` alone recovers **`+0.35 dB` (`crowd @ 4000k`)** and **`+0.27 dB` (`pan @ 4000k`, mostly `nonbase_qpel +0.29 dB`)**; falling back to flat distance-1 P-frames (`layers.count = 1`) recovers **`+0.47 dB` (`crowd`)** and **`+0.41 dB` (`pan`)**.
   - Note: weakening `cbr_base_step_ratio` from `0.35` to `0.65` while keeping `layers = 3` *hurts* (`-0.34 to -0.50 dB`) because distance-4 T0 frames degrade.

2. **Sub-pel search & fixed SATD gates — `#2 cause, +0.20 to +0.56 dB YUV`:**
   - `decision.winner_walk_steps = 0` (`winner_walk2`: `+0.20 dB` `crowd`, `+0.56 dB` `pan`), `decision.refine_cheap_vector = 0` (`+0.10 to +0.24 dB`).
   - Fixed `h264.split_gate_satd` / `intra4x4_gate_satd` (`h264.split_gate_lambda = 0` pinned in `faster.toml` and `veryfast.toml`): restoring lambda-scaled `h264.split_gate_lambda = 0.8` gains **`+0.22 dB` (`crowd`)**, **`+0.11 dB` (`pan`)**, **`+0.11 dB` (`lowlight`)** at 4000 kbps.
   - On `h264 veryfast`, `h264.partition_search = 0` costs an additional `+0.10 to +0.21 dB` at 4000 kbps, and `h264.subpel_skip_variance = 10` (fixed) costs `+0.18 dB` on `lowlight @ 4000k`.

3. **Minor contributors on H.264 `faster` at 4000 kbps:**
   - `decision.final_stage = 0`: `+0.10 to +0.11 dB`.
   - `quant.trellis = 0`: `+0.05 to +0.14 dB`.
   - `analysis.{static,one_position,skip_hme}_sad_per_pixel`: `+0.00 to +0.09 dB`.

---

## 2. Implemented Rate-Aware Shared Engine Fixes (Default Off, 387/387 Byte-Identical)

1. **Shared `gate.partial_transform_min_quantizer` wired to VP9 and HEVC** (already wired in AV1; default `0.0`):
   - [nc-vp9/src/backend.rs](../../crates/nc-vp9/src/backend.rs) & [nc-hevc/src/backend.rs](../../crates/nc-hevc/src/backend.rs): disables `partial_forward` frequency truncation when relative quantizer drops below the threshold.
2. **Shared `decision.nonbase_min_quantizer` (default `0.0`):**
   - [nc-encoder/src/params.rs](../../crates/nc-encoder/src/params.rs) & [nc-encoder/src/encoder.rs](../../crates/nc-encoder/src/encoder.rs): when `last_relative_quantizer < decision.nonbase_min_quantizer`, `frame_role()` returns `FrameRole::Base`, bypassing `nonbase.*` cheap overrides at high bitrates across all codecs.
3. **Shared `layers.adaptive_min_quantizer` (default `0.0`):**
   - [nc-encoder/src/params.rs](../../crates/nc-encoder/src/params.rs), [nc-encoder/src/structure.rs](../../crates/nc-encoder/src/structure.rs), & [nc-encoder/src/encoder.rs](../../crates/nc-encoder/src/encoder.rs): tracks committed relative quantizer EMA in `TemporalStructure` and falls back from layered mini-GOPs to flat distance-1 P-frames below the threshold.

---

## 3. Measurements: Both Upper Regime and Low Regime (Quality and CPU Separate)

### Upper Regime (600 / 1200 / 2400 / 4000 kbps, 4 720p pool clips, BD-PSNR-YUV vs preset base + 2400k Callgrind ΔIr)

| Codec / Preset | Candidate Params | Upper BD-rate (mean) | Per-clip (`crowd17` / `pan` / `crowd20` / `lowlight`) | 2400k Callgrind ΔIr |
|---|---|---|---|---|
| `av1 veryfast` | `gate.partial_transform_min_quantizer=0.55` | **`-15.82%`** | `-5.1%` / `-25.7%` / `-1.5%` / `-31.0%` | `≤ +2.16%` |
| `av1 veryfast` | `gate.partial_transform_min_quantizer=0.60` | **`-18.00%`** | `-5.4%` / `-30.4%` / `-3.4%` / `-32.9%` | `≤ +2.16%` |
| `av1 veryfast` | `gate.partial_transform_min_quantizer=0.70` | **`-19.98%`** | `-6.0%` / `-37.7%` / `-4.3%` / `-31.8%` | `+2.16%` |
| `av1 veryfast` | `pf=0.70 + layers.adaptive_min_quantizer=0.65` | **`-22.69%`** | `-9.6%` / `-36.7%` / `-6.7%` / `-37.7%` | `+7.32%` |
| `h264 faster` | `decision.nonbase_min_quantizer=0.70` | **`-1.39%`** | `-3.0%` / `-2.5%` / `+0.0%` / `+0.0%` | `+3.90%` |
| `h264 faster` | `layers.adaptive_min_quantizer=0.70 + h264.split_gate_lambda=0.8` | **`-4.00%`** | `-10.0%` / `-3.6%` / `-0.0%` / `-2.4%` | `+20.94%` |
| `vp9 veryfast` | `gate.partial_transform_min_quantizer=0.70` | **`-1.03%`** | `+0.1%` / `-0.3%` / `-3.9%` / `+0.0%` | `+0.16%` |
| `hevc faster` | `gate.partial_transform_min_quantizer=0.70` | `-0.07%` | `-0.0%` / `-0.2%` / `-0.0%` / `-0.0%` | `-0.14%` |

### Low Regime (`nceval.quick`, pure `J` at `k = 0`) and Low-Regime CPU (`aws_cpu_ab --decision-plan`, Intel `c7i.large`)

| Codec / Preset | Candidate Params | L1 Pure `J` (24 clips, 150f) | Probe Mean `J` (1/150 & 1/300) | Verdict | Intel Cycles (`key1` / `f600-k150` / `f900-k300`) |
|---|---|---|---|---|---|
| `av1 veryfast` | **`gate.partial_transform_min_quantizer=0.55`** | **`-1.57% [-3.15, -0.29]`** | **`-0.09% [-0.41, +0.41]`** (1/150 `-0.68%`, 1/300 `+0.50%`) | **CONTINUE** | *(at 0.70 upper bound: `-0.49%` / `-1.46%` / `-1.84%`)* |
| `av1 veryfast` | `gate.partial_transform_min_quantizer=0.70` | `-0.27% [-2.52, +0.99]` | `+1.38% [+0.11, +3.48]` (`1/300 crowd +8.87%`) | KILL (threshold too high for low-rate T0) | `-0.49%` / `-1.46%` / `-1.84%` |
| `av1 veryfast` | `pf=0.70 + layers.adaptive_min_quantizer=0.65` | `-1.16% [-4.19, +0.79]` | `+1.95% [+0.43, +4.16]` (`1/300 crowd +8.95%`) | KILL (threshold too high for low-rate T0) | `+0.79%` / `-0.28%` / `-1.72%` |
| `h264 faster` | **`decision.nonbase_min_quantizer=0.70`** | **`-0.54% [-1.15, -0.07]`** | **`-0.03% [-0.04, -0.01]`** (worst class `+0.00%`) | **CONTINUE** | **`-0.60%` / `-1.40%` / `-0.37%`** (`+0.00%` insn) |
| `h264 faster` | `layers.adaptive_min_quantizer=0.70 + h264.split_gate_lambda=0.8` | `+0.80% [-0.89, +2.91]` (static guard `+7.25%`) | **`-0.22% [-0.48, +0.04]`** | CONTINUE (lower `adaptive_min_quantizer` to `0.55–0.60` for static L1) | `-0.65%` / `-0.66%` / `-0.28%` |

---

## 4. Carry-Over Per Codec

| Codec | `partial_forward` truncation (`gate.partial_transform_min_quantizer`) | `nonbase.*` / `layers.adaptive_min_quantizer` | Fixed depth-removal / SATD gates |
|---|---|---|---|
| **AV1** | **Huge (`-15.8%` to `-20.0%` upper BD-rate)** on `veryfast`, `faster` (light), `l2-fast`, `l3-fast`. `0.55` passes low-regime `nceval.quick` cleanly (`-1.57%` L1). | Applies (`-2.7%` additional upper BD-rate on `veryfast`). | `planner.depth_removal_min_quantizer = 0` on `veryfast` / `faster` light (secondary once `partial_forward` is gated). |
| **VP9** | **Applies (`-1.03%` upper BD-rate, `+0.16%` Ir)** on `veryfast`, `faster`, `fast` (nonbase); wired in this branch. | Applies (`vp9/veryfast`, `vp9/faster` use `layers = 3` + `nonbase.*`). | `planner.depth_removal_min_quantizer = 0` on `veryfast` / `faster` / `fast` light. |
| **HEVC** | **Wired, small effect (`-0.07%` upper BD-rate)** on `faster`, `fast` (light) because HEVC's TU tree already splits active 32×32 TUs. | Applies (`hevc/faster`, `hevc/fast` use `layers = 3` + `nonbase.*`). | `planner.depth_removal_min_quantizer = 0` on `faster` / `fast` light. |
| **H.264** | Does not apply (max transform 8×8). | **Major (`-1.4%` to `-4.0%` upper BD-rate)** on `veryfast`, `faster`, `fast`. `nonbase_min_quantizer=0.70` wins both low (`-0.54%` L1) and upper (`-1.39%`). | `h264.split_gate_lambda = 0` pinned on `veryfast`, `faster`, `fast`; fixed `subpel_skip_variance = 10` on `veryfast`. |
| **VVC** | Does not apply (no `partial_forward`). | Applies where `layers = 3` + `nonbase.*` are active. | Presets already keep `depth_removal_min_quantizer`, which is why VVC held up best in the upper regime (`-20.8` on `faster`). |
| **AV2** | Does not apply today (no `partial_forward` param). | Applies where `layers = 3` + `nonbase.*` are active. | `av2/faster` disables `inter_tx_split`; check `TX_64X64` high-rate behavior if `partial_forward` is ever added. |

## Verification & Budget
- Unit tests: `nanocodec-encoder` `structure::tests` 14/14 passed.
- Byte-identity while off: `387/387 streams identical` (`3900962da` vs `5166726b9`).
- AWS spend: 4 `c7i.large` spot CPU A/Bs (`~$0.05`) + 5 warm-box `nceval.quick` runs (`~$0.75`), well under the `$6.00` cap.
