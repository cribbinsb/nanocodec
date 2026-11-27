# Exact-Filter Finalist Refinement (`decision.exact_finalists`)

Date: 2026-11-23  
Branch: `feature/gemini-h264-refine` (base `3c3644588`, commit `19e86b562`, unmerged, no preset edits)  
Byte-identity while default-off: **387/387 streams identical** (`tools/dev/preset_identity.py` vs `3c3644588`).

## Summary

Following Brief 12's finding that precomputed 6-tap half-pel planes gained `-0.9%` pure quality on `h264 veryfast` at `+16.2%` instructions, we implemented shared engine finalist refinement (`decision.exact_finalists`, default `0.0` off): rank with today's bilinear sub-pel path, then re-rank only the top `N` (`1..4`) unique finalists on the codec's exact luma filter (pruning fractional runner-ups more than `1.15×` behind the bilinear winner's cost) **only where sub-pel search actually runs** (never on `static_skip` or `subpel_skip_variance` blocks), and cache the winning fractional prediction via `CodecBackend::cache_exact_luma` for direct reuse by the backend.

### Quality (`nceval.quick`, pure `J` at `k = 0`) and CPU (`aws_cpu_ab --decision-plan`, Intel `c7i.large` on the screens' own encodes)

| Config | Preset | L1 Pure `J` (24 clips, 150f) | Probe 1/150 `J` (600f) | Probe 1/300 `J` (900f) | Probe Mean `J` | Intel Cycles (`key1` / `f600-k150` / `f900-k300`) | Intel Insn (`key1` / `f600-k150` / `f900-k300`) |
|---|---|---|---|---|---|---|---|
| `decision.exact_finalists=1` | `h264 veryfast` | `-0.05% [-0.32, +0.18]` | `-0.48% [-0.82, -0.08]` | `-0.05% [-0.31, +0.30]` | `-0.26% [-0.48, +0.00]` | `-0.46%` / `+0.07%` / `+1.52%` | `+0.96%` / `-0.01%` / `+1.09%` |
| `decision.exact_finalists=2` | `h264 veryfast` | **`-1.83% [-2.35, -1.43]`** (full ladder `-2.1%`) | **`-1.86% [-2.98, -0.47]`** (pulse `-0.11 dB`) | **`-2.02% [-2.55, -1.49]`** (pulse `-0.13 dB`) | **`-1.94% [-2.56, -1.20]`** | **`+2.25%` / `+1.95%` / `+2.40%`** | `+2.79%` / `+2.58%` / `+1.53%` |
| `decision.exact_finalists=3` | `h264 veryfast` | **`-2.13% [-2.70, -1.65]`** (full ladder `-2.4%`) | **`-2.26% [-3.59, -0.87]`** (pulse `-0.18 dB`) | **`-2.32% [-3.01, -1.66]`** (pulse `-0.13 dB`) | **`-2.29% [-3.04, -1.52]`** | **`+3.65%` / `+3.30%` / `+3.97%`** | `+4.01%` / `+3.56%` / `+2.46%` |
| `h264.exact_ranking=0, decision.exact_finalists=2` | `h264 faster` | `+0.64% [-0.53, +1.98]` | **`-0.67% [-1.09, -0.20]`** (pulse `+0.01 dB`) | **`-0.52% [-0.86, -0.44]`** (pulse `-0.02 dB`) | **`-0.60% [-0.86, -0.36]`** | **`-3.01%` / `-4.52%` / `-0.63%`** | `-4.95%` / `-3.35%` / `-2.17%` |

- **Class breakdown on `h264 veryfast` (`exact_finalists=2`, full L1 ladder):** all 8 classes improve (`static -3.8%`, `pan -3.4%`, `videocall -3.0%`, `texture -2.6%`, `crowd -1.6%`, `bodycam -0.8%`, `lowlight -0.8%`, `drone -0.6%`; worse on only 3/24 clips).
- **Class breakdown on `h264 veryfast` (`exact_finalists=3`, full L1 ladder):** all 8 classes improve (`static -3.9%`, `pan -3.6%`, `videocall -3.5%`, `texture -2.6%`, `crowd -1.8%`, `drone -1.3%`, `bodycam -1.2%`, `lowlight -1.0%`; worse on only 2/24 clips).

---

## 1. Split: Ranking Accuracy vs Final Vector Choice

Comparing `exact_finalists=1` (same MV as bilinear; re-evaluates only the winning fractional MV on exact 6-tap to update inter-vs-intra distortion and cache the prediction) against `exact_finalists=2` and `=3` (which also re-rank the top 2–3 finalists on exact 6-tap):

- **Ranking accuracy alone (`ef1`):** `-0.05%` L1 / `-0.26%` probe (**~3–13%** of the gain) at essentially zero net CPU (`-0.46%` to `+1.52%` cycles, mean `+0.38%`).
- **Final vector choice (`ef2 - ef1`):** **`-1.78%` L1 / `-1.68%` probe** (**~87–97%** of the gain) for `+1.8%` additional cycles.
- **Third finalist (`ef3 - ef2`):** `-0.30%` L1 / `-0.35%` probe for `+1.4%` additional cycles.

The gain exceeds Brief 12's half-pel planes (`-0.9%`) because `rerank_exact_finalists` compares exact 6-tap `distortion + rate` directly across the top finalists (including sharp half/quarter-pel neighbours that bilinear blurred past `subpel_move_bias`), recovering the best sub-pel vector without precomputing full-frame half-pel planes.

---

## 2. Codec Audit (Shared Engine Code vs Per-Codec Applicability)

All five codecs already implement `CodecBackend::luma_filter()` and `cache_exact_luma()`:
- **H.264 (`nc-h264`):** **Applies.** `veryfast` sets `h264.exact_ranking = 0` (bilinear ranking, 6-tap prediction), while `faster` and slower presets default to `h264.exact_ranking = 1`. Implemented `exact_luma_filter() -> Some(Filter::H264Luma)`. On `faster`, switching to `h264.exact_ranking=0, decision.exact_finalists=2` saves `-0.6%` to `-4.5%` Intel cycles while improving 600f/900f probe `J` by `-0.60%`.
- **VP9 (`nc-vp9`):** **Applies.** `luma_filter()` returns `None` when `vp9.exact_ranking < 0.5` (bilinear ranking, 8-tap prediction). Implemented `exact_luma_filter()` returning the frame's VP9 filter (`Vp9` / `Vp9Smooth` / `Vp9Sharp`), automatically sharing the exact same `nc-encoder::decision` path and `cache_exact_luma` reuse.
- **HEVC (`nc-hevc`):** **Does not apply today.** `luma_filter()` unconditionally returns `Some(Filter::HevcLuma)` (already ranks on exact 8-tap everywhere).
- **VVC (`nc-vvc`):** **Does not apply today.** `luma_filter()` unconditionally returns `Some(Filter::HevcLuma)`.
- **AV1 (`nc-av1`):** **Does not apply today.** `luma_filter()` unconditionally returns `Some(Filter::Av1)`.

---

## 3. Callgrind Instructions (`35f`, `1280×720`, `600 kbps`, `--threads 1`)

| Clip | Preset | Base Ir | `ef1` | `ef2` | `ef3` | `exact_ranking=1` / `bilin` |
|---|---|---|---|---|---|---|
| `crowd_mot17_09` | `h264 veryfast` | 1,077.36M | 1,093.94M (+1.54%) | 1,130.46M (+4.93%) | 1,155.32M (+7.24%) | `exact1`: 1,192.42M (+10.68%) |
| `pan_jaad_0101` | `h264 veryfast` | 1,007.47M | 1,027.04M (+1.94%) | 1,071.27M (+6.33%) | 1,097.21M (+8.91%) | `exact1`: 1,102.15M (+9.40%) |
| `crowd_mot17_09` | `h264 faster` | 1,474.07M | `bilin+ef1`: 1,399.57M (-5.05%) | `bilin+ef2`: 1,416.52M (-3.90%) | `bilin+ef3`: 1,432.96M (-2.79%) | `bilin`: 1,389.08M (-5.77%) |
| `pan_jaad_0101` | `h264 faster` | 1,473.90M | `bilin+ef1`: 1,378.60M (-6.47%) | `bilin+ef2`: 1,405.08M (-4.67%) | `bilin+ef3`: 1,422.51M (-3.49%) | `bilin`: 1,362.72M (-7.54%) |

## Verification & Budget
- Unit tests: `nanocodec-encoder` `decision::tests` 12/12 passing.
- Byte-identity while off: `387/387 streams identical` (`3c3644588` vs `19e86b562`).
- AWS spend: 4 `c7i.large` spot CPU A/B runs (~40 min total, `~$0.05`) + 4 warm-box `nceval.quick` runs (~12 min total on shared warm `c7i.24xlarge`), well under the `$5.00` cap.
