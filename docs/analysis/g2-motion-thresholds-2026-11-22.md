# Brief 10: Heuristic-Containing Linear Thresholds for G2 Motion & Sub-Pel Exits

- **Date:** 2026-11-22
- **Branch:** `feature/gemini-thresholds` (commit `b4d88d47f`, unmerged; landing freeze respected, no preset files modified)
- **Implementation:**
  - `crates/nc-analysis/src/motion.rs`, `crates/nc-analysis/src/analyzer.rs` — `ZeroGateShift`, `WindowExitShift` (`Option<...>` on `SearchConfig`, `None` when all zero)
  - `crates/nc-encoder/src/params.rs` — 19 registered shift parameters (all default `0.0`)
  - `crates/nc-encoder/src/decision.rs` — reuses `crate::linear_threshold::LinearThreshold` for `effective_subpel_seed_gate` and `effective_skip_no_refine`
  - `crates/nc-encoder/src/encoder.rs` — zero-overhead per-frame folding of `nonbase`, `noise_sigma`, and `log_lambda`
  - `eval/tune_spaces/motion-exits-g2.txt` — 19-parameter Optuna tune space
- **Byte-Identity Proof (`tools/dev/preset_identity.py` on `nc-remote`):**
  - Base `75353c332` (`cfccf548`) vs `b4d88d47f` across all HEVC and AV1 presets: **`144/144 streams identical`** (57 s).
  - Unit tests: `nanocodec-analysis` (27/27) + `nanocodec-encoder` (328/328) passed; `clippy --tests -- -D warnings` clean.

---

## 1. Selected G2 Exits (4 Highest-Leverage Exits, 19 Parameters)

1. **`decision.codec_subpel_seed_lambdas`** (Inventory #2 overall; 6 params: `bias`, `nonbase`, `lambda`, `noise`, `size`, `zero_mv`)
2. **`decision.skip_no_refine_lambdas`** (4 params: `bias`, `nonbase`, `lambda`, `noise`)
3. **`analysis.{static,one_position,skip_hme}_sad_per_pixel`** (Inventory #1 overall; 5 params: multiplicative `(1 + shift).max(0)` on `bias`, `nonbase`, `lambda`, `noise`, `neighbour_mv`)
4. **`analysis.window_exit_sad_per_pixel`** (4 params: `bias`, `nonbase`, `lambda`, `noise`)

With all 19 terms at `0.0`, `LinearThreshold::new` / `Option<...>` returns `None`, computing zero features and producing bit-exact shipped streams.

---

## 2. Pilot & Study Costs (AWS `c7i.24xlarge` Warm Boxes, Budget Cap `$10.00`)

| Run | Mode | Trials | Wall Time | AWS Cost |
|---|---|---:|---:|---:|
| HEVC `medium` pilot (`g2-motion-hevc-medium`) | 150f | 10 | 20.8 min | `$0.19` (`$0.019`/trial) |
| HEVC `medium` study (`g2-motion-hevc-medium`) | 150f | 90 more (100 total) | 43.5 min | `$0.38` (`$0.004`/trial) |
| AV1 `medium` study (`g2-motion-av1-medium`) | 150f | 100 | 44.0 min | `$0.44` (`$0.004`/trial) |
| HEVC `medium` long-window (`g2-motion-hevc-medium-600f`) | **600f, `k=0.1`** | 60 (warm-started) | 60.1 min | `$0.61` (`$0.07` + `$0.54`) |
| AV1 `medium` long-window (`g2-motion-av1-medium-600f`) | **600f, `k=0.1`** | 60 (warm-started) | 58.4 min | `$0.52` (`$0.07` + `$0.45`) |
| Held-out screens (16 `nceval.screen` runs: L3 + 600f) | held-out | 8 candidates × 2 | ~60 min | `~$0.65` (shared warm boxes) |
| **Total Brief 10 AWS spend** | | | | **`~$2.79` / `$10.00` cap** |

---

## 3. Held-Out Screen Results (`nceval.screen --clips adopt --levels final`, `k = 0.1` CPU-Charged L3 + 600-Frame Final Check)

### 3.1 Long-Window (`--window-frames 600 --k 0.1`) Study Finalists — Primary Results

| Candidate | x86 ΔCPU (callgrind) | Held-Out L3 (`k=0.1`, 108 clips, 150f) | Held-Out 600f Final Check (39 clips, 10s keys) | Verdict |
|---|---:|---|---|---|
| **`hevc_600f_t15`** | **`-2.21%`** | **`BETTER` `-0.61%` `[-0.76, -0.45]` (0 guards)** | **`BETTER` `-0.09%` `[-0.28, +0.11]` (0 guards)** | **PASSES BOTH (`BETTER` / `BETTER`)** |
| `hevc_600f_t54` | `-0.14%` | `AMBIGUOUS` `+0.09%` `[-0.02, +0.20]` (0 guards) | `AMBIGUOUS` `+0.38%` `[-0.07, +0.83]` (0 guards) | Ambiguous |
| `av1_600f_t46` | `-0.95%` | `AMBIGUOUS` `+0.01%` `[-0.43, +0.51]` (0 guards) | `AMBIGUOUS` `-0.05%` `[-0.35, +0.25]` (0 guards) | Ambiguous (flat on AV1) |

- **`hevc_600f_t15` periodic-key pulse (held-out):**
  - L3 (150f): 2 s keys `+0.00 dB`, 10 s keys `-0.09 dB`.
  - Final check (600f): 2 s keys `-0.02 dB`, 10 s keys `-0.04 dB`.

### 3.2 Earlier 150-Frame Study Finalists (Superseded by 600f Objective; Not Prepared for Landing)

| Candidate | x86 ΔCPU | Held-Out L3 (`k=0.1`, 108 clips) | Held-Out 600f Final Check (39 clips, 10s keys) | Verdict |
|---|---:|---|---|---|
| `hevc_t76` | `-2.67%` | `BETTER` `-0.59%` `[-0.73, -0.43]` (0 guards) | `AMBIGUOUS` `+0.20%` `[-0.31, +0.70]` (0 guards) | Superseded by `hevc_600f_t15` |
| `hevc_t59` | `-1.60%` | `WORSE` `-0.46%` (pulse guard `+0.26 > +0.25`) | `AMBIGUOUS` `+0.08%` `[-0.07, +0.23]` (0 guards) | Fail (L3 pulse) |
| `hevc_t19` | `-4.40%` | `BETTER` `-0.30%` `[-0.49, -0.08]` (0 guards) | `WORSE` `+0.73%` `[+0.39, +1.07]` (0 guards) | Fail (600f) |
| `av1_t65` | `-0.70%` | `AMBIGUOUS` `-0.03%` `[-0.19, +0.12]` (0 guards) | `AMBIGUOUS` `-0.01%` `[-0.38, +0.37]` (0 guards) | Ambiguous |
| `av1_t28` | `-3.11%` | `AMBIGUOUS` `+0.34%` `[-0.12, +0.87]` (0 guards) | `WORSE` `+0.33%` `[+0.14, +0.51]` (0 guards) | Fail (600f) |

---

## 4. Winning Point Parameters (`hevc_600f_t15`, Ready Once Landing Freeze Lifts & Rented-Core CPU A/B Runs)

```toml
# HEVC medium — g2-motion-hevc-medium-600f trial #15
# Held-out L3 (k=0.1, dcpu=-2.21%): BETTER -0.61% [-0.76, -0.45], 0 guards
# Held-out 600f final check:        BETTER -0.09% [-0.28, +0.11], 0 guards
analysis.window_exit_shift_bias = -0.103662
analysis.window_exit_shift_lambda = -0.291233
analysis.window_exit_shift_noise = 0.394298
analysis.window_exit_shift_nonbase = 0.444587
analysis.zero_gate_shift_bias = -0.0209617
analysis.zero_gate_shift_lambda = 0.23367
analysis.zero_gate_shift_neighbour_mv = -0.061071
analysis.zero_gate_shift_noise = -0.521622
analysis.zero_gate_shift_nonbase = 0.749672
decision.skip_no_refine_shift_bias = 0.0491521
decision.skip_no_refine_shift_lambda = -0.0429776
decision.skip_no_refine_shift_noise = 0.0211403
decision.skip_no_refine_shift_nonbase = -0.0568535
decision.subpel_seed_shift_bias = -0.0157777
decision.subpel_seed_shift_lambda = -0.028532
decision.subpel_seed_shift_noise = 0.0119113
decision.subpel_seed_shift_nonbase = 0.0466947
decision.subpel_seed_shift_size = -0.00550283
decision.subpel_seed_shift_zero_mv = 0.0416859
```

**Why `hevc_600f_t15` works:** It pushes motion/sub-pel exits higher on **non-base frames** (`zero_gate_shift_nonbase = +0.75`, `window_exit_shift_nonbase = +0.44`, `subpel_seed_shift_nonbase = +0.047`), on **zero-MV seeds** (`+0.042`), and on **noisy frames** (`window_exit_shift_noise = +0.39`), while keeping **base frames** slightly *stricter* than today (`bias < 0`) so 300-frame GOP reference quality does not degrade. On **AV1 `medium`**, where `codec_subpel_seed_lambdas = 0.03` and `skip_no_refine_lambdas = 0.25` are already active, G2 linear shifts did not beat the shipped rung on held-out L3 or 600f.
