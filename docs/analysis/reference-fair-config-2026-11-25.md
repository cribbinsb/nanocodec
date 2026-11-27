# Reference encoders' CBR configuration under the BP-4.0 rule (2026-11-25)

The mapping of `docs/benchmark-procedure.md` section 3 ("The configuration rule", b10b2d62a)
onto every family of the fairness record, flag by flag, with the measured effect of each
change. It recommends the record changes (section 6); it applies none of them.

## Owner decisions this applies (2026-11-25)

1. **CBR only.** The benchmark has no Fixed QP mode (338f52fbf). Every Fixed QP arm of this
   audit was dropped; VVenC (charted in Fixed QP only) is out of scope.
2. **The configuration rule (b10b2d62a):**
   - each encoder's own real-time defaults;
   - two constraints only: no added latency (no lookahead, no reordering; temporal layers
     allowed) and the 1 s buffer, starting full, drained at the target;
   - four measurement conditions: the same 8-bit input with one key at its start, the same
     target, one thread, every frame coded on main rows (`+drop` rows separate);
   - everything else default: structure, cadence, scene-cut keys, quantizer range, AQ and
     tools.
3. **Quantizer range: libvpx and libaom run their full range, max 63, on main rows.** The 52
   clamp of their real-time examples presupposes frame dropping. `+drop` rows keep the
   shipped configuration, including its max-q.
4. **Rule 1 for libvpx and libaom is the project's own shipped real-time configuration**
   (their real-time example programs), except settings that presuppose dropping
   (coordinator's reading for the owner).
   - The procedure's rule-1 list names `libvpx --rt`. The bare CLI defaults were measured
     too (section 4): they cost libvpx about 10 %.

## 1. Method

- **Clips:** 24, from the tuning pool (`nceval.pool` draw `refaudit-2026-11-25`, stratified
  and family-disjoint), 150 frames from frame 0.
  - The AOM-CTC sequences and every sealed set are excluded (OldTownCross was swapped out).
  - 21 clips carry all four rungs and 3 carry fewer, because of the source-rate cap: 84 CBR
    points per family.
  - List: `bench-method-records/refaudit-2026-11-25/clips.json`.
- **Commands:**
  - the record's own (`tools/bench/plan.py --arch aarch64`, the bp3 aarch64 anchors; ours
    889309c99);
  - each arm is the record command with the stated change
    (`bench-method-records/refaudit-2026-11-25/tools/make_arms.py` and the stage-3 additions
    in `jobs.json`).
- **Encoding:** Graviton 4.
  - Main arms: one c8g.metal-24xl, 95 encodes at once.
  - Stage-3 arms (libaom example configuration, Kvazaar lp-g8d4t1, openh264, x264 scene-cut):
    one c8g.8xlarge, because bare-metal spot had no capacity at 09:38-09:48.
  - Every stream decoded to 150 frames (V9); 0 failed encodes.
- **Scoring on the PC:**
  - decoding by the pinned ffmpeg 8.0.1 (libdav1d for AV1);
  - VMAF v1 and MS-SSIM on the GPU's VMAFx server (`nceval.metrics`);
  - PSNR per plane from exact sums;
  - 3,316 CBR streams scored, 0 refused.
- **BD:** BD-J is the mean of the BD-rates on PSNR-YUV (6:1:1), VMAF v1 and MS-SSIM. It is
  computed by `report.clip_bd` (BP-3.14 rules, monotone envelope) against **x264 medium's
  default-tune curve for every metric**. The audit coded no `~psnr`/`~ssim` twins, so this
  differs from the report's matched twins.
- **Reading the BD numbers:**
  - **"own set":** the clips on which the arm has a BD.
  - **"common":** the clips where both the record and the arm have one.
  - **"direct":** arm against record, as a diagnostic only (the headline is never pairwise).
- **Compliance:** `report.replay` at the target (1 s, starting full). The figure is the
  number of points whose peak exceeds 1 buffer (and 1.15).
- **Time:**
  - **Same-machine arms** (libvpx, SVT, Kvazaar lp-g8d3t1, libaom "aomenc defaults"): the
    ratio of wall seconds per point to the record's, coded in the same run. It is a rough
    CPU proxy at full mixed load, not a BP-4 timing.
  - **Stage-3 arms** ran on a different machine, so their time ratios are **not comparable**
    and are not quoted.
- **Records:**
  - durable: `~/nanocodec-data/bench-method-records/refaudit-2026-11-25/` (encodes, scores,
    tools, tables `analysis-all.txt`, `common.txt`, `behave-*.txt`);
  - streams: deleted after scoring.
- **Cost:**
  - **AWS:** $1.68 (c8g.metal-24xl $1.60, most of it the 17 GB download including the
    dropped Fixed QP streams; c8g.8xlarge $0.04 twice).
  - **Scoring:** on the local GPU, no AWS cost.

## 2. Behaviour, read from the streams

- **Bitstreams** (every rung-500 stream of every CBR family and arm, 23 per family;
  `behave-main.txt`, `behave-s4.txt`):
  - **No reordering anywhere.** H.264/HEVC packets in decode order have non-decreasing
    presentation times. AV1/VP9 frame headers (ffmpeg `trace_headers`) carry no
    `show_frame = 0` and no `show_existing_frame`.
  - **Keys:** one per stream (the start), except where scene-cut detection is switched on
    (x264 scene-cut and openh264 scene-change arms, below).
  - SVT-AV1's default 2-level structure references past frames only (no hidden frames).
- **One thread** (x86 builds, one encode each; `threads.txt`):
  - peak thread count 1 for every family except x265 (5) and Kvazaar (2);
  - their CPU time over wall time is 1.022 (x265 medium) and 1.036 (Kvazaar veryfast), one
    720p CBR encode each: helper threads, one core of work;
  - SVT-AV1 M12: 0.999.

## 3. Results (CBR, 24 clips, 84 points per family)

| Family: change | BD-J vs x264 medium, common clips (record → arm) | direct arm vs record | clips with a BD (record → arm) | points over 1 buffer (record → arm) | time (same run) |
|---|---|---|---|---|---|
| libvpx cpu7: max-q 52 → 63 | 18 clips: -15.1 → -12.6 % (median) | +3.9 % | 18 → 23 | 44 → 14 of 84 | x0.90 |
| libvpx cpu9: max-q 52 → 63 | 19: -13.4 → -11.2 % | +2.3 % | 19 → 23 | 44 → 14 | x0.92 |
| libvpx cpu7: CLI defaults only (`--rt`, no example settings) | 18: -15.1 → -6.1 % | +11.0 % | 18 → 23 | 44 → 33 | x0.97 |
| libvpx cpu7: CLI defaults + `--aq-mode=3` | 18: -15.1 → -15.2 % | +1.6 % | 18 → 23 | 44 → 17 | x0.90 |
| libaom cpu8: example → `aomenc --rt` + example RT settings, initial 1000 ms, max-q 63 (**mapping**) | 18: -28.6 → -27.8 % | -0.5 % | 18 → 23 | 35 → 10 | n/a (other machine) |
| libaom cpu10: same (**mapping**) | 18: -23.2 → -22.7 % | -1.0 % | 18 → 23 | 36 → 10 | n/a |
| libaom cpu8: `aomenc --rt` defaults only | 17: -28.2 → -31.1 % | -3.3 % | 18 → 22 | 35 → 17 | x1.24 |
| libaom cpu8: same, max-q 52 | 17: -28.2 → -30.5 % | -4.0 % | 18 → 18 | 35 → 36 | x1.33 |
| SVT-AV1 M8: drop `--hierarchical-levels 0` (default 2 levels) | 22: -32.8 → -32.2 % | +7.2 % | 22 → 22 | 4 → 2 | x1.04 |
| SVT-AV1 M10: same | 22: -24.3 → -25.1 % | +7.0 % | 22 → 22 | 1 → 3 | x1.03 |
| SVT-AV1 M12: same | 23: -19.7 → -17.6 % | +4.2 % | 23 → 23 | 1 → 2 | x1.00 |
| SVT-AV1 M13: same | 23: -14.7 → -14.5 % | +4.2 % | 23 → 23 | 1 → 1 | x1.03 |
| Kvazaar ultrafast: `--gop 0` → `lp-g8d4t1` (**mapping**) | 23: -18.7 → -17.2 % | -8.6 % | 23 → 23 | 16 → 16 | n/a |
| Kvazaar superfast: same | 23: -25.8 → -24.3 % | -6.9 % | 23 → 23 | 15 → 18 | n/a |
| Kvazaar veryfast: same | 22: -28.5 → -29.6 % | -6.1 % | 22 → 23 | 17 → 19 | n/a |
| Kvazaar veryfast: `lp-g8d3t1` (shallower, documentation) | 21: -28.7 → -29.4 % | -8.8 % | 22 → 22 | 17 → 20 | x0.92 |
| openh264: `--scene-change 0` → 1 (default) | 23: +26.5 → +26.5 % (mean +30.6 → +33.6) | +0.0 % | 23 → 23 | 25 → 26 | n/a |
| openh264: `--adaptive-quant 0` → 1 (default) | identical streams on all 84 points | 0 | — | — | — |
| x264 medium: drop `--no-scenecut`, `--min-keyint` | identical BD (one stream of 23 gained a second key) | +0.0 % | 23 → 23 | 0 → 0 | — |

How to read the table:
- Where the clip counts on the common set and the own set differ, the "common clips" column
  holds only the clips where both have a BD. The arms' own-set medians are in
  `analysis-all.txt`.
- Per-rung achieved/target rate (median over clips):

  | Family | Record (300 / 500 / 800 / 1200) | Arm |
  |---|---|---|
  | libvpx cpu7 | 1.84 / 1.24 / 1.10 / 1.09 | max-q 63: 1.10 / 1.09 / 1.08 / 1.08 |
  | libaom cpu8 | 1.47 / 1.10 / 1.02 / 1.01 | mapping: 1.11 / 1.09 / 1.09 / 1.08 |
  | SVT, Kvazaar, openh264 | 0.95-1.05 at every rung | 0.95-1.05 at every rung |

**The common set over the 13 families together** (`common.txt`; x265 medium unchanged,
openh264 with scene-change on):

| Commands | Clips where every family has a BD | Lost clips, by family |
|---|---|---|
| Record | 18 of 24 | libvpx cpu7: 6; libvpx cpu9: 5; libaom cpu8/cpu10: 6 each; SVT M8/M10: 2 each; x265: 2; others: 1 or 2 |
| Mapped | **22 of 24** | 1 or 2 per family; x265: 2; SVT M8/M10: 2 each |

**Answer to the owner's point on max-q 52 (selection).** It is not the audit's clip selection:
with max-q 52, libvpx and libaom have a BD on only 18-19 of 24 clips. What happens:
- With max-q 52, the bottom rung overshoots the target by 1.5-1.9x (median).
- Those clips' curves then lie above x264's quality range and drop out of their common set.
  The headline is computed on the clips that remain.
- On those remaining clips the clamped command looks 2-4 % better than max-q 63, because it
  spends up to 84 % more than the target on them, and BD-rate counts that only as rate.
- At max-q 63 it meets the rate (1.08-1.10x), keeps 23 of 24 clips, and its points over one
  buffer fall from 44 to 14 of 84.

The clamp's apparent BD advantage is overshoot plus selection, not coding efficiency.

## 4. Mapping, family by family

Sources:
- x264 b35605ac, x265 e444744c, libvpx v1.16.0 (1024874c), libaom v3.15.3 (b2347b8, our
  `tools/ref/libaom`), SVT-AV1 v4.3.0, openh264 v2.7.0, Kvazaar v2.3.2 (6040962b);
- ours: build 889309c99, this repository.

| Family | Flag | Rule | Source for the default or the constraint | Keep / change | Measured effect |
|---|---|---|---|---|---|
| ours | `--temporal-layers 1` | 4 (default) | `crates/nc-cli/src/main.rs:466` "(default 1)" | keep (redundant) | — |
| ours | `--strict-buffer --param rate.key_cap_buffer={KEY_CAP}` | 2 (1 s buffer) | `params.rs:1375` `key_cap_buffer` default 1.0; the strict bound is the owner's BP-3.13 decision | keep | — |
| ours | `--param rate.first_key_lookahead_off=1` | 2 (no lookahead) | `params.rs:1334-1335`: turns off the start-up lookahead's first-key cap (default 0 = on) | keep | — |
| ours | `--param hold.enable=0` | 3 (every frame coded) | `params.rs:135-139` | keep | — |
| ours | `--peak-bitrate {KBPS}`, `--buffer-ms 1000` | 2 (buffer, peak = target) | CLI default peak 2x (`--help`) | keep | — |
| x264 | `--tune zerolatency` | 1 | `common/base.c:675-683`: lookahead 0, sync lookahead 0, no B-frames, sliced threads, mb-tree off | keep | — |
| x264 | `--bframes 0 --rc-lookahead 0` | 2 (restate zerolatency) | `base.c:677-679` | keep (redundant) | — |
| x264 | `--no-scenecut` | **violates rule 4** (scene-cut keys are the encoder's choice) | default threshold 40 (`base.c:388`); scene-cut runs without lookahead (`encoder.c:1618` keeps low-res analysis whenever scenecut is on; `slicetype.c:1523`) | **drop** | one stream of 23 gained a key; BD identical |
| x264 | `--min-keyint {KEYINT}` | violates rule 4 (it forbids scene-cut keys) | — | **drop** (`--keyint {KEYINT}` stays: one key at the start) | as above |
| x264 | `--vbv-maxrate/--vbv-bufsize {KBPS} --vbv-init {VBV_INIT}` | 2 | — | keep | — |
| x264 | `--threads 1` | 3 | — | keep | — |
| x265 | `--tune zerolatency` | 1 | `source/common/param.cpp:670-680`: no B-frames, lookahead 0, **scenecut 0**, no hist scene-cut, cuTree off, frame threads 1 | keep | — |
| x265 | `--no-scenecut`, `--bframes 0`, `--rc-lookahead 0` | 4 (equal to zerolatency's defaults) | `param.cpp:674-677` | keep (redundant) | — |
| x265 | `--min-keyint {KEYINT}` | harmless (scene-cut is off under zerolatency) | — | keep | — |
| x265 | `--frame-threads 1 --pools none --no-wpp` | 3 (one thread): WPP and frame threads exist only for parallelism | — | keep | — |
| x265 | `{X265_ASM}` = `--asm avx512` on x86_64 | **violates rule 4** (a speed flag, not a constraint) | `source/common/cpu.cpp:158-226`: `cpu_detect` adds AVX-512 only when asked (`--asm avx512`) | **drop** (output identical by design; speed only, not measurable on Graviton) | — |
| openh264 | `--usage camera-realtime` | 1 | — | keep | — |
| openh264 | `--complexity medium/high` | rung (the family's speed setting, like `--preset`) | library default LOW (`param_svc.h:142`); harness default MEDIUM | keep as the rung | — |
| openh264 | `--scene-change 0` | **violates rule 4** | `param_svc.h:166` `bEnableSceneChangeDetect = true` | **change to 1** (or drop the flag) | median BD-J unchanged; mean +3.0 points; 3 of 23 streams gained keys (up to 9) |
| openh264 | `--adaptive-quant 0` | violates rule 4 | `param_svc.h:168` AQ true | **change to 1** | identical streams (84/84) |
| openh264 | `--background-detection 1 --denoise 0 --ltr 0 --temporal-layers 1` | 4 (equal to defaults) | `param_svc.h:165, 167, 170` | keep (redundant) | — |
| openh264 | `--frame-skip {SKIP}`, `--threads 1` | 3 | — | keep | — |
| SVT-AV1 | `--rtc 1 --pred-struct 1` | 1 | — | keep | — |
| SVT-AV1 | `--hierarchical-levels 0` | **violates rule 4** | `Source/Lib/Globals/enc_handle.c:4570-4589`: low-delay CBR defaults to 2 and is capped at 2 | **drop** | BD-J vs record +4.2 to +7.2 % (worse) at every preset; compliance unchanged (1-4 points of 84 over a buffer) |
| SVT-AV1 | `--buf-initial-sz {INIT_MS}` | 2 (starting full) | default 600 (`Docs/Parameters.md:118`) | keep | — |
| SVT-AV1 | `--buf-sz 1000` | 2 | default 1000 (`Parameters.md:117`) | keep | — |
| SVT-AV1 | `--buf-optimal-sz 600` | 4 (equal to the default) | default 600 (`Parameters.md:119`) | keep (redundant) | — |
| SVT-AV1 | `--scd 0` | 4 (equal to the default) | default 0 (`Parameters.md:259`) | keep (redundant) | — |
| SVT-AV1 | `--lp 1`, `--keyint {KEYINT}` | 3 | — | keep | — |
| libvpx | `--rt --cpu-used`, `--aq-mode=3`, `--noise-sensitivity=0`, `--static-thresh=1`, `--gf-cbr-boost=0`, `--frame-boost=0`, `--undershoot-pct=50 --overshoot-pct=50`, `--min-q=2`, `--max-intra-rate=1000`, `--resize-allowed=0` | 1 (the project's shipped real-time configuration) | `examples/vpx_temporal_svc_encoder.c`: min-q 2 (812), under/overshoot 50 (815-816), resize off (822), AQ 3 (910), GF CBR boost 0 (911), periodic boost 0 (913), denoiser off (914), static threshold 1 (915), max intra 1000 % (941-943) | keep | removing them all (CLI defaults): +10-11 % BD-J, worse compliance; most of that is AQ 3 (CLI + AQ 3: +1.6 to +2.6 %) |
| libvpx | `--max-q=52` | **owner decision: 63 on main rows** (52 presupposes dropping: `vpx_temporal_svc_encoder.c:814`, with `rc_dropframe_thresh` from argv, 810) | — | **main rows 63; `+drop` rows keep 52** | +2.3/+3.9 % direct on the clamped set's clips; clips with a BD 18-19 → 23; over one buffer 44 → 14 of 84 |
| libvpx | `--buf-sz=1000 --buf-initial-sz={INIT_MS} --buf-optimal-sz=600` | 2; optimal level as the example (818) | — | keep | — |
| libvpx | `--lag-in-frames=0`, `--threads=1`, `--passes=1`, `--drop-frame={DROP}` | 2, 3 | — | keep | — |
| libvpx | `--error-resilient=0` | 4 (default 0) | example: `g_error_resilient` from argv (832) | keep (redundant) | — |
| libaom | `svc_encoder_rtc` with `-sl 1 -tl 1 -lm 0 -aq 1 -th 1` | **violates rule 2:** the example hard-codes a 600 ms initial buffer (`examples/svc_encoder_rtc.cc:1911`) and max-q 52 (1908), not the full 1 s | — | **main rows: `aomenc --rt` with the example's real-time settings**, initial 1000 ms, max-q 63; **`+drop` rows: the example as shipped** | -0.5 % (cpu8) and -1.0 % (cpu10) direct; clips with a BD 18 → 23; over one buffer 35-36 → 10 of 84 |
| libaom | (example settings carried to aomenc) | 1 | `svc_encoder_rtc.cc`: min-q 2 (1907), under/overshoot 50 (1909-1910), buffer optimal 600 (1912), max intra 300 % (1720), AQ 3 from `-aq 1` (2037), GF CBR boost 0 (2038), CDEF on (2039), loop-filter control 1 (2040), warped/OBMC/global motion/order hint/TPL off (2041-2045), deltaq 0 (2046), cost updates 3 (2053-2056, non-GOOD_QUALITY build), CDF update mode 1 (2058), CfL/smooth/angle-delta/filter intra off and default-tx-only intra (2061-2065) | carried as aomenc flags | as above |
| libaom | example's drop controls (`MAX_CONSEC_FRAME_DROP_MS_CBR`, `SVC_FRAME_DROP_MODE`, `POSTENCODE_DROP_RTC`) | presuppose dropping | `svc_encoder_rtc.cc:2080-2085` | main rows: not carried; `+drop`: shipped example | — |
| Kvazaar | `--gop 0` | **override, needed:** Kvazaar's default GOP is a 16-picture random-access pyramid with reordering (`src/cfg.c:137-138`, `kvz_gop_ra16`), which rule 2 forbids | — | **change to `--gop lp-g8d4t1`** | -6.1 to -8.6 % direct; clips with a BD 22-23 → 23; compliance about equal (15-17 → 16-19 of 84) |

**How `lp-g8d4t1` was resolved from source** (Kvazaar has no default low-delay GOP; its
low-delay configurations are the `lp-g{G}d{D}t{T}` family, README "LP-GOP syntax"):
- **t = 1:** every picture is decodable, with no reference skipping (the README's t). The
  references are the previous pictures (`cfg.c:1464-1491`). There is no reordering, as rule 2
  requires.
- **d = 4:** the deepest low-delay GOP Kvazaar's own CBR rate control supports. Its per-layer
  bit weights exist for low-delay GOPs of up to 4 layers (`src/encoder.c`: the 4-layer
  low-delay table from doi:10.1109/TIP.2014.2336550). With 5 layers it refuses ("Unsupported
  amount of layers (5) for lowdelay GOP", `encoder.c:800`; measured: all 252 encodes of
  `lp-g16d5t1` failed). The default RA16 has 5 levels.
- **g = 8:** the shortest GOP that holds 4 layers (layer k: POC mod 2^(4-k), `cfg.c:1445-1461`).
  It is also the README's own example length (`lp-g8d4t2`).
- **QP offsets:** the layer number, +1 to +4 (`cfg.c:1466`), Kvazaar's own.

## 5. CPU: the libaom CLI against the example

- **Same run, same machine, full mixed load:** `aomenc --rt` with its defaults took x1.24
  (cpu8) and x1.14 (cpu10) the example's wall time per point, median x1.34 / x1.23. That is
  a rough proxy.
- **With the example's settings** the aomenc arm ran on another machine (no bare-metal
  capacity), so it has no time ratio here.
- **The earlier measurement** (BP-2 response, N6): the CLI with the example's flags was 6-11 %
  slower, attributed to its wider reference set.
- **The real figure** comes from the BP-4 timing run on the changed command.

## 6. Recommended `docs/benchmark-fairness.toml` changes (CBR; not applied)

**x264** (`[[encoder]] family = "x264-{preset}[~{tune}]"`, `cbr`):

```diff
-cbr = [..., "--bframes", "0", "--rc-lookahead", "0", "--no-scenecut", "--keyint", "{KEYINT}", "--min-keyint", "{KEYINT}", "--bitrate", ...]
+cbr = [..., "--bframes", "0", "--rc-lookahead", "0", "--keyint", "{KEYINT}", "--bitrate", ...]
```

- The `why` entry for `--bframes 0 --rc-lookahead 0 --no-scenecut --keyint/--min-keyint`
  becomes `--bframes 0 --rc-lookahead 0 --keyint {KEYINT}`.
- Its reason: zerolatency's own values restated (rule 2, no reordering, no lookahead), and
  one key at the start (rule 3). Scene-cut keys are x264's own choice (rule 4).

**x265:**

```diff
-ARCH_ARGS = {"X265_ASM": {"x86_64": ["--asm", "avx512"], "aarch64": []}}   # eval/nceval/fairness.py:150
+ARCH_ARGS = {"X265_ASM": {"x86_64": [], "aarch64": []}}
```

- Better still, remove `{X265_ASM}` from both x265 commands and its `why` entry.
- Note: x265's default dispatch on Granite Rapids leaves AVX-512 off. Its speed falls; its
  streams do not change.

**openh264** (`eval/nceval/fairness.py:64-66` and the `cbr` command):

```diff
-    for name, complexity, cabac, aq in (("openh264-baseline", "medium", "0", "0"),
-                                        ("openh264-baseline~cabac", "high", "1", "0"),
-                                        ("openh264-baseline~aq", "medium", "0", "1")):
+    for name, complexity, cabac, aq in (("openh264-baseline", "medium", "0", "1"),
+                                        ("openh264-baseline~cabac", "high", "1", "1")):
-..., "--adaptive-quant", "{AQ}", "--scene-change", "0", ...
+..., "--adaptive-quant", "{AQ}", "--scene-change", "1", ...
```

- The `~aq` twin goes, because AQ on is now the main row. Its streams were already identical.
- The `why` entry for `--scene-change` names rule 4 (the library default,
  `param_svc.h:166`).

**SVT-AV1:**

```diff
-cbr = ["{BIN}/SvtAv1EncApp", "--rtc", "1", "--preset", "{PRESET}", "--pred-struct", "1", "--hierarchical-levels", "0", "--rc", "2", ...]
+cbr = ["{BIN}/SvtAv1EncApp", "--rtc", "1", "--preset", "{PRESET}", "--pred-struct", "1", "--rc", "2", ...]
```

- The `why` entry `--rtc 1 --pred-struct 1 --hierarchical-levels 0` becomes
  `--rtc 1 --pred-struct 1`.
- Its reason: SVT's real-time low-delay mode with its own CBR structure (2 levels,
  `enc_handle.c:4570-4589`), all references past (verified in the streams, section 2).
- Measured +4 to +7 % against flat P: the default is what rule 4 requires, not what scores
  best.

**libvpx:**

```diff
-..., "--min-q=2", "--max-q=52", "--overshoot-pct=50", ...
+..., "--min-q=2", "--max-q={MAX_Q}", "--overshoot-pct=50", ...
```

- `fairness.py` gains `MAX_Q`: "63" on the main rows (`DROP` "0"), and the example's "52" on
  the `+drop` rows, beside `VPX_AOM_DROP_THRESHOLD`.
- The `why` entry for the example settings cites `examples/vpx_temporal_svc_encoder.c` lines
  812-943 (section 4). The max-q entry names the owner's decision.

**libaom:**
- **Main rows:** a new CBR command for `aomenc`.
- **`+drop` rows:** keep `svc_encoder_rtc`.
- `fairness.py` then needs the libaom family's tool chosen per row (main: `aomenc`; `+drop`:
  `svc_encoder_rtc`) instead of the one `tool` key (`fairness.py:74-79`).

```toml
cbr = ["{BIN}/aomenc", "--rt", "--passes=1", "--lag-in-frames=0", "--threads=1",
       "--cpu-used={PRESET}", "--end-usage=cbr", "--target-bitrate={KBPS}", "--buf-sz=1000",
       "--buf-initial-sz={INIT_MS}", "--buf-optimal-sz=600", "--undershoot-pct=50",
       "--overshoot-pct=50", "--min-q=2", "--max-q=63", "--aq-mode=3", "--gf-cbr-boost=0",
       "--max-intra-rate=300", "--enable-cdef=1", "--loopfilter-control=1",
       "--enable-warped-motion=0", "--enable-obmc=0", "--enable-global-motion=0",
       "--enable-order-hint=0", "--enable-tpl-model=0", "--deltaq-mode=0",
       "--coeff-cost-upd-freq=3", "--mode-cost-upd-freq=3", "--mv-cost-upd-freq=3",
       "--dv-cost-upd-freq=3", "--cdf-update-mode=1", "--enable-cfl-intra=0",
       "--enable-smooth-intra=0", "--enable-angle-delta=0", "--enable-filter-intra=0",
       "--use-intra-default-tx-only=1", "--drop-frame=0", "--kf-min-dist={KEYINT}",
       "--kf-max-dist={KEYINT}", "--i420", "--width={W}", "--height={H}",
       "--fps={FPS_NUM}/{FPS_DEN}", "--limit={N}", "--ivf", "-o", "{OUTPUT}", "{INPUT}"]
```

- With `aomenc`, libaom also has `--tune=ssim`: its `~ssim` twin becomes possible in CBR, and
  `untuned_modes = ["cbr"]` can go.

**Kvazaar:**

```diff
-cbr = [..., "--gop", "0", "--period", "{KEYINT}", "--no-open-gop", ...]
+cbr = [..., "--gop", "lp-g8d4t1", "--period", "{KEYINT}", "--no-open-gop", ...]
```

The `why` entry: rule 2 (Kvazaar's default RA16 reorders), with the resolution of section 4.

**Every changed family is coded again once per platform** (procedure section 3): libaom
cpu8/cpu10 (main rows), libvpx cpu7/cpu9 (main rows), SVT-AV1 M8-M13, Kvazaar
ultrafast-veryfast, openh264 (main and ~cabac), and x264 every preset and tune. x264 is the
zero line, so every BD moves slightly. x265 changes speed only.

## 7. Limits of this audit

- **Small scale:** 24 tuning-pool clips, one platform (Graviton 4, aarch64 builds), 150
  frames.
- **Few scene cuts in these clips:** the scene-cut arms are a functional check, not an effect
  size on cut-heavy content.
- **Zero line:** BD-J against x264 medium's default tune for all three metrics (no twins).
- **Time:** wall time at full mixed load, a proxy. The stage-3 arms have no time ratio.
- **x265's AVX-512 change** is speed only and was not measured (no x86 run).
