# Second-reference and compound usage across codecs and rungs (2026-11-24)

**Question (owner).** Is the second-reference / compound usage correct and consistent
across codecs? Read-only review: nothing is changed here.

**Method.** Resolved settings per codec and rung from the sources in the order the engine
applies them (registry default, the backend's `DEFAULT_PARAMS`, `presets/slowest.toml` on
each codec's slowest rung, then `presets/<codec>/<rung>.toml`), head `53ab431aa`. The
backends' capabilities (`supports_second_reference`, `supports_compound_reference`,
`supports_primary_pair`, `SECOND_REFERENCE_WITH_LAYERS`) read from each `backend.rs`.
Stream-level check: every rung of every codec coded at CBR 600 kbps, 60 frames, on
`crowd_mot17_09` and `static_otw_homes390` (720p), with the encoder's label dump; per
frame, the leaves coded from the second reference (`MODE_SECONDARY`) and compound / pair
(`MODE_COMPOUND`, `MODE_PRIMARY_PAIR`). Script:
`~/nanocodec-data/scratch/lowdelay-bipred/review/` (resolve.py, nanocodec-review-refs.py).

## What the settings mean

- `structure.unified` 1: only high-quality (HQ, cadence-base) frames carry the second
  reference (the previous HQ frame), and only where the cadence strength is at least
  `structure.secondary_min_depth` (0.75); content decides (crowd: often none).
- `refs.second` 2 with `structure.unified` 0: every inter frame carries the last HQ frame.
- `refs.second` 2 with `structure.unified` 1: **the plan masks it** (`TierPlan::planned`):
  HQ frames only, exactly as `structure.unified` 1 alone.
- `refs.second_full`: the full mode decision against the second reference (1) or the
  cheap ranking (0). The fixed-QP profile (`FIXED_QP_PARAMS`) sets it to 1 for HEVC, AV1
  and VP9 (with `structure.secondary_min_depth` 0 for AV1 and VP9).
- Compound (`refs.second_compound_margin` 0.15): only where the backend supports it.
- References kept per frame: at most two (the previous frame in slot 0, the HQ frame in
  the long-term slot), plus the T0 slot under temporal layers.

## Table

Frames with a second-reference or compound leaf out of 60, and the share of inter leaf
area coded from it (second reference / compound), crowd | static.

| codec | rung | refs.second | unified | full | hold | second ref on | crowd | static |
|---|---|---|---|---|---|---|---|---|
| H.264 | fast, medium, slow | 0 | 0 | 0 | 0 | never | 0 | 0 |
| H.264 | slower (slowest) | 2 | 0 | 1 | 0 | every frame | 18, 0.8 % | 50, 8.8 % |
| HEVC | faster, fast | 0 | 1 | 0 | 0 | HQ frames | fast 0 | fast 7, 3.2 % |
| HEVC | medium | 0 | 1 | 0 | 0 | HQ frames | 0 | 7, 2.2 % |
| HEVC | slow | 0 | 1 | 0 | 0 | HQ frames | 5, 1.1 % | 7, 3.5 % |
| HEVC | slower | **2** | 1 | 0 | 0 | HQ frames (refs.second masked) | 5, 1.0 % | 7, 2.7 % |
| HEVC | veryslow (slowest) | 2 | 0 | 1 | 0 | every frame | 51, 9.0 % | 47, 11.1 % |
| VVC | faster, fast | 0 | 1 | 0 | 0 | HQ frames | fast 0 | fast 7, 3.1 % |
| VVC | medium, slow | 0 | 1 | 0 | 0 | HQ frames | 5, 1.6 / 1.5 % | 7, 3.5 / 3.7 % |
| VVC | slower | 0 (pinned) | 1 | 0 | 0 | HQ frames | 5, 1.4 % | 7, 2.8 % |
| VVC | veryslow (slowest) | 2 | 0 | 1 | 0 | every frame | 51, 9.2 % | 47, 12.3 % |
| AV1 | veryfast, faster, fast | 0 | **0 (pinned)** | 0 | 0 | never | fast 0 | fast 0 |
| AV1 | medium | 0 | 1 | 0 | 0 | HQ frames | 0 | 7, 0.8 % + 0.7 % comp |
| AV1 | slow | 2 | 0 | 0 | 5 s | every frame, held | 54, 4.2 % + 0.7 % comp | 50, 1.2 % + 4.6 % comp |
| AV1 | slower (slowest) | 2 | 0 | 1 | 0 | every frame | 51, 2.9 % + 0.8 % comp | 46, 1.0 % + 4.0 % comp |
| VP9 | veryfast, faster, fast | 0 | **0 (pinned)** | 0 | 0 | never | fast 0 | fast 0 |
| VP9 | medium | 0 | 1 | 0 | 0 | HQ frames | 0 | 7, 0.5 % |
| VP9 | slow | 2 | 0 | 0 | 5 s | every frame, held | 54, 2.5 % | 47, 2.9 % |
| VP9 | slower (slowest) | 2 | 0 | 1 | 0 | every frame | 51, 2.4 % | 45, 1.5 % |
| AV2 | faster | 0 | **0 (pinned)** | 0 | 0 | never | — | — |
| AV2 | fast | 0 | 1 | 0 | 0 | HQ frames | 0 | 0 |
| AV2 | medium | 0 | 1 | 0 | 0 | HQ frames | 5, 1.8 % | 7, 1.7 % |
| AV2 | slow | 0 | 1 | 0 | 0 | HQ frames | **0** | 7, 1.2 % |
| AV2 | slower (slowest) | 2 | 0 | 1 | 0 | every frame | 51, 6.2 % | 47, 8.9 % |

(Faster / veryfast rungs not coded: their settings equal the fast rung's in each row.)

## Syntax and backend use

| codec | syntax allows | our encoder uses | compound / bi-prediction |
|---|---|---|---|
| H.264 (High) | up to 16 refs, `ref_idx_l0`, long-term refs; B slices with two lists | 2 refs (`ref_idx_l0` 1 = long-term HQ frame) | none: P slices only (decoder supports B) |
| HEVC (Main) | up to 16 L0 refs (DPB by level); B / GPB slices | 2 refs (second short-term RPS entry) | GPB bi-prediction implemented, off (`decision.bi_pair`); `supports_compound_reference` only in B slices |
| VVC (Main 10) | multiple L0/L1 refs, B slices (BCW, GPM...) | 2 refs (RPL entry 1) | none: P only |
| AV1 | 7 refs of 8 buffers (LAST..ALTREF), compound (average, distance, wedge, diff) | LAST + HQ frame | average compound (`av1.compound` 1 everywhere) where a frame carries the second reference |
| VP9 | 3 refs (LAST, GOLDEN, ALTREF), compound only between refs of different sign bias | LAST + GOLDEN (HQ) | none: sign bias 0 on all refs |
| AV2 | up to 7 ranked refs, compound | rank 0 + rank 1 (`read_single_ref`) | none: not implemented (`backend.rs` "No compound prediction between the two") |

## Flagged

1. **Dead setting: HEVC slower `refs.second = 2`.** With `structure.unified = 1` on the same
   rung the plan masks it: HEVC slower carries the second reference on exactly HEVC slow's
   frames (crowd 5/60 at 24, 32, 40, 48, 56; static 7/60 at 8..56). The same masking made
   `refs.second=2` alone a no-op in the 2026-11-23 quicks (HEVC medium/fast, AV1 medium,
   VVC fast: J exactly 0.00 %).
2. **Unscreened pins, not codec reasons: AV1 veryfast/faster/fast, VP9 veryfast/faster/fast,
   AV2 faster** pin `structure.unified = 0` with "pinned before Default's unified reference
   plan" (dates 2026-11-14/15). Their HEVC and VVC siblings (fast, faster) carry the HQ
   plan (`structure.unified = 1` set explicitly), and so do AV2 fast and every medium rung.
   No codec reason is stated (shared-first rule); the fast rungs of AV1/VP9/AV2 have never
   been screened with it.
3. **H.264 never carries a second reference below its slowest rung.** H.264's Default does not
   enable the unified plan that every other codec's Default does, so H.264 fast/medium/slow
   code no second reference while H.264 slower codes it on every frame (via slowest.toml).
   The ledger has measurements against it (`h-264-second-reference-last-hq-frame`: CBR
   −1.2 % at +23 %; unused-tools A/B #12 +1.0 %), but no comment in H.264's
   `DEFAULT_PARAMS` states the codec reason for leaving the plan off.
4. **Slow-rung inconsistency across codecs.** AV1 and VP9 slow carry the second reference on
   every frame (held 5 s, cheap ranking); HEVC, VVC and AV2 slow carry it on HQ frames only.
   The HEVC every-frame variant was screened and rejected at slow on 2026-11-24 (CPU +16 to
   +18 % for L1 −0.93 %); VVC and AV2 slow were not screened.
5. **Ladder non-monotonic on content: AV2 slow < AV2 medium on crowd** (0/60 frames vs 5/60
   at the same settings for the second reference: `structure.unified` 1, `refs.second` 0).
   Static is equal (7/60 both). Cause not known yet: AV2 slow differs from medium in its
   planner and propagation settings (`layers.propagation 0`, RD split), which may change
   the cadence strength the plan's depth gate reads; test: AV2 slow vs medium with the
   cadence depth logged per HQ frame.
6. **Second-reference share falls between slow and slower in AV1 and VP9** (crowd: AV1 4.2 %
   → 2.9 %, VP9 2.5 % → 2.4 %; static: AV1 compound 4.6 % → 4.0 %, VP9 2.9 % → 1.5 %).
   Slow holds the reference (`refs.second_hold 5`), slower (the shared slowest bundle)
   does not. Not wrong in itself, but the slowest rung drops the hold that the rung before
   it adopted; no stated reason.
7. **Compound coverage.** Only AV1 codes compound. AV2's syntax allows it but our backend
   does not implement it, and `docs/analysis/unused-tools-2026-11-21.md` lists AV2
   "Compound (two past refs) USED Slow `refs.second`": that row is wrong, AV2 slow codes
   single-reference blocks from rank 0 or 1 only. VP9 compound needs a sign-bias change
   (queued in that doc). HEVC GPB bi-prediction exists but is off on every rung (screened
   2026-11-23/24: does not pay at its search cost).
8. **`refs.second_compound_margin` is inert outside AV1** (HEVC only in B slices, which no
   preset enables); harmless, but it is listed as an engine parameter for every codec.

## Not flagged

- VVC slower pins `refs.second = 0` explicitly; it carries the HQ plan like its siblings.
- Each codec's slowest rung gets the shared every-frame bundle (`slowest.toml`) with a
  measured justification in that file.
- The fixed-QP profile's `refs.second_full = 1` (HEVC, AV1, VP9) only acts where a frame
  carries the second reference.
