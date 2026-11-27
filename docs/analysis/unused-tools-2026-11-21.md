# Unused coding tools, per codec (2026-11-21)

**Question (owner).** For each codec, which coding tools of the target profile do we not
use, and which of them could help? Inventory from the spec tool lists and the reference
encoders (JM/HM, VTM/VVenC, libaom/SVT-AV1, libvpx, AVM), checked against our encoders'
header writers, the shipped presets (`crates/nc-encoder/presets/`), the codec defaults
(`nanocodec params`, build 99e9e7fb = head b27e2a537's code), our decoders' parsers and the
ideas ledger. Then a fixed-QP A/B of the best implemented-but-off items on today's draw.

**Statuses.** USED (presets), OFF (implemented, off in every shipped preset), NI (not
implemented in the encoder), EX (excluded by design: B slices / bi-prediction, lookahead,
slices/tiles/WPP, > 8-bit, interlace, 4:4:4-only tools). **Verdicts** for OFF / NI items:
TRY NOW (cheap, plausible), QUEUE (worth it, larger), SKIP (reason).

**Decoder work.** Small everywhere: H.264 is a full progressive High decoder (CAVLC/CABAC,
all I/P macroblock types incl. sub-8x8, scaling matrices, explicit WP, long-term refs);
HEVC passes 147/147 HEVC_v1 conformance streams (AMP, PCM, lossless, scaling lists, WP,
transform skip); VVC decodes 138/138 VTM/VVenC tool streams bit-exactly (only palette is
refused, `validate_tools`); AV1 decodes all 242 libaom Main vectors, but its frame gate
(`nc-av1/src/decoder/subset.rs`) still refuses quantizer matrices (l. 97) and loop-filter
deltas (l. 168); VP9 passes 303/303 Profile 0 vectors; AV2's general decoder is 30/30 on
AVM v1.0.0 streams. So decoder work is "none" unless stated.

**Reference ablations used, and their limits.** VVC: the two VVenC ablations in
`docs/handoff/vvc.md` ("vr": one tool removed from VVenC restricted to about our tools;
"oracle": one tool added to a QT-only base), low delay, our clips. AV1: Meta's "Analysis of
AV1 Coding Tools" (libaom, **random access**, PSNR-Y tool-off / tool-on, written "FB
off / on"): it overstates each tool for low delay. HEVC: JCTVC-C149 (AMP off, 2010) and
Vanne et al. (AMP/SMP, low-delay B). AV2: Nalci et al. "Transform and Entropy Coding in
AV2" (IST). Where no source gives a number, the gain is "unknown". No numbers were invented;
every ledger figure carries its slug.

## Summary: ranked

### TRY NOW

Implemented but off and cheap; A/B on today's draw.

Fixed QP `nceval.fixedqp ab`, draw fixedqp-2026-11-21 quick (24 clips, 150 frames, matched
QP), J = PSNR-YUV / VMAF v1 / MS-SSIM thirds, 90 % CI; cost = x86 P-frame instructions
(`nceval.fixedqp cost`, callgrind); J_k = J + k * cost (k 0.1 Default, 0 Slow). Anchor:
build 99e9e7fb (head b27e2a537's code) with its own presets on both sides.

| # | codec, preset: change | prior evidence | fixed QP J | P instr | J_k | read |
|---|---|---|---:|---:|---:|---|
| 1 | AV2 Default: `av2.ref_mv_bank=1 av2.max_drl_bits=3` (Slow has 1 / 5) | -0.41 / -0.32 alone (`av2-cheap-fixed-qp-medium-candidates`) | **-1.7 [-2.4, -1.2]** (P/V/M -1.7/-1.7/-1.8; 2/24 worse, worst +0.6) | +4.8 % | -1.2 | **screen (CBR L1-L3 + final-check)** |
| 2 | AV2 Default: `av2.ref_mv_bank=1` | -0.41 [-0.75, -0.07] at +0.6 % | -0.9 [-1.4, -0.4] | +0.7 % | -0.8 | in #1 |
| 3 | AV2 Default: `av2.max_drl_bits=3` | -0.32 [-0.60, -0.04] at +1.6 % | -0.8 [-1.1, -0.5] | +1.3 % | -0.7 | in #1 |
| 4 | VVC Slow: `vvc.sao_search_interval=1 vvc.sao_stats_row_step=1` | Slow re-pick: with the SAO part -4.5 % vs shipped -4.2 % | **-0.4 [-0.6, -0.3]** (5/24 worse, worst +0.4) | +2.8 % | -0.4 | **screen (k 0)** |
| 5 | AV2 Default: `av2.interp_switchable=1` | fixed QP -1.20 [-2.28, -0.24] at +1.7 %; CBR (2026-11-09) +0.4..+0.9 | -0.3 [-0.6, +0.0] (PSNR -1.4, VMAF +1.0; 9/24 worse) | +1.9 % | -0.1 | no (VMAF reads the smooth filter's blur, as on 2026-11-09) |
| 6 | VP9 Default: `decision.intra_gate_variance=0` | held, VC-1K worst six -2.6 [-6.0, -0.3] | -0.0 [-0.2, +0.1] | +0.2 % | +0.0 | no (worst-six gain does not reach the pool) |
| 7 | AV2 Default: `av2.deblock_delta=1` | held, worst six -0.5 [-1.0, +0.1] | -0.0 [-0.3, +0.2] (11/24 worse) | -0.2 % | -0.0 | no |
| 8 | AV1 Default: `decision.intra_gate_lambdas=0.03 decision.intra_test_dc=1` | held, worst six -2.6 (VMAF +6.1) | +0.2 [-0.3, +0.6] (VMAF +1.4) | +0.1 % | +0.2 | no |
| 9 | VP9 Default: `decision.intra_gate_lambdas=0.03` | held, worst six -0.7 | +0.3 [+0.1, +0.6] | -0.2 % | +0.3 | no |
| 10 | AV1 Default: `av1.loop_filter_level_scale=1.5` | held, worst six -0.7 [-1.1, -0.3] | +0.6 [+0.1, +1.0] | -0.8 % | +0.5 | no |
| 11 | AV1 Default: `av1.segment_qp=1` | never measured | +0.8 [+0.6, +1.0] | +0.2 % | +0.8 | no (segment ids cost more than the delta-q they replace; not traced) |
| 12 | H.264 Slow: `refs.second=2 refs.second_full=1` (Slow pins 0) | shared lever -1.5..-2.0 elsewhere; inside the all-on bundle -5.1 with vs -5.6 without | +1.0 [-0.2, +1.6] (MS-SSIM +1.2) | +31.1 % | +1.0 | no: keep the pin |

On #12: the pin's comment says H.264 "cannot signal [the second reference] beside the
Default's temporal layers" and codes the stream flat, but `SECOND_REFERENCE_WITH_LAYERS` is
`true` in `nc-h264/src/backend.rs` since 2c97f3be2 (2026-11-15). Whether the stream still goes
flat was not checked; the result supports keeping the pin either way, but the comment's reason
should be re-checked by whoever next edits that file.

**Adoption (same day).** #1 landed as the AV2 Default (4e5f3bdf5): pool k 0.1 L3 -1.00 %
[-1.26, -0.74], final-check -0.68 % [-1.09, -0.54], no guard (`docs/handoff/av2.md`, "MV bank
and DRL 3 at the Default"). Carried to AV2 Fast and Faster at k 0.6 it lost (L1 +0.95 % /
+2.07 %); both pin the old values. Their cheaper forms at k 0.6 (whole-encode dCPU): Fast
bank alone (+1.1 %) L3 +0.42 % WORSE; Fast DRL 2 (+0.8 %) L3 +0.18 % AMBIGUOUS (final-check
quality -0.80 %); Faster bank alone (+1.4 %) L1 +0.67 % [+0.26, +1.10]; Faster DRL 2 (+0.9 %)
L1 +0.02 %, not carried further. None landed.

**Follow-ups (same day, fixed QP 24 clips then the pool checks):**

| change | fixed QP J | P instr | pool | outcome |
|---|---:|---:|---|---|
| VVC Slow: SAO every picture and row (#4) | -0.4 [-0.6, -0.3] | +2.8 % | k 0: L3 -0.59 % [-1.00, -0.19], final-check -0.63 % [-0.87, -0.20] | landed 3a8fbea1e |
| HEVC Default: SAO statistics every 2nd row (`hevc.sao_stats_row_step` 2) | -1.0 [-1.2, -0.6] | +2.6 % (whole encode +2.75 %) | k 0.1: L3 -0.43 % [-0.68, -0.19], final-check -0.94 % [-1.11, -0.65] | landed c55484228 |
| HEVC Default: SAO statistics every row | -1.2 [-1.3, -0.9] | +7.9 % | - | rows 2 kept (cheaper) |
| HEVC Default: SAO every picture + band (+ every row) | -0.2 / -1.9 | +9.8 / +35.0 % | - | no |
| VVC Default: SAO every picture / + every row | -0.3 / -0.5 | +2.6 / +11.1 % | - | no |
| VVC Default: SAO statistics every 2nd row | -0.5 [-0.8, -0.3] | +1.3 % | in the pool checks | see the ledger |
| AV2 Default: CDEF search every 4th / 2nd row (period 8) | -0.6 / -0.9 | +6.7 / +19.9 % | - | no at k 0.1 |
| VP9 Default / Slow: loop-filter reference/mode deltas (new code, `vp9.lf_ref_deltas`) | -0.1 / +0.1 | - | - | no; branch `unused-tools-vp9-lf-deltas` |

Not carried: the SAO row step to HEVC Fast/Faster (k 0.6: -1.0 + 0.6 x 2.6 > 0); AV1's CDEF
search already samples every row at its Default.

Not A/B'd here: HEVC Default `hevc.sao_band 1` + `sao_search_interval 1` (VC-1K -0.0 at
+2.6 % for band alone; Slow already has both); VVC Slow `decision.final_stage 1` (another
agent's, held today at fixed QP -0.9, `final-decision-stage-on-the-slow-prese`); AV1
superres (`av1.superres_denom`, starved CBR only, no fixed-QP meaning); film grain (visual
judgement only).

### QUEUE

Needs code; ranked by expected value.

| # | item | codecs | evidence | enc cost | dec work |
|---|---|---|---|---|---|
| 1 | RD (coded-cost) partition search with rectangular shapes: HEVC 2NxN/Nx2N then AMP; AV1/VP9/AV2 HORZ/VERT (then 4-way/H/V for AV1/AV2) | HEVC, AV1, VP9, AV2 | VVC quadtree RD split -9.1 % at +99 % instr (`vvc-mtt-under-the-rd-partition-search`); FB rect off +5.11 % (RA); AMP off +1.0..3.8 % (JCTVC-C149), -0.8 % at +14 % (HM-6); our cheap-proxy shapes lose (tools-that-hurt §2, §4, §5) | high (Slow) | none |
| 2 | AV1 switchable interpolation filter (`is_filter_switchable` 0 today) | AV1 | VP9's fixed regular filter rejected (Faster J +3.5, Fast guards); AV2's switchable -0.3 today (VMAF +1.0); libaom RT keeps a nonrd filter search | low-medium (VP9's search exists) | none |
| 3 | VVC intra tools on keys: MIP, intra dual tree | VVC | VVenC vr: MIP -1.9, dual tree -2.6 (pan/static -4.5); oracle -0.6 | medium (key frames) | none |
| 4 | AV2 CCSO search (cheap, SIMD) | AV2 | AVM oracle -2.84 J; ours -0.8 at +539 % (Slow) | very high today | none |
| 5 | AV2 temporal MV fields; flexible MV resolution / adaptive MVD | AV2 | VVC TMVP -0.4..-1.9 ours, AMVR -2.5 ours (analogues); no AV2 number | medium | none |
| 6 | AV1 transform types (reduced set) for Slow | AV1 | FB reduced set off +2.61 %, flip+IDTX off +1.62 % (RA); AV2 inter types -0.7 at +7 % | +8-15 % | none |
| 7 | AV2 IST | AV2 | Nalci et al. low delay VMAF -1.24 % (adoption-time) | medium-high | none |
| 8 | VVC SBT, LFNST | VVC | VVenC oracle SBT -1.7 (vr noise); LFNST -1.1 vr / -0.4 oracle | medium | none |
| 9 | Weighted prediction (drifting exposure) | H.264 (HEVC, VVC, AV2 BAWP) | H.264 upper bound -1.4 % on 6 hard clips, one dark clip -7.4 % (`weighted-prediction-for-vc-1k-brightness-drift`); HEVC ~0 overall | low | none (H.264/HEVC/VVC); AV2 BAWP none |
| 10 | Small header-level tools: VP9 loop-filter ref/mode deltas; HEVC strong intra smoothing; AV1 intra edge filter | VP9, HEVC, AV1 | libvpx defaults LF deltas on (INTRA +1, GOLDEN/ALTREF -1); AV1 edge filter FB -0.36 % at +2.5 %; HEVC unknown | low | none |
| 11 | AV1 order hints, then skip mode / ref-frame MVs | AV1 | FB MFMV on -1.06 % (RA, decode +17 %); SVT rtc sets mfmv off; skip mode unknown in LD | low-medium | none |
| 12 | VVC affine beside the second reference; SbTMVP | VVC | ours affine -0.69 [-1.19,-0.19] fixed QP; VVenC vr +0.7 / oracle -3.9; SbTMVP vr -0.9 | +15-41 % | none |
| 13 | VP9 compound (GOLDEN sign bias 1) | VP9 | our AV1 average compound -0.72 pp; libvpx RT uses compound only with lag | medium | none |

## Per-codec inventory

### H.264 (High, profile_idc 100, CABAC; IDR + P)

| tool | status | presets / param | evidence | verdict |
|---|---|---|---|---|
| CABAC | USED all | `h264.cavlc` 0 | CAVLC +10/+14/+19 % bytes at QP 32/38/44 (`h-264-cavlc-slice-data`) | - |
| 8x8 transform; RD 4x4/8x8 choice | USED all; RD choice Slow only (`h264.transform_size_rd` 1) | | RD choice +23 % P instr | - |
| Intra 8x8 | USED (Default/Veryfast every frame, Fast/Faster keys; Slow all 9 modes) | `h264.intra8x8` | fixed QP BD -7.75 % (`h-264-intra-8x8-i-nxn…`) | - |
| Intra 4x4 | USED all (shortlist in fast rungs; RD over 9 modes in Slow) | | Slow fixed QP -4.4 % at x1.87 | - |
| 16x8/8x16/8x8 | USED, SATD-gated (Veryfast off) | `h264.partition_search` | open search +1.9 % unless coded-cost decided (tools-that-hurt §2) | - |
| Sub-8x8 (8x4/4x8/4x4) | NI (only P_L0_8x8 written) | | x264 keeps p4x4 off by default; our 8x8 splits already fit noise | SKIP |
| Multiple refs | partial: `refs.second` (last HQ frame, long-term slot), 0 everywhere; Slow pins 0 | `refs.second` | CBR -1.2 % at +23 % (`h-264-second-reference-last-hq-frame`); x264 ref=1 on VC-1K only +0.8 | tried: +1.0 % (A/B #12), keep the pin |
| Long-term refs | USED opt-in for loss recovery | `ltr.interval` | resilience | - |
| Weighted prediction | NI | | upper bound -1.4 % (one dark clip -7.4 %) | QUEUE (low) |
| Scaling matrices | NI (flat) | | perceptual tool; unknown, likely <= 0 on PSNR/VMAF | SKIP |
| Deblock offsets | OFF | `h264.deblock_alpha_offset`, `beta_offset` | +1/+1 +0.1 %, +2/+2 +0.2 % | SKIP |
| mb_qp_delta | USED by the reuse term; variance AQ OFF | `rate.aq_strength` 0 | variance AQ rejected 6 times (+3.7 % at 1.0) | SKIP (AQ) |
| Trellis | USED Default/Slow; off in fast rungs | `quant.trellis` | 8 states -0.2..-0.5 for +33..128 % | - |
| RD vector refinement | OFF | `h264.rd_refine` | -2.5 % at +96 %; on Slow -0.1 for +39 points | SKIP |
| Direct, B, bi-pred, interlace, slices | EX | | | - |

### HEVC (Main, 8-bit 4:2:0; CTU 64, CU 8-64, TU 4-32)

| tool | status | presets / param | evidence | verdict |
|---|---|---|---|---|
| CU depth, 64 CTU | USED; 8x8 inter CUs off | `planner.inter_min_log2` 4 | 8x8 +8..10 % by proxy; needs the RD split (QUEUE 1) | QUEUE |
| TU tree, 32x32 TU | USED (Faster TU 16, no inter split) | | TU16 +4.9 % BD | - |
| Intra NxN, 4x4 DST | USED Default/Slow (off Fast/Faster) | `hevc.intra_nxn` | key BD -6.9 % | - |
| 2NxN / Nx2N PUs | NI (2Nx2N only; engine shape hooks unwired, no-op audit item 7) | | Vanne et al. AMP+SMP 0.3-1.3 % (LDB) | QUEUE (1) |
| AMP | NI | | JCTVC-C149 off +1.0..3.8 %; HM-6 -0.8 % at +14 % time; x265 off below slow | QUEUE (after 2NxN) |
| TMVP | USED | | | - |
| Merge candidates | USED 4 (Faster 2); 5 within noise | `hevc.merge_candidates` | L3 -1.01 % for 4 | - |
| SAO edge / band | edge USED all; band Slow only | `hevc.sao_band` | band VC-1K -0.0 [-0.3,+0.3] at +2.6 % | SKIP at Default |
| Sign data hiding | USED (constant) | | | - |
| cu_qp_delta | USED, CTB quant groups, reuse term; variance AQ off | | AQ -0.09 % held out, VMAF guard | SKIP (AQ) |
| RDOQ | USED Default/Slow | `quant.trellis` | | - |
| Multiple refs | USED Slow (second ref as a short-term RPS entry) | `refs.second` 2 | CBR L3 -1.65 % at +24.8 % | - |
| Weighted prediction | NI | | worst six +0.4 %, drifting bodycam -4.2 % | SKIP (unless drift clips are a target) |
| Scaling lists, transform skip, PCM, lossless | NI | | no gain on camera content (TS: screen content) | SKIP |
| Strong intra smoothing | NI (SPS flag 0; the shared `predict_smoothed` exists) | | HM default on; unknown, 32x32 intra only | QUEUE (10, small) |
| Deblock offsets | OFF | `hevc.deblock_beta_offset`, `tc_offset` | 2/2 final-check +0.31 %, guard | SKIP |
| Final RD stage | OFF (pricing hooks wired 2026-11-21) | `decision.final_stage` | Slow fixed QP -0.3 (held) | other agent |
| WPP, tiles, B, RExt | EX | | | - |

### VVC (Main 10 profile, coded 8-bit)

| tool | status | presets / param | evidence | verdict |
|---|---|---|---|---|
| QT, 64 CTU, 64-point transform | USED | `vvc.tx64` | worst six -3.5 | - |
| MTT (BT/TT) | OFF | `vvc.mtt` 0 | Slow +4.6 %; under the RD split mtt 2 -10.7 % at +283 % vs QT RD -9.1 % at +99 %; VVenC -4.1/-6.3 vr | SKIP until the RD split is cheaper |
| Intra dual tree | NI | | VVenC vr -2.6 (pan/static -4.5), oracle -0.6 | QUEUE (3) |
| 128 CTU | NI | | unknown | SKIP |
| Affine (+PROF) | OFF; refused beside the second ref | `vvc.affine` | fixed QP -0.69 [-1.19,-0.19] at +15-17 %; VVenC vr +0.7 | QUEUE (12) |
| MMVD | NI (branch rejected) | | J +0.3 (`vvc-mmvd`) | SKIP |
| CIIP | NI | | VVenC within noise | SKIP |
| SbTMVP | NI | | VVenC vr -0.9 | QUEUE (12, low) |
| TMVP, AMVR, HMVP, merge 6 | USED | | TMVP -1.9, AMVR -2.5 | - |
| LMCS | NI | | no ablation found at low delay | SKIP (unknown) |
| ALF luma | USED (Fast off) | | -2.9 | - |
| CCALF | NI | | VVenC 0.0 | SKIP |
| SAO | edge USED (every 4th picture/row); band OFF | `vvc.sao_search_interval`, `vvc.sao_band` | Slow re-pick: with the SAO part -4.5 % at x2.44 vs shipped -4.2 % at x2.13; band -0.0 (held) | TRY NOW (Slow): -0.4 at +2.8 % (A/B #4); band SKIP |
| Dependent quantisation | OFF | `vvc.dep_quant` | +6.6 % [+5.8,+8.0]; VVenC: noise | SKIP |
| SDH, trellis | USED | | | - |
| JCCR | NI | | VVenC -0.1 | SKIP |
| MTS | OFF | `vvc.mts` | Slow 0.0 at +79 %; implicit +0.6 | SKIP |
| LFNST, SBT | NI | | LFNST -1.1 vr; SBT -1.7 oracle | QUEUE (8) |
| ISP | USED on keys (Default/Slow) | | P-frame ISP adds nothing | - |
| MIP | NI | | VVenC vr -1.9 | QUEUE (3) |
| MRL | NI | | VVenC -0.6 / -0.5 | SKIP (small) |
| CCLM | NI (branch rejected) | | J +0.25; VVenC 0.0 / -0.5 | SKIP |
| IBC, TS, BDPCM, palette, ACT | NI / EX | screen content, 4:4:4 | VVenC codes no IBC CU on camera content | SKIP / EX |
| Weighted prediction | NI | | unknown (fades) | QUEUE (9, low) |
| RPR | NI | | owner: scaled references are a bodge | EX |
| Second reference | USED Slow | `refs.second` | Default held -1.75 L3 at Jetson +8.2 % | - |
| GPM, BDOF, DMVR, SMVD, BCW | EX (bi-prediction / B slices) | | | - |

### AV1 (Main, 8-bit 4:2:0)

Header facts (`nc-av1/src/obu.rs` ll. 533-540): 128x128 SB, intra edge filter,
inter-intra, masked compound, warped motion, dual filter and order hints are all written 0;
`is_filter_switchable` 0 (l. 900). No order hint rules out skip mode, ref-frame MVs and
distance-weighted compound until it is written (decoder supports all three).

| tool | status | presets / param | evidence | verdict |
|---|---|---|---|---|
| Average compound (two past refs) | USED Default (unified HQ frames), Slow, L2/L3 | `av1.compound` | -0.72 pp PSNR BD for <1 % | - |
| Distance-weighted compound | NI (order hint) | | FB on -0.58 % | SKIP |
| Wedge / diff-weighted | NI | | FB wedge -1.64 % at 128 % enc time; libaom RT prunes by the average | SKIP |
| Inter-intra | NI | | FB off +0.33 %; SVT off M9+ | SKIP |
| OBMC | NI (search-free version removed) | | J_real +2.8 % (`av1-search-free-obmc`); SVT off under CBR | SKIP |
| Local warp / global motion | NI | | FB warp off +0.86 %; libaom RT off at speed >= 7; GM FB +0.06 % | SKIP |
| Switchable interp | NI (regular only) | | see QUEUE 2 | QUEUE (2) |
| Dual filter | NI | | FB -0.02 % | SKIP |
| Palette, IntraBC | NI | screen content | | SKIP |
| CfL, filter intra, angle deltas, smooth, Paeth | USED on keys (Default/Slow), P intra via `av1.inter_intra_search` | | P-frame CfL rejected | - |
| Intra edge filter | NI | | FB on -0.36 % at +2.5 % | QUEUE (10) |
| Transform types | NI (DCT only) | | FB reduced set off +2.61 % (RA) | QUEUE (6) |
| 64x64 tx, tx split (one level) | USED | | | - |
| Rectangular / 4-way / H / V partitions | NI | | FB rect off +5.11 % (RA) | QUEUE (1) |
| 128 SB | NI | | FB off +0.72 % | SKIP |
| CDEF | USED | | | - |
| Wiener restoration | OFF (Slow trial rejected) | `av1.restoration` | fixed QP -0.88 at +146 %; Slow +0.3 at +142 % | SKIP |
| SGR | NI | | libaom RT and SVT rtc off | SKIP |
| Superres | OFF | `av1.superres_denom` 8 | never measured; overlaps adaptive resolution | probe under starved CBR (not fixed QP) |
| Segmentation (quantizer) | OFF | `av1.segment_qp` | never measured | tried: +0.8 % (A/B #11) |
| Delta q | USED (reuse term) | | | - |
| Delta lf | NI; our decoder refuses it | | unknown | SKIP |
| Film grain | OFF | `grain.enable` | held, visual only | visual A/B |
| Reference scaling | USED (`resolution.adaptive`, spatial layers) | | | - |
| Skip mode, ref-frame MVs | NI (order hint) | | FB MFMV on -1.06 % (decode +17 %); SVT rtc mfmv off | QUEUE (11) |
| 7 references | partial (LAST + HQ) | | FB full set -0.83 % at 141 % | SKIP |
| Quantizer matrices | NI; decoder refuses | | unknown | SKIP |
| Deblock level scale | USED 1.0 | `av1.loop_filter_level_scale` | x1.5 worst six -0.7 | tried x1.5: +0.6 % (A/B #10) |
| disable_cdf_update | OFF | `av1.cdf_update` | non-base J +1.6/+1.8 | SKIP |
| 1/8 MVs | NI | | VP9 1/8 +1.7 % | SKIP |

### VP9 (profile 0)

| tool | status | presets / param | evidence | verdict |
|---|---|---|---|---|
| Compound | NI (sign bias 0) | `refs.second_compound_margin` inert | libvpx RT only with lag | QUEUE (13) |
| LAST + GOLDEN | USED (Default unified, Slow every frame) | | Slow second ref -1.0 % | - |
| Switchable interp | USED | `vp9.interp_filter` 3 | fixed regular rejected | - |
| TX sizes / tx_mode select | USED | `vp9.inter_tx_depth` | depth 0 +6.7 % | - |
| Segmentation (quantizer) | USED via reuse term | | | - |
| LF ref/mode deltas | NI (`mode_ref_delta_enabled` 0) | | libvpx default on | QUEUE (10) |
| LF level / sharpness | USED (scale 0.6274 Default) | | sharpness rejected | - |
| Reference scaling | USED in spatial layers | | | - |
| 1/8 MVs | OFF | `vp9.high_precision_qindex` 0 | +1.7 % even after the fix | SKIP |
| Rectangular partitions | NI (edges only) | | libvpx RT variance partitioning uses them | QUEUE (1) |
| Intra modes (10) | USED on keys; P intra DC only | | | - |
| P-frame intra gate | OFF at 6.857 | `decision.intra_gate_variance` | held, worst six -2.6 | tried: 0.0 / +0.3 % (A/B #6, #9) |
| Error resilient / frame parallel | USED with layers | | | - |
| Probability adaptation | USED (backward + coef updates) | | mode/MV forward updates <= 0.64 % | - |

### AV2 (AVM)

| tool | status | presets / param | evidence | verdict |
|---|---|---|---|---|
| Square partitions to 64 | USED | | | - |
| 128/256 SB, rect/extended/uneven partitions, SDP | NI | | VVC MTT paid only under full RD | QUEUE (1) |
| Base intra modes, angle deltas | USED | | | - |
| CfL | NI | | AV1 key CfL -0.90 key-only, mixed | SKIP |
| DIP, MRL, edge filter, IBP, MHCCP | NI | | no number found | unknown |
| Palette, IBC, FSC, CCTX | NI | screen content / untested | | SKIP |
| Intra DCT/ADST, tx split | USED | | | - |
| Inter transform types | OFF | `av2.inter_tx_types` | -0.7 at +7.1 %; in Slow +0.2 | SKIP |
| IST | NI | | Nalci et al. LD VMAF -1.24 % | QUEUE (7) |
| TCQ | USED Slow | `av2.tcq` | Default -2.27 at +40.9 % | - |
| Parity hiding | USED | | | - |
| Skip mode | USED Slow | `av2.skip_mode` | -0.3 at +12-17 % | - |
| Second reference (single-reference blocks from rank 0 or 1) | USED Slow (HQ frames) | `refs.second` | Default held -1.46 L3 at +6.5 % | - |
| Compound (two past refs) | NI (2026-11-24 correction: the backend codes no compound, `backend.rs`; the row said USED) | | none | QUEUE (low) |
| Masked/wedge compound, CWP, inter-intra | NI | | none | QUEUE (low) |
| Warp variants, global motion | NI | | none (VVC affine analogue -0.7) | QUEUE (low) |
| OPFL, refine-MV, TIP | NI | | bi-directional tools (AVM slides) | EX (likely) |
| BAWP | NI | | none | QUEUE (9) |
| Adaptive MVD, flexible MV resolution | NI | | VVC AMVR -2.5 (analogue) | QUEUE (5) |
| Ref MV bank, DRL size | USED Slow; Default 0 / 1 | `av2.ref_mv_bank`, `av2.max_drl_bits` | -1.7 together at +4.8 % (A/B #1) | TRY NOW (screen) |
| Temporal MV fields | NI | | analogues only | QUEUE (5) |
| Switchable interp | OFF | `av2.interp_switchable` | -0.3 at +1.9 %, VMAF +1.0 (A/B #5) | SKIP |
| Cross-frame CDFs | USED | | -3.4 | - |
| Deblocking, delta | USED; delta 0 | `av2.deblock_delta` | -0.0 (A/B #7) | SKIP |
| CDEF | USED | | full search in Slow -1.1 at +46 % | - |
| CCSO | OFF | `av2.ccso` | Slow -0.8 at +539 %; oracle -2.84 | QUEUE (4) |
| GDF, Wiener-NS, PC-Wiener | NI | | oracle -1.73 (GDF, ~10 % encoder instr), -0.52 / -0.07 | SKIP (decided) |

## Records

Ledger: the A/B and cost rows were written by `nceval.fixedqp ab` / `cost` (spool). Ideas:
`unused-tools-*` slugs, one per A/B row (held for #1 and #4, rejected for the rest) and one
held record per QUEUE item. Results: `~/nanocodec-data/work/unused-tools/ab_*.json`,
`cost_*.json`.

## Filter-search density by frame role (owner request, 2026-11-21)

Question: do the frame-level filter searches (HEVC/VVC SAO statistics, VVC ALF statistics,
AV1/AV2 CDEF search) pay more on some frames than others, so that one density per preset
leaves gain or CPU on the table? Roles: key frames (`key.<name>`, new engine prefix, backend
per-frame parameters only), base frames (every 4th frame at the Defaults, plus keys) and
non-base frames (`nonbase.<name>`). The Defaults are flat (all frames referenced), so
"non-reference" frames do not exist there.

Fixed QP (draw fixedqp-2026-11-21, 24 clips, matched QP, rented box) against each codec's
Default at head; "low" is the BD over the curves' lower-rate half (coarser steps); P instr.
from `nceval.fixedqp cost` (x86 callgrind, key + 12 P frames, so key-only changes show
little there).

| codec: setting (Default value) | J [90 % CI] | low half | P instr |
|---|---:|---:|---:|
| HEVC: SAO rows 1 everywhere (2) | -0.2 [-0.4, -0.1] | -0.25 | +5.2 % |
| HEVC: rows 1 on base frames | -0.1 [-0.3, +0.0] | -0.11 | +3.3 % |
| HEVC: rows 1 on keys | +0.0 [-0.2, +0.2] | | +0.2 % |
| HEVC: rows 4 on non-base | +0.1 [-0.1, +0.2] | +0.05 | -0.9 % |
| HEVC: rows 1 base, 4 non-base | -0.2 [-0.3, +0.0] | -0.16 | +2.2 % |
| AV2: CDEF search period 4 everywhere (8) | -0.6 [-0.8, -0.4] | -0.51 | +6.7 % |
| AV2: period 4 on base frames | -0.3 [-0.4, -0.0] | -0.16 | +2.9 % |
| AV2: period 2 on base frames | -0.7 [-0.9, -0.5] | -0.64 | +8.2 % |
| AV2: period 1 on base frames | -0.8 [-1.1, -0.6] | -0.76 | +19.5 % |
| AV2: period 1 on keys | -0.2 [-0.3, +0.1] | | +0.1 % |
| AV2: period 1 keys, 4 base, 8 non-base | -0.5 [-0.6, -0.3] | | +3.0 % |
| VVC: ALF rows 1 everywhere (2) | -0.9 [-1.2, -0.7] | -0.87 | +3.1 % |
| VVC: ALF rows 1 on base frames | -0.9 [-1.2, -0.6] | -0.83 | +0.7 % |
| VVC: ALF rows 1 on keys | -0.2 [-0.4, -0.0] | | +0.2 % |
| VVC: ALF rows 4 on non-base | +0.0 [+0.0, +0.0] | 0.0 | -0.0 % |
| AV1: CDEF search rows 2 on non-base (1) | +0.1 [-0.1, +0.2] | +0.06 | -1.2 % |
| AV1: rows 4 on non-base | +0.3 [+0.1, +0.4] | +0.22 | -1.5 % |

Readings:
- **The density gain sits on the base frames.** VVC's ALF statistics on base frames alone
  keep the whole -0.9 % for a quarter of the cost (whole encode: Zen 5 +1.47 %
  [+0.84, +2.11], Graviton3 +1.17 %, Graviton4 +1.40 %); non-base density does not move the
  stream at all (non-base frames take ALF filters from the bank). AV2's CDEF on base frames
  reaches most of the uniform gain (period 2 base -0.7 % at +8.2 % against period 2
  everywhere -0.9 % at +19.9 %), but not enough to pay at k 0.1.
- **Keys alone buy little** at these 150-frame windows (one key): -0.2 % for VVC ALF and
  AV2 CDEF, 0.0 for HEVC SAO. Their whole-encode cost is being measured (AV2).
- **Sparser statistics on non-base frames cost quality** everywhere they matter (HEVC
  rows 4 +0.1, AV1 CDEF rows 4 +0.3 %) for 1-1.5 % of P instructions: no saving there.
- **By quantiser:** the lower-rate half moves with the whole curve in every row (no row gains
  clearly more at coarse steps), so no quantiser gate is proposed from this data.
- VVC's SAO settings are not per-frame parameters (they live in the stream-fixed group):
  a role split for VVC SAO needs that hook first.

**Landed:** VVC Default and Slow ALF statistics on every row of key and base frames
(0cb218bd0; nceval.train Default k 0.1 L3 -0.74 % [-1.14, -0.33], final-check -0.55 %
[-0.88, -0.20]; Slow k 0 L3 -0.22 %, final-check -0.63 %).

**Keys plus base, AV2 CDEF** (whole encode from `tools/dev/aws_cpu_ab.py`, Zen 5 /
Graviton3 / Graviton4 cycles):

| AV2 Default: CDEF search | J | whole encode |
|---|---:|---|
| period 1 on keys | -0.2 [-0.3, +0.1] | +0.47 / +0.81 / +0.85 % |
| period 1 keys, 4 base, 8 non-base | -0.5 [-0.6, -0.3] | +1.93 / +2.19 / +2.17 % |

In `nceval.train` (quick k 0.1: L1 -0.62 % [-0.98, -0.27], probe +0.10 %).

**Fast rungs: filters they have off, on key and base frames only** (fixed QP against
each rung; P instr.; whole encode where measured):

| rung: change | J | cost |
|---|---:|---|
| AV1 Faster: deblocking on base frames (off) | -1.9 [-3.0, -1.0] | +2.68 / +2.01 / +2.20 % |
| AV1 Faster: deblocking on keys | -0.2 [-0.3, -0.0] | +0.1 % P |
| AV1 Faster: deblocking everywhere | -2.8 [-4.3, -1.6] | +5.5 % P |
| AV1 Veryfast: deblocking on base frames | -2.5 [-3.4, -1.8] | +2.79 / +2.42 / +2.75 % |
| AV1 Veryfast: deblocking everywhere | -3.1 [-4.3, -2.1] | +5.0 % P |
| AV1 Fast (deblocking on): off on non-base frames | +1.6 [+0.6, +2.7] | -4.34 / -3.98 / -4.78 % |
| HEVC Fast: chroma SAO on base frames (off) | +0.1 [-0.1, +0.3] | +1.5 % P |
| HEVC Faster: chroma SAO on base frames | -0.2 [-0.4, -0.1] | +2.3 % P |
| AV2 Fast: chroma deblocking on base frames (off) | -0.4 [-0.6, -0.2] | +1.3 % P |
| AV2 Faster: chroma deblocking on base frames | -0.4 [-0.6, -0.3] | +1.6 % P |

At k 0.6 the AV1 rows pay (Faster about -0.3 J_k on fixed QP, Veryfast about -0.9; the
Fast row about -1.0 from its CPU saving); the HEVC and AV2 chroma rows do not (+0.4..+0.9).
The AV1 deblocking rows raise VMAF v1 by 2-3 % (it reads the filter's smoothing as a loss)
while PSNR improves 4-5 % and MS-SSIM 3-4 %. Quick screens (k 0.6): Faster L1 -1.83 %
[-3.04, -0.55], probe -1.37 %; Veryfast L1 -1.69 % [-3.56, +0.41], probe -2.32 %; Fast
(non-base off) L1 +0.69 % (videocall +10.7 %), probe -0.92 %. All in `nceval.train`.

**AV1 Veryfast** deblocking on key and base frames landed too (f8346dccb): its L3
periodic-key pulse guard (bodycam_antare_0815, +0.78 dB) came from better keys. On that
clip (150 frames, a key every 50, CBR 105 / 211 / 422) key PSNR-Y rose +1.08 / +0.92 /
+0.58 dB, the P frames before the keys +0.49 / +0.30 / +0.12 dB and after them +0.48 /
+0.27 / +0.21 dB. With keys unfiltered the pulse stayed at the anchor's, but the whole-clip
gain fell from +0.51 to +0.40 dB at 105 kbps.

**VVC SAO by role** (09173bfcf made the `vvc.sao_*` search settings per-frame; fixed QP
against the Default, rows 4 everywhere):

| VVC Default: SAO statistics | J |
|---|---:|
| every row everywhere | -0.4 [-0.7, -0.2] |
| every row on base frames | -0.3 [-0.5, -0.1] |
| every 2nd row on base frames | -0.3 [-0.6, -0.0] |
| every row on keys | +0.0 [-0.1, +0.2] |

## Where the filter searches' cost goes (diagnostic for content-adaptive density)

Inclusive x86 instructions (callgrind, 60 frames, CBR 600, 720p, each codec's Default,
build box), the largest filter functions as a share of the whole encode:

| codec | crowd_mot17_09 | static_cam22 |
|---|---|---|
| HEVC SAO decision (`sao::decide`, of it edge statistics) | 6.4 % (5.8 %) | 7.5 % (6.9 %) |
| VVC ALF decision (`alf::Analysis::decide`; statistics alone) | 8.4 % (3.0 %) | 8.9 % (3.2 %) |
| AV1 CDEF strength choice (`cdef::choose_superblocks`) | 11.4 % | 5.7 % |
| AV2 CDEF search and filter (`cdef::search_and_filter`) | 16.5 % | 10.9 % |
| H.264 / VP9 deblocking (no search) | 5.1 % / 5.7 % | 3.5 % / 5.4 % |

The searches are material: 6-16 % of the encode. The static clip, where most CTUs end up
unchanged, still spends 7.5 % on HEVC SAO and 8.9 % on VVC ALF. The existing per-CTU
reuse switches cover part of this: `hevc.sao_skip_unchanged` (all-skip CTUs keep SAO off
without statistics; on in HEVC Fast/Faster, off at the Default) measured at the HEVC
Default J +0.0 [-0.2, +0.2] at fixed QP. Its whole-encode saving is in the ledger.
VVC has `sao_skip_unchanged` and `alf_skip_unchanged` on at its Default already.

**Landed from this round:** VVC Default SAO statistics every 2nd row on key and base
pictures (nceval.train k 0.1: L3 -0.23 % [-0.52, +0.06], final-check -0.30 %
[-0.53, -0.03]; whole encode +0.1..+0.2 %), and HEVC Default `hevc.sao_skip_unchanged` 1
(bdf0b4906; L3 -0.20 % [-0.39, -0.01], final-check +0.20 % [-0.07, +0.44] AMBIGUOUS, a
CPU saving of 1.6-2.2 %). The closed-loop density controller (previous frame's filter
usage with a floor) is the next step: the shares above say it is worth building, and the
per-CTU outcome share (filter off or unchanged) needs encoder-side counters first, which
the codecs do not record today.


## The ceiling of usage-driven filter-search density (2026-11-21)

Counters (`nanocodec encode --filter-stats FILE`, `nc_encoder::FilterOutcome`): per picture,
the units (CTBs, CDEF filter blocks) whose statistics the search gathered, and how many of
them ended with the filter off, or on with the previous picture's co-located parameters
(SAO types, ALF filter and bank, CDEF strengths). Each codec's Default, 60 frames, CBR 600,
720p, one clip per class; the search share is the search function's inclusive x86
instructions (callgrind, build box). Ceiling = share x (off + unchanged) / searched: what a
perfect predictor of "off" and "unchanged" could save, assuming each searched unit costs the
same.

| codec: filter | class | search share of encode | searched units: off | unchanged | ceiling |
|---|---|---:|---:|---:|---:|
| AV1 CDEF | bodycam | 14.9 % | 51 % | 26 % | 11.5 % |
| AV1 CDEF | crowd | 11.8 % | 52 % | 39 % | 10.8 % |
| AV1 CDEF | lowlight | 5.6 % | 81 % | 10 % | 5.0 % |
| AV1 CDEF | pan | 5.1 % | 74 % | 11 % | 4.3 % |
| AV1 CDEF | static | 5.8 % | 78 % | 15 % | 5.3 % |
| AV2 CDEF | bodycam | 18.0 % | 42 % | 12 % | 9.7 % |
| AV2 CDEF | crowd | 16.4 % | 53 % | 8 % | 10.0 % |
| AV2 CDEF | lowlight | 9.8 % | 77 % | 2 % | 7.8 % |
| AV2 CDEF | pan | 9.6 % | 73 % | 4 % | 7.3 % |
| AV2 CDEF | static | 10.9 % | 74 % | 5 % | 8.6 % |
| HEVC SAO | bodycam | 4.5 % | 78 % | 3 % | 3.6 % |
| HEVC SAO | crowd | 4.5 % | 72 % | 1 % | 3.3 % |
| HEVC SAO | lowlight | 4.0 % | 56 % | 2 % | 2.3 % |
| HEVC SAO | pan | 4.1 % | 58 % | 4 % | 2.5 % |
| HEVC SAO | static | 3.6 % | 52 % | 6 % | 2.1 % |
| VVC ALF | bodycam | 4.6 % | 69 % | 1 % | 3.3 % |
| VVC SAO | bodycam | 2.1 % | 91 % | 1 % | 1.9 % |
| VVC ALF | crowd | 10.5 % | 52 % | 3 % | 5.8 % |
| VVC SAO | crowd | 2.0 % | 90 % | 0 % | 1.8 % |
| VVC ALF | lowlight | 4.0 % | 68 % | 0 % | 2.7 % |
| VVC SAO | lowlight | 1.9 % | 64 % | 1 % | 1.3 % |
| VVC ALF | pan | 4.5 % | 75 % | 0 % | 3.4 % |
| VVC SAO | pan | 2.0 % | 70 % | 2 % | 1.4 % |
| VVC ALF | static | 13.2 % | 66 % | 0 % | 8.7 % |
| VVC SAO | static | 1.8 % | 69 % | 1 % | 1.3 % |

Notes: the AV1 and AV2 shares include the filtering those functions apply in place (the
search and the filter are one pass there), so their ceilings are upper bounds of the
statistics part; AV1 counts the per-superblock CDEF choice only. HEVC's share is with the
Default's new `sao_skip_unchanged` (all-skip CTUs already skipped).

Readings: "off" is the bulk everywhere (51-91 % of searched units end with the filter
off); "unchanged" matters only for AV1 CDEF (10-39 %). The ceiling is 2-4 % of the encode
for HEVC SAO and VVC SAO, 3-9 % for VVC ALF, 4-12 % for AV1 CDEF and 7-10 % for AV2 CDEF.
A controller wins only the share of "off" units it can predict from the previous picture
(co-located off, all-skip rows) at a floor and periodic full refresh; the next step is to
measure that predictability (co-located off -> off) with the same counters before
building the controller.

**Correction (2026-11-21, `docs/handoff/filter-effort.md`):** the AV1 and AV2 rows above
counted filter blocks with no coded 8x8 unit as "searched, off"; those blocks (50-75 % of
them) cost the search nothing. Among blocks with work AV2 CDEF ends off in 0-2 % and AV1 in
7-39 %, so their ceilings are 1.0-3.2 % (AV2) and 3.1-10.0 % (AV1, mostly "unchanged" blocks
that must still be filtered). The predictability measurement and the controller are in the
handoff.
