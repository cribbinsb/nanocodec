# First key frame size: investigation summary (2026-11-25)

Short version of `first-key-2026-11-25.md` (same directory), which holds the tables and method.
All figures: benchmark CBR conditions (strict 1.0 buffer, first-key lookahead off, peak = target),
pool clips (draws 2026-11-24 and 2026-11-25), `medium` presets unless stated. Class figures rest
on 1–4 clips each.

## Question

Our first key frame is 13–24 mean frame sizes; SVT-AV1, x264 and x265 use 2.5–3.6. Can we make
it clearly smaller at about neutral BD-J, e.g. a moderate I frame followed by refinement P frames
where the scene is static?

## Answer

No design tried is neutral, and the measurements say none is available without lookahead.
Recommendation: keep the first key's sizing.

## What a smaller first key costs

| First key cap (mean frames) | AV1 medium | H.264 medium |
|---|---|---|
| 10 | +3.1 % | +3.6 % |
| 8 | +4.9 % | +4.4 % |
| 6 | +6.9 % | +6.4 % |

- Cap 6 in the other codecs: HEVC +5.0 %, VP9 +7.8 %, VVC +7.3 %, AV2 +6.0 %. AV1 veryfast +11.4 %.
- By class at cap 6 (AV1 / H.264): static +18 / +19 %, texture +13 / +16 %, lowlight +7 / +6 %,
  crowd +4 / +4 %, pan +1 / +3 %, videocall +1 / +1 %, bodycam −1 / −2 %.
- Where the quality goes: the cap takes about 1.0 dB from frames 0–29 and returns about
  0.05–0.07 dB to the rest; the last 30 frames gain about 0.2 dB and 0.9 VMAF. The score counts
  every frame of a 5 s clip equally, so the large key wins.

## The frame-1 refinement idea (moderate I, then a refining frame)

- **A refinement P frame works, but is not free.** On a still clip, a coarse key plus a P frame
  that refines it costs 11–17 % more bits than coding the picture once at the fine quantizer, and
  is cheaper than re-coding frame 1 as intra. A P frame at the key's own quantizer is coarser than
  a key (2.2 dB on H.264, 0.6 dB on AV1 medium), so the refinement must be about 3 QP finer.
- **Coding frame 1 as a second intra frame is the worst option:** coarse key + intra at the fine
  quantizer costs the full fine key again (H.264: 19.4 + 50.0 kB against 49.9 kB for one fine key).
- **Two-step on the whole 24-clip draw, cap 6 plus one refinement frame:** as gated by camera
  motion +5.8 to +7.8 % (AV1) and +3.5 to +5.5 % (H.264); with a perfect per-clip choice still
  +2.6 % and +1.2 %. It recovers 51–64 % of the cap's loss; break-even needs 84–87 %.
- **The clips that gain cannot be picked from frame 0:** they are the ones whose key ends up 7 dB
  or more above what the stream then holds, which is set by motion.
- Not built: a per-cell refinement integrated with rate control. Arithmetic from these
  measurements puts it at about +0.1 to +0.3 % (derived, not measured).

## Two findings that matter more than the key

1. **AV1 veryfast P frames cannot code fine detail.** `av1.partial_forward=2` with
   `av1.inter_tx_split=0` stops large inter blocks coding high frequencies at any quantizer: P
   frames reach 28–34 dB where the key reaches 37–39 dB. Releasing either setting restores
   3–3.5 dB. Quick screen: a quantizer gate on the partial transform is −2.1 % at no CPU cost;
   transform split is −6.5 % for +8 to +13 % CPU. This, not key size, is the prime suspect for
   veryfast's end-of-clip deficit against SVT-AV1 (being screened and landed on branch
   `tx-ceiling`).
2. **The own-rate buffer overshoot is not caused by the first key.** In the benchmark run, 956 of
   13,055 points exceed one buffer at their own rate; only 37 of them peak at frame 0. The cause
   is that streams run 2–9 % under target, unevenly. A 0.95 ceiling with debit margin 0 fixes it
   on 5 s clips (16 → 3 points of 98 on AV1) but fails long windows (+1.4 to +3.1 %). No
   rate-control lever removed it.

## Limits

- The refinement overhead and the veryfast ceiling rest on three probe clips; the screens on one
  draw each.
- No final-check or long single-key run: no first-key candidate survived its quick screen.

## Where

- Full document, code and ledger entries: branch `first-key`, head 49e7b3647.
- Reports and logs: `~/nanocodec-data/work/first-key/`. Probe scripts:
  `~/nanocodec-data/scratch/first-key/`.
- AWS cost: at most $2.49.
