# BP-3 run lessons (2026-11-23/24)

What the first full BP-3 run cost, where it overran and why, and what the next run should do.

**Sources.**
- The AWS ledger (`~/nanocodec-data/work/aws/ledger.md`).
- Each run's box logs (`session cost` lines), `boxes.jsonl` and encode records, under
  `~/nanocodec-data/bench/bp3-*`.

**Labels.**
- Figures marked *est.* are estimates.
- Everything else is read from those records.
- Times are UTC.

## Numbers first

- **Total.** The benchmark's AWS spend since 2026-11-23 20:00 is **$79.43** over 969 box
  sessions (ledger, purposes `bench` and `bench-method`), plus **$19.75** for 57 orphaned boxes
  booked at recovery.
- **c8i plan (BP-3.10).** Estimated at **$36**, now projected at **~$45** (owner-approved).
  - $40.4 had been spent at 12:00.
  - What remains is AOM-CTC Fixed QP scoring, its tail, and the history.
- **Reports.** "Ours" CBR came out at 10:12 and "ours" Fixed QP at 11:05; both were planned for
  about 09:20. VC-1K and AOM-CTC CBR were planned for 09:30-10:00 and are still pending at 12:30.
- **Where the money went:**

  | Item | Cost | Share of the run |
  |---|---|---|
  | c8a.2xlarge timing | $32.30 | its speeds are now superseded |
  | c8i.2xlarge | $21.43 | |
  | c8a.large | $9.92 | its streams are used; its speeds are not |
  | c7a, c6i and c8g scoring | $11.75 | |
  | c8g history | $2.82 | |

## 1. Planned against actual, per stage (c8i round, from 06:54)

| Stage | Dataset | Plan | Actual | Wall plan → actual | Why it overran |
|---|---|---|---|---|---|
| Re-time | "ours" CBR (34,689 jobs) | ~$1 | **$1.04** (25 + 18 + 12 + 2 boxes) | 1 h → 3.3 h | 5 reclaims and one V7 refusal; three fill-in passes, the last two caused by V6 per box and by V7 on short encodes (both since fixed) |
| Re-time | "ours" Fixed QP (53,082) | ~$1.5 | **$1.63** | 1.5 h → 3.6 h | reclaims; fill-in |
| Re-time | AOM-CTC CBR (12,852) | ~$1 | **$1.26** | 1 h → 3.4 h | reclaims; two fill-ins |
| Encode and time | AOM-CTC Fixed QP (18,888) | ~$5 | **$12.83** so far | 2 h → 5.5 h+ | whole-clip shards left 5 boxes with up to 315 min of work; the first queue driver starved boxes with slow jobs (10-minute stalls, 9 lost boxes) |
| Encode and time | VC-1K CBR (37,611) | ~$3.5 | **$3.58** | 1.5 h → 2.5 h | fill-in after 4 reclaims |
| Score | VC-1K CBR | ~$2 | **$3.35** so far | 0.5 h → 5 h+ | 2 of 10 boxes reclaimed with no replacement; the Arm boxes reclaimed |
| Score | AOM-CTC CBR (run 2) | ~$1.5 | **$2.41** so far | 0.5 h → 4 h+ | the disk stall (all 8 boxes); then all 8 Arm boxes reclaimed |
| History | c8g, 6 arms | ~$3 | **$2.71** (2 arms dropped) | 1 h → 3 h | the starvation bug; the vCPU cap stopped 3 arms |
| History | VC-1K Fixed QP on c8i (5,640) | ~$0.7 | **$0.95** | 1 h → 3 h | the starvation bug |

## 2. Failures and delays

| # | What | Cost | Delay | Fix | Landed |
|---|---|---|---|---|---|
| 1 | The PC's public IP changed at 02:46 and 70 boxes lost SSH | $19.75 for 57 orphans (with #2) | ~1 h | `aws_session` re-adds the PC's address when a poll fails | yes (faf0ac2be) |
| 2 | A PC reboot left orphaned boxes | in #1 | ~1 h | `recover.py` (scratch); every box has a hard lifetime | partly: recovery is not a landed tool |
| 3 | Timing machine changed twice (c8a.large: 12.6 % box spread; then c8a.2xlarge on demand; then c8i spot) | c8a.2xlarge $32.3 and c8a.large $9.9 of superseded speeds | ~10 h | measure stability first (the 35-minute loops), then choose; BP-3.10 | yes |
| 4 | Whole-clip assignment: the slowest box set the end time (AOM-CTC Fixed QP: 315 min against a 106 min median) | ~$5 *est.* | ~3 h | BP-3.11 queue, units of up to 8 jobs | yes |
| 5 | Records came back only at a box's end, so a reclaim lost everything | 12 boxes of work in the first hour *est.* $3 | ~1.5 h | 30 s checkpoints, spot notice, replacement | yes |
| 6 | Feed starvation: coded jobs waiting for references counted as held work | 37 lost boxes, *est.* $1.5 | ~1 h | count only uncoded jobs; an idle box references | yes (cc9edf1b7) |
| 7 | V6 judged per box: one 2.4 % window refused 1,302 jobs | $0.25 re-time | 20 min | per reference window (220 of 29,406 jobs refused instead of ~2,600) | yes |
| 8 | V7 clock check on 0.02-0.04 s encodes: one sample, same jobs refused on every re-run | $0.10 | 40 min | clock check only on encodes ≥ 0.25 s | yes |
| 9 | Scoring disk stall: 90 workers' 1080p decodes filled the 40 GB root disk | $1.7 *est.* | ~2 h | decodes on tmpfs, workers capped by memory | yes |
| 10 | Scoring boxes reclaimed with no replacement (c6i 0.9/box-h; c8g.8xlarge 2.3/box-h, all in 1a) | ~$1 *est.* | ~1.5 h | self-replacing scoring slots; scores checkpointed every 30 s | yes (a2e8c7e89) |
| 11 | Launch lock held until a box ran (~15 s each), so 100 slots queued | — | ~15 min | released once the instance is pending | yes |
| 12 | Disk floor: 76 GB free against a 200 GiB floor blocked launches | — | ~15 min | prune Fixed QP streams after scoring; floor checked in `fleet.py` | yes |
| 13 | The fleets took 1000 of 1200 vCPU and starved the dev screens | dev time | — | `--vcpu-cap` 1000, checked under the launch lock | yes |
| 14 | On-demand by default for timing; spill to dearer regions | c8a.2xlarge on demand at $0.46/h; spill 2.8-3.6× | — | spot by default; spill only with `NC_AWS_SPILL=1` | yes |
| 15 | c8i spot in eu-north-1a: 35 reclaims in 7.4 box-h against 0.12 per box-h in 1b | most of #5 and #10 | — | `fleet.AVOID_ZONES` | yes (this commit) |
| 16 | The c7a scoring type is excluded by the cost guide | $7.62 | — | c7i/c6i, or c8g with the Arm tools | yes |

## 3. Unit costs (measured)

**Encode and time (c8i.2xlarge spot, about $0.075/h, one encode at a time):**
- "ours" CBR, 720p class: **$0.03 per 1,000 encodes**.
- "ours" Fixed QP: $0.031 per 1,000.
- VC-1K CBR: $0.095 per 1,000 (streams kept).
- AOM-CTC CBR: $0.098 per 1,000.
- AOM-CTC Fixed QP (1080p, VVenC timed on the subset): $0.68 per 1,000 as run. *Est.* $0.35 per
  1,000 without the starvation and tail losses.

**Score (decode, cross-decode and libvmaf on every frame):**

| Box | 720p (VC-1K) | 1080p (AOM-CTC) |
|---|---|---|
| c6i.24xlarge | 5,507 streams per box-hour, **$0.09 per 1,000** | 2,215 per box-hour, $0.22 per 1,000 |
| c8g.8xlarge, Arm tools | 2,431 per box-hour, **$0.067 per 1,000** | 1,113 per box-hour, **$0.147 per 1,000** |

The box-hours include setup. Arm gives identical scores: 24 of 24 streams across six codecs,
per frame. Arm is about 1.5× cheaper on 1080p, not the 2.6× seen for 720p encodes, because
memory caps the workers (load 25 on 32 vCPU).

**Spot reclaims, per box-hour** (box logs, today):

| Box | Overall | eu-north-1a | eu-north-1b | eu-north-1c |
|---|---|---|---|---|
| c8i.2xlarge | 0.37 | **4.7** | 0.12 | 0.17 |
| c8g.2xlarge | 0.08 | | | |
| c6i.24xlarge | 0.89 | | | |
| c8g.8xlarge | 2.3 | all of them | | |
| c8a (on demand) | 0 | | | |

**Overheads:**
- **Setup per box:** 2-4 minutes from launch to the first encode (apt, tmpfs, clip upload).
  On "ours" CBR the boxes were busy 8.2 of 10.6 box-hours (77 %).
- **References and canaries:** 6.1 % and 1.7 % of encode time on "ours" CBR, 4.2 % and 1.6 %
  on AOM-CTC Fixed QP. On small fill-in boxes they are 15 %, because canaries run per box.

**Packing test** (3 loaded against 7 unloaded c8i boxes, about 20 min):
- Within-box reference sd: 1.5-1.7 % loaded, 0.5-0.7 % unloaded.
- Range: 6-7 % loaded, 1-2 % unloaded.
- The clock was the same; the variation was in instructions per cycle.
- This does not yet support packing timed boxes.

## 4. The next full run: checklist

Each item is code or a procedure rule; the last column says where it stands.

| # | Item | Where | State |
|---|---|---|---|
| 1 | Pre-flight before any launch: spot quota free, benchmark vCPU under the cap, disk ≥ 200 GiB, the PC's IP in the security group, spot prices against the budget | `fleet.py`; task T18 | quota/cap, disk and IP done; a single `preflight` step still to do |
| 2 | One queue for everything, scoring included (units, checkpoints, replacement) | `fleet.py` | done for encode; scoring is per slot with replacement |
| 3 | Score while encoding: a timed box streams finished jobs to scoring boxes, so no stream round-trips through the PC | task T19 | to do. Saves the PC disk and ~1 h of wall time |
| 4 | Scoring on Graviton from the start (c8g.8xlarge in 1b/1c, Arm tools, per-pass parity check of 24 streams) | `fleet.py --score-bin` | done |
| 5 | 1080p memory-aware scoring workers | `score.worker_cap` | done |
| 6 | Avoid eu-north-1a for timing (c8i: 4.6 reclaims per box-hour there, and dearer); not for scoring, where 1a is cheapest even with its reclaims ($0.164/h, 1.55 reclaims per box-hour, against $0.238-0.249/h in 1b/1c) | `fleet.AVOID_ZONES` | done |
| 7 | No re-timing where V11 already holds: time the run that codes the streams, on the timing machine, and score those streams (the plan's re-time stage cost 3 h of wall time today because the quality came from another series) | procedure | the next run codes on c8i directly |
| 8 | Right-size the fleets: total ≤ 1000 vCPU, sized so each stage ends together (~2 h) | `--vcpu-cap` | the cap is done; sizing is a plan step |
| 9 | Fewer reference points? References are 4-6 % of time. Every 20 jobs gives windows of ~1-4 minutes, and V6 caught real 5-12 % episodes lasting minutes. Keep every 20. | — | keep |
| 10 | Canaries per box cost 9 % on small fill-in boxes: run them only on boxes expected to code ≥ 100 jobs | task T20 | to do |
| 11 | A `recover` step for orphans after a PC crash | task T21 | to do (scratch tool) |

**Estimate for the next full run** (superseded on wall time by "Why it was slow" below) (all three datasets, CBR and Fixed QP, about 157,000 jobs plus
the anchor extras; unit costs above plus 15 % for reclaims and setup):
- Encode and time on c8i: about $15.
- Score on Arm: about $13.
- **Total about $32, wall time about 2.5 h** with score-while-encoding and fleets under 1000 vCPU.
  Without item 3, about 3.5 h.

## 5. An incremental run after a code change

Only our encoders re-run; the competitors' points are reused.

- **Share of the work:**
  - Our families are 37-38 % of the jobs in every dataset.
  - *Est.* about 45 % of the encode time: our slower presets weigh more than their job count.
- **What has to re-run:**
  - Our encodes, timed on c8i with the same references, plus their scoring.
  - x264 medium's timing references run on every box anyway. They tie the new speeds to the
    old ones within the measured 1.4-1.5 % box spread.
- **Estimate:** about **$12-14 and 1.5 h** for all three datasets in both modes. "ours" CBR
  alone is about $1.5 and 40 minutes.
- **Caveats:**
  - A new build changes the PGO training, so it needs a release PGO build first.
  - The reused competitor points must come from the same procedure version and machine series.

## 6. Why it was slow, and the fixes

Dollars came out roughly on plan; wall time was 2.5-10× over plan at every stage. There are
five causes.

| # | Cause | Today's cost in time | Fix | State |
|---|---|---|---|---|
| 1 | Every stage first ran at full scale. The scoring disk stall, V7 on short encodes, V6 per box, the starved queue, the long-encode stall rule and the missing score-box replacement each first showed up on hundreds of boxes. | most of the 2.5-10× | `fleet.py rehearse RUN...`: the whole pipeline (stage, encode and time on 2 boxes with one forced reclaim, validity as records arrive, Arm scoring, the report) on whole clips of each run: the largest clip (1080p, the slowest encodes) and the smallest (the shortest encodes). BP rule: no full run unless the rehearsal passed on the same tools commit. | code landed and unit-tested; the first live rehearsal waits for spend approval (a few cents) |
| 2 | Validity was judged only at report time, so each refusal cost another 30-40 min fill-in pass (2-3 passes per report) | ~2 h per report | the driver settles a job only by the report's full rule (bracketed, outside refused V6 windows, not V7-slow, canaries in line); refused jobs go back to the queue while the fleet is up, and V13 is met when the fleet ends | landed (05557b4ee) |
| 3 | Stages ran one after another: encode, then re-time, then fill-ins, then scoring, then history | ~3 h | the next run codes on c8i directly (no re-time stage). History starts with the main fleets, under the vCPU cap. Score while encoding (T19) | the plan rule is in the procedure; T19 is not implemented yet |
| 4 | No measured progress: every ETA today was a hand estimate, and wrong | decisions taken late | `fleet.py status RUN... --plan-start X`: per run and stage, done/total (valid), the rate over the last 10 min from the records' end times, the ETA at that rate, the boxes up, unbooked cost, spend against the plan and the projected total | landed; rates start with records written from now on (records now carry end times) |
| 5 | Mid-run holds for approval: two spend holds, the disk floor, quota | ~1.5 h | `fleet.py preflight RUN...`: disk, quota and the benchmark cap, the PC's address, spot prices and reclaim rates by zone, the unit-cost estimate and an envelope (+25 %, this run's overrun after fixes) to approve once before launch | landed (T18) |

Also found while finishing: the long-encode stall rule. A box was given up when its records
stopped for 10 minutes, but a 1080p encode can take longer, so boxes were replaced mid-encode
and their jobs redone; the AOM-CTC Fixed QP c8g arm cycled 24 boxes. Now a box is given up only
when its log, which gets a heartbeat every minute, is silent (landed eba7ab871).

**Expected wall time of the next full run, with these in place.** The vCPU cap bounds it, not
the stage order. Measured rates:
- Encode-and-time needs about 63 encode-hours, plus 4-6 % references and 77 % box use: about
  82 c8i box-hours, or 650 vCPU-hours.
- Scoring 157,000 streams on c8g.8xlarge (76 streams per vCPU-hour at 720p, 35 at 1080p): about
  2,550 vCPU-hours.

Under the 1000-vCPU cap that is at least 3.2 h. Adding ramp-up, reclaims (about 6 min per
replaced box) and the reports: **3.5-4.5 h**, and about **$32-40**. Scoring is 80 % of the
vCPU-hours, so the next lever is the scoring work itself, for example cross-decoding our
streams in one decoder rather than all. Raising the cap to the full 1200 vCPU when no dev
screens run cuts the time by about 15 %. The 2.5 h in section 4 assumed no cap and is
superseded.
