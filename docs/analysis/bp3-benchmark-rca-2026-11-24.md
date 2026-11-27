# Root-cause analysis: the BP-3 benchmark run (2026-11-23/24)

The coordinator wrote this for the owner. The run's own numbers (unit costs, the plan-against-actual table, the
16-item failure list) are in `docs/analysis/bp3-run-lessons-2026-11-24.md`; this document groups the mistakes by
class and records what now stops each class.

## Summary

- **What was asked:** re-measure every result under one reviewed procedure, with speed and quality from the same
  run of each encode.
- **What happened:**
  - The run took about 19 hours of wall time, against a first estimate of about 3.
  - Benchmark spend was $79.43 from 2026-11-23 20:00 to midday on 2026-11-24, and about $82 by the end. $42 of
    that was a superseded timing series, and $19.75 was orphaned boxes.
  - On the day, the reports arrived 2 to 5 hours after the times given to the owner.
  - Every time estimate given to the owner was wrong, and seven estimates for the last report failed in a row.
- **The c8i round itself:**
  - Its money stayed on plan: $35.21 against $36.
  - Its wall time was 2.5 to 10 times the plan at every stage.
- **The common root:** almost every failure was found at full scale, one at a time, after it had cost time. The
  coordinator then fixed the instance, not the class.

## The classes

Each class gives what happened, why, what now prevents the class, and how that is enforced. "Enforced by" names a
check in code that refuses or fails; a rule in prose alone is marked as such.

### A. A stage was first exercised at full scale

- **Instances:**
  - the scoring disk stall (90 workers' 1080p decodes filled a 40 GB disk);
  - the clock check refusing 0.02 s encodes on every retry;
  - the reference check refusing whole boxes;
  - the job queue starving boxes;
  - the stall detector killing boxes during long 1080p encodes;
  - scoring boxes with no replacement after a reclaim;
  - two workers scoring one stream after a re-code;
  - the launch lock held for 15 s per box.
- **Why:** there was no end-to-end trial. Each stage was pipe-cleaned alone, if at all, and never with 1080p
  clips, very short encodes, a reclaim and the report together.
- **Prevention:** `fleet.py rehearse` (9f5823a55) runs the whole pipeline on whole sample clips of every dataset,
  through to the report, with one forced reclaim. BP-3.11 requires it before any full run.
- **Enforced by:** the procedure rule today. To do (task R1): `fleet.py run` refuses a full launch unless a
  rehearsal pass is recorded for the same tools commit.

### B. Finished work could be lost

- **Instances:**
  - spot reclaims lost a whole box's records, because they came back only at the end;
  - a reclaim lost a whole dev batch;
  - CPU comparisons lost whole shards;
  - a local driver killed by its memory limit lost a held-out round;
  - a driver's own timeout killed a run;
  - the coordinator's spend stops killed boxes without their closing timing references, which invalidated 214
    coded jobs;
  - the PC's address change and reboot orphaned 57 boxes ($19.75).
- **Why:** tools assumed that a box, a driver and the PC live until the end of the run.
- **Prevention, landed:**
  - four rules for every spot tool, written into `docs/aws-cost-guide.md` (924628914): copy work off as it
    finishes; act on the interruption notice; retry only what is missing; have a reclaim test;
  - applied to every launch path: `aws_measure` (493dd7eff, a4a832359), `aws_cpu_ab` (ef812fa30), the build box
    (dc404a7ef), scoring slots (a2e8c7e89), chart fleets;
  - any stop of a box, from any cause, first runs its closing references and returns its records (84cbe1ba3);
  - a rerun of a held-out round reuses its finished parts (8d291a746); a run pins its code at launch
    (85b788218);
  - the PC's address is re-added automatically (faf0ac2be); every box has a hard lifetime.
- **Enforced by:** a simulated-reclaim test per tool. To do:
  - R2: a CI check that every tool which launches instances goes through `aws_session` and has a reclaim test;
  - R3: reference pairs spaced by encode time, not job count, so the exposure is minutes;
  - R4: the orphan-recovery tool landed (it is a scratch script today), and fleet drivers that re-attach after
    a PC restart.

### C. Validity was judged at the end

- **Instances:** each report needed two or three extra "fill-in" passes of 30 to 40 minutes, because refused
  timings were found only when the report was written.
- **Why:** the checks lived in the report, not in the driver.
- **Prevention:**
  - the driver settles a job only by the report's full validity rule, and re-queues refused jobs while the
    fleet is up (05557b4ee);
  - the reference check judges windows, not boxes (18a1f1f0f);
  - the clock check applies only to encodes of 0.25 s or more (1275c61e2);
  - `fleet.py missing` uses the report's rule.
- **Enforced by:** unit tests on the shared rule; the rehearsal exercises it.

### D. Estimates were not measurements

- **Instances:** every completion time given to the owner. The last report was promised for 11:45, 11:15, 12:15,
  13:20, 14:30, 14:45 and 15:10.
- **Why:**
  - each estimate was the known remaining work divided by the normal rate;
  - the delays were never in the known work: the report needs every point, so the slowest stragglers set the
    end;
  - the coordinator relayed each new figure without looking at the record of earlier ones, and reported slips
    at the deadline instead of when they began.
- **Prevention:** `fleet.py status` (9f5823a55) reports done/total, the measured rate, the ETA from that rate and
  the spend.
- **Enforced by:** the coordinator's rule (memory): quote only measured ETAs, as a range with the run's
  actual-to-estimate ratio beside it, and report a slip at the first check that shows it. To do (R5): `status`
  ignores box setup time in its rate, and counts the same jobs as `missing`.

### E. A number was acted on without checking what it measured

- **Instances:**
  - "plan spend" was the whole account's ledger since 06:54, not the benchmark's. The coordinator enforced
    ceilings at $42, $46 and $48 against it while the benchmark's own spend was $35. Those stops cost about
    1.5 hours and caused the invalid timings in class B.
  - Before this run: charts that timed fixed-QP encodes against CBR quality (the reason for the publish freeze).
- **Why:** the coordinator did not read how the figure was computed. CLAUDE.md already requires that.
- **Prevention:** the single-run rule and the stream-identity check (BP-3) for the timing case.
- **Enforced by:** to do (R6): the ledger records a purpose per box, and `fleet.py status` and the preflight
  compute plan spend from the benchmark's own rows only.

### F. Machines and methods were chosen without measuring or pricing them

- **Instances:**
  - the timing machine changed twice (c8a.large had a 12.6 % box-to-box spread; then c8a.2xlarge on demand at
    $0.46/h, not offered in the region with quota; then c8i spot): $42 of superseded timing;
  - x86 where Graviton does the same work for less;
  - on-demand fallback and spill to regions costing 2.8 to 3.6 times more;
  - a c7a build box;
  - scoring on paid cloud CPU after the local GPU was switched off following the 02:19 crash.
- **Why:** choices were made by reasoning and habit, and a failure was answered by abandoning the resource
  rather than bounding it.
- **Prevention:**
  - `docs/aws-cost-guide.md`, measured per workload, with Arm first; `tools/dev/aws_choose.py` before a launch;
  - spot, eu-north-1 and Graviton as tool defaults; on-demand and spill opt-in (e54d4190b, b4031c3f7);
  - build-box types from the guide (72a811838);
  - dev scoring on the local GPU with one bounded server (d581b0311, 7ca32049a), and automatic fallback;
  - every brief carries a "Machines and cost" section.
- **Enforced by:** the tool defaults and their tests. To do (R2, the same CI check): no launcher sets an
  instance type, region or lifecycle outside `aws_session` and the guide's table.

### G. Shared limits were hit mid-run

- **Instances:**
  - the PC's disk floor blocked launches; the work directory was over its budget and blocked local scoring;
  - the fleets took the whole spot quota and starved the dev screens;
  - no spot capacity in the zones left after excluding the reclaim-prone one;
  - the PC locked up from too many local driver processes; the GPU crash.
- **Why:** there was no check of shared resources before launch, and no caps.
- **Prevention:**
  - `fleet.py preflight` (9f5823a55): disk, quota, the vCPU cap, the PC's address, prices and reclaim rates by
    zone, and one cost envelope to approve before launch;
  - a 1000-vCPU cap on benchmark fleets (d4dc4bedb);
  - caches with size caps, pruned by their own tools (426950d68);
  - one driver per 20 boxes (952cdce6b); the memory guard; GPU scoring behind a lock and a free-memory gate.
- **Enforced by:** the preflight (procedure rule today; R1 makes `run` require it) and the tools' own guards.
  To do (R7): the preflight checks spot capacity in the allowed zones, and a fleet short of capacity falls
  back to the cheaper zone.

### H. Stages ran in series, and mid-run approvals stopped the run

- **Instances:** encode, then re-time, then fill-ins, then scoring, then history; four spend holds on one report.
- **Prevention:**
  - the next run encodes on the timing machine directly (no re-time stage); history starts with the main
    fleets;
  - one approved envelope from the preflight, with a stated margin, and no mid-run stops inside it.
- **Enforced by:** the procedure. To do (R8): score while encoding (task T19).

### I. The coordinator fixed instances, not classes

- **Instances:**
  - "design for failure" was applied to the benchmark fleet only, and a dev batch was lost to a reclaim two
    hours later;
  - the cost rule was applied to one agent's machines, and others kept using dear ones;
  - the local GPU was switched off instead of bounded.
- **Why:** an owner instruction was read as a note about the tool under discussion.
- **Prevention (coordinator rules, in memory and in every brief):**
  - when the owner states a principle, list every place it applies in the first reply and fix them all the
    same day;
  - answer from existing data first; design the smallest test that decides;
  - check where a job's compute goes before starting it.
- **Enforced by:** not enforceable in code. The CI check (R2) and the rehearsal gate (R1) exist so that the
  classes above do not depend on it.

## Open actions

| # | Action | Owner | State |
|---|---|---|---|
| R1 | `fleet.py run` refuses a full launch without a rehearsal pass and an approved preflight for the same tools commit | benchmark agent | **landed f24c559a7**: records in `bench/gate.jsonl` (`rehearse` writes the pass, `preflight --approve USD` the approval), committed tools required, `--ungated REASON` override printed and recorded; tested |
| R2 | CI check: every instance launcher uses `aws_session`, has a reclaim test, and sets no type/region/lifecycle outside the guide | cost agent | ordered 2026-11-24 |
| R3 | Timing references spaced by encode time | benchmark agent | **landed 84cbe1ba3**: a pair every 20 jobs or 5 min of encode time, and right after a long encode; any stop closes a box with references (driver STOP); tested |
| R4 | Orphan recovery as a landed tool; fleet drivers re-attach after a PC restart | benchmark agent | ordered |
| R10 | Longest first: the queue hands out jobs in descending expected duration; units sized in expected seconds (a long job alone); fleet size from expected seconds ÷ target wall time; `status` ETA from remaining expected seconds; the rehearsal includes the longest jobs | benchmark agent | **landed**: `fleet.expected_seconds` (each job's measured time across all runs, else its family's per-pixel rate), `WorkQueue` longest first with `UNIT_SECONDS`, `fleet_size` (`run --boxes 0 --target-minutes`), status ETA from expected seconds; tested |
| R5 | `status`: rate ignores setup; counts agree with `missing` | benchmark agent | **landed**: each box's rate counts from its first record; `missing` and `status` share one validity count (`fleet.valid_ids`, merged and box records); `status --group` prints the dataset's combined count; tested |
| R6 | Ledger purpose per box; plan spend from the plan's own rows | benchmark and cost agents | **landed**: `run --plan NAME` tags boxes Purpose `bench-<plan>` (the cap and spend count every `bench*`); `status --since ISO` and `preflight` report the benchmark's own ledger rows (`fleet.purpose_spend`, purpose `bench`); ledger-minus-start is gone; tested |
| R7 | Preflight checks zone capacity; fallback to the cheaper zone | benchmark agent | **landed**: preflight prints the AWS spot placement score per zone and the largest fleet that scores 5/10 in an allowed zone (a warning: the score read 1/10 for 50 c8i while 100+ ran); once a timing fleet finds no capacity in the allowed zones, it allows the avoided zone; tested |
| R8 | Score while encoding (T19) | benchmark agent | queued |
| R9 | First live rehearsal, then an incremental "ours" run as the proof | benchmark agent | after this run closes |
| Rev-1 | Canaries short, never the queue head | benchmark agent | **landed 63e11e067**; tested |
| Rev-9 | Default fleet budget after sizing (`--boxes 0` gave $0) | benchmark agent | **landed 63e11e067**; tested |
| Rev-2 | Any SSH or API error is a lost box, replaced | benchmark agent | **landed**: `fleet.LOST_BOX_ERRORS` (+ botocore errors) in `slot`; failed slots refuse the run; fault-injection test |
| Rev-8 | Unique box dirs per driver start (a restart never overwrites records) | benchmark agent | **landed** (box names carry the driver's start); the journal and re-attach are R4 |
| Rev-12 | Retry limit counts only boxes in a row that settled nothing | benchmark agent | **landed**; tested |
| Rev-4 | Final count from the report's rule (retroactive refusals) | benchmark agent | **landed** (run's final count is `missing`); per-poll unsettling is with Rev-6 |
| Rev-14 | Fake-session fault-injection tests of the slot loop | benchmark agent | **started**: transient errors, barren attempts, refusal |
| Rev-10 | Streams verified (size and SHA-256) on arrival before a job settles | benchmark agent | **landed**: `fleet.stream_arrived`, hashed once (a mark beside the streams); tested |
| Rev-5 | Poison jobs and hangs: job-level failures, attempt counts, quarantine after 2 boxes, a kill at 5 x expected seconds (at least 10 min), a heartbeat naming the current job | benchmark agent | **landed**: `box.JobFailure`, `WorkQueue.fail`/`quarantined`, `JOB_TIMEOUT_FACTOR`, heartbeat "alive: <job> for N s"; tested |
| Rev-18 | Procedure drift (score type, status flags, units) | benchmark agent | **landed 63e11e067** |

This document is updated as each action lands.
