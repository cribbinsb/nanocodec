# Hidden-gaps audit, 2026-11-27

A read-only adversarial pass over checks that look fine but are hollow,
prompted by the AV1 decoder "passing" conformance while rejecting valid Main
tools. Decoder `Unsupported` gates against profiles were covered by another agent
and are out of scope here. Repository at `8cfcb9d6d`. Probes used the cached
landed build `charts/builds/bcd8f4ca` (no compile) on small synthetic clips.
Severity order: wrong output or false claim, then crash, then hollow check, then
hygiene. "Unverified" marks what I could not check.

## 1. Wrong output or false claim

**1.1 The C API says it catches panics, but shipped builds abort.**
`Cargo.toml:71-74` sets `panic = "abort"` for `[profile.release]`, and
`libnanocodec.a/.so` ship from that profile (`tools/release/release.py:283`,
`tools/ci/capi_check.sh:20`). Under abort, `catch_unwind` in `guarded()`
(`crates/nc-capi/src/ffi.rs:261-263`) never returns `NC_ERROR_INTERNAL`: any
panic kills the host process. Three places state the opposite:
`crates/nc-capi/src/lib.rs:18` ("panics are caught at the boundary"),
`docs/ARCHITECTURE.md:97`, and `docs/FAQ.md:303` ("The C ABI catches panics at
the boundary"). The header's `NC_ERROR_INTERNAL` (`include/nanocodec.h:30`)
documents a status that cannot occur.
*Fix:* build the C library with its own profile (`inherits = "release"`,
`panic = "unwind"`) and add a forced-panic case to `capi_check.sh`, or correct
the three docs to say that a panic aborts.

**1.2 The README claims long-term references for VVC, but the encoder refuses them.**
`README.md:35`: "Temporal layers, long-term references, intra refresh and key
frames on request (H.264, HEVC, VVC, AV1)". Running `nanocodec encode --codec vvc
--ltr-interval 4` exits 1 with "long-term reference: not supported by this codec
yet". The same line also implies that intra refresh combines with temporal
layers, but the encoder refuses intra refresh with fixed layers
(`tools/ci/landing_conformance.py:350-356`, "not yet with temporal layers").
*Fix:* name the codecs for each feature, and say that intra refresh does not
combine with fixed temporal layers.

**1.3 `--still-camera` changes nothing for VP9, VVC and AV2 on still, noisy content.**
The test clip was synthetic: 128x96, 30 frames, a static gradient with ±3
luma noise, 150 kbps. With `--still-camera`, H.264 went from 23,760 to 10,234
bytes, HEVC from 23,264 to 11,088 and AV1 from 25,141 to 13,808. The VP9, VVC
and AV2 streams were byte-identical with and without the flag. All six backends
implement `try_static_skip`, and the engine gates the noise map on pre-analysis
(`crates/nc-encoder/src/encoder.rs:2388-2393`). The cause is unverified: I did not
establish it, nor test real camera content. The help text
(`crates/nc-cli/src/help.rs:241-244`) promises a lower bitrate with no codec
limit. *Fix:* find why the flag has no effect for these three codecs. Then make
it work, or refuse it for those codecs.

## 2. Crash

I found no crash.
- **Encoder options:** 1,296 combinations, covering 6 codecs × {none,
  `--temporal-layers 2/3`, `--scalability L1T3`} × {none, `--ltr-interval 3/7`}
  × {CBR 150, CBR 5, `--qp 40`} × {none, intra period, intra refresh,
  `--keyframe`, `--simulate-loss`, `--key-offsets`}. No panic or abort; every
  refusal printed a `nanocodec:` message.
- **Decoders:** 1,800 mutated streams (300 per codec: bit flips, byte
  overwrites, truncation of our own 12-frame streams) through `nanocodec
  decode`. No exit status other than 0 or 1.

Both are shallow smoke tests, not fuzzing (see 3.9).

## 3. Hollow checks

**3.1 The tests half of `land.sh --verify` has not passed for two days.**
Evidence from `~/nanocodec-data/run/`:
- `verified-history`: the last `tests` line is `c15638cab
  2026-11-25T10:32:00Z`, and `verified-head` is the same commit. The newest
  `land-logs/verify-<hash>.log` is from that time.
- Since then 289 commits have landed on main, 123 of them touching codec or
  DSP crates. The streams part kept running (the last success was `bcd8f4caa`
  at 07:31 today).

What has gone unchecked since then (`run_checks` via `check.sh --quick`):
- the workspace integration tests, 36 files under `crates/*/tests/`: the
  libaom golden tests in `nc-checkasm/tests`, the `malformed.rs` decoder tests,
  the `nc-cli` behaviour tests;
- host checkasm (AVX2, AVX-512) and the whole-workspace clippy;
- the browser module tests and the eval suite.

The pre-land checks (`tools/dev/land.sh:291-312`) run only `cargo test --lib
--bins` on the touched crates and build nc-checkasm without running it, so they
cover none of the above. Nothing schedules the verifier: there is no cron or
timer entry for it.
*Fix:* run `tools/dev/land.sh --verify tests` now. Then schedule both parts
(cron beside `nc-remote reap`) and alert when `verified-head` is more than a few
hours old.

**3.2 The latest streams verify failed on infrastructure, so the head is unverified.**
- `run/verify-failed` (07:49) points at `3db6eec8d`. Its log shows the box
  failing with "fatal: not a git repository", exit 128 after 0 s, then fetching
  the previous run's binary, whose `--version` says `bcd8f4caa`.
- The failure was recorded correctly. Only the build-cache version check
  noticed the stale binary, though: a remote failure leaves the last run's
  binary in the fetch path.
- `3db6eec8d` and `02975de9e` are unverified. `8cfcb9d6d` was still running
  when I looked.

*Fix:* clear the remote `target/gate/nanocodec` before each verify, and treat
a remote exit 128 as an infrastructure retry, not a stream failure.

**3.3 No automated check covers the NEON, SVE2, dotprod or I8MM kernels.**
- The verifier runs checkasm only on the x86 build box: `nc-remote` types
  `c6i.24xlarge` and `c7i.24xlarge`, `nc-remote:113`. Its log prints `neon: not
  available on this machine/build, skipped` and passes
  (`crates/nc-checkasm/src/main.rs:158-161`).
- `check.sh` only cross-clippies aarch64 (`tools/ci/check.sh:154-155`), and
  only in full mode, which nothing runs (3.5).
- Arm checkasm runs are per-agent and manual (Jetson entries in
  `docs/handoff/wasm-perf.md`). The Jetson's Cortex-A78AE has no SVE2, so the
  SVE2 tier is tested only when someone happens to use Graviton4.
- Several Arm kernels landed today (the `astra-*-sve2-*` and `*-neon-*`
  worktrees).

*Fix:* add a Graviton4 (c8g, the cheapest per `tools/dev/aws_choose.py`)
checkasm part to verify. Add a `--require` ISA list to nc-checkasm so a
missing tier fails instead of being skipped.

**3.4 Streams are checked on one SIMD tier only.**
- Landing conformance encodes on the build box at its best tier (AVX-512).
  Neither `landing_conformance.py`, `nceval/conformance.py` nor `land.sh` sets
  `NC_DSP_ISA`.
- `tools/ci/isa_identity.sh` (scalar = AVX2 = AVX-512; scalar = NEON = SVE2)
  has no automated caller and no recorded run in `docs/`.
- So the streams users code on AVX2 (the 5950X) and on NEON (Jetson,
  Graviton) are never checked against the standard decoders. Only
  kernel-level checkasm stands between them and a mismatch.

*Fix:* run `isa_identity.sh` in the streams part (x86), and on the Arm box
of 3.3.

**3.5 No automation runs `check.sh` in full mode.**
The verifier always calls `check.sh --quick` (`tools/dev/land.sh:614-615`). So
these never run automatically:
- WASM SIMD128 and relaxed-SIMD checkasm (`check.sh:129-145`), on what is the
  primary browser decode path;
- the aarch64 clippy;
- the `analysis` feature tests (`check.sh:149-151`; pre-land only runs `cargo
  check --all-features`);
- the H.264 decoder conformance step.

*Fix:* have the tests part run full `check.sh`, or move the WASM checkasm
and analysis tests into quick mode.

**3.6 No automated check runs the official decoder conformance suites, and the H.264 suite passes when its data is missing.**
- The six suites (`tools/ci/{h264,hevc,av1,vp9,vvc}_conformance.sh`, AV2
  vectors) run only on demand. The landing check tests only our own
  encoder's streams, which is how the AV1 gaps stayed hidden.
- `tools/ci/h264_conformance.sh:18-20` prints "skipped" and exits 0 when
  `conformance/h264/streams` is absent. Its `check.sh` step title says so.
  `vvc_conformance.sh` correctly fails on missing data.

*Fix:* add a verify part that runs all six suites against checked-in
pass-count manifests (`*_conformance*.json`), with missing data treated as a
failure.

**3.7 No automated conformance check covers long-term references, loss recovery, timing or colour signalling.**
- `landing_conformance.py` `cases()` (`:361-395`) has no LTR case, no
  single-layer `--simulate-loss`, no `--keyframe` request and no colour or VUI
  case. Loss appears only in SVC `s1loss`/`s2loss`.
- `tools/ci/realuse_conformance.sh` covers these, but nothing runs it
  automatically, and I found no dated pass record (unverified when it last
  passed). Its LTR and loss rows (`:107-110`) list `h264 hevc av1 av2`
  without VP9, although VP9 accepts `--ltr-interval` and its stream changes
  with it.

*Fix:* add `ltr` and `loss` kinds (all codecs that accept them) to
`landing_conformance.py`.

**3.8 No tool runs the 39 `#[ignore]` tests.**
`grep -- --ignored` over `tools/` finds nothing. The ignored tests include:
- 14 VVC encoder-vs-standard-decoder tests (`crates/nc-vvc/tests/conformance.rs:39-747`);
- 12 VVC official-corpus decoder tests (`decoder_blocks/*_tests.rs`,
  `decoder_headers/sps.rs:385`, `nc-web/src/vvc_corpus_tests.rs:116`);
- 5 VVdeC-trace kernel tests (DMVR, BDOF, PROF, intra; `nc-dsp/src/vvc_*`);
- H.264 and HEVC external-fixture tests.

These are the independent-reference tests, and they only run when someone
remembers to.
*Fix:* add a build-box step that runs `cargo test -- --ignored` with the corpus
and decoder paths set, or delete the tests nobody will run.

**3.9 Fuzzing has gone stale, and nothing replays the corpus.**
- The last session was round 3, 2026-11-23 at `98069a8b3`
  (`~/nanocodec-data/fuzz/round3-2026-11-23`). The fuzz targets have not
  changed since 2026-11-13.
- Since 2026-11-23 the decoder crates have taken 7 to 29 `src` commits each.
- `fuzz/replay` is not in `check.sh` or the verifier, so a regression on a
  known crash input is caught only if that input was also made a unit test.

*Fix:* replay the kept corpora in the tests part, and add a short fuzz
session per release.

**3.10 The libaom golden hashes cannot be reproduced.**
`crates/nc-checkasm/tests/av1_inverse_golden.rs:7-11` says its `GOLDEN` hashes
come from "a throwaway C harness linked against libaom ... (not kept in the
repository)", and `av1_block_intra_golden.rs` and the other `*_golden.rs`
files follow the same pattern. The reference is independent but cannot be
regenerated, so nobody can show that a later edit to `GOLDEN` still came from
libaom. The AV2 tests cite `tools/codegen/check_av2_mv_refs.py`, which is
better. *Fix:* commit the C harnesses under `tools/codegen/`.

## 4. Hygiene

**4.1 `--key-qp-offset` is silently ignored under `--bitrate`.**
With `--bitrate 200`, the stream is byte-identical with and without it, on all
six codecs. Under `--qp` it changes every codec's stream. The help says
"fixed-QP key frames", but there is no error, unlike `--min-qp`, which says it
"requires --bitrate". *Fix:* refuse `--key-qp-offset` without `--qp`.

**4.2 `--max-qp` accepts values that `--qp` rejects.**
| Codec | `--max-qp` accepted up to | `--qp` accepts |
| --- | --- | --- |
| H.264, HEVC | 64 | at most 51 |
| AV1, VP9 | 256 | 1..=255 |

The rate model's maximum is widened by the extra rate steps
(`crates/nc-encoder/src/encoder.rs:816-819`), so the user-facing scale of
`--max-qp` is not the codec's quantiser scale that the help names.
*Fix:* validate against the codec range, or document the extended steps.

**4.3 Safety and licensing rules hold.** This is a clean result.
- `unsafe` appears only in nc-dsp and nc-capi.
- Every other `lib.rs`/`main.rs` has `#![forbid(unsafe_code)]`.
- All 4,052 `unsafe {` blocks in nc-dsp and nc-capi have a `// SAFETY:`
  comment. A scan found two without one in the 8 lines above, and both have
  long SAFETY comments just above that window (`dispatch.rs:2420`,
  `warp_avx2/fused.rs:182`).
- No GPL-derived code found. The x264 references are cited ideas
  (`intra_estimate.rs:11`: "an idea, no code").
- All 830 registered `param_group!` fields are read somewhere (a field-name
  scan, so a crude check).

## Public docs against internal gaps

Gaps recorded internally but stated wrongly or not at all in public docs:
- C API panic handling (1.1).
- VVC long-term references, and intra refresh with temporal layers (1.2).
- `--still-camera`: the per-codec effect (1.3), and AV1 running over target on
  long still streams (1.048 against 1.032, recorded only in
  `crates/nc-encoder/src/still_camera.rs` module doc).

The README's decoder gaps (AV1 tools, interlaced H.264) are stated as of
`02975de9e`.
