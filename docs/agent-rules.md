# nanocodec — working rules

> This is the rulebook the AI coding agents that build nanocodec work under (the repository's agent instructions, read first by every agent).
> Published copy: machine-specific and internal details (local paths, hosts, accounts, internal-only notes) were removed or generalised; the rules are unchanged. Snapshot at release 0.5.0 (2026-11-20).

Start here: the internal status, procedures and per-area handoff notes (where
things stand; how to build, measure, profile, and the multi-agent rules). These
are not in the public repository. Keep them current: they are how work is handed over.

Extreme-speed, low-latency video encoder for a constrained subset of
H.264 / HEVC / AV1, with initial VP9 / AV2 support, plus matching subset decoders. Live camera content,
low bitrate, forward prediction only. Full design: `docs/ARCHITECTURE.md`.

## Code quality is a hard requirement

The project MUST be cleanly structured, with well-written, understandable code.
Speed is not an excuse for unreadable code. Speed lives in small, isolated
kernels; everything else must read clearly. When a fast hack conflicts with
clean structure, choose clean structure.

### Language and layout

Rust (stable, edition 2021+), as a Cargo workspace. All maintenance is done by
AI agents. Rust was chosen (2026-11-06) so that the compiler enforces safety
and module boundaries that conventions alone would not keep.

```
crates/
  nc-common/      buffers/planes, bit writer/reader, YUV I/O, errors, logging
  nc-dsp/         pixel kernels: scalar reference + x86 (AVX2) / aarch64 (NEON)
                  / wasm32 (SIMD128) versions, runtime dispatch. The ONLY crate
                  that may contain `unsafe` (besides nc-capi).
  nc-analysis/    pyramid, motion estimation, pre-analysis, AQ
  nc-encoder/     shared engine: mode decision, rate control, ref/GOP manager,
                  the `CodecBackend` trait
  nc-h264/ nc-hevc/ nc-av1/ nc-vp9/ nc-av2/   one backend each: syntax, entropy, transforms,
                  loop filters, and that codec's subset decoder
  nc-capi/        C ABI (staticlib/cdylib + generated header) for embedding
  nc-cli/         command-line encoder/decoder
  nc-checkasm/    SIMD-vs-reference tests and kernel benchmarks
tools/  eval/  docs/
```

### Rules

- **One-way dependencies, enforced by Cargo:** codec crates → nc-encoder /
  nc-analysis → nc-dsp → nc-common. nc-encoder never depends on a codec
  crate; it sees codecs only through the `CodecBackend` trait.
- **One encode path per codec (owner rule):** a codec encodes only
  through its `CodecBackend`, driven by `nc_encoder::Encoder`: no second,
  standalone, legacy or test-only encoder, in `src/`, `examples/` or tests
  (tests code their streams through the engine). `tools/ci/check_encode_paths.py`
  enforces it.
- **`unsafe` is confined to nc-dsp kernels** (and the C ABI in nc-capi). Every
  other crate has `#![forbid(unsafe_code)]`. Each `unsafe` block carries a
  `// SAFETY:` comment.
- **No global mutable state.** All state lives in explicit context structs; the
  SIMD dispatch table belongs to each encoder/decoder instance.
- **Small modules and functions** with one job each. Every file starts with a
  `//!` doc comment saying what it does and where it fits.
- **Comments explain why, not what.** Bitstream code cites the spec section
  (e.g. `H.264 §7.3.4`, `HEVC §7.3.8.5`, `AV1 §5.11.5`).
- **Every SIMD kernel has a scalar reference** and a bit-exactness
  (checkasm-style) test in nc-checkasm. SIMD modules contain kernels only,
  never control logic.
- **One kernel per pattern across codecs (owner rule):**
  - **Sharing is the default.** Similar kernels in different codecs (transforms,
    quant, intra predictors, MC filters, loop filters, metrics) share one
    implementation, parameterised by codec tables, rounding or shifts. Accept a
    small measured cost for this.
  - **A codec-specific copy states why in its module doc:** the arithmetic
    difference that stops sharing, and the shared family it belongs to.
  - **Family kernels follow one pattern:** the same data layout, SIMD technique
    and ISA tiers (AVX2, AVX-512, NEON, simd128). A better technique found for
    one codec is carried to the rest of its family.
  - **Record:** `docs/analysis/encode-kernel-sharing.md`.
- **Every speed heuristic / early termination** has a comment: what it skips,
  why that is safe, and its measured speed and BD-rate effect.
- **Shared first (owner rule):** a heuristic, cost model, search or
  early exit that is not bitstream syntax has one implementation, in the
  engine (nc-encoder / nc-analysis) or nc-dsp; the codec supplies only its
  syntax-dependent inputs through `CodecBackend`.
  - Prefer the shared mechanism even at equal or slightly worse measured
    results (about +0.3 % J, no guard tripped): it is more robust and harder
    to overfit. Count non-shared code as a cost.
  - A codec-local heuristic states its codec reason (the syntax or tool
    difference) in its doc comment.
  - A codec's value for a shared parameter is justified by a codec reason in
    its `DEFAULT_PARAMS` comment, or it is the shared default; a value every
    codec sets becomes the engine default.
  - See `docs/analysis/shared-vs-specific-audit-2026-11-15.md` for the
    candidates still to share.
- **Every change is considered for every codec and preset (owner rule):**
  when any change lands for one codec or preset (speed-up, quality lever,
  heuristic, preset value, bug fix), check whether the same pattern exists in
  the other codecs (H.264, HEVC, AV1, VP9, AV2, VVC) and in the other presets of
  each. "Considered" means a common-sense judgement, not a forced evaluation:
  where it plausibly helps, carry it over (shared code they all call, a copy
  only where the codecs' arithmetic differs; screened per preset) or queue it in
  the ideas ledger; where the answer is an obvious no, a one-line reason in the
  commit is enough.
- **No magic numbers:** named constants and enums.
- **Narrowest data type (owner rule; amended for decoders):** encoders code
  8-bit only; decoders also take high bit depth for conformance, with u16 samples only for
  streams deeper than 8 bits (the code is generic over `nc_common::Sample`, compiled once
  for u8 and once for u16; described in an internal handoff note). Store and compute in the
  narrowest type the value range allows (u8 samples, i16 coefficients, residuals and
  transform intermediates, u16 small-block SADs), citing the spec clamp or the arithmetic
  bound. Widen only where a value can exceed it (products before their shift, large
  accumulators, a 10-bit bound that 16 bits no longer holds); see
  `docs/analysis/integer-width-audit-2026-11-14.md` for the places where 16 bits is not
  safe.
- **Hot loops:** avoid per-element bounds checks with slices/iterators or one
  up-front length check; verify with benchmarks. No `unsafe` outside nc-dsp to
  "go faster".
- **Borrowed code** (openh264, kvazaar, SVT-AV1, libaom, dav1d — BSD-style
  only) is reimplemented in idiomatic Rust, not transliterated wholesale. Keep
  the licence attribution in the module doc comment and in `THIRD_PARTY.md`.
  **Never copy GPL/LGPL code** (x264, x265, ffmpeg, libde265); ideas from
  papers are fine.
- Tooling: `cargo fmt` (checked-in rustfmt.toml), `cargo clippy --all-targets
  -- -D warnings`, `cargo test`, Miri where feasible for nc-common, and fuzz
  targets for the decoders. No dead code, no commented-out code.
- Toolchain: rustup (development PC and Jetson). Targets: x86_64,
  aarch64 (Jetson), wasm32-unknown-unknown (+simd128).

## Key design decisions (see docs/ARCHITECTURE.md for detail)

- **SVT-AV1 real-time (M12 `--rtc`) is the primary guide and the bar.** Our AV1
  must match or beat it on BD-rate and CPU; its real-time pipeline is the
  reference architecture for the shared engine (docs/ARCHITECTURE.md §2).

- All three codecs are built together on one shared analysis/mode-decision
  engine; each codec plugs in through the `CodecBackend` trait.
- Single-threaded for now (one core should encode 1080p), but keep code
  thread-ready (no globals, per-row state).
- No slices: one slice/tile per frame. The loss model is "frame arrives whole
  or not at all". Frame-level resilience (temporal layers, intra refresh,
  LTR + feedback, keyframe on request) is required.
- Bitstreams are strictly standard-conformant: no custom syntax, ever (scope
  and testability). Our WASM decoder is the primary browser decode path and
  only needs to handle our own subset; WebCodecs is an optional extra.
- Codecs: H.264, HEVC and AV1, plus initial VP9 and AV2 on `feature/vp9-av2`.
  An internal handoff note lists their narrower supported subset. EVC was
  considered and rejected.

## Working in parallel (multiple agents)

- **Each agent works in its own git worktree** (one directory per agent,
  outside the shared checkout), rebased on the branch. Never run
  `git checkout`, `restore`, `reset` or `stash` on another tree's files, and
  never edit or `git add` in the shared checkout's index.
- **Each agent builds into its own target directory** and points the
  harness at it (`NC_NANOCODEC_BIN`), so it never measures another agent's
  build.
- **To measure an existing commit, never compile:**
  `NC_NANOCODEC_BIN=$(tools/dev/binary_for.sh <commit>)` (the landed-commit
  cache, else built on the build box). Build your own target dir only for code
  you have changed, and then on the build box.
- **Load:** measure only through the shared streamed point engine
  (`nceval` tune/sweep/sanity, `tools/dev/fastbd.py`, `nceval.point`): it
  writes no video files and takes machine-wide CPU slots, so never sleep on
  `uptime`. One heavy job per agent; ≤ 8 workers and `cargo -j8` while other
  agents are busy, but on an idle machine (load under about 8) use the cores
  (up to nearly all of them): waiting on a slow build beside idle cores is the wrong
  trade (owner rule). No private point loops that write YUVs.
- **Timing runs only on a quiet machine;** otherwise report callgrind
  instruction counts. Code layout alone moves whole-encode time by up to
  ~2 % on Zen 5 (sd 1.5 %, measured in an internal handoff note): a smaller gap between
  builds whose executed code is unchanged is noise unless it holds over
  at least four shuffled links per side.
- **Heavy work runs on the remote build box (owner rule):** builds,
  tests, clippy, callgrind, conformance suites and fuzzing go through
  `nc-remote run -- <command>` (`tools/dev/nc_remote.py`: one shared cloud
  build machine, your worktree synced, your own target dir kept there; usage in
  the internal procedures). Local work is landing, verify, wall-clock timing and
  the GPU (screens, VMAF). Clips may be copied to the project's own cloud machines whenever
  that is faster (owner rule); delete them there afterwards, and never
  publish clips or frames (datasets licensed for non-commercial use stay
  numbers-only).

## Testing

- **Have we tried it? (owner rule).** Before starting an idea, run
  `python3 -m nceval.levers ideas search WORDS`; every brief's final step records
  each idea it tried, landed or not, with `python3 -m nceval.levers ideas add`
  (outcome, one-line reason with the key numbers, branch, doc section, ledger
  rows). The ledger is the one record: never a parallel markdown table
  (usage in the internal procedures).

- **Speed and common sense (owner rule).** Check the one thing that
  covers this change's risk, first, then stop:
  - **Exact speed-ups need no speed proof (owner rule).** Proof is not needed
    for a small, clean change that leaves streams identical and is plainly faster:
    less work, fewer copies or checks, a better kernel. Its only check is identity
    (`preset_identity` or the unit tests). Land it, and do not reject it on noisy
    timings. Many small gains add up. Measure only changes that alter streams, or
    whose speed is genuinely in doubt. Speed is in doubt (so take one release
    callgrind count) when a similar attempt measured slower before, or a kernel
    is restructured (new data layout, packing or lane width), not just trimmed.
  - **Never overfit (owner rule, top priority).** Adoption evidence
    comes only from the pool levels on the day's rotating draw (the default
    `nceval.screen --clips adopt`: L1 quick, L2 adopt, L3 final over the
    whole ~190-clip graded pool, rotated clips and start frames, decided on a
    bootstrap interval of per-clip J). No agent picks its clips; fixed sets
    (`--clips train|val|names`) only rank ideas. L3 is required before any
    preset/Default change lands. Run `python3 -m nceval.levels final-check`
    for every preset bundle, starting it alongside L1 rather than after L3:
    its long windows and key pulse decided both of the 2026-11-16 Fast
    candidates after hours of passing screens. The sealed held-out test sets stay
    sealed for milestones.
  - **A codec `DEFAULT_PARAMS` or engine-default change reaches every preset
    that does not pin the value.** Run `tools/dev/preset_identity.py` over all
    presets and list each preset whose streams change in the commit message.
    Pin the old value in any preset that was not screened (2026-11-16: AV1 Fast
    and Faster silently inherited Default-only settings, +45–52 % key-frame
    work).
  - Anything that changes frame structure, references or rate control: run
    `crates/nc-cli/tests/static_recovery.rs` and a long single-key run
    *before* screening.
  - Guards are judgement calls, not tripwires: a guard missed by a few
    tenths while the score and most metrics improve gets a quick look at
    what it flags (e.g. the worst frames) and is adopted if nothing is
    visibly wrong; say why in the commit. The periodic-key pulse guard is
    relative (key against the P frames before it): when it trips, compare
    absolute key and after-key quality with the anchor; a pulse that comes
    from better P frames, with the key at most a few tenths worse at a few
    rungs, is not a block (owner direction: frontier, not a no-loss bar).
  - Guards apply to the final bundle, not each step: a change that gains
    overall but misses a guard can be paired with another that wins the
    guard back (single-change screening otherwise stalls at local maxima).
  - The score is PSNR-YUV, VMAF v1 and MS-SSIM in equal thirds
    (`eval/nceval/jreal.py`). Before quoting what any tool measures, read
    the code that computes it.

- **Bounded disk usage (hard rule).** Never fill the disk.
  - The eval harness enforces a 100 GiB budget on its work directory and
    stops a sweep if fewer than 200 GiB are free (`eval/nceval/diskguard.py`).
  - Keep only bitstreams; delete decoded and reconstructed YUVs as soon as they
    have been checked.
  - Keep one result version per curve.
  - Delete scratch files when done, with explicit literal paths only:
    `rm -- /abs/path/a /abs/path/b` or `rm -r -- /abs/literal/dir`. Never
    rm with globs, variables, command substitution, loops or
    `find -delete` (each needs owner approval); for long lists, have a
    script remove its computed list path by path.
  - Never write to the source clip store (read-only).

- **Correctness and evaluation are separate tasks.**
  - **Conformance is checked right after landing (owner rule: land
    first, verify beside; occasional rewinds are accepted for speed):**
    `tools/dev/land.sh --verify` runs
    `tools/ci/landing_conformance.py` (also run on demand) for any change
    that can alter a stream (encoder, codec, DSP crates, presets). It codes
    short streams for every shipped preset of every codec (layers, holds,
    intra refresh, starved keys, the main coding tools) and requires each to
    decode bit-exactly (per-frame hashes, no decoded files) against the
    encoder's reconstruction in both standard decoders (H.264 JM + ffmpeg,
    HEVC HM + ffmpeg, VVC VVdeC + ffmpeg, AV1 aomdec + dav1d, VP9 vpxdec +
    ffmpeg, AV2 avmdec + dav2d) and our own. Any error or mismatch is fixed
    forward or reverted at once. Mirroring and publishing never wait for
    verify (owner rule); a verify failure in a page/eval test is fixed
    beside, only a real stream mismatch reverts. Verified streams are cached
    by bitstream hash. Docs/README-only changes skip the landing queue.
  - **Evaluation never decodes our streams:** screens, boards, sweeps,
    charts and tune score our encoder's reconstruction (`--display-recon`);
    `--gate sample|full` remain for explicit use only.
  - Published numbers come from verified heads (builds that passed the
    landing conformance check).

- Platforms: the development PC (Ryzen 9 5950X, AVX2) and a Jetson Orin Nano
  (Cortex-A78AE NEON). WASM is tested on the PC.
- Test clips: the owner's local clip collections (not distributed). Many are already
  compressed, so **only encode a clip at ≤ 50% of its source bitrate**.
- Compare against openh264 (the baseline) using BD-rate on VMAF, XPSNR,
  MS-SSIM and PSNR, plus single-core encode speed.
