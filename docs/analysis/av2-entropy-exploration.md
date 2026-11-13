# AV2 entropy exploration

## Implementation follow-up

The audit below describes the initial fixed-CDF implementation. Its principal
recommendation is now enabled by default in the shared backend and stateful
encoder (`av2.cdf_adaptation=1`): per-tile model
state and trained rates, matching decoder/header support, and transactional
model rollback during motion RD trials. The 125-frame adaptive I/P suite passes
both AVM and dav2d; the adaptive CLI CQP/CBR suite passes 27 frames. Model updates,
motion symbols and header prefixes also have original-C oracle checks.

CDF checkpoints hold an inline snapshot (4,542 bytes after adding TX8 models), avoiding a heap
allocation for every searched block. Host and WASM SIMD strict Clippy pass.
Adaptive streams now reach the source-rate budget on all three quick360 clips;
fixed probabilities had a roughly 946 kb/s floor even at q255. At q208, a bounded
Callgrind key-plus-two-P probe measured P-frame instructions increasing from
45.70M to 46.79M on static Chirla (+2.38%) and 50.21M to 51.98M on JAAD pan
(+3.52%). Fixed/adaptive decoded pixels match exactly in both probes; every
stream passed both external decoders. These are instruction counts, not timings,
and the short probes are not quality curves. Source binary SHA and raw counts
are in `~/nanocodec-data/results/av2-adaptive-instructions-20261108/instructions.json`.
That binary predates the inline checkpoint allocation cleanup (irrelevant to
zero-motion trials). Low-level fixed-CDF helpers remain for differential tests;
`av2.cdf_adaptation=0` permits controlled comparisons. Cross-frame context
inheritance remains disabled. The audit's detailed state/identity cautions still
apply when extending the supported toolset.

## Initial audit

Audit date: 2026-11-08. No production changes or performance measurements made.
Reference: final AVM v1.0.0, commit
`966a7d7cd6fcf60360caf5dc413b2aeeb65e144d`, under
`~/nanocodec-data/anchors/vp9-av2/src/avm-v1.0.0`.
The pinned implementation takes precedence over the development paper.

## Recommendation

Implement **within-tile CDF adaptation with the normative trained rates** next.
Keep primary-reference context inheritance disabled initially. This changes
coding probabilities without requiring more motion/transform search or changing
reconstructed pixels when coding decisions are held fixed. It is a plausible
compression improvement, not a measured gain or a guaranteed speed improvement.
The arithmetic primitive already exists; integrating state is the substantial
remaining work. Simply clearing the header flag produces an invalid stream.

## Paper evidence actually reviewed

Read [Nalci et al., Transform and Entropy Coding in AV2](https://arxiv.org/html/2601.02712v1),
§III-A–O, §IV-A–C/Tables XI–XIII and Appendix A. Relevant details are PARA's
trained context/time-dependent rates (§III-I), ATC's frequency-dependent
alphabets/context neighborhoods (§III-J), and adaptive truncated Rice tails
(§III-K). FSC (§III-L), parity hiding (§III-M), and lossless refinement
(§III-O) require additional coding paths.

Table XII reports indicative overall AI/LD VMAF BD-rate contributions:
PARA −0.51%/−0.29%; ATC −0.78%/−0.19%; TCQ −1.49%/−0.71%;
IST −5.13%/−1.24%. These are adoption-time/tool-off comparisons, not
fixed-CDF-versus-adaptive comparisons, not additive, and not predictions for
nanocodec. Table XI uses `--cpu-used=0`; our subset is different. Table XIII
shows stronger lossless opportunities from larger identity transforms than
RBR alone. Appendix A reports entropy simplification reducing an early
worst-case throughput regression from 12% to about 5%, not an adaptation-only
cost. No isolated Rice result is provided.

## What is already implemented

| Feature | Current native implementation | Remaining scope |
| --- | --- | --- |
| AV2 arithmetic coder and true bypass bits | `entropy.rs`; AV2-specific probability scaling, bypass coding, bounded decoder | Bypass currently loops over bits; a grouped implementation is an independent exact-speed experiment |
| PARA update primitive | `Cdf`, `Writer::symbol`, `Reader::symbol` | Picture syntax uses `fixed`; trained rates and persistent picture/tile models are absent |
| ATC for supported 4×4 blocks | `coeff.rs`: diagonal scan; low-frequency luma/chroma split; separate base/escape alphabets; neighborhood contexts; low-range pass | New transform shapes/types need their own context derivation; this is not a dormant flag |
| Adaptive high-range Rice | `hr_parameter`, `write_high`, `read_high`; average resets per transform block and updates after high-range values; bounded Exp-Golomb escape | Already active even while CDF updates are disabled; do not confuse Rice adaptation with probability adaptation |
| DCT/WHT and inferred transform syntax | Reduced transform set 2, 4×4 coefficients, appropriate lossless inter transform symbol | No general transform selection/IST/IDTX/FSC |
| TCQ, parity hiding, CCTX, lossless DPCM | Disabled/unsupported in the picture subset | Additional syntax, reconstruction or coefficient-selection work; not low-risk header toggles |

Evidence: [entropy.rs](../../crates/nc-av2/src/entropy.rs),
[coeff.rs](../../crates/nc-av2/src/coeff.rs),
[coefficient tables](../../crates/nc-av2/src/coeff_tables.rs),
[intra modes](../../crates/nc-av2/src/intra_modes.rs),
[frame header](../../crates/nc-av2/src/frame_header.rs).
The adaptive coder's existing 20,000-symbol mixed-symbol/bypass test compares
against `tests/data/avm-entropy.bin`; this establishes the primitive, not correct
picture-level model indexing.

## Exact adaptation integration required

1. **Preserve full original model initialization.** AVM
   `av2/common/entropymode.c:av2_setup_past_independence` initializes coefficient,
   mode and MV probabilities. `av2/common/entropy.c:av2_default_coef_probs`
   chooses coefficient categories at qindex ≤90, ≤140, ≤190, otherwise category
   3. Extract only the models reachable by our subset, including the three
   stored rate offsets and a zero initial count. The current
   `examples/gen_coeff_tables.py` deliberately strips these fields. Extract
   mode/MV/partition rates too, including models whose initial CDF is uniform.
2. **Use final AVM's exact state layout/updates.** In `avm_dsp/prob.h`,
   `CDF_SIZE(N)=N+4`: N inverse boundaries (last zero), count, three rate
   offsets. Before each update choose interval 0/1/2 for counts 0–15/16–31/32;
   `rate=2+stored_offset[interval]`. Each nonterminal boundary moves toward
   32768 if its index precedes the symbol, otherwise toward zero, using integer
   right shifts. Increment count, saturating at 32. `AVM_PARA2/3` add 2/3/4 to
   the raw triplet; `AVM_PARA4..8` add 3/4/5. Do not pass raw training offsets
   into `Cdf::new`, or add the alphabet/time adjustments twice. Native `adapt`
   matches this rule.
3. **Represent model identity, not just equal probabilities.** The fixed
   implementation can reuse identical slices where the adaptive decoder keeps
   separate models. Concrete traps include the uniform Y/UV DPCM models,
   intra/inter transform-partition models, and lossless transform-size/type
   models. Conversely `eob_extra_cdf` is one shared model: do not create a new
   one per token or transform. The original sign tables have separate plane
   axes, but `encodetxb.c` codes chroma DC signs as bypass in our current subset;
   only Y adapts `dc_sign_cdf`. Do not introduce chroma sign adaptation merely
   because a chroma default table exists. Audit every collapsed index and
   inline uniform CDF in coefficient, mode, MV, DRL, partition, skip and
   transform syntax.
4. **Own mutable state per tile.** AVM `decoder/decodeframe.c` initializes
   `tile_data->tctx = *cm->fc`, and `xd->tile_ctx` points to that local copy.
   Start one native context from defaults at each picture's sole tile, pass
   mutable references through the complete traversal, and preserve updates
   between blocks/planes. Never recreate a CDF per call/block/superblock.
   All syntax using `avm_write_symbol`/`avm_read_symbol` must update in order;
   literal bypass bits do not. A coefficient-only toggle is insufficient.
5. **Signal the same policy on both paths.** Key/inter headers currently write
   `disable_cdf_update=1`, and parsers require it. Add an explicit header field
   and dispatch fixed/adaptive behavior symmetrically. Keep an explicit fixed
   path for regressions. AVM's bitreader/bitwriter gates updates using
   `allow_update_cdf = !disable_cdf_update`.
6. **Keep lifetime simple initially.** Our inter headers signal primary NONE
   and disable cross-frame context. AVM `set_primary_ref_frame_and_ctx` then
   calls `av2_setup_past_independence`, so each picture starts from defaults
   even with within-tile adaptation enabled. Do not carry the last tile's
   probabilities into the next picture. Future cross-frame adaptation needs
   reference-associated frame contexts, correct refresh/inheritance semantics,
   and any applicable averaging/reset rules. Commit decoder state only after
   a complete valid packet, preserving existing malformed-packet rollback.

## Cost and efficient validation

An adapted N-symbol call adds N−1 small integer shift/add/subtract updates,
one saturated counter update, and mutable-state traffic. It also adds a
loop-carried probability dependency. Initializing only the active qindex
category avoids copying four coefficient categories per frame. Each typed
native model needs `2*N+4` bytes before any aggregate alignment; the full
working set also depends on restoring currently collapsed context axes.
Do not infer its size from the paper's full-codec memory figure.

Keep model selection statically indexed, avoid per-symbol allocation/hash-map
lookup, and validate generated CDFs at construction. The existing public
`fixed` coder validates each CDF on every symbol; a private trusted path could
remove repeated validation after independent correctness coverage. Preserve
validation for externally supplied tables.

The first check should freeze decisions and compare fixed/adaptive byte counts
for a few already-used tiny textured, smooth and moving pictures. Require
identical native reconstructions and exact AVM plus dav2d decoding, exercising
all active syntax and >32 visits to selected contexts. Add original-C traces
covering count transitions 15→16 and 31→32, all used alphabets, and independently used
model separation. Include a frame-to-frame reset check and failed-packet state
check. Only then run one bounded representative speed/size comparison through
the current shared test framework. A broad quality sweep is unnecessary for
this first change because unchanged decisions preserve reconstructed pixels.

After adaptation, prioritize measured entropy hot spots and inexpensive
coefficient decisions. FSC/IDTX plus lossless refinement is a separate useful
screen-content experiment; TCQ/IST require a more substantial implementation
and search budget. Treat parity hiding's encoder-side coefficient adjustment
as real work even though its decoder arithmetic is small.

## Final-source precedence example

The paper's §III-G describes lossless step 32/WHT scaling 4. Our independently
C-checked final-AVM transform path uses step 64 and WHT scaling 8. Follow
`av2/common/quant_common.c` and the pinned transform oracle, not those earlier
numbers. The same discipline applies to entropy initialization, syntax and
model indexing; paper descriptions alone are insufficient conformance proof.
