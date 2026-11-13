# nanocodec vs SVT-AV1: generated code, kernel by kernel

What the compiler makes of our Rust kernels, set against the code SVT-AV1
v4.3.0 actually runs (its run-time dispatch picks AVX2 on the 5950X and
`neon` / `neon_dotprod` on the Orin). Our functions come from the release
`nc-checkasm` binary. SVT's come from the linked `svt_bench` binary: its
static library is LTO bitcode, so the code only exists after linking.
Statistics are from `asm_stats.py` (Intel syntax on x86); "stack" counts
memory operands on `rsp`/`rbp`/`sp`, i.e. spills and stack arrays.

## Timings after the fixes below (`run.sh`, best of alternating rounds)

| Kernel | x86 SVT ns | x86 ours ns | ratio | Orin SVT ns | Orin ours ns | ratio |
|---|---|---|---|---|---|---|
| SAD 16×16 | 5.1 | 5.2 | 1.03× | 14.8 | 14.5 | 0.98× |
| SAD 64×64 | 30.8 | 39.1 | 1.27× | 193.6 | 191.8 | 0.99× |
| AV1 8-tap 2-D 16×16 | 85.8 | 104.9 | 1.22× | 425.4 | 628.0 | 1.48× |
| AV1 8-tap 2-D 64×64 | 1065.2 | 1515.5 | 1.42× | 6098.5 | 8558.1 | 1.40× |
| forward DCT 8×8 | 34.3 | 17.9 | **0.52×** | 94.9 | 109.5 | 1.15× |
| forward DCT 32×32 | 363.3 | 738.4 | 2.03× | 2890.4 | 3091.8 | 1.07× |
| inverse DCT + add 8×8 | 31.3 | 31.5 | 1.01× | 126.2 | 203.2 | 1.61× |
| inverse DCT + add 32×32 | 321.8 | 778.9 | 2.42× | 2595.2 | 6568.1 | 2.53× |
| quantise 32×32 | 226.4 | 296.5 | 1.31× | 1462.0 | 1561.9 | 1.07× |
| sub-pixel variance 16×16 | 25.8 | 31.9 | 1.24× | 73.4 | 210.2 | 2.86× |

The PC had other agents' jobs running (load 3–13); the Orin was idle.

## Codegen problems found and fixed

| Problem | Where | Symptom in the assembly | Fix | Effect |
|---|---|---|---|---|
| A closure lost the target feature | `transpose_tile`'s `array::from_fn(\|i\| load(..))` | an out-of-line `core::array::try_from_fn` and a non-inlined `core_arch::x86::avx::_mm256_loadu_si256` *call* per transposed row | explicit loads, then register-level transposes (`store_transposed`) | fwd DCT 8×8 34 → 18 ns, inv 8×8 46 → 31 ns |
| Zeroed scratch arrays | `[0i16; 71 * w]` prediction scratch, `[0i16; 65 * w]` sub-pel scratch, `[0i32; 1024]` ×2 DCT buffers | `call memset` of 2–8 KiB per call | rolling windows of rows kept in registers (prediction, sub-pel variance); one `MaybeUninit` buffer between DCT passes | 4×4 prediction 133 → 53 ns; DCT one buffer pass fewer |
| `transmute` by value | DCT buffer `MaybeUninit` → `[i32; N*N]` | `call memcpy` of 4 KiB | borrow it as a slice instead | 32×32 inverse 4 KiB copy gone |
| Two passes for 64-wide SAD | `sad_w64` = two `sad::<32>` calls | two row loops | one loop, both halves, two accumulators | inner loop now matches SVT's (below) |
| `vpmulld` per filter tap | vertical 8-tap accumulate | widen + `vpmulld` (10-cycle) per tap | paired taps through `vpmaddwd` on interleaved rows (`acc_mul_add2`) | 2-D predict 16×16 1.62× → 1.22× |
| Bounds checks per load | residual / variance rows | 10 of the 37 instructions in the NEON sub-pel loop were `cmp`/`b.hi`/`b.cc` | slice each whole row once, then load within it | NEON sub-pel loop branches 22 → 7 (≈3% faster: the checks were well predicted) |

None of the remaining gaps is a Rust codegen issue: the inner loops below
are what we asked for. They are algorithm and instruction-set choices.

## Side by side

### SAD 64×64, x86 inner loop (both: 2 rows per iteration, 17 instructions)

```
ours (nc_dsp::pixel::avx2::sad_w64)          SVT (svt_aom_sad64x64_avx2)
vmovdqu  ymm2,[r10-0x20]                     vmovdqu  ymm0,[rdx]
vmovdqu  ymm3,[r10]                          vmovdqu  ymm3,[rdx+0x20]
vpsadbw  ymm2,ymm2,[rbx-0x20]                vpsadbw  ymm0,ymm0,[rdi]
vpaddq   ymm1,ymm2,ymm1                      vpsadbw  ymm3,ymm3,[rdi+0x20]
vpsadbw  ymm2,ymm3,[rbx]                     vmovdqu  ymm1,[rdx+rcx]
vpaddq   ymm0,ymm2,ymm0                      vmovdqu  ymm2,[rdx+rcx+0x20]
... same for row + 1 ...                     ... vpaddd ×4, vmovdqa ...
add/add/cmp/jne                              sub eax,1 / jne
```
Identical work per row. The remaining 1.27× on x86 (parity on the Orin)
is outside the loop: our call goes through `Dsp::sad`'s view checks and a
generic-height loop where SVT's is a fixed 64-row function. That is a few
nanoseconds per call.

### SAD 16×16, NEON (ours loops, SVT fully unrolls; same per-row code)

```
ours (nc_dsp::pixel::neon::sad_wide::<16>)   SVT (svt_aom_sad16x16_neon)
ldr    q1,[x0]                               ldr    q2,[x0]
ldr    q2,[x3]                               ldr    q3,[x2]
uabd   v1.16b,v1.16b,v2.16b                  uabd   v3.16b,v2.16b,v3.16b
uadalp v0.8h,v1.16b                          uadalp v0.8h,v3.16b
subs/add/add/b.ne  (13 instrs total)         ... ×16 rows unrolled (100 instrs)
```

### AV1 8-tap 2-D prediction
- **x86:** our vertical pass is now `vpunpcklwd`/`vpunpckhwd` + `vpmaddwd`
  (two taps per instruction) like SVT's. Our horizontal pass still widens
  every tap (`vpmovzxbw` + `vpmullw` + `vpaddw`, 26 of each in the
  function), where SVT shuffles byte pairs and uses `vpmaddubsw` (two taps
  per instruction, on bytes). Tap broadcasts (`vpbroadcastw` ×24) are
  rematerialised inside loops.
- **Orin:** SVT runs `svt_av1_convolve_2d_sr_neon_dotprod`: the Orin has the
  dot-product extension (`asimddp`) and SVT's horizontal pass is `usdot`/`sdot`
  four taps at a time. Ours uses baseline NEON (`umull`/`mla`). Matching it
  means a dot-product kernel variant behind a run-time feature check.

### Transforms
- **forward 8×8:** ours is half SVT's time: SVT's AVX2 8×8 runs 1277
  instructions with 324 stack accesses (its 32-bit generic stage functions);
  ours keeps the 8 rows in registers.
- **32-point (both directions):** our 32-point butterflies operate on arrays
  of 32 vectors (`input`, `output`, `step` in `dct.rs`), so x86 keeps them
  on the stack (≈1000–1400 stack accesses per call; 16 ymm registers cannot
  hold 32). SVT's forward splits the 32-point transform into stage helpers
  over 16-bit halves. **dav1d's inverse (which SVT uses) is hand-written
  16-bit asm** (`svt_dav1d_inv_txfm_add_dct_dct_32x32_8bpc_avx2`: 16
  coefficients per register, 5 stack accesses), twice our lanes per
  instruction. 16-bit inverse arithmetic is exact for conformant
  coefficients, since AV1 requires every stage to fit 16 bits; ours uses 32
  bits and clamps, which is also exact for out-of-range inputs.
- **8×8 inverse on NEON (1.61×):** the same 16-bit vs 32-bit difference.

### Quantise 32×32 (x86 1.31×)
SVT's `quantize_b` processes 16-bit lanes with `vpmulhw` (libaom's
multiply-shift quantiser); ours is exact division by reciprocal on 32-bit
lanes (`vpmuludq` on even/odd lanes plus blend). Parity on the Orin.

### Sub-pixel variance 16×16 (x86 1.24×, Orin 2.86×)
SVT filters bytes: `vpmaddubsw` on x86, `umull`/`umlal` on 16-byte NEON
vectors with a narrowing rounding shift back to bytes. Ours widens to
16-bit lanes first (`uxtl`, `mul`, `mla`, `sshr` on 8 lanes), so NEON does
twice the vector operations per pixel. After the row-slicing fix the
loop has no avoidable overhead; closing the gap needs a byte-domain
bilinear lane operation.

## What would close the remaining gaps (none are Rust-specific)
1. **16-bit inverse DCTs** (dav1d-style), valid for conformant coefficients:
   the largest remaining gap (2.4–2.5×).
2. **32-point butterflies in two 16-point halves** so the working set fits
   the register file.
3. **Byte-domain filters:** `pmaddubsw` horizontal passes on x86, `usdot` /
   `umull` on NEON (dot-product variant selected at run time), and a byte
   bilinear for sub-pixel variance.
