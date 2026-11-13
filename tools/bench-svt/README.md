# bench-svt: our kernels against SVT-AV1's

Compares `nc-dsp` kernels with SVT-AV1 v4.3.0's own optimised versions
(the anchors' `libSvtAv1Enc.a`, run-time dispatched exactly as the encoder
uses them) on identical inputs.

```
tools/bench-svt/run.sh [core]     build both benches, run them on one pinned core,
                                  print ns/call side by side (ROUNDS=3 by default)
tools/bench-svt/asm_stats.py BIN SYMBOL [--loop]
                                  instruction statistics of one function, and
                                  its largest innermost loop
```

- `svt_bench.c` calls SVT's kernels (SAD, `convolve_2d_sr`, forward DCT,
  `inv_txfm_add`, `quantize_b`, sub-pixel variance). Its inputs are generated
  with `nc-checkasm`'s generator and seed, so both benches see the same
  bytes. SVT's SIMD needs 32-byte aligned coefficient buffers.
- `compare.py` joins the outputs: SVT time, our best SIMD time, ratio.
- Run on an idle machine. `run.sh` alternates the two benches over several
  rounds and keeps each kernel's best time, which filters out some
  interference, not all.
- `ASM.md`: the instruction-level comparison and its findings.

Requires the anchors build (`tools/anchors/build_anchors.sh`) on the
machine, gcc, and the Rust toolchain in `~/.cargo`.
