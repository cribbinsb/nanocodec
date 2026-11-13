//! Full AVX2 warp dataflow from dav1d 1.5.4 `src/x86/mc_avx2.asm`
//! (`warp_affine_8x8_8bpc`, BSD-2-Clause; see `THIRD_PARTY.md`).
//!
//! The packed-byte tap transpose, scalar phase progression and alternating
//! row-pair schedule stay in fixed registers. The equivalent intrinsic port
//! saved about a third of the baseline time but spilled phase state and
//! missed the reference target. This bounded assembly keeps the coherent
//! reference schedule; Rust checks window/table extents and owns dispatch.
//!
//! Horizontal rounded values occupy the high word of each dword. The
//! vertical transpose widens signed filter bytes by placing them in the
//! high byte of words (multiplication by 256), avoiding eight i16 loads.
//! Horizontal magnitude is at most ceil(255*222/8)=7077; hence vertical
//! magnitude times 256 is below 7077*222*256 < 2^31. Rounding and shifting
//! stay in signed i32 for every round1=1..11, preserving negative and >255
//! outputs. No final pixel clipping or word saturation is performed here.
//!
//! Register contract: ymm1..6 retain row pairs; ymm0/7 hold outputs;
//! ymm8..10 are scratch, ymm11 is zero, ymm12/13 hold shuffles, ymm14/15
//! horizontal constants. r8/r9 hold alpha/adjusted beta, r10/r11 gamma/
//! adjusted delta, r12/r13 running phases; r14 table and r15 source stride.
//! The helper at label20 shares horizontal code; label30 emits two rows.

#![allow(unsafe_code)]
#![allow(
    clippy::many_single_char_names,
    reason = "the normative affine phase variables"
)]

use super::super::warp::{PHASE_ROUND, PHASE_SHIFT, WARPED_FILTERS_PAIRED};
use super::super::warp_kernels::{WINDOW_SIDE, WINDOW_STRIDE, WarpOut, WarpPhases};
static SHUFFLE_A: [u8; 32] = [
    0, 2, 4, 6, 1, 3, 5, 7, 1, 3, 5, 7, 2, 4, 6, 8, 4, 6, 8, 10, 5, 7, 9, 11, 5, 7, 9, 11, 6, 8,
    10, 12,
];
static SHUFFLE_B: [u8; 32] = [
    2, 4, 6, 8, 3, 5, 7, 9, 3, 5, 7, 9, 4, 6, 8, 10, 6, 8, 10, 12, 7, 9, 11, 13, 7, 9, 11, 13, 8,
    10, 12, 14,
];
/// One extra row makes the high-lane 16-byte load valid for the final
/// eight-byte filter. Its extra bytes are discarded by the transpose.
const FILTER_COUNT: usize = WARPED_FILTERS_PAIRED.len();
#[repr(align(64))]
struct AlignedFilters([[i8; 8]; FILTER_COUNT + 1]);
static FILTERS: AlignedFilters = AlignedFilters({
    let mut rows = [[0; 8]; FILTER_COUNT + 1];
    let mut i = 0;
    while i < FILTER_COUNT {
        rows[i] = WARPED_FILTERS_PAIRED[i];
        i += 1;
    }
    rows
});
/// Horizontal half-sum weights (two copies of 2^13) and upper-word rounding.
static HORIZONTAL_CONSTANTS: [i32; 2] = [0x2000_2000, 1 << 15];
/// Affine extrema occur at rectangle corners. i64 checks also prevent any
/// i32 phase arithmetic overflow before unchecked table addressing. Outside
/// this range the original clamped AVX2 kernel retains its behavior.
fn valid(phases: &WarpPhases) -> bool {
    let bound = |start: i32, dx: i32, dy: i32, first: i64, last: i64| {
        let base = i64::from(start) + i64::from(PHASE_ROUND);
        [first, last].into_iter().all(|r| {
            [0, 7].into_iter().all(|c| {
                let phase = base + i64::from(dx) * c + i64::from(dy) * r;
                (0..(FILTER_COUNT as i64) << PHASE_SHIFT).contains(&phase)
            })
        })
    };
    let (x, a, b) = phases.x;
    let (y, g, d) = phases.y;
    bound(x, a, b, -3, 11) && bound(y, g, d, 0, 7)
}

/// The assembly reads these fixed repr(C) offsets. The scalar state is kept
/// in registers while processing rows; only final rounding is reloaded.
#[repr(C)]
struct Config {
    alpha: i32,
    beta: i32,
    gamma: i32,
    delta: i32,
    x: i32,
    y: i32,
    shift: i32,
    round: i32,
    end: *mut i32,
}
// Keep the assembly state layout checked at compile time.
const _: () = {
    assert!(core::mem::offset_of!(Config, alpha) == 0);
    assert!(core::mem::offset_of!(Config, beta) == 4);
    assert!(core::mem::offset_of!(Config, gamma) == 8);
    assert!(core::mem::offset_of!(Config, delta) == 12);
    assert!(core::mem::offset_of!(Config, x) == 16);
    assert!(core::mem::offset_of!(Config, y) == 20);
    assert!(core::mem::offset_of!(Config, shift) == 24);
    assert!(core::mem::offset_of!(Config, round) == 28);
    assert!(core::mem::offset_of!(Config, end) == 32);
};
#[allow(
    clippy::too_many_lines,
    reason = "one assembly block keeps the full row-pair register schedule intact"
)]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn warp8x8(
    (src, stride): (&[u8], usize),
    phases: &WarpPhases,
    round1: u32,
    out: &mut WarpOut,
) {
    let span = stride
        .checked_mul(WINDOW_SIDE - 1)
        .and_then(|n| n.checked_add(WINDOW_STRIDE))
        .expect("warp window extent");
    assert!(src.len() >= span, "warp window too short");
    if !valid(phases) {
        // SAFETY: AVX2 enabled; the original fallback checks its source and clamps taps.
        unsafe { super::clamped((src, stride), phases, round1, out) };
        return;
    }
    let (x, a, b) = phases.x;
    let (y, g, d) = phases.y;
    let config = Config {
        alpha: a,
        beta: b - 3 * a,
        gamma: g,
        delta: d - 3 * g,
        x: x + PHASE_ROUND - 3 * b,
        y: y + PHASE_ROUND,
        shift: (round1 + 8) as i32,
        round: 1 << (round1 + 7),
        end: out.as_mut_ptr().wrapping_add(64),
    };
    // SAFETY: AVX2 is enabled. The checked span covers all fifteen 16-byte
    // source loads; the i64 corner proof bounds every table index. FILTERS
    // includes the extra eight bytes needed by its final 16-byte load. Four
    // iterations write exactly eight i32 outputs twice into the 64-value out.
    // All modified GPRs/YMMs are declared; rsi/r14/r15 remain unchanged.
    // Local calls push/pop only their return address and restore rsp before
    // exit. Without nostack the compiler reserves its red zone against these
    // return-address writes. Memory/flags effects remain enabled. No external
    // function is called and no unwind can originate inside the assembly.
    unsafe {
        core::arch::asm!(
            "movsxd r8, dword ptr [rsi]",
            "movsxd r9, dword ptr [rsi+4]",
            "movsxd r10, dword ptr [rsi+8]",
            "movsxd r11, dword ptr [rsi+12]",
            "mov r12d, dword ptr [rsi+16]",
            "mov r13d, dword ptr [rsi+20]",
            "vmovdqu ymm12, ymmword ptr [rip+{sa}]",
            "vmovdqu ymm13, ymmword ptr [rip+{sb}]",
            "vpbroadcastd ymm14, dword ptr [rip+{hc}]",
            "vpbroadcastd ymm15, dword ptr [rip+{hc}+4]",
            "vpxor ymm11, ymm11, ymm11",
            "call 20f",
            "vpsrld ymm1, ymm0, 16",
            "call 20f",
            "vpsrld ymm4, ymm0, 16",
            "call 20f",
            "vpblendw ymm1, ymm1, ymm0, 0xaa",
            "call 20f",
            "vpblendw ymm4, ymm4, ymm0, 0xaa",
            "call 20f",
            "vpsrld ymm2, ymm1, 16",
            "vpblendw ymm2, ymm2, ymm0, 0xaa",
            "call 20f",
            "vpsrld ymm5, ymm4, 16",
            "vpblendw ymm5, ymm5, ymm0, 0xaa",
            "call 20f",
            "vpsrld ymm3, ymm2, 16",
            "vpblendw ymm3, ymm3, ymm0, 0xaa",
            // Six retained row pairs: 02/24/46 and 13/35/57. Produce two
            // vertical outputs, then rotate by two rows without spilling them.
            "30:",
            "call 20f",
            "vpsrld ymm6, ymm5, 16",
            "vpblendw ymm6, ymm6, ymm0, 0xaa",
            "lea eax, [r13+r10*4]",
            "lea ecx, [r13+r10*1]",
            "shr r13d, {phase_shift}",
            "shr eax, {phase_shift}",
            "vmovq xmm8, [r14+r13  *8]",
            "vinserti128 ymm8, ymm8, xmmword ptr [r14+rax*8], 1",
            "lea eax, [rcx+r10*4]",
            "lea r13d, [rcx+r10*1]",
            "shr ecx, {phase_shift}",
            "shr eax, {phase_shift}",
            "vmovq xmm0, [r14+rcx*8]",
            "vinserti128 ymm0, ymm0, xmmword ptr [r14+rax*8], 1",
            "lea eax, [r13+r10*4]",
            "lea ecx, [r13+r10*1]",
            "shr r13d, {phase_shift}",
            "shr eax, {phase_shift}",
            "vmovq xmm9, [r14+r13  *8]",
            "vinserti128 ymm9, ymm9, xmmword ptr [r14+rax*8], 1",
            "lea eax, [rcx+r10*4]",
            "lea r13d, [rcx+r11]",
            "shr ecx, {phase_shift}",
            "shr eax, {phase_shift}",
            "vpunpcklwd ymm8, ymm8, ymm0",
            "vmovq xmm0, [r14+rcx*8]",
            "vinserti128 ymm0, ymm0, xmmword ptr [r14+rax*8], 1",
            "vpunpcklwd ymm0, ymm9, ymm0",
            "vpunpckldq ymm9, ymm8, ymm0",
            "vpunpckhdq ymm0, ymm8, ymm0",
            "vpunpcklbw ymm8, ymm11, ymm9",
            "vpunpckhbw ymm9, ymm11, ymm9",
            "vpmaddwd ymm1, ymm1, ymm8",
            "vpmaddwd ymm9, ymm9, ymm3",
            "vpunpcklbw ymm8, ymm11, ymm0",
            "vpunpckhbw ymm0, ymm11, ymm0",
            "vpmaddwd ymm8, ymm8, ymm4",
            "vpmaddwd ymm0, ymm0, ymm6",
            "vpaddd ymm1, ymm1, ymm9",
            "vpaddd ymm0, ymm0, ymm8",
            "vpaddd ymm7, ymm0, ymm1",
            "call 20f",
            "vmovdqa ymm1, ymm2",
            "vmovdqa ymm2, ymm3",
            "vpsrld ymm3, ymm3, 16",
            "vpblendw ymm3, ymm3, ymm0, 0xaa",
            "lea eax, [r13+r10*4]",
            "lea ecx, [r13+r10*1]",
            "shr r13d, {phase_shift}",
            "shr eax, {phase_shift}",
            "vmovq xmm8, [r14+r13  *8]",
            "vinserti128 ymm8, ymm8, xmmword ptr [r14+rax*8], 1",
            "lea eax, [rcx+r10*4]",
            "lea r13d, [rcx+r10*1]",
            "shr ecx, {phase_shift}",
            "shr eax, {phase_shift}",
            "vmovq xmm0, [r14+rcx*8]",
            "vinserti128 ymm0, ymm0, xmmword ptr [r14+rax*8], 1",
            "lea eax, [r13+r10*4]",
            "lea ecx, [r13+r10*1]",
            "shr r13d, {phase_shift}",
            "shr eax, {phase_shift}",
            "vmovq xmm9, [r14+r13  *8]",
            "vinserti128 ymm9, ymm9, xmmword ptr [r14+rax*8], 1",
            "lea eax, [rcx+r10*4]",
            "lea r13d, [rcx+r11]",
            "shr ecx, {phase_shift}",
            "shr eax, {phase_shift}",
            "vpunpcklwd ymm8, ymm8, ymm0",
            "vmovq xmm0, [r14+rcx*8]",
            "vinserti128 ymm0, ymm0, xmmword ptr [r14+rax*8], 1",
            "vpunpcklwd ymm0, ymm9, ymm0",
            "vpunpckldq ymm9, ymm8, ymm0",
            "vpunpckhdq ymm0, ymm8, ymm0",
            "vpunpcklbw ymm8, ymm11, ymm9",
            "vpunpckhbw ymm9, ymm11, ymm9",
            "vpmaddwd ymm4, ymm4, ymm8",
            "vpmaddwd ymm9, ymm9, ymm6",
            "vpunpcklbw ymm8, ymm11, ymm0",
            "vpunpckhbw ymm0, ymm11, ymm0",
            "vpmaddwd ymm8, ymm8, ymm1",
            "vpmaddwd ymm0, ymm0, ymm3",
            "vpaddd ymm4, ymm4, ymm9",
            "vpaddd ymm0, ymm0, ymm8",
            "vpaddd ymm0, ymm0, ymm4",
            "vmovdqa ymm4, ymm5",
            "vmovdqa ymm5, ymm6",
            "vpbroadcastd ymm9, dword ptr [rsi+28]",
            "vmovd xmm8, dword ptr [rsi+24]",
            "vpaddd ymm7, ymm7, ymm9",
            "vpaddd ymm0, ymm0, ymm9",
            "vpsrad ymm7, ymm7, xmm8",
            "vpsrad ymm0, ymm0, xmm8",
            "vmovdqu ymmword ptr [rdi], ymm7",
            "vmovdqu ymmword ptr [rdi+32], ymm0",
            "add rdi, 64",
            "cmp rdi, qword ptr [rsi+32]",
            "jne 30b",
            "jmp 99f",
            // Shared horizontal body. r12 advances by beta each call after
            // four paired-column filter loads; output is rounded in high words.
            "20:",
            "lea eax, [r12+r8*4]",
            "lea ecx, [r12+r8*1]",
            "shr r12d, {phase_shift}",
            "shr eax, {phase_shift}",
            "vmovq xmm8, [r14+r12  *8]",
            "vinserti128 ymm8, ymm8, xmmword ptr [r14+rax*8], 1",
            "lea eax, [rcx+r8*4]",
            "lea r12d, [rcx+r8*1]",
            "shr ecx, {phase_shift}",
            "shr eax, {phase_shift}",
            "vmovq xmm0, [r14+rcx*8]",
            "vinserti128 ymm0, ymm0, xmmword ptr [r14+rax*8], 1",
            "lea eax, [r12+r8*4]",
            "lea ecx, [r12+r8*1]",
            "shr r12d, {phase_shift}",
            "shr eax, {phase_shift}",
            "vmovq xmm9, [r14+r12  *8]",
            "vinserti128 ymm9, ymm9, xmmword ptr [r14+rax*8], 1",
            "lea eax, [rcx+r8*4]",
            "lea r12d, [rcx+r9]",
            "shr ecx, {phase_shift}",
            "shr eax, {phase_shift}",
            "vpunpcklqdq ymm8, ymm8, ymm0",
            "vmovq xmm0, [r14+rcx*8]",
            "vinserti128 ymm0, ymm0, xmmword ptr [r14+rax*8], 1",
            "vpunpcklqdq ymm9, ymm9, ymm0",
            "vbroadcasti128 ymm10, [rdx]",
            "vpshufb ymm0, ymm10, ymm12",
            "vpmaddubsw ymm0, ymm0, ymm8",
            "vpshufb ymm10, ymm10, ymm13",
            "vpmaddubsw ymm10, ymm10, ymm9",
            "add rdx, r15",
            "vphaddw ymm0, ymm0, ymm10",
            "vpmaddwd ymm0, ymm0, ymm14",
            "vpaddd ymm0, ymm0, ymm15",
            "ret",
            "99:",
            // LLVM does not infer the AVX/SSE transition from opaque asm.
            "vzeroupper",
            phase_shift=const PHASE_SHIFT,
            sa=sym SHUFFLE_A,sb=sym SHUFFLE_B,hc=sym HORIZONTAL_CONSTANTS,
            in("rsi") &raw const config,inout("rdx") src.as_ptr()=>_,inout("rdi") out.as_mut_ptr()=>_,
            in("r14") FILTERS.0.as_ptr(),in("r15") stride,
            out("rax") _,out("rcx") _,out("r8") _,out("r9") _,out("r10") _,out("r11") _,out("r12") _,out("r13") _,
            out("ymm0") _,
            out("ymm1") _,
            out("ymm2") _,
            out("ymm3") _,
            out("ymm4") _,
            out("ymm5") _,
            out("ymm6") _,
            out("ymm7") _,
            out("ymm8") _,
            out("ymm9") _,
            out("ymm10") _,
            out("ymm11") _,
            out("ymm12") _,
            out("ymm13") _,
            out("ymm14") _,
            out("ymm15") _,
        );
    }
}
