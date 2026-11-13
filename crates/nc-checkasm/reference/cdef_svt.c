//! Direct native-byte CDEF comparison with SVT-AV1; no codec code is copied.
// Compile with SVT's cdef.c and ISA cdef_filter_block file, function/data
// sections enabled, then link with --gc-sections. This needs no full library.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cdef.h"
#include "common_dsp_rtcd.h"

#define SIDE 32
#define ORIGIN 8
#define REPEATS 5
#define TARGET_NS 20000000.0
static uint8_t input[CDEF_BSTRIDE * SIDE];
static uint8_t output[SIDE * SIDE];
static uint8_t reference[SIDE * SIDE];
static volatile uint32_t sink;

typedef void (*kernel_fn)(uint8_t*, int32_t, const uint8_t*, int32_t, int32_t,
                          int32_t, int32_t, int32_t, int32_t, uint8_t);
#if ARCH_X86_64
#define KERNEL svt_cdef_filter_block_8bit_avx2
#else
#define KERNEL svt_cdef_filter_block_8bit_neon
#endif
extern void KERNEL(uint8_t*, int32_t, const uint8_t*, int32_t, int32_t,
                   int32_t, int32_t, int32_t, int32_t, uint8_t);

static double now_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec * 1e9 + value.tv_nsec;
}

static void invoke(kernel_fn function, int side, int primary, int secondary, uint8_t* dst) {
    function(dst + ORIGIN * SIDE + ORIGIN, SIDE,
             input + ORIGIN * CDEF_BSTRIDE + ORIGIN, primary, secondary,
             3, 4, side == 8 ? BLOCK_8X8 : BLOCK_4X4, 0, 1);
}

static void calls(uint64_t count, int side, int primary, int secondary) {
    for (uint64_t i = 0; i < count; ++i) {
        invoke(KERNEL, side, primary, secondary, output);
        sink += output[ORIGIN * SIDE + ORIGIN];
    }
}

static double timed(int side, int primary, int secondary) {
    uint64_t count = 16;
    for (;;) {
        double start = now_ns();
        calls(count, side, primary, secondary);
        double elapsed = now_ns() - start;
        if (elapsed >= TARGET_NS / 4) {
            count = (uint64_t)(count * TARGET_NS / elapsed);
            if (!count) count = 1;
            break;
        }
        count *= 4;
    }
    double best = 1e30;
    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        double start = now_ns();
        calls(count, side, primary, secondary);
        double ns = (now_ns() - start) / count;
        if (ns < best) best = ns;
    }
    return best;
}

int main(int argc, char** argv) {
    for (int row = 0; row < SIDE; ++row)
        for (int col = 0; col < SIDE; ++col)
            input[row * CDEF_BSTRIDE + col] = 120 + (col >= ORIGIN) * 4 + (row * SIDE + col) % 3;
    const int strengths[3][2] = {{8, 2}, {15, 0}, {0, 4}};
    const char* names[] = {"cdef_filter_u8", "cdef_filter_u8_pri", "cdef_filter_u8_sec"};
    if (argc == 4) {
        const int side = atoi(argv[1]);
        const int kind = atoi(argv[2]);
        if ((side != 4 && side != 8) || kind < 0 || kind >= 3) return 2;
        calls(strtoull(argv[3], NULL, 10), side, strengths[kind][0], strengths[kind][1]);
        printf("%u\n", output[ORIGIN * SIDE + ORIGIN]);
        return 0;
    }
    for (int side = 4; side <= 8; side += 4) {
        for (int kind = 0; kind < 3; ++kind) {
            memset(output, 0, sizeof(output));
            memset(reference, 0, sizeof(reference));
            invoke(KERNEL, side, strengths[kind][0], strengths[kind][1], output);
            invoke(svt_cdef_filter_block_8bit_c, side, strengths[kind][0], strengths[kind][1], reference);
            if (memcmp(output, reference, sizeof(output))) {
                fprintf(stderr, "SVT scalar mismatch: %dx%d %s\n", side, side, names[kind]);
                return 1;
            }
            printf("%s %dx%d %.2f\n", names[kind], side, side,
                   timed(side, strengths[kind][0], strengths[kind][1]));
        }
    }
    return 0;
}
