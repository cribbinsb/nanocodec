//! Same-input libvpx comparison for nc-checkasm VP9 interpolation benchmarks.
/* Build against libvpx's build directory, source directory, and libvpx.a.
 * Calls public-to-libvpx kernel entry points; no implementation is copied.
 * Source values, phases and block sizes match bench_vp9_mc.rs. Extra physical
 * padding satisfies libvpx SIMD overreads; our Rust API needs no such padding.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vpx_dsp_rtcd.h"
#include "vp9/common/vp9_filter.h"
#define MAX_SIDE 64
#define TAPS 8
#define SIDE (MAX_SIDE + TAPS - 1)
#define EXTRA (4 * SIDE)
#define REPEATS 5
#define TARGET_NS 20000000.0
static uint8_t src[SIDE*SIDE+EXTRA], dst[MAX_SIDE*MAX_SIDE];
static volatile uint32_t sink;
static double now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static void run(int side,int kind,int axis) {
    const int stride=side+TAPS-1;
    const uint8_t *p=src+3*stride+3;
    const InterpKernel *filter=vp9_filter_kernels[kind];
    if(axis==0) vpx_convolve_copy(p,stride,dst,side,filter,0,16,0,16,side,side);
    else if(axis==1) vpx_convolve8_horiz(p,stride,dst,side,filter,7,16,0,16,side,side);
    else if(axis==2) vpx_convolve8_vert(p,stride,dst,side,filter,0,16,7,16,side,side);
    else vpx_convolve8(p,stride,dst,side,filter,7,16,9,16,side,side);
    sink+=dst[0];
}
static double timed(int side,int kind,int axis) {
    uint64_t calls=16;
    for(;;) {
        double t=now(); for(uint64_t i=0;i<calls;i++)run(side,kind,axis);double elapsed=now()-t;
        if(elapsed>=TARGET_NS/4 || calls>=(1u<<30)) {calls=(uint64_t)(calls*TARGET_NS/(elapsed>1?elapsed:1));if(!calls)calls=1;break;} calls*=4;
    }
    double best=1e99;
    for(int j=0;j<REPEATS;j++) {double t=now();for(uint64_t i=0;i<calls;i++)run(side,kind,axis);double ns=(now()-t)/calls;if(ns<best)best=ns;}
    return best;
}
int main(int argc, char **argv) {
    vpx_dsp_rtcd();
    if(argc==5) {
        const int side=atoi(argv[1]),kind=atoi(argv[2]),axis=atoi(argv[3]),calls=atoi(argv[4]);
        const int stride=side+TAPS-1;
        for(int i=0;i<(int)sizeof(src);i++)src[i]=(uint8_t)(i*37+i/stride*19);
        for(int i=0;i<calls;i++)run(side,kind?BILINEAR:EIGHTTAP,axis);
        printf("%u\n",sink);
        return 0;
    }
    const char *names[2][4]={{"vp9_mc_copy","vp9_mc_h","vp9_mc_v","vp9_mc_2d"},{"vp9_bilinear_copy","vp9_bilinear_h","vp9_bilinear_v","vp9_bilinear_2d"}};
    for(int side=4;side<=MAX_SIDE;side*=2) {
        const int stride=side+TAPS-1;
        for(int i=0;i<(int)sizeof(src);i++)src[i]=(uint8_t)(i*37+i/stride*19);
        for(int kind=0;kind<2;kind++)for(int axis=0;axis<4;axis++)printf("%s %dx%d %.2f\n",names[kind][axis],side,side,timed(side,kind?BILINEAR:EIGHTTAP,axis));
    }
    return 0;
}
