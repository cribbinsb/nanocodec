//! Same-input libvpx deblocking comparison; BSD-3-Clause source functions remain external.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifndef VP9_ISA
#define VP9_ISA sse2
#endif
#define JOIN_(a,b) a##b
#define JOIN(a,b) JOIN_(a,b)
#define DECL(name) extern void JOIN(name,VP9_ISA)(uint8_t *,int,const uint8_t *,const uint8_t *,const uint8_t *)
DECL(vpx_lpf_horizontal_4_); DECL(vpx_lpf_vertical_4_);
DECL(vpx_lpf_horizontal_8_); DECL(vpx_lpf_vertical_8_);
DECL(vpx_lpf_horizontal_16_); DECL(vpx_lpf_vertical_16_);
typedef void (*filter_fn)(uint8_t *,int,const uint8_t *,const uint8_t *,const uint8_t *);
enum { SIDE=32, AREA=SIDE*SIDE, EDGE=12, REPEATS=5 };
static const double TARGET_NS=20000000.0;
static uint8_t input[AREA], output[AREA];
static _Alignas(16) uint8_t boundary[16], inside[16], variance[16];
static volatile uint32_t sink;
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec*1e9+t.tv_nsec; }
static void call(filter_fn f) {
    memcpy(output,input,AREA);
    f(output+EDGE*SIDE+EDGE,SIDE,boundary,inside,variance);
    sink+=output[EDGE*SIDE+EDGE];
}
static double bench(filter_fn f) {
    uint64_t n=16;
    for (;;) {
        double start=now_ns();for(uint64_t i=0;i<n;i++)call(f);double elapsed=now_ns()-start;
        if(elapsed>=TARGET_NS/4) { n=(uint64_t)(n*TARGET_NS/elapsed);if(!n)n=1;break; }n*=4;
    }
    double best=1e30;
    for(int r=0;r<REPEATS;r++) { double start=now_ns();for(uint64_t i=0;i<n;i++)call(f);double ns=(now_ns()-start)/n;if(ns<best)best=ns; }
    return best;
}
int main(void) {
    const filter_fn f[3][2] = {
        {JOIN(vpx_lpf_horizontal_4_,VP9_ISA),JOIN(vpx_lpf_vertical_4_,VP9_ISA)},
        {JOIN(vpx_lpf_horizontal_8_,VP9_ISA),JOIN(vpx_lpf_vertical_8_,VP9_ISA)},
        {JOIN(vpx_lpf_horizontal_16_,VP9_ISA),JOIN(vpx_lpf_vertical_16_,VP9_ISA)}
    };
    const int widths[3]={4,8,16};const char *patterns[4]={"reject","narrow","flat","mixed"};
    memset(boundary,100,16);memset(inside,32,16);memset(variance,2,16);
    for(int w=0;w<3;w++)for(int vertical=0;vertical<2;vertical++)for(int pattern=0;pattern<4;pattern++) {
        for(int i=0;i<AREA;i++) {
            int across=vertical?i%SIDE:i/SIDE;int line=vertical?i/SIDE:i%SIDE;
            input[i]=(uint8_t)(pattern==0?i*37+i/SIDE*11:pattern==1?120+(across%3)*2:pattern==2?120+(across>=EDGE)*4:line%2==0?120+(across>=EDGE)*4:i*37);
        }
        printf("vp9_filter%d_%c_%s %.2f\n",widths[w],vertical?'v':'h',patterns[pattern],bench(f[w][vertical]));
    }
    return 0;
}
