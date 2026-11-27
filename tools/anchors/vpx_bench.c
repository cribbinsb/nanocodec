/*
 * vpx-bench: libvpx's VP9 decoder (vpxdec's library) timed in process, a reference
 * of tools/dev/decoder_speed.py. IVF frames in memory; each repeat opens a fresh
 * one-thread decoder (untimed) and times vpx_codec_decode over every frame, the
 * end-of-stream flush and vpx_codec_get_frame until drained; frames are not copied.
 *
 *   vpx-bench [--repeat N] STREAM.ivf
 *
 * Built against the pinned libvpx by tools/anchors/build_anchors.sh (decbench).
 */
#include <vpx/vp8dx.h>
#include <vpx/vpx_decoder.h>

#include "decode_bench.h"

static long drain(vpx_codec_ctx_t *codec)
{
    long frames = 0;
    vpx_codec_iter_t iter = NULL;
    while (vpx_codec_get_frame(codec, &iter))
        frames++;
    return frames;
}

int main(int argc, char **argv)
{
    int repeats;
    Stream stream;
    bench_read(bench_args(argc, argv, &repeats), &stream);
    bench_split_ivf(&stream);
    for (int repeat = 0; repeat < repeats; repeat++) {
        vpx_codec_ctx_t codec;
        vpx_codec_dec_cfg_t cfg = {.threads = 1};
        if (vpx_codec_dec_init(&codec, vpx_codec_vp9_dx(), &cfg, 0))
            bench_die("cannot open the decoder");
        long frames = 0;
        double start = bench_seconds();
        for (size_t i = 0; i < stream.count; i++) {
            if (vpx_codec_decode(&codec, stream.units[i].data, (unsigned)stream.units[i].size, NULL, 0))
                bench_die(vpx_codec_error(&codec));
            frames += drain(&codec);
        }
        vpx_codec_decode(&codec, NULL, 0, NULL, 0);
        frames += drain(&codec);
        double seconds = bench_seconds() - start;
        vpx_codec_destroy(&codec);
        bench_report(repeat, frames, seconds);
    }
    return 0;
}
