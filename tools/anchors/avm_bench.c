/*
 * avm-bench: the AV2 reference decoder (AVM, avmdec's library) timed in process, a
 * reference of tools/dev/decoder_speed.py. IVF temporal units in memory; each
 * repeat opens a fresh one-thread decoder (untimed) and times avm_codec_decode
 * over every unit, the end-of-stream flush and avm_codec_get_frame until drained;
 * frames are not copied.
 *
 *   avm-bench [--repeat N] STREAM.ivf
 *
 * Built against the pinned AVM by tools/anchors/build_avm_bench.sh.
 */
#include <avm/avm_decoder.h>
#include <avm/avmdx.h>

#include "decode_bench.h"

static long drain(avm_codec_ctx_t *codec)
{
    long frames = 0;
    avm_codec_iter_t iter = NULL;
    while (avm_codec_get_frame(codec, &iter))
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
        avm_codec_ctx_t codec;
        avm_codec_dec_cfg_t cfg = {.threads = 1};
        if (avm_codec_dec_init(&codec, avm_codec_av2_dx(), &cfg, 0))
            bench_die("cannot open the decoder");
        long frames = 0;
        double start = bench_seconds();
        for (size_t i = 0; i < stream.count; i++) {
            if (avm_codec_decode(&codec, stream.units[i].data, stream.units[i].size, NULL))
                bench_die(avm_codec_error(&codec));
            frames += drain(&codec);
        }
        avm_codec_decode(&codec, NULL, 0, NULL);
        frames += drain(&codec);
        double seconds = bench_seconds() - start;
        avm_codec_destroy(&codec);
        bench_report(repeat, frames, seconds);
    }
    return 0;
}
