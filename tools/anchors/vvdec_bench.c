/*
 * vvdec-bench: VVdeC timed in process, a reference of tools/dev/decoder_speed.py.
 * The Annex-B stream's NAL units (with start codes) are each copied into an access
 * unit before timing, as vvdecapp feeds them one at a time; each repeat opens a
 * fresh one-thread decoder (untimed) and times vvdec_decode over every NAL unit
 * and vvdec_flush until end of stream; frames are unreferenced, not copied.
 *
 *   vvdec-bench [--repeat N] STREAM.vvc
 *
 * Built against the pinned VVdeC by tools/anchors/build_anchors.sh (decbench).
 */
#include <vvdec/vvdec.h>

#include "decode_bench.h"

static long take(vvdecDecoder *decoder, vvdecFrame *frame)
{
    if (!frame)
        return 0;
    vvdec_frame_unref(decoder, frame);
    return 1;
}

int main(int argc, char **argv)
{
    int repeats;
    Stream stream;
    bench_read(bench_args(argc, argv, &repeats), &stream);
    bench_split_annexb(&stream);
    vvdecAccessUnit **units = malloc(stream.count * sizeof *units);
    if (!units)
        bench_die("out of memory");
    for (size_t i = 0; i < stream.count; i++) {
        units[i] = vvdec_accessUnit_alloc();
        vvdec_accessUnit_alloc_payload(units[i], (int)stream.units[i].size);
        memcpy(units[i]->payload, stream.units[i].data, stream.units[i].size);
        units[i]->payloadUsedSize = (int)stream.units[i].size;
    }
    for (int repeat = 0; repeat < repeats; repeat++) {
        vvdecParams params;
        vvdec_params_default(&params);
        params.threads = 1;
        vvdecDecoder *decoder = vvdec_decoder_open(&params);
        if (!decoder)
            bench_die("cannot open the decoder");
        long frames = 0;
        double start = bench_seconds();
        for (size_t i = 0; i < stream.count; i++) {
            vvdecFrame *frame = NULL;
            int err = vvdec_decode(decoder, units[i], &frame);
            if (err != VVDEC_OK && err != VVDEC_TRY_AGAIN)
                bench_die(vvdec_get_error_msg(err));
            frames += take(decoder, frame);
        }
        for (;;) {
            vvdecFrame *frame = NULL;
            int err = vvdec_flush(decoder, &frame);
            if (err != VVDEC_OK && err != VVDEC_EOF)
                bench_die(vvdec_get_error_msg(err));
            frames += take(decoder, frame);
            if (err == VVDEC_EOF || !frame)
                break;
        }
        double seconds = bench_seconds() - start;
        vvdec_decoder_close(decoder);
        bench_report(repeat, frames, seconds);
    }
    return 0;
}
