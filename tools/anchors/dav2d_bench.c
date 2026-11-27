/*
 * dav2d-bench: dav2d (the VideoLAN AV2 decoder) timed in process, a reference of
 * tools/dev/decoder_speed.py. IVF temporal units in memory; each repeat opens a
 * fresh decoder with one thread and a frame delay of 1 (dav2d --threads 1
 * --framedelay 1; untimed) and times dav2d_send_data / dav2d_get_picture over
 * every unit and the drain; pictures are unreferenced, not copied. The data is
 * wrapped, never copied.
 *
 *   dav2d-bench [--repeat N] STREAM.ivf
 *
 * Built against the pinned dav2d by tools/anchors/build_anchors.sh (decbench).
 */
#include <errno.h>

#include <dav2d/dav2d.h>

#include "decode_bench.h"

static void keep(const uint8_t *buf, void *cookie)
{
    (void)buf; /* the stream outlives every pass */
    (void)cookie;
}

/* Every picture the decoder has ready; dies on a decode error. */
static long drain(Dav2dContext *decoder)
{
    long frames = 0;
    Dav2dPicture picture = {0};
    int err;
    while (!(err = dav2d_get_picture(decoder, &picture))) {
        frames++;
        dav2d_picture_unref(&picture);
    }
    if (err != DAV2D_ERR(EAGAIN))
        bench_die("decode error");
    return frames;
}

int main(int argc, char **argv)
{
    int repeats;
    Stream stream;
    bench_read(bench_args(argc, argv, &repeats), &stream);
    bench_split_ivf(&stream);
    for (int repeat = 0; repeat < repeats; repeat++) {
        Dav2dSettings settings;
        dav2d_default_settings(&settings);
        settings.n_threads = 1;
        settings.max_frame_delay = 1;
        Dav2dContext *decoder = NULL;
        if (dav2d_open(&decoder, &settings))
            bench_die("cannot open the decoder");
        long frames = 0;
        double start = bench_seconds();
        for (size_t i = 0; i < stream.count; i++) {
            Dav2dData data = {0};
            if (dav2d_data_wrap(&data, stream.units[i].data, stream.units[i].size, keep, NULL))
                bench_die("cannot wrap a unit");
            while (data.sz) {
                int err = dav2d_send_data(decoder, &data);
                if (err && err != DAV2D_ERR(EAGAIN))
                    bench_die("decode error");
                frames += drain(decoder);
            }
        }
        frames += drain(decoder);
        double seconds = bench_seconds() - start;
        dav2d_close(&decoder);
        bench_report(repeat, frames, seconds);
    }
    return 0;
}
