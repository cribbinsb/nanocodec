/*
 * libavcodec-bench: FFmpeg's decoders timed in process, the reference side of
 * tools/dev/decoder_speed.py.
 *
 *   libavcodec-bench [--decoder NAME] [--repeat N] STREAM
 *
 * The stream is demuxed (and parsed into frames) by libavformat once, untimed,
 * and every packet kept in memory. Each repeat then opens a fresh one-thread
 * decoder (untimed) and times only the decode loop: avcodec_send_packet for
 * every packet, the end-of-stream flush and avcodec_receive_frame until the
 * decoder is drained; frames are unreferenced, never copied or written. It is
 * the same span `nanocodec decode --bench` times for ours, so neither side is
 * charged process start-up, probing, demuxing or output.
 *
 * Prints one JSON object per repeat: {"repeat", "frames", "seconds"}.
 * --decoder picks a decoder by name (libdav1d for AV1); the default is the
 * stream's codec's default decoder.
 *
 * Built against the pinned static decode-only FFmpeg by
 * tools/anchors/build_anchors.sh (build_ffmpeg).
 */
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_REPEATS 1

static void die(const char *what, int err)
{
    char text[AV_ERROR_MAX_STRING_SIZE] = "";
    if (err < 0)
        av_strerror(err, text, sizeof text);
    fprintf(stderr, "libavcodec-bench: %s %s\n", what, text);
    exit(1);
}

static double now_seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Every packet of the stream's video track, in memory. */
typedef struct Packets {
    AVPacket **items;
    size_t count, capacity;
    AVCodecParameters *params;
} Packets;

static void read_packets(const char *path, Packets *packets)
{
    AVFormatContext *format = NULL;
    int err = avformat_open_input(&format, path, NULL, NULL);
    if (err < 0)
        die("cannot open", err);
    if ((err = avformat_find_stream_info(format, NULL)) < 0)
        die("cannot probe", err);
    int video = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (video < 0)
        die("no video stream", video);
    packets->params = avcodec_parameters_alloc();
    avcodec_parameters_copy(packets->params, format->streams[video]->codecpar);
    AVPacket *packet = av_packet_alloc();
    while (av_read_frame(format, packet) >= 0) {
        if (packet->stream_index == video) {
            if (packets->count == packets->capacity) {
                packets->capacity = packets->capacity ? 2 * packets->capacity : 256;
                packets->items = realloc(packets->items, packets->capacity * sizeof *packets->items);
                if (!packets->items)
                    die("out of memory", 0);
            }
            packets->items[packets->count++] = av_packet_clone(packet);
        }
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
    avformat_close_input(&format);
}

static AVCodecContext *open_decoder(const Packets *packets, const char *name)
{
    const AVCodec *codec = name ? avcodec_find_decoder_by_name(name)
                                : avcodec_find_decoder(packets->params->codec_id);
    if (!codec)
        die("no such decoder", 0);
    AVCodecContext *context = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(context, packets->params);
    context->thread_count = 1;
    int err = avcodec_open2(context, codec, NULL);
    if (err < 0)
        die("cannot open decoder", err);
    return context;
}

/* Receive every frame the decoder has ready; returns how many. */
static long drain(AVCodecContext *context, AVFrame *frame)
{
    long frames = 0;
    int err;
    while ((err = avcodec_receive_frame(context, frame)) >= 0) {
        frames++;
        av_frame_unref(frame);
    }
    if (err != AVERROR(EAGAIN) && err != AVERROR_EOF)
        die("decode error", err);
    return frames;
}

/* One timed decode of every packet; the frame count through *frames. */
static double decode_all(AVCodecContext *context, const Packets *packets, long *frames)
{
    AVFrame *frame = av_frame_alloc();
    long shown = 0;
    double start = now_seconds();
    for (size_t i = 0; i < packets->count; i++) {
        int err = avcodec_send_packet(context, packets->items[i]);
        if (err < 0)
            die("decode error", err);
        shown += drain(context, frame);
    }
    avcodec_send_packet(context, NULL);
    shown += drain(context, frame);
    double elapsed = now_seconds() - start;
    av_frame_free(&frame);
    *frames = shown;
    return elapsed;
}

int main(int argc, char **argv)
{
    const char *decoder = NULL, *path = NULL;
    int repeats = DEFAULT_REPEATS;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--decoder") && i + 1 < argc)
            decoder = argv[++i];
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc)
            repeats = atoi(argv[++i]);
        else if (!path && argv[i][0] != '-')
            path = argv[i];
        else
            die("usage: libavcodec-bench [--decoder NAME] [--repeat N] STREAM", 0);
    }
    if (!path || repeats < 1)
        die("usage: libavcodec-bench [--decoder NAME] [--repeat N] STREAM", 0);
    av_log_set_level(AV_LOG_ERROR);
    Packets packets = {0};
    read_packets(path, &packets);
    for (int repeat = 0; repeat < repeats; repeat++) {
        AVCodecContext *context = open_decoder(&packets, decoder);
        long frames = 0;
        double seconds = decode_all(context, &packets, &frames);
        avcodec_free_context(&context);
        printf("{\"repeat\": %d, \"frames\": %ld, \"seconds\": %.9f}\n", repeat, frames, seconds);
    }
    return 0;
}
