/*
 * decode_bench.h: the shared part of the in-process decoder harnesses
 * (vpx_bench.c, vvdec_bench.c, avm_bench.c; libavcodec_bench.c reads through
 * libavformat instead). Each harness reads its stream into memory, splits it
 * into the units its decoder takes (IVF frames, or Annex-B NAL units with their
 * start codes), all untimed, then times only its decode loop over those units
 * per repeat: the method of tools/dev/decoder_speed.py.
 *
 *   HARNESS [--repeat N] STREAM
 *
 * prints one JSON object per repeat: {"repeat", "frames", "seconds"}.
 */
#ifndef DECODE_BENCH_H
#define DECODE_BENCH_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define IVF_FILE_HEADER_BYTES 32
#define IVF_FRAME_HEADER_BYTES 12

typedef struct Unit {
    const uint8_t *data;
    size_t size;
} Unit;

typedef struct Stream {
    uint8_t *bytes;
    size_t size;
    Unit *units;
    size_t count;
} Stream;

static void bench_die(const char *what)
{
    fprintf(stderr, "decode bench: %s\n", what);
    exit(1);
}

static double bench_seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* --repeat N STREAM, in any order; N defaults to 1. */
static const char *bench_args(int argc, char **argv, int *repeats)
{
    const char *path = NULL;
    *repeats = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--repeat") && i + 1 < argc)
            *repeats = atoi(argv[++i]);
        else if (!path && argv[i][0] != '-')
            path = argv[i];
        else
            bench_die("usage: HARNESS [--repeat N] STREAM");
    }
    if (!path || *repeats < 1)
        bench_die("usage: HARNESS [--repeat N] STREAM");
    return path;
}

static void bench_read(const char *path, Stream *stream)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        bench_die("cannot open the stream");
    fseek(f, 0, SEEK_END);
    stream->size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    stream->bytes = malloc(stream->size ? stream->size : 1);
    if (!stream->bytes || fread(stream->bytes, 1, stream->size, f) != stream->size)
        bench_die("cannot read the stream");
    fclose(f);
    stream->units = malloc((stream->size / 3 + 1) * sizeof *stream->units);
    if (!stream->units)
        bench_die("out of memory");
    stream->count = 0;
}

/* The frames of an IVF file (frame header: 4-byte little-endian size, 8-byte pts). */
static void bench_split_ivf(Stream *s)
{
    if (s->size < IVF_FILE_HEADER_BYTES || memcmp(s->bytes, "DKIF", 4))
        bench_die("not an IVF file");
    size_t header = s->bytes[6] | (size_t)s->bytes[7] << 8;
    for (size_t at = header; at + IVF_FRAME_HEADER_BYTES <= s->size;) {
        const uint8_t *h = s->bytes + at;
        size_t size = h[0] | (size_t)h[1] << 8 | (size_t)h[2] << 16 | (size_t)h[3] << 24;
        at += IVF_FRAME_HEADER_BYTES;
        if (at + size > s->size)
            bench_die("truncated IVF frame");
        s->units[s->count++] = (Unit){s->bytes + at, size};
        at += size;
    }
}

/* The NAL units of an Annex-B stream, each with its start code (00 00 01 or 00 00 00 01). */
static void bench_split_annexb(Stream *s)
{
    size_t start = SIZE_MAX;
    for (size_t i = 0; i + 3 <= s->size; i++) {
        if (s->bytes[i] || s->bytes[i + 1] || s->bytes[i + 2] != 1)
            continue;
        size_t code = i > 0 && !s->bytes[i - 1] ? i - 1 : i;
        if (start != SIZE_MAX)
            s->units[s->count++] = (Unit){s->bytes + start, code - start};
        start = code;
        i += 2;
    }
    if (start == SIZE_MAX)
        bench_die("no Annex-B start code");
    s->units[s->count++] = (Unit){s->bytes + start, s->size - start};
}

static void bench_report(int repeat, long frames, double seconds)
{
    printf("{\"repeat\": %d, \"frames\": %ld, \"seconds\": %.9f}\n", repeat, frames, seconds);
    fflush(stdout);
}

#endif
