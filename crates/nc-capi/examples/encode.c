/*
 * Encode a raw NV12 or I420 file with the nanocodec C API.
 *
 *   encode CODEC FORMAT WIDTH HEIGHT FRAMES KBPS IN OUT
 *
 * CODEC: h264 | hevc | av1; FORMAT: nv12 | i420; KBPS 0 = fixed quantizer.
 * Writes the packets back to back (Annex-B for H.264/HEVC, raw OBUs for
 * AV1) and stamps frames at 30 fps. A live application would also call
 * nc_encoder_request_keyframe, nc_encoder_set_bitrate and
 * nc_encoder_set_intra_refresh between frames. Under rate control it holds
 * frame 0 back one frame for the start-up lookahead, as the CLI does.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanocodec.h"

static int fail(const char *what, nc_status status) {
    fprintf(stderr, "%s failed: %d\n", what, (int)status);
    return 1;
}

/* Read one packed frame of `size` bytes; 0 at the end of the input. */
static int read_frame(FILE *in, uint8_t *buffer, size_t size) {
    return fread(buffer, 1, size, in) == size;
}

/* Describe packed frame `index` in `buffer`, stamped at 30 fps. */
static nc_frame wrap_frame(uint8_t *buffer, int nv12, uint32_t width, uint32_t height, int index) {
    size_t luma = (size_t)width * height, chroma = luma / 4;
    nc_frame frame = {0};
    frame.format = nv12 ? NC_FORMAT_NV12 : NC_FORMAT_I420;
    frame.planes[0] = buffer;
    frame.strides[0] = width;
    frame.planes[1] = buffer + luma;
    frame.strides[1] = nv12 ? width : width / 2;
    frame.planes[2] = buffer + luma + chroma;
    frame.strides[2] = width / 2;
    frame.timestamp_us = (int64_t)index * 1000000 / 30;
    frame.has_timestamp = 1;
    return frame;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: encode h264|hevc|av1 nv12|i420 W H FRAMES KBPS IN OUT\n");
        return 2;
    }
    uint32_t codec = strcmp(argv[1], "av1") == 0    ? NC_CODEC_AV1
                     : strcmp(argv[1], "hevc") == 0 ? NC_CODEC_HEVC
                                                    : NC_CODEC_H264;
    int nv12 = strcmp(argv[2], "nv12") == 0;
    uint32_t width = (uint32_t)atoi(argv[3]), height = (uint32_t)atoi(argv[4]);
    int frames = atoi(argv[5]);
    uint32_t kbps = (uint32_t)atoi(argv[6]);

    nc_encoder_config config;
    nc_status status = nc_encoder_config_default(&config, codec, width, height);
    if (status != NC_OK) return fail("config", status);
    config.target_kbps = kbps;

    nc_encoder *encoder = NULL;
    if ((status = nc_encoder_create(&config, NULL, 0, &encoder)) != NC_OK)
        return fail("create", status);

    FILE *in = fopen(argv[7], "rb"), *out = fopen(argv[8], "wb");
    if (!in || !out) return 1;
    size_t luma = (size_t)width * height, chroma = luma / 4, size = luma + 2 * chroma;
    /* Two buffers: the frame being encoded and, at start-up, the next one. */
    uint8_t *buffers[2] = {malloc(size), malloc(size)};

    int have = read_frame(in, buffers[0], size), next_read = 0;
    /* Start-up lookahead: hold frame 0 until frame 1 is available. */
    uint8_t wants = 0;
    if ((status = nc_encoder_wants_look_ahead(encoder, &wants)) != NC_OK)
        return fail("wants_look_ahead", status);
    if (have && wants && frames > 1 && read_frame(in, buffers[1], size)) {
        nc_frame next = wrap_frame(buffers[1], nv12, width, height, 1);
        if ((status = nc_encoder_look_ahead(encoder, &next)) != NC_OK)
            return fail("look_ahead", status);
        next_read = 1;
    }
    for (int i = 0; i < frames && have; i++) {
        nc_frame frame = wrap_frame(buffers[i % 2], nv12, width, height, i);
        nc_packet packet;
        if ((status = nc_encoder_encode(encoder, &frame, &packet)) != NC_OK)
            return fail("encode", status);
        if (packet.size > 0) fwrite(packet.data, 1, packet.size, out);
        have = next_read || read_frame(in, buffers[(i + 1) % 2], size);
        next_read = 0;
    }
    nc_packet tail;
    nc_encoder_flush(encoder, &tail);
    nc_encoder_destroy(encoder);
    free(buffers[0]);
    free(buffers[1]);
    fclose(in);
    fclose(out);
    return 0;
}
