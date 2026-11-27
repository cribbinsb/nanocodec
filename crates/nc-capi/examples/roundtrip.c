/*
 * Minimal encode and decode loop with the nanocodec C API: code a moving
 * synthetic picture at 300 kbps, decode every packet with our decoder, and
 * report the decoded pictures' mean luma PSNR against the source.
 *
 *   roundtrip [h264|hevc|av1|vp9] [rtp]      (default h264)
 *
 * With `rtp`, every packet travels as RTP packets (nc_rtp_sender) through
 * a receiver (nc_rtp_receiver) before it is decoded.
 *
 * Exits non-zero if a call fails or a packet does not decode. The project
 * site's C API page shows this file; tools/ci/capi_check.sh builds and runs it.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanocodec.h"

enum { WIDTH = 320, HEIGHT = 180, FRAMES = 30, KBPS = 300 };

/* Fill an I420 frame: a diagonal gradient that moves two pixels a frame. */
static void draw(uint8_t *y, uint8_t *u, uint8_t *v, int index) {
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
            y[row * WIDTH + col] = (uint8_t)(16 + (col + row + 2 * index) % 220);
    memset(u, 128, WIDTH / 2 * HEIGHT / 2);
    memset(v, 128, WIDTH / 2 * HEIGHT / 2);
}

/* Luma PSNR of a decoded picture against the source plane. */
static double luma_psnr(const nc_picture *picture, const uint8_t *source) {
    double sse = 0;
    for (uint32_t row = 0; row < picture->height; row++)
        for (uint32_t col = 0; col < picture->width; col++) {
            int d = picture->planes[0][row * picture->strides[0] + col] - source[row * WIDTH + col];
            sse += d * d;
        }
    return sse == 0 ? 99.0 : 10 * log10(255.0 * 255.0 * WIDTH * HEIGHT / sse);
}

static uint32_t codec_named(const char *name) {
    if (strcmp(name, "hevc") == 0) return NC_CODEC_HEVC;
    if (strcmp(name, "av1") == 0) return NC_CODEC_AV1;
    if (strcmp(name, "vp9") == 0) return NC_CODEC_VP9;
    return NC_CODEC_H264;
}

int main(int argc, char **argv) {
    uint32_t codec = codec_named(argc > 1 ? argv[1] : "h264");

    nc_encoder_config config;
    nc_encoder *encoder = NULL;
    nc_decoder *decoder = NULL;
    if (nc_encoder_config_default(&config, codec, WIDTH, HEIGHT) != NC_OK) return 1;
    config.target_kbps = KBPS; /* 0 would code at the fixed config.quantizer */
    if (nc_encoder_create(&config, NULL, 0, &encoder) != NC_OK) return 1;
    if (nc_decoder_create(codec, &decoder) != NC_OK) return 1;

    int over_rtp = argc > 2 && strcmp(argv[2], "rtp") == 0;
    nc_rtp_sender *sender = NULL;
    nc_rtp_receiver *receiver = NULL;
    if (over_rtp) {
        /* AV1 carries the Dependency Descriptor as header extension 1. */
        uint8_t descriptor_id = codec == NC_CODEC_AV1 ? 1 : 0;
        nc_rtp_sender_config rtp = {codec, 96, descriptor_id, 0x1234u, 1200};
        if (nc_rtp_sender_create(&rtp, &sender) != NC_OK) return 1;
        if (nc_rtp_receiver_create(codec, descriptor_id, 0, &receiver) != NC_OK) return 1;
    }

    static uint8_t y[WIDTH * HEIGHT], u[WIDTH * HEIGHT / 4], v[WIDTH * HEIGHT / 4];
    nc_frame frame = {0};
    frame.format = NC_FORMAT_I420;
    frame.planes[0] = y, frame.planes[1] = u, frame.planes[2] = v;
    frame.strides[0] = WIDTH, frame.strides[1] = WIDTH / 2, frame.strides[2] = WIDTH / 2;
    frame.has_timestamp = 1;

    size_t bytes = 0;
    int decoded = 0;
    double psnr = 0;
    for (int i = 0; i < FRAMES; i++) {
        draw(y, u, v, i);
        frame.timestamp_us = (int64_t)i * 1000000 / 30;
        nc_packet packet;
        if (nc_encoder_encode(encoder, &frame, &packet) != NC_OK) return 1;
        if (packet.size == 0) continue; /* dropped or held: nothing to send */
        bytes += packet.size;

        const uint8_t *unit = packet.data;
        size_t unit_size = packet.size;
        if (over_rtp) {
            /* Sender: RTP packets of at most 1200 bytes; receiver: the unit back. */
            size_t count = 0;
            if (nc_rtp_sender_packetize(sender, encoder, &count) != NC_OK) return 1;
            for (size_t k = 0; k < count; k++) {
                const uint8_t *datagram;
                size_t datagram_size;
                if (nc_rtp_sender_packet(sender, k, &datagram, &datagram_size) != NC_OK) return 1;
                if (nc_rtp_receiver_push(receiver, datagram, datagram_size) != NC_OK) return 1;
            }
            /* This loop hands over every packet of the frame at once, so it
             * releases the frame now; a network receiver pops whenever a
             * unit is ready instead. */
            if (nc_rtp_receiver_flush(receiver) != NC_OK) return 1;
            if (nc_rtp_receiver_pop(receiver, &unit, &unit_size, NULL, NULL) != NC_OK) return 1;
            if (unit_size == 0) return 1;
        }

        /* The receiving side: one outcome per packet. */
        nc_decode_result result;
        if (nc_decoder_decode(decoder, unit, unit_size, packet.index, NULL, 0, &result) != NC_OK)
            return 1;
        if (result.outcome != NC_DECODED) {
            fprintf(stderr, "frame %d frozen (outcome %u)\n", i, result.outcome);
            return 1;
        }
        nc_picture picture;
        if (nc_decoder_picture(decoder, &picture) != NC_OK) return 1;
        psnr += luma_psnr(&picture, y);
        decoded++;
        nc_encoder_acknowledge(encoder, packet.index); /* receiver feedback */
    }
    nc_rtp_receiver_destroy(receiver);
    nc_rtp_sender_destroy(sender);
    nc_decoder_destroy(decoder);
    nc_encoder_destroy(encoder);
    printf("%d of %d frames decoded, %zu bytes, mean luma PSNR %.1f dB\n", decoded, FRAMES, bytes,
           decoded ? psnr / decoded : 0.0);
    return decoded > 0 ? 0 : 1;
}
