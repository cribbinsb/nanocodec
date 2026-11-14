/*
 * nanocodec: low-latency H.264 / HEVC / AV1 encoder and decoder, C API.
 *
 * Link with the static or shared library built from crates/nc-capi
 * (libnanocodec.a / libnanocodec.so; the static library also needs
 * -lpthread -ldl -lm). examples/encode.c shows the whole life cycle.
 *
 * Every function returns an nc_status; nothing is ever thrown or aborted
 * across this interface. An encoder is not thread-safe: use one per thread.
 *
 * This header mirrors crates/nc-capi/src/ffi.rs; tools/ci/capi_check.sh
 * compiles and runs the example against it.
 */
#ifndef NANOCODEC_H
#define NANOCODEC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nc_status {
    NC_OK = 0,
    NC_ERROR_INVALID_ARGUMENT = -1, /* null pointer, unknown enum value */
    NC_ERROR_CONFIG = -2,           /* the configuration was rejected */
    NC_ERROR_FRAME = -3,            /* frame size or layout mismatch */
    NC_ERROR_RATE_CONTROL = -4,     /* invalid bitrate settings */
    NC_ERROR_INTERNAL = -5          /* internal error: destroy the encoder */
} nc_status;

/* nc_encoder_config.codec */
#define NC_CODEC_H264 0u /* output: Annex-B access units */
#define NC_CODEC_HEVC 1u /* output: Annex-B access units */
#define NC_CODEC_AV1 2u  /* output: AV1 temporal units (OBUs) */
#define NC_CODEC_VP9 3u  /* output: VP9 frame / Annex-B superframe packets */

/* nc_frame.format: 8-bit 4:2:0 layouts */
#define NC_FORMAT_I420 0u /* planes: Y, Cb, Cr */
#define NC_FORMAT_YV12 1u /* planes: Y, Cr, Cb */
#define NC_FORMAT_NV12 2u /* planes: Y, interleaved CbCr */

typedef struct nc_encoder nc_encoder;

typedef struct nc_encoder_config {
    uint32_t codec;
    uint32_t width, height;     /* visible size, even */
    uint32_t fps_num, fps_den;  /* nominal frame rate */
    uint32_t quantizer;         /* fixed QP (AV1: qindex), or the bitrate seed */
    uint32_t intra_period;      /* frames between key frames */
    uint32_t temporal_layers;   /* 1..3 (hierarchical P) */
    uint32_t intra_refresh_period; /* frames per refresh sweep; 0 = periodic keys */
    uint32_t ltr_interval;      /* frames between long-term reference marks; 0 = none */
    uint32_t target_kbps;       /* 0 = fixed quantizer */
    uint32_t peak_kbps;         /* 0 = twice the target */
    uint32_t buffer_ms;         /* 0 = 1000 ms */
    uint8_t frame_drop;         /* drop frames rather than overflow the buffer */
    uint8_t colour[3];          /* primaries, transfer, matrix (ISO/IEC 23091-2); 0,0,0 = unsignalled */
    uint8_t full_range;         /* full-range samples */
    uint8_t timing;             /* header timing: 0 nominal, 1 constant, 2 variable rate */
    uint8_t film_grain;         /* AV1 film grain synthesis; other codecs: NC_ERROR_CONFIG */
    float film_grain_strength;  /* relative to the measured source noise; 0 = the default */
    uint8_t still_camera;       /* still-camera noise handling over the preset: noise skip and
                                   skip-aware budget (AV1 also the noise split); lowers the bitrate
                                   on still noisy scenes; off in every preset; on AV1 a long still
                                   stream may run slightly over target; nc_param entries still win */
} nc_encoder_config;

typedef struct nc_param {
    const char *name; /* as listed by `nanocodec params`, e.g. "scene.detect" */
    float value;
} nc_param;

typedef struct nc_frame {
    uint32_t format;
    const uint8_t *planes[3]; /* NV12 uses two */
    size_t strides[3];        /* bytes between rows */
    int64_t timestamp_us;     /* capture time, if has_timestamp */
    uint8_t has_timestamp;    /* budgets rate control on real frame intervals */
} nc_frame;

typedef struct nc_packet {
    const uint8_t *data; /* valid until the next call on this encoder; NULL if size 0 */
    size_t size;         /* 0: frame dropped or held, send nothing */
    uint64_t index;
    int64_t timestamp_us;
    uint8_t keyframe;
    uint8_t temporal_layer;
    uint8_t dropped;     /* rate control dropped it, or it repeats the picture */
    uint32_t quantizer;
} nc_packet;

/* Per-layer sidecar entry for spatial SVC packets (nc_encoder_packet_layers). */
typedef struct nc_layer_frame {
    uint8_t spatial_id;             /* 0 = base (S0), 1 = S1, 2 = S2 */
    uint8_t temporal_layer;         /* 0..2 */
    uint8_t droppable;              /* non-zero: droppable without breaking its spatial layer */
    uint8_t inter_layer_ref;        /* non-zero: predicts from spatial_id - 1 in this packet */
    uint32_t operating_point_mask;  /* bit i = 1 when operating point i includes this layer */
    uint32_t width, height;         /* visible size of this layer picture */
    size_t byte_offset;             /* offset in nc_packet.data */
    size_t byte_len;                /* byte length of this layer frame */
    uint32_t quantizer;
} nc_layer_frame;

/* Defaults for `codec`: 30 fps, starting quantizer, key frame every 300. */
nc_status nc_encoder_config_default(nc_encoder_config *config, uint32_t codec,
                                    uint32_t width, uint32_t height);

/* Create an encoder; `params` may be NULL when `param_count` is 0. */
nc_status nc_encoder_create(const nc_encoder_config *config, const nc_param *params,
                            size_t param_count, nc_encoder **out);

void nc_encoder_destroy(nc_encoder *encoder);

/* Encode one frame; the packet is borrowed until the next call. */
nc_status nc_encoder_encode(nc_encoder *encoder, const nc_frame *frame, nc_packet *packet);

/* Spatial-layer sidecar entries of the latest encoded packet (0 on single-layer
 * streams); borrowed until the next call on this encoder. */
nc_status nc_encoder_packet_layers(const nc_encoder *encoder,
                                   const nc_layer_frame **layers, size_t *count);

/*
 * Optional one-frame start-up lookahead. Before the first frame, if
 * nc_encoder_wants_look_ahead sets *wants to 1 (rate control on, and a codec
 * that sizes its first key from start-up camera motion), the caller may wait
 * for frame 1, pass it to nc_encoder_look_ahead, then encode frames 0 and 1
 * in order with nc_encoder_encode as usual. This costs one frame of start-up
 * latency and lets rate control give a moving scene a smaller first key
 * (about 4 dB less first-key quality jump on pans). Skipping it is always
 * valid; the encoder still never buffers frames.
 */
nc_status nc_encoder_wants_look_ahead(const nc_encoder *encoder, uint8_t *wants);
nc_status nc_encoder_look_ahead(nc_encoder *encoder, const nc_frame *next);

/* Never returns data (frames are never buffered); for callers written for buffering encoders. */
nc_status nc_encoder_flush(nc_encoder *encoder, nc_packet *packet);

nc_status nc_encoder_request_keyframe(nc_encoder *encoder);

/* Live bitrate change (also switches fixed-quantizer encoders to rate control). */
nc_status nc_encoder_set_bitrate(nc_encoder *encoder, uint32_t target_kbps, uint32_t peak_kbps);

/* Live target, peak (0 = twice the target) and buffer (ms; 0 = unchanged)
 * change from the next frame on, e.g. a new bandwidth estimate. */
nc_status nc_encoder_set_rate(nc_encoder *encoder, uint32_t target_kbps, uint32_t peak_kbps,
                              uint32_t buffer_ms);

/* Receiver feedback, by nc_packet.index: decoded, or lost (the next frame
 * then recovers from the acknowledged long-term reference, or is a key). */
nc_status nc_encoder_acknowledge(nc_encoder *encoder, uint64_t index);
nc_status nc_encoder_report_loss(nc_encoder *encoder, uint64_t index);
nc_status nc_encoder_acknowledge_layer(nc_encoder *encoder, uint64_t index, uint8_t spatial_id);
nc_status nc_encoder_report_layer_loss(nc_encoder *encoder, uint64_t index, uint8_t spatial_id);

/* Resolution change from the next frame on (a key frame with new parameter
 * sets / sequence header); later frames must have the new size. */
nc_status nc_encoder_set_resolution(nc_encoder *encoder, uint32_t width, uint32_t height);

/* Untimed frames from the next one on are den/num seconds apart (frames
 * with timestamps follow their own times; a rate change needs nothing). */
nc_status nc_encoder_set_frame_rate(nc_encoder *encoder, uint32_t num, uint32_t den);

/* Intra refresh: period >= 2 frames per sweep, 0 = off. */
nc_status nc_encoder_set_intra_refresh(nc_encoder *encoder, uint32_t period);

/*
 * Decoder (our subset decoder for our own streams). Each packet gets one
 * outcome: decoded, or frozen (the picture stays the last decoded one) with
 * the reason and the frame index to pass to nc_encoder_report_loss on the
 * sending side. A frozen packet is not an API error: the call returns NC_OK.
 * H.264, HEVC and AV1 streams reveal a missing reference themselves
 * (frame_num, POC, AV1 frame ids); a caller that has the transport's
 * dependency signal (the frames a frame predicts from) passes it too.
 */
typedef struct nc_decoder nc_decoder;

typedef enum nc_decode_outcome {
    NC_DECODED = 0,                  /* a new picture */
    NC_FROZEN_MISSING_REFERENCE = 1, /* an earlier frame was lost or refused */
    NC_FROZEN_MISSING_KEY = 2,       /* no key frame decoded yet */
    NC_FROZEN_CORRUPT = 3            /* malformed or truncated packet */
} nc_decode_outcome;

typedef struct nc_decode_result {
    uint32_t outcome;      /* nc_decode_outcome */
    uint64_t report_index; /* frozen: the frame to report lost (the missing
                              reference if `references` named it, else this one) */
} nc_decode_result;

typedef struct nc_picture {
    const uint8_t *planes[3]; /* Y, Cb, Cr (8-bit 4:2:0); valid until the next decoder call */
    size_t strides[3];
    uint32_t width, height;
} nc_picture;

nc_status nc_decoder_create(uint32_t codec, nc_decoder **out);
void nc_decoder_destroy(nc_decoder *decoder);

/* Select the operating point (0..31 on AV1, e.g. 0..8 in L3T3; 0..2 on VP9). */
nc_status nc_decoder_set_operating_point(nc_decoder *decoder, uint8_t operating_point);

/* Decode coded frame `index` (nc_packet.index); `references` may be NULL
 * when `reference_count` is 0. */
nc_status nc_decoder_decode(nc_decoder *decoder, const uint8_t *data, size_t size,
                            uint64_t index, const uint64_t *references,
                            size_t reference_count, nc_decode_result *result);

/* The picture on screen; NC_ERROR_FRAME before the first decoded one. */
nc_status nc_decoder_picture(const nc_decoder *decoder, nc_picture *picture);

/* Spatial layer id (0..2) of the picture currently on screen (0 on single-layer). */
nc_status nc_decoder_spatial_id(const nc_decoder *decoder, uint8_t *spatial_id);

#ifdef __cplusplus
}
#endif

#endif /* NANOCODEC_H */
