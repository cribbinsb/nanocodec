/*
 * openh264-harness: drive the openh264 encoder with explicit, fully specified
 * parameters, for use as an evaluation anchor (ENCODER_PLAN.md §4.2.1).
 *
 * The stock h264enc CLI cannot set every field we need (complexity, entropy
 * mode, fixed QP with temporal layers, detection features), so this program
 * maps command-line flags one-to-one onto SEncParamExt.
 *
 * Input: raw I420 YUV. Output: an Annex-B H.264 elementary stream.
 * Every input frame must produce an output frame (frame skip is off); a
 * skipped frame is an error, because metrics compare frame by frame.
 *
 * openh264 is BSD-2 (Cisco). This file only uses its public API.
 */

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <wels/codec_api.h>

typedef struct HarnessOptions {
    const char *input_path;
    const char *output_path;
    int width;
    int height;
    float fps;
    int frames;
    EUsageType usage;
    ECOMPLEXITY_MODE complexity;
    bool cabac;
    int qp;
    int bitrate_kbps; /* > 0: openh264's own bitrate control; --qp is then unused */
    int intra_period;
    int temporal_layers;
    bool adaptive_quant;
    bool scene_change;
    bool background_detection;
    bool denoise;
    bool long_term_reference;
    bool frame_skip;
    int threads;
} HarnessOptions;

static void usage_and_exit(const char *message)
{
    if (message)
        fprintf(stderr, "openh264-harness: %s\n", message);
    fprintf(stderr,
            "usage: openh264-harness --input in.yuv --output out.264\n"
            "  --width W --height H --fps F --frames N (--qp Q | --bitrate kbps)\n"
            "  --intra-period N\n"
            "  [--usage camera-realtime] [--complexity low|medium|high] [--cabac 0|1]\n"
            "  [--temporal-layers 1-4] [--adaptive-quant 0|1] [--scene-change 0|1]\n"
            "  [--background-detection 0|1] [--denoise 0|1] [--ltr 0|1]\n"
            "  [--frame-skip 0|1] [--threads N]\n");
    exit(2);
}

static int parse_int(const char *flag, const char *value, int min, int max)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(value, &end, 10);
    if (errno || *end != '\0' || v < min || v > max) {
        fprintf(stderr, "openh264-harness: bad value '%s' for %s (expected %d..%d)\n",
                value, flag, min, max);
        exit(2);
    }
    return (int)v;
}

static bool parse_bool(const char *flag, const char *value)
{
    return parse_int(flag, value, 0, 1) != 0;
}

static ECOMPLEXITY_MODE parse_complexity(const char *value)
{
    if (!strcmp(value, "low"))
        return LOW_COMPLEXITY;
    if (!strcmp(value, "medium"))
        return MEDIUM_COMPLEXITY;
    if (!strcmp(value, "high"))
        return HIGH_COMPLEXITY;
    usage_and_exit("--complexity must be low, medium or high");
    return LOW_COMPLEXITY;
}

static EUsageType parse_usage(const char *value)
{
    if (!strcmp(value, "camera-realtime"))
        return CAMERA_VIDEO_REAL_TIME;
    usage_and_exit("--usage: only camera-realtime is supported");
    return CAMERA_VIDEO_REAL_TIME;
}

static HarnessOptions parse_options(int argc, char **argv)
{
    HarnessOptions o = {
        .usage = CAMERA_VIDEO_REAL_TIME,
        .complexity = MEDIUM_COMPLEXITY,
        .temporal_layers = 1,
        .background_detection = true,
        .threads = 1,
        .qp = -1,
    };
    for (int i = 1; i < argc; i += 2) {
        const char *flag = argv[i];
        if (i + 1 >= argc)
            usage_and_exit("missing value after last flag");
        const char *v = argv[i + 1];

        if (!strcmp(flag, "--input"))                     o.input_path = v;
        else if (!strcmp(flag, "--output"))               o.output_path = v;
        else if (!strcmp(flag, "--width"))                o.width = parse_int(flag, v, 16, 8192);
        else if (!strcmp(flag, "--height"))               o.height = parse_int(flag, v, 16, 8192);
        else if (!strcmp(flag, "--fps"))                  o.fps = strtof(v, NULL);
        else if (!strcmp(flag, "--frames"))               o.frames = parse_int(flag, v, 1, 1 << 24);
        else if (!strcmp(flag, "--usage"))                o.usage = parse_usage(v);
        else if (!strcmp(flag, "--complexity"))           o.complexity = parse_complexity(v);
        else if (!strcmp(flag, "--cabac"))                o.cabac = parse_bool(flag, v);
        else if (!strcmp(flag, "--qp"))                   o.qp = parse_int(flag, v, 0, 51);
        else if (!strcmp(flag, "--bitrate"))              o.bitrate_kbps = parse_int(flag, v, 1, 1 << 20);
        else if (!strcmp(flag, "--intra-period"))         o.intra_period = parse_int(flag, v, 1, 1 << 16);
        else if (!strcmp(flag, "--temporal-layers"))      o.temporal_layers = parse_int(flag, v, 1, 4);
        else if (!strcmp(flag, "--adaptive-quant"))       o.adaptive_quant = parse_bool(flag, v);
        else if (!strcmp(flag, "--scene-change"))         o.scene_change = parse_bool(flag, v);
        else if (!strcmp(flag, "--background-detection")) o.background_detection = parse_bool(flag, v);
        else if (!strcmp(flag, "--denoise"))              o.denoise = parse_bool(flag, v);
        else if (!strcmp(flag, "--ltr"))                  o.long_term_reference = parse_bool(flag, v);
        else if (!strcmp(flag, "--frame-skip"))           o.frame_skip = parse_bool(flag, v);
        else if (!strcmp(flag, "--threads"))              o.threads = parse_int(flag, v, 1, 64);
        else {
            fprintf(stderr, "openh264-harness: unknown flag %s\n", flag);
            usage_and_exit(NULL);
        }
    }
    if (!o.input_path || !o.output_path || !o.width || !o.height || o.fps <= 0.0f ||
        !o.frames || (o.qp < 0) == (o.bitrate_kbps == 0) || !o.intra_period)
        usage_and_exit("missing required option");
    if (o.width % 2 || o.height % 2)
        usage_and_exit("width and height must be even (I420)");
    return o;
}

/* Translate options into openh264's parameter block. Without --bitrate, RC is
 * off and QP is fixed; with it, openh264's realtime bitrate mode runs over
 * its full QP range (frame skip still follows --frame-skip). */
static void fill_params(const HarnessOptions *o, SEncParamExt *p)
{
    p->iUsageType = o->usage;
    p->iPicWidth = o->width;
    p->iPicHeight = o->height;
    p->fMaxFrameRate = o->fps;
    bool rate_control = o->bitrate_kbps > 0;
    int bitrate_bps = o->bitrate_kbps * 1000;
    p->iRCMode = rate_control ? RC_BITRATE_MODE : RC_OFF_MODE;
    p->iTargetBitrate = rate_control ? bitrate_bps : 0;
    /* No separate peak cap: the comparison scores buffer behaviour itself. */
    p->iMaxBitrate = UNSPECIFIED_BIT_RATE;
    /* Full H.264 QP range for rate control. Not 0: a zero bound makes
     * openh264 replace the range with 12..42 (encoder_ext.cpp,
     * MAX_LOW_BR_QP), which caps how far it can cut the rate. */
    p->iMinQp = rate_control ? 1 : o->qp;
    p->iMaxQp = rate_control ? 51 : o->qp;
    p->iComplexityMode = o->complexity;
    p->iEntropyCodingModeFlag = o->cabac ? 1 : 0;
    p->uiIntraPeriod = (unsigned int)o->intra_period;
    p->iTemporalLayerNum = o->temporal_layers;
    p->iSpatialLayerNum = 1;
    p->bEnableAdaptiveQuant = o->adaptive_quant;
    p->bEnableSceneChangeDetect = o->scene_change;
    p->bEnableBackgroundDetection = o->background_detection;
    p->bEnableDenoise = o->denoise;
    p->bEnableLongTermReference = o->long_term_reference;
    p->bEnableFrameSkip = o->frame_skip;
    p->iMultipleThreadIdc = (unsigned short)o->threads;
    /* Keep SPS/PPS ids constant across IDRs, as in the product (CMAF). */
    p->eSpsPpsIdStrategy = CONSTANT_ID;

    SSpatialLayerConfig *layer = &p->sSpatialLayers[0];
    layer->iVideoWidth = o->width;
    layer->iVideoHeight = o->height;
    layer->fFrameRate = o->fps;
    layer->iSpatialBitrate = rate_control ? bitrate_bps : 0;
    layer->iMaxSpatialBitrate = UNSPECIFIED_BIT_RATE;
    if (!rate_control)
        layer->iDLayerQp = o->qp;
    layer->sSliceArgument.uiSliceMode = SM_SINGLE_SLICE;
}

/* Write every NAL of every layer of one encoded frame. Returns bytes written. */
static size_t write_frame(FILE *out, const SFrameBSInfo *info)
{
    size_t total = 0;
    for (int l = 0; l < info->iLayerNum; ++l) {
        const SLayerBSInfo *layer = &info->sLayerInfo[l];
        size_t layer_bytes = 0;
        for (int n = 0; n < layer->iNalCount; ++n)
            layer_bytes += (size_t)layer->pNalLengthInByte[n];
        if (fwrite(layer->pBsBuf, 1, layer_bytes, out) != layer_bytes) {
            perror("openh264-harness: write");
            exit(1);
        }
        total += layer_bytes;
    }
    return total;
}

int main(int argc, char **argv)
{
    const HarnessOptions o = parse_options(argc, argv);

    FILE *in = fopen(o.input_path, "rb");
    FILE *out = fopen(o.output_path, "wb");
    if (!in || !out) {
        perror("openh264-harness: open");
        return 1;
    }

    ISVCEncoder *encoder = NULL;
    if (WelsCreateSVCEncoder(&encoder) || !encoder) {
        fprintf(stderr, "openh264-harness: cannot create encoder\n");
        return 1;
    }
    SEncParamExt params;
    (*encoder)->GetDefaultParams(encoder, &params);
    fill_params(&o, &params);
    if ((*encoder)->InitializeExt(encoder, &params) != cmResultSuccess) {
        fprintf(stderr, "openh264-harness: InitializeExt rejected the parameters\n");
        return 1;
    }

    const size_t luma = (size_t)o.width * o.height;
    const size_t frame_bytes = luma * 3 / 2;
    uint8_t *frame = malloc(frame_bytes);
    if (!frame) {
        fprintf(stderr, "openh264-harness: out of memory\n");
        return 1;
    }

    SSourcePicture pic = {
        .iColorFormat = videoFormatI420,
        .iPicWidth = o.width,
        .iPicHeight = o.height,
        .iStride = { o.width, o.width / 2, o.width / 2 },
        .pData = { frame, frame + luma, frame + luma + luma / 4 },
    };

    int status = 0;
    for (int i = 0; i < o.frames; ++i) {
        if (fread(frame, 1, frame_bytes, in) != frame_bytes) {
            fprintf(stderr, "openh264-harness: input ended at frame %d of %d\n", i, o.frames);
            status = 1;
            break;
        }
        pic.uiTimeStamp = (long long)(i * 1000.0 / o.fps);
        SFrameBSInfo info;
        memset(&info, 0, sizeof(info));
        if ((*encoder)->EncodeFrame(encoder, &pic, &info) != cmResultSuccess) {
            fprintf(stderr, "openh264-harness: EncodeFrame failed at frame %d\n", i);
            status = 1;
            break;
        }
        if (info.eFrameType == videoFrameTypeSkip) {
            fprintf(stderr, "openh264-harness: frame %d was skipped; every frame must be "
                            "coded for frame-by-frame metrics\n", i);
            status = 1;
            break;
        }
        write_frame(out, &info);
    }

    (*encoder)->Uninitialize(encoder);
    WelsDestroySVCEncoder(encoder);
    free(frame);
    fclose(in);
    if (fclose(out) != 0) {
        perror("openh264-harness: close output");
        status = 1;
    }
    return status;
}
