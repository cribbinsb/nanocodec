# Foreign-encoder check (2026-11-27)

nanocodec 73d931d14; 423 streams; per-frame MD5 against the reference decoders (tools/ci/foreign_encoder_check.py).

## H.264

78 pass, 3 rejected

| encoder | setting | size | bits | verdict | detail |
|---|---|---|---|---|---|
| x264 | preset ultrafast | 416x240 | 8 | pass |  |
| x264 | preset ultrafast | 202x98 | 8 | pass |  |
| x264 | preset superfast | 416x240 | 8 | pass |  |
| x264 | preset superfast | 202x98 | 8 | pass |  |
| x264 | preset veryfast | 416x240 | 8 | pass |  |
| x264 | preset veryfast | 202x98 | 8 | pass |  |
| x264 | preset faster | 416x240 | 8 | pass |  |
| x264 | preset faster | 202x98 | 8 | pass |  |
| x264 | preset fast | 416x240 | 8 | pass |  |
| x264 | preset fast | 202x98 | 8 | pass |  |
| x264 | preset medium | 416x240 | 8 | pass |  |
| x264 | preset medium | 202x98 | 8 | pass |  |
| x264 | preset slow | 416x240 | 8 | pass |  |
| x264 | preset slow | 202x98 | 8 | pass |  |
| x264 | preset slower | 416x240 | 8 | pass |  |
| x264 | preset slower | 202x98 | 8 | pass |  |
| x264 | preset veryslow | 416x240 | 8 | pass |  |
| x264 | preset veryslow | 202x98 | 8 | pass |  |
| x264 | tune film | 416x240 | 8 | pass |  |
| x264 | tune film | 202x98 | 8 | pass |  |
| x264 | tune zerolatency | 416x240 | 8 | pass |  |
| x264 | tune zerolatency | 202x98 | 8 | pass |  |
| x264 | tune animation | 416x240 | 8 | pass |  |
| x264 | tune animation | 202x98 | 8 | pass |  |
| x264 | bframes 0 | 416x240 | 8 | pass |  |
| x264 | bframes 0 | 202x98 | 8 | pass |  |
| x264 | bframes 3 b-pyramid none | 416x240 | 8 | pass |  |
| x264 | bframes 3 b-pyramid none | 202x98 | 8 | pass |  |
| x264 | bframes 8 b-pyramid strict | 416x240 | 8 | pass |  |
| x264 | bframes 8 b-pyramid strict | 202x98 | 8 | pass |  |
| x264 | weightp 1 | 416x240 | 8 | pass |  |
| x264 | weightp 1 | 202x98 | 8 | pass |  |
| x264 | weightp 0 no-weightb | 416x240 | 8 | pass |  |
| x264 | weightp 0 no-weightb | 202x98 | 8 | pass |  |
| x264 | cqm jvt | 416x240 | 8 | pass |  |
| x264 | cqm jvt | 202x98 | 8 | pass |  |
| x264 | no-8x8dct | 416x240 | 8 | pass |  |
| x264 | no-8x8dct | 202x98 | 8 | pass |  |
| x264 | cavlc | 416x240 | 8 | pass |  |
| x264 | cavlc | 202x98 | 8 | pass |  |
| x264 | profile baseline | 416x240 | 8 | pass |  |
| x264 | profile baseline | 202x98 | 8 | pass |  |
| x264 | profile main | 416x240 | 8 | pass |  |
| x264 | profile main | 202x98 | 8 | pass |  |
| x264 | slices 4 | 416x240 | 8 | pass |  |
| x264 | slices 4 | 202x98 | 8 | pass |  |
| x264 | ref 16 | 416x240 | 8 | pass |  |
| x264 | ref 16 | 202x98 | 8 | pass |  |
| x264 | keyint 10 open-gop | 416x240 | 8 | pass |  |
| x264 | keyint 10 open-gop | 202x98 | 8 | pass |  |
| x264 | intra only | 416x240 | 8 | pass |  |
| x264 | intra only | 202x98 | 8 | pass |  |
| x264 | aq-mode 3 | 416x240 | 8 | pass |  |
| x264 | aq-mode 3 | 202x98 | 8 | pass |  |
| x264 | no-deblock | 416x240 | 8 | pass |  |
| x264 | no-deblock | 202x98 | 8 | pass |  |
| x264 | nal-hrd cbr | 416x240 | 8 | pass |  |
| x264 | nal-hrd cbr | 202x98 | 8 | pass |  |
| x264 | lossless qp 0 | 416x240 | 8 | rejected | H264 lossless transform bypass |
| x264 | lossless qp 0 | 202x98 | 8 | rejected | H264 lossless transform bypass |
| x264 | preset placebo | 202x98 | 8 | pass |  |
| x264 | preset medium | 416x240 | 10 | pass |  |
| x264 | preset medium | 202x98 | 10 | pass |  |
| x264 | preset ultrafast | 416x240 | 10 | pass |  |
| x264 | preset ultrafast | 202x98 | 10 | pass |  |
| x264 | preset veryslow | 416x240 | 10 | pass |  |
| x264 | preset veryslow | 202x98 | 10 | pass |  |
| ffmpeg-libx264 | defaults | 416x240 | 8 | pass |  |
| ffmpeg-libx264 | defaults | 202x98 | 8 | pass |  |
| ffmpeg-libx264 | defaults | 416x240 | 10 | pass |  |
| ffmpeg-libx264 | defaults | 202x98 | 10 | pass |  |
| ffmpeg-libx264 | interlaced tff (MBAFF) | 416x240 | 8 | rejected | H264 MBAFF frames (known: interlaced is not decoded yet) |
| ffmpeg-libx264 | fake-interlaced | 416x240 | 8 | pass |  (known: interlaced is not decoded yet) |
| openh264 | defaults | 416x240 | 8 | pass |  |
| openh264 | defaults | 202x98 | 8 | pass |  |
| openh264 | rc off | 416x240 | 8 | pass |  |
| openh264 | rc off | 202x98 | 8 | pass |  |
| openh264 | temporal layers 3 | 416x240 | 8 | pass |  |
| openh264 | temporal layers 3 | 202x98 | 8 | pass |  |
| openh264 | slices 4 | 416x240 | 8 | pass |  |
| openh264 | slices 4 | 202x98 | 8 | pass |  |

## HEVC

104 pass, 2 encoder-error

| encoder | setting | size | bits | verdict | detail |
|---|---|---|---|---|---|
| x265 | preset ultrafast | 416x240 | 8 | pass |  |
| x265 | preset ultrafast | 202x98 | 8 | pass |  |
| x265 | preset superfast | 416x240 | 8 | pass |  |
| x265 | preset superfast | 202x98 | 8 | pass |  |
| x265 | preset veryfast | 416x240 | 8 | pass |  |
| x265 | preset veryfast | 202x98 | 8 | pass |  |
| x265 | preset faster | 416x240 | 8 | pass |  |
| x265 | preset faster | 202x98 | 8 | pass |  |
| x265 | preset fast | 416x240 | 8 | pass |  |
| x265 | preset fast | 202x98 | 8 | pass |  |
| x265 | preset medium | 416x240 | 8 | pass |  |
| x265 | preset medium | 202x98 | 8 | pass |  |
| x265 | preset slow | 416x240 | 8 | pass |  |
| x265 | preset slow | 202x98 | 8 | pass |  |
| x265 | preset slower | 416x240 | 8 | pass |  |
| x265 | preset slower | 202x98 | 8 | pass |  |
| x265 | preset veryslow | 416x240 | 8 | pass |  |
| x265 | preset veryslow | 202x98 | 8 | pass |  |
| x265 | tune psnr | 416x240 | 8 | pass |  |
| x265 | tune psnr | 202x98 | 8 | pass |  |
| x265 | tune ssim | 416x240 | 8 | pass |  |
| x265 | tune ssim | 202x98 | 8 | pass |  |
| x265 | tune grain | 416x240 | 8 | pass |  |
| x265 | tune grain | 202x98 | 8 | pass |  |
| x265 | tune zerolatency | 416x240 | 8 | pass |  |
| x265 | tune zerolatency | 202x98 | 8 | pass |  |
| x265 | tune fastdecode | 416x240 | 8 | pass |  |
| x265 | tune fastdecode | 202x98 | 8 | pass |  |
| x265 | scaling-list default | 416x240 | 8 | pass |  |
| x265 | scaling-list default | 202x98 | 8 | pass |  |
| x265 | no-wpp | 416x240 | 8 | pass |  |
| x265 | no-wpp | 202x98 | 8 | pass |  |
| x265 | slices 4 | 416x240 | 8 | pass |  |
| x265 | slices 4 | 202x98 | 8 | pass |  |
| x265 | no-sao | 416x240 | 8 | pass |  |
| x265 | no-sao | 202x98 | 8 | pass |  |
| x265 | sao-non-deblock limit-sao | 416x240 | 8 | pass |  |
| x265 | sao-non-deblock limit-sao | 202x98 | 8 | encoder-error | Command '['/home/ubuntu/nanocodec-data/scratch/foreign-enc/tools/x265', '--input', '/home/ubuntu/nanocodec-data/scratch/foreign-enc/clips2/box_202x98_8.yuv', '- |
| x265 | deblock -2:-2 | 416x240 | 8 | pass |  |
| x265 | deblock -2:-2 | 202x98 | 8 | pass |  |
| x265 | no-deblock | 416x240 | 8 | pass |  |
| x265 | no-deblock | 202x98 | 8 | encoder-error | Command '['/home/ubuntu/nanocodec-data/scratch/foreign-enc/tools/x265', '--input', '/home/ubuntu/nanocodec-data/scratch/foreign-enc/clips2/box_202x98_8.yuv', '- |
| x265 | amp rect | 416x240 | 8 | pass |  |
| x265 | amp rect | 202x98 | 8 | pass |  |
| x265 | ctu 32 | 416x240 | 8 | pass |  |
| x265 | ctu 32 | 202x98 | 8 | pass |  |
| x265 | ctu 16 | 416x240 | 8 | pass |  |
| x265 | ctu 16 | 202x98 | 8 | pass |  |
| x265 | bframes 8 b-pyramid | 416x240 | 8 | pass |  |
| x265 | bframes 8 b-pyramid | 202x98 | 8 | pass |  |
| x265 | ref 5 weightb | 416x240 | 8 | pass |  |
| x265 | ref 5 weightb | 202x98 | 8 | pass |  |
| x265 | keyint 10 (CRA, RASL) | 416x240 | 8 | pass |  |
| x265 | keyint 10 (CRA, RASL) | 202x98 | 8 | pass |  |
| x265 | closed gop keyint 10 | 416x240 | 8 | pass |  |
| x265 | closed gop keyint 10 | 202x98 | 8 | pass |  |
| x265 | intra only | 416x240 | 8 | pass |  |
| x265 | intra only | 202x98 | 8 | pass |  |
| x265 | tskip | 416x240 | 8 | pass |  |
| x265 | tskip | 202x98 | 8 | pass |  |
| x265 | constrained-intra | 416x240 | 8 | pass |  |
| x265 | constrained-intra | 202x98 | 8 | pass |  |
| x265 | no-strong-intra-smoothing | 416x240 | 8 | pass |  |
| x265 | no-strong-intra-smoothing | 202x98 | 8 | pass |  |
| x265 | rd 6 rdoq 2 | 416x240 | 8 | pass |  |
| x265 | rd 6 rdoq 2 | 202x98 | 8 | pass |  |
| x265 | lossless | 416x240 | 8 | pass |  |
| x265 | lossless | 202x98 | 8 | pass |  |
| x265 | hrd vbv | 416x240 | 8 | pass |  |
| x265 | hrd vbv | 202x98 | 8 | pass |  |
| x265 | no-sign-hide | 416x240 | 8 | pass |  |
| x265 | no-sign-hide | 202x98 | 8 | pass |  |
| x265 | tu depth 4 | 416x240 | 8 | pass |  |
| x265 | tu depth 4 | 202x98 | 8 | pass |  |
| x265 | preset placebo | 202x98 | 8 | pass |  |
| ffmpeg-libx265 | defaults | 416x240 | 8 | pass |  |
| ffmpeg-libx265 | defaults | 202x98 | 8 | pass |  |
| ffmpeg-libx265 | defaults | 416x240 | 10 | pass |  |
| ffmpeg-libx265 | defaults | 202x98 | 10 | pass |  |
| ffmpeg-libx265 | preset ultrafast | 416x240 | 10 | pass |  |
| ffmpeg-libx265 | preset ultrafast | 202x98 | 10 | pass |  |
| ffmpeg-libx265 | preset slow | 416x240 | 10 | pass |  |
| ffmpeg-libx265 | preset slow | 202x98 | 10 | pass |  |
| ffmpeg-libx265 | scaling-list default | 416x240 | 10 | pass |  |
| ffmpeg-libx265 | scaling-list default | 202x98 | 10 | pass |  |
| kvazaar | defaults | 416x240 | 8 | pass |  |
| kvazaar | defaults | 202x98 | 8 | pass |  |
| kvazaar | preset ultrafast | 416x240 | 8 | pass |  |
| kvazaar | preset ultrafast | 202x98 | 8 | pass |  |
| kvazaar | preset medium | 416x240 | 8 | pass |  |
| kvazaar | preset medium | 202x98 | 8 | pass |  |
| kvazaar | preset veryslow | 416x240 | 8 | pass |  |
| kvazaar | preset veryslow | 202x98 | 8 | pass |  |
| kvazaar | no-wpp | 416x240 | 8 | pass |  |
| kvazaar | no-wpp | 202x98 | 8 | pass |  |
| kvazaar | gop 8 | 416x240 | 8 | pass |  |
| kvazaar | gop 8 | 202x98 | 8 | pass |  |
| kvazaar | gop lp-g4d3t1 | 416x240 | 8 | pass |  |
| kvazaar | gop lp-g4d3t1 | 202x98 | 8 | pass |  |
| kvazaar | smp amp | 416x240 | 8 | pass |  |
| kvazaar | smp amp | 202x98 | 8 | pass |  |
| kvazaar | tr-skip | 416x240 | 8 | pass |  |
| kvazaar | tr-skip | 202x98 | 8 | pass |  |
| kvazaar | intra only | 416x240 | 8 | pass |  |
| kvazaar | intra only | 202x98 | 8 | pass |  |
| kvazaar | tiles 2x2 | 640x360 | 8 | pass |  |

## VVC

31 pass

| encoder | setting | size | bits | verdict | detail |
|---|---|---|---|---|---|
| vvenc | preset faster | 416x240 | 8 | pass |  |
| vvenc | preset faster | 202x98 | 8 | pass |  |
| vvenc | preset fast | 416x240 | 8 | pass |  |
| vvenc | preset fast | 202x98 | 8 | pass |  |
| vvenc | preset medium | 416x240 | 8 | pass |  |
| vvenc | preset medium | 202x98 | 8 | pass |  |
| vvenc | preset slow | 416x240 | 8 | pass |  |
| vvenc | preset slow | 202x98 | 8 | pass |  |
| vvenc | preset slower | 416x240 | 8 | pass |  |
| vvenc | preset slower | 202x98 | 8 | pass |  |
| vvenc | internal 8-bit | 416x240 | 8 | pass |  |
| vvenc | internal 8-bit | 202x98 | 8 | pass |  |
| vvenc | wpp | 416x240 | 8 | pass |  |
| vvenc | wpp | 202x98 | 8 | pass |  |
| vvenc | no alf/ccalf | 416x240 | 8 | pass |  |
| vvenc | no alf/ccalf | 202x98 | 8 | pass |  |
| vvenc | lmcs | 416x240 | 8 | pass |  |
| vvenc | lmcs | 202x98 | 8 | pass |  |
| vvenc | intra period 8 | 416x240 | 8 | pass |  |
| vvenc | intra period 8 | 202x98 | 8 | pass |  |
| vvenc | rpr on | 416x240 | 8 | pass |  |
| vvenc | rpr on | 202x98 | 8 | pass |  |
| vvenc | no dep-quant | 416x240 | 8 | pass |  |
| vvenc | no dep-quant | 202x98 | 8 | pass |  |
| vvenc | ibc on | 416x240 | 8 | pass |  |
| vvenc | ibc on | 202x98 | 8 | pass |  |
| vvenc | tiles 2x2 | 640x360 | 8 | pass |  |
| vvenc | preset faster | 416x240 | 10 | pass |  |
| vvenc | preset faster | 202x98 | 10 | pass |  |
| vvenc | preset medium | 416x240 | 10 | pass |  |
| vvenc | preset medium | 202x98 | 10 | pass |  |

## AV1

121 pass, 7 rejected

| encoder | setting | size | bits | verdict | detail |
|---|---|---|---|---|---|
| aomenc | good cpu-used 0 | 416x240 | 8 | pass |  |
| aomenc | good cpu-used 0 | 202x98 | 8 | pass |  |
| aomenc | good cpu-used 1 | 416x240 | 8 | pass |  |
| aomenc | good cpu-used 1 | 202x98 | 8 | pass |  |
| aomenc | good cpu-used 2 | 416x240 | 8 | pass |  |
| aomenc | good cpu-used 2 | 202x98 | 8 | pass |  |
| aomenc | good cpu-used 3 | 416x240 | 8 | pass |  |
| aomenc | good cpu-used 3 | 202x98 | 8 | pass |  |
| aomenc | good cpu-used 4 | 416x240 | 8 | pass |  |
| aomenc | good cpu-used 4 | 202x98 | 8 | pass |  |
| aomenc | good cpu-used 5 | 416x240 | 8 | pass |  |
| aomenc | good cpu-used 5 | 202x98 | 8 | pass |  |
| aomenc | good cpu-used 6 | 416x240 | 8 | pass |  |
| aomenc | good cpu-used 6 | 202x98 | 8 | pass |  |
| aomenc | rt cpu-used 5 | 416x240 | 8 | pass |  |
| aomenc | rt cpu-used 5 | 202x98 | 8 | pass |  |
| aomenc | rt cpu-used 6 | 416x240 | 8 | pass |  |
| aomenc | rt cpu-used 6 | 202x98 | 8 | pass |  |
| aomenc | rt cpu-used 7 | 416x240 | 8 | pass |  |
| aomenc | rt cpu-used 7 | 202x98 | 8 | pass |  |
| aomenc | rt cpu-used 8 | 416x240 | 8 | pass |  |
| aomenc | rt cpu-used 8 | 202x98 | 8 | pass |  |
| aomenc | rt cpu-used 9 | 416x240 | 8 | pass |  |
| aomenc | rt cpu-used 9 | 202x98 | 8 | pass |  |
| aomenc | rt cpu-used 10 | 416x240 | 8 | pass |  |
| aomenc | rt cpu-used 10 | 202x98 | 8 | pass |  |
| aomenc | enable-qm | 416x240 | 8 | rejected | AV1 quantizer matrices |
| aomenc | enable-qm | 202x98 | 8 | rejected | AV1 quantizer matrices |
| aomenc | deltaq-mode 1 | 416x240 | 8 | pass |  |
| aomenc | deltaq-mode 1 | 202x98 | 8 | pass |  |
| aomenc | deltaq-mode 1 delta-lf-mode 1 | 416x240 | 8 | rejected | AV1 block loop filter deltas |
| aomenc | deltaq-mode 1 delta-lf-mode 1 | 202x98 | 8 | rejected | AV1 block loop filter deltas |
| aomenc | aq-mode 1 | 416x240 | 8 | pass |  |
| aomenc | aq-mode 1 | 202x98 | 8 | pass |  |
| aomenc | aq-mode 2 | 416x240 | 8 | pass |  |
| aomenc | aq-mode 2 | 202x98 | 8 | pass |  |
| aomenc | aq-mode 3 | 416x240 | 8 | pass |  |
| aomenc | aq-mode 3 | 202x98 | 8 | pass |  |
| aomenc | rt aq-mode 3 | 416x240 | 8 | pass |  |
| aomenc | rt aq-mode 3 | 202x98 | 8 | pass |  |
| aomenc | lossless | 416x240 | 8 | pass |  |
| aomenc | lossless | 202x98 | 8 | pass |  |
| aomenc | error-resilient | 416x240 | 8 | pass |  |
| aomenc | error-resilient | 202x98 | 8 | pass |  |
| aomenc | kf-max-dist 10 | 416x240 | 8 | pass |  |
| aomenc | kf-max-dist 10 | 202x98 | 8 | pass |  |
| aomenc | lag 0 | 416x240 | 8 | pass |  |
| aomenc | lag 0 | 202x98 | 8 | pass |  |
| aomenc | superres | 416x240 | 8 | pass |  |
| aomenc | superres | 202x98 | 8 | pass |  |
| aomenc | resize | 416x240 | 8 | pass |  |
| aomenc | resize | 202x98 | 8 | pass |  |
| aomenc | screen content | 416x240 | 8 | pass |  |
| aomenc | screen content | 202x98 | 8 | pass |  |
| aomenc | film grain | 416x240 | 8 | pass |  |
| aomenc | film grain | 202x98 | 8 | pass |  |
| aomenc | end-usage cbr | 416x240 | 8 | pass |  |
| aomenc | end-usage cbr | 202x98 | 8 | pass |  |
| aomenc | good cpu-used 4 | 416x240 | 10 | pass |  |
| aomenc | good cpu-used 4 | 202x98 | 10 | pass |  |
| aomenc | rt cpu-used 8 | 416x240 | 10 | pass |  |
| aomenc | rt cpu-used 8 | 202x98 | 10 | pass |  |
| aomenc | good cpu-used 4 | 201x99 | 8 | rejected | AV1 odd frame size |
| svtav1 | preset 0 | 416x240 | 8 | pass |  |
| svtav1 | preset 0 | 202x98 | 8 | pass |  |
| svtav1 | preset 1 | 416x240 | 8 | pass |  |
| svtav1 | preset 1 | 202x98 | 8 | pass |  |
| svtav1 | preset 2 | 416x240 | 8 | pass |  |
| svtav1 | preset 2 | 202x98 | 8 | pass |  |
| svtav1 | preset 3 | 416x240 | 8 | pass |  |
| svtav1 | preset 3 | 202x98 | 8 | pass |  |
| svtav1 | preset 4 | 416x240 | 8 | pass |  |
| svtav1 | preset 4 | 202x98 | 8 | pass |  |
| svtav1 | preset 5 | 416x240 | 8 | pass |  |
| svtav1 | preset 5 | 202x98 | 8 | pass |  |
| svtav1 | preset 6 | 416x240 | 8 | pass |  |
| svtav1 | preset 6 | 202x98 | 8 | pass |  |
| svtav1 | preset 7 | 416x240 | 8 | pass |  |
| svtav1 | preset 7 | 202x98 | 8 | pass |  |
| svtav1 | preset 8 | 416x240 | 8 | pass |  |
| svtav1 | preset 8 | 202x98 | 8 | pass |  |
| svtav1 | preset 9 | 416x240 | 8 | pass |  |
| svtav1 | preset 9 | 202x98 | 8 | pass |  |
| svtav1 | preset 10 | 416x240 | 8 | pass |  |
| svtav1 | preset 10 | 202x98 | 8 | pass |  |
| svtav1 | preset 11 | 416x240 | 8 | pass |  |
| svtav1 | preset 11 | 202x98 | 8 | pass |  |
| svtav1 | preset 12 | 416x240 | 8 | pass |  |
| svtav1 | preset 12 | 202x98 | 8 | pass |  |
| svtav1 | rtc preset 8 | 416x240 | 8 | pass |  |
| svtav1 | rtc preset 8 | 202x98 | 8 | pass |  |
| svtav1 | rtc preset 10 | 416x240 | 8 | pass |  |
| svtav1 | rtc preset 10 | 202x98 | 8 | pass |  |
| svtav1 | rtc preset 12 | 416x240 | 8 | pass |  |
| svtav1 | rtc preset 12 | 202x98 | 8 | pass |  |
| svtav1 | enable-qm | 416x240 | 8 | rejected | AV1 quantizer matrices |
| svtav1 | enable-qm | 202x98 | 8 | rejected | AV1 quantizer matrices |
| svtav1 | variance boost | 416x240 | 8 | pass |  |
| svtav1 | variance boost | 202x98 | 8 | pass |  |
| svtav1 | aq-mode 1 | 416x240 | 8 | pass |  |
| svtav1 | aq-mode 1 | 202x98 | 8 | pass |  |
| svtav1 | film grain 8 | 416x240 | 8 | pass |  |
| svtav1 | film grain 8 | 202x98 | 8 | pass |  |
| svtav1 | keyint 10 | 416x240 | 8 | pass |  |
| svtav1 | keyint 10 | 202x98 | 8 | pass |  |
| svtav1 | low delay | 416x240 | 8 | pass |  |
| svtav1 | low delay | 202x98 | 8 | pass |  |
| svtav1 | screen content | 416x240 | 8 | pass |  |
| svtav1 | screen content | 202x98 | 8 | pass |  |
| svtav1 | fast-decode 1 | 416x240 | 8 | pass |  |
| svtav1 | fast-decode 1 | 202x98 | 8 | pass |  |
| svtav1 | preset 8 | 416x240 | 10 | pass |  |
| svtav1 | preset 8 | 202x98 | 10 | pass |  |
| svtav1 | preset 4 | 416x240 | 10 | pass |  |
| svtav1 | preset 4 | 202x98 | 10 | pass |  |
| aomenc | aom tiles 2x2 | 640x360 | 8 | pass |  |
| aomenc | aom rt tiles 4 cols | 640x360 | 8 | pass |  |
| svtav1 | svt tiles 2x2 | 640x360 | 8 | pass |  |
| ffmpeg-libaom | defaults | 416x240 | 8 | pass |  |
| ffmpeg-libaom | defaults | 202x98 | 8 | pass |  |
| ffmpeg-libsvtav1 | defaults | 416x240 | 8 | pass |  |
| ffmpeg-libsvtav1 | defaults | 202x98 | 8 | pass |  |
| ffmpeg-librav1e | defaults | 416x240 | 8 | pass |  |
| ffmpeg-librav1e | defaults | 202x98 | 8 | pass |  |
| ffmpeg-librav1e | speed 10 | 416x240 | 8 | pass |  |
| ffmpeg-librav1e | speed 10 | 202x98 | 8 | pass |  |
| ffmpeg-librav1e | defaults | 416x240 | 10 | pass |  |
| ffmpeg-librav1e | defaults | 202x98 | 10 | pass |  |

## VP9

64 pass

| encoder | setting | size | bits | verdict | detail |
|---|---|---|---|---|---|
| vpxenc | good cpu-used 0 | 416x240 | 8 | pass |  |
| vpxenc | good cpu-used 0 | 202x98 | 8 | pass |  |
| vpxenc | good cpu-used 1 | 416x240 | 8 | pass |  |
| vpxenc | good cpu-used 1 | 202x98 | 8 | pass |  |
| vpxenc | good cpu-used 2 | 416x240 | 8 | pass |  |
| vpxenc | good cpu-used 2 | 202x98 | 8 | pass |  |
| vpxenc | good cpu-used 3 | 416x240 | 8 | pass |  |
| vpxenc | good cpu-used 3 | 202x98 | 8 | pass |  |
| vpxenc | good cpu-used 4 | 416x240 | 8 | pass |  |
| vpxenc | good cpu-used 4 | 202x98 | 8 | pass |  |
| vpxenc | good cpu-used 5 | 416x240 | 8 | pass |  |
| vpxenc | good cpu-used 5 | 202x98 | 8 | pass |  |
| vpxenc | rt cpu-used 5 | 416x240 | 8 | pass |  |
| vpxenc | rt cpu-used 5 | 202x98 | 8 | pass |  |
| vpxenc | rt cpu-used 6 | 416x240 | 8 | pass |  |
| vpxenc | rt cpu-used 6 | 202x98 | 8 | pass |  |
| vpxenc | rt cpu-used 7 | 416x240 | 8 | pass |  |
| vpxenc | rt cpu-used 7 | 202x98 | 8 | pass |  |
| vpxenc | rt cpu-used 8 | 416x240 | 8 | pass |  |
| vpxenc | rt cpu-used 8 | 202x98 | 8 | pass |  |
| vpxenc | rt cpu-used 9 | 416x240 | 8 | pass |  |
| vpxenc | rt cpu-used 9 | 202x98 | 8 | pass |  |
| vpxenc | aq-mode 1 | 416x240 | 8 | pass |  |
| vpxenc | aq-mode 1 | 202x98 | 8 | pass |  |
| vpxenc | aq-mode 2 | 416x240 | 8 | pass |  |
| vpxenc | aq-mode 2 | 202x98 | 8 | pass |  |
| vpxenc | aq-mode 3 | 416x240 | 8 | pass |  |
| vpxenc | aq-mode 3 | 202x98 | 8 | pass |  |
| vpxenc | rt aq-mode 3 cbr | 416x240 | 8 | pass |  |
| vpxenc | rt aq-mode 3 cbr | 202x98 | 8 | pass |  |
| vpxenc | lossless | 416x240 | 8 | pass |  |
| vpxenc | lossless | 202x98 | 8 | pass |  |
| vpxenc | error-resilient | 416x240 | 8 | pass |  |
| vpxenc | error-resilient | 202x98 | 8 | pass |  |
| vpxenc | frame-parallel | 416x240 | 8 | pass |  |
| vpxenc | frame-parallel | 202x98 | 8 | pass |  |
| vpxenc | no alt-ref | 416x240 | 8 | pass |  |
| vpxenc | no alt-ref | 202x98 | 8 | pass |  |
| vpxenc | kf-max-dist 10 | 416x240 | 8 | pass |  |
| vpxenc | kf-max-dist 10 | 202x98 | 8 | pass |  |
| vpxenc | rt resize | 416x240 | 8 | pass |  |
| vpxenc | rt resize | 202x98 | 8 | pass |  |
| vpxenc | screen content | 416x240 | 8 | pass |  |
| vpxenc | screen content | 202x98 | 8 | pass |  |
| vpxenc | good cpu-used 2 | 201x99 | 8 | pass |  |
| vpxenc | tiles 2 cols | 640x360 | 8 | pass |  |
| vpxenc | tiles 2 cols 2 rows | 640x360 | 8 | pass |  |
| vpxenc | rt tiles 2 cols | 640x360 | 8 | pass |  |
| vp9svc | 2 spatial 1 temporal | 416x240 | 8 | pass |  |
| vp9svc | 2 spatial 1 temporal | 640x360 | 8 | pass |  |
| vp9svc | 2 spatial 3 temporal | 416x240 | 8 | pass |  |
| vp9svc | 2 spatial 3 temporal | 640x360 | 8 | pass |  |
| vp9svc | 3 spatial 1 temporal | 416x240 | 8 | pass |  |
| vp9svc | 3 spatial 1 temporal | 640x360 | 8 | pass |  |
| vp9svc | 1 spatial 3 temporal | 416x240 | 8 | pass |  |
| vp9svc | 1 spatial 3 temporal | 640x360 | 8 | pass |  |
| ffmpeg-libvpx-vp9 | defaults | 416x240 | 8 | pass |  |
| ffmpeg-libvpx-vp9 | defaults | 202x98 | 8 | pass |  |
| ffmpeg-libvpx-vp9 | defaults | 416x240 | 10 | pass |  |
| ffmpeg-libvpx-vp9 | defaults | 202x98 | 10 | pass |  |
| ffmpeg-libvpx-vp9 | good cpu-used 2 | 416x240 | 10 | pass |  |
| ffmpeg-libvpx-vp9 | good cpu-used 2 | 202x98 | 10 | pass |  |
| ffmpeg-libvpx-vp9 | realtime cpu-used 8 | 416x240 | 10 | pass |  |
| ffmpeg-libvpx-vp9 | realtime cpu-used 8 | 202x98 | 10 | pass |  |

## AV2

4 pass, 4 rejected, 5 crash

| encoder | setting | size | bits | verdict | detail |
|---|---|---|---|---|---|
| avmenc | defaults | 202x98 | 8 | crash | after 0/5 pictures: exit -6: thread 'main' (1095945) panicked at crates/nc-dsp/src/edge.rs:37:26: / range end index 31568 out of range for slice of length 31360 |
| avmenc | cpu-used 5 | 416x240 | 8 | pass |  |
| avmenc | cpu-used 5 | 202x98 | 8 | crash | after 0/10 pictures: exit -6: thread 'main' (1009254) panicked at crates/nc-dsp/src/edge.rs:37:26: / range end index 31568 out of range for slice of length 3136 |
| avmenc | cpu-used 9 | 416x240 | 8 | pass |  |
| avmenc | cpu-used 9 | 202x98 | 8 | crash | after 0/10 pictures: exit -6: thread 'main' (1009163) panicked at crates/nc-dsp/src/edge.rs:37:26: / range end index 31568 out of range for slice of length 3136 |
| avmenc | cpu-used 9 lag 0 | 416x240 | 8 | pass |  |
| avmenc | cpu-used 9 lag 0 | 202x98 | 8 | crash | after 0/10 pictures: exit -6: thread 'main' (1009824) panicked at crates/nc-dsp/src/block_copy.rs:205:9: / block copy rectangle too small / note: run with `RUST |
| avmenc | cpu-used 9 kf-max-dist 4 | 416x240 | 8 | pass |  |
| avmenc | cpu-used 9 kf-max-dist 4 | 202x98 | 8 | crash | after 0/10 pictures: exit -6: thread 'main' (1009408) panicked at crates/nc-dsp/src/edge.rs:37:26: / range end index 31568 out of range for slice of length 3136 |
| avmenc | cpu-used 9 | 201x99 | 8 | rejected | AV2 4:2:0 frame of odd width or height |
| avmenc | cpu-used 9 | 416x240 | 10 | rejected | AV2 bit depth other than eight |
| avmenc | cpu-used 9 | 202x98 | 10 | rejected | AV2 bit depth other than eight |
| avmenc | tiles 2x2 | 640x360 | 8 | rejected | AV2 multiple tiles |

