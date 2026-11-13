# BWC SOC hardware encoder: which coding tools its streams use

What a body-worn-camera SoC's hardware encoder ("BWC SOC") actually codes,
read from its streams by our instrumented decoders (`nanocodec stat`), next to
x265 / x264 medium, openh264 and our own encoder on the same input. **One clip
(netflix-food-market, 1280x720, 30 fps, 300 frames, one IDR then P frames) at
1000 / 1500 / 2000 kbps.** Nothing here generalises beyond that.

Streams: the BWC SOC streams were supplied by the owner (EVBR recording rate
control, HEVC Main / H.264 High). The x265, x264 and openh264 streams are in
`web/streams-ref` (`manifest.json`, chart CBR settings, 1 s VBV). Ours were
coded at commit `b53145b5` with `--bitrate R --buffer-ms 1000 --intra-period
300`, at Default and `--preset hevc-fast` / `h264-fast`.

Method: `nanocodec stat --codec hevc|h264 --input S --json out.json` decodes
the whole stream with analysis on. The decoder pushes one record per coding,
prediction, transform and SAO block (`nc_common::stream_stats`). The tallies
come from those records and `eval/nceval/streamstats.py` summarises them.
Shares are % of luma area unless they say "% CTB" or count. "KEY" is the IDR
frame and "P" is all 299 P frames. "Frame QP" is the area-mean coding-block
QP of a frame; the slice QP is only the starting point of the QP deltas.
Bits are arithmetic-coded bits counted at the parse points.

Cross-check: ffmpeg `-debug qp` on the BWC SOC H.264 1000k key frame gives a
mean macroblock QP of 24.1225 over 3600 macroblocks, the same as our records.
The record-based HEVC tallies match an earlier, independent counter-based
version of the instrumentation on every row.

## HEVC

| | SOC-1000 | SOC-1500 | SOC-2000 | x265m-1000 | x265m-1500 | x265m-2000 | ncDef-1000 | ncDef-1500 | ncDef-2000 | ncFast-1000 | ncFast-1500 | ncFast-2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CTB / min CB / max TB / AMP | 64/16/16/0 | 64/16/16/0 | 64/16/16/0 | 64/8/32/0 | 64/8/32/0 | 64/8/32/0 | 64/8/32/0 | 64/8/32/0 | 64/8/32/0 | 64/8/32/0 | 64/8/32/0 | 64/8/32/0 |
| max TU depth intra/inter | 1/0 | 1/0 | 1/0 | 0/0 | 0/0 | 0/0 | 2/2 | 2/2 | 2/2 | 0/2 | 0/2 | 0/2 |
| merge cands / L0 refs | 2/1 | 2/1 | 2/1 | 3/3 | 3/3 | 3/3 | 2/1 | 2/1 | 2/1 | 2/1 | 2/1 | 2/1 |
| sign hiding / tskip / dQP depth | 0/0/2 | 0/0/2 | 0/0/2 | 1/0/1 | 1/0/1 | 1/0/1 | 1/0/0 | 1/0/0 | 1/0/0 | 1/0/0 | 1/0/0 | 1/0/0 |
| key bytes / mean P bytes / P CV% | 62285/3967/13.8 | 62285/6055/14.6 | 62285/8123/15.3 | 10644/4156/36.1 | 17128/6216/38.9 | 23558/8275/39.8 | 40989/3974/75.6 | 56500/5985/75.7 | 76093/7943/73.0 | 40724/3977/64.4 | 57079/5966/64.5 | 77422/7960/64.7 |
| key QP / P QP (area-mean CB QP per frame) mean [range] sd | 24.4/38.2 [28.8, 43.9] 2.6 | 24.4/35.6 [26.0, 42.3] 2.78 | 24.4/33.7 [24.9, 40.8] 2.86 | 37.8/35.7 [33.1, 42.5] 2.54 | 34.3/33.0 [30.4, 40.4] 2.79 | 31.8/31.1 [28.3, 38.7] 2.9 | 28.0/37.3 [28.0, 48.0] 3.54 | 25.0/34.7 [24.0, 46.0] 3.83 | 22.0/32.9 [23.0, 45.0] 4.06 | 28.0/38.2 [28.0, 48.0] 3.33 | 25.0/35.5 [25.0, 46.0] 3.57 | 22.0/33.6 [22.0, 45.0] 3.78 |
| P frame QP: mean abs step / reversals >0.5 | 0.41/24 | 0.41/18 | 0.43/24 | 0.22/3 | 0.21/2 | 0.2/2 | 1.25/113 | 1.27/105 | 1.27/104 | 1.27/113 | 1.24/102 | 1.25/104 |
| P slice-QP changes (of 298) | 73 | 71 | 57 | 49 | 63 | 66 | 169 | 167 | 168 | 172 | 167 | 163 |
| KEY CB size % area | 16:69.4 32:30.6 | 16:69.4 32:30.6 | 16:69.4 32:30.6 | 8:6 16:31.3 32:62.7 | 8:11.1 16:38 32:50.9 | 8:15.3 16:45.3 32:39.4 | 8:22.1 16:42.7 32:34.3 64:0.9 | 8:28.1 16:43 32:28 64:0.9 | 8:33.8 16:40.2 32:25.6 64:0.4 | 16:100 | 16:100 | 16:100 |
| KEY intra shape % | whole:64.1 quarter:35.9 | whole:64.1 quarter:35.9 | whole:64.1 quarter:35.9 | whole:98.9 quarter:1.1 | whole:97.5 quarter:2.5 | whole:96.5 quarter:3.5 | whole:99.2 quarter:0.8 | whole:98 quarter:2 | whole:96.4 quarter:3.6 | whole:100 | whole:100 | whole:100 |
| KEY intra PB size % | 8:35.9 16:33.5 32:30.6 | 8:35.9 16:33.5 32:30.6 | 8:35.9 16:33.5 32:30.6 | 4:1.1 8:4.9 16:31.3 32:62.7 | 4:2.5 8:8.6 16:38 32:50.9 | 4:3.5 8:11.7 16:45.3 32:39.4 | 4:0.8 8:21.2 16:42.7 32:34.3 64:0.9 | 4:2 8:26.1 16:43 32:28 64:0.9 | 4:3.6 8:30.1 16:40.2 32:25.6 64:0.4 | 16:100 | 16:100 | 16:100 |
| KEY TB size % | 4:14.6 8:21.3 16:64.1 | 4:14.6 8:21.3 16:64.1 | 4:14.6 8:21.3 16:64.1 | 4:1.1 8:4.9 16:31.3 32:62.7 | 4:2.5 8:8.6 16:38 32:50.9 | 4:3.5 8:11.7 16:45.3 32:39.4 | 4:4 8:19.6 16:41.3 32:35.1 | 4:7 8:23.5 16:40.8 32:28.7 | 4:10.6 8:26.6 16:37.3 32:25.4 | 16:100 | 16:100 | 16:100 |
| KEY intra class % (planar/dc/angular) | planar:36.1 dc:20 angular:43.9 | planar:36.1 dc:20 angular:43.9 | planar:36.1 dc:20 angular:43.9 | planar:24.6 dc:15.2 angular:60.2 | planar:26.9 dc:17.2 angular:55.9 | planar:26.4 dc:18.6 angular:55 | planar:26 dc:24.3 angular:49.7 | planar:24.7 dc:22.4 angular:52.9 | planar:25.7 dc:20.9 angular:53.4 | planar:26.5 dc:21.2 angular:52.2 | planar:25.2 dc:20.8 angular:54.1 | planar:26.6 dc:19.6 angular:53.8 |
| KEY distinct luma modes by PB size | 16:35 32:21 8:35 | 16:35 32:21 8:35 | 16:35 32:21 8:35 | 16:35 32:35 4:35 8:35 | 16:35 32:35 4:35 8:35 | 16:35 32:33 4:35 8:35 | 16:35 32:30 4:35 64:2 8:35 | 16:35 32:30 4:35 64:2 8:35 | 16:35 32:27 4:35 64:1 8:35 | 16:35 | 16:35 | 16:35 |
| KEY luma TB area coded % | 89.6 | 89.6 | 89.6 | 81.0 | 83.1 | 84.1 | 87.6 | 89.0 | 90.5 | 89.4 | 91.8 | 92.8 |
| KEY CBs with nonzero dQP / coded, range | 954/2698 [-7, 7] | 954/2698 [-7, 7] | 954/2698 [-7, 7] | 451/822 [-4, 4] | 475/856 [-4, 5] | 478/868 [-4, 4] | 0/221 [0, 0] | 0/221 [0, 0] | 0/221 [0, 0] | 0/221 [0, 0] | 0/221 [0, 0] | 0/221 [0, 0] |
| KEY CB QP p10-p90 | 22-27 | 22-27 | 22-27 | 36-39 | 33-36 | 30-33 | 28-28 | 25-25 | 22-22 | 28-28 | 25-25 | 22-22 |
| P CB size % area | 16:5.3 32:90.6 64:4.1 | 16:5.7 32:91.8 64:2.5 | 16:5.9 32:92.3 64:1.8 | 8:4.9 16:21 32:46.7 64:27.4 | 8:8 16:25.7 32:46.5 64:19.8 | 8:10.8 16:28.5 32:44.8 64:15.9 | 16:49 32:39 64:12 | 16:62.2 32:30.2 64:7.6 | 16:70.2 32:24.3 64:5.6 | 16:16.2 32:73 64:10.8 | 16:18 32:75.5 64:6.5 | 16:18.7 32:76.8 64:4.6 |
| P inter (non-skip) CB size % | 16:2.8 32:97.2 | 16:2.7 32:97.3 | 16:2.5 32:97.5 | 8:3.4 16:16.2 32:47.4 64:33 | 8:6 16:21.1 32:50.2 64:22.7 | 8:8.6 16:24.1 32:49.5 64:17.8 | 16:49.8 32:37.4 64:12.8 | 16:64.9 32:27.4 64:7.7 | 16:73.2 32:21.3 64:5.5 | 16:11.6 32:79.9 64:8.4 | 16:13.2 32:81.8 64:5 | 16:13.9 32:82.6 64:3.5 |
| P skip CB size % | 16:6.6 32:83.1 64:10.3 | 16:8.3 32:83.1 64:8.6 | 16:10.2 32:82 64:7.8 | 8:5.9 16:25.5 32:45.8 64:22.8 | 8:9.5 16:31.2 32:42.1 64:17.2 | 8:12.8 16:34.5 32:38.4 64:14.3 | 16:46 32:41.9 64:12.1 | 16:57.9 32:33.9 64:8.2 | 16:65.7 32:28 64:6.3 | 16:19.8 32:65.1 64:15.1 | 16:24.6 32:65.1 64:10.3 | 16:28.1 32:64 64:8 |
| P intra/inter/skip % area | 3.2/57.1/39.8 | 3.5/67.2/29.3 | 3.8/73.3/22.9 | 1.9/50.1/48.0 | 2.4/54.5/43.1 | 2.8/57.6/39.6 | 6.9/40.7/52.4 | 7.1/45.4/47.5 | 7.4/49.3/43.3 | 5.4/57.7/36.9 | 5.3/65.1/29.6 | 5.4/69.8/24.8 |
| P inter partition % | whole:99.4 halfH:0.3 halfV:0.3 | whole:99.3 halfH:0.4 halfV:0.3 | whole:99.2 halfH:0.4 halfV:0.3 quarter:0.1 | whole:100 | whole:100 | whole:100 | whole:100 | whole:100 | whole:100 | whole:100 | whole:100 | whole:100 |
| P inter PB size % area | 8x16:0.1 16x8:0.2 16:4 32:91.4 64:4.2 | 8x16:0.2 16x8:0.2 16:3.9 32:93 64:2.6 | 8:0.1 8x16:0.2 16x8:0.3 16:3.8 32:93.8 64:1.9 | 8:4.6 16:20.8 32:46.6 64:28 | 8:7.6 16:25.5 32:46.6 64:20.3 | 8:10.3 16:28.3 32:45 64:16.4 | 16:47.6 32:39.9 64:12.4 | 16:61.3 32:30.7 64:8 | 16:69.7 32:24.5 64:5.9 | 16:14.8 32:74.2 64:11 | 16:16.7 32:76.6 64:6.7 | 16:17.6 32:77.7 64:4.7 |
| P PB signal % (inter=explicit MVD) | intra:3.2 inter:20.4 skip:39.8 merge:36.7 | intra:3.5 inter:24.7 skip:29.3 merge:42.5 | intra:3.8 inter:27 skip:22.9 merge:46.3 | intra:1.9 inter:29.1 skip:48 merge:21 | intra:2.4 inter:31.1 skip:43.1 merge:23.5 | intra:2.8 inter:32.6 skip:39.6 merge:25 | intra:6.9 inter:22.2 skip:52.4 merge:18.5 | intra:7.1 inter:25.4 skip:47.5 merge:20 | intra:7.4 inter:28.4 skip:43.3 merge:21 | intra:5.4 inter:26.9 skip:36.9 merge:30.8 | intra:5.3 inter:32.4 skip:29.6 merge:32.7 | intra:5.4 inter:36.1 skip:24.8 merge:33.6 |
| P MV int/half/quarter % | int:4.8 half:51.5 quarter:43.7 | int:4.6 half:50.8 quarter:44.5 | int:4.5 half:50.3 quarter:45.2 | int:17.1 half:16.4 quarter:66.5 | int:16.2 half:16.9 quarter:66.9 | int:16.2 half:17 quarter:66.8 | int:47.4 half:25.1 quarter:27.5 | int:41.2 half:25.6 quarter:33.2 | int:38.3 half:25.9 quarter:35.8 | int:50.5 half:29.9 quarter:19.6 | int:45.3 half:30.5 quarter:24.2 | int:41.7 half:30.5 quarter:27.8 |
| P zero MV % | 2.8 | 2.7 | 2.5 | 7.0 | 6.1 | 5.7 | 5.7 | 5.4 | 5.3 | 9.3 | 8.3 | 7.7 |
| P MV size % (0/<1px/1-2/2-4/4-8/8-16/16-32/>32) | 2.8/25.9/7.4/5.5/16.9/24.1/15.7/1.8 | 2.7/25.7/7.8/5.5/16.8/24.5/15.1/1.9 | 2.5/26.5/7.4/5.4/16.7/24.4/15.4/1.8 | 7/14.2/10/7.1/17/24.2/17.2/3.4 | 6.1/14.9/10/7.1/17.2/24.1/17.1/3.5 | 5.7/14.8/10.3/7/17.1/24.4/17.1/3.5 | 5.7/13.6/11.9/7.2/17.4/24.9/17/2.3 | 5.4/14.6/11.5/7/17.6/25/16.7/2.2 | 5.3/14.7/11.7/7/17.6/25/16.7/2.1 | 9.3/12.1/12.1/6.6/16.8/24.4/16.6/2.2 | 8.3/13.1/11.8/6.7/17/24.3/16.5/2.3 | 7.7/13.8/11.6/6.6/17.1/24.4/16.5/2.2 |
| P ref_idx % | 0:100 | 0:100 | 0:100 | 0:95.2 1:2.9 2:2 | 0:94.1 1:3.7 2:2.2 | 0:93.2 1:4.4 2:2.5 | 0:100 | 0:100 | 0:100 | 0:100 | 0:100 | 0:100 |
| P MVD size % (0/<1px/1-2px/larger) | 5.2/58.9/25/10.9 | 5.4/62.7/22.6/9.4 | 5.3/64.8/21.3/8.7 | 4.3/50.6/24.8/20.2 | 4.7/54.7/21.8/18.8 | 4.8/57.2/20.5/17.5 | 9.8/46.2/23.4/20.7 | 9.1/53.3/20.8/16.8 | 8.8/56.9/19/15.2 | 9.8/39.9/31/19.5 | 9.1/45.3/28.6/17 | 8.3/48.8/27/15.9 |
| P intra class % | planar:49.8 dc:22.6 angular:27.6 | planar:48.5 dc:22.4 angular:29.1 | planar:47.1 dc:23 angular:29.8 | planar:22 dc:9.5 angular:68.6 | planar:20 dc:10 angular:69.9 | planar:20.6 dc:9.8 angular:69.6 | planar:35.6 dc:37.9 angular:26.5 | planar:34.8 dc:39.6 angular:25.6 | planar:34.2 dc:40.7 angular:25.1 | dc:100 | dc:100 | dc:100 |
| P intra PB size % | 8:7.5 16:26.8 32:65.6 | 8:11.5 16:30.7 32:57.8 | 8:13.9 16:32.2 32:53.9 | 8:18.1 16:33.3 32:48.7 | 8:24.3 16:34 32:41.8 | 8:28.1 16:33.9 32:38 | 16:66.7 32:26.8 64:6.5 | 16:73.5 32:23.1 64:3.3 | 16:76 32:22 64:2 | 16:41.2 32:51.8 64:7 | 16:40.2 32:55.8 64:4 | 16:37.5 32:60.1 64:2.4 |
| P TB size inter % | 8:0.4 16:99.6 | 8:0.6 16:99.4 | 8:0.8 16:99.2 | 8:3 16:15.2 32:81.9 | 8:5.3 16:19.9 32:74.8 | 8:7.7 16:22.9 32:69.3 | 8:5.2 16:69.4 32:25.4 | 8:6.4 16:77.6 32:16 | 8:7.2 16:81.8 32:11 | 16:84.6 32:15.4 | 16:89.3 32:10.7 | 16:91.7 32:8.3 |
| P TB depth inter % | 0:2.1 1:97.9 | 0:1.9 1:98.1 | 0:1.7 1:98.3 | 0:64.8 1:35.2 | 0:75.6 1:24.4 | 0:81 1:19 | 0:62 1:31.7 2:6.3 | 0:70.6 1:24.8 2:4.6 | 0:75.1 1:21.3 2:3.6 | 0:23.2 1:72 2:4.8 | 0:21.8 1:74.8 2:3.5 | 0:20.7 1:76.8 2:2.6 |
| P luma area with coded residual % | 31.9 | 42.3 | 50.2 | 40.5 | 46.8 | 51.0 | 23.7 | 29.5 | 34.5 | 36.0 | 44.3 | 50.7 |
| P non-skip CBs with any / chroma residual % | 84.5/16.8 | 87.9/19.4 | 90.4/21.8 | 88.1/16.8 | 89.1/17.4 | 89.5/17.3 | 71.0/21.6 | 73.9/22.6 | 76.2/22.8 | 88.8/22.3 | 91.5/26.6 | 93.2/28.3 |
| P coded TB % Y/Cb/Cr (of TB area) | 62.6/3.7/4.6 | 68.1/4.4/5.1 | 72.0/5.1/5.7 | 87.8/6.6/8.0 | 91.9/7.7/9.0 | 93.9/8.1/9.6 | 67.0/12.4/14.0 | 73.1/13.8/15.5 | 77.3/14.4/15.8 | 63.4/8.2/8.9 | 68.2/9.9/10.5 | 71.8/10.7/11.2 |
| P nonzero levels per 1000 px | 4.4 | 7.4 | 10.5 | 3.8 | 6.2 | 8.8 | 4.6 | 7.8 | 11.2 | 4.5 | 7.6 | 10.9 |
| P dQP nonzero per frame / range | 328.6 [-12, 12] | 398.9 [-13, 11] | 444.4 [-13, 12] | 200.3 [-6, 5] | 252.8 [-6, 5] | 286.5 [-6, 6] | 0.6 [-7, 2] | 0.4 [-2, 2] | 0.4 [-2, 2] | 0.6 [-2, 3] | 1.0 [-3, 3] | 1.2 [-3, 2] |
| P CB QP p10-p90 | 34-43 | 31-40 | 29-39 | 33-41 | 30-38 | 28-37 | 35-44 | 32-41 | 30-40 | 36-44 | 33-41 | 31-40 |
| P merge_idx used | 0:61.3 1:38.7 | 0:60.2 1:39.8 | 0:59.6 1:40.4 | 0:57 1:28.6 2:14.4 | 0:56.2 1:29.6 2:14.2 | 0:55.7 1:30.2 2:14.1 | 0:62.9 1:37.1 | 0:61.9 1:38.1 | 0:61.2 1:38.8 | 0:65.7 1:34.3 | 0:65.2 1:34.8 | 0:64.8 1:35.2 |
| P SAO luma off/band/edge % CTB | 0:33.3 1:12.2 2:54.5 | 0:22.5 1:14 2:63.5 | 0:17.2 1:13.6 2:69.2 | 0:26.3 1:6.3 2:67.5 | 0:18.8 1:5 2:76.2 | 0:14.6 1:4.5 2:80.9 | 0:52.1 2:47.9 | 0:37.4 2:62.6 | 0:29.2 2:70.8 | 0:52.3 2:47.7 | 0:38.3 2:61.7 | 0:31.6 2:68.4 |
| P SAO chroma off/band/edge % CTB | 0:90.9 1:3.3 2:5.8 | 0:87.2 1:2.9 2:9.9 | 0:82.1 1:2.7 2:15.2 | 0:92 1:0.9 2:7.1 | 0:86.2 1:1 2:12.8 | 0:82.7 1:0.9 2:16.4 | 0:98.6 2:1.4 | 0:96.1 2:3.9 | 0:92.3 2:7.7 | 0:98.6 2:1.4 | 0:96.6 2:3.4 | 0:93.4 2:6.6 |
| P SAO mergeL/mergeU/explicit % | 0:49.6 1:17.4 2:33 | 0:43.8 1:18.2 2:38 | 0:40.4 1:17.9 2:41.7 | 0:70.1 1:17.5 2:12.4 | 0:61.5 1:19.8 2:18.7 | 0:54.7 1:21 2:24.3 | 0:66.7 1:17.7 2:15.6 | 0:57.4 1:20 2:22.6 | 0:51.9 1:20.5 2:27.6 | 0:67.3 1:17.9 2:14.9 | 0:58.2 1:20.1 2:21.7 | 0:53.1 1:20.4 2:26.6 |
| P bits % residual/MVD/SAO | 67.8/5.2/4.5 | 75.2/4.1/3.7 | 79.8/3.3/3.1 | 60.2/9.7/2.0 | 63.9/9.2/1.9 | 66.8/8.8/1.8 | 57.2/12.0/2.1 | 64.5/10.7/2.0 | 69.6/9.5/1.8 | 68.4/8.2/2.1 | 75.2/6.8/1.9 | 79.8/5.8/1.8 |

Value keys: shapes are `whole` 2Nx2N, `halfH` 2NxN, `halfV` Nx2N, `quarter`
NxN (intra NxN = four PBs). MV size buckets are in luma samples. "Merge" PBs
are merged with residual, "inter" PBs carry an MVD. The TB depth of inter
blocks includes the implicit split of a CU larger than the max TB. Chroma
intra mode syntax, from a separate counter run: BWC SOC and ours code the
derived mode (DM) 100% of the time; x265 codes DM for 84-86%.

## H.264

| | SOC-1000 | SOC-1500 | SOC-2000 | x264m-1000 | x264m-1500 | x264m-2000 | oh264-1000 | oh264-1500 | oh264-2000 | ncDef-1000 | ncDef-1500 | ncDef-2000 | ncFast-1000 | ncFast-1500 | ncFast-2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| profile / CABAC / 8x8 transform | 100/1/1 | 100/1/1 | 100/1/1 | 100/1/1 | 100/1/1 | 100/1/1 | 66/0/0 | 66/0/0 | 66/0/0 | 100/1/1 | 100/1/1 | 100/1/1 | 100/1/1 | 100/1/1 | 100/1/1 |
| max ref frames / L0 refs | 1/1 | 1/1 | 1/1 | 3/5 | 3/5 | 3/5 | 1/1 | 1/1 | 1/1 | 2/1 | 2/1 | 2/1 | 2/1 | 2/1 | 2/1 |
| deblock idc / alpha / beta | [0]/[0]/[0] | [0]/[0]/[0] | [0]/[0]/[0] | [0]/[0]/[0] | [0]/[0]/[0] | [0]/[0]/[0] | [0]/[0]/[0] | [0]/[0]/[0] | [0]/[0]/[0] | [0]/[0]/[0] | [0]/[0]/[0] | [0]/[0]/[0] | [0, 1]/[0]/[0] | [0, 1]/[0]/[0] | [0, 1]/[0]/[0] |
| weighted pred / chroma QP offset | [0]/[0] | [0]/[0] | [0]/[0] | [1]/[-2] | [1]/[-2] | [1]/[-2] | [0]/[0] | [0]/[0] | [0]/[0] | [0]/[-3] | [0]/[-3] | [0]/[-3] | [0]/[-3] | [0]/[-3] | [0]/[-3] |
| key bytes / mean P bytes / P CV% | 64054/3960/19.0 | 64054/6035/23.2 | 64054/8038/28.7 | 11187/4182/29.5 | 17700/6252/36.2 | 23911/8331/39.5 | 32910/4069/31.6 | 32910/6158/36.3 | 40878/8225/40.5 | 28731/4027/76.8 | 41127/6019/83.7 | 56977/7992/81.0 | 31554/4016/78.8 | 45049/6016/78.9 | 62495/7989/77.7 |
| key QP / P QP (area-mean CB QP per frame) mean [range] sd | 24.1/36.4 [28.3, 43.4] 3.1 | 24.1/33.1 [25.5, 41.4] 3.43 | 24.1/31.1 [25.4, 39.7] 3.52 | 37.9/36.8 [33.7, 43.2] 2.2 | 34.3/33.4 [30.6, 39.9] 2.61 | 31.8/31.2 [28.4, 38.4] 2.83 | 32.0/39.6 [29.6, 46.7] 3.31 | 32.0/35.8 [26.1, 43.9] 3.5 | 30.0/33.5 [23.7, 42.5] 3.69 | 32.0/37.9 [29.0, 49.0] 3.96 | 29.0/34.9 [26.0, 48.0] 4.42 | 26.0/32.7 [24.0, 46.0] 4.59 | 32.0/38.9 [30.0, 50.0] 3.63 | 29.0/35.9 [26.0, 48.0] 3.94 | 26.0/33.9 [24.0, 46.0] 4.17 |
| P frame QP: mean abs step / reversals >0.5 | 0.38/11 | 0.39/9 | 0.4/17 | 0.38/18 | 0.33/12 | 0.32/9 | 0.57/30 | 0.53/31 | 0.56/42 | 1.68/122 | 1.64/122 | 1.73/125 | 1.55/107 | 1.58/113 | 1.6/113 |
| P slice-QP changes (of 298) | 58 | 57 | 36 | 73 | 65 | 58 | 161 | 165 | 162 | 183 | 171 | 183 | 147 | 159 | 160 |
| KEY CB size % area | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 |
| KEY intra shape % | whole:23.8 quarter:72.7 sixteenth:3.6 | whole:23.8 quarter:72.7 sixteenth:3.6 | whole:23.8 quarter:72.7 sixteenth:3.6 | whole:24.2 quarter:72.6 sixteenth:3.3 | whole:18.7 quarter:74.3 sixteenth:7 | whole:16.5 quarter:75 sixteenth:8.5 | whole:64.6 sixteenth:35.4 | whole:64.6 sixteenth:35.4 | whole:56.2 sixteenth:43.8 | whole:72.7 quarter:24.8 sixteenth:2.5 | whole:72.3 quarter:24.7 sixteenth:3 | whole:72.3 quarter:24.7 sixteenth:3 | whole:95.6 sixteenth:4.4 | whole:95.6 sixteenth:4.4 | whole:95.7 sixteenth:4.3 |
| KEY intra PB size % | 4:3.6 8:72.7 16:23.8 | 4:3.6 8:72.7 16:23.8 | 4:3.6 8:72.7 16:23.8 | 4:3.3 8:72.6 16:24.2 | 4:7 8:74.3 16:18.7 | 4:8.5 8:75 16:16.5 | 4:35.4 16:64.6 | 4:35.4 16:64.6 | 4:43.8 16:56.2 | 4:2.5 8:24.8 16:72.7 | 4:3 8:24.7 16:72.3 | 4:3 8:24.7 16:72.3 | 4:4.4 16:95.6 | 4:4.4 16:95.6 | 4:4.3 16:95.7 |
| KEY TB size % | 4:27.3 8:72.7 | 4:27.3 8:72.7 | 4:27.3 8:72.7 | 4:27.4 8:72.6 | 4:25.7 8:74.3 | 4:25 8:75 | 4:100 | 4:100 | 4:100 | 4:75.2 8:24.8 | 4:75.3 8:24.7 | 4:75.3 8:24.7 | 4:100 | 4:100 | 4:100 |
| KEY intra class % (planar/dc/angular) | planar:7.1 dc:27.2 angular:65.7 | planar:7.1 dc:27.2 angular:65.7 | planar:7.1 dc:27.2 angular:65.7 | planar:5.4 dc:20 angular:74.6 | planar:3.4 dc:17.2 angular:79.4 | planar:3.1 dc:18.3 angular:78.7 | planar:8.5 dc:26.1 angular:65.4 | planar:8.5 dc:26.1 angular:65.4 | planar:7.9 dc:24.7 angular:67.4 | planar:13.1 dc:35.3 angular:51.6 | planar:13.4 dc:35.1 angular:51.6 | planar:12.8 dc:36.2 angular:51 | planar:18.9 dc:38 angular:43.1 | planar:19.5 dc:40.2 angular:40.3 | planar:18.6 dc:40.4 angular:41 |
| KEY distinct luma modes by PB size | 16:4 4:9 8:9 | 16:4 4:9 8:9 | 16:4 4:9 8:9 | 16:4 4:9 8:9 | 16:4 4:9 8:9 | 16:4 4:9 8:9 | 16:4 4:9 | 16:4 4:9 | 16:4 4:9 | 16:4 4:9 8:9 | 16:4 4:9 8:9 | 16:4 4:9 8:9 | 16:4 4:9 | 16:4 4:9 | 16:4 4:9 |
| KEY chroma mode % (H.264: 0 DC 1 H 2 V 3 plane) | 0:34.5 1:26.9 2:21.8 3:16.8 | 0:34.5 1:26.9 2:21.8 3:16.8 | 0:34.5 1:26.9 2:21.8 3:16.8 | 0:85.1 1:8.9 2:5.6 3:0.4 | 0:77.5 1:12.7 2:8.9 3:0.9 | 0:71.9 1:15.5 2:10.8 3:1.9 | 0:72.9 1:14 2:11 3:2.1 | 0:72.9 1:14 2:11 3:2.1 | 0:63.2 1:19.3 2:14.7 3:2.7 | 0:40.5 1:29.4 2:19.2 3:10.9 | 0:41.9 1:26.9 2:19.5 3:11.8 | 0:36.9 1:30.1 2:22.9 3:10.1 | 0:40.5 1:29.4 2:19.2 3:10.9 | 0:41.9 1:26.9 2:19.5 3:11.8 | 0:36.9 1:30.1 2:22.9 3:10.1 |
| KEY luma TB area coded % | 86.4 | 86.4 | 86.4 | 37.1 | 47.6 | 57.3 | 57.7 | 57.7 | 60.0 | 72.6 | 79.9 | 84.0 | 73.2 | 79.9 | 83.9 |
| KEY CBs with nonzero dQP / coded, range | 940/3595 [-10, 10] | 940/3595 [-10, 10] | 940/3595 [-10, 10] | 1039/3073 [-21, 20] | 1118/3366 [-17, 19] | 1195/3459 [-17, 20] | 0/3589 [0, 0] | 0/3589 [0, 0] | 0/3599 [0, 0] | 0/3600 [0, 0] | 0/3600 [0, 0] | 0/3600 [0, 0] | 0/3600 [0, 0] | 0/3600 [0, 0] | 0/3600 [0, 0] |
| KEY CB QP p10-p90 | 22-27 | 22-27 | 22-27 | 34-42 | 30-38 | 27-36 | 32-32 | 32-32 | 30-30 | 32-32 | 29-29 | 26-26 | 32-32 | 29-29 | 26-26 |
| P CB size % area | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 |
| P inter (non-skip) CB size % | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 |
| P skip CB size % | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 | 16:100 |
| P intra/inter/skip % area | 7.9/31.9/60.2 | 6.1/45.2/48.6 | 5.7/53.2/41.1 | 3.2/38.0/58.8 | 3.5/47.1/49.4 | 3.9/53.7/42.4 | 18.7/18.0/63.3 | 16.7/28.5/54.7 | 15.6/35.8/48.6 | 5.8/33.5/60.8 | 4.3/44.3/51.3 | 4.0/51.1/44.9 | 4.6/30.1/65.3 | 3.6/39.5/56.9 | 3.3/45.5/51.1 |
| P inter partition % | whole:83.7 halfH:10 halfV:6.3 | whole:82.2 halfH:11.5 halfV:6.2 quarter:0.1 | whole:81.4 halfH:12.5 halfV:5.9 quarter:0.2 | whole:88 halfH:5.1 halfV:3.4 quarter:3.6 | whole:82.4 halfH:7.7 halfV:5.1 quarter:4.8 | whole:76.5 halfH:10.4 halfV:6.7 quarter:6.4 | whole:93 halfH:2.7 halfV:2.2 quarter:2.1 | whole:88.8 halfH:4.2 halfV:3.3 quarter:3.6 | whole:85.4 halfH:5.5 halfV:4.1 quarter:5 | whole:97.1 halfH:1.6 halfV:1.1 quarter:0.2 | whole:95.5 halfH:2.5 halfV:1.6 quarter:0.4 | whole:94.1 halfH:3.2 halfV:2 quarter:0.7 | whole:99.7 halfH:0.2 halfV:0.1 | whole:99.7 halfH:0.2 halfV:0.1 | whole:99.6 halfH:0.2 halfV:0.1 quarter:0.1 |
| P inter PB size % area | 8x16:2.2 16x8:3.5 16:94.4 | 8:0.1 8x16:3 16x8:5.5 16:91.4 | 8:0.1 8x16:3.4 16x8:7 16:89.5 | 8:1.4 8x16:1.3 16x8:2 16:95.3 | 8:2.3 8x16:2.5 16x8:3.7 16:91.4 | 8:3.6 8x16:3.7 16x8:5.8 16:86.9 | 8:0.5 8x16:0.5 16x8:0.6 16:98.4 | 8:1.2 8x16:1.1 16x8:1.4 16:96.2 | 8:2.1 8x16:1.7 16x8:2.3 16:93.8 | 8:0.1 8x16:0.4 16x8:0.6 16:99 | 8:0.2 8x16:0.7 16x8:1.2 16:97.9 | 8:0.4 8x16:1.1 16x8:1.7 16:96.8 | 16x8:0.1 16:99.9 | 8x16:0.1 16x8:0.1 16:99.9 | 8x16:0.1 16x8:0.1 16:99.8 |
| P MV int/half/quarter % | int:34.3 half:24.5 quarter:41.2 | int:28 half:27.3 quarter:44.8 | int:25.1 half:28.1 quarter:46.8 | int:18.8 half:14.7 quarter:66.5 | int:17.7 half:13.5 quarter:68.8 | int:17.1 half:13.3 quarter:69.6 | int:46.5 half:9.7 quarter:43.7 | int:37.5 half:11.5 quarter:51 | int:33.9 half:12.4 quarter:53.7 | int:56.9 half:18.7 quarter:24.4 | int:49.2 half:19.8 quarter:31 | int:44.3 half:19.8 quarter:35.9 | int:58.5 half:28.7 quarter:12.8 | int:51.6 half:31.6 quarter:16.8 | int:48.1 half:32.9 quarter:19 |
| P zero MV % | 28.1 | 21.7 | 19.2 | 9.6 | 8.1 | 7.4 | 38.7 | 27.7 | 23.1 | 25.3 | 20.2 | 17.0 | 29.9 | 24.3 | 21.0 |
| P MV size % (0/<1px/1-2/2-4/4-8/8-16/16-32/>32) | 28.1/6.3/7.1/5.9/15.3/21.2/14.6/1.4 | 21.7/9.1/8/5.8/16.2/22.5/15.1/1.5 | 19.2/10.6/8.2/5.7/16.5/22.9/15.2/1.6 | 9.6/12.6/9.3/7.1/17.3/24.5/16.9/2.7 | 8.1/13.6/9.4/7.1/17.4/24.7/16.9/2.7 | 7.4/14.2/9.6/7.2/17.3/24.8/16.8/2.7 | 38.7/4.7/5.5/5.7/13.1/18.4/13/0.8 | 27.7/6.9/8.1/6.2/14.9/20.8/14.3/1 | 23.1/8.1/9.1/6.5/15.6/21.7/14.7/1.2 | 25.3/4/6.1/8.4/16.3/22.2/15.6/2.1 | 20.2/5.4/7.3/8.6/17.1/23.2/16/2.3 | 17/6.6/7.9/8.8/17.5/23.7/16.2/2.4 | 29.9/2.3/5.8/8.1/15.6/21.1/15.3/2 | 24.3/3.3/7.5/8.5/16.4/22.2/15.6/2.1 | 21/4/8.4/8.7/16.9/22.8/15.9/2.4 |
| P ref_idx % | 0:100 | 0:100 | 0:100 | 0:88 1:7.3 2:3.5 3:1.2 | 0:85 1:10.1 2:3.6 3:1.3 | 0:83.2 1:11.5 2:3.9 3:1.4 | 0:100 | 0:100 | 0:100 | 0:100 | 0:100 | 0:100 | 0:100 | 0:100 | 0:100 |
| P MVD size % (0/<1px/1-2px/larger) | 35.8/25.5/15.3/23.4 | 37.2/32.6/12.8/17.3 | 37.7/36.1/11.5/14.7 | 24.3/28.9/18.2/28.5 | 22.2/37.4/17.4/23 | 21.4/42/16.2/20.5 | 15/34.5/21.3/29.2 | 14/44.7/19.3/22 | 14.1/49.4/17.5/18.9 | 25.6/33.9/17.4/23.1 | 25.5/39/16/19.5 | 24.4/42.6/15.1/17.8 | 33.7/23.1/20.2/22.9 | 32.4/28.2/19.1/20.2 | 31.3/31.3/18.4/19 |
| P intra class % | planar:29.3 dc:30.9 angular:39.8 | planar:21.1 dc:31.4 angular:47.5 | planar:15.9 dc:30.8 angular:53.3 | planar:4.8 dc:27.6 angular:67.6 | planar:4.1 dc:23.9 angular:72 | planar:4 dc:21.9 angular:74.1 | planar:9.8 dc:17.8 angular:72.3 | planar:10.5 dc:20 angular:69.4 | planar:10.7 dc:21.2 angular:68.1 | planar:17.1 dc:39.6 angular:43.3 | planar:15.3 dc:40 angular:44.6 | planar:14.4 dc:39.6 angular:45.9 | planar:21 dc:40.6 angular:38.4 | planar:19.8 dc:43 angular:37.2 | planar:19.2 dc:43.7 angular:37.1 |
| P intra PB size % | 4:0.1 8:26.7 16:73.2 | 4:0.4 8:45.6 16:54 | 4:0.8 8:59.4 16:39.8 | 4:3.1 8:60.5 16:36.4 | 4:5.3 8:62.6 16:32.1 | 4:6.8 8:64.8 16:28.4 | 4:1.7 16:98.3 | 4:4.9 16:95.1 | 4:8.3 16:91.7 | 4:2 8:18.5 16:79.5 | 4:3.5 8:26.6 16:69.9 | 4:4 8:30.6 16:65.3 | 4:7.3 16:92.7 | 4:12.5 16:87.5 | 4:14.9 16:85.1 |
| P TB size inter % | 4:74.3 8:25.7 | 4:66.8 8:33.2 | 4:62.3 8:37.7 | 4:59.4 8:40.6 | 4:54.5 8:45.5 | 4:52.2 8:47.8 | 4:100 | 4:100 | 4:100 | 4:57.8 8:42.2 | 4:49.5 8:50.5 | 4:44 8:56 | 4:56.7 8:43.3 | 4:49.6 8:50.4 | 4:44.3 8:55.7 |
| P luma area with coded residual % | 8.8 | 11.8 | 15.2 | 7.5 | 11.3 | 14.4 | 8.9 | 10.7 | 12.4 | 10.8 | 14.8 | 18.5 | 10.9 | 15.1 | 19.1 |
| P non-skip CBs with any / chroma residual % | 52.2/10.1 | 56.9/11.2 | 62.4/12.4 | 52.3/11.4 | 60.1/13.5 | 65.5/15.8 | 71.6/11.6 | 69.4/11.6 | 70.6/12.6 | 56.8/14.7 | 60.7/16.6 | 64.3/17.8 | 58.5/16.3 | 61.6/18.0 | 64.8/18.6 |
| P coded TB % Y/Cb/Cr (of TB area) | 22.1/-/- | 23.0/-/- | 25.7/-/- | 18.3/-/- | 22.2/-/- | 25.1/-/- | 24.2/-/- | 23.7/-/- | 24.2/-/- | 27.4/-/- | 30.4/-/- | 33.6/-/- | 31.5/-/- | 35.1/-/- | 39.1/-/- |
| P nonzero levels per 1000 px | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| P dQP nonzero per frame / range | 436.7 [-13, 13] | 572.4 [-13, 13] | 663.9 [-13, 12] | 308.2 [-20, 23] | 435.1 [-21, 23] | 534.5 [-21, 23] | 5.7 [-4, 2] | 5.9 [-4, 2] | 5.8 [-3, 2] | 0.2 [-6, 6] | 0.4 [-9, 9] | 0.4 [-8, 8] | 0.4 [-6, 6] | 0.5 [-7, 7] | 0.4 [-7, 7] |
| P CB QP p10-p90 | 30-42 | 27-40 | 25-38 | 32-42 | 28-39 | 26-37 | 36-44 | 32-41 | 30-40 | 35-46 | 32-44 | 29-42 | 36-46 | 33-44 | 31-42 |
| P bits % in intra/inter/skip MBs | intra:24.5 inter:70.5 skip:4.5 | intra:19.1 inter:77.3 skip:3.3 | intra:18 inter:79.2 skip:2.5 | intra:12 inter:83 skip:4.4 | intra:12.3 inter:84 skip:3.2 | intra:12.9 inter:84.3 skip:2.5 | intra:47.2 inter:43.2 | intra:36.9 inter:56.2 | intra:31.6 inter:63.1 | intra:22.5 inter:73.1 skip:3.9 | intra:19.2 inter:77.6 skip:2.9 | intra:17.8 inter:79.7 skip:2.2 | intra:21.3 inter:74.6 skip:3.7 | intra:18.1 inter:78.8 skip:2.7 | intra:16.9 inter:80.7 skip:2.2 |

Value keys: `whole` 16x16 / I_16x16, `halfH` 16x8, `halfV` 8x16, `quarter`
P_8x8 / I_8x8, `sixteenth` I_4x4. "Planar" for H.264 is the I_16x16 / chroma
plane mode. P_Skip counts in "zero MV" when its predicted vector is zero.
H.264 MVDs are magnitudes, so the sign is not kept.

## Reading (this clip, these three rates)

HEVC:

- **Block sizes:** the BWC SOC codes P frames almost entirely as 32x32 CUs
  (91-94% of the area, with 2Nx2N partitions on 99%+). 16x16 CUs cover ~5%
  and 64x64 appears only as skip. The SPS rules out 8x8 CUs (min CB 16) and
  32x32 transforms (max TB 16), so every 32x32 CU is coded as four 16x16
  transforms. It never chooses a transform split. x265 medium uses 8-64 CUs,
  with 64x64 on 16-28% of P area and 32x32 transforms on 69-82%. Ours uses
  mostly 16x16, with an RQT split on 4-6%.
- **Partitions:** it uses a few rectangular 2NxN / Nx2N partitions (~0.6%),
  never AMP (disabled). x265 medium and ours use 2Nx2N only. So partitions
  are not where it is weak.
- **Motion:** one reference picture (x265: 3, 5-7% of area on refs 1-2) and
  2 merge candidates (x265: 3). It does sub-pel everywhere, but with a
  strong half-pel bias: 51% half / 44% quarter / 5% integer against x265's
  16 / 67 / 17. That looks like a half-pel stage with limited quarter-pel
  refinement. Its MV size distribution matches x265's, so its search range
  looks adequate. It picks almost no zero vectors (2.5-2.8% vs 6-7%) and
  uses less skip (24-41% vs 40-49%) but more merge-with-residual.
- **Intra:** all 35 luma modes appear. In P frames it leans on planar / DC
  (71% vs 31% for x265). On the key frame it uses NxN (8x8 PBs, 4x4 TBs) on
  36% of the area, where x265 uses 1-4%. It never codes an explicit chroma
  mode (neither do we).
- **Key frame:** the same frame (62285 bytes, mean QP 24.4) at all three
  rates: a fixed key-frame QP that ignores the target. It costs half a
  second's budget at 1000k (x265: 10.6 kB, QP ~38). Its P frames then run
  about 2.5 QP coarser than x265 at the same rate. Its H.264 P frames match
  x264's QP, though; there the key frame is paid for elsewhere.
- **Quantisation:** real adaptive QP. A quarter to a third of 16x16 quant
  groups carry a non-zero delta, up to ±12-13, wider than x265 (±6). Frame
  QP is steady, with a mean frame-to-frame step of 0.4 against x265's 0.2
  and our 1.25. Frame sizes vary least of the three (CV 14-15% vs 36-40%
  x265 and 73-76% ours).
- **Off:** sign hiding and transform skip are both off. SAO is on and fully
  used (band and edge on luma, less merging than x265). Deblocking runs at
  default offsets.
- **Bits:** 68-80% of P bits are residual (x265: 60-67%), 3-5% MVD (x265:
  9-10%). The large fixed blocks with no transform split leave fewer side
  bits and spend more on residual.

H.264:

- **Same encoder pattern:** the key frame is identical at all rates (64054
  bytes, mean QP 24.1). One reference frame, CABAC, 8x8 transform on,
  deblocking at default offsets, no weighted prediction.
- **Partitions:** it uses 16x16, 16x8 and 8x16 (16-19% rectangular). It
  almost never uses P_8x8 (≤0.2%), so it never uses sub-8x8 partitions.
  x264 medium uses P_8x8 on 4-6%.
- **Motion:** the same half-pel bias as its HEVC (25-28% half vs x264's
  13-15%). It has many more zero vectors (19-28% vs 7-10%) and zero MVDs
  (36-38% vs 21-24%). Only one reference (x264: 12-17% of area on refs 1-3).
- **Intra:** it uses I_8x8, I_4x4 and I_16x16 with all 9 directional modes.
  It uses I_16x16 more than x264 in P frames (40-73% vs 28-36% of intra
  area). The key-frame mix (72.7% I_8x8) is almost the same as x264's.
- **Transform:** 4x4 on 62-74% of inter TB area against 52-59% for x264.
- **Quantisation:** heavy adaptive QP: 437-664 macroblocks per frame with a
  non-zero mb_qp_delta (±13). x264 has 308-535 (±23). The frame QP is about
  as steady as x264's (step 0.4 vs 0.3-0.4).

Ours: our Default's large frame-QP swings and frame sizes (CV 73-84%,
mean frame-QP step 1.25-1.7 against 0.2-0.4 for x26x and the BWC SOC) are
deliberate, not unstable rate control. It is the pseudo-layer quantizer
cadence (`crates/nc-encoder/src/qp_cadence.rs`): in a flat low-delay P
stream every fourth frame gets "base" treatment with a finer QP, and later
frames inherit it wherever they copy or skip, as SVT-AV1's real-time flat
IPP does. The demo's QP overlay should make the pattern visible.

What the streams cannot tell us: the search pattern and range limit, whether
decisions are rate-distortion optimised or threshold-based, what drives the
adaptive QP (activity, motion or an ROI), any lookahead, and hardware cost.

## Reproduce

`stat` exists only in an analysis build of the CLI (docs/PROCEDURES.md "Inspect a
stream"): `CARGO_TARGET_DIR=~/nanocodec-data/targets/analysis cargo build --release
-p nc-cli --features analysis`.

```
nanocodec stat --codec hevc --input stream.ivf --json stats.json
cd eval && python3 -m nceval.streamstats --codec hevc stream.ivf
```
