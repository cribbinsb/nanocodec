# Sample clip attribution

The demo's sample clips are short excerpts of openly licensed video, redistributed
here under those licences; each excerpt carries the licence of its source. Every
clip ships in two sizes, `<id>-360p.mp4` (640×360, read at the demo's 384×216 and
640×360 settings) and `<id>-720p.mp4` (1280×720, read at 1280×720).

Changes made to every clip: cut to the segment below, audio removed, scaled with
a Lanczos filter, and encoded with ffmpeg/libx264 (`-preset veryslow`, the CRF
below, `aq-mode=3`, keyframe every 2 s) as H.264 High, yuv420p, BT.709, MP4 with
faststart, for browser playback. H.264 was kept rather than AV1 because Safari
plays AV1 only on devices with an AV1 hardware decoder.

## Blender Foundation open movies (CC BY 3.0)

Licence: Creative Commons Attribution 3.0,
<https://creativecommons.org/licenses/by/3.0/>. The Tears of Steel excerpts
were also centre-cropped from 2.4:1 (1920×800) to 16:9 (1422×800) before scaling.

| Files | Work | Author | Source | Segment | CRF 360p / 720p |
|---|---|---|---|---|---|
| `tos-canal-*.mp4` | *Tears of Steel* (2012) | (CC) Blender Foundation, mango.blender.org | <https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip> | 4:18.0–4:30.0 (12 s, 24 fps) | 22 / 24 |
| `tos-bridge-*.mp4` | *Tears of Steel* (2012) | (CC) Blender Foundation, mango.blender.org | <https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip> | 0:26.0–0:39.0 (13 s, 24 fps) | 22 / 24 |
| `bbb-tree-*.mp4` | *Big Buck Bunny* (2008) | (c) copyright 2008, Blender Foundation, peach.blender.org | <https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov.zip> | 2:07.0–2:21.0 (14 s, 24 fps) | 22 / 24 |

The Tears of Steel soundtrack is under a different licence (CC BY-ND 3.0); no
audio is included.

## SVT Open Content Video Test Suite 2022 – Natural Complexity (CC BY 4.0)

*SVT Open Content Video Test Suite 2022 – Natural Complexity* by Sveriges
Television AB (SVT), Video Core team, is licensed under Creative Commons
Attribution 4.0 International, <https://creativecommons.org/licenses/by/4.0/>.

- Suite page and licence statement: <https://svt.github.io/en/content/>
- Documentation (v1.3), whose "Intentions, license and restrictions of use"
  section states the CC BY 4.0 licence:
  <https://svt.github.io/docs/SVT_Open_Content_Video_Test_Suite_2022_Natural_Complexity_v1-3-reduced.pdf>
- Masters: `ftp://svtopencontent.svt.se/pub/svt_videotestsuite_natural_complexity/REC709/`
  (SDR Rec. 709 mezzanines, 3840×2160p50, JPEG 2000 lossless, YUV 4:4:4 12-bit)

Additional changes: 3840×2160 downscaled to 1280×720 and 640×360, every second
frame dropped (50 → 25 fps), 4:4:4 12-bit converted to 4:2:0 8-bit.

| Files | Sequence | Master file | Segment | CRF 360p / 720p |
|---|---|---|---|---|
| `svt-waterfall-*.mp4` | waterfall (SDR) | `waterfall_JPEG2000_SDR_3840x2160p50_YUV444_12bit_Lossless.mov` | 0:10.0–0:15.0 (5 s) | 25 / 28 |
| `svt-midnight-sun-*.mp4` | midnight_sun (SDR) | `midnight_sun_JPEG2000_SDR_3840x2160p50_YUV444_12bit_Lossless.mov` | 0:10.0–0:15.0 (5 s) | 25 / 28 |
| `svt-smoke-sauna-*.mp4` | smoke_sauna (SDR) | `smoke_sauna_JPEG2000_SDR_3840x2160p50_YUV444_12bit_Lossless.mov` | 0:10.0–0:15.0 (5 s) | 25 / 28 |

SVT's older *High Definition Multi Format Test Set* (2006: CrowdRun, ParkJoy,
DucksTakeOff, IntoTree, OldTownCross) is not included: its terms limit use to
"developing, testing and presenting technology standards" and are not an open
licence.

These clips are not drawn from the project's evaluation sets, whose licences do
not allow redistribution.
