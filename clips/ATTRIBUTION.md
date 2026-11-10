# Sample clip attribution

The demo's sample clips are short excerpts of openly licensed, public-domain or
freely redistributable video, redistributed here under their terms; each excerpt
carries the licence or terms of its source. The demo is non-commercial and no fee
is charged for any of these files. Every
clip ships in two sizes, `<id>-360p.mp4` (640×360, read at the demo's 384×216 and
640×360 settings) and `<id>-720p.mp4` (1280×720, read at 1280×720); the 1.9:1
Netflix clips are 640×338 and 1280×676.

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
| `svt-waterfall-*.mp4` | waterfall (SDR) | `waterfall_JPEG2000_SDR_3840x2160p50_YUV444_12bit_Lossless.mov` | 0:10.0–0:20.0 (10 s) | 25 / 28 |
| `svt-midnight-sun-*.mp4` | midnight_sun (SDR) | `midnight_sun_JPEG2000_SDR_3840x2160p50_YUV444_12bit_Lossless.mov` | 0:10.0–0:20.0 (10 s) | 25 / 28 |
| `svt-smoke-sauna-*.mp4` | smoke_sauna (SDR) | `smoke_sauna_JPEG2000_SDR_3840x2160p50_YUV444_12bit_Lossless.mov` | 0:10.0–0:20.0 (10 s) | 25 / 28 |

## Xiph.org derf collection: JCT-VC / Vidyo conferencing sequences (public domain)

The Xiph.org Video Test Media page (derf's collection),
<https://media.xiph.org/video/derf/>, labels each of these sequences
"16:9 | 600 frames | public domain" (checked 2026-11-10; the page gives no
per-file notes for them). They were originally contributed by Vidyo to the
JCT-VC HEVC test set (class E, videoconferencing). Source files: lossless
1280×720p60 YUV 4:2:0 8-bit y4m, `https://media.xiph.org/video/derf/y4m/<file>`.

Additional changes: every second frame dropped (60 → 30 fps); the 720p file is
not rescaled.

| Files | Sequence | Source file | Segment | CRF 360p / 720p |
|---|---|---|---|---|
| `xiph-johnny-*.mp4` | Johnny | `Johnny_1280x720_60.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `xiph-kristen-sara-*.mp4` | KristenAndSara | `KristenAndSara_1280x720_60.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `xiph-four-people-*.mp4` | FourPeople | `FourPeople_1280x720_60.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `xiph-vidyo1-*.mp4` | vidyo1 | `vidyo1_720p_60fps.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `xiph-vidyo3-*.mp4` | vidyo3 | `vidyo3_720p_60fps.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `xiph-vidyo4-*.mp4` | vidyo4 | `vidyo4_720p_60fps.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |

## SVT High Definition Multi Format Test Set (2006)

These excerpts come from *The SVT High Definition Multi Format Test Set*,
Version 1.0, Feb. 2006, by Sveriges Television AB (SVT), taken from the SVT
programme 'Fairytale'. They are distributed here under the set's own terms, not
an open licence. This project uses them for developing, testing and presenting
implementations of video coding standards (H.264, HEVC, AV1, VP9, AV2, VVC), and
no fee is charged. The set's documentation, which its terms require to accompany
any distributed part, ships next to the clips:
[`SVT_MultiFormat_v10.pdf`](SVT_MultiFormat_v10.pdf) (from
<https://media.xiph.org/video/derf/vqeg.its.bldrdoc.gov/HDTV/SVT_MultiFormat/SVT_MultiFormat_v10.pdf>).

Terms, verbatim from that document (§1 and §2):

> All sequences from this set may be distributed freely (complete or in parts) as long as the
> copyright and the restrictions of the use of the sequences are not violated.
> This document has to be distributed with any part of the test set that is distributed.
> It is not allowed to charge a fee for the distribution of the sequences. Anybody receiving
> material from this data set is kindly asked to further distribute these sequences to other parties
> to reach a super distribution effect.
>
> **2. Copyright and Restrictions of Use**
>
> Individuals and organizations extracting sequences from this archive agree that the sequences
> and all intellectual property rights therein remain the property of Sveriges Television AB
> (SVT), Sweden. These sequences may only be used for the purpose of developing, testing and
> presenting technology standards. SVT makes no warranties with respect to the materials and
> expressly disclaim any warranties regarding their fitness for any purpose.

Source files: the Xiph.org derf collection's y4m conversions,
`https://media.xiph.org/video/derf/y4m/<file>` (originally
`ftp://vqeg.its.bldrdoc.gov/HDTV/SVT_MultiFormat/`). Additional changes: every
second frame dropped (50 → 25 fps); crowd_run downscaled from 1080p (the others
are SVT's own 720p versions and are not rescaled at 720p).

| Files | Sequence | Source file | Segment | CRF 360p / 720p |
|---|---|---|---|---|
| `svt-old-town-cross-*.mp4` | OldTownCross | `old_town_cross_420_720p50.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `svt-in-to-tree-*.mp4` | IntoTree | `in_to_tree_420_720p50.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `svt-crowd-run-*.mp4` | CrowdRun | `crowd_run_1080p50.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 25 / 28 |
| `svt-park-joy-*.mp4` | ParkJoy | `park_joy_420_720p50.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 25 / 28 |
| `svt-ducks-take-off-*.mp4` | DucksTakeOff | `ducks_take_off_420_720p50.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 25 / 28 |

## Netflix Open Content: El Fuente and Chimera (CC BY-NC-ND 4.0)

Licence: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International, <https://creativecommons.org/licenses/by-nc-nd/4.0/>. The demo is
non-commercial and no fee is charged.

Each file is a format conversion of an unmodified, contiguous excerpt (10 s, 600
consecutive frames, or the whole sequence where it is shorter; every frame kept at
the source's frame timing, full picture with no cropping): 10-bit converted to 8-bit, downscaled with a Lanczos filter from
4096×2160 to 1280×676 (`-720p.mp4`) and 640×338 (`-360p.mp4`) so the 1.9:1 frame
is kept whole (the demo letterboxes it), and encoded to H.264 as described above
(keyframe every 120 frames). Nothing else is changed. Source files: the Xiph.org
derf collection, `https://media.xiph.org/video/derf/<folder>/Netflix_<Name>_4096x2160_60fps_10bit_420.y4m`;
only the needed frames were fetched.

| Files | Sequence | Source folder | Source frames | CRF 360p / 720p |
|---|---|---|---|---|
| `netflix-narrator-*.mp4` | Narrator (*El Fuente*) | `ElFuente/` | 0–299 (all 300, 5.0 s) | 22 / 24 |
| `netflix-dinner-scene-*.mp4` | DinnerScene (*Chimera*) | `Chimera/` | 60–659 (10 s) | 22 / 24 |
| `netflix-bar-scene-*.mp4` | BarScene (*Chimera*) | `Chimera/` | 60–659 (10 s) | 22 / 24 |
| `netflix-crosswalk-*.mp4` | Crosswalk (*El Fuente*) | `ElFuente/` | 0–299 (all 300, 5.0 s) | 25 / 28 |
| `netflix-food-market-*.mp4` | FoodMarket (*El Fuente*) | `ElFuente/` | 0–599 (all 600, 10 s) | 25 / 28 |
| `netflix-tango-*.mp4` | Tango (*El Fuente*) | `ElFuente/` | 0–293 (all 294, 4.9 s) | 22 / 24 |
| `netflix-boxing-practice-*.mp4` | BoxingPractice (*El Fuente*) | `ElFuente/` | 0–253 (all 254, 4.2 s) | 22 / 24 |
| `netflix-dancers-*.mp4` | Dancers (*Chimera*) | `Chimera/` | 60–659 (10 s) | 22 / 24 |
| `netflix-toddler-fountain-*.mp4` | ToddlerFountain (*Chimera*) | `Chimera/` | 60–659 (10 s) | 25 / 28 |

Copyright notices, verbatim from each sequence's `Netflix_<Name>_Copyright.txt` on the derf page:

`netflix-narrator-*.mp4` (from <https://media.xiph.org/video/derf/ElFuente/Netflix_Narrator_Copyright.txt>):

```text
Digital Video Sequence "Narrator"
Copyright (C)  NETFLIX INC.  2015

4096x2160p, 10-bit, YCbCr 4:2:0, BT.709, 59.94 fps, 300 frames

COPYRIGHT AND LICENSE INFORMATION
December, 2015

NETFLIX INC.
100 Winchester Circle, Los Gatos, CA 95032, USA

The video sequence provided above and all intellectual property rights therein
remain the property of Netflix Inc. This video sequence is licensed under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-nd/4.0/
```

`netflix-dinner-scene-*.mp4` (from <https://media.xiph.org/video/derf/Chimera/Netflix_DinnerScene_Copyright.txt>):

```text
Digital Video Sequence "Dinner Scene"
Copyright (C)  NETFLIX INC.  2015

4096x2160p, 10-bit, YCbCr 4:2:0, BT.709, 59.94 fps, 1199 frames

COPYRIGHT AND LICENSE INFORMATION
December, 2015

NETFLIX INC.
100 Winchester Circle, Los Gatos, CA 95032, USA

The video sequence provided above and all intellectual property rights therein
remain the property of Netflix Inc. This video sequence is licensed under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-nd/4.0/
```

`netflix-bar-scene-*.mp4` (from <https://media.xiph.org/video/derf/Chimera/Netflix_BarScene_Copyright.txt>):

```text
Digital Video Sequence "Bar Scene"
Copyright (C)  NETFLIX INC.  2015

4096x2160p, 10-bit, YCbCr 4:2:0, BT.709, 59.94 fps, 1199 frames

COPYRIGHT AND LICENSE INFORMATION
December, 2015

NETFLIX INC.
100 Winchester Circle, Los Gatos, CA 95032, USA

The video sequence provided above and all intellectual property rights therein
remain the property of Netflix Inc. This video sequence is licensed under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-nd/4.0/
```

`netflix-crosswalk-*.mp4` (from <https://media.xiph.org/video/derf/ElFuente/Netflix_Crosswalk_Copyright.txt>):

```text
Digital Video Sequence "Crosswalk"
Copyright (C)  NETFLIX INC.  2015

4096x2160p, 10-bit, YCbCr 4:2:0, BT.709, 59.94 fps, 300 frames

COPYRIGHT AND LICENSE INFORMATION
December, 2015

NETFLIX INC.
100 Winchester Circle, Los Gatos, CA 95032, USA

The video sequence provided above and all intellectual property rights therein
remain the property of Netflix Inc. This video sequence is licensed under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-nd/4.0/
```

`netflix-food-market-*.mp4` (from <https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket_Copyright.txt>):

```text
Digital Video Sequence "Food Market"
Copyright (C)  NETFLIX INC.  2015

4096x2160p, 10-bit, YCbCr 4:2:0, BT.709, 59.94 fps, 600 frames

COPYRIGHT AND LICENSE INFORMATION
December, 2015

NETFLIX INC.
100 Winchester Circle, Los Gatos, CA 95032, USA

The video sequence provided above and all intellectual property rights therein
remain the property of Netflix Inc. This video sequence is licensed under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-nd/4.0/
```

`netflix-tango-*.mp4` (from <https://media.xiph.org/video/derf/ElFuente/Netflix_Tango_Copyright.txt>):

```text
Digital Video Sequence "Tango"
Copyright (C)  NETFLIX INC.  2015

4096x2160p, 10-bit, YCbCr 4:2:0, BT.709, 59.94 fps, 294 frames

COPYRIGHT AND LICENSE INFORMATION
December, 2015

NETFLIX INC.
100 Winchester Circle, Los Gatos, CA 95032, USA

The video sequence provided above and all intellectual property rights therein
remain the property of Netflix Inc. This video sequence is licensed under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-nd/4.0/
```

`netflix-boxing-practice-*.mp4` (from <https://media.xiph.org/video/derf/ElFuente/Netflix_BoxingPractice_Copyright.txt>):

```text
Digital Video Sequence "Boxing Practice"
Copyright (C)  NETFLIX INC.  2015

4096x2160p, 10-bit, YCbCr 4:2:0, BT.709, 59.94 fps, 254 frames

COPYRIGHT AND LICENSE INFORMATION
December, 2015

NETFLIX INC.
100 Winchester Circle, Los Gatos, CA 95032, USA

The video sequence provided above and all intellectual property rights therein
remain the property of Netflix Inc. This video sequence is licensed under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-nd/4.0/
```

`netflix-dancers-*.mp4` (from <https://media.xiph.org/video/derf/Chimera/Netflix_Dancers_Copyright.txt>):

```text
Digital Video Sequence "Dancers"
Copyright (C)  NETFLIX INC.  2015

4096x2160p, 10-bit, YCbCr 4:2:0, BT.709, 59.94 fps, 1199 frames

COPYRIGHT AND LICENSE INFORMATION
December, 2015

NETFLIX INC.
100 Winchester Circle, Los Gatos, CA 95032, USA

The video sequence provided above and all intellectual property rights therein
remain the property of Netflix Inc. This video sequence is licensed under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-nd/4.0/
```

`netflix-toddler-fountain-*.mp4` (from <https://media.xiph.org/video/derf/Chimera/Netflix_ToddlerFountain_Copyright.txt>):

```text
Digital Video Sequence "Toddler Fountain"
Copyright (C)  NETFLIX INC.  2015

4096x2160p, 10-bit, YCbCr 4:2:0, BT.709, 59.94 fps, 1199 frames

COPYRIGHT AND LICENSE INFORMATION
December, 2015

NETFLIX INC.
100 Winchester Circle, Los Gatos, CA 95032, USA

The video sequence provided above and all intellectual property rights therein
remain the property of Netflix Inc. This video sequence is licensed under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-nd/4.0/
```

Only material whose terms allow redistribution is shipped here; the project's
main evaluation sets (`/mldata`) do not. Johnny and FourPeople are also in the
small `adv12` screening set (AOM CTC copies); being public domain, they may be
redistributed.
