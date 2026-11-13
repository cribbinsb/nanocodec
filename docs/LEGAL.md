# Credits, licences, patents and disclaimers

This page explains, in plain language, what nanocodec is built from, what its licence does and does not give
you, and what you remain responsible for. **It is general information, not legal advice.** If you plan to ship
or deploy a video codec, talk to a lawyer who knows codec licensing.

## In one paragraph

nanocodec is experimental, research-grade software, provided "as is" with no warranty. Its source code is
under the BSD 3-Clause licence, which is a **copyright** licence only. The video standards it implements
(H.264, HEVC, VVC, AV1, VP9, AV2) are covered by **patents held by third parties**. Nothing here (the source
code, the binaries, or the browser demo) gives you a licence to any of those patents. If you use nanocodec,
in any form, obtaining whatever patent licences your use needs is your responsibility.

## Credits

nanocodec was built by studying, and reimplementing ideas and code from, open-source projects under BSD-style
and other permissive licences. We are grateful to their authors. In particular:

- **[SVT-AV1](https://gitlab.com/AOMediaCodec/SVT-AV1)** (Alliance for Open Media; originally Intel and
  Netflix). Its real-time pipeline is the main guide for nanocodec's shared engine, and its `--rtc` presets
  are the bar we measure ourselves against. It is an excellent encoder.
- **libaom**, **dav1d** and **libvpx** (AV1 and VP9), **openh264** (H.264), **kvazaar** (HEVC), **VVenC** and
  **VVdeC** (VVC), **AVM** (AV2), **Thor**, **libyuv** and **libvmaf**.

The full list, with each project's licence, copyright notice and the nanocodec modules that draw on it, is in
[THIRD_PARTY.md](../THIRD_PARTY.md). Modules that reimplement another project's algorithm also name their
source in their own documentation. If you believe an attribution is missing or wrong, please open an issue or
write to <bernandocribbenza@gmail.com> and we will fix it.

nanocodec is an independent project. It is not affiliated with, sponsored by or endorsed by any of these
projects, any standards body, or any company named on this page. Product and project names are trademarks of
their owners and are used only to identify them.

## GPL and LGPL software: compared against, never copied

Our charts compare nanocodec with x264, x265 and FFmpeg, which are GPL or LGPL licensed. We run those programs
as separate, unmodified external tools to produce comparison streams and to check that our streams decode
correctly. **No GPL or LGPL source code is copied into, linked into or distributed with nanocodec.** The
project's rules forbid it, and contributions that include such code will be rejected.

The demo's "pre-encoded" comparison streams are the output of those external encoders (the exact program,
version and command line are shown beside each stream). An encoder's output is not covered by the encoder's
software licence.

## Software licence

nanocodec's own source code is under the [BSD 3-Clause licence](../LICENSE). In short: you may use, modify and
redistribute it, in source or binary form, provided you keep the copyright notice, the licence text and the
third-party notices in [THIRD_PARTY.md](../THIRD_PARTY.md). The licence disclaims all warranties and
liability; read its full text.

## Patents

**The BSD licence grants rights under copyright only. It grants no patent rights**, and the nanocodec authors
do not grant, and are not in a position to grant, a licence to anyone else's patents.

Implementing or using a video coding standard may require patent licences from third parties:

- **H.264 (AVC), HEVC (H.265) and VVC (H.266)** are covered by many patents. Licences are offered through
  patent pools and by individual patent holders who are not in a pool. Fees and terms depend on what you do
  (products sold, content distributed, volumes).
- **AV1, VP9 and AV2** are developed with the aim of being royalty-free. The Alliance for Open Media offers its
  members' essential patents under the AOM Patent License 1.0, subject to that licence's own terms and
  conditions; nanocodec is not an AOMedia product and we make no statement that those terms cover your use.
  Third parties have also asserted that they hold patents on these formats and offer pool licences for them.

Starting points (not a complete list; patent holders outside these bodies may also have claims, and a link
here is not a statement that you do or do not need that licence):

| Body | Formats | Link |
|---|---|---|
| Via Licensing Alliance (Via LA, formerly MPEG LA) | H.264/AVC, HEVC, VVC pools | <https://www.via-la.com/> |
| Access Advance | HEVC and VVC pools | <https://accessadvance.com/> |
| Alliance for Open Media | AOM Patent License 1.0 (AV1, AV2) | <https://aomedia.org/license/patent-license/> |
| WebM Project (Google) | VP9 patent cross-licence | <https://www.webmproject.org/license/> |
| Sisvel | pools asserted for AV1 and VP9 | <https://www.sisvel.com/> |
| ITU-T / ISO / IEC patent declarations | declarations filed against H.264, HEVC, VVC | <https://www.itu.int/en/ITU-T/ipr/> |

We make **no representation** about which patents apply to nanocodec, whether any use of it infringes, or
whether any use is licensed. Whether you need a licence, and from whom, depends on your country, your product
and how you use it. **Downloading, building, running or redistributing nanocodec, including using the browser
demo, does not give you any patent licence, and you are responsible for obtaining any licence you need.**

## The browser demo

The [live demo](https://cribbinsb.github.io/nanocodec/) is a technology demonstration for evaluation and
research. It runs entirely in your browser: your webcam or file is encoded and decoded on your own device and
is not uploaded. Everything on this page applies to the demo too: it comes with no warranty, and using it
conveys no patent licence and no right to use the codecs commercially.

The sample clips belong to their own rights holders and are provided under their own licences, some of which
forbid commercial use or modification; see the
[clip attribution](https://cribbinsb.github.io/nanocodec/clips/ATTRIBUTION.md). Some sample clips (and the
pre-encoded streams made from them) carry NonCommercial or ShareAlike terms (MOTChallenge: CC BY-NC-SA 3.0);
those terms apply to those video files only, not to nanocodec's code.

## No warranty

nanocodec is experimental and is developed largely by AI agents under human direction. It may contain errors.
Published measurements are our own, made as described in [the results pages](results/README.md), and may
contain mistakes; other encoders are compared at the settings stated there and may perform differently in
other configurations. The software and all accompanying material are provided "as is", without warranty of
any kind, as set out in the [LICENCE](../LICENSE).

## Contact

Questions about credits, licensing or this page, and reports of a missing attribution or of content that
should not be here: <bernandocribbenza@gmail.com>, or open an issue on the
[GitHub repository](https://github.com/cribbinsb/nanocodec/issues).
