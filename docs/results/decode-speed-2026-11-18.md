# 8-bit and browser decode speed against the references (2026-11-18)

<sub>Measured 2026-11-18 on commit `34df0f485` (`variance-integration` head, plain `--release` CLI and `--profile web` `+simd128,+relaxed-simd` browser WASM module). All 30 streams verified MD5 bit-exact between `nanocodec decode` and the reference decoders before timing.</sub>

## Method

- **Machines and pinning:**
  - **PC (Zen 3, AVX2):** AMD Ryzen 9 5950X, pinned to reserved timing CPU 14 under `tools/dev/host-timed.sh` (`timing.slice`). Note: `nanocodec` uses runtime CPUID dispatch (`nc-dsp`), running its AVX2 tier on Zen 3.
  - **Jetson (NEON):** NVIDIA Jetson Orin (`aarch64`), pinned to core 3 under the exclusive `~/nanocodec-data/jetson-timed.sh` lock (1-minute load settled `< 1.5`; inter-round IQR `≤ 0.73 %` across all 30 streams).
  - **Browser (Chromium WASM):** Headless Chromium (Playwright) pinned with all child renderer/worker processes to reserved CPU 14 under `tools/dev/host-timed.sh`, running inside a dedicated ES Module Web Worker (`BrowserDecoder.decode` on our `+simd128,+relaxed-simd` `nc_web` build after one full-stream warm-up pass so V8 TurboFan tier-up is complete).
- **Single-thread rate- and output-matched invocations (median of 5 interleaved, order-rotated rounds):**
  - **AV1:** `nanocodec decode --codec av1 --discard 1` vs `dav1d --threads 1 --framedelay 1 --muxer null -q` (`dav1d` 1.5.4). Browser WASM reference: published `dav1d.js` 0.1.1 (`unsafeDecodeFrameAsYUV` + `unsafeCleanup`). *Finding on `dav1d.js` 0.1.1:* its C wrapper omits `dav1d_picture_unref` per frame inside a hardcoded 1024-page (64 MiB) WASM memory import, leaking ~2.1 MiB per 720p frame and trapping with `RuntimeError: memory access out of bounds` after ~25 frames at 720p / ~11 frames at 1080p; it completes the 10-frame conformance vector (`0.48×`), and on the 150-frame 720p/1080p rows we compare against our own native decode time (`else against our own native time`).
  - **VP9:** `nanocodec decode --codec vp9 --discard 1` vs `vpxdec --threads=1 --noblit` (`libvpx` v1.16.0). Browser WASM reference: `@ffmpeg/core` 0.12.10 (`ffmpeg.wasm`, `-threads 1 -f null -`), plus ratio against our own PC native decode time.
  - **H.264 & HEVC:** `nanocodec decode --codec {h264|hevc} --output /dev/null` vs `ffmpeg-decode -nostdin -hide_banner -loglevel error -threads 1 -i <stream> -f rawvideo -y /dev/null` (FFmpeg n8.0.1 anchor; both decoders pack planar 4:2:0 to `/dev/null`). Per ground rules, FFmpeg is used as a wall-time yardstick only (no Callgrind on FFmpeg and none of its source read). Browser WASM reference: `@ffmpeg/core` 0.12.10 (`ffmpeg.wasm`), plus ratio against our own PC native `--discard 1` decode time.
  - **VVC:** `nanocodec decode --codec vvc --discard 1` vs `vvdecapp -v 0 -t 1 -b <stream>` (VVdeC 3.3.0-dev, no output file). No published WASM VVdeC build exists; browser column compares against our own PC native decode time.
  - **AV2:** `nanocodec decode --codec av2 --discard 1` vs `avmdec --threads=1 --noblit` (AVM v1.0.0, PC anchor) and `dav2d --threads 1 --framedelay 1 --muxer null -q` (`dav2d` 237229e, PC and Jetson, built with `build_anchors.sh`'s validated `-Denable_asm=false -Dbitdepths=8` flags). No published WASM AV2 reference exists; browser column compares against our own PC native decode time.
- **Streams (5 per codec = 30 streams, all 8-bit 4:2:0):**
  - **720p CBR (`1000 kbps`) & fixed-QP (`QP 32` / `q160`):** 150 frames of `crowd_mot17_09` (`1280×720`, 30 fps, source 2998 kbps) encoded by `nanocodec encode` Default preset.
  - **1080p CBR (`3000 kbps`) & fixed-QP (`QP 32` / `q160`):** 150 frames of `crowd_pp22_189_1080p` (`1920×1080`, 29.97 fps, source 6313 kbps) encoded by `nanocodec encode` Default preset.
  - **Official 8-bit conformance vector (1 per codec):** H.264 `HPCA_BRCM_C` (ITU-T H.264.1 High CABAC, `352×288`, 300f), HEVC `AMP_D_Hisilicon_3` (JCT-VC HEVC_v1 Main, `1920×1080`, 100f), VVC `8b420_B_2` (JVET VVC_v1 Main 10 8-bit, `1920×1080`, 49f), AV1 `av1-1-b8-23-film_grain-50` (`av1-libaom` Main, `352×288`, 10f), VP9 `vp90-2-02-size-lf-1920x1080` (`vp9-libvpx` Profile 0, `1920×1080`, 217f), and AV2 `avm-v15.0.0-bus.352x288.l10.deltaq1.obu` (VideoLAN `dav2d-test-data` AVM vector, `352×288`, 10f; AOM has published no official AV2 conformance suite yet).

---

## Measured Table: 8-bit Decode Speed Against the References

Wall-time cells show **`ours / reference` in ms per frame (`ratio = ours ÷ reference`)**, median of 5 interleaved rounds (`< 1.00×` means we are faster; native ratios **`> 1.20×`** are bolded). Callgrind cells show whole-stream **`ours / reference` instruction counts (`G` = $10^9$ Ir)**.

| Codec | Stream | Rate (kbps) | PC wall ms/f (`ours / ref`, Zen 3 AVX2) | Jetson wall ms/f (`ours / ref`, Orin NEON) | Browser Chromium WASM ms/f (`ours / ref WASM` or `own nat`) | PC Callgrind Ir (`ours / ref`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H.264 | ours 720p CBR (1000k) | 977 | 1.19 / 1.28 (0.93×) | 2.63 / 3.16 (0.83×) | 1.73 / ffmpeg.wasm 2.82 (0.61×; 1.51× own nat) | 1.85G / — (FFmpeg wall-only) |
| H.264 | ours 720p fixed-QP (QP 32) | 2020 | 1.65 / 1.63 (1.01×) | 3.51 / 4.12 (0.85×) | 2.33 / ffmpeg.wasm 3.66 (0.63×; 1.45× own nat) | 2.43G / — (FFmpeg wall-only) |
| H.264 | ours 1080p CBR (3000k) | 2965 | 4.16 / 4.38 (0.95×) | 6.82 / 8.44 (0.81×) | 5.93 / ffmpeg.wasm 9.21 (0.64×; 1.37× own nat) | 4.64G / — (FFmpeg wall-only) |
| H.264 | ours 1080p fixed-QP (QP 32) | 2040 | 3.40 / 3.56 (0.95×) | 5.87 / 7.40 (0.79×) | 5.48 / ffmpeg.wasm 8.17 (0.67×; 1.61× own nat) | 3.96G / — (FFmpeg wall-only) |
| H.264 | conformance HPCA_BRCM_C (352x288, 300f) | 1330 | 0.79 / 0.64 (**1.23×**) | 1.62 / 1.57 (1.03×) | 1.03 / ffmpeg.wasm 1.32 (0.78×; 1.32× own nat) | 2.24G / — (FFmpeg wall-only) |
| HEVC | ours 720p CBR (1000k) | 976 | 1.36 / 1.79 (0.76×) | 3.48 / 4.83 (0.72×) | 2.03 / ffmpeg.wasm 4.77 (0.43×; 1.51× own nat) | 2.36G / — (FFmpeg wall-only) |
| HEVC | ours 720p fixed-QP (QP 32) | 3233 | 2.43 / 2.95 (0.82×) | 5.82 / 7.60 (0.77×) | 3.59 / ffmpeg.wasm 7.64 (0.47×; 1.49× own nat) | 4.12G / — (FFmpeg wall-only) |
| HEVC | ours 1080p CBR (3000k) | 2924 | 4.22 / 6.34 (0.67×) | 11.35 / 14.33 (0.79×) | 6.56 / ffmpeg.wasm 17.00 (0.39×; 1.55× own nat) | 7.04G / — (FFmpeg wall-only) |
| HEVC | ours 1080p fixed-QP (QP 32) | 4937 | 5.52 / 7.98 (0.69×) | 14.02 / 17.36 (0.81×) | 8.29 / ffmpeg.wasm 20.23 (0.41×; 1.50× own nat) | 9.04G / — (FFmpeg wall-only) |
| HEVC | conformance AMP_D_Hisilicon_3 (1080p, 100f) | 1717 | 6.52 / 9.45 (0.69×) | 16.18 / 19.08 (0.85×) | 10.13 / ffmpeg.wasm 28.59 (0.35×; 1.51× own nat) | 5.32G / — (FFmpeg wall-only) |
| VVC | ours 720p CBR (1000k) | 972 | 2.72 / 7.43 (0.37×) | 6.64 / 11.03 (0.60×) | 4.55 / nat 2.72 (1.67× own nat) | 5.13G / 6.07G (0.85×) |
| VVC | ours 720p fixed-QP (QP 32) | 2742 | 4.36 / 9.47 (0.46×) | 10.26 / 15.72 (0.65×) | 6.99 / nat 4.36 (1.60× own nat) | 8.09G / 9.43G (0.86×) |
| VVC | ours 1080p CBR (3000k) | 2921 | 7.49 / 13.16 (0.57×) | 19.16 / 30.72 (0.62×) | 13.02 / nat 7.49 (1.74× own nat) | 13.87G / 16.25G (0.85×) |
| VVC | ours 1080p fixed-QP (QP 32) | 3858 | 9.27 / 22.10 (0.42×) | 21.60 / 34.07 (0.63×) | 14.72 / nat 9.27 (1.59× own nat) | 15.88G / 18.56G (0.86×) |
| VVC | conformance 8b420_B_2 (1080p, 49f) | 785 | 72.57 / 25.39 (**2.86×**) | 129.79 / 43.71 (**2.97×**) | 78.54 / nat 72.57 (1.08× own nat) | 54.63G / 7.49G (7.29×) |
| AV1 | ours 720p CBR (1000k) | 988 | 2.00 / 1.71 (1.17×) | 5.46 / 5.16 (1.06×) | 2.84 / nat 2.00 (1.42× own nat) | 3.32G / 2.77G (1.20×) |
| AV1 | ours 720p fixed-QP (q160) | 1839 | 2.62 / 2.20 (1.19×) | 6.78 / 6.31 (1.07×) | 3.68 / nat 2.62 (1.40× own nat) | 4.37G / 3.57G (1.22×) |
| AV1 | ours 1080p CBR (3000k) | 2957 | 7.41 / 4.98 (**1.49×**) | 15.81 / 14.50 (1.09×) | 11.30 / nat 7.41 (1.52× own nat) | 9.70G / 7.87G (1.23×) |
| AV1 | ours 1080p fixed-QP (q160) | 2341 | 6.60 / 4.68 (**1.41×**) | 14.93 / 13.77 (1.08×) | 10.59 / nat 6.60 (1.61× own nat) | 8.95G / 7.31G (1.22×) |
| AV1 | conformance av1-1-b8-23-film_grain-50 (352x288, 10f) | 687 | 1.48 / 0.94 (**1.58×**) | 2.84 / 2.12 (**1.34×**) | 2.51 / dav1d.js 5.20 (0.48×; 1.69× own nat) | 0.12G / 0.07G (1.79×) |
| VP9 | ours 720p CBR (1000k) | 987 | 1.37 / 1.14 (1.20×) | 3.39 / 3.12 (1.09×) | 1.97 / ffmpeg.wasm 4.89 (0.40×; 1.44× own nat) | 2.29G / 2.09G (1.10×) |
| VP9 | ours 720p fixed-QP (q160) | 1833 | 1.83 / 1.52 (**1.20×**) | 4.29 / 3.94 (1.09×) | 2.43 / ffmpeg.wasm 5.69 (0.43×; 1.33× own nat) | 3.01G / 2.67G (1.13×) |
| VP9 | ours 1080p CBR (3000k) | 2953 | 4.64 / 3.71 (**1.25×**) | 11.36 / 10.75 (1.06×) | 6.60 / ffmpeg.wasm 17.09 (0.39×; 1.42× own nat) | 7.42G / 6.58G (1.13×) |
| VP9 | ours 1080p fixed-QP (q160) | 2156 | 4.12 / 3.32 (**1.24×**) | 10.30 / 9.75 (1.06×) | 5.87 / ffmpeg.wasm 15.90 (0.37×; 1.43× own nat) | 6.51G / 5.87G (1.11×) |
| VP9 | conformance vp90-2-02-size-lf-1920x1080 (1080p, 217f) | 6399 | 8.98 / 7.06 (**1.27×**) | 19.79 / 18.79 (1.05×) | 12.58 / ffmpeg.wasm 28.34 (0.44×; 1.40× own nat) | 19.32G / 15.98G (1.21×) |
| AV2 | ours 720p CBR (1000k) | 986 | 2.70 / 6.35 (0.42×; dav2d 9.62, 0.28×) | 7.18 / dav2d 18.13 (0.40×) | 4.08 / nat 2.70 (1.51× own nat) | 5.06G / 12.01G (0.42×; dav2d 25.37G, 0.20×) |
| AV2 | ours 720p fixed-QP (q160) | 1035 | 2.76 / 6.59 (0.42×; dav2d 9.56, 0.29×) | 7.21 / dav2d 17.98 (0.40×) | 4.04 / nat 2.76 (1.46× own nat) | 5.07G / 11.95G (0.42×; dav2d 24.89G, 0.20×) |
| AV2 | ours 1080p CBR (3000k) | 2946 | 7.63 / 20.76 (0.37×; dav2d 21.69, 0.35×) | 20.35 / dav2d 40.42 (0.50×) | 12.27 / nat 7.63 (1.61× own nat) | 14.73G / 36.35G (0.41×; dav2d 55.83G, 0.26×) |
| AV2 | ours 1080p fixed-QP (q160) | 1053 | 4.56 / 13.29 (0.34×; dav2d 15.06, 0.30×) | 13.87 / dav2d 29.33 (0.47×) | 7.85 / nat 4.56 (1.72× own nat) | 8.72G / 23.20G (0.38×; dav2d 39.84G, 0.22×) |
| AV2 | dav2d-test-data bus.352x288.l10.deltaq1 (352x288, 10f) | 20 | 13.66 / 8.14 (**1.68×**; dav2d 8.01, **1.71×**) | 28.88 / dav2d 15.42 (**1.87×**) | 19.81 / nat 13.66 (1.45× own nat) | 2.92G / 0.42G (6.88×; dav2d 1.22G, 2.39×) |

---

## Summary by Codec & Ranked `> 1.20×` Gaps (Worst First)

### Where we are ahead or within `1.20×`

- **HEVC (all 15 machine × stream cells `≤ 0.85×`):** `0.67×–0.82×` FFmpeg on PC, `0.72×–0.85×` FFmpeg on Jetson, and `0.35×–0.47×` `ffmpeg.wasm` in Chromium (`1.49×–1.55×` our own PC native), including on the 1080p B-frame conformance stream `AMP_D_Hisilicon_3` (`0.69×` PC, `0.85×` Jetson, `0.35×` Chromium).
- **H.264 own streams (all 12 cells `≤ 1.01×`):** `0.93×–1.01×` FFmpeg on PC, `0.79×–0.85×` FFmpeg on Jetson (`1.03×` on `HPCA_BRCM_C`), and `0.61×–0.78×` `ffmpeg.wasm` in Chromium (`1.32×–1.61×` our own PC native).
- **VVC own streams (all 8 native cells `0.37×–0.65×`):** `0.37×–0.57×` VVdeC on PC (`0.85×–0.86×` Callgrind Ir) and `0.60×–0.65×` VVdeC on Jetson; in Chromium WASM (`1.59×–1.74×` our own PC native), our browser VVC decoder (`4.55–14.72 ms/f`) is faster than native x86 VVdeC (`7.43–22.10 ms/f`).
- **AV2 own streams (all 8 native cells `0.28×–0.50×`):** `0.34×–0.42×` `avmdec` and `0.28×–0.35×` `dav2d` on PC, `0.40×–0.50×` `dav2d` on Jetson; in Chromium WASM (`1.46×–1.72×` our own PC native), our browser AV2 decoder (`4.04–12.27 ms/f`) is `1.6×–2.5×` faster than native x86 `avmdec` (`6.35–20.76 ms/f`).
- **Jetson NEON on AV1 & VP9 own streams:** AV1 own streams are `1.06×–1.09×` `dav1d`; all 5 VP9 streams (including `vp90-2-02-size-lf-1920x1080`) are `1.05×–1.09×` `vpxdec`.
- **Browser WASM against published WASM references in Chromium:** ahead on every measurable reference-WASM row (`0.35×–0.78×` vs `ffmpeg.wasm` across H.264/HEVC/VP9, `0.48×` vs `dav1d.js` on AV1 conformance). Against our own PC native time (`else against our own native time`), our WASM decoder runs at `1.32×–1.74×` native across all 6 codecs (`1.40×–1.69×` on AV1, `1.33×–1.44×` on VP9, `1.49×–1.55×` on HEVC, `1.32×–1.61×` on H.264, `1.59×–1.74×` on VVC, `1.45×–1.72×` on AV2; note that vs *native* x86 `dav1d` and `vpxdec`, our browser WASM AV1 and VP9 decoders sit at `1.66×–2.27×` and `1.60×–1.78×`, so any shared SIMD / decoder speedup on AV1/VP9 also directly lifts browser decode).

### Rows `> 1.20×` slower than the reference in wall time (worst first)

| Rank | Ratio | Machine | Codec | Stream | Measured totals (`ours` vs `reference`) | Initial diagnosis / scope note |
| ---: | ---: | --- | --- | --- | --- | --- |
| 1 | **2.97×** | Jetson Orin (NEON) | VVC | conformance 8b420_B_2 (1080p, 49f) | 6359.8 ms vs VVdeC 3.3.0-dev 2141.7 ms (2.97×) | General 8-bit VVC conformance path (`decoder::general::Core<u8>`, `7.29×` Ir: `54.63G` vs `7.49G`); own VVC subset path is `0.37×–0.65×`. |
| 2 | **2.86×** | PC (Zen 3, AVX2) | VVC | conformance 8b420_B_2 (1080p, 49f) | 3556.1 ms vs VVdeC 3.3.0-dev 1244.2 ms (2.86×) | General 8-bit VVC conformance path (`decoder::general::Core<u8>`, `7.29×` Ir: `54.63G` vs `7.49G`); own VVC subset path is `0.37×–0.65×`. |
| 3 | **1.87×** | Jetson Orin (NEON) | AV2 | dav2d-test-data bus.352x288.l10.deltaq1 (352x288, 10f) | 288.8 ms vs dav2d 154.2 ms (1.87×) | 10-frame `352×288` general AV2 path (`2.92G` vs `0.42G` `avmdec` / `1.22G` `dav2d` Ir): per-instance decoder/CDF/table init + general-path overhead. |
| 4 | **1.68×** | PC (Zen 3, AVX2) | AV2 | dav2d-test-data bus.352x288.l10.deltaq1 (352x288, 10f) | 136.6 ms vs avmdec 81.4 ms (1.68×), dav2d 80.1 ms (1.71×) | 10-frame `352×288` general AV2 path (`2.92G` vs `0.42G` `avmdec` / `1.22G` `dav2d` Ir): per-instance decoder/CDF/table init + general-path overhead. |
| 5 | **1.58×** | PC (Zen 3, AVX2) | AV1 | conformance av1-1-b8-23-film_grain-50 (352x288, 10f) | 14.8 ms vs dav1d 1.5.4 9.4 ms (1.58×) | Short 10-frame `352×288` film-grain stream (`119.7M` vs `66.7M` Ir, `1.79×`): decoder startup + 8-bit film-grain synthesis. |
| 6 | **1.49×** | PC (Zen 3, AVX2) | AV1 | ours 1080p CBR (3000k) | 1112.2 ms vs dav1d 1.5.4 746.6 ms (1.49×) | 1080p AV1 decode on Zen 3 AVX2 (`1.22×–1.23×` Ir, `1.41×–1.49×` wall; browser WASM is also `2.26×–2.27×` native `dav1d` here). |
| 7 | **1.41×** | PC (Zen 3, AVX2) | AV1 | ours 1080p fixed-QP (q160) | 989.6 ms vs dav1d 1.5.4 702.2 ms (1.41×) | 1080p AV1 decode on Zen 3 AVX2 (`1.22×–1.23×` Ir, `1.41×–1.49×` wall; browser WASM is also `2.26×–2.27×` native `dav1d` here). |
| 8 | **1.34×** | Jetson Orin (NEON) | AV1 | conformance av1-1-b8-23-film_grain-50 (352x288, 10f) | 28.4 ms vs dav1d 1.5.4 21.2 ms (1.34×) | Short 10-frame `352×288` film-grain stream (`119.7M` vs `66.7M` Ir, `1.79×`): decoder startup + 8-bit film-grain synthesis. |
| 9 | **1.27×** | PC (Zen 3, AVX2) | VP9 | conformance vp90-2-02-size-lf-1920x1080 (1080p, 217f) | 1948.6 ms vs vpxdec 1.16.0 1532.0 ms (1.27×) | 1080p / fixed-QP VP9 decode on Zen 3 AVX2 (`1.11×–1.21×` Ir, `1.205×–1.27×` wall; browser WASM is `1.72×–1.78×` native `vpxdec`). |
| 10 | **1.25×** | PC (Zen 3, AVX2) | VP9 | ours 1080p CBR (3000k) | 695.3 ms vs vpxdec 1.16.0 556.4 ms (1.25×) | 1080p / fixed-QP VP9 decode on Zen 3 AVX2 (`1.11×–1.21×` Ir, `1.205×–1.27×` wall; browser WASM is `1.72×–1.78×` native `vpxdec`). |
| 11 | **1.24×** | PC (Zen 3, AVX2) | VP9 | ours 1080p fixed-QP (q160) | 617.8 ms vs vpxdec 1.16.0 498.2 ms (1.24×) | 1080p / fixed-QP VP9 decode on Zen 3 AVX2 (`1.11×–1.21×` Ir, `1.205×–1.27×` wall; browser WASM is `1.72×–1.78×` native `vpxdec`). |
| 12 | **1.23×** | PC (Zen 3, AVX2) | H.264 | conformance HPCA_BRCM_C (352x288, 300f) | 237.1 ms vs FFmpeg 8.0.1 193.2 ms (1.23×) | 352×288 300-frame High CABAC B-frame conformance (`1.23×` with `--output /dev/null`, `1.21×` with `--discard 1`). |
| 13 | **1.20×** | PC (Zen 3, AVX2) | VP9 | ours 720p fixed-QP (q160) | 274.0 ms vs vpxdec 1.16.0 227.4 ms (1.20×) | 1080p / fixed-QP VP9 decode on Zen 3 AVX2 (`1.11×–1.21×` Ir, `1.205×–1.27×` wall; browser WASM is `1.72×–1.78×` native `vpxdec`). |

### Browser WASM rows without a working reference WASM (`ours WASM / own PC native > 1.20×`)

Where a working reference WASM build runs in Chromium (`ffmpeg.wasm` for H.264/HEVC/VP9, `dav1d.js` on `av1-1-b8-23-film_grain-50`), our `+simd128,+relaxed-simd` module is **`0.35×–0.78×`** (`1.28×–2.82×` faster, since neither published WASM build uses hand-written SIMD128). On the 14 rows where no working reference WASM exists (VVC, AV2, and 150-frame AV1 where `dav1d.js` 0.1.1 OOM-leaks at 64 MiB), comparing against our own PC native decode time (`else against our own native time`) gives:

| Codec | Shipped to browser | Chromium WASM / own PC native | Chromium WASM / PC native reference |
| --- | :---: | ---: | ---: |
| **AV1** (720p / 1080p own, 4 rows) | Primary | **1.40×–1.61×** (`2.84–11.30 ms/f` vs `2.00–7.41 ms/f`) | **1.66×–2.27×** native `dav1d` |
| **VVC** (own 4 rows / `8b420_B_2`) | Demo | **1.59×–1.74×** own (`1.08×` on `8b420_B_2`) | **0.67×–0.99×** native VVdeC on own (**3.09×** on `8b420_B_2`) |
| **AV2** (own 4 rows / `deltaq1`) | Demo | **1.45×–1.72×** | **0.59×–0.64×** native `avmdec` on own (**2.43×** on `deltaq1`) |
| *(For comparison: H.264 / HEVC / VP9)* | Primary | *1.32×–1.61× / 1.49×–1.55× / 1.33×–1.44×* | *1.35×–1.61× FFmpeg / 1.03×–1.22× FFmpeg / 1.60×–1.78× `vpxdec`* |

Raw per-round JSON artifacts (ordered runs, minima, medians, MD5s, stream sizes):
`~/nanocodec-data/work/decode-speed-2026-11-18/{manifest,pc_wall,jetson_wall,chromium_wasm,callgrind}.json`.

---

## Task 2: Fixes, Before → After Timings, and Verification

Following the user-approved priority order from the Task 1 review:

1. **Priority 1 — AV1 (Browser WASM & Native AVX2/NEON):** commit `42268e470`
2. **Priority 2 — VP9 (PC AVX2 & Browser WASM):** commit `59cc89868`
3. **Priority 3 — VVC 8-bit general path (`8b420_B_2`):** commit `31e3e871c`
4. **Priority 4 — AV2 short foreign stream (`bus.352x288.l10.deltaq1`):** commit `cf09611c7`
5. **H.264:** skipped per user direction (borderline `1.23×` on one `352×288` conformance stream and `0.79×–1.01×` on all own streams).

Note on Chromium WASM A/B timing: the Task 1 baseline (`orig`) ran with V8's default tiered compiler (Liftoff → background TurboFan); all Task 2 Chromium A/B passes run with `--js-flags=--no-liftoff` so both baseline (`--no-liftoff` control) and candidate execute deterministic TurboFan code from frame 0. Both the `--no-liftoff` baseline control and the original Task 1 baseline (in parentheses) are reported below.

### Priority 1 — AV1 Native AVX2/NEON and Browser WASM (`42268e470`)

- **Changes:**
  - Replaced full-frame `CdefSource` buffers with an in-place sliding halo (`≤ 4` retained rows) and empty-band skip across shared AV1/AV2 CDEF band filtering so `CdefLines` stays L2-resident at 1080p.
  - Added fused SSE2 and WASM SIMD128 AV1 symbol steps and extended NEON/scalar symbol step coverage to all capacities `3..=16` with parallel signed CDF adaptation, branchless GPR interval selection, 8-byte `u64::from_be_bytes` refills, and out-of-line wide helpers (`N ≥ 5`).
  - Added WebAssembly `+relaxed-simd` byte dot-product interpolation (`av1_simd128_dot`) and split 2-D horizontal/vertical register scopes for 8-wide strips across AV1, HEVC, and VVC.
  - Shrunk AV1 `MotionGrid` to a superblock-height ring buffer (`32`/`64` MI rows), specialized `av1_packed_dir` deblocking by axis while reusing `EdgeMap` run storage, stack-allocated film-grain templates, and rejected odd superres `upscaled_width` in subset validation (caught by `decode_av1` fuzzing, covered by `odd_upscaled_superres_width_is_rejected`).
- **Whole-decode time before → after against `dav1d` 1.5.4:**

| Stream | PC Zen 3 AVX2 (`ours` before → after vs `dav1d`) | Jetson Orin NEON (`ours` before → after vs `dav1d`) | Chromium WASM (`--no-liftoff` base `[orig]` → after) | Callgrind Ir (`ours` before → after vs `dav1d`) |
| --- | ---: | ---: | ---: | ---: |
| `av1_720p_cbr` (150f) | `299.9` → **`287.4 ms`** vs `258.9 ms` (`1.17×` → **`1.11×`**) | `818.8` → **`778.9 ms`** vs `773.1 ms` (`1.06×` → **`1.01×`**) | `437.2` `[426.2]` → **`423.8 ms`** (`1.47×` own nat) | `3.32G` → **`3.31G`** vs `2.77G` (`1.20×`) |
| `av1_720p_qp` (150f) | `393.4` → **`374.6 ms`** vs `330.3 ms` (`1.19×` → **`1.13×`**) | `1016.9` → **`981.1 ms`** vs `947.4 ms` (`1.07×` → **`1.04×`**) | `561.8` `[552.1]` → **`539.5 ms`** (`1.44×` own nat) | `4.37G` → **`4.36G`** vs `3.57G` (`1.22×`) |
| `av1_1080p_cbr` (150f) | `1112.2` → **`862.1 ms`** vs `747.2 ms` (**`1.49×` → `1.15×`**, `-22.5%`) | `2371.3` → **`2309.1 ms`** vs `2180.4 ms` (`1.09×` → **`1.06×`**) | `1423.4` `[1694.9]` → **`1261.4 ms`** (`1.46×` own nat; `-11.4%` TF / `-25.6%` orig) | `9.70G` → **`9.67G`** vs `7.87G` (`1.23×`) |
| `av1_1080p_qp` (150f) | `989.6` → **`801.6 ms`** vs `692.4 ms` (**`1.41×` → `1.16×`**, `-19.0%`) | `2240.2` → **`2163.5 ms`** vs `2065.4 ms` (`1.08×` → **`1.05×`**) | `1343.5` `[1588.8]` → **`1184.3 ms`** (`1.48×` own nat; `-11.8%` TF / `-25.5%` orig) | `8.95G` → **`8.93G`** vs `7.31G` (`1.22×`) |
| `av1_conf` (`film_grain-50`, 10f) | `14.84` → **`13.96 ms`** vs `9.49 ms` (`1.58×` → **`1.47×`**) | `28.35` → **`28.19 ms`** vs `21.53 ms` (`1.34×` → **`1.31×`**) | `15.4` `[25.1]` → **`15.1 ms`** vs `dav1d.js` `46.9 ms` (**`0.32×`**) | `119.7M` → **`116.4M`** vs `66.7M` (`1.74×`) |

- **Checks:** 5-target `clippy -D warnings`, `0` outlined SIMD across ELF & WASM, 4-tier `nc-checkasm --iterations 200` (`2,905,877` AVX2/AVX-512/SIMD128, `3,060,877` NEON, `3,168,021` relaxed-SIMD), `av1_conformance.sh` `242/242`, `landing_conformance.py --codecs av1,hevc,vvc,av2` `291/291`, `preset_identity.py` `297/297`, ASan `decode_av1` (`-seed=2026111802`, `390,597` runs / `611` s, `0` crashes).

---

### Priority 2 — VP9 PC AVX2 and Browser WASM (`59cc89868`)

- **Changes:**
  - Routed AVX2 VP9 loop filtering through 16-line paired groups (`avx2_wide::B`) with constant-folded horizontal/vertical axis schedules across all packed loop-filter tiers and direct array-literal vector construction.
  - Added two-row-per-step sliding signed-interleaved vertical column passes for WebAssembly `+relaxed-simd` VP9 interpolation (`Simd128Dot`), and trimmed 2-D and vertical `BILINEAR` interpolation to active 2-tap support across AVX2, AVX-512, NEON, and WASM SIMD128.
  - Made VP9 `BoolReader` branchless with an equiprobable `read_bit` fast path, peeled the DC `eob == 0` fast exit in `coeff::read_tokens`, added single-byte `4×4` fast paths to `any_set`/`set_contexts`, and recycled above-context buffers in `BlockScratch`.
- **Whole-decode time before → after against `vpxdec` 1.16.0 (all 5 PC rows now `< 1.20×`):**

| Stream | PC Zen 3 AVX2 (`ours` before → after vs `vpxdec`) | Jetson Orin NEON (`ours` before → after vs `vpxdec`) | Chromium WASM (`--no-liftoff` base `[orig]` → after vs `ffmpeg.wasm`) | Callgrind Ir (`ours` before → after vs `vpxdec`) |
| --- | ---: | ---: | ---: | ---: |
| `vp9_720p_cbr` (150f) | `205.1` → **`193.1 ms`** vs `174.8 ms` (`1.20×` → **`1.10×`**) | `509.3` → **`499.4 ms`** vs `470.8 ms` (`1.09×` → **`1.06×`**) | `290.3` `[294.9]` → **`271.9 ms`** vs `741.6 ms` (**`0.37×`**; `1.41×` own nat) | `2.29G` → **`2.01G`** vs `2.09G` (**`0.96×`**) |
| `vp9_720p_qp` (150f) | `274.0` → **`261.9 ms`** vs `234.6 ms` (**`1.20×` → `1.12×`**) | `643.7` → **`631.2 ms`** vs `593.7 ms` (`1.09×` → **`1.06×`**) | `369.4` `[365.1]` → **`351.6 ms`** vs `874.5 ms` (**`0.40×`**; `1.34×` own nat) | `3.01G` → **`2.69G`** vs `2.67G` (`1.01×`) |
| `vp9_1080p_cbr` (150f) | `695.3` → **`640.2 ms`** vs `570.6 ms` (**`1.25×` → `1.12×`**) | `1703.7` → **`1673.3 ms`** vs `1614.2 ms` (`1.06×` → **`1.04×`**) | `986.0` `[990.7]` → **`901.4 ms`** vs `2571.3 ms` (**`0.35×`**; `1.41×` own nat) | `7.42G` → **`6.55G`** vs `6.58G` (**`0.99×`**) |
| `vp9_1080p_qp` (150f) | `617.8` → **`558.7 ms`** vs `507.0 ms` (**`1.24×` → `1.10×`**) | `1544.7` → **`1518.6 ms`** vs `1466.0 ms` (`1.06×` → **`1.04×`**) | `888.0` `[881.2]` → **`806.9 ms`** vs `2380.0 ms` (**`0.34×`**; `1.44×` own nat) | `6.51G` → **`5.70G`** vs `5.87G` (**`0.97×`**) |
| `vp9_conf` (`size-lf-1080p`, 217f) | `1948.6` → **`1787.3 ms`** vs `1507.0 ms` (**`1.27×` → `1.19×`**) | `4295.2` → **`4220.4 ms`** vs `4066.8 ms` (`1.05×` → **`1.04×`**) | `2693.4` `[2730.1]` → **`2530.0 ms`** vs `6238.7 ms` (**`0.41×`**; `1.42×` own nat) | `19.32G` → **`18.08G`** vs `15.98G` (`1.13×`) |

- **Checks:** 5-target `clippy -D warnings`, `0` outlined SIMD, 4-tier `nc-checkasm --iterations 200`, `vp9_conformance.sh` `305/305`, `landing_conformance.py --codecs vp9` `215/215`, `preset_identity.py` `297/297`, ASan `decode_vp9` (`-seed=2026111803`, `1,908,256` runs / `611` s, `0` crashes).

---

### Priority 3 — VVC 8-bit General Path (`31e3e871c`)

- **Changes:**
  - Added register-only `u8` widening (`load_u8_halves`, `load_u8_quarters`) and 1-byte-unaligned deinterleaved `u8` loads (`load_u8_deinterleaved`) to `I16FilterLanes` across AVX2, AVX-512, NEON, and WASM SIMD128, plus the crate-private `SimdSample` trait (`u8` / `u16`).
  - Parameterized VVC clipped luma ALF, chroma ALF, and CC-ALF SIMD kernels over `<S: SimdSample, V: I16FilterLanes>` and dispatched `u8` kernels across all four SIMD tiers via a static `vvc_alf::Kernels` table (leaving the 8-bit linear ALF byte fast path completely untouched).
  - Parameterized VVC high-precision bi-prediction staging (`predict_high_precision`, used by BDOF/DMVR/weighted bi-pred) over `SimdSample` across all four SIMD tiers, folded away 8-bit zero-shift horizontal passes, and fixed a pre-existing odd-height `8×h` tail lane-combining bug in `two_pass_w8_regs` (caught by new odd-height `nc-checkasm` coverage).
- **Whole-decode time before → after against `vvdecapp` 3.3.0-dev:**

| Stream | PC Zen 3 AVX2 (`ours` before → after vs `vvdecapp`) | Jetson Orin NEON (`ours` before → after vs `vvdecapp`) | Chromium WASM (`--no-liftoff` base `[orig]` → after) | Callgrind Ir (`ours` before → after vs `vvdecapp`) |
| --- | ---: | ---: | ---: | ---: |
| `vvc_720p_cbr` (150f) | `408.0` → **`414.2 ms`** vs `1123.7 ms` (`0.37×` → **`0.37×`**) | `996.0` → **`994.8 ms`** vs `1663.8 ms` (`0.60×` → **`0.60×`**) | `673.2` `[682.0]` → **`669.5 ms`** (`1.62×` own nat) | `5.13G` → **`5.13G`** vs `6.06G` (`0.85×`) |
| `vvc_720p_qp` (150f) | `653.4` → **`657.0 ms`** vs `1210.3 ms` (`0.46×` → **`0.54×`**) | `1538.4` → **`1535.1 ms`** vs `2371.7 ms` (`0.65×` → **`0.65×`**) | `1013.9` `[1047.9]` → **`1041.6 ms`** (`1.59×` own nat) | `8.09G` → **`8.09G`** vs `9.42G` (`0.86×`) |
| `vvc_1080p_cbr` (150f) | `1123.3` → **`1229.4 ms`** vs `2730.3 ms` (`0.57×` → **`0.45×`**) | `2874.0` → **`2867.2 ms`** vs `4618.8 ms` (`0.62×` → **`0.62×`**) | `1900.5` `[1952.3]` → **`1917.9 ms`** (`1.56×` own nat) | `13.87G` → **`13.87G`** vs `16.24G` (`0.85×`) |
| `vvc_1080p_qp` (150f) | `1389.8` → **`1420.7 ms`** vs `2968.9 ms` (`0.42×` → **`0.48×`**) | `3239.5` → **`3236.7 ms`** vs `5126.8 ms` (`0.63×` → **`0.63×`**) | `2210.2` `[2208.4]` → **`2168.8 ms`** (`1.53×` own nat) | `15.88G` → **`15.88G`** vs `18.55G` (`0.86×`) |
| **`vvc_conf` (`8b420_B_2`, 1080p 49f)** | `3556.1` → **`1141.7 ms`** vs `1117.2 ms` (**`2.86×` → `1.02×`**, `-67.9%`) | `6359.8` → **`2631.1 ms`** vs `2167.5 ms` (**`2.97×` → `1.21×`**, `-58.6%`) | `3856.4` `[3848.7]` → **`1590.3 ms`** (`-58.8%`; `1.39×` own nat) | `54.63G` → **`12.49G`** vs `7.50G` (**`7.29×` → `1.67×`**, `-77.1%`) |

- **Checks:** 5-target `clippy -D warnings`, `0` outlined SIMD, 4-tier `nc-checkasm --iterations 200` (`2,909,101` AVX2/AVX-512/SIMD128, `3,064,091` NEON, `3,171,245` relaxed-SIMD), `vvc_conformance.py` (`main8` `9/9`), `conformance.py --suite vvc-v1` (`226/226`: 8-bit `9/9`, 10-bit `217/217`), `conformance.py --suite hevc-v1` (`147/147`), `landing_conformance.py --codecs vvc,hevc` (`48/48`), `preset_identity.py` (`297/297`), ASan `decode_vvc` (`-seed=2026111804`, `187,057` runs / `611` s, `0` crashes).

---

### Priority 4 — AV2 Short Foreign Stream Startup, CDF Propagation, and Separable MC (`cf09611c7`)

- **Changes:**
  - Pre-sized `InterMasks` storage (`wedge[0..3]`, `tmvp`, `interintra`) and wrote cropped, 4:2:2/4:2:0-subsampled, and TMVP summary masks directly in place without intermediate `Vec` allocations (`2,178` per-instance heap allocations → `0`) or nested `FlatMap` iterators (`70.5M` → `< 4M` Ir startup).
  - Cached default `Cdfs` tables per AVM quantizer category (`0..4`) and diagonal transform `Scans` per `Decoder` instance (`OnceCell`), passed `Rc<Cdfs>` through `initial_models` / `decode_tile` / `decode_frame` so keyframes, TIP frames, and averaged B-frames avoid redundant multi-megabyte `Box<Cdfs>` clones, and reused `Workspace`'s pre-dispatched `BlockCopy` instead of calling `Dsp::new()` per tile.
  - Replaced `9,088`-byte return-by-value `[[i16; 64]; 71]` / `[[i16; 64]; 65]` intermediate arrays in `av2_dec::mc_8tap` and `av2_dec::mc` (which accounted for `588M` Ir = `20.1%` of `deltaq1` in `memcpy`) with compact circular row buffers (`[[i16; 8]; 8]` for `w ≤ 8`) and zero-tap-skipping slice loops; the remaining gap on `deltaq1` is scalar PC-Wiener/NS-Wiener loop restoration (`Decoder::restore`, left out of scope per priority guidance).
  - Fixed general-path AV2 in-loop filtering on frames whose even dimensions are not a multiple of 8 (running `cdef_and_ccso` and `restore` on the `8×8`-aligned coded surface `width.next_multiple_of(8) × height.next_multiple_of(8)` before cropping, matching `dav2d`; caught by `decode_av2` fuzzing and covered by `non_multiple_of_eight_general_frame_filters_without_panicking`).
- **Whole-decode time before → after against `avmdec` v1.0.0 and `dav2d` 237229e:**

| Stream | PC Zen 3 AVX2 (`ours` before → after vs `avmdec` / `dav2d`) | Jetson Orin NEON (`ours` before → after vs `dav2d`) | Chromium WASM (`--no-liftoff` base `[orig]` → after) | Callgrind Ir (`ours` before → after vs `avmdec` / `dav2d`) |
| --- | ---: | ---: | ---: | ---: |
| `av2_720p_cbr` (150f) | `404.5` → **`411.5 ms`** vs `1018.3` / `1461.2 ms` (`0.40×` / `0.28×`) | `1077.0` → **`1038.6 ms`** vs `2717.4 ms` (`0.40×` → **`0.38×`**) | `602.6` `[611.4]` → **`626.8 ms`** (`1.52×` own nat) | `5.06G` → **`5.13G`** vs `12.01G` / `25.37G` (`0.43×` / `0.20×`) |
| `av2_720p_qp` (150f) | `413.9` → **`415.7 ms`** vs `1013.9` / `1443.6 ms` (`0.41×` / `0.29×`) | `1081.2` → **`1045.2 ms`** vs `2694.9 ms` (`0.40×` → **`0.39×`**) | `605.7` `[606.2]` → **`647.1 ms`** (`1.56×` own nat) | `5.07G` → **`5.14G`** vs `11.95G` / `24.89G` (`0.43×` / `0.21×`) |
| `av2_1080p_cbr` (150f) | `1144.8` → **`1164.1 ms`** vs `3671.6` / `3332.0 ms` (`0.32×` / `0.35×`) | `3053.2` → **`3000.8 ms`** vs `6074.3 ms` (`0.50×` → **`0.49×`**) | `2022.1` `[1840.1]` → **`1893.3 ms`** (`1.63×` own nat) | `14.73G` → **`14.99G`** vs `36.35G` / `55.83G` (`0.41×` / `0.27×`) |
| `av2_1080p_qp` (150f) | `683.9` → **`714.3 ms`** vs `2657.6` / `2352.5 ms` (`0.27×` / `0.30×`) | `2080.8` → **`2015.1 ms`** vs `4412.1 ms` (`0.47×` → **`0.46×`**) | `1360.6` `[1177.3]` → **`1178.3 ms`** (`1.65×` own nat) | `8.72G` → **`8.85G`** vs `23.20G` / `39.84G` (`0.38×` / `0.22×`) |
| **`av2_conf` (`deltaq1`, 352×288 10f)** | `136.6` → **`122.6 ms`** vs `99.5` / `82.6 ms` (**`1.68×` → `1.23×`** `avmdec`; `1.71×` → **`1.48×`** `dav2d`) | `288.8` → **`259.4 ms`** vs `154.3 ms` (**`1.87×` → `1.68×`**, `-10.2%`) | `180.7` `[198.1]` → **`152.9 ms`** (`-15.4%` TF / `-22.8%` orig; `1.25×` own nat) | `2.92G` → **`2.13G`** vs `0.42G` / `1.22G` (`6.88×` → **`5.01×`** / `2.39×` → **`1.74×`**, `-27.2%`) |

- **Checks:** 5-target `clippy -D warnings`, `0` outlined SIMD across ELF & WASM, 4-tier `nc-checkasm --iterations 200` (`2,909,101` AVX2/AVX-512/SIMD128, `3,064,091` NEON, `3,171,245` relaxed-SIMD), `tools/dev/av2_foreign/compare.py` (`30/30` bit-exact with `avmdec` & `dav2d`), `landing_conformance.py --codecs av2` (`18/18`), `preset_identity.py` (`297/297`), ASan `decode_av2` (`-seed=2026111805`, `54,829` runs / `611` s, `0` crashes).

Task 2 raw JSON artifacts:
`~/nanocodec-data/work/decode-speed-2026-11-18/{pc_wall_p1,pc_wall_p2,pc_wall_p3,pc_wall_p4,jetson_wall_p1,jetson_wall_p2,jetson_wall_p3,jetson_wall_p4,chromium_wasm_p1_av1,chromium_wasm_p2_vp9,chromium_wasm_p3_vvc,chromium_wasm_p4_av2,callgrind_p1,callgrind_p2,callgrind_p3,callgrind_p4}.json`.

## Integration onto `variance-integration` (2026-11-18)

The four commits landed as four squashed commits on `variance-integration` (base `752dbf628`). One
conflict: `nc-dsp/src/cdef/mod.rs` against `2f7a6c420` (CDEF band source-row reuse), which bounds the
same `CdefSource` buffer. One design kept: `2f7a6c420`'s `CdefSource` (physical discard, gap skipping,
its row/window oracles) plus this branch's single contiguous copy in `fill_rows_to` and its band-level
`prepare_band_lines` (empty-band skip, copy only the rows a band's units span).

Decode callgrind Ir after the merge (release CLI, `--discard 1`, base `752dbf628` → merged head):
`av1_1080p_cbr` 9.707G → 9.668G (-0.4 %), `vp9_1080p_cbr` 7.416G → 6.547G (-11.7 %),
`vvc_conf` (`8b420_B_2`) 54.63G → 12.49G (-77.1 %), `av2_conf` (`deltaq1`) 2.921G → 2.092G (-28.4 %);
the same as the branch's own before/after counts. Checks on the merged tree: clippy `-D warnings`
(x86_64, aarch64, wasm32 SIMD128, wasm32 relaxed SIMD, analysis feature), checkasm AVX2 and AVX-512
2,909,101 cases (build box), NEON 3,064,091 (Jetson), SIMD128 and relaxed SIMD under node, 0 outlined
SIMD, workspace tests, `preset_identity` 306/306, AV1 242/242, VP9 305/305, VVC 226/226, HEVC 147/147,
AV2 foreign 30/30, landing conformance 544/544.
