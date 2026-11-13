# Anchor encoders, reference decoders and metric tools

Pinned builds of the external tools used by the evaluation suite
(`eval/README.md` "Methodology"). Every version (git tag + commit hash) is in
`versions.env`.

| Tool | Version | Installed binaries | Role |
|---|---|---|---|
| openh264 | v2.7.0 | `h264enc`, `h264dec`, `share/openh264/*.cfg`, `include/wels`, `lib/libopenh264.a` | baseline encoder; extra H.264 decoder |
| openh264-harness | (in repo) | `openh264-harness` | our openh264 driver, `tools/openh264-harness/`, built against the pinned openh264 |
| x264 | stable @ b35605a (core 165) | `x264` | H.264 reference point |
| x265 | 4.2 | `x265` | HEVC reference |
| kvazaar | v2.3.2 | `kvazaar` | HEVC reference |
| SVT-AV1 | v4.3.0 | `SvtAv1EncApp` | AV1 reference |
| libaom | v3.15.3 | `aomenc`, `aomdec` | AV1 RTC reference; AV1 reference decoder |
| libaom (`aomsvc`, opt-in) | v3.15.3 | `svc_encoder_rtc` | AV1 spatial SVC reference (the real-time SVC example WebRTC's AV1 SVC follows; `eval/nceval/svc_chart.py`) |
| dav1d | 1.5.4 | `dav1d` | AV1 decoder |
| JM | 19.0 | `ldecod` | H.264 reference decoder |
| HM | 19.0 | `TAppDecoder` | HEVC reference decoder |
| libvmaf | v3.2.1 | `vmaf`, `share/vmaf/model/` (incl. `vmaf_v0.6.1neg.json`) | metrics |
| clang-format | 23.1.5 (pip) | `clang-format` | code formatting |

## Where

`$HOME/nanocodec-data/anchors/<platform>/` with platform `x86_64` (PC) or
`aarch64` (Jetson):

```
src/<tool>-<ref>/      pristine shallow git checkouts
build/<tool>-<ref>/    build trees; .stamp marks a finished build
build/logs/<tool>.log  full log of the last fetch/build of each tool
bin/ lib/ include/ share/
venv/                  meson, ninja, clang-format (pip, pinned)
```

Binaries are statically linked against the tools' own libraries (only libc,
libm, libstdc++ remain), so `bin/` can be used directly.

## Build / rebuild

```sh
tools/anchors/build_anchors.sh                 # everything, default prefix
tools/anchors/build_anchors.sh x265 hm         # just these
tools/anchors/build_anchors.sh --force libaom  # rebuild even if up to date
tools/anchors/smoke_test.sh                    # versions + encode/decode check
```

The build is idempotent: a tool is rebuilt only if its pinned commit or its
`build_<tool>` recipe changed (for the harness: its sources or the pinned
openh264). To bump a tool, edit its `_REF` and `_COMMIT`
in `versions.env` (`git ls-remote <url> 'refs/tags/<tag>^{}'`).

**Jetson** (8 GB shared RAM, production service on port 8077 — keep the
build gentle). The repo isn't cloned there; copy `tools/anchors` and
`tools/openh264-harness` over, keeping them side by side:

```sh
rsync -a tools/anchors tools/openh264-harness \
    $NC_JETSON:~/nanocodec-tools/
ssh $NC_JETSON 'cd ~/nanocodec-tools/anchors &&
    PATH=$HOME/.local/bin:$PATH nice -n 19 ./build_anchors.sh -j 4'
```

`PATH` must put `~/.local/bin` first: non-login ssh shells don't, and
`/usr/bin/cmake` there is 3.22 (cmake 4.1 lives in `~/.local/bin`).

`smoke_test.sh` encodes 10 frames of `testsrc2` (352×288) with each encoder
at fixed QP (plus `openh264-harness` in the "mid" config with 2 temporal
layers, CAVLC and CABAC), decodes every stream with the reference decoder(s) and ffmpeg
(H.264: JM + ffmpeg + openh264; HEVC: HM + ffmpeg; AV1: aomdec + dav1d +
ffmpeg), requires identical MD5s, and runs `vmaf` with the NEG model.

## Running anchors single-threaded

The plan requires every anchor to run on one thread, pinned with
`taskset -c <cpu>`:

| Tool | Single-thread flags |
|---|---|
| h264enc | `-threadIdc 1` (cfg: `MultipleThreadIdc 1`) |
| openh264-harness | `--threads 1` (the default) |
| x264 | `--threads 1` (lookahead then runs in the same thread) |
| x265 | `--pools none --frame-threads 1 --no-wpp` |
| kvazaar | `--threads 0 --owf 0` (0 = all work on the main thread) |
| SvtAv1EncApp | `--lp 1` — still starts helper threads; only `taskset` truly limits it to one core |
| aomenc | `--threads=1` (row-mt has no effect with one thread) |
| aomdec | `--threads=1` |
| dav1d | `--threads 1` |
| ffmpeg-decode (gate decode, pinned n8.0.1 build) | `-threads 1` |
| ldecod, TAppDecoder | single-threaded already; HM may link OpenMP, so set `OMP_NUM_THREADS=1` |
| vmaf | `--threads 1` |

### x264 / x265 context anchors (Mark, 2026-11-08)

One-time context comparisons on VAL-150 and in the scoreboard; GPL tools run
as external binaries only (no code is copied). Pinned standalone builds from
`versions.env`: x264 stable @ b35605a (reports `0.165.x`, core 165) and x265
4.2 @ e444744 (8-bit, AVX2). Adapters in `eval/nceval/encoders/anchors.py`:

| Anchor | Command (per point; fixed QP, keyint = our intra period) |
|---|---|
| x264-ultrafast, x264-veryfast (also x264-superfast) | `x264 --preset <p> --tune zerolatency --bframes 0 --rc-lookahead 0 --no-scenecut --keyint N --min-keyint N --threads 1 --qp Q --input-res WxH --fps F --frames n --output out.264 in.yuv` |
| x265-ultrafast, x265-superfast | `x265 --preset <p> --tune zerolatency --bframes 0 --rc-lookahead 0 --no-scenecut --keyint N --min-keyint N --frame-threads 1 --pools none --no-wpp --qp Q --input-res WxH --input-depth 8 --fps F --frames n --input in.yuv --output out.hevc` |

Curves are fingerprinted by the pinned commit and the command, so they only
re-run if the anchor build or flags change.

## openh264 CLI coverage

`h264enc` (command line or cfg file) can set everything the "mid" and
"preview" configs need: usage type (`-utype` / `UsageType`), complexity
(`-complexity` / `ComplexityMode`), entropy (`-cabac` /
`EntropyCodingModeFlag`), RC mode (`-rc`, `-1` = off), fixed QP (`-lqp
<layer> <qp>` / `InitialQP` in the layer cfg), temporal layers (`-numtl`),
frame skip (`-fs`), background detection (`-bgd`), AQ (`-aq`), scene change
(`-scene`), denoise (`-denois`), LTR (`-ltr`), SPS/PPS id strategy (`-spsid`,
0 = `CONSTANT_ID`), threads, intra period (`-iper`), min/max QP, target/max
bitrate, profile (`ProfileIdc` in the layer cfg) and recon output (`-drec`).

Watch-outs:
- `h264enc` does **not** start from the library defaults: it overrides
  them (RC quality mode, frame skip on, `INCREASING_ID`, 4 spatial layers…),
  and the stock `testbin/welsenc.cfg` turns LTR on. Always pass every
  setting explicitly (and `-numl 1` with that layer's size/fps).
- The layer cfg's `ProfileIdc 66` (Baseline) must become 77 or 0 for a
  CABAC run.
- "Fixed QP" with RC off is not one QP: openh264 adds per-temporal-layer QP
  cascading and, with AQ on, a per-frame/per-MB AQ delta in P frames.

What the CLI **cannot** do (reasons for `openh264-harness`; the current
harness does not do these yet either):
- change settings mid-stream (bitrate, frame rate, GOP — the preview
  path's still-scene GOP/fps adaptation) or force an IDR on demand
  (`ForceIntraFrame`), LTR recovery requests (`ENCODER_LTR_RECOVERY_REQUEST`);
- feed real per-frame timestamps (frames are timed from the fixed input
  frame rate, which matters for `RC_TIMESTAMP_MODE` and the preview path's
  irregular sampling);
- report per-frame stats (frame type, size, QP, temporal id) in a
  machine-readable form, or time just the encode calls.

## Known quirks

- **conda on the PC:** an active conda env puts its own cmake/ffmpeg/libs
  first on `PATH`. HM's optional OpenSSL search found conda's `libssl.so`
  and broke the static link; the recipe turns that search off. No other
  tool links anything outside the prefix and the system libc/libstdc++.
  Its FFmpeg 7.1.1 also mis-decodes VVC ALF (ticket 11325), so the
  conformance gates never use an `ffmpeg` from `PATH`: they run
  `bin/ffmpeg-decode`, a static decode-only FFmpeg n8.0.1 with the pinned
  dav1d (`build_anchors.sh ffmpeg`, opt-in). It is deliberately not named
  `ffmpeg`: metric preprocessing (XPSNR, the display-VMAF upscale) and the
  VP9/VVC curve fingerprints still resolve `ffmpeg` via `find_tool`, and
  renaming it would change their identity.
- **x265 on the Jetson:** its SVE/SVE2 kernels need `-march=armv9-a`, which
  gcc 11 rejects; they are disabled on aarch64 (the A78AE has no SVE).
- **Jetson Python:** no `ensurepip`, so the venv is created without pip and
  borrows the user pip, with `--ignore-installed` to keep the pinned
  packages inside the venv.
- **JM** needs `-fcommon` (globals defined in headers) with gcc ≥ 10; JM and
  HM write into their source tree, so they are built from a copy in
  `build/`. JM is 19.0 as specified; JM-19.1 also exists.
- **HM** must be run with `-d 8` to write 8-bit output (otherwise it writes
  the internal bit depth).
- **Version strings:** x264 reports `0.165.x` (its version script needs full
  history) and `vmaf --version` prints the commit (`f85a853`); trust
  `versions.env` for the exact commit.
- **CMake 4** refuses projects that declare `cmake_minimum_required` < 3.5;
  every CMake build passes `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
- **libvmaf** is the CPU build; a CUDA build (`-Denable_cuda=true`) is not
  set up.


## VP9 / AV2 anchors and decoders (opt-in)

`libvpx`, `avm` and `dav2d` are explicit opt-in builds (not part of the default
all-tools build), pinned in `versions.env`. Build them into an isolated prefix
so existing performance runs are unaffected:

```sh
tools/anchors/build_anchors.sh libvpx avm dav2d --jobs 2 --prefix "$HOME/nanocodec-data/anchors/vp9-av2"
```

libvpx installs `vpxenc` and `vpxdec`; `vpxsvc` (same pin, separate build)
installs only `vp9_spatial_svc_encoder`, the VP9 spatial SVC reference for the
SVC chart's later VP9 row (`tools/anchors/build_anchors.sh vpxsvc --prefix
"$HOME/nanocodec-data/anchors/vp9-av2"`); AVM installs `avmenc` and `avmdec` only.
The evaluator anchors are `vpx-vp9-rt-s6/s8/s9` (flat, forward-predicted
Profile 0 at fixed quantizer, one thread, no lag, no alternate reference, no AQ)
and `avm-ld-s9` (AVM `--good` single-pass low delay, not an RTC preset).

Conformance decoders (`eval/nceval/conformance.py`):

- **VP9:** FFmpeg's native `vp9` decoder plus libvpx `vpxdec`, two independent
  implementations. The pinned `vpxdec` reproduces the official libvpx test-vector
  MD5s.
- **AV2:** AVM `avmdec` (the reference) plus `dav2d`. dav2d is pinned to a `main`
  commit, not a release; its own README calls it an early project not for
  production. A disagreement fails the gate, so it cannot produce a false pass,
  but the conformance claim rests on avmdec. The recipe disables assembly and
  builds only 8-bit support, so the binary is not a decoder-speed anchor.

AVM's default encoder enables ML partition pruning and builds TensorFlow
dependencies, so the first full build is substantial. Use `NC_ANCHORS` to select
the private prefix; never link these libraries into the native Rust/WASM product.
