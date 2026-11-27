# Results

Where nanocodec's measured results live. Negative BD-rate always means fewer bits than the comparison encoder
for the same score.

Until 2026-11-23 this page carried its own copies of the tables (presets, decoders, NVENC, the browser, per
metric, per class), each from an earlier build. Several had gone stale beside the regenerated data
([results audit, finding 15](../analysis/results-audit-2026-11-23.md#15-stale-tables-beside-current-ones)), so
they were removed: the current figures are only in the generated files and pages below, and the older figures
only in their dated records, each of which names its build and date. The earlier version of this page is in the
repository history.

## Current results

The current results are the benchmark procedure's (BP-4, CBR only; [benchmark-procedure.md](../benchmark-procedure.md)):
the interactive pages at **https://cribbinsb.github.io/nanocodec/results/** (overview, by codec, rate
compliance, method, history), built from the procedure's reports by `tools/bench/preview.py` with the pages in
`tools/bench/pages/`, and the README's figures and chart (`tools/bench/readme_chart.py`), from the same reports.

## Earlier records

The files in this directory are the dated records of the measurements before BP-4: the pre-BP-3 CBR charts,
the withdrawn Fixed QP comparison, speed on the earlier timing machines, buffer use, rate-control cost,
spatial scalability, decoder and browser speed, and conformance. Their pages (speed, buffer, Fixed QP, SVC,
clips, yardsticks, rate control) are no longer published; `python3 -m nceval.site` still renders their data
into `web/results/data/*.json`. Each record names its build and date.

| What | Data here |
|---|---|
| CBR bits against speed (pre-BP-3): fresh held-out, all held-out, development and VC-1K clips | `bd-vs-speed*.csv`, `bd-vs-speed*.md`, `bd-vs-speed-timing*.json` |
| Fixed QP (withdrawn at BP-4) | `fixed-qp-*`, `standards-ctc-2026-11-14/` |
| Encode speed, browser, decoders, hardware encoders | `bd-vs-speed-timing*.json`, `bd-vs-speed-hardware.csv`, `decoder-speed-*/`, `wasm-yardstick-*/`, `wasm-speed.csv` |
| Buffer use of every encoder in the CBR charts | `cbr-buffer-use-*.json`, `cbr-buffer-cost-*.json` |
| Rate-control cost (CBR against fixed QP) | `rc-cost-*.csv` |
| Spatial scalability | `svc-chart.*` |
| Day-by-day history of the pre-BP-3 headline chart | `history-*/` |

## Presets

The preset names and their mapping from the names used before 2026-11-21 are in the main README's "Presets"
section. Each preset's current BD-J and fps are in the [by-codec table](https://cribbinsb.github.io/nanocodec/results/codecs.html).

## Decoders

What each decoder supports and how many conformance streams it passes, with denominators, is stated in the
[FAQ](../FAQ.md#what-is-not-done) and the README, from the dated conformance record below. Decoder speed against
FFmpeg, dav1d, vpxdec, VVdeC and avmdec, natively and in the browser, is in the newest `decoder-speed-*/` and
`wasm-yardstick-*/` records here. Fuzzing: [fuzzing.md](fuzzing.md).

## Per metric

Every chart's table gives BD-PSNR-YUV and BD-VMAF beside BD-J (the [by-codec table](https://cribbinsb.github.io/nanocodec/results/codecs.html)
and the `bd-vs-speed*.md` tables here). The per-metric pairs measured on 2026-11-11 (build 730e4e16, 11
development clips, preset names of that date) are in the repository history of this page.

## Known weaknesses

Each was measured at the date given, on an earlier build; none has been re-measured at 0.6.0.

- **Fine pristine texture** (water, foliage, grass), above all against x264 and x265; on the 2026-11-10 board our
  HEVC, AV1, VP9, AV2 and VVC also lost Riverbed to SVT-AV1 M12.
- **Flicker:** on 2026-11-11 (build 730e4e16) our H.264 presets flickered more than x264 `veryfast` in mean
  frame-to-frame PSNR-Y change.
- **Panning key frames** stood out more than SVT-AV1's (2026-11-11 board, pan class).
- **Bodycam worst frames under CBR:** on bodycam_bc1 our worst-1% P-frame PSNR-Y was worse than SVT-AV1 M12's
  (AV1 and HEVC, 2026-11-09).
- **VP9** trailed our HEVC on the 2026-11-11 board, most on crowd and low light and on VMAF v1.
- **VVC and AV2** are young: rate control and quality work are ongoing.
- **Rates:** our savings are largest at the low rates the charts use; over 600 kbps–4 Mbps at 720p30 they were
  about half as large on 10–11 clips at 2026-11-13 builds ([rate-control audit §5](rate-control-audit-2026-11-13.md#5-rate-ladder-the-low-rungs-favour-everyone-against-x264-higher-rates-shrink-every-gap)).
- **Window length:** over 600 frames with a key every 10 s our savings were 6–10 points smaller than over 150
  frames at commit 8b492382 ([independent report](independent-2026-11-13.md#update-2026-11-13-later-the-new-rate-control-and-a-20-s-window)).

## Dated records

Each is the state at its date and build, and is not updated.

| Record | Date, build | What |
|---|---|---|
| [results-audit-2026-11-23.md](../analysis/results-audit-2026-11-23.md) | 2026-11-23, release 0.6.0 | adversarial audit of the published results, with the fix status of each finding |
| [cpu-targets-2026-11-21.md](cpu-targets-2026-11-21.md) | 2026-11-21 | encode speed on the CPU targets |
| [decoder-yardstick-2026-11-22.md](decoder-yardstick-2026-11-22.md), [decode-speed-2026-11-18.md](decode-speed-2026-11-18.md), `decoder-speed-2026-11-13/`, `decoder-speed-2026-11-19/` | as named | decoder speed against the reference decoders |
| [wasm-speed.md](wasm-speed.md), `wasm-yardstick-2026-11-12/`, `wasm-yardstick-2026-11-19/` | 2026-11-10 to 11-19 | browser encode and decode speed, ffmpeg.wasm |
| [competitor-yardsticks.md](competitor-yardsticks.md) | 2026-11-12 onwards | every speed claim against an independent implementation |
| [vc1k-2026-11-18.md](vc1k-2026-11-18.md) | 2026-11-18 | the VC-1K video-call clip set |
| [graviton-speed-2026-11-17.md](graviton-speed-2026-11-17.md), [zen3-speed-2026-11-17.md](zen3-speed-2026-11-17.md) | 2026-11-17 | encode speed on Graviton4 and Zen 3 |
| [fixed-qp-lens-2026-11-15.md](fixed-qp-lens-2026-11-15.md), [fixed-qp-speed-2026-11-16.md](fixed-qp-speed-2026-11-16.md), [standards-ctc-2026-11-14.md](standards-ctc-2026-11-14.md) | 2026-11-14 to 11-16 | fixed-QP lens on the CTC sequences |
| [independent-2026-11-13.md](independent-2026-11-13.md) | 2026-11-13, 8b492382 | independent re-run on 209 held-out clips, per metric, 95 % bootstrap intervals, 600-frame window |
| [rate-control-audit-2026-11-13.md](rate-control-audit-2026-11-13.md) | 2026-11-13 | how each gap depends on rate counting, buffer, window, ladder and reference settings |
| [heldout-chart-2026-11-12.md](heldout-chart-2026-11-12.md) | 2026-11-12 | the first 71-clip held-out check |
| [development-chart.md](development-chart.md), [svc-chart.md](svc-chart.md) | as stated in each | development-clip and SVC charts |
| [reproducibility.md](reproducibility.md) | — | how to reproduce one chart number and the independent test |
| [fuzzing.md](fuzzing.md) | per round | decoder fuzzing rounds and what was not tested |
