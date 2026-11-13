# clipset — building raw YUV clip sets

`extract.py` turns a selection file into raw 8-bit I420 clips plus the clip
set JSON that `nceval` reads.

```
python3 clipset/extract.py clipset/smoke.json
# -> ~/nanocodec-data/clips/smoke/<clip>.yuv and ~/nanocodec-data/clips/smoke.json
```

Selection entries: `name`, `source` (must be in the inventory), `start`
(seconds), `frames`, `width`, `height`, optional `fps` (frame-rate
conversion; avoid upsampling — it duplicates frames and makes clips easy),
`content_class` (one of `nceval/clips.py: CONTENT_CLASSES`) and `split`
(`train` / `val` / `test`).

- Scaling uses lanczos.
- The source bitrate is read from the inventory and scaled down linearly by
  pixel rate when downscaling. That makes the 50% rule stricter, never looser.
- Each output's SHA-256 is recorded. Clips whose file still matches are not
  rebuilt.

`smoke.json` is a 3-clip, 60-frame set for checking the harness end to end.
It is not a benchmark.

Pristine sources (2026-11-09): raw AOM CTC / Xiph clips are kept as y4m in
`~/nanocodec-data/clips/pristine/` and registered in the inventory
(`source_group` `pristine/xiph-aom-ctc`, uncompressed bitrate), so the
`texture_*` entries of `main.json` extract like any other source.

Caution: `extract.py` rewrites `<set>.json` from the selection file alone.
Clips added to `main.json` by other tools (the `*_noise_*` clips from
`add_noise.py`) are not in the selection and would be dropped: back the set
file up and merge them back, or re-run `add_noise.py` afterwards.
