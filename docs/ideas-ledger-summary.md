# Ideas ledger: summary (2026-11-20)

The ideas ledger records every idea an agent tried, landed or not, with its outcome and a one-line
reason with the key numbers. Agents search it before starting an idea and add to it when they finish.
This page gives the counts only.

## Totals

| | Ideas |
|---|---:|
| Tried (all records) | 2,405 |
| Adopted | 1,074 |
| Rejected | 1,069 |
| Held (parked, not decided) | 102 |
| Partial (part adopted) | 84 |
| Unclear (no decision recorded) | 76 |

Dates run from 2026-11-06 to 2026-11-20. The idea records were introduced on 2026-11-18: 2,088 of the
records were backfilled that day from the project's handoff and analysis notes, and 317 were written by
agents as they finished (`ideas add`).

| Source of the record | Ideas | Adopted | Rejected | Held | Partial | Unclear |
|---|---:|---:|---:|---:|---:|---:|
| Backfilled from notes (2026-11-18) | 2,088 | 958 | 923 | 68 | 71 | 68 |
| Recorded live (`ideas add`) | 317 | 116 | 146 | 34 | 13 | 8 |

## By codec

An idea tried in several codecs is counted under each, so the codec rows sum to more than the total.

| Codec | Ideas | Adopted | Rejected | Held | Partial | Unclear |
|---|---:|---:|---:|---:|---:|---:|
| AV1 | 994 | 428 | 444 | 41 | 46 | 35 |
| HEVC | 537 | 263 | 209 | 21 | 26 | 18 |
| H.264 | 495 | 226 | 212 | 20 | 27 | 10 |
| AV2 | 465 | 254 | 173 | 17 | 12 | 9 |
| VVC | 356 | 196 | 126 | 16 | 13 | 5 |
| VP9 | 322 | 190 | 96 | 12 | 20 | 4 |

| Scope | Ideas | Adopted | Rejected | Held | Partial | Unclear |
|---|---:|---:|---:|---:|---:|---:|
| One codec | 1,893 | 789 | 909 | 85 | 45 | 65 |
| Two or more codecs | 408 | 221 | 131 | 15 | 35 | 6 |
| No codec recorded (engine, tooling) | 104 | 64 | 29 | 2 | 4 | 5 |

The ledger has no speed-versus-quality field, so there is no split by that.

## Commands

Run from `eval/` at commit c311f42cf, 2026-11-20 evening:

```
python3 -m nceval.levers ideas | tail -1                    # totals
python3 -m nceval.levers ideas --codec av1 | tail -1        # likewise h264, hevc, vvc, vp9, av2
```

The scope rows and the record-source split use the same loader the CLI uses
(`nceval.levers_ideas.load_ideas()`), grouped by the number of codecs and by each record's `source` field.

The CLI reads the committed ledger (`eval/results/levers.jsonl`) plus a local spool of records not yet
committed. The committed ledger alone, at c311f42cf, holds 2,100 ideas (adopted 962, rejected 929,
partial 72, held 69, unclear 68); the figures above include the spool, so a checkout of the repository gives
the smaller numbers until the spool is committed.
