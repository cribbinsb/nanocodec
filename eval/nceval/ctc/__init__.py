"""The standards lens: our encoders under codec-standards common test conditions.

A second, independent evaluation beside the project's own CBR charts
(docs/analysis/standards-evaluation-methodology-2026-11-14.md):

- `sequences`  the public CTC sequences, checksums and licence tiers;
- `conditions` the AOM CTC v8 low-delay and JVET LD-P class E conditions,
               quantiser points and the encoder roster;
- `reference_encoders`  HM, VTM, VVenC and libaom-speed-0 adapters;
- `plan`       jobs and a staged run directory for the worker;
- `worker`     the stdlib-only script that encodes, verifies and scores on a
               rented machine (or here, for the pipe-clean);
- `report`     BD-rate tables in the CTCs' own form, CSVs and the page data.

The sequences are an evaluation lens only: sealed against tuning
(`nceval.pool.SEALED_SETS`).
"""
