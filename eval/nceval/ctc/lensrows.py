"""Our rows of the results site's fixed-QP lens data, re-measured at a release.

    python3 -m nceval.ctc lens-variants --data web/results/data/fixed-qp-lens.json \
        --families nanocodec-av1,... [--json OUT]
    python3 -m nceval.ctc lens-rows --data web/results/data/fixed-qp-lens.json \
        --compare COMPARE.json --build COMMIT
    python3 -m nceval.ctc lens-rows --data ... --compare COMPARE.json --reported-only

The lens (web/results/data/fixed-qp-lens.json and its CSV) shows each of our encoders in
up to four columns (docs/results/fixed-qp-lens-2026-11-15.md "Columns"): the preset at
`--qp` with its fixed-QP profile (`profile`, the overview's chart), the same with the
profile off (`cbr_settings`), a fully static pattern (`strict`, the Defaults), and every
picture and block at the given quantizer (`constant`, `rate.constant_qp`: the flat row,
pinned as the references are). A release codes every (encoder, column) of ours the data
has, plus the RELEASE_COLUMNS for a preset that lacks them, with `nceval.ctc local` (the
variant's parameters over the preset;
`lens-variants` lists them, one label per distinct parameter set), compares them with
`nceval.ctc compare --json`, and `lens-rows` writes the results into the data and its
CSV. The references' rows stay as they are (their code does not change with ours).

A new zero line (`--anchor`, e.g. x264-medium+psnr, `nceval.ctc local --x264 psnr-qp`)
re-bases every row, the references' too, on a `compare --anchor` of the same curves: each
row's figures and sequence count are replaced, its build and fps kept, and the anchor
gains its own row at zero.

XPSNR-YUV is a reported figure (never in Weighted or J). `--reported-only` adds the
reported figures to rows already in the data, ours and the references', from a re-run
that coded the same streams again: every other figure of a row must come out as the
data has it, within one unit of its rounding (the re-run's anchor curve is at full
precision, the stored one rounded in its points CSV: a figure at a rounding edge moves
by 0.1 with identical streams); else nothing is written. So the
rows' build and timing stay as they were. A reference whose re-run is a different
build of its encoder (`--remeasured`, e.g. VVenC rebuilt from its tag coding a few
points differently) instead takes every figure but fps from the re-run.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from ..plotted import OWN_PREFIX

#: Each column's parameters over the preset, by codec ("*": any codec). Strict: H.264
#: flat with the unified plan and the second reference on every high-quality frame;
#: AV1 and VP9 the full cadence; HEVC, VVC and AV2's fixed-QP profile already is that
#: pattern (the parameter sets of the 2026-11-15 lens run, work/fixedqp2/lens/cut/*).
COLUMN_PARAMS = {
    "profile": {"*": ()},
    "cbr_settings": {"*": ("rate.fixed_qp_profile=0",)},
    "constant": {"*": ("rate.constant_qp=1",)},
    "strict": {"h264": ("layers.count=1", "layers.adaptive=0", "structure.unified=1",
                        "structure.secondary_min_depth=0", "refs.second_full=1",
                        "rate.cadence_static_scaling=0"),
               "av1": ("rate.cadence_static_scaling=0",),
               "vp9": ("rate.cadence_static_scaling=0",),
               "hevc": (), "vvc": (), "av2": ()},
}
#: The column every shipped preset has (the overview's).
BASE_COLUMN = "profile"
#: The columns a release codes for every preset: the overview's and the flat one.
RELEASE_COLUMNS = (BASE_COLUMN, "constant")
#: The compare figures each row field takes, and their rounding (the data's).
FIELDS = {"weighted": "Weighted", "j": "J", "psnr_yuv": "PSNR-YUV", "vmaf": "VMAF-v1",
          "ms_ssim": "MS-SSIM"}
#: Figures a row gains when re-measured or re-based (None until then): PSNR-Y, the
#: frame-averaged luma PSNR as HM prints it (nceval.ctc.report "Y-fa").
ADDED_FIELDS = {"psnr_y": "PSNR-Y"}
#: Reported figures (nceval.ctc.report REPORTED_METRICS): None where a curve lacks them.
REPORTED_FIELDS = {"xpsnr": "XPSNR-YUV"}
#: A reported field's companion: the sequences it is over (compare's `<metric> n`, those
#: where it can be computed), e.g. `xpsnr_clips`.
COUNT_SUFFIX = "_clips"
FPS_DIGITS, FIGURE_DIGITS = 3, 1
#: How far a reproduced figure may move: one unit of the data's rounding (module doc).
REPRODUCED_WITHIN = 10.0 ** -FIGURE_DIGITS + 1e-9


def codec_of(family: str) -> str:
    return family.removeprefix(OWN_PREFIX).partition("@")[0]


def column_params(family: str, column: str) -> tuple[str, ...]:
    by_codec = COLUMN_PARAMS[column]
    return by_codec.get(codec_of(family), by_codec.get("*", ()))


def label(family: str, column: str) -> str:
    """The `nceval.ctc local` label coding `family` in `column`: one per distinct
    parameter set (a column whose parameters are the profile's reuses its curve)."""
    params = column_params(family, column)
    if params == column_params(family, BASE_COLUMN):
        column = BASE_COLUMN
    return f"rel-{column}"


def variants(data: dict, families: list[str]) -> list[dict]:
    """The runs a release codes: each family's columns in the data, the RELEASE_COLUMNS
    always, one run per distinct label."""
    columns = {f: set(RELEASE_COLUMNS) for f in families}
    for row in data["rows"]:
        if row["test"] in columns:
            columns[row["test"]].add(row["column"])
    runs = {}
    for family, wanted in columns.items():
        for column in sorted(wanted):
            name = label(family, column)
            runs[(family, name)] = {"encoder": family, "label": name,
                                    "params": list(column_params(family, column))}
    return list(runs.values())


def apply(data: dict, compare_rows: list[dict], build: str,
          column_builds: dict[str, str] | None = None) -> list[dict]:
    """Our rows of `data` from `compare` rows (tests `<encoder>+<label>` against the data's
    anchor); a family without a row in a column it was coded for gains one. Returns the
    rows written. Every row of ours must be re-measured: none stays on an older build.
    `column_builds`: columns coded on another build than `build` (a column whose switch
    the release lacks, e.g. `constant` before rate.constant_qp), kept as the data's
    `column_builds` for the captions (nceval.site)."""
    anchor = data["rows"][0]["anchor"]
    measured = {row["test"]: row for row in compare_rows if row["anchor"] == anchor}
    rows = {(r["test"], r["column"]): r for r in data["rows"]}
    # ours only: a compare of every curve (a re-base) also lists the references
    families = {test.partition("+")[0] for test in measured if test.startswith(OWN_PREFIX)}
    wanted = [(r["test"], r["column"]) for r in data["rows"] if r["test"].startswith(OWN_PREFIX)]
    wanted += [(f, c) for f in sorted(families) for c in RELEASE_COLUMNS
               if (f, c) not in rows and (f, c) not in wanted]
    missing = [f"{t} {c}" for t, c in wanted if f"{t}+{label(t, c)}" not in measured]
    if missing:
        raise SystemExit(f"lens-rows: not measured: {', '.join(missing)}")
    written = []
    for test, column in wanted:
        source = measured[f"{test}+{label(test, column)}"]
        row = rows.get((test, column))
        if row is None:
            row = {"test": test, "column": column, "anchor": anchor}
            data["rows"].append(row)
        row.update({"n": source["n"], "fps": round(source["fps"], FPS_DIGITS),
                    **figures(source, FIELDS | ADDED_FIELDS), **reported(source)})
        written.append(row)
    column_builds = column_builds or {}
    unknown = sorted(set(column_builds) - set(COLUMN_PARAMS))
    if unknown:
        raise SystemExit(f"lens-rows: no such column: {', '.join(unknown)}")
    others = "".join(f", column {c} at {b}" for c, b in sorted(column_builds.items()))
    data["build_commit"] = (f"{build} for the nanocodec rows{others}; "
                            "the references as measured")
    data["column_builds"] = dict(sorted(column_builds.items()))
    return written


def rebase(data: dict, compare_rows: list[dict], anchor: str) -> list[dict]:
    """Every row of `data` against a new zero line `anchor` (module doc), from `compare
    --anchor ANCHOR` rows of the same curves: ours under `<encoder>+<label>`, a reference
    under its name. Every row must be there; the anchor's own row (zero) is added.
    Returns the rows written."""
    measured = {row["test"]: row for row in compare_rows if row["anchor"] == anchor}
    name = lambda row: (f"{row['test']}+{label(row['test'], row['column'])}"
                        if row["test"].startswith(OWN_PREFIX) else row["test"])
    missing = sorted({name(r) for r in data["rows"] if r["test"] != anchor} - set(measured))
    if missing:
        raise SystemExit(f"lens-rows: no comparison against {anchor}: {', '.join(missing)}")
    data["rows"] = [r for r in data["rows"] if r["test"] != anchor]
    for row in data["rows"]:
        source = measured[name(row)]
        row.update({"anchor": anchor, "n": source["n"],
                    **figures(source, FIELDS | ADDED_FIELDS), **reported(source)})
    columns = sorted({r["column"] for r in data["rows"]})
    zero = {f: 0.0 for f in (*FIELDS, *ADDED_FIELDS)}
    sequences = max(r["n"] for r in data["rows"])
    # the zero row carries XPSNR only where some row has it (else the views would keep
    # an XPSNR metric with nothing but the zero on it: nceval.site with_xpsnr)
    scored = any(r.get("xpsnr") is not None for r in data["rows"])
    data["rows"] += [{"test": anchor, "column": column, "anchor": anchor, "n": sequences,
                      "fps": None, **zero, "xpsnr": 0.0 if scored else None,
                      "xpsnr" + COUNT_SUFFIX: sequences if scored else None}
                     for column in columns]
    return list(data["rows"])


def figures(source: dict, fields: dict[str, str]) -> dict[str, float | None]:
    """A compare row's figures in the data's fields and rounding."""
    return {f: None if source.get(name) is None else round(source[name], FIGURE_DIGITS)
            for f, name in fields.items()}


def reported(source: dict) -> dict[str, float | int | None]:
    """A compare row's reported figures, each with its sequence count."""
    out = {}
    for field, name in REPORTED_FIELDS.items():
        value = source.get(name)
        out[field] = None if value is None else round(value, FIGURE_DIGITS)
        out[field + COUNT_SUFFIX] = None if value is None else source.get(f"{name} n")
    return out


def apply_reported(data: dict, compare_rows: list[dict],
                   remeasured: frozenset[str] = frozenset()) -> list[dict]:
    """The reported figures of every row of `data` from a re-run of its curve (ours under
    `<encoder>+<label>`, a reference under its name); each row's other figures must equal
    the re-run's, except a `remeasured` reference's, which take the re-run's. Rows
    without a re-run are left alone. Returns the rows written."""
    anchor = data["rows"][0]["anchor"]
    measured = {row["test"]: row for row in compare_rows if row["anchor"] == anchor}
    updates, differing = [], []
    for row in data["rows"]:
        name = (f"{row['test']}+{label(row['test'], row['column'])}"
                if row["test"].startswith(OWN_PREFIX) else row["test"])
        if name not in measured:
            continue
        source = measured[name]
        expected = {"n": row["n"], **{f: row[f] for f in FIELDS}}
        got = {"n": source["n"], **figures(source, FIELDS)}
        if row["test"] in remeasured:
            updates.append((row, {**got, **reported(source)}))
            continue
        if got["n"] != expected["n"] or any(
                abs(got[f] - expected[f]) > REPRODUCED_WITHIN for f in FIELDS):
            differing.append(f"{row['test']} {row['column']}: data {expected}, re-run {got}")
        updates.append((row, reported(source)))
    if differing:
        raise SystemExit("lens-rows: the re-run did not reproduce:\n" + "\n".join(differing))
    for row, values in updates:
        row.update(values)
    return [row for row, _ in updates]


def rewrite_csv(text: str, rows: list[dict]) -> str:
    """The lens CSV with `rows` (test, column) replaced or added, its line endings kept."""
    reader = list(csv.DictReader(io.StringIO(text)))
    fields = list(reader[0])
    reported_fields = [name for f in REPORTED_FIELDS for name in (f, f + COUNT_SUFFIX)]
    fields += [f for f in [*ADDED_FIELDS, *reported_fields]
               if f not in fields and any(f in r for r in rows)]
    by_key = {(r["test"], r["column"]): r for r in reader}
    for row in rows:
        key = (row["test"], row["column"])
        if key not in by_key:
            by_key[key] = {}
            reader.append(by_key[key])
        by_key[key].update({f: "" if row.get(f) is None else str(row[f]) for f in fields})
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields,
                            lineterminator="\r\n" if "\r\n" in text else "\n")
    writer.writeheader()
    writer.writerows(reader)
    return out.getvalue()


def variants_command(args: argparse.Namespace) -> None:
    runs = variants(json.loads(Path(args.data).read_text()), args.families.split(","))
    text = json.dumps(runs, indent=1) + "\n"
    if args.json:
        Path(args.json).write_text(text)
    print(text, end="")


def rows_command(args: argparse.Namespace) -> None:
    path = Path(args.data)
    data = json.loads(path.read_text())
    compare_rows = json.loads(Path(args.compare).read_text())
    if args.reported_only:
        remeasured = frozenset(args.remeasured.split(",")) if args.remeasured else frozenset()
        if any(name.startswith(OWN_PREFIX) for name in remeasured):
            raise SystemExit("lens-rows: --remeasured is for references; ours use --build")
        written = apply_reported(data, compare_rows, remeasured)
    elif args.build or args.anchor:
        written = rebase(data, compare_rows, args.anchor) if args.anchor else []
        if args.build:
            column_builds = dict(spec.split("=", 1) for spec in args.column_build)
            measured = {(r["test"], r["column"])
                        for r in apply(data, compare_rows, args.build, column_builds)}
            written = [r for r in written if (r["test"], r["column"]) not in measured]
            written += [r for r in data["rows"] if (r["test"], r["column"]) in measured]
    else:
        raise SystemExit("lens-rows: --build (our re-measured rows), --anchor (a new zero "
                         "line) or --reported-only")
    path.write_text(json.dumps(data, indent=1) + "\n")
    repo = path.resolve().parents[3]  # web/results/data/<file> -> the repository
    csv_path = repo / data["source"][0]
    csv_path.write_bytes(rewrite_csv(csv_path.read_bytes().decode(), written).encode())
    what = ("with reported figures" if args.reported_only
            else f"against {args.anchor}" if args.anchor and not args.build
            else f"of ours at {args.build}" + (f", every row against {args.anchor}"
                                               if args.anchor else ""))
    print(f"{len(written)} lens rows {what}")


def main_lens_rows(sub) -> None:
    v = sub.add_parser("lens-variants", help="the runs a release codes for our lens rows")
    v.add_argument("--data", required=True)
    v.add_argument("--families", required=True)
    v.add_argument("--json")
    v.set_defaults(handler=variants_command)
    r = sub.add_parser("lens-rows", help="our re-measured lens rows into the site's data")
    r.add_argument("--data", required=True)
    r.add_argument("--compare", required=True, help="nceval.ctc compare --json output")
    r.add_argument("--build", help="our build: every row of ours re-measured on it")
    r.add_argument("--column-build", action="append", default=[], metavar="COLUMN=COMMIT",
                   help="with --build: a column coded on another build (e.g. constant=feb16365)")
    r.add_argument("--anchor", help="re-base every row on this zero line (compare --anchor "
                   "of the same curves; e.g. x264-medium+psnr)")
    r.add_argument("--reported-only", action="store_true",
                   help="add the reported figures (XPSNR) to the rows the re-run reproduces")
    r.add_argument("--remeasured", help="with --reported-only: references whose rows take "
                   "every figure from the re-run (a rebuilt encoder that did not reproduce)")
    r.set_defaults(handler=rows_command)
