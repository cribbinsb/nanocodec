"""Our rows of the results site's fixed-QP lens data, re-measured at a release.

    python3 -m nceval.ctc lens-variants --data web/results/data/fixed-qp-lens.json \
        --families nanocodec-av1,... [--json OUT]
    python3 -m nceval.ctc lens-rows --data web/results/data/fixed-qp-lens.json \
        --compare COMPARE.json --build COMMIT

The lens (web/results/data/fixed-qp-lens.json and its CSV) shows each of our encoders in
up to three columns (docs/results/fixed-qp-lens-2026-11-15.md "Columns"): the preset at
`--qp` with its fixed-QP profile (`profile`, the overview's chart), the same with the
profile off (`cbr_settings`), and a fully static pattern (`strict`, the Defaults). A
release codes every (encoder, column) of ours the data has, plus `profile` for a preset
it lacks, with `nceval.ctc local` (the variant's parameters over the preset;
`lens-variants` lists them, one label per distinct parameter set), compares them with
`nceval.ctc compare --json`, and `lens-rows` writes the results into the data and its
CSV. The references' rows stay as they are (their code does not change with ours).
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
    "strict": {"h264": ("layers.count=1", "layers.adaptive=0", "structure.unified=1",
                        "structure.secondary_min_depth=0", "refs.second_full=1",
                        "rate.cadence_static_scaling=0"),
               "av1": ("rate.cadence_static_scaling=0",),
               "vp9": ("rate.cadence_static_scaling=0",),
               "hevc": (), "vvc": (), "av2": ()},
}
#: The column every shipped preset has (the overview's).
BASE_COLUMN = "profile"
#: The compare figures each row field takes, and their rounding (the data's).
FIELDS = {"weighted": "Weighted", "j": "J", "psnr_yuv": "PSNR-YUV", "vmaf": "VMAF-v1",
          "ms_ssim": "MS-SSIM"}
FPS_DIGITS, FIGURE_DIGITS = 3, 1


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
    """The runs a release codes: each family's columns in the data, `profile` always,
    one run per distinct label."""
    columns = {f: {BASE_COLUMN} for f in families}
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


def apply(data: dict, compare_rows: list[dict], build: str) -> list[dict]:
    """Our rows of `data` from `compare` rows (tests `<encoder>+<label>` against the data's
    anchor); a family without a row in a column it was coded for gains one. Returns the
    rows written. Every row of ours must be re-measured: none stays on an older build."""
    anchor = data["rows"][0]["anchor"]
    measured = {row["test"]: row for row in compare_rows if row["anchor"] == anchor}
    rows = {(r["test"], r["column"]): r for r in data["rows"]}
    families = {test.partition("+")[0] for test in measured}
    wanted = [(r["test"], r["column"]) for r in data["rows"] if r["test"].startswith(OWN_PREFIX)]
    wanted += [(f, BASE_COLUMN) for f in sorted(families) if (f, BASE_COLUMN) not in rows]
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
                    **{f: round(source[name], FIGURE_DIGITS) for f, name in FIELDS.items()}})
        written.append(row)
    data["build_commit"] = f"{build} for every nanocodec row; the references as measured"
    return written


def rewrite_csv(text: str, rows: list[dict]) -> str:
    """The lens CSV with `rows` (test, column) replaced or added, its line endings kept."""
    reader = list(csv.DictReader(io.StringIO(text)))
    fields = list(reader[0])
    by_key = {(r["test"], r["column"]): r for r in reader}
    for row in rows:
        key = (row["test"], row["column"])
        if key not in by_key:
            by_key[key] = {}
            reader.append(by_key[key])
        by_key[key].update({f: str(row.get(f, "")) for f in fields})
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
    written = apply(data, json.loads(Path(args.compare).read_text()), args.build)
    path.write_text(json.dumps(data, indent=1) + "\n")
    repo = path.resolve().parents[3]  # web/results/data/<file> -> the repository
    csv_path = repo / data["source"][0]
    csv_path.write_bytes(rewrite_csv(csv_path.read_bytes().decode(), written).encode())
    print(f"{len(written)} lens rows of ours at {args.build}")


def main_lens_rows(sub) -> None:
    v = sub.add_parser("lens-variants", help="the runs a release codes for our lens rows")
    v.add_argument("--data", required=True)
    v.add_argument("--families", required=True)
    v.add_argument("--json")
    v.set_defaults(handler=variants_command)
    r = sub.add_parser("lens-rows", help="our re-measured lens rows into the site's data")
    r.add_argument("--data", required=True)
    r.add_argument("--compare", required=True, help="nceval.ctc compare --json output")
    r.add_argument("--build", required=True)
    r.set_defaults(handler=rows_command)
