"""Idea records in the lever ledger: "have we tried X, and what happened?" in one place.

The ledger's measurement rows (nceval.levers) say what one run measured; an idea
record says what was tried and how it ended, including ideas that never left an
unlanded branch and so have no rows. Both live in the same files
(`eval/results/levers.jsonl` and the spool), told apart by `kind`:

    python3 -m nceval.levers ideas                       # every idea, newest first
    python3 -m nceval.levers ideas search causal tpl     # all words, any field
    python3 -m nceval.levers ideas search layer --outcome rejected --codec av1
    python3 -m nceval.levers ideas show reuse-forecast   # the record + its ledger rows
    python3 -m nceval.levers ideas add --name "Reuse forecast" --what "..." \\
        --outcome rejected --reason "L2 J +0.4 [+0.1, +0.7]" --codec av1,hevc \\
        --branch aq-rescreen --doc docs/handoff/block-aq.md#reuse-forecast \\
        --link aq.reuse_forecast

Record schema (kind "idea"; absent values are left out when stored):

    idea      slug naming the idea; the latest record of a slug is its current state
              (an `add` with an existing slug is a revision: unset fields carry over)
    name what short name and one- or two-line description
    first last  dates the idea was tried (YYYY-MM-DD)
    codecs presets  where it was tried (lists)
    branch commits  where the code is or was (commits: list)
    params    registered parameters involved (list)
    outcome   adopted | rejected | held | partial | unclear
    reason    one line with the key numbers
    docs      doc sections, "path#section"
    rows      ledger measurement row ids this idea covers
    date source  when and by what the record was written
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

from . import levers

KIND = "idea"
OUTCOMES = ("adopted", "rejected", "held", "partial", "unclear")
FIELDS = ("id", "kind", "idea", "name", "what", "first", "last", "codecs", "presets", "branch",
          "commits", "params", "outcome", "reason", "docs", "rows", "date", "source")
LISTS = ("codecs", "presets", "commits", "params", "docs", "rows")
#: Fields `search` matches against (all words must occur, case-insensitive).
SEARCHED = ("idea", "name", "what", "codecs", "presets", "branch", "commits", "params",
            "outcome", "reason", "docs")
#: Ledger rows linked by one --link before `add` refuses as too broad.
MAX_LINKED = 400
NAME_WIDTH = 38


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def make_idea(**fields) -> dict:
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown idea fields: {sorted(unknown)}")
    idea = {f: fields.get(f) for f in FIELDS}
    if not idea["name"]:
        raise ValueError("an idea needs a name")
    if idea["outcome"] not in OUTCOMES:
        raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}: {idea['outcome']!r}")
    idea["kind"] = KIND
    idea["idea"] = idea["idea"] or slug(idea["name"])
    for name in LISTS:
        idea[name] = list(dict.fromkeys(idea[name] or [])) or None
    idea["date"] = idea["date"] or levers.today()
    content = {k: v for k, v in idea.items() if k != "id"}
    idea["id"] = "idea-" + hashlib.sha256(
        json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
    return idea


def read_ideas(path) -> list[dict]:
    """The idea records of one ledger file, in file order."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            stored = json.loads(line)
            if stored.get("kind") == KIND:
                out.append({f: stored.get(f) for f in FIELDS})
    return out


def load_ideas() -> list[dict]:
    """The current record of every idea: the union of ledger and spool, and of each slug's
    revisions the latest by date (file order breaks ties, the spool after the ledger)."""
    latest: dict[str, dict] = {}
    for path in (levers.LEDGER, levers.spool_path()):
        for idea in read_ideas(path):
            held = latest.get(idea["idea"])
            if held is None or idea["date"] >= held["date"]:
                latest[idea["idea"]] = idea
    return sorted(latest.values(), key=lambda i: tried(i) or "", reverse=True)


def tried(idea: dict) -> str | None:
    """When the idea was last tried (None when undated: not when it was recorded)."""
    return idea.get("last") or idea.get("first")


def matches(idea: dict, words: list[str], outcome: str | None, codec: str | None) -> bool:
    if outcome and idea["outcome"] != outcome:
        return False
    if codec and codec not in (idea["codecs"] or []):
        return False
    text = " ".join(json.dumps(idea.get(f)) for f in SEARCHED).lower()
    return all(word.lower() in text for word in words)


def print_table(ideas: list[dict]) -> None:
    print(f"{'outcome':8s} {'last':10s} {'codecs':14s} {'rows':>4s}  {'idea':{NAME_WIDTH}s} reason")
    for idea in ideas:
        when = tried(idea) or "-"
        codecs = ",".join(idea["codecs"] or []) or "-"
        print(f"{idea['outcome']:8s} {when:10s} {codecs[:14]:14s} {len(idea['rows'] or []):4d}  "
              f"{idea['idea'][:NAME_WIDTH]:{NAME_WIDTH}s} {(idea['reason'] or '')[:110]}")


def linked_rows(text: str, codecs: list[str] | None) -> list[str]:
    """Ids of ledger rows whose change label or overridden parameter names contain TEXT."""
    needle = text.lower()
    return [r["id"] for r in levers.load()
            if (not codecs or r["codec"] in codecs)
            and (needle in str(r["change"] or "").lower()
                 or any(needle in name.lower() for name in r["overrides"] or {}))]


def split(values: list[str] | None) -> list[str] | None:
    """Repeatable comma-separated options as one list (None when not given)."""
    if not values:
        return None
    return [v.strip() for value in values for v in value.split(",") if v.strip()]


def add(args: argparse.Namespace) -> None:
    key = args.idea or slug(args.name or "")
    previous = next((i for i in load_ideas() if i["idea"] == key), None)
    fields = {f: (previous or {}).get(f) for f in FIELDS if f not in ("id", "date", "source")}
    given = {"name": args.name, "what": args.what, "first": args.first, "last": args.last,
             "branch": args.branch, "outcome": args.outcome, "reason": args.reason,
             **{name: split(getattr(args, name)) for name in LISTS}}
    for name, value in given.items():
        if value is None:
            continue
        fields[name] = (fields[name] or []) + value if name in LISTS and args.append else value
    fields["idea"] = key
    known = {r["id"] for r in levers.load()}
    unknown = [r for r in fields["rows"] or [] if r not in known]
    if unknown:
        raise SystemExit(f"ideas add: unknown ledger row ids {unknown}")
    for text in args.link or []:
        found = linked_rows(text, fields["codecs"])
        if len(found) > MAX_LINKED:
            raise SystemExit(f"ideas add: --link {text!r} matches {len(found)} rows; narrow it")
        print(f"--link {text!r}: {len(found)} ledger row(s)")
        fields["rows"] = (fields["rows"] or []) + found
    try:
        idea = make_idea(**fields, source=args.source or "ideas add")
    except ValueError as error:
        raise SystemExit(f"ideas add: {error}") from error
    added = levers.append([idea])
    state = "unchanged" if not added else "revised" if previous else "new"
    print(f"{state}: {idea['idea']} ({idea['outcome']}, {len(idea['rows'] or [])} rows) "
          f"in {levers.spool_path()}")


def show(args: argparse.Namespace) -> None:
    ideas = [i for i in load_ideas() if args.idea in (i["idea"], i["name"])]
    if not ideas:
        raise SystemExit(f"no idea {args.idea!r}; try: ideas search {args.idea}")
    idea = ideas[0]
    for name in FIELDS:
        if idea.get(name) and name != "rows":
            print(f"{name:8s} {idea[name]}")
    rows = {r["id"]: r for r in levers.load()}
    print(f"rows     {len(idea['rows'] or [])}")
    for row_id in idea["rows"] or []:
        row = rows.get(row_id)
        if row:
            j = "n/a" if row["j"] is None else f"{row['j']:+.2f}"
            print(f"  {row_id} {row['codec']:5s} {row['base']:8s} {row['level'][:12]:12s} "
                  f"J {j:>6s}  {row['verdict'] or '':14.14s} {levers.label(row)[:70]}")


def listing(args: argparse.Namespace) -> None:
    words = getattr(args, "words", None) or []
    outcome, codec = getattr(args, "outcome", None), getattr(args, "codec", None)
    found = [i for i in load_ideas() if matches(i, words, outcome, codec)]
    print_table(found)
    counts = {o: sum(i["outcome"] == o for i in found) for o in OUTCOMES}
    print(f"{len(found)} idea(s): " + ", ".join(f"{o} {n}" for o, n in counts.items() if n))


def register(sub) -> None:
    """The `ideas` command of nceval.levers."""
    filters = argparse.ArgumentParser(add_help=False)
    filters.add_argument("--outcome", choices=OUTCOMES, default=argparse.SUPPRESS)
    filters.add_argument("--codec", default=argparse.SUPPRESS)
    ideas = sub.add_parser("ideas", parents=[filters],
                           help="idea records: what was tried and how it ended")
    ideas.set_defaults(ideas_run=listing)
    actions = ideas.add_subparsers(dest="ideas_command")
    search = actions.add_parser("search", parents=[filters], help="ideas matching all WORDS")
    search.add_argument("words", nargs="+")
    search.set_defaults(ideas_run=listing)
    detail = actions.add_parser("show", help="one idea in full, with its ledger rows")
    detail.add_argument("idea", help="slug or name")
    detail.set_defaults(ideas_run=show)
    new = actions.add_parser("add", help="record an idea, or revise one (same --idea/--name)")
    new.add_argument("--name", help="short name (required for a new idea)")
    new.add_argument("--idea", help="slug (default: from --name); an existing slug revises")
    new.add_argument("--what", help="one- or two-line description")
    new.add_argument("--outcome", choices=OUTCOMES)
    new.add_argument("--reason", help="one line with the key numbers")
    new.add_argument("--first", help="first date tried, YYYY-MM-DD")
    new.add_argument("--last", help="last date tried, YYYY-MM-DD")
    new.add_argument("--branch")
    for name, what in (("codecs", "h264,hevc,..."), ("presets", "medium,fast,..."),
                       ("commits", "commit hashes"), ("params", "registered parameter names"),
                       ("docs", "path#section"), ("rows", "ledger row ids")):
        new.add_argument(f"--{name.removesuffix('s')}", dest=name, action="append",
                         help=f"{what}; comma-separated or repeated")
    new.add_argument("--link", action="append",
                     help="also link the ledger rows whose change label or parameter names "
                     "contain this text (within --codec when given)")
    new.add_argument("--append", action="store_true",
                     help="on a revision, add to the list fields instead of replacing them")
    new.add_argument("--source", help="what wrote the record (default 'ideas add')")
    new.set_defaults(ideas_run=add)


def run(args: argparse.Namespace) -> None:
    args.ideas_run(args)
