#!/usr/bin/env python3
"""Cheap checks of documentation and result files (land.sh's docs class).

    python3 tools/dev/check_docs.py <file>...
    python3 tools/dev/check_docs.py --all      # every tracked file (land.sh --verify)

Markdown: every relative link and image target (`[text](path)`,
`![alt](path)`, `<img src="path">`) must exist in the repository, and a
`#anchor` into a Markdown file must name one of its headings (GitHub's slugs)
or an `<a name|id>`; URLs and mail links are not followed. A symlinked
Markdown file (the crates' THIRD_PARTY.md copies, there for `cargo package`) is
the same document as its target and is checked once, at the target, where its
relative links resolve. SVG must parse as XML and JSON as JSON, so a truncated
chart or results file cannot land. Other files pass. Exits non-zero listing
every problem.

`--all` is what verify runs: a change to one file (a rename, a deleted section)
breaks links in files it did not touch, which the changed-files check misses.
"""

import functools
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Inline links and images; the target stops at whitespace or the closing
# parenthesis (an optional "title" may follow).
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_SRC_RE = re.compile(r"""<img\b[^>]*\bsrc\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
FENCE_RE = re.compile(r"^(```|~~~)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "data:")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
HTML_ANCHOR_RE = re.compile(r"""<a\s+[^>]*\b(?:name|id)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def prose_lines(text):
    """The lines outside fenced code blocks."""
    in_fence = False
    for line in text.splitlines():
        if FENCE_RE.match(line.lstrip()):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield line


def markdown_targets(text):
    """Link and image targets outside fenced and inline code."""
    for line in prose_lines(text):
        line = re.sub(r"`[^`]*`", "", line)
        yield from MD_LINK_RE.findall(line)
        yield from HTML_SRC_RE.findall(line)


def heading_slug(heading):
    """GitHub's anchor for a heading: code spans and links reduced to their text,
    lower case, punctuation other than `-` and `_` dropped, spaces to `-`."""
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


@functools.cache
def markdown_anchors(path):
    """The anchors of a Markdown file: heading slugs (repeats numbered -1, -2, ...
    as GitHub does) and explicit `<a name|id>` targets."""
    anchors, seen = set(), {}
    for line in prose_lines(path.read_text(errors="replace")):
        if heading := HEADING_RE.match(line):
            slug = heading_slug(heading.group(1))
            anchors.add(f"{slug}-{seen[slug]}" if slug in seen else slug)
            seen[slug] = seen.get(slug, 0) + 1
        anchors.update(HTML_ANCHOR_RE.findall(line))
    return frozenset(anchors)


def check_markdown(path, root):
    if path.is_symlink():
        return []  # checked at its target (module docstring)
    problems = []
    for target in markdown_targets(path.read_text(errors="replace")):
        if target.startswith(EXTERNAL_PREFIXES):
            continue
        relative, _, anchor = target.split("?", 1)[0].partition("#")
        if not relative:
            resolved = path
        else:
            resolved = (root / relative.lstrip("/")) if relative.startswith("/") else (path.parent / relative)
            if not resolved.exists():
                problems.append(f"{path.relative_to(root)}: broken link {target}")
                continue
        if anchor and resolved.suffix == ".md" and resolved.is_file() \
                and anchor.lower() not in markdown_anchors(resolved.resolve()):
            problems.append(f"{path.relative_to(root)}: no heading for anchor {target}")
    return problems


def check_svg(path, root):
    try:
        ET.parse(path)
    except ET.ParseError as error:
        return [f"{path.relative_to(root)}: not well-formed SVG ({error})"]
    return []


def check_json(path, root):
    try:
        json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return [f"{path.relative_to(root)}: not valid JSON ({error})"]
    return []


CHECKS = {".md": check_markdown, ".svg": check_svg, ".json": check_json}


def tracked(root):
    """Every tracked file this checker knows (git's list)."""
    patterns = [f"*{suffix}" for suffix in CHECKS]
    listed = subprocess.run(["git", "ls-files", "-z", "--", *patterns], cwd=root,
                            capture_output=True, text=True, check=True).stdout
    return [name for name in listed.split("\0") if name]


def main(args):
    root = Path.cwd()
    if args == ["--all"]:
        args = tracked(root)
    problems = []
    for name in args:
        path = root / name
        check = CHECKS.get(path.suffix)
        if check and path.is_file():
            problems += check(path, root)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
