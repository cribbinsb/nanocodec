#!/usr/bin/env python3
"""Cheap checks of documentation and result files (land.sh's docs class).

    python3 tools/dev/check_docs.py <file>...

Markdown: every relative link and image target (`[text](path)`,
`![alt](path)`, `<img src="path">`) must exist in the repository; URLs,
`#anchors` and mail links are not followed. SVG must parse as XML and JSON as
JSON, so a truncated chart or results file cannot land. Other files pass.
Exits non-zero listing every problem.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Inline links and images; the target stops at whitespace or the closing
# parenthesis (an optional "title" may follow).
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_SRC_RE = re.compile(r"""<img\b[^>]*\bsrc\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
FENCE_RE = re.compile(r"^(```|~~~)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#", "data:")


def markdown_targets(text):
    """Link and image targets outside fenced and inline code."""
    in_fence = False
    for line in text.splitlines():
        if FENCE_RE.match(line.lstrip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = re.sub(r"`[^`]*`", "", line)
        yield from MD_LINK_RE.findall(line)
        yield from HTML_SRC_RE.findall(line)


def check_markdown(path, root):
    problems = []
    for target in markdown_targets(path.read_text(errors="replace")):
        if target.startswith(EXTERNAL_PREFIXES):
            continue
        relative = target.split("#", 1)[0].split("?", 1)[0]
        if not relative:
            continue
        resolved = (root / relative.lstrip("/")) if relative.startswith("/") else (path.parent / relative)
        if not resolved.exists():
            problems.append(f"{path.relative_to(root)}: broken link {target}")
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


def main(args):
    root = Path.cwd()
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
