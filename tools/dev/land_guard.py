#!/usr/bin/env python3
"""Refuse a landing that silently reverts other agents' work (used by land.sh).

    python3 tools/dev/land_guard.py <integration-ref>     (in the candidate worktree)

2fae95f reverted 55 files: its author squashed with `git reset --soft
<integration>` after the integration branch had moved, so the commit carried
its old tree. The tree was self-consistent, so every check passed. After such
a reset the fork point is the integration head, so history shows nothing; this
guard catches the mistake by content.

The rule. land.sh runs the guard after rebasing, so the candidate's diff
against the integration head is exactly what landing would change. A stale
tree shows up in that diff as old text put back. For a recent upstream commit
C (WINDOW_DAYS, non-merge) that touched a file the candidate changes, the
candidate undoes one of C's hunks when either

  - it deletes every line of the hunk that still survives on the integration
    head (found with `git blame`, so lines upstream itself later rewrote are
    not C's any more) and puts back only old text: the lines C replaced,
    nothing, or text from the file before a recent commit (a tree stale by
    several commits); the deleted lines must not reappear elsewhere in the
    candidate (a move); or
  - C deleted lines without replacement and the candidate inserts them again,
    verbatim, next to one of the lines that surrounded them.

and the candidate undoes C when it undoes at least UNDONE_SHARE of C's hunks
that survive, counted over every file C touched: a stale tree lacks all of C
except where its author's own edits overlap. Anything else is the candidate's
own work: editing or rewriting lines C added (the replacement is new text),
deleting part of a hunk, moving lines, appending next to C's append, or
restoring a part of C (a condition dropped in a rewrite, an old doc line).
Hunks made only of punctuation and blank lines are ignored: they match
everywhere.

A deliberate revert of C must say so: naming C's sha (7+ characters, e.g.
`Reverts: <sha>`) in a candidate commit message lets it pass.

Deleting a whole file counts against the commit that created it only: a
stale tree lacks just the files created after its base, so deleting an older
file is deliberate and its later edits say nothing.

Validated on the real incident: replaying 2fae95f against its parent flags
all 54 files 6eb3cbd later restored and none of its 7 intended files. The
nine candidates the earlier hunk-matching guard refused on 2026-11-08/09 that
are still in agents' reflogs, and the 28 landings annotated `Reverts:` to get
past it, pass without annotation, except the genuine whole reverts among
them (c0f9287c of c326b513, 33fc7bc2 of afaa553, 1143e937 of b197bd3), which
are still caught. Synthetic cases of each kind are in
eval/tests/test_land_guard.py.

Known limits: changes older than WINDOW_DAYS; changes introduced only in a
merge commit's conflict resolution (merges are skipped); and a stale hunk
merged into one diff hunk with the candidate's own adjacent edit (the
replacement is then partly new text). Binary files are checked as a whole:
C's binary patch applies to the candidate but not to the integration head.
Exit 0 if clean, 1 listing the reverted (file, commit) pairs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache

WINDOW_DAYS = 21
# A commit counts as undone when at least this share of its surviving hunks
# is. Replays: the 2fae95f stale tree undid 9/16 to all of each commit it
# reverted (one 1/2); the 2026-11-09 refusals of agents' own work undid at
# most 1/3 (a helper dropped in a rewrite), mostly 1/13 to 1/107.
UNDONE_SHARE = 0.5
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def git(*args: str, check: bool = True) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check,
                          errors="replace").stdout


def substantive(line: str) -> bool:
    """Whether a line says something: blank and punctuation-only lines match anywhere."""
    return any(ch.isalnum() for ch in line)


def contains_run(haystack: list[str], needle: list[str]) -> bool:
    """Whether `needle` occurs in `haystack` as consecutive lines."""
    if not needle:
        return True
    width = len(needle)
    return any(haystack[i:i + width] == needle for i in range(len(haystack) - width + 1))


@dataclass(frozen=True)
class Hunk:
    """One hunk of a zero-context diff: `old` lines from `old_start` become `new`."""

    old_start: int  # 1-based; for a pure insertion, the line it follows (0 = file start)
    old: tuple[str, ...]
    new_start: int  # likewise in the new version
    new: tuple[str, ...]

    def old_span(self) -> range:
        return range(self.old_start, self.old_start + len(self.old))

    def new_span(self) -> range:
        return range(self.new_start, self.new_start + len(self.new))


def parse_hunks(patch: str) -> list[Hunk]:
    """The hunks of a `git diff -U0` patch for one file."""
    result, current = [], None
    for line in patch.splitlines():
        header = HUNK_HEADER.match(line)
        if header:
            if current:
                result.append(Hunk(current[0], tuple(current[1]), current[2], tuple(current[3])))
            current = (int(header[1]), [], int(header[3]), [])
        elif current and line[:1] == "-":
            current[1].append(line[1:])
        elif current and line[:1] == "+":
            current[3].append(line[1:])
    if current:
        result.append(Hunk(current[0], tuple(current[1]), current[2], tuple(current[3])))
    return result


def diff_hunks(old: str, new: str, path: str) -> list[Hunk]:
    return parse_hunks(git("diff", "-U0", "--no-color", "--no-ext-diff", old, new, "--", path,
                           check=False))


def is_binary(old: str, new: str, path: str) -> bool:
    numstat = git("diff", "--numstat", old, new, "--", path, check=False)
    return numstat.startswith("-\t-\t")


@lru_cache(maxsize=None)
def file_lines(tree: str, path: str) -> tuple[str, ...]:
    return tuple(git("show", f"{tree}:{path}", check=False).splitlines())


def blame(tree: str, path: str) -> list[tuple[str, int]]:
    """(commit, line number in that commit's version) for each line of `path` at `tree`."""
    origins, out = [], git("blame", "--porcelain", tree, "--", path, check=False)
    for line in out.splitlines():
        fields = line.split(" ")
        if len(fields) >= 3 and len(fields[0]) == 40 and not line.startswith("\t"):
            origins.append((fields[0], int(fields[1])))
    return origins


def applies(patch: str, tree: str) -> bool:
    """Whether `patch` applies forward to `tree` (checked in a throwaway index)."""
    with tempfile.NamedTemporaryFile(prefix="land-guard-index-") as index:
        env = {**os.environ, "GIT_INDEX_FILE": index.name}
        subprocess.run(["git", "read-tree", tree], env=env, check=True, capture_output=True)
        return subprocess.run(["git", "apply", "--cached", "--check", "-"], input=patch,
                              env=env, capture_output=True, text=True).returncode == 0


class Candidate:
    """The candidate's change against the integration head (`base`..`head`)."""

    def __init__(self, base: str, head: str):
        self.base, self.head = base, head
        self.files = [f for f in git("diff", "--name-only", base, head).splitlines() if f]
        self.deleted = set(git("diff", "--name-only", "--diff-filter=D", base, head).split())
        self.messages = git("log", "--format=%B", f"{base}..{head}")
        self._hunks: dict[str, list[Hunk]] = {}
        self._blame: dict[str, list[tuple[str, int]]] = {}
        added = git("diff", "-U0", "--no-color", "--no-ext-diff", base, head)
        # Every line the candidate adds, in any file: a deleted line found here moved.
        self.added_anywhere = {line[1:].strip() for line in added.splitlines()
                               if line.startswith("+") and not line.startswith("+++")}

    def hunks(self, path: str) -> list[Hunk]:
        if path not in self._hunks:
            self._hunks[path] = diff_hunks(self.base, self.head, path)
        return self._hunks[path]

    def base_origins(self, path: str) -> list[tuple[str, int]]:
        if path not in self._blame:
            self._blame[path] = blame(self.base, path)
        return self._blame[path]

    def announced(self, commit: str) -> bool:
        return commit[:7] in self.messages


@lru_cache(maxsize=None)
def upstream_commits(base: str, path: str) -> tuple[str, ...]:
    """Recent non-merge commits on the integration branch that touched `path`."""
    return tuple(git("log", "--no-merges", "--format=%H", f"--since={WINDOW_DAYS}.days",
                     base, "--", path).split())


def surviving_lines(candidate: Candidate, path: str, commit: str, hunk: Hunk) -> list[int]:
    """Integration-head line numbers (1-based) that `hunk` of `commit` added and still hold."""
    span = hunk.new_span()
    return [number for number, (origin, line) in enumerate(candidate.base_origins(path), 1)
            if origin == commit and line in span]


def covering(hunks: list[Hunk], lines: list[int]) -> list[Hunk] | None:
    """The candidate hunks that delete all of `lines`, or None if any line is kept."""
    found = []
    for number in lines:
        hunk = next((h for h in hunks if number in h.old_span()), None)
        if hunk is None:
            return None
        if hunk not in found:
            found.append(hunk)
    return found


def old_versions(commits: tuple[str, ...], path: str) -> list[tuple[str, ...]]:
    """The file before each recent upstream commit: where a stale tree's text comes from."""
    return [file_lines(f"{commit}^", path) for commit in commits]


def addition_survives(candidate: Candidate, path: str, commit: str, hunk: Hunk) -> list[int]:
    """The integration-head lines `hunk` added that still hold, or [] if none say anything."""
    survivors = surviving_lines(candidate, path, commit, hunk)
    base_text = file_lines(candidate.base, path)
    return survivors if any(substantive(base_text[n - 1]) for n in survivors) else []


def addition_undone(candidate: Candidate, path: str, survivors: list[int], hunk: Hunk,
                    recent: tuple[str, ...]) -> bool:
    """The candidate deletes all surviving lines of `hunk` and puts back only old text."""
    replacing = covering(candidate.hunks(path), survivors)
    if replacing is None:
        return False  # some of it is kept: an edit, not a revert
    base_text = file_lines(candidate.base, path)
    deleted = [base_text[n - 1].strip() for n in survivors if substantive(base_text[n - 1])]
    if all(line in candidate.added_anywhere for line in deleted):
        return False  # the lines moved
    put_back = [line for h in replacing for line in h.new]
    if hunk.old and contains_run(put_back, list(hunk.old)):
        return True  # the lines the hunk replaced are back
    # Nothing, or text the file had before some recent commit: a tree stale by
    # several commits holds the oldest of them.
    return not put_back or any(contains_run(list(version), put_back)
                               for version in old_versions(recent, path))


def neighbours(text: tuple[str, ...], first: int, count: int) -> tuple[str | None, str | None]:
    """The lines just before and after lines first..first+count-1 (1-based; None at an end)."""
    before = text[first - 2] if first >= 2 else None
    after = text[first + count - 1] if first + count - 1 < len(text) else None
    return before, after


def deletion_survives(candidate: Candidate, hunk: Hunk, path: str) -> bool:
    """`hunk` deleted lines that say something and the integration head still lacks them."""
    return (any(substantive(line) for line in hunk.old)
            and not contains_run(list(file_lines(candidate.base, path)), list(hunk.old)))


def deletion_undone(candidate: Candidate, path: str, commit: str, hunk: Hunk) -> bool:
    """The candidate inserts the lines `hunk` of `commit` deleted again where they were."""
    was_before, was_after = neighbours(file_lines(f"{commit}^", path), hunk.old_start,
                                       len(hunk.old))
    base_text = file_lines(candidate.base, path)
    for mine in candidate.hunks(path):
        if not contains_run(list(mine.new), list(hunk.old)):
            continue
        # A pure insertion's old_start is the line it follows.
        first = mine.old_start + 1 if not mine.old else mine.old_start
        before, after = neighbours(base_text, first, len(mine.old))
        if before == was_before or after == was_after:
            return True
    return False


def created(commit: str, path: str) -> bool:
    """Whether `commit` added `path`."""
    return git("diff-tree", "--no-commit-id", "-r", "--name-only", "--diff-filter=A", commit,
               "--", path).strip() == path


@dataclass
class Tally:
    """How many of a commit's hunks still on the integration head the candidate undoes."""

    surviving: int = 0
    undone: int = 0


def file_tally(candidate: Candidate, commit: str, path: str) -> Tally:
    """Count `commit`'s surviving hunks in `path` and those the candidate undoes."""
    tally = Tally()
    if path in candidate.deleted and not created(commit, path):
        # A stale tree only lacks files created after its base, and their
        # creators are checked; deleting an older file is deliberate, so its
        # later edits say nothing (the subsystem rewrites bcc00be2, 70090bec).
        return tally
    if is_binary(f"{commit}^", commit, path):
        binary = git("diff", "--binary", f"{commit}^", commit, "--", path, check=False)
        if not applies(binary, candidate.base):
            tally.surviving = 1
            tally.undone = int(applies(binary, candidate.head))
        return tally
    changed = path in candidate.files
    recent = upstream_commits(candidate.base, path) if changed else ()
    for hunk in diff_hunks(f"{commit}^", commit, path):
        if hunk.new:
            survivors = addition_survives(candidate, path, commit, hunk)
            if not survivors:
                continue
            undone = changed and addition_undone(candidate, path, survivors, hunk, recent)
        else:
            if not deletion_survives(candidate, hunk, path):
                continue
            undone = changed and deletion_undone(candidate, path, commit, hunk)
        tally.surviving += 1
        tally.undone += int(undone)
    return tally


def undone_files(candidate: Candidate, commit: str) -> list[str]:
    """The files in which the candidate undoes `commit`, if it undoes half of it or more.

    A stale tree lacks every change of an upstream commit it predates, except
    where the candidate's own edits overlap them; restoring a part of a commit
    (a feature dropped in a rewrite, a doc line put back) is the candidate's
    own work. So a commit counts as undone only when at least UNDONE_SHARE
    of its hunks still on the integration head, over every file it touched,
    are undone.
    """
    paths = git("diff-tree", "--no-commit-id", "-r", "--name-only", commit).split()
    mine = [path for path in paths if path in candidate.files]
    tallies = {path: file_tally(candidate, commit, path) for path in mine}
    undone = sum(tally.undone for tally in tallies.values())
    surviving = sum(tally.surviving for tally in tallies.values())
    # Files the candidate leaves alone only add surviving hunks: stop once
    # they settle it (most commits are settled by undone == 0 already).
    for path in (path for path in paths if path not in tallies):
        if undone < UNDONE_SHARE * surviving:
            break
        surviving += file_tally(candidate, commit, path).surviving
    if not undone or undone < UNDONE_SHARE * surviving:
        return []
    return [path for path, tally in tallies.items() if tally.undone]


def reverted_pairs(base: str, head: str = "HEAD") -> list[tuple[str, str]]:
    candidate = Candidate(base, head)
    commits = dict.fromkeys(commit for path in candidate.files
                            for commit in upstream_commits(base, path))
    return [(path, commit) for commit in commits if not candidate.announced(commit)
            for path in undone_files(candidate, commit)]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    base = sys.argv[1]
    files = git("diff", "--name-only", base, "HEAD").split()
    print(f"== candidate changes {len(files)} file(s) vs {base}:")
    print(git("diff", "--stat", base, "HEAD"), end="")
    pairs = reverted_pairs(base)
    if not pairs:
        return 0
    print(f"\nland: REFUSED: the candidate undoes {len(pairs)} upstream change(s) "
          f"on {base} (a stale tree, e.g. after `git reset --soft`):", file=sys.stderr)
    for path, commit in pairs:
        subject = git("log", "-1", "--format=%h %s", commit).strip()
        print(f"  {path}: reverts {subject}", file=sys.stderr)
    print("Rebuild the commit on the current tree (cherry-pick your own changes onto it); "
          "for a deliberate revert, name the commit (`Reverts: <sha>`) in the message.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
