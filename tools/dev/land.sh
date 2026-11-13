#!/usr/bin/env bash
# Land an agent's branch on variance-integration: the ONLY landing route
# (docs/PROCEDURES.md "Landing").
#
#   tools/dev/land.sh [branch] [--with BRANCH...]   run from the agent's own worktree
#
# --with queues other branches into the same train, after this one (their
# commits as their refs stand; their worktrees are not touched). It is for
# the coordinator waving a backlog through as one train; their owners then
# rebase, which drops the landed commits.
#
# Twice a branch rebased after it was built broke the integration branch (a
# red test, a commit that did not compile). This script therefore checks
# exactly the commit it lands:
#   1. rebase the branch onto the integration branch in its own worktree
#      (conflicts show up here, before any waiting), queue it (a ticket in
#      the landing queue) and take the landing lock (one lander at a time,
#      machine-wide);
#   2. the lock holder lands a MERGE TRAIN: every queued candidate, oldest
#      first (at most TRAIN_MAX), rebased in turn in the lander worktree onto
#      the integration head plus the candidates before it, each refused if it
#      undoes recent upstream changes (tools/dev/land_guard.py: a stale tree,
#      e.g. after `git reset --soft`). A candidate that conflicts with an
#      earlier one in the train waits for the next train; one that conflicts
#      with the integration head alone is refused. Landers of an older
#      land.sh blocked on the lock are adopted into the queue (adopt_waiters);
#   3. the checks the train's changed paths need run once, on the train's
#      head (tools/dev/land_scope.py classifies them; the table is in
#      docs/PROCEDURES.md "Landing"): tools/ci/check.sh --quick, limited to
#      the changed crates and their dependents, for Rust; the eval pytest
#      suite for eval/ and presets/; cheap link, parse and syntax checks for
#      docs and scripts; the full gate for build or gate files and anything
#      unclassified; and, for any change that can alter a stream, the landing
#      conformance check (tools/ci/landing_conformance.py: every shipped
#      preset's streams, bit-exact in the standard decoders and ours;
#      evaluation never decodes, so this is where conformance is proved). The
#      head of a train contains every candidate's commits, so each landed
#      commit sequence passed the same checks a lone landing would run (the
#      union over the train is a superset);
#   4. if they pass, fast-forward the integration worktree to the train's
#      head: every candidate lands. If they fail, bisect over the train's
#      prefixes: the longest passing prefix lands, the first candidate whose
#      prefix fails is refused with its log (an interaction failure is the
#      later candidate's, as in a serial landing), and the candidates after
#      it form the next train. If the integration branch moved during the
#      checks (a landing outside this script), retry (MAX_ATTEMPTS); when it
#      moved only in files the train does not touch, `cargo check
#      --workspace` suffices (both sides were fully checked on their own).
#   Every lander, leader or not, ends by reading its ticket's result: a
#   landed candidate's branch is moved onto the integration branch (in its
#   own worktree) and the script exits 0; a refused one prints its log.
#
# Each stage prints its wall time ("-- <stage>: <s>s"). The Python checks (the
# eval suite and the cheap checks) run beside the Rust checks (run_checks).
#
# Why one lander worktree and one target dir: the checks build the candidate
# in three profiles (clippy, tests, the gate-profile browser module). In an
# agent's own target dir, keyed by branch, every new branch started cold and
# every older one was hours behind integration: 3-10 min per landing, where
# the same checks on a warm target take 30-60 s (2026-11-11, nc-h264 one-line
# change: cold 2.8 min at -j4, warm 30 s). The lander tree always sits on the
# last landed commit, so each landing rebuilds only its own diff.
#
# The whole script is one function, read before it runs: rebasing the
# agent's worktree can rewrite this file mid-run.
#
# Environment: NC_LAND_TARGET (integration branch, default
# variance-integration), NC_INTEGRATION_TREE (its worktree, default
# ~/nanocodec-data/worktrees/variance-integration), NC_LANDER_TREE (the
# lander worktree, created on first use; owned by this script, never edit it;
# default ~/nanocodec-data/worktrees/lander), NC_LAND_TARGET_DIR (its cargo
# target dir, default ~/nanocodec-data/targets/lander), NC_LAND_LOCK (default ~/nanocodec-data/run/land.lock),
# NC_LAND_QUEUE (default ~/nanocodec-data/run/land-queue), NC_LAND_TRAIN_MAX
# (candidates per train, default 6; 1 lands one at a time), NC_LAND_FULL=1
# (run the full gate whatever changed), CARGO_BUILD_JOBS (default 8: warm
# landings are short bursts), NC_LAND_PYTEST_WORKERS (eval pytest workers,
# default 4: the suite runs beside the Rust checks).
set -euo pipefail

MAX_ATTEMPTS=10
DEFAULT_TRAIN_MAX=6
DEFAULT_BUILD_JOBS=8  # the per-agent cap (CLAUDE.md "Working in parallel")
DEFAULT_PYTEST_WORKERS=4
#: Lines of a refused candidate's check log printed by its own lander.
FAILURE_TAIL=60
#: Days a train's check logs are kept.
LOG_DAYS=3

main() {
    data_root=${NC_DATA_ROOT:-$HOME/nanocodec-data}
    # fast (default): land after formatting, `cargo check` and the cheap file
    # checks; everything else runs afterwards in `land.sh --verify`, beside the
    # next landings instead of in front of them. full: every check first.
    mode=${NC_LAND_MODE:-fast}
    local verify_only=0
    if [[ "${1:-}" == --verify ]]; then verify_only=1; shift; fi
    target=${NC_LAND_TARGET:-variance-integration}
    integration=${NC_INTEGRATION_TREE:-$data_root/worktrees/variance-integration}
    lander=${NC_LANDER_TREE:-$data_root/worktrees/lander}
    lock=${NC_LAND_LOCK:-$data_root/run/land.lock}
    queue=${NC_LAND_QUEUE:-$data_root/run/land-queue}
    train_max=${NC_LAND_TRAIN_MAX:-$DEFAULT_TRAIN_MAX}
    branch=$(git rev-parse --abbrev-ref HEAD)
    if [[ $# -gt 0 && "$1" != --with ]]; then branch=$1; shift; fi
    local -a with=()
    if [[ $# -gt 0 ]]; then
        [[ "$1" == --with ]] || die "usage: land.sh [branch] [--with BRANCH...]"
        shift
        with=("$@")
    fi
    # Agents' own CARGO_TARGET_DIR is deliberately ignored: the warm lander
    # target is the point.
    export CARGO_TARGET_DIR=${NC_LAND_TARGET_DIR:-$data_root/targets/lander}
    export PATH="$HOME/.cargo/bin:$PATH"
    cargo_shim  # every cargo call in a memory-capped scope (tools/dev/cargo-capped.sh)
    export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-$DEFAULT_BUILD_JOBS}
    export BUILD_JOBS=${BUILD_JOBS:-$CARGO_BUILD_JOBS}
    pytest_parallel=(-n "${NC_LAND_PYTEST_WORKERS:-$DEFAULT_PYTEST_WORKERS}")
    python3 -c 'import xdist' 2>/dev/null || pytest_parallel=()  # serial without pytest-xdist
    if [[ $verify_only == 1 ]]; then verify_target; exit $?; fi

    [[ "$(git rev-parse --abbrev-ref HEAD)" == "$branch" ]] ||
        die "check out $branch in this worktree first (never land another tree's branch)"
    [[ -z "$(git status --porcelain --untracked-files=no)" ]] ||
        die "uncommitted changes: commit or drop them first"
    [[ "$(git -C "$integration" rev-parse --abbrev-ref HEAD)" == "$target" ]] ||
        die "$integration is not on $target"

    printf '== land %s: rebase onto %s %s\n' "$branch" "$target" "$(git rev-parse --short=9 "$target")"
    git rebase --quiet "$target" || die "rebase conflict: resolve it, then run land.sh again"
    sha=$(git rev-parse HEAD)
    [[ "$sha" != "$(git rev-parse "$target")" ]] || die "nothing to land: $branch equals $target"

    mkdir -p "$queue/logs" "$(dirname "$lock")"
    prune_logs
    ticket=$queue/$(date +%s%N)-$$
    printf '%s %s %d\n' "$branch" "$sha" $$ >"$ticket.ticket"
    local other index=0
    for other in "${with[@]}"; do  # after ours in the queue: the same name, a later suffix
        index=$((index + 1))
        printf '%s %s %d\n' "$other" "$(git rev-parse --verify "$other^{commit}")" $$ \
            >"$ticket.w$(printf %03d $index).ticket"
    done

    # One lander at a time. flock(1) releases the lock when this script exits,
    # however it exits, so a crashed lander never blocks the next one.
    exec 9>>"$lock"
    if ! flock -n 9; then
        printf '== queued; waiting for the landing lock (held by: %s)\n' \
            "$(cat "$lock.holder" 2>/dev/null || echo unknown)"
        flock 9
    fi
    if [[ ! -e "$lander/.git" ]]; then
        printf '== creating the lander worktree %s (its first landing builds cold)\n' "$lander"
        git worktree add --quiet --detach "$lander" "$target"
    fi

    checked_base="" checked_train=""  # the last fully checked train and its base
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        ! done_here || finish
        lead_train "$attempt"
    done
    ! done_here || finish
    die "$target kept moving: gave up after $MAX_ATTEMPTS attempts"
}

# `land.sh --verify`: the full checks (tests, clippy, checkasm, the browser
# module, the eval suite, landing conformance) for everything landed since the
# last verified head, in its own worktree and target directory, so it never
# holds the landing lock. On success it records the head in run/verified-head
# (the coordinator mirrors and publishes only verified heads); on failure it
# writes run/verify-failed (head, base, log) and exits non-zero: fix forward or
# revert, then run it again. One verifier at a time.
verify_target() {
    local state=$data_root/run/verified-head failed=$data_root/run/verify-failed
    local tree=${NC_VERIFIER_TREE:-$data_root/worktrees/verifier} head base log scope status=0
    mkdir -p "$queue/logs"
    exec 8>>"$data_root/run/verify.lock"
    flock -n 8 || { printf '== a verifier is already running\n'; return 0; }
    head=$(git rev-parse "$target")
    base=$(cat "$state" 2>/dev/null || git rev-parse "$target~1")
    if [[ "$head" == "$base" ]]; then
        printf '== verified: %s %s (nothing new)\n' "$target" "${head:0:9}"
        return 0
    fi
    [[ -e "$tree/.git" ]] || git worktree add --quiet --detach "$tree" "$target"
    git -C "$tree" checkout --quiet --force --detach "$head"
    export CARGO_TARGET_DIR=${NC_VERIFY_TARGET_DIR:-$data_root/targets/verifier}
    mode=full
    log=$queue/logs/verify-${head:0:9}.log
    scope=$(cd "$tree" && python3 tools/dev/land_scope.py "$base") || return 1
    printf '== verifying %s %s..%s (log %s)\n' "$target" "${base:0:9}" "${head:0:9}" "$log"
    (cd "$tree" && run_checks "$scope") >"$log" 2>&1 || status=$?
    if ((status == 0)); then
        printf '%s\n' "$head" >"$state"
        rm -f "$failed"
        printf '== verified: %s %s\n' "$target" "${head:0:9}"
    else
        printf '%s %s %s\n' "$head" "$base" "$log" >"$failed"
        printf '== VERIFY FAILED at %s (since %s): the last %d lines of %s\n' \
            "${head:0:9}" "${base:0:9}" "$FAILURE_TAIL" "$log" >&2
        tail -n "$FAILURE_TAIL" "$log" >&2
    fi
    return "$status"
}

# Put a `cargo` on PATH that runs every cargo call (check.sh's, the browser
# build's, ours) through tools/dev/cargo-capped.sh with its own reserved slot:
# land.sh's builds are serialised by the landing lock, so they do not queue
# behind agents' builds for the shared slots, but are memory-capped like them.
cargo_shim() {
    shim_dir=$(mktemp -d)
    trap 'rm -rf -- "$shim_dir"; [[ -z "${ticket:-}" ]] || rm -f "$ticket.ticket" "$ticket".w*' EXIT
    ln -s "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cargo-capped.sh" "$shim_dir/cargo"
    export PATH="$shim_dir:$PATH" NC_BUILD_RESERVED=1
}

die() { printf 'land: %s\n' "$*" >&2; exit 1; }

# The ticket's result ("landed <sha> <log>", "refused <log>" or "conflict"):
# report it, move a landed branch onto the integration branch, and exit.
# Whether this lander's own ticket and every --with ticket have a result.
done_here() {
    local other
    [[ -e "$ticket.result" ]] || return 1
    for other in "$ticket".w*.ticket; do
        [[ ! -e "$other" || -e "${other%.ticket}.result" ]] || return 1
    done
}

finish() {
    local verdict landed log other name
    for other in "$ticket".w*.ticket; do
        [[ -e "$other" ]] || continue
        read -r name _ <"$other"
        printf '== --with %s: %s\n' "$name" "$(cat "${other%.ticket}.result")"
        rm -f "$other" "${other%.ticket}.result"
    done
    read -r verdict landed log <"$ticket.result"
    rm -f "$ticket.result"
    case $verdict in
        landed)
            # Only commits made since queueing remain (normally none).
            git rebase --quiet --onto "$target" "$sha" ||
                die "landed as ${landed:0:9}, but moving $branch onto $target failed: rebase it yourself"
            printf '== landed %s on %s (checks: %s)\n' "$(git log --oneline -1 "$landed")" "$target" "$log"
            exit 0 ;;
        conflict)
            die "rebase conflict with $target: rebase, resolve it, then run land.sh again" ;;
        *)
            printf '== refused: the last %d lines of %s\n' "$FAILURE_TAIL" "$landed" >&2
            tail -n "$FAILURE_TAIL" "$landed" >&2
            die "$branch refused (failed checks or reverted upstream work; log $landed): not landed" ;;
    esac
}

# Landers of an older land.sh (no queue) blocked on the landing lock join the
# train too: for each process waiting on the lock (/proc/locks) whose parent is
# a land.sh without a ticket, a ticket is written on its behalf, named by that
# lander's start time (so the queue stays oldest first), with its worktree's
# branch and HEAD read read-only. That lander never reads the result: when it
# gets the lock, its own rebase finds its commits already landed and it ends
# with an empty landing, or it re-runs the same checks and fails as the train
# found. Nothing about it is touched.
adopt_waiters() {
    local inode waiter parent args dir name sha started hz boot
    inode=$(stat -c %i "$lock")
    hz=$(getconf CLK_TCK)
    boot=$(awk '/^btime/ {print $2}' /proc/stat)
    for waiter in $(awk -v ino=":$inode" '$0 ~ "->" && index($0, ino " ") {print $(NF - 3)}' /proc/locks); do
        parent=$(awk '{print $4}' "/proc/$waiter/stat" 2>/dev/null) || continue
        args=$(tr '\0' ' ' <"/proc/$parent/cmdline" 2>/dev/null) || continue
        [[ "$args" == *land.sh* ]] || continue
        if grep -qsx "[^ ]* [^ ]* $parent" "$queue"/*.ticket; then continue; fi
        dir=$(readlink "/proc/$parent/cwd") || continue
        name=${args#*land.sh}
        name=${name// /}
        name=${name:-$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null)} || continue
        sha=$(git -C "$dir" rev-parse --verify --quiet "$name^{commit}") || continue
        started=$(awk '{print $22}' "/proc/$parent/stat")
        printf '%s %s %d\n' "$name" "$sha" "$parent" \
            >"$queue/$(((boot * hz + started) * (1000000000 / hz)))-$parent.ticket"
        printf '== adopted %s (pid %d, an older land.sh waiting for the lock) into the queue\n' "$name" "$parent"
    done
}

# The queued tickets (oldest first) whose landers are alive and which have no
# result yet, at most train_max. A dead lander's ticket is dropped.
queued_tickets() {
    local file name pid shown=0
    for file in $(ls "$queue"/*.ticket 2>/dev/null | LC_ALL=C sort); do
        read -r name _ pid <"$file" || continue
        if ! kill -0 "$pid" 2>/dev/null; then rm -f "$file" "${file%.ticket}.result"; continue; fi
        [[ ! -e "${file%.ticket}.result" ]] || continue
        printf '%s\n' "${file%.ticket}"
        shown=$((shown + 1))
        ((shown < train_max)) || break
    done
}

# The queued tickets as `rank:ticket`, in the order the train takes them: by
# rank, then arrival. Rank 0 is a docs-only candidate (lands first, at once),
# 2 one that changes the page (last: a page test failure then refuses only the
# page changes, and the bisection lands everything before them), 1 the rest
# (tools/dev/land_scope.py --rank).
ranked_tickets() {
    local ticket_of candidate rank
    for ticket_of in $(queued_tickets); do
        read -r _ candidate _ <"$ticket_of.ticket"
        rank=$(git diff --name-only --no-renames "$(git merge-base "$target" "$candidate")" "$candidate" |
            (cd "$lander" && python3 tools/dev/land_scope.py --rank) 2>/dev/null) || rank=1
        [[ "$rank" =~ ^[012]$ ]] || rank=1
        printf '%s:%s\n' "$rank" "$ticket_of"
    done | LC_ALL=C sort -s -t: -k1,1n
}

# Check logs older than LOG_DAYS, removed one by one.
prune_logs() {
    local old
    while IFS= read -r old; do rm -f -- "$old"; done < <(find "$queue/logs" -type f -mtime +"$LOG_DAYS")
}

result() { printf '%s\n' "$2" >"$1.result"; }  # result TICKET TEXT

# One train from the queue (step 2-4 above); results go to the tickets.
lead_train() {
    local base head ticket_of name candidate log n lo hi mid rank docs=0
    local -a names=() heads=() tickets=() ranks=()
    base=$(git rev-parse "$target")
    git -C "$lander" checkout --quiet --force --detach "$base"
    head=$base
    adopt_waiters
    for ticket_of in $(ranked_tickets); do
        rank=${ticket_of%%:*} ticket_of=${ticket_of#*:}
        read -r name candidate _ <"$ticket_of.ticket"
        if ! git -C "$lander" rebase --quiet --onto "$head" "$base" "$candidate" >/dev/null 2>&1; then
            git -C "$lander" rebase --abort 2>/dev/null || true
            git -C "$lander" checkout --quiet --force --detach "$head"
            if ((${#names[@]} == 0)); then
                result "$ticket_of" conflict
            else
                printf '== %s conflicts with the train before it: it waits for the next train\n' "$name"
            fi
            continue
        fi
        log=$queue/logs/$(basename "$ticket_of")-guard.log
        if ! (cd "$lander" && python3 tools/dev/land_guard.py "$head") >"$log" 2>&1; then
            cat "$log"
            printf '== %s reverts upstream work (listed above): refused\n' "$name"
            result "$ticket_of" "refused $log"
            git -C "$lander" checkout --quiet --force --detach "$head"
            continue
        fi
        if [[ "$(git -C "$lander" rev-parse HEAD)" == "$head" ]]; then
            printf '== %s: its commits are already on %s\n' "$name" "$target"
            result "$ticket_of" "landed $head already"
            continue
        fi
        fmt_fixup "$name"
        head=$(git -C "$lander" rev-parse HEAD)
        names+=("$name") heads+=("$head") tickets+=("$ticket_of") ranks+=("$rank")
    done
    n=${#heads[@]}
    ((n > 0)) || return 0

    # Docs-only candidates lead the train (ranked_tickets). They need seconds of
    # checks and cannot break the code behind them, so they land at once instead
    # of waiting for, and being bisected with, the rest of the train.
    while ((docs < n)) && [[ "${ranks[docs]}" == 0 ]]; do docs=$((docs + 1)); done
    if ((docs > 0 && docs < n)); then
        log=$queue/logs/$(basename "${tickets[0]}")-a$1-docs.log
        if check_prefix "${heads[docs - 1]}" "$base" "$log" && land_prefix "$docs" "$base" "$log"; then
            base=${heads[docs - 1]}
            names=("${names[@]:docs}") heads=("${heads[@]:docs}") tickets=("${tickets[@]:docs}")
            n=${#heads[@]}
        fi
    fi
    printf '%s: train of %d (%s, since %s)\n' "$branch" "$n" "${names[*]}" "$(date +%H:%M:%S)" >"$lock.holder"
    printf '\n== train %d (attempt %d) on %s %s: %s\n' "$n" "$1" "$target" "${base:0:9}" "${names[*]}"

    log=$queue/logs/$(basename "${tickets[0]}")-a$1-p$n.log
    if [[ -n "$checked_base" && "$checked_train" == "${names[*]}" ]] &&
        ! overlaps "$checked_base" "$base" "${heads[n - 1]}"; then
        printf '== %s moved only in files the train does not touch: build check only\n' "$target"
        if ! check_build "${heads[n - 1]}" "$base"; then
            refuse_from 0 "$log"
            return 0
        fi
    elif ! check_prefix "${heads[n - 1]}" "$base" "$log"; then
        # Bisect: the longest passing prefix (lo) lands, candidate hi (1-based) is refused.
        lo=0 hi=$n
        while ((hi - lo > 1)); do
            mid=$(((lo + hi) / 2))
            printf '\n== bisect: checking the first %d of %d\n' "$mid" "$n"
            local mid_log=$queue/logs/$(basename "${tickets[0]}")-a$1-p$mid.log
            if check_prefix "${heads[mid - 1]}" "$base" "$mid_log"; then lo=$mid; else hi=$mid log=$mid_log; fi
        done
        ((lo == 0)) || land_prefix "$lo" "$base" "$log" || return 0
        printf '== %s fails the checks on top of the %d before it: refused\n' "${names[hi - 1]}" "$lo"
        result "${tickets[hi - 1]}" "refused $log"
        return 0
    fi
    checked_base=$base checked_train="${names[*]}"
    land_prefix "$n" "$base" "$log" || true
}

# Format the lander's HEAD (`cargo fmt --all`) and commit any change as a train
# fixup after candidate $1. Branches that are fmt-clean alone can combine into
# an unformatted head (module and `use` lists both sides extended, 2026-11-12):
# a formatting-only difference is never a reason to refuse a validated branch,
# and the fixup lands with it, so the formatting check still passes on exactly
# what lands.
fmt_fixup() {
    (cd "$lander" && cargo fmt --all 2>/dev/null) || return 0  # a parse error: the checks report it
    [[ -n "$(git -C "$lander" status --porcelain --untracked-files=no)" ]] || return 0
    git -C "$lander" commit --quiet -a --author="land.sh <land.sh@nanocodec>" \
        -m "train: cargo fmt after $1 (combined with the branches landed before it)"
    printf '== %s: formatting of the combined head fixed (train fixup commit)\n' "$1"
}

# Fast-forward the integration branch to the head of the first $1 candidates,
# if it is still at $2; results "landed" for them. Fails if it moved.
land_prefix() {
    local k=$1 i tip=${heads[$1 - 1]}
    if [[ "$(git rev-parse "$target")" != "$2" ]] ||
        ! git -C "$integration" merge --ff-only --quiet "$tip"; then
        printf '== %s moved during the checks; rebuilding the train\n' "$target"
        return 1
    fi
    for ((i = 0; i < k; i++)); do result "${tickets[i]}" "landed ${heads[i]} $3"; done
    printf '== landed %d on %s: %s\n' "$k" "$target" "$(git log --oneline -1 "$tip")"
}

# Results "refused" for the train's candidates from index $1 on (a failed build check).
refuse_from() {
    local i
    for ((i = $1; i < ${#tickets[@]}; i++)); do result "${tickets[i]}" "refused $2"; done
}

# Whether integration commits between $1 and $2 touch any file the train
# (head $3, on top of $2) changes.
overlaps() {
    local arrived mine
    arrived=$(git diff --name-only "$1" "$2" | LC_ALL=C sort -u)
    mine=$(git diff --name-only "$2" "$3" | LC_ALL=C sort -u)
    [[ -n "$(LC_ALL=C comm -12 <(printf '%s\n' "$arrived") <(printf '%s\n' "$mine") | sed '/^$/d')" ]]
}

# `cargo check --workspace` on head $1 (when the train touches Rust at all).
check_build() {
    git -C "$lander" checkout --quiet --force --detach "$1"
    local scope
    scope=$(cd "$lander" && python3 tools/dev/land_scope.py "$2") || return 1
    grep -qx 'rust none' <<<"$scope" || (cd "$lander" && timed "cargo check" cargo check --workspace --quiet)
}

# The checks the changes from $2 to head $1 need, logged to $3.
check_prefix() {
    local scope status
    git -C "$lander" checkout --quiet --force --detach "$1"
    printf '== changed paths by class (tools/dev/land_scope.py)\n'
    scope=$(cd "$lander" && python3 tools/dev/land_scope.py "$2") || return 1
    printf '== checks on %s in %s (CARGO_TARGET_DIR=%s, %s jobs; log %s)\n' \
        "${1:0:9}" "$lander" "$CARGO_TARGET_DIR" "$CARGO_BUILD_JOBS" "$3"
    status=0
    (cd "$lander" && run_checks "$scope") 2>&1 | tee "$3" || status=$?
    return "$status"
}

# `timed NAME CMD...`: run CMD, then print its wall time.
timed() {
    local name=$1 began=$SECONDS status=0
    shift
    "$@" || status=$?
    printf -- '-- %s: %ds%s\n' "$name" $((SECONDS - began)) "$([[ $status == 0 ]] || echo ' FAILED')"
    return "$status"
}

# The checks the plan in $1 (land_scope.py output) asks for, on the lander
# worktree's HEAD (run from inside it). Two independent groups run side by
# side, each with its own output buffered and printed whole: the cheap checks
# and the eval suite (Python only, no cargo); the Rust checks (check.sh, the
# browser smoke for a web-only change, then the conformance check, whose
# gate build shares the lander target with check.sh's: built in parallel in
# a second target dir it would compile the encoder twice, on a machine whose
# landings are already compile-bound). Fails if either group fails, after
# both have finished.
run_checks() {
    local -A plan=()
    local key values root=$PWD scratch status=0 group
    while read -r key values; do plan[$key]=$values; done <<<"$1"
    if [[ "${NC_LAND_FULL:-0}" == 1 ]]; then plan[rust]=all plan[eval]=1 plan[web]=1 plan[page]=1 plan[conformance]=1; fi
    local -A pid=() failed=()
    scratch=$(mktemp -d)
    python_checks plan >"$scratch/python" 2>&1 &
    pid[python]=$!
    { rust_checks plan && conformance_check plan; } >"$scratch/rust" 2>&1 &
    pid[rust]=$!
    for group in python rust; do wait "${pid[$group]}" || { failed[$group]=1 status=1; }; done
    # a failed group's output comes last, where a refused lander's tail shows it
    for group in python rust; do [[ -n "${failed[$group]:-}" ]] || print_group "$scratch" "$group"; done
    for group in python rust; do [[ -z "${failed[$group]:-}" ]] || print_group "$scratch" "$group" FAILED; done
    rm -r -- "$scratch"
    return "$status"
}

print_group() {  # print_group DIR GROUP [FAILED]
    [[ ! -s "$1/$2" ]] || { printf '\n==== %s checks%s\n' "$2" "${3:+ $3}"; cat "$1/$2"; }
}

python_checks() {
    local -n p=$1
    [[ -z "${p[docs]}" ]] || timed docs python3 tools/dev/check_docs.py ${p[docs]} || return 1
    [[ -z "${p[python]}" ]] || timed "python byte-compile" python3 -m py_compile ${p[python]} || return 1
    if [[ -n "${p[shell]}" ]]; then
        for file in ${p[shell]}; do timed "bash -n $file" bash -n "$file" || return 1; done
    fi
    if [[ -n "${p[toml]}" ]]; then
        timed "toml parse" python3 -c 'import sys, tomllib; [tomllib.load(open(f, "rb")) for f in sys.argv[1:]]' \
            ${p[toml]} || return 1
    fi
    if [[ "${p[eval]}" == 1 && $mode != fast ]]; then
        (cd eval && timed "eval tests (${pytest_parallel[*]:-serial})" \
            python3 -m pytest -q -p no:cacheprovider "${pytest_parallel[@]}" tests) || return 1
    fi
}

rust_checks() {
    local -n p=$1
    [[ -z "${p[fuzz]}" ]] || timed "fuzz rustfmt" rustfmt --check --edition 2024 ${p[fuzz]} || return 1
    # check.sh runs the browser module itself when its selection includes nc-web.
    # The page tests in headless Chromium (minutes, and timing-sensitive on a loaded
    # machine: they failed four unrelated trains on 2026-11-13) run only when the
    # page itself changed; other changes get the module's node tests, and
    # tools/web/publish.sh runs the page tests before any build goes live.
    export NC_PAGE_TESTS=${p[page]:-1}
    local depth=--quick
    [[ $mode != fast ]] || depth=--fast
    case ${p[rust]} in
        none) ;;
        all) timed check.sh tools/ci/check.sh "$depth" || return 1 ;;
        *) timed check.sh tools/ci/check.sh "$depth" --crates "${p[rust]}" || return 1 ;;
    esac
    [[ $mode != fast ]] || return 0
    if [[ "${p[web]}" == 1 && "${p[rust]}" != all && " ${p[rust]} " != *" nc-web "* ]]; then
        timed "browser module and product smoke (web/ changed)" tools/ci/browser_smoke.sh || return 1
    fi
}

conformance_check() {
    local -n p=$1
    [[ "${p[conformance]}" == 1 && $mode != fast ]] || return 0
    # The gate profile: release code generation, incremental, no LTO (same streams).
    printf '== landing conformance (every shipped preset vs the standard decoders and ours)\n'
    timed "conformance build" cargo build --profile gate --quiet -p nc-cli || return 1
    NC_NANOCODEC_BIN="$CARGO_TARGET_DIR/gate" timed "landing conformance" \
        python3 tools/ci/landing_conformance.py || return 1
}

main "$@"
exit
