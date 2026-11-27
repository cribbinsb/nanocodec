#!/usr/bin/env bash
# Land an agent's branch on main: the ONLY landing route
# (docs/PROCEDURES.md "Landing").
#
#   tools/dev/land.sh [branch]     run from the agent's own worktree, on that branch
#   tools/dev/land.sh --verify [streams|tests]
#                                  the post-landing checks (the coordinator runs
#                                  them; verify_target below)
#
# Land first, verify beside (owner, 2026-11-13). There is no landing lock, no
# queue and no train (owner, 2026-11-15: on that day the queue cost agents
# ~21 h, median 7 min and worst 47 min a landing, behind other landers' checks):
#   1. rebase the branch onto the integration head in the agent's own tree;
#   2. the pre-land checks, in that tree and the shared landing target dir
#      (below), only what cannot wait (about 90 s): `cargo check --profile gate` of the
#      crates the diff touches and every crate depending on them, the unit
#      tests (`--lib --bins`) of the touched crates, in the gate profile,
#      byte-compile of touched Python and `bash -n` of touched shell scripts;
#   3. compare-and-swap the integration ref onto the candidate
#      (`git update-ref NEW OLD`: atomic, it fails if anyone landed meanwhile).
#      If the head moved, rebase again and retry; the checks re-run only when
#      the commits that arrived touch a file the candidate changes.
# Everything else (workspace-wide fmt and clippy, workspace tests, checkasm, the browser module,
# the eval suite, landing conformance) runs in `land.sh --verify`. Only a
# real stream mismatch there reverts a landing; other failures are fixed
# forward.
#
# The revert guard (tools/dev/land_guard.py) refuses a candidate that puts back the
# old text of recently landed commits (a stale tree after `git reset --soft` onto a
# moved integration head), naming the commits and files and the fix; `land.sh
# --reverts SHA` lands a deliberate revert.
#
# The integration worktree (the one with main checked out, normally the shared
# checkout) is not needed
# to land; after a landing it is moved to the new head when it was clean at an
# older commit (a stale index otherwise failed every later fast-forward).
#
# The pre-land cargo calls share one warm target dir, LAND_TARGET_NAME under
# ~/nanocodec-data/targets, taken by one lander at a time (a flock; landers
# already queue on the ref swap), at most LAND_BUILD_JOBS jobs, each cargo
# call in a scope of LAND_MEM_MAX (throttled from LAND_MEM_HIGH). 2026-11-27:
# landers checking from their own cold target dirs, several at once at 4-6.6 GB
# of rustc each, ran the PC out of memory.
#
# NC_LAND_MODE=full runs every check (as the verifier does) in the agent's
# own tree before landing, for a change considered risky.
#
# The whole script is one function, read before it runs: rebasing the
# agent's worktree can rewrite this file mid-run.
#
# Environment: NC_LAND_TARGET (integration branch, default main),
# NC_INTEGRATION_TREE (its worktree, default the worktree that has it checked
# out), NC_LAND_CHECK_BUDGET (seconds the pre-land checks should take, default
# 90: over it they warn), CARGO_BUILD_JOBS (the verifier's; default 8).
set -euo pipefail

MAX_ATTEMPTS=20
DEFAULT_BUILD_JOBS=8  # the per-agent cap (CLAUDE.md "Working in parallel")
#: The pre-land checks' shared target dir (under targets/), jobs and memory (header).
LAND_TARGET_NAME=land-shared
LAND_BUILD_JOBS=8
LAND_MEM_MAX=8G
LAND_MEM_HIGH=7G
DEFAULT_CHECK_BUDGET=90
DEFAULT_PYTEST_WORKERS=4
#: Seconds the revert guard may take (git blame of the touched files).
GUARD_TIMEOUT=60
#: Seconds a lander waits to resync the integration worktree (a few git calls).
SYNC_LOCK_TIMEOUT=30
#: First-parent commits searched for the one a stale integration index matches.
STALE_SEARCH_DEPTH=200
#: Lines of a failed verify log printed.
FAILURE_TAIL=60
#: Days check logs are kept.
LOG_DAYS=3

main() {
    data_root=${NC_DATA_ROOT:-$HOME/nanocodec-data}
    mode=${NC_LAND_MODE:-fast}
    target=${NC_LAND_TARGET:-main}
    logs=$data_root/run/land-logs
    export PATH="$HOME/.cargo/bin:$PATH"
    cargo_shim
    export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-$DEFAULT_BUILD_JOBS}
    export BUILD_JOBS=${BUILD_JOBS:-$CARGO_BUILD_JOBS}
    pytest_parallel=(-n "${NC_LAND_PYTEST_WORKERS:-$DEFAULT_PYTEST_WORKERS}")
    python3 -c 'import xdist' 2>/dev/null || pytest_parallel=()  # serial without pytest-xdist
    mkdir -p "$logs"
    prune_logs
    if [[ "${1:-}" == --verify ]]; then verify_target "${2:-}"; exit $?; fi
    if [[ "${1:-}" == --run-part ]]; then run_part_here "${2:-}"; exit $?; fi
    integration=${NC_INTEGRATION_TREE:-$(git worktree list --porcelain |
        awk -v branch="branch refs/heads/$target" '/^worktree /{tree=substr($0, 10)} $0 == branch {print tree}')}

    local reverts=()  # `--reverts SHA`: a deliberate revert of SHA (repeatable)
    while [[ "${1:-}" == --reverts ]]; do
        [[ -n "${2:-}" ]] || die "--reverts needs the reverted commit's sha"
        reverts+=("$2")
        shift 2
    done
    branch=$(git rev-parse --abbrev-ref HEAD)
    [[ $# -eq 0 || "$1" == "$branch" ]] ||
        die "check out $1 in this worktree first (never land another tree's branch)"
    [[ -z "$(git status --porcelain --untracked-files=no)" ]] ||
        die "uncommitted changes: commit or drop them first"
    export CARGO_TARGET_DIR=$data_root/targets/$LAND_TARGET_NAME CARGO_BUILD_JOBS=$LAND_BUILD_JOBS \
        NC_BUILD_MEM=$LAND_MEM_MAX NC_BUILD_MEM_HIGH=$LAND_MEM_HIGH
    local sha
    for sha in "${reverts[@]}"; do  # named in the message, the guard lets that revert pass
        git log -1 --format=%B | grep -q "Reverts: $sha" ||
            git -c core.hooksPath=/dev/null commit --quiet --amend --no-edit --trailer "Reverts: $sha"
    done

    local began=$SECONDS base checked_base="" attempt
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        base=$(git rev-parse "refs/heads/$target")
        drop_site_commit
        git rebase --quiet "$base" || { git rebase --abort 2>/dev/null || true
            die "rebase conflict with $target: rebase, resolve it, then run land.sh again"; }
        if [[ "$(git rev-parse HEAD)" == "$base" ]]; then
            printf '== nothing to land: %s is already on %s\n' "$branch" "$target"
            exit 0
        fi
        regenerate_site "$base"
        if [[ -z "$checked_base" ]]; then
            printf '== land %s on %s %s\n' "$branch" "$target" "${base:0:9}"
            revert_guard "$base"
            train_guard "$base"
            pre_land_checks "$base"
        elif touches_same_files "$checked_base" "$base"; then
            printf '== %s moved in files %s changes: checking again\n' "$target" "$branch"
            pre_land_checks "$base"
        fi
        checked_base=$base
        if git update-ref -m "land.sh: $branch" "refs/heads/$target" HEAD "$base" 2>/dev/null; then
            printf '== landed %s on %s in %ds (verify beside: land.sh --verify)\n' \
                "$(git log --oneline -1 HEAD)" "$target" $((SECONDS - began))
            conditions_notice "$base"
            sync_integration_tree
            exit 0
        fi
        printf '== %s moved (attempt %d): rebasing again\n' "$target" "$attempt"
    done
    die "$target kept moving: gave up after $MAX_ATTEMPTS attempts"
}

# Our benchmark conditions changed (the lock moved): every dev baseline point is stale.
# Re-encode them once, centrally, now, instead of the next screen paying mid-task (owner,
# 2026-11-25; tools/dev/rebaseline.py).
conditions_notice() {
    if ! git diff --quiet "$1" HEAD -- eval/tests/test_conditions_lock.py; then
        printf '\n!!! this landing changes our locked benchmark conditions\n'
        printf '!!! run NOW: systemd-run --user --scope -p MemoryMax=2G python3 tools/dev/rebaseline.py\n'
        printf '!!! (re-encodes every dev baseline point once; until then every screen re-encodes them)\n\n'
    fi
    # a failed or missing rebaseline is never silent (nceval.baseline_state)
    (cd eval && python3 -c 'from nceval import baseline_state as b; n = b.notice(); n and print(n)') ||
        true
}

# Put a `cargo` on PATH that runs every cargo call through
# tools/dev/cargo-capped.sh, memory-capped. The reserved slot: a lander never
# waits behind other agents' builds for the shared slots.
cargo_shim() {
    shim_dir=$(mktemp -d "${TMPDIR:-/tmp}/nanocodec-land-XXXXXX")
    trap 'rm -rf -- "$shim_dir"' EXIT
    ln -s "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cargo-capped.sh" "$shim_dir/cargo"
    export PATH="$shim_dir:$PATH" NC_BUILD_RESERVED=1
}

die() { printf 'land: %s\n' "$*" >&2; exit 1; }

# Check logs older than LOG_DAYS, removed one by one.
prune_logs() {
    local old
    while IFS= read -r old; do rm -f -- "$old"; done < <(find "$logs" -type f -mtime +"$LOG_DAYS")
    while IFS= read -r old; do rmdir -- "$old"; done < <(find "$logs" -mindepth 1 -type d -empty)
}

# `timed NAME CMD...`: run CMD, then print its wall time.
timed() {
    local name=$1 began=$SECONDS status=0
    shift
    "$@" || status=$?
    printf -- '-- %s: %ds%s\n' "$name" $((SECONDS - began)) "$([[ $status == 0 ]] || echo ' FAILED')"
    return "$status"
}

# Whether the commits from $1 to $2 (arrived on the integration branch since
# the last check) touch any file HEAD changes on top of $2.
touches_same_files() {
    local arrived mine
    arrived=$(git diff --name-only --no-renames "$1" "$2" | LC_ALL=C sort -u)
    mine=$(git diff --name-only --no-renames "$2" HEAD | LC_ALL=C sort -u)
    [[ -n "$(LC_ALL=C comm -12 <(printf '%s\n' "$arrived") <(printf '%s\n' "$mine") | sed '/^$/d')" ]]
}

# The revert guard, as a warning: its findings become a `Land-warning:`
# trailer on HEAD's commit message.
# The revert guard (tools/dev/land_guard.py): a candidate whose diff puts back the
# old text of recently landed commits is refused, with the commits and files it would
# revert and the fix (2026-11-21: twice in a day a `git reset --soft` squash onto a
# moved integration head carried its stale tree, once 45 files, and only luck caught
# it; the guard only warned then). `land.sh --reverts SHA` lands a deliberate revert.
revert_guard() {
    local found status=0
    found=$(timeout "$GUARD_TIMEOUT" python3 tools/dev/land_guard.py "$1" 2>&1 >/dev/null) ||
        status=$?
    ((status == 0)) && return 0
    if ((status == 124)); then
        printf '== WARNING: land_guard.py did not finish in %ss: landing unchecked\n' "$GUARD_TIMEOUT"
        return 0
    fi
    if [[ $found != *"Fix: rebuild the commit"* ]]; then  # the guard itself failed
        printf '%s\n== WARNING: land_guard.py failed (exit %s): landing unchecked\n' \
            "$found" "$status" >&2
        return 0
    fi
    printf '%s\n' "$found" >&2
    die "refused: this commit reverts other agents' work (above); see the fix"
}

# A preset change whose nceval.train verdict tripped a guard (PASS-GUARD) lands only
# with a `Guard-justification:` line in its commit message (tools/dev/land_train_guards.py;
# 2026-11-23: a tripped worst-1 % guard went unread and H.264 veryfast was reverted).
train_guard() {
    local found status=0
    found=$(python3 tools/dev/land_train_guards.py "$1" 2>&1 >/dev/null) || status=$?
    ((status == 0)) && return 0
    if [[ $found != *"Fix: "* ]]; then  # the check itself failed
        printf '%s\n== WARNING: land_train_guards.py failed (exit %s): guards unchecked\n' \
            "$found" "$status" >&2
        return 0
    fi
    printf '%s\n' "$found" >&2
    die "refused: nceval.train guards tripped without a justification (above)"
}

#: Author of the commits land.sh makes on a lander's branch (regenerated files).
SITE_AUTHOR="land.sh <land.sh@nanocodec>"
SITE_SUBJECT="site: regenerate the generated results data (python3 -m nceval.site)"
#: What eval/nceval/site.py writes.
SITE_OUTPUTS=(web/results/data web/docs/examples)

# The generated site data (eval/nceval/site.py), rewritten
# and committed on top of the branch when the branch changes one of their
# inputs since base $1: they are generated at landing, never left stale for
# the verifier to find (2026-11-15: a stale chart.json failed a 56-min verify).
# A failure to generate only warns.
regenerate_site() {
    [[ -f eval/nceval/site.py ]] || return 0
    (cd eval && timed "site data" python3 -m nceval.site --changed-since "$1" >/dev/null) || {
        printf '== WARNING: python3 -m nceval.site failed: generated files not refreshed\n'; return 0; }
    git add -A -- "${SITE_OUTPUTS[@]}"
    git diff --cached --quiet && return 0
    git -c core.hooksPath=/dev/null commit --quiet --author="$SITE_AUTHOR" -m "$SITE_SUBJECT"
    printf '== generated files refreshed: %s\n' "$(git log --oneline -1 HEAD)"
}

# Before rebasing again: drop the previous attempt's regeneration commit (it is
# made again on the new head, instead of conflicting with someone else's).
drop_site_commit() {
    [[ "$(git log -1 --format='%an <%ae>%n%s' HEAD)" == "$SITE_AUTHOR"$'\n'"$SITE_SUBJECT" ]] || return 0
    git reset --quiet --hard HEAD~1
}

# What cannot wait (header, step 2), on HEAD against base $1.
pre_land_checks() {
    local lock_fd
    mkdir -p "$data_root/locks"
    exec {lock_fd}>"$data_root/locks/$LAND_TARGET_NAME"
    if ! flock -n "$lock_fd"; then
        printf '== waiting for another lander'"'"'s checks (shared target %s)\n' "$CARGO_TARGET_DIR"
        flock "$lock_fd"
    fi
    local began=$SECONDS key values budget=${NC_LAND_CHECK_BUDGET:-$DEFAULT_CHECK_BUDGET}
    local -A plan=()
    local scope
    scope=$(python3 tools/dev/land_scope.py "$1" 2>/dev/null) || die "tools/dev/land_scope.py failed: not landed"
    while read -r key values; do plan[$key]=$values; done <<<"$scope"
    [[ -z "${plan[python]:-}" ]] || timed "python byte-compile" python3 -m py_compile ${plan[python]} ||
        die "python byte-compile failed: not landed"
    local file
    for file in ${plan[shell]:-}; do bash -n "$file" || die "bash -n $file failed: not landed"; done
    local -a checked=() tested=()
    case ${plan[rust]:-none} in
        none) ;;
        all) checked=(--workspace) ;;
        *) for key in ${plan[rust]}; do checked+=(-p "$key"); done ;;
    esac
    for key in ${plan[touched]:-}; do tested+=(-p "$key"); done
    if ((${#checked[@]})); then
        timed "cargo check (gate) ${checked[*]}" cargo check --profile gate --quiet "${checked[@]}" ||
            die "cargo check failed: not landed"
    fi
    # Formatting and lints of the touched crates: seconds here, and each one
    # left to the verifier stopped its stream check for hours (2026-11-15).
    if ((${#tested[@]})); then
        timed "fmt ${tested[*]}" cargo fmt --check "${tested[@]}" ||
            die "cargo fmt --check failed (run cargo fmt): not landed"
        timed "clippy (gate) ${tested[*]}" cargo clippy --profile gate --quiet --all-targets "${tested[@]}" -- -D warnings ||
            die "clippy failed: not landed"
        # Feature builds (`analysis`): code behind a feature broke the verifier's
        # web build twice on 2026-11-16 while the default build passed here.
        timed "cargo check --all-features ${tested[*]}" cargo check --profile gate --quiet --all-features "${tested[@]}" ||
            die "cargo check --all-features failed: not landed"
    fi
    if ((${#tested[@]})); then
        # `--lib` is an error for a package with no library (the CLI), so it
        # is passed only when every touched package has one.
        local lib=--lib
        for key in ${plan[touched]:-}; do
            cargo metadata --no-deps --format-version 1 --offline 2>/dev/null |
                python3 -c 'import json,sys; p=[x for x in json.load(sys.stdin)["packages"] if x["name"]==sys.argv[1]]; sys.exit(0 if p and any("lib" in t["kind"] for t in p[0]["targets"]) else 1)' "$key" ||
                lib=
        done
        timed "unit tests ${tested[*]}" cargo test --profile gate --quiet $lib --bins "${tested[@]}" ||
            die "unit tests failed: not landed"
    fi
    # Kernel changes: no SIMD outlined from its #[target_feature] context. The
    # verifier ran this alone, so a closure in a kernel (2a1aacb7f, 2026-11-24) failed
    # every verify after it until fixed forward.
    if [[ " ${plan[touched]:-} " == *" nanocodec-dsp "* ]]; then
        timed "build nc-checkasm (gate)" cargo build --profile gate --quiet -p nc-checkasm ||
            die "nc-checkasm build failed: not landed"
        timed "outlined SIMD (nc-checkasm)" python3 tools/ci/check_outlined_simd.py \
            "${CARGO_TARGET_DIR:-target}/gate/nc-checkasm" ||
            die "SIMD code outside its #[target_feature] context: not landed"
    fi
    if [[ $mode == full ]]; then
        printf '== NC_LAND_MODE=full: every check, in this tree (target %s)\n' "$CARGO_TARGET_DIR"
        run_checks "$scope" || die "full checks failed: not landed"
    fi
    exec {lock_fd}>&-
    printf -- '-- pre-land checks: %ds\n' $((SECONDS - began))
    (((SECONDS - began) <= budget)) ||
        printf '== note: the pre-land checks took over the %ds budget\n' "$budget"
}

# After a landing: move the integration worktree to the new head, when its
# index is the tree of an older commit and it has no unstaged edits (stale,
# not someone's work). update-ref moves the branch it has checked out, not
# its files. Landing does not depend on this tree: a problem is only logged.
sync_integration_tree() {
    local tree=$integration head index commit
    [[ -e "$tree/.git" ]] || return 0
    [[ "$(git -C "$tree" symbolic-ref -q HEAD)" == "refs/heads/$target" ]] || return 0
    exec 7>>"$data_root/run/integration-sync.lock"
    if ! flock -w "$SYNC_LOCK_TIMEOUT" 7; then
        printf '== note: %s not synced: its lock was held %ds (by: %s)\n' "$tree" "$SYNC_LOCK_TIMEOUT" \
            "$(cat "$data_root/run/integration-sync.holder" 2>/dev/null || echo unknown)"
        return 0
    fi
    printf '%s pid %d since %s\n' "$branch" $$ "$(date +%T)" >"$data_root/run/integration-sync.holder"
    git -C "$tree" update-index -q --refresh >/dev/null || true
    if ! git -C "$tree" diff --quiet; then
        printf '== note: %s has unstaged edits: not synced\n' "$tree"
        return 0
    fi
    head=$(git rev-parse "refs/heads/$target")
    index=$(git -C "$tree" write-tree)
    [[ "$index" != "$(git rev-parse "$head^{tree}")" ]] || return 0
    for commit in $(git rev-list --first-parent -n "$STALE_SEARCH_DEPTH" "$head"); do
        [[ "$(git rev-parse "$commit^{tree}")" == "$index" ]] || continue
        if git -C "$tree" read-tree -u -m "$commit" "$head"; then
            printf '%s %s: synced %s from %s to %s\n' "$(date +%FT%T)" "$branch" "$tree" "${commit:0:9}" \
                "${head:0:9}" | tee -a "$logs/integration-sync.log"
        fi
        return 0
    done
    printf '== note: %s matches none of the last %d commits of %s: left alone\n' \
        "$tree" "$STALE_SEARCH_DEPTH" "$target" | tee -a "$logs/integration-sync.log"
}

# `land.sh --verify [streams|tests]` (default both, in that order): checks of
# everything landed since the last head each part verified, in its own worktree
# and target dir, so neither holds up landing nor the other part.
#   streams: the landing conformance check alone (minutes). Success records the
#     head in run/verified-streams-head (mirrors and publishing use it);
#     failure writes run/verify-failed (head, base, log): a stream mismatch,
#     the only failure that reverts a landing.
#   tests: fmt, clippy, workspace tests, checkasm, the browser module, the eval
#     suite (tens of minutes). Success records run/verified-head and empties
#     run/fix-forward; failure appends `head base log` to run/fix-forward: its
#     owner fixes forward, nothing is reverted (2026-11-15: the two failed
#     verifies that day were a stale generated file and a renamed label).
# Each success also appends `PART HEAD DATE` to run/verified-history, never
# pruned: a release picks its commit from it (tools/release/release.py), long
# after the check logs are gone.
# Both parts copy the binaries they built for the head (the gate CLI, nc-checkasm)
# into ~/nanocodec-data/charts/builds/<commit>/ (cache_binaries): tools/dev/binary_for.sh.
# A part already running elsewhere is skipped (the next call covers its heads).
verify_target() {
    local status=0
    case ${1:-both} in
        streams) verify_streams || status=$? ;;
        tests) verify_tests || status=$? ;;
        both) verify_streams || status=$?; verify_tests || status=$? ;;
        *) die "usage: land.sh --verify [streams|tests]" ;;
    esac
    return "$status"
}

verify_streams() {
    local state=$data_root/run/verified-streams-head failed=$data_root/run/verify-failed
    local head base log scope tree status=0
    exec 8>>"$data_root/run/verify-streams.lock"
    flock -n 8 || { printf '== a streams verifier is already running\n'; return 0; }
    head=$(git rev-parse "$target")
    base=$(cat "$state" 2>/dev/null || cat "$data_root/run/verified-head" 2>/dev/null || git rev-parse "$target~1")
    if [[ "$head" == "$base" ]]; then
        printf '== streams verified: %s %s (nothing new)\n' "$target" "${head:0:9}"
        return 0
    fi
    verifier_tree "${NC_VERIFIER_STREAMS_TREE:-$data_root/worktrees/verifier-streams}" "$head"
    log=$logs/verify-streams-${head:0:9}.log
    scope=$(cd "$tree" && python3 tools/dev/land_scope.py "$base") || return 1
    printf '== verifying streams %s %s..%s (log %s)\n' "$target" "${base:0:9}" "${head:0:9}" "$log"
    local target_dir=${NC_VERIFY_STREAMS_TARGET_DIR:-$data_root/targets/verifier-streams}
    verify_part streams "$tree" "$head" "$scope" "$target_dir" >"$log" 2>&1 || status=$?
    if ((status == 0)); then
        printf '%s\n' "$head" >"$state"
        record_verified streams "$head"
        rm -f "$failed"
        printf '== streams verified: %s %s\n' "$target" "${head:0:9}"
    else
        printf '%s %s %s\n' "$head" "$base" "$log" >"$failed"
        report_failure "STREAMS FAILED (revert or fix at once)" "$head" "$base" "$log"
    fi
    return "$status"
}

verify_tests() {
    local state=$data_root/run/verified-head list=$data_root/run/fix-forward
    local head base log scope tree status=0
    exec 9>>"$data_root/run/verify.lock"
    flock -n 9 || { printf '== a tests verifier is already running\n'; return 0; }
    head=$(git rev-parse "$target")
    base=$(cat "$state" 2>/dev/null || git rev-parse "$target~1")
    if [[ "$head" == "$base" ]]; then
        printf '== tests verified: %s %s (nothing new)\n' "$target" "${head:0:9}"
        return 0
    fi
    verifier_tree "${NC_VERIFIER_TREE:-$data_root/worktrees/verifier}" "$head"
    log=$logs/verify-${head:0:9}.log
    # the conformance check is the streams part's
    scope=$(cd "$tree" && python3 tools/dev/land_scope.py "$base") || return 1
    scope=$(sed 's/^conformance .*/conformance 0/' <<<"$scope")
    printf '== verifying tests %s %s..%s (log %s)\n' "$target" "${base:0:9}" "${head:0:9}" "$log"
    printf '== live: tail -f %s.parts/{python,rust}\n' "$log"
    local target_dir=${NC_VERIFY_TARGET_DIR:-$data_root/targets/verifier} began_at
    began_at=$(date +%s)
    # NC_WEB_TESTS_SINCE: the page tests run only where these changes reach (tools/ci/web_tests.py).
    NC_CHECK_PARTS=$log.parts NC_WEB_TESTS_SINCE=$base \
        verify_part tests "$tree" "$head" "$scope" "$target_dir" --since "$began_at" \
        >"$log" 2>&1 || status=$?
    if ((status == 0)); then
        printf '%s\n' "$head" >"$state"
        record_verified tests "$head"
        # Only this part's lines go: the nightly verifier's (tools/dev/verify_schedule.py,
        # logs under land-logs/nightly-*) stay until their own checks pass.
        if [[ -f "$list" ]]; then
            grep -v ' [^ ]*/verify-[0-9a-f]*\.log$' "$list" >"$list.new" || true
            mv "$list.new" "$list"
        fi
        printf '== tests verified: %s %s\n' "$target" "${head:0:9}"
    else
        printf '%s %s %s\n' "$head" "$base" "$log" >>"$list"
        report_failure "TESTS FAILED (fix forward; listed in $list)" "$head" "$base" "$log"
    fi
    return "$status"
}

# `verify_part PART TREE HEAD SCOPE TARGET_DIR [--since EPOCH]`: one verify part's checks.
# On the build box by default (owner, 2026-11-21: the PC carries no material compute):
# nc-remote syncs the verifier tree, the clips landing conformance reads and the
# verified-stream cache, runs `land.sh --run-part PART` there, and brings back the
# cache (new verified streams) and the gate binaries for the landed-commit cache;
# only the verdict (the exit status) and the log decide here, where the verified
# head and its history line are written. NC_VERIFY_LOCAL=1 runs the checks here.
verify_part() {
    local part=$1 tree=$2 head=$3 scope=$4 target_dir=$5 status=0
    shift 5
    if [[ "${NC_VERIFY_LOCAL:-0}" == 1 ]]; then
        (cd "$tree" && CARGO_TARGET_DIR=$target_dir run_part "$part" "$scope") || status=$?
        cache_binaries "$tree" "$head" "$target_dir" "$@"
        return "$status"
    fi
    local fetched=$tree/remote-out/verify-target clips=() clip binaries=(--fetch target/gate/nanocodec)
    [[ "$part" == tests ]] && binaries+=(--fetch target/gate/nc-checkasm)
    # The browser smoke decodes every conformance corpus, and the box's AMI has all but
    # H.264's (fetched on the PC): sent each time, only changes go up (2026-11-27: the
    # box held partial H.264 folders and 216 decodes failed on missing streams).
    local corpora=() corpus
    if [[ "$part" == tests ]]; then
        for corpus in streams supplementary jm; do
            corpora+=(--sync "$data_root/conformance/h264/$corpus")
        done
    fi
    local changed=()  # the box's tree has no git: the page tests' selection goes with it
    if [[ -n "${NC_WEB_TESTS_SINCE:-}" ]]; then
        changed=(--env "NC_WEB_TESTS_CHANGED=$(git -C "$tree" diff --name-only "$NC_WEB_TESTS_SINCE" HEAD | tr '\n' ' ')")
    fi
    local verified=$data_root/run/verified
    local cache=${verified#"$HOME"/}  # the same place under the box's home
    for clip in $(cd "$tree" && python3 tools/ci/landing_conformance.py --list-clips); do
        clips+=(--clip "$clip")
    done
    rm -rf "$fetched"
    mkdir -p "$verified"
    "$HOME/nanocodec-data/bin/nc-remote" run --worktree "$tree" --name "verifier-$part" \
        --env "NC_VERIFY_SCOPE=$(base64 -w0 <<<"$scope")" \
        "${changed[@]}" --env NC_CHECK_STRICT=1 --env NC_CHECKASM_REQUIRE=avx2,avx512 \
        --sync "$verified" --fetch "~/$cache" "${corpora[@]}" \
        "${clips[@]}" "${binaries[@]}" \
        --fetch-to "$fetched" -- "NC_VERIFIED_DIR=\$HOME/$cache NC_CHECK_GROUPS='rust conformance' \
NC_CONFORMANCE_WORKERS=\$(( \$(nproc) / 4 )) \
bash tools/dev/land.sh --run-part $part" || status=$?
    if [[ "$part" == tests ]]; then  # the eval suite needs this repository's git history
        (cd "$tree" && NC_CHECK_GROUPS=python run_part tests "$scope") || status=$?
    fi
    [[ -d "$fetched/target/gate" ]] && cache_binaries "$tree" "$head" "$fetched/target" "$@"
    return "$status"
}

# `run_part PART SCOPE`: the checks of a verify part, in the current tree.
run_part() {
    case $1 in
        streams) timed "streams" conformance_check_plan "$2" ;;
        tests) run_checks "$2" ;;
        *) die "unknown verify part $1" ;;
    esac
}

# `land.sh --run-part PART` on the build box (verify_part): the scope comes in
# NC_VERIFY_SCOPE (base64), CARGO_TARGET_DIR is the job's.
run_part_here() {
    run_part "$1" "$(base64 -d <<<"${NC_VERIFY_SCOPE:?no NC_VERIFY_SCOPE}")"
}

# `cache_binaries TREE HEAD TARGET_DIR [--since EPOCH]`: the binaries the verifier
# has just built for HEAD, copied into the landed-commit cache (charts/builds/<commit>/,
# tools/dev/build_cache.py), so measuring HEAD later needs no compile. Whatever the
# checks' outcome; a binary that is not HEAD's is refused there. Only a note on failure.
cache_binaries() {
    (cd "$1" && python3 tools/dev/build_cache.py store "$2" "$3/gate" "${@:4}") ||
        printf '== note: build_cache.py store failed for %s\n' "${2:0:9}"
}

# `record_verified PART HEAD`: the durable record of one verified head.
record_verified() {
    printf '%s %s %s\n' "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$data_root/run/verified-history"
}

# `verifier_tree DIR HEAD`: the verifier worktree DIR (created on first use) at
# HEAD; sets the caller's $tree.
verifier_tree() {
    tree=$1
    [[ -e "$tree/.git" ]] || git worktree add --quiet --detach "$tree" "$target"
    git -C "$tree" checkout --quiet --force --detach "$2"
}

report_failure() {  # report_failure WHAT HEAD BASE LOG
    printf '== %s at %s (since %s): the last %d lines of %s\n' \
        "$1" "${2:0:9}" "${3:0:9}" "$FAILURE_TAIL" "$4" >&2
    tail -n "$FAILURE_TAIL" "$4" >&2
}

# The conformance check for the plan in $1 (land_scope.py output).
conformance_check_plan() {
    local -A plan=()
    local key values
    while read -r key values; do plan[$key]=$values; done <<<"$1"
    conformance_check plan
}

# Every check the plan in $1 (land_scope.py output) asks for, on the current
# tree (the verifier, or NC_LAND_MODE=full). Three independent groups run side
# by side, each writing its own file (NC_CHECK_PARTS, beside the log, so a
# running check can be followed: `tail -f LOG.parts/rust`; else a temporary
# directory): the cheap checks and the eval suite (Python only); the Rust
# checks (check.sh; the browser smoke for a web-only change); the conformance
# check, in the same target dir (cargo's lock serialises its gate build with
# check.sh's, so the encoder is compiled once). Fails if any group fails,
# after all have finished; a failed group's output is printed last.
run_checks() {
    local -A plan=()
    local key values scratch status=0 group
    while read -r key values; do plan[$key]=$values; done <<<"$1"
    if [[ "${NC_LAND_FULL:-0}" == 1 ]]; then plan[rust]=all plan[eval]=1 plan[web]=1 plan[page]=1 plan[conformance]=all plan[checkasm]=1; fi
    local -A pid=() failed=()
    scratch=${NC_CHECK_PARTS:-$(mktemp -d "${TMPDIR:-/tmp}/nanocodec-land-XXXXXX")}
    mkdir -p "$scratch"
    # NC_CHECK_GROUPS: a subset (verify_part runs the Rust checks on the build box and
    # the Python ones here, where the eval suite has the repository's git history).
    local -a groups=(${NC_CHECK_GROUPS:-python rust conformance})
    for group in "${groups[@]}"; do
        case $group in
            python) python_checks plan >"$scratch/python" 2>&1 & ;;
            rust) rust_checks plan >"$scratch/rust" 2>&1 & ;;
            conformance) conformance_check plan >"$scratch/conformance" 2>&1 & ;;
        esac
        pid[$group]=$!
    done
    for group in "${groups[@]}"; do wait "${pid[$group]}" || { failed[$group]=1 status=1; }; done
    for group in "${groups[@]}"; do [[ -n "${failed[$group]:-}" ]] || print_group "$scratch" "$group"; done
    for group in "${groups[@]}"; do [[ -z "${failed[$group]:-}" ]] || print_group "$scratch" "$group" FAILED; done
    [[ -n "${NC_CHECK_PARTS:-}" ]] || rm -r -- "$scratch"
    return "$status"
}

print_group() {  # print_group DIR GROUP [FAILED]
    [[ ! -s "$1/$2" ]] || { printf '\n==== %s checks%s\n' "$2" "${3:+ $3}"; cat "$1/$2"; }
}

python_checks() {
    local -n p=$1
    # Every tracked document, whatever changed (under a second): a rename or a deleted
    # section breaks links in files the change did not touch.
    timed docs python3 tools/dev/check_docs.py --all || return 1
    [[ -z "${p[python]}" ]] || timed "python byte-compile" python3 -m py_compile ${p[python]} || return 1
    if [[ -n "${p[shell]}" ]]; then
        for file in ${p[shell]}; do timed "bash -n $file" bash -n "$file" || return 1; done
    fi
    if [[ -n "${p[toml]}" ]]; then
        timed "toml parse" python3 -c 'import sys, tomllib; [tomllib.load(open(f, "rb")) for f in sys.argv[1:]]' \
            ${p[toml]} || return 1
    fi
    if [[ "${p[eval]}" == 1 ]]; then
        (cd eval && timed "eval tests (${pytest_parallel[*]:-serial})" \
            python3 -m pytest -q -p no:cacheprovider "${pytest_parallel[@]}" tests \
                $(for skipped in ${p[eval_skip]:-}; do printf -- '--ignore=tests/%s ' "$skipped"; done)) ||
            return 1
    fi
}

rust_checks() {
    local -n p=$1
    [[ -z "${p[fuzz]}" ]] || timed "fuzz rustfmt" rustfmt --check --edition 2024 ${p[fuzz]} || return 1
    # check.sh runs the browser module itself when its selection includes nc-web.
    # The page tests in headless Chromium (minutes, and timing-sensitive on a loaded
    # machine) run only when the page itself changed; tools/web/publish.sh runs
    # them before any build goes live.
    export NC_PAGE_TESTS=${p[page]:-1} NC_BROWSER_TESTS=${p[web]:-1} NC_CHECKASM=${p[checkasm]:-1}
    case ${p[rust]} in
        none) ;;
        all) timed check.sh tools/ci/check.sh --quick || return 1 ;;
        *) timed check.sh tools/ci/check.sh --quick --crates "${p[rust]}" || return 1 ;;
    esac
    if [[ "${p[web]}" == 1 && "${p[rust]}" != all && " ${p[rust]} " != *" nc-web "* ]]; then  # web/ only
        timed "browser module and product smoke (web/ changed)" tools/ci/browser_smoke.sh || return 1
    fi
    # Random legal decisions against every decoder (check.sh runs it only without
    # --quick): a VP9 writer abort under it went unseen for a day (95874a490 to
    # 4fc9a1de4, 2026-11-21) because no verify ran it. The gate build is the one the
    # landed-commit cache stores.
    if [[ "${p[rust]}" != none ]]; then
        timed "syntax stress build" cargo build --profile gate --quiet -p nanocodec-cli || return 1
        NC_NANOCODEC_BIN="${CARGO_TARGET_DIR:-target}/gate" timed "syntax stress" \
            python3 tools/ci/syntax_stress.py --quick --workers 8 || return 1
    fi
}

conformance_check() {
    local -n p=$1
    [[ "${p[conformance]}" != 0 ]] || return 0
    local codecs=()
    [[ "${p[conformance]}" == all ]] || codecs=(--codecs "${p[conformance]// /,}")
    # The gate profile: release code generation, incremental, no LTO (same streams).
    printf '== landing conformance (every shipped preset vs the standard decoders and ours)\n'
    timed "conformance build" cargo build --profile gate --quiet -p nanocodec-cli || return 1
    NC_NANOCODEC_BIN="$CARGO_TARGET_DIR/gate" timed "landing conformance" \
        python3 tools/ci/landing_conformance.py "${codecs[@]}" \
            --workers "${NC_CONFORMANCE_WORKERS:-4}" || return 1
}

main "$@"
exit
