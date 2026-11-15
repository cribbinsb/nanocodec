#!/usr/bin/env bash
# Land an agent's branch on variance-integration: the ONLY landing route
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
#   2. the pre-land checks, in that tree and the agent's own target dir, only
#      what cannot wait (about 90 s): `cargo check --profile gate` of the
#      crates the diff touches and every crate depending on them, the unit
#      tests (`--lib --bins`) of the touched crates, in the gate profile (warm
#      in agents' targets: the harness binary; gate-test was cold everywhere,
#      minutes a first landing), byte-compile of touched Python and `bash -n`
#      of touched shell scripts;
#   3. compare-and-swap the integration ref onto the candidate
#      (`git update-ref NEW OLD`: atomic, it fails if anyone landed meanwhile).
#      If the head moved, rebase again and retry; the checks re-run only when
#      the commits that arrived touch a file the candidate changes.
# Everything else (fmt, clippy, workspace tests, checkasm, the browser module,
# the eval suite, landing conformance) runs in `land.sh --verify`. Only a
# real stream mismatch there reverts a landing; other failures are fixed
# forward.
#
# The revert guard (tools/dev/land_guard.py: a stale tree after `git reset
# --soft` put back other agents' work) no longer refuses: what it finds is
# added to the commit message as a `Land-warning:` trailer and printed.
#
# The integration worktree (variance-integration checked out) is not needed
# to land; after a landing it is moved to the new head when it was clean at an
# older commit (a stale index otherwise failed every later fast-forward).
#
# NC_LAND_MODE=full runs every check (as the verifier does) in the agent's
# own tree before landing, for a change considered risky.
#
# The whole script is one function, read before it runs: rebasing the
# agent's worktree can rewrite this file mid-run.
#
# Environment: NC_LAND_TARGET (integration branch, default
# variance-integration), NC_INTEGRATION_TREE (its worktree, default
# ~/nanocodec-data/worktrees/variance-integration), CARGO_TARGET_DIR (the
# agent's target dir, default ~/nanocodec-data/targets/<worktree name>),
# NC_LAND_CHECK_BUDGET (seconds the pre-land checks should take, default 90:
# over it they warn), CARGO_BUILD_JOBS (default 8).
set -euo pipefail

MAX_ATTEMPTS=20
DEFAULT_BUILD_JOBS=8  # the per-agent cap (CLAUDE.md "Working in parallel")
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
    target=${NC_LAND_TARGET:-variance-integration}
    integration=${NC_INTEGRATION_TREE:-$data_root/worktrees/variance-integration}
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

    branch=$(git rev-parse --abbrev-ref HEAD)
    [[ $# -eq 0 || "$1" == "$branch" ]] ||
        die "check out $1 in this worktree first (never land another tree's branch)"
    [[ -z "$(git status --porcelain --untracked-files=no)" ]] ||
        die "uncommitted changes: commit or drop them first"
    export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$data_root/targets/$(basename "$PWD")}

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
            guard_warning "$base"
            pre_land_checks "$base"
        elif touches_same_files "$checked_base" "$base"; then
            printf '== %s moved in files %s changes: checking again\n' "$target" "$branch"
            pre_land_checks "$base"
        fi
        checked_base=$base
        if git update-ref -m "land.sh: $branch" "refs/heads/$target" HEAD "$base" 2>/dev/null; then
            printf '== landed %s on %s in %ds (verify beside: land.sh --verify)\n' \
                "$(git log --oneline -1 HEAD)" "$target" $((SECONDS - began))
            sync_integration_tree
            exit 0
        fi
        printf '== %s moved (attempt %d): rebasing again\n' "$target" "$attempt"
    done
    die "$target kept moving: gave up after $MAX_ATTEMPTS attempts"
}

# Put a `cargo` on PATH that runs every cargo call through
# tools/dev/cargo-capped.sh, memory-capped. The reserved slot: a lander never
# waits behind other agents' builds for the shared slots.
cargo_shim() {
    shim_dir=$(mktemp -d)
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
guard_warning() {
    local found
    found=$(timeout "$GUARD_TIMEOUT" python3 tools/dev/land_guard.py "$1" 2>&1 >/dev/null) && return 0
    [[ -n "$found" ]] || found="land_guard.py did not finish in ${GUARD_TIMEOUT}s"
    printf '== WARNING (landing anyway; recorded in the commit):\n%s\n' "$found"
    git -c core.hooksPath=/dev/null commit --quiet --amend --no-edit \
        --trailer "Land-warning: $(grep -m1 -o 'undoes [0-9]* upstream change(s)\|did not finish.*' <<<"$found" ||
            echo 'revert guard flagged this commit') (tools/dev/land_guard.py)"
}

#: Author of the commits land.sh makes on a lander's branch (regenerated files).
SITE_AUTHOR="land.sh <land.sh@nanocodec>"
SITE_SUBJECT="site: regenerate the generated results data (python3 -m nceval.site)"
#: What eval/nceval/site.py writes.
SITE_OUTPUTS=(web/results/data web/docs/examples README.md)

# The generated site data and README block (eval/nceval/site.py), rewritten
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
    if ((${#tested[@]})); then
        timed "unit tests ${tested[*]}" cargo test --profile gate --quiet --lib --bins "${tested[@]}" ||
            die "unit tests failed: not landed"
    fi
    if [[ $mode == full ]]; then
        printf '== NC_LAND_MODE=full: every check, in this tree (target %s)\n' "$CARGO_TARGET_DIR"
        run_checks "$scope" || die "full checks failed: not landed"
    fi
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
    (cd "$tree" && CARGO_TARGET_DIR=${NC_VERIFY_STREAMS_TARGET_DIR:-$data_root/targets/verifier-streams} \
        timed "streams" conformance_check_plan "$scope") >"$log" 2>&1 || status=$?
    if ((status == 0)); then
        printf '%s\n' "$head" >"$state"
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
    # NC_WEB_TESTS_SINCE: the page tests run only where these changes reach (tools/ci/web_tests.py).
    (cd "$tree" && CARGO_TARGET_DIR=${NC_VERIFY_TARGET_DIR:-$data_root/targets/verifier} \
        NC_CHECK_PARTS=$log.parts NC_WEB_TESTS_SINCE=$base run_checks "$scope") \
        >"$log" 2>&1 || status=$?
    if ((status == 0)); then
        printf '%s\n' "$head" >"$state"
        rm -f "$list"
        printf '== tests verified: %s %s\n' "$target" "${head:0:9}"
    else
        printf '%s %s %s\n' "$head" "$base" "$log" >>"$list"
        report_failure "TESTS FAILED (fix forward; listed in $list)" "$head" "$base" "$log"
    fi
    return "$status"
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
    scratch=${NC_CHECK_PARTS:-$(mktemp -d)}
    mkdir -p "$scratch"
    python_checks plan >"$scratch/python" 2>&1 &
    pid[python]=$!
    rust_checks plan >"$scratch/rust" 2>&1 &
    pid[rust]=$!
    conformance_check plan >"$scratch/conformance" 2>&1 &
    pid[conformance]=$!
    local -a groups=(python rust conformance)
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
    [[ -z "${p[docs]}" ]] || timed docs python3 tools/dev/check_docs.py ${p[docs]} || return 1
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
        python3 tools/ci/landing_conformance.py "${codecs[@]}" || return 1
}

main "$@"
exit
