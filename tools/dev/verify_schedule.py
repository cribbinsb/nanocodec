#!/usr/bin/env python3
"""The scheduled verifier: runs every correctness check on main's head whenever it moved.

    python3 tools/dev/verify_schedule.py tick      the timer's entry point (hourly)
    python3 tools/dev/verify_schedule.py status    one line: the last result of every check
    python3 tools/dev/verify_schedule.py hourly    the hourly part now (if main moved)
    python3 tools/dev/verify_schedule.py nightly [--only x86|arm]
                                                   the nightly part now (whatever the hour;
                                                   --only: one half, e.g. after an error)
    python3 tools/dev/verify_schedule.py install   (re)write and start the systemd user timer

Why: on 2026-11-27 the checks looked fine but did not run (docs/analysis/
hidden-gaps-audit-2026-11-27.md 3.1-3.9): nothing scheduled `land.sh --verify`, and the
official suites, the Arm kernels and the #[ignore]d tests ran only by hand. This is the
one scheduler. It only orchestrates; every heavy job runs on AWS.

  hourly   `land.sh --verify` (streams, then tests) on the x86 build box, when main moved
           since the last hourly run. land.sh keeps its own records (verified heads,
           run/verify-failed, run/fix-forward) as before.
  nightly  once a UTC day from NIGHTLY_HOUR_UTC, when main moved since the last nightly
           run, as its own systemd unit (so the hourly part never waits on it), two
           halves side by side on one checkout of the head:
             x86  `tools/ci/nightly_checks.sh x86` on the build box (nc-remote);
             arm  `tools/dev/verify_arm.py` (a Graviton4 spot box).

Records: run/verify-status.json holds each check's last result (pass, fail, or error: it
could not run), the commit, the time, the log, and the last commit it passed at. A
nightly check that fails appends `HEAD BASE LOG` to run/fix-forward (land.sh's
convention; BASE is the last commit it passed at, or `-`); its lines go when it passes
again. Logs: run/land-logs/scheduled-hourly-<head>.log and run/land-logs/nightly-<head>/.

Cost (aws_choose.py, 2026-11-27): the Arm box, c8g.8xlarge spot in eu-north-1 at
$0.19/h for ~0.5-0.75 h, ~$0.15; the x86 half, about an hour of one build-box slot
(c7i.24xlarge spot $0.63/h for the whole box, shared with every agent's jobs; ~$0.65
on a night the box would otherwise be idle). About $0.80 a night; an hourly verify
costs build-box time only when main moved, i.e. while agents are landing and the box is
up anyway.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
RUN = DATA / "run"
LOGS = RUN / "land-logs"
STATUS = RUN / "verify-status.json"
FIX_FORWARD = RUN / "fix-forward"
NC_REMOTE = DATA / "bin/nc-remote"
NIGHTLY_TREE = DATA / "worktrees/verifier-nightly"
TARGET = "main"
#: The nightly part starts with the first tick at or after this UTC hour (the PC's quiet
#: hours; the next working day sees its results).
NIGHTLY_HOUR_UTC = 1
UNIT = "nanocodec-verify"
NIGHTLY_UNIT = "nanocodec-verify-nightly"
#: The clips ISA identity codes on x86 (preset_identity.py CLIPS) and the data the x86
#: checks read beside the build box's anchors and other corpora. The H.264 corpus is
#: synced (4 GB once, then only changes): the box's copy was empty on 2026-11-27, which
#: the WASM suite caught.
IDENTITY_CLIPS = ("crowd_mot17_09", "static_otw_homes390", "pan_bdd_00c12bd0")
X86_SYNC = ("work/av2-foreign", "fuzz/round3-2026-11-23", "fuzz/regressions",
            "conformance/h264/streams", "conformance/h264/supplementary")
#: land.sh --verify's line when a part is already running elsewhere (its heads are
#: covered by that run, so nothing is recorded here).
ALREADY_RUNNING = "verifier is already running"
#: verify_arm.py's exit status when the run could not complete.
ARM_INCOMPLETE = 2


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def main_head() -> str:
    return git("rev-parse", f"refs/heads/{TARGET}")


class Records:
    """run/verify-status.json and run/fix-forward, read and written under one lock."""

    def __init__(self) -> None:
        self.lock = open(RUN / "verify-status.lock", "a")  # noqa: SIM115 (held for life)

    def __enter__(self) -> dict:
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        try:
            self.status = json.loads(STATUS.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.status = {}
        self.status.setdefault("checks", {})
        self.status.setdefault("parts", {})
        return self.status

    def __exit__(self, *exc) -> None:
        temporary = STATUS.with_suffix(".partial")
        temporary.write_text(json.dumps(self.status, indent=1, sort_keys=True) + "\n")
        temporary.replace(STATUS)
        fcntl.flock(self.lock, fcntl.LOCK_UN)


def record(status: dict, check: str, result: str, commit: str, log: Path,
           fix_forward: bool) -> None:
    """One check's result. A nightly (`fix_forward`) failure is listed in run/fix-forward,
    and its earlier lines go once it passes."""
    entry = status["checks"].setdefault(check, {})
    base = entry.get("passed_at", "-")
    entry.update(result=result, commit=commit, at=utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                 log=str(log))
    if result == "pass":
        entry["passed_at"] = commit
    if not fix_forward:
        return
    lines = FIX_FORWARD.read_text().splitlines() if FIX_FORWARD.is_file() else []
    mine = f"/{check.split('/', 1)[1]}.log"
    kept = [line for line in lines if not ("/nightly-" in line and line.endswith(mine))]
    if result != "pass":
        kept.append(f"{commit} {base} {log}")
    if kept != lines:
        FIX_FORWARD.write_text("".join(f"{line}\n" for line in kept))


def hourly() -> int:
    """`land.sh --verify` on main's head, when it moved since the last hourly run."""
    head = main_head()
    with Records() as status:
        if status["parts"].get("hourly", {}).get("commit") == head:
            print(f"hourly: {head[:9]} already verified")
            return 0
    log = LOGS / f"scheduled-hourly-{head[:9]}.log"
    print(f"hourly: land.sh --verify at {head[:9]} (log {log})", flush=True)
    with log.open("w") as out:
        code = subprocess.run(["bash", "tools/dev/land.sh", "--verify"], cwd=REPO, stdout=out,
                              stderr=subprocess.STDOUT).returncode
    text = log.read_text()
    complete = True
    with Records() as status:
        for part, state, part_log in (
                ("streams", "verified-streams-head", f"verify-streams-{head[:9]}.log"),
                ("tests", "verified-head", f"verify-{head[:9]}.log")):
            if f"a {part} {ALREADY_RUNNING}" in text:
                complete = False
                continue
            verified = (RUN / state).read_text().strip() if (RUN / state).is_file() else ""
            result = "pass" if verified == head else "fail"
            record(status, f"hourly/{part}", result, head, LOGS / part_log, fix_forward=False)
        if complete:
            status["parts"]["hourly"] = {"commit": head, "exit": code}
    return code


def nightly_due(head: str) -> bool:
    """Whether the nightly part should start now (and marks a day without a change)."""
    now = utc_now()
    today = now.date().isoformat()
    with Records() as status:
        last = status["parts"].get("nightly", {})
        if now.hour < NIGHTLY_HOUR_UTC or last.get("date") == today:
            return False
        if last.get("commit") == head:
            last["date"] = today
            print(f"nightly: main has not moved since {head[:9]}")
            return False
    return True


def start_nightly() -> None:
    """The nightly part as its own transient unit (skipped while one runs)."""
    if subprocess.run(["systemctl", "--user", "is-active", "--quiet", NIGHTLY_UNIT]).returncode == 0:
        print("nightly: already running")
        return
    subprocess.run(["systemd-run", "--user", "--unit", NIGHTLY_UNIT, "--collect",
                    f"--working-directory={REPO}", f"--setenv=PATH={os.environ['PATH']}",
                    sys.executable, str(Path(__file__).resolve()), "nightly"], check=True)
    print(f"nightly: started as {NIGHTLY_UNIT}")


def checkout(tree: Path, commit: str) -> None:
    """The nightly worktree (created on first use) at `commit`."""
    if not (tree / ".git").exists():
        git("worktree", "add", "--quiet", "--detach", str(tree), commit)
    subprocess.run(["git", "-C", str(tree), "checkout", "--quiet", "--force", "--detach", commit],
                   check=True)


def run_x86(tree: Path, out: Path) -> int:
    clips = [x for c in IDENTITY_CLIPS
             for x in ("--clip", str(DATA / f"clips/main/{c}.yuv"))]
    syncs = [x for s in X86_SYNC for x in ("--sync", str(DATA / s))]
    with out.with_name(f"{out.name}-run.log").open("w") as log:
        return subprocess.run([str(NC_REMOTE), "run", "--worktree", str(tree),
                               "--name", "verifier-nightly", *syncs, *clips,
                               "--fetch", "remote-out/nightly", "--fetch-to", str(out / "fetch"),
                               "--", "rm -rf remote-out/nightly; bash tools/ci/nightly_checks.sh x86"],
                              stdout=log, stderr=subprocess.STDOUT).returncode


def run_arm(tree: Path, out: Path) -> int:
    with out.with_name(f"{out.name}-run.log").open("w") as log:
        return subprocess.run([sys.executable, str(tree / "tools/dev/verify_arm.py"), str(tree),
                               str(out / "fetch/remote-out/nightly")],
                              stdout=log, stderr=subprocess.STDOUT).returncode


def nightly(only: str | None = None) -> int:
    """Both nightly halves (or `only` one) on one checkout of main's head; every check
    recorded."""
    head = main_head()
    with Records() as status:
        status["parts"]["nightly"] = {"commit": head, "date": utc_now().date().isoformat(),
                                      "state": "running"}
    checkout(NIGHTLY_TREE, head)
    root = LOGS / f"nightly-{head[:9]}"
    halves = {name: run for name, run in (("x86", run_x86), ("arm", run_arm))
              if only in (None, name)}
    codes: dict[str, int] = {}
    threads = []
    for name, run in halves.items():
        (root / name).mkdir(parents=True, exist_ok=True)
        thread = threading.Thread(
            target=lambda n=name, r=run: codes.__setitem__(n, r(NIGHTLY_TREE, root / n)))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    failed = 0
    with Records() as status:
        for name in halves:
            results = root / name / "fetch/remote-out/nightly"
            summary = results / "summary"
            lines = summary.read_text().split("\n") if summary.is_file() else []
            if not any(lines):
                record(status, f"nightly/{name}-run", "error", head, root / f"{name}-run.log",
                       fix_forward=True)
                failed = 1
                continue
            record(status, f"nightly/{name}-run", "pass", head, root / f"{name}-run.log",
                   fix_forward=True)
            for line in filter(None, lines):
                check, result, _seconds = line.split()
                ok = result == "PASS"
                failed |= not ok
                record(status, f"nightly/{check}", "pass" if ok else "fail", head,
                       results / f"{check}.log", fix_forward=True)
        status["parts"]["nightly"]["state"] = f"done {codes}"
    print(summary_line())
    return failed


def summary_line() -> str:
    """The one-line status: main's head, the hourly parts, the nightly checks."""
    try:
        status = json.loads(STATUS.read_text())
    except FileNotFoundError:
        return "verify: no record yet (verify_schedule.py tick has not run)"
    head = main_head()
    checks = status.get("checks", {})
    hourly = " ".join(f"{part} {c['result'].upper()} {c['commit'][:9]}"
                      for part, c in sorted(checks.items()) if part.startswith("hourly/"))
    nightly = {k.split("/", 1)[1]: c for k, c in checks.items() if k.startswith("nightly/")}
    bad = sorted(k for k, c in nightly.items() if c["result"] != "pass")
    part = status.get("parts", {}).get("nightly", {})
    return (f"verify {TARGET} {head[:9]} | hourly: {hourly or 'none yet'} | nightly "
            f"{part.get('date', '-')} {part.get('commit', '-')[:9]} ({part.get('state', '-')}): "
            f"{len(nightly) - len(bad)} pass" + (f", {len(bad)} not: {' '.join(bad)}" if bad else ""))


def tick() -> int:
    """The timer: the nightly part when due (its own unit), then the hourly part."""
    head = main_head()
    if nightly_due(head):
        start_nightly()
    return hourly()


def install() -> None:
    """The systemd user timer (hourly, persistent) running `tick` from this checkout."""
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    path = ":".join([str(Path.home() / ".cargo/bin"), str(Path(sys.executable).parent),
                     "/usr/local/bin", "/usr/bin", "/bin"])
    (units / f"{UNIT}.service").write_text(f"""[Unit]
Description=nanocodec scheduled verifier (tools/dev/verify_schedule.py)

[Service]
Type=oneshot
WorkingDirectory={REPO}
Environment=PATH={path}
ExecStart={sys.executable} {REPO}/tools/dev/verify_schedule.py tick
""")
    (units / f"{UNIT}.timer").write_text(f"""[Unit]
Description=nanocodec scheduled verifier, hourly

[Timer]
OnCalendar=hourly
RandomizedDelaySec=120
Persistent=true

[Install]
WantedBy=timers.target
""")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", f"{UNIT}.timer"], check=True)
    print(f"installed {units / UNIT}.timer")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("tick", "status", "hourly", "nightly", "install"))
    ap.add_argument("--only", choices=("x86", "arm"), help="nightly: this half only")
    args = ap.parse_args()
    command = args.command
    if command == "status":
        print(summary_line())
        return 0
    if command == "install":
        install()
        return 0
    if command == "nightly":
        return nightly(args.only)
    return {"tick": tick, "hourly": hourly}[command]()


if __name__ == "__main__":
    sys.exit(main())
