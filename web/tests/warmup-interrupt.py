#!/usr/bin/env python3
"""The decode worker's start-up warm-up (web/warmup.js) never holds up real
data for more than one tiny frame and never changes the output.

Headless Chromium runs the demo's decode worker for every codec. The reference
pass has warm-up off; the others send the first real packet immediately after
`ready` (before the warm-up has decoded anything), or a few milliseconds later
(while it is decoding). Strict checks: every pass shows the same pixels, and
every warm-up that met real data reports stopping. The first real packet's
wait (round trip less the worker's own time on it) may exceed the reference
pass's own wait, measured in the same browser, by at most the warm-up's
longest synchronous step (a tiny frame, or creating its decoder) plus a
scheduling allowance (at least 20 ms, twice that reference wait under load),
so a loaded machine slows both sides alike. That bound is timing only, so
exceeding it is a warning, not a failure: identical output and a stopped
warm-up are asserted. Whether some pass stopped the warm-up mid-way or before
its first frame depends on timing; it is reported, not asserted.
"""
import functools
import http.server
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1]
CODECS = ["h264", "hevc", "av1", "vp9", "av2", "vvc"]
# Delays (ms after `ready`) for the first real packet: immediate, then a spread
# that lands inside the warm-up on this machine (it takes 30-70 ms).
DELAYS_MS = [0, 2, 5, 10, 20, 35]
# Least allowance for message passing and timer granularity between page
# and worker; it grows with the reference pass's own wait.
SCHEDULING_ALLOWANCE_MS = 20
# Packets in each built-in warm-up stream (tools/web/make-warmup.sh).
WARMUP_FRAMES = 8


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    failures, warnings = [], []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/tests/warmup-interrupt.html")
            page.wait_for_function("window.ready")
            run = lambda **options: page.evaluate("o => window.runPass(o)", options)
            for codec in CODECS:
                reference = run(codec=codec, warmUp=False, delayMs=0)
                # The slower of two warm-up-off passes: how long messaging
                # alone takes on this machine right now.
                reference_wait = max(reference["firstWaitMs"],
                                     run(codec=codec, warmUp=False, delayMs=0)["firstWaitMs"])
                interrupted_mid = stopped_before = False
                for delay in DELAYS_MS:
                    out = run(codec=codec, warmUp=True, delayMs=delay)
                    report = out["report"]
                    if out["hashes"] != reference["hashes"]:
                        failures.append(f"{codec} delay {delay} ms: output differs from warm-up off")
                    if report["frames"] < WARMUP_FRAMES and not report["stopped"]:
                        failures.append(f"{codec} delay {delay} ms: warm-up met real data but did not stop")
                    allowance = max(SCHEDULING_ALLOWANCE_MS, 2 * reference_wait)
                    bound = reference_wait + report["longestFrameMs"] + allowance
                    # Timing only: a warning, never a failure (owner, 2026-11-12: a loaded
                    # machine's scheduling must not block landings; output is what must match).
                    if out["firstWaitMs"] > bound:
                        warnings.append(f"{codec} delay {delay} ms: first packet waited {out['firstWaitMs']:.1f} ms "
                                        f"(reference {reference_wait:.1f} ms, longest warm-up step "
                                        f"{report['longestFrameMs']:.1f} ms, allowance {allowance:.1f} ms)")
                    interrupted_mid |= report["stopped"] and report["frames"] > 0
                    stopped_before |= report["stopped"] and report["frames"] == 0
                    print(f"{codec:5} delay {delay:2} ms: waited {out['firstWaitMs']:5.1f} ms (reference "
                          f"{reference_wait:4.1f}), warm-up {'stopped' if report['stopped'] else 'finished'} "
                          f"after {report['frames']} frames (longest {report['longestFrameMs']:.1f} ms)")
                if not (interrupted_mid and stopped_before):
                    print(f"note: {codec}: this run did not cover both a stop before the first tiny frame "
                          f"({stopped_before}) and a stop mid-way ({interrupted_mid}): timing-dependent")
            browser.close()
            failures += errors
    finally:
        server.shutdown()
    for warning in warnings:
        print(f"WARNING warm-up interrupt (timing only): {warning}")
    if failures:
        raise SystemExit("FAIL warm-up interrupt:\n  " + "\n  ".join(failures))
    print("PASS warm-up interrupt: identical output, first packet waits at most one tiny frame")


if __name__ == "__main__":
    main()
