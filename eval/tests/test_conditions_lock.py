"""The benchmark's conditions for our encoder are locked (owner, 2026-11-25).

Dev screens and the benchmark both run ours with `nceval.fairness.own_conditions("cbr")`.
Changing them invalidates every cached baseline point and every result measured under the
old ones: on 2026-11-25 a same-day flip and flip-back of the buffer bound cost an agent an hour
of computing on conditions that no longer applied. A change to these conditions must update
LOCKED here, in the same commit, with the owner's decision (docs/benchmark-procedure.md
"Settled decisions") in the comment. Never update it for any other reason.
"""

from nceval import fairness

#: D4 (strict buffer, final); no start-up lookahead; D21 (owner, 2026-11-25): hold on with
#: all-skip output, so every frame is still coded; no encoder-info SEI.
LOCKED = ["--buffer-ms", "1000", "--peak-bitrate", "{KBPS}", "--strict-buffer",
          "--param", "rate.first_key_lookahead_off=1", "--param", "hold.enable=1",
          "--param", "hold.code_skip=1",
          "--param", "stream.encoder_info=0"]


def test_our_benchmark_conditions_are_the_locked_ones():
    assert fairness.own_conditions("cbr") == LOCKED, (
        "our benchmark conditions changed: only with an owner decision recorded in the "
        "procedure's Settled decisions, and LOCKED updated in the same commit")
