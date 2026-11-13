"""Automatic sanity report: catches the bug classes relative checks miss.

Comparing a change only against our own previous default, averaging away
outliers and never looking at codec ordering, tool activity or per-frame
trends let several large bugs survive for weeks (docs/PROCEDURES.md "Sanity
report"). This package checks absolute expectations instead:

  ordering  bits(AV1) <= bits(HEVC) <= bits(H.264) at matched quality, per clip
            and rate region (ordering.py)
  outliers  per-clip BD-rate against the reference encoders (outliers.py)
  tools     coding-tool activity of our streams against the reference's
            (activity.py, from av1_tools.py / avc_tools.py)
  decay     sustained per-frame quality decline over P frames (decay.py)
  harness   anchor rate targets, degenerate curves, anchor command lines and
            clip source caps (harness.py)

Run: python3 -m nceval sanity [--config FILE] [--only measure|checks]
"""
