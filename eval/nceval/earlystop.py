"""Early stopping inside a screen stage: stop once a WORSE verdict is certain.

A real-use screen stage (nceval.realuse) measures every clip of the stage
before nceval.jreal.objective decides it. Most candidates are rejected
(stored screens, 2026-11-11: 149 of 222 whole-set screens WORSE), and many
of them are rejected by a margin that the first few clips already settle.
`certain_worse` is called each time a clip's curves are complete; when it
returns a reason, the stage's remaining points are cancelled and the stage
is decided WORSE with the clips it has.

Only WORSE can be certain early. BETTER needs every guard clear, and a clip
not yet measured can still trip a class guard. WORSE is certain when the
final objective must reject whatever the remaining clips measure:

  class guard   a class whose clips are all measured has a class mean
                quality or quality_low over CLASS_GUARD_PCT + the stage
                margin (the objective's hard rule, scaled by the stage sigma);
  rate guard    rate losses exceed rate wins plus every rung not yet
                measured (each could at most be a win);
  buffer guard  non-compliant test points exceed the anchor's plus every
                anchor point not yet measured;
  J_real        its lower bound exceeds the stage margin: every clip not yet
                measured counted at QUALITY_FLOOR_PCT, a gain larger than any
                stored screen's clip has shown.

The first three are exact (the final verdict is WORSE by the objective's
own rules); the fourth holds as long as no clip gains more than the floor.
Replaying every stored screen stage (tools/dev/earlystop_replay.py) found
no early stop whose final verdict was not WORSE.
"""

from __future__ import annotations

from dataclasses import dataclass

from .jreal import CLASS_GUARD_PCT, MARGIN_SIGMAS

#: The most a clip not yet measured is assumed to gain (quality and
#: quality_low, %): the stored screens' best clip was -34 % (2026-11-11).
QUALITY_FLOOR_PCT = -50.0
QUALITY_TERMS = ("quality", "quality_low")


@dataclass(frozen=True)
class StageScope:
    """Everything a stage will measure: each clip's class and ladder rungs."""
    classes: dict[str, str]  # clip -> content class
    rungs: dict[str, int]  # clip -> points per arm

    def order(self) -> list[str]:
        """Clips in the order that completes classes soonest: classes with
        the fewest clips first (one clip settles them), then by name."""
        sizes: dict[str, int] = {}
        for content_class in self.classes.values():
            sizes[content_class] = sizes.get(content_class, 0) + 1
        return sorted(self.classes, key=lambda c: (sizes[self.classes[c]], self.classes[c], c))


def certain_worse(rows: dict[str, dict], scope: StageScope, sigma: float,
                  dcpu_pct: float | None = None, k_cpu: float = 0.0) -> str | None:
    """Why the stage must end WORSE given the clips measured so far, or None.

    `rows` are nceval.realuse.compare rows (paired) of the measured clips;
    `sigma` is the stage's sigma (0 for a whole set).
    """
    if not rows:
        return None
    margin = MARGIN_SIGMAS * sigma
    pending = [clip for clip in scope.classes if clip not in rows]
    for content_class in sorted(set(scope.classes.values())):
        members = [c for c, k in scope.classes.items() if k == content_class]
        if any(c not in rows for c in members):
            continue
        for term in QUALITY_TERMS:
            values = [rows[c][term] for c in members if rows[c].get(term) is not None]
            if values and sum(values) / len(values) > CLASS_GUARD_PCT + margin:
                return (f"class {content_class} {term} {sum(values) / len(values):+.2f} > "
                        f"{CLASS_GUARD_PCT + margin:+.2f} with every clip measured")
    total = lambda key: sum(row.get(key, 0) for row in rows.values())
    open_rungs = sum(scope.rungs[c] for c in pending)
    if total("rate_losses") > total("rate_wins") + open_rungs:
        return (f"rate: {total('rate_losses')} rungs overshot > {total('rate_wins')} won "
                f"+ {open_rungs} still open")
    if total("test_noncompliant") > total("anchor_noncompliant") + open_rungs:
        return (f"buffer: {total('test_noncompliant')} non-compliant points > anchor's "
                f"{total('anchor_noncompliant')} + {open_rungs} still open")
    lower = [_balanced_lower_bound(rows, scope, term) for term in QUALITY_TERMS]
    if None not in lower:
        j_lower = sum(lower) / len(lower) + k_cpu * (dcpu_pct or 0.0)
        if j_lower - margin > 0:
            return (f"J_real >= {j_lower:+.2f} > margin {margin:.2f} whatever the "
                    f"{len(pending)} clips still open measure")
    return None


def _balanced_lower_bound(rows: dict[str, dict], scope: StageScope, term: str) -> float | None:
    """The least class-balanced mean of `term` the finished stage can have.

    A clip not yet measured counts at QUALITY_FLOOR_PCT, or not at all (its
    BD can come out unusable): whichever is lower for its class. A measured
    class without any value drops out, as in jreal.aggregate.
    """
    bounds = []
    for content_class in sorted(set(scope.classes.values())):
        members = [c for c, k in scope.classes.items() if k == content_class]
        known = [rows[c][term] for c in members if c in rows and rows[c].get(term) is not None]
        open_clips = sum(1 for c in members if c not in rows)
        candidates = [(sum(known) + r * QUALITY_FLOOR_PCT) / (len(known) + r)
                      for r in range(open_clips + 1) if len(known) + r]
        if candidates:
            bounds.append(min(candidates))
    if not bounds:
        return None
    # Above the floor, a class that drops out only raises the mean.
    return sum(bounds) / len(bounds)
