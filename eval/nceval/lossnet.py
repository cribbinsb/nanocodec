"""Network models of nceval.losssim: RTP packetization and packet loss.

Packetization: a frame travels in RTP packets of at most MAX_PAYLOAD_BYTES
of payload; each packet also carries PACKET_OVERHEAD_BYTES of headers
(RTP 12, a dependency-descriptor header extension 8, UDP 8, IPv4 20), which
count in the bitrate on the wire. A frame is whole or lost: one lost packet
loses it (owner, 2026-11-11: at our rates most inter frames are one packet,
so no partial-frame concealment is modelled).

Loss models, each a per-packet draw from its own seeded generator:

  none                 nothing is lost
  bernoulli:R          independent loss with probability R (uniform loss)
  ge:R:B               Gilbert-Elliott two-state burst loss with mean loss
                       rate R and mean burst length B packets: every packet
                       of the Bad state is lost, none of the Good state
                       (the simplified Gilbert model ITU-T G.1050 and the
                       3GPP MTSI test conditions parameterise this way,
                       by loss rate and mean burst length)
  trace:PATH           a fixed trace, one 0/1 (1 = lost) per packet,
                       whitespace-separated, repeated cyclically
  burst2/5/10          presets: ge:0.02:2, ge:0.05:3, ge:0.10:4
  uniform2/5/10        presets: bernoulli:0.02 / 0.05 / 0.10
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

#: Largest RTP payload per packet (a common real-time video MTU budget).
MAX_PAYLOAD_BYTES = 1200
#: Header bytes per packet: RTP 12 + dependency descriptor extension 8 +
#: UDP 8 + IPv4 20.
PACKET_OVERHEAD_BYTES = 12 + 8 + 8 + 20

PRESETS = {
    "burst2": "ge:0.02:2", "burst5": "ge:0.05:3", "burst10": "ge:0.10:4",
    "uniform2": "bernoulli:0.02", "uniform5": "bernoulli:0.05", "uniform10": "bernoulli:0.10",
}


def packet_count(frame_bytes: int) -> int:
    """Packets a frame of `frame_bytes` takes (at least one)."""
    return max(1, math.ceil(frame_bytes / MAX_PAYLOAD_BYTES))


def wire_bytes(frame_bytes: int) -> int:
    """Bytes a frame of `frame_bytes` puts on the wire, headers included."""
    return frame_bytes + packet_count(frame_bytes) * PACKET_OVERHEAD_BYTES


class LossModel(Protocol):
    def lost(self) -> bool:
        """Draw the next packet: True if it is lost."""


@dataclass
class NoLoss:
    def lost(self) -> bool:
        return False


class Bernoulli:
    def __init__(self, rate: float, rng: random.Random):
        if not 0 <= rate < 1:
            raise ValueError(f"loss rate {rate} outside [0, 1)")
        self.rate, self._rng = rate, rng

    def lost(self) -> bool:
        return self._rng.random() < self.rate


class GilbertElliott:
    """Two states; Bad loses every packet, Good none.

    With P(Good→Bad) = p and P(Bad→Good) = r the mean loss rate is
    p / (p + r) and the mean burst length 1 / r.
    """

    def __init__(self, rate: float, burst: float, rng: random.Random):
        if not 0 < rate < 1 or burst < 1:
            raise ValueError(f"Gilbert-Elliott needs 0 < rate < 1, burst >= 1 ({rate}, {burst})")
        self.to_good = 1.0 / burst
        self.to_bad = rate * self.to_good / (1.0 - rate)
        self._rng = rng
        # Start in the stationary distribution, so the first packets see
        # the mean rate too.
        self._bad = rng.random() < rate

    def lost(self) -> bool:
        lost = self._bad
        flip = self.to_good if self._bad else self.to_bad
        if self._rng.random() < flip:
            self._bad = not self._bad
        return lost


class Trace:
    def __init__(self, path: Path, offset: int):
        self._bits = [token == "1" for token in path.read_text().split()]
        if not self._bits:
            raise ValueError(f"{path}: empty loss trace")
        self._position = offset % len(self._bits)

    def lost(self) -> bool:
        lost = self._bits[self._position]
        self._position = (self._position + 1) % len(self._bits)
        return lost


def loss_model(spec: str, seed: int) -> LossModel:
    """The model `spec` names (module docstring), seeded with `seed`."""
    spec = PRESETS.get(spec, spec)
    kind, _, rest = spec.partition(":")
    rng = random.Random(seed)
    if kind == "none":
        return NoLoss()
    if kind == "bernoulli":
        return Bernoulli(float(rest), rng)
    if kind == "ge":
        rate, burst = rest.split(":")
        return GilbertElliott(float(rate), float(burst), rng)
    if kind == "trace":
        return Trace(Path(rest), seed)
    raise ValueError(f"unknown loss model {spec!r} (none, bernoulli:R, ge:R:B, trace:PATH, "
                     f"{', '.join(PRESETS)})")
