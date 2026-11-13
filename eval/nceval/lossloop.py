"""The closed loop of nceval.losssim: encoder, lossy network, receiver, feedback.

    session (Rust: `nanocodec session`)        this module (Python)
    ──────────────────────────────────        ─────────────────────────────
    encoder ── frame ──────────────────────►  packetize, lose packets (forward
                                              model), retransmit (RTX, optional)
    receiver (our decoder) ◄── receive N A ─  jitter buffer: frames in order,
                                              when whole or given up
    display pipe ── one picture per tick ──►  GPU metrics (nceval.metrics)
    encoder ◄── ack N / loss N ────────────  feedback channel: one-way delay,
                                              its own loss model

Time runs in ticks of one input frame. A frame sent at tick s reaches the
receiver at s (a constant forward delay is left out of display times), and
feedback the receiver sends then reaches the encoder one round trip later
(2 x the one-way delay). The receiver acknowledges every frame it decodes,
reports every frame it loses, and repeats its latest loss report each
round trip while its picture stays frozen (the report itself may be lost),
as RTC receivers repeat picture-loss indications. A frame the decoder
refuses for a missing reference is reported too when no loss is pending
(a loss only the stream revealed).

Retransmission (NACK + RTX, as WebRTC): the receiver asks for a frame's
missing packets; each request costs a round trip and both it and the
resent packets can be lost again; after `rtx_attempts` rounds the frame is
given up (and reported lost). Frames behind it wait in the jitter buffer.

Every frame the receiver shows as decoded is hash-checked against the
encoder's reconstruction by the session ("good" or "mismatch").
"""

from __future__ import annotations

import math
import statistics
import subprocess
from collections import deque
from dataclasses import dataclass, field

from .framepipe import FrameSink, PipeOutput, ProcessFailed, ProcessSet
from .lossnet import LossModel, packet_count, wire_bytes
from .yuv import YuvFormat

SESSION_TIMEOUT_SECONDS = 1800
#: Longest reference distance of the temporal structures (L1T3 base frames
#: predict 4 back): an inter frame whose every reference is further back
#: predicts from the long-term reference, a recovery frame.
MAX_PATTERN_DISTANCE = 4


@dataclass(frozen=True)
class Feedback:
    """The feedback channel and receiver reporting."""
    one_way_ms: float
    #: Loss model of feedback packets (each ACK / loss report / NACK is one).
    loss: LossModel
    #: NACK + RTX rounds per frame (0: no retransmission).
    rtx_attempts: int = 0


@dataclass(frozen=True)
class SpatialTuInfo:
    """Per-TU spatial layer sizes and receiver operating-point bounds from `nanocodec session`."""
    op_max_spatial: int
    op_max_temporal: int
    s0_bytes: int
    s1_bytes: int
    s2_bytes: int = 0
    num_layers: int = 2
    s1_repair: bool = False
    s2_repair: bool = False


def _parse_spatial_extra(extra: list[str]) -> SpatialTuInfo | None:
    if len(extra) < 2:
        return None
    op_part = next((t[3:] for t in extra if t.startswith("op=")), None)
    lay_part = next((t[7:] for t in extra if t.startswith("layers=")), None)
    if op_part is None or lay_part is None:
        return None
    max_s, max_t = (int(v) for v in op_part.split(":"))
    parts = [int(v) for v in lay_part.split(":")]
    s0_b = parts[0]
    s1_b = parts[1] if len(parts) > 1 else 0
    s2_b = parts[2] if len(parts) > 2 else 0
    s1_repair = "repair=s1" in extra
    s2_repair = "repair=s2" in extra
    return SpatialTuInfo(
        max_s, max_t, s0_b, s1_b, s2_bytes=s2_b, num_layers=len(parts),
        s1_repair=s1_repair, s2_repair=s2_repair,
    )


@dataclass
class SentFrame:
    tick: int
    index: int
    kind: str  # key | inter
    layer: int
    bytes: int
    references: tuple[int, ...]
    packets: int
    lost_packets: int = 0
    #: Tick at which the receiver has it whole (or gives it up).
    ready: int = 0
    arrived: bool = True
    status: str = ""
    rtx_rounds: int = 0
    #: Spatial SVC per-layer wire accounting (0 on single-layer streams).
    s0_bytes: int = 0
    enh_bytes: int = 0
    enh_packets: int = 0
    enh_lost_packets: int = 0
    enh_packets_dropped: int = 0
    enh_arrived: bool = True
    enh_repair: bool = False
    s2_bytes: int = 0
    s2_packets: int = 0
    s2_lost_packets: int = 0
    s2_arrived: bool = True
    s2_repair: bool = False
    num_spatial_layers: int = 2
    is_spatial: bool = False
    dropped_by_op: bool = False

    @property
    def recovery(self) -> bool:
        """An inter frame predicting only from the long-term reference."""
        return self.kind == "inter" and bool(self.references) and all(
            self.index - r > MAX_PATTERN_DISTANCE for r in self.references)


class Session:
    """`nanocodec session` with its display written to `sinks`."""

    def __init__(self, command: list[str], fmt: YuvFormat, sinks: list[FrameSink]):
        self._output = PipeOutput(fmt, sinks)
        self._processes = ProcessSet()
        try:
            self._process = self._processes.spawn(
                [*command, "--display", self._output.path], "session",
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, pass_fds=self._output.pass_fds)
        except BaseException:
            self._output.close_unstarted()
            raise
        self._output.start()
        self.frames_encoded = 0
        self.last_spatial_info: SpatialTuInfo | None = None
        #: (frames encoded before it, lost frame) of every loss report the
        #: encoder received (nceval.losssim checks).
        self.reports_applied: list[tuple[int, int]] = []

    def _send(self, line: str) -> None:
        self._process.stdin.write(f"{line}\n".encode())

    def _ask(self, line: str) -> list[str]:
        self._send(line)
        self._process.stdin.flush()
        reply = self._process.stdout.readline().decode().split()
        if not reply:
            self._process.wait(10)
            raise ProcessFailed(f"session ended: {self._processes.stderr_text(self._process)[-400:]}")
        return reply

    def acknowledge(self, index: int, spatial_id: int | None = None) -> None:
        if spatial_id is None:
            self._send(f"ack {index}")
        else:
            self._send(f"ack {index} {spatial_id}")

    def report_loss(self, index: int, spatial_id: int | None = None) -> None:
        self.reports_applied.append((self.frames_encoded, index))
        if spatial_id is None:
            self._send(f"loss {index}")
        else:
            self._send(f"loss {index} {spatial_id}")

    def encode(self) -> tuple[int, str, int, int, tuple[int, ...]]:
        """(index, kind, layer, bytes, references) of the next frame."""
        _, index, kind, layer, size, refs, *extra = self._ask("encode")
        self.frames_encoded += 1
        self.last_spatial_info = _parse_spatial_extra(extra)
        references = () if refs == "-" else tuple(int(r) for r in refs.split(","))
        return int(index), kind, int(layer), int(size), references

    def receive(self, index: int, arrived: bool) -> str:
        return self._ask(f"receive {index} {int(arrived)}")[2]

    def receive_spatial(self, index: int, s0_arrived: bool, s1_arrived: bool,
                        s2_arrived: bool | None = None) -> str:
        if s2_arrived is None:
            return self._ask(f"receive {index} {int(s0_arrived)} {int(s1_arrived)}")[2]
        return self._ask(f"receive {index} {int(s0_arrived)} {int(s1_arrived)} {int(s2_arrived)}")[2]

    def show(self) -> None:
        self._send("show")

    def close(self) -> int:
        """End the session; the number of pictures it showed."""
        self._process.stdin.close()
        try:
            self._process.wait(SESSION_TIMEOUT_SECONDS)
        finally:
            if self._process.poll() is None:
                self._process.kill()
            self._output.join(SESSION_TIMEOUT_SECONDS)
        if self._process.returncode:
            raise ProcessFailed(f"session failed ({self._process.returncode}): "
                                f"{self._processes.stderr_text(self._process)[-400:]}")
        return self._output.frames

    def kill(self) -> None:
        self._processes.kill_all()


@dataclass
class Trace:
    """Everything one run did, tick by tick."""
    frames: list[SentFrame] = field(default_factory=list)
    #: Per tick: "good" (the newest frame sent is on screen), "frozen",
    #: "corrupt" (a mismatching picture is on screen) or "blank" (nothing
    #: decoded yet).
    screen: list[str] = field(default_factory=list)
    #: Per tick on spatial operating points with `max_spatial_id >= 1`: "good"
    #: when `S1` is up to date, "frozen" while only base plays.
    enhancement_screen: list[str] = field(default_factory=list)
    #: Per tick on 3-layer operating points with `max_spatial_id == 2`: "good"
    #: when `S2` is up to date, "frozen" when `S2` (or a lower layer) freezes.
    top_enhancement_screen: list[str] = field(default_factory=list)
    uncoded: int = 0  # input frames the encoder held or dropped
    acks_sent: int = 0
    reports_sent: int = 0
    feedback_lost: int = 0
    #: Loss reports the encoder received: (input frames before it, lost frame).
    reports_applied: list[tuple[int, int]] = field(default_factory=list)


#: Session statuses of a frame the receiver refused (`FreezeReason` names).
FROZEN_STATUSES = ("missing-reference", "missing-key", "corrupt")


def run_loop(session: Session, ticks: int, fps: float, forward: LossModel,
             feedback: Feedback, enhancement_forward: LossModel | None = None,
             top_enhancement_forward: LossModel | None = None) -> Trace:
    """Drive `session` for `ticks` input frames through the network models."""
    enh_net = enhancement_forward if enhancement_forward is not None else forward
    top_net = top_enhancement_forward if top_enhancement_forward is not None else enh_net
    tick_ms = 1000.0 / fps
    rtt_ticks = max(1, math.ceil(2 * feedback.one_way_ms / tick_ms)) if feedback.one_way_ms else 0
    report_every = rtt_ticks + 1
    trace = Trace()
    to_encoder: list[tuple[int, str, int, int | None]] = []  # (arrival tick, kind, index, spatial_id)
    buffer: deque[SentFrame] = deque()
    latest_sent = -1
    latest_good = -1
    latest_enh_sent = -1
    latest_enh_good = -1
    latest_top_sent = -1
    latest_top_good = -1
    track_enh = False
    track_top = False
    corrupt = False
    lost_unrepaired: int | None = None  # newest frame reported lost, not yet repaired
    last_report = -1
    enh_lost_unrepaired: int | None = None  # newest S1 frame reported lost, not yet repaired
    last_enh_report = -1
    top_lost_unrepaired: int | None = None  # newest S2 frame reported lost, not yet repaired
    last_top_report = -1

    def send_feedback(tick: int, kind: str, index: int, spatial_id: int | None = None) -> None:
        if kind == "ack":
            trace.acks_sent += 1
        else:
            trace.reports_sent += 1
        if feedback.loss.lost():
            trace.feedback_lost += 1
            return
        to_encoder.append((tick + rtt_ticks, kind, index, spatial_id))

    for tick in range(ticks):
        arrived_now = [f for f in to_encoder if f[0] <= tick]
        to_encoder = [f for f in to_encoder if f[0] > tick]
        for _, kind, index, spatial_id in sorted(
            arrived_now, key=lambda x: (x[0], x[1], x[2], -1 if x[3] is None else x[3])
        ):
            fn = session.acknowledge if kind == "ack" else session.report_loss
            if spatial_id is None:
                fn(index)
            else:
                fn(index, spatial_id)
        index, kind, layer, size, references = session.encode()
        spatial: SpatialTuInfo | None = getattr(session, "last_spatial_info", None)
        if kind in ("held", "dropped"):
            trace.uncoded += 1
        elif spatial is not None:
            frame = _build_spatial_frame(tick, index, kind, layer, references, spatial,
                                         forward, enh_net, top_net, feedback, rtt_ticks)
            trace.frames.append(frame)
            buffer.append(frame)
            if not frame.dropped_by_op:
                latest_sent = index
                if spatial.op_max_spatial >= 1:
                    track_enh = True
                    if spatial.s1_bytes > 0:
                        latest_enh_sent = index
                if spatial.op_max_spatial >= 2:
                    track_top = True
                    if spatial.s2_bytes > 0:
                        latest_top_sent = index
        else:
            frame = SentFrame(tick, index, kind, layer, size, references, packet_count(size))
            transmit(frame, forward, feedback, rtt_ticks)
            trace.frames.append(frame)
            buffer.append(frame)
            latest_sent = index
        while buffer and buffer[0].ready <= tick:
            frame = buffer.popleft()
            if frame.is_spatial:
                if frame.num_spatial_layers >= 3:
                    frame.status = session.receive_spatial(
                        frame.index, frame.arrived, frame.enh_arrived, frame.s2_arrived)
                else:
                    frame.status = session.receive_spatial(
                        frame.index, frame.arrived, frame.enh_arrived)
            else:
                frame.status = session.receive(frame.index, frame.arrived)
            if frame.status == "skipped-op":
                continue
            if frame.status == "good":
                latest_good, corrupt = frame.index, False
                if frame.is_spatial and frame.enh_bytes > 0:
                    latest_enh_good = frame.index
                if frame.is_spatial and frame.s2_bytes > 0:
                    latest_top_good = frame.index
                send_feedback(tick, "ack", frame.index)
            elif frame.status in ("s1-only", "s2-frozen"):
                # S0 and S1 both decoded bit-exactly and keep playing; only S2 freezes.
                latest_good, corrupt = frame.index, False
                if frame.enh_bytes > 0:
                    latest_enh_good = frame.index
                send_feedback(tick, "ack", frame.index, 0)
                send_feedback(tick, "ack", frame.index, 1)
                if frame.status == "s1-only" and (
                    top_lost_unrepaired is None or frame.s2_repair or frame.enh_repair
                ):
                    top_lost_unrepaired = frame.index
                    last_top_report = tick
                    send_feedback(tick, "loss", frame.index, 2)
            elif frame.status in ("base-only", "enh-frozen"):
                # Base decoded bit-exactly and keeps playing; S1 (and S2 if 3-layer) freezes.
                latest_good, corrupt = frame.index, False
                send_feedback(tick, "ack", frame.index, 0)
                if frame.status == "base-only" and (enh_lost_unrepaired is None or frame.enh_repair):
                    enh_lost_unrepaired = frame.index
                    last_enh_report = tick
                    send_feedback(tick, "loss", frame.index, 1)
            elif frame.status == "mismatch":
                corrupt = True
            elif frame.status == "lost" or (frame.status in FROZEN_STATUSES
                                             and lost_unrepaired is None):
                # A refused frame the transport did not report (the decoder
                # found the gap in the stream itself) is reported the same.
                lost_unrepaired = frame.index
                last_report = tick
                send_feedback(tick, "loss", frame.index)
        frozen = latest_good < latest_sent
        if not frozen:
            lost_unrepaired = None
        elif lost_unrepaired is not None and tick - last_report >= report_every:
            last_report = tick
            send_feedback(tick, "loss", lost_unrepaired)
        session.show()
        trace.screen.append("blank" if latest_good < 0 and not corrupt else
                            "corrupt" if corrupt else "frozen" if frozen else "good")
        enh_frozen = False
        if track_enh:
            enh_frozen = latest_enh_good < latest_enh_sent or frozen
            if not enh_frozen:
                enh_lost_unrepaired = None
            elif not frozen and enh_lost_unrepaired is not None and tick - last_enh_report >= report_every:
                last_enh_report = tick
                send_feedback(tick, "loss", enh_lost_unrepaired, 1)
            trace.enhancement_screen.append(
                "blank" if latest_enh_good < 0 and not corrupt else
                "corrupt" if corrupt else "frozen" if enh_frozen else "good")
        if track_top:
            top_frozen = latest_top_good < latest_top_sent or enh_frozen or frozen
            if not top_frozen:
                top_lost_unrepaired = None
            elif (not frozen and not enh_frozen and top_lost_unrepaired is not None
                  and tick - last_top_report >= report_every):
                last_top_report = tick
                send_feedback(tick, "loss", top_lost_unrepaired, 2)
            trace.top_enhancement_screen.append(
                "blank" if latest_top_good < 0 and not corrupt else
                "corrupt" if corrupt else "frozen" if top_frozen else "good")
    trace.reports_applied = list(session.reports_applied)
    return trace


def _build_spatial_frame(tick: int, index: int, kind: str, layer: int,
                         references: tuple[int, ...], spatial: SpatialTuInfo,
                         forward: LossModel, enh_forward: LossModel,
                         top_forward: LossModel,
                         feedback: Feedback, rtt_ticks: int) -> SentFrame:
    """Packetize and transmit one spatial TU according to the subscribed operating point."""
    s1_pkts = packet_count(spatial.s1_bytes) if spatial.s1_bytes > 0 else 0
    s2_pkts = packet_count(spatial.s2_bytes) if spatial.s2_bytes > 0 else 0
    if layer > spatial.op_max_temporal:
        return SentFrame(
            tick, index, kind, layer, 0, references, 0, ready=tick, arrived=False,
            enh_arrived=False, s2_arrived=False,
            enh_packets_dropped=packet_count(spatial.s0_bytes) + s1_pkts + s2_pkts,
            enh_repair=spatial.s1_repair, s2_repair=spatial.s2_repair,
            num_spatial_layers=spatial.num_layers, is_spatial=True, dropped_by_op=True,
        )
    if spatial.op_max_spatial == 0:
        # Subscribed to base operating point: drop S1 and S2 packets before the link.
        frame = SentFrame(
            tick, index, kind, layer, spatial.s0_bytes, references,
            packet_count(spatial.s0_bytes), s0_bytes=spatial.s0_bytes,
            enh_packets_dropped=s1_pkts + s2_pkts, enh_arrived=False, s2_arrived=False,
            enh_repair=spatial.s1_repair, s2_repair=spatial.s2_repair,
            num_spatial_layers=spatial.num_layers, is_spatial=True,
        )
        transmit(frame, forward, feedback, rtt_ticks)
        return frame
    if spatial.op_max_spatial == 1:
        # Subscribed to S1 operating point (2-layer full or 3-layer middle): drop S2 packets.
        frame = SentFrame(
            tick, index, kind, layer, spatial.s0_bytes + spatial.s1_bytes, references,
            packet_count(spatial.s0_bytes), s0_bytes=spatial.s0_bytes,
            enh_bytes=spatial.s1_bytes, enh_packets=s1_pkts,
            enh_packets_dropped=s2_pkts, s2_arrived=False,
            enh_repair=spatial.s1_repair, s2_repair=spatial.s2_repair,
            num_spatial_layers=spatial.num_layers, is_spatial=True,
        )
        transmit(frame, forward, feedback, rtt_ticks)
        if s1_pkts > 0:
            _transmit_enhancement(frame, enh_forward, feedback, rtt_ticks)
        return frame
    # Subscribed to top S2 operating point (3-layer full): transmit S0, S1, and S2.
    frame = SentFrame(
        tick, index, kind, layer,
        spatial.s0_bytes + spatial.s1_bytes + spatial.s2_bytes, references,
        packet_count(spatial.s0_bytes), s0_bytes=spatial.s0_bytes,
        enh_bytes=spatial.s1_bytes, enh_packets=s1_pkts,
        s2_bytes=spatial.s2_bytes, s2_packets=s2_pkts,
        enh_repair=spatial.s1_repair, s2_repair=spatial.s2_repair,
        num_spatial_layers=spatial.num_layers, is_spatial=True,
    )
    transmit(frame, forward, feedback, rtt_ticks)
    if s1_pkts > 0:
        _transmit_enhancement(frame, enh_forward, feedback, rtt_ticks)
    if s2_pkts > 0:
        _transmit_top_enhancement(frame, top_forward, feedback, rtt_ticks)
    return frame


def _transmit_top_enhancement(frame: SentFrame, forward: LossModel, feedback: Feedback,
                              rtt_ticks: int) -> None:
    missing = sum(forward.lost() for _ in range(frame.s2_packets))
    frame.s2_lost_packets = missing
    rounds = 0
    while missing and rounds < feedback.rtx_attempts:
        rounds += 1
        if feedback.loss.lost():
            continue
        missing = sum(forward.lost() for _ in range(missing))
    frame.rtx_rounds = max(frame.rtx_rounds, rounds)
    frame.ready = max(frame.ready, frame.tick + rounds * rtt_ticks)
    frame.s2_arrived = missing == 0


def _transmit_enhancement(frame: SentFrame, forward: LossModel, feedback: Feedback,
                          rtt_ticks: int) -> None:
    missing = sum(forward.lost() for _ in range(frame.enh_packets))
    frame.enh_lost_packets = missing
    rounds = 0
    while missing and rounds < feedback.rtx_attempts:
        rounds += 1
        if feedback.loss.lost():
            continue
        missing = sum(forward.lost() for _ in range(missing))
    frame.rtx_rounds = max(frame.rtx_rounds, rounds)
    frame.ready = max(frame.ready, frame.tick + rounds * rtt_ticks)
    frame.enh_arrived = missing == 0


def transmit(frame: SentFrame, forward: LossModel, feedback: Feedback, rtt_ticks: int) -> None:
    """Draw the frame's packets and, with RTX, its retransmissions: set when
    it is whole at the receiver (`ready`), or that it is given up."""
    missing = sum(forward.lost() for _ in range(frame.packets))
    frame.lost_packets = missing
    frame.ready = frame.tick
    rounds = 0
    while missing and rounds < feedback.rtx_attempts:
        rounds += 1
        if feedback.loss.lost():  # the NACK was lost: this round brings nothing
            continue
        missing = sum(forward.lost() for _ in range(missing))
    frame.rtx_rounds = rounds
    frame.ready = frame.tick + rounds * rtt_ticks
    frame.arrived = missing == 0


def stall_stats(screen: list[str], tick_ms: float) -> dict:
    """Freeze statistics of a screen trace."""
    stalls, run = [], 0
    for state in screen:
        if state == "good":
            if run:
                stalls.append(run)
            run = 0
        else:
            run += 1
    if run:
        stalls.append(run)
    first_good = next((t for t, s in enumerate(screen) if s == "good"), None)
    # The start (nothing shown yet) counts as time to first picture, not a stall.
    later = stalls[1:] if screen and screen[0] != "good" and stalls else stalls
    not_good = sum(s != "good" for s in screen)
    return {
        "frozen_frames": not_good,
        "frozen_share": not_good / len(screen) if screen else 0.0,
        "corrupt_frames": sum(s == "corrupt" for s in screen),
        "stalls": len(later),
        "longest_stall_ms": max(later, default=0) * tick_ms,
        "mean_stall_ms": statistics.fmean(later) * tick_ms if later else 0.0,
        "p95_stall_ms": percentile(later, 95) * tick_ms if later else 0.0,
        "first_picture_ms": None if first_good is None else first_good * tick_ms,
    }


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return float(ordered[k])


def frame_stats(trace: Trace, fps: float, intra_period: int) -> dict:
    """Sizes, packets and what recovery cost."""
    frames = trace.frames
    seconds = (len(trace.screen) or 1) / fps
    keys = [f for f in frames if f.kind == "key"]
    inter = [f for f in frames if f.kind == "inter"]
    scheduled = [f for f in keys if f.tick % intra_period == 0]
    response_keys = [f for f in keys if f.tick % intra_period != 0]
    recovery = [f for f in inter if f.recovery]
    enh_repairs = [f for f in inter if (f.enh_repair or f.s2_repair) and not f.recovery]
    typical_inter = statistics.median([f.bytes for f in inter]) if inter else 0
    typical_key = statistics.median([f.bytes for f in scheduled]) if scheduled else 0
    # Recovery spend: what response keys, recovery frames and S1/S2 repairs cost
    # above an ordinary inter frame.
    extra = sum(max(0, f.bytes - typical_inter) for f in response_keys + recovery + enh_repairs)

    def frame_wire_bytes(f: SentFrame) -> int:
        if f.dropped_by_op:
            return 0
        if f.is_spatial:
            return (wire_bytes(f.s0_bytes)
                    + (wire_bytes(f.enh_bytes) if f.enh_packets > 0 else 0)
                    + (wire_bytes(f.s2_bytes) if f.s2_packets > 0 else 0))
        return wire_bytes(f.bytes)

    def packets(group: list[SentFrame]) -> dict:
        counts = [f.packets + f.enh_packets + f.s2_packets for f in group]
        return {"mean": statistics.fmean(counts) if counts else 0.0,
                "p50": percentile(counts, 50), "p95": percentile(counts, 95),
                "max": max(counts, default=0)}

    statuses = {}
    for f in frames:
        statuses[f.status or "pending"] = statuses.get(f.status or "pending", 0) + 1
    out = {
        "frames_sent": len(frames),
        "uncoded_frames": trace.uncoded,
        "payload_kbps": sum(f.bytes for f in frames) * 8 / seconds / 1000,
        "wire_kbps": sum(frame_wire_bytes(f) for f in frames) * 8 / seconds / 1000,
        "rtx_kbps": sum((f.lost_packets + f.enh_lost_packets + f.s2_lost_packets) * 1200
                        for f in frames if f.rtx_rounds) * 8 / seconds / 1000,
        "keys": len(keys),
        "response_keys": len(response_keys),
        "recovery_frames": len(recovery),
        "key_bytes_median": typical_key,
        "response_key_bytes_mean": statistics.fmean([f.bytes for f in response_keys])
        if response_keys else 0.0,
        "recovery_bytes_mean": statistics.fmean([f.bytes for f in recovery]) if recovery else 0.0,
        "inter_bytes_median": typical_inter,
        "recovery_kbps": extra * 8 / seconds / 1000,
        "packets_inter": packets(inter),
        "packets_key": packets(keys),
        "frames_lost": sum(not f.arrived and not f.dropped_by_op for f in frames),
        "packets_lost": sum(f.lost_packets + f.enh_lost_packets + f.s2_lost_packets for f in frames),
        "statuses": statuses,
        "mismatches": statuses.get("mismatch", 0),
        "decode_errors": statuses.get("corrupt", 0),
        "acks_sent": trace.acks_sent,
        "reports_sent": trace.reports_sent,
        "feedback_lost": trace.feedback_lost,
    }
    if any(f.is_spatial for f in frames):
        out["enhancement_packets_dropped"] = sum(f.enh_packets_dropped for f in frames)
        out["enhancement_packets_lost"] = sum(f.enh_lost_packets for f in frames)
        out["top_enhancement_packets_lost"] = sum(f.s2_lost_packets for f in frames)
        out["enhancement_repairs"] = sum(f.enh_repair for f in frames)
        out["s2_repairs"] = sum(f.s2_repair for f in frames)
        out["base_only_frames"] = statuses.get("base-only", 0) + statuses.get("enh-frozen", 0)
        out["s1_only_frames"] = statuses.get("s1-only", 0) + statuses.get("s2-frozen", 0)
        if trace.enhancement_screen:
            out["enhancement_frozen_frames"] = sum(s != "good" for s in trace.enhancement_screen)
        if trace.top_enhancement_screen:
            out["top_enhancement_frozen_frames"] = sum(
                s != "good" for s in trace.top_enhancement_screen
            )
    return out


def enhancement_recovery_times(trace: Trace, tick_ms: float) -> list[float]:
    """Per enhancement-layer loss event after the first enhancement picture:
    from the tick its first lost enhancement frame was sent to the next tick
    with the newest enhancement frame on screen."""
    screen = trace.enhancement_screen
    if not screen:
        return []
    lost_ticks = sorted(
        f.tick for f in trace.frames
        if f.is_spatial and f.enh_bytes > 0 and (not f.arrived or not f.enh_arrived)
    )
    times = []
    first_good = next((t for t, s in enumerate(screen) if s == "good"), len(screen))
    covered = -1
    for start in lost_ticks:
        if start <= covered or start < first_good:
            continue
        end = next((t for t in range(start, len(screen)) if screen[t] == "good"), None)
        if end is None:
            break
        times.append((end - start) * tick_ms)
        covered = end
    return times


def recovery_times(trace: Trace, tick_ms: float) -> list[float]:
    """Per loss event after the first picture: from the tick its first lost
    frame was sent to the next tick with the newest frame on screen."""
    lost_ticks = sorted(f.tick for f in trace.frames if not f.arrived)
    times = []
    screen = trace.screen
    first_good = next((t for t, s in enumerate(screen) if s == "good"), len(screen))
    covered = -1
    for start in lost_ticks:
        if start <= covered or start < first_good:
            continue
        end = next((t for t in range(start, len(screen)) if screen[t] == "good"), None)
        if end is None:
            break
        times.append((end - start) * tick_ms)
        covered = end
    return times


def first_quality_tick(screen: list[str], vmaf: list[float], margin: float) -> int | None:
    """First tick whose shown picture is within `margin` VMAF points of the
    run's settled quality (median of the good ticks of its second half)."""
    settled = [v for s, v in zip(screen[len(screen) // 2:], vmaf[len(vmaf) // 2:]) if s == "good"]
    if not settled:
        return None
    target = statistics.median(settled) - margin
    return next((t for t, (s, v) in enumerate(zip(screen, vmaf)) if s == "good" and v >= target),
                None)
