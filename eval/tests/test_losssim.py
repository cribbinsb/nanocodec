"""nceval.lossnet models and the nceval.lossloop closed loop on a fake session."""

from __future__ import annotations

import random

import pytest

from nceval.lossloop import (Feedback, SentFrame, SpatialTuInfo, enhancement_recovery_times,
                             first_quality_tick, frame_stats, recovery_times, run_loop,
                             stall_stats, transmit)
from nceval.lossnet import (GilbertElliott, NoLoss, loss_model, packet_count, wire_bytes)


def test_packets_and_wire_bytes():
    assert packet_count(0) == 1
    assert packet_count(1200) == 1
    assert packet_count(1201) == 2
    assert wire_bytes(2500) == 2500 + 3 * 48


def test_gilbert_elliott_matches_rate_and_burst():
    model = GilbertElliott(0.05, 3.0, random.Random(7))
    draws = [model.lost() for _ in range(400_000)]
    assert sum(draws) / len(draws) == pytest.approx(0.05, abs=0.004)
    bursts, run = [], 0
    for lost in draws:
        if lost:
            run += 1
        elif run:
            bursts.append(run)
            run = 0
    assert sum(bursts) / len(bursts) == pytest.approx(3.0, rel=0.05)


def test_models_are_seeded_and_presets_resolve(tmp_path):
    a, b = loss_model("burst5", 3), loss_model("burst5", 3)
    assert [a.lost() for _ in range(1000)] == [b.lost() for _ in range(1000)]
    trace = tmp_path / "trace.txt"
    trace.write_text("0 1 1 0")
    model = loss_model(f"trace:{trace}", 0)
    assert [model.lost() for _ in range(6)] == [False, True, True, False, False, True]
    with pytest.raises(ValueError):
        loss_model("nonsense", 1)


class AlwaysLost:
    def lost(self) -> bool:
        return True


def test_retransmission_rounds_cost_round_trips():
    frame = SentFrame(tick=10, index=10, kind="key", layer=0, bytes=6000, references=(),
                      packets=5)
    # Every first transmission lost, every retransmission arrives.
    draws = iter([True] * 5 + [False] * 5)

    class Scripted:
        def lost(self) -> bool:
            return next(draws)

    transmit(frame, Scripted(), Feedback(50.0, NoLoss(), rtx_attempts=3), rtt_ticks=3)
    assert (frame.arrived, frame.ready, frame.rtx_rounds) == (True, 13, 1)
    lost = SentFrame(tick=0, index=0, kind="key", layer=0, bytes=100, references=(), packets=1)
    transmit(lost, AlwaysLost(), Feedback(50.0, NoLoss(), rtx_attempts=2), rtt_ticks=3)
    assert (lost.arrived, lost.ready) == (False, 6)


class FakeSession:
    """A flat chain with key-frame recovery and a transport receiver."""

    def __init__(self):
        self.next = 0
        self.key_requested = True
        self.repaired_from = 0
        self.good: set[int] = set()
        self.reports: list[int] = []
        self.reports_applied: list[tuple[int, int]] = []

    def acknowledge(self, index: int) -> None:
        pass

    def report_loss(self, index: int) -> None:
        self.reports.append(index)
        if index >= self.repaired_from:
            self.key_requested = True

    def encode(self):
        index, self.next = self.next, self.next + 1
        if self.key_requested:
            self.key_requested, self.repaired_from = False, index
            return index, "key", 0, 5000, ()
        return index, "inter", 0, 500, (index - 1,)

    def receive(self, index: int, arrived: bool) -> str:
        if not arrived:
            return "lost"
        self.good.add(index)  # every reference is checked by the caller below
        return "good"

    def show(self) -> None:
        pass


class ChainSession(FakeSession):
    def receive(self, index: int, arrived: bool) -> str:
        if not arrived:
            return "lost"
        if index > 0 and index - 1 not in self.good and index != self.repaired_from:
            return "missing-reference"
        self.good.add(index)
        return "good"


def test_loop_freezes_reports_and_recovers_with_a_key():
    session = ChainSession()
    # The key takes 5 packets, each inter frame one: draw 9 is frame 5's.
    draws = iter([False] * 9 + [True])

    class Scripted:
        def lost(self) -> bool:
            return next(draws, False)

    # Only frame 5 (one packet) is lost; its report arrives one round trip
    # (2 ticks) later and the next frame is a key.
    trace = run_loop(session, 12, 30.0, Scripted(), Feedback(1000 / 30, NoLoss()))
    assert session.reports[0] == 5
    kinds = [f.kind for f in trace.frames]
    assert kinds[0] == "key" and kinds[7] == "key" and kinds.count("key") == 2
    assert trace.screen[:5] == ["good"] * 5
    assert trace.screen[5:7] == ["frozen", "frozen"]
    assert trace.screen[7:] == ["good"] * 5
    stats = stall_stats(trace.screen, 1000 / 30)
    assert stats["stalls"] == 1 and stats["first_picture_ms"] == 0
    assert recovery_times(trace, 1000 / 30) == [pytest.approx(2000 / 30)]


def test_repeated_reports_while_frozen_do_not_storm():
    session = ChainSession()
    # The first two reports are lost: the receiver repeats each round trip.
    feedback_draws = iter([True, True])

    class FeedbackLoss:
        def lost(self) -> bool:
            return next(feedback_draws, False)

    draws = iter([False] * 7 + [True])

    class Forward:
        def lost(self) -> bool:
            return next(draws, False)

    trace = run_loop(session, 40, 30.0, Forward(), Feedback(1000 / 30, FeedbackLoss()))
    assert [f.kind for f in trace.frames].count("key") == 2
    assert trace.feedback_lost >= 2


def test_first_quality_tick():
    screen = ["blank", "good", "good", "good", "good", "good"]
    vmaf = [0, 40, 70, 80, 81, 80]
    assert first_quality_tick(screen, vmaf, 6.0) == 3


class SpatialFakeSession:
    """Two-layer spatial SVC session with S1-only repair on S1 loss and key TU on S0 loss."""

    def __init__(self, op_max_spatial: int):
        self.next = 0
        self.op_max_spatial = op_max_spatial
        self.key_requested = True
        self.s1_repair_requested = False
        self.s0_repaired_from = 0
        self.s1_repaired_from = 0
        self.s0_good: set[int] = set()
        self.s1_good: set[int] = set()
        self.s1_repairs: set[int] = set()
        self.last_spatial_info: SpatialTuInfo | None = None
        self.reports_applied: list[tuple[int, int]] = []

    def acknowledge(self, index: int, spatial_id: int | None = None) -> None:
        pass

    def report_loss(self, index: int, spatial_id: int | None = None) -> None:
        self.reports_applied.append((self.next, index))
        if spatial_id is None or spatial_id == 0:
            if index >= self.s0_repaired_from:
                self.key_requested = True
        elif spatial_id == 1 and index >= self.s1_repaired_from:
            self.s1_repair_requested = True

    def encode(self):
        idx = self.next
        self.next += 1
        key = False
        s1_repair = False
        if self.key_requested:
            key = True
            self.key_requested = False
            self.s1_repair_requested = False
            self.s0_repaired_from = idx
            self.s1_repaired_from = idx
        elif self.s1_repair_requested:
            s1_repair = True
            self.s1_repair_requested = False
            self.s1_repaired_from = idx
            self.s1_repairs.add(idx)
        s0_b = 2000 if key else 400
        s1_b = 3600 if key else (2000 if s1_repair else 1400)
        self.last_spatial_info = SpatialTuInfo(
            self.op_max_spatial, 0, s0_b, s1_b, s1_repair=s1_repair
        )
        return idx, ("key" if key else "inter"), 0, s0_b + s1_b, (() if key else (idx - 1,))

    def receive_spatial(self, index: int, s0_arrived: bool, s1_arrived: bool) -> str:
        if not s0_arrived:
            return "lost"
        key = index == self.s0_repaired_from
        if not key and (index - 1) not in self.s0_good:
            return "missing-reference"
        self.s0_good.add(index)
        if self.op_max_spatial == 0:
            assert not s1_arrived
            return "good"
        if s1_arrived and (key or index in self.s1_repairs or (index - 1) in self.s1_good):
            self.s1_good.add(index)
            return "good"
        return "enh-frozen" if s1_arrived else "base-only"

    def show(self) -> None:
        pass


def test_spatial_base_operating_point_drops_enhancement_packets():
    session = SpatialFakeSession(op_max_spatial=0)
    trace = run_loop(session, 8, 30.0, NoLoss(), Feedback(25.0, NoLoss()))
    stats = frame_stats(trace, 30.0, 300)
    assert trace.screen == ["good"] * 8
    assert stats["enhancement_packets_dropped"] > 0
    assert all(f.enh_packets == 0 for f in trace.frames)


def test_spatial_enhancement_loss_freezes_enhancement_only():
    session = SpatialFakeSession(op_max_spatial=1)
    # Lose S1 on frame 2 (key at 0 has 3 S1 packets; frame 1 has 2; frame 2's first S1 packet is draw 5).
    enh_draws = iter([False] * 5 + [True] + [False] * 20)

    class EnhLoss:
        def lost(self) -> bool:
            return next(enh_draws, False)

    trace = run_loop(session, 8, 30.0, NoLoss(), Feedback(25.0, NoLoss()),
                     enhancement_forward=EnhLoss())
    stats = frame_stats(trace, 30.0, 300)
    # Base keeps playing every single tick without stalling and without a response key.
    assert trace.screen == ["good"] * 8
    assert stall_stats(trace.screen, 1000 / 30)["frozen_frames"] == 0
    assert stats["keys"] == 1 and stats["response_keys"] == 0
    assert stats["enhancement_repairs"] == 1
    # Feedback arrives after 1 RTT (2 ticks), so S1 repairs on TU 4 without a key TU.
    assert trace.enhancement_screen == [
        "good", "good", "frozen", "frozen", "good", "good", "good", "good"]
    assert stats["base_only_frames"] == 2
    assert stats["enhancement_frozen_frames"] == 2
    assert enhancement_recovery_times(trace, 1000 / 30) == [pytest.approx(2000 / 30)]


class ThreeLayerSpatialFakeSession:
    """Three-layer SVC session verifying S2-only loss, S1+S2 loss, and lower-OP isolation."""

    def __init__(self, op_max_spatial: int):
        self.next = 0
        self.op_max_spatial = op_max_spatial
        self.key_requested = True
        self.s1_repair_requested = False
        self.s2_repair_requested = False
        self.s0_repaired_from = 0
        self.s1_repaired_from = 0
        self.s2_repaired_from = 0
        self.s0_good: set[int] = set()
        self.s1_good: set[int] = set()
        self.s2_good: set[int] = set()
        self.s1_repairs: set[int] = set()
        self.s2_repairs: set[int] = set()
        self.last_spatial_info: SpatialTuInfo | None = None
        self.reports_applied: list[tuple[int, int]] = []

    def acknowledge(self, index: int, spatial_id: int | None = None) -> None:
        pass

    def report_loss(self, index: int, spatial_id: int | None = None) -> None:
        self.reports_applied.append((self.next, index))
        if spatial_id is None or spatial_id == 0:
            if index >= self.s0_repaired_from:
                self.key_requested = True
        elif spatial_id == 1 and index >= self.s1_repaired_from:
            self.s1_repair_requested = True
        elif spatial_id == 2 and index >= self.s2_repaired_from:
            self.s2_repair_requested = True

    def encode(self):
        idx = self.next
        self.next += 1
        key = s1_repair = s2_repair = False
        if self.key_requested:
            key = True
            self.key_requested = self.s1_repair_requested = self.s2_repair_requested = False
            self.s0_repaired_from = self.s1_repaired_from = self.s2_repaired_from = idx
        elif self.s1_repair_requested:
            s1_repair = True
            self.s1_repair_requested = self.s2_repair_requested = False
            self.s1_repaired_from = self.s2_repaired_from = idx
            self.s1_repairs.add(idx)
            self.s2_repairs.add(idx)
        elif self.s2_repair_requested:
            s2_repair = True
            self.s2_repair_requested = False
            self.s2_repaired_from = idx
            self.s2_repairs.add(idx)
        s0_b, s1_b, s2_b = (1000, 1000, 1000) if not key else (2000, 2000, 2000)
        self.last_spatial_info = SpatialTuInfo(
            self.op_max_spatial, 0, s0_b, s1_b, s2_bytes=s2_b, num_layers=3,
            s1_repair=s1_repair, s2_repair=s2_repair,
        )
        return idx, ("key" if key else "inter"), 0, s0_b + s1_b + s2_b, (() if key else (idx - 1,))

    def receive_spatial(self, index: int, s0_arrived: bool, s1_arrived: bool,
                        s2_arrived: bool | None = None) -> str:
        assert s2_arrived is not None
        if not s0_arrived:
            return "lost"
        key = index == self.s0_repaired_from
        if not key and (index - 1) not in self.s0_good:
            return "missing-reference"
        self.s0_good.add(index)
        if self.op_max_spatial == 0:
            assert not s1_arrived and not s2_arrived
            return "good"
        s1_ok = s1_arrived and (key or index in self.s1_repairs or (index - 1) in self.s1_good)
        if not s1_ok:
            return "enh-frozen" if s1_arrived else "base-only"
        self.s1_good.add(index)
        if self.op_max_spatial == 1:
            assert not s2_arrived
            return "good"
        s2_ok = s2_arrived and (key or index in self.s2_repairs or (index - 1) in self.s2_good)
        if not s2_ok:
            return "s2-frozen" if s2_arrived else "s1-only"
        self.s2_good.add(index)
        return "good"

    def show(self) -> None:
        pass


def test_three_layer_s2_loss_freezes_only_top_view_and_s1_loss_freezes_s1_and_s2():
    # 1. Losing S2 on frame 2 freezes ONLY S2 (`top_enhancement_screen`); S0 and S1 keep playing!
    s2_session = ThreeLayerSpatialFakeSession(op_max_spatial=2)
    # Key at 0 takes 2 S2 packets, frame 1 takes 1 -> draw index 3 is frame 2's S2 packet.
    s2_draws = iter([False, False, False, True] + [False] * 20)

    class S2Loss:
        def lost(self) -> bool:
            return next(s2_draws, False)

    trace_s2 = run_loop(
        s2_session, 8, 30.0, NoLoss(), Feedback(25.0, NoLoss()),
        enhancement_forward=NoLoss(), top_enhancement_forward=S2Loss(),
    )
    stats_s2 = frame_stats(trace_s2, 30.0, 300)
    assert trace_s2.screen == ["good"] * 8
    assert trace_s2.enhancement_screen == ["good"] * 8
    assert trace_s2.top_enhancement_screen == [
        "good", "good", "frozen", "frozen", "good", "good", "good", "good"]
    assert stats_s2["s2_repairs"] == 1 and stats_s2["enhancement_repairs"] == 0
    assert stats_s2["response_keys"] == 0 and stats_s2["mismatches"] == 0

    # 2. Losing S1 on frame 2 freezes S1 AND S2 while S0 keeps playing!
    s1_session = ThreeLayerSpatialFakeSession(op_max_spatial=2)
    s1_draws = iter([False, False, False, True] + [False] * 20)

    class S1Loss:
        def lost(self) -> bool:
            return next(s1_draws, False)

    trace_s1 = run_loop(
        s1_session, 8, 30.0, NoLoss(), Feedback(25.0, NoLoss()),
        enhancement_forward=S1Loss(), top_enhancement_forward=NoLoss(),
    )
    stats_s1 = frame_stats(trace_s1, 30.0, 300)
    assert trace_s1.screen == ["good"] * 8
    assert trace_s1.enhancement_screen == [
        "good", "good", "frozen", "frozen", "good", "good", "good", "good"]
    assert trace_s1.top_enhancement_screen == [
        "good", "good", "frozen", "frozen", "good", "good", "good", "good"]
    assert stats_s1["enhancement_repairs"] == 1 and stats_s1["response_keys"] == 0

    # 3. Subscribed to middle OP (`op_max_spatial == 1`), 100% S2 loss has zero effect!
    mid_session = ThreeLayerSpatialFakeSession(op_max_spatial=1)
    trace_mid = run_loop(
        mid_session, 8, 30.0, NoLoss(), Feedback(25.0, NoLoss()),
        enhancement_forward=NoLoss(), top_enhancement_forward=AlwaysLost(),
    )
    assert trace_mid.screen == ["good"] * 8
    assert trace_mid.enhancement_screen == ["good"] * 8
    assert not trace_mid.top_enhancement_screen


class KsvcThreeLayerFakeSession:
    """Three-layer K-SVC session (`il=0:1:1` on key TUs, `il=0:0:0` on non-key TUs)."""

    def __init__(self, op_max_spatial: int):
        self.next = 0
        self.op_max_spatial = op_max_spatial
        self.key_requested = True
        self.s1_repair_requested = False
        self.s2_repair_requested = False
        self.s0_repaired_from = 0
        self.s1_repaired_from = 0
        self.s2_repaired_from = 0
        self.s0_good: set[int] = set()
        self.s1_good: set[int] = set()
        self.s2_good: set[int] = set()
        self.s1_repairs: set[int] = set()
        self.s2_repairs: set[int] = set()
        self.layer_reports: list[tuple[int, int | None]] = []
        self.last_spatial_info: SpatialTuInfo | None = None
        self.reports_applied: list[tuple[int, int]] = []

    def acknowledge(self, index: int, spatial_id: int | None = None) -> None:
        pass

    def report_loss(self, index: int, spatial_id: int | None = None) -> None:
        self.reports_applied.append((self.next, index))
        self.layer_reports.append((index, spatial_id))
        if spatial_id is None or spatial_id == 0:
            if index >= self.s0_repaired_from:
                self.key_requested = True
        elif spatial_id == 1 and index >= self.s1_repaired_from:
            self.s1_repair_requested = True
        elif spatial_id == 2 and index >= self.s2_repaired_from:
            self.s2_repair_requested = True

    def encode(self):
        idx = self.next
        self.next += 1
        key = s1_repair = s2_repair = False
        if self.key_requested:
            key = True
            self.key_requested = self.s1_repair_requested = self.s2_repair_requested = False
            self.s0_repaired_from = self.s1_repaired_from = self.s2_repaired_from = idx
        else:
            if self.s1_repair_requested:
                s1_repair = True
                self.s1_repair_requested = False
                self.s1_repaired_from = idx
                self.s1_repairs.add(idx)
            if self.s2_repair_requested:
                s2_repair = True
                self.s2_repair_requested = False
                self.s2_repaired_from = idx
                self.s2_repairs.add(idx)
        s0_b, s1_b, s2_b = (2000, 2000, 2000) if key else (500, 900, 1100)
        il_refs = (0, 1, 1) if key else (0, 0, 0)
        self.last_spatial_info = SpatialTuInfo(
            self.op_max_spatial, 0, s0_b, s1_b, s2_bytes=s2_b, num_layers=3,
            s1_repair=s1_repair, s2_repair=s2_repair, il_refs=il_refs,
        )
        return idx, ("key" if key else "inter"), 0, s0_b + s1_b + s2_b, (() if key else (idx - 1,))

    def receive_spatial(self, index: int, s0_arrived: bool, s1_arrived: bool,
                        s2_arrived: bool | None = None) -> str:
        assert s2_arrived is not None
        key = index == self.s0_repaired_from
        if not key and self.op_max_spatial == 2:
            # Selective forwarder dropped S0 and S1 before the link on every non-key TU!
            assert not s0_arrived and not s1_arrived
            if not s2_arrived:
                return "lost"
            if index in self.s2_repairs or (index - 1) in self.s2_good:
                self.s2_good.add(index)
                return "good"
            return "missing-reference"
        if not key and self.op_max_spatial == 1:
            assert not s0_arrived and not s2_arrived
            if not s1_arrived:
                return "lost"
            if index in self.s1_repairs or (index - 1) in self.s1_good:
                self.s1_good.add(index)
                return "good"
            return "missing-reference"
        if not s0_arrived:
            return "lost"
        self.s0_good.add(index)
        if self.op_max_spatial == 0:
            return "good"
        if not s1_arrived:
            return "base-only"
        self.s1_good.add(index)
        if self.op_max_spatial == 1:
            return "good"
        if not s2_arrived:
            return "s1-only"
        self.s2_good.add(index)
        return "good"

    def show(self) -> None:
        pass


def test_ksvc_selective_forwarding_recovers_s2_and_s1_without_lower_layer_resend():
    # Top 720p receiver (`op_max_spatial == 2`): on non-key TUs only S2 is sent,
    # and an S2 loss on TU 2 triggers `loss 2 2` (NOT a whole-stream keyframe!)
    # and recovers on TU 4 via standalone S2 repair without resending S0 or S1.
    s2_session = KsvcThreeLayerFakeSession(op_max_spatial=2)
    # Key at TU 0 takes 2 S2 packets, TU 1 takes 1 -> draw 3 is TU 2's S2 packet.
    s2_draws = iter([False, False, False, True] + [False] * 20)

    class S2Loss:
        def lost(self) -> bool:
            return next(s2_draws, False)

    trace_s2 = run_loop(
        s2_session, 8, 30.0, AlwaysLost() if False else NoLoss(),
        Feedback(25.0, NoLoss()),
        enhancement_forward=NoLoss(), top_enhancement_forward=S2Loss(),
    )
    stats_s2 = frame_stats(trace_s2, 30.0, 300)
    assert s2_session.layer_reports == [(2, 2)]
    assert stats_s2["keys"] == 1 and stats_s2["response_keys"] == 0
    assert stats_s2["s2_repairs"] == 1 and stats_s2["enhancement_repairs"] == 0
    assert all(f.packets == 0 and f.enh_packets == 0 for f in trace_s2.frames[1:])
    assert trace_s2.screen == [
        "good", "good", "frozen", "frozen", "good", "good", "good", "good"]

    # Middle 360p receiver (`op_max_spatial == 1`): 100% S0 loss on non-key TUs has
    # zero effect because S0 is dropped before the link on non-key TUs!
    s1_session = KsvcThreeLayerFakeSession(op_max_spatial=1)
    s0_nonkey_loss = iter([False, False] + [True] * 20)

    class NonKeyS0Loss:
        def lost(self) -> bool:
            return next(s0_nonkey_loss, True)

    trace_s1 = run_loop(
        s1_session, 8, 30.0, NonKeyS0Loss(), Feedback(25.0, NoLoss()),
        enhancement_forward=NoLoss(),
    )
    assert trace_s1.screen == ["good"] * 8
    assert trace_s1.enhancement_screen == ["good"] * 8
