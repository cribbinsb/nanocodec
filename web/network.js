// The in-page network emulator between a view's RTP sender and receiver
// (rtp-link.js): per packet it decides loss, queueing behind a bandwidth cap,
// delay, jitter, reordering and duplication, on the source timeline (ms), so
// a run is a pure function of its packets, settings and seed (no wall clock,
// no Math.random): the same clip and settings lose the same packets.
//
// Loss models: independent random loss, or Gilbert-Elliott bursts (two
// states; Bad loses every packet, Good none; the mean loss rate and mean
// burst length in packets set the transition probabilities: the simplified
// Gilbert model of ITU-T G.1050 / 3GPP MTSI test conditions, as
// eval/nceval/lossnet.py `ge:R:B`). Loss can apply to every packet, to base
// layer packets only (spatial 0, temporal 0) or to enhancement layer packets
// only, so layered streams' loss behaviour can be shown on its own.

/** Bytes a packet costs on the wire beyond its RTP bytes: IPv4 20 + UDP 8. */
export const UDP_IP_OVERHEAD_BYTES = 28;
const BITS_PER_BYTE = 8;
/** Extra delay a reordered packet gets beyond the jitter, ms. */
const REORDER_DELAY_MS = 15;

/** Network presets: loss, delay and jitter of typical links (one-way). */
export const NETWORK_PRESETS = {
  good: { label: 'Good · wired', lossPercent: 0, burstLength: 1, delayMs: 15, jitterMs: 2 },
  wifi: { label: 'Wi-Fi · 1 % bursts', lossPercent: 1, burstLength: 2, delayMs: 25, jitterMs: 10 },
  mobile: { label: 'Mobile · 3 % bursts', lossPercent: 3, burstLength: 3, delayMs: 50, jitterMs: 20 },
  bad: { label: 'Bad · 8 % bursts', lossPercent: 8, burstLength: 4, delayMs: 100, jitterMs: 35 },
};
export const DEFAULT_PRESET = 'mobile';
/** Where loss applies (`lossScope`). */
export const LOSS_SCOPES = { all: 'All packets', base: 'Base layer only', enhancement: 'Enhancement layers only' };

/** Every setting, at the default preset's values. `rttMs` null: twice the one-way delay. */
export const DEFAULT_NETWORK = {
  lossModel: 'burst', ...withoutLabel(NETWORK_PRESETS[DEFAULT_PRESET]), lossScope: 'all',
  reorderPercent: 0, duplicatePercent: 0, bandwidthKbps: 0, queueMs: 200, mtu: 1200, rttMs: null, seed: 1,
};
/** A perfect network (the plain RTP link): nothing lost, delayed or reordered. */
export const PERFECT_NETWORK = { ...DEFAULT_NETWORK, lossPercent: 0, delayMs: 0, jitterMs: 0 };

function withoutLabel({ label, ...settings }) { return settings; }

/** The round trip loss feedback takes: `rttMs`, or twice the one-way delay. */
export const roundTripMs = settings => settings.rttMs ?? 2 * settings.delayMs;
/** How long feedback takes from the receiver back to the sender: the round trip less the forward delay. */
export const feedbackDelayMs = settings => Math.max(0, roundTripMs(settings) - settings.delayMs);
/**
 * How long the receiver's jitter buffer waits for a missing packet: past the
 * forward delay and the jitter (and a reordered packet's extra delay), a gap
 * is loss (nc-rtp `set_playout_deadline`).
 */
export const playoutDelayMs = settings =>
  settings.delayMs + settings.jitterMs + (settings.reorderPercent > 0 ? REORDER_DELAY_MS : 0) + 5;

/** A small seeded generator (mulberry32): uniform numbers in [0, 1). */
export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Whether a packet of `layer` ({ spatial, temporal }) is subject to loss under `scope`. */
export function inLossScope(scope, layer) {
  const base = layer.spatial === 0 && layer.temporal === 0;
  return scope === 'all' || (scope === 'base' ? base : !base);
}

export class NetworkEmulator {
  constructor(settings) {
    this.settings = { ...DEFAULT_NETWORK, ...settings };
    this.random = seededRandom(this.settings.seed);
    this.bad = false;
    // When the bandwidth-capped link finishes sending its queue (ms).
    this.busyUntil = 0;
    // The latest arrival of an in-order packet: jitter does not reorder by itself.
    this.lastArrival = 0;
    // Packets on their way: { arrival, packet }, delivered by `due`.
    this.inFlight = [];
    this.stats = { sent: 0, lost: 0, overflowed: 0, duplicated: 0, reordered: 0, wireBytes: 0, layers: {} };
  }
  /** Change the settings from the next packet on (packets on their way keep their arrival). */
  configure(settings) { this.settings = { ...this.settings, ...settings }; }
  /** Per-layer counters ({ sent, lost }), keyed `S0T1` etc. */
  layerStats(layer) {
    const key = `S${layer.spatial}T${layer.temporal}`;
    this.stats.layers[key] ??= { sent: 0, lost: 0 };
    return this.stats.layers[key];
  }
  /**
   * Whether the loss model loses the next in-scope packet. Gilbert-Elliott: a
   * mean burst of B packets leaves Bad with probability 1 / B; entering it with
   * r / (1 - R) * R (r = 1 / B) gives a mean loss rate R.
   */
  loses() {
    const rate = this.settings.lossPercent / 100;
    if (rate <= 0) return false;
    if (this.settings.lossModel !== 'burst' || this.settings.burstLength <= 1) return this.random() < rate;
    const leave = 1 / this.settings.burstLength, enter = Math.min(1, rate * leave / (1 - rate));
    this.bad = this.bad ? this.random() >= leave : this.random() < enter;
    return this.bad;
  }
  /**
   * Send `packet` (bytes) of `layer` at `nowMs`. Returns what happened to it:
   * 'lost', 'overflow' (the bandwidth cap's queue was full) or 'sent'.
   */
  send(packet, layer, nowMs) {
    const s = this.settings, counters = this.layerStats(layer), wireBytes = packet.byteLength + UDP_IP_OVERHEAD_BYTES;
    this.stats.sent++; counters.sent++;
    let departure = nowMs;
    if (s.bandwidthKbps > 0) {
      const start = Math.max(nowMs, this.busyUntil);
      if (start - nowMs > s.queueMs) { this.stats.overflowed++; counters.lost++; return 'overflow'; }
      departure = start + wireBytes * BITS_PER_BYTE / s.bandwidthKbps;
      this.busyUntil = departure;
    }
    this.stats.wireBytes += wireBytes;
    if (inLossScope(s.lossScope, layer) && this.loses()) { this.stats.lost++; counters.lost++; return 'lost'; }
    this.schedule(packet, departure);
    if (s.duplicatePercent > 0 && this.random() < s.duplicatePercent / 100) {
      this.stats.duplicated++;
      this.schedule(packet.slice(), departure);
    }
    return 'sent';
  }
  /** Put a packet on its way: delay plus jitter, kept in order unless it is reordered. */
  schedule(packet, departure) {
    const s = this.settings;
    let arrival = departure + s.delayMs + this.random() * s.jitterMs;
    if (s.reorderPercent > 0 && this.random() < s.reorderPercent / 100) {
      this.stats.reordered++;
      arrival += REORDER_DELAY_MS;
    } else {
      arrival = Math.max(arrival, this.lastArrival);
      this.lastArrival = arrival;
    }
    this.inFlight.push({ arrival, packet });
  }
  /** The packets that have arrived by `nowMs`, in arrival order. */
  due(nowMs) {
    const arrived = this.inFlight.filter(p => p.arrival <= nowMs).sort((a, b) => a.arrival - b.arrival);
    this.inFlight = this.inFlight.filter(p => p.arrival > nowMs);
    return arrived.map(p => p.packet);
  }
}
