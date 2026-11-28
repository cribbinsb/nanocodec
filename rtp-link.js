// The receiving end of a view's RTP link, in its decode worker
// (decode-worker.js): the sender's packets (encode-worker.js packetizes each
// coded frame with nc-web `RtpSender`) go through an optional selective
// forwarding server that drops layers by the Dependency Descriptor
// (`RtpLayerFilter`), then the network emulator (network.js), then nc-web
// `RtpReceiver` (reorder window, playout deadline, decodability by the
// descriptor's frame dependencies). The receiver's loss feedback goes back
// to the encoder one feedback delay later (`feedback`): an acknowledgement of
// every decoded frame with long-term reference feedback on, a report of every
// lost one with it or with spatial layers (the encoder repairs a lost
// enhancement layer from the layer below), and a key frame request (a PLI)
// when decoding stopped, repeated every round trip while the picture stays frozen.
//
// Every codec's packets carry the Dependency Descriptor (nc-rtp writes it for
// any format), so the receiver knows each frame's number and the frames it
// predicts from, and drops what it cannot decode instead of handing it to the
// decoder: what the view shows is what the decoder really outputs.
import { NetworkEmulator, PERFECT_NETWORK, feedbackDelayMs, playoutDelayMs, roundTripMs } from './network.js';

/** The RFC 8285 extension id of the Dependency Descriptor, sender and receiver alike. */
export const DEPENDENCY_DESCRIPTOR_ID = 1;
/** The RTP video clock: 90 kHz. */
const RTP_TICKS_PER_MS = 90;
const RTP_TIMESTAMP_RANGE = 2 ** 32;
/** The wire bitrate covers this much source time. */
const WIRE_WINDOW_MS = 1000;
/**
 * A run of this many frames without a new picture is a freeze; a shorter one is a
 * skip: a lost top-temporal-layer frame (one frame), or an L1T3 middle-layer frame
 * and the top-layer frame that predicts from it (two).
 */
const FREEZE_FRAMES = 3;
/** The timeline's lanes, top to bottom (link-panel.js draws them). */
export const TIMELINE_LANES = {
  network: { label: 'Packet lost', colour: '#b45a4f' },
  feedback: { label: 'Feedback', colour: '#c08a2c' },
  encoder: { label: 'Key / repair coded', colour: '#226b51' },
  receiver: { label: 'Frozen', colour: '#6d5689' },
};

/** The RTP timestamp of source time `ms` (nc-rtp `rtp_timestamp`). */
export const rtpTimestamp = ms => Math.round(ms * RTP_TICKS_PER_MS) % RTP_TIMESTAMP_RANGE;

export class RtpLink {
  /**
   * `link`: the view's link settings (link-panel.js `settings`); `layers`: the
   * stream's `{ spatialLayers, temporalLayers }`; `frameMs`: one frame interval.
   */
  constructor(module, codec, link, layers, frameMs) {
    this.link = link; this.layers = layers; this.frameMs = frameMs;
    this.settings = link.mode === 'lossy' ? link.network : { ...PERFECT_NETWORK, mtu: link.network.mtu };
    this.network = new NetworkEmulator(this.settings);
    this.receiver = new module.RtpReceiver(codec, DEPENDENCY_DESCRIPTOR_ID, false);
    const { spatial, temporal } = link.sfu;
    const forwardsAll = spatial >= layers.spatialLayers - 1 && temporal >= layers.temporalLayers - 1;
    this.filter = forwardsAll ? null : new module.RtpLayerFilter(DEPENDENCY_DESCRIPTOR_ID, spatial, temporal);
    this.counts = { sfuDropped: 0, framesDecoded: 0, framesFrozen: 0, framesDropped: 0, keyRequests: 0, lossReports: 0, malformed: 0 };
    this.recoveries = [];
    // Source times of the frames sent, and of the newest decoded one, for freezes.
    this.sentAt = [];
    this.lastDecodedAt = null;
    this.freezeStart = null; this.lastKeyRequest = -Infinity;
    this.wire = [];
    this.events = [];
    this.now = 0;
    // The receiver's count of units it did not deliver, at the last `feedback`.
    this.droppedSeen = 0;
  }
  /** The network settings changed while running (live). */
  configure(network) {
    if (this.link.mode !== 'lossy') return;
    this.settings = { ...network, mtu: this.settings.mtu };
    this.network.configure(this.settings);
  }
  event(lane, at, details = {}) { this.events.push({ lane, at, ...details }); }
  /**
   * Send one coded frame's packets (`rtp`: `{ packets, layers }`, from the encode
   * worker) at source time `atMs`: through the SFU, onto the network.
   */
  send(rtp, atMs, key) {
    this.sentAt.push(atMs);
    if (key) this.event('encoder', atMs, { what: 'key' });
    rtp.packets.forEach((buffer, i) => {
      const [spatial, temporal] = rtp.layers[i];
      let packet = new Uint8Array(buffer);
      if (this.filter) {
        packet = this.filter.filter(packet);
        if (!packet) { this.counts.sfuDropped++; return; }
      }
      this.wire.push({ at: atMs, bytes: packet.byteLength });
      if (this.network.send(packet, { spatial, temporal }, atMs) !== 'sent') this.event('network', atMs, { layer: `S${spatial}T${temporal}` });
    });
  }
  /** The encoder applied loss feedback to the frame coded at `atMs` (encode-worker.js). */
  repaired(atMs) { this.event('encoder', atMs, { what: 'repair' }); }
  /**
   * Source time `nowMs`: the packets that have arrived go to the receiver, and
   * gaps the playout deadline has passed become loss. Returns the units ready
   * for the decoder: `{ unit, timestamp (RTP), key, frameIds }`.
   */
  receive(nowMs) {
    this.now = nowMs;
    for (const packet of this.network.due(nowMs)) {
      try { this.receiver.push(packet); } catch { this.counts.malformed++; }
    }
    const deadline = nowMs - playoutDelayMs(this.settings);
    if (deadline >= 0) this.receiver.set_playout_deadline(rtpTimestamp(deadline));
    const units = [];
    for (let unit = this.receiver.pop(); unit; unit = this.receiver.pop()) {
      units.push({ unit, timestamp: this.receiver.timestamp(), key: this.receiver.key(), frameIds: [...this.receiver.frame_ids()] });
    }
    return units;
  }
  /**
   * After the decoder took this tick's units (`decodedIds`: the frame numbers it
   * decoded; `refused`: it refused one), note freezes and recoveries and return the
   * feedback for the encoder, each item `{ kind: 'ack' | 'loss' | 'key', frame, at }`
   * with `at` the source time it reaches the encoder.
   */
  feedback(decodedIds, refused) {
    const now = this.now, at = now + feedbackDelayMs(this.settings), items = [];
    const { ltr, keyRequests } = this.link.resilience;
    // Loss reports drive LTR repairs and, with spatial layers, the enhancement-layer repair.
    const reports = ltr || this.layers.spatialLayers > 1;
    if (decodedIds.length) {
      this.counts.framesDecoded += 1;
      if (this.freezeStart !== null) {
        this.recoveries.push(now - this.freezeStart);
        this.event('receiver', this.freezeStart, { until: now });
        this.freezeStart = null;
      }
      this.lastDecodedAt = now;
      if (ltr) for (const frame of decodedIds) items.push({ kind: 'ack', frame, at });
    }
    const lost = [...this.receiver.lost_frames()];
    if (reports) {
      for (const frame of lost) items.push({ kind: 'loss', frame, at });
      this.counts.lossReports += lost.length;
      if (lost.length) this.event('feedback', now, { arrives: at, what: `loss of ${lost.join(', ')}` });
    }
    // Units the receiver did not hand to the decoder: lost, or undecodable after a loss.
    this.counts.framesDropped += this.receiver.dropped_units - this.droppedSeen;
    this.droppedSeen = this.receiver.dropped_units;
    const frozen = this.updateFreeze();
    // With long-term reference feedback the loss reports ask for the repair; a key
    // request then only follows a freeze that outlasts a round trip (the repair was lost too).
    const stopped = this.receiver.keyframe_needed() || refused;
    const repeat = frozen && now - this.lastKeyRequest >= roundTripMs(this.settings) + this.frameMs
      && now - this.freezeStart >= roundTripMs(this.settings) + this.frameMs;
    if (keyRequests && ((stopped && !reports) || repeat)) {
      items.push({ kind: 'key', at });
      this.counts.keyRequests++;
      this.lastKeyRequest = now;
      this.event('feedback', now, { arrives: at, what: 'key request' });
    }
    return items;
  }
  /**
   * Whether the picture is frozen: the newest frame due for playout by now was sent
   * FREEZE_FRAMES or more frames after the newest decoded one. A shorter gap is a skip.
   */
  updateFreeze() {
    const due = this.sentAt.filter(at => at <= this.now - playoutDelayMs(this.settings));
    this.sentAt = this.sentAt.filter(at => at > this.now - playoutDelayMs(this.settings) - this.frameMs * FREEZE_FRAMES * 4);
    const newestDue = due.at(-1);
    if (newestDue === undefined || this.lastDecodedAt === null) return false;
    const behind = newestDue - (this.lastDecodedAt - playoutDelayMs(this.settings));
    const frozen = this.freezeStart !== null || behind >= FREEZE_FRAMES * this.frameMs;
    if (frozen && this.lastDecodedAt < this.now) {
      this.freezeStart ??= this.now;
      this.counts.framesFrozen++;
      return true;
    }
    return false;
  }
  /**
   * The state the page shows (link-panel.js `LinkMonitor`): counters, the layer on
   * screen and the events since the last call.
   */
  state(shownLayer) {
    const since = this.now - WIRE_WINDOW_MS;
    this.wire = this.wire.filter(w => w.at > since);
    const wireBytes = this.wire.reduce((sum, w) => sum + w.bytes + 28, 0);
    const n = this.network.stats;
    const events = this.events; this.events = [];
    if (this.freezeStart !== null) events.push({ lane: 'receiver', at: this.freezeStart, until: this.now });
    return { now: this.now, packetsSent: n.sent, packetsLost: n.lost + n.overflowed, packetsLate: this.receiver.late_packets,
      layerLoss: n.layers, wireKbps: wireBytes * 8 / WIRE_WINDOW_MS, frozen: this.freezeStart !== null, shownLayer,
      recoveries: this.recoveries.slice(-20), ...this.counts, events };
  }
}
