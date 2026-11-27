// Exact frames of a clip file for the comparison (app.js), decoded on demand with
// WebCodecs' VideoDecoder from the file's own samples (mp4.js), instead of
// sampling a playing <video>: frame callbacks can come late or not at all, and a
// slow page would then code other frames than a fast one. Here frame i is always
// the clip's i-th picture, so a run's streams do not depend on the machine's
// speed: a slow pipeline only slows the clip's time down (app.js).
//
// Frames are asked for in presentation order and mostly one after another; a
// request behind the decoder's position (a loop back to the start, a step) resets
// the decoder at the last key frame at or before it.
//
// Feeding is driven by the decoder's outputs and bounded twice: at most
// MAX_PENDING_SAMPLES in the decoder, and, beyond the frame waited for, at most
// DECODE_AHEAD_FRAMES decoded or decoding. Feeding on a timer whatever the
// decoder's backlog (as this file once did) queued the rest of the clip on a
// slow machine: ~550 samples and as many unclosed 720p VideoFrames, and the
// decoder then stalled for 1-9 s at a time (the demo's periodic pauses;
// tests/clip-pauses.py).
import { readMp4, trackKbps } from './mp4.js';

const MICROSECONDS_PER_SECOND = 1e6;
// The most samples in the decoder at once: H.264's largest DPB (16 frames, Annex
// A.3.1), so a conforming decoder outputs a frame before it holds this many. The
// hosted clips (two B-frames) come out 2-3 samples behind the input.
const MAX_PENDING_SAMPLES = 16;
// Frames decoded (or decoding) ahead of the last one handed out, so the next
// request is usually ready at once and the clip decoder overlaps the codecs: four
// ticks of a 60 fps clip, of which the page shows every other frame.
const DECODE_AHEAD_FRAMES = 8;

export class ClipSource {
  /**
   * The clip at `url` as a frame source of `fps` frames per second (null: the
   * file's own), or null when
   * this browser cannot decode it this way (no VideoDecoder, not an MP4 this
   * reader handles, codec unsupported): the caller then samples the <video>.
   */
  static async open(url, fps) {
    if (typeof VideoDecoder !== 'function') return null;
    const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
    const track = readMp4(bytes);
    if (!track || !track.samples.length) return null;
    // A file of unknown rate: its typical frame duration.
    fps ??= Math.round(track.timescale / track.samples.map(s => s.duration).sort((a, b) => a - b)[track.samples.length >> 1]);
    const config = { codec: track.codec, description: track.description, codedWidth: track.width, codedHeight: track.height,
      optimizeForLatency: true };
    if (!(await VideoDecoder.isConfigSupported(config)).supported) return null;
    return new ClipSource(bytes, track, config, fps);
  }
  constructor(bytes, track, config, fps) {
    this.bytes = bytes; this.config = config; this.fps = fps; this.timescale = track.timescale;
    /** The file's video bitrate as it plays (mp4.js `trackKbps`), for the views' over-half-the-source note. */
    this.kbps = trackKbps(track, fps);
    // Decode order, each with its presentation index.
    this.samples = track.samples.map(sample => ({ ...sample, index: Math.round(sample.pts * fps / track.timescale) }));
    this.frameCount = this.samples.length;
    // Decoded frames by presentation index, waiting to be taken.
    this.ready = new Map();
    this.waiters = new Map();
    this.error = null;
    this.decoder = this.createDecoder(0);
  }
  /** A new decoder, fed from sample `next` on. */
  createDecoder(next) {
    // The next sample to feed, the samples fed whose frames have not come out,
    // and the last frame index handed out since this decoder started.
    this.next = next; this.pending = 0; this.lastTaken = -1;
    // Whether the decoder was flushed after the last sample, and whether that
    // flush is done (every frame it will give is out).
    this.flushed = false; this.drained = false;
    const decoder = new VideoDecoder({
      output: frame => this.received(frame),
      error: error => { this.error = error; this.rejectWaiters(error); },
    });
    decoder.configure(this.config);
    return decoder;
  }
  received(frame) {
    this.pending--;
    const index = Math.round(frame.timestamp * this.fps / MICROSECONDS_PER_SECOND);
    const waiter = this.waiters.get(index);
    if (waiter) { this.waiters.delete(index); waiter.resolve(frame); }
    // Frames before one already taken or waited for are not asked for before a restart.
    else if (index <= this.lastTaken || [...this.waiters.keys()].some(wanted => wanted > index)) frame.close();
    else this.ready.set(index, frame);
    this.feed();
  }
  /** Frame `index` (0 to frameCount - 1) as a VideoFrame the caller owns and closes. */
  async frame(index) {
    if (this.error) throw this.error;
    if (this.ready.has(index)) {
      const frame = this.take(index);
      this.feed();
      return frame;
    }
    const position = this.samples.findIndex(s => s.index === index);
    if (position < 0) throw new Error(`The clip has no frame ${index}`);
    // At or behind a frame already taken (a loop back, a step back): decode again
    // from a key frame. Later frames already fed are still on their way out.
    if (index <= this.lastTaken) this.restart(position);
    else this.discardBefore(index);
    if (this.drained) throw new Error(`The decoder gave no frame ${index}`);
    const frame = new Promise((resolve, reject) => this.waiters.set(index, { resolve, reject }));
    this.feed();
    return this.handOut(index, await frame);
  }
  /**
   * Feed the decoder while it has room (MAX_PENDING_SAMPLES) and, unless a frame
   * is waited for, while fewer than DECODE_AHEAD_FRAMES are decoded or decoding;
   * after the last sample, flush it so the frames it holds for reordering come out.
   */
  feed() {
    while (this.next < this.samples.length && this.pending < MAX_PENDING_SAMPLES
      && (this.waiters.size || this.ready.size + this.pending < DECODE_AHEAD_FRAMES)) {
      const sample = this.samples[this.next++];
      this.decoder.decode(new EncodedVideoChunk({ type: sample.key ? 'key' : 'delta',
        timestamp: Math.round(sample.index * MICROSECONDS_PER_SECOND / this.fps),
        data: this.bytes.subarray(sample.offset, sample.offset + sample.size) }));
      this.pending++;
    }
    if (this.next === this.samples.length && !this.flushed) {
      this.flushed = true;
      const decoder = this.decoder;
      // A restart closes the decoder, which rejects the flush: nothing to do then.
      decoder.flush().then(() => {
        if (decoder !== this.decoder) return;
        this.drained = true;
        this.rejectWaiters(null);
      }, () => {});
    }
  }
  /** Fail every waiting request: with `error`, or (null) as a frame the drained decoder never gave. */
  rejectWaiters(error) {
    for (const [index, { reject }] of this.waiters) reject(error ?? new Error(`The decoder gave no frame ${index}`));
    this.waiters.clear();
  }
  take(index) {
    const frame = this.ready.get(index);
    this.ready.delete(index);
    return this.handOut(index, frame);
  }
  /** Hand frame `index` out; frames before it will not be asked for again before a restart. */
  handOut(index, frame) {
    this.lastTaken = index;
    this.discardBefore(index);
    return frame;
  }
  /** Close the decoded frames before `index`. */
  discardBefore(index) {
    for (const [other, held] of this.ready) if (other < index) { held.close(); this.ready.delete(other); }
  }
  /** Restart decoding at the last key frame at or before sample `position`. */
  restart(position) {
    let key = position;
    while (key > 0 && !this.samples[key].key) key--;
    this.decoder.close();
    for (const frame of this.ready.values()) frame.close();
    this.ready.clear();
    this.decoder = this.createDecoder(key);
  }
  /** Release the decoder and its frames; a request still waiting fails (a stopped comparison's pump, app.js). */
  close() {
    this.rejectWaiters(new Error('The clip source was closed'));
    for (const frame of this.ready.values()) frame.close();
    this.ready.clear();
    if (this.decoder.state !== 'closed') this.decoder.close();
  }
}
