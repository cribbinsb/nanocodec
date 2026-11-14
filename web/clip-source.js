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
import { readMp4 } from './mp4.js';

const MICROSECONDS_PER_SECOND = 1e6;
// Samples fed per step while waiting for a frame: enough for B-frame reordering
// without holding many decoded frames (a decoder stalls when too many are unclosed).
const AHEAD_SAMPLES = 4;
// How long to wait for the decoder's output before feeding more (ms).
const OUTPUT_WAIT_MS = 10;

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
    // Decode order, each with its presentation index.
    this.samples = track.samples.map(sample => ({ ...sample, index: Math.round(sample.pts * fps / track.timescale) }));
    this.frameCount = this.samples.length;
    // Decoded frames by presentation index, waiting to be taken.
    this.ready = new Map();
    this.waiters = new Map();
    this.next = 0;
    // The last frame index handed out since the decoder (re)started.
    this.lastTaken = -1;
    this.error = null;
    // Whether the decoder was flushed after the last sample (its last frames are out).
    this.flushed = false;
    this.decoder = this.createDecoder();
  }
  createDecoder() {
    const decoder = new VideoDecoder({
      output: frame => this.received(frame),
      error: error => { this.error = error; for (const { reject } of this.waiters.values()) reject(error); this.waiters.clear(); },
    });
    decoder.configure(this.config);
    return decoder;
  }
  received(frame) {
    const index = Math.round(frame.timestamp * this.fps / MICROSECONDS_PER_SECOND);
    const waiter = this.waiters.get(index);
    if (waiter) { this.waiters.delete(index); waiter.resolve(frame); } else this.ready.set(index, frame);
  }
  /** Frame `index` (0 to frameCount - 1) as a VideoFrame the caller owns and closes. */
  async frame(index) {
    if (this.error) throw this.error;
    if (this.ready.has(index)) return this.take(index);
    const position = this.samples.findIndex(s => s.index === index);
    if (position < 0) throw new Error(`The clip has no frame ${index}`);
    // At or behind a frame already taken (a loop back, a step back): decode again
    // from a key frame. Later frames already fed are still on their way out.
    if (index <= this.lastTaken) this.restart(position);
    const frame = new Promise((resolve, reject) => this.waiters.set(index, { resolve, reject })).then(f => this.handOut(index, f));
    while (this.waiters.has(index)) {
      if (this.next < this.samples.length) this.decodeAhead();
      else if (this.flushed) throw new Error(`The decoder gave no frame ${index}`);
      else { await this.decoder.flush(); this.flushed = true; }
      await Promise.race([frame, new Promise(resolve => setTimeout(resolve, OUTPUT_WAIT_MS))]);
      if (this.error) throw this.error;
    }
    return frame;
  }
  /** Feed the decoder the next few samples (reordered B-frames need some ahead). */
  decodeAhead() {
    for (const end = Math.min(this.samples.length, this.next + AHEAD_SAMPLES); this.next < end; this.next++) {
      const sample = this.samples[this.next];
      this.decoder.decode(new EncodedVideoChunk({ type: sample.key ? 'key' : 'delta',
        timestamp: Math.round(sample.index * MICROSECONDS_PER_SECOND / this.fps),
        data: this.bytes.subarray(sample.offset, sample.offset + sample.size) }));
    }
  }
  take(index) {
    const frame = this.ready.get(index);
    this.ready.delete(index);
    return this.handOut(index, frame);
  }
  /** Hand frame `index` out; frames before it will not be asked for again before a restart. */
  handOut(index, frame) {
    this.lastTaken = index;
    for (const [other, held] of this.ready) if (other < index) { held.close(); this.ready.delete(other); }
    return frame;
  }
  /** Restart decoding at the last key frame at or before sample `position`. */
  restart(position) {
    let key = position;
    while (key > 0 && !this.samples[key].key) key--;
    this.decoder.close();
    for (const frame of this.ready.values()) frame.close();
    this.ready.clear();
    this.decoder = this.createDecoder();
    this.next = key; this.flushed = false; this.lastTaken = -1;
  }
  close() {
    for (const frame of this.ready.values()) frame.close();
    this.ready.clear();
    if (this.decoder.state !== 'closed') this.decoder.close();
  }
}
