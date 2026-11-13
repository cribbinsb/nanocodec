// Where captured pictures come from and which tick of the shared source
// timeline (timeline.js) each one is. The page asks for the next tick it may
// capture and takes a picture of that instant (`snapshot`: a VideoFrame, which
// the capture worker converts for each view at its own size, capture-worker.js).
//
// Video sources (sample clips, local files) stage every presented frame:
// `requestVideoFrameCallback` names the frame's media time, and a VideoFrame
// made in that callback holds exactly that frame, so the tick of the pixels is
// known (a picture taken later could already show the next frame). A frame is
// captured once per tick it covers (timeline.js `ticksOfFrame`). Browsers
// without the callback or VideoFrame take the picture at capture time and the
// media time from `currentTime` (close, not exact), and cannot step frames
// (`canStep`).
// Sample clips loop: each loop continues the tick count, so ticks only grow.
// The built-in scene is drawn at the tick's own time; the webcam has no
// timeline of its own, so its ticks follow the wall clock.
import { drawScene } from './scene.js';
import { SOURCE_FPS, frameAtTick, loopTicks, tickAt, tickMs, ticksOfFrame } from './timeline.js';

// A media time this far below the last one means the clip looped (ms).
const LOOP_BACKSTEP_MS = 500;
// Seek to the middle of a frame's display interval, so rounding cannot land on its neighbour.
const FRAME_MIDPOINT = 0.5;
// A step whose frame is not presented within this time failed (ms): the server
// cannot seek (it ignores HTTP range requests) or the video stalled.
const SEEK_TIMEOUT_MS = 5000;
// After 'seeked', how long a frame callback may take before the page reads the
// element's picture itself (ms).
const SEEKED_GRACE_MS = 100;
const SEEK_FAILED = 'Could not step: the video did not show the next frame (the server may not support seeking).';

/**
 * A transferable picture of `source` (video element or canvas) for the capture
 * worker: a VideoFrame (WebCodecs), else a promised ImageBitmap.
 */
export function snapshot(source) {
  return typeof VideoFrame === 'function' ? new VideoFrame(source, { timestamp: 0 }) : createImageBitmap(source);
}

/**
 * `snapshot` of a video element, or null while it has no frame to give: just
 * after a source switch, a webcam start, a seek or a loop wrap the element can
 * briefly hold none (VideoFrame then throws InvalidStateError). The caller skips
 * that tick; any other error is a real failure and is thrown.
 */
export function videoSnapshot(video) {
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) return null;
  try {
    return snapshot(video);
  } catch (error) {
    if (error.name === 'InvalidStateError') return null;
    throw error;
  }
}

export class FrameSource {
  /** `onError(error)`: staging a presented frame failed for a reason other than a missing frame. */
  constructor(video, onError) {
    this.video = video; this.onError = onError;
    // The test scene is drawn here, at the largest view size (a software canvas,
    // so its pixels do not depend on the GPU).
    this.surface = document.createElement('canvas');
    this.surfaceContext = this.surface.getContext('2d', { willReadFrequently: true });
    this.canStage = 'requestVideoFrameCallback' in HTMLVideoElement.prototype && typeof VideoFrame === 'function';
    this.watching = false;
    // The staged frame: a VideoFrame of the last presented video frame.
    this.staged = null;
    this.reset('scene');
  }
  /**
   * Start a new timeline for `kind` ('clip', 'file', 'camera' or 'scene'); `fps`
   * is a clip's own frame rate (others are taken as SOURCE_FPS).
   */
  reset(kind, fps = SOURCE_FPS) {
    this.kind = kind; this.fps = fps;
    this.presented = null; this.loop = 0; this.lastMediaMs = -Infinity;
    this.waiting = null;
    this.started = performance.now();
    this.staged?.close(); this.staged = null;
    if (this.staging && !this.watching) this.watch();
  }
  /** Whether the source is a video with its own media timeline. */
  get timed() { return this.kind === 'clip' || this.kind === 'file'; }
  /** Whether presented frames are staged (timed sources where the browser names each frame). */
  get staging() { return this.timed && this.canStage; }
  /** Whether Step can show a chosen tick: the scene draws any; a video needs staged frames. */
  get canStep() { return this.kind === 'scene' || this.staging; }
  /** Ticks in one loop of the clip (Infinity for sources that do not loop). */
  get loopLength() {
    return this.kind === 'clip' && this.video.duration ? loopTicks(Math.round(this.video.duration * this.fps), this.fps) : Infinity;
  }
  watch() {
    this.watching = true;
    const onFrame = (_, metadata) => {
      if (this.staging && this.video.videoWidth) {
        try {
          const frame = videoSnapshot(this.video);
          if (frame) {
            this.staged?.close();
            this.staged = frame;
            this.stage(metadata.mediaTime * 1000);
          }
        } catch (error) { this.onError(error); }
      }
      this.video.requestVideoFrameCallback(onFrame);
    };
    this.video.requestVideoFrameCallback(onFrame);
  }
  /** Note which source frame (of which loop) the video presents. */
  stage(mediaMs) {
    if (mediaMs < this.lastMediaMs - LOOP_BACKSTEP_MS && this.kind === 'clip') ++this.loop;
    this.lastMediaMs = mediaMs;
    this.presented = { frame: Math.round(mediaMs * this.fps / 1000), loop: this.loop };
    if (this.waiting) this.checkStep();
  }
  /** The first tick after `after` that can be captured now, or null if there is none yet. */
  nextTick(after) {
    if (!this.timed) return Math.max(tickAt(performance.now() - this.started), after + 1);
    if (!this.staging) this.stage(this.video.currentTime * 1000);
    if (!this.presented) return null;
    const ticks = ticksOfFrame(this.presented.frame, this.fps);
    if (!ticks) return null;
    const offset = this.loopStart(this.presented.loop);
    const tick = Math.max(ticks[0] + offset, after + 1);
    return tick <= ticks[1] + offset ? tick : null;
  }
  /**
   * The source at `tick`: `{ picture, element, width, height }`, the picture for
   * the capture worker (`snapshot`), the element it shows (for the preview) and
   * its size. The scene is drawn for `tick` at `width` x `height`, the largest
   * view size; a video is taken at its own size. Null while the video has no
   * frame to give (`videoSnapshot`): the caller skips the tick.
   */
  capture(tick, width, height) {
    if (this.kind === 'scene') {
      if (this.surface.width !== width || this.surface.height !== height) { this.surface.width = width; this.surface.height = height; }
      drawScene(this.surfaceContext, width, height, tickMs(tick) / 1000);
      return { picture: snapshot(this.surface), element: this.surface, width, height };
    }
    const { video } = this;
    const picture = this.staging ? this.staged?.clone() : videoSnapshot(video);
    if (!picture) return null;
    return { picture, element: video, width: video.videoWidth, height: video.videoHeight };
  }
  /**
   * Show the frame of `tick` on a paused source and resolve to the tick shown:
   * a video seeks to that frame and waits until the browser presents it (the
   * frame callback names it; a drawImage right after 'seeked' may still copy the
   * previous frame), the scene draws any tick. If the browser presents a later
   * frame instead, the first tick of that frame is shown.
   */
  async advanceTo(tick) {
    if (!this.timed) return tick;
    const length = this.loopLength, loop = Number.isFinite(length) ? Math.floor(tick / length) : 0;
    const target = { frame: frameAtTick(tick - this.loopStart(loop), this.fps), loop };
    this.loop = loop; this.lastMediaMs = -Infinity;
    const reached = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => { this.waiting = null; reject(new Error(SEEK_FAILED)); }, SEEK_TIMEOUT_MS);
      this.waiting = { target, resolve: () => { clearTimeout(timeout); resolve(); } };
    });
    // A seek to the frame the paused video already shows (it may have run on past the
    // pause instant) presents no new frame, so no frame callback comes: once the seek
    // is done and none came, the element's picture is the target frame.
    this.video.addEventListener('seeked', () => setTimeout(() => this.stageSeeked(target), SEEKED_GRACE_MS), { once: true });
    this.video.currentTime = (target.frame + FRAME_MIDPOINT) / this.fps;
    await reached;
    return ticksOfFrame(this.presented.frame, this.fps)[0] + this.loopStart(this.presented.loop);
  }
  /** Stage the seeked-to frame if a step still waits for it (see advanceTo). */
  stageSeeked(target) {
    if (this.waiting?.target !== target) return;
    const frame = videoSnapshot(this.video);
    if (!frame) return;
    this.staged?.close();
    this.staged = frame;
    this.stage(target.frame * 1000 / this.fps);
  }
  /** The first tick of loop `loop` of the clip. */
  loopStart(loop) { return Number.isFinite(this.loopLength) ? loop * this.loopLength : 0; }
  /** Resolve a waiting step once the frame just staged is (or is past) its frame. */
  checkStep() {
    const { target, resolve } = this.waiting, { frame, loop } = this.presented;
    if (loop < target.loop || (loop === target.loop && frame < target.frame) || !ticksOfFrame(frame, this.fps)) return;
    this.waiting = null;
    resolve();
  }
}
