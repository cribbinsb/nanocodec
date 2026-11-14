// Sources whose every frame is known in advance (a sample clip or file decoded
// with clip-source.js, the built-in test scene), driven by the comparison rather
// than by the clock: the page asks for each tick a view codes, in order, and gets
// exactly that instant's picture. Nothing is skipped when the pipeline is slow;
// the clip's time just runs slower than real time (app.js says by how much), so a
// run's pictures, streams and numbers are those of a machine that keeps up.
// Live sources (the webcam) are sampled by capture.js instead.
import { ClipSource } from './clip-source.js';
import { drawScene } from './scene.js';
import { snapshot } from './capture.js';
import { frameAtTick, loopTicks, tickMs } from './timeline.js';

export class ExactSource {
  /**
   * The exact source for `kind`: 'scene', or a clip/file at `url` (frames per second
   * `fps`, `loops`: whether it repeats); null when the browser cannot decode the file
   * frame by frame (the page then samples the playing video).
   */
  static async open(kind, { url, fps, loops } = {}) {
    if (kind === 'scene') return new ExactSource(null, Infinity);
    const clip = await ClipSource.open(url, fps);
    return clip && new ExactSource(clip, loops ? loopTicks(clip.frameCount, fps) : Infinity);
  }
  constructor(clip, loopLength) {
    this.clip = clip;
    /** Ticks in one loop of the clip (Infinity for the scene and files, which do not loop). */
    this.loopLength = loopLength;
    this.surface = document.createElement('canvas');
    this.surfaceContext = this.surface.getContext('2d', { willReadFrequently: true });
  }
  /** Whether `tick` is past the end of a file that does not loop. */
  ended(tick) { return Boolean(this.clip) && !Number.isFinite(this.loopLength) && frameAtTick(tick, this.clip.fps) >= this.clip.frameCount; }
  /**
   * The source at `tick`: `{ picture, element, width, height }` like capture.js
   * FrameSource.capture. The scene is drawn at `width` x `height` (the largest view);
   * a clip's frame comes at its own size.
   */
  async picture(tick, width, height) {
    if (!this.clip) {
      if (this.surface.width !== width || this.surface.height !== height) { this.surface.width = width; this.surface.height = height; }
      drawScene(this.surfaceContext, width, height, tickMs(tick) / 1000);
      return { picture: snapshot(this.surface), element: this.surface, width, height };
    }
    const local = Number.isFinite(this.loopLength) ? tick % this.loopLength : tick;
    const frame = await this.clip.frame(frameAtTick(local, this.clip.fps));
    return { picture: frame, element: frame, width: frame.displayWidth, height: frame.displayHeight };
  }
  close() { this.clip?.close(); }
}
