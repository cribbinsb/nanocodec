// Sources whose every frame is known in advance (a sample clip or file decoded
// with clip-source.js, the built-in test scene), driven by the comparison rather
// than by the clock: the page asks for each tick a view codes, in order, and gets
// exactly that instant's picture. Nothing is skipped when the pipeline is slow;
// the clip's time just runs slower than real time (app.js says by how much), so a
// run's pictures, streams and numbers are those of a machine that keeps up.
// Live sources (the webcam) are sampled by capture.js instead.
//
// A sample clip ships in several renditions (clips.js `clipRendition`); each
// view codes the one for its own height, as the pre-encoded streams were made
// (tools/web/make-reference-streams.py `clip_file`), so views that straddle a
// rendition boundary have one exact source each (`ViewSources`), advanced on
// the same ticks.
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

/**
 * The exact sources of a comparison: one per distinct source among the views
 * (a clip's rendition for each view's height, else one for all), with which
 * view codes from which.
 */
export class ViewSources {
  /**
   * `urls[i]`: view i's source (a rendition URL, a file URL, or 'scene'); null
   * when the browser cannot decode one of them frame by frame (the page then
   * samples the playing video).
   */
  static async open(kind, urls, { fps, loops }) {
    const distinct = [...new Set(urls)];
    const sources = await Promise.all(distinct.map(url => ExactSource.open(kind, { url, fps, loops })));
    if (sources.some(source => !source)) { for (const source of sources) source?.close(); return null; }
    const lengths = new Set(sources.map(source => source.loopLength));
    if (lengths.size > 1) throw new Error(`The clip's renditions differ in length (${[...lengths].join(', ')} ticks)`);
    return new ViewSources(distinct.map((url, i) => ({ url, source: sources[i], views: urls.flatMap((u, view) => (u === url ? [view] : [])) })));
  }
  constructor(entries) { this.entries = entries; }
  /** Ticks in one loop (every source has the same). */
  get loopLength() { return this.entries[0].source.loopLength; }
  ended(tick) { return this.entries[0].source.ended(tick); }
  /**
   * The pictures of `tick` for the views that code it (`coding`): one shot
   * (ExactSource.picture, plus `views`, the view indices it is for) per source
   * a coding view reads; the scene is drawn at `width` x `height`.
   */
  async pictures(tick, coding, width, height) {
    const wanted = this.entries.map(entry => ({ ...entry, views: entry.views.filter(view => coding[view]) })).filter(entry => entry.views.length);
    return Promise.all(wanted.map(async ({ source, views }) => ({ ...(await source.picture(tick, width, height)), views })));
  }
  close() { for (const { source } of this.entries) source.close(); }
}
