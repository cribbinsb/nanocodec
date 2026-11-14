// A lighter configuration for a live view whose encoder is too slow for its
// frame rate on this machine (app.js shows it under the session line). The
// codecs, not the page, set the rate on slower machines
// (docs/handoff/browser-demo.md, "Where the page's time goes"), so instead of
// leaving the visitor at "0.4x real time" the note says how long this machine
// takes to encode a frame of the view's settings and offers, as buttons, the
// heaviest lighter preset and the largest smaller picture size whose estimated
// encode time fits the frame interval. Nothing is applied without a click. Next to a pre-encoded
// stream the note only informs: that view's settings are the stream's, and the
// comparison is meant to stay as set up.
//
// The estimate scales the view's measured per-frame encode time (the encode
// worker's, with its source conversion) by PRESET_COST and by picture area
// (SIZE_COST_EXPONENT). It
// only picks the choices and is never shown: the delivered rate also depends on
// decoding, scoring, capture and the other view, which do not scale with this
// view's settings and, on few cores, share them (one CPU, 720p30 H.264 vs AV1:
// 640x360 was estimated to encode at 44 fps and the comparison then delivered
// 26 fps). The note shows only what was measured.
import { CODECS } from './codecs.js';
import { RESOLUTIONS } from './view-settings.js';

/**
 * Each preset's encode time relative to Default, per codec: the WASM module in
 * Node, 60 frames of mot17-mall-walk at 640x360 and at 1280x720, CBR at 300 kbps
 * per 640x360 area, two rounds each (2026-11-14; the machine was loaded, so the
 * ratios are good to about 15%). Spatial SVC presets are not estimated.
 */
export const PRESET_COST = {
  h264: { default: 1, fast: 0.53, faster: 0.49, fastest: 0.45 },
  hevc: { default: 1, fast: 0.61, faster: 0.47 },
  av1: { default: 1, fast: 0.71, faster: 0.45 },
  vvc: { default: 1, fast: 0.55 },
  vp9: { default: 1, fast: 0.7, faster: 0.5, fastest: 0.45 },
  av2: { default: 1 },
};
/**
 * Encode time scales with picture area to this power. Less than linear: the
 * view's measured time includes scaling the clip's rendition to the view and
 * per-frame work that does not shrink with the picture. Measured in the page on
 * one CPU (speed-hint.py, 2026-11-14): 1280x720 -> 854x480 (area / 2.25) cut
 * the encode time 1.27x for AV1 Default and 1.55x for H.264 Default.
 */
const SIZE_COST_EXPONENT = 0.5;
/** The share of the frame interval an estimated encode may take (decode, capture and the page need the rest). */
const FIT_SHARE = 0.85;
/** A view is slow when it encodes below this share of its frame rate (app.js KEEPING_UP is the same). */
const SLOW_SHARE = 0.95;
/** Coded frames a view needs before its encode time is trusted (start-up and warm-up excluded). */
export const MIN_FRAMES = 30;
/** How often the note is recomputed while frames arrive. */
const UPDATE_INTERVAL_MS = 1000;
const PRESET_LABELS = { default: 'Default', fast: 'Fast', faster: 'Faster', fastest: 'Fastest' };

const area = (width, height) => width * height;

/**
 * The lighter choices for a live view `config` that encodes at `encodeMs` per
 * frame, each `{ label, change }` (`change`: `{ preset }` or `{ width, height }`). An
 * option is the heaviest that fits the frame interval, else the lightest there is;
 * none when nothing lighter exists.
 */
export function lighterChoices({ codec, preset, width, height, fps }, encodeMs) {
  const costs = PRESET_COST[codec], budget = FIT_SHARE * 1000 / fps, options = [];
  const atPreset = name => encodeMs * costs[name] / costs[preset];
  const atSize = (w, h) => encodeMs * (area(w, h) / area(width, height)) ** SIZE_COST_EXPONENT;
  if (preset in costs) {
    const lighter = Object.keys(costs).filter(name => (name === 'default' || CODECS[codec][name]) && costs[name] < costs[preset])
      .sort((a, b) => costs[b] - costs[a]);
    const pick = lighter.find(name => atPreset(name) <= budget) ?? lighter.at(-1);
    if (pick) options.push({ label: `Lighter: ${PRESET_LABELS[pick]}`, change: { preset: pick } });
  }
  const smaller = RESOLUTIONS.filter(([w, h]) => area(w, h) < area(width, height)).sort((a, b) => area(...b) - area(...a));
  const size = smaller.find(([w, h]) => atSize(w, h) <= budget) ?? smaller.at(-1);
  if (size) {
    const [w, h] = size;
    options.push({ label: `Smaller: ${w}×${h}`, change: { width: w, height: h } });
  }
  return options;
}

/** The note: hidden while every live view keeps up. */
export class SpeedHint {
  /** `element`: where the note goes; `onApply(index, change)`: a button was clicked for view `index`. */
  constructor(element, onApply) {
    this.element = element;
    this.onApply = onApply;
    this.updatedAt = 0;
  }
  clear() { this.element.hidden = true; this.element.replaceChildren(); this.updatedAt = 0; }
  /**
   * Recompute (at most every UPDATE_INTERVAL_MS) from `views`, one per panel:
   * `{ letter, config, encodeMs, frames }` (`encodeMs` null for a pre-encoded view).
   * `exact`: the source is exact, so a slow view slows the clip down instead of skipping frames.
   */
  update(views, exact) {
    const now = performance.now();
    if (now - this.updatedAt < UPDATE_INTERVAL_MS) return;
    this.updatedAt = now;
    const slow = views
      .map((view, index) => ({ ...view, index, load: view.encodeMs * view.config.fps / 1000 }))
      .filter(view => view.config.source === 'live' && view.frames >= MIN_FRAMES && view.encodeMs && view.load > 1 / SLOW_SHARE)
      .sort((a, b) => b.load - a.load)[0];
    if (!slow) { this.element.hidden = true; return; }
    const besideStream = views.some(view => view.config.source === 'reference');
    const { codec, preset, width, height, fps } = slow.config;
    const options = lighterChoices(slow.config, slow.encodeMs);
    const presetLabel = PRESET_LABELS[preset] ?? preset;
    const consequence = exact ? 'so the clip runs slower than real time' : 'so frames are skipped';
    const text = document.createElement('span');
    text.textContent = `Version ${slow.letter} (${CODECS[codec].label} ${presetLabel}, ${width}×${height}) takes ${Math.round(slow.encodeMs)} ms`
      + ` to encode a frame on this machine, and ${fps} fps leaves ${Math.round(1000 / fps)} ms, ${consequence}.`;
    const parts = [text];
    if (besideStream) {
      parts.push(document.createTextNode(' The other view plays a pre-encoded stream, so the settings stay as they are.'));
    } else if (options.length) {
      parts.push(document.createTextNode(' Try: '));
      for (const option of options) {
        const button = document.createElement('button');
        button.className = 'speed-option';
        button.textContent = option.label;
        button.addEventListener('click', () => { this.clear(); this.onApply(slow.index, option.change); });
        parts.push(button);
      }
    }
    this.element.replaceChildren(...parts);
    this.element.hidden = false;
  }
}
