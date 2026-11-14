// The analysis mode of the comparison page: coding-tool overlays drawn over
// each view, a statistics panel per view, larger layouts, and zoom and pan
// shared by both views (analysis.js reads and draws the decoder's records).
//
// Off by default; the page then looks and decodes as without it (the decode
// workers do not ask their decoders for records). On:
// - each view gets a canvas layer over its picture, redrawn per frame and per
//   zoom change, in picture coordinates so vectors stay crisp at any zoom;
//   the picture itself is scaled by a CSS transform, unsmoothed past 1:1;
// - zoom (wheel, pinch, buttons) and pan (drag) are one state in normalised
//   picture coordinates, applied to both views, so they show the same region;
// - hovering a block (or tapping it) shows its record;
// - the layout can show both views stacked at full width, or one view large;
//   each view can go fullscreen;
// - the magnifier (loupe.js) draws the overlay too (`drawInto`);
// - the PSNR map (error-map.js) colours each block of a fixed grid by its PSNR
//   against the source the view coded, for one plane at a time, at 16x16 luma
//   blocks or, zoomed in to FINE_GRID_ZOOM and beyond, 8x8 (fixed blocks rather
//   than each codec's partition, so the two views compare block for block); the
//   statistics show the frame's PSNR of each plane and the hover record the block's.
import { ANALYSIS_CODECS, FIELD, HEAT_LAYERS, LAYERS, LAYER_LABELS, PRED_COLOURS, RunningSummary, describeBlock, differenceSummary, drawOverlay, recordsAt, summarise } from './analysis.js';
import { PLANES, PSNR_SCALE, YUV_WEIGHTS, describeError, errorPixels, formatPsnr, psnrColour, yuvPsnr } from './error-map.js';
import { tickMs } from './timeline.js';

const MIN_ZOOM = 1, MAX_ZOOM = 32;
// One wheel notch (deltaY 100) zooms by this factor.
const WHEEL_ZOOM_PER_100 = 1.25;
const BUTTON_ZOOM = 2;
// A pointer that moves less than this between down and up is a tap.
const TAP_SLOP_PX = 6;
// The statistics text is rewritten at most this often (as panel.js metrics).
const STATS_INTERVAL_MS = 250;
const PRECISIONS = ['integer', 'half', 'quarter', 'eighth'];
const LAYOUTS = { side: 'Side by side', stacked: 'Stacked', 'expand-A': 'A large', 'expand-B': 'B large' };
const DEFAULT_LAYERS = ['blocks', 'vectors'];
// The PSNR map shows 8x8 blocks from this shared zoom on (16x16 below it).
const FINE_GRID_ZOOM = 4;
// Colour stops of the PSNR map's scale bar.
const SCALE_STOPS = 5;
const escapeHtml = text => String(text).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
// PSNR-YUV's label in the statistics, with its weighting.
const YUV_LABEL = `YUV (${YUV_WEIGHTS.join(':')})`;
const psnrText = psnr => (psnr ? [...PLANES, YUV_LABEL].map(name => `${name} ${formatPsnr(psnr[name])}`).join(' · ') : '—');
/** A PSNR map's frame PSNR of each plane and PSNR-YUV, by name. */
const framePsnr = errors => ({ ...Object.fromEntries(PLANES.map((plane, i) => [plane, errors.frame[i]])), [YUV_LABEL]: yuvPsnr(errors.frame) });
// Phone widths stack the views when analysis is on (styles.css uses the same width).
const NARROW_QUERY = '(max-width: 700px)';
const UNSUPPORTED = 'Analysis is not available for this codec.';

const percent = value => `${(100 * value).toFixed(value < 0.1 ? 1 : 0)}%`;
const shareText = (map, order) => Object.entries(map ?? {})
  .sort(([a], [b]) => (order ? order.indexOf(a) - order.indexOf(b) : (parseInt(b) || 0) - (parseInt(a) || 0)))
  .filter(([, v]) => v >= 0.001).map(([k, v]) => `${k} ${percent(v)}`).join(' · ') || '—';

/** The PSNR map's colour scale as a CSS gradient, low to high, opaque. */
function scaleGradient() {
  const stops = Array.from({ length: SCALE_STOPS }, (_, i) => {
    const [r, g, b] = psnrColour(PSNR_SCALE.low + (PSNR_SCALE.high - PSNR_SCALE.low) * i / (SCALE_STOPS - 1));
    return `rgb(${r} ${g} ${b})`;
  });
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}

/** The page-wide analysis state and controls; one AnalysisView per panel. */
export class AnalysisController {
  /** `element`: the #analysis section; `onEnabled(on)`: tell the decoders. `loupeActive()`: the magnifier owns pointer input. */
  constructor(element, versions, panels, { onEnabled, loupeActive }) {
    this.element = element; this.versions = versions; this.onEnabled = onEnabled; this.loupeActive = loupeActive;
    this.enabled = false; this.layers = new Set(DEFAULT_LAYERS); this.layout = 'side';
    // The PSNR map's plane: 0 Y, 1 Cb, 2 Cr (error-map.js PLANES).
    this.plane = 0;
    this.zoom = { scale: MIN_ZOOM, u: 0.5, v: 0.5 };
    element.innerHTML = `<div class="analysis-bar"><span class="control-label">ANALYSIS</span><button class="analysis-toggle" aria-pressed="false">Analysis off</button>
      <div class="segmented analysis-options analysis-layers" role="group" aria-label="Overlay layers">${LAYERS.map(layer =>
        `<button data-layer="${layer}" aria-pressed="false">${LAYER_LABELS[layer]}</button>`).join('')}</div>
      <div class="segmented analysis-options" role="group" aria-label="Layout">${Object.entries(LAYOUTS).map(([value, label]) =>
        `<button data-layout="${value}" aria-pressed="false">${label}</button>`).join('')}</div>
      <div class="analysis-options analysis-zoom"><button data-zoom="out" aria-label="Zoom out">−</button><output>1×</output><button data-zoom="in" aria-label="Zoom in">+</button><button data-zoom="reset">Fit</button></div>
      <span class="analysis-hint analysis-options">Scroll or pinch to zoom · drag to pan · hover or tap a block</span>
      <span class="analysis-legend analysis-options">${['intra', 'inter', 'skip', 'merge'].map((name, i) => `<i style="--c:${PRED_COLOURS[i]}"></i>${name}`).join(' ')}</span>
      <div class="analysis-options analysis-psnr" hidden><div class="segmented" role="group" aria-label="PSNR map plane">${PLANES.map((plane, i) =>
        `<button data-plane="${i}" aria-pressed="false">${plane}</button>`).join('')}</div>
        <span class="analysis-psnr-scale">PSNR vs source (dB): ${PSNR_SCALE.low} or less <i style="--scale:${scaleGradient()}"></i> ${PSNR_SCALE.high} or more · <output></output></span></div></div>`;
    this.toggle = element.querySelector('.analysis-toggle');
    this.toggle.addEventListener('click', () => this.setEnabled(!this.enabled));
    for (const button of element.querySelectorAll('[data-layer]')) button.addEventListener('click', () => this.toggleLayer(button.dataset.layer));
    for (const button of element.querySelectorAll('[data-layout]')) button.addEventListener('click', () => this.setLayout(button.dataset.layout));
    for (const button of element.querySelectorAll('[data-plane]')) button.addEventListener('click', () => { this.plane = +button.dataset.plane; this.showControls(); this.redraw(); });
    element.querySelector('[data-zoom="in"]').addEventListener('click', () => this.zoomBy(BUTTON_ZOOM));
    element.querySelector('[data-zoom="out"]').addEventListener('click', () => this.zoomBy(1 / BUTTON_ZOOM));
    element.querySelector('[data-zoom="reset"]').addEventListener('click', () => this.setZoom({ scale: MIN_ZOOM, u: 0.5, v: 0.5 }));
    this.zoomLabel = element.querySelector('.analysis-zoom output');
    this.views = panels.map(panel => new AnalysisView(panel, this));
    // When both views are analysed: one line comparing them, under the views.
    this.difference = document.createElement('p'); this.difference.className = 'analysis-difference'; this.difference.hidden = true;
    versions.after(this.difference);
    this.showControls();
  }
  setEnabled(on) {
    this.enabled = on;
    this.toggle.setAttribute('aria-pressed', String(on));
    this.toggle.textContent = on ? 'Analysis on' : 'Analysis off';
    this.element.classList.toggle('active', on);
    if (on && this.layout === 'side' && matchMedia(NARROW_QUERY).matches) this.layout = 'stacked';
    if (!on) this.zoom = { scale: MIN_ZOOM, u: 0.5, v: 0.5 };
    for (const view of this.views) view.setEnabled(on);
    this.showControls();
    this.onEnabled(on);
  }
  toggleLayer(layer) {
    if (this.layers.has(layer)) this.layers.delete(layer);
    else {
      if (HEAT_LAYERS.includes(layer)) for (const heat of HEAT_LAYERS) this.layers.delete(heat);
      this.layers.add(layer);
    }
    this.showControls(); this.redraw();
  }
  setLayout(layout) { this.layout = layout; this.showControls(); this.redraw(); }
  /** Zoom by `factor` about the normalised point (u, v) (default: the centre of the shown region). */
  zoomBy(factor, at = null) {
    const { scale, u, v } = this.zoom;
    const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, scale * factor));
    // Keep the picture point under `at` where it is on screen.
    const point = at ?? { u, v };
    const ratio = scale / next;
    this.setZoom({ scale: next, u: point.u + (u - point.u) * ratio, v: point.v + (v - point.v) * ratio });
  }
  /** Pan by a normalised picture offset. */
  panBy(du, dv) { this.setZoom({ ...this.zoom, u: this.zoom.u + du, v: this.zoom.v + dv }); }
  setZoom(zoom) {
    // The shown region stays inside the picture.
    const half = 0.5 / zoom.scale, clamp = value => Math.max(half, Math.min(1 - half, value));
    this.zoom = { scale: zoom.scale, u: clamp(zoom.u), v: clamp(zoom.v) };
    this.zoomLabel.textContent = `${this.zoom.scale.toFixed(this.zoom.scale < 10 ? 1 : 0).replace(/\.0$/, '')}×`;
    this.showControls();
    this.redraw();
  }
  /** The analysis state for the debug report (debug-report.js). */
  debugState() {
    return { enabled: this.enabled, layers: [...this.layers], layout: this.layout, zoom: { ...this.zoom }, plane: PLANES[this.plane], fineGrid: this.fineGrid };
  }
  /** Whether the PSNR map shows its fine grid (8x8) rather than the coarse one (16x16). */
  get fineGrid() { return this.zoom.scale >= FINE_GRID_ZOOM; }
  redraw() { for (const view of this.views) view.place(); }
  /** The difference line, when both views have statistics. */
  showDifference() {
    const means = this.views.map(view => view.state !== 'unsupported' && view.running.mean());
    this.difference.hidden = !this.enabled || !means.every(Boolean);
    if (this.difference.hidden) return;
    const codecs = this.views.map(view => view.panel.current?.codec);
    this.difference.textContent = differenceSummary(means[0], means[1], codecs, this.views.map(view => view.panel.letter));
  }
  showControls() {
    for (const button of this.element.querySelectorAll('[data-layer]')) button.setAttribute('aria-pressed', String(this.layers.has(button.dataset.layer)));
    for (const button of this.element.querySelectorAll('[data-layout]')) {
      button.setAttribute('aria-pressed', String(this.layout === button.dataset.layout));
      button.classList.toggle('selected', this.layout === button.dataset.layout);
    }
    for (const layout of Object.keys(LAYOUTS)) this.versions.classList.toggle(`layout-${layout}`, this.enabled && this.layout === layout);
    this.element.querySelector('.analysis-psnr').hidden = !this.layers.has('psnr');
    for (const button of this.element.querySelectorAll('[data-plane]')) button.setAttribute('aria-pressed', String(+button.dataset.plane === this.plane));
    const block = this.fineGrid ? '8×8' : '16×16';
    this.element.querySelector('.analysis-psnr-scale output').textContent = `${PLANES[this.plane]}, ${block} luma blocks (8×8 from ${FINE_GRID_ZOOM}× zoom)`;
  }
}

/** One view's overlay layer, statistics, hover record and zoom. */
class AnalysisView {
  constructor(panel, controller) {
    this.panel = panel; this.controller = controller;
    panel.analysis = this;
    this.records = null; this.state = 'off'; this.running = new RunningSummary(); this.summary = null;
    // The latest PSNR map (error-map.js `errorMap`), its image at the drawn grid
    // and plane (made when first drawn), and the mean PSNR of each plane.
    this.errors = null; this.errorImage = null; this.errorImageKey = ''; this.psnr = new RunningSummary();
    this.drawRequested = false; this.statsShownAt = 0; this.statsTimer = 0; this.hover = null; this.pinned = false;
    this.built = false;
  }
  /** The layer, tip, fullscreen button and statistics, made when analysis is first switched on (the page is unchanged until then). */
  build() {
    if (this.built) return;
    this.built = true;
    const panel = this.panel;
    this.layer = document.createElement('canvas'); this.layer.className = 'analysis-layer';
    this.tip = document.createElement('div'); this.tip.className = 'analysis-tip'; this.tip.hidden = true;
    this.fullscreen = document.createElement('button');
    this.fullscreen.className = 'analysis-fullscreen'; this.fullscreen.textContent = '⛶';
    this.fullscreen.setAttribute('aria-label', `Version ${panel.letter} fullscreen`);
    this.fullscreen.addEventListener('click', () => this.toggleFullscreen());
    panel.viewport.append(this.layer, this.tip, this.fullscreen);
    this.stats = document.createElement('div'); this.stats.className = 'analysis-stats';
    panel.viewport.after(this.stats);
    new ResizeObserver(() => this.place()).observe(panel.viewport);
    this.listen();
  }
  get enabled() { return this.controller.enabled; }
  setEnabled(on) {
    if (on) this.build();
    if (!this.built) return;
    for (const element of [this.layer, this.stats, this.fullscreen]) element.hidden = !on;
    this.panel.viewport.classList.toggle('analysing', on);
    if (!on) { this.records = null; this.errors = null; this.hideTip(); }
    this.running.reset(); this.psnr.reset(); this.summary = null; this.state = on ? 'waiting' : 'off';
    this.place(); this.showStats();
  }
  /** A new comparison started: forget the previous stream's statistics. */
  reset() {
    this.running.reset(); this.psnr.reset(); this.summary = null; this.records = null; this.errors = null;
    this.state = this.enabled ? 'waiting' : 'off';
    if (this.built) this.showStats();
  }
  /**
   * A decoded frame was shown for source tick `tick`: its records
   * (`result.analysis`, an ArrayBuffer of packed records) and PSNR map
   * (`result.errorMap`) if any.
   */
  update(result, tick) {
    if (!this.enabled) return;
    this.tick = tick;
    this.state = result.analysisState ?? (ANALYSIS_CODECS.has(this.panel.current?.codec) ? 'waiting' : 'unsupported');
    if (result.analysis) {
      this.records = new Int16Array(result.analysis);
      this.summary = summarise(this.records, this.panel.current?.codec);
      this.running.add(this.summary);
    } else if (!result.repeat) {
      this.records = null;
    }
    if (result.errorMap) {
      this.errors = result.errorMap; this.errorImageKey = '';
      this.psnr.add({ psnr: framePsnr(this.errors) });
    } else if (!result.repeat) {
      this.errors = null;
    }
    // A view that fell back from WebGL has a new canvas: zoom it too.
    if (this.placedCanvas !== this.panel.canvas) this.place();
    this.requestDraw();
    this.scheduleStats();
    if (this.hover && this.pinned) this.showTip(this.hover);
  }
  /** Picture size in samples: the canvas holds the decoded picture at its own size. */
  pictureSize() { return { width: this.panel.canvas.width, height: this.panel.canvas.height }; }
  /** Viewport pixels per picture pixel, and the picture origin's viewport position, at the shared zoom. */
  mapping() {
    const box = this.panel.viewport.getBoundingClientRect(), { width, height } = this.pictureSize();
    if (!width || !height || !box.width) return null;
    const fit = Math.min(box.width / width, box.height / height);
    const fx = (box.width - width * fit) / 2, fy = (box.height - height * fit) / 2;
    const { scale, u, v } = this.enabled ? this.controller.zoom : { scale: 1, u: 0.5, v: 0.5 };
    const tx = box.width / 2 - (fx + u * width * fit) * scale, ty = box.height / 2 - (fy + v * height * fit) * scale;
    return { box, fit, scale, tx, ty, s: fit * scale, ox: fx * scale + tx, oy: fy * scale + ty, width, height };
  }
  /** Apply the zoom to the picture and redraw the layer (zoom, layout or size changed). */
  place() {
    if (!this.built) return;
    const canvas = this.panel.canvas, map = this.enabled ? this.mapping() : null;
    this.placedCanvas = canvas;
    if (!map || map.scale === 1) {
      canvas.style.transform = ''; canvas.style.transformOrigin = '';
    } else {
      canvas.style.transformOrigin = '0 0';
      canvas.style.transform = `translate(${map.tx}px, ${map.ty}px) scale(${map.scale})`;
    }
    // Unsmoothed pixels once a picture pixel covers more than one device pixel.
    canvas.style.imageRendering = map && map.s * devicePixelRatio > 1 ? 'pixelated' : '';
    this.requestDraw();
  }
  requestDraw() {
    if (this.drawRequested) return;
    this.drawRequested = true;
    requestAnimationFrame(() => { this.drawRequested = false; this.draw(); });
  }
  draw() {
    if (!this.built) return;
    const context = this.layer.getContext('2d'), map = this.enabled ? this.mapping() : null;
    if (!map) return;
    const ratio = devicePixelRatio, w = Math.round(map.box.width * ratio), h = Math.round(map.box.height * ratio);
    if (this.layer.width !== w || this.layer.height !== h) { this.layer.width = w; this.layer.height = h; }
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, w, h);
    if (!this.records) return;
    const scale = map.s * ratio;
    context.setTransform(scale, 0, 0, scale, map.ox * ratio, map.oy * ratio);
    const clip = { x: -map.ox / map.s, y: -map.oy / map.s, width: map.box.width / map.s, height: map.box.height / map.s };
    this.drawErrors(context);
    drawOverlay(context, this.records, { codec: this.panel.current?.codec, layers: this.controller.layers, scale, clip });
    if (this.hover) this.outline(context, this.hover, scale);
  }
  /** The magnifier's pane: draw the overlay for normalised `region` into a `width` x `height` context. */
  drawInto(context, region, width, height) {
    if (!this.enabled || !this.records) return;
    const picture = this.pictureSize(), scale = width / (region.width * picture.width);
    context.save();
    context.setTransform(scale, 0, 0, scale, -region.x * picture.width * scale, -region.y * picture.height * scale);
    const clip = { x: region.x * picture.width, y: region.y * picture.height, width: region.width * picture.width, height: height / scale };
    this.drawErrors(context);
    drawOverlay(context, this.records, { codec: this.panel.current?.codec, layers: this.controller.layers, scale, clip });
    context.restore();
  }
  /**
   * The PSNR map under the outlines, when its layer is on: one image pixel per
   * block, scaled unsmoothed onto the blocks (in picture coordinates, through
   * the context's transform) and clipped to the picture, as edge blocks overhang it.
   */
  drawErrors(context) {
    if (!this.errors || !this.controller.layers.has('psnr')) return;
    const { fineGrid: fine, plane } = this.controller, key = `${fine}:${plane}`;
    if (this.errorImageKey !== key) {
      const { rgba, columns, rows } = errorPixels(this.errors, fine, plane);
      this.errorImage ??= document.createElement('canvas');
      this.errorImage.width = columns; this.errorImage.height = rows;
      this.errorImage.getContext('2d').putImageData(new ImageData(rgba, columns, rows), 0, 0);
      this.errorImageKey = key;
    }
    // The map is at its source's size, which a smaller spatial layer is scaled to (decode-worker.js).
    const picture = this.pictureSize(), grid = fine ? this.errors.fine : this.errors.coarse;
    const sx = picture.width / this.errors.width, sy = picture.height / this.errors.height;
    context.save();
    context.beginPath(); context.rect(0, 0, picture.width, picture.height); context.clip();
    context.imageSmoothingEnabled = false;
    context.drawImage(this.errorImage, 0, 0, grid.columns * grid.block * sx, grid.rows * grid.block * sy);
    context.restore();
  }
  outline(context, found, scale) {
    const r = found.coding;
    if (r < 0) return;
    context.save();
    context.lineWidth = 2 / scale; context.strokeStyle = '#fff';
    const f = field => this.records[r + FIELD[field]];
    context.strokeRect(f('x'), f('y'), f('w'), f('h'));
    context.restore();
  }
  /** Picture coordinates of a pointer event, or null outside the picture. */
  pictureAt(event) {
    const map = this.mapping();
    if (!map) return null;
    const x = (event.clientX - map.box.left - map.ox) / map.s, y = (event.clientY - map.box.top - map.oy) / map.s;
    return x >= 0 && y >= 0 && x < map.width && y < map.height ? { x, y, map } : null;
  }
  listen() {
    const area = this.panel.viewport;
    let drag = null;
    const active = () => this.enabled && !this.controller.loupeActive();
    area.addEventListener('wheel', event => {
      if (!active()) return;
      event.preventDefault();
      const at = this.pictureAt(event);
      this.controller.zoomBy(WHEEL_ZOOM_PER_100 ** (-event.deltaY / 100), at && { u: at.x / at.map.width, v: at.y / at.map.height });
    }, { passive: false });
    const pointers = new Map();
    area.addEventListener('pointerdown', event => {
      if (!active() || event.target === this.fullscreen) return;
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      drag = { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, pinch: null };
      area.setPointerCapture(event.pointerId);
    });
    area.addEventListener('pointermove', event => {
      if (!active()) return;
      if (pointers.has(event.pointerId)) {
        pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        const map = this.mapping();
        if (pointers.size === 2 && map) {
          // Pinch: zoom by the change of the two pointers' distance.
          const [a, b] = [...pointers.values()], distance = Math.hypot(a.x - b.x, a.y - b.y);
          if (drag.pinch) this.controller.zoomBy(distance / drag.pinch);
          drag.pinch = distance;
        } else if (drag && map) {
          this.controller.panBy(-(event.clientX - drag.x) / (map.s * map.width), -(event.clientY - drag.y) / (map.s * map.height));
          drag.x = event.clientX; drag.y = event.clientY;
        }
        return;
      }
      if (!this.pinned) this.hoverAt(event);
    });
    const end = event => {
      if (!pointers.delete(event.pointerId) || !drag) return;
      const tap = Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < TAP_SLOP_PX && !drag.pinch;
      if (pointers.size === 0) drag = null;
      if (tap) { this.pinned = !this.pinned || event.pointerType !== 'mouse'; this.hoverAt(event); }
    };
    area.addEventListener('pointerup', end);
    area.addEventListener('pointercancel', end);
    area.addEventListener('pointerleave', () => { if (!this.pinned) this.hideTip(); });
  }
  hoverAt(event) {
    const at = this.records && this.pictureAt(event);
    if (!at) { this.hideTip(); return; }
    this.hover = recordsAt(this.records, at.x, at.y);
    this.hoverPoint = { x: at.x, y: at.y };
    this.tipAt = { x: event.clientX - at.map.box.left, y: event.clientY - at.map.box.top };
    this.showTip(this.hover);
    this.requestDraw();
  }
  showTip(found) {
    const lines = describeBlock(this.records, found, this.panel.current?.codec);
    const error = this.errors && describeError(this.errors, this.hoverPoint.x, this.hoverPoint.y, this.controller.fineGrid, this.pictureSize());
    if (error) lines.push(error);
    if (!lines.length) { this.hideTip(); return; }
    this.tip.textContent = lines.join('\n');
    this.tip.hidden = false;
    const box = this.panel.viewport.getBoundingClientRect();
    const left = Math.min(this.tipAt.x + 12, box.width - this.tip.offsetWidth - 4), top = Math.min(this.tipAt.y + 12, box.height - this.tip.offsetHeight - 4);
    Object.assign(this.tip.style, { left: `${Math.max(4, left)}px`, top: `${Math.max(4, top)}px` });
  }
  hideTip() { this.tip.hidden = true; this.hover = null; this.pinned = false; this.requestDraw(); }
  toggleFullscreen() {
    if (document.fullscreenElement === this.panel.viewport) document.exitFullscreen();
    else this.panel.viewport.requestFullscreen?.();
  }
  scheduleStats() {
    if (this.statsTimer) return;
    const wait = this.statsShownAt + STATS_INTERVAL_MS - performance.now();
    if (wait <= 0) this.showStats();
    else this.statsTimer = setTimeout(() => { this.statsTimer = 0; this.showStats(); }, wait);
  }
  showStats() {
    this.statsShownAt = performance.now();
    this.controller.showDifference();
    if (!this.enabled) { this.stats.textContent = ''; return; }
    if (this.state === 'unsupported') { this.stats.textContent = UNSUPPORTED; return; }
    const now = this.summary, mean = this.running.mean(), psnrMean = this.psnr.mean();
    if (!now) { this.stats.textContent = 'Analysis: loading the analysis decoder…'; return; }
    const row = (label, value, average) => `<tr><th>${label}</th><td>${value}</td><td>${average}</td></tr>`;
    const qp = s => s?.qp == null ? '—' : `${s.qp.toFixed(1)}${s.qpRange ? ` [${s.qpRange[0]}–${s.qpRange[1]}]` : ''}`;
    this.stats.innerHTML = `<p class="analysis-note">Analysis on: decode timing is not representative (the decoder records every block).</p><table><thead><tr><th>By area</th><td>This frame</td><td>Mean of ${this.running.frames}</td></tr></thead><tbody>${[
      row(`PSNR vs source (dB) · ${this.errors ? `${this.errors.width}×${this.errors.height}` : '—'}`, psnrText(this.errors && framePsnr(this.errors)),
        psnrText(psnrMean?.psnr)),
      row('Source frame compared', Number.isFinite(this.tick) ? `tick ${this.tick} · ${(tickMs(this.tick) / 1000).toFixed(3)} s` : '—', ''),
      row('Source rendition', this.panel.current?.rendition ?? 'not a sample clip', ''),
      row('Prediction', shareText(now.pred, ['intra', 'inter', 'merge', 'skip']), shareText(mean.pred, ['intra', 'inter', 'merge', 'skip'])),
      row('Block size', shareText(now.cbSize), shareText(mean.cbSize)),
      row('Partition', shareText(now.shape), shareText(mean.shape)),
      row('Transform', shareText(now.tbSize), shareText(mean.tbSize)),
      row('Coded residual', percent(now.coded), percent(mean.coded ?? 0)),
      row('MV precision', shareText(now.mvPrecision, PRECISIONS), shareText(mean.mvPrecision, PRECISIONS)),
      row('Zero MV', now.zeroMv == null ? '—' : percent(now.zeroMv), mean.zeroMv == null ? '—' : percent(mean.zeroMv)),
      row('Intra modes', shareText(now.intraClass, ['planar', 'DC', 'angular']), shareText(mean.intraClass, ['planar', 'DC', 'angular'])),
      row('QP', qp(now), mean.qp == null ? '—' : mean.qp.toFixed(1)),
      row('QP changes (blocks)', String(now.qpDeltas), mean.qpDeltas?.toFixed(0) ?? '—'),
      row('Bits / pixel', now.bitsPerPixel.toFixed(3), (mean.bitsPerPixel ?? 0).toFixed(3)),
    ].join('')}</tbody></table><p class="analysis-note">PSNR: per plane against the source frame this view coded, at its own size, on the source timeline's tick shown (30 ticks a second; views showing the same tick were scored against the same instant); PSNR-YUV weights the planes ${YUV_WEIGHTS.join(':')}, as the PSNR-YUV above.${
      this.panel.current?.entry?.sourceNote ? ` Source note: ${escapeHtml(this.panel.current.entry.sourceNote)}` : ''}</p>`;
  }
}
