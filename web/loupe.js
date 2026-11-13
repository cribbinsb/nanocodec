// Magnifier for the decoded views: one source region shown from both decoded
// outputs (and the original frame), side by side, with nearest-neighbour zoom.
// It only copies pixels from the existing canvases, on pointer moves and once
// per displayed frame while switched on, so the codec workers never wait on it.

// Size of each magnified pane in canvas pixels; zoom is relative to source pixels,
// so at 8x a pane shows a 48×27 pixel region of a version's output.
const PANE_WIDTH = 384, PANE_HEIGHT = 216;
const ZOOM_LEVELS = [2, 4, 8], DEFAULT_ZOOM = 4;
// A trackpad emits a burst of wheel events per gesture; one zoom step per burst.
const WHEEL_STEP_INTERVAL_MS = 150;
// A pointer that moves less than this between down and up is a tap (pin/unpin), not a drag.
const TAP_SLOP_PX = 6;

export class Loupe {
  constructor(element, panels) {
    this.element = element; this.panels = panels;
    this.enabled = false; this.pinned = false; this.zoom = DEFAULT_ZOOM;
    this.center = { x: 0.5, y: 0.5 }; this.source = null; this.sourceDrawn = null;
    this.drawRequested = false; this.lastWheel = 0; this.tapStart = null;
    this.toggle = element.querySelector('.loupe-toggle');
    this.pinButton = element.querySelector('.loupe-pin');
    this.zoomButtons = [...element.querySelectorAll('[data-zoom]')];
    this.sourceCanvas = document.createElement('canvas');
    // Views are read through `panel.canvas` each time: a view falling back from WebGL gets a new canvas.
    // `overlay`: the view's analysis layer for the region (analysis-view.js), when on.
    this.panes = [...panels.map(panel => ({ from: () => panel.canvas, label: () => `${panel.letter} · ${panel.title}`,
      overlay: (context, region) => panel.analysis?.drawInto(context, region, PANE_WIDTH, PANE_HEIGHT) })),
      { from: () => this.sourceCanvas, label: () => 'Original' }];
    const grid = element.querySelector('.loupe-panes');
    for (const pane of this.panes) {
      const figure = document.createElement('figure');
      figure.innerHTML = `<canvas width="${PANE_WIDTH}" height="${PANE_HEIGHT}"></canvas><figcaption></figcaption>`;
      grid.append(figure);
      pane.context = figure.querySelector('canvas').getContext('2d');
      pane.caption = figure.querySelector('figcaption');
    }
    this.markers = panels.map(panel => {
      const marker = document.createElement('div'); marker.className = 'loupe-marker'; marker.hidden = true;
      panel.viewport.append(marker); return marker;
    });
    this.toggle.addEventListener('click', () => this.setEnabled(!this.enabled));
    this.pinButton.addEventListener('click', () => this.setPinned(!this.pinned));
    for (const button of this.zoomButtons) button.addEventListener('click', () => this.setZoom(+button.dataset.zoom));
    for (const panel of panels) this.listen(panel);
    this.setZoom(this.zoom);
  }
  // Pointer events come from the view's viewport, which outlives its canvas.
  listen(panel) {
    const area = panel.viewport;
    area.addEventListener('pointermove', event => {
      if (!this.enabled || this.pinned) return;
      this.moveTo(panel.canvas, event);
    });
    area.addEventListener('pointerdown', event => {
      if (!this.enabled) return;
      this.tapStart = { x: event.clientX, y: event.clientY };
      if (!this.pinned) this.moveTo(panel.canvas, event);
    });
    area.addEventListener('pointerup', event => {
      if (!this.enabled || !this.tapStart) return;
      const moved = Math.hypot(event.clientX - this.tapStart.x, event.clientY - this.tapStart.y);
      this.tapStart = null;
      if (moved >= TAP_SLOP_PX) return;
      this.moveTo(panel.canvas, event, true);
      this.setPinned(!this.pinned);
    });
    area.addEventListener('wheel', event => {
      if (!this.enabled) return;
      event.preventDefault();
      if (event.timeStamp - this.lastWheel < WHEEL_STEP_INTERVAL_MS) return;
      this.lastWheel = event.timeStamp;
      const index = ZOOM_LEVELS.indexOf(this.zoom) + (event.deltaY < 0 ? 1 : -1);
      this.setZoom(ZOOM_LEVELS[Math.max(0, Math.min(ZOOM_LEVELS.length - 1, index))]);
    }, { passive: false });
  }
  setEnabled(value) {
    this.enabled = value; this.tapStart = null;
    this.toggle.setAttribute('aria-pressed', String(value));
    this.toggle.textContent = value ? 'Magnifier on' : 'Magnifier off';
    this.element.classList.toggle('active', value);
    // Touch drags over the views pan the magnifier only while it is on; otherwise they scroll the page.
    for (const panel of this.panels) panel.viewport.style.touchAction = value ? 'none' : '';
    if (!value) this.setPinned(false);
    this.requestDraw();
  }
  setPinned(value) {
    this.pinned = value;
    this.pinButton.setAttribute('aria-pressed', String(value));
    this.requestDraw();
  }
  setZoom(zoom) {
    this.zoom = zoom;
    for (const button of this.zoomButtons) {
      const selected = +button.dataset.zoom === zoom;
      button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', String(selected));
    }
    this.requestDraw();
  }
  // The decoder pixels were redrawn; `source` is the original frame they were encoded from, or null after a reset.
  frameShown(source) { this.source = source; this.requestDraw(); }
  // Normalized frame coordinates of the pointer, so views of different resolutions map to the same region.
  moveTo(canvas, event, force = false) {
    if (this.pinned && !force) return;
    const view = contentRect(canvas);
    this.center = { x: clamp01((event.clientX - view.left) / view.width), y: clamp01((event.clientY - view.top) / view.height) };
    this.requestDraw();
  }
  requestDraw() {
    if (!this.enabled && this.markers.every(marker => marker.hidden)) return;
    if (this.drawRequested) return;
    this.drawRequested = true;
    requestAnimationFrame(() => { this.drawRequested = false; this.draw(); });
  }
  // The region in normalized coordinates: PANE/zoom pixels of version A's output, kept inside the frame.
  region() {
    const reference = this.panels[0].canvas;
    const width = Math.min(1, PANE_WIDTH / this.zoom / reference.width);
    const height = Math.min(1, PANE_HEIGHT / this.zoom / reference.height);
    const x = Math.max(0, Math.min(1 - width, this.center.x - width / 2));
    const y = Math.max(0, Math.min(1 - height, this.center.y - height / 2));
    return { x, y, width, height };
  }
  draw() {
    for (const marker of this.markers) marker.hidden = !this.enabled;
    if (!this.enabled) return;
    this.updateSource();
    const region = this.region();
    for (const pane of this.panes) {
      const { context } = pane, from = pane.from(), label = pane.label();
      if (pane.caption.textContent !== label) pane.caption.textContent = label;
      context.clearRect(0, 0, PANE_WIDTH, PANE_HEIGHT);
      if (!from.width || !from.height) continue;
      context.imageSmoothingEnabled = false;
      context.drawImage(from, region.x * from.width, region.y * from.height, region.width * from.width, region.height * from.height, 0, 0, PANE_WIDTH, PANE_HEIGHT);
      pane.overlay?.(context, region);
    }
    this.panels.forEach((panel, i) => placeMarker(this.markers[i], panel.canvas, region));
  }
  // The original frame (a colour.js SourcePicture) is drawn into a canvas once per new frame so it can be scaled like the views.
  updateSource() {
    if (this.sourceDrawn === this.source) return;
    this.sourceDrawn = this.source;
    if (!this.source) { this.sourceCanvas.width = 0; return; }
    this.sourceCanvas.width = this.source.width; this.sourceCanvas.height = this.source.height;
    this.sourceCanvas.getContext('2d').putImageData(this.source.image, 0, 0);
  }
}

function clamp01(value) { return Math.max(0, Math.min(1, value)); }

// Where the frame is drawn inside the canvas element: the views use `object-fit: contain`.
function contentRect(canvas) {
  const box = canvas.getBoundingClientRect();
  const scale = Math.min(box.width / canvas.width, box.height / canvas.height);
  const width = canvas.width * scale, height = canvas.height * scale;
  return { left: box.left + (box.width - width) / 2, top: box.top + (box.height - height) / 2, width, height };
}

// Outline the magnified region on a view; the marker sits in the view's positioned viewport.
function placeMarker(marker, canvas, region) {
  const view = contentRect(canvas), parent = marker.parentElement.getBoundingClientRect();
  Object.assign(marker.style, {
    left: `${view.left - parent.left + region.x * view.width}px`, top: `${view.top - parent.top + region.y * view.height}px`,
    width: `${region.width * view.width}px`, height: `${region.height * view.height}px`,
  });
}
