// Zoom and pan of the views: one state in normalised picture coordinates
// (`scale`, and the picture point `u`, `v` at the view's centre) applied to
// every view, so views of different sizes always show the same region. The
// picture is scaled by a CSS transform; once a picture pixel covers more than
// one device pixel it is drawn unsmoothed (nearest neighbour), so coding
// artefacts stay visible. Analysis overlays (analysis-view.js) draw through
// the same mapping, so they line up at any zoom.
//
// Input on a view: wheel or pinch zooms about the pointer, drag pans,
// double-click resets; with the view focused, + and - zoom, the arrow keys pan,
// 0 or Escape resets. While the views are not at the default fit, each shows a
// chip ("2.5× · Reset") that resets them when pressed.

const MIN_ZOOM = 1, MAX_ZOOM = 32;
// One wheel notch (deltaY 100) zooms by this factor.
const WHEEL_ZOOM_PER_100 = 1.25;
// A key press zooms by this factor, or pans by this fraction of the shown region.
const KEY_ZOOM = 1.5, KEY_PAN = 0.1;
// A pointer that moves less than this between down and up is a tap, not a drag.
const TAP_SLOP_PX = 6;
const FIT = Object.freeze({ scale: MIN_ZOOM, u: 0.5, v: 0.5 });
const KEY_PANS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };

export class ViewZoom {
  /**
   * `panels`: the views (panel.js; each has `viewport` and `canvas`). `onChange()`:
   * the zoom changed (overlays redraw). `onTap(panel, event)`: a tap or click on a view
   * that was not a drag (the analysis hover record pins on it).
   */
  constructor(panels, { onChange = () => {}, onTap = () => {} } = {}) {
    this.panels = panels; this.onChange = onChange; this.onTap = onTap;
    this.state = { ...FIT };
    this.chips = panels.map(panel => {
      const chip = document.createElement('button');
      chip.type = 'button'; chip.className = 'zoom-chip'; chip.hidden = true;
      chip.addEventListener('click', event => { event.stopPropagation(); this.reset(); });
      // The chip's own pointer input is not a pan or tap on the view.
      chip.addEventListener('pointerdown', event => event.stopPropagation());
      panel.viewport.append(chip);
      return chip;
    });
    for (const panel of panels) this.listen(panel);
    this.show();
  }
  /** Whether the views show the whole picture (no zoom). */
  get atFit() { return this.state.scale === MIN_ZOOM; }
  reset() { this.set(FIT); }
  /** Zoom by `factor` about the normalised point `at` ({ u, v }; default: the centre of the shown region). */
  zoomBy(factor, at = null) {
    const { scale, u, v } = this.state;
    const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, scale * factor));
    // The picture point under `at` stays where it is on screen.
    const point = at ?? { u, v }, ratio = scale / next;
    this.set({ scale: next, u: point.u + (u - point.u) * ratio, v: point.v + (v - point.v) * ratio });
  }
  /** Pan by a normalised picture offset. */
  panBy(du, dv) { this.set({ ...this.state, u: this.state.u + du, v: this.state.v + dv }); }
  set({ scale, u, v }) {
    // The shown region stays inside the picture.
    const half = 0.5 / scale, clamp = value => Math.max(half, Math.min(1 - half, value));
    this.state = { scale, u: clamp(u), v: clamp(v) };
    this.show();
    this.onChange();
  }
  /** Apply the zoom to every view and update the chips. */
  show() {
    for (const panel of this.panels) this.place(panel);
    const text = `${this.state.scale.toFixed(this.state.scale < 10 ? 1 : 0).replace(/\.0$/, '')}× · Reset`;
    for (const chip of this.chips) {
      chip.hidden = this.atFit;
      chip.textContent = text;
      chip.setAttribute('aria-label', `Zoomed ${text.replace(' · Reset', '')}: reset to the whole picture`);
    }
  }
  /**
   * Viewport pixels per picture pixel and the picture origin's viewport position
   * for `panel` at the shared zoom (the canvas fills the viewport, its picture fitted
   * inside with `object-fit: contain`), or null before it has a size.
   */
  mapping(panel) {
    const box = panel.viewport.getBoundingClientRect(), { width, height } = panel.canvas;
    if (!width || !height || !box.width) return null;
    const fit = Math.min(box.width / width, box.height / height);
    const fx = (box.width - width * fit) / 2, fy = (box.height - height * fit) / 2;
    const { scale, u, v } = this.state;
    const tx = box.width / 2 - (fx + u * width * fit) * scale, ty = box.height / 2 - (fy + v * height * fit) * scale;
    return { box, fit, scale, tx, ty, s: fit * scale, ox: fx * scale + tx, oy: fy * scale + ty, width, height };
  }
  /** Transform `panel`'s canvas for the zoom (also after the panel replaced its canvas, or resized). */
  place(panel) {
    const canvas = panel.canvas, map = this.mapping(panel);
    panel.zoomedCanvas = canvas;
    if (!map || map.scale === MIN_ZOOM) {
      canvas.style.transform = ''; canvas.style.transformOrigin = '';
    } else {
      canvas.style.transformOrigin = '0 0';
      canvas.style.transform = `translate(${map.tx}px, ${map.ty}px) scale(${map.scale})`;
    }
    canvas.style.imageRendering = map && map.s * devicePixelRatio > 1 ? 'pixelated' : '';
    // Zoomed, touch drags pan the picture; at fit they scroll the page (pinches still reach us).
    panel.viewport.style.touchAction = this.atFit ? 'pan-y' : 'none';
  }
  /** A view showed a frame: place its canvas if it is a new one (a WebGL fallback replaces it). */
  frameShown(panel) { if (panel.zoomedCanvas !== panel.canvas) this.place(panel); }
  /** Normalised picture coordinates of a pointer event on `panel`, or null outside the picture. */
  pointAt(panel, event) {
    const map = this.mapping(panel);
    if (!map) return null;
    const x = (event.clientX - map.box.left - map.ox) / map.s, y = (event.clientY - map.box.top - map.oy) / map.s;
    return x >= 0 && y >= 0 && x < map.width && y < map.height ? { u: x / map.width, v: y / map.height } : null;
  }
  listen(panel) {
    const area = panel.viewport;
    area.tabIndex = 0;
    area.setAttribute('aria-label', `Version ${panel.letter} picture: scroll or pinch to zoom, drag to pan, double-click or Escape to reset; with focus, + and − zoom, arrow keys pan`);
    new ResizeObserver(() => this.place(panel)).observe(area);
    area.addEventListener('wheel', event => {
      // At fit, scrolling down (zooming out) scrolls the page instead.
      if (this.atFit && event.deltaY >= 0) return;
      event.preventDefault();
      this.zoomBy(WHEEL_ZOOM_PER_100 ** (-event.deltaY / 100), this.pointAt(panel, event));
    }, { passive: false });
    area.addEventListener('dblclick', () => this.reset());
    area.addEventListener('keydown', event => {
      if (event.key === '+' || event.key === '=') this.zoomBy(KEY_ZOOM);
      else if (event.key === '-') this.zoomBy(1 / KEY_ZOOM);
      else if ((event.key === 'Escape' || event.key === '0') && !this.atFit) this.reset();
      else if (KEY_PANS[event.key] && !this.atFit) {
        const [dx, dy] = KEY_PANS[event.key];
        this.panBy(dx * KEY_PAN / this.state.scale, dy * KEY_PAN / this.state.scale);
      } else return;
      event.preventDefault();
    });
    const pointers = new Map();
    let drag = null;
    area.addEventListener('pointerdown', event => {
      if (event.button > 0) return;
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      drag = { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, pinch: null };
      area.setPointerCapture(event.pointerId);
    });
    area.addEventListener('pointermove', event => {
      if (!pointers.has(event.pointerId) || !drag) return;
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      const map = this.mapping(panel);
      if (!map) return;
      if (pointers.size === 2) {
        // Pinch: zoom by the change of the two pointers' distance.
        const [a, b] = [...pointers.values()], distance = Math.hypot(a.x - b.x, a.y - b.y);
        if (drag.pinch) this.zoomBy(distance / drag.pinch);
        drag.pinch = distance;
      } else if (!this.atFit) {
        this.panBy(-(event.clientX - drag.x) / (map.s * map.width), -(event.clientY - drag.y) / (map.s * map.height));
      }
      drag.x = event.clientX; drag.y = event.clientY;
    });
    const end = event => {
      if (!pointers.delete(event.pointerId) || !drag) return;
      const tap = Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < TAP_SLOP_PX && !drag.pinch;
      if (pointers.size === 0) drag = null;
      if (tap && event.type === 'pointerup') this.onTap(panel, event);
    };
    area.addEventListener('pointerup', end);
    area.addEventListener('pointercancel', end);
  }
  /** The zoom for the debug report. */
  debugState() { return { ...this.state }; }
}
