// How the two views are laid out: side by side, stacked at full width, or one
// view large (the other's picture hidden, its panel kept). One control on the
// page, with or without analysis (which only adds its overlays and statistics
// to the views). The choice is remembered per viewer in localStorage; without
// one the page opens side by side, or stacked on a narrow screen. Storage can
// be missing or refused (private windows, blocked site data): the page then
// works the same, it just does not remember.

export const LAYOUTS = { side: 'Side by side', stacked: 'Stacked', 'expand-A': 'A large', 'expand-B': 'B large' };
const STORAGE_KEY = 'nanocodec.layout';
// Phone widths open stacked (styles.css stacks side-by-side views there anyway).
const NARROW_QUERY = '(max-width: 700px)';

function remembered() {
  try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}
function remember(layout) {
  try { localStorage.setItem(STORAGE_KEY, layout); } catch { /* not remembered: storage is unavailable */ }
}

export class Layout {
  /** `select`: the layout control; `versions`: the views' container, which gets a `layout-*` class. `onChange()`: after a change. */
  constructor(select, versions, onChange = () => {}) {
    this.select = select; this.versions = versions; this.onChange = onChange;
    select.innerHTML = Object.entries(LAYOUTS).map(([value, label]) => `<option value="${value}">${label}</option>`).join('');
    const saved = remembered();
    this.apply(saved in LAYOUTS ? saved : matchMedia(NARROW_QUERY).matches ? 'stacked' : 'side');
    select.addEventListener('change', () => { remember(select.value); this.apply(select.value); });
  }
  apply(layout) {
    this.value = layout; this.select.value = layout;
    for (const name of Object.keys(LAYOUTS)) this.versions.classList.toggle(`layout-${name}`, name === layout);
    this.onChange();
  }
}
