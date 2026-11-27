// Shared helpers of the results pages: loading the generated data (data/*.json,
// written by `python3 -m nceval.site` from docs/results), number formatting, and
// building DOM. Every figure a page shows passes through here from that data;
// the pages' HTML holds none of its own. Text from the data is inserted with
// textContent, never as HTML.

export const REPO = 'https://github.com/cribbinsb/nanocodec/blob/main/';

export async function loadData(name) {
  const response = await fetch(`data/${name}.json`);
  if (!response.ok) throw new Error(`data/${name}.json: HTTP ${response.status}`);
  return response.json();
}

const MINUS = '−';

/** A BD-rate in percent with its sign ("−44.7%", "+4.7%", "0"). */
export function pct(value, digits = 1) {
  if (value === null || value === undefined) return '–';
  if (value === 0) return '0';
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return `${(0).toFixed(digits)}%`;
  return `${value < 0 ? MINUS : '+'}${text}%`;
}

/** Frames per second: one decimal below 100, whole numbers above. */
export function fps(value) {
  if (value === null || value === undefined) return '–';
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

export function ratio(value) {
  return value === null || value === undefined ? '–' : `${value.toFixed(2)}×`;
}

/** An element with attributes and children (strings become text nodes). */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/**
 * A table: columns are {label, num?, cell(row) -> string|Node}; rowClass(row)
 * gives each row's class. Wrapped for horizontal scrolling on phones.
 */
export function table(columns, rows, rowClass = () => '') {
  const head = el('tr', {}, columns.map((c) => el('th', { class: c.num ? 'num' : '', scope: 'col' }, c.label)));
  const body = rows.map((row) => el('tr', { class: rowClass(row) },
    columns.map((c) => el('td', { class: c.num ? 'num' : '' }, c.cell(row)))));
  return el('div', { class: 'table-wrap' }, el('table', {}, el('thead', {}, head), el('tbody', {}, body)));
}

/** A segmented control: options [{value, label, disabled?, title?}]; onChange(value). */
export function segmented(name, options, selected, onChange) {
  const group = el('div', { class: 'segmented', role: 'group', 'aria-label': name });
  const buttons = options.map((o) => el('button', {
    type: 'button', 'aria-pressed': String(o.value === selected), disabled: o.disabled, title: o.title,
    'data-value': o.value,
    onclick: () => {
      for (const b of buttons) b.setAttribute('aria-pressed', String(b === buttons[options.indexOf(o)]));
      onChange(o.value);
    },
  }, o.label));
  group.append(...buttons);
  return group;
}

export function control(label, widget) {
  return el('label', { class: 'control' }, el('span', {}, label), widget);
}

/** The "Data:" line under a table or chart: the checked-in files behind it. */
export function sources(paths) {
  return el('p', { class: 'sources' }, 'Data: ', paths.flatMap((p, i) => [
    i ? ', ' : '', el('a', { href: REPO + p }, el('code', {}, p.replace(/^docs\/results\//, ''))),
  ]));
}

/** The "Measured on" line of a table or chart: the date, our builds with their released
 * crate versions, the machines and the references (nceval.site writes the caption from
 * docs/results/provenance.json and the charts' build records); `bufferNote` (a CBR chart's
 * {caption, link, href}) adds the buffer line, its heading linked to the buffer page. */
export function measuredNote(caption, bufferNote = null) {
  return el('p', { class: 'note measured' }, caption,
    bufferNote && [` ${bufferNote.caption.slice(0, -bufferNote.link.length - 1)}`,
      el('a', { href: bufferNote.href }, bufferNote.link), '.']);
}

/** Report a page that could not load its data, instead of failing silently. */
export function fail(container, error) {
  container.replaceChildren(el('p', { class: 'disclaimer', role: 'alert' },
    `Could not load the results data (${error.message}). `,
    el('a', { href: REPO + 'docs/results/README.md' }, 'The results are also in the repository.')));
  console.error(error);
}

export const CODEC_NAMES = { h264: 'H.264', hevc: 'HEVC', vvc: 'VVC', av1: 'AV1', vp9: 'VP9', av2: 'AV2' };
export const METRIC_NAMES = { j: 'BD-J (PSNR-YUV, VMAF v1, MS-SSIM)', psnr: 'BD-PSNR-YUV', vmaf: 'BD-VMAF v1', ms_ssim: 'BD-MS-SSIM' };
export const METRIC_SHORT = { j: 'J', psnr: 'PSNR-YUV', vmaf: 'VMAF v1', ms_ssim: 'MS-SSIM' };
/** Shown where a view records a metric but this encoder's points do not have it yet. */
export const NOT_MEASURED = 'not measured';
export const ZERO_NAMES = {
  'x264-medium~psnr': 'x264 medium --tune zerolatency,psnr',
  'x264-medium': 'x264 medium, default tuning',
};
