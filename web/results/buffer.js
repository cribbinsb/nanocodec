// Rate control and the buffer (buffer.html): what each encoder is asked for in the CBR
// charts, how full each one's 1 s buffer gets on the published charts (at the target and
// at each stream's own rate), and our position. Data: data/buffer.json (nceval.site
// buffer_data, from nceval.buffer_use and nceval.buffer_note); every figure comes from it.
import {
  CODEC_NAMES, control, el, fail, loadData, measuredNote, segmented, sources, table,
} from './common.js';

const OURS = 'nanocodec-';
const state = { view: 'heldout-psnr' };

const share = (x) => `${x.toFixed(0)} %`;
const peak = (x) => x.toFixed(2);
const isOurs = (f) => f.startsWith(OURS);

function encoderName(data, family) {
  if (isOurs(family)) return `nanocodec ${CODEC_NAMES[family.slice(OURS.length)] || family} (every preset)`;
  return data.labels[family] || family;
}

function shareRange(values) {
  const low = Math.min(...values), high = Math.max(...values);
  return share(low) === share(high) ? share(low) : `${low.toFixed(0)}–${share(high)}`;
}

function asked(data, root) {
  root.querySelector('.asked-intro').textContent = `The harness's command lines at one rung (T = the target in kbps; `
    + `tool paths shortened). Each buffer is ${data.buffer_ms} ms at the target rate.`;
  root.querySelector('.requests').replaceChildren(el('ul', { class: 'prose' }, data.requests.map((r) =>
    el('li', {}, el('strong', {}, `${r.encoders}: `), r.note, ' ', el('code', {}, r.command)))));
}

function measure(data, root) {
  const limits = data.limits.map((l) => `${l}`).join(', ');
  root.querySelector('.measure').replaceChildren(
    el('li', {}, el('strong', {}, 'The bucket: '),
      `each point's stored frame sizes (${data.frames} frames, one key frame, every bit counted, our first key frame's borrowing included) are `
      + 'replayed through a leaky bucket that starts empty and is drained at a constant rate; its peak is given in buffers '
      + '(1 = one full second queued). Nothing is re-encoded.'),
    el('li', {}, el('strong', {}, 'At the target: '),
      'drained at the target and holding one second of the target, as the harness has always measured it. Points more '
      + `than ${data.rate_floor_pct} % over their target are left out of these columns and counted `
      + 'apart: they are the lowest rate an encoder can reach on a clip (libvpx and libaom on the lowest rungs, openh264), '
      + 'and a target-rate bucket reads them as many buffers, which says nothing about buffer control.'),
    el('li', {}, el('strong', {}, 'At the stream\'s own rate: '),
      'drained at the stream\'s own average rate and holding one second of it, over every point. The charts plot each '
      + 'point at its achieved rate, so this is the buffer at the rate it is judged at. An encoder that undershoots its '
      + 'target has, at the target, a larger buffer than one second of its own rate (half the target: two seconds); '
      + 'measured at its own rate, every encoder gets the same one second. We consider this the fair measure.'),
    el('li', {}, el('strong', {}, 'Shares: '),
      `the share of points whose peak passes ${limits} buffers: one buffer, our new re-code ceiling, and the level the `
      + 'harness flags as breaking the buffer.'),
  );
}

function useTable(data, root) {
  const v = data.views.find((x) => x.view === state.view);
  const families = Object.keys(v.rows).sort((a, b) => Number(isOurs(b)) - Number(isOurs(a)) || a.localeCompare(b));
  const over = (o) => data.limits.map((l) => share(o[`${l}`])).join(' / ');
  root.querySelector('.view-note').textContent = `${v.title}: ${v.clips} clips, ${v.points} points `
    + `(every rung the chart reads, low rungs included)${v.missing ? `, ${v.missing} points not in the store left out` : ''}; `
    + `our builds ${v.builds.join(', ')}. Over ${data.limits.join(' / ')} buffers.`;
  root.querySelector('.use-table').replaceChildren(table([
    { label: 'Encoder', cell: (f) => [el('span', { class: `swatch ${isOurs(f) ? 'ours' : 'ref'}` }), encoderName(data, f)] },
    { label: 'Points', num: true, cell: (f) => String(v.rows[f].points) },
    { label: 'Rate ÷ target, median (5th–95th pct)', num: true,
      cell: (f) => `${v.rows[f].ratio_median.toFixed(3)} (${peak(v.rows[f].ratio_p5)}–${peak(v.rows[f].ratio_p95)})` },
    { label: `Over target by > ${data.rate_floor_pct} %`, num: true, cell: (f) => share(v.rows[f].rate_floor_pct) },
    { label: 'At target: peak p95 / max', num: true, cell: (f) => `${peak(v.rows[f].target_p95)} / ${peak(v.rows[f].target_max)}` },
    { label: 'At target: points over', num: true, cell: (f) => over(v.rows[f].target_over) },
    { label: 'At own rate: peak p95 / max', num: true, cell: (f) => `${peak(v.rows[f].own_p95)} / ${peak(v.rows[f].own_max)}` },
    { label: 'At own rate: points over', num: true, cell: (f) => over(v.rows[f].own_over) },
  ], families, (f) => (isOurs(f) ? 'ours' : '')), measuredNote(data.measured.caption), sources(data.source));
  findings(data, v, families, root);
}

function findings(data, v, families, root) {
  const ours = families.filter(isOurs);
  const unbounded = families.filter((f) => data.unbounded.includes(f));
  const bounded = families.filter((f) => !isOurs(f) && !unbounded.includes(f));
  const ceiling = `${data.position.ceiling}`;
  const range = (list, key, limit) => shareRange(list.map((f) => v.rows[f][key][limit]));
  const rises = bounded.filter((f) => v.rows[f].own_p95 > v.rows[f].target_p95)
    .map((f) => `${encoderName(data, f)} ${peak(v.rows[f].target_p95)} → ${peak(v.rows[f].own_p95)}`);
  root.querySelector('.findings').replaceChildren(
    el('li', {}, el('strong', {}, 'Undershooting buys buffer: '),
      rises.length ? `measured at their own rate instead of the target, the 95th-percentile peak rises for ${rises.join('; ')}.`
        : 'on this clip set no reference\'s 95th-percentile peak rises when measured at its own rate.'),
    el('li', {}, el('strong', {}, `nanocodec on the ${data.position.release_before} charts: `),
      `${range(ours, 'own_over', '1')} of points over one buffer at their own rate and ${range(ours, 'own_over', ceiling)} over ${ceiling}.`),
    el('li', {}, el('strong', {}, 'The references with a buffer setting: '),
      `${range(bounded, 'own_over', '1')} of points over one buffer and ${range(bounded, 'own_over', ceiling)} over ${ceiling}; `
      + `their highest peaks reach ${peak(Math.max(...bounded.map((f) => v.rows[f].own_max)))} buffers.`),
    unbounded.length && el('li', {}, el('strong', {}, `${unbounded.map((f) => encoderName(data, f)).join(', ')} (no buffer setting): `),
      `${range(unbounded, 'own_over', '1')} of points over one buffer.`),
  );
}

/** The highest peak at the target of one arm of the cost measurement. */
function highest(data, arm) {
  return Math.max(...data.cost.rows.map((r) => r.at_target[arm].max));
}

function position(data, root) {
  const p = data.position;
  root.querySelector('.position').replaceChildren(
    el('li', {}, el('strong', {}, `Up to ${p.release_before}: `),
      `nanocodec treated its buffer as a target: a frame was re-coded only when it would leave more than ${p.ceiling_before} `
      + 'buffers queued, and its first key frame borrowed from the following seconds. On the published charts it ran the '
      + 'loosest buffer of the encoders with a buffer setting (the table above), which is worth quality in a short window.'),
    el('li', {}, el('strong', {}, `From ${p.release}: `),
      `the default re-codes any frame that would leave more than ${p.ceiling} buffers queued (${p.parameter}) and caps a key `
      + `frame at ${p.key_cap} of a buffer (${p.key_parameter}). That is about where SVT-AV1, libaom, libvpx and x264 are on `
      + `their worst points in the table above. Measured at the target, our highest peak in the measurement below is `
      + `${peak(highest(data, 'ceiling'))} buffers; measured at our own rate, a stream that ends under its target fills higher `
      + '(the table below), as the references\' do.'),
    el('li', {}, el('strong', {}, 'Strict mode: '),
      `${p.strict_flag} (the C API's ${p.strict_c_field}) re-codes any frame that would leave more than ${p.strict_ceiling} `
      + `buffer queued, for a receiver that needs a hard VBV (highest peak at the target below: ${peak(highest(data, 'strict'))}).`),
    el('li', {}, el('strong', {}, 'Why a ceiling, not a rule that drops points: '),
      'the charts could instead leave out every point that breaks a buffer limit. That would make the comparison depend on '
      + 'which encoder loses which points, a selection that can favour one side, so every point stays on its curve and we '
      + 'hold our own buffer instead.'),
    el('li', {}, el('strong', {}, 'What it costs: '), data.cost_sentence,
      ` The CBR charts will be re-measured on ${p.release}.`),
  );
  costTable(data, root);
}

function costTable(data, root) {
  const c = data.cost;
  const arms = ['before', 'ceiling', 'strict'];
  const peaks = (r, where) => arms.map((a) => `${peak(r[where][a].p95)} / ${peak(r[where][a].max)}`).join('; ');
  const bd = (x) => `${x > 0 ? '+' : x < 0 ? '−' : ''}${Math.abs(x).toFixed(2)} %`;
  root.querySelector('.cost-note').textContent = `${c.tool}, build ${c.build}, ${c.preset} presets, draw ${c.draw}: `
    + Object.entries(c.windows).map(([w, text]) => `${w} frames, ${text}`).join('; ') + `. Arms: `
    + Object.values(c.arms).join('; ') + `. ${c.runs}. BD-J against ${data.position.release_before}'s arm (positive costs bits); peaks in buffers, `
    + `p95 / max for ${data.position.release_before}; ${data.position.ceiling}; strict.`;
  root.querySelector('.cost-table').replaceChildren(table([
    { label: 'Codec', cell: (r) => CODEC_NAMES[r.codec] },
    { label: 'Frames', num: true, cell: (r) => String(r.frames) },
    { label: `BD-J, ${data.position.ceiling} buffers`, num: true, cell: (r) => bd(r.j_ceiling) },
    { label: 'BD-J, strict', num: true, cell: (r) => bd(r.j_strict) },
    { label: 'Peak at target', num: true, cell: (r) => peaks(r, 'at_target') },
    { label: 'Peak at own rate', num: true, cell: (r) => peaks(r, 'at_own_rate') },
  ], c.rows), sources([data.source[1]]));
}

async function main() {
  const root = document.getElementById('buffer');
  try {
    const data = await loadData('buffer');
    root.querySelector('.statement').textContent = data.statement;
    asked(data, root);
    measure(data, root);
    root.querySelector('.view-controls').append(control('Clip set', segmented('Clip set',
      data.views.map((v) => ({ value: v.view, label: v.title })), state.view,
      (value) => { state.view = value; useTable(data, root); })));
    useTable(data, root);
    position(data, root);
  } catch (error) {
    fail(root, error);
  }
}

main();
