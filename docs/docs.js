// The usage pages (web/docs/): fill the parts generated from the code.
// - `[data-source="examples/x"]` blocks get that file's text: the examples
//   are copies of the checked, compiled and run sources (python3 -m nceval.site
//   writes them), so a page never shows code of its own.
// - `#options` gets every command's grouped options and `.presets` the
//   codecs' presets, from data/cli.json (written from crates/nc-cli/src/help.rs
//   by its `cli_reference_is_current` test).
// Text is inserted with textContent, never as HTML.

const CODEC_NAMES = { h264: 'H.264', hevc: 'HEVC', vvc: 'VVC', av1: 'AV1', vp9: 'VP9', av2: 'AV2' };

function element(tag, attrs = {}, text = '') {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text) node.textContent = text;
  return node;
}

async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.text();
}

async function fillSources() {
  await Promise.all([...document.querySelectorAll('[data-source]')].map(async (block) => {
    block.textContent = await fetchText(block.dataset.source);
  }));
}

/** One table per option group, under a heading per command. */
function renderOptions(target, commands) {
  for (const command of commands) {
    const section = element('section', { id: `cmd-${command.name}` });
    section.append(element('h2', {}, `nanocodec ${command.name}`), element('p', {}, command.about));
    for (const group of command.groups) {
      if (command.groups.length > 1) section.append(element('h3', {}, group.title));
      const wrap = element('div', { class: 'table-wrap' });
      const table = element('table', { class: 'options' });
      const head = element('tr');
      head.append(element('th', {}, 'Option'), element('th', {}, 'What it does'));
      table.append(head);
      for (const option of group.options) {
        const row = element('tr');
        const flag = element('td');
        flag.append(element('code', {}, option.value ? `${option.flag} ${option.value}` : option.flag));
        row.append(flag, element('td', {}, option.help));
        table.append(row);
      }
      wrap.append(table);
      section.append(wrap);
    }
    target.append(section);
  }
}

/** Each codec's presets, in the order the CLI lists them. */
function renderPresets(target, presets) {
  const wrap = element('div', { class: 'table-wrap' });
  const table = element('table');
  const head = element('tr');
  head.append(element('th', {}, 'Codec'), element('th', {}, '--preset'));
  table.append(head);
  for (const [codec, names] of Object.entries(presets)) {
    const row = element('tr');
    const cell = element('td');
    names.forEach((name, i) => {
      if (i) cell.append(' ');
      cell.append(element('code', {}, name));
    });
    row.append(element('td', {}, CODEC_NAMES[codec] ?? codec), cell);
    table.append(row);
  }
  wrap.append(table);
  target.append(wrap);
}

async function fillReference() {
  const options = document.getElementById('options');
  const presets = document.querySelectorAll('.presets');
  if (!options && !presets.length) return;
  const reference = JSON.parse(await fetchText('data/cli.json'));
  if (options) renderOptions(options, reference.commands);
  presets.forEach((target) => renderPresets(target, reference.presets));
}

Promise.all([fillSources(), fillReference()]).then(() => {
  document.body.dataset.ready = 'true';
}).catch((error) => {
  const note = element('p', { class: 'disclaimer', role: 'alert' }, `Could not load this page's data: ${error.message}`);
  document.querySelector('main').prepend(note);
  document.body.dataset.ready = 'error';
});
