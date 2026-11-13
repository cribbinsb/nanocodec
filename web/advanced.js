// A panel's collapsed "Advanced" section: curated toggles for what the selected
// codec supports (film grain, persistence AQ, temporal layers, second reference)
// and a free-form `name=value` editor checked against the engine's parameter
// registry (options.js). Its settings apply over the panel's preset.
import { loadOptions, PERSISTENCE_AQ, SECOND_REFERENCE, layerCount, settingLines } from './options.js';

const TIPS = {
  filmGrain: 'Experimental AV1 film grain synthesis: the encoder measures the texture that coding smoothed away and signals grain parameters; the decoder re-synthesizes the grain on its output. The WASM decoder\'s grain synthesis is still slow scalar code (no SIMD yet), so decode time rises, roughly doubling on textured clips. PSNR is measured on the grained output and drops by construction. Clean content is gated off and gets no grain.',
  blockQp: 'Persistence AQ (persist.strength=2): finer block quantizers where past frames predict a block well, coarser where nothing does. Off by default: it lost on BD-rate overall. Still blocks are refined to the quality their reuse warrants by default (persist.reuse).',
  temporalLayers: 'Temporal layers (layers.count): a hierarchical-P structure whose upper layers a receiver may drop. Costs some compression; not with the second reference.',
  secondReference: 'Second reference (refs.second=1): inter blocks may also predict from the last high-quality frame, at a second mode decision per block. Not with temporal layers.',
};
const escapeHtml = text => text.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

export class Advanced {
  /** `changed()` restarts the panel's encoder with the new settings. */
  constructor(letter, codec, changed) {
    this.codec = codec; this.changed = changed;
    this.api = null; this.supported = null; this.registry = []; this.applied = '';
    this.element = document.createElement('details'); this.element.className = 'advanced';
    this.element.innerHTML = `<summary>Advanced</summary>
      <p class="advanced-loading">Loading the codec's options…</p>
      <div class="advanced-toggles" hidden>
        <label class="toggle" data-option="filmGrain" title="${escapeHtml(TIPS.filmGrain)}"><input class="film-grain" type="checkbox">Film grain · slow decode</label>
        <label class="toggle" data-option="blockQp" title="${escapeHtml(TIPS.blockQp)}"><input class="persist-aq" type="checkbox">Persistence AQ</label>
        <label class="compact-control" data-option="temporalLayers" title="${escapeHtml(TIPS.temporalLayers)}">Temporal layers<select class="layers"><option value="1">1 · flat</option><option value="2">2 · L1T2</option><option value="3">3 · L1T3</option></select></label>
        <label class="toggle" data-option="secondReference" title="${escapeHtml(TIPS.secondReference)}"><input class="second-ref" type="checkbox">Second reference</label>
      </div>
      <label class="param-label" for="params-${letter}">Parameter overrides <small>name=value per line, over the preset</small></label>
      <div class="param-find"><input class="param-search" list="param-names-${letter}" placeholder="Find a parameter…" aria-label="Find a parameter"><button type="button" class="param-add" disabled>Add</button></div>
      <datalist id="param-names-${letter}"></datalist>
      <p class="param-doc"></p>
      <textarea id="params-${letter}" class="param-text" rows="3" spellcheck="false" placeholder="e.g. rate.aq_strength=1"></textarea>
      <ul class="param-errors" aria-live="polite"></ul>`;
    const $ = selector => this.element.querySelector(selector);
    this.filmGrain = $('.film-grain'); this.persist = $('.persist-aq'); this.layers = $('.layers'); this.secondRef = $('.second-ref');
    this.search = $('.param-search'); this.addButton = $('.param-add'); this.names = $('datalist'); this.doc = $('.param-doc');
    this.text = $('.param-text'); this.errors = $('.param-errors');
    this.element.addEventListener('toggle', () => { if (this.element.open) this.load(); });
    for (const control of [this.filmGrain, this.persist, this.layers, this.secondRef]) control.addEventListener('change', () => { this.syncToggles(); changed(); });
    this.search.addEventListener('input', () => this.describe());
    this.addButton.addEventListener('click', () => this.addSearched());
    this.text.addEventListener('change', () => { if (this.check()) changed(); });
  }
  async load() {
    try { this.api = await loadOptions(); this.setCodec(this.codec); } catch (error) {
      this.element.querySelector('.advanced-loading').textContent = `Could not load the codec's options: ${error.message}`;
    }
  }
  /** Show what `codec` supports and re-check the overrides for it (synchronous once loaded). */
  setCodec(codec) {
    this.codec = codec;
    if (!this.api) return;
    this.supported = this.api.codecOptions(codec);
    this.registry = this.api.parameterRegistry(codec);
    for (const label of this.element.querySelectorAll('[data-option]')) {
      const offered = this.supported[label.dataset.option];
      label.hidden = !offered;
      if (!offered) { for (const input of label.querySelectorAll('input')) input.checked = false; for (const select of label.querySelectorAll('select')) select.value = '1'; }
    }
    this.syncToggles();
    this.names.innerHTML = this.registry.map(p => `<option value="${p.name}"></option>`).join('');
    this.element.querySelector('.advanced-loading').hidden = true;
    this.element.querySelector('.advanced-toggles').hidden = false;
    this.describe();
    this.applied = '';
    this.check();
  }
  /** The engine refuses the second reference with temporal layers. */
  syncToggles() {
    const layered = +this.layers.value > 1;
    this.secondRef.disabled = layered;
    if (layered) this.secondRef.checked = false;
  }
  describe() {
    const spec = this.registry.find(p => p.name === this.search.value.trim());
    this.addButton.disabled = !spec;
    this.doc.textContent = spec ? `${spec.doc} · default ${spec.default}, range ${spec.min} to ${spec.max}` : '';
  }
  addSearched() {
    const spec = this.registry.find(p => p.name === this.search.value.trim());
    if (!spec) return;
    this.text.value = `${this.text.value.trimEnd()}${this.text.value.trim() ? '\n' : ''}${spec.name}=${spec.default}`;
    this.search.value = ''; this.describe();
    this.text.focus();
  }
  /** Check the editor's lines; only a valid text is applied. Returns whether it changed what applies. */
  check() {
    if (!this.api) return false;
    const errors = this.api.checkParameters(this.codec, this.text.value);
    this.errors.innerHTML = errors.map(e => `<li>Line ${e.line}: ${escapeHtml(e.message)}</li>`).join('');
    if (errors.length) this.errors.insertAdjacentHTML('beforeend', '<li class="not-applied">Not applied: fix these lines first.</li>');
    this.text.classList.toggle('invalid', errors.length > 0);
    if (errors.length || this.text.value === this.applied) return false;
    this.applied = this.text.value;
    return true;
  }
  /**
   * The encoder settings: `filmGrain`, and `overrides` as `name=value` lines, the
   * toggles' settings first, then the editor's (so they win).
   */
  settings() {
    const offered = this.supported ?? {};
    const lines = [
      ...(offered.blockQp && this.persist.checked ? PERSISTENCE_AQ : []),
      ...(offered.temporalLayers ? layerCount(+this.layers.value) : []),
      ...(offered.secondReference && this.secondRef.checked ? SECOND_REFERENCE : []),
      ...settingLines(this.applied),
    ];
    return { filmGrain: !!offered.filmGrain && this.filmGrain.checked, overrides: lines.join('\n') };
  }
}
