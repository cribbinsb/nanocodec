// One comparison view: its source (a live nanocodec encode or a pre-encoded
// reference stream), its own picture size and frame rate, its encoder settings,
// the decoded picture, its statistics and its download.
import { download, streamBlob, MAX_RECORDING_BYTES } from './streams.js';
import { CODECS, streamExtension } from './codecs.js';
import { createRenderer } from './renderer.js';
import { Advanced } from './advanced.js';
import { cliCommand, cliPreset } from './options.js';
import { FRAME_RATES, tickMs, tickStep } from './timeline.js';
import { fetchStream, loadManifest, referenceStream, streamGroups, streamTitle } from './reference.js';
import { trace } from './trace.js';
import { BUFFER_CHOICES_MS, DEFAULT_FPS, DEFAULT_RESOLUTION, GOP_SECONDS, RATE_LIMITS_KBPS, RESOLUTIONS } from './view-settings.js';

// Statistics cover this many recent coded frames.
const HISTORY_FRAMES = 90;
// The metrics, rate line, frame-size chart and settings text are redrawn at most
// this often (and once more after the last frame), not on every frame: they
// write and lay out DOM text, read a computed style and repaint a 90-bar chart.
// Headless Chromium, default page, 3 interleaved loads: Panel.frame 0.72 ->
// 0.56 ms mean (median 0.6 -> 0.4 ms) per view per frame on the page thread.
const METRICS_INTERVAL_MS = 250;
const DEFAULT_GOP_SECONDS = 30, DEFAULT_BUFFER_MS = 1000;
const DEFAULT_TARGET_KBPS = 300, DEFAULT_PEAK_KBPS = 600;
const SOURCES = { live: 'live', reference: 'reference' };
const escapeHtml = text => String(text).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const options = (items, selected) => items.map(([value, label]) => `<option value="${value}"${String(value) === String(selected) ? ' selected' : ''}>${label}</option>`).join('');
const rateInput = (name, value) => `<input class="${name}" type="number" required min="${RATE_LIMITS_KBPS.min}" max="${RATE_LIMITS_KBPS.max}" step="${RATE_LIMITS_KBPS.step}" value="${value}">`;

export class Panel {
  /**
   * `changed(kind)`: a setting changed ('rate' for live rate changes, 'reference'
   * when another pre-encoded stream was chosen, else a restart). `onMatch(entry)`:
   * the visitor asked the other view to match this view's stream. `clipTitle(id)`:
   * a sample clip's name, for the stream picker. `display`: the preferred renderer ('auto', 'webgl2', 'webgl' or 'cpu'; renderer.js).
   */
  constructor(letter, codec, container, { changed, onMatch, clipTitle }, display) {
    this.letter = letter; this.frames = []; this.history = []; this.bytes = 0; this.full = false;
    this.changed = changed; this.clipTitle = clipTitle;
    // The selected pre-encoded stream's manifest record, and its loaded packets.
    this.reference = null; this.streams = [];
    this.loaded = new Map();
    // Stream fetches by stream id, started once each (prepare).
    this.fetches = new Map();
    this.element = document.createElement('article'); this.element.className = 'version';
    this.element.innerHTML = `
      <div class="version-top"><span class="version-letter">${letter}</span><h2>Version ${letter}</h2><select class="source-picker" aria-label="Version ${letter} source"><option value="${SOURCES.live}">Live · nanocodec</option><option value="${SOURCES.reference}">Pre-encoded · other encoders</option></select><select class="codec-picker" aria-label="Version ${letter} codec">${Object.entries(CODECS).map(([value, { label }]) => `<option value="${value}">${label}</option>`).join('')}</select></div>
      <div class="viewport"><canvas aria-label="Version ${letter} decoded output"></canvas><div class="empty-overlay"><div class="empty-symbol">◉</div><strong>Your decoded video appears here</strong><span>Choose a source and start comparing</span></div><span class="frame-badge">WASM DECODER · READY</span></div>
      <div class="metrics"><div class="metric"><span>Bitrate</span><p><b data-metric="bitrate">—</b><small>kbps</small></p></div><div class="metric"><span>Encode</span><p><b data-metric="encode">—</b><small>ms</small></p></div><div class="metric"><span>Decode</span><p><b data-metric="decode">—</b><small>ms</small></p></div><div class="metric"><span>PSNR-YUV</span><p><b data-metric="psnr">—</b><small>dB</small></p></div></div>
      <div class="picture-options"><label class="compact-control">Resolution<select class="resolution">${options(RESOLUTIONS.map(([w, h]) => [`${w}x${h}`, `${w} × ${h}`]), DEFAULT_RESOLUTION.join('x'))}</select></label><label class="compact-control">Frame rate<select class="fps">${options(FRAME_RATES.map(fps => [fps, `${fps} fps`]), DEFAULT_FPS)}</select></label></div>
      <div class="reference-options" hidden><label class="compact-control">Pre-encoded stream<select class="stream-picker"></select></label><p class="reference-status" role="status"></p><div class="reference-info"><p class="reference-label"></p><dl class="reference-facts"></dl><p class="reference-version"></p><div class="reference-command"><code class="reference-cli"></code><button type="button" class="copy-command">Copy</button></div><button type="button" class="match">Match this in the other version</button></div></div>
      <div class="options"><label class="compact-control">Rate mode<select class="rate-mode"><option value="cbr" selected>Target bitrate</option><option value="qp">Fixed quantizer</option></select></label><div class="rate-options"><label class="compact-control">Target (kbps)${rateInput('target-rate', DEFAULT_TARGET_KBPS)}</label><label class="compact-control">Peak (kbps)${rateInput('peak-rate', DEFAULT_PEAK_KBPS)}</label><label class="compact-control">Buffer<select class="buffer-ms">${options(BUFFER_CHOICES_MS.map(ms => [ms, ms < 1000 ? `${ms} ms` : `${ms / 1000} second${ms > 1000 ? 's' : ''}`]), DEFAULT_BUFFER_MS)}</select></label></div><div class="quality-option" hidden><label class="quality-label" for="qp-${letter}">Compression / quality <output></output></label><input id="qp-${letter}" type="range" step="1"><div class="range-ends"><span>More detail · larger stream</span><span>Smaller stream</span></div></div><label class="compact-control">Encoder preset<select class="preset"><option value="default">Default</option><option value="fast">Fast · quality tradeoff</option><option value="faster">Faster · H.264, HEVC, AV1 and VP9</option><option value="fastest">Fastest · H.264 and VP9</option><option value="svc">SVC · full + half-size layers (AV1, VP9)</option><option value="svc-fast">SVC Fast · full + half-size layers (AV1, VP9)</option><option value="svc3">SVC 3 layers · full, half and quarter size (AV1)</option><option value="svc3-fast">SVC 3 layers Fast · full, half and quarter size (AV1)</option></select></label><label class="compact-control layer-control" hidden>Shown layer<select class="operating-point"><option value="0" selected>Full size (operating point 0)</option><option value="1">Half size (operating point 1)</option><option value="2">Quarter size (operating point 2)</option></select></label><label class="compact-control">Keyframe interval<select class="gop">${options(GOP_SECONDS.map(s => [s, `${s} second${s > 1 ? 's' : ''}`]), DEFAULT_GOP_SECONDS)}</select></label></div>
      <p class="rate-status" aria-live="off">Target 300 kbps · waiting for frames</p><div class="history"><div class="history-label"><span>FRAME SIZE · LAST ${HISTORY_FRAMES} FRAMES</span><span class="history-max">—</span></div><canvas width="540" height="42" aria-label="Recent compressed frame sizes"></canvas></div>
      <div class="version-bottom"><span class="frame-count">No frames encoded</span><button class="download" disabled>Download stream ↓</button></div>
      <details class="reproduce"><summary>Settings · CLI command</summary><p class="effective"></p><code class="cli"></code></details>`;
    container.append(this.element);
    const $ = selector => this.element.querySelector(selector);
    this.source = $('.source-picker');
    this.codec = $('.codec-picker'); this.codec.value = codec;
    this.resolution = $('.resolution'); this.fps = $('.fps');
    this.streamPicker = $('.stream-picker');
    this.qp = $('input[type=range]'); this.preset = $('.preset');
    this.gop = $('.gop');
    // Which SVC operating point this view decodes: switched while running, without
    // restarting the encoder (pipeline.js passes it to the decode worker).
    this.operatingPoint = $('.operating-point');
    this.preset.addEventListener('change', () => this.updateLayers());
    this.advanced = new Advanced(letter, codec, changed); $('.options').append(this.advanced.element);
    this.reproduce = $('.reproduce');
    this.reproduce.addEventListener('toggle', () => this.describeSettings()); this.canvas = $('.viewport canvas');
    this.viewport = $('.viewport'); this.useRenderer(display); this.badge = $('.frame-badge');
    this.chart = $('.history canvas'); this.downloadButton = $('.download');
    this.rateMode = $('.rate-mode');
    this.targetRate = $('.target-rate');
    this.peakRate = $('.peak-rate');
    this.bufferMs = $('.buffer-ms');
    for (const control of [this.rateMode, this.targetRate, this.peakRate, this.bufferMs]) control.addEventListener('change', () => {
      const invalid = [this.targetRate, this.peakRate].find(input => !input.checkValidity());
      if (invalid) { invalid.reportValidity(); return; }
      if (+this.peakRate.value < +this.targetRate.value) this.peakRate.value = this.targetRate.value;
      this.showRateMode();
      changed('rate');
    });
    this.updateCodec();
    this.codec.addEventListener('change', () => { this.updateCodec(); changed(); });
    this.qp.addEventListener('input', () => this.updateLabel());
    for (const control of [this.qp, this.preset, this.gop, this.resolution, this.fps]) control.addEventListener('change', () => changed());
    this.source.addEventListener('change', () => this.sourceChanged());
    this.streamPicker.addEventListener('change', () => { this.choose(this.streams.find(s => s.id === this.streamPicker.value)); changed('reference'); });
    $('.copy-command').addEventListener('click', () => navigator.clipboard?.writeText(this.reference?.command ?? ''));
    $('.match').addEventListener('click', () => { if (this.reference) onMatch(this.reference); });
    this.downloadButton.addEventListener('click', () => this.save());
  }
  /** Draw with `preferred` (renderer.js), falling back to the CPU path if GL set-up fails. */
  useRenderer(preferred) {
    try {
      this.renderer = createRenderer(this.canvas, { preferred, onLost: () => this.fallBack() });
      this.viewport.dataset.display = this.renderer.kind;
    } catch (error) {
      console.warn(`Version ${this.letter}: no WebGL display, using the CPU path`, error);
      this.fallBack();
    }
  }
  /** The CPU path on a fresh canvas (one that had a GL context cannot draw 2D); size, label and class are cloned. */
  fallBack() {
    const fresh = this.canvas.cloneNode();
    this.canvas.replaceWith(fresh); this.canvas = fresh;
    this.renderer = createRenderer(fresh, { preferred: 'cpu' });
    this.viewport.dataset.display = this.renderer.kind;
  }
  /** Whether this view plays a pre-encoded stream. */
  get preEncoded() { return this.source.value === SOURCES.reference; }
  /** The view's name in captions: codec, or the reference encoder. */
  get title() { return this.preEncoded && this.reference ? this.reference.label : this.codec.selectedOptions[0].text; }
  /** Switch between live and pre-encoded; the stream list loads on first use. */
  async sourceChanged() {
    this.showSource();
    if (!this.preEncoded) { this.changed(); return; }
    try {
      this.status('Loading the list of pre-encoded streams…');
      this.streams = await loadManifest();
    } catch (error) { this.status(error.message, true); return; }
    this.status('');
    this.streamPicker.innerHTML = streamGroups(this.streams).map(([clip, entries]) => `<optgroup label="${escapeHtml(this.clipTitle(clip))}">${
      entries.map(entry => `<option value="${entry.id}">${escapeHtml(streamTitle(entry))}</option>`).join('')}</optgroup>`).join('');
    this.choose(this.reference ?? this.streams[0]);
    this.changed('reference');
  }
  /** Select a stream (manifest record); its file loads when the comparison starts. */
  choose(entry) {
    trace('choose', this.letter, entry.id);
    this.reference = entry;
    this.streamPicker.value = entry.id;
    if (this.codec.value !== entry.codec) { this.codec.value = entry.codec; this.updateCodec(); }
    const facts = {
      Encoder: `${entry.label}${entry.chartConfiguration ? ' · the chart configuration' : ' · not a chart configuration'}`,
      Clip: this.clipTitle(entry.clip), Size: `${entry.width} × ${entry.height}`, 'Frame rate': `${entry.fps} fps`,
      Bitrate: `target ${entry.targetKbps} kbps · actual ${entry.actualKbps} kbps`,
      'Key interval': `${entry.keyInterval} frames`, Source: entry.sourceFile,
      ...(entry.notice ? { Licence: entry.notice } : {}),
    };
    const $ = selector => this.element.querySelector(selector);
    $('.reference-label').textContent = 'Encoded offline by this third-party encoder at the real-time settings our charts use (one thread, low delay, CBR); its other settings may give better results. Decoded here by nanocodec’s WASM decoder.';
    $('.reference-facts').innerHTML = Object.entries(facts).map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`).join('');
    const { version, source, gpu } = entry.version;
    $('.reference-version').textContent = `Version: ${version}${source?.commit ? ` · source ${source.ref} ${source.commit.slice(0, 12)}` : ''}${gpu ? ` · GPU ${gpu}` : ''}`;
    $('.reference-cli').textContent = entry.command;
  }
  /** Show the controls of the selected source (live encoder settings, or the stream picker and its facts). */
  showSource() {
    const preEncoded = this.preEncoded;
    this.element.querySelector('.reference-options').hidden = !preEncoded;
    this.element.querySelector('.picture-options').hidden = preEncoded;
    this.element.querySelector('.options').hidden = preEncoded;
    this.codec.disabled = preEncoded;
    this.element.classList.toggle('pre-encoded', preEncoded);
    if (!preEncoded) this.reference = null;
  }
  /** Pick the first stream of `clip` if the selected one is of another clip; false if there is none. */
  followClip(clip) {
    if (!this.preEncoded || this.reference?.clip === clip) return true;
    const entry = this.streams.find(s => s.clip === clip);
    if (entry) this.choose(entry);
    return Boolean(entry);
  }
  /** Leave pre-encoded mode (its stream cannot play on the current source); the caller restarts. */
  goLive() { this.source.value = SOURCES.live; this.showSource(); }
  /** Load the selected stream's file (once per stream), with progress in the panel. */
  /**
   * Load the stream of the settings this view was last reset with (`current`), once
   * per stream, with progress in the panel. A start uses the stream of its own
   * settings even if the visitor picks another meanwhile: that choice restarts.
   */
  async prepare() {
    const entry = this.current.entry;
    if (!entry || this.loaded.has(entry.id)) return;
    // A restart while the file is still arriving waits for that same fetch.
    if (!this.fetches.has(entry.id)) {
      this.status(`Fetching ${entry.file}…`);
      this.fetches.set(entry.id, fetchStream(entry, fraction => this.status(`Fetching ${entry.file} · ${(fraction * 100).toFixed(0)}%`))
        .then(bytes => referenceStream(entry, bytes)));
    }
    try {
      this.loaded.set(entry.id, await this.fetches.get(entry.id));
      this.status('');
      trace('fetched', this.letter, entry.id);
    } catch (error) {
      this.fetches.delete(entry.id);
      this.status(error.message, true);
      throw new Error(`Version ${this.letter}: ${error.message}`);
    }
  }
  /** The loaded stream of this view's current settings (after prepare), or null for a live view. */
  get stream() { return this.current?.entry ? this.loaded.get(this.current.entry.id) ?? null : null; }

  status(text, failed = false) {
    const line = this.element.querySelector('.reference-status');
    line.textContent = text; line.classList.toggle('failed', failed);
  }
  /** Apply live settings that match a pre-encoded stream (view-settings.js `matchingSettings`). */
  match(settings) {
    this.source.value = SOURCES.live; this.showSource();
    this.resolution.value = `${settings.width}x${settings.height}`; this.fps.value = String(settings.fps);
    if (this.codec.value !== settings.codec) { this.codec.value = settings.codec; this.updateCodec(); }
    this.rateMode.value = settings.rateMode; this.targetRate.value = settings.targetKbps; this.peakRate.value = settings.peakKbps;
    this.bufferMs.value = String(settings.bufferMs); this.gop.value = String(settings.gopSeconds);
    this.showRateMode();
  }
  showRateMode() {
    this.element.querySelector('.rate-options').hidden = this.rateMode.value !== 'cbr';
    this.element.querySelector('.quality-option').hidden = this.rateMode.value === 'cbr';
  }
  updateCodec() {
    const { fast, faster, fastest, svc, svc3, quantizer } = CODECS[this.codec.value];
    this.advanced.setCodec(this.codec.value);
    this.preset.disabled = !fast;
    this.preset.querySelector('option[value="faster"]').disabled = !faster;
    this.preset.querySelector('option[value="fastest"]').disabled = !fastest;
    for (const value of ['svc', 'svc-fast']) this.preset.querySelector(`option[value="${value}"]`).disabled = !svc;
    for (const value of ['svc3', 'svc3-fast']) this.preset.querySelector(`option[value="${value}"]`).disabled = !svc3;
    const unavailable = (!faster && this.preset.value === 'faster') || (!fastest && this.preset.value === 'fastest')
      || (!svc && this.preset.value.startsWith('svc')) || (!svc3 && this.preset.value.startsWith('svc3'));
    if (this.preset.disabled || unavailable) this.preset.value = 'default';
    this.updateLayers();
    this.qp.min = String(quantizer.min); this.qp.max = String(quantizer.max);
    this.qp.value = String(quantizer.initial); this.updateLabel();
  }
  /** The layer picker shows with the SVC presets only (other streams have one operating point);
   * the quarter-size operating point exists in three-layer streams only. */
  updateLayers() {
    const layered = this.preset.value.startsWith('svc'), threeLayers = this.preset.value.startsWith('svc3');
    this.element.querySelector('.layer-control').hidden = !layered;
    this.operatingPoint.querySelector('option[value="2"]').disabled = !threeLayers;
    if (!layered || (!threeLayers && this.operatingPoint.value === '2')) this.operatingPoint.value = '0';
  }
  updateLabel() { this.element.querySelector('output').textContent = `${CODECS[this.codec.value].quantizer.name} ${this.qp.value}`; }
  /**
   * This view's settings: picture size and frame rate, and the live encoder's settings
   * or the stream's (`entry`). A running view reads only these, not the controls,
   * which may already show the next settings.
   */
  config() {
    if (this.preEncoded && this.reference) {
      const { codec, width, height, fps, targetKbps, keyInterval } = this.reference;
      return { source: SOURCES.reference, entry: this.reference, codec, width, height, fps, step: tickStep(fps), rateMode: 'cbr', targetKbps, keyInterval };
    }
    const [width, height] = this.resolution.value.split('x').map(Number), fps = +this.fps.value;
    return { source: SOURCES.live, codec: this.codec.value, width, height, fps, step: tickStep(fps), qp: +this.qp.value,
      preset: this.preset.value, ...this.advanced.settings(), keyInterval: +this.gop.value * fps, ...this.rateSettings() };
  }
  /** The live rate settings, which change without a restart. */
  rateSettings() {
    return { rateMode: this.rateMode.value, targetKbps: +this.targetRate.value, peakKbps: +this.peakRate.value, bufferMs: +this.bufferMs.value };
  }
  reset(config) {
    this.current = config; this.frames = []; this.history = []; this.bytes = 0; this.full = false; this.count = 0;
    this.element.querySelector('.rate-status').textContent = config.source === SOURCES.reference ? 'Pre-encoded offline · waiting for frames'
      : config.rateMode === 'cbr' ? `Target ${config.targetKbps} kbps · waiting for frames` : 'Fixed quantizer';
    this.shown = false; this.element.classList.remove('error'); this.downloadButton.disabled = true;
    this.canvas.width = config.width; this.canvas.height = config.height;
    this.element.querySelector('.empty-overlay').hidden = false;
    this.badge.textContent = 'WASM DECODER · STARTING';
    delete this.badge.dataset.tick;
    for (const e of this.element.querySelectorAll('[data-metric]')) e.textContent = '—';
    this.element.querySelector('.frame-count').textContent = 'No frames encoded';
    this.chart.getContext('2d').clearRect(0, 0, this.chart.width, this.chart.height);
    clearTimeout(this.metricsTimer); this.metricsTimer = 0; this.metricsShownAt = -Infinity;
    this.element.querySelector('.reproduce summary').textContent = config.source === SOURCES.reference ? 'Settings · encoder command' : 'Settings · CLI command';
    this.describeSettings();
  }
  /** Show the frame decoded for source tick `tick`; returns a pixel buffer the worker may reuse. */
  frame(result, showOriginal, original, tick) {
    const sw = result.streamWidth ?? result.width, sh = result.streamHeight ?? result.height;
    if (sw !== this.current.width || sh !== this.current.height || !result.width || !result.height || result.width > sw || result.height > sh) {
      throw new Error('Decoder output dimensions differ from the input');
    }
    const free = this.renderer.show(result);
    this.shown = true; this.original = original;
    if (showOriginal) this.renderer.showOriginal(original?.image ?? null);
    this.element.querySelector('.empty-overlay').hidden = true;
    // `earlier`: frames coded with this one (start-up lookahead, or stream frames a capture skipped).
    for (const coded of [...(result.earlier ?? []), result]) this.record(coded);
    this.badge.dataset.tick = String(tick);
    this.label(showOriginal, result);
    this.scheduleMetrics(result);
    return free;
  }
  /** Show `result`'s metrics now if METRICS_INTERVAL_MS has passed, else once that interval ends. */
  scheduleMetrics(result) {
    this.metricsResult = result;
    if (this.metricsTimer) return;
    const wait = this.metricsShownAt + METRICS_INTERVAL_MS - performance.now();
    if (wait <= 0) this.showMetrics();
    else this.metricsTimer = setTimeout(() => { this.metricsTimer = 0; this.showMetrics(); }, wait);
  }
  /** The running averages, frame count, rate line, frame-size chart and settings for the latest result. */
  showMetrics() {
    const result = this.metricsResult;
    this.metricsShownAt = performance.now();
    const encode = this.current.entry ? null : this.average('encode');
    for (const [key, value] of Object.entries({ bitrate: this.average('bytes') * 8 * this.current.fps / 1000, encode, decode: this.average('decode'), psnr: this.average('psnr') })) {
      this.element.querySelector(`[data-metric="${key}"]`).textContent = value === null ? 'n/a' : value.toFixed(key === 'bitrate' ? 0 : 1);
    }
    this.element.querySelector('.frame-count').textContent = this.current.entry
      ? `${this.count} stream frames decoded · stream ${(this.current.entry.bytes / 1024).toFixed(0)} KiB`
      : `${this.count} frames · ${(this.bytes / 1024).toFixed(0)} KiB saved${this.full ? ' · buffer full' : ''}`;
    this.element.querySelector('.rate-status').textContent = this.rateStatus(result);
    this.drawHistory();
    this.describeSettings();
  }
  /** The badge: what is shown, the frame type, its number in this view and its source time. */
  label(showOriginal, result = this.lastResult) {
    this.lastResult = result;
    if (!result) return;
    const sw = result.streamWidth ?? result.width, sh = result.streamHeight ?? result.height;
    const scaled = result.width !== sw || result.height !== sh ? ` · ${result.width}×${result.height}` : '';
    const what = showOriginal ? 'ORIGINAL SOURCE' : this.current.entry ? 'PRE-ENCODED · WASM DECODED' : 'WASM DECODED';
    const seconds = (tickMs(+this.badge.dataset.tick) / 1000).toFixed(2);
    this.badge.textContent = `${what} · ${result.repeat ? 'REPEAT' : result.key ? 'KEY' : 'INTER'}${scaled} · #${this.count} · ${seconds} s`;
  }
  rateStatus(result) {
    const quantizer = CODECS[this.current.codec].quantizer.name;
    if (this.current.entry) return `Pre-encoded offline · target ${this.current.entry.targetKbps} kbps · whole stream ${this.current.entry.actualKbps} kbps`;
    if (this.current.rateMode !== 'cbr') return `Fixed ${quantizer} ${this.current.qp}`;
    return `Target ${this.current.targetKbps} kbps · ${quantizer} ${result.quantizer} · buffer ${(result.bufferFullness * 100).toFixed(0)}%${result.bufferOverflows ? ` · ${result.bufferOverflows} buffer overruns` : ''}`;
  }
  /** Mean of one recorded field over the recent frames that have it. */
  average(key) {
    const values = this.history.map(f => f[key]).filter(Number.isFinite);
    return values.length ? values.reduce((s, v) => s + v, 0) / values.length : null;
  }
  /** Mean ms per frame each of this view's workers is busy: encoding (none for a pre-encoded stream), and decoding with PSNR and display hand-off. */
  stageTimes() { return { encode: this.average('encode'), decode: this.average('decodeStage') }; }
  /** Count, chart and save one coded frame. */
  record(coded) {
    ++this.count;
    this.history.push({ bytes: coded.packet.byteLength, encode: coded.encodeMs, decode: coded.decodeMs, decodeStage: coded.decodeStageMs, psnr: coded.psnr, key: coded.key });
    if (this.history.length > HISTORY_FRAMES) this.history.shift();
    // A pre-encoded view downloads its whole stream file instead.
    if (this.current.entry) { this.downloadButton.disabled = false; return; }
    // A repeat (frame hold) codes nothing: the saved stream skips it, its
    // IVF timestamps (and an Annex-B player's last picture) cover the time.
    if (coded.repeat) return;
    if (!this.full && this.bytes + coded.packet.byteLength <= MAX_RECORDING_BYTES) {
      this.frames.push({ data: coded.packet, timestamp: coded.timestamp }); this.bytes += coded.packet.byteLength;
      this.downloadButton.disabled = false;
    } else this.full = true;
  }
  drawHistory() {
    const ctx = this.chart.getContext('2d'), { width, height } = this.chart;
    ctx.clearRect(0, 0, width, height);
    const max = Math.max(1, ...this.history.map(f => f.bytes));
    const color = getComputedStyle(this.element).getPropertyValue('--chart');
    this.history.forEach((f, i) => { ctx.fillStyle = f.key ? '#344d40' : color; const h = Math.max(1, f.bytes / max * height); ctx.fillRect(i * width / HISTORY_FRAMES, height - h, width / HISTORY_FRAMES - 1.5, h); });
    this.element.querySelector('.history-max').textContent = `${(max / 1024).toFixed(1)} KiB peak`;
  }
  /** The effective settings and the equivalent CLI command (or, pre-encoded, the encoder's command), when the section is open. */
  describeSettings() {
    if (!this.reproduce.open || !this.current) return;
    if (this.current.source === SOURCES.reference) {
      this.reproduce.querySelector('.effective').textContent = `${this.current.entry.label}, ${this.current.entry.version.version}, run offline on`
        + ` ${this.current.entry.sourceFile} as I420 (tools/web/make-reference-streams.py):`;
      this.reproduce.querySelector('.cli').textContent = this.current.entry.command;
      return;
    }
    const config = this.current, frames = this.frames.length;
    const overrides = config.overrides ? config.overrides.split('\n').join(', ') : 'none';
    this.reproduce.querySelector('.effective').textContent = `Preset ${cliPreset(config.codec, config.preset) ?? 'default'}`
      + `${config.filmGrain ? ' · film grain' : ''} · overrides: ${overrides}. The command takes an I420 file of the`
      + ` ${frames} recorded frames (the page codes its limited-range BT.601 conversion); after a live rate change it shows the latest.`;
    this.reproduce.querySelector('.cli').textContent = cliCommand(config, frames);
  }
  save() {
    if (this.current.entry) {
      if (this.stream) download(new Blob([this.stream.bytes]), this.current.entry.file);
      return;
    }
    if (!this.frames.length) return;
    const { codec, width, height } = this.current;
    download(streamBlob(codec, width, height, this.frames), `nanocodec-${this.letter.toLowerCase()}-${codec}.${streamExtension(codec)}`);
  }
}
