// One comparison view: what it shows (a live nanocodec encode, a pre-encoded
// reference stream, or the original source uncompressed), its own picture size
// and frame rate, its encoder settings (the common ones visible, the rest under
// Advanced), the picture, its statistics, its CLI command and its download.
import { download, streamBlob, MAX_RECORDING_BYTES } from './streams.js';
import { CODECS, DEFAULT_PRESET, LAYERED_PRESETS, PRESETS, streamExtension } from './codecs.js';
import { createRenderer } from './renderer.js';
import { Advanced } from './advanced.js';
import { cliCommand } from './options.js';
import { FRAME_RATES, SOURCE_FPS, tickAt, tickMs, tickStep } from './timeline.js';
import { fetchStream, keyCadence, loadManifest, secondsLabel, pickerGroups, referenceStream, streamTitle } from './reference.js';
import { trace } from './trace.js';
import { CHART_PRESETS, resultsQuery } from './deep-link.js';
import { BITRATE_FORMULA, PSNR_FORMULA, shownNumbers } from './debug-report.js';
import { BUFFER_CHOICES_MS, DEFAULT_FPS, DEFAULT_RESOLUTION, GOP_SECONDS, KEY_AT_CLIP_START, PEAK_TO_TARGET, RATE_LIMITS_KBPS, RESOLUTIONS, clipStartKeys, overSourceShare } from './view-settings.js';

// Statistics cover this many recent coded frames.
const HISTORY_FRAMES = 90;
// The debug report's record of recent coded frames (debug-report.js): 30 s at 30 fps.
const DEBUG_LOG_FRAMES = 900;
// The metrics, rate line, frame-size chart and settings text are redrawn at most
// this often (and once more after the last frame), not on every frame: they
// write and lay out DOM text, read a computed style and repaint a 90-bar chart.
// Headless Chromium, default page, 3 interleaved loads: Panel.frame 0.72 ->
// 0.56 ms mean (median 0.6 -> 0.4 ms) per view per frame on the page thread.
const METRICS_INTERVAL_MS = 250;
// The key interval of a live view on a source that does not loop (on a clip: KEY_AT_CLIP_START).
const DEFAULT_GOP_SECONDS = 30, DEFAULT_BUFFER_MS = 1000;
// The page source that loops (app.js `kind`), the only one offered KEY_AT_CLIP_START.
const LOOPING_SOURCE = 'clip';
const DEFAULT_TARGET_KBPS = 300;
// An Original view keeps only this many recent pictures for the debug report's
// hashes (a 720p picture is 1.4 MB; coded frames keep DEBUG_LOG_FRAMES).
const ORIGINAL_LOG_FRAMES = 10;
const SOURCES = { live: 'live', reference: 'reference', original: 'original' };
const STILL_CAMERA_TIP = 'For live nanocodec encodes: blocks a still camera only changes by sensor noise are skipped, and frames that are mostly skipped get a smaller budget, so still noisy scenes need fewer bits. On by default for the webcam only, off for clips, the test scene and files (as in the engine\'s presets: --still-camera); on AV1 a long still stream can run slightly over its target.';
// The page source whose live encodes get the still-camera noise handling by default (app.js `kind`).
const STILL_CAMERA_SOURCE = 'camera';
const onOff = on => (on ? 'on' : 'off');
const bufferLabel = ms => (ms < 1000 ? `${ms} ms` : `${ms / 1000} s`);
// Spatial layers (SVC) are a mode beside the speed preset: one layer, or a full-size
// layer over a half-size one (and a quarter-size one), each an operating point.
const SPATIAL_LAYERS = [[1, '1 · full size only'], [2, '2 · full + half size'], [3, '3 · full, half and quarter size']];
const presetLabel = preset => PRESETS.find(([value]) => value === preset)[1].replace(' (default)', '');
const ORIGINAL_STATUS = 'Uncompressed source · nothing is coded';
// Why a control does not apply (it stays in place, disabled: both views keep one layout).
const RATE_ONLY_TIP = 'For target bitrate rate control (Advanced, Rate control)';
const QUANTIZER_ONLY_TIP = 'For fixed quantizer rate control (Advanced, Rate control)';
/** Disable `controls` inside `box` (kept in place, dimmed, with `why` as its tooltip) or enable them. */
function inapplicable(box, controls, off, why) {
  for (const control of controls) control.disabled = off;
  box.classList.toggle('inapplicable', off);
  box.title = off ? why : '';
}
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
      <div class="version-top"><span class="version-letter">${letter}</span><h2>Version ${letter}</h2><div class="segmented view-source" role="group" aria-label="What version ${letter} shows"><button type="button" data-view-source="${SOURCES.live}" aria-pressed="true" title="Encoded and decoded live in your browser by nanocodec">nanocodec live</button><button type="button" data-view-source="${SOURCES.reference}" aria-pressed="false" title="A stream coded offline by another encoder, decoded here">Pre-encoded reference</button><button type="button" data-view-source="${SOURCES.original}" aria-pressed="false" title="The source pictures before any compression">Original (uncompressed)</button></div></div>
      <p class="source-unavailable" hidden>Pre-encoded reference streams exist for the sample clips only.</p>
      <div class="viewport"><canvas aria-label="Version ${letter} picture"></canvas><div class="empty-overlay"><div class="empty-symbol">◉</div><strong>The picture appears here</strong><span>Starting the comparison…</span></div><span class="frame-badge">STARTING</span></div>
      <div class="metrics"><div class="metric"><span>Bitrate</span><p><b data-metric="bitrate">—</b><small>kbps</small></p></div><div class="metric"><span>Encode</span><p><b data-metric="encode">—</b><small>ms</small></p></div><div class="metric"><span>Decode</span><p><b data-metric="decode">—</b><small>ms</small></p></div><div class="metric"><span>PSNR-YUV</span><p><b data-metric="psnr">—</b><small>dB</small></p></div></div>
      <div class="picture-options"><label class="compact-control live-only">Codec<select class="codec-picker" aria-label="Version ${letter} codec">${Object.entries(CODECS).map(([value, { label }]) => `<option value="${value}">${label}</option>`).join('')}</select></label><label class="compact-control live-only">Preset<select class="preset">${options(PRESETS, DEFAULT_PRESET)}</select></label><label class="compact-control live-only spatial-layers-control">Spatial layers<select class="spatial-layers">${options(SPATIAL_LAYERS, 1)}</select></label><label class="compact-control">Resolution<select class="resolution">${options(RESOLUTIONS.map(([w, h]) => [`${w}x${h}`, `${w} × ${h}`]), DEFAULT_RESOLUTION.join('x'))}</select></label><label class="compact-control">Frame rate<select class="fps">${options(FRAME_RATES.map(fps => [fps, `${fps} fps`]), DEFAULT_FPS)}</select></label><label class="compact-control live-only target-option">Bitrate (kbps)${rateInput('target-rate', DEFAULT_TARGET_KBPS)}</label><div class="quality-option live-only"><label class="quality-label" for="qp-${letter}">Fixed quality <output></output></label><input id="qp-${letter}" type="range" step="1"><div class="range-ends"><span>More detail · larger stream</span><span>Smaller stream</span></div></div></div>
      <p class="compare-link live-only"><a href="results/">How does this preset compare?</a></p>
      <div class="reference-options" hidden><label class="compact-control">Stream (encoder · codec · size · rate · bitrate)<select class="stream-picker"></select></label><p class="reference-hint" hidden>This clip has no reference streams: the list shows other sample clips’ streams, and choosing one switches the page to that clip.</p><p class="reference-status" role="status"></p><button type="button" class="match">Match nanocodec to this stream</button><p class="match-note">Sets the other version to live nanocodec at this stream’s codec, size, frame rate and bitrate.</p><details class="reference-about"><summary>About this stream</summary><p class="reference-label"></p><dl class="reference-facts"></dl><p class="reference-version"></p></details></div>
      <p class="original-note" hidden>The source pictures at this size and frame rate, uncompressed: exactly what the encoders are given. Nothing is coded, so there is no bitrate or quality to show.</p>
      <p class="rate-status" aria-live="off">Target 300 kbps · waiting for frames</p><div class="history"><div class="history-label"><span>FRAME SIZE · LAST ${HISTORY_FRAMES} FRAMES</span><span class="history-max">—</span></div><canvas width="540" height="42" aria-label="Recent compressed frame sizes"></canvas></div>
      <div class="version-bottom"><span class="frame-count">No frames yet</span><button class="download" disabled>Download stream ↓</button></div>
      <div class="reproduce"><span class="reproduce-label">CLI</span><code class="cli"></code><button type="button" class="copy-command" aria-label="Copy the command line">Copy</button></div>`;
    // The Advanced section's general encoder settings; the codec's own options follow them (advanced.js).
    const general = document.createElement('div'); general.className = 'advanced-general';
    general.innerHTML = `<label class="compact-control">Rate control<select class="rate-mode"><option value="cbr" selected>Target bitrate</option><option value="qp">Fixed quantizer</option></select></label><div class="peak-options"><label class="compact-control">Peak rate (kbps)${rateInput('peak-rate', DEFAULT_TARGET_KBPS * PEAK_TO_TARGET)}</label><label class="compact-control">Rate buffer<select class="buffer-ms">${options(BUFFER_CHOICES_MS.map(ms => [ms, ms < 1000 ? `${ms} ms` : `${ms / 1000} second${ms > 1000 ? 's' : ''}`]), DEFAULT_BUFFER_MS)}</select></label></div><label class="compact-control">Keyframe interval<select class="gop">${options([[KEY_AT_CLIP_START, 'On clip start'], ...GOP_SECONDS.map(s => [s, `${s} second${s > 1 ? 's' : ''}`])], KEY_AT_CLIP_START)}</select></label><label class="compact-control layer-control">Shown layer<select class="operating-point"><option value="0" selected>Full size (operating point 0)</option><option value="1">Half size (operating point 1)</option><option value="2">Quarter size (operating point 2)</option></select></label><label class="toggle" title="${escapeHtml(STILL_CAMERA_TIP)}"><input class="still-camera" type="checkbox">Still-camera noise handling</label>`;
    container.append(this.element);
    const $ = selector => this.element.querySelector(selector);
    // Live nanocodec or a reference encoder's stream (SOURCES), chosen with the buttons.
    this.sourceChoice = SOURCES.live;
    this.sourceButtons = [...this.element.querySelectorAll('.view-source button')];
    // The sample clip the page plays (null for the webcam, a file or the scene: no reference streams).
    this.clip = null;
    this.codec = $('.codec-picker'); this.codec.value = codec;
    this.advanced = new Advanced(letter, codec, changed, general);
    $('.picture-options').after(this.advanced.element);
    this.effective = document.createElement('p'); this.effective.className = 'effective';
    this.advanced.element.append(this.effective);
    this.stillCamera = $('.still-camera');
    // The toggle follows the page source (setSourceKind) until the visitor sets it.
    this.stillCameraSet = false;
    this.stillCamera.addEventListener('change', () => { this.stillCameraSet = true; changed(); });
    this.resolution = $('.resolution'); this.fps = $('.fps');
    this.streamPicker = $('.stream-picker');
    this.qp = $('input[type=range]'); this.preset = $('.preset'); this.spatialLayers = $('.spatial-layers');
    this.gop = $('.gop');
    // The key interval follows the page source (setSourceKind) until the visitor sets it.
    this.gopSet = false;
    this.gop.addEventListener('change', () => { this.gopSet = true; });
    // Which SVC operating point this view decodes: switched while running, without
    // restarting the encoder (pipeline.js passes it to the decode worker).
    this.operatingPoint = $('.operating-point');
    // Going back to the full layer asks the encoder for a key frame (pipeline.js).
    this.operatingPoint.addEventListener('change', () => this.intervene('operating-point', +this.badge.dataset.tick, { operatingPoint: +this.operatingPoint.value }));
    this.compareLink = $('.compare-link a');
    for (const control of [this.preset, this.spatialLayers]) control.addEventListener('change', () => { this.updateChoices(); this.updateCompareLink(); });
    this.reproduce = $('.reproduce'); this.canvas = $('.viewport canvas');
    this.viewport = $('.viewport'); this.useRenderer(display); this.badge = $('.frame-badge');
    this.chart = $('.history canvas'); this.downloadButton = $('.download');
    this.rateMode = $('.rate-mode');
    this.targetRate = $('.target-rate');
    this.peakRate = $('.peak-rate');
    this.bufferMs = $('.buffer-ms');
    // The peak rate follows the target (PEAK_TO_TARGET) until it is set itself.
    this.peakSet = false;
    this.peakRate.addEventListener('change', () => { this.peakSet = true; });
    for (const control of [this.rateMode, this.targetRate, this.peakRate, this.bufferMs]) control.addEventListener('change', () => {
      const invalid = [this.targetRate, this.peakRate].find(input => !input.checkValidity());
      if (invalid) { invalid.reportValidity(); return; }
      if (!this.peakSet) this.peakRate.value = String(Math.min(RATE_LIMITS_KBPS.max, +this.targetRate.value * PEAK_TO_TARGET));
      if (+this.peakRate.value < +this.targetRate.value) this.peakRate.value = this.targetRate.value;
      this.showRateMode();
      changed('rate');
    });
    this.updateCodec();
    this.codec.addEventListener('change', () => { this.updateCodec(); changed(); });
    this.qp.addEventListener('input', () => this.updateLabel());
    for (const control of [this.qp, this.preset, this.spatialLayers, this.gop, this.resolution, this.fps]) control.addEventListener('change', () => changed());
    this.markSource();
    for (const button of this.sourceButtons) button.addEventListener('click', () => { if (button.dataset.viewSource !== this.sourceChoice) this.setSource(button.dataset.viewSource); });
    this.streamPicker.addEventListener('change', () => { this.choose(this.streams.find(s => s.id === this.streamPicker.value)); changed('reference'); });
    $('.copy-command').addEventListener('click', () => navigator.clipboard?.writeText(this.reproduce.querySelector('.cli').textContent));
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
  get preEncoded() { return this.sourceChoice === SOURCES.reference; }
  /** Whether this view shows the original source, uncompressed. */
  get showsOriginal() { return this.sourceChoice === SOURCES.original; }
  /** Choose live or reference (the buttons), then apply it. */
  setSource(source) {
    this.sourceChoice = source;
    this.sourceChanged();
  }
  /**
   * The page now plays sample clip `clip` (null: another source, where only live
   * encodes exist): the reference choice is offered or disabled with the reason,
   * and the stream list puts that clip's streams first.
   */
  setClip(clip) {
    this.clip = clip;
    this.sourceButtons[1].disabled = !clip;
    this.sourceButtons[1].title = clip ? 'A stream coded offline by another encoder, decoded here' : 'Pre-encoded reference streams exist for the sample clips only';
    this.element.querySelector('.source-unavailable').hidden = Boolean(clip);
    if (this.streams.length) this.fillStreamPicker();
  }
  /**
   * The page source is now `kind` (app.js: 'clip', 'scene', 'camera' or 'file'): the
   * still-camera noise handling defaults on for the webcam only, and keys "On clip start"
   * (offered for a clip only) on a clip and every DEFAULT_GOP_SECONDS elsewhere, unless the visitor set them.
   */
  setSourceKind(kind) {
    if (!this.stillCameraSet) this.stillCamera.checked = kind === STILL_CAMERA_SOURCE;
    const loops = kind === LOOPING_SOURCE, clipStart = this.gop.querySelector(`option[value="${KEY_AT_CLIP_START}"]`);
    clipStart.hidden = clipStart.disabled = !loops;
    if (!this.gopSet || (!loops && this.gop.value === KEY_AT_CLIP_START)) this.gop.value = String(loops ? KEY_AT_CLIP_START : DEFAULT_GOP_SECONDS);
  }
  /** The stream picker: the current clip's streams by encoder, then other clips' (choosing one switches the clip). */
  fillStreamPicker() {
    const option = (entry, text) => `<option value="${entry.id}">${escapeHtml(text)}</option>`;
    const { encoders, otherClips } = pickerGroups(this.streams, this.clip);
    this.streamPicker.innerHTML = [
      ...encoders.map(([label, entries]) => `<optgroup label="${escapeHtml(label)}">${entries.map(e => option(e, streamTitle(e))).join('')}</optgroup>`),
      ...otherClips.map(([clip, entries]) => `<optgroup label="${escapeHtml(`Other clip: ${this.clipTitle(clip)} (switches the clip)`)}">${
        entries.map(e => option(e, streamTitle(e))).join('')}</optgroup>`),
    ].join('');
    if (this.reference) this.streamPicker.value = this.reference.id;
    this.element.querySelector('.reference-hint').hidden = encoders.length > 0;
  }
  /** What this view shows, in a sentence (the page's summary line). */
  get summary() {
    const c = this.config();
    const size = `${c.width}×${c.height}, ${c.fps} fps`;
    if (c.source === SOURCES.original) return `the original, uncompressed (${size})`;
    const codec = CODECS[c.codec].label;
    if (c.source === SOURCES.reference) return `${c.entry.label}, pre-encoded ${codec} (${size}, ${c.targetKbps} kbps)`;
    const layers = c.spatialLayers > 1 ? ` with ${c.spatialLayers} spatial layers` : '';
    return `nanocodec ${codec} ${presetLabel(c.preset)}${layers}, encoding live (${size}, ${c.rateMode === 'cbr' ? `${c.targetKbps} kbps` : `fixed ${CODECS[c.codec].quantizer.name} ${c.qp}`})`;
  }
  /** Switch between live, pre-encoded and original; the stream list loads on first use. */
  async sourceChanged() {
    this.showSource();
    if (!this.preEncoded) { this.changed(); return; }
    try { await this.loadStreams(); } catch (error) { this.status(error.message, true); return; }
    this.choose(this.reference ?? this.streams.find(s => s.clip === this.clip) ?? this.streams[0]);
    this.changed('reference');
  }
  /** Show pre-encoded stream `id` (the page's opening state); the caller starts the comparison. */
  async showStream(id) {
    this.sourceChoice = SOURCES.reference; this.showSource();
    await this.loadStreams();
    this.choose(this.streams.find(s => s.id === id));
  }
  /** The manifest of pre-encoded streams (loaded once), into the picker. */
  async loadStreams() {
    if (!this.streams.length) {
      this.status('Loading the list of pre-encoded streams…');
      this.streams = await loadManifest();
      this.status('');
    }
    this.fillStreamPicker();
  }
  /** Select a stream (manifest record); its file loads when the comparison starts. */
  choose(entry) {
    trace('choose', this.letter, entry.id);
    this.reference = entry;
    this.streamPicker.value = entry.id;
    if (this.codec.value !== entry.codec) { this.codec.value = entry.codec; this.updateCodec(); }
    const configuration = entry.chartConfiguration ? 'the chart configuration' : 'not a chart configuration';
    const facts = {
      Encoder: `${entry.label} · ${configuration}`,
      Clip: this.clipTitle(entry.clip), Size: `${entry.width} × ${entry.height}`, 'Frame rate': `${entry.fps} fps`,
      Bitrate: `target ${entry.targetKbps} kbps · actual ${entry.actualKbps} kbps`,
      ...(entry.rateControl ? { 'Rate control': entry.rateControl } : {}),
      'Key frames': `${entry.keyFrames === 1 ? `one, at the start of the ${entry.frames}-frame clip` : `every ${entry.keyInterval} frames`} · shown: ${keyCadence(entry)}`,
      Source: entry.sourceFile,
      ...(entry.notice ? { Licence: entry.notice } : {}),
    };
    const $ = selector => this.element.querySelector(selector);
    $('.reference-label').textContent = 'Encoded offline by this third-party encoder at the real-time settings our charts use (one thread, low delay); its other settings may give better results. Decoded here by nanocodec’s WASM decoder.';
    $('.reference-facts').innerHTML = Object.entries(facts).map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`).join('');
    const { version, source, gpu } = entry.version;
    $('.reference-version').textContent = `Version: ${version}${source?.commit ? ` · source ${source.ref} ${source.commit.slice(0, 12)}` : ''}${gpu ? ` · GPU ${gpu}` : ''}`;
  }
  /**
   * Show the controls of the selected source: a live encode's settings, a
   * pre-encoded stream's picker and facts, or the original's size and rate.
   */
  showSource() {
    const preEncoded = this.preEncoded, original = this.showsOriginal, live = !preEncoded && !original;
    const $$ = selector => this.element.querySelectorAll(selector);
    this.element.querySelector('.reference-options').hidden = !preEncoded;
    this.element.querySelector('.original-note').hidden = !original;
    this.element.querySelector('.picture-options').hidden = preEncoded;
    for (const element of $$('.live-only')) element.hidden = !live;
    this.advanced.element.hidden = !live;
    if (live) this.showRateMode();
    this.markSource();
    this.element.classList.toggle('pre-encoded', preEncoded);
    this.element.classList.toggle('original-view', original);
    if (!preEncoded) this.reference = null;
  }
  /** Show which source is chosen on its buttons. */
  markSource() {
    for (const button of this.sourceButtons) {
      const selected = button.dataset.viewSource === this.sourceChoice;
      button.setAttribute('aria-pressed', String(selected)); button.classList.toggle('selected', selected);
    }
  }
  /** Pick the first stream of `clip` if the selected one is of another clip; false if there is none (known once the list has loaded). */
  followClip(clip) {
    if (!this.preEncoded || this.reference?.clip === clip) return true;
    // The stream list is still loading (a slow network or machine): keep the choice,
    // sourceChanged picks a stream once it arrives and restarts on it.
    if (!this.streams.length) return true;
    const entry = this.streams.find(s => s.clip === clip);
    if (entry) this.choose(entry);
    return Boolean(entry);
  }
  /** Leave pre-encoded mode (its stream cannot play on the current source); the caller restarts. */
  goLive() { this.sourceChoice = SOURCES.live; this.showSource(); }
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
    this.sourceChoice = SOURCES.live; this.showSource();
    this.resolution.value = `${settings.width}x${settings.height}`; this.fps.value = String(settings.fps);
    if (this.codec.value !== settings.codec) { this.codec.value = settings.codec; this.updateCodec(); }
    this.rateMode.value = settings.rateMode; this.targetRate.value = settings.targetKbps; this.peakRate.value = settings.peakKbps;
    this.bufferMs.value = String(settings.bufferMs); this.gop.value = String(settings.gop); this.gopSet = true;
    this.peakSet = false;
    this.showRateMode();
  }
  /**
   * Apply a demo link's settings (deep-link.js `parseDemoQuery`) to this live view:
   * codec, preset (one spatial layer), size, frame rate and target bitrate, each only where the page offers it.
   */
  applyLink(link) {
    this.sourceChoice = SOURCES.live; this.showSource();
    if (link.codec in CODECS && this.codec.value !== link.codec) { this.codec.value = link.codec; this.updateCodec(); }
    this.spatialLayers.value = '1'; this.updateChoices();
    const preset = this.preset.querySelector(`option[value="${link.preset}"]`);
    if (preset && !preset.disabled) { this.preset.value = link.preset; this.updateChoices(); }
    this.updateCompareLink();
    if (RESOLUTIONS.some(([w, h]) => w === link.width && h === link.height)) this.resolution.value = `${link.width}x${link.height}`;
    if (FRAME_RATES.includes(link.fps)) this.fps.value = String(link.fps);
    if (link.kbps >= RATE_LIMITS_KBPS.min && link.kbps <= RATE_LIMITS_KBPS.max) {
      this.rateMode.value = 'cbr'; this.targetRate.value = String(link.kbps);
      this.peakRate.value = String(Math.min(RATE_LIMITS_KBPS.max, link.kbps * PEAK_TO_TARGET)); this.peakSet = false;
      this.showRateMode();
    }
  }
  /** The link to this codec and preset on the results overview (spatial layers: the overview itself). */
  updateCompareLink() {
    const preset = this.preset.value;
    this.compareLink.href = +this.spatialLayers.value === 1 && CHART_PRESETS.includes(preset)
      ? `results/${resultsQuery({ codec: this.codec.value, preset })}` : 'results/index.html';
  }
  /** Show the original source in this view (a demo link's view A). */
  showOriginal() { this.sourceChoice = SOURCES.original; this.showSource(); }
  /**
   * The bitrate target (with peak and buffer) or the fixed quantizer's slider, by the rate
   * control chosen under Advanced: the other stays in place, disabled, saying why.
   */
  showRateMode() {
    const cbr = this.rateMode.value === 'cbr';
    inapplicable(this.element.querySelector('.target-option'), [this.targetRate], !cbr, RATE_ONLY_TIP);
    inapplicable(this.element.querySelector('.peak-options'), [this.peakRate, this.bufferMs], !cbr, RATE_ONLY_TIP);
    inapplicable(this.element.querySelector('.quality-option'), [this.qp], cbr, QUANTIZER_ONLY_TIP);
  }
  updateCodec() {
    const { presets, spatialLayers, quantizer } = CODECS[this.codec.value];
    this.advanced.setCodec(this.codec.value);
    if (!presets.includes(this.preset.value)) this.preset.value = DEFAULT_PRESET;
    if (+this.spatialLayers.value > 1 && !spatialLayers.includes(+this.spatialLayers.value)) this.spatialLayers.value = '1';
    this.updateChoices();
    this.updateCompareLink();
    this.qp.min = String(quantizer.min); this.qp.max = String(quantizer.max);
    this.qp.value = String(quantizer.initial); this.updateLabel();
  }
  /**
   * The preset and spatial-layer choices the codec offers: the others are hidden.
   * Layered streams are coded only at LAYERED_PRESETS, so each picker disables the
   * other's choices that cannot go with its own.
   */
  updateChoices() {
    const { presets, spatialLayers } = CODECS[this.codec.value];
    const layered = +this.spatialLayers.value > 1, layerable = LAYERED_PRESETS.includes(this.preset.value);
    for (const option of this.preset.options) {
      option.hidden = !presets.includes(option.value);
      option.disabled = option.hidden || (layered && !LAYERED_PRESETS.includes(option.value));
    }
    for (const option of this.spatialLayers.options) {
      const count = +option.value;
      option.hidden = count > 1 && !spatialLayers.includes(count);
      option.disabled = option.hidden || (count > 1 && !layerable);
    }
    inapplicable(this.element.querySelector('.spatial-layers-control'), [this.spatialLayers], spatialLayers.length === 0,
      `${CODECS[this.codec.value].label} has no spatial layers here: AV1 and VP9 do`);
    this.updateLayers();
  }
  /** The shown-layer picker shows with spatial layers only (other streams have one operating point);
   * the quarter-size operating point exists in three-layer streams only. */
  updateLayers() {
    const layered = +this.spatialLayers.value > 1, threeLayers = +this.spatialLayers.value === 3;
    inapplicable(this.element.querySelector('.layer-control'), [this.operatingPoint], !layered, 'Choose 2 or 3 spatial layers to pick the layer shown');
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
    if (this.showsOriginal) return { source: SOURCES.original, width, height, fps, step: tickStep(fps) };
    return { source: SOURCES.live, codec: this.codec.value, width, height, fps, step: tickStep(fps), qp: +this.qp.value,
      preset: this.preset.value, spatialLayers: +this.spatialLayers.value, ...this.advanced.settings(), ...this.keySettings(fps), ...this.rateSettings(),
      stillCamera: this.stillCamera.checked };
  }
  /**
   * The key interval: seconds, or KEY_AT_CLIP_START (`keyAtClipStart`), whose interval is
   * known once the clip's loop is (`atLoop`; until then DEFAULT_GOP_SECONDS, which no encoder starts with).
   */
  keySettings(fps) {
    if (this.gop.value !== KEY_AT_CLIP_START) return { keyInterval: +this.gop.value * fps, keyAtClipStart: false, loopTicks: null };
    return { keyInterval: DEFAULT_GOP_SECONDS * fps, keyAtClipStart: true, loopTicks: null };
  }
  /** A live view's settings once the clip's loop is known (`loopTicks` ticks): keys at each clip start. */
  atLoop(config, loopTicks) {
    if (config.source !== SOURCES.live || !config.keyAtClipStart) return config;
    return { ...config, loopTicks, keyInterval: clipStartKeys(loopTicks, config.step, config.fps).keyInterval };
  }
  /** The live rate settings, which change without a restart. */
  rateSettings() {
    return { rateMode: this.rateMode.value, targetKbps: +this.targetRate.value, peakKbps: +this.peakRate.value, bufferMs: +this.bufferMs.value };
  }
  reset(config) {
    this.current = config; this.frames = []; this.history = []; this.bytes = 0; this.full = false; this.count = 0;
    // Ticks of this view's rate the source skipped, and the last tick it got.
    this.skipped = 0; this.lastTick = null;
    // For the debug report: recent coded frames with their packets, and what changed the live encoder mid-run.
    this.log = []; this.interventions = [];
    // Ticks of the frames coded as key frames on request, and of the first frames
    // coded at changed rate settings ({ tick, ...settings }), for the whole run (debug report).
    this.requestedKeys = []; this.rateChanges = [];
    // The rate settings the run started with (`current` follows live rate changes).
    this.initialRate = { rateMode: config.rateMode, targetKbps: config.targetKbps, peakKbps: config.peakKbps, bufferMs: config.bufferMs };
    this.element.querySelector('.rate-status').textContent = config.source === SOURCES.original ? ORIGINAL_STATUS
      : (config.source === SOURCES.reference ? 'Pre-encoded offline · waiting for frames'
        : config.rateMode === 'cbr' ? `Target ${config.targetKbps} kbps · waiting for frames` : 'Fixed quantizer') + this.settingsNote();
    this.shown = false; this.element.classList.remove('error'); this.downloadButton.disabled = true;
    this.canvas.width = config.width; this.canvas.height = config.height;
    this.element.querySelector('.empty-overlay').hidden = false;
    this.badge.textContent = 'STARTING';
    delete this.badge.dataset.tick;
    for (const e of this.element.querySelectorAll('[data-metric]')) e.textContent = '—';
    this.element.querySelector('.frame-count').textContent = 'No frames yet';
    this.chart.getContext('2d').clearRect(0, 0, this.chart.width, this.chart.height);
    clearTimeout(this.metricsTimer); this.metricsTimer = 0; this.metricsShownAt = -Infinity;
    this.reproduce.hidden = config.source === SOURCES.original;
    this.reproduce.querySelector('.reproduce-label').textContent = config.source === SOURCES.reference ? 'Encoder command' : 'CLI';
    this.describeSettings();
  }
  /** Show the frame decoded for source tick `tick`; returns a pixel buffer the worker may reuse. */
  frame(result, original, tick) {
    const sw = result.streamWidth ?? result.width, sh = result.streamHeight ?? result.height;
    if (sw !== this.current.width || sh !== this.current.height || !result.width || !result.height || result.width > sw || result.height > sh) {
      throw new Error('Decoder output dimensions differ from the input');
    }
    const free = this.renderer.show(result);
    this.shown = true; this.original = original;
    this.element.querySelector('.empty-overlay').hidden = true;
    // `earlier`: frames coded with this one (start-up lookahead, or stream frames a capture skipped).
    for (const coded of [...(result.earlier ?? []), result]) this.record(coded);
    this.analysis?.update(result, tick);
    this.badge.dataset.tick = String(tick);
    this.label(result);
    this.scheduleMetrics(result);
    return free;
  }
  /**
   * Show `result` (the picture on screen, decoded again by a replaced decoder:
   * pipeline.js `switchDecoders`) with its records; nothing is counted, and the
   * original and tick stay. Returns a pixel buffer the worker may reuse.
   */
  redisplay(result) {
    const free = this.renderer.show(result);
    this.analysis?.update(result, +this.badge.dataset.tick);
    this.label(result);
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
    // The original codes nothing: none of the numbers exist.
    const original = this.current.source === SOURCES.original;
    const encode = this.current.entry ? null : this.average('encode');
    const values = original ? { bitrate: null, encode: null, decode: null, psnr: null }
      : { bitrate: this.bitrate(), encode, decode: this.average('decode'), psnr: this.average('psnr') };
    for (const [key, value] of Object.entries(values)) {
      this.element.querySelector(`[data-metric="${key}"]`).textContent = value === null ? 'n/a' : value.toFixed(key === 'bitrate' ? 0 : 1);
    }
    this.element.querySelector('.frame-count').textContent = original ? `${this.count} source frames shown${this.skipNote()}`
      : this.current.entry ? `${this.count} stream frames decoded · stream ${(this.current.entry.bytes / 1024).toFixed(0)} KiB`
        : `${this.count} frames · ${(this.bytes / 1024).toFixed(0)} KiB saved${this.full ? ' · buffer full' : ''}${this.skipNote()}`;
    this.element.querySelector('.rate-status').textContent = this.rateStatus(result);
    this.drawHistory();
    this.describeSettings();
  }
  /** The badge: what is shown, the frame type, its number in this view and its source time. */
  label(result) {
    const sw = result.streamWidth ?? result.width, sh = result.streamHeight ?? result.height;
    const scaled = result.width !== sw || result.height !== sh ? ` · ${result.width}×${result.height}` : '';
    const seconds = (tickMs(+this.badge.dataset.tick) / 1000).toFixed(2);
    if (this.current.source === SOURCES.original) { this.badge.textContent = `ORIGINAL · UNCOMPRESSED · #${this.count} · ${seconds} s`; return; }
    const what = this.current.entry ? 'PRE-ENCODED · WASM DECODED' : 'WASM DECODED';
    this.badge.textContent = `${what} · ${result.repeat ? 'REPEAT' : result.key ? 'KEY' : 'INTER'}${scaled} · #${this.count} · ${seconds} s`;
  }
  rateStatus(result) {
    if (this.current.source === SOURCES.original) return ORIGINAL_STATUS;
    const quantizer = CODECS[this.current.codec].quantizer.name, note = this.settingsNote();
    if (this.current.entry) return `Pre-encoded offline · target ${this.current.entry.targetKbps} kbps · whole stream ${this.current.entry.actualKbps} kbps${note}`;
    if (this.current.rateMode !== 'cbr') return `Fixed ${quantizer} ${this.current.qp}${note}`;
    return `Target ${this.current.targetKbps} kbps · ${quantizer} ${result.quantizer} · buffer ${(result.bufferFullness * 100).toFixed(0)}%${result.bufferOverflows ? ` · ${result.bufferOverflows} buffer overruns` : ''}${note}`;
  }
  /**
   * The settings a live and a pre-encoded view could otherwise differ in unseen, for
   * the rate line: when key frames come, the rate buffer (and a live view's peak) and
   * the still-camera noise handling. Size, frame rate and codec are in the summary line.
   */
  settingsNote() {
    const c = this.current;
    const target = c.entry ? c.entry.targetKbps : c.rateMode === 'cbr' ? c.targetKbps : null;
    const source = target !== null && overSourceShare(target, c.sourceKbps)
      ? ` · over half the clip file's ${c.sourceKbps.toFixed(0)} kbps: its own coding may show` : '';
    if (c.entry) return ` · ${keyCadence(c.entry)} · ${bufferLabel(c.entry.bufferMs)} buffer · still-camera n/a${source}`;
    const rate = c.rateMode === 'cbr' ? ` · peak ${c.peakKbps} kbps, ${bufferLabel(c.bufferMs)} buffer` : '';
    return ` · ${this.liveKeys()}${rate} · still-camera ${onOff(c.stillCamera)}${source}`;
  }
  /** When a live view's keys come, as its rate line says. */
  liveKeys() {
    const c = this.current;
    if (!c.keyAtClipStart) return `key every ${c.keyInterval / c.fps} s`;
    if (!c.loopTicks) return 'key at each clip start';
    const { extraSeconds } = clipStartKeys(c.loopTicks, c.step, c.fps);
    return `key at each clip start, every ${secondsLabel(c.loopTicks / SOURCE_FPS)}${extraSeconds ? ` (and every ${extraSeconds} s: the loop is not a whole number of frames at ${c.fps} fps)` : ''}`;
  }
  /**
   * Kilobits per second of source time over the recent frames: their bytes over
   * the time they cover (first to last timestamp plus one frame interval). Every
   * coded frame counts, so where frames were skipped (a live source the page
   * could not keep up with) the rate is per real second, not per frame x fps.
   */
  bitrate() {
    if (!this.history.length) return null;
    const bytes = this.history.reduce((sum, f) => sum + f.bytes, 0);
    const seconds = (this.history.at(-1).timestamp - this.history[0].timestamp + 1000 / this.current.fps) / 1000;
    return bytes * 8 / seconds / 1000;
  }
  /** For a live view that skipped frames: how many, and the rate it delivers against its configured one. */
  skipNote() {
    if (!this.skipped || this.history.length < 2) return '';
    const seconds = (this.history.at(-1).timestamp - this.history[0].timestamp) / 1000;
    return ` · ${this.skipped} skipped: configured ${this.current.fps} fps, delivering ${((this.history.length - 1) / seconds).toFixed(1)}`;
  }
  /**
   * A frame of tick `tick` went to this view: count the ticks of its rate since
   * the previous one that it never got (a live source the page could not keep up
   * with skips them; an exact source never does).
   */
  sent(tick) {
    if (this.lastTick !== null) this.skipped += Math.max(0, Math.floor((tick - this.lastTick) / this.current.step) - 1);
    this.lastTick = tick;
  }
  /** Mean of one recorded field over the recent frames that have it. */
  average(key) {
    const values = this.history.map(f => f[key]).filter(Number.isFinite);
    return values.length ? values.reduce((s, v) => s + v, 0) / values.length : null;
  }
  /** Mean ms per frame each of this view's workers is busy: encoding (none for a pre-encoded stream), and decoding with PSNR and display hand-off. */
  stageTimes() { return { encode: this.average('encode'), decode: this.average('decodeStage') }; }
  /**
   * This view for the debug report (debug-report.js `buildReport`): its effective
   * configuration as the workers got it, its stream, every number it shows, the
   * window its bitrate and PSNR are computed over, interventions and coded frames.
   */
  debugState() {
    const { entry, ...config } = this.current;
    return {
      letter: this.letter, config: { ...config, operatingPoint: +this.operatingPoint.value, analysis: Boolean(this.analysis?.enabled) },
      streamId: entry?.id ?? null, streamFile: entry?.file ?? null, streamBytes: this.stream?.bytes ?? null,
      shown: shownNumbers(this.element),
      window: { fps: this.current.fps, bitrateFormula: BITRATE_FORMULA, psnrFormula: PSNR_FORMULA,
        frames: this.history.map(({ bytes, timestamp, psnr, encode, decode }) => ({ bytes, timestamp, psnr, encodeMs: encode, decodeMs: decode })) },
      initialRate: this.initialRate, interventions: this.interventions, requestedKeys: this.requestedKeys, rateChanges: this.rateChanges, log: this.log,
    };
  }
  /** The metrics worker scored the frame of `tick` (decode-worker.js): note it in the log and the statistics. */
  scoredMsSsim({ tick, value }) {
    const entry = this.log.findLast(frame => frame.tick === tick);
    if (entry) entry.msSsim = value;
    this.analysis?.scoredMsSsim(tick, value);
  }
  /** Note something that changed the live encoder mid-run (rate, key frame request), for the debug report. */
  intervene(kind, tick, details = {}) { this.interventions.push({ kind, tick, ...details }); }
  /** Count, chart and save one coded frame (for an Original view, one shown source frame). */
  record(coded) {
    if (this.current.source === SOURCES.original) { this.recordOriginal(coded); return; }
    ++this.count;
    if (coded.keyRequested) this.requestedKeys.push(tickAt(coded.timestamp));
    if (coded.rateApplied) this.rateChanges.push({ tick: tickAt(coded.timestamp), ...coded.rateApplied });
    this.log.push({ tick: tickAt(coded.timestamp), key: coded.key, repeat: coded.repeat, quantizer: coded.quantizer, bytes: coded.packet.byteLength,
      packet: coded.packet, encodeMs: coded.encodeMs, decodeMs: coded.decodeMs, psnr: coded.psnr, planes: coded.errorMap?.frame ?? null,
      colour: coded.colour ?? null });
    if (this.log.length > DEBUG_LOG_FRAMES) this.log.shift();
    this.history.push({ bytes: coded.packet.byteLength, encode: coded.encodeMs, decode: coded.decodeMs, decodeStage: coded.decodeStageMs, psnr: coded.psnr, key: coded.key, timestamp: coded.timestamp });
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
  /**
   * One source frame an Original view showed: counted, timed (the worker's
   * placement) and kept, with its picture, for the debug report's hashes.
   */
  recordOriginal(shown) {
    ++this.count;
    const picture = new Uint8Array(shown.source);
    this.log.push({ tick: tickAt(shown.timestamp), key: false, repeat: false, quantizer: null, bytes: picture.byteLength, packet: picture,
      encodeMs: null, decodeMs: null, psnr: null, planes: null, colour: shown.colour ?? null });
    if (this.log.length > ORIGINAL_LOG_FRAMES) this.log.shift();
    this.history.push({ bytes: 0, encode: null, decode: null, decodeStage: shown.decodeStageMs, psnr: null, key: false, timestamp: shown.timestamp });
    if (this.history.length > HISTORY_FRAMES) this.history.shift();
  }
  drawHistory() {
    const ctx = this.chart.getContext('2d'), { width, height } = this.chart;
    ctx.clearRect(0, 0, width, height);
    const max = Math.max(1, ...this.history.map(f => f.bytes));
    const color = getComputedStyle(this.element).getPropertyValue('--chart');
    this.history.forEach((f, i) => { ctx.fillStyle = f.key ? '#344d40' : color; const h = Math.max(1, f.bytes / max * height); ctx.fillRect(i * width / HISTORY_FRAMES, height - h, width / HISTORY_FRAMES - 1.5, h); });
    this.element.querySelector('.history-max').textContent = `${(max / 1024).toFixed(1)} KiB peak`;
  }
  /**
   * The command line that reproduces this view offline: the equivalent
   * `nanocodec encode` (its effective settings under Advanced), or the
   * reference encoder's command (how it was run in the line's tooltip).
   */
  describeSettings() {
    if (!this.current || this.current.source === SOURCES.original) return;
    const cli = this.reproduce.querySelector('.cli');
    if (this.current.source === SOURCES.reference) {
      const { entry } = this.current;
      cli.title = `${entry.label}, ${entry.version.version}, run offline on ${entry.sourceFile} as I420 (tools/web/make-reference-streams.py)`;
      cli.textContent = entry.command;
      return;
    }
    const config = this.current, frames = this.frames.length;
    const overrides = config.overrides ? config.overrides.split('\n').join(', ') : 'none';
    this.effective.textContent = `Effective: preset ${config.preset}${config.spatialLayers > 1 ? ` · ${config.spatialLayers} spatial layers` : ''}`
      + `${config.filmGrain ? ' · film grain' : ''} · overrides: ${overrides}. The CLI command below the view takes an I420 file of the`
      + ` ${frames} recorded frames (for a sample clip, ffmpeg's decode of its file at this size); after a live rate change it shows the latest.`;
    cli.title = '';
    // The colour of the frames the encoder last coded (the source decides it, colour.js).
    cli.textContent = cliCommand(config, frames, this.log.findLast(frame => frame.colour)?.colour);
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
