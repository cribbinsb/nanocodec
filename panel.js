import { download, streamBlob, MAX_RECORDING_BYTES } from './streams.js';
import { CODECS, streamExtension } from './codecs.js';
import { createRenderer } from './renderer.js';
import { Advanced } from './advanced.js';
import { cliCommand, cliPreset } from './options.js';
export class Panel {
  /** `display`: the preferred renderer ('auto', 'webgl2', 'webgl' or 'cpu'; renderer.js). */
  constructor(letter, codec, container, changed, display) {
    this.letter = letter; this.frames = []; this.history = []; this.bytes = 0; this.full = false;
    this.element = document.createElement('article'); this.element.className = 'version';
    this.element.innerHTML = `
      <div class="version-top"><span class="version-letter">${letter}</span><h2>Version ${letter}</h2><select class="codec-picker" aria-label="Version ${letter} codec">${Object.entries(CODECS).map(([value, { label }]) => `<option value="${value}">${label}</option>`).join('')}</select></div>
      <div class="viewport"><canvas aria-label="Version ${letter} decoded output"></canvas><div class="empty-overlay"><div class="empty-symbol">◉</div><strong>Your decoded video appears here</strong><span>Choose a source and start comparing</span></div><span class="frame-badge">WASM DECODER · READY</span></div>
      <div class="options"><label class="compact-control">Rate mode<select class="rate-mode"><option value="cbr" selected>Target bitrate</option><option value="qp">Fixed quantizer</option></select></label><div class="rate-options"><label class="compact-control">Target (kbps)<input class="target-rate" type="number" required min="10" max="100000" step="10" value="300"></label><label class="compact-control">Peak (kbps)<input class="peak-rate" type="number" required min="10" max="100000" step="10" value="600"></label><label class="compact-control">Buffer<select class="buffer-ms"><option value="250">250 ms</option><option value="500">500 ms</option><option value="1000" selected>1 second</option><option value="2000">2 seconds</option></select></label></div><div class="quality-option" hidden><label class="quality-label" for="qp-${letter}">Compression / quality <output></output></label><input id="qp-${letter}" type="range" step="1"><div class="range-ends"><span>More detail · larger stream</span><span>Smaller stream</span></div></div><label class="compact-control">Encoder preset<select class="preset"><option value="default">Default</option><option value="fast">Fast · quality tradeoff</option><option value="faster">Faster · H.264 and AV1</option></select></label><label class="compact-control">Keyframe interval<select class="gop"><option value="1">1 second</option><option value="2">2 seconds</option><option value="5">5 seconds</option><option value="10">10 seconds</option><option value="30" selected>30 seconds</option></select></label></div>
      <div class="metrics"><div class="metric"><span>Bitrate</span><p><b data-metric="bitrate">—</b><small>kbps</small></p></div><div class="metric"><span>Encode</span><p><b data-metric="encode">—</b><small>ms</small></p></div><div class="metric"><span>Decode</span><p><b data-metric="decode">—</b><small>ms</small></p></div><div class="metric"><span>PSNR-YUV</span><p><b data-metric="psnr">—</b><small>dB</small></p></div></div>
      <p class="rate-status" aria-live="off">Target 300 kbps · waiting for frames</p><div class="history"><div class="history-label"><span>FRAME SIZE · LAST 90 FRAMES</span><span class="history-max">—</span></div><canvas width="540" height="42" aria-label="Recent compressed frame sizes"></canvas></div>
      <div class="version-bottom"><span class="frame-count">No frames encoded</span><button class="download" disabled>Download stream ↓</button></div>
      <details class="reproduce"><summary>Settings · CLI command</summary><p class="effective"></p><code class="cli"></code></details>`;
    container.append(this.element);
    this.element.querySelector(".options").before(this.element.querySelector(".metrics"));
    this.codec = this.element.querySelector('.codec-picker'); this.codec.value = codec;
    this.qp = this.element.querySelector('input[type=range]'); this.preset = this.element.querySelector('.preset');
    this.gop = this.element.querySelector('.gop');
    this.advanced = new Advanced(letter, codec, changed); this.element.querySelector('.options').append(this.advanced.element);
    this.reproduce = this.element.querySelector('.reproduce');
    this.reproduce.addEventListener('toggle', () => this.describeSettings()); this.canvas = this.element.querySelector('.viewport canvas');
    this.viewport = this.element.querySelector('.viewport'); this.useRenderer(display); this.badge = this.element.querySelector('.frame-badge');
    this.chart = this.element.querySelector('.history canvas'); this.downloadButton = this.element.querySelector('.download');
    this.rateMode = this.element.querySelector('.rate-mode');
    this.targetRate = this.element.querySelector('.target-rate');
    this.peakRate = this.element.querySelector('.peak-rate');
    this.bufferMs = this.element.querySelector('.buffer-ms');
    for (const control of [this.rateMode, this.targetRate, this.peakRate, this.bufferMs]) control.addEventListener('change', () => {
      const invalid = [this.targetRate, this.peakRate].find(input => !input.checkValidity());
      if (invalid) { invalid.reportValidity(); return; }
      if (+this.peakRate.value < +this.targetRate.value) this.peakRate.value = this.targetRate.value;
      this.element.querySelector('.rate-options').hidden = this.rateMode.value !== 'cbr';
      this.element.querySelector('.quality-option').hidden = this.rateMode.value === 'cbr';
      changed('rate');
    });
    this.updateCodec();
    this.codec.addEventListener('change', () => { this.updateCodec(); changed(); });
    this.qp.addEventListener('input', () => this.updateLabel());
    for (const control of [this.qp, this.preset, this.gop]) control.addEventListener('change', changed);
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
  updateCodec() {
    const { fast, faster, quantizer } = CODECS[this.codec.value];
    this.advanced.setCodec(this.codec.value);
    this.preset.disabled = !fast;
    this.preset.querySelector('option[value="faster"]').disabled = !faster;
    if (this.preset.disabled || (!faster && this.preset.value === 'faster')) this.preset.value = 'default';
    this.qp.min = String(quantizer.min); this.qp.max = String(quantizer.max);
    this.qp.value = String(quantizer.initial); this.updateLabel();
  }
  updateLabel() { this.element.querySelector('output').textContent = `${CODECS[this.codec.value].quantizer.name} ${this.qp.value}`; }
  config(width, height, fps) { return { codec: this.codec.value, width, height, fps, qp: +this.qp.value, preset: this.preset.value, ...this.advanced.settings(), keyInterval: +this.gop.value * fps, rateMode: this.rateMode.value, targetKbps: +this.targetRate.value, peakKbps: +this.peakRate.value, bufferMs: +this.bufferMs.value }; }
  reset(config) {
    this.current = config; this.frames = []; this.history = []; this.bytes = 0; this.full = false; this.count = 0;
    this.element.querySelector('.rate-status').textContent = config.rateMode === 'cbr' ? `Target ${config.targetKbps} kbps · waiting for frames` : 'Fixed quantizer';
    this.shown = false; this.element.classList.remove('error'); this.downloadButton.disabled = true;
    this.canvas.width = config.width; this.canvas.height = config.height;
    this.element.querySelector('.empty-overlay').hidden = false;
    this.badge.textContent = 'WASM DECODER · STARTING';
    for (const e of this.element.querySelectorAll('[data-metric]')) e.textContent = '—';
    this.element.querySelector('.frame-count').textContent = 'No frames encoded';
    this.chart.getContext('2d').clearRect(0, 0, this.chart.width, this.chart.height);
    this.describeSettings();
  }
  /** Show a decoded frame; returns a pixel buffer the worker may reuse. */
  frame(result, showOriginal, original) {
    if (result.width !== this.current.width || result.height !== this.current.height) throw new Error('Decoder output dimensions differ from the input');
    const free = this.renderer.show(result);
    this.shown = true;
    if (showOriginal) this.renderer.showOriginal(original);
    this.element.querySelector('.empty-overlay').hidden = true;
    // `previous`: frame 0, coded together with frame 1 after the start-up lookahead.
    for (const coded of [result.previous, result]) if (coded) this.record(coded);
    this.badge.textContent = `${showOriginal ? 'ORIGINAL SOURCE' : 'WASM DECODED'} · ${result.repeat ? 'REPEAT' : result.key ? 'KEY' : 'INTER'} · #${this.count}`;
    for (const [key, value] of Object.entries({ bitrate: this.average('bytes') * 8 * this.current.fps / 1000, encode: this.average('encode'), decode: this.average('decode'), psnr: this.average('psnr') })) {
      this.element.querySelector(`[data-metric="${key}"]`).textContent = value.toFixed(key === 'bitrate' ? 0 : 1);
    }
    this.element.querySelector('.frame-count').textContent = `${this.count} frames · ${(this.bytes / 1024).toFixed(0)} KiB saved${this.full ? ' · buffer full' : ''}`;
    this.element.querySelector('.rate-status').textContent = this.current.rateMode === 'cbr'
      ? `Target ${this.current.targetKbps} kbps · ${CODECS[this.current.codec].quantizer.name} ${result.quantizer} · buffer ${(result.bufferFullness * 100).toFixed(0)}%${result.bufferOverflows ? ` · ${result.bufferOverflows} buffer overruns` : ''}`
      : `Fixed ${CODECS[this.current.codec].quantizer.name} ${this.current.qp}`;
    this.drawHistory();
    this.describeSettings();
    return free;
  }
  /** Mean of one recorded field over the recent frames. */
  average(key) { return this.history.reduce((s, f) => s + f[key], 0) / this.history.length; }
  /** Mean ms per frame each of this view's workers is busy: encoding, and decoding with PSNR and display hand-off. */
  stageTimes() { return { encode: this.average('encode'), decode: this.average('decodeStage') }; }
  /** Count, chart and save one coded frame. */
  record(coded) {
    ++this.count;
    this.history.push({ bytes: coded.packet.byteLength, encode: coded.encodeMs, decode: coded.decodeMs, decodeStage: coded.decodeStageMs, psnr: coded.psnr, key: coded.key });
    if (this.history.length > 90) this.history.shift();
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
    this.history.forEach((f, i) => { ctx.fillStyle = f.key ? '#344d40' : color; const h = Math.max(1, f.bytes / max * height); ctx.fillRect(i * width / 90, height - h, width / 90 - 1.5, h); });
    this.element.querySelector('.history-max').textContent = `${(max / 1024).toFixed(1)} KiB peak`;
  }
  /** The effective settings and the equivalent CLI command (when the section is open). */
  describeSettings() {
    if (!this.reproduce.open || !this.current) return;
    const config = this.current, frames = this.frames.length;
    const overrides = config.overrides ? config.overrides.split('\n').join(', ') : 'none';
    this.reproduce.querySelector('.effective').textContent = `Preset ${cliPreset(config.codec, config.preset) ?? 'default'}`
      + `${config.filmGrain ? ' · film grain' : ''} · overrides: ${overrides}. The command takes an I420 file of the`
      + ` ${frames} recorded frames (the page codes its limited-range BT.601 conversion); after a live rate change it shows the latest.`;
    this.reproduce.querySelector('.cli').textContent = cliCommand(config, frames);
  }
  save() {
    if (!this.frames.length) return;
    const { codec, width, height } = this.current;
    download(streamBlob(codec, width, height, this.frames), `nanocodec-${this.letter.toLowerCase()}-${codec}.${streamExtension(codec)}`);
  }
}
