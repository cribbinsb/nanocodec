import { download, streamBlob, MAX_RECORDING_BYTES } from './streams.js';
export class Panel {
  constructor(letter, codec, container, changed) {
    this.letter = letter; this.frames = []; this.history = []; this.bytes = 0; this.full = false;
    this.element = document.createElement('article'); this.element.className = 'version';
    this.element.innerHTML = `
      <div class="version-top"><span class="version-letter">${letter}</span><h2>Version ${letter}</h2><select class="codec-picker" aria-label="Version ${letter} codec"><option value="h264">H.264</option><option value="hevc">HEVC</option><option value="av1">AV1</option><option value="vp9">VP9 · experimental</option><option value="av2">AV2 · experimental</option></select></div>
      <div class="viewport"><canvas aria-label="Version ${letter} decoded output"></canvas><div class="empty-overlay"><div class="empty-symbol">◉</div><strong>Your decoded video appears here</strong><span>Choose a source and start comparing</span></div><span class="frame-badge">WASM DECODER · READY</span></div>
      <div class="options"><label class="compact-control">Rate mode<select class="rate-mode"><option value="cbr" selected>Target bitrate</option><option value="qp">Fixed quantizer</option></select></label><div class="rate-options"><label class="compact-control">Target (kbps)<input class="target-rate" type="number" required min="10" max="100000" step="10" value="300"></label><label class="compact-control">Peak (kbps)<input class="peak-rate" type="number" required min="10" max="100000" step="10" value="600"></label><label class="compact-control">Buffer<select class="buffer-ms"><option value="250">250 ms</option><option value="500">500 ms</option><option value="1000" selected>1 second</option><option value="2000">2 seconds</option></select></label></div><div class="quality-option" hidden><label class="quality-label" for="qp-${letter}">Compression / quality <output></output></label><input id="qp-${letter}" type="range" step="1"><div class="range-ends"><span>More detail · larger stream</span><span>Smaller stream</span></div></div><label class="compact-control">Encoder preset<select class="preset"><option value="default">Default</option><option value="fast">Fast · quality tradeoff</option></select></label><label class="compact-control">Keyframe interval<select class="gop"><option value="1">1 second</option><option value="2">2 seconds</option><option value="5">5 seconds</option><option value="10">10 seconds</option><option value="30" selected>30 seconds</option></select></label></div>
      <div class="metrics"><div class="metric"><span>Bitrate</span><p><b data-metric="bitrate">—</b><small>kbps</small></p></div><div class="metric"><span>Encode</span><p><b data-metric="encode">—</b><small>ms</small></p></div><div class="metric"><span>Decode</span><p><b data-metric="decode">—</b><small>ms</small></p></div><div class="metric"><span>PSNR-YUV</span><p><b data-metric="psnr">—</b><small>dB</small></p></div></div>
      <p class="rate-status" aria-live="off">Target 300 kbps · waiting for frames</p><div class="history"><div class="history-label"><span>FRAME SIZE · LAST 90 FRAMES</span><span class="history-max">—</span></div><canvas width="540" height="42" aria-label="Recent compressed frame sizes"></canvas></div>
      <div class="version-bottom"><span class="frame-count">No frames encoded</span><button class="download" disabled>Download stream ↓</button></div>`;
    container.append(this.element);
    this.element.querySelector(".options").before(this.element.querySelector(".metrics"));
    this.codec = this.element.querySelector('.codec-picker'); this.codec.value = codec;
    this.qp = this.element.querySelector('input[type=range]'); this.preset = this.element.querySelector('.preset');
    this.gop = this.element.querySelector('.gop'); this.canvas = this.element.querySelector('.viewport canvas');
    this.context = this.canvas.getContext('2d'); this.badge = this.element.querySelector('.frame-badge');
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
  updateCodec() {
    this.preset.disabled = ['vp9', 'av2'].includes(this.codec.value);
    if (this.preset.disabled) this.preset.value = 'default';
    this.qp.min = this.codec.value === 'av1' ? '1' : '0';
    this.qp.max = ['av1', 'vp9', 'av2'].includes(this.codec.value) ? '255' : '51';
    this.qp.value = ['av1', 'vp9', 'av2'].includes(this.codec.value) ? '180' : '32'; this.updateLabel();
  }
  updateLabel() { this.element.querySelector('output').textContent = `${['av1', 'vp9', 'av2'].includes(this.codec.value) ? 'qindex' : 'QP'} ${this.qp.value}`; }
  config(width, height, fps) { return { codec: this.codec.value, width, height, fps, qp: +this.qp.value, fast: this.preset.value === 'fast', keyInterval: +this.gop.value * fps, rateMode: this.rateMode.value, targetKbps: +this.targetRate.value, peakKbps: +this.peakRate.value, bufferMs: +this.bufferMs.value }; }
  reset(config) {
    this.current = config; this.frames = []; this.history = []; this.bytes = 0; this.full = false; this.count = 0;
    this.element.querySelector('.rate-status').textContent = config.rateMode === 'cbr' ? `Target ${config.targetKbps} kbps · waiting for frames` : 'Fixed quantizer';
    this.latest = null; this.element.classList.remove('error'); this.downloadButton.disabled = true;
    this.canvas.width = config.width; this.canvas.height = config.height;
    this.element.querySelector('.empty-overlay').hidden = false;
    this.badge.textContent = 'WASM DECODER · STARTING';
    for (const e of this.element.querySelectorAll('[data-metric]')) e.textContent = '—';
    this.element.querySelector('.frame-count').textContent = 'No frames encoded';
    this.chart.getContext('2d').clearRect(0, 0, this.chart.width, this.chart.height);
  }
  frame(result, showOriginal, original) {
    if (result.width !== this.current.width || result.height !== this.current.height) throw new Error('Decoder output dimensions differ from the input');
    this.latest = new ImageData(new Uint8ClampedArray(result.rgba.buffer), result.width, result.height);
    this.context.putImageData(showOriginal ? original : this.latest, 0, 0);
    this.element.querySelector('.empty-overlay').hidden = true;
    this.badge.textContent = `${showOriginal ? 'ORIGINAL SOURCE' : 'WASM DECODED'} · ${result.key ? 'KEY' : 'INTER'} · #${++this.count}`;
    this.history.push({ bytes: result.packet.byteLength, encode: result.encodeMs, decode: result.decodeMs, psnr: result.psnr, key: result.key });
    if (this.history.length > 90) this.history.shift();
    if (!this.full && this.bytes + result.packet.byteLength <= MAX_RECORDING_BYTES) {
      this.frames.push({ data: result.packet, timestamp: result.timestamp }); this.bytes += result.packet.byteLength;
      this.downloadButton.disabled = false;
    } else this.full = true;
    const average = key => this.history.reduce((s, f) => s + f[key], 0) / this.history.length;
    for (const [key, value] of Object.entries({ bitrate: average('bytes') * 8 * this.current.fps / 1000, encode: average('encode'), decode: average('decode'), psnr: average('psnr') })) {
      this.element.querySelector(`[data-metric="${key}"]`).textContent = value.toFixed(key === 'bitrate' ? 0 : 1);
    }
    this.element.querySelector('.frame-count').textContent = `${this.count} frames · ${(this.bytes / 1024).toFixed(0)} KiB saved${this.full ? ' · buffer full' : ''}`;
    this.element.querySelector('.rate-status').textContent = this.current.rateMode === 'cbr'
      ? `Target ${this.current.targetKbps} kbps · ${['av1', 'vp9', 'av2'].includes(this.current.codec) ? 'qindex' : 'QP'} ${result.quantizer} · buffer ${(result.bufferFullness * 100).toFixed(0)}%${result.bufferOverflows ? ` · ${result.bufferOverflows} buffer overruns` : ''}`
      : `Fixed ${['av1', 'vp9', 'av2'].includes(this.current.codec) ? 'qindex' : 'QP'} ${this.current.qp}`;
    this.drawHistory();
  }
  drawHistory() {
    const ctx = this.chart.getContext('2d'), { width, height } = this.chart;
    ctx.clearRect(0, 0, width, height);
    const max = Math.max(1, ...this.history.map(f => f.bytes));
    const color = getComputedStyle(this.element).getPropertyValue('--chart');
    this.history.forEach((f, i) => { ctx.fillStyle = f.key ? '#344d40' : color; const h = Math.max(1, f.bytes / max * height); ctx.fillRect(i * width / 90, height - h, width / 90 - 1.5, h); });
    this.element.querySelector('.history-max').textContent = `${(max / 1024).toFixed(1)} KiB peak`;
  }
  save() {
    if (!this.frames.length) return;
    const { codec, width, height } = this.current;
    download(streamBlob(codec, width, height, this.frames), `nanocodec-${this.letter.toLowerCase()}-${codec}.${['av1', 'vp9', 'av2'].includes(codec) ? 'ivf' : codec === 'h264' ? '264' : 'hevc'}`);
  }
}
