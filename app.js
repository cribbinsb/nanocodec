import { Panel } from './panel.js';
import { Loupe } from './loupe.js';
import { drawScene } from './scene.js';
import { CLIPS, CLIP_GROUPS, DEFAULT_CLIP, clipById, clipUrl } from './clips.js';

const $ = selector => document.querySelector(selector);
const pauseButton = $('#pause');
const startButton = $('#start'), video = $('#input-video'), notice = $('#notice');
const surface = document.createElement('canvas'), capture = surface.getContext('2d', { willReadFrequently: true });
const preview = $('#source-preview').getContext('2d');
let kind = 'clip', clip = DEFAULT_CLIP, file = null, fileUrl = null, loadedSource = null, camera = null;
let running = false, preparing = false, generation = 0, workers = [], timer = 0, restartTimer = 0;
let frameId = 0, results = [null, null], sourceBuffers = [null, null], original = null, showOriginal = false, delivered = 0, started = 0;
let pendingRateChange = false;
let paused = false, pausedAt = 0, inFlight = false;
let settings, nextDeadline = 0, firstDeliveredAt = 0;
const workerTimeouts = new Set();
// `?display=cpu|webgl|webgl2` overrides the frame renderer's automatic choice (renderer.js), e.g. to compare the paths.
const display = new URLSearchParams(location.search).get('display') || 'auto';
const panels = [new Panel('A', 'h264', $('#versions'), settingsChanged, display), new Panel('B', 'av1', $('#versions'), settingsChanged, display)];
const loupe = new Loupe($('#loupe'), panels);

function message(text = '') { notice.textContent = text; notice.hidden = !text; }
function stop(keepSource = false) {
  ++generation; clearTimeout(timer); clearTimeout(restartTimer);
  for (const timeout of workerTimeouts) clearTimeout(timeout); workerTimeouts.clear();
  for (const worker of workers) worker.terminate(); workers = [];
  running = false; preparing = false; paused = false; inFlight = false;
  pauseButton.disabled = true; pauseButton.textContent = 'Pause';
  pauseButton.setAttribute('aria-pressed', 'false');
  results = [null, null]; sourceBuffers = [null, null];
  $("#keyframe").disabled = true;
  document.body.classList.remove('running'); startButton.innerHTML = 'Start comparison <span aria-hidden="true">↗</span>';
  $('#runtime-label').textContent = 'WASM encode + decode';
  if (!keepSource) {
    video.pause();
    if (camera) { for (const track of camera.getTracks()) track.stop(); camera = null; video.srcObject = null; }
  }
}
function pause() {
  if (!running || paused) return;
  paused = true; pausedAt = performance.now(); clearTimeout(timer);
  if (kind !== 'scene') video.pause();
  pauseButton.textContent = 'Resume'; pauseButton.setAttribute('aria-pressed', 'true');
  $('#runtime-label').textContent = 'Paused';
}
async function resume() {
  if (!running || !paused) return;
  const token = generation;
  pauseButton.disabled = true;
  try {
    if (kind !== 'scene') await video.play();
    if (token !== generation) return;
    const now = performance.now(), gap = now - pausedAt;
    started += gap;
    if (delivered) firstDeliveredAt += gap;
    paused = false; nextDeadline = now;
    pauseButton.disabled = false; pauseButton.textContent = 'Pause';
    pauseButton.setAttribute('aria-pressed', 'false');
    $('#runtime-label').textContent = 'Encoding + decoding live';
    if (!inFlight) scheduleFrame(token);
  } catch (error) { if (token === generation) fail(error); }
}
function fail(error) {
  stop();
  const detail = error instanceof Error ? error.message : String(error);
  message(detail); $('#session-stats').textContent = 'Comparison stopped';
}
function settingsChanged(change) {
  if (change === 'rate' && running) { pendingRateChange = true; return; }
  if (!running && !preparing) { resetViews(); $("#session-stats").textContent = "Settings changed · start a new comparison"; return; }
  clearTimeout(restartTimer);
  restartTimer = setTimeout(() => { stop(true); start(); }, 200);
}
function resetViews() {
  const [width, height] = $("#resolution").value.split("x").map(Number), fps = +$("#fps").value;
  for (const panel of panels) panel.reset(panel.config(width, height, fps));
  original = null; loupe.frameShown(null);
}
async function metadata() {
  if (video.readyState >= 1) return;
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(new Error('The video did not load. Try a file supported by your browser.')), 15000);
    const loaded = () => finish();
    const failed = () => finish(new Error('Your browser could not open this video. Try an MP4 or WebM file.'));
    function finish(error) {
      clearTimeout(timeout); video.removeEventListener('loadedmetadata', loaded); video.removeEventListener('error', failed);
      if (error) reject(error); else resolve();
    }
    video.addEventListener('loadedmetadata', loaded, { once: true }); video.addEventListener('error', failed, { once: true });
  });
}
async function prepareSource(token) {
  if (kind === 'scene') return;
  if (kind === 'camera') {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS or localhost and a browser with camera support. You can still use the test scene or a video file.');
    if (!camera) {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      if (token !== generation) { stream.getTracks().forEach(t => t.stop()); return; }
      camera = stream; loadedSource = null; video.srcObject = stream;
      for (const track of stream.getTracks()) track.addEventListener('ended', () => { if (camera === stream) fail('The webcam disconnected or camera access was stopped.'); });
    }
  } else if (kind === 'clip') {
    // The URL is the source identity: a resolution change that picks another variant reloads.
    const url = clipUrl(clip, settings.height);
    loadVideo(url, url);
  } else {
    if (!file) throw new Error('Choose a video file first.');
    if (loadedSource !== file) {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
      fileUrl = URL.createObjectURL(file);
    }
    loadVideo(file, fileUrl);
    if (video.ended) video.currentTime = 0;
  }
  await metadata();
  if (token !== generation) return;
  try { await video.play(); } catch (error) {
    // Muted inline video may autoplay, but a browser can still refuse until the user interacts.
    if (error.name === 'NotAllowedError') throw new Error('The browser blocked automatic playback. Press Start comparison to play the clip.');
    throw error;
  }
}
// Sample clips loop so the comparison keeps running; local files stop at their end.
function loadVideo(source, url) {
  if (loadedSource === source) return;
  loadedSource = source; video.srcObject = null; video.loop = kind === 'clip'; video.src = url; video.load();
}
function initWorker(panel, index, token) {
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }); workers.push(worker);
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('The WASM worker did not start. Check the build and browser console.')), 30000);
    workerTimeouts.add(timeout);
    worker.onerror = event => {
      clearTimeout(timeout);
      const error = new Error(event.message || 'Could not load the WASM codec. Build the demo with tools/web/build.sh, then serve the web directory over HTTP.');
      if (preparing) reject(error); else if (token === generation) fail(error);
    };
    worker.onmessage = ({ data }) => {
      if (token !== generation) return;
      if (data.type === 'ready') { clearTimeout(timeout); workerTimeouts.delete(timeout); resolve(); }
      else if (data.type === 'error') {
        clearTimeout(timeout); panel.element.classList.add('error');
        const error = new Error(`Version ${panel.letter}: ${data.message}`);
        if (preparing) reject(error); else fail(error);
      } else if (data.type === 'frame' && data.id === frameId && running) {
        results[index] = data;
        if (results.every(Boolean)) finishFrame(token);
      }
    };
    worker.postMessage({ type: 'init', config: panel.current });
  });
}
async function start() {
  stop(true); const token = generation;
  preparing = true; message(); startButton.textContent = 'Cancel'; $('#session-stats').textContent = 'Loading codec workers…';
  const [width, height] = $('#resolution').value.split('x').map(Number), fps = +$('#fps').value;
  settings = { width, height, fps }; surface.width = width; surface.height = height;
  for (const panel of panels) panel.reset(panel.config(width, height, fps));
  loupe.frameShown(null);
  try {
    await prepareSource(token);
    if (token !== generation) return;
    await Promise.all(panels.map((p, i) => initWorker(p, i, token)));
    if (token !== generation) return;
    pendingRateChange = false;
    $("#keyframe").disabled = false; pauseButton.disabled = false;
    running = true; preparing = false; delivered = 0; firstDeliveredAt = 0; frameId = 0; started = performance.now(); nextDeadline = started;
    document.body.classList.add('running'); startButton.textContent = 'Stop comparison'; $('#runtime-label').textContent = 'Encoding + decoding live';
    scheduleFrame(token);
  } catch (error) { if (token === generation) fail(error); }
}
function scheduleFrame(token) {
  if (token !== generation || !running || paused) return;
  timer = setTimeout(() => sendFrame(token), Math.max(0, nextDeadline - performance.now()));
}
function sendFrame(token) {
  try { captureFrame(token); } catch (error) { if (token === generation) fail(error); }
}
function captureFrame(token) {
  if (token !== generation || !running || paused) return;
  if (kind !== 'scene' && video.readyState < 2) { nextDeadline = performance.now() + 100; scheduleFrame(token); return; }
  if (kind === 'file' && video.ended) { stop(); $('#session-stats').textContent = `${delivered} frames compared · video ended`; return; }
  const now = performance.now(), { width, height, fps } = settings;
  nextDeadline = Math.max(nextDeadline + 1000 / fps, now);
  if (kind === 'scene') drawScene(capture, width, height, (now - started) / 1000);
  else {
    capture.fillStyle = '#000'; capture.fillRect(0, 0, width, height);
    const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
    const w = video.videoWidth * scale, h = video.videoHeight * scale;
    capture.drawImage(video, (width - w) / 2, (height - h) / 2, w, h);
  }
  original = capture.getImageData(0, 0, width, height);
  preview.drawImage(surface, 0, 0, 160, 90);
  const timestamp = kind === 'file' ? video.currentTime * 1000 : now - started;
  if (pendingRateChange) {
    panels.forEach((panel, i) => {
      panel.current = panel.config(width, height, fps); panel.history = [];
      workers[i].postMessage({ type: 'rate', config: panel.current });
    });
    pendingRateChange = false;
  }
  results = [null, null]; ++frameId; inFlight = true;
  workers.forEach((worker, i) => {
    // Reuse the buffer this worker returned with its last result.
    let rgba = sourceBuffers[i];
    sourceBuffers[i] = null;
    if (!rgba || rgba.byteLength !== original.data.length) rgba = new ArrayBuffer(original.data.length);
    new Uint8ClampedArray(rgba).set(original.data);
    // Each frame names the format the view's renderer draws, which changes if a GL context is lost.
    worker.postMessage({ type: 'frame', id: frameId, timestamp, rgba, display: panels[i].renderer.format }, [rgba]);
  });
}
function finishFrame(token) {
  inFlight = false;
  try {
    panels.forEach((panel, i) => {
      sourceBuffers[i] = results[i].source;
      // A held frame 0 (start-up lookahead) is shown with the next result.
      if (results[i].held) return;
      const replaced = panel.frame(results[i], showOriginal, original);
      if (replaced) workers[i].postMessage({ type: 'recycle', buffer: replaced }, [replaced]);
    });
  }
  catch (error) { fail(error); return; }
  loupe.frameShown(original);
  ++delivered;
  const now = paused ? pausedAt : performance.now();
  if (delivered === 1) firstDeliveredAt = now;
  const deliveryRate = delivered > 1 ? (delivered - 1) * 1000 / Math.max(now - firstDeliveredAt, 1) : 0;
  $('#session-stats').textContent = `${delivered} frames compared · ${deliveryRate.toFixed(1)} fps delivered · bitrate @ ${settings.fps} fps`;
  scheduleFrame(token);
}
function sourceChanged(next) {
  stop(); kind = next; message(); resetViews();
  preview.clearRect(0, 0, 160, 90);
  if (kind === "scene") drawScene(preview, 160, 90, 0);
  for (const button of document.querySelectorAll('[data-source]')) { const selected = button.dataset.source === kind; button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', String(selected)); }
  const names = { clip: `${clip.film} · ${clip.title}`, scene: 'Moving shapes, fine texture and a smooth gradient', camera: 'Webcam · camera access requested when you start', file: file?.name || 'Choose a local video file' };
  $('#source-name').textContent = names[kind];
  clipCredit.hidden = kind !== 'clip'; clipCredit.textContent = clip.credit; clipCredit.title = clip.notice ?? '';
  $('#session-stats').textContent = 'Ready when you are';
}
for (const button of document.querySelectorAll('[data-source]')) button.addEventListener('click', () => { sourceChanged(button.dataset.source); if (kind === 'file') $('#file').click(); });
const clipPicker = $('#clip'), clipCredit = $('#clip-credit');
for (const group of CLIP_GROUPS) {
  const optgroup = document.createElement('optgroup'); optgroup.label = group;
  for (const { id, title } of CLIPS.filter(c => c.group === group)) optgroup.append(new Option(title, id, false, id === clip.id));
  clipPicker.append(optgroup);
}
clipPicker.addEventListener('change', () => { clip = clipById(clipPicker.value); sourceChanged('clip'); start(); });
$('#file').addEventListener('change', event => { const selected = event.target.files[0]; if (selected) { file = selected; sourceChanged('file'); } });
for (const control of [$('#resolution'), $('#fps')]) control.addEventListener('change', settingsChanged);
$('#keyframe').addEventListener('click', () => { if (running) for (const worker of workers) worker.postMessage({ type: 'keyframe' }); });
pauseButton.addEventListener('click', () => { if (paused) resume(); else pause(); });
startButton.addEventListener('click', () => { if (running || preparing) stop(); else start(); });
const originalButton = $('#original');
function toggleOriginal(value) {
  showOriginal = value; originalButton.setAttribute('aria-pressed', String(value));
  if (!original) return;
  for (const panel of panels) if (panel.shown) {
    if (value) panel.renderer.showOriginal(original); else panel.renderer.showLatest();
    panel.badge.textContent = value ? 'ORIGINAL SOURCE' : 'WASM DECODED';
  }
  loupe.frameShown(original);
}
originalButton.addEventListener('pointerdown', e => { originalButton.setPointerCapture(e.pointerId); toggleOriginal(true); });
for (const event of ['pointerup', 'pointercancel', 'lostpointercapture', 'blur']) originalButton.addEventListener(event, () => toggleOriginal(false));
originalButton.addEventListener('keydown', e => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); toggleOriginal(true); } });
originalButton.addEventListener('keyup', () => toggleOriginal(false));
video.addEventListener('error', () => { if (running) fail('The input video could not be read.'); });
window.addEventListener('pagehide', () => { stop(); if (fileUrl) URL.revokeObjectURL(fileUrl); });
document.addEventListener('visibilitychange', () => { if (document.hidden && (running || preparing)) { stop(); message('Comparison paused while this tab is hidden. Start again to resume.'); } });
// Open on a playing sample clip. A page loaded in a background tab starts once it is shown.
function autostart() {
  if (document.hidden) document.addEventListener('visibilitychange', autostart, { once: true });
  else start();
}
sourceChanged('clip');
autostart();
