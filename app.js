// The comparison page: the source (sample clip, test scene, webcam or file),
// two views (panel.js), the capture loop on the shared source timeline
// (timeline.js, capture.js), and pause, frame stepping and the session line.
//
// Captures run at the timeline's SOURCE_FPS; each view codes the ticks of its
// own frame rate and is given the source drawn at its own resolution. Pause
// stops on the next captured tick that every view codes, and Step moves to the
// next such tick, so paused views always show the same source instant.
import { Panel } from './panel.js';
import { Comparison } from './pipeline.js';
import { Loupe } from './loupe.js';
import { drawScene } from './scene.js';
import { FrameSource } from './capture.js';
import { CLIPS, CLIP_GROUPS, DEFAULT_CLIP, clipById, clipUrl } from './clips.js';
import { SOURCE_FPS, nextCommonTick, tickMs } from './timeline.js';
import { matchingSettings } from './view-settings.js';
import { trace } from './trace.js';

const $ = selector => document.querySelector(selector);
const pauseButton = $('#pause'), stepButton = $('#step');
const startButton = $('#start'), video = $('#input-video'), notice = $('#notice');
const preview = $('#source-preview').getContext('2d');
const PREVIEW_WIDTH = 160, PREVIEW_HEIGHT = 90;
// How long to wait for a video that is not ready to show a frame (ms).
const NOT_READY_RETRY_MS = 100;
// Delivery within this fraction of the target frame rate counts as keeping up.
const KEEPING_UP = 0.95;
let kind = 'clip', clip = DEFAULT_CLIP, file = null, fileUrl = null, loadedSource = null, camera = null;
let running = false, preparing = false, generation = 0, comparison = null, timer = 0, restartTimer = 0;
let showOriginal = false, delivered = 0;
let pendingRateChange = false;
// The last comparison ended in a failure (not a Stop): a settings change starts a new one.
let stoppedByFailure = false;
// A settings change has scheduled a restart that has not begun yet (restartTimer).
let restartPending = false;
// `pausing`: Pause was pressed; the views stop at the next tick they all code.
let paused = false, pausing = false, pausedAt = 0, stepping = false;
let nextDeadline = 0, firstDeliveredAt = 0, lastTick = -1;
// The original frame (largest view size) of the instant on screen, for the magnifier.
let original = null;
const frames = new FrameSource(video, error => { if (running || preparing) fail(error); });
// `?display=cpu|webgl|webgl2` overrides the frame renderer's automatic choice (renderer.js), e.g. to compare the paths.
const display = new URLSearchParams(location.search).get('display') || 'auto';
const clipTitle = id => clipById(id).title;
const panels = ['A', 'B'].map((letter, i) => {
  const panel = new Panel(letter, i ? 'av1' : 'h264', $('#versions'), {
    changed: change => settingsChanged(change, panel),
    onMatch: entry => matchOther(panel, entry),
    clipTitle,
  }, display);
  return panel;
});
const loupe = new Loupe($('#loupe'), panels);

function message(text = '') { notice.textContent = text; notice.hidden = !text; }
function setPauseButton(label, pressed) { pauseButton.textContent = label; pauseButton.setAttribute('aria-pressed', String(pressed)); }
function stop(keepSource = false) {
  ++generation; clearTimeout(timer); clearTimeout(restartTimer); stoppedByFailure = false; restartPending = false;
  comparison?.terminate(); comparison = null;
  running = false; preparing = false; paused = false; pausing = false;
  pauseButton.disabled = true; setPauseButton('Pause', false); stepButton.disabled = true;
  $("#keyframe").disabled = true;
  document.body.classList.remove('running'); startButton.innerHTML = 'Start comparison <span aria-hidden="true">↗</span>';
  $('#runtime-label').textContent = 'WASM encode + decode';
  if (!keepSource) {
    video.pause();
    if (camera) { for (const track of camera.getTracks()) track.stop(); camera = null; video.srcObject = null; }
  }
}
/** Stop capturing once the views have reached a common instant. */
function pause() {
  if (!running || paused || pausing) return;
  pausing = true; setPauseButton('Pausing…', true); trace('pause requested', lastTick);
}
function enterPause() {
  trace('paused', lastTick);
  pausing = false; paused = true; pausedAt = performance.now(); clearTimeout(timer);
  if (kind !== 'scene') video.pause();
  setPauseButton('Resume', true);
  stepButton.disabled = !frames.canStep;
  $('#runtime-label').textContent = 'Paused';
}
async function resume() {
  if (!running || !paused) return;
  const token = generation;
  pauseButton.disabled = true; stepButton.disabled = true;
  try {
    if (kind !== 'scene') await video.play();
    if (token !== generation) return;
    const now = performance.now(), gap = now - pausedAt;
    if (delivered) firstDeliveredAt += gap;
    // Wall-clock sources continue from the instant on screen.
    frames.started = now - tickMs(lastTick + 1);
    paused = false; nextDeadline = now;
    pauseButton.disabled = false; setPauseButton('Pause', false);
    $('#runtime-label').textContent = 'Encoding + decoding live';
    scheduleFrame(token);
  } catch (error) { if (token === generation) fail(error); }
}
/** While paused, show the next instant every view codes. */
async function step() {
  if (!paused || stepping || comparison.full) return;
  const token = generation, tick = nextCommonTick(lastTick, panels.map(p => p.current.step));
  stepping = true; stepButton.disabled = true;
  try {
    const shown = await frames.advanceTo(tick);
    if (token === generation) capture(shown);
  } catch (error) { if (token === generation) fail(error); }
  stepping = false;
  if (token === generation && paused) stepButton.disabled = false;
}
function fail(error) {
  const detail = error instanceof Error ? error.message : String(error);
  trace('fail', detail, restartPending ? '(restart pending)' : '');
  // Settings changed after the failing comparison started: their restart is still
  // due (stop() cancels the timer), and it runs with the new settings.
  if (restartPending) { start(); return; }
  stop(); stoppedByFailure = true;
  message(detail); $('#session-stats').textContent = 'Comparison stopped';
}
/** A panel's setting changed: rate changes apply live, a new stream may switch the clip, others restart. */
function settingsChanged(change, panel) {
  trace('settings', change ?? 'restart', panel?.letter, running ? 'running' : preparing ? 'starting' : stoppedByFailure ? 'failed' : 'stopped');
  if (change === 'rate' && running) { pendingRateChange = true; return; }
  if (change === 'reference' && panel.reference && (kind !== 'clip' || panel.reference.clip !== clip.id)) {
    clip = clipById(panel.reference.clip); clipPicker.value = clip.id;
    sourceChanged('clip'); start(); return;
  }
  describeViews();
  // A comparison the visitor stopped waits for Start; one running, starting or
  // stopped by a failure restarts with the new settings (start() reads them all).
  if (!running && !preparing && !stoppedByFailure) { resetViews(); $("#session-stats").textContent = "Settings changed · start a new comparison"; return; }
  clearTimeout(restartTimer);
  restartPending = true;
  restartTimer = setTimeout(() => { trace('restart'); stop(true); start(); }, 200);
}
/** "Match this in the other view": the other view becomes a live encode at the stream's settings. */
function matchOther(panel, entry) {
  const other = panels.find(p => p !== panel);
  other.match(matchingSettings(entry));
  settingsChanged('match', other);
}
function resetViews() {
  for (const panel of panels) panel.reset(panel.config());
  original = null; loupe.frameShown(null);
}
/** Say when the views' numbers are not directly comparable. */
function describeViews() {
  const configs = panels.map(p => p.config()), notes = [];
  if ((configs[0].width !== configs[1].width || configs[0].height !== configs[1].height)) {
    notes.push('The views differ in resolution: each PSNR is against the source at that view’s own size, so the two are not directly comparable.');
  }
  if (configs[0].fps !== configs[1].fps) {
    notes.push('The views differ in frame rate: bitrate and timings are per view at its own rate; a lower rate holds each picture until its next frame.');
  }
  $('#view-note').textContent = notes.join(' ');
  $('#view-note').hidden = !notes.length;
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
    // The URL is the source identity: a size change that picks another variant reloads.
    const url = clipUrl(clip, Math.max(...panels.map(p => p.current.height)));
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
  // A pre-encoded stream starts at the clip's first frame, like the stream itself.
  if (panels.some(p => p.current.entry)) video.currentTime = 0;
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
/** Pre-encoded streams play only on their own sample clip. */
function checkReferences() {
  for (const panel of panels) {
    if (!panel.preEncoded) continue;
    if (kind === 'clip' && panel.followClip(clip.id)) continue;
    panel.goLive();
    message(`Version ${panel.letter} is live again: pre-encoded streams exist only for some sample clips${kind === 'clip' ? ', not this one' : ''}.`);
  }
}
async function start() {
  stop(true); const token = generation;
  trace('start', token, panels.map(p => `${p.letter}:${p.reference?.id ?? p.config().codec}`).join(' '));
  preparing = true; message(); startButton.textContent = 'Cancel'; $('#session-stats').textContent = 'Loading codec workers…';
  checkReferences();
  for (const panel of panels) panel.reset(panel.config());
  describeViews();
  original = null; loupe.frameShown(null);
  try {
    await prepareSource(token);
    if (token !== generation) return;
    await Promise.all(panels.map(panel => panel.prepare()));
    if (token !== generation) return;
    frames.reset(kind, kind === 'clip' ? clip.fps : SOURCE_FPS);
    for (const panel of panels) {
      if (panel.current.entry && panel.stream.loopTicks !== frames.loopLength) {
        throw new Error(`Version ${panel.letter}: the stream was made for a ${panel.stream.loopTicks}-tick loop, the clip has ${frames.loopLength}`);
      }
    }
    comparison = new Comparison(panels, {
      onFrame: (results, originals, tick) => finishFrame(token, results, originals, tick),
      onError: error => { if (token === generation) fail(error); },
    });
    await comparison.start();
    if (token !== generation) return;
    pendingRateChange = false;
    $("#keyframe").disabled = false; pauseButton.disabled = false;
    running = true; preparing = false; delivered = 0; firstDeliveredAt = 0; lastTick = -1; nextDeadline = performance.now();
    document.body.classList.add('running'); startButton.textContent = 'Stop comparison'; $('#runtime-label').textContent = 'Encoding + decoding live';
    scheduleFrame(token);
  } catch (error) { if (token === generation) fail(error); }
}
// Captures follow the source timeline. While the pipeline is full none is
// scheduled: the next frame completing calls this again, so a slow pipeline
// skips ticks rather than queueing frames (pipeline.js).
function scheduleFrame(token) {
  clearTimeout(timer);
  if (token !== generation || !running || paused || comparison.full) return;
  timer = setTimeout(() => sendFrame(token), Math.max(0, nextDeadline - performance.now()));
}
function sendFrame(token) {
  try { captureNext(token); } catch (error) { if (token === generation) fail(error); }
}
function captureNext(token) {
  if (token !== generation || !running || paused) return;
  if (kind !== 'scene' && video.readyState < 2) { nextDeadline = performance.now() + NOT_READY_RETRY_MS; scheduleFrame(token); return; }
  if (kind === 'file' && video.ended) { stop(); $('#session-stats').textContent = `${delivered} frames compared · video ended`; return; }
  nextDeadline = Math.max(nextDeadline + 1000 / SOURCE_FPS, performance.now());
  const tick = frames.nextTick(lastTick);
  if (tick !== null) capture(tick);
  scheduleFrame(token);
}
/** Capture `tick` for every view that codes it; the capture worker converts it for each view's size. */
function capture(tick) {
  const coding = comparison.coding(tick);
  if (!coding.some(Boolean)) { lastTick = tick; return; }
  const largest = panels.map(p => p.current).reduce((a, b) => (b.width > a.width ? b : a));
  const shot = frames.capture(tick, largest.width, largest.height);
  // No frame yet (a source or webcam just starting): try again on the next tick.
  if (!shot) return;
  lastTick = tick;
  const { picture, element, width, height } = shot;
  const scale = Math.min(PREVIEW_WIDTH / width, PREVIEW_HEIGHT / height);
  preview.fillStyle = '#000'; preview.fillRect(0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT);
  preview.drawImage(element, (PREVIEW_WIDTH - width * scale) / 2, (PREVIEW_HEIGHT - height * scale) / 2, width * scale, height * scale);
  if (pendingRateChange) {
    panels.forEach((panel, i) => {
      if (panel.current.source !== 'live') return;
      // Only the rate changes here: other settings restart the comparison.
      panel.current = { ...panel.current, ...panel.rateSettings() }; panel.history = [];
      comparison.updateRate(i);
    });
    pendingRateChange = false;
  }
  comparison.send(tick, picture, width, height, coding);
  if (pausing && coding.every(Boolean)) enterPause();
}
/** Every view that codes `tick` decoded it: show them and update the session line. */
function finishFrame(token, results, originals, tick) {
  try {
    panels.forEach((panel, i) => {
      // A held frame 0 (start-up lookahead) is shown with the next result; a view
      // that does not code this tick keeps its picture.
      if (!results[i] || results[i].held) return;
      const replaced = panel.frame(results[i], showOriginal, originals[i], tick);
      if (replaced) comparison.recycle(i, replaced);
    });
  }
  catch (error) { fail(error); return; }
  // Frame 0 held for the start-up lookahead returns no source.
  const shown = originals.filter(Boolean);
  if (shown.length) original = shown.reduce((a, b) => (b.width > a.width ? b : a));
  loupe.frameShown(original);
  // Delivery is counted in frames of the faster view, the rate the page aims for.
  const fastest = panels.reduce((a, b) => (b.current.fps > a.current.fps ? b : a));
  if (!results[panels.indexOf(fastest)]) { scheduleFrame(token); return; }
  ++delivered;
  const now = paused ? pausedAt : performance.now();
  if (delivered === 1) firstDeliveredAt = now;
  const deliveryRate = delivered > 1 ? (delivered - 1) * 1000 / Math.max(now - firstDeliveredAt, 1) : 0;
  $('#session-stats').textContent = `${delivered} frames compared · ${deliveryRate.toFixed(1)} fps delivered${limit(deliveryRate)} · bitrate per view at its own frame rate`;
  scheduleFrame(token);
}
/**
 * The busiest worker (encode and decode per view), which sets the delivered
 * rate once it falls below the fastest view's frame rate.
 */
function limit(deliveryRate) {
  if (!deliveryRate) return '';
  const stages = panels.flatMap(panel => {
    const { encode, decode } = panel.stageTimes();
    return [{ name: `encode ${panel.letter}`, ms: encode }, { name: `decode ${panel.letter}`, ms: decode }].filter(s => s.ms !== null);
  });
  const slowest = stages.reduce((a, b) => (b.ms > a.ms ? b : a));
  const stage = `${slowest.name} (${slowest.ms.toFixed(0)} ms)`;
  const target = Math.max(...panels.map(p => p.current.fps));
  return deliveryRate >= KEEPING_UP * target ? ` · keeping up, slowest ${stage}` : ` · limited by ${stage}`;
}
function sourceChanged(next) {
  stop(); kind = next; message(); resetViews();
  preview.clearRect(0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT);
  if (kind === "scene") drawScene(preview, PREVIEW_WIDTH, PREVIEW_HEIGHT, 0);
  for (const button of document.querySelectorAll('[data-source]')) { const selected = button.dataset.source === kind; button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', String(selected)); }
  const speed = clip.speed ? ` · plays ${clip.speed.toFixed(2)}× real time (retimed to 30 fps)` : '';
  const names = { clip: `${clip.film} · ${clip.title}${speed}`, scene: 'Moving shapes, fine texture and a smooth gradient', camera: 'Webcam · camera access requested when you start', file: file?.name || 'Choose a local video file' };
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
$('#keyframe').addEventListener('click', () => { if (running) comparison.requestKeyframes(); });
pauseButton.addEventListener('click', () => { if (paused) resume(); else pause(); });
stepButton.addEventListener('click', step);
startButton.addEventListener('click', () => { if (running || preparing) stop(); else start(); });
const originalButton = $('#original');
function toggleOriginal(value) {
  showOriginal = value; originalButton.setAttribute('aria-pressed', String(value));
  for (const panel of panels) if (panel.shown) {
    if (value) panel.renderer.showOriginal(panel.original); else panel.renderer.showLatest();
    panel.label(value);
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
