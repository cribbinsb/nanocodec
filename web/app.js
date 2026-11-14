// The comparison page: the source (sample clip, test scene, webcam or file),
// two views (panel.js), the capture loop on the shared source timeline
// (timeline.js, capture.js), and pause, frame stepping and the session line.
//
// Each view codes the ticks of its own frame rate and is given the source drawn
// at its own resolution. Sample clips, files the browser can decode frame by
// frame and the test scene are exact sources (exact-source.js): the page asks
// for every tick a view codes, in order, as fast as the pipeline takes them, and
// the clip's time runs slower than real time when it cannot keep up; nothing is
// skipped, so pictures, streams and numbers do not depend on the machine. The
// webcam (and a file the browser can only play) is sampled live on the timeline
// (capture.js). Pause stops on the next tick every view codes, and Step moves to
// the next such tick, so paused views always show the same source instant.
//
// Switching analysis on or off while comparing keeps the session: captures
// wait until none is in flight, the views' decoders move to the other module
// and decode the picture on screen again (pipeline.js `switchDecoders`), and a
// paused comparison stays paused on that instant while a playing one continues
// from it.
import { Panel } from './panel.js';
import { Comparison } from './pipeline.js';
import { Loupe } from './loupe.js';
import { AnalysisController } from './analysis-view.js';
import { drawScene } from './scene.js';
import { FrameSource } from './capture.js';
import { ExactSource } from './exact-source.js';
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
// Delivery under this fraction of the slowest stage's own rate is not that stage's limit.
const STAGE_BOUND = 0.8;
// Said of a clip or file the browser cannot decode frame by frame (exact-source.js), so it is sampled while it plays.
const APPROXIMATE = 'this browser/file cannot be read frame by frame: frames may be skipped and the numbers are approximate';
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
// Analysis was switched while comparing: no captures until the decoders have
// moved to the other module (`switchDecoders`); `switchBusy` while they move.
let switchWanted = false, switchBusy = false;
// The original frame (largest view size) of the instant on screen, for the magnifier.
let original = null;
const frames = new FrameSource(video, error => { if (running || preparing) fail(error); });
// The exact source of the current clip, file or scene (null: sampled live), the
// identity it was opened for, and whether the pump is fetching a picture.
let exact = null, exactFor = null, pumping = false;
// Exact sources: the wall time of tick 0, so the clip never runs faster than real
// time; moved later whenever the pipeline falls behind, so it never catches up in a burst.
let clockOrigin = 0;
// Capture follows the source: a staged frame on a tick some view codes is captured
// as soon as it is presented (capture.js), not on the next timer tick.
frames.wanted = tick => !comparison || comparison.coding(tick).some(Boolean);
frames.onCandidate = () => { if (running && !paused) { nextDeadline = performance.now(); scheduleFrame(generation); } };
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
// Coding-tool overlays and statistics (analysis-view.js); off until switched on.
const analysis = new AnalysisController($('#analysis'), $('#versions'), panels, {
  onEnabled: () => analysisSwitched(),
  loupeActive: () => loupe.enabled,
});

function message(text = '') { notice.textContent = text; notice.hidden = !text; }
function setPauseButton(label, pressed) { pauseButton.textContent = label; pauseButton.setAttribute('aria-pressed', String(pressed)); }
function stop(keepSource = false) {
  ++generation; clearTimeout(timer); clearTimeout(restartTimer); stoppedByFailure = false; restartPending = false;
  switchWanted = false; switchBusy = false;
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
  if (!exact && kind !== 'scene') video.pause();
  setPauseButton('Resume', true);
  stepButton.disabled = !exact && !frames.canStep;
  $('#runtime-label').textContent = 'Paused';
}
async function resume() {
  if (!running || !paused) return;
  pauseButton.disabled = true; stepButton.disabled = true;
  await play(pausedAt);
}
/** Play on from the instant on screen, held since `heldAt` (performance.now()). */
async function play(heldAt) {
  const token = generation;
  try {
    if (!exact && kind !== 'scene') await video.play();
    if (token !== generation) return;
    const now = performance.now(), gap = now - heldAt;
    if (delivered) firstDeliveredAt += gap;
    // Wall-clock sources continue from the instant on screen.
    frames.started = now - tickMs(lastTick + 1);
    paused = false; nextDeadline = now;
    pauseButton.disabled = false; setPauseButton('Pause', false);
    $('#runtime-label').textContent = 'Encoding + decoding live';
    scheduleFrame(token);
  } catch (error) { if (token === generation) fail(error); }
}
/**
 * Analysis was switched on or off. A running comparison keeps going on the
 * same instant (see the file comment); a starting one restarts with the new
 * module, a stopped one waits for Start. A live view past its re-decode limit
 * (pipeline.js MAX_REPLAY_BYTES) cannot switch in place: the comparison restarts.
 */
function analysisSwitched() {
  if (!running) { settingsChanged('analysis'); return; }
  if (!comparison.canSwitchDecoders) { trace('analysis switch restarts: a view is past its re-decode limit'); settingsChanged('analysis'); return; }
  trace('analysis switch', analysis.enabled ? 'on' : 'off', paused ? 'paused' : 'playing');
  switchWanted = true;
  pauseButton.disabled = true; stepButton.disabled = true;
  $('#runtime-label').textContent = 'Switching decoders…';
  switchDecodersWhenIdle();
}
/** Switch the decoders once nothing is in flight and no pause or step is under way (called again as those end). */
async function switchDecodersWhenIdle() {
  if (!switchWanted || switchBusy || pausing || stepping || !comparison.idle) return;
  const token = generation, playing = !paused, heldAt = performance.now();
  switchWanted = false; switchBusy = true;
  clearTimeout(timer);
  if (playing && !exact && kind !== 'scene') video.pause();
  try {
    const results = await comparison.switchDecoders();
    if (token !== generation) return;
    results.forEach((result, i) => {
      if (!result) return;
      const replaced = panels[i].redisplay(result, showOriginal);
      if (replaced) comparison.recycle(i, replaced);
    });
  } catch (error) { if (token === generation) fail(error); return; }
  switchBusy = false;
  // Switched again meanwhile: switch back before going on.
  switchWanted = comparison.decodersStale;
  if (switchWanted) { switchDecodersWhenIdle(); return; }
  if (playing) { await play(heldAt); return; }
  pauseButton.disabled = false; stepButton.disabled = !exact && !frames.canStep;
  $('#runtime-label').textContent = 'Paused';
}
/**
 * Exact sources: send every tick a view codes, in order, while the pipeline has
 * room (a completed frame calls this again through scheduleFrame).
 */
async function pump(token) {
  if (pumping) return;
  pumping = true;
  // A decoder switch (or pause) may begin while a picture is decoded: re-checked before sending.
  const mayCapture = () => token === generation && running && !paused && !((switchWanted || switchBusy) && !pausing);
  try {
    while (mayCapture() && !comparison.full) {
      const tick = nextCodedTick(lastTick);
      if (exact.ended(tick)) { stop(); $('#session-stats').textContent = `${delivered} frames compared · video ended`; return; }
      const due = clockOrigin + tickMs(tick), now = performance.now();
      if (due > now) { timer = setTimeout(() => pump(token), due - now); return; }
      clockOrigin = Math.max(clockOrigin, now - tickMs(tick + 1));
      const shot = await exact.picture(tick, ...largestView());
      if (!mayCapture()) { shot.picture.close?.(); return; }
      send(tick, shot);
    }
  } catch (error) { if (token === generation) fail(error); } finally { pumping = false; }
}
/** The first tick after `after` that some view codes. */
function nextCodedTick(after) {
  let tick = after + 1;
  while (!comparison.coding(tick).some(Boolean)) ++tick;
  return tick;
}
/** The largest view size, at which the scene is drawn. */
function largestView() {
  const largest = panels.map(p => p.current).reduce((a, b) => (b.width > a.width ? b : a));
  return [largest.width, largest.height];
}
/** While paused, show the next instant every view codes. */
async function step() {
  if (!paused || stepping || comparison.full) return;
  const token = generation, tick = nextCommonTick(lastTick, panels.map(p => p.current.step));
  stepping = true; stepButton.disabled = true;
  try {
    if (exact) {
      const shot = await exact.picture(tick, ...largestView());
      if (token === generation) send(tick, shot); else shot.picture.close?.();
    } else {
      const shown = await frames.advanceTo(tick);
      if (token === generation) capture(shown);
    }
  } catch (error) { if (token === generation) fail(error); }
  stepping = false;
  if (token === generation && paused && !switchWanted) stepButton.disabled = false;
  switchDecodersWhenIdle();
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
  for (const panel of panels) { panel.reset(panel.config()); panel.analysis.reset(); }
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
/** The exact source for the current clip, file or scene, reused while the source is the same (null: none). */
async function openExact() {
  const url = kind === 'clip' ? clipUrl(clip, Math.max(...panels.map(p => p.current.height))) : kind === 'file' ? fileUrl : kind;
  if (exactFor === url) return exact;
  exact?.close();
  exact = kind === 'camera' ? null : await ExactSource.open(kind, { url, fps: kind === 'clip' ? clip.fps : null, loops: kind === 'clip' });
  exactFor = url;
  return exact;
}
async function prepareSource(token) {
  if (kind === 'file' && file && loadedSource !== file) {
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    fileUrl = URL.createObjectURL(file); loadedSource = null;
  }
  if (await openExact()) { video.pause(); return; }
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
  for (const panel of panels) { panel.reset(panel.config()); panel.analysis.reset(); }
  describeViews();
  original = null; loupe.frameShown(null);
  try {
    await prepareSource(token);
    if (token !== generation) return;
    await Promise.all(panels.map(panel => panel.prepare()));
    if (token !== generation) return;
    if (!exact) frames.reset(kind, kind === 'clip' ? clip.fps : SOURCE_FPS);
    const loopLength = (exact ?? frames).loopLength;
    for (const panel of panels) {
      if (panel.current.entry && panel.stream.loopTicks !== loopLength) {
        throw new Error(`Version ${panel.letter}: the stream was made for a ${panel.stream.loopTicks}-tick loop, the clip has ${loopLength}`);
      }
    }
    // Sampled live (webcam, or a file read while it plays): frames may be skipped,
    // so each carries its capture time to the encoder (encode-worker.js).
    for (const panel of panels) panel.current = { ...panel.current, timed: !exact };
    comparison = new Comparison(panels, {
      onFrame: (results, originals, tick) => finishFrame(token, results, originals, tick),
      onError: error => { if (token === generation) fail(error); },
    });
    await comparison.start();
    if (token !== generation) return;
    pendingRateChange = false;
    $("#keyframe").disabled = false; pauseButton.disabled = false;
    running = true; preparing = false; delivered = 0; firstDeliveredAt = 0; lastTick = -1; nextDeadline = clockOrigin = performance.now();
    document.body.classList.add('running'); startButton.textContent = 'Stop comparison'; $('#runtime-label').textContent = 'Encoding + decoding live';
    scheduleFrame(token);
  } catch (error) { if (token === generation) fail(error); }
}
// Captures follow the source timeline. While the pipeline is full none is
// scheduled: the next frame completing calls this again, so a slow pipeline
// skips ticks rather than queueing frames (pipeline.js).
function scheduleFrame(token) {
  clearTimeout(timer);
  // A decoder switch waits for the frames in flight; a pause under way still reaches its instant.
  if (token !== generation || !running || paused || comparison.full || ((switchWanted || switchBusy) && !pausing)) return;
  if (exact) { pump(token); return; }
  timer = setTimeout(() => sendFrame(token), Math.max(0, nextDeadline - performance.now()));
}
function sendFrame(token) {
  try { captureNext(token); } catch (error) { if (token === generation) fail(error); }
}
function captureNext(token) {
  if (token !== generation || !running || paused || ((switchWanted || switchBusy) && !pausing)) return;
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
  const shot = frames.capture(tick, ...largestView());
  // No frame yet (a source or webcam just starting): try again on the next tick.
  if (shot) send(tick, shot, coding);
}
/** Show the picture of `tick` in the preview and send it to the views that code it. */
function send(tick, { picture, element, width, height }, coding = comparison.coding(tick)) {
  lastTick = tick;
  panels.forEach((panel, i) => { if (coding[i]) panel.sent(tick); });
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
  switchDecodersWhenIdle();
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
  $('#session-stats').textContent = exact
    ? `${delivered} frames compared · ${deliveryRate.toFixed(1)} fps${pace(deliveryRate)} · bitrate per second of source time`
    : `${delivered} frames compared · ${deliveryRate.toFixed(1)} fps delivered${limit(deliveryRate)} · bitrate per second of source time${
      kind === 'camera' ? '' : ` · ${APPROXIMATE}`}`;
  scheduleFrame(token);
}
/**
 * The busiest worker (encode and decode per view), which sets the delivered
 * rate once it falls below the fastest view's frame rate.
 */
/** The busiest stage: { name, ms } over the encode and decode workers of both views. */
function slowestStage() {
  const stages = panels.flatMap(panel => {
    const { encode, decode } = panel.stageTimes();
    return [{ name: `encode ${panel.letter}`, ms: encode }, { name: `decode ${panel.letter}`, ms: decode }].filter(s => s.ms !== null);
  });
  return stages.reduce((a, b) => (b.ms > a.ms ? b : a));
}
/**
 * Exact sources: how fast the clip's time runs. Below real time nothing is
 * skipped, so the pictures and numbers are those of a machine that keeps up.
 */
function pace(deliveryRate) {
  if (!deliveryRate) return '';
  const target = Math.max(...panels.map(p => p.current.fps)), speed = deliveryRate / target, slowest = slowestStage();
  if (speed >= KEEPING_UP) return ` · real time, slowest ${slowest.name} (${slowest.ms.toFixed(0)} ms)`;
  return ` · running at ${speed.toFixed(2)}× real time (slowest ${slowest.name}, ${slowest.ms.toFixed(0)} ms): pictures and numbers are unaffected`;
}
function limit(deliveryRate) {
  if (!deliveryRate) return '';
  const slowest = slowestStage(), stage = `${slowest.name} (${slowest.ms.toFixed(0)} ms)`;
  const target = Math.max(...panels.map(p => p.current.fps));
  if (deliveryRate >= KEEPING_UP * target) return ` · keeping up, slowest ${stage}`;
  // Stages overlap (pipeline.js), so the slowest one allows 1000 / ms frames per
  // second; well below that, the views wait for source frames (capture takes every
  // presented frame on a coded tick, capture.js), e.g. a browser dropping frames.
  if (deliveryRate < STAGE_BOUND * 1000 / slowest.ms) return ` · waiting for source frames (slowest stage ${stage})`;
  return ` · limited by ${stage}`;
}
function sourceChanged(next) {
  stop(); kind = next; message(); resetViews();
  exact?.close(); exact = null; exactFor = null;
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
    if (value) panel.renderer.showOriginal(panel.original?.image ?? null); else panel.renderer.showLatest();
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
