// The comparison page: the source (sample clip, test scene, webcam or file),
// two views (panel.js), the capture loop on the shared source timeline
// (timeline.js, capture.js), and pause, frame stepping and the session line.
//
// The comparison always runs: it starts on page open and restarts by itself,
// RESTART_DELAY_MS after the last of a burst of changes, whenever the source
// or a view's setting changes (rate changes apply live, without a restart).
// Each start supersedes the one before (`generation`), so a burst of changes
// under load ends in one comparison with the final settings.
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
import { ViewZoom } from './view-zoom.js';
import { Layout } from './layout.js';
import { AnalysisController } from './analysis-view.js';
import { FrameSource } from './capture.js';
import { ViewSources } from './exact-source.js';
import { CLIPS, CLIP_GROUPS, DEFAULT_CLIP, clipById, clipUrl } from './clips.js';
import { SOURCE_FPS, nextCommonTick, tickMs } from './timeline.js';
import { matchingSettings } from './view-settings.js';
import { parseDemoQuery } from './deep-link.js';
import { trace, traceTail } from './trace.js';
import { buildReport, clipSeconds, formatReport, sha256 } from './debug-report.js';
import { download } from './streams.js';
import { RELAXED_SIMD } from './wasm.js';
import { SpeedHint } from './speed-hint.js';

const $ = selector => document.querySelector(selector);
const pauseButton = $('#pause'), stepButton = $('#step');
const video = $('#input-video'), notice = $('#notice');
// A change restarts the comparison this long after the last change of a burst.
const RESTART_DELAY_MS = 300;
// The opening comparison: a pre-encoded x264 stream in A, and B matched to it
// live (nanocodec H.264 Default at its size, frame rate and bitrate).
const OPENING_STREAM = 'xiph-four-people-x264-medium-854x480-30fps-400k';
// `?views=live` opens instead on two live views at the panels' defaults (H.264
// and AV1, 854x480, 15 fps, 300 kbps) on DEFAULT_CLIP: the page tests that
// measure live encoders start there.
const LIVE_VIEWS = new URLSearchParams(location.search).get('views') === 'live';
// How long to wait for a video that is not ready to show a frame (ms).
const NOT_READY_RETRY_MS = 100;
// Delivery within this fraction of the target frame rate counts as keeping up.
const KEEPING_UP = 0.95;
// Delivery under this fraction of the slowest stage's own rate is not that stage's limit.
const STAGE_BOUND = 0.8;
// Said of a clip or file the browser cannot decode frame by frame (exact-source.js), so it is sampled while it plays.
const APPROXIMATE = 'sampled while it plays (this browser cannot read it frame by frame): numbers approximate';
let kind = 'clip', clip = DEFAULT_CLIP, file = null, fileUrl = null, loadedSource = null, camera = null;
let running = false, preparing = false, generation = 0, comparison = null, timer = 0, restartTimer = 0;
let delivered = 0;
let pendingRateChange = false;
// A change has scheduled a restart that has not begun yet (restartTimer).
let restartPending = false;
// The comparison stopped (video ended, failure, hidden tab): the Pause button restarts it.
let ended = false;
// `pausing`: Pause was pressed; the views stop at the next tick they all code.
let paused = false, pausing = false, pausedAt = 0, stepping = false;
let nextDeadline = 0, firstDeliveredAt = 0, lastTick = -1;
// Analysis was switched while comparing: no captures until the decoders have
// moved to the other module (`switchDecoders`); `switchBusy` while they move.
let switchWanted = false, switchBusy = false;
const frames = new FrameSource(video, error => { if (running || preparing) fail(error); });
// The exact sources of the current clip, file or scene (exact-source.js
// ViewSources: a clip's rendition for each view's height; null: sampled live),
// the identity they were opened for, and whether the pump is fetching pictures.
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
const speedHint = new SpeedHint($('#speed-hint'), applyLighter);
// Zoom and pan shared by both views (view-zoom.js); the analysis layers draw through it.
const zoom = new ViewZoom(panels, { onChange: () => analysis.zoomChanged(), onTap: (panel, event) => analysis.tapped(panel, event) });
const layout = new Layout($('#layout'), $('#versions'), () => zoom.show());
// Coding-tool overlays and statistics (analysis-view.js); off until switched on.
const analysis = new AnalysisController($('#analysis-toggle'), $('#analysis'), $('#versions'), panels, {
  onEnabled: () => analysisSwitched(), zoom,
});

// What tools/web/build.sh wrote about the module build (commit, profile), for the debug report.
const BUILD_INFO_URL = new URL('./pkg/build.json', import.meta.url);
/** The page's whole state as a debug report (debug-report.js), downloaded as Markdown; `includeStreams`: embed the coded streams. */
async function downloadReport(includeStreams) {
  const build = await fetch(BUILD_INFO_URL).then(response => (response.ok ? response.json() : {}), () => ({}));
  const loopLength = exact?.loopLength ?? frames.loopLength;
  const text = selector => $(selector).textContent;
  const report = await buildReport({
    build: { commit: build.commit ?? 'unknown', dirty: Boolean(build.dirty), profile: build.profile ?? 'unknown',
      module: RELAXED_SIMD ? 'SIMD128 + relaxed SIMD (pkg/relaxed)' : 'SIMD128 (pkg)' },
    page: { url: location.href, time: new Date().toISOString(), userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency },
    source: await sourceState(loopLength),
    state: { playback: paused ? 'paused' : running ? 'playing' : preparing ? 'starting' : 'stopped', tick: lastTick,
      clipSeconds: clipSeconds(lastTick, loopLength), loop: Number.isFinite(loopLength) ? Math.floor(lastTick / loopLength) : 0,
      session: text('#session-stats'), runtime: text('#runtime-label'), notice: notice.hidden ? null : notice.textContent,
      viewNote: $('#view-note').hidden ? null : text('#view-note'), pipeline: comparison && delivered ? pipelineState() : null,
      delivered, analysis: analysis.debugState(), layout: layout.value, zoom: zoom.debugState() },
    events: traceTail(),
  }, panels.map(panel => panel.debugState()), { includeStreams });
  download(new Blob([formatReport(report)], { type: 'text/markdown' }), `nanocodec-debug-${report.page.time.replace(/[:.]/g, '-')}.md`);
}
/** A decoded clip's container facts (clip-source.js), for the debug report. */
const containerFacts = source => source && { codec: source.config.codec, width: source.config.codedWidth, height: source.config.codedHeight,
  frames: source.frameCount, fps: source.fps, timescale: source.timescale };
/**
 * The source for the debug report: what it is, the hash of each file the views
 * code (a clip's rendition per view, exact-source.js), and whether replay can regenerate it.
 */
async function sourceState(loopLength) {
  const base = { kind, path: kind === 'camera' ? 'webcam, sampled live' : exact ? 'exact: decoded frame by frame' : 'sampled while the video plays',
    loopLength: Number.isFinite(loopLength) ? loopLength : null };
  if (kind === 'clip') {
    const files = [...new Set(panels.map(panel => panel.current.rendition))];
    const renditions = await Promise.all(files.map(async name => {
      const decoded = exact?.entries.find(entry => entry.url.endsWith(`/${name}`))?.source.clip;
      const bytes = decoded?.bytes ?? new Uint8Array(await (await fetch(`clips/${name}`)).arrayBuffer());
      return { file: name, sha256: await sha256(bytes), container: containerFacts(decoded) };
    }));
    return { ...base, clip: clip.id, fps: clip.fps, renditions, reproducible: true };
  }
  if (kind === 'file') {
    const decoded = exact?.entries[0].source.clip;
    return { ...base, file: file?.name ?? null, size: file?.size ?? null, type: file?.type ?? null, sha256: file ? await sha256(await file.arrayBuffer()) : null,
      fps: decoded?.fps ?? null, container: containerFacts(decoded), reproducible: false,
      reason: 'a local file is never embedded; replay checks the statistics and any embedded streams' };
  }
  const reason = kind === 'camera' ? 'a webcam picture is never embedded; replay checks the statistics and any embedded streams'
    : 'the test scene is drawn by the browser; replay checks the statistics and any embedded streams';
  return { ...base, fps: kind === 'scene' ? SOURCE_FPS : null, reproducible: false, reason };
}
function message(text = '') { notice.textContent = text; notice.hidden = !text; }
function setPauseButton(label, pressed) { pauseButton.textContent = label; pauseButton.setAttribute('aria-pressed', String(pressed)); }
/**
 * End the comparison (its workers, and unless `keepSource` the camera and
 * video). A restart already scheduled still runs.
 */
function stop(keepSource = false) {
  ++generation; clearTimeout(timer);
  switchWanted = false; switchBusy = false;
  comparison?.terminate(); comparison = null; speedHint.clear();
  running = false; preparing = false; paused = false; pausing = false; ended = false;
  pauseButton.disabled = true; setPauseButton('Pause', false); stepButton.disabled = true;
  $("#keyframe").disabled = true;
  document.body.classList.remove('running');
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
      const replaced = panels[i].redisplay(result);
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
 *
 * One pump reads the sources at a time (a clip source decodes one request at a
 * time). A pump of a stopped comparison can still be waiting for a picture when
 * the next comparison starts, whose pump then returns at once: with nothing in
 * flight no completed frame would call it again, so the stale pump, once its
 * picture is in, starts the current comparison's pump.
 */
async function pump(token) {
  if (pumping) return;
  pumping = true;
  // A decoder switch (or pause) may begin while a picture is decoded: re-checked before sending.
  const mayCapture = () => token === generation && running && !paused && !((switchWanted || switchBusy) && !pausing);
  try {
    while (mayCapture() && !comparison.full) {
      const tick = nextCodedTick(lastTick);
      if (exact.ended(tick)) { videoEnded(); return; }
      const due = clockOrigin + tickMs(tick), now = performance.now();
      if (due > now) { timer = setTimeout(() => pump(token), due - now); return; }
      clockOrigin = Math.max(clockOrigin, now - tickMs(tick + 1));
      const shots = await exact.pictures(tick, comparison.coding(tick), ...largestView());
      if (!mayCapture()) { closeShots(shots); return; }
      send(tick, shots);
    }
  } catch (error) { if (token === generation) fail(error); } finally {
    pumping = false;
    if (token !== generation) scheduleFrame(generation);
  }
}
/** The first tick after `after` that some view codes. */
function nextCodedTick(after) {
  let tick = after + 1;
  while (!comparison.coding(tick).some(Boolean)) ++tick;
  return tick;
}
/** Release pictures that will not be sent. */
function closeShots(shots) { for (const shot of shots) shot.picture.close?.(); }
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
      const shots = await exact.pictures(tick, comparison.coding(tick), ...largestView());
      if (token === generation) send(tick, shots); else closeShots(shots);
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
  // due, and it runs with the new settings.
  stop(true);
  if (restartPending) return;
  showEnded('Restart');
  message(detail); $('#session-stats').textContent = 'Comparison stopped';
}
/** The comparison has stopped by itself: the Pause button, as `label`, starts it again. */
function showEnded(label) {
  ended = true; pauseButton.disabled = false; setPauseButton(label, false);
}
/** A local file reached its end. */
function videoEnded() {
  stop(); showEnded('Play again');
  $('#session-stats').textContent = `${delivered} frames compared · video ended`;
}
/** Start a new comparison RESTART_DELAY_MS after the last of a burst of calls (see the file comment). */
function scheduleRestart(reason) {
  trace('restart scheduled', reason);
  clearTimeout(restartTimer);
  restartPending = true;
  restartTimer = setTimeout(() => { restartPending = false; trace('restart', reason); start(); }, RESTART_DELAY_MS);
}
/** A panel's setting changed: rate changes apply live, a new stream may switch the clip, others restart. */
function settingsChanged(change, panel) {
  trace('settings', change ?? 'restart', panel?.letter, running ? 'running' : preparing ? 'starting' : 'stopped');
  if (change === 'rate' && running) { pendingRateChange = true; describeViews(); return; }
  if (change === 'reference' && panel.reference && (kind !== 'clip' || panel.reference.clip !== clip.id)) {
    clip = clipById(panel.reference.clip); clipPicker.value = clip.id;
    sourceChanged('clip'); return;
  }
  describeViews();
  scheduleRestart(change ?? 'settings');
}
/** "Match nanocodec to this stream": the other view becomes a live encode at the stream's settings. */
function matchOther(panel, entry) {
  const other = panels.find(p => p !== panel);
  other.match(matchingSettings(entry));
  settingsChanged('match', other);
}
/**
 * A lighter configuration from the speed note: view `index` takes the preset, or
 * every live view at its size takes the smaller size (the views stay comparable).
 * Either restarts the comparison, as changing the control does.
 */
function applyLighter(index, change) {
  const panel = panels[index];
  if (change.preset) { panel.preset.value = change.preset; panel.preset.dispatchEvent(new Event('change')); return; }
  const { width, height } = panel.current;
  for (const other of panels) {
    if (other.current.source !== 'live' || other.current.width !== width || other.current.height !== height) continue;
    other.resolution.value = `${change.width}x${change.height}`;
    other.resolution.dispatchEvent(new Event('change'));
  }
}
/** Say what is compared (the header sentence), and when the views' numbers are not directly comparable. */
function describeViews() {
  const source = { clip: `the clip ${clip.title}`, scene: 'the test scene', camera: 'your webcam', file: file?.name ?? 'a video file' }[kind];
  $('#summary').textContent = `Comparing ${panels[0].summary} in A with ${panels[1].summary} in B, on ${source}.`;
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
/** Each view's source: a clip's rendition for the view's height (clips.js clipUrl), the file, or the scene. */
function viewSourceUrls() {
  return panels.map(panel => (kind === 'clip' ? clipUrl(clip, panel.current.height) : kind === 'file' ? fileUrl : kind));
}
/** The exact sources for the current clip, file or scene, reused while they are the same (null: none). */
async function openExact() {
  const urls = viewSourceUrls(), identity = urls.join(' ');
  if (exactFor === identity) return exact;
  exact?.close();
  exact = kind === 'camera' ? null : await ViewSources.open(kind, urls, { fps: kind === 'clip' ? clip.fps : null, loops: kind === 'clip' });
  exactFor = identity;
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
    // One playing video for every view: the tallest view's rendition. The URL is the
    // source identity: a size change that picks another variant reloads.
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
    if (error.name === 'NotAllowedError') throw new Error('The browser blocked automatic playback. Press Restart to play the video.');
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
    describeViews();
  }
}
async function start() {
  clearTimeout(restartTimer); restartPending = false;
  stop(true); const token = generation;
  if (document.hidden) { trace('start deferred: the tab is hidden'); return; }
  if (kind === 'file' && !file) { $('#session-stats').textContent = 'Choose a video file to compare'; return; }
  trace('start', token, panels.map(p => `${p.letter}:${p.reference?.id ?? p.config().codec ?? p.config().source}`).join(' '));
  preparing = true; message(); $('#session-stats').textContent = 'Loading codec workers…';
  checkReferences();
  for (const panel of panels) { panel.reset(panel.config()); panel.analysis.reset(); }
  describeViews();
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
    // The clip file each view codes (and is scored against): its own height's rendition on
    // the exact path, the playing video's otherwise.
    const renditions = exact ? viewSourceUrls() : panels.map(() => clipUrl(clip, Math.max(...panels.map(p => p.current.height))));
    panels.forEach((panel, i) => {
      panel.current = { ...panel.current, timed: !exact, rendition: kind === 'clip' ? renditions[i].split('/').pop() : null,
        sourceKbps: kind === 'clip' && exact ? exact.sourceKbps(i) : null };
      // "On clip start" keys where the clip loops (panel.js `atLoop`, encode-worker.js).
      if (kind === 'clip' && Number.isFinite(loopLength)) panel.current = panel.atLoop(panel.current, loopLength);
    });
    comparison = new Comparison(panels, {
      onFrame: (results, originals, tick) => finishFrame(token, results, originals, tick),
      onError: error => { if (token === generation) fail(error); },
    });
    await comparison.start();
    if (token !== generation) return;
    pendingRateChange = false;
    $("#keyframe").disabled = false; pauseButton.disabled = false;
    running = true; preparing = false; delivered = 0; firstDeliveredAt = 0; lastTick = -1; nextDeadline = clockOrigin = performance.now();
    document.body.classList.add('running'); $('#runtime-label').textContent = 'Encoding + decoding live';
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
  if (kind === 'file' && video.ended) { videoEnded(); return; }
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
  if (shot) send(tick, [{ ...shot, views: coding.flatMap((c, i) => (c ? [i] : [])) }], coding);
}
/** Send the views that code `tick` their pictures (`shots`: one per source, each with the `views` it is for). */
function send(tick, shots, coding = comparison.coding(tick)) {
  lastTick = tick;
  panels.forEach((panel, i) => { if (coding[i]) panel.sent(tick); });
  if (pendingRateChange) {
    panels.forEach((panel, i) => {
      if (panel.current.source !== 'live') return;
      // Only the rate changes here: other settings restart the comparison.
      panel.current = { ...panel.current, ...panel.rateSettings() }; panel.history = [];
      panel.intervene('rate', tick, panel.rateSettings());
      comparison.updateRate(i);
    });
    pendingRateChange = false;
  }
  comparison.send(tick, shots, coding);
  if (pausing && coding.every(Boolean)) enterPause();
}
/** Every view that codes `tick` decoded it: show them and update the session line. */
function finishFrame(token, results, originals, tick) {
  try {
    panels.forEach((panel, i) => {
      // A held frame 0 (start-up lookahead) is shown with the next result; a view
      // that does not code this tick keeps its picture.
      if (!results[i] || results[i].held) return;
      const replaced = panel.frame(results[i], originals[i], tick);
      if (replaced) comparison.recycle(i, replaced);
      zoom.frameShown(panel);
    });
  }
  catch (error) { fail(error); return; }
  switchDecodersWhenIdle();
  // Delivery is counted in frames of the faster view, the rate the page aims for.
  const fastest = panels.reduce((a, b) => (b.current.fps > a.current.fps ? b : a));
  if (!results[panels.indexOf(fastest)]) { scheduleFrame(token); return; }
  ++delivered;
  const now = paused ? pausedAt : performance.now();
  if (delivered === 1) firstDeliveredAt = now;
  const deliveryRate = delivered > 1 ? (delivered - 1) * 1000 / Math.max(now - firstDeliveredAt, 1) : 0;
  const session = $('#session-stats');
  session.textContent = `${delivered} frames compared · ${deliveryRate.toFixed(1)} fps${exact ? pace(deliveryRate) : limit(deliveryRate)}${
    exact || kind === 'camera' ? '' : ` · ${APPROXIMATE}`}`;
  if (deliveryRate) session.title = stageLine(deliveryRate);
  speedHint.update(panels.map(panel => ({ letter: panel.letter, config: panel.current, encodeMs: panel.stageTimes().encode, frames: panel.count })),
    Boolean(exact));
  scheduleFrame(token);
}
/**
 * Each view's workers and their mean ms per frame (encode; decode, or an
 * Original view's placing of the source, with PSNR and display hand-off).
 */
function stages() {
  return panels.flatMap(panel => {
    const { encode, decode } = panel.stageTimes();
    const shown = panel.current.source === 'original' ? 'original' : 'decode';
    return [{ name: `encode ${panel.letter}`, ms: encode }, { name: `${shown} ${panel.letter}`, ms: decode }].filter(s => s.ms !== null);
  });
}
/** The busiest stage, which sets the delivered rate once it falls below the fastest view's frame rate. */
const slowestStage = () => stages().reduce((a, b) => (b.ms > a.ms ? b : a));
const targetFps = () => Math.max(...panels.map(p => p.current.fps));
/**
 * Exact sources: how fast the clip's time runs, said only below real time.
 * Nothing is skipped then, so the pictures and numbers are those of a machine that keeps up.
 */
function pace(deliveryRate) {
  const speed = deliveryRate / targetFps();
  return deliveryRate && speed < KEEPING_UP ? ` · running at ${speed.toFixed(2)}× real time, so no frame is skipped` : '';
}
/** Live sources: said only when this machine delivers fewer frames than the views code. */
function limit(deliveryRate) {
  return deliveryRate && deliveryRate < KEEPING_UP * targetFps() ? ` · below ${targetFps()} fps: frames are skipped` : '';
}
/** The detail behind the session line (its tooltip, and the debug report): which worker limits delivery. */
function stageLine(deliveryRate) {
  const slowest = slowestStage(), stage = `${slowest.name} (${slowest.ms.toFixed(0)} ms)`;
  if (deliveryRate >= KEEPING_UP * targetFps()) return `Keeping up · slowest ${stage}`;
  // Stages overlap (pipeline.js), so the slowest one allows 1000 / ms frames per
  // second; well below that, the views wait for source frames (capture takes every
  // presented frame on a coded tick, capture.js), e.g. a browser dropping frames.
  if (!exact && deliveryRate < STAGE_BOUND * 1000 / slowest.ms) return `Waiting for source frames · slowest stage ${stage}`;
  return `Limited by ${stage}`;
}
/** The pipeline's speed for the debug report. */
function pipelineState() {
  const now = paused ? pausedAt : performance.now();
  const deliveryRate = delivered > 1 ? (delivered - 1) * 1000 / Math.max(now - firstDeliveredAt, 1) : 0;
  return { deliveredFps: deliveryRate, targetFps: targetFps(), stages: stages(), summary: deliveryRate ? stageLine(deliveryRate) : null };
}
/** The source changed (or the clip): stop at once, and restart on it (a video file once one is chosen). */
function sourceChanged(next) {
  stop(); kind = next; message();
  // Reference streams exist for the sample clips only.
  for (const panel of panels) panel.setClip(kind === 'clip' ? clip.id : null);
  for (const panel of panels) panel.setSourceKind(kind);
  exact?.close(); exact = null; exactFor = null;
  for (const button of document.querySelectorAll('[data-source]')) { const selected = button.dataset.source === kind; button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', String(selected)); }
  const speed = clip.speed ? ` · plays ${clip.speed.toFixed(2)}× real time (retimed to 30 fps)` : '';
  const names = { clip: `${clip.film} · ${clip.title}${speed}`, scene: 'Moving shapes, fine texture and a smooth gradient', camera: 'Webcam', file: file?.name || 'Choose a local video file' };
  $('#source-name').textContent = names[kind];
  clipCredit.hidden = kind !== 'clip'; clipCredit.textContent = clip.credit; clipCredit.title = clip.notice ?? '';
  $('#session-stats').textContent = 'Starting…';
  describeViews();
  scheduleRestart(`source ${kind}`);
}
for (const button of document.querySelectorAll('[data-source]')) button.addEventListener('click', () => { sourceChanged(button.dataset.source); if (kind === 'file') $('#file').click(); });
const clipPicker = $('#clip'), clipCredit = $('#clip-credit');
for (const group of CLIP_GROUPS) {
  const optgroup = document.createElement('optgroup'); optgroup.label = group;
  for (const { id, title } of CLIPS.filter(c => c.group === group)) optgroup.append(new Option(title, id, false, id === clip.id));
  clipPicker.append(optgroup);
}
clipPicker.addEventListener('change', () => { clip = clipById(clipPicker.value); sourceChanged('clip'); });
$('#file').addEventListener('change', event => { const selected = event.target.files[0]; if (selected) { file = selected; sourceChanged('file'); } });
$('#keyframe').addEventListener('click', () => {
  if (!running) return;
  comparison.requestKeyframes();
  for (const panel of panels) if (panel.current.source === 'live') panel.intervene('keyframe', lastTick + 1);
});
// Debug report: a dialog asks whether to embed the coded streams, then the report downloads.
const debugDialog = $('#debug-dialog');
$('#debug-report').addEventListener('click', () => debugDialog.showModal());
// A choice in the More menu closes it.
for (const item of document.querySelectorAll('.more-menu button')) item.addEventListener('click', () => { $('details.more').open = false; });
debugDialog.addEventListener('close', () => {
  if (debugDialog.returnValue !== 'download') return;
  downloadReport($('#debug-streams').checked).catch(error => message(`Debug report failed: ${error.message}`));
});
// Page errors go to the event log the debug report keeps.
window.addEventListener('error', event => trace('page error', event.message));
window.addEventListener('unhandledrejection', event => trace('unhandled rejection', event.reason?.message ?? String(event.reason)));
pauseButton.addEventListener('click', () => { if (ended) start(); else if (paused) resume(); else pause(); });
stepButton.addEventListener('click', step);
video.addEventListener('error', () => { if (running) fail('The input video could not be read.'); });
window.addEventListener('pagehide', () => { stop(); if (fileUrl) URL.revokeObjectURL(fileUrl); });
// A hidden tab stops the comparison (and the camera); it restarts when shown again.
// A start while hidden (page opened in a background tab) waits for that too.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { if (running || preparing) { stop(); trace('stopped: the tab is hidden'); } return; }
  if (!running && !preparing && !ended) scheduleRestart('tab shown');
});
/**
 * A results-page link (deep-link.js): view A shows the linked reference stream
 * (view B matched to it) or the original source, and view B takes the linked
 * nanocodec settings. False if the URL carries no link.
 */
async function openLink() {
  const link = parseDemoQuery(location.search);
  if (!link) return false;
  trace('link', JSON.stringify(link));
  if (link.clip && CLIPS.some(c => c.id === link.clip)) { clip = clipById(link.clip); clipPicker.value = clip.id; }
  let shown = false;
  if (link.stream) {
    try {
      await panels[0].showStream(link.stream);
      clip = clipById(panels[0].reference.clip); clipPicker.value = clip.id;
      panels[1].match(matchingSettings(panels[0].reference));
      shown = true;
    } catch (error) {
      trace('linked stream unavailable', error.message);
    }
  }
  if (!shown) panels[0].showOriginal();
  panels[1].applyLink(link);
  return true;
}
/** The opening comparison (a results link, else OPENING_STREAM), then the first start. */
async function openPage() {
  if (await openLink()) { sourceChanged('clip'); return; }
  if (!LIVE_VIEWS) {
    try {
      await panels[0].showStream(OPENING_STREAM);
      clip = clipById(panels[0].reference.clip); clipPicker.value = clip.id;
      panels[1].match(matchingSettings(panels[0].reference));
    } catch (error) {
      trace('opening stream unavailable', error.message);
      panels[0].goLive();
    }
  }
  sourceChanged('clip');
}
openPage();
