// The debug report: one Markdown file holding the page's whole state, so a
// claimed issue can be reproduced elsewhere (tools/web/replay-debug-report.py
// re-encodes, decodes and re-scores it natively and checks every number).
//
// Human-readable sections come first; the machine-readable state is one fenced
// JSON block (REPORT_SCHEMA) at the end. It holds the build (the commit
// tools/web/build.sh wrote into pkg/build.json), the source (clip id and file
// hash, the exact-source frame mapping; a webcam or local file is described,
// never embedded: they may show a private room), each view's effective
// configuration, every number shown in its panel and statistics, recent coded
// frames (tick, source frame, type, quantizer, bytes, SHA-256 of the access
// unit, timings, PSNR as the page computed it), the bitrate window and its
// formula, what changed the encoders mid-run, and the page's event log tail.
// The coded access units since each view's last key frame are added only when
// asked (`includeStreams`), base64, so the default report stays small.
import { frameAtTick, tickMs } from './timeline.js';

/** The report's schema name and version: the replay tool refuses others. */
export const REPORT_SCHEMA = 'nanocodec-debug-report/1';
/** Frames of each view's history in the report: those since its last key frame, at least MIN_HISTORY_FRAMES. */
export const MIN_HISTORY_FRAMES = 300;
/** Coded bytes of one view the optional stream payload may hold. */
export const MAX_STREAM_BYTES = 8 * 1024 * 1024;
/** The panel bitrate's formula (panel.js `bitrate`), stated in the report. */
export const BITRATE_FORMULA = 'kbps = sum(bytes) * 8 / ((last.timestamp - first.timestamp + 1000 / fps) / 1000) / 1000 over the window';
/** The panel PSNR-YUV's formula (panel.js `average('psnr')`). */
export const PSNR_FORMULA = 'mean of each frame\'s PSNR-YUV (Y:Cb:Cr weighted 6:1:1, each plane capped at 100 dB) over the window';
const JSON_FENCE = '```json';

/** Lowercase hex SHA-256 of `bytes` (Web Crypto). */
export async function sha256(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
}

/** Base64 of `bytes`, in chunks (String.fromCharCode takes a bounded argument list). */
function base64(bytes) {
  const CHUNK = 32768;
  let text = '';
  for (let i = 0; i < bytes.length; i += CHUNK) text += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  return btoa(text);
}

/** The frames of a view's log the report keeps: from its last key frame, and at least MIN_HISTORY_FRAMES. */
export function historyWindow(log) {
  const lastKey = log.findLastIndex(frame => frame.key);
  const start = Math.min(lastKey < 0 ? 0 : lastKey, Math.max(0, log.length - MIN_HISTORY_FRAMES));
  return log.slice(start);
}

/** Each frame's source frame index on a clip of `fps` with a `loopLength`-tick loop (null without a clip). */
export function sourceFrameOf(tick, source) {
  if (!source.fps || !Number.isFinite(source.loopLength)) return null;
  return frameAtTick(tick % source.loopLength, source.fps);
}

/**
 * The report object: `page` holds what app.js gathers (build, page, source,
 * state, events), `views` each panel's debug state (`viewState`). Hashes every
 * kept access unit; adds the streams since the last key when `includeStreams`.
 */
export async function buildReport(page, views, { includeStreams = false } = {}) {
  const reported = [];
  for (const view of views) {
    const window = historyWindow(view.log);
    const frames = [];
    for (const frame of window) {
      const { packet, ...rest } = frame;
      frames.push({ ...rest, sourceFrame: sourceFrameOf(frame.tick, page.source), sha256: await sha256(packet) });
    }
    const { log, streamBytes, ...described } = view;
    const entry = { ...described, ...(streamBytes ? { streamSha256: await sha256(streamBytes) } : {}), frames };
    if (includeStreams) entry.stream = streamSinceKey(log);
    reported.push(entry);
  }
  return { schema: REPORT_SCHEMA, ...page, views: reported };
}

/** The coded access units from the last key frame in `log`, base64 (repeats code nothing), or why there are none. */
function streamSinceKey(log) {
  const lastKey = log.findLastIndex(frame => frame.key);
  if (lastKey < 0) return { complete: false, note: 'no key frame in the kept history' };
  const units = log.slice(lastKey).filter(frame => !frame.repeat);
  const bytes = units.reduce((sum, frame) => sum + frame.packet.byteLength, 0);
  if (bytes > MAX_STREAM_BYTES) return { complete: false, note: `${bytes} bytes since the last key frame, over the ${MAX_STREAM_BYTES}-byte limit` };
  return { complete: true, firstTick: units[0].tick, units: units.map(frame => ({ tick: frame.tick, data: base64(frame.packet) })) };
}

/** The numbers a panel shows, exactly as shown: metrics, rate line, frame count, badge, statistics table, notes. */
export function shownNumbers(element) {
  const text = selector => element.querySelector(selector)?.textContent ?? null;
  const metrics = Object.fromEntries([...element.querySelectorAll('[data-metric]')].map(e => [e.dataset.metric, e.textContent]));
  const statistics = Object.fromEntries([...element.querySelectorAll('.analysis-stats tbody tr')]
    .map(row => [row.querySelector('th').textContent, [...row.querySelectorAll('td')].map(cell => cell.textContent)]));
  return { metrics, rateStatus: text('.rate-status'), frameCount: text('.frame-count'), badge: text('.frame-badge'),
    historyPeak: text('.history-max'), statistics, analysisNotes: [...element.querySelectorAll('.analysis-stats p')].map(p => p.textContent) };
}

/** A short human summary line of a view. */
function viewLine(view) {
  const c = view.config;
  const what = view.config.source === 'reference' ? `pre-encoded \`${view.streamId}\``
    : `live nanocodec ${c.codec} ${c.preset}${c.stillCamera ? ' + still-camera noise handling' : ''}`;
  return `${what} · ${c.width}×${c.height} · ${c.fps} fps · ${c.rateMode === 'cbr' ? `target ${c.targetKbps} kbps` : `fixed quantizer ${c.qp}`}${c.rendition ? ` · codes ${c.rendition}` : ''}`;
}

/** The report as Markdown: readable sections, then the whole state as one JSON block. */
export function formatReport(report) {
  const { build, page, source, state } = report;
  const lines = [
    '# nanocodec debug report', '',
    `Schema \`${report.schema}\` · ${page.time} · demo commit \`${build.commit}\`${build.dirty ? ' (local changes)' : ''}`, '',
    'Reproduce: `python3 tools/web/replay-debug-report.py <this file>`', '',
    '## Build and browser', '',
    `- Page: ${page.url}`, `- Browser: ${page.userAgent}`, `- Logical processors: ${page.hardwareConcurrency}`,
    `- Codec module: ${build.module}${state.analysis.enabled ? ' (decoders on the analysis module)' : ''} · profile ${build.profile}`,
    `- Source path: ${source.path} (the session line below gives the speed)`, '',
    '## Source', '',
    `- ${source.kind}${source.clip ? ` \`${source.clip}\` (${source.fps} fps, loop ${source.loopLength} ticks)` : ''}${source.sha256 ? ` · SHA-256 \`${source.sha256}\`` : ''}`,
    ...(source.renditions ?? []).map(r => `- rendition \`${r.file}\` · SHA-256 \`${r.sha256}\``),
    `- Reproducible from this report: ${source.reproducible ? 'yes' : `no (${source.reason})`}`, '',
    '## Views', '',
    ...report.views.map(view => `- **${view.letter}**: ${viewLine(view)} · ${view.frames.length} frames kept${view.stream ? ` · stream ${view.stream.complete ? `${view.stream.units.length} access units embedded` : view.stream.note}` : ''}`),
    '',
    ...report.views.flatMap(view => [`### Version ${view.letter} as shown`, '', '| Shown | Value |', '|---|---|',
      ...Object.entries(view.shown.metrics).map(([k, v]) => `| ${k} | ${v} |`),
      `| rate | ${view.shown.rateStatus} |`, `| frames | ${view.shown.frameCount} |`, `| badge | ${view.shown.badge} |`,
      ...Object.entries(view.shown.statistics).map(([k, v]) => `| ${k} | ${v.join(' / ')} |`), '']),
    '## State', '',
    `- ${state.playback} · tick ${state.tick} · clip time ${state.clipSeconds} s · loop ${state.loop}`,
    `- Session line: ${state.session}`,
    `- Analysis: ${state.analysis.enabled ? `on · layers ${state.analysis.layers.join(', ')} · layout ${state.analysis.layout} · zoom ${state.analysis.zoom.scale}×` : 'off'}`, '',
    '## Event log (most recent last)', '', '```', ...report.events, '```', '',
    '## Machine-readable state', '', JSON_FENCE, JSON.stringify(report, null, 1), '```', '',
  ];
  return lines.join('\n');
}

/** The state object of a report's Markdown (its last JSON block); throws for another schema. */
export function parseReport(markdown) {
  const start = markdown.lastIndexOf(JSON_FENCE);
  if (start < 0) throw new Error('No machine-readable state in this report');
  const body = markdown.slice(start + JSON_FENCE.length, markdown.indexOf('\n```', start + JSON_FENCE.length));
  const report = JSON.parse(body);
  if (report.schema !== REPORT_SCHEMA) throw new Error(`Report schema ${report.schema}, expected ${REPORT_SCHEMA}`);
  return report;
}

/** Clip time in seconds of `tick` on a `loopLength`-tick loop (the tick's own time without a loop). */
export function clipSeconds(tick, loopLength) {
  return tickMs(Number.isFinite(loopLength) ? tick % loopLength : tick) / 1000;
}
