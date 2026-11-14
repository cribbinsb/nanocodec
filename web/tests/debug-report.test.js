// The debug report (debug-report.js): its schema, the history window, the
// source frame mapping, and a Markdown round trip through its JSON block.
import test from 'node:test';
import assert from 'node:assert/strict';
import { MIN_HISTORY_FRAMES, REPORT_SCHEMA, buildReport, clipSeconds, formatReport, historyWindow, parseReport, sourceFrameOf } from '../debug-report.js';

const frame = (tick, key = false) => ({ tick, key, repeat: false, quantizer: 30, bytes: 3, packet: new Uint8Array([tick, 1, 2]),
  encodeMs: 1, decodeMs: 1, psnr: 40, planes: null, colour: 'bt709' });

test('the history keeps the frames since the last key frame, and at least MIN_HISTORY_FRAMES', () => {
  const long = Array.from({ length: 1000 }, (_, i) => frame(i, i === 100));
  assert.equal(historyWindow(long)[0].tick, 100);
  const recentKey = Array.from({ length: 1000 }, (_, i) => frame(i, i === 990));
  assert.equal(historyWindow(recentKey).length, MIN_HISTORY_FRAMES);
  assert.equal(historyWindow([frame(0), frame(1)]).length, 2);
});

test('source frames follow the clip loop and frame rate', () => {
  const source = { fps: 60, loopLength: 300 };
  assert.equal(sourceFrameOf(0, source), 0);
  assert.equal(sourceFrameOf(301, source), 2);
  assert.equal(sourceFrameOf(5, { fps: null, loopLength: null }), null);
  assert.equal(clipSeconds(330, 300), 1);
});

test('a report round-trips through its Markdown, with hashes and the optional streams', async () => {
  const view = { letter: 'A', config: { source: 'live', codec: 'h264', preset: 'default', width: 64, height: 64, fps: 30, rateMode: 'cbr', targetKbps: 300 },
    streamId: null, streamBytes: null, shown: { metrics: { bitrate: '300' }, rateStatus: '', frameCount: '', badge: '', statistics: {} },
    window: { fps: 30, frames: [] }, interventions: [], log: [frame(0, true), frame(1)] };
  const page = { build: { commit: 'abc', dirty: false, profile: 'release', module: 'SIMD128 (pkg)' },
    page: { url: 'http://x/', time: 't', userAgent: 'ua', hardwareConcurrency: 4 },
    source: { kind: 'clip', clip: 'c', fps: 30, loopLength: 300, path: 'exact', reproducible: true },
    state: { playback: 'paused', tick: 1, clipSeconds: 0, loop: 0, session: 's', analysis: { enabled: false } }, events: ['1 start'] };
  const report = await buildReport(page, [view], { includeStreams: true });
  assert.equal(report.schema, REPORT_SCHEMA);
  assert.equal(report.views[0].frames[0].sha256, 'ae4b3280e56e2faf83f414a6e3dabe9d5fbe18976544c05fed121accb85b53fc');
  assert.equal(report.views[0].frames[1].sourceFrame, 1);
  assert.equal('packet' in report.views[0].frames[0], false);
  assert.equal(report.views[0].stream.units.length, 2);
  const markdown = formatReport(report);
  assert.match(markdown, /^# nanocodec debug report/);
  assert.deepEqual(parseReport(markdown), JSON.parse(JSON.stringify(report)));
  assert.throws(() => parseReport(markdown.replaceAll(REPORT_SCHEMA, 'other/1')));
});
