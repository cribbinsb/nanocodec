// Links between the demo and the results pages (deep-link.js): what one page
// writes, the other reads back unchanged, and every chart reference family that
// the demo has pre-encoded streams of finds them.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  CHART_PRESETS, chartFamily, demoQuery, manifestEncoder, parseDemoQuery, parseResultsQuery, resultsQuery,
} from '../deep-link.js';

const web = new URL('../', import.meta.url);
const chart = JSON.parse(readFileSync(new URL('results/data/chart.json', web)));
const manifest = JSON.parse(readFileSync(new URL('streams-ref/manifest.json', web)));

test('a demo link reads back as written', () => {
  const link = { codec: 'av1', preset: 'fast', width: 1280, height: 720, fps: 30, kbps: 600 };
  assert.deepEqual(parseDemoQuery(demoQuery(link)), link);
  const stream = { codec: 'h264', preset: 'default', stream: 'xiph-johnny-x264-medium-1280x720-30fps-300k' };
  assert.deepEqual(parseDemoQuery(demoQuery(stream)), stream);
});

test('a results link reads back as written, and names a chart family', () => {
  for (const preset of CHART_PRESETS) {
    assert.deepEqual(parseResultsQuery(resultsQuery({ codec: 'hevc', preset })), { codec: 'hevc', preset });
  }
  assert.equal(chartFamily({ codec: 'hevc', preset: 'default' }), 'nanocodec-hevc');
  assert.equal(chartFamily({ codec: 'h264', preset: 'faster' }), 'nanocodec-h264@h264-faster');
});

test('every preset of ours on the chart has a link that names it', () => {
  for (const [family, point] of Object.entries(chart.points).filter(([, p]) => p.ours)) {
    const link = parseResultsQuery(resultsQuery({ codec: point.codec, preset: point.preset.toLowerCase() }));
    assert.equal(chartFamily(link), family);
  }
});

test('links without settings, or with bad ones, ask for nothing', () => {
  assert.equal(parseDemoQuery(''), null);
  assert.equal(parseDemoQuery('?views=live'), null);
  assert.equal(parseDemoQuery('?codec=av1&preset=svc'), null);
  assert.equal(parseDemoQuery('?codec=<b>'), null);
  assert.deepEqual(parseDemoQuery('?codec=av1&size=big&fps=-3&stream=../x'), { codec: 'av1', preset: 'default' });
  assert.equal(parseResultsQuery('?preset=fast'), null);
});

test('chart reference families find the demo\'s streams of the same encoder', () => {
  const encoders = new Set(manifest.streams.map((s) => s.encoder));
  const linked = Object.keys(chart.points).filter((f) => !chart.points[f].ours && encoders.has(manifestEncoder(f)));
  for (const family of ['x264-medium', 'svtav1-m12', 'libvpx-vp9-cpu7', 'libaom-av1-cpu8', 'openh264-baseline', 'x265-medium']) {
    assert.ok(linked.includes(family), family);
  }
  assert.equal(manifestEncoder('x264-medium~psnr'), 'x264-medium');
  assert.equal(manifestEncoder('nvenc-h264'), null);
});
