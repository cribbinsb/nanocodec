// The lighter configurations the speed note offers (speed-hint.js): the heaviest
// lighter preset and the largest smaller size whose estimated encode time fits
// the frame interval, else the lightest there is.
import test from 'node:test';
import assert from 'node:assert/strict';
import { PRESET_COST, lighterChoices } from '../speed-hint.js';
import { CODECS } from '../codecs.js';

const view = (codec, preset, width, height, fps) => ({ codec, preset, width, height, fps });

test('the heaviest preset and size that fit are offered', () => {
  // AV1 Default 720p30 at 40 ms: Faster (18 ms) fits 28.3 ms, Fast (28.4 ms) just misses;
  // 854x480 (17.8 ms) is the largest smaller size that fits.
  const options = lighterChoices(view('av1', 'default', 1280, 720, 30), 40);
  assert.deepEqual(options.map(o => o.change), [{ preset: 'faster' }, { width: 854, height: 480 }]);
  assert.deepEqual(options.map(o => o.label), ['Lighter: Faster', 'Smaller: 854×480']);
});

test('nothing fits: the lightest preset and the smallest size', () => {
  const options = lighterChoices(view('h264', 'default', 1280, 720, 30), 400);
  assert.deepEqual(options.map(o => o.change), [{ preset: 'fastest' }, { width: 384, height: 216 }]);
});

test('only presets the page offers for the codec, and none for AV2 or the smallest size', () => {
  const vvc = lighterChoices(view('vvc', 'default', 640, 360, 30), 60);
  assert.deepEqual(vvc[0].change, { preset: 'fast' });
  assert.deepEqual(lighterChoices(view('av2', 'default', 384, 216, 30), 100), []);
  assert.deepEqual(lighterChoices(view('h264', 'fastest', 384, 216, 30), 100), []);
});

test('an SVC preset gets size choices only', () => {
  const options = lighterChoices(view('av1', 'svc', 1280, 720, 30), 60);
  assert.deepEqual(options.map(o => Object.keys(o.change)), [['width', 'height']]);
});

test('every codec has a cost for each preset the page offers, lighter than Default', () => {
  for (const [codec, flags] of Object.entries(CODECS)) {
    const costs = PRESET_COST[codec];
    assert.equal(costs.default, 1, codec);
    for (const preset of ['fast', 'faster', 'fastest'].filter(name => flags[name])) {
      assert.ok(costs[preset] > 0 && costs[preset] < 1, `${codec} ${preset}`);
    }
  }
});
