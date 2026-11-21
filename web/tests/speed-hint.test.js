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
  const options = lighterChoices(view('av1', 'medium', 1280, 720, 30), 40);
  assert.deepEqual(options.map(o => o.change), [{ preset: 'faster' }, { width: 854, height: 480 }]);
  assert.deepEqual(options.map(o => o.label), ['Lighter: Faster', 'Smaller: 854×480']);
});

test('nothing fits: the lightest preset and the smallest size', () => {
  const options = lighterChoices(view('h264', 'medium', 1280, 720, 30), 400);
  assert.deepEqual(options.map(o => o.change), [{ preset: 'veryfast' }, { width: 384, height: 216 }]);
});

test('only presets the page offers for the codec, and no size below the smallest', () => {
  // VVC Fast (0.63) is not light enough at 60 ms for 30 fps; Faster (0.55) is the lightest.
  const vvc = lighterChoices(view('vvc', 'medium', 640, 360, 30), 60);
  assert.deepEqual(vvc[0].change, { preset: 'faster' });
  // AV2 offers Fast and Faster since 2026-11-18 (d6bd02130); at the smallest size only a preset is left.
  assert.deepEqual(lighterChoices(view('av2', 'medium', 384, 216, 30), 100).map(o => o.change),
                   [{ preset: 'faster' }]);
  assert.deepEqual(lighterChoices(view('h264', 'veryfast', 384, 216, 30), 100), []);
});

test('a spatially layered stream gets size choices only', () => {
  const options = lighterChoices({ ...view('av1', 'medium', 1280, 720, 30), spatialLayers: 2 }, 60);
  assert.deepEqual(options.map(o => Object.keys(o.change)), [['width', 'height']]);
});

test('every estimated preset is one the page offers for the codec, lighter than Medium (Slow and Slower heavier)', () => {
  for (const [codec, { presets }] of Object.entries(CODECS)) {
    const costs = PRESET_COST[codec];
    assert.equal(costs.medium, 1, codec);
    for (const [preset, cost] of Object.entries(costs).filter(([name]) => name !== 'medium')) {
      assert.ok(presets.includes(preset), `${codec} ${preset}`);
      assert.ok(['slow', 'slower', 'veryslow'].includes(preset) ? cost > 1 : cost > 0 && cost < 1, `${codec} ${preset}`);
    }
  }
});

test('a view coded at Slower that falls behind is offered Medium', () => {
  const choices = lighterChoices({ codec: 'hevc', preset: 'slower', width: 1280, height: 720, fps: 30 }, 60);
  assert.ok(choices.some(choice => choice.change.preset === 'medium'), JSON.stringify(choices));
});
