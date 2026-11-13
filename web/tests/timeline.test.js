// The shared source timeline (timeline.js): frame-rate steps, common instants
// for pause and stepping, and how a video's own frames map onto the 30 Hz ticks.
import test from 'node:test';
import assert from 'node:assert/strict';
import { FRAME_RATES, SOURCE_FPS, codesTick, commonStep, frameAtTick, loopTicks, nextCommonTick, tickStep, ticksOfFrame } from '../timeline.js';

test('every offered frame rate divides the source rate', () => {
  assert.deepEqual(FRAME_RATES, [30, 15, 10, 6, 5]);
  assert.deepEqual(FRAME_RATES.map(tickStep), [1, 2, 3, 5, 6]);
  assert.ok(FRAME_RATES.every(Number.isInteger));
  assert.throws(() => tickStep(25));
  assert.throws(() => tickStep(60));
});

test('a common instant is one every view codes', () => {
  assert.equal(commonStep([1, 2]), 2);
  assert.equal(commonStep([2, 3]), 6);
  assert.equal(commonStep([4, 6]), 12);
  // 30 fps against 15 fps: the next 15 fps instant.
  assert.equal(nextCommonTick(7, [1, 2]), 8);
  assert.equal(nextCommonTick(8, [1, 2]), 10);
  // 10 fps against 6 fps.
  assert.equal(nextCommonTick(0, [3, 5]), 15);
  for (const tick of [15, 30]) assert.ok([3, 5].every(step => codesTick(step, tick)));
});

test('a 25 fps clip repeats a frame every sixth tick, as a 30 Hz sampler sees it', () => {
  const shown = Array.from({ length: 12 }, (_, tick) => frameAtTick(tick, 25));
  assert.deepEqual(shown, [0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9]);
  assert.deepEqual(ticksOfFrame(0, 25), [0, 1]);
  assert.deepEqual(ticksOfFrame(1, 25), [2, 2]);
  // Every frame's ticks show that frame.
  for (const fps of [24, 25, 30, 60]) for (let frame = 0; frame < 120; frame++) {
    const ticks = ticksOfFrame(frame, fps);
    if (ticks) for (let tick = ticks[0]; tick <= ticks[1]; tick++) assert.equal(frameAtTick(tick, fps), frame);
  }
  // A 60 fps clip: only every second frame falls on a tick.
  assert.equal(ticksOfFrame(1, 60), null);
  assert.equal(loopTicks(250, 25), 10 * SOURCE_FPS);
  assert.equal(loopTicks(288, 24), 12 * SOURCE_FPS);
});
