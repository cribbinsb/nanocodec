// plotDomain (web/results/domain.js): the results scatter's axes cover the bulk of
// the points, not one far outlier.
import test from 'node:test';
import assert from 'node:assert/strict';
import { plotDomain } from '../results/domain.js';

// Twenty encoders from 40 to 400 fps and -40 % to +40 %, ours the first five.
const bulk = Array.from({ length: 20 }, (_, i) => ({
  x: 40 * 10 ** (i / 19), y: -40 + (80 * ((i * 7) % 20)) / 19, kind: i < 5 ? 'ours' : 'ref',
}));
const inside = (d, p) => p.x >= d.x[0] && p.x <= d.x[1] && p.y >= d.y[0] && p.y <= d.y[1];

test('a far-slow and a far-worse reference fall outside; the bulk stays inside', () => {
  const slow = { x: 0.7, y: -45, kind: 'ref' };
  const poor = { x: 400, y: 250, kind: 'ref' };
  const d = plotDomain([...bulk, slow, poor]);
  assert.ok(bulk.every((p) => inside(d, p)));
  assert.ok(!inside(d, slow) && !inside(d, poor));
});

test('a point just past the bulk widens the axes instead of being pinned', () => {
  const near = { x: 500, y: 55, kind: 'ref' };
  assert.ok(inside(plotDomain([...bulk, near]), near));
});

test('our points and the reference frontier inside our speed range always stay', () => {
  const oursFar = { x: 0.5, y: -60, kind: 'ours' };
  const d = plotDomain([...bulk, oursFar]);
  assert.ok(inside(d, oursFar));
});

test('too few points for quartiles: every point sets the range', () => {
  const few = [{ x: 1, y: 0, kind: 'ref' }, { x: 1000, y: 200, kind: 'ref' }, { x: 10, y: 5, kind: 'ours' }];
  assert.ok(few.every((p) => inside(plotDomain(few), p)));
});
