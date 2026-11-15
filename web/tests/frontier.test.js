// paretoFront and frontierLine (web/results/frontier.js), the overview's Pareto frontier.
import test from 'node:test';
import assert from 'node:assert/strict';
import { dominates, paretoFront, frontierLine } from '../results/frontier.js';

test('a point beaten on both axes is off the frontier', () => {
  const a = { x: 100, y: -40 };
  const b = { x: 50, y: -30 }; // slower and worse than a
  const c = { x: 200, y: -10 }; // faster but worse: a trade-off, it stays
  assert.equal(dominates(a, b), true);
  assert.deepEqual(paretoFront([a, b, c]), [a, c]);
});

test('ties: equal on one axis and better on the other dominates; identical points both stay', () => {
  const a = { x: 100, y: -40 };
  const sameSpeedWorse = { x: 100, y: -30 };
  const twin = { x: 100, y: -40 };
  assert.equal(dominates(a, sameSpeedWorse), true);
  assert.deepEqual(paretoFront([a, sameSpeedWorse, twin]), [a, twin]);
});

test('a single point is its own frontier; the line joins the frontier points directly', () => {
  const p = { x: 10, y: 5 };
  assert.deepEqual(paretoFront([p]), [p]);
  assert.deepEqual(frontierLine([{ x: 10, y: -40 }, { x: 50, y: -20 }]), [[10, -40], [50, -20]]);
});
