// Coding-tool analysis records (analysis.js): the packed layout shared with
// nc-common `pack_records`, the summary by area, intra directions and the
// block lookup behind the hover record.
import test from 'node:test';
import assert from 'node:assert/strict';
import { FIELD, HALFWORDS, KIND, PRED, RunningSummary, describeBlock, intraDirection, recordCount, recordsAt, summarise } from '../analysis.js';

/** Pack records given as field objects (the nc-common layout). */
function pack(list) {
  const out = new Int16Array(list.length * HALFWORDS);
  list.forEach((record, i) => { for (const [name, value] of Object.entries(record)) out[i * HALFWORDS + FIELD[name]] = value; });
  return out;
}

// A 32x16 picture: an intra 16x16 block (planar) and a skipped one beside it
// with a half-sample vector, one coded 16x16 transform in the intra block.
const RECORDS = pack([
  { kind: KIND.coding, pred: PRED.intra, x: 0, y: 0, w: 16, h: 16, qp: 30, bits: 512 },
  { kind: KIND.prediction, pred: PRED.intra, x: 0, y: 0, w: 16, h: 16, mode: 0, cls: 0, ref0: -1, ref1: -1 },
  { kind: KIND.transform, pred: PRED.intra, x: 0, y: 0, w: 16, h: 16, flags: 1, qp: 30 },
  { kind: KIND.coding, pred: PRED.skip, x: 16, y: 0, w: 16, h: 16, qp: 34, bits: 2 },
  { kind: KIND.prediction, pred: PRED.skip, x: 16, y: 0, w: 16, h: 16, mv0x: 2, mv0y: -4, ref1: -1 },
]);

test('the layout matches nc-common pack_records', () => {
  // packs_records_in_the_documented_layout (stream_stats.rs) checks the same order.
  assert.equal(HALFWORDS, 24);
  assert.deepEqual(Object.values(FIELD), [...Array(HALFWORDS).keys()]);
  assert.equal(recordCount(RECORDS), 5);
  assert.equal(recordCount(null), 0);
});

test('a frame summary is by luma area', () => {
  const s = summarise(RECORDS);
  assert.deepEqual(s.pred, { intra: 0.5, skip: 0.5 });
  assert.deepEqual(s.cbSize, { 16: 1 });
  assert.deepEqual(s.mvPrecision, { half: 1 });
  assert.equal(s.zeroMv, 0);
  assert.equal(s.coded, 0.5);
  assert.equal(s.qp, 32);
  assert.deepEqual(s.qpRange, [30, 34]);
  assert.equal(s.bitsPerPixel, 514 / 512);
  assert.equal(summarise(new Int16Array(0)), null);
  const running = new RunningSummary();
  running.add(s); running.add(s);
  assert.equal(running.mean().qp, 32);
  assert.deepEqual(running.mean().pred, { intra: 0.5, skip: 0.5 });
});

test('intra directions follow each standard', () => {
  assert.equal(intraDirection('hevc', 0, 8), null);
  assert.equal(intraDirection('hevc', 1, 8), null);
  assert.deepEqual(intraDirection('hevc', 10, 8), [32, -0]);
  assert.deepEqual(intraDirection('hevc', 26, 8), [-0, 32]);
  assert.deepEqual(intraDirection('hevc', 2, 8), [32, -32]);
  assert.deepEqual(intraDirection('h264', 0, 4), [0, 1]);
  assert.equal(intraDirection('h264', 2, 4), null);
  assert.equal(intraDirection('h264', 3, 16), null);
  assert.equal(intraDirection('av1', 0, 8), null);
  const vertical = intraDirection('av1', 1, 8);
  assert.ok(Math.abs(vertical[0]) < 1e-9 && vertical[1] === -1);
});

test('AV1 vectors are in eighth samples', () => {
  const av1 = pack([{ kind: KIND.coding, pred: PRED.inter, w: 8, h: 8 },
    { kind: KIND.prediction, pred: PRED.inter, w: 8, h: 8, mv0x: 1, mv0y: 8, ref1: -1 }]);
  assert.deepEqual(summarise(av1, 'av1').mvPrecision, { eighth: 1 });
  assert.deepEqual(summarise(av1, 'hevc').mvPrecision, { quarter: 1 });
});

test('the block under a point and its description', () => {
  const found = recordsAt(RECORDS, 20, 5);
  assert.equal(found.coding, 3 * HALFWORDS);
  assert.equal(found.prediction, 4 * HALFWORDS);
  assert.equal(found.transform, -1);
  const lines = describeBlock(RECORDS, found);
  assert.match(lines[0], /^skip 16×16 at \(16, 0\)/);
  assert.match(lines.at(-1), /MV \(0.5, -1\) px · ref 0 · mode 0/);
  assert.deepEqual(recordsAt(RECORDS, 40, 5), { coding: -1, prediction: -1, transform: -1 });
});
