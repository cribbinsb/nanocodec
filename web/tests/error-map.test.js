// The PSNR map (error-map.js): block PSNR from nc-web's per-block squared
// errors at the fine and coarse grids, edge blocks, frame PSNR per plane and
// the cap for identical samples.
import test from 'node:test';
import assert from 'node:assert/strict';
import { MAX_PSNR_DB, describeError, errorMap, psnr, psnrColour } from '../error-map.js';

const PEAK_SQUARED = 255 * 255;
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-4, `${a} vs ${b}`);

// A 24x10 picture: luma 3x2 blocks of 8 (the right column 8 wide, the bottom row
// 2 high); chroma 12x5 in 3x2 blocks of 4 (bottom row 1 high).
const WIDTH = 24, HEIGHT = 10, BLOCKS = 6;
function sums({ y = [], cb = [], cr = [] }) {
  const out = new Uint32Array(3 * BLOCKS);
  [y, cb, cr].forEach((plane, p) => plane.forEach((value, i) => { out[p * BLOCKS + i] = value; }));
  return out;
}

test('PSNR of a block is 10 log10(255^2 n / SSE), capped for identical samples', () => {
  close(psnr(64, 64), 10 * Math.log10(PEAK_SQUARED));
  assert.equal(psnr(0, 64), MAX_PSNR_DB);
  assert.ok(Number.isFinite(psnr(0, 64)));
});

test('the fine grid counts each block’s own samples, edge blocks included', () => {
  const map = errorMap(sums({ y: [64, 0, 0, 0, 0, 16], cb: [16], cr: [0, 0, 0, 4] }), WIDTH, HEIGHT);
  assert.deepEqual([map.fine.columns, map.fine.rows, map.fine.block], [3, 2, 8]);
  const [y, cb, cr] = map.fine.planes;
  close(y[0], psnr(64, 64));
  // The bottom-right luma block is 8x2.
  close(y[5], psnr(16, 16));
  assert.equal(y[1], MAX_PSNR_DB);
  close(cb[0], psnr(16, 16));
  // The bottom-left chroma block is 4x1.
  close(cr[3], psnr(4, 4));
});

test('the coarse grid merges 2x2 fine blocks, and frame PSNR covers each whole plane', () => {
  const map = errorMap(sums({ y: [64, 32, 8, 16, 0, 0], cb: [4, 0, 0, 0, 0, 0] }), WIDTH, HEIGHT);
  assert.deepEqual([map.coarse.columns, map.coarse.rows, map.coarse.block], [2, 1, 16]);
  // 16x10 samples on the left, 8x10 on the right.
  close(map.coarse.planes[0][0], psnr(64 + 32 + 16, 160));
  close(map.coarse.planes[0][1], psnr(8, 80));
  close(map.frame[0], psnr(64 + 32 + 8 + 16, WIDTH * HEIGHT));
  close(map.frame[1], psnr(4, 12 * 5));
  assert.equal(map.frame[2], MAX_PSNR_DB);
});

test('a sum count that does not fit the picture is refused', () => {
  assert.throws(() => errorMap(new Uint32Array(5), WIDTH, HEIGHT));
});

test('colours run from red at the low end of the scale to blue at the top, identical as the top', () => {
  const [r1, , b1] = psnrColour(10), [r2, , b2] = psnrColour(MAX_PSNR_DB);
  assert.ok(r1 > b1 && b2 > r2);
  assert.deepEqual(psnrColour(MAX_PSNR_DB), psnrColour(60));
});

test('the hover line names the block, its planes and the cap', () => {
  const map = errorMap(sums({ y: [64, 0, 0, 0, 0, 0] }), WIDTH, HEIGHT);
  const line = describeError(map, 3, 3, true, { width: WIDTH, height: HEIGHT });
  assert.match(line, /^PSNR vs source, 8×8 block at \(0, 0\): Y 48\.\d\d · Cb 100 \(identical\) · Cr 100 \(identical\) dB$/);
});
