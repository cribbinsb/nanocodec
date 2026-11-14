// The PSNR map of analysis mode: error per block of a decoded picture against
// the exact source frame its view coded, in dB on a fixed scale.
//
// The decode worker gets from the analysis build of the decoder (nc-web
// `BrowserDecoder.error_map`, nc-dsp's SSE kernel) the sum of squared errors of
// every block of a fixed grid: MAP_BLOCK x MAP_BLOCK luma samples and the
// co-located half-size chroma blocks (one grid for all codecs, so two views
// compare block for block, unlike each codec's own partition). `errorMap`
// (run in the worker, off the page's thread) turns them into per-block PSNR of
// each plane at that grid and at a COARSE_FACTOR times coarser one, and the
// frame's PSNR of each plane. Identical blocks and planes get MAX_PSNR_DB, the
// cap nc-web's frame PSNR also uses, never Infinity, and draw as the top of
// the scale.
//
// The map is drawn on the fixed PSNR_SCALE (red at its low end, blue at its
// top), so two views compare directly.

/** Luma samples per side of a block of the fine grid (nc-web error_map.rs MAP_BLOCK). */
export const MAP_BLOCK = 8;
/** Fine blocks per side of a coarse block: 16x16 luma. */
export const COARSE_FACTOR = 2;
/** The PSNR of identical samples, and the cap of every value. */
export const MAX_PSNR_DB = 100;
/** The planes, in the order of the sums. */
export const PLANES = ['Y', 'Cb', 'Cr'];
/** PSNR-YUV's plane weights, Y:Cb:Cr, as nc-web `pixels::psnr` (the page's PSNR-YUV number). */
export const YUV_WEIGHTS = [6, 1, 1];
const PEAK_SQUARED = 255 * 255;
/** The map's fixed colour scale in dB: at or below `low` red, at or above `high` blue. */
export const PSNR_SCALE = { low: 20, high: 50 };
// The ramp of analysis.js `heat` (hue 0 red .. 240 blue, 90% saturation, 50% lightness) as RGB, and its opacity.
const SATURATION = 0.9, LIGHTNESS = 0.5, RED_TO_BLUE_DEGREES = 240, MAP_ALPHA = 128;

/** PSNR in dB of `samples` samples with squared error `sse` (MAX_PSNR_DB when exact). */
export function psnr(sse, samples) {
  return sse === 0 ? MAX_PSNR_DB : Math.min(MAX_PSNR_DB, 10 * Math.log10(PEAK_SQUARED * samples / sse));
}

/** Columns and rows of blocks `block` samples wide over a `width` x `height` plane. */
const gridOf = (width, height, block) => ({ columns: Math.ceil(width / block), rows: Math.ceil(height / block) });

/**
 * One plane's block PSNR at `factor` fine blocks per side: `sums` are its fine
 * blocks' squared errors (row by row), `width` x `height` its size, `block`
 * its fine block side. Edge blocks count only their samples inside the plane.
 */
function planeMap(sums, width, height, block, factor) {
  const fine = gridOf(width, height, block), side = block * factor, { columns, rows } = gridOf(width, height, side);
  const sse = new Float64Array(columns * rows);
  for (let y = 0; y < fine.rows; y++) {
    for (let x = 0; x < fine.columns; x++) sse[Math.floor(y / factor) * columns + Math.floor(x / factor)] += sums[y * fine.columns + x];
  }
  const values = new Float32Array(columns * rows);
  for (let y = 0; y < rows; y++) {
    const h = Math.min(side, height - y * side);
    for (let x = 0; x < columns; x++) values[y * columns + x] = psnr(sse[y * columns + x], Math.min(side, width - x * side) * h);
  }
  return values;
}

/**
 * The PSNR map of a `width` x `height` picture (4:2:0) from its block sums
 * (a Uint32Array: luma blocks, then Cb, then Cr): `{ width, height, frame,
 * fine, coarse }`, `frame` the PSNR of each plane, `fine` and `coarse`
 * `{ block, columns, rows, planes }` with one Float32Array of block PSNR per
 * plane, `block` in luma samples.
 */
export function errorMap(sums, width, height) {
  const chroma = { width: Math.ceil(width / 2), height: Math.ceil(height / 2) };
  const sizes = [{ width, height, block: MAP_BLOCK }, { ...chroma, block: MAP_BLOCK / 2 }, { ...chroma, block: MAP_BLOCK / 2 }];
  const count = gridOf(width, height, MAP_BLOCK).columns * gridOf(width, height, MAP_BLOCK).rows;
  if (sums.length !== count * PLANES.length) throw new Error(`error map: ${sums.length} sums for a ${width}x${height} picture`);
  const planeSums = sizes.map((_, i) => sums.subarray(i * count, (i + 1) * count));
  const grid = factor => ({ block: MAP_BLOCK * factor, ...gridOf(width, height, MAP_BLOCK * factor),
    planes: sizes.map((size, i) => planeMap(planeSums[i], size.width, size.height, size.block, factor)) });
  const frame = sizes.map((size, i) => psnr(planeSums[i].reduce((a, b) => a + b, 0), size.width * size.height));
  return { width, height, frame, fine: grid(1), coarse: grid(COARSE_FACTOR) };
}

/** PSNR-YUV from the PSNR of each plane (`errorMap` `frame`): their YUV_WEIGHTS mean, as nc-web computes it. */
export function yuvPsnr(frame) {
  return frame.reduce((sum, value, i) => sum + YUV_WEIGHTS[i] * value, 0) / YUV_WEIGHTS.reduce((a, b) => a + b, 0);
}

/** The buffers of an `errorMap` result, to transfer it from the worker. */
export function errorMapBuffers(map) {
  return [map.fine, map.coarse].flatMap(grid => grid.planes.map(plane => plane.buffer));
}

/** A PSNR as text: two decimals, or the cap named as identical samples. */
export function formatPsnr(value) {
  return value >= MAX_PSNR_DB ? `${MAX_PSNR_DB} (identical)` : value.toFixed(2);
}

/** The RGBA colour of `value` dB on PSNR_SCALE (clamped at both ends). */
export function psnrColour(value) {
  const t = Math.max(0, Math.min(1, (value - PSNR_SCALE.low) / (PSNR_SCALE.high - PSNR_SCALE.low)));
  return [...hslToRgb(RED_TO_BLUE_DEGREES * t, SATURATION, LIGHTNESS), MAP_ALPHA];
}

/** HSL (hue in degrees, saturation and lightness 0..1) as 8-bit RGB (CSS Color 4 `hsl()`). */
function hslToRgb(hue, saturation, lightness) {
  const a = saturation * Math.min(lightness, 1 - lightness);
  const channel = n => {
    const k = (n + hue / 30) % 12;
    return Math.round(255 * (lightness - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [channel(0), channel(8), channel(4)];
}

/** One grid of `map` (the fine one, or the coarse one) as RGBA, one pixel per block, for `plane` (0 Y, 1 Cb, 2 Cr). */
export function errorPixels(map, fine, plane) {
  const grid = fine ? map.fine : map.coarse, values = grid.planes[plane];
  const rgba = new Uint8ClampedArray(values.length * 4);
  values.forEach((value, i) => rgba.set(psnrColour(value), i * 4));
  return { rgba, columns: grid.columns, rows: grid.rows, block: grid.block };
}

/**
 * The hover line for picture point (x, y) of a `picture` `{ width, height }`
 * (the map may be at another size: a smaller spatial layer is scored against
 * its source upscaled): the block's PSNR of every plane at the drawn grid.
 */
export function describeError(map, x, y, fine, picture) {
  const grid = fine ? map.fine : map.coarse;
  const column = Math.floor(x * map.width / picture.width / grid.block), row = Math.floor(y * map.height / picture.height / grid.block);
  if (column < 0 || row < 0 || column >= grid.columns || row >= grid.rows) return '';
  const values = grid.planes.map((plane, p) => `${PLANES[p]} ${formatPsnr(plane[row * grid.columns + column])}`);
  return `PSNR vs source, ${grid.block}×${grid.block} block at (${column * grid.block}, ${row * grid.block}): ${values.join(' · ')} dB`;
}
