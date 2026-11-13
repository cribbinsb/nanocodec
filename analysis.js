// Coding-tool analysis of decoded pictures: reading the decoder's packed block
// records, summarising them and drawing them over the picture.
//
// With analysis on, the decode worker passes, with each shown frame, the
// decoder's block records (nc-common `stream_stats`: one record per coding,
// prediction, transform and loop-filter block, packed by `pack_records` into
// HALFWORDS signed 16-bit values each, in the FIELD order below; nc-web
// `BrowserDecoder.analysis_frame`). Everything here works on that flat
// Int16Array: no per-block objects, so a 720p frame's few thousand records
// cost one pass to summarise and one to draw.
//
// Drawing is in picture coordinates through the context's transform (the
// caller maps picture to screen pixels, zoom and pan included), with line
// widths divided by the scale so outlines stay one screen pixel at any zoom.

/** Codecs whose decoders fill the records (others show a notice). */
export const ANALYSIS_CODECS = new Set(['h264', 'hevc', 'av1']);
/** Values per record and their order (nc-common `pack_records`). */
export const HALFWORDS = 24;
export const FIELD = {
  kind: 0, pred: 1, shape: 2, depth: 3, x: 4, y: 5, w: 6, h: 7, mode: 8, mode2: 9, cls: 10, flags: 11,
  qp: 12, qpDelta: 13, ref0: 14, ref1: 15, mv0x: 16, mv0y: 17, mv1x: 18, mv1y: 19, mvd0x: 20, mvd0y: 21,
  nonzero: 22, bits: 23,
};
export const KIND = { coding: 0, prediction: 1, transform: 2, filter: 3 };
export const PRED = { intra: 0, inter: 1, skip: 2, merge: 3 };
const PRED_NAMES = ['intra', 'inter', 'skip', 'merge'];
const SHAPE_NAMES = ['2Nx2N', '2NxN', 'Nx2N', 'NxN', '2NxnU', '2NxnD', 'nLx2N', 'nRx2N', '4x4 (16)', 'PCM'];
const INTRA_CLASS = { planar: 0, dc: 1, angular: 2 };
const FLAG = { codedY: 1, codedCb: 2, codedCr: 4, transformSkip: 8, qpDeltaCoded: 16, bi: 32 };
/** Each codec's record conventions: motion vector units per luma sample, and
 * the QP heat map's scale (H.264 / HEVC QP; AV1 qindex, 0..255). */
const CODEC_UNITS = {
  h264: { mv: 4, qp: { low: 15, high: 51 } },
  hevc: { mv: 4, qp: { low: 15, high: 51 } },
  av1: { mv: 8, qp: { low: 40, high: 255 } },
};
const units = codec => CODEC_UNITS[codec] ?? CODEC_UNITS.hevc;
/** The overlay layers a viewer can switch on. */
export const LAYERS = ['blocks', 'transforms', 'vectors', 'intra', 'qp', 'bits'];
export const LAYER_LABELS = { blocks: 'Blocks', transforms: 'Transforms', vectors: 'Motion vectors', intra: 'Intra modes', qp: 'QP map', bits: 'Bits map' };
export const PRED_COLOURS = ['#ff5a36', '#3b8bff', '#a8b3ad', '#27c27a'];
// Heat maps use fixed scales so two views compare directly.
const BITS_PER_PIXEL_SCALE = { low: Math.log2(1 / 64), high: Math.log2(4) };
const HEAT_ALPHA = 0.5;
// Fill of a coding block in the block layer; outlines are opaque.
const BLOCK_FILL_ALPHA = 0.16;
// Vectors shorter than this many screen pixels are drawn as dots.
const MIN_VECTOR_PX = 1.5;
// Arrow head length in screen pixels.
const ARROW_PX = 4;
// An intra direction mark is drawn when its block is at least this many screen pixels wide.
const MIN_INTRA_MARK_PX = 6;
// HEVC intraPredAngle by mode (H.265 Table 8-5), modes 2..34.
const HEVC_ANGLE = [32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32];
const HEVC_FIRST_ANGULAR = 2, HEVC_FIRST_VERTICAL = 18;
// H.264 Intra 4x4 / 8x8 prediction directions (H.264 Table 8-2), as lines through the block (x, y):
// 0 vertical, 1 horizontal, 2 DC, 3 diagonal down-left, 4 diagonal down-right, 5 vertical-right,
// 6 horizontal-down, 7 vertical-left, 8 horizontal-up.
const H264_NXN_DIRECTION = [[0, 1], [1, 0], null, [1, -1], [1, 1], [0.5, 1], [1, 0.5], [-0.5, 1], [1, -0.5]];
// Intra_16x16 modes (Table 8-4): 0 vertical, 1 horizontal, 2 DC, 3 plane.
const H264_16X16_DIRECTION = [[0, 1], [1, 0], null, null];
const H264_MB = 16;
// AV1 directional y modes (V_PRED .. D67_PRED, symbols 1..8) by nominal angle in degrees (AV1 §7.11.2).
const AV1_DIRECTIONS = { 1: 90, 2: 180, 3: 45, 4: 135, 5: 113, 6: 157, 7: 203, 8: 67 };

/** The number of records in `records` (an Int16Array of packed records). */
export function recordCount(records) { return records ? Math.floor(records.length / HALFWORDS) : 0; }

/** The direction of an intra mode as a line `[dx, dy]` in picture coordinates, or null (planar, DC). */
export function intraDirection(codec, mode, size) {
  if (codec === 'av1') {
    const degrees = AV1_DIRECTIONS[mode];
    if (degrees === undefined) return null;
    // AV1 angles: 90 is vertical (predicted from above), 180 horizontal.
    const radians = degrees * Math.PI / 180;
    return [Math.cos(radians), -Math.sin(radians)];
  }
  if (codec === 'hevc') {
    if (mode < HEVC_FIRST_ANGULAR) return null;
    const angle = HEVC_ANGLE[mode - HEVC_FIRST_ANGULAR];
    // Horizontal modes predict from the left column, vertical ones from the top row.
    return mode < HEVC_FIRST_VERTICAL ? [32, -angle] : [-angle, 32];
  }
  return (size >= H264_MB ? H264_16X16_DIRECTION : H264_NXN_DIRECTION)[mode] ?? null;
}

/** A colour on a blue-to-red ramp for `t` in 0..1. */
function heat(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const hue = 240 - 240 * clamped;
  return `hsla(${hue.toFixed(0)}, 90%, 50%, ${HEAT_ALPHA})`;
}

/**
 * Shares of one picture's records, by luma area: coding block sizes and
 * prediction, partition shapes, motion vector precision, transform sizes,
 * coded residual, mean QP, bits per pixel. `null` for no records.
 */
export function summarise(records, codec = 'hevc') {
  const mvUnits = units(codec).mv;
  const n = recordCount(records);
  if (!n) return null;
  const s = { area: 0, cbSize: {}, pred: {}, shape: {}, mvPrecision: {}, tbSize: {}, tbArea: 0, codedArea: 0,
    qpSum: 0, qpMin: Infinity, qpMax: -Infinity, qpDeltas: 0, bits: 0, inter: 0, zeroMv: 0, intraClass: {} };
  const add = (map, key, value) => { map[key] = (map[key] ?? 0) + value; };
  for (let i = 0; i < n; i++) {
    const r = i * HALFWORDS, kind = records[r + FIELD.kind], area = records[r + FIELD.w] * records[r + FIELD.h];
    if (kind === KIND.coding) {
      const pred = PRED_NAMES[records[r + FIELD.pred]], qp = records[r + FIELD.qp];
      s.area += area;
      add(s.cbSize, sizeName(records[r + FIELD.w], records[r + FIELD.h]), area);
      add(s.pred, pred, area);
      if (pred !== 'skip') add(s.shape, SHAPE_NAMES[records[r + FIELD.shape]] ?? '?', area);
      s.qpSum += qp * area; s.qpMin = Math.min(s.qpMin, qp); s.qpMax = Math.max(s.qpMax, qp);
      if (records[r + FIELD.flags] & FLAG.qpDeltaCoded && records[r + FIELD.qpDelta]) s.qpDeltas++;
      s.bits += records[r + FIELD.bits];
    } else if (kind === KIND.prediction) {
      if (records[r + FIELD.pred] === PRED.intra) {
        add(s.intraClass, ['planar', 'DC', 'angular', 'other'][records[r + FIELD.cls]], area);
        continue;
      }
      const x = records[r + FIELD.mv0x], y = records[r + FIELD.mv0y];
      s.inter += area;
      if (!x && !y) s.zeroMv += area;
      add(s.mvPrecision, PRECISION_NAMES[Math.max(precision(x, mvUnits), precision(y, mvUnits))], area);
    } else if (kind === KIND.transform) {
      s.tbArea += area;
      add(s.tbSize, sizeName(records[r + FIELD.w], records[r + FIELD.h]), area);
      if (records[r + FIELD.flags] & FLAG.codedY) s.codedArea += area;
    }
  }
  return {
    cbSize: shares(s.cbSize), pred: shares(s.pred), shape: shares(s.shape), intraClass: shares(s.intraClass),
    mvPrecision: shares(s.mvPrecision), zeroMv: s.inter ? s.zeroMv / s.inter : null, tbSize: shares(s.tbSize),
    coded: s.area ? s.codedArea / s.area : 0, qp: s.area ? s.qpSum / s.area : null, qpRange: [s.qpMin, s.qpMax],
    qpDeltas: s.qpDeltas, bitsPerPixel: s.area ? s.bits / s.area : 0,
  };
}

/** 0 integer, 1 half, 2 quarter sample. */
const PRECISION_NAMES = ['integer', 'half', 'quarter', 'eighth'];
/** 0 integer, 1 half, 2 quarter, 3 eighth sample, for vectors of `per` units per sample. */
function precision(v, per) {
  const fraction = ((v % per) + per) % per;
  if (fraction === 0) return 0;
  let step = per / 2, level = 1;
  while (fraction % step !== 0) { step /= 2; level++; }
  return level;
}

function sizeName(w, h) { return w === h ? String(w) : `${w}x${h}`; }

function shares(map) {
  const total = Object.values(map).reduce((a, b) => a + b, 0);
  return Object.fromEntries(Object.entries(map).map(([k, v]) => [k, total ? v / total : 0]));
}

/** Averages of `summarise` results over the frames seen since `reset`. */
export class RunningSummary {
  constructor() { this.reset(); }
  reset() { this.frames = 0; this.sums = {}; this.scalars = {}; }
  add(summary) {
    if (!summary) return;
    this.frames++;
    for (const [key, value] of Object.entries(summary)) {
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        const target = this.sums[key] ??= {};
        for (const [k, v] of Object.entries(value)) target[k] = (target[k] ?? 0) + v;
      } else if (typeof value === 'number') {
        this.scalars[key] = (this.scalars[key] ?? 0) + value;
      }
    }
  }
  /** The mean summary (shares averaged per frame), or null before the first frame. */
  mean() {
    if (!this.frames) return null;
    const result = {};
    for (const [key, map] of Object.entries(this.sums)) {
      result[key] = Object.fromEntries(Object.entries(map).map(([k, v]) => [k, v / this.frames]));
    }
    for (const [key, value] of Object.entries(this.scalars)) result[key] = value / this.frames;
    return result;
  }
}

/**
 * Draw `layers` (a Set of LAYERS names) of `records` on `context`, whose
 * transform maps picture coordinates to the target; `scale` is target pixels
 * per picture pixel (for one-pixel lines and size thresholds), `clip`
 * `{ x, y, width, height }` the visible picture region (records outside it
 * are skipped).
 */
export function drawOverlay(context, records, { codec, layers, scale, clip }) {
  const n = recordCount(records);
  if (!n || !layers.size) return;
  const px = 1 / scale, qpScale = units(codec).qp, mvUnits = units(codec).mv;
  const visible = r => records[r + FIELD.x] < clip.x + clip.width && records[r + FIELD.y] < clip.y + clip.height
    && records[r + FIELD.x] + records[r + FIELD.w] > clip.x && records[r + FIELD.y] + records[r + FIELD.h] > clip.y;
  context.save();
  context.lineWidth = px;
  // Heat maps first, under the outlines.
  if (layers.has('qp') || layers.has('bits')) {
    for (let i = 0; i < n; i++) {
      const r = i * HALFWORDS;
      if (records[r + FIELD.kind] !== KIND.coding || !visible(r)) continue;
      const w = records[r + FIELD.w], h = records[r + FIELD.h];
      const t = layers.has('qp')
        ? (records[r + FIELD.qp] - qpScale.low) / (qpScale.high - qpScale.low)
        : (Math.log2(Math.max(records[r + FIELD.bits], 1) / (w * h)) - BITS_PER_PIXEL_SCALE.low)
          / (BITS_PER_PIXEL_SCALE.high - BITS_PER_PIXEL_SCALE.low);
      context.fillStyle = heat(t);
      context.fillRect(records[r + FIELD.x], records[r + FIELD.y], w, h);
    }
  }
  if (layers.has('transforms')) {
    context.setLineDash([2 * px, 2 * px]);
    for (let i = 0; i < n; i++) {
      const r = i * HALFWORDS;
      if (records[r + FIELD.kind] !== KIND.transform || !visible(r)) continue;
      context.strokeStyle = records[r + FIELD.flags] & FLAG.codedY ? '#fff9' : '#fff4';
      context.strokeRect(records[r + FIELD.x] + px / 2, records[r + FIELD.y] + px / 2, records[r + FIELD.w] - px, records[r + FIELD.h] - px);
    }
    context.setLineDash([]);
  }
  if (layers.has('blocks')) {
    for (let i = 0; i < n; i++) {
      const r = i * HALFWORDS;
      if (records[r + FIELD.kind] !== KIND.coding || !visible(r)) continue;
      const colour = PRED_COLOURS[records[r + FIELD.pred]];
      const x = records[r + FIELD.x], y = records[r + FIELD.y], w = records[r + FIELD.w], h = records[r + FIELD.h];
      context.globalAlpha = BLOCK_FILL_ALPHA; context.fillStyle = colour; context.fillRect(x, y, w, h);
      context.globalAlpha = 1; context.strokeStyle = colour; context.strokeRect(x + px / 2, y + px / 2, w - px, h - px);
    }
    // Prediction blocks that split their coding block: thin lines inside it.
    context.strokeStyle = '#fffc';
    for (let i = 0; i < n; i++) {
      const r = i * HALFWORDS;
      if (records[r + FIELD.kind] !== KIND.prediction || records[r + FIELD.pred] === PRED.intra || !visible(r)) continue;
      context.strokeRect(records[r + FIELD.x] + px, records[r + FIELD.y] + px, records[r + FIELD.w] - 2 * px, records[r + FIELD.h] - 2 * px);
    }
  }
  if (layers.has('vectors')) drawVectors(context, records, n, px, visible, mvUnits);
  if (layers.has('intra')) drawIntra(context, records, n, px, codec, scale, visible);
  context.restore();
}

function drawVectors(context, records, n, px, visible, mvUnits) {
  context.strokeStyle = '#ffe14d'; context.fillStyle = '#ffe14d';
  context.beginPath();
  for (let i = 0; i < n; i++) {
    const r = i * HALFWORDS;
    if (records[r + FIELD.kind] !== KIND.prediction || records[r + FIELD.pred] === PRED.intra || !visible(r)) continue;
    const cx = records[r + FIELD.x] + records[r + FIELD.w] / 2, cy = records[r + FIELD.y] + records[r + FIELD.h] / 2;
    // The vector points to where the block was predicted from.
    const dx = records[r + FIELD.mv0x] / mvUnits, dy = records[r + FIELD.mv0y] / mvUnits;
    const length = Math.hypot(dx, dy);
    if (length < MIN_VECTOR_PX * px) { context.rect(cx - px, cy - px, 2 * px, 2 * px); continue; }
    const ex = cx + dx, ey = cy + dy, head = ARROW_PX * px, ux = dx / length, uy = dy / length;
    context.moveTo(cx, cy); context.lineTo(ex, ey);
    context.moveTo(ex - head * (ux - uy * 0.5), ey - head * (uy + ux * 0.5)); context.lineTo(ex, ey);
    context.lineTo(ex - head * (ux + uy * 0.5), ey - head * (uy - ux * 0.5));
  }
  context.stroke();
}

function drawIntra(context, records, n, px, codec, scale, visible) {
  context.strokeStyle = '#ff9f80'; context.fillStyle = '#ff9f80';
  context.font = `${10 * px}px ui-monospace, monospace`; context.textAlign = 'center'; context.textBaseline = 'middle';
  context.beginPath();
  for (let i = 0; i < n; i++) {
    const r = i * HALFWORDS;
    if (records[r + FIELD.kind] !== KIND.prediction || records[r + FIELD.pred] !== PRED.intra || !visible(r)) continue;
    const w = records[r + FIELD.w], h = records[r + FIELD.h];
    if (w * scale < MIN_INTRA_MARK_PX) continue;
    const cx = records[r + FIELD.x] + w / 2, cy = records[r + FIELD.y] + h / 2;
    const direction = intraDirection(codec, records[r + FIELD.mode], w);
    if (!direction) {
      if (w * scale >= 2 * MIN_INTRA_MARK_PX) context.fillText(records[r + FIELD.cls] === INTRA_CLASS.dc ? 'DC' : 'P', cx, cy);
      continue;
    }
    const half = 0.4 * w / Math.hypot(direction[0], direction[1]);
    context.moveTo(cx - direction[0] * half, cy - direction[1] * half);
    context.lineTo(cx + direction[0] * half, cy + direction[1] * half);
  }
  context.stroke();
}

/** The records of the blocks containing picture point (x, y): `{ coding, prediction, transform }` record offsets (or -1). */
export function recordsAt(records, x, y) {
  const found = { coding: -1, prediction: -1, transform: -1 };
  const names = ['coding', 'prediction', 'transform'];
  for (let i = 0, n = recordCount(records); i < n; i++) {
    const r = i * HALFWORDS, kind = records[r + FIELD.kind];
    if (kind > KIND.transform) continue;
    const rx = records[r + FIELD.x], ry = records[r + FIELD.y];
    if (x >= rx && y >= ry && x < rx + records[r + FIELD.w] && y < ry + records[r + FIELD.h]) found[names[kind]] = r;
  }
  return found;
}

/** A block's record as text lines (hover / tap). */
export function describeBlock(records, { coding, prediction, transform }, codec = 'hevc') {
  const mvUnits = units(codec).mv;
  const lines = [];
  const v = (r, field) => records[r + FIELD[field]];
  if (coding >= 0) {
    const pred = PRED_NAMES[v(coding, 'pred')];
    lines.push(`${pred} ${v(coding, 'w')}×${v(coding, 'h')} at (${v(coding, 'x')}, ${v(coding, 'y')}) · depth ${v(coding, 'depth')}`);
    if (pred !== 'skip') lines.push(`partition ${SHAPE_NAMES[v(coding, 'shape')] ?? '?'}`);
    const delta = v(coding, 'flags') & FLAG.qpDeltaCoded ? ` (Δ${v(coding, 'qpDelta')})` : '';
    lines.push(`QP ${v(coding, 'qp')}${delta} · ${v(coding, 'bits')} bits`);
  }
  if (prediction >= 0) {
    if (v(prediction, 'pred') === PRED.intra) {
      lines.push(`intra mode ${v(prediction, 'mode')} (${['planar', 'DC', 'angular', 'other'][v(prediction, 'cls')]}), chroma ${v(prediction, 'mode2')} · PB ${v(prediction, 'w')}×${v(prediction, 'h')}`);
    } else {
      const mv = [v(prediction, 'mv0x') / mvUnits, v(prediction, 'mv0y') / mvUnits];
      const how = v(prediction, 'pred') === PRED.inter ? `MVD (${v(prediction, 'mvd0x') / mvUnits}, ${v(prediction, 'mvd0y') / mvUnits})` : `mode ${v(prediction, 'mode')}`;
      lines.push(`PB ${v(prediction, 'w')}×${v(prediction, 'h')} · MV (${mv[0]}, ${mv[1]}) px · ref ${v(prediction, 'ref0')} · ${how}`);
    }
  }
  if (transform >= 0) {
    const flags = v(transform, 'flags');
    const planes = ['Y', 'Cb', 'Cr'].filter((_, i) => flags & (1 << i)).join('+') || 'none';
    lines.push(`TB ${v(transform, 'w')}×${v(transform, 'h')} · depth ${v(transform, 'depth')} · coded ${planes}${flags & FLAG.transformSkip ? ' · transform skip' : ''}`);
  }
  return lines;
}
