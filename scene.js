// The built-in synthetic test scenes, drawn in the browser: no hosted bytes and no
// licences. Each is a pure function of the picture size and the source time (a
// tick's, timeline.js), so a scene is an exact source (exact-source.js) and two
// views, a step or a restart see the same pixels. Each stresses one scenario
// (`tip`); its constants are its `params`, which the debug report records with its id.
// Noise and placement come from an integer hash of position and frame, not Math.random.

/** The scene a page opens with (and a link without `scene`). */
export const DEFAULT_SCENE = 'study';
// Frames per second of the noise and flash schedules (timeline.js SOURCE_FPS).
const SCENE_FPS = 30;

/** A deterministic value in [0, 1) for integers a, b, c (a 32-bit integer hash). */
function hash(a, b = 0, c = 0) {
  let h = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x632be59b, 0xc2b2ae35) ^ Math.imul(c + 0x27d4eb2f, 0x165667b1);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  return ((h ^ (h >>> 15)) >>> 0) / 4294967296;
}

/** Moving shapes, fine texture and a smooth gradient (the page's first scene). */
function drawStudy(ctx, width, height, seconds) {
  ctx.fillStyle = '#102b2c'; ctx.fillRect(0, 0, width, height);
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, '#183c3c'); gradient.addColorStop(.58, '#6a8975'); gradient.addColorStop(1, '#d5d5b3');
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
  ctx.save(); ctx.translate(width * .57, height * .5);
  ctx.rotate(seconds * .12);
  for (let i = 0; i < 42; i++) {
    ctx.strokeStyle = i % 2 ? '#e3dbb994' : '#183f3866';
    ctx.lineWidth = Math.max(.6, width / 900);
    ctx.strokeRect(-width * .27 + i * 2.5, -height * .3 + i * 1.3, width * .54 - i * 5, height * .6 - i * 2.6);
  }
  ctx.restore();
  const x = width * (.5 + .3 * Math.sin(seconds * .7));
  const y = height * (.48 + .18 * Math.cos(seconds * .9));
  ctx.fillStyle = '#f2b566'; ctx.beginPath(); ctx.arc(x, y, height * .14, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#faf0c8'; ctx.lineWidth = Math.max(1, width / 500);
  for (let r = 1; r < 8; r++) { ctx.beginPath(); ctx.arc(x, y, height * (.14 + r * .009), 0, Math.PI * 2); ctx.stroke(); }
  ctx.fillStyle = '#0e2a2b'; ctx.fillRect(width * .07, height * .16, width * .23, height * .67);
  for (let row = 0; row < 20; row++) for (let col = 0; col < 20; col++) {
    ctx.fillStyle = (col + row) % 2 ? '#6c998a' : '#bdd2b3';
    ctx.fillRect(width * .09 + col * width * .009, height * .24 + row * height * .019, Math.max(1, width * .004), Math.max(1, height * .009));
  }
  ctx.fillStyle = '#e7e9d4'; ctx.font = `${Math.max(10, width / 45)}px monospace`;
  ctx.fillText('NANO / MOTION STUDY', width * .07, height * .12);
  ctx.font = `${Math.max(9, width / 55)}px monospace`;
  ctx.fillText(`${seconds.toFixed(2).padStart(7, '0')} s`, width * .72, height * .9);
}

// Pixels as 32-bit words (canvas RGBA bytes on a little-endian machine, as every
// browser platform is): opaque grey v is OPAQUE | v * GREY.
const OPAQUE = 0xff000000, GREY = 0x010101;

// Version 2 (2026-11-22): drawn from separable row and column tables (3 ms at 720p, was 31).
const ZONE = { version: 2, contrast: 110, driftSeconds: 9, driftShare: .12, phasePerSecond: 2 };
/**
 * A zone plate: rings whose frequency rises to the Nyquist limit half a picture from its
 * drifting centre. cos(a + b) = cos a cos b - sin a sin b with a = pi (x - cx)^2 / S + phase
 * and b = pi (y - cy)^2 / S, so each pixel costs two products from per-column and per-row tables.
 */
function drawZonePlate(ctx, width, height, seconds) {
  const scale = Math.max(width, height), phase = ZONE.phasePerSecond * seconds;
  const cx = width * (.5 + ZONE.driftShare * Math.sin(2 * Math.PI * seconds / ZONE.driftSeconds));
  const cy = height * (.5 + ZONE.driftShare * Math.cos(2 * Math.PI * seconds / ZONE.driftSeconds));
  const colCos = new Float64Array(width), colSin = new Float64Array(width);
  for (let x = 0; x < width; x++) {
    const a = Math.PI * (x - cx) ** 2 / scale + phase;
    colCos[x] = ZONE.contrast * Math.cos(a); colSin[x] = ZONE.contrast * Math.sin(a);
  }
  const image = ctx.createImageData(width, height), pixels = new Uint32Array(image.data.buffer);
  for (let y = 0, at = 0; y < height; y++) {
    const b = Math.PI * (y - cy) ** 2 / scale, rowCos = Math.cos(b), rowSin = Math.sin(b);
    // 128 +- contrast stays inside 0..255; + .5 rounds.
    for (let x = 0; x < width; x++, at++) pixels[at] = OPAQUE | (((128.5 + colCos[x] * rowCos - colSin[x] * rowSin) | 0) * GREY);
  }
  ctx.putImageData(image, 0, 0);
}

const PAN = { pixelsPerSecond: 24, periodSeconds: 20 };
/** A slow pan over wide, low-contrast gradients: a sky, a haze and soft blobs. */
function drawGradientPan(ctx, width, height, seconds) {
  const unit = width / 1280, offset = PAN.pixelsPerSecond * unit * (seconds % PAN.periodSeconds);
  const sky = ctx.createLinearGradient(0, 0, 0, height);
  sky.addColorStop(0, '#20324a'); sky.addColorStop(.55, '#5d7491'); sky.addColorStop(1, '#a9a3a0');
  ctx.fillStyle = sky; ctx.fillRect(0, 0, width, height);
  for (let i = 0; i < 7; i++) {
    const x = ((i * .23 * width - offset) % (width * 1.6) + width * 1.6) % (width * 1.6) - width * .3;
    const y = height * (.25 + .5 * hash(i, 1));
    const r = height * (.25 + .25 * hash(i, 2));
    const blob = ctx.createRadialGradient(x, y, 0, x, y, r);
    blob.addColorStop(0, `rgba(${200 + 40 * hash(i, 3)}, ${170 + 40 * hash(i, 4)}, 150, .35)`);
    blob.addColorStop(1, 'rgba(200, 180, 150, 0)');
    ctx.fillStyle = blob; ctx.fillRect(0, 0, width, height);
  }
  const haze = ctx.createLinearGradient(-offset, 0, width * 2 - offset, 0);
  haze.addColorStop(0, 'rgba(255,255,255,0)'); haze.addColorStop(.5, 'rgba(255,255,255,.12)'); haze.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = haze; ctx.fillRect(0, 0, width, height);
}

// Version 2 (2026-11-22): one 32-bit hash per pixel added to a cached still background
// as whole words (about 3 ms at 720p, was 33). `amplitude` bounds each channel's noise.
const NOISE = { version: 2, amplitude: 14 };
// The noise per channel, 0..NOISE_SPAN before the bias: COMMON_SPAN shared by the channels
// (luma-like) plus OWN_SPAN of each channel's own.
const COMMON_SPAN = 20, OWN_SPAN = 8, NOISE_SPAN = COMMON_SPAN + OWN_SPAN;
// The still background per picture size: the study at 0 s with each channel held to
// amplitude..255-amplitude and lowered by amplitude, so adding noise never carries
// between channels; drawn once (a pure function of the size: caching changes no pixel).
const noiseBackgrounds = new Map();
/** A 32-bit hash of pixel index i in frame n (one multiply-xorshift round per input). */
function pixelHash(i, n) {
  let h = Math.imul(i, 0x9e3779b1) ^ Math.imul(n + 1, 0x85ebca77);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  return h ^ (h >>> 12);
}
/** The biased background words of the study at 0 s for this size (see noiseBackgrounds). */
function noiseBackground(ctx, width, height) {
  const key = `${width}x${height}`;
  if (!noiseBackgrounds.has(key)) {
    drawStudy(ctx, width, height, 0);
    const data = ctx.getImageData(0, 0, width, height).data, low = NOISE.amplitude, high = 255 - NOISE.amplitude;
    for (let at = 0; at < data.length; at += 4) {
      for (let c = 0; c < 3; c++) data[at + c] = Math.min(high, Math.max(low, data[at + c])) - low;
    }
    noiseBackgrounds.set(key, new Uint32Array(data.buffer));
  }
  return noiseBackgrounds.get(key);
}
/**
 * The study scene stopped at 0 s under fresh sensor-like noise every frame: one hash per
 * pixel, its low byte the noise the channels share and three 4-bit fields each channel's
 * own, added to the background word (channels stay in 0..255, NOISE_SPAN <= 2 * amplitude).
 */
function drawSensorNoise(ctx, width, height, seconds) {
  const background = noiseBackground(ctx, width, height), frame = Math.round(seconds * SCENE_FPS);
  const image = ctx.createImageData(width, height), pixels = new Uint32Array(image.data.buffer);
  for (let i = 0; i < pixels.length; i++) {
    const h = pixelHash(i, frame);
    const common = ((h & 255) * (COMMON_SPAN + 1)) >>> 8;
    const red = (((h >>> 8) & 15) * (OWN_SPAN + 1)) >>> 4, green = (((h >>> 12) & 15) * (OWN_SPAN + 1)) >>> 4;
    const blue = (((h >>> 16) & 15) * (OWN_SPAN + 1)) >>> 4;
    pixels[i] = background[i] + common * GREY + red + (green << 8) + (blue << 16);
  }
  ctx.putImageData(image, 0, 0);
}

const TEXT = { linesPerSecond: 1.5, lines: 400 };
const WORDS = ['encoder', 'frame', 'key', 'delta', 'buffer', 'rate', 'motion', 'block', 'residual', 'skip', 'quant', 'loop', 'filter', 'tile'];
/** A text line of the scrolling log: deterministic words and numbers. */
function textLine(n) {
  const words = Array.from({ length: 4 + Math.floor(hash(n, 7) * 6) }, (_, i) => WORDS[Math.floor(hash(n, i) * WORDS.length)]);
  return `${String(n).padStart(4, '0')}  ${words.join(' ')}  ${(hash(n, 9) * 1000).toFixed(1)}`;
}
/** Screen content: a static window frame and menu beside a log scrolling up. */
function drawScrollingText(ctx, width, height, seconds) {
  const unit = height / 720, line = 22 * unit;
  ctx.fillStyle = '#f4f4f0'; ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = '#2b5797'; ctx.fillRect(0, 0, width, 40 * unit);
  ctx.fillStyle = '#ffffff'; ctx.font = `${18 * unit}px sans-serif`; ctx.fillText('nanocodec · session log', 16 * unit, 27 * unit);
  ctx.fillStyle = '#e3e6ea'; ctx.fillRect(0, 40 * unit, 220 * unit, height);
  ctx.fillStyle = '#1d1d1d'; ctx.font = `${16 * unit}px sans-serif`;
  ['Overview', 'Streams', 'Rate control', 'Key frames', 'Analysis', 'Settings'].forEach((item, i) => ctx.fillText(item, 20 * unit, (80 + 34 * i) * unit));
  ctx.save(); ctx.beginPath(); ctx.rect(240 * unit, 56 * unit, width - 256 * unit, height - 72 * unit); ctx.clip();
  const scroll = seconds * TEXT.linesPerSecond * line, first = Math.floor(scroll / line);
  ctx.font = `${15 * unit}px monospace`;
  for (let n = first; n < first + height / line + 1; n++) {
    ctx.fillStyle = n % 5 === 0 ? '#b3261e' : '#202124';
    ctx.fillText(textLine(n % TEXT.lines), 250 * unit, 56 * unit + (n + 1) * line - scroll);
  }
  ctx.restore();
}

const GLOBAL = { panPerSecond: 60, zoomShare: .35, zoomSeconds: 7, rotation: .35, rotationSeconds: 11, tile: 48 };
/** A textured plane under a combined pan, zoom and rotation of the whole picture. */
function drawGlobalMotion(ctx, width, height, seconds) {
  const unit = width / 1280, tile = GLOBAL.tile * unit;
  ctx.fillStyle = '#333'; ctx.fillRect(0, 0, width, height);
  ctx.save();
  ctx.translate(width / 2, height / 2);
  ctx.rotate(GLOBAL.rotation * Math.sin(2 * Math.PI * seconds / GLOBAL.rotationSeconds));
  const zoom = 1 + GLOBAL.zoomShare * Math.sin(2 * Math.PI * seconds / GLOBAL.zoomSeconds);
  ctx.scale(zoom, zoom);
  ctx.translate(-GLOBAL.panPerSecond * unit * seconds, -.4 * GLOBAL.panPerSecond * unit * seconds);
  // The tiles around the view's centre, enough to cover it at the smallest zoom and any rotation.
  const reach = Math.ceil(Math.hypot(width, height) / (tile * (1 - GLOBAL.zoomShare)) / 2) + 1;
  const ci = Math.floor(GLOBAL.panPerSecond * unit * seconds / tile), cj = Math.floor(.4 * GLOBAL.panPerSecond * unit * seconds / tile);
  ctx.font = `${tile * .4}px monospace`;
  for (let j = cj - reach; j <= cj + reach; j++) for (let i = ci - reach; i <= ci + reach; i++) {
    const shade = Math.floor(60 + 160 * hash(i, j));
    ctx.fillStyle = `rgb(${shade}, ${Math.floor(80 + 120 * hash(j, i, 1))}, ${Math.floor(70 + 150 * hash(i, j, 2))})`;
    ctx.fillRect(i * tile, j * tile, tile, tile);
    ctx.fillStyle = (i + j) % 2 ? '#111' : '#eee';
    ctx.beginPath(); ctx.arc((i + .5) * tile, (j + .5) * tile, tile * .18, 0, Math.PI * 2); ctx.fill();
    if ((i - j) % 3 === 0) ctx.fillText(String((i * 7 + j) & 63), i * tile + tile * .1, j * tile + tile * .9);
  }
  ctx.restore();
}

const CUTS = { shotSeconds: 2, flashAt: 4, flashFrames: 3, fadeSeconds: 1 };
/**
 * Shots cut every CUTS.shotSeconds over an 8 s cycle: the study, the moving plane,
 * a white flash, the gradient pan, then a fade to black and back in.
 */
function drawCutsAndFades(ctx, width, height, seconds) {
  const cycle = 4 * CUTS.shotSeconds, t = seconds % cycle, shot = Math.floor(t / CUTS.shotSeconds);
  [drawStudy, drawGlobalMotion, drawGradientPan, drawStudy][shot](ctx, width, height, seconds);
  const frame = Math.round((t - CUTS.flashAt) * SCENE_FPS);
  if (frame >= 0 && frame < CUTS.flashFrames) { ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, width, height); }
  const fade = t - (cycle - 2 * CUTS.fadeSeconds);
  if (fade >= 0) {
    ctx.fillStyle = `rgba(0, 0, 0, ${1 - Math.abs(fade - CUTS.fadeSeconds) / CUTS.fadeSeconds})`;
    ctx.fillRect(0, 0, width, height);
  }
}

const CROWD = { movers: 160, maxSpeed: 70, size: 14 };
/** A static paved ground under many small figures, each on its own straight path, wrapping at the edges. */
function drawCrowd(ctx, width, height, seconds) {
  const unit = width / 1280;
  drawGround(ctx, width, height);
  for (let i = 0; i < CROWD.movers; i++) {
    const speed = CROWD.maxSpeed * unit * (.2 + .8 * hash(i, 1)), angle = 2 * Math.PI * hash(i, 2);
    const x = ((hash(i, 3) * width + Math.cos(angle) * speed * seconds) % width + width) % width;
    const y = ((hash(i, 4) * height + Math.sin(angle) * speed * seconds) % height + height) % height;
    const size = CROWD.size * unit * (.7 + .6 * hash(i, 5));
    ctx.fillStyle = `hsl(${Math.floor(360 * hash(i, 6))}, 55%, ${30 + Math.floor(40 * hash(i, 7))}%)`;
    ctx.beginPath(); ctx.ellipse(x, y, size * .45, size * .8, angle, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#e8c9a8'; ctx.beginPath(); ctx.arc(x + Math.cos(angle) * size * .5, y + Math.sin(angle) * size * .5, size * .3, 0, Math.PI * 2); ctx.fill();
  }
}

/** Grey paving stones with fine joints: detailed and still. */
function drawGround(ctx, width, height) {
  const unit = width / 1280, stone = 40 * unit;
  ctx.fillStyle = '#4a4a46'; ctx.fillRect(0, 0, width, height);
  for (let j = 0; j * stone < height; j++) for (let i = -1; i * stone < width; i++) {
    const shade = Math.floor(110 + 50 * hash(i, j, 9));
    ctx.fillStyle = `rgb(${shade}, ${shade - 4}, ${shade - 10})`;
    ctx.fillRect(i * stone + (j % 2) * stone / 2 + unit, j * stone + unit, stone - 2 * unit, stone - 2 * unit);
  }
}

const MOVER = { radius: 18, xSeconds: 6, ySeconds: 4.4 };
/** A still detailed scene with one small ball on a slow Lissajous path. */
function drawOneMover(ctx, width, height, seconds) {
  drawGround(ctx, width, height);
  const x = width * (.5 + .4 * Math.sin(2 * Math.PI * seconds / MOVER.xSeconds));
  const y = height * (.5 + .35 * Math.sin(2 * Math.PI * seconds / MOVER.ySeconds));
  ctx.fillStyle = '#d23c2a'; ctx.beginPath(); ctx.arc(x, y, MOVER.radius * width / 1280, 0, Math.PI * 2); ctx.fill();
}

// The waving hand's 10 s cycle: wave, hold still, leave the frame, stay out, come back.
const HAND = {
  version: 1, waveSeconds: 5, waves: 4, swingRadians: .5, holdSeconds: 2.5, leaveSeconds: 1, outSeconds: 1, returnSeconds: .5,
  armLength: .5, armWidth: .34, palmRadius: .17, blurSeconds: 1 / 90, blurAlpha: .35,
};
const HAND_CYCLE = HAND.waveSeconds + HAND.holdSeconds + HAND.leaveSeconds + HAND.outSeconds + HAND.returnSeconds;
// The room behind the hand per picture size, drawn once (a pure function of the size:
// caching it changes no pixel).
const roomBackgrounds = new Map();

/** A still room: a textured wall, a window, a bookshelf, a poster with fine print and a desk edge. */
function drawRoom(ctx, width, height) {
  const u = height / 720;
  const wall = ctx.createLinearGradient(0, 0, width, height);
  wall.addColorStop(0, '#c9bfa8'); wall.addColorStop(1, '#a89c86');
  ctx.fillStyle = wall; ctx.fillRect(0, 0, width, height);
  for (let i = 0; i < 2400; i++) {  // plaster speckle
    ctx.fillStyle = hash(i, 11) < .5 ? 'rgba(80, 70, 55, .10)' : 'rgba(255, 250, 235, .12)';
    ctx.fillRect(hash(i, 12) * width, hash(i, 13) * height, 2 * u, 2 * u);
  }
  // A window with frame bars and a bright, slightly blue sky.
  ctx.fillStyle = '#5b4a36'; ctx.fillRect(width * .06, height * .08, width * .26, height * .42);
  const sky = ctx.createLinearGradient(0, height * .1, 0, height * .48);
  sky.addColorStop(0, '#dfeaf2'); sky.addColorStop(1, '#a9c3d6');
  ctx.fillStyle = sky; ctx.fillRect(width * .07, height * .1, width * .24, height * .38);
  ctx.fillStyle = '#5b4a36'; ctx.fillRect(width * .185, height * .1, 6 * u, height * .38); ctx.fillRect(width * .07, height * .285, width * .24, 6 * u);
  // A bookshelf with many book spines: hard vertical edges in many colours.
  const shelfX = width * .62, shelfW = width * .34;
  ctx.fillStyle = '#4a3423'; ctx.fillRect(shelfX, height * .06, shelfW, height * .62);
  for (let row = 0; row < 4; row++) {
    const top = height * (.08 + row * .15), shelfH = height * .13;
    for (let x = shelfX + 8 * u, b = 0; x < shelfX + shelfW - 14 * u; b++) {
      const w = (8 + 14 * hash(row, b, 1)) * u, h = shelfH * (.7 + .3 * hash(row, b, 2));
      ctx.fillStyle = `hsl(${Math.floor(360 * hash(row, b, 3))}, ${30 + Math.floor(40 * hash(row, b, 4))}%, ${25 + Math.floor(40 * hash(row, b, 5))}%)`;
      ctx.fillRect(x, top + shelfH - h, w - u, h);
      x += w;
    }
    ctx.fillStyle = '#2e2016'; ctx.fillRect(shelfX, top + shelfH, shelfW, 5 * u);
  }
  // A poster with fine print.
  ctx.fillStyle = '#f1ede2'; ctx.fillRect(width * .38, height * .12, width * .18, height * .3);
  ctx.fillStyle = '#7a2e2a'; ctx.fillRect(width * .395, height * .14, width * .15, height * .09);
  ctx.fillStyle = '#333'; ctx.font = `${9 * u}px sans-serif`;
  for (let line = 0; line < 9; line++) ctx.fillText(textLine(line + 40).slice(6, 34), width * .395, height * (.27 + line * .016));
  // A desk edge along the bottom.
  ctx.fillStyle = '#6b4d33'; ctx.fillRect(0, height * .86, width, height * .14);
  ctx.fillStyle = '#3d2b1c'; ctx.fillRect(0, height * .86, width, 4 * u);
}

/** Where the hand is at `t` s into its cycle: { shown, x, y, angle, spread }, x and y the elbow (a share of width / height). */
function handPose(t) {
  const ease = v => v * v * (3 - 2 * v);  // smoothstep: starts and stops with zero speed
  let at = t;
  if (at < HAND.waveSeconds) {
    // The swing's own sine accelerates and decelerates; the envelope eases it in and out.
    const envelope = Math.sin(Math.PI * at / HAND.waveSeconds);
    const phase = 2 * Math.PI * HAND.waves * at / HAND.waveSeconds;
    return { shown: true, x: .5 + .04 * Math.sin(phase), y: 1, angle: HAND.swingRadians * envelope * Math.sin(phase), spread: .5 + .5 * Math.abs(Math.sin(phase)) };
  }
  at -= HAND.waveSeconds;
  if (at < HAND.holdSeconds) return { shown: true, x: .5, y: 1, angle: 0, spread: .5 };
  at -= HAND.holdSeconds;
  if (at < HAND.leaveSeconds) { const v = ease(at / HAND.leaveSeconds); return { shown: true, x: .5 + .75 * v, y: 1 + .3 * v, angle: -.4 * v, spread: .5 }; }
  at -= HAND.leaveSeconds;
  if (at < HAND.outSeconds) return { shown: false };
  at -= HAND.outSeconds;
  const v = 1 - ease(at / HAND.returnSeconds);
  return { shown: true, x: .5 - .75 * v, y: 1 + .3 * v, angle: .4 * v, spread: .5 };
}

/** A forearm from the elbow at the bottom edge up to an open palm with five fingers. */
function drawHand(ctx, width, height, pose) {
  const h = height;
  ctx.save();
  ctx.translate(pose.x * width, pose.y * height);
  ctx.rotate(pose.angle);
  const skin = ctx.createLinearGradient(-HAND.armWidth * h / 2, 0, HAND.armWidth * h / 2, 0);
  skin.addColorStop(0, '#9c6b4e'); skin.addColorStop(.5, '#d9a27f'); skin.addColorStop(1, '#a8755a');
  ctx.fillStyle = skin;
  const length = HAND.armLength * h, half = HAND.armWidth * h / 2;
  ctx.beginPath(); ctx.moveTo(-half, 0); ctx.lineTo(-half * .8, -length); ctx.lineTo(half * .8, -length); ctx.lineTo(half, 0); ctx.closePath(); ctx.fill();
  const palm = HAND.palmRadius * h;
  ctx.beginPath(); ctx.ellipse(0, -length - palm * .4, palm * .85, palm, 0, 0, Math.PI * 2); ctx.fill();
  for (let f = 0; f < 5; f++) {
    const spread = (f - 2) * (.25 + .15 * pose.spread), finger = palm * (f === 0 || f === 4 ? .7 : .95);
    ctx.save(); ctx.translate(0, -length - palm * .8); ctx.rotate(spread);
    ctx.beginPath(); ctx.ellipse(0, -finger, palm * .22, finger, 0, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }
  ctx.fillStyle = 'rgba(60, 35, 25, .25)';  // a sleeve cuff near the elbow
  ctx.fillRect(-half, -length * .25, 2 * half, length * .08);
  ctx.restore();
}

/**
 * A hand waving in front of a still room (the classic webcam case): it sweeps back and
 * forth with eased acceleration and a little motion blur, holds still, leaves the frame,
 * stays out (the uncovered room must code cheaply again) and comes back; a 10 s cycle.
 */
function drawWavingHand(ctx, width, height, seconds) {
  const key = `${width}x${height}`;
  if (!roomBackgrounds.has(key)) {
    const room = document.createElement('canvas'); room.width = width; room.height = height;
    drawRoom(room.getContext('2d'), width, height);
    roomBackgrounds.set(key, room);
  }
  ctx.drawImage(roomBackgrounds.get(key), 0, 0);
  const t = seconds % HAND_CYCLE, pose = handPose(t);
  if (!pose.shown) return;
  // Motion blur: the poses a third of a frame before and after, faint, under the sharp one;
  // when the hand is still they coincide with it.
  ctx.save(); ctx.globalAlpha = HAND.blurAlpha;
  for (const dt of [-HAND.blurSeconds, HAND.blurSeconds]) {
    const ghost = handPose(Math.min(HAND_CYCLE - 1e-6, Math.max(0, t + dt)));
    if (ghost.shown) drawHand(ctx, width, height, ghost);
  }
  ctx.restore();
  drawHand(ctx, width, height, pose);
}

/** The scenes in picker order: id, name, what it stresses, its constants and its drawing. */
export const SCENES = [
  { id: DEFAULT_SCENE, name: 'Motion study', tip: 'Moving shapes, fine texture and a smooth gradient: a general mix.', params: {}, draw: drawStudy },
  { id: 'zone-plate', name: 'Moving zone plate', tip: 'Rings up to the finest detail a picture holds: aliasing, transforms and ringing.', params: ZONE, draw: drawZonePlate },
  { id: 'gradient-pan', name: 'Slow pan over gradients', tip: 'Wide, faint gradients panning slowly: banding and blocking at low rates.', params: PAN, draw: drawGradientPan },
  { id: 'sensor-noise', name: 'Still scene, sensor noise', tip: 'A still picture under fresh noise each frame: the still-camera noise handling and skip decisions.', params: NOISE, draw: drawSensorNoise },
  { id: 'scrolling-text', name: 'Scrolling text', tip: 'A window with a scrolling log: screen content and sharp edges.', params: TEXT, draw: drawScrollingText },
  { id: 'global-motion', name: 'Pan, zoom and rotation', tip: 'The whole picture moving at once: motion search and non-translational motion.', params: GLOBAL, draw: drawGlobalMotion },
  { id: 'cuts-fades', name: 'Cuts, flashes and fades', tip: 'Shot changes, a white flash and a fade: key frame decisions and rate control.', params: CUTS, draw: drawCutsAndFades },
  { id: 'crowd', name: 'Many small movers', tip: 'A synthetic crowd of independent movers: block partitioning and the motion field.', params: CROWD, draw: drawCrowd },
  { id: 'waving-hand', name: 'Hand waving at a webcam', tip: 'A large hand waving before a still room, then holding still, leaving and returning: occlusion, uncovered background, large motion and the return to still.', params: HAND, draw: drawWavingHand },
  { id: 'one-mover', name: 'Still scene, one mover', tip: 'One small ball on a still background: skip efficiency, the bits spent where nothing changes.', params: MOVER, draw: drawOneMover },
];

/** The scene with this id, or the default for an unknown one. */
export function sceneById(id) { return SCENES.find(scene => scene.id === id) ?? SCENES[0]; }

/** Draw scene `id` (default: DEFAULT_SCENE) at `seconds` of source time. */
export function drawScene(ctx, width, height, seconds, id = DEFAULT_SCENE) {
  sceneById(id).draw(ctx, width, height, seconds);
}
