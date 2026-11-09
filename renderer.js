// Draws decoded frames on a view's canvas. The fast path uploads the decoder's
// packed I420 planes as three one-channel textures and converts them to RGB in a
// fragment shader (WebGL2, else WebGL1), so the worker moves 1.5 bytes per pixel
// and skips its RGBA conversion. Without WebGL the worker converts to RGBA in
// WASM and the frame is copied with putImageData (the CPU path).
//
// The shader repeats nc-dsp's display conversion (`convert` module: limited-range
// BT.601 in 8.8 fixed point, each chroma sample covering a 2x2 luma block, no
// chroma interpolation) in exact integer-valued float arithmetic, so both paths
// show the same pixels. `preserveDrawingBuffer` keeps the last frame readable,
// so the magnifier can copy from the canvas whenever it draws.
//
// Fallbacks: no hardware WebGL2 -> WebGL1 -> CPU; GL set-up failure or a
// context lost for longer than RESTORE_TIMEOUT_MS -> the owner's CPU fallback
// (panel.js).

/** What a renderer asks its worker for: `'i420'` (packed Y, Cb, Cr) or `'rgba'`. */
export const FORMATS = { i420: 'i420', rgba: 'rgba' };

// nc-dsp `convert::bt601`: R = (298 (Y-16) + 409 (Cr-128) + 128) >> 8, and so on.
// Sums reach ~1.4e5, exact only in highp floats: a GPU without highp fragment
// precision fails to compile this, and its view takes the CPU path.
const FRAGMENT_SHADER = `
precision highp float;
uniform sampler2D u_y, u_cb, u_cr, u_rgba;
uniform vec2 u_size, u_chroma_size;
uniform bool u_show_rgba;
float sample8(sampler2D plane, vec2 texel, vec2 size) {
  return floor(texture2D(plane, (texel + 0.5) / size).r * 255.0 + 0.5);
}
float channel(float fixed8) { return clamp(floor(fixed8 / 256.0), 0.0, 255.0) / 255.0; }
void main() {
  // Pixel indices with row 0 at the top, as in the decoded picture.
  vec2 pixel = floor(vec2(gl_FragCoord.x, u_size.y - gl_FragCoord.y));
  if (u_show_rgba) { gl_FragColor = texture2D(u_rgba, (pixel + 0.5) / u_size); return; }
  vec2 chroma = floor(pixel / 2.0);
  float y = 298.0 * (sample8(u_y, pixel, u_size) - 16.0) + 128.0;
  float u = sample8(u_cb, chroma, u_chroma_size) - 128.0;
  float v = sample8(u_cr, chroma, u_chroma_size) - 128.0;
  gl_FragColor = vec4(channel(y + 409.0 * v), channel(y - 100.0 * u - 208.0 * v), channel(y + 516.0 * u), 1.0);
}`;
// One triangle covering the viewport; the fragment shader works in pixels.
const VERTEX_SHADER = `
attribute vec2 a_position;
void main() { gl_Position = vec4(a_position, 0.0, 1.0); }`;
const COVER_TRIANGLE = new Float32Array([-1, -1, 3, -1, -1, 3]);
const CONTEXT_OPTIONS = { alpha: false, antialias: false, depth: false, stencil: false, preserveDrawingBuffer: true };
const PLANE_UNITS = ['u_y', 'u_cb', 'u_cr'], RGBA_UNIT = 3;
// How long a lost context may take to come back before the CPU path takes over.
// Chromium restores about 1 s after a GPU process restart (headless Chromium
// with SwiftShader lost its first contexts once, ~0.1 s after creation).
const RESTORE_TIMEOUT_MS = 3000;

/**
 * A renderer for `canvas`. `preferred`:
 * - 'auto' (or anything not listed): WebGL2, else WebGL1, only on a hardware GPU
 *   (`failIfMajorPerformanceCaveat`: software GL such as SwiftShader or
 *   llvmpipe is slower than the CPU path, ~1.4x per 720p frame), else the CPU path;
 * - 'webgl2' or 'webgl': that GL version or lower, even in software (tests, comparisons);
 * - 'cpu': the CPU path.
 * A canvas keeps the first context type it gives out, so when GL set-up throws
 * or `onLost` reports a context that stayed lost, the caller replaces the
 * canvas and makes a CPU renderer for the new one.
 */
export function createRenderer(canvas, { preferred = 'auto', onLost = () => {} } = {}) {
  const explicit = ['webgl2', 'webgl', 'cpu'].indexOf(preferred);
  for (const kind of ['webgl2', 'webgl'].slice(Math.max(explicit, 0))) {
    const gl = canvas.getContext(kind, { ...CONTEXT_OPTIONS, failIfMajorPerformanceCaveat: explicit < 0 });
    if (gl) return new GlRenderer(canvas, gl, kind, onLost);
  }
  return new CpuRenderer(canvas);
}

/** Packed RGBA from the worker, drawn with putImageData. */
class CpuRenderer {
  constructor(canvas) {
    this.kind = 'cpu'; this.format = FORMATS.rgba;
    this.context = canvas.getContext('2d'); this.latest = null;
  }
  /** Show a decoded frame; returns a pixel buffer the worker may reuse. */
  show({ format, pixels, width, height }) {
    // A frame converted for a lost GL renderer: keep the previous picture.
    if (format !== this.format) return pixels.buffer;
    const replaced = this.latest;
    this.latest = new ImageData(new Uint8ClampedArray(pixels.buffer), width, height);
    this.context.putImageData(this.latest, 0, 0);
    // The latest frame stays referenced for showLatest; the replaced one is free.
    return replaced && replaced.data.buffer;
  }
  showOriginal(image) { this.context.putImageData(image, 0, 0); }
  showLatest() { if (this.latest) this.context.putImageData(this.latest, 0, 0); }
}

/** Packed I420 from the worker, converted by the fragment shader. */
class GlRenderer {
  constructor(canvas, gl, kind, onLost) {
    this.kind = kind; this.format = FORMATS.i420; this.gl = gl; this.lost = false;
    this.webgl2 = kind === 'webgl2';
    this.setUp();
    // A context may be lost and restored (GPU process restart, driver reset):
    // frames are skipped meanwhile, and the CPU path takes over if it stays lost.
    let fallBack = 0;
    canvas.addEventListener('webglcontextlost', event => {
      event.preventDefault(); // allows the restore
      this.lost = true;
      fallBack = setTimeout(onLost, RESTORE_TIMEOUT_MS);
    });
    canvas.addEventListener('webglcontextrestored', () => {
      clearTimeout(fallBack);
      try { this.setUp(); this.lost = false; } catch { onLost(); }
    });
  }
  /** Create the program, the covering triangle and the four textures (also after a restore). */
  setUp() {
    const gl = this.gl;
    const program = linkProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
    gl.useProgram(program);
    this.uniforms = Object.fromEntries(['u_size', 'u_chroma_size', 'u_show_rgba'].map(name => [name, gl.getUniformLocation(program, name)]));
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, COVER_TRIANGLE, gl.STATIC_DRAW);
    const position = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    // Rows are packed without padding, and chroma rows may be odd-sized.
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.textures = [...PLANE_UNITS, 'u_rgba'].map((name, unit) => {
      gl.uniform1i(gl.getUniformLocation(program, name), unit);
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
      for (const parameter of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, parameter, gl.NEAREST);
      for (const parameter of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, parameter, gl.CLAMP_TO_EDGE);
      return { width: 0, height: 0 };
    });
    this.hasFrame = false;
  }
  show({ format, pixels, width, height }) {
    if (format !== this.format || this.lost) return pixels.buffer;
    const gl = this.gl, chromaWidth = Math.ceil(width / 2), chromaHeight = Math.ceil(height / 2);
    const lumaSize = width * height, chromaSize = chromaWidth * chromaHeight;
    this.upload(0, width, height, pixels.subarray(0, lumaSize));
    this.upload(1, chromaWidth, chromaHeight, pixels.subarray(lumaSize, lumaSize + chromaSize));
    this.upload(2, chromaWidth, chromaHeight, pixels.subarray(lumaSize + chromaSize, lumaSize + 2 * chromaSize));
    gl.uniform2f(this.uniforms.u_size, width, height);
    gl.uniform2f(this.uniforms.u_chroma_size, chromaWidth, chromaHeight);
    this.hasFrame = true;
    this.draw(false);
    // The textures hold the frame now, so its buffer goes straight back.
    return pixels.buffer;
  }
  showOriginal(image) {
    if (this.lost) return;
    const gl = this.gl, slot = this.textures[RGBA_UNIT];
    gl.activeTexture(gl.TEXTURE0 + RGBA_UNIT);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    slot.width = image.width; slot.height = image.height;
    gl.uniform2f(this.uniforms.u_size, image.width, image.height);
    this.draw(true);
  }
  showLatest() {
    if (this.lost || !this.hasFrame) return;
    const luma = this.textures[0];
    this.gl.uniform2f(this.uniforms.u_size, luma.width, luma.height);
    this.draw(false);
  }
  /** Replace one plane texture's samples, reallocating only when its size changes. */
  upload(unit, width, height, samples) {
    const gl = this.gl, slot = this.textures[unit];
    const [internal, format] = this.webgl2 ? [gl.R8, gl.RED] : [gl.LUMINANCE, gl.LUMINANCE];
    gl.activeTexture(gl.TEXTURE0 + unit);
    if (slot.width !== width || slot.height !== height) {
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, gl.UNSIGNED_BYTE, samples);
      slot.width = width; slot.height = height;
    } else gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, format, gl.UNSIGNED_BYTE, samples);
  }
  draw(showRgba) {
    const gl = this.gl;
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.uniform1i(this.uniforms.u_show_rgba, showRgba ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

function linkProgram(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}
