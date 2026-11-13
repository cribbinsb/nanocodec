// The colour of the samples the demo codes and shows: which matrix and range
// a captured frame's YUV planes use (capture-worker.js), the same 8.8
// fixed-point YCbCr -> RGB conversion as nc-dsp's `DisplayMatrix` for the
// WebGL shader (renderer.js) and for the original frame shown on demand
// (`SourcePicture`), so all three display paths give the same pixels.
//
// Names are those of nc-web `SourceFrame`: 'bt601' and 'bt709' (limited
// range), 'bt601-full' and 'bt709-full'. RGBA input (the test scene, frames
// the browser gives only as RGBA) is converted to limited-range BT.601.

/** The colour of RGBA input after nc-dsp's `rgba_to_i420`. */
export const RGBA_COLOUR = 'bt601';

/**
 * 8.8 weights per colour, nc-dsp `DisplayMatrix::weights`:
 * R = (y (Y - offset) + rCr Cr' + 128) >> 8, G = (y (Y - offset) - gCb Cb'
 * - gCr Cr' + 128) >> 8, B = (y (Y - offset) + bCb Cb' + 128) >> 8, clamped.
 */
export const DISPLAY_WEIGHTS = {
  bt601: { y: 298, offset: 16, rCr: 409, gCb: 100, gCr: 208, bCb: 516 },
  bt709: { y: 298, offset: 16, rCr: 459, gCb: 55, gCr: 136, bCb: 541 },
  'bt601-full': { y: 256, offset: 0, rCr: 359, gCb: 88, gCr: 183, bCb: 454 },
  'bt709-full': { y: 256, offset: 0, rCr: 403, gCb: 48, gCr: 120, bCb: 475 },
};

/** WebCodecs `VideoColorSpace.matrix` values of each colour family. */
const MATRICES = { bt709: 'bt709', smpte170m: 'bt601', bt470bg: 'bt601' };
/** Pictures taller than this without a matrix are taken as HD (BT.709), as players do. */
const SD_MAX_HEIGHT = 576;

/**
 * The colour of a VideoFrame's YUV samples, or null when the demo cannot
 * code them as they are (another matrix such as BT.2020, or RGB): such
 * frames take the RGBA path. A frame without a matrix is BT.709 above SD
 * height, else BT.601; without a range, limited.
 */
export function colourOf(frame) {
  const { matrix, fullRange } = frame.colorSpace ?? {};
  const family = matrix ? MATRICES[matrix] : (frame.visibleRect.height > SD_MAX_HEIGHT ? 'bt709' : 'bt601');
  if (!family) return null;
  return fullRange ? `${family}-full` : family;
}

/** Packed I420 (Y, Cb, Cr; rows unpadded) of `width` x `height` as RGBA bytes, exactly as nc-dsp `i420_to_rgba`. */
export function i420ToRgba(i420, width, height, colour) {
  const w = DISPLAY_WEIGHTS[colour];
  const chromaWidth = Math.ceil(width / 2), uBase = width * height, vBase = uBase + chromaWidth * Math.ceil(height / 2);
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const chromaRow = (y >> 1) * chromaWidth;
    for (let x = 0; x < width; x++) {
      const c = chromaRow + (x >> 1);
      const luma = w.y * (i420[y * width + x] - w.offset) + 128, u = i420[uBase + c] - 128, v = i420[vBase + c] - 128;
      const o = 4 * (y * width + x);
      // Uint8ClampedArray clamps; >> 8 is the arithmetic shift of the integer sums.
      rgba[o] = (luma + w.rCr * v) >> 8;
      rgba[o + 1] = (luma - w.gCb * u - w.gCr * v) >> 8;
      rgba[o + 2] = (luma + w.bCb * u) >> 8;
      rgba[o + 3] = 255;
    }
  }
  return rgba;
}

/**
 * A view's source frame (the picture it coded), as the decode worker returns
 * it: packed I420 of the view size and its colour. The RGBA image (for
 * "hold original" and the magnifier) is converted only when first asked for.
 */
export class SourcePicture {
  constructor(i420, width, height, colour) {
    this.i420 = i420; this.width = width; this.height = height; this.colour = colour;
    this.converted = null;
  }
  get image() {
    this.converted ??= new ImageData(i420ToRgba(this.i420, this.width, this.height, this.colour), this.width, this.height);
    return this.converted;
  }
}
