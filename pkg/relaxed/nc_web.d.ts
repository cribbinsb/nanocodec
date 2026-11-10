/* tslint:disable */
/* eslint-disable */
/**
 * An independent subset decoder; no encoder instance or reconstruction is needed.
 */
export class BrowserDecoder {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Pack the latest frame's visible samples as I420 (Y, then Cb, then Cr,
   * rows without padding: `packed_frame_len` bytes) into the reused display
   * buffer and return its byte offset in WASM memory, read as for
   * `rgba_frame`. A WebGL display uploads the three planes and converts
   * them in a shader, moving 1.5 instead of 4 bytes per pixel.
   * # Errors
   * Rejects calls made before a frame has been decoded.
   */
  i420_frame(): number;
  /**
   * Convert the latest frame to RGBA (`width * height * 4` bytes) in a
   * buffer this decoder reuses, and return its byte offset in the module's
   * WASM memory. JS reads it through a view of `memory.buffer`, which is
   * valid until the next call into the module (memory may grow); copying
   * from it into a reused canvas buffer avoids a per-frame allocation.
   * This is the CPU display path; `i420_frame` feeds a GPU conversion.
   * # Errors
   * Rejects calls made before a frame has been decoded.
   */
  rgba_frame(): number;
  /**
   * The latest decoded I420 samples, suitable for independent conformance checks.
   * # Errors
   * Rejects calls made before a frame has been decoded.
   */
  decoded_i420(): Uint8Array;
  /**
   * Create a decoder for per-frame Annex-B (H.264, HEVC, VVC), raw OBU
   * (AV1, AV2) or VP9 frame packets.
   * # Errors
   * Rejects codec names other than h264, hevc, av1, vp9, av2 and vvc.
   */
  constructor(codec: string);
  /**
   * Decode one packet; the picture becomes the latest decoded frame.
   * # Errors
   * Rejects oversized packets and malformed or unsupported codec syntax.
   */
  decode(packet: Uint8Array): void;
  /**
   * Visible width of the latest decoded frame, or zero before the first frame.
   */
  readonly width: number;
  /**
   * Visible height of the latest decoded frame, or zero before the first frame.
   */
  readonly height: number;
}
/**
 * One browser encoder. A separate decoder instance consumes only its packets.
 */
export class BrowserEncoder {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Per-frame PSNR-YUV against the actual decoder output, using6:1:1 weighting.
   * # Errors
   * Requires a decoded frame with the encoder's visible dimensions.
   */
  quality(decoder: BrowserDecoder): number;
  /**
   * Hand over the second RGBA frame before encoding the first; both are
   * still encoded in order afterwards.
   * # Errors
   * Rejects input whose byte count differs from the configured dimensions.
   */
  look_ahead(rgba: Uint8Array): void;
  /**
   * Request a keyframe on the next input, preserving rate-control history.
   */
  request_keyframe(): void;
  /**
   * Enable or update realtime bitrate control without restarting references.
   * # Errors
   * Rejects invalid targets, peak rates or buffer durations.
   */
  set_rate_control(target_kbps: number, peak_kbps: number, buffer_ms: number): void;
  /**
   * Return to the fixed quantizer supplied at construction.
   */
  disable_rate_control(): void;
  /**
   * Create a fixed-quantizer encoder. Dimensions and buffering are bounded.
   * # Errors
   * Rejects invalid codecs, configurations, or dimensions above1080p.
   */
  constructor(codec: string, width: number, height: number, qp: number, fps: number, key_interval: number, fast: boolean);
  /**
   * Convert one RGBA frame and encode an independent access unit / AV1 temporal unit.
   * # Errors
   * Rejects input whose byte count differs from the configured dimensions.
   */
  encode(rgba: Uint8Array): Uint8Array;
  /**
   * Fraction of the configured bitrate buffer currently occupied.
   */
  readonly buffer_fullness: number;
  /**
   * Number of frames that exceeded the modeled bitrate buffer capacity.
   */
  readonly buffer_overflows: number;
  /**
   * Whether the encoder wants the second frame before the first
   * (`look_ahead`): only before the first frame, under rate control, for
   * codecs whose rate model sizes the first key from start-up camera
   * motion. The caller then holds frame 0 back one frame.
   */
  readonly wants_look_ahead: boolean;
  /**
   * Whether the last encoded packet is a key frame.
   */
  readonly key: boolean;
  /**
   * Quantizer reported for the latest frame, in the codec's native scale.
   */
  readonly quantizer: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_browserdecoder_free: (a: number, b: number) => void;
  readonly __wbg_browserencoder_free: (a: number, b: number) => void;
  readonly browserdecoder_decode: (a: number, b: number, c: number) => [number, number];
  readonly browserdecoder_decoded_i420: (a: number) => [number, number, number, number];
  readonly browserdecoder_height: (a: number) => number;
  readonly browserdecoder_i420_frame: (a: number) => [number, number, number];
  readonly browserdecoder_new: (a: number, b: number) => [number, number, number];
  readonly browserdecoder_rgba_frame: (a: number) => [number, number, number];
  readonly browserdecoder_width: (a: number) => number;
  readonly browserencoder_buffer_fullness: (a: number) => number;
  readonly browserencoder_buffer_overflows: (a: number) => number;
  readonly browserencoder_disable_rate_control: (a: number) => void;
  readonly browserencoder_encode: (a: number, b: number, c: number) => [number, number, number, number];
  readonly browserencoder_key: (a: number) => number;
  readonly browserencoder_look_ahead: (a: number, b: number, c: number) => [number, number];
  readonly browserencoder_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
  readonly browserencoder_quality: (a: number, b: number) => [number, number, number];
  readonly browserencoder_quantizer: (a: number) => number;
  readonly browserencoder_request_keyframe: (a: number) => void;
  readonly browserencoder_set_rate_control: (a: number, b: number, c: number, d: number) => [number, number];
  readonly browserencoder_wants_look_ahead: (a: number) => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
