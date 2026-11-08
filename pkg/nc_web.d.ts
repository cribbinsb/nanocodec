/* tslint:disable */
/* eslint-disable */
/**
 * An independent subset decoder; no encoder instance or reconstruction is needed.
 */
export class BrowserDecoder {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * The latest decoded I420 samples, suitable for independent conformance checks.
   * # Errors
   * Rejects calls made before a frame has been decoded.
   */
  decoded_i420(): Uint8Array;
  /**
   * Create a decoder for per-frame Annex-B or raw AV1 OBU packets.
   * # Errors
   * Rejects codec names other than h264, hevc, av1, vp9 and av2.
   */
  constructor(codec: string);
  /**
   * Decode compressed bytes and return RGBA pixels for a browser canvas.
   * # Errors
   * Rejects oversized packets and malformed or unsupported codec syntax.
   */
  decode(packet: Uint8Array): Uint8Array;
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
  readonly browserdecoder_decode: (a: number, b: number, c: number) => [number, number, number, number];
  readonly browserdecoder_decoded_i420: (a: number) => [number, number, number, number];
  readonly browserdecoder_height: (a: number) => number;
  readonly browserdecoder_new: (a: number, b: number) => [number, number, number];
  readonly browserdecoder_width: (a: number) => number;
  readonly browserencoder_buffer_fullness: (a: number) => number;
  readonly browserencoder_buffer_overflows: (a: number) => number;
  readonly browserencoder_disable_rate_control: (a: number) => void;
  readonly browserencoder_encode: (a: number, b: number, c: number) => [number, number, number, number];
  readonly browserencoder_key: (a: number) => number;
  readonly browserencoder_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
  readonly browserencoder_quality: (a: number, b: number) => [number, number, number];
  readonly browserencoder_quantizer: (a: number) => number;
  readonly browserencoder_request_keyframe: (a: number) => void;
  readonly browserencoder_set_rate_control: (a: number, b: number, c: number, d: number) => [number, number];
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
