/* tslint:disable */
/* eslint-disable */
/**
 * `codec`'s registered parameters, as a JSON array of `{name, default,
 * min, max, doc}` (defaults are the codec's own).
 * # Errors
 * Rejects unknown codec names.
 */
export function parameter_registry(codec: string): string;
/**
 * What the page may offer for `codec`, as JSON `{filmGrain, blockQp,
 * temporalLayers, secondReference}`.
 * # Errors
 * Rejects unknown codec names.
 */
export function codec_options(codec: string): string;
/**
 * The bad lines of a `name=value` override list for `codec`, as a JSON
 * array of `{line, message}` (empty when every line is valid).
 * # Errors
 * Rejects unknown codec names.
 */
export function check_parameters(codec: string, text: string): string;
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
   * Decode coded frame `index` and report its outcome, never throwing
   * for stream problems. `references` is the transport's dependency
   * signal (the frames it predicts from; over RTP, the AV1 dependency
   * descriptor), which RTP receivers should pass; else empty: H.264 and
   * HEVC streams, and AV1 streams with frame ids (`av1.frame_ids`, or a
   * long-term reference), then reveal a missing reference themselves; VP9
   * and AV2 streams cannot (`docs/handoff/loss-resilience.md`).
   * # Errors
   * Rejects indices that are not non-negative integers.
   */
  decode_frame(packet: Uint8Array, index: number, references: Float64Array): DecodeOutcome;
  /**
   * The latest decoded I420 samples, suitable for independent conformance checks.
   * # Errors
   * Rejects calls made before a frame has been decoded.
   */
  decoded_i420(): Uint8Array;
  /**
   * Select the operating point (`0..=31` on AV1, `0..=1` on VP9 where `> 0`
   * selects base-only `S0` decoding; default `0`) before or between packets.
   * # Errors
   * Rejects non-zero operating points on codecs other than AV1 and VP9, and
   * values above 31.
   */
  set_operating_point(op: number): void;
  /**
   * Create a decoder for per-frame Annex-B (H.264, HEVC, VVC), raw OBU
   * (AV1, AV2) or VP9 frame packets.
   * # Errors
   * Rejects codec names other than h264, hevc, av1, vp9, av2 and vvc.
   */
  constructor(codec: string);
  /**
   * PSNR-YUV of the latest decoded frame against the RGBA source frame it
   * was coded from, which is converted here exactly as the encoder
   * converts its input (6:1:1 weighting, `BrowserEncoder::quality`). When
   * the decoded frame is a smaller spatial layer or adaptive-resolution
   * picture of the same stream, it is upscaled to the stream dimensions
   * (`nc_dsp::display`) before scoring.
   * # Errors
   * Rejects calls made before a frame has been decoded, and a source
   * whose byte count matches neither the decoded frame nor the stream.
   */
  psnr(rgba: Uint8Array): number;
  /**
   * Decode one packet; the picture becomes the latest decoded frame.
   * # Errors
   * Every packet that does not decode, with its freeze reason
   * (`missing-reference: ...`, `missing-key: ...`, `corrupt: ...`); a
   * player uses `decode_frame` instead.
   */
  decode(packet: Uint8Array): void;
  /**
   * The `spatial_id` (`0..=3`) of the latest shown AV1 or VP9 frame (`0` on
   * single-layer streams and other codecs).
   */
  readonly spatial_id: number;
  /**
   * Maximum visible width of the stream (`max_frame_width` on AV1, or the
   * current frame width on other codecs).
   */
  readonly stream_width: number;
  /**
   * Maximum visible height of the stream (`max_frame_height` on AV1, or the
   * current frame height on other codecs).
   */
  readonly stream_height: number;
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
   * Smaller spatial/scaled frames of the same stream are upscaled to the
   * source dimensions (`nc_dsp::display`) before scoring.
   * # Errors
   * Requires a decoded frame belonging to a stream with the encoder's
   * visible dimensions.
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
   * `preset_name` is "default", "fast", (H.264, HEVC, AV1 and VP9) "faster" or
   * (H.264 and VP9) "fastest";
   * `overrides` (optional) holds `name=value` lines applied over the
   * preset ([`check_parameters`] reports bad ones); `film_grain` (AV1 only)
   * turns on film grain synthesis, which the decoder applies to its
   * output.
   * # Errors
   * Rejects invalid codecs, presets, overrides, configurations, film
   * grain outside AV1, or dimensions above1080p.
   */
  constructor(codec: string, width: number, height: number, qp: number, fps: number, key_interval: number, preset_name: string, overrides: string | null | undefined, film_grain: boolean);
  /**
   * Convert one RGBA frame and encode an access unit / AV1 temporal unit:
   * empty when the input repeated the picture ([`BrowserEncoder::repeat`]).
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
   * Whether the last input coded nothing (its packet is empty): it
   * repeated the picture (frame hold) or rate control dropped it. Send
   * nothing; the decoder keeps showing its last frame, which is the
   * picture for this input.
   */
  readonly repeat: boolean;
  /**
   * Quantizer reported for the latest frame, in the codec's native scale.
   */
  readonly quantizer: number;
}
/**
 * One packet's outcome (`nc_common::outcome::FrameOutcome`): decoded, or
 * frozen. On a frozen packet the page keeps showing the picture it shows
 * (the decoder may no longer hold it) and sends a loss report naming
 * `report_index`.
 */
export class DecodeOutcome {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * The coded-frame index to report lost (the missing reference when
   * the dependency list named it, else this frame), or -1 when decoded.
   */
  readonly report_index: number;
  /**
   * Why the picture froze: `missing-reference` (an earlier frame was
   * lost or refused), `missing-key` (no key frame yet) or `corrupt`
   * (malformed or truncated packet); empty when decoded.
   */
  readonly reason: string;
  /**
   * Whether a new picture was decoded.
   */
  readonly decoded: boolean;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_browserdecoder_free: (a: number, b: number) => void;
  readonly __wbg_browserencoder_free: (a: number, b: number) => void;
  readonly __wbg_decodeoutcome_free: (a: number, b: number) => void;
  readonly browserdecoder_decode: (a: number, b: number, c: number) => [number, number];
  readonly browserdecoder_decode_frame: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
  readonly browserdecoder_decoded_i420: (a: number) => [number, number, number, number];
  readonly browserdecoder_height: (a: number) => number;
  readonly browserdecoder_i420_frame: (a: number) => [number, number, number];
  readonly browserdecoder_new: (a: number, b: number) => [number, number, number];
  readonly browserdecoder_psnr: (a: number, b: number, c: number) => [number, number, number];
  readonly browserdecoder_rgba_frame: (a: number) => [number, number, number];
  readonly browserdecoder_set_operating_point: (a: number, b: number) => [number, number];
  readonly browserdecoder_spatial_id: (a: number) => number;
  readonly browserdecoder_stream_height: (a: number) => number;
  readonly browserdecoder_stream_width: (a: number) => number;
  readonly browserdecoder_width: (a: number) => number;
  readonly browserencoder_buffer_fullness: (a: number) => number;
  readonly browserencoder_buffer_overflows: (a: number) => number;
  readonly browserencoder_disable_rate_control: (a: number) => void;
  readonly browserencoder_encode: (a: number, b: number, c: number) => [number, number, number, number];
  readonly browserencoder_key: (a: number) => number;
  readonly browserencoder_look_ahead: (a: number, b: number, c: number) => [number, number];
  readonly browserencoder_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => [number, number, number];
  readonly browserencoder_quality: (a: number, b: number) => [number, number, number];
  readonly browserencoder_quantizer: (a: number) => number;
  readonly browserencoder_repeat: (a: number) => number;
  readonly browserencoder_request_keyframe: (a: number) => void;
  readonly browserencoder_set_rate_control: (a: number, b: number, c: number, d: number) => [number, number];
  readonly browserencoder_wants_look_ahead: (a: number) => number;
  readonly check_parameters: (a: number, b: number, c: number, d: number) => [number, number, number, number];
  readonly codec_options: (a: number, b: number) => [number, number, number, number];
  readonly decodeoutcome_decoded: (a: number) => number;
  readonly decodeoutcome_reason: (a: number) => [number, number];
  readonly decodeoutcome_report_index: (a: number) => number;
  readonly parameter_registry: (a: number, b: number) => [number, number, number, number];
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
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
