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
   * Convert the latest frame to RGBA (`width * height * 4` bytes) with
   * `colour` (the matrix and range of the samples, `SourceFrame` names)
   * in a buffer this decoder reuses, and return its byte offset in the
   * module's WASM memory. JS reads it through a view of `memory.buffer`,
   * which is valid until the next call into the module (memory may grow);
   * copying from it into a reused canvas buffer avoids a per-frame
   * allocation. This is the CPU display path; `i420_frame` feeds a GPU
   * conversion.
   * # Errors
   * Rejects unknown colours, and calls made before a frame has been
   * decoded.
   */
  rgba_frame(colour: string): number;
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
   * Switch coding-tool analysis on (from the next decoded picture) or
   * off; returns whether this codec's decoder supports it (H.264, HEVC,
   * AV1 and VP9 so far). Only in the analysis build of the module; every
   * picture's block records are then kept for `analysis_frame`.
   */
  set_analysis(on: boolean): boolean;
  /**
   * Pack the latest decoded picture's analysis records
   * (`nc_common::stream_stats::pack_records`: `analysis_halfwords` `i16`
   * values per record) into a reused buffer and return its byte offset
   * in WASM memory, read as for `rgba_frame`; `analysis_length` is the
   * number of `i16` values. Without analysis the buffer is empty.
   */
  analysis_frame(): number;
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
   * PSNR-YUV of the latest decoded frame against the source frame it was
   * coded from, packed I420 (`SourceFrame::packed`; 6:1:1 weighting).
   * When the decoded frame is a smaller spatial layer or
   * adaptive-resolution picture of the same stream, it is upscaled to the
   * stream dimensions (`nc_dsp::display`) before scoring.
   * # Errors
   * Rejects calls made before a frame has been decoded, and a source
   * whose byte count matches neither the decoded frame nor the stream.
   */
  psnr(i420: Uint8Array): number;
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
   * The number of `i16` values `analysis_frame` packed.
   */
  readonly analysis_length: number;
  /**
   * `i16` values per packed analysis record.
   */
  readonly analysis_halfwords: number;
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
   * Hand over the second source frame before encoding the first; both
   * are still encoded in order afterwards.
   * # Errors
   * Rejects a frame of another size or colour than the configuration's.
   */
  look_ahead(source: SourceFrame): void;
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
   * output. `colour` (optional, [`SourceFrame`] names, default `bt601`)
   * is the matrix and range of the input, which the stream signals
   * (AV2, whose subset has no colour syntax, signals nothing).
   * # Errors
   * Rejects invalid codecs, presets, overrides, configurations, film
   * grain outside AV1, unknown colours, or dimensions above1080p.
   */
  constructor(codec: string, width: number, height: number, qp: number, fps: number, key_interval: number, preset_name: string, overrides: string | null | undefined, film_grain: boolean, colour?: string | null);
  /**
   * Encode one source frame as an access unit / AV1 temporal unit:
   * empty when the input repeated the picture ([`BrowserEncoder::repeat`]).
   * # Errors
   * Rejects a frame of another size or colour than the configuration's.
   */
  encode(source: SourceFrame): Uint8Array;
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
   * The colour (`SourceFrame` names) the encoder expects and signals.
   */
  readonly colour: string;
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
/**
 * A source frame converted for one view: the encoder's input picture
 * (`BrowserEncoder::encode`) and, packed, the reference the decode worker
 * scores and the page shows as the original.
 */
export class SourceFrame {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Fill the frame from a `width` x `height` YUV picture in `data`, laid
   * out as `WebCodecs` `VideoFrame.copyTo` reports it: `layout` holds each
   * plane's byte offset and stride (`[offset, stride, ...]`); `format` is
   * its `VideoPixelFormat` (`I420`, `I420A` or `NV12`) and `colour` its
   * matrix and range (`bt601`, `bt709`, `bt601-full`, `bt709-full`).
   * # Errors
   * Rejects other formats or colours, and layouts that do not fit `data`.
   */
  from_planes(data: Uint8Array, format: string, colour: string, width: number, height: number, layout: Uint32Array): void;
  /**
   * An empty (black, BT.601) frame of the view size.
   * # Errors
   * Rejects sizes the encoders refuse (`BrowserEncoder::new`).
   */
  constructor(width: number, height: number);
  /**
   * Pack the frame as I420 (Y, then Cb, then Cr, rows without padding) in
   * a buffer this frame reuses and return its byte offset in the module's
   * WASM memory, read as for `BrowserDecoder::rgba_frame`
   * (`packed_length` bytes).
   */
  packed(): number;
  /**
   * Fill the frame from packed RGBA of the view size (already letterboxed
   * by the page's canvas), as limited-range BT.601.
   * # Errors
   * Rejects input whose byte count differs from the view size.
   */
  from_rgba(rgba: Uint8Array): void;
  /**
   * Byte count of `packed`.
   */
  readonly packed_length: number;
  /**
   * The colour (`from_planes` names) of the samples.
   */
  readonly colour: string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_browserdecoder_free: (a: number, b: number) => void;
  readonly __wbg_browserencoder_free: (a: number, b: number) => void;
  readonly __wbg_decodeoutcome_free: (a: number, b: number) => void;
  readonly __wbg_sourceframe_free: (a: number, b: number) => void;
  readonly browserdecoder_analysis_frame: (a: number) => number;
  readonly browserdecoder_analysis_halfwords: (a: number) => number;
  readonly browserdecoder_analysis_length: (a: number) => number;
  readonly browserdecoder_decode: (a: number, b: number, c: number) => [number, number];
  readonly browserdecoder_decode_frame: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
  readonly browserdecoder_decoded_i420: (a: number) => [number, number, number, number];
  readonly browserdecoder_height: (a: number) => number;
  readonly browserdecoder_i420_frame: (a: number) => [number, number, number];
  readonly browserdecoder_new: (a: number, b: number) => [number, number, number];
  readonly browserdecoder_psnr: (a: number, b: number, c: number) => [number, number, number];
  readonly browserdecoder_rgba_frame: (a: number, b: number, c: number) => [number, number, number];
  readonly browserdecoder_set_analysis: (a: number, b: number) => number;
  readonly browserdecoder_set_operating_point: (a: number, b: number) => [number, number];
  readonly browserdecoder_spatial_id: (a: number) => number;
  readonly browserdecoder_stream_height: (a: number) => number;
  readonly browserdecoder_stream_width: (a: number) => number;
  readonly browserdecoder_width: (a: number) => number;
  readonly browserencoder_buffer_fullness: (a: number) => number;
  readonly browserencoder_buffer_overflows: (a: number) => number;
  readonly browserencoder_colour: (a: number) => [number, number];
  readonly browserencoder_disable_rate_control: (a: number) => void;
  readonly browserencoder_encode: (a: number, b: number) => [number, number, number, number];
  readonly browserencoder_key: (a: number) => number;
  readonly browserencoder_look_ahead: (a: number, b: number) => [number, number];
  readonly browserencoder_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number, number];
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
  readonly sourceframe_colour: (a: number) => [number, number];
  readonly sourceframe_from_planes: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
  readonly sourceframe_from_rgba: (a: number, b: number, c: number) => [number, number];
  readonly sourceframe_new: (a: number, b: number) => [number, number, number];
  readonly sourceframe_packed: (a: number) => number;
  readonly sourceframe_packed_length: (a: number) => number;
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
