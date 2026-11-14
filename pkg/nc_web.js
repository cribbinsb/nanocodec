let wasm;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}
/**
 * `codec`'s registered parameters, as a JSON array of `{name, default,
 * min, max, doc}` (defaults are the codec's own).
 * # Errors
 * Rejects unknown codec names.
 * @param {string} codec
 * @returns {string}
 */
export function parameter_registry(codec) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(codec, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.parameter_registry(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * What the page may offer for `codec`, as JSON `{filmGrain, blockQp,
 * temporalLayers, secondReference}`.
 * # Errors
 * Rejects unknown codec names.
 * @param {string} codec
 * @returns {string}
 */
export function codec_options(codec) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(codec, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.codec_options(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * The bad lines of a `name=value` override list for `codec`, as a JSON
 * array of `{line, message}` (empty when every line is valid).
 * # Errors
 * Rejects unknown codec names.
 * @param {string} codec
 * @param {string} text
 * @returns {string}
 */
export function check_parameters(codec, text) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(codec, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.check_parameters(ptr0, len0, ptr1, len1);
        var ptr3 = ret[0];
        var len3 = ret[1];
        if (ret[3]) {
            ptr3 = 0; len3 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred4_0 = ptr3;
        deferred4_1 = len3;
        return getStringFromWasm0(ptr3, len3);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let cachedFloat64ArrayMemory0 = null;

function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

let cachedUint32ArrayMemory0 = null;

function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

const BrowserDecoderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_browserdecoder_free(ptr >>> 0, 1));
/**
 * An independent subset decoder; no encoder instance or reconstruction is needed.
 */
export class BrowserDecoder {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BrowserDecoderFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_browserdecoder_free(ptr, 0);
    }
    /**
     * Pack the latest frame's visible samples as I420 (Y, then Cb, then Cr,
     * rows without padding: `packed_frame_len` bytes) into the reused display
     * buffer and return its byte offset in WASM memory, read as for
     * `rgba_frame`. A WebGL display uploads the three planes and converts
     * them in a shader, moving 1.5 instead of 4 bytes per pixel.
     * # Errors
     * Rejects calls made before a frame has been decoded.
     * @returns {number}
     */
    i420_frame() {
        const ret = wasm.browserdecoder_i420_frame(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
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
     * @param {string} colour
     * @returns {number}
     */
    rgba_frame(colour) {
        const ptr0 = passStringToWasm0(colour, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.browserdecoder_rgba_frame(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * The `spatial_id` (`0..=3`) of the latest shown AV1 or VP9 frame (`0` on
     * single-layer streams and other codecs).
     * @returns {number}
     */
    get spatial_id() {
        const ret = wasm.browserdecoder_spatial_id(this.__wbg_ptr);
        return ret;
    }
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
     * @param {Uint8Array} packet
     * @param {number} index
     * @param {Float64Array} references
     * @returns {DecodeOutcome}
     */
    decode_frame(packet, index, references) {
        const ptr0 = passArray8ToWasm0(packet, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(references, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.browserdecoder_decode_frame(this.__wbg_ptr, ptr0, len0, index, ptr1, len1);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return DecodeOutcome.__wrap(ret[0]);
    }
    /**
     * The latest decoded I420 samples, suitable for independent conformance checks.
     * # Errors
     * Rejects calls made before a frame has been decoded.
     * @returns {Uint8Array}
     */
    decoded_i420() {
        const ret = wasm.browserdecoder_decoded_i420(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Maximum visible width of the stream (`max_frame_width` on AV1, or the
     * current frame width on other codecs).
     * @returns {number}
     */
    get stream_width() {
        const ret = wasm.browserdecoder_stream_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Maximum visible height of the stream (`max_frame_height` on AV1, or the
     * current frame height on other codecs).
     * @returns {number}
     */
    get stream_height() {
        const ret = wasm.browserdecoder_stream_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Select the operating point (`0..=31` on AV1, `0..=1` on VP9 where `> 0`
     * selects base-only `S0` decoding; default `0`) before or between packets.
     * # Errors
     * Rejects non-zero operating points on codecs other than AV1 and VP9, and
     * values above 31.
     * @param {number} op
     */
    set_operating_point(op) {
        const ret = wasm.browserdecoder_set_operating_point(this.__wbg_ptr, op);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Create a decoder for per-frame Annex-B (H.264, HEVC, VVC), raw OBU
     * (AV1, AV2) or VP9 frame packets.
     * # Errors
     * Rejects codec names other than h264, hevc, av1, vp9, av2 and vvc.
     * @param {string} codec
     */
    constructor(codec) {
        const ptr0 = passStringToWasm0(codec, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.browserdecoder_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        BrowserDecoderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * PSNR-YUV of the latest decoded frame against the source frame it was
     * coded from, packed I420 (`SourceFrame::packed`; 6:1:1 weighting).
     * When the decoded frame is a smaller spatial layer or
     * adaptive-resolution picture of the same stream, it is upscaled to the
     * stream dimensions (`nc_dsp::display`) before scoring.
     * # Errors
     * Rejects calls made before a frame has been decoded, and a source
     * whose byte count matches neither the decoded frame nor the stream.
     * @param {Uint8Array} i420
     * @returns {number}
     */
    psnr(i420) {
        const ptr0 = passArray8ToWasm0(i420, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.browserdecoder_psnr(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Visible width of the latest decoded frame, or zero before the first frame.
     * @returns {number}
     */
    get width() {
        const ret = wasm.browserdecoder_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Decode one packet; the picture becomes the latest decoded frame.
     * # Errors
     * Every packet that does not decode, with its freeze reason
     * (`missing-reference: ...`, `missing-key: ...`, `corrupt: ...`); a
     * player uses `decode_frame` instead.
     * @param {Uint8Array} packet
     */
    decode(packet) {
        const ptr0 = passArray8ToWasm0(packet, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.browserdecoder_decode(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Visible height of the latest decoded frame, or zero before the first frame.
     * @returns {number}
     */
    get height() {
        const ret = wasm.browserdecoder_height(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) BrowserDecoder.prototype[Symbol.dispose] = BrowserDecoder.prototype.free;

const BrowserEncoderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_browserencoder_free(ptr >>> 0, 1));
/**
 * One browser encoder. A separate decoder instance consumes only its packets.
 */
export class BrowserEncoder {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BrowserEncoderFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_browserencoder_free(ptr, 0);
    }
    /**
     * Hand over the second source frame before encoding the first; both
     * are still encoded in order afterwards.
     * # Errors
     * Rejects a frame of another size or colour than the configuration's.
     * @param {SourceFrame} source
     */
    look_ahead(source) {
        _assertClass(source, SourceFrame);
        const ret = wasm.browserencoder_look_ahead(this.__wbg_ptr, source.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Fraction of the configured bitrate buffer currently occupied.
     * @returns {number}
     */
    get buffer_fullness() {
        const ret = wasm.browserencoder_buffer_fullness(this.__wbg_ptr);
        return ret;
    }
    /**
     * Number of frames that exceeded the modeled bitrate buffer capacity.
     * @returns {number}
     */
    get buffer_overflows() {
        const ret = wasm.browserencoder_buffer_overflows(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Request a keyframe on the next input, preserving rate-control history.
     */
    request_keyframe() {
        wasm.browserencoder_request_keyframe(this.__wbg_ptr);
    }
    /**
     * Enable or update realtime bitrate control without restarting references.
     * # Errors
     * Rejects invalid targets, peak rates or buffer durations.
     * @param {number} target_kbps
     * @param {number} peak_kbps
     * @param {number} buffer_ms
     */
    set_rate_control(target_kbps, peak_kbps, buffer_ms) {
        const ret = wasm.browserencoder_set_rate_control(this.__wbg_ptr, target_kbps, peak_kbps, buffer_ms);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Whether the encoder wants the second frame before the first
     * (`look_ahead`): only before the first frame, under rate control, for
     * codecs whose rate model sizes the first key from start-up camera
     * motion. The caller then holds frame 0 back one frame.
     * @returns {boolean}
     */
    get wants_look_ahead() {
        const ret = wasm.browserencoder_wants_look_ahead(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Return to the fixed quantizer supplied at construction.
     */
    disable_rate_control() {
        wasm.browserencoder_disable_rate_control(this.__wbg_ptr);
    }
    /**
     * Whether the last encoded packet is a key frame.
     * @returns {boolean}
     */
    get key() {
        const ret = wasm.browserencoder_key(this.__wbg_ptr);
        return ret !== 0;
    }
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
     * @param {string} codec
     * @param {number} width
     * @param {number} height
     * @param {number} qp
     * @param {number} fps
     * @param {number} key_interval
     * @param {string} preset_name
     * @param {string | null | undefined} overrides
     * @param {boolean} film_grain
     * @param {string | null} [colour]
     */
    constructor(codec, width, height, qp, fps, key_interval, preset_name, overrides, film_grain, colour) {
        const ptr0 = passStringToWasm0(codec, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(preset_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        var ptr2 = isLikeNone(overrides) ? 0 : passStringToWasm0(overrides, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len2 = WASM_VECTOR_LEN;
        var ptr3 = isLikeNone(colour) ? 0 : passStringToWasm0(colour, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len3 = WASM_VECTOR_LEN;
        const ret = wasm.browserencoder_new(ptr0, len0, width, height, qp, fps, key_interval, ptr1, len1, ptr2, len2, film_grain, ptr3, len3);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        BrowserEncoderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * The colour (`SourceFrame` names) the encoder expects and signals.
     * @returns {string}
     */
    get colour() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.browserencoder_colour(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Encode one source frame as an access unit / AV1 temporal unit:
     * empty when the input repeated the picture ([`BrowserEncoder::repeat`]).
     * # Errors
     * Rejects a frame of another size or colour than the configuration's.
     * @param {SourceFrame} source
     * @returns {Uint8Array}
     */
    encode(source) {
        _assertClass(source, SourceFrame);
        const ret = wasm.browserencoder_encode(this.__wbg_ptr, source.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Whether the last input coded nothing (its packet is empty): it
     * repeated the picture (frame hold) or rate control dropped it. Send
     * nothing; the decoder keeps showing its last frame, which is the
     * picture for this input.
     * @returns {boolean}
     */
    get repeat() {
        const ret = wasm.browserencoder_repeat(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * [`BrowserEncoder::encode`] for a frame captured at `timestamp_us`
     * (microseconds, any origin): a live source whose frames may be skipped.
     * Rate control budgets the frame for the time since the previous one, so
     * the bitrate holds in real time.
     * # Errors
     * As [`BrowserEncoder::encode`].
     * @param {SourceFrame} source
     * @param {number} timestamp_us
     * @returns {Uint8Array}
     */
    encode_at(source, timestamp_us) {
        _assertClass(source, SourceFrame);
        const ret = wasm.browserencoder_encode_at(this.__wbg_ptr, source.__wbg_ptr, timestamp_us);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Quantizer reported for the latest frame, in the codec's native scale.
     * @returns {number}
     */
    get quantizer() {
        const ret = wasm.browserencoder_quantizer(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) BrowserEncoder.prototype[Symbol.dispose] = BrowserEncoder.prototype.free;

const DecodeOutcomeFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decodeoutcome_free(ptr >>> 0, 1));
/**
 * One packet's outcome (`nc_common::outcome::FrameOutcome`): decoded, or
 * frozen. On a frozen packet the page keeps showing the picture it shows
 * (the decoder may no longer hold it) and sends a loss report naming
 * `report_index`.
 */
export class DecodeOutcome {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(DecodeOutcome.prototype);
        obj.__wbg_ptr = ptr;
        DecodeOutcomeFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecodeOutcomeFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decodeoutcome_free(ptr, 0);
    }
    /**
     * The coded-frame index to report lost (the missing reference when
     * the dependency list named it, else this frame), or -1 when decoded.
     * @returns {number}
     */
    get report_index() {
        const ret = wasm.decodeoutcome_report_index(this.__wbg_ptr);
        return ret;
    }
    /**
     * Why the picture froze: `missing-reference` (an earlier frame was
     * lost or refused), `missing-key` (no key frame yet) or `corrupt`
     * (malformed or truncated packet); empty when decoded.
     * @returns {string}
     */
    get reason() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.decodeoutcome_reason(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Whether a new picture was decoded.
     * @returns {boolean}
     */
    get decoded() {
        const ret = wasm.decodeoutcome_decoded(this.__wbg_ptr);
        return ret !== 0;
    }
}
if (Symbol.dispose) DecodeOutcome.prototype[Symbol.dispose] = DecodeOutcome.prototype.free;

const SourceFrameFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_sourceframe_free(ptr >>> 0, 1));
/**
 * A source frame converted for one view: the encoder's input picture
 * (`BrowserEncoder::encode`) and, packed, the reference the decode worker
 * scores and the page shows as the original.
 */
export class SourceFrame {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SourceFrameFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_sourceframe_free(ptr, 0);
    }
    /**
     * Fill the frame from a `width` x `height` YUV picture in `data`, laid
     * out as `WebCodecs` `VideoFrame.copyTo` reports it: `layout` holds each
     * plane's byte offset and stride (`[offset, stride, ...]`); `format` is
     * its `VideoPixelFormat` (`I420`, `I420A` or `NV12`) and `colour` its
     * matrix and range (`bt601`, `bt709`, `bt601-full`, `bt709-full`).
     * # Errors
     * Rejects other formats or colours, and layouts that do not fit `data`.
     * @param {Uint8Array} data
     * @param {string} format
     * @param {string} colour
     * @param {number} width
     * @param {number} height
     * @param {Uint32Array} layout
     */
    from_planes(data, format, colour, width, height, layout) {
        const ptr0 = passArray8ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(format, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(colour, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passArray32ToWasm0(layout, wasm.__wbindgen_malloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.sourceframe_from_planes(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, width, height, ptr3, len3);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Byte count of `packed`.
     * @returns {number}
     */
    get packed_length() {
        const ret = wasm.sourceframe_packed_length(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * An empty (black, BT.601) frame of the view size.
     * # Errors
     * Rejects sizes the encoders refuse (`BrowserEncoder::new`).
     * @param {number} width
     * @param {number} height
     */
    constructor(width, height) {
        const ret = wasm.sourceframe_new(width, height);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        SourceFrameFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * The colour (`from_planes` names) of the samples.
     * @returns {string}
     */
    get colour() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.sourceframe_colour(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Pack the frame as I420 (Y, then Cb, then Cr, rows without padding) in
     * a buffer this frame reuses and return its byte offset in the module's
     * WASM memory, read as for `BrowserDecoder::rgba_frame`
     * (`packed_length` bytes).
     * @returns {number}
     */
    packed() {
        const ret = wasm.sourceframe_packed(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Fill the frame from packed RGBA of the view size (already letterboxed
     * by the page's canvas), as limited-range BT.601.
     * # Errors
     * Rejects input whose byte count differs from the view size.
     * @param {Uint8Array} rgba
     */
    from_rgba(rgba) {
        const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.sourceframe_from_rgba(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) SourceFrame.prototype[Symbol.dispose] = SourceFrame.prototype.free;

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg___wbindgen_throw_b855445ff6a94295 = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_externrefs;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
        ;
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('nc_web_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
