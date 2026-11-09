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

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
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

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
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
     * Convert the latest frame to RGBA (`width * height * 4` bytes) in a
     * buffer this decoder reuses, and return its byte offset in the module's
     * WASM memory. JS reads it through a view of `memory.buffer`, which is
     * valid until the next call into the module (memory may grow); copying
     * from it into a reused canvas buffer avoids a per-frame allocation.
     * This is the CPU display path; `i420_frame` feeds a GPU conversion.
     * # Errors
     * Rejects calls made before a frame has been decoded.
     * @returns {number}
     */
    rgba_frame() {
        const ret = wasm.browserdecoder_rgba_frame(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
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
     * Create a decoder for per-frame Annex-B or raw AV1 OBU packets.
     * # Errors
     * Rejects codec names other than h264, hevc, av1, vp9 and av2.
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
     * Rejects oversized packets and malformed or unsupported codec syntax.
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
     * Per-frame PSNR-YUV against the actual decoder output, using6:1:1 weighting.
     * # Errors
     * Requires a decoded frame with the encoder's visible dimensions.
     * @param {BrowserDecoder} decoder
     * @returns {number}
     */
    quality(decoder) {
        _assertClass(decoder, BrowserDecoder);
        const ret = wasm.browserencoder_quality(this.__wbg_ptr, decoder.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Hand over the second RGBA frame before encoding the first; both are
     * still encoded in order afterwards.
     * # Errors
     * Rejects input whose byte count differs from the configured dimensions.
     * @param {Uint8Array} rgba
     */
    look_ahead(rgba) {
        const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.browserencoder_look_ahead(this.__wbg_ptr, ptr0, len0);
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
     * # Errors
     * Rejects invalid codecs, configurations, or dimensions above1080p.
     * @param {string} codec
     * @param {number} width
     * @param {number} height
     * @param {number} qp
     * @param {number} fps
     * @param {number} key_interval
     * @param {boolean} fast
     */
    constructor(codec, width, height, qp, fps, key_interval, fast) {
        const ptr0 = passStringToWasm0(codec, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.browserencoder_new(ptr0, len0, width, height, qp, fps, key_interval, fast);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        BrowserEncoderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Convert one RGBA frame and encode an independent access unit / AV1 temporal unit.
     * # Errors
     * Rejects input whose byte count differs from the configured dimensions.
     * @param {Uint8Array} rgba
     * @returns {Uint8Array}
     */
    encode(rgba) {
        const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.browserencoder_encode(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v2;
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
