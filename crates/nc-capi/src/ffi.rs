//! The `extern "C"` functions declared in `include/nanocodec.h`.
//!
//! Each function checks its pointers, turns caller memory into borrowed
//! Rust views and calls [`Session`]; nothing here decides anything. Panics
//! never cross the ABI: they are caught and reported as
//! [`NcStatus::Internal`].

#![allow(
    unsafe_code,
    reason = "the C ABI: raw pointers from the caller (CLAUDE.md allows unsafe in nc-capi)"
)]

use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use nc_common::outcome::{FrameOutcome, FreezeReason};
use nc_common::{NV12_CHROMA_BYTES, PlaneIndex, PlaneView, RawFrame, chroma_len};
use nc_encoder::{Codec, FrameRate, FrameType, ParamOverrides, StreamSignal};

use crate::receiver::Receiver;
use crate::session::{
    Session, SessionConfig, SessionError, colour_info, default_quantizer, timing_signal,
};

/// `nc_status`: the result of every call.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcStatus {
    /// Success.
    Ok = 0,
    /// A null pointer, unknown enum value or invalid number.
    InvalidArgument = -1,
    /// The engine rejected the configuration.
    Config = -2,
    /// The frame does not match the configured size or layout.
    Frame = -3,
    /// Invalid rate-control settings.
    RateControl = -4,
    /// An internal error (a caught panic); the encoder must be destroyed.
    Internal = -5,
}

impl From<SessionError> for NcStatus {
    fn from(e: SessionError) -> Self {
        match e {
            SessionError::Config(_) => Self::Config,
            SessionError::RateControl(_) => Self::RateControl,
            SessionError::Frame => Self::Frame,
        }
    }
}

/// `nc_codec`.
pub const NC_CODEC_H264: u32 = 0;
/// `nc_codec`.
pub const NC_CODEC_HEVC: u32 = 1;
/// `nc_codec`.
pub const NC_CODEC_AV1: u32 = 2;
/// `nc_codec`.
pub const NC_CODEC_VP9: u32 = 3;
/// `nc_pixel_format`: Y, Cb, Cr planes.
pub const NC_FORMAT_I420: u32 = 0;
/// `nc_pixel_format`: Y, Cr, Cb planes.
pub const NC_FORMAT_YV12: u32 = 1;
/// `nc_pixel_format`: Y plane, interleaved Cb/Cr plane.
pub const NC_FORMAT_NV12: u32 = 2;

/// `nc_param`: one heuristic parameter override (`nanocodec params`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NcParam {
    /// NUL-terminated name, e.g. `"scene.detect"`.
    pub name: *const c_char,
    /// Value.
    pub value: f32,
}

/// `nc_encoder_config`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NcEncoderConfig {
    /// `nc_codec`.
    pub codec: u32,
    /// Visible width (even).
    pub width: u32,
    /// Visible height (even).
    pub height: u32,
    /// Nominal frame rate numerator.
    pub fps_num: u32,
    /// Nominal frame rate denominator.
    pub fps_den: u32,
    /// Fixed quantizer, or the rate controller's seed (QP; AV1 qindex).
    pub quantizer: u32,
    /// Frames between key frames.
    pub intra_period: u32,
    /// Temporal layers, 1–3.
    pub temporal_layers: u32,
    /// Frames per intra-refresh sweep; 0 = periodic key frames.
    pub intra_refresh_period: u32,
    /// Frames between long-term reference marks; 0 = none (AV1 so far).
    pub ltr_interval: u32,
    /// Target bitrate in kbps; 0 = fixed quantizer.
    pub target_kbps: u32,
    /// Peak bitrate in kbps; 0 = twice the target.
    pub peak_kbps: u32,
    /// Rate-control buffer in ms; 0 = default (1000).
    pub buffer_ms: u32,
    /// Non-zero: drop frames rather than overflow the buffer.
    pub frame_drop: u8,
    /// Colour primaries, transfer, matrix code points (ISO/IEC 23091-2);
    /// all zero = unsignalled.
    pub colour: [u8; 3],
    /// Non-zero: full-range samples.
    pub full_range: u8,
    /// Header timing: 0 nominal, 1 constant, 2 variable frame rate.
    pub timing: u8,
    /// Non-zero: AV1 film grain synthesis; other codecs reject it
    /// (`NC_ERROR_CONFIG`).
    pub film_grain: u8,
    /// Film grain strength relative to the measured source noise; 0 = the
    /// default.
    pub film_grain_strength: f32,
}

/// `nc_frame`: one borrowed input frame.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NcFrame {
    /// `nc_pixel_format`.
    pub format: u32,
    /// Plane pointers in the format's order (NV12: two).
    pub planes: [*const u8; 3],
    /// Bytes between rows of each plane.
    pub strides: [usize; 3],
    /// Capture time in microseconds, used when `has_timestamp` is non-zero.
    pub timestamp_us: i64,
    /// Non-zero: `timestamp_us` is valid.
    pub has_timestamp: u8,
}

/// `nc_packet`: one coded frame, borrowed from the encoder until its next call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NcPacket {
    /// Bitstream bytes (Annex-B access unit or AV1 temporal unit); null
    /// when `size` is 0.
    pub data: *const u8,
    /// Byte count; 0 for a dropped frame.
    pub size: usize,
    /// Coded-frame index.
    pub index: u64,
    /// Capture time in microseconds.
    pub timestamp_us: i64,
    /// Non-zero for a key (IDR) frame.
    pub keyframe: u8,
    /// Temporal layer (0 = base).
    pub temporal_layer: u8,
    /// Non-zero: no frame to send (rate control dropped it, or it would not
    /// have changed the picture and the decoder keeps showing the last one).
    pub dropped: u8,
    /// Frame quantizer.
    pub quantizer: u32,
}

/// `nc_layer_frame`: one spatial-layer picture entry in the latest packet.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NcLayerFrame {
    /// Spatial layer (`0` = base `S0`, `1` = `S1`, `2` = `S2`).
    pub spatial_id: u8,
    /// Temporal layer (`0..=2`).
    pub temporal_layer: u8,
    /// Non-zero when droppable without breaking its spatial layer's reference chain.
    pub droppable: u8,
    /// Bit `i` is `1` when operating point `i` includes this layer picture.
    pub operating_point_mask: u32,
    /// Visible width of this layer picture.
    pub width: u32,
    /// Visible height of this layer picture.
    pub height: u32,
    /// Start offset in `nc_packet.data`.
    pub byte_offset: usize,
    /// Byte length of this layer frame.
    pub byte_len: usize,
    /// Layer frame quantizer.
    pub quantizer: u32,
}

/// `nc_decode_outcome`: decoded; the picture is new.
pub const NC_DECODED: u32 = 0;
/// `nc_decode_outcome`: frozen, a reference is missing (an earlier frame
/// was lost or refused).
pub const NC_FROZEN_MISSING_REFERENCE: u32 = 1;
/// `nc_decode_outcome`: frozen, no key frame decoded yet.
pub const NC_FROZEN_MISSING_KEY: u32 = 2;
/// `nc_decode_outcome`: frozen, the packet is malformed or truncated.
pub const NC_FROZEN_CORRUPT: u32 = 3;

/// `nc_decode_result`: one packet's outcome.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NcDecodeResult {
    /// `nc_decode_outcome`.
    pub outcome: u32,
    /// When frozen: the coded-frame index to pass to
    /// `nc_encoder_report_loss` (the missing reference if the caller's
    /// dependency list named it, else the refused frame itself).
    pub report_index: u64,
}

impl From<FrameOutcome> for NcDecodeResult {
    fn from(outcome: FrameOutcome) -> Self {
        match outcome {
            FrameOutcome::Decoded => Self::default(),
            FrameOutcome::Frozen { reason, report } => Self {
                outcome: match reason {
                    FreezeReason::MissingReference => NC_FROZEN_MISSING_REFERENCE,
                    FreezeReason::MissingKey => NC_FROZEN_MISSING_KEY,
                    FreezeReason::Corrupt => NC_FROZEN_CORRUPT,
                },
                report_index: report.unwrap_or_default(),
            },
        }
    }
}

/// `nc_picture`: the decoder's picture on screen, borrowed until the next
/// call on the decoder. 8-bit 4:2:0, planes Y, Cb, Cr.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NcPicture {
    /// Plane pointers.
    pub planes: [*const u8; 3],
    /// Bytes between rows of each plane.
    pub strides: [usize; 3],
    /// Luma width.
    pub width: u32,
    /// Luma height.
    pub height: u32,
}

/// `nc_decoder`: opaque to C.
#[derive(Debug)]
pub struct NcDecoder(Receiver);

/// `nc_encoder`: opaque to C.
#[derive(Debug)]
pub struct NcEncoder {
    session: Session,
    layers: Vec<NcLayerFrame>,
}

/// Run `body`, turning a panic into [`NcStatus::Internal`].
fn guarded(body: impl FnOnce() -> NcStatus) -> NcStatus {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(NcStatus::Internal)
}

/// Fill `config` with defaults for `codec`: 30 fps, the CLI's starting
/// quantizer, a key frame every 300 frames, fixed quantizer.
///
/// # Safety
///
/// `config` must be null or point to writable memory for one config.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_config_default(
    config: *mut NcEncoderConfig,
    code: u32,
    width: u32,
    height: u32,
) -> NcStatus {
    let Some(codec) = codec_from(code) else {
        return NcStatus::InvalidArgument;
    };
    // SAFETY: the caller guarantees `config` is null or valid for writes.
    let Some(config) = (unsafe { config.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    let defaults = SessionConfig::new(codec, 0, 0);
    *config = NcEncoderConfig {
        codec: code,
        width,
        height,
        fps_num: defaults.frame_rate.num,
        fps_den: defaults.frame_rate.den,
        quantizer: default_quantizer(codec),
        intra_period: defaults.intra_period,
        temporal_layers: u32::from(defaults.temporal_layers),
        intra_refresh_period: 0,
        ltr_interval: 0,
        target_kbps: 0,
        peak_kbps: 0,
        buffer_ms: 0,
        frame_drop: 0,
        colour: [0; 3],
        full_range: 0,
        timing: 0,
        film_grain: 0,
        film_grain_strength: 0.0,
    };
    NcStatus::Ok
}

/// Create an encoder; `params` (`param_count` entries, may be null when 0)
/// override heuristic parameters. On success `*out` owns the encoder;
/// release it with [`nc_encoder_destroy`].
///
/// # Safety
///
/// `config` and `out` must be valid; `params` must point to `param_count`
/// entries whose names are NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_create(
    config: *const NcEncoderConfig,
    params: *const NcParam,
    param_count: usize,
    out: *mut *mut NcEncoder,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees both are null or valid.
        let (Some(config), Some(out)) = (unsafe { config.as_ref() }, unsafe { out.as_mut() })
        else {
            return NcStatus::InvalidArgument;
        };
        let params = if param_count == 0 {
            &[][..]
        } else if params.is_null() {
            return NcStatus::InvalidArgument;
        } else {
            // SAFETY: non-null, and the caller guarantees `param_count` entries.
            unsafe { std::slice::from_raw_parts(params, param_count) }
        };
        let Some(session_config) = session_config(config, params) else {
            return NcStatus::InvalidArgument;
        };
        match Session::new(&session_config) {
            Ok(session) => {
                *out = Box::into_raw(Box::new(NcEncoder {
                    session,
                    layers: Vec::new(),
                }));
                NcStatus::Ok
            }
            Err(e) => e.into(),
        }
    })
}

/// Create a decoder for `codec` (`nc_codec`); release it with
/// [`nc_decoder_destroy`].
///
/// # Safety
///
/// `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_decoder_create(codec: u32, out: *mut *mut NcDecoder) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees `out` is null or valid.
        let Some(out) = (unsafe { out.as_mut() }) else {
            return NcStatus::InvalidArgument;
        };
        let Some(receiver) = codec_from(codec).and_then(Receiver::new) else {
            return NcStatus::InvalidArgument;
        };
        *out = Box::into_raw(Box::new(NcDecoder(receiver)));
        NcStatus::Ok
    })
}

/// Destroy a decoder (null is ignored).
///
/// # Safety
///
/// `decoder` must come from [`nc_decoder_create`] and not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_decoder_destroy(decoder: *mut NcDecoder) {
    if !decoder.is_null() {
        // SAFETY: created by `Box::into_raw` in `nc_decoder_create`, and the
        // caller hands back ownership.
        drop(unsafe { Box::from_raw(decoder) });
    }
}

/// Select the scalable operating point (`0..=31` on AV1, e.g. `0..=8` in `L3T3`;
/// `0..=1` on VP9).
///
/// # Safety
///
/// `decoder` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_decoder_set_operating_point(
    decoder: *mut NcDecoder,
    operating_point: u8,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees `decoder` is null or valid.
        let Some(decoder) = (unsafe { decoder.as_mut() }) else {
            return NcStatus::InvalidArgument;
        };
        match decoder.0.set_operating_point(operating_point) {
            Ok(()) => NcStatus::Ok,
            Err(_) => NcStatus::InvalidArgument,
        }
    })
}

/// Decode coded frame `index` (`nc_packet.index`) of `size` bytes into
/// `*result`. `references` (`reference_count` entries, may be null when 0)
/// is the transport's dependency signal when the caller has it. A frozen
/// outcome is not an error (`NC_OK`): the picture stays the last good one.
///
/// # Safety
///
/// `decoder` and `result` must be valid; `data` must cover `size` bytes and
/// `references` `reference_count` entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_decoder_decode(
    decoder: *mut NcDecoder,
    data: *const u8,
    size: usize,
    index: u64,
    references: *const u64,
    reference_count: usize,
    result: *mut NcDecodeResult,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees both are null or valid.
        let (Some(decoder), Some(result)) =
            (unsafe { decoder.as_mut() }, unsafe { result.as_mut() })
        else {
            return NcStatus::InvalidArgument;
        };
        // SAFETY: the caller guarantees each array covers its count.
        let (Some(packet), Some(references)) = (unsafe { slice(data, size) }, unsafe {
            slice(references, reference_count)
        }) else {
            return NcStatus::InvalidArgument;
        };
        *result = decoder.0.decode(packet, index, references).into();
        NcStatus::Ok
    })
}

/// The picture on screen (the last decoded one), borrowed until the next
/// call on `decoder`; `NC_ERROR_FRAME` before the first decoded picture.
///
/// # Safety
///
/// `decoder` and `picture` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_decoder_picture(
    decoder: *const NcDecoder,
    picture: *mut NcPicture,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees both are null or valid.
        let (Some(decoder), Some(out)) = (unsafe { decoder.as_ref() }, unsafe { picture.as_mut() })
        else {
            return NcStatus::InvalidArgument;
        };
        let Some(shown) = decoder.0.picture() else {
            return NcStatus::Frame;
        };
        let view = |plane| shown.plane(plane).view();
        let (Ok(width), Ok(height)) = (u32::try_from(shown.width()), u32::try_from(shown.height()))
        else {
            return NcStatus::Internal;
        };
        *out = NcPicture {
            planes: PlaneIndex::ALL.map(|plane| view(plane).data().as_ptr()),
            strides: PlaneIndex::ALL.map(|plane| view(plane).stride()),
            width,
            height,
        };
        NcStatus::Ok
    })
}

/// Spatial layer id (`0..=2`) of the picture currently on screen (`NC_ERROR_FRAME`
/// before the first decoded picture; `0` on single-layer streams).
///
/// # Safety
///
/// `decoder` and `spatial_id` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_decoder_spatial_id(
    decoder: *const NcDecoder,
    spatial_id: *mut u8,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees both pointers are null or valid.
        let (Some(decoder), Some(out)) =
            (unsafe { decoder.as_ref() }, unsafe { spatial_id.as_mut() })
        else {
            return NcStatus::InvalidArgument;
        };
        if decoder.0.picture().is_none() {
            return NcStatus::Frame;
        }
        *out = decoder.0.spatial_id();
        NcStatus::Ok
    })
}

/// `count` elements at `data` (null allowed when `count` is 0).
///
/// # Safety
///
/// A non-null `data` must cover `count` elements.
unsafe fn slice<'a, T>(data: *const T, count: usize) -> Option<&'a [T]> {
    if count == 0 {
        return Some(&[]);
    }
    if data.is_null() {
        return None;
    }
    // SAFETY: non-null and covering `count` elements (caller guarantee).
    Some(unsafe { std::slice::from_raw_parts(data, count) })
}

/// Destroy an encoder (null is ignored).
///
/// # Safety
///
/// `encoder` must come from [`nc_encoder_create`] and not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_destroy(encoder: *mut NcEncoder) {
    if !encoder.is_null() {
        // SAFETY: created by `Box::into_raw` in `nc_encoder_create`, and the
        // caller hands back ownership.
        drop(unsafe { Box::from_raw(encoder) });
    }
}

/// Encode one frame into `*packet` (borrowed until the next call on this
/// encoder). A dropped or held frame gives `size` 0 and `dropped` 1.
///
/// # Safety
///
/// `encoder`, `frame` and `packet` must be valid; each plane pointer must
/// cover its plane's rows at its stride for the configured size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_encode(
    encoder: *mut NcEncoder,
    frame: *const NcFrame,
    packet: *mut NcPacket,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees all three are null or valid.
        let (Some(encoder), Some(frame), Some(packet)) = (
            unsafe { encoder.as_mut() },
            unsafe { frame.as_ref() },
            unsafe { packet.as_mut() },
        ) else {
            return NcStatus::InvalidArgument;
        };
        let (width, height) = encoder.session.size();
        // SAFETY: the caller guarantees the planes cover the frame.
        let Some(raw) = (unsafe { raw_frame(frame, width, height) }) else {
            return NcStatus::Frame;
        };
        let timestamp = (frame.has_timestamp != 0).then_some(frame.timestamp_us);
        match encoder.session.encode(&raw, timestamp) {
            Ok(coded) => {
                let m = coded.metadata;
                *packet = NcPacket {
                    data: if coded.data.is_empty() {
                        std::ptr::null()
                    } else {
                        coded.data.as_ptr()
                    },
                    size: coded.data.len(),
                    index: m.index,
                    timestamp_us: m.timestamp_us,
                    keyframe: u8::from(m.frame_type == FrameType::Key && !m.dropped && !m.held),
                    temporal_layer: m.temporal_layer,
                    dropped: u8::from(m.dropped || m.held),
                    quantizer: m.quantizer,
                };
                encoder.layers = coded
                    .layers
                    .iter()
                    .map(|l| NcLayerFrame {
                        spatial_id: l.spatial_id,
                        temporal_layer: l.temporal_layer,
                        droppable: u8::from(l.droppable),
                        operating_point_mask: l.operating_point_mask,
                        width: l.width as u32,
                        height: l.height as u32,
                        byte_offset: l.byte_offset,
                        byte_len: l.byte_len,
                        quantizer: l.quantizer,
                    })
                    .collect();
                NcStatus::Ok
            }
            Err(e) => e.into(),
        }
    })
}

/// Spatial-layer sidecar entries of the latest packet encoded by
/// [`nc_encoder_encode`], borrowed until the next call on `encoder`.
///
/// # Safety
///
/// `encoder`, `layers` and `count` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_packet_layers(
    encoder: *const NcEncoder,
    layers: *mut *const NcLayerFrame,
    count: *mut usize,
) -> NcStatus {
    // SAFETY: the caller guarantees all three are null or valid.
    let (Some(encoder), Some(layers), Some(count)) = (
        unsafe { encoder.as_ref() },
        unsafe { layers.as_mut() },
        unsafe { count.as_mut() },
    ) else {
        return NcStatus::InvalidArgument;
    };
    *count = encoder.layers.len();
    *layers = if encoder.layers.is_empty() {
        std::ptr::null()
    } else {
        encoder.layers.as_ptr()
    };
    NcStatus::Ok
}

/// Set `*wants` to 1 if the encoder wants the second frame before the
/// first ([`nc_encoder_look_ahead`]), else 0. Only before the first frame,
/// under rate control, for codecs that size the first key from start-up
/// camera motion.
///
/// # Safety
///
/// `encoder` and `wants` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_wants_look_ahead(
    encoder: *const NcEncoder,
    wants: *mut u8,
) -> NcStatus {
    // SAFETY: the caller guarantees both are null or valid.
    let (Some(encoder), Some(wants)) = (unsafe { encoder.as_ref() }, unsafe { wants.as_mut() })
    else {
        return NcStatus::InvalidArgument;
    };
    guarded(|| {
        *wants = u8::from(encoder.session.wants_look_ahead());
        NcStatus::Ok
    })
}

/// Before the first frame: hand over the second frame, which the caller
/// still encodes in turn with [`nc_encoder_encode`]. The encoder copies it.
///
/// # Safety
///
/// `encoder` and `next` must be valid; each plane pointer must cover its
/// plane's rows at its stride for the configured size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_look_ahead(
    encoder: *mut NcEncoder,
    next: *const NcFrame,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees both are null or valid.
        let (Some(encoder), Some(next)) = (unsafe { encoder.as_mut() }, unsafe { next.as_ref() })
        else {
            return NcStatus::InvalidArgument;
        };
        let (width, height) = encoder.session.size();
        // SAFETY: the caller guarantees the planes cover the frame.
        let Some(raw) = (unsafe { raw_frame(next, width, height) }) else {
            return NcStatus::Frame;
        };
        match encoder.session.look_ahead(&raw) {
            Ok(()) => NcStatus::Ok,
            Err(e) => e.into(),
        }
    })
}

/// Flush: the encoder holds no frames (the start-up lookahead only reads
/// one ahead of time), so there is
/// never anything left; `*packet` is set to an empty packet. Provided so
/// callers written for buffering encoders work unchanged.
///
/// # Safety
///
/// `encoder` and `packet` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_flush(
    encoder: *mut NcEncoder,
    packet: *mut NcPacket,
) -> NcStatus {
    // SAFETY: the caller guarantees both are null or valid.
    let (Some(_), Some(packet)) = (unsafe { encoder.as_mut() }, unsafe { packet.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    *packet = NcPacket {
        data: std::ptr::null(),
        size: 0,
        index: 0,
        timestamp_us: 0,
        keyframe: 0,
        temporal_layer: 0,
        dropped: 0,
        quantizer: 0,
    };
    NcStatus::Ok
}

/// Code the next frame as a key frame.
///
/// # Safety
///
/// `encoder` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_request_keyframe(encoder: *mut NcEncoder) -> NcStatus {
    // SAFETY: the caller guarantees `encoder` is null or valid.
    let Some(encoder) = (unsafe { encoder.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    guarded(|| {
        encoder.session.request_keyframe();
        NcStatus::Ok
    })
}

/// Receiver feedback: coded frame `index` (`nc_packet.index`) was decoded.
///
/// # Safety
///
/// `encoder` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_acknowledge(encoder: *mut NcEncoder, index: u64) -> NcStatus {
    // SAFETY: the caller guarantees `encoder` is null or valid.
    let Some(encoder) = (unsafe { encoder.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    guarded(|| {
        encoder.session.acknowledge(index);
        NcStatus::Ok
    })
}

/// Receiver feedback: coded frame `index` was lost. The next frame predicts
/// from the acknowledged long-term reference, or is a key frame.
///
/// # Safety
///
/// `encoder` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_report_loss(encoder: *mut NcEncoder, index: u64) -> NcStatus {
    // SAFETY: the caller guarantees `encoder` is null or valid.
    let Some(encoder) = (unsafe { encoder.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    guarded(|| {
        encoder.session.report_loss(index);
        NcStatus::Ok
    })
}

/// Receiver feedback: spatial layer `spatial_id` (`0..=2`) of temporal unit
/// `index` (`nc_packet.index`) was decoded.
///
/// # Safety
///
/// `encoder` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_acknowledge_layer(
    encoder: *mut NcEncoder,
    index: u64,
    spatial_id: u8,
) -> NcStatus {
    // SAFETY: the caller guarantees `encoder` is null or valid.
    let Some(encoder) = (unsafe { encoder.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    if usize::from(spatial_id) >= nc_encoder::MAX_SPATIAL_LAYERS {
        return NcStatus::InvalidArgument;
    }
    guarded(|| {
        encoder.session.acknowledge_layer(index, spatial_id);
        NcStatus::Ok
    })
}

/// Receiver feedback: spatial layer `spatial_id` (`0..=2`) of temporal unit
/// `index` (`nc_packet.index`) was lost.
///
/// # Safety
///
/// `encoder` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_report_layer_loss(
    encoder: *mut NcEncoder,
    index: u64,
    spatial_id: u8,
) -> NcStatus {
    // SAFETY: the caller guarantees `encoder` is null or valid.
    let Some(encoder) = (unsafe { encoder.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    if usize::from(spatial_id) >= nc_encoder::MAX_SPATIAL_LAYERS {
        return NcStatus::InvalidArgument;
    }
    guarded(|| {
        encoder.session.report_layer_loss(index, spatial_id);
        NcStatus::Ok
    })
}

/// Change the target bitrate (and peak; 0 = twice the target) from the
/// next frame on, without restarting; also switches a fixed-quantizer
/// encoder to rate control.
///
/// # Safety
///
/// `encoder` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_set_bitrate(
    encoder: *mut NcEncoder,
    target_kbps: u32,
    peak_kbps: u32,
) -> NcStatus {
    // SAFETY: the caller guarantees `encoder` is null or valid.
    let Some(encoder) = (unsafe { encoder.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    guarded(
        || match encoder.session.set_bitrate(target_kbps, peak_kbps) {
            Ok(()) => NcStatus::Ok,
            Err(e) => e.into(),
        },
    )
}

/// Change the target bitrate, peak (0 = twice the target) and buffer
/// (`buffer_ms`, 0 = unchanged) from the next frame on, without restarting;
/// also switches a fixed-quantizer encoder to rate control.
///
/// # Safety
///
/// `encoder` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_set_rate(
    encoder: *mut NcEncoder,
    target_kbps: u32,
    peak_kbps: u32,
    buffer_ms: u32,
) -> NcStatus {
    // SAFETY: the caller guarantees `encoder` is null or valid.
    let Some(encoder) = (unsafe { encoder.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    guarded(
        || match encoder.session.set_rate(target_kbps, peak_kbps, buffer_ms) {
            Ok(()) => NcStatus::Ok,
            Err(e) => e.into(),
        },
    )
}

/// Code frames from the next one on at `width` x `height` (visible size):
/// a key frame with new parameter sets or sequence header; frames passed to
/// `nc_encoder_encode` must then have the new size.
///
/// # Safety
///
/// `encoder` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_set_resolution(
    encoder: *mut NcEncoder,
    width: u32,
    height: u32,
) -> NcStatus {
    // SAFETY: the caller guarantees `encoder` is null or valid.
    let Some(encoder) = (unsafe { encoder.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    guarded(|| {
        match encoder
            .session
            .set_resolution(width as usize, height as usize)
        {
            Ok(()) => NcStatus::Ok,
            Err(e) => e.into(),
        }
    })
}

/// Enable (`period` ≥ 2 frames per sweep), change or disable (0) intra
/// refresh from the next frame on.
///
/// # Safety
///
/// `encoder` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_set_intra_refresh(
    encoder: *mut NcEncoder,
    period: u32,
) -> NcStatus {
    // SAFETY: the caller guarantees `encoder` is null or valid.
    let Some(encoder) = (unsafe { encoder.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    guarded(|| match encoder.session.set_intra_refresh(period) {
        Ok(()) => NcStatus::Ok,
        Err(e) => e.into(),
    })
}

/// Untimed frames from the next one on are `den / num` seconds apart (a
/// camera changing its rate without timestamps; timestamped frames follow
/// their own times).
///
/// # Safety
///
/// `encoder` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_encoder_set_frame_rate(
    encoder: *mut NcEncoder,
    num: u32,
    den: u32,
) -> NcStatus {
    // SAFETY: the caller guarantees `encoder` is null or valid.
    let Some(encoder) = (unsafe { encoder.as_mut() }) else {
        return NcStatus::InvalidArgument;
    };
    guarded(|| match encoder.session.set_frame_rate(num, den) {
        Ok(()) => NcStatus::Ok,
        Err(e) => e.into(),
    })
}

fn codec_from(code: u32) -> Option<Codec> {
    match code {
        NC_CODEC_H264 => Some(Codec::H264),
        NC_CODEC_HEVC => Some(Codec::Hevc),
        NC_CODEC_AV1 => Some(Codec::Av1),
        NC_CODEC_VP9 => Some(Codec::Vp9),
        _ => None,
    }
}

/// The session configuration of a C config, or `None` if a field is invalid.
fn session_config(config: &NcEncoderConfig, params: &[NcParam]) -> Option<SessionConfig> {
    let codec = codec_from(config.codec)?;
    let mut overrides = ParamOverrides::default();
    for param in params {
        if param.name.is_null() {
            return None;
        }
        // SAFETY: non-null, and the caller guarantees a NUL-terminated string.
        let name = unsafe { CStr::from_ptr(param.name) }.to_str().ok()?;
        overrides.set(name, param.value);
    }
    Some(SessionConfig {
        codec,
        width: usize::try_from(config.width).ok()?,
        height: usize::try_from(config.height).ok()?,
        frame_rate: FrameRate {
            num: config.fps_num,
            den: config.fps_den,
        },
        quantizer: config.quantizer,
        intra_period: config.intra_period,
        temporal_layers: u8::try_from(config.temporal_layers).ok()?,
        intra_refresh_period: config.intra_refresh_period,
        ltr_interval: config.ltr_interval,
        target_kbps: config.target_kbps,
        peak_kbps: config.peak_kbps,
        buffer_ms: config.buffer_ms,
        frame_drop: config.frame_drop != 0,
        signal: StreamSignal {
            colour: colour_info(config.colour, config.full_range != 0),
            timing: timing_signal(config.timing)?,
        },
        film_grain: config.film_grain != 0,
        film_grain_strength: film_grain_strength(config.film_grain_strength).ok()?,
        params: overrides,
    })
}

/// A negative or non-finite `nc_encoder_config.film_grain_strength`.
struct InvalidStrength;

/// `nc_encoder_config.film_grain_strength`: 0 is the default (`None`), a
/// positive value a strength.
fn film_grain_strength(strength: f32) -> Result<Option<f32>, InvalidStrength> {
    if strength == 0.0 {
        Ok(None)
    } else if strength.is_finite() && strength > 0.0 {
        Ok(Some(strength))
    } else {
        Err(InvalidStrength)
    }
}

/// A plane of `width` x `height` samples at `stride`, borrowed from C.
///
/// # Safety
///
/// `data` must be null or cover `(height - 1) * stride + width` bytes.
unsafe fn plane<'a>(
    data: *const u8,
    stride: usize,
    width: usize,
    height: usize,
) -> Option<PlaneView<'a>> {
    if data.is_null() || stride < width {
        return None;
    }
    let len = stride
        .checked_mul(height.checked_sub(1)?)?
        .checked_add(width)?;
    // SAFETY: non-null and covering `len` bytes (caller guarantee).
    let bytes = unsafe { std::slice::from_raw_parts(data, len) };
    PlaneView::new(bytes, stride, width, height)
}

/// The borrowed frame `frame` describes, for a `width` x `height` encoder.
///
/// # Safety
///
/// The frame's plane pointers must cover their planes (see [`plane`]).
unsafe fn raw_frame<'a>(frame: &NcFrame, width: usize, height: usize) -> Option<RawFrame<'a>> {
    let (cw, ch) = (chroma_len(width), chroma_len(height));
    let [p0, p1, p2] = frame.planes;
    let [s0, s1, s2] = frame.strides;
    let first_chroma_width = match frame.format {
        NC_FORMAT_NV12 => NV12_CHROMA_BYTES * cw,
        NC_FORMAT_I420 | NC_FORMAT_YV12 => cw,
        _ => return None,
    };
    // SAFETY: the caller guarantees the luma plane covers its rows.
    let y = unsafe { plane(p0, s0, width, height) }?;
    // SAFETY: the caller guarantees the first chroma plane covers its rows.
    let first = unsafe { plane(p1, s1, first_chroma_width, ch) }?;
    if frame.format == NC_FORMAT_NV12 {
        return Some(RawFrame::nv12(y, first));
    }
    // SAFETY: the caller guarantees the second chroma plane covers its rows.
    let second = unsafe { plane(p2, s2, cw, ch) }?;
    Some(if frame.format == NC_FORMAT_I420 {
        RawFrame::i420(y, first, second)
    } else {
        RawFrame::yv12(y, first, second)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One packet through `nc_decoder_decode`.
    fn decode(
        decoder: *mut NcDecoder,
        data: &[u8],
        index: u64,
        references: &[u64],
    ) -> NcDecodeResult {
        let mut result = NcDecodeResult::default();
        // SAFETY: valid decoder, slices and result.
        let status = unsafe {
            nc_decoder_decode(
                decoder,
                data.as_ptr(),
                data.len(),
                index,
                references.as_ptr(),
                references.len(),
                &raw mut result,
            )
        };
        assert_eq!(status, NcStatus::Ok);
        result
    }

    /// Through the C ABI, for every codec: a dropped key, a dropped frame
    /// (refused from the stream alone, and named by the dependency list),
    /// a truncated frame and garbage, each with its outcome and the frame to
    /// report; the picture stays the last decoded one while frozen.
    #[test]
    fn decoder_outcomes_cross_the_abi() {
        let (w, h) = (64, 48);
        for code in [NC_CODEC_H264, NC_CODEC_HEVC, NC_CODEC_AV1] {
            let codec = codec_from(code).unwrap();
            let mut config = SessionConfig::new(codec, w, h);
            // A flat chain: frame 3 predicts from frame 2.
            config.params.set("layers.count", 1.0);
            config.params.set("layers.adaptive", 0.0);
            if codec == Codec::Av1 {
                // AV1 detects a loss in band only with frame ids.
                config.params.set(nc_av1::FRAME_IDS_PARAM, 1.0);
            }
            let mut session = Session::new(&config).unwrap();
            let packets: Vec<Vec<u8>> = (0..6u8)
                .map(|n| {
                    // A moving texture: inter frames predict from the last.
                    let frame: Vec<u8> = (0..w * h * 3 / 2)
                        .map(|i| ((i % w + 2 * usize::from(n)) * 7 + (i / w) * 13) as u8)
                        .collect();
                    let raw = RawFrame::packed_i420(&frame, w, h).unwrap();
                    session.encode(&raw, None).unwrap().data.clone()
                })
                .collect();
            let mut decoder = std::ptr::null_mut();
            // SAFETY: valid out pointer.
            let status = unsafe { nc_decoder_create(code, &raw mut decoder) };
            assert_eq!(status, NcStatus::Ok);
            let mut picture = NcPicture {
                planes: [std::ptr::null(); 3],
                strides: [0; 3],
                width: 0,
                height: 0,
            };
            // SAFETY: valid decoder and picture.
            let status = unsafe { nc_decoder_picture(decoder, &raw mut picture) };
            assert_eq!(status, NcStatus::Frame, "{codec:?}: no picture yet");
            let r = decode(decoder, &packets[1], 1, &[]);
            assert_eq!(
                (r.outcome, r.report_index),
                (NC_FROZEN_MISSING_KEY, 1),
                "{codec:?}"
            );
            assert_eq!(decode(decoder, &packets[0], 0, &[]).outcome, NC_DECODED);
            assert_eq!(decode(decoder, &packets[1], 1, &[0]).outcome, NC_DECODED);
            // Frame 2 is lost.
            let r = decode(decoder, &packets[3], 3, &[]);
            assert_eq!(
                (r.outcome, r.report_index),
                (NC_FROZEN_MISSING_REFERENCE, 3),
                "{codec:?}"
            );
            let r = decode(decoder, &packets[3], 3, &[2]);
            assert_eq!(
                (r.outcome, r.report_index),
                (NC_FROZEN_MISSING_REFERENCE, 2)
            );
            let truncated = &packets[4][..packets[4].len() / 2];
            let r = decode(decoder, truncated, 4, &[]);
            assert_ne!(r.outcome, NC_DECODED, "{codec:?} truncated");
            let r = decode(decoder, &[0x5a; 97], 5, &[]);
            assert_ne!(r.outcome, NC_DECODED, "{codec:?} garbage");
            // SAFETY: valid decoder and picture.
            let status = unsafe { nc_decoder_picture(decoder, &raw mut picture) };
            assert_eq!(
                (status, picture.width, picture.height),
                (NcStatus::Ok, 64, 48)
            );
            // SAFETY: created above, not used again.
            unsafe { nc_decoder_destroy(decoder) };
        }
    }

    fn encode_av1_l3t3_key_packet(w: u32, h: u32) -> Vec<u8> {
        let mut cfg = NcEncoderConfig {
            codec: 0,
            width: 0,
            height: 0,
            fps_num: 0,
            fps_den: 0,
            quantizer: 0,
            intra_period: 0,
            temporal_layers: 0,
            intra_refresh_period: 0,
            ltr_interval: 0,
            target_kbps: 0,
            peak_kbps: 0,
            buffer_ms: 0,
            frame_drop: 0,
            colour: [0; 3],
            full_range: 0,
            timing: 0,
            film_grain: 0,
            film_grain_strength: 0.0,
        };
        // SAFETY: valid cfg pointer.
        let status = unsafe { nc_encoder_config_default(&raw mut cfg, NC_CODEC_AV1, w, h) };
        assert_eq!(status, NcStatus::Ok);
        cfg.temporal_layers = 3;
        let params = [
            NcParam {
                name: c"spatial.layers".as_ptr(),
                value: 3.0,
            },
            NcParam {
                name: c"layers.adaptive".as_ptr(),
                value: 0.0,
            },
            NcParam {
                name: c"hold.enable".as_ptr(),
                value: 0.0,
            },
        ];
        let mut enc = std::ptr::null_mut();
        // SAFETY: valid config, params and out pointer.
        let status = unsafe {
            nc_encoder_create(&raw const cfg, params.as_ptr(), params.len(), &raw mut enc)
        };
        assert_eq!(status, NcStatus::Ok);

        let y_plane = vec![120u8; (w * h) as usize];
        let uv_plane = vec![128u8; ((w / 2) * (h / 2)) as usize];
        let frame = NcFrame {
            format: NC_FORMAT_I420,
            planes: [y_plane.as_ptr(), uv_plane.as_ptr(), uv_plane.as_ptr()],
            strides: [w as usize, (w / 2) as usize, (w / 2) as usize],
            has_timestamp: 0,
            timestamp_us: 0,
        };
        let mut pkt = NcPacket {
            data: std::ptr::null(),
            size: 0,
            index: 0,
            timestamp_us: 0,
            keyframe: 0,
            temporal_layer: 0,
            dropped: 0,
            quantizer: 0,
        };
        // SAFETY: valid encoder, frame and packet pointers.
        let status = unsafe { nc_encoder_encode(enc, &raw const frame, &raw mut pkt) };
        assert_eq!(status, NcStatus::Ok);
        let mut layers_ptr: *const NcLayerFrame = std::ptr::null();
        let mut layer_count = 0usize;
        // SAFETY: valid encoder and output pointers.
        let status =
            unsafe { nc_encoder_packet_layers(enc, &raw mut layers_ptr, &raw mut layer_count) };
        assert_eq!(status, NcStatus::Ok);
        assert_eq!(layer_count, 3);
        // SAFETY: non-null slice of `layer_count` items borrowed from `enc`.
        let layers = unsafe { std::slice::from_raw_parts(layers_ptr, layer_count) };
        assert_eq!(layers[0].operating_point_mask, 0x1ff);
        assert_eq!(layers[1].operating_point_mask, 0x03f);
        assert_eq!(layers[2].operating_point_mask, 0x007);
        // SAFETY: slice of `pkt.size` packet bytes borrowed from `enc`.
        let data = unsafe { std::slice::from_raw_parts(pkt.data, pkt.size) }.to_vec();

        // SAFETY: valid encoder pointer.
        let ack_s2 = unsafe { nc_encoder_acknowledge_layer(enc, 0, 2) };
        assert_eq!(ack_s2, NcStatus::Ok);
        // SAFETY: valid encoder pointer.
        let loss_s3 = unsafe { nc_encoder_report_layer_loss(enc, 0, 3) };
        assert_eq!(loss_s3, NcStatus::InvalidArgument);
        // SAFETY: created above, not used again.
        unsafe { nc_encoder_destroy(enc) };
        data
    }

    #[test]
    fn av1_l3t3_operating_points_and_layer_masks_cross_the_abi() {
        let data = encode_av1_l3t3_key_packet(160, 96);
        for op in 0..9u8 {
            let mut dec = std::ptr::null_mut();
            // SAFETY: valid out pointer.
            let status = unsafe { nc_decoder_create(NC_CODEC_AV1, &raw mut dec) };
            assert_eq!(status, NcStatus::Ok);
            // SAFETY: valid decoder pointer.
            let status = unsafe { nc_decoder_set_operating_point(dec, op) };
            assert_eq!(status, NcStatus::Ok);
            assert_eq!(decode(dec, &data, 0, &[]).outcome, NC_DECODED);
            let mut pic = NcPicture {
                planes: [std::ptr::null(); 3],
                strides: [0; 3],
                width: 0,
                height: 0,
            };
            let mut sid = 99u8;
            // SAFETY: valid decoder and out pointers.
            let pic_status = unsafe { nc_decoder_picture(dec, &raw mut pic) };
            assert_eq!(pic_status, NcStatus::Ok);
            // SAFETY: valid decoder and out pointers.
            let sid_status = unsafe { nc_decoder_spatial_id(dec, &raw mut sid) };
            assert_eq!(sid_status, NcStatus::Ok);
            let expected_sid = 2 - op / 3;
            let expected_dim = match expected_sid {
                0 => (40, 24),
                1 => (80, 48),
                _ => (160, 96),
            };
            assert_eq!(sid, expected_sid, "op {op}");
            assert_eq!((pic.width, pic.height), expected_dim, "op {op}");
            // SAFETY: created above, not used again.
            unsafe { nc_decoder_destroy(dec) };
        }
    }
}
