//! The `extern "C"` RTP functions declared in `include/nanocodec.h`: a
//! sender that packetizes an encoder's latest packet and a receiver that
//! depacketizes for a decoder. Each checks its pointers and calls
//! [`crate::rtp`]; nothing here decides anything. Panics are caught
//! ([`guarded`]).

#![allow(
    unsafe_code,
    reason = "the C ABI: raw pointers from the caller (CLAUDE.md allows unsafe in nc-capi)"
)]

use crate::ffi::{NcEncoder, NcStatus, codec_from, guarded};
use crate::rtp::{Receiver, Sender, SenderConfig};

/// RFC 8285 one-byte extension ids are 1-14; 0 means "not negotiated".
const NO_EXTENSION: u8 = 0;

/// `nc_rtp_sender_config`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NcRtpSenderConfig {
    /// `nc_codec` of the encoder whose packets are sent.
    pub codec: u32,
    /// RTP payload type (96-127 by convention).
    pub payload_type: u8,
    /// Dependency Descriptor extension id (1-14); 0 = not sent.
    pub dependency_descriptor_id: u8,
    /// RTP SSRC.
    pub ssrc: u32,
    /// Largest RTP packet in bytes; 0 = 1200.
    pub mtu: u32,
}

/// `nc_rtp_sender`: opaque.
#[derive(Debug)]
pub struct NcRtpSender {
    sender: Sender,
}

/// `nc_rtp_receiver`: opaque.
#[derive(Debug)]
pub struct NcRtpReceiver {
    receiver: Receiver,
}

/// The extension id of a C field (0: none).
fn extension_id(id: u8) -> Option<u8> {
    (id != NO_EXTENSION).then_some(id)
}

/// Create a sender in `*out`.
///
/// # Safety
///
/// `config` and `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_rtp_sender_create(
    config: *const NcRtpSenderConfig,
    out: *mut *mut NcRtpSender,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees both are null or valid.
        let (Some(config), Some(out)) = (unsafe { config.as_ref() }, unsafe { out.as_mut() })
        else {
            return NcStatus::InvalidArgument;
        };
        let Some(codec) = codec_from(config.codec) else {
            return NcStatus::InvalidArgument;
        };
        match Sender::new(SenderConfig {
            codec,
            payload_type: config.payload_type,
            ssrc: config.ssrc,
            mtu: config.mtu as usize,
            dependency_descriptor_id: extension_id(config.dependency_descriptor_id),
        }) {
            Ok(sender) => {
                *out = Box::into_raw(Box::new(NcRtpSender { sender }));
                NcStatus::Ok
            }
            Err(_) => NcStatus::Config,
        }
    })
}

/// Destroy a sender (null is ignored).
///
/// # Safety
///
/// `sender` must be null or from [`nc_rtp_sender_create`], not yet destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_rtp_sender_destroy(sender: *mut NcRtpSender) {
    if !sender.is_null() {
        // SAFETY: created by `Box::into_raw` in `nc_rtp_sender_create`.
        drop(unsafe { Box::from_raw(sender) });
    }
}

/// Packetize `encoder`'s latest packet; `*count` packets are then
/// available from [`nc_rtp_sender_packet`] (0 for a dropped or held frame).
///
/// # Safety
///
/// `sender`, `encoder` and `count` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_rtp_sender_packetize(
    sender: *mut NcRtpSender,
    encoder: *const NcEncoder,
    count: *mut usize,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees all three are null or valid.
        let (Some(sender), Some(encoder), Some(count)) = (
            unsafe { sender.as_mut() },
            unsafe { encoder.as_ref() },
            unsafe { count.as_mut() },
        ) else {
            return NcStatus::InvalidArgument;
        };
        match sender.sender.packetize(&encoder.session) {
            Ok(n) => {
                *count = n;
                NcStatus::Ok
            }
            Err(_) => NcStatus::Rtp,
        }
    })
}

/// Packet `index` of the latest [`nc_rtp_sender_packetize`], borrowed until
/// the next call on `sender`.
///
/// # Safety
///
/// `sender`, `data` and `size` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_rtp_sender_packet(
    sender: *const NcRtpSender,
    index: usize,
    data: *mut *const u8,
    size: *mut usize,
) -> NcStatus {
    // SAFETY: the caller guarantees all three are null or valid.
    let (Some(sender), Some(data), Some(size)) = (
        unsafe { sender.as_ref() },
        unsafe { data.as_mut() },
        unsafe { size.as_mut() },
    ) else {
        return NcStatus::InvalidArgument;
    };
    let Some(packet) = sender.sender.packet(index) else {
        return NcStatus::InvalidArgument;
    };
    *data = packet.as_ptr();
    *size = packet.len();
    NcStatus::Ok
}

/// Create a receiver in `*out` for `codec` (`nc_codec`).
/// `dependency_descriptor_id` 0 = not negotiated; `deliver_after_loss`
/// non-zero keeps delivering after a loss (intra refresh, LTR recovery).
///
/// # Safety
///
/// `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_rtp_receiver_create(
    codec: u32,
    dependency_descriptor_id: u8,
    deliver_after_loss: u8,
    out: *mut *mut NcRtpReceiver,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees `out` is null or valid.
        let (Some(codec), Some(out)) = (codec_from(codec), unsafe { out.as_mut() }) else {
            return NcStatus::InvalidArgument;
        };
        match Receiver::new(
            codec,
            extension_id(dependency_descriptor_id),
            deliver_after_loss != 0,
        ) {
            Ok(receiver) => {
                *out = Box::into_raw(Box::new(NcRtpReceiver { receiver }));
                NcStatus::Ok
            }
            Err(_) => NcStatus::Config,
        }
    })
}

/// Destroy a receiver (null is ignored).
///
/// # Safety
///
/// `receiver` must be null or from [`nc_rtp_receiver_create`], not yet
/// destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_rtp_receiver_destroy(receiver: *mut NcRtpReceiver) {
    if !receiver.is_null() {
        // SAFETY: created by `Box::into_raw` in `nc_rtp_receiver_create`.
        drop(unsafe { Box::from_raw(receiver) });
    }
}

/// Hand one received RTP datagram to the receiver. A malformed packet gives
/// [`NcStatus::Rtp`] and is dropped; the stream goes on.
///
/// # Safety
///
/// `receiver` must be valid and `data` must cover `size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_rtp_receiver_push(
    receiver: *mut NcRtpReceiver,
    data: *const u8,
    size: usize,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees `receiver` is null or valid.
        let Some(receiver) = (unsafe { receiver.as_mut() }) else {
            return NcStatus::InvalidArgument;
        };
        if data.is_null() {
            return NcStatus::InvalidArgument;
        }
        // SAFETY: the caller guarantees `data` covers `size` bytes.
        let datagram = unsafe { std::slice::from_raw_parts(data, size) };
        match receiver.receiver.push(datagram) {
            Ok(()) => NcStatus::Ok,
            Err(_) => NcStatus::Rtp,
        }
    })
}

/// End of stream: release the units still held for reordering.
///
/// # Safety
///
/// `receiver` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_rtp_receiver_flush(receiver: *mut NcRtpReceiver) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees `receiver` is null or valid.
        let Some(receiver) = (unsafe { receiver.as_mut() }) else {
            return NcStatus::InvalidArgument;
        };
        receiver.receiver.flush();
        NcStatus::Ok
    })
}

/// The next unit for the decoder, borrowed until the next call on
/// `receiver`: `*size` 0 when none is ready. `timestamp` and `keyframe`
/// may be null.
///
/// # Safety
///
/// `receiver`, `data` and `size` must be valid; `timestamp` and `keyframe`
/// null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_rtp_receiver_pop(
    receiver: *mut NcRtpReceiver,
    data: *mut *const u8,
    size: *mut usize,
    timestamp: *mut u32,
    keyframe: *mut u8,
) -> NcStatus {
    guarded(|| {
        // SAFETY: the caller guarantees these are null or valid.
        let (Some(receiver), Some(data), Some(size)) = (
            unsafe { receiver.as_mut() },
            unsafe { data.as_mut() },
            unsafe { size.as_mut() },
        ) else {
            return NcStatus::InvalidArgument;
        };
        let Some(unit) = receiver.receiver.pop() else {
            *data = std::ptr::null();
            *size = 0;
            return NcStatus::Ok;
        };
        *data = unit.data.as_ptr();
        *size = unit.data.len();
        // SAFETY: the caller guarantees each is null or valid.
        if let Some(timestamp) = unsafe { timestamp.as_mut() } {
            *timestamp = unit.timestamp;
        }
        // SAFETY: as above.
        if let Some(keyframe) = unsafe { keyframe.as_mut() } {
            *keyframe = u8::from(unit.keyframe);
        }
        NcStatus::Ok
    })
}

/// Set `*requested` to 1 once after decoding stopped for want of a key
/// frame (send a PLI/FIR, or `nc_encoder_request_keyframe` locally), else 0.
///
/// # Safety
///
/// `receiver` and `requested` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nc_rtp_receiver_keyframe_request(
    receiver: *mut NcRtpReceiver,
    requested: *mut u8,
) -> NcStatus {
    // SAFETY: the caller guarantees both are null or valid.
    let (Some(receiver), Some(requested)) =
        (unsafe { receiver.as_mut() }, unsafe { requested.as_mut() })
    else {
        return NcStatus::InvalidArgument;
    };
    *requested = u8::from(receiver.receiver.take_keyframe_request());
    NcStatus::Ok
}
