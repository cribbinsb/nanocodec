//! The carry of the byte-oriented range encoders that keep finished bytes
//! in a 64-bit window and write several at once (AV1's `od_ec`, AV2's AVM
//! coder, VP9's bool coder): a carry out of the window adds one to the
//! number the bytes already written spell. Only this step is shared; the
//! coders' arithmetic differs (VP9's 8-bit range split per bit, AV1's and
//! AV2's 15-bit CDF intervals with their own scaling), and so do their
//! windows and flush points.

/// Add one to the big-endian number `out` (a carry out of an encoder's
/// window): trailing `0xff` bytes become `0x00` until one takes the carry.
/// The encoders start with a byte that a carry never passes (VP9's zero
/// marker bit, `od_ec`'s range bound), so it always stops inside `out`.
pub fn propagate_carry(out: &mut [u8]) {
    for byte in out.iter_mut().rev() {
        let (sum, overflow) = byte.overflowing_add(1);
        *byte = sum;
        if !overflow {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::propagate_carry;

    #[test]
    fn carries_through_trailing_ff_bytes() {
        let mut out = [0x12, 0xff, 0xff];
        propagate_carry(&mut out);
        assert_eq!(out, [0x13, 0x00, 0x00]);
        let mut out = [0x12, 0x34];
        propagate_carry(&mut out);
        assert_eq!(out, [0x12, 0x35]);
    }
}
