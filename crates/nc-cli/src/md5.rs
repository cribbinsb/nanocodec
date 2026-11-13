//! MD5 (RFC 1321) for per-frame decode hashes: the libvpx test vectors'
//! `.md5` files list the MD5 of each output frame (`vpxdec --md5 --i420`),
//! and `nanocodec decode --frame-md5` prints the same lines to compare.
//! Integrity checking only; not for security.

/// Per-round left rotations.
const SHIFTS: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];
/// Initial state (A, B, C, D).
const INITIAL: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
const BLOCK_BYTES: usize = 64;
/// Bytes of the message length appended to the padding.
const LENGTH_BYTES: usize = 8;

/// `K[i] = floor(abs(sin(i + 1)) * 2^32)`.
fn constants() -> [u32; 64] {
    std::array::from_fn(|i| ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32)
}

/// Incremental MD5.
pub(crate) struct Md5 {
    state: [u32; 4],
    k: [u32; 64],
    pending: Vec<u8>,
    length: u64,
}

impl Md5 {
    pub(crate) fn new() -> Self {
        Self {
            state: INITIAL,
            k: constants(),
            pending: Vec::with_capacity(BLOCK_BYTES),
            length: 0,
        }
    }

    pub(crate) fn update(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        if !self.pending.is_empty() {
            let take = (BLOCK_BYTES - self.pending.len()).min(data.len());
            self.pending.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.pending.len() == BLOCK_BYTES {
                let block: [u8; BLOCK_BYTES] = self.pending[..].try_into().expect("one block");
                self.compress(&block);
                self.pending.clear();
            }
        }
        let (blocks, remainder) = data.as_chunks::<BLOCK_BYTES>();
        for block in blocks {
            self.compress(block);
        }
        self.pending.extend_from_slice(remainder);
    }

    pub(crate) fn finish(mut self) -> [u8; 16] {
        let bits = self.length.wrapping_mul(8);
        let mut tail = vec![0x80];
        let used = (self.pending.len() + 1) % BLOCK_BYTES;
        let zeros = (BLOCK_BYTES + BLOCK_BYTES - LENGTH_BYTES - used) % BLOCK_BYTES;
        tail.resize(1 + zeros, 0);
        tail.extend_from_slice(&bits.to_le_bytes());
        self.update(&tail);
        let mut digest = [0; 16];
        for (out, word) in digest.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            *out = word.to_le_bytes();
        }
        digest
    }

    /// One 64-byte block (RFC 1321 §3.4).
    fn compress(&mut self, block: &[u8; BLOCK_BYTES]) {
        let words: [u32; 16] = std::array::from_fn(|i| {
            u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().expect("word"))
        });
        let mut state = self.state;
        for (round, (&shift, &constant)) in SHIFTS.iter().zip(&self.k).enumerate() {
            let [a, b, c, d] = state;
            let (mix, word) = match round / 16 {
                0 => ((b & c) | (!b & d), round),
                1 => ((d & b) | (!d & c), (5 * round + 1) % 16),
                2 => (b ^ c ^ d, (3 * round + 5) % 16),
                _ => (c ^ (b | !d), (7 * round) % 16),
            };
            let rotated = a
                .wrapping_add(mix)
                .wrapping_add(constant)
                .wrapping_add(words[word])
                .rotate_left(shift);
            state = [d, b.wrapping_add(rotated), b, c];
        }
        for (total, value) in self.state.iter_mut().zip(state) {
            *total = total.wrapping_add(value);
        }
    }
}

/// Lowercase hexadecimal of a digest.
pub(crate) fn hex(digest: &[u8; 16]) -> String {
    use std::fmt::Write;
    digest
        .iter()
        .fold(String::with_capacity(32), |mut text, byte| {
            write!(text, "{byte:02x}").expect("writing to a String cannot fail");
            text
        })
}

/// Lowercase MD5 hex digest of `picture`'s visible I420 planes (`Y`, `U`, `V`).
pub(crate) fn picture(picture: &nc_common::Picture) -> String {
    let mut hasher = Md5::new();
    let (width, height) = (picture.width(), picture.height());
    for plane in nc_common::PlaneIndex::ALL {
        let ss = usize::from(plane != nc_common::PlaneIndex::Y);
        let (w, h) = ((width + ss) >> ss, (height + ss) >> ss);
        let view = picture.plane(plane).view();
        for y in 0..h {
            hasher.update(&view.row(y)[..w]);
        }
    }
    hex(&hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn md5(data: &[u8]) -> String {
        let mut hasher = Md5::new();
        hasher.update(data);
        hex(&hasher.finish())
    }

    #[test]
    fn rfc_1321_test_suite() {
        assert_eq!(md5(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(md5(b"message digest"), "f96b697d7cb7938d525a2f31aaf161d0");
        assert_eq!(
            md5(
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"
            ),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }

    #[test]
    fn incremental_updates_match_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let mut hasher = Md5::new();
        for chunk in data.chunks(37) {
            hasher.update(chunk);
        }
        assert_eq!(hex(&hasher.finish()), md5(&data));
    }
}
