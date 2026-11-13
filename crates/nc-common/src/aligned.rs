//! A heap byte buffer whose first element sits on a chosen alignment.
//!
//! SIMD kernels run fastest when rows start on cache-line boundaries. Rather
//! than using the raw allocator (which needs `unsafe`), the buffer
//! over-allocates a `Vec<u8>` and starts at the first aligned byte inside it.
//! The `Vec` is never resized, so the aligned start never moves.

/// Zero-initialised bytes, aligned to `align`.
///
/// `Clone` is implemented by hand: a derived clone would copy the `Vec` to a
/// new address where the stored offset no longer lands on an aligned byte.
#[derive(Debug)]
pub(crate) struct AlignedBuf {
    storage: Vec<u8>,
    offset: usize,
    len: usize,
    align: usize,
}

impl AlignedBuf {
    /// `len` zero bytes whose first byte is a multiple of `align` (a power of two).
    pub(crate) fn zeroed(len: usize, align: usize) -> Self {
        debug_assert!(align.is_power_of_two());
        let storage = vec![0u8; len + align - 1];
        let addr = storage.as_ptr() as usize;
        let offset = addr.next_multiple_of(align) - addr;
        Self {
            storage,
            offset,
            len,
            align,
        }
    }

    // Inlined: every plane region goes through these (out of line they
    // cost ~11 instructions a call, ~17M per 720p decode).
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.storage[self.offset..self.offset + self.len]
    }

    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.storage[self.offset..self.offset + self.len]
    }
}

impl Clone for AlignedBuf {
    fn clone(&self) -> Self {
        let mut copy = Self::zeroed(self.len, self.align);
        copy.as_mut_slice().copy_from_slice(self.as_slice());
        copy
    }
}

#[cfg(test)]
mod tests {
    use super::AlignedBuf;

    #[test]
    fn start_is_aligned_and_length_exact() {
        for align in [16, 32, 64, 128] {
            let buf = AlignedBuf::zeroed(1000, align);
            assert_eq!(buf.as_slice().as_ptr() as usize % align, 0);
            assert_eq!(buf.as_slice().len(), 1000);
            assert!(buf.as_slice().iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn clone_keeps_contents() {
        let mut buf = AlignedBuf::zeroed(10, 64);
        buf.as_mut_slice()[3] = 7;
        let copy = buf.clone();
        assert_eq!(copy.as_slice()[3], 7);
        assert_eq!(copy.as_slice().as_ptr() as usize % 64, 0);
    }
}
