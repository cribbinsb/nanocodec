//! Raw I420 frame I/O: tightly packed Y, then U, then V, 8 bits per sample.
//!
//! This is the format of every evaluation clip (see `eval/`) and of the
//! decoders' reconstruction output, so reading and writing it exactly
//! matters for bit-exact conformance checks.

use std::io::{self, Read, Write};

use crate::error::{Error, Result};
use crate::picture::{Picture, PlaneIndex};

/// Read the next frame into `picture`. Returns `Ok(false)` at a clean end of
/// input (no bytes left) and an error if the input ends mid-frame.
///
/// Only the visible area is written; call [`Picture::extend_borders`] if the
/// border is needed.
///
/// # Errors
///
/// [`Error::TruncatedFrame`] if the input ends part-way through a frame, and
/// [`Error::Io`] for read failures.
pub fn read_frame<R: Read>(reader: &mut R, picture: &mut Picture) -> Result<bool> {
    let expected = picture.packed_frame_len();
    let mut got = 0;
    for index in PlaneIndex::ALL {
        let mut view = picture.plane_mut(index).view_mut();
        for y in 0..view.height() {
            let row = view.row_mut(y);
            let n = read_up_to(reader, row)?;
            got += n;
            if n < row.len() {
                return if got == 0 {
                    Ok(false)
                } else {
                    Err(Error::TruncatedFrame { expected, got })
                };
            }
        }
    }
    Ok(true)
}

/// Write the visible area of `picture` as one packed I420 frame.
///
/// # Errors
///
/// [`Error::Io`] if the writer fails.
pub fn write_frame<W: Write>(writer: &mut W, picture: &Picture) -> Result<()> {
    for index in PlaneIndex::ALL {
        for row in picture.plane(index).view().rows() {
            writer.write_all(row)?;
        }
    }
    Ok(())
}

/// Fill as much of `buf` as the reader can supply; fewer bytes means end of input.
fn read_up_to<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

#[cfg(test)]
mod tests {
    use super::{read_frame, write_frame};
    use crate::error::Error;
    use crate::picture::Picture;

    fn packed_frame(width: usize, height: usize, seed: u8) -> Vec<u8> {
        (0..width * height * 3 / 2)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    #[test]
    fn round_trip_is_exact() {
        let (w, h) = (6, 4);
        let mut input = packed_frame(w, h, 1);
        input.extend(packed_frame(w, h, 2));
        let mut reader = input.as_slice();

        let mut pic = Picture::new(w, h, 8).unwrap();
        let mut output = Vec::new();
        while read_frame(&mut reader, &mut pic).unwrap() {
            write_frame(&mut output, &pic).unwrap();
        }
        assert_eq!(output, input);
    }

    #[test]
    fn partial_frame_is_an_error() {
        let input = packed_frame(4, 4, 0);
        let mut reader = &input[..input.len() - 1];
        let mut pic = Picture::new(4, 4, 0).unwrap();
        match read_frame(&mut reader, &mut pic) {
            Err(Error::TruncatedFrame { expected, got }) => {
                assert_eq!((expected, got), (24, 23));
            }
            other => panic!("expected TruncatedFrame, got {other:?}"),
        }
    }

    #[test]
    fn empty_input_is_clean_end() {
        let mut reader: &[u8] = &[];
        let mut pic = Picture::new(4, 4, 0).unwrap();
        assert!(!read_frame(&mut reader, &mut pic).unwrap());
    }
}
