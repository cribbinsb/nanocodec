//! Raw I420 frame I/O: tightly packed Y, then U, then V, 8 bits per sample
//! (`yuv420p`); decoded pictures of more than 8 bits are written as 16-bit
//! little-endian samples at their own depth (`yuv420p10le` for 10-bit, as
//! HM and ffmpeg write them).
//!
//! This is the format of every evaluation clip (see `eval/`) and of the
//! decoders' reconstruction output, so reading and writing it exactly
//! matters for bit-exact conformance checks.

use std::io::{self, IoSlice, IoSliceMut, Read, Write};

use crate::error::{Error, Result};
use crate::picture::{Picture, PlaneIndex};
use crate::sample::Sample;
use crate::view::PlaneView;

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
        let (width, height, stride) = (view.width(), view.height(), view.stride());
        // A whole plane is large enough to bypass BufReader's intermediate
        // buffer. Scatter directly into the padded picture's visible rows.
        let mut rows: Vec<_> = view
            .data_mut()
            .chunks_mut(stride)
            .take(height)
            .map(|row| IoSliceMut::new(&mut row[..width]))
            .collect();
        let n = read_up_to_vectored(reader, &mut rows)?;
        got += n;
        if n < width * height {
            return if got == 0 {
                Ok(false)
            } else {
                Err(Error::TruncatedFrame { expected, got })
            };
        }
    }
    Ok(true)
}

/// Write the visible area of `picture` as one packed I420 frame (16-bit
/// little-endian samples for `Picture<u16>`).
///
/// # Errors
///
/// [`Error::Io`] if the writer fails.
pub fn write_frame<W: Write, T: Sample>(writer: &mut W, picture: &Picture<T>) -> Result<()> {
    write_planes(
        writer,
        PlaneIndex::ALL.map(|index| picture.plane(index).view()),
    )
}

/// Write the rows of `planes`, in order, as packed samples (bytes for `u8`,
/// 16-bit little-endian for `u16`): a whole frame, or a monochrome picture's
/// luma alone.
///
/// # Errors
///
/// [`Error::Io`] if the writer fails.
pub fn write_planes<'a, W: Write, T: Sample>(
    writer: &mut W,
    planes: impl IntoIterator<Item = PlaneView<'a, T>>,
) -> Result<()> {
    if T::BYTES == 1 {
        // Present a complete frame to BufWriter, so large raw output can
        // bypass its buffer rather than copy every short sample row.
        let mut rows = Vec::new();
        for plane in planes {
            rows.reserve(plane.height());
            for row in plane.rows() {
                rows.push(IoSlice::new(
                    T::borrowed_le_bytes(row).expect("byte samples"),
                ));
            }
        }
        return write_vectored_all(writer, &mut rows).map_err(Error::Io);
    }
    let mut scratch = Vec::new();
    for plane in planes {
        for row in plane.rows() {
            writer.write_all(T::le_bytes(row, &mut scratch))?;
        }
    }
    Ok(())
}

/// Write every slice, including writers that accept only part of one row.
/// Shared by packed picture output and cropped decoder output.
///
/// # Errors
/// Returns the underlying write error, or `WriteZero` if progress stops.
pub fn write_vectored_all<W: Write>(writer: &mut W, rows: &mut [IoSlice<'_>]) -> io::Result<()> {
    let mut rows = rows;
    IoSlice::advance_slices(&mut rows, 0);
    while !rows.is_empty() {
        match writer.write_vectored(rows) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => IoSlice::advance_slices(&mut rows, n),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Fill as many rows as the reader supplies; fewer bytes means end of input.
fn read_up_to_vectored<R: Read>(reader: &mut R, rows: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
    let mut filled = 0;
    let mut rows = rows;
    while !rows.is_empty() {
        match reader.read_vectored(rows) {
            Ok(0) => break,
            Ok(n) => {
                filled += n;
                IoSliceMut::advance_slices(&mut rows, n);
            }
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
    use crate::picture::{Picture, PlaneIndex};

    fn packed_frame(width: usize, height: usize, seed: u8) -> Vec<u8> {
        (0..width * height * 3 / 2)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    struct ShortIo {
        input: std::io::Cursor<Vec<u8>>,
        output: Vec<u8>,
        calls: usize,
    }

    impl std::io::Read for ShortIo {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            self.calls += 1;
            if self.calls.is_multiple_of(5) {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let end = out.len().min(7);
            std::io::Read::read(&mut self.input, &mut out[..end])
        }
    }

    impl std::io::Write for ShortIo {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            self.calls += 1;
            if self.calls.is_multiple_of(5) {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let n = data.len().min(7);
            self.output.extend_from_slice(&data[..n]);
            Ok(n)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn vectored_rows_handle_short_interrupted_io_and_truncation() {
        let (w, h) = (6, 130);
        let data = packed_frame(w, h, 19);
        let mut io = ShortIo {
            input: std::io::Cursor::new(data.clone()),
            output: Vec::new(),
            calls: 0,
        };
        let mut picture = Picture::new(w, h, 8).unwrap();
        assert!(read_frame(&mut io, &mut picture).unwrap());
        assert!(!read_frame(&mut io, &mut picture).unwrap());
        write_frame(&mut io, &picture).unwrap();
        assert_eq!(io.output, data);
        io.input = std::io::Cursor::new(data[..data.len() - 1].to_vec());
        assert!(
            matches!(read_frame(&mut io, &mut picture), Err(Error::TruncatedFrame { expected, got })
            if expected == data.len() && got == data.len() - 1)
        );
    }

    #[test]
    fn full_planes_bypass_the_intermediate_read_buffer() {
        use std::io::{BufReader, IoSliceMut, Read};
        struct VectorsOnly(std::io::Cursor<Vec<u8>>, usize);
        impl Read for VectorsOnly {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                panic!("plane reads should use the vectored entry");
            }
            fn read_vectored(&mut self, rows: &mut [IoSliceMut<'_>]) -> std::io::Result<usize> {
                self.1 += 1;
                self.0.read_vectored(rows)
            }
        }
        let data = packed_frame(32, 16, 7);
        let mut reader =
            BufReader::with_capacity(64, VectorsOnly(std::io::Cursor::new(data.clone()), 0));
        let mut picture = Picture::new(32, 16, 8).unwrap();
        assert!(read_frame(&mut reader, &mut picture).unwrap());
        assert_eq!(reader.get_ref().1, 3, "one direct call per plane");
        let mut result = Vec::new();
        write_frame(&mut result, &picture).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn vectored_writer_skips_empty_slices_and_reports_no_progress() {
        use std::io::IoSlice;
        let mut output = Vec::new();
        super::write_vectored_all(&mut output, &mut [IoSlice::new(&[])]).unwrap();
        let mut full: &mut [u8] = &mut [];
        assert_eq!(
            super::write_vectored_all(&mut full, &mut [IoSlice::new(b"x")])
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::WriteZero
        );
    }

    #[test]
    fn round_trip_is_exact() {
        let (w, h) = (6, 4);
        let mut input = packed_frame(w, h, 1);
        input.extend(packed_frame(w, h, 2));
        let mut reader = input.as_slice();

        let mut pic = Picture::<u8>::new(w, h, 8).unwrap();
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
        let mut pic = Picture::<u8>::new(4, 4, 0).unwrap();
        match read_frame(&mut reader, &mut pic) {
            Err(Error::TruncatedFrame { expected, got }) => {
                assert_eq!((expected, got), (24, 23));
            }
            other => panic!("expected TruncatedFrame, got {other:?}"),
        }
    }

    #[test]
    fn deep_pictures_are_written_little_endian() {
        let mut pic = Picture::<u16>::new(2, 2, 0).unwrap();
        pic.plane_mut(PlaneIndex::Y).view_mut().fill(0x3ff);
        pic.plane_mut(PlaneIndex::V).view_mut().fill(0x102);
        let mut out = Vec::new();
        write_frame(&mut out, &pic).unwrap();
        assert_eq!(out, [0xff, 3, 0xff, 3, 0xff, 3, 0xff, 3, 0, 0, 2, 1]);
    }

    #[test]
    fn empty_input_is_clean_end() {
        let mut reader: &[u8] = &[];
        let mut pic = Picture::<u8>::new(4, 4, 0).unwrap();
        assert!(!read_frame(&mut reader, &mut pic).unwrap());
    }
}
