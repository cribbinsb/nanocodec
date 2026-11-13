//! The reconstruction as a player shows it: a picture coded smaller than
//! the stream (adaptive coded resolution) is scaled to the stream size,
//! its render size, by the player model (`nc_dsp::display`), so that
//! `--recon` always holds stream-size frames for scoring.

use nc_common::{Picture, PlaneIndex};
use nc_dsp::display::display_scale_plane;

/// A reusable stream-size buffer for scaled frames.
#[derive(Debug, Default)]
pub(crate) struct Display {
    scaled: Option<Picture>,
}

impl Display {
    /// `picture` as shown at the `stream` size: itself when it has that
    /// size, else scaled into this buffer.
    pub(crate) fn shown<'a>(
        &'a mut self,
        picture: &'a Picture,
        (width, height): (usize, usize),
    ) -> Result<&'a Picture, String> {
        if (picture.width(), picture.height()) == (width, height) {
            return Ok(picture);
        }
        if self
            .scaled
            .as_ref()
            .is_none_or(|s| (s.width(), s.height()) != (width, height))
        {
            self.scaled = Some(Picture::new(width, height, 0).map_err(|e| e.to_string())?);
        }
        let scaled = self.scaled.as_mut().expect("allocated above");
        for plane in PlaneIndex::ALL {
            display_scale_plane(
                &picture.plane(plane).view(),
                &mut scaled.plane_mut(plane).view_mut(),
            );
        }
        Ok(scaled)
    }
}
