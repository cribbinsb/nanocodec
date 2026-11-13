//! Film grain synthesis signalling (AV1 §5.9.30, §7.18.3): the encoder
//! measures the source's noise on flat blocks (`nc_analysis::grain`),
//! caps it at the noise the final reconstruction lost there, turns the
//! model into `film_grain_params()` and codes them, so decoders lay
//! matching grain over the shown picture. References stay grain-free.
//!
//! Low delay: every shown frame takes a new `grain_seed` (libaom's
//! sequence: +3381, skipping 0; `av1/encoder/encoder_utils.c`,
//! BSD-2-Clause), so the pattern never freezes. New parameters are coded
//! on key frames and when a periodic re-measurement changes them; other
//! inter frames load them from their reference slot (`update_grain = 0`,
//! `film_grain_params_ref_idx`), which holds the parameters of the frame
//! that last refreshed it. The encoder tracks every slot's parameters for
//! that, so temporal layers and long-term references stay correct.

use nc_common::{BitWriter, Picture};
use nc_dsp::av1_film_grain::{Av1FilmGrainParams, Av1ScalingPoints, av1_grain_template_rms};
use nc_encoder::grain::{BIN_WIDTH, INTENSITY_BINS, MeasureSettings, NoiseModel, NoiseStats};
use nc_encoder::{Codec, ConfigError, ParamOverrides};

nc_encoder::param_group! {
    /// Film grain synthesis (AV1 `film_grain_params`).
    pub struct GrainParams ("grain") {
        /// Off by default until judged by eye (synthesized grain lowers
        /// PSNR, MS-SSIM and VMAF NEG by construction).
        enable: 0.0, [0.0, 1.0],
            "film grain synthesis: 1 measures the texture coding removed and signals AV1 film grain so decoders re-synthesize it",
        strength: 1.0, [0.0, 4.0],
            "synthesized noise level relative to the measured source noise: lower adds subtler grain",
        lag: 3.0, [0.0, 3.0],
            "auto-regressive lag of the grain filter (rounded): higher reproduces coarser, more structured texture",
        flat_structure: 2.0, [0.25, 16.0],
            "noise is measured only on blocks whose blurred source, less its shading plane, deviates by at most this many levels: higher also reads fine texture as noise",
        max_kurtosis: 5.0, [3.0, 20.0],
            "blocks whose fine detail is more heavy-tailed than this (3 is Gaussian noise) are lines or specks, not noise, and are not measured",
        loss_cap: 1.0, [0.0, 4.0],
            "grain never exceeds this multiple of the noise coding removed on the measured blocks: at high rates the coded picture keeps the noise itself",
        period: 16.0, [1.0, 64.0],
            "frames between noise measurements (rounded); changed parameters are re-signalled only then",
        smoothing: 0.5, [0.05, 1.0],
            "weight of a new measurement in the running noise statistics: higher follows changes faster",
        chroma: 0.0, [0.0, 1.0],
            "1 also synthesizes chroma grain (with its luma-correlated part); 0 keeps grain luma-only",
        min_strength: 1.0, [0.0, 32.0],
            "grain standard deviation to synthesize (source noise capped by the coding loss, 8-bit luma levels, peak over intensities) below which no grain is synthesized: the clean-content gate",
    }
}

/// The front ends' film grain option (the CLI's `--film-grain [STRENGTH]`,
/// the C API's `film_grain`, the browser demo's checkbox): `grain.enable`,
/// plus `grain.strength` when one is given (otherwise any earlier override,
/// or the default, stands). Only AV1 signals film grain here.
///
/// # Errors
///
/// [`ConfigError::Unsupported`] for any codec but AV1.
pub fn enable_film_grain(
    codec: Codec,
    strength: Option<f32>,
    params: &mut ParamOverrides,
) -> Result<(), ConfigError> {
    if codec != Codec::Av1 {
        return Err(ConfigError::Unsupported(
            "film grain synthesis is only available for AV1",
        ));
    }
    params.set("grain.enable", 1.0);
    if let Some(strength) = strength {
        params.set("grain.strength", strength);
    }
    Ok(())
}

/// `NUM_REF_FRAMES`: reference slots whose parameters inter frames load.
const SLOTS: usize = 8;
/// First `grain_seed` and its per-frame step (libaom's).
const INITIAL_SEED: u16 = 7391;
const SEED_STEP: u16 = 3381;
/// Most luma / chroma scaling points (`num_y_points` <= 14,
/// `num_cb_points` <= 10).
const MAX_LUMA_POINTS: usize = 14;
const MAX_CHROMA_POINTS: usize = 10;
/// `ar_coeff_shift_minus_6` + 6 range: coefficients in [-2, 2) at 6, [-0.25,
/// 0.25) at 9.
const AR_SHIFT_RANGE: (i32, i32) = (6, 9);
/// `grain_scaling_minus_8` + 8 range.
const SCALING_SHIFT_RANGE: (u32, u32) = (8, 11);
/// Chroma scaling as a function of luma alone (libaom's choice):
/// `CbMult = 0`, `CbLumaMult = 64`, `CbOffset = 0` once the spec's 128 / 128
/// / 256 biases are removed (§7.18.3.5).
const CHROMA_AS_LUMA: (u8, u8, u16) = (128, 192, 256);
/// Chroma grain is coded only when its strength reaches this fraction of
/// the luma gate.
const CHROMA_GATE_FRACTION: f64 = 0.5;
/// Seeds whose templates the grain gain is averaged over: each frame draws
/// a new seed, and one template's rms varies by several percent with it.
const GAIN_SEEDS: usize = 2;
/// Iterations of the chroma luma-coefficient rescaling (it changes the
/// chroma template's gain, which rescales it again).
const CHROMA_RESCALE_ITERATIONS: usize = 2;

/// What one frame's `film_grain_params()` codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameGrain {
    /// `apply_grain = 0`.
    Off,
    /// Parameters loaded from reference `slot` with a new seed
    /// (`update_grain = 0`; inter frames only).
    Reuse {
        /// `grain_seed`.
        seed: u16,
        /// `film_grain_params_ref_idx`.
        slot: u8,
    },
    /// New parameters (`update_grain = 1`, implied on key frames).
    New(Av1FilmGrainParams),
}

impl FrameGrain {
    /// `film_grain_params()` (AV1 §5.9.30) of a shown frame.
    pub fn write(&self, w: &mut BitWriter, inter: bool) {
        match self {
            Self::Off => w.put_bit(false), // apply_grain
            Self::Reuse { seed, slot } => {
                debug_assert!(inter, "key frames always code new grain parameters");
                w.put_bit(true); // apply_grain
                w.put_bits(u32::from(*seed), 16); // grain_seed
                w.put_bit(false); // update_grain
                w.put_bits(u32::from(*slot), 3); // film_grain_params_ref_idx
            }
            Self::New(p) => {
                w.put_bit(true); // apply_grain
                w.put_bits(u32::from(p.grain_seed), 16); // grain_seed
                if inter {
                    w.put_bit(true); // update_grain
                }
                write_parameters(w, p);
            }
        }
    }
}

/// `film_grain_params()` after `update_grain`.
fn write_parameters(w: &mut BitWriter, p: &Av1FilmGrainParams) {
    let points = |w: &mut BitWriter, count: usize, value: &[u8], scaling: &[u8]| {
        w.put_bits(count as u32, 4); // num_*_points
        for (&v, &s) in value.iter().zip(scaling).take(count) {
            w.put_bits(u32::from(v), 8); // point_*_value
            w.put_bits(u32::from(s), 8); // point_*_scaling
        }
    };
    points(w, p.y_points.count, &p.y_points.value, &p.y_points.scaling);
    w.put_bit(p.chroma_scaling_from_luma); // chroma_scaling_from_luma
    // 4:2:0 without luma points infers no chroma points.
    if !p.chroma_scaling_from_luma && p.y_points.count > 0 {
        points(
            w,
            p.cb_points.count,
            &p.cb_points.value,
            &p.cb_points.scaling,
        );
        points(
            w,
            p.cr_points.count,
            &p.cr_points.value,
            &p.cr_points.scaling,
        );
    }
    w.put_bits(u32::from(p.grain_scaling_minus_8), 2); // grain_scaling_minus_8
    w.put_bits(u32::from(p.ar_coeff_lag), 2); // ar_coeff_lag
    let luma_positions = ar_positions(usize::from(p.ar_coeff_lag));
    let chroma_positions = luma_positions + usize::from(p.y_points.count > 0);
    if p.y_points.count > 0 {
        for &c in &p.ar_coeffs_y_plus_128[..luma_positions] {
            w.put_bits(u32::from(c), 8); // ar_coeffs_y_plus_128
        }
    }
    for (count, coeffs) in [
        (p.cb_points.count, &p.ar_coeffs_cb_plus_128),
        (p.cr_points.count, &p.ar_coeffs_cr_plus_128),
    ] {
        if p.chroma_scaling_from_luma || count > 0 {
            for &c in &coeffs[..chroma_positions] {
                w.put_bits(u32::from(c), 8); // ar_coeffs_cb/cr_plus_128
            }
        }
    }
    w.put_bits(u32::from(p.ar_coeff_shift_minus_6), 2); // ar_coeff_shift_minus_6
    w.put_bits(u32::from(p.grain_scale_shift), 2); // grain_scale_shift
    for (count, (mult, luma_mult, offset)) in [
        (p.cb_points.count, p.cb_mult),
        (p.cr_points.count, p.cr_mult),
    ] {
        if count > 0 {
            w.put_bits(u32::from(mult), 8); // cb/cr_mult
            w.put_bits(u32::from(luma_mult), 8); // cb/cr_luma_mult
            w.put_bits(u32::from(offset), 9); // cb/cr_offset
        }
    }
    w.put_bit(p.overlap); // overlap_flag
    w.put_bit(p.clip_to_restricted_range); // clip_to_restricted_range
}

/// `numPosLuma` = `2 · lag · (lag + 1)`.
fn ar_positions(lag: usize) -> usize {
    2 * lag * (lag + 1)
}

/// The encoder's film grain state.
#[derive(Debug, Clone)]
pub struct GrainState {
    params: GrainParams,
    lag: usize,
    period: u64,
    /// Running noise statistics (committed).
    stats: Option<NoiseStats>,
    /// Parameters of the latest measurement (seed 0), `None` when the gate
    /// turned grain off (committed).
    current: Option<Av1FilmGrainParams>,
    /// Each reference slot's parameters (seed 0; committed).
    slots: [Option<Av1FilmGrainParams>; SLOTS],
    /// The next frame's seed (committed).
    seed: u16,
    /// Frames committed.
    frames: u64,
    /// This frame's choice, with the statistics and parameters committing
    /// it makes current.
    planned: Option<Planned>,
}

/// One frame's planned grain, applied by [`GrainState::commit`].
#[derive(Debug, Clone)]
struct Planned {
    syntax: FrameGrain,
    stats: Option<NoiseStats>,
    current: Option<Av1FilmGrainParams>,
    /// The parameters the frame's refreshed slots now hold.
    saved: Option<Av1FilmGrainParams>,
}

impl GrainState {
    /// The state for `params`, or `None` when grain synthesis is off.
    #[must_use]
    pub fn new(params: GrainParams) -> Option<Self> {
        (params.enable >= 0.5).then(|| Self {
            params,
            lag: params.lag.round() as usize,
            period: (params.period.round() as u64).max(1),
            stats: None,
            current: None,
            slots: [None; SLOTS],
            seed: INITIAL_SEED,
            frames: 0,
            planned: None,
        })
    }

    /// Plan the grain of the frame just reconstructed (`recon`, final
    /// after loop filters) from `source`: a key frame, or an inter frame
    /// predicting from `reference_slot`. Derived from committed state only,
    /// so a recode plans the same way.
    pub fn plan(
        &mut self,
        source: &Picture,
        recon: &Picture,
        size: (usize, usize),
        reference_slot: Option<u8>,
    ) {
        let key = reference_slot.is_none();
        let measure = key || self.frames.is_multiple_of(self.period);
        let (stats, current) = if measure {
            let settings = MeasureSettings {
                lag: self.lag,
                flat_structure: f64::from(self.params.flat_structure),
                max_kurtosis: f64::from(self.params.max_kurtosis),
                chroma: self.params.chroma >= 0.5,
            };
            let frame = NoiseStats::measure(source, recon, size, &settings);
            let stats = match &self.stats {
                Some(old) => {
                    let mut blended = old.clone();
                    blended.blend(&frame, f64::from(self.params.smoothing));
                    blended
                }
                None => frame,
            };
            let current = stats
                .model()
                .and_then(|m| synthesis_params(&m, &self.params));
            (Some(stats), current)
        } else {
            (self.stats.clone(), self.current)
        };
        let syntax = match (current, reference_slot) {
            (None, _) => FrameGrain::Off,
            (Some(p), Some(slot)) if self.slots[usize::from(slot)] == Some(p) => {
                FrameGrain::Reuse {
                    seed: self.seed,
                    slot,
                }
            }
            (Some(p), _) => FrameGrain::New(Av1FilmGrainParams {
                grain_seed: self.seed,
                ..p
            }),
        };
        self.planned = Some(Planned {
            syntax,
            stats,
            current,
            saved: current,
        });
    }

    /// This frame's `film_grain_params()` choice (after [`GrainState::plan`]).
    #[must_use]
    pub fn syntax(&self) -> FrameGrain {
        self.planned.as_ref().map_or(FrameGrain::Off, |p| p.syntax)
    }

    /// Commit the planned frame: `refreshed(slot)` says which slots it
    /// refreshed.
    pub fn commit(&mut self, refreshed: impl Fn(usize) -> bool) {
        let Some(planned) = self.planned.take() else {
            return;
        };
        self.stats = planned.stats;
        self.current = planned.current;
        for (slot, saved) in self.slots.iter_mut().enumerate() {
            if refreshed(slot) {
                *saved = planned.saved;
            }
        }
        self.seed = match self.seed.wrapping_add(SEED_STEP) {
            0 => INITIAL_SEED,
            seed => seed,
        };
        self.frames += 1;
    }
}

/// Mean template rms of `p` (Y, Cb, Cr) over the first [`GAIN_SEEDS`] of
/// the seed sequence.
fn template_gain(p: &Av1FilmGrainParams) -> [f64; 3] {
    let mut total = [0.0; 3];
    let mut seed = INITIAL_SEED;
    for _ in 0..GAIN_SEEDS {
        let rms = av1_grain_template_rms(&Av1FilmGrainParams {
            grain_seed: seed,
            ..*p
        });
        for (t, r) in total.iter_mut().zip(rms) {
            *t += r / GAIN_SEEDS as f64;
        }
        seed = seed.wrapping_add(SEED_STEP);
    }
    total
}

/// Quantised AR coefficients (`+ 128`) of `coeffs` at `shift`.
fn quantise_ar<const N: usize>(coeffs: &[f64], shift: i32) -> [u8; N] {
    let mut out = [128u8; N];
    for (o, &c) in out.iter_mut().zip(coeffs) {
        *o = ((c * f64::from(1 << shift)).round().clamp(-128.0, 127.0) as i32 + 128) as u8;
    }
    out
}

/// The scaling points of `strength` (noise standard deviation per
/// intensity bin, already scaled) for a template of rms `rms` at scaling
/// shift `shift`: one point at each measured bin's centre.
fn scaling_points<const N: usize>(
    strength: &[Option<f64>; INTENSITY_BINS],
    rms: f64,
    shift: u32,
) -> Av1ScalingPoints<N> {
    let mut points = Av1ScalingPoints::<N>::default();
    for (bin, s) in strength.iter().enumerate() {
        let Some(s) = *s else { continue };
        if points.count == N {
            break;
        }
        points.value[points.count] = (bin * BIN_WIDTH + BIN_WIDTH / 2) as u8;
        points.scaling[points.count] = (s * f64::from(1u32 << shift) / rms)
            .round()
            .clamp(0.0, 255.0) as u8;
        points.count += 1;
    }
    points
}

/// The largest scaling shift at which `peak` (noise standard deviation)
/// still fits 8-bit scaling for a template of rms `rms`.
fn scaling_shift(peak: f64, rms: f64) -> u32 {
    (SCALING_SHIFT_RANGE.0..=SCALING_SHIFT_RANGE.1)
        .rev()
        .find(|&shift| peak * f64::from(1u32 << shift) / rms <= 255.0)
        .unwrap_or(SCALING_SHIFT_RANGE.0)
}

/// Largest measured value of a strength function.
fn peak(strength: &[Option<f64>; INTENSITY_BINS]) -> f64 {
    strength.iter().flatten().copied().fold(0.0, f64::max)
}

/// Mean measured value of a strength function.
fn mean(strength: &[Option<f64>; INTENSITY_BINS]) -> f64 {
    let measured: Vec<f64> = strength.iter().flatten().copied().collect();
    if measured.is_empty() {
        0.0
    } else {
        measured.iter().sum::<f64>() / measured.len() as f64
    }
}

/// `film_grain_params()` for `model` (seed 0), or `None` when its luma
/// noise is below the gate.
fn synthesis_params(model: &NoiseModel, params: &GrainParams) -> Option<Av1FilmGrainParams> {
    // The gate reads the measured noise (capped by the loss), so
    // `strength` does not move it.
    let measured = model.grain_strength(f64::from(params.loss_cap));
    let gate = f64::from(params.min_strength);
    if peak(&measured[0]) < gate.max(f64::MIN_POSITIVE) {
        return None;
    }
    let chroma_on =
        [1, 2].map(|c| params.chroma >= 0.5 && peak(&measured[c]) >= CHROMA_GATE_FRACTION * gate);
    let gain = f64::from(params.strength);
    let strength = measured.map(|plane| plane.map(|s| s.map(|s| s * gain)));
    let luma_peak = peak(&strength[0]);

    // Every coefficient, the chroma luma terms included, shares one shift.
    let positions = ar_positions(model.lag);
    let largest = model
        .luma_ar
        .iter()
        .chain(model.chroma_ar.iter().flatten())
        .fold(1e-4f64, |m, &c| m.max(c.abs()));
    let ar_shift =
        (7 - (largest.log2().floor() as i32 + 1)).clamp(AR_SHIFT_RANGE.0, AR_SHIFT_RANGE.1);

    let mut p = Av1FilmGrainParams {
        grain_seed: 0,
        ar_coeff_lag: model.lag as u8,
        ar_coeffs_y_plus_128: quantise_ar(&model.luma_ar, ar_shift),
        ar_coeff_shift_minus_6: (ar_shift - AR_SHIFT_RANGE.0) as u8,
        grain_scale_shift: 0,
        cb_mult: CHROMA_AS_LUMA,
        cr_mult: CHROMA_AS_LUMA,
        overlap: true,
        // Clipping to the restricted range would also clip every source
        // highlight above 235 on the frames that carry grain, and only on
        // those: visible flicker on limited-range content that overshoots.
        clip_to_restricted_range: false,
        ..Av1FilmGrainParams::default()
    };
    // Placeholder points so every template is generated for its gain.
    p.y_points.count = 1;
    p.cb_points.count = usize::from(chroma_on[0]);
    p.cr_points.count = usize::from(chroma_on[1]);

    // The chroma filter's luma term was fitted between noise levels; the
    // templates carry unit grain, so it scales by the luma / chroma noise
    // per template unit (libaom's y_corr), which in turn depends on the
    // chroma template's gain.
    let luma_unit = |rms: [f64; 3]| mean(&strength[0]) / rms[0].max(1e-6);
    let mut rms = template_gain(&p);
    let rescales = if chroma_on.contains(&true) {
        CHROMA_RESCALE_ITERATIONS
    } else {
        0
    };
    for _ in 0..rescales {
        for c in 0..2 {
            let chroma_unit = mean(&strength[c + 1]) / rms[c + 1].max(1e-6);
            let mut coeffs = model.chroma_ar[c].clone();
            coeffs[positions] *= if chroma_unit > 1e-6 {
                luma_unit(rms) / chroma_unit
            } else {
                0.0
            };
            let quantised = quantise_ar(&coeffs, ar_shift);
            if c == 0 {
                p.ar_coeffs_cb_plus_128 = quantised;
            } else {
                p.ar_coeffs_cr_plus_128 = quantised;
            }
        }
        rms = template_gain(&p);
    }

    let chroma_peak = [1, 2].map(|c| {
        if chroma_on[c - 1] {
            peak(&strength[c]) / rms[c]
        } else {
            0.0
        }
    });
    let shift = scaling_shift(
        (luma_peak / rms[0]).max(chroma_peak[0]).max(chroma_peak[1]),
        1.0,
    );
    p.grain_scaling_minus_8 = (shift - SCALING_SHIFT_RANGE.0) as u8;
    p.y_points = scaling_points::<MAX_LUMA_POINTS>(&strength[0], rms[0], shift);
    p.cb_points = if chroma_on[0] {
        scaling_points::<MAX_CHROMA_POINTS>(&strength[1], rms[1], shift)
    } else {
        Av1ScalingPoints::default()
    };
    p.cr_points = if chroma_on[1] {
        scaling_points::<MAX_CHROMA_POINTS>(&strength[2], rms[2], shift)
    } else {
        Av1ScalingPoints::default()
    };
    // Unused coefficient positions stay at the neutral 128.
    if p.cb_points.count == 0 {
        p.ar_coeffs_cb_plus_128 = [128; 25];
    }
    if p.cr_points.count == 0 {
        p.ar_coeffs_cr_plus_128 = [128; 25];
    }
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::decode::BitReader;

    fn model() -> NoiseModel {
        let mut strength = [[None; INTENSITY_BINS]; 3];
        strength[0][2] = Some(3.0);
        strength[0][5] = Some(4.0);
        strength[1][2] = Some(1.0);
        NoiseModel {
            lag: 3,
            luma_ar: vec![0.03; 24],
            chroma_ar: [vec![0.02; 25], vec![0.0; 25]],
            noise: strength,
            // Coding removed more than the noise: the cap does not bind.
            loss: strength.map(|plane| plane.map(|s| s.map(|s| 2.0 * s))),
        }
    }

    /// Full-strength luma and chroma grain, gated at 1 level.
    fn params() -> GrainParams {
        GrainParams {
            strength: 1.0,
            chroma: 1.0,
            min_strength: 1.0,
            ..GrainParams::default()
        }
    }

    #[test]
    fn the_film_grain_option_is_av1_only() {
        let spec = |name| {
            GrainParams::SPECS
                .iter()
                .find(|s| s.name == name)
                .expect("spec")
        };
        let mut overrides = ParamOverrides::default();
        enable_film_grain(Codec::Av1, Some(0.4), &mut overrides).expect("AV1");
        assert_eq!(overrides.value(spec("grain.enable")), 1.0);
        assert_eq!(overrides.value(spec("grain.strength")), 0.4);
        for codec in [Codec::H264, Codec::Hevc, Codec::Vvc, Codec::Vp9, Codec::Av2] {
            assert!(enable_film_grain(codec, None, &mut ParamOverrides::default()).is_err());
        }
    }

    #[test]
    fn quiet_content_has_no_grain() {
        let mut m = model();
        m.noise[0] = [Some(0.2); INTENSITY_BINS];
        assert_eq!(synthesis_params(&m, &params()), None);
    }

    #[test]
    fn grain_is_capped_at_the_noise_coding_removed() {
        let mut m = model();
        // A high rate kept most of the noise: only 2 levels were lost at the
        // bin whose source noise is 4.
        m.loss[0][5] = Some(2.0);
        let p = synthesis_params(&m, &params()).expect("grain");
        let rms = av1_grain_template_rms(&Av1FilmGrainParams { grain_seed: 1, ..p })[0];
        let shift = u32::from(p.grain_scaling_minus_8) + 8;
        let level = f64::from(p.y_points.scaling[1]) * rms / f64::from(1u32 << shift);
        assert!((level - 2.0).abs() < 0.2, "noise level {level}");
        // Nothing lost anywhere: no grain.
        m.loss[0] = [Some(0.0); INTENSITY_BINS];
        assert_eq!(synthesis_params(&m, &params()), None);
    }

    #[test]
    fn synthesized_luma_noise_matches_the_target() {
        let p = synthesis_params(&model(), &params()).expect("grain");
        assert_eq!(p.y_points.count, 2);
        assert_eq!(p.cb_points.count, 1);
        assert_eq!(p.cr_points.count, 0);
        // A seed outside the averaged ones: the gain is the filter's.
        let rms = av1_grain_template_rms(&Av1FilmGrainParams { grain_seed: 1, ..p })[0];
        let shift = u32::from(p.grain_scaling_minus_8) + 8;
        let level = f64::from(p.y_points.scaling[1]) * rms / f64::from(1u32 << shift);
        assert!((level - 4.0).abs() < 0.2, "noise level {level}");
    }

    #[test]
    fn parameters_round_trip_through_the_syntax_length() {
        let p = synthesis_params(&model(), &params()).expect("grain");
        let mut w = BitWriter::new();
        FrameGrain::New(p).write(&mut w, true);
        w.byte_align_zero();
        let bits: usize = 1 + 16 + 1 // apply, seed, update
            + 4 + 2 * 16 + 1 // luma points, chroma_scaling_from_luma
            + 4 + 16 + 4 // cb, cr points
            + 2 + 2 + 24 * 8 + 25 * 8 // shifts, lag, luma and cb coefficients
            + 2 + 2 + 8 + 8 + 9 + 1 + 1;
        let bytes = w.into_bytes();
        assert_eq!(bytes.len(), bits.div_ceil(8));
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(1).expect("bit"), 1);
    }
}
