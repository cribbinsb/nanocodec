//! Instruction-set selection.
//!
//! x86-64 detects AVX2 and AVX-512F/BW/VL at run time. `AArch64` always has NEON. WebAssembly has
//! no run-time detection: SIMD128 is available exactly when the crate was
//! compiled with `-C target-feature=+simd128`.

use std::fmt;

/// An instruction set a kernel implementation targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Isa {
    /// Portable Rust; the reference every other implementation must match.
    Scalar,
    /// x86-64 AVX2 (implies SSE4.1/SSSE3).
    Avx2,
    /// x86-64 AVX-512F/BW/VL, with AVX2 fallback kernels.
    Avx512,
    /// `AArch64` Advanced SIMD.
    Neon,
    /// WebAssembly 128-bit SIMD.
    Simd128,
}

impl Isa {
    /// Every ISA, scalar first.
    pub const ALL: [Isa; 5] = [Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon, Isa::Simd128];

    /// Whether this ISA can run here (this CPU, this build).
    #[must_use]
    pub fn is_supported(self) -> bool {
        match self {
            Isa::Scalar => true,
            Isa::Avx2 => avx2_supported(),
            Isa::Avx512 => avx512_supported(),
            Isa::Neon => cfg!(target_arch = "aarch64"),
            Isa::Simd128 => cfg!(all(target_arch = "wasm32", target_feature = "simd128")),
        }
    }

    /// The fastest supported ISA.
    #[must_use]
    pub fn best() -> Isa {
        [Isa::Avx512, Isa::Avx2, Isa::Neon, Isa::Simd128]
            .into_iter()
            .find(|isa| isa.is_supported())
            .unwrap_or(Isa::Scalar)
    }

    /// Short lowercase name, for logs and benchmark tables.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
            Isa::Neon => "neon",
            Isa::Simd128 => "simd128",
        }
    }
}

impl fmt::Display for Isa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_supported() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_supported() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn avx512_supported() -> bool {
    avx2_supported()
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

#[cfg(not(target_arch = "x86_64"))]
fn avx512_supported() -> bool {
    false
}

impl std::str::FromStr for Isa {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|isa| isa.name() == value)
            .ok_or_else(|| format!("unknown DSP ISA {value:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::Isa;

    #[test]
    fn names_roundtrip_and_unknown_names_fail() {
        for isa in Isa::ALL {
            assert_eq!(isa.name().parse::<Isa>(), Ok(isa));
        }
        assert!("avx-512".parse::<Isa>().is_err());
    }

    #[test]
    fn avx512_requires_the_fallback_tier() {
        if Isa::Avx512.is_supported() {
            assert!(Isa::Avx2.is_supported());
        }
        for isa in Isa::ALL {
            assert_eq!(crate::Dsp::with_isa(isa).is_some(), isa.is_supported());
        }
    }

    #[test]
    fn scalar_is_always_supported_and_best_is_supported() {
        assert!(Isa::Scalar.is_supported());
        assert!(Isa::best().is_supported());
    }
}
