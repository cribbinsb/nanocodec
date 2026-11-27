//! Instruction-set selection.
//!
//! x86-64 detects AVX2 and the AVX-512 Ice Lake feature set at run time.
//! `AArch64` always has NEON and checks for SVE2 on Linux. WebAssembly has
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
    /// x86-64 AVX-512 at the Ice Lake / Zen 4 level (the features `avx512_supported` checks, GFNI included), with
    /// AVX2 fallback kernels. Owner decision 2026-11-16: the same tier as dav1d's and SVT-AV1's
    /// `avx512icl`, so their kernels (which use GFNI and VBMI) port as they are. CPUs with
    /// AVX-512 but without that set (Skylake-X, Cascade Lake, Cooper Lake) run the AVX2 tier.
    Avx512,
    /// `AArch64` Advanced SIMD.
    Neon,
    /// `AArch64` SVE2 with NEON fallbacks on Linux.
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    Sve2,
    /// WebAssembly 128-bit SIMD.
    Simd128,
}

impl Isa {
    /// Every ISA, scalar first.
    pub const ALL: [Isa; if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
        6
    } else {
        5
    }] = [
        Isa::Scalar,
        Isa::Avx2,
        Isa::Avx512,
        Isa::Neon,
        #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
        Isa::Sve2,
        Isa::Simd128,
    ];

    /// Whether this ISA can run here (this CPU, this build).
    #[must_use]
    pub fn is_supported(self) -> bool {
        match self {
            Isa::Scalar => true,
            Isa::Avx2 => avx2_supported(),
            Isa::Avx512 => avx512_supported(),
            Isa::Neon => cfg!(target_arch = "aarch64"),
            #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
            Isa::Sve2 => sve2_supported(),
            Isa::Simd128 => cfg!(all(target_arch = "wasm32", target_feature = "simd128")),
        }
    }

    /// The fastest supported ISA.
    #[must_use]
    pub fn best() -> Isa {
        [
            Isa::Avx512,
            Isa::Avx2,
            #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
            Isa::Sve2,
            Isa::Neon,
            Isa::Simd128,
        ]
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
            #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
            Isa::Sve2 => "sve2",
            Isa::Simd128 => "simd128",
        }
    }
}

impl fmt::Display for Isa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Granite Rapids server silicon measured on c8i (family 6, model AD).
#[cfg(target_arch = "x86_64")]
pub(crate) fn granite_rapids() -> bool {
    use std::arch::x86_64::__cpuid;

    let vendor = __cpuid(0);
    if vendor.eax < 1
        || [vendor.ebx, vendor.edx, vendor.ecx]
            != [
                u32::from_le_bytes(*b"Genu"),
                u32::from_le_bytes(*b"ineI"),
                u32::from_le_bytes(*b"ntel"),
            ]
    {
        return false;
    }
    let signature = __cpuid(1).eax;
    let family = (signature >> 8) & 0xf;
    let model = ((signature >> 12) & 0xf0) | ((signature >> 4) & 0xf);
    family == 6 && model == 0xad
}

#[cfg(target_arch = "x86_64")]
fn avx2_supported() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_supported() -> bool {
    false
}

/// The AVX-512 tier's required features: dav1d's `avx512icl` set (Intel Ice Lake and later, AMD
/// Zen 4 and later). A kernel in the tier may `#[target_feature(enable = ...)]` any of them.
#[cfg(target_arch = "x86_64")]
fn avx512_supported() -> bool {
    use std::arch::is_x86_feature_detected as has;
    // Include the BMI requirements inherited by dav1d's AVX2 CPU flag;
    // Rust's AVX-512 detection also checks OS ZMM/opmask state.
    avx2_supported()
        && has!("avx512f")
        && has!("avx512cd")
        && has!("avx512bw")
        && has!("avx512dq")
        && has!("avx512vl")
        && has!("avx512ifma")
        && has!("avx512vbmi")
        && has!("avx512vbmi2")
        && has!("avx512vnni")
        && has!("avx512bitalg")
        && has!("avx512vpopcntdq")
        && has!("gfni")
        && has!("vaes")
        && has!("vpclmulqdq")
        && has!("bmi1")
        && has!("bmi2")
}

#[cfg(not(target_arch = "x86_64"))]
fn avx512_supported() -> bool {
    false
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn sve2_supported() -> bool {
    // Rust's Linux detector reads HWCAP2_SVE2.
    std::arch::is_aarch64_feature_detected!("sve2")
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

    /// Run under SDE SKX as well as native: the older AVX-512 combination
    /// must choose AVX2 globally, rather than install mixed feature contracts.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn older_avx512_cpus_select_avx2() {
        if std::arch::is_x86_feature_detected!("avx512f")
            && !std::arch::is_x86_feature_detected!("gfni")
        {
            assert_eq!(Isa::best(), Isa::Avx2);
            assert!(crate::Dsp::with_isa(Isa::Avx512).is_none());
        }
    }

    #[test]
    fn scalar_is_always_supported_and_best_is_supported() {
        assert!(Isa::Scalar.is_supported());
        assert!(Isa::best().is_supported());
    }
}
