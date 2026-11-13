//! Packed-byte lanes for loop filters (VP9 `vp9_loop_filter`, AV1
//! `deblock::av1_packed`): one byte vector per edge position holds eight
//! independent lines, with libvpx/libaom's byte masks and narrow filter
//! (BSD-3-Clause / BSD-2-Clause; attribution in `THIRD_PARTY.md`) and a
//! widening to 16-bit words for the smoothing filters. Vertical edges
//! transpose 8x8 byte tiles in registers.
#![allow(clippy::inline_always)]
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2_wide;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon_wide;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod simd128;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod simd128_wide;

/// Lines per basic group; paired adapters contain two independent groups.
pub(crate) const LANES: usize = 8;
/// Taps per side of the 4/8-tap filters (p3..q3).
pub(crate) const INNER: usize = 4;
/// Offset between unsigned samples and the narrow filter's signed bytes.
const SIGN: u8 = 128;

/// Widened byte lanes, with one or two independent eight-line groups.
pub(crate) trait Words: Copy {
    fn splat(v: i16) -> Self;
    fn add(self, b: Self) -> Self;
    fn sub(self, b: Self) -> Self;
    fn shr<const N: i32>(self) -> Self;
}

/// Byte lanes, one per edge line. Ordinary adapters hold eight; paired
/// adapters hold two independent groups. These memory methods access one
/// eight-byte group; [`PairedBytes`] provides explicit paired memory access.
pub(crate) trait Bytes: Copy {
    type W: Words;
    fn splat(v: u8) -> Self;
    unsafe fn load(p: *const u8) -> Self;
    unsafe fn store(self, p: *mut u8);
    /// Store only the low two bytes to writable memory. ISA must be enabled.
    unsafe fn store2(self, p: *mut u8);
    /// Four readable bytes, with unused lanes zeroed. ISA must be enabled.
    unsafe fn load4(p: *const u8) -> Self;
    /// Store only the low four bytes to writable memory. ISA must be enabled.
    unsafe fn store4(self, p: *mut u8);
    fn max(self, b: Self) -> Self;
    fn sub_sat(self, b: Self) -> Self;
    fn add_sat(self, b: Self) -> Self;
    fn signed_add(self, b: Self) -> Self;
    fn signed_sub(self, b: Self) -> Self;
    fn and(self, b: Self) -> Self;
    fn xor(self, b: Self) -> Self;
    fn eq(self, b: Self) -> Self;
    fn shr<const N: i32>(self) -> Self;
    fn signed_shr<const N: i32>(self) -> Self;
    fn zip_lo(self, b: Self) -> Self;
    fn zip_hi(self, b: Self) -> Self;
    fn any(self) -> bool;
    fn widen(self) -> Self::W;
    fn narrow(v: Self::W) -> Self;
    /// The 8x8 byte transpose of each independent group in `rows` (vector
    /// `i`'s byte `j` becomes vector `j`'s byte `i` within each group).
    /// The default interleaves with [`Bytes::zip_lo`]/[`Bytes::zip_hi`];
    /// an ISA overrides it where wider interleaves are cheaper.
    #[inline(always)]
    fn transpose(rows: [Self; LANES]) -> [Self; LANES] {
        let mut rows = rows;
        for _ in 0..3 {
            let mut next = rows;
            for i in 0..LANES / 2 {
                next[2 * i] = rows[i].zip_lo(rows[i + LANES / 2]);
                next[2 * i + 1] = rows[i].zip_hi(rows[i + LANES / 2]);
            }
            rows = next;
        }
        rows
    }
}

/// Two independent eight-byte groups sharing SIMD arithmetic. Ordinary
/// narrow memory methods retain their eight-byte contract; wide tile IO uses
/// these explicit paired loads/stores instead.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
pub(crate) trait PairedBytes: Bytes {
    /// Caller supplies two readable eight-byte spans and enables the ISA.
    unsafe fn load_pair(first: *const u8, second: *const u8) -> Self;
    /// Caller supplies two writable eight-byte spans and enables the ISA.
    unsafe fn store_pair(self, first: *mut u8, second: *mut u8);
}

/// A small word table split into byte planes, for register lookups by width.
/// Unused entries are zero; packing constants here avoids rebuilding vectors
/// inside each edge's decision/ramp kernel.
pub(crate) struct WordTable {
    low: [u8; 16],
    high: [u8; 16],
}
impl WordTable {
    pub const fn new(values: &[i16]) -> Self {
        assert!(values.len() <= 16);
        let mut table = Self {
            low: [0; 16],
            high: [0; 16],
        };
        let mut i = 0;
        while i < values.len() {
            table.low[i] = values[i] as u8;
            table.high[i] = (values[i] as u16 >> 8) as u8;
            i += 1;
        }
        table
    }
}

/// Word lanes with the masks, products and memory access of a decision
/// cascade (AV2 deblocking, `deblock::av2_decide`). Masks are all ones or
/// all zeros per lane.
pub(crate) trait MaskWords: Words {
    /// Look up each word lane in a constant table; indices must be 0..=15.
    fn lookup(self, table: &WordTable) -> Self;
    /// Eight values from `src`.
    fn load(src: &[i16; LANES]) -> Self;
    fn store(self, dst: &mut [i16; LANES]);
    fn abs(self) -> Self;
    fn min(self, b: Self) -> Self;
    fn max(self, b: Self) -> Self;
    /// The low 16 bits of each product.
    fn mul(self, b: Self) -> Self;
    /// `(self · b + 2^14) >> 15` per lane.
    fn mul_q15_round(self, b: Self) -> Self;
    /// Mask of the lanes where `self > b` (signed).
    fn gt(self, b: Self) -> Self;
    /// Mask of the lanes where `self == b`.
    fn eq(self, b: Self) -> Self;
    fn and(self, b: Self) -> Self;
    fn or(self, b: Self) -> Self;
    /// `self & !b`.
    fn and_not(self, b: Self) -> Self;
    /// Per lane, `if_set` where the mask `self` is set, else `if_clear`.
    fn select(self, if_set: Self, if_clear: Self) -> Self;
    /// Whether any lane is nonzero.
    fn any(self) -> bool;
}

/// Byte lanes of four interleaved lines: 32 bytes where byte `4k + l` is
/// line `l` of lane `k` (a horizontal AV2 edge row: four lines per
/// segment), and two half-loads.
pub(crate) trait QuadBytes: Bytes<W: MaskWords> {
    /// The four lines of the 32 bytes at `src`, one vector each.
    ///
    /// # Safety
    ///
    /// `src` points to 32 readable bytes; the ISA is enabled.
    unsafe fn load_lines4(src: *const u8) -> [Self; 4];
    /// Interleave `lines` back into the 32 bytes at `dst`
    /// ([`QuadBytes::load_lines4`] inverted).
    ///
    /// # Safety
    ///
    /// `dst` points to 32 writable bytes; the ISA is enabled.
    unsafe fn store_lines4(lines: [Self; 4], dst: *mut u8);
    /// Lanes 0..3 from the four bytes at `low`, lanes 4..7 from `high`.
    ///
    /// # Safety
    ///
    /// Both point to four readable bytes; the ISA is enabled.
    unsafe fn load_halves(low: *const u8, high: *const u8) -> Self;
}

#[inline(always)]
pub(crate) fn difference<B: Bytes>(a: B, b: B) -> B {
    a.sub_sat(b).max(b.sub_sat(a))
}
#[inline(always)]
pub(crate) fn select<B: Bytes>(old: B, new: B, mask: B) -> B {
    old.xor(old.xor(new).and(mask))
}
#[inline(always)]
pub(crate) fn within<B: Bytes>(v: B, limit: u8) -> B {
    v.sub_sat(B::splat(limit)).eq(B::splat(0))
}

#[inline(always)]
pub(crate) fn transpose<B: Bytes>(rows: [B; LANES]) -> [B; LANES] {
    B::transpose(rows)
}

/// Load a tile as positions (each vector holds the independent edge lines).
#[inline(always)]
pub(crate) unsafe fn load<B: Bytes>(
    data: *const u8,
    edge: usize,
    step: usize,
    along: usize,
    before: usize,
) -> [B; LANES] {
    // SAFETY: dispatch validated the complete edge footprint and eight lines.
    unsafe {
        let values = core::array::from_fn(|i| {
            if along == 1 {
                B::load(data.add(edge - before * step + i * step))
            } else {
                B::load(data.add(edge - before + i * along))
            }
        });
        if along == 1 {
            values
        } else {
            transpose(values)
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn store<B: Bytes>(
    data: *mut u8,
    edge: usize,
    step: usize,
    along: usize,
    before: usize,
    values: [B; LANES],
    changed: core::ops::Range<usize>,
) {
    let rows = if along == 1 {
        values
    } else {
        transpose(values)
    };
    // SAFETY: dispatch validated every byte in this eight-line tile.
    unsafe {
        for (i, value) in rows.into_iter().enumerate() {
            if along == 1 && !changed.contains(&i) {
                continue;
            }
            let at = if along == 1 {
                edge - before * step + i * step
            } else {
                edge - before + i * along
            };
            value.store(data.add(at));
        }
    }
}

#[inline(always)]
pub(crate) fn narrow<B: Bytes>(p: &mut [B; LANES], q: &mut [B; LANES], mask: B, hev: B) {
    let sign = B::splat(SIGN);
    let (p1, p0, q0, q1) = (
        p[1].xor(sign),
        p[0].xor(sign),
        q[0].xor(sign),
        q[1].xor(sign),
    );
    let delta = q0.signed_sub(p0);
    // Three equal-sign saturated additions equal one clamp after 3*delta.
    let f = p1
        .signed_sub(q1)
        .and(hev)
        .signed_add(delta)
        .signed_add(delta)
        .signed_add(delta)
        .and(mask);
    let f1 = f.signed_add(B::splat(4)).signed_shr::<3>();
    let f2 = f.signed_add(B::splat(3)).signed_shr::<3>();
    q[0] = q0.signed_sub(f1).xor(sign);
    p[0] = p0.signed_add(f2).xor(sign);
    let outer = f1
        .signed_add(B::splat(1))
        .signed_shr::<1>()
        .and(hev.xor(B::splat(255)));
    q[1] = q1.signed_sub(outer).xor(sign);
    p[1] = p1.signed_add(outer).xor(sign);
}
