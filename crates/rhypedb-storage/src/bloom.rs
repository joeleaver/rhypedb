use std::io::{self, Read, Write};

use xxhash_rust::xxh3::xxh3_64_with_seed;

/// Bloom filter for SST files.
///
/// Stores hashed user keys to enable fast "definitely not in the file"
/// checks. False positives are possible (~1% at the default settings),
/// but false negatives are impossible.
///
/// Uses double hashing: two hashes with different seeds combined as
/// `h(i) = h1 + i * h2` to derive k pseudo-independent hash positions.
/// The hash function is selected per filter:
/// - `HashAlgo::Fnv1a` — original byte-by-byte FNV-1a (kept for backward
///   compatibility with v2 SST files written before xxh3 landed).
/// - `HashAlgo::Xxh3` — xxh3 64-bit with seeds. ~4× faster than FNV-1a
///   on the SIMD path; used for newly-written v3 SST files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgo {
    Fnv1a = 0,
    Xxh3 = 1,
}

impl HashAlgo {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(HashAlgo::Fnv1a),
            1 => Some(HashAlgo::Xxh3),
            _ => None,
        }
    }
    pub fn to_byte(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<u8>,
    num_bits: u32, // total number of bits (m); always a power of two for newly-built filters
    k: u8,         // number of hash functions
    /// Set when `num_bits` is a power of two — lets `add`/`contains` use a
    /// bitmask (`& mask`) instead of a costly `divq` for modulo. Legacy
    /// on-disk filters where `num_bits` is non-power-of-two fall back to the
    /// modulo path.
    mask: Option<u32>,
    /// Which hash function this filter uses. Tracked per-filter so legacy
    /// v2 SSTs (FNV-1a) keep working alongside new v3 SSTs (xxh3) in the
    /// same LSM tree without compaction.
    algo: HashAlgo,
}

/// Default bits per entry. With k=7, this gives ~1% false positive rate.
pub const DEFAULT_BITS_PER_ENTRY: usize = 10;

/// Default number of hash functions. Optimal for 10 bits/entry.
pub const DEFAULT_K: u8 = 7;

impl BloomFilter {
    /// Create an empty bloom filter sized for the expected number of entries.
    pub fn new(expected_entries: usize) -> Self {
        Self::with_params(expected_entries, DEFAULT_BITS_PER_ENTRY, DEFAULT_K)
    }

    pub fn with_params(expected_entries: usize, bits_per_entry: usize, k: u8) -> Self {
        Self::with_params_and_algo(expected_entries, bits_per_entry, k, HashAlgo::Xxh3)
    }

    pub fn with_params_and_algo(
        expected_entries: usize,
        bits_per_entry: usize,
        k: u8,
        algo: HashAlgo,
    ) -> Self {
        // Round up to a power of two so `add`/`contains` can index with a
        // bitmask instead of `%`. Worst-case ~2× memory overhead — for the
        // 100K SSTs this works out to ~125 KB extra per file. Larger bit array
        // also strictly improves the false-positive rate at the same k.
        let raw_bits = (expected_entries.max(1) * bits_per_entry).max(1);
        let num_bits = raw_bits.next_power_of_two();
        let num_bytes = num_bits.div_ceil(8);
        Self {
            bits: vec![0u8; num_bytes],
            num_bits: num_bits as u32,
            k,
            mask: Some((num_bits - 1) as u32),
            algo,
        }
    }

    pub fn algo(&self) -> HashAlgo {
        self.algo
    }

    /// Add a key to the filter.
    pub fn add(&mut self, key: &[u8]) {
        let (h1, h2) = self.double_hash(key);
        if let Some(mask) = self.mask {
            for i in 0..self.k {
                let bit = combined_hash(h1, h2, i) as u32 & mask;
                self.set_bit(bit as usize);
            }
        } else {
            for i in 0..self.k {
                let bit = combined_hash(h1, h2, i) % self.num_bits as u64;
                self.set_bit(bit as usize);
            }
        }
    }

    /// Check whether a key might be in the filter. Returns true if all k
    /// hash positions are set (might be present; false positive possible),
    /// or false if any position is unset (definitely not present).
    pub fn contains(&self, key: &[u8]) -> bool {
        if self.num_bits == 0 {
            return true;
        }
        let (h1, h2) = self.double_hash(key);
        if let Some(mask) = self.mask {
            for i in 0..self.k {
                let bit = combined_hash(h1, h2, i) as u32 & mask;
                if !self.get_bit(bit as usize) {
                    return false;
                }
            }
        } else {
            for i in 0..self.k {
                let bit = combined_hash(h1, h2, i) % self.num_bits as u64;
                if !self.get_bit(bit as usize) {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the (h1, h2) double-hash pair using this filter's algorithm.
    /// Inlined into add/contains so the per-key dispatch cost is zero.
    #[inline]
    fn double_hash(&self, key: &[u8]) -> (u64, u64) {
        match self.algo {
            HashAlgo::Fnv1a => fnv_double_hash(key),
            HashAlgo::Xxh3 => xxh3_double_hash(key),
        }
    }

    pub fn num_bits(&self) -> u32 {
        self.num_bits
    }

    pub fn k(&self) -> u8 {
        self.k
    }

    pub fn byte_size(&self) -> usize {
        self.bits.len()
    }

    fn set_bit(&mut self, bit: usize) {
        let byte = bit / 8;
        let mask = 1u8 << (bit % 8);
        self.bits[byte] |= mask;
    }

    fn get_bit(&self, bit: usize) -> bool {
        let byte = bit / 8;
        let mask = 1u8 << (bit % 8);
        (self.bits[byte] & mask) != 0
    }

    /// Serialize the bloom payload: [num_bits: u32 BE][k: u8][bits: ceil(num_bits/8) bytes].
    ///
    /// The `algo` field is NOT written here — it's stored alongside the
    /// bloom block in the SST footer/header so that v2 SSTs (no algo byte;
    /// always FNV-1a) and v3 SSTs (algo byte) can both be parsed by reading
    /// the SST version first.
    pub fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        w.write_all(&self.num_bits.to_be_bytes())?;
        w.write_all(&[self.k])?;
        w.write_all(&self.bits)
    }

    /// Deserialize a bloom payload. `algo` is supplied by the caller from
    /// the surrounding SST format (v2 → `Fnv1a`, v3 → algo byte in header).
    pub fn read_from(r: &mut dyn Read, algo: HashAlgo) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let num_bits = u32::from_be_bytes(buf4);

        let mut buf1 = [0u8; 1];
        r.read_exact(&mut buf1)?;
        let k = buf1[0];

        let num_bytes = (num_bits as usize).div_ceil(8);
        let mut bits = vec![0u8; num_bytes];
        r.read_exact(&mut bits)?;

        // Pick the fast path (bitmask) for power-of-two filters, fall back to
        // modulo for legacy v2 files that pre-date the power-of-two rounding.
        let mask = if num_bits > 0 && num_bits.is_power_of_two() {
            Some(num_bits - 1)
        } else {
            None
        };

        Ok(Self { bits, num_bits, k, mask, algo })
    }
}

/// FNV-1a 64-bit hash with the given seed. Kept for legacy v2 SST files.
fn fnv1a_64(data: &[u8], seed: u64) -> u64 {
    let mut hash = seed;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// FNV-1a double-hash: two seeds chosen to be distinct and well-mixed.
fn fnv_double_hash(key: &[u8]) -> (u64, u64) {
    let h1 = fnv1a_64(key, 0xcbf29ce484222325);
    let h2 = fnv1a_64(key, 0x9e3779b97f4a7c15);
    // h2 must be odd to avoid degenerate behavior modulo m.
    (h1, h2 | 1)
}

/// xxh3 double-hash: two seeded passes over the key. xxh3 processes 8+ bytes
/// per cycle in its SIMD-friendly inner loop, so this is ~4× faster than
/// `fnv_double_hash` for the ~26-byte object/edge keys we typically index.
fn xxh3_double_hash(key: &[u8]) -> (u64, u64) {
    let h1 = xxh3_64_with_seed(key, 0xcbf29ce484222325);
    let h2 = xxh3_64_with_seed(key, 0x9e3779b97f4a7c15);
    (h1, h2 | 1)
}

fn combined_hash(h1: u64, h2: u64, i: u8) -> u64 {
    h1.wrapping_add((i as u64).wrapping_mul(h2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_false_negatives() {
        let mut filter = BloomFilter::new(1000);
        let keys: Vec<Vec<u8>> = (0..1000).map(|i| format!("key-{i}").into_bytes()).collect();

        for k in &keys {
            filter.add(k);
        }
        for k in &keys {
            assert!(filter.contains(k), "key {:?} should be in filter", k);
        }
    }

    #[test]
    fn low_false_positive_rate() {
        let mut filter = BloomFilter::new(1000);
        for i in 0..1000 {
            filter.add(format!("present-{i}").as_bytes());
        }

        let mut false_positives = 0;
        let trials = 10_000;
        for i in 0..trials {
            if filter.contains(format!("absent-{i}").as_bytes()) {
                false_positives += 1;
            }
        }
        let fp_rate = false_positives as f64 / trials as f64;
        eprintln!("bloom filter false positive rate: {fp_rate:.4} ({false_positives}/{trials})");
        assert!(
            fp_rate < 0.03,
            "false positive rate {fp_rate} too high (expected < 3%)"
        );
    }

    #[test]
    fn empty_filter_returns_false() {
        let filter = BloomFilter::new(100);
        assert!(!filter.contains(b"anything"));
    }

    #[test]
    fn serialization_roundtrip() {
        let mut filter = BloomFilter::new(500);
        let keys: Vec<Vec<u8>> = (0..500).map(|i| format!("k{i}").into_bytes()).collect();
        for k in &keys {
            filter.add(k);
        }

        let mut buf = Vec::new();
        filter.write_to(&mut buf).unwrap();

        let restored = BloomFilter::read_from(&mut &buf[..], filter.algo()).unwrap();
        assert_eq!(restored.num_bits, filter.num_bits);
        assert_eq!(restored.k, filter.k);

        for k in &keys {
            assert!(restored.contains(k));
        }
    }

    #[test]
    fn different_sized_keys() {
        let mut filter = BloomFilter::new(100);
        let short = b"a";
        let long = &vec![0u8; 1024];

        filter.add(short);
        filter.add(long);
        assert!(filter.contains(short));
        assert!(filter.contains(long));
    }

    #[test]
    fn new_filter_uses_power_of_two_bits_and_mask() {
        // 100 entries × 10 bits = 1000 raw bits → rounded up to 1024 (2^10).
        // Mask should be num_bits - 1 = 1023 = 0b1111111111.
        let filter = BloomFilter::new(100);
        assert_eq!(filter.num_bits, 1024);
        assert_eq!(filter.mask, Some(1023));
    }

    #[test]
    fn legacy_non_power_of_two_filter_round_trips_via_modulo() {
        // Hand-build a filter with non-power-of-two num_bits (simulating a v2
        // file produced before the power-of-two rounding landed). The reader
        // path must fall back to modulo addressing — verify the bit set/get
        // line up so contains() doesn't false-negative.
        let num_bits = 997u32; // prime, definitely not a power of two
        let k = 7u8;
        let num_bytes = (num_bits as usize).div_ceil(8);
        let mut legacy = BloomFilter {
            bits: vec![0u8; num_bytes],
            num_bits,
            k,
            mask: None, // forces modulo path
            algo: HashAlgo::Fnv1a, // legacy v2 SSTs use FNV
        };

        for i in 0..200 {
            legacy.add(format!("k{i}").as_bytes());
        }
        for i in 0..200 {
            assert!(
                legacy.contains(format!("k{i}").as_bytes()),
                "legacy filter must not false-negative on inserted key {i}"
            );
        }

        // Round-trip through serialization — the deserialized filter must
        // detect non-power-of-two and keep using the modulo path, preserving
        // no-false-negative semantics.
        let mut buf = Vec::new();
        legacy.write_to(&mut buf).unwrap();
        let restored = BloomFilter::read_from(&mut &buf[..], HashAlgo::Fnv1a).unwrap();
        assert_eq!(restored.num_bits, num_bits);
        assert!(restored.mask.is_none(), "legacy filter must stay on modulo path");
        for i in 0..200 {
            assert!(restored.contains(format!("k{i}").as_bytes()));
        }
    }

    #[test]
    fn xxh3_filter_default_and_roundtrip() {
        // New filters default to xxh3. Serializing + deserializing while
        // telling read_from the algo must preserve no-false-negatives.
        let mut filter = BloomFilter::new(500);
        assert_eq!(filter.algo(), HashAlgo::Xxh3, "default should be xxh3");

        let keys: Vec<Vec<u8>> = (0..500).map(|i| format!("xxh-{i}").into_bytes()).collect();
        for k in &keys {
            filter.add(k);
        }
        let mut buf = Vec::new();
        filter.write_to(&mut buf).unwrap();
        let restored = BloomFilter::read_from(&mut &buf[..], filter.algo()).unwrap();
        assert_eq!(restored.algo(), HashAlgo::Xxh3);
        for k in &keys {
            assert!(restored.contains(k), "xxh3 false-negative on {:?}", k);
        }
    }

    #[test]
    fn xxh3_filter_low_false_positive_rate() {
        let mut filter = BloomFilter::new(1000);
        for i in 0..1000 {
            filter.add(format!("present-{i}").as_bytes());
        }
        let trials = 10_000;
        let mut fp = 0;
        for i in 0..trials {
            if filter.contains(format!("absent-{i}").as_bytes()) {
                fp += 1;
            }
        }
        let fp_rate = fp as f64 / trials as f64;
        eprintln!("xxh3 bloom FPR: {fp_rate:.4} ({fp}/{trials})");
        // With ~16 bits/entry (pow-of-2 rounded from 10) and k=7, FPR should
        // be well under 1% — even tighter than the FNV path.
        assert!(fp_rate < 0.01, "xxh3 FPR {fp_rate} too high (expected < 1%)");
    }

    #[test]
    fn handles_empty_key() {
        let mut filter = BloomFilter::new(10);
        filter.add(b"");
        assert!(filter.contains(b""));
    }
}
