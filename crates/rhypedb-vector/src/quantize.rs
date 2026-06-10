use std::io;

use rand::Rng;

use crate::serial::{
    read_byte_vec, read_f32, read_f32_vec, read_u32, read_u8, write_byte_vec, write_f32,
    write_f32_slice, write_u32, write_u8,
};

/// Sample from N(0,1) via Box-Muller transform.
fn sample_normal(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.random_range(1e-10..1.0f32);
    let u2: f32 = rng.random_range(0.0..std::f32::consts::TAU);
    (-2.0 * u1.ln()).sqrt() * u2.cos()
}

/// Configuration for TurboQuant vector compression.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    pub dimensions: u32,
    pub bits: u8, // 2, 3, or 4 bits per dimension
}

impl TurboQuantConfig {
    pub fn new(dimensions: u32, bits: u8) -> Self {
        assert!(matches!(bits, 2..=4), "bits must be 2, 3, or 4");
        Self { dimensions, bits }
    }

    pub fn compressed_size(&self) -> usize {
        let total_bits = self.dimensions as usize * self.bits as usize;
        total_bits.div_ceil(8)
    }

    pub fn total_size(&self) -> usize {
        // quantized data + norm (f32)
        self.compressed_size() + 4
    }
}

/// A trained TurboQuant quantizer for a specific vector collection.
///
/// Implements the MSE variant (TurboQuant_mse) of arXiv:2504.19874:
/// 1. Normalize to unit sphere, store norm
/// 2. Random orthogonal rotation (QR of Gaussian matrix) → coords ~ Beta
/// 3. Lloyd-Max scalar quantization with the analytic Beta-distribution codebook
///
/// The inner-product estimate is the rotated-query · centroids dot, rescaled by
/// the stored norm. (The paper's TurboQuant_prod variant adds a 1-bit QJL
/// residual correction to make the *dot-product* estimate unbiased — valuable
/// for aggregate inner products like attention, but for k-NN ranking at 2–4
/// bits it added ~2× build/search cost for no recall gain, so it is intentionally
/// omitted. See the perf/turboquant-mse change notes.)
pub struct TurboQuantizer {
    config: TurboQuantConfig,
    rotation_matrix: Vec<f32>, // d x d, row-major — orthogonal via QR
    codebook: Codebook,
}

/// Precomputed Lloyd-Max codebook for Beta-distributed values.
#[derive(Debug, Clone)]
struct Codebook {
    centroids: Vec<f32>,
    boundaries: Vec<f32>, // len = centroids.len() - 1 (interior boundaries)
}

impl TurboQuantizer {
    /// Build a quantizer for the given config. The codebook is computed
    /// analytically from the Beta distribution determined by the dimension,
    /// so no training samples are needed.
    pub fn new(config: TurboQuantConfig) -> Self {
        let dims = config.dimensions as usize;
        let num_centroids = 1usize << config.bits;

        let rotation_matrix = generate_rotation_matrix(dims);
        let codebook = compute_beta_codebook(dims, num_centroids);

        Self {
            config,
            rotation_matrix,
            codebook,
        }
    }

    /// Compress a single vector.
    pub fn compress(&self, vector: &[f32]) -> CompressedVector {
        let dims = self.config.dimensions as usize;
        assert_eq!(vector.len(), dims);

        // Step 1: Normalize to unit sphere.
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let unit: Vec<f32> = if norm > 0.0 {
            vector.iter().map(|x| x / norm).collect()
        } else {
            vec![0.0; dims]
        };

        // Step 2: Rotate.
        let rotated = mat_vec_mul(&self.rotation_matrix, &unit, dims);

        // Step 3: Lloyd-Max quantize using precomputed Beta codebook.
        let mut indices = Vec::with_capacity(dims);
        for &val in &rotated {
            indices.push(self.codebook.quantize(val) as u8);
        }

        CompressedVector::from_parts(bit_pack(&indices, self.config.bits), norm)
    }

    /// Decompress to approximate f32 values (MSE reconstruction only, no QJL correction).
    pub fn decompress(&self, compressed: &CompressedVector) -> Vec<f32> {
        let dims = self.config.dimensions as usize;

        let indices = bit_unpack(compressed.data(), self.config.bits, dims);
        let rotated: Vec<f32> = indices
            .iter()
            .map(|&i| self.codebook.centroids[i as usize])
            .collect();

        // Inverse rotation (transpose of orthogonal matrix).
        let unit = mat_vec_mul_transpose(&self.rotation_matrix, &rotated, dims);

        // Rescale by original norm.
        unit.iter().map(|x| x * compressed.norm).collect()
    }

    /// Project a full-precision query through the rotation ONCE so every
    /// candidate distance can reuse it: `⟨q, Rᵀ·c⟩ = ⟨R·q, c⟩`, turning the
    /// per-distance cost from an O(d²) inverse rotation into an O(d) dot product.
    pub fn prepare_query(&self, query: &[f32]) -> PreparedQuery {
        let dims = self.config.dimensions as usize;
        PreparedQuery {
            rq: mat_vec_mul(&self.rotation_matrix, query, dims),
            q_norm: query.iter().map(|x| x * x).sum::<f32>().sqrt(),
        }
    }

    /// Prepare an already-*stored* (compressed) vector for use as a query,
    /// skipping the O(d²) inverse rotation that `decompress` + `prepare_query`
    /// would perform.
    ///
    /// A decompressed vector is `x̂ = ‖x‖·Rᵀ·c`, where `c` is the per-dimension
    /// centroid vector. Because R is orthonormal, `R·x̂ = ‖x‖·(R·Rᵀ)·c = ‖x‖·c`
    /// (the rotated query the MSE term needs — with NO matmul). Used on the HNSW
    /// build path, where neighbor pruning re-projects stored nodes. Equivalent to
    /// `prepare_query(&decompress(compressed))` up to the tiny floating-point
    /// deviation of R·Rᵀ from the identity.
    pub fn prepare_stored(&self, compressed: &CompressedVector) -> PreparedQuery {
        let mut out = PreparedQuery { rq: Vec::new(), q_norm: 0.0 };
        self.prepare_stored_into(compressed, &mut out);
        out
    }

    /// Like [`Self::prepare_stored`], but reuses the buffer in `out` instead of
    /// allocating a fresh `rq` vector. The HNSW build path prunes up to ~2·m
    /// stored neighbors per insert, each needing a projected query; threading one
    /// scratch `PreparedQuery` through that loop removes those per-prune
    /// allocations at millions of inserts. Produces values identical to
    /// `prepare_stored`.
    pub fn prepare_stored_into(&self, compressed: &CompressedVector, out: &mut PreparedQuery) {
        let dims = self.config.dimensions as usize;

        // rq = ‖x‖ · c  (this is exactly R·x̂, the rotated query the MSE term
        // consumes — produced without the inverse-rotation matmul). Decode the
        // packed indices straight into the scaled centroid values, reusing the
        // buffer, with no intermediate index `Vec`.
        unpack_scaled_centroids(
            compressed.data(),
            self.config.bits,
            dims,
            &self.codebook.centroids,
            compressed.norm,
            &mut out.rq,
        );
        out.q_norm = out.rq.iter().map(|x| x * x).sum::<f32>().sqrt();
    }

    /// MSE inner-product estimate reusing a [`PreparedQuery`]. O(d) per call —
    /// no matrix multiplies and no full `decompress`.
    pub fn inner_product_estimate_prepared(
        &self,
        prepared: &PreparedQuery,
        compressed: &CompressedVector,
    ) -> f32 {
        debug_assert_eq!(prepared.rq.len(), self.config.dimensions as usize);

        // MSE term: ⟨q, x̂_mse⟩ = ‖x‖ · ⟨R·q, cv⟩, where cv[i] is the centroid for
        // the quantized rotated coordinate i (decompress's pre-inverse-rotation
        // values). Avoids materializing x̂_mse (which would need the R^T multiply).
        //
        // The decode is fused into the dot product: instead of `bit_unpack`-ing
        // the indices into a heap `Vec<u8>` and then zipping, we walk the packed
        // bytes and accumulate `centroids[idx] · rq[i]` in one allocation-free
        // pass. Summation is strictly position-ordered, so the result is
        // bit-identical to the previous materialized path.
        mse_dot_fused(compressed.data(), self.config.bits, &self.codebook.centroids, &prepared.rq)
            * compressed.norm
    }

    /// Compute the inner product estimate between a full-precision query and a
    /// compressed vector. Convenience wrapper that prepares the query then
    /// delegates; prefer preparing once and reusing
    /// [`Self::inner_product_estimate_prepared`] across many candidates.
    pub fn inner_product_estimate(
        &self,
        query: &[f32],
        compressed: &CompressedVector,
    ) -> f32 {
        let prepared = self.prepare_query(query);
        self.inner_product_estimate_prepared(&prepared, compressed)
    }

    /// Compute approximate distance using the MSE estimate. Wrapper that
    /// prepares the query once; prefer [`Self::distance_estimate_prepared`] when
    /// scoring many candidates against the same query.
    pub fn distance_estimate(
        &self,
        query: &[f32],
        compressed: &CompressedVector,
        metric: crate::distance::Metric,
    ) -> f32 {
        let prepared = self.prepare_query(query);
        self.distance_estimate_prepared(&prepared, compressed, metric)
    }

    /// Approximate distance reusing a [`PreparedQuery`].
    pub fn distance_estimate_prepared(
        &self,
        prepared: &PreparedQuery,
        compressed: &CompressedVector,
        metric: crate::distance::Metric,
    ) -> f32 {
        match metric {
            crate::distance::Metric::DotProduct => {
                -self.inner_product_estimate_prepared(prepared, compressed)
            }
            crate::distance::Metric::Cosine => {
                let ip = self.inner_product_estimate_prepared(prepared, compressed);
                let denom = prepared.q_norm * compressed.norm;
                if denom == 0.0 {
                    1.0
                } else {
                    1.0 - ip / denom
                }
            }
            crate::distance::Metric::L2 => {
                // ‖q - x‖² = ‖q‖² + ‖x‖² - 2⟨q, x⟩
                let ip = self.inner_product_estimate_prepared(prepared, compressed);
                prepared.q_norm * prepared.q_norm + compressed.norm * compressed.norm - 2.0 * ip
            }
        }
    }

    pub fn config(&self) -> &TurboQuantConfig {
        &self.config
    }

    pub fn write_to(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_u32(w, self.config.dimensions)?;
        write_u8(w, self.config.bits)?;
        write_f32_slice(w, &self.rotation_matrix)?;
        self.codebook.write_to(w)
    }

    pub fn read_from(r: &mut dyn io::Read) -> io::Result<Self> {
        let dimensions = read_u32(r)?;
        let bits = read_u8(r)?;
        let config = TurboQuantConfig::new(dimensions, bits);
        let dims = dimensions as usize;
        let rotation_matrix = read_f32_vec(r, dims * dims)?;
        let codebook = Codebook::read_from(r)?;
        Ok(Self {
            config,
            rotation_matrix,
            codebook,
        })
    }
}

/// A query projected through the rotation once, so that many candidate distances
/// can be estimated against it in O(d) each. Built by
/// [`TurboQuantizer::prepare_query`].
#[derive(Debug, Clone)]
pub struct PreparedQuery {
    /// R · query — the rotated query (consumed by the MSE estimate).
    pub rq: Vec<f32>,
    /// ‖query‖ — cached for the cosine/L2 conversions.
    pub q_norm: f32,
}

/// A compressed vector: the bit-packed Lloyd-Max quantized indices plus the
/// original L2 norm. The MSE-variant quantizer keeps no QJL residual, so there
/// are no sign bits or residual norm (see [`TurboQuantizer`]).
#[derive(Debug, Clone)]
pub struct CompressedVector {
    data: Box<[u8]>, // bit-packed quantized indices
    pub norm: f32,   // original vector L2 norm
}

impl CompressedVector {
    /// Bit-packed quantized indices.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Resident heap bytes of the packed codes, excluding the inline struct
    /// fields (which the caller counts as node overhead).
    pub fn heap_bytes(&self) -> usize {
        self.data.len()
    }

    fn from_parts(data: Vec<u8>, norm: f32) -> Self {
        Self { data: data.into_boxed_slice(), norm }
    }

    pub fn total_bytes(&self) -> usize {
        self.data.len() + 4 // + f32 norm
    }

    pub fn write_to(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_byte_vec(w, &self.data)?;
        write_f32(w, self.norm)
    }

    pub fn read_from(r: &mut dyn io::Read) -> io::Result<Self> {
        let data = read_byte_vec(r)?;
        let norm = read_f32(r)?;
        Ok(Self::from_parts(data, norm))
    }
}

impl Codebook {
    fn write_to(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_u32(w, self.centroids.len() as u32)?;
        write_f32_slice(w, &self.centroids)?;
        write_f32_slice(w, &self.boundaries)
    }

    fn read_from(r: &mut dyn io::Read) -> io::Result<Self> {
        let num_centroids = read_u32(r)? as usize;
        let centroids = read_f32_vec(r, num_centroids)?;
        let num_boundaries = num_centroids.saturating_sub(1);
        let boundaries = read_f32_vec(r, num_boundaries)?;
        Ok(Self {
            centroids,
            boundaries,
        })
    }

    fn quantize(&self, value: f32) -> usize {
        // Binary search on decision boundaries. `total_cmp` gives a total order
        // even for non-finite `value` (NaN sorts above every finite boundary),
        // so this never panics the way `partial_cmp(..).unwrap()` would —
        // callers should still reject non-finite vectors before quantizing, but
        // this keeps the quantizer panic-free as a backstop. Boundaries are all
        // finite and ascending, so `total_cmp` agrees with their sort order.
        match self.boundaries.binary_search_by(|b| b.total_cmp(&value)) {
            Ok(i) => i + 1,
            Err(i) => i,
        }
    }
}

// --- Codebook computation from Beta distribution ---

/// Compute the Lloyd-Max optimal codebook for the Beta distribution that
/// arises after rotating a d-dimensional unit vector.
///
/// PDF: f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
/// Supported on [-1, 1].
fn compute_beta_codebook(dims: usize, num_centroids: usize) -> Codebook {
    let grid_size = 10000;
    let grid: Vec<f32> = (0..grid_size)
        .map(|i| -1.0 + 2.0 * (i as f32 + 0.5) / grid_size as f32)
        .collect();

    let pdf: Vec<f32> = grid.iter().map(|&x| beta_pdf(x, dims)).collect();
    let dx = 2.0 / grid_size as f32;

    // CDF for quantile initialization.
    let mut cdf = Vec::with_capacity(grid_size);
    let mut cum = 0.0f32;
    for &p in &pdf {
        cum += p * dx;
        cdf.push(cum);
    }
    // Normalize CDF.
    let total = cum;
    for c in &mut cdf {
        *c /= total;
    }

    // Initialize centroids at quantile midpoints.
    let mut centroids: Vec<f32> = (0..num_centroids)
        .map(|i| {
            let target = (i as f32 + 0.5) / num_centroids as f32;
            let idx = cdf.partition_point(|&c| c < target).min(grid_size - 1);
            grid[idx]
        })
        .collect();

    // Lloyd-Max iterations.
    for _ in 0..100 {
        // Compute decision boundaries (midpoints between consecutive centroids).
        let mut boundaries = Vec::with_capacity(num_centroids - 1);
        for i in 0..num_centroids - 1 {
            boundaries.push((centroids[i] + centroids[i + 1]) / 2.0);
        }

        // Update centroids: conditional expectation within each region.
        let mut new_centroids = vec![0.0f32; num_centroids];
        let mut changed = false;

        for (c_idx, centroid) in new_centroids.iter_mut().enumerate() {
            let lo = if c_idx == 0 {
                -1.0
            } else {
                boundaries[c_idx - 1]
            };
            let hi = if c_idx == num_centroids - 1 {
                1.0
            } else {
                boundaries[c_idx]
            };

            // Numerical integration: E[X | lo < X < hi]
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for (j, &x) in grid.iter().enumerate() {
                if x >= lo && x < hi {
                    let w = pdf[j] * dx;
                    num += x * w;
                    den += w;
                }
            }

            *centroid = if den > 0.0 {
                num / den
            } else {
                centroids[c_idx]
            };

            if (*centroid - centroids[c_idx]).abs() > 1e-10 {
                changed = true;
            }
        }

        centroids = new_centroids;
        if !changed {
            break;
        }
    }

    // Final boundaries.
    let boundaries: Vec<f32> = (0..num_centroids - 1)
        .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
        .collect();

    Codebook {
        centroids,
        boundaries,
    }
}

/// Beta PDF for rotated unit vector coordinates in d dimensions.
/// f(x) = C · (1 - x²)^((d-3)/2) on [-1, 1]
fn beta_pdf(x: f32, dims: usize) -> f32 {
    let x2 = (x * x) as f64;
    if x2 >= 1.0 {
        return 0.0;
    }
    let exponent = (dims as f64 - 3.0) / 2.0;
    // We skip the normalization constant since Lloyd-Max only needs relative densities.
    (1.0 - x2).powf(exponent) as f32
}

// --- Matrix operations ---

/// Generate random orthogonal matrix via QR decomposition of Gaussian matrix.
fn generate_rotation_matrix(dims: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    let mut matrix = vec![0.0f32; dims * dims];

    // Fill with i.i.d. N(0,1).
    for val in &mut matrix {
        *val = sample_normal(&mut rng);
    }

    // QR decomposition via modified Gram-Schmidt (numerically stable variant).
    // For each column, orthogonalize against all previous columns.
    // We work in row-major but treat rows as our vectors to orthogonalize.
    for i in 0..dims {
        // Orthogonalize row i against rows 0..i
        for j in 0..i {
            let dot: f32 = (0..dims)
                .map(|k| matrix[i * dims + k] * matrix[j * dims + k])
                .sum();
            for k in 0..dims {
                matrix[i * dims + k] -= dot * matrix[j * dims + k];
            }
        }

        // Normalize row i.
        let norm: f32 = (0..dims)
            .map(|k| matrix[i * dims + k] * matrix[i * dims + k])
            .sum::<f32>()
            .sqrt();
        if norm > 1e-10 {
            for k in 0..dims {
                matrix[i * dims + k] /= norm;
            }
        }
    }

    matrix
}

/// Matrix–vector product `out[i] = Σ_j matrix[i*dims + j] · vector[j]` for a
/// row-major `dims×dims` matrix, allocating the result.
fn mat_vec_mul(matrix: &[f32], vector: &[f32], dims: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dims];
    mat_vec_mul_into(matrix, vector, dims, &mut result);
    result
}

/// Like [`mat_vec_mul`] but writes into a caller-owned buffer, so the build
/// hot path can reuse one allocation across the many `prepare_stored`
/// projections it performs per insert. Each output is the dot product of one
/// contiguous matrix row with `vector`, computed by the SIMD [`dot`] primitive.
fn mat_vec_mul_into(matrix: &[f32], vector: &[f32], dims: usize, out: &mut [f32]) {
    debug_assert_eq!(vector.len(), dims);
    debug_assert_eq!(out.len(), dims);
    debug_assert_eq!(matrix.len(), dims * dims);
    for (i, o) in out.iter_mut().enumerate() {
        *o = dot(&matrix[i * dims..i * dims + dims], vector);
    }
}

fn mat_vec_mul_transpose(matrix: &[f32], vector: &[f32], dims: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dims];
    for (i, res) in result.iter_mut().enumerate() {
        *res = (0..dims).map(|j| matrix[j * dims + i] * vector[j]).sum();
    }
    result
}

// --- SIMD dot product (the build hot path's matvec primitive) ---

/// Dot product of two equal-length `f32` slices.
///
/// Uses a runtime-detected AVX2+FMA path on `x86_64` and a portable
/// multiple-accumulator fallback everywhere else. Both *reassociate* the
/// summation (and the AVX2 path fuses each multiply-add), so the result is NOT
/// bit-identical to a naive left-to-right `.sum()` — it differs by at most a
/// few ULPs. That is fine for every caller here: this dot feeds the rotation /
/// QJL projection matvecs, whose outputs are either quantized or handed to the
/// already-approximate TurboQuant distance estimator, where a sub-ULP
/// perturbation sits far below the quantization noise floor (recall is verified
/// to be preserved within HNSW run-to-run variance). The distance kernel
/// ([`mse_dot_fused`]/[`correction_dot_fused`]) likewise reassociates for ILP;
/// the index *decode* there stays exact, only the accumulation order changes.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: `dot_avx2` is only entered when AVX2+FMA are detected at
            // runtime. It reads `a` and `b` exclusively through bounds-respecting
            // unaligned loads (`_mm256_loadu_ps`) and scalar indexing within
            // `a.len() == b.len()` (debug-asserted above).
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

/// Portable dot product with eight independent accumulators. Breaking the
/// loop-carried dependency a single accumulator would impose lets the compiler
/// pack the lanes into SSE/NEON and keep several FMAs in flight; this is also
/// the reference the SIMD path is checked against. Reassociated, so not
/// bit-identical to a naive `.sum()`.
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        // `chunks_exact(8)` yields length-8 slices; the array conversion lets
        // the optimizer drop the bounds checks on the unrolled inner loop.
        let x: &[f32; 8] = x.try_into().unwrap();
        let y: &[f32; 8] = y.try_into().unwrap();
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut total =
        ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        total += x * y;
    }
    total
}

/// True iff this CPU has both AVX2 and FMA. Cached after the first probe so the
/// per-call dispatch is a single relaxed load.
#[cfg(target_arch = "x86_64")]
fn has_avx2_fma() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    // u8::MAX = "not yet probed"; 0 = absent; 1 = present.
    static CACHE: AtomicU8 = AtomicU8::new(u8::MAX);
    let cached = CACHE.load(Ordering::Relaxed);
    if cached != u8::MAX {
        return cached == 1;
    }
    let detected = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    CACHE.store(detected as u8, Ordering::Relaxed);
    detected
}

/// AVX2+FMA dot product: four 8-wide accumulators (32 floats/iter) for
/// instruction-level parallelism, an 8-wide drain, then a scalar tail.
///
/// # Safety
/// The caller must ensure AVX2 and FMA are available (see [`has_avx2_fma`]).
/// `a.len()` must equal `b.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    // SAFETY: every intrinsic and pointer read below stays within `i < n` and
    // `n == a.len() == b.len()`; loads are unaligned (`loadu`), so no alignment
    // requirement. The enclosing `#[target_feature(enable = "avx2,fma")]` is
    // honored because callers gate on `has_avx2_fma()`.
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 32 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i + 24)),
                acc3,
            );
            i += 32;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }
        // Combine the four 256-bit accumulators, then horizontally reduce.
        let mut total = hsum256(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
        // Scalar remainder (n not a multiple of 8).
        while i < n {
            total += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        total
    }
}

/// Horizontal sum of a `__m256` (8 lanes → scalar). AVX2 implies the AVX/SSE3
/// shuffles used here, so it is callable from any `avx2`-enabled context.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // All register-only shuffles/adds (no memory access), so safe in an avx2 ctx.
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let mut s = _mm_add_ps(lo, hi); // 4 partial sums
    let shuf = _mm_movehdup_ps(s); // [s1, s1, s3, s3]
    s = _mm_add_ps(s, shuf); // [s0+s1, _, s2+s3, _]
    let hi64 = _mm_movehl_ps(shuf, s); // bring s2+s3 down to lane 0
    s = _mm_add_ss(s, hi64); // (s0+s1) + (s2+s3)
    _mm_cvtss_f32(s)
}

// --- Fused decode + dot product (allocation-free hot kernel) ---

/// `Σ_i centroids[idx_i] · rq[i]`, decoding `bits`-wide quantized indices
/// straight out of the packed byte stream — no intermediate `Vec`.
///
/// Dispatches to a runtime-detected AVX2 path (gather the centroids for 8 decoded
/// indices, then fused-multiply-add against `rq`) or a portable multiple-
/// accumulator scalar fallback. Both keep several accumulators in flight to break
/// the single-accumulator dependency chain, so they REASSOCIATE the sum — the
/// result differs from a strict position-ordered `.sum()` by a few ULPs. The
/// index *decode* stays exact (see [`decode8_indices`]); only the accumulation
/// order changes. This is the hottest function in HNSW search/build; recall is
/// verified preserved within HNSW run-to-run variance (e2e).
#[inline]
fn mse_dot_fused(data: &[u8], bits: u8, centroids: &[f32], rq: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: gated on runtime AVX2+FMA. The decoded gather indices are
            // masked to `0..2^bits <= centroids.len()` so the gather is in-bounds,
            // and `rq` is read only within full 8-lane groups (`8*g + 8 <= dims`).
            return unsafe { mse_dot_avx2(data, bits, centroids, rq) };
        }
    }
    mse_dot_scalar(data, bits, centroids, rq)
}

/// Decode the 8 quantized indices of group `g` (logical positions `8g..8g+8`)
/// from the `bits` bytes holding them, MSB-first — matching [`bit_pack`]. Valid
/// only for full 8-index groups (`8·bits` bits = `bits` whole bytes, so groups
/// are byte-aligned). Every returned value is masked into `0..2^bits`, so it is
/// always a safe `centroids` index regardless of the input bytes.
#[inline]
fn decode8_indices(data: &[u8], bits: u8, g: usize) -> [i32; 8] {
    match bits {
        2 => {
            let b0 = data[2 * g];
            let b1 = data[2 * g + 1];
            [
                ((b0 >> 6) & 3) as i32,
                ((b0 >> 4) & 3) as i32,
                ((b0 >> 2) & 3) as i32,
                (b0 & 3) as i32,
                ((b1 >> 6) & 3) as i32,
                ((b1 >> 4) & 3) as i32,
                ((b1 >> 2) & 3) as i32,
                (b1 & 3) as i32,
            ]
        }
        3 => {
            let b0 = data[3 * g];
            let b1 = data[3 * g + 1];
            let b2 = data[3 * g + 2];
            [
                ((b0 >> 5) & 7) as i32,
                ((b0 >> 2) & 7) as i32,
                ((((b0 & 3) << 1) | (b1 >> 7)) & 7) as i32,
                ((b1 >> 4) & 7) as i32,
                ((b1 >> 1) & 7) as i32,
                ((((b1 & 1) << 2) | (b2 >> 6)) & 7) as i32,
                ((b2 >> 3) & 7) as i32,
                (b2 & 7) as i32,
            ]
        }
        4 => {
            let b0 = data[4 * g];
            let b1 = data[4 * g + 1];
            let b2 = data[4 * g + 2];
            let b3 = data[4 * g + 3];
            [
                (b0 >> 4) as i32,
                (b0 & 0xF) as i32,
                (b1 >> 4) as i32,
                (b1 & 0xF) as i32,
                (b2 >> 4) as i32,
                (b2 & 0xF) as i32,
                (b3 >> 4) as i32,
                (b3 & 0xF) as i32,
            ]
        }
        _ => unreachable!("TurboQuant bits must be 2, 3, or 4"),
    }
}

/// Portable multiple-accumulator `mse_dot`: decode 8 indices per group and FMA
/// them into 8 independent lanes (so the optimizer keeps several in flight), then
/// a bit-walked tail for the `< 8` remainder. Reassociated — see [`mse_dot_fused`].
fn mse_dot_scalar(data: &[u8], bits: u8, centroids: &[f32], rq: &[f32]) -> f32 {
    let dims = rq.len();
    let groups = dims / 8;
    let mut a = [0.0f32; 8];
    for g in 0..groups {
        let idx = decode8_indices(data, bits, g);
        let base = 8 * g;
        for l in 0..8 {
            a[l] += centroids[idx[l] as usize] * rq[base + l];
        }
    }
    let mut acc = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
    let done = groups * 8;
    if done < dims {
        // Full groups consume `groups * bits` whole bytes, so the tail resumes on
        // a byte boundary and the bit-walker reads a clean sub-slice.
        acc += mse_dot_generic(&data[groups * bits as usize..], bits, centroids, &rq[done..]);
    }
    acc
}

/// AVX2+FMA `mse_dot`: the centroid codebook is tiny (`2^bits <= 16` entries), so
/// look it up with an in-register cross-lane permute (`permutevar8x32`) instead of
/// a memory gather — gather is microcoded and measured ~30% SLOWER here for a LUT
/// this small. Per group of 8: decode the indices, permute the codebook to get the
/// 8 centroids, then FMA against 8 `rq` values; two 256-bit accumulators in flight.
/// 4-bit needs both halves of the 16-entry codebook (two permutes blended on the
/// index's bit 3).
///
/// # Safety
/// Caller must ensure AVX2+FMA (see [`has_avx2_fma`]). All ops are register-only
/// except the `rq` loads, which stay within full 8-lane groups (`8*g + 8 <= dims`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mse_dot_avx2(data: &[u8], bits: u8, centroids: &[f32], rq: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let dims = rq.len();
    let groups = dims / 8;
    let rptr = rq.as_ptr();
    // Load the codebook into one (<=8 entries: 2-/3-bit) or two (16 entries:
    // 4-bit) registers via a zero-padded stack copy, so no out-of-bounds read of
    // the short `centroids` slice.
    let mut lut = [0.0f32; 16];
    lut[..centroids.len()].copy_from_slice(centroids);
    let four_bit = bits == 4;
    // SAFETY: see fn-level note. `lut` is a fixed 16-float array (two full
    // loadu_ps); the decoded-index array is 8×i32 = 32 bytes (one loadu_si256);
    // `rptr` loads stay within the full groups; permute/blend are register-only.
    unsafe {
        let lo = _mm256_loadu_ps(lut.as_ptr());
        let hi = _mm256_loadu_ps(lut.as_ptr().add(8));
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        // Permute the codebook by the (in-range) decoded indices in `iv`.
        let lookup = |iv: __m256i| -> __m256 {
            let clo = _mm256_permutevar8x32_ps(lo, iv);
            if four_bit {
                let chi = _mm256_permutevar8x32_ps(hi, iv); // permute masks index to &7
                // Select the high half where index bit 3 is set: shift bit 3 to the
                // sign bit and use it as the blend mask.
                let sel = _mm256_castsi256_ps(_mm256_slli_epi32::<28>(iv));
                _mm256_blendv_ps(clo, chi, sel)
            } else {
                clo
            }
        };

        let mut g = 0usize;
        while g + 2 <= groups {
            let i0 = decode8_indices(data, bits, g);
            let i1 = decode8_indices(data, bits, g + 1);
            let iv0 = _mm256_loadu_si256(i0.as_ptr().cast());
            let iv1 = _mm256_loadu_si256(i1.as_ptr().cast());
            acc0 = _mm256_fmadd_ps(lookup(iv0), _mm256_loadu_ps(rptr.add(8 * g)), acc0);
            acc1 = _mm256_fmadd_ps(lookup(iv1), _mm256_loadu_ps(rptr.add(8 * (g + 1))), acc1);
            g += 2;
        }
        while g < groups {
            let i0 = decode8_indices(data, bits, g);
            let iv0 = _mm256_loadu_si256(i0.as_ptr().cast());
            acc0 = _mm256_fmadd_ps(lookup(iv0), _mm256_loadu_ps(rptr.add(8 * g)), acc0);
            g += 1;
        }
        let mut acc = hsum256(_mm256_add_ps(acc0, acc1));
        let done = groups * 8;
        if done < dims {
            acc += mse_dot_generic(&data[groups * bits as usize..], bits, centroids, &rq[done..]);
        }
        acc
    }
}

/// Generic bit-width fused decode+dot. Mirrors [`bit_unpack`]'s MSB-first walk
/// exactly, accumulating in place. Used for the non-byte-aligned 3-bit case.
#[inline]
fn mse_dot_generic(data: &[u8], bits: u8, centroids: &[f32], rq: &[f32]) -> f32 {
    let bits = bits as usize;
    let mut acc = 0.0f32;
    let mut bit_pos = 0usize;
    for &rqi in rq {
        let mut val = 0u8;
        for _ in 0..bits {
            val <<= 1;
            if (data[bit_pos / 8] >> (7 - (bit_pos % 8))) & 1 == 1 {
                val |= 1;
            }
            bit_pos += 1;
        }
        acc += centroids[val as usize] * rqi;
    }
    acc
}

// --- Bit packing ---

fn bit_pack(indices: &[u8], bits: u8) -> Vec<u8> {
    let total_bits = indices.len() * bits as usize;
    let num_bytes = total_bits.div_ceil(8);
    let mut packed = vec![0u8; num_bytes];

    let mut bit_pos = 0usize;
    for &idx in indices {
        for b in (0..bits).rev() {
            if (idx >> b) & 1 == 1 {
                packed[bit_pos / 8] |= 1 << (7 - (bit_pos % 8));
            }
            bit_pos += 1;
        }
    }

    packed
}

fn bit_unpack(packed: &[u8], bits: u8, count: usize) -> Vec<u8> {
    let mut indices = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        let mut val = 0u8;
        for _ in 0..bits {
            val <<= 1;
            if (packed[bit_pos / 8] >> (7 - (bit_pos % 8))) & 1 == 1 {
                val |= 1;
            }
            bit_pos += 1;
        }
        indices.push(val);
    }
    indices
}

/// Decode `count` `bits`-wide packed indices into `out[i] = centroids[idx_i] *
/// scale`, reusing `out`'s buffer and never materializing the intermediate
/// index `Vec`. Walks the packed bytes MSB-first, exactly like [`bit_unpack`]
/// followed by the centroid lookup + scale. Used by
/// [`TurboQuantizer::prepare_stored_into`] on the build hot path.
#[inline]
fn unpack_scaled_centroids(
    data: &[u8],
    bits: u8,
    count: usize,
    centroids: &[f32],
    scale: f32,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.reserve(count);
    let bits = bits as usize;
    let mut bit_pos = 0usize;
    for _ in 0..count {
        let mut val = 0u8;
        for _ in 0..bits {
            val <<= 1;
            if (data[bit_pos / 8] >> (7 - (bit_pos % 8))) & 1 == 1 {
                val |= 1;
            }
            bit_pos += 1;
        }
        out.push(centroids[val as usize] * scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance;

    #[test]
    fn config_sizes() {
        let c2 = TurboQuantConfig::new(1536, 2);
        assert_eq!(c2.compressed_size(), 384);

        let c3 = TurboQuantConfig::new(1536, 3);
        assert_eq!(c3.compressed_size(), 576);

        let c4 = TurboQuantConfig::new(1536, 4);
        assert_eq!(c4.compressed_size(), 768);
    }

    #[test]
    fn bit_pack_roundtrip_2bit() {
        let indices: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let packed = bit_pack(&indices, 2);
        let unpacked = bit_unpack(&packed, 2, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn bit_pack_roundtrip_3bit() {
        let indices: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let packed = bit_pack(&indices, 3);
        let unpacked = bit_unpack(&packed, 3, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn bit_pack_roundtrip_4bit() {
        let indices: Vec<u8> = (0..16).collect();
        let packed = bit_pack(&indices, 4);
        let unpacked = bit_unpack(&packed, 4, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn beta_codebook_centroid_count() {
        let cb = compute_beta_codebook(64, 8);
        assert_eq!(cb.centroids.len(), 8);
        assert_eq!(cb.boundaries.len(), 7);
    }

    #[test]
    fn beta_codebook_centroids_span_range() {
        let cb = compute_beta_codebook(64, 4);
        assert!(cb.centroids.iter().any(|&c| c < 0.0));
        assert!(cb.centroids.iter().any(|&c| c > 0.0));
    }

    #[test]
    fn beta_codebook_boundaries_are_ordered() {
        let cb = compute_beta_codebook(128, 8);
        for w in cb.boundaries.windows(2) {
            assert!(w[0] < w[1], "boundaries not sorted: {:?}", cb.boundaries);
        }
    }

    #[test]
    fn compress_preserves_direction() {
        let dims = 32;
        let config = TurboQuantConfig::new(dims, 4);
        let quantizer = TurboQuantizer::new(config);

        let mut rng = rand::rng();
        let original: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();

        let compressed = quantizer.compress(&original);
        let decompressed = quantizer.decompress(&compressed);

        let cosine_sim = distance::dot_product(&original, &decompressed)
            / (distance::dot_product(&original, &original).sqrt()
                * distance::dot_product(&decompressed, &decompressed).sqrt());

        assert!(
            cosine_sim > 0.8,
            "cosine similarity too low: {cosine_sim}"
        );
    }

    #[test]
    fn norm_is_preserved() {
        let dims = 64;
        let config = TurboQuantConfig::new(dims, 3);
        let quantizer = TurboQuantizer::new(config);

        let mut rng = rand::rng();
        let original: Vec<f32> = (0..dims).map(|_| rng.random_range(-5.0..5.0)).collect();
        let original_norm = original.iter().map(|x| x * x).sum::<f32>().sqrt();

        let compressed = quantizer.compress(&original);
        assert!(
            (compressed.norm - original_norm).abs() < 1e-5,
            "norm mismatch: {} vs {}",
            compressed.norm,
            original_norm
        );
    }

    #[test]
    fn mse_estimate_correlates_with_true_inner_product() {
        // TurboQuant_mse's estimate is biased (no QJL unbiasing) but must track
        // the true inner product well enough to rank — check Spearman correlation.
        let dims = 64;
        let quantizer = TurboQuantizer::new(TurboQuantConfig::new(dims, 4));
        let mut rng = rand::rng();
        let query: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();

        let targets: Vec<Vec<f32>> = (0..200)
            .map(|_| (0..dims).map(|_| rng.random_range(-1.0f32..1.0)).collect())
            .collect();
        let mut exact: Vec<(u64, f32)> = targets
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, distance::dot_product(&query, t)))
            .collect();
        let mut est: Vec<(u64, f32)> = targets
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, quantizer.inner_product_estimate(&query, &quantizer.compress(t))))
            .collect();
        // Higher inner product = more similar, so sort descending.
        exact.sort_by(|a, b| b.1.total_cmp(&a.1));
        est.sort_by(|a, b| b.1.total_cmp(&a.1));
        let exact_order: Vec<u64> = exact.iter().map(|(i, _)| *i).collect();
        let est_order: Vec<u64> = est.iter().map(|(i, _)| *i).collect();
        let rho = crate::index::rank_correlation(&exact_order, &est_order);
        assert!(rho > 0.5, "mse estimate ranks poorly: rho={rho}");
    }

    #[test]
    fn l2_distance_estimate_reasonable() {
        let dims = 64;
        let config = TurboQuantConfig::new(dims, 4);
        let quantizer = TurboQuantizer::new(config);

        let mut rng = rand::rng();
        let query: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
        let target: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();

        let true_dist = distance::l2_squared(&query, &target);
        let compressed = quantizer.compress(&target);
        let estimated = quantizer.distance_estimate(&query, &compressed, distance::Metric::L2);

        let ratio = estimated / true_dist;
        assert!(
            ratio > 0.3 && ratio < 3.0,
            "L2 estimate off: true={true_dist}, est={estimated}, ratio={ratio}"
        );
    }

    #[test]
    fn compression_ratio() {
        let config = TurboQuantConfig::new(1536, 3);
        let original_size = 1536 * 4; // f32 = 4 bytes
        let compressed_size = config.total_size();
        let ratio = original_size as f32 / compressed_size as f32;
        assert!(ratio > 7.0, "compression ratio {ratio} should be >7x");
    }

    #[test]
    fn no_training_needed() {
        // TurboQuant codebook is analytical — no samples required.
        let config = TurboQuantConfig::new(128, 3);
        let _quantizer = TurboQuantizer::new(config);
    }

    #[test]
    fn compressed_vector_serialization_roundtrip() {
        let dims = 64;
        let config = TurboQuantConfig::new(dims, 4);
        let quantizer = TurboQuantizer::new(config);

        let mut rng = rand::rng();
        let original: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
        let compressed = quantizer.compress(&original);

        let mut buf = Vec::new();
        compressed.write_to(&mut buf).unwrap();
        let restored = CompressedVector::read_from(&mut &buf[..]).unwrap();

        assert_eq!(compressed.data(), restored.data());
        assert_eq!(compressed.norm, restored.norm);
    }

    #[test]
    fn quantizer_serialization_roundtrip() {
        let dims = 32;
        let config = TurboQuantConfig::new(dims, 3);
        let quantizer = TurboQuantizer::new(config);

        let mut buf = Vec::new();
        quantizer.write_to(&mut buf).unwrap();
        let restored = TurboQuantizer::read_from(&mut &buf[..]).unwrap();

        assert_eq!(quantizer.config.dimensions, restored.config.dimensions);
        assert_eq!(quantizer.config.bits, restored.config.bits);
        assert_eq!(quantizer.rotation_matrix, restored.rotation_matrix);
        assert_eq!(quantizer.codebook.centroids, restored.codebook.centroids);
        assert_eq!(quantizer.codebook.boundaries, restored.codebook.boundaries);
    }

    /// The fused kernel must match the materialize-then-zip reference to within
    /// floating-point reassociation error, for every bit width and for dims that
    /// are odd / not a multiple of 8 (exercising the tail paths). The multi-
    /// accumulator / AVX2-gather kernel reassociates the sum (so NOT bit-identical),
    /// but the *index decode* is exact (see `decode8_indices_matches_bit_unpack`);
    /// the tolerance is conditioning-aware (`Σ|term|`), far below the quantization
    /// noise floor, and a real decode/gather bug diverges by `O(magnitude)`.
    #[test]
    fn fused_kernel_matches_materialized() {
        let mut rng = rand::rng();
        for &dims in &[16usize, 30, 31, 32, 33, 64, 127, 384] {
            for &bits in &[2u8, 3, 4] {
                let quantizer = TurboQuantizer::new(TurboQuantConfig::new(dims as u32, bits));
                for _ in 0..25 {
                    let query: Vec<f32> =
                        (0..dims).map(|_| rng.random_range(-3.0..3.0)).collect();
                    let target: Vec<f32> =
                        (0..dims).map(|_| rng.random_range(-3.0..3.0)).collect();
                    let prepared = quantizer.prepare_query(&query);
                    let cv = quantizer.compress(&target);

                    // Reference: materialize the indices and sum directly.
                    let indices = bit_unpack(cv.data(), bits, dims);
                    let ref_ip: f32 = indices
                        .iter()
                        .zip(prepared.rq.iter())
                        .map(|(&i, &rqi)| quantizer.codebook.centroids[i as usize] * rqi)
                        .sum::<f32>()
                        * cv.norm;

                    // Conditioning bound: rounding error scales with Σ|term|.
                    let mag: f32 = indices
                        .iter()
                        .zip(prepared.rq.iter())
                        .map(|(&i, &rqi)| (quantizer.codebook.centroids[i as usize] * rqi).abs())
                        .sum::<f32>()
                        * cv.norm;
                    let tol = 1e-4 * mag + 1e-5;

                    let got = quantizer.inner_product_estimate_prepared(&prepared, &cv);
                    assert!(
                        (got - ref_ip).abs() <= tol,
                        "fused kernel diverged: dims={dims} bits={bits} got={got} \
                         ref={ref_ip} tol={tol}"
                    );
                }
            }
        }
    }

    /// The fused helpers must match the materialized helpers to within
    /// reassociation tolerance (they now use multiple accumulators).
    #[test]
    fn fused_helpers_match_materialized() {
        let mut rng = rand::rng();
        for &dims in &[1usize, 7, 8, 9, 15, 16, 17, 100] {
            // MSE: for each bit width, random indices + random rq.
            for &bits in &[2u8, 3, 4] {
                let max_idx = 1u8 << bits;
                let indices: Vec<u8> =
                    (0..dims).map(|_| rng.random_range(0..max_idx)).collect();
                let centroids: Vec<f32> =
                    (0..max_idx).map(|_| rng.random_range(-1.0..1.0)).collect();
                let rq: Vec<f32> = (0..dims).map(|_| rng.random_range(-2.0..2.0)).collect();
                let packed = bit_pack(&indices, bits);

                let reference: f32 = indices
                    .iter()
                    .zip(rq.iter())
                    .map(|(&i, &r)| centroids[i as usize] * r)
                    .sum();
                let mag: f32 = indices
                    .iter()
                    .zip(rq.iter())
                    .map(|(&i, &r)| (centroids[i as usize] * r).abs())
                    .sum();
                let fused = mse_dot_fused(&packed, bits, &centroids, &rq);
                assert!(
                    (fused - reference).abs() <= 1e-4 * mag + 1e-6,
                    "mse_dot_fused mismatch: dims={dims} bits={bits} fused={fused} ref={reference}"
                );
            }
        }
    }

    /// `decode8_indices` must reproduce `bit_unpack`'s indices EXACTLY for every
    /// full 8-index group — the decode is the part that must stay bit-exact (only
    /// the subsequent accumulation reassociates). Covers all bit widths.
    #[test]
    fn decode8_indices_matches_bit_unpack() {
        let mut rng = rand::rng();
        for &bits in &[2u8, 3, 4] {
            let max_idx = 1u8 << bits;
            let n = 8 * 6; // several full groups
            for _ in 0..50 {
                let indices: Vec<u8> = (0..n).map(|_| rng.random_range(0..max_idx)).collect();
                let packed = bit_pack(&indices, bits);
                for g in 0..(n / 8) {
                    let got = decode8_indices(&packed, bits, g);
                    for l in 0..8 {
                        assert_eq!(
                            got[l] as u8,
                            indices[8 * g + l],
                            "decode8 mismatch bits={bits} group={g} lane={l}"
                        );
                    }
                }
            }
        }
    }

    /// The fast `prepare_stored` (direct scaled-centroid decode) must match the
    /// original `prepare_query(&decompress(cv))` path (inverse rotation) to within
    /// the floating-point deviation of R·Rᵀ from identity. We assert on
    /// the resulting stored-vs-stored distance estimates across all metrics —
    /// that's the value the build path actually consumes.
    #[test]
    fn prepare_stored_matches_decompress_path() {
        let mut rng = rand::rng();
        for &dims in &[64usize, 128] {
            for &bits in &[2u8, 3, 4] {
                let q = TurboQuantizer::new(TurboQuantConfig::new(dims as u32, bits));
                for _ in 0..15 {
                    let a_vec: Vec<f32> =
                        (0..dims).map(|_| rng.random_range(-2.0..2.0)).collect();
                    let b_vec: Vec<f32> =
                        (0..dims).map(|_| rng.random_range(-2.0..2.0)).collect();
                    let a = q.compress(&a_vec);
                    let b = q.compress(&b_vec);

                    let old_prepared = q.prepare_query(&q.decompress(&a));
                    let new_prepared = q.prepare_stored(&a);

                    for metric in [
                        distance::Metric::DotProduct,
                        distance::Metric::Cosine,
                        distance::Metric::L2,
                    ] {
                        let old_d =
                            q.distance_estimate_prepared(&old_prepared, &b, metric);
                        let new_d =
                            q.distance_estimate_prepared(&new_prepared, &b, metric);
                        // `prepare_stored` equals `prepare_query(decompress)` only up
                        // to R's f32 orthonormality residual (R·Rᵀ ≈ I), which is an
                        // ABSOLUTE error on the inner product — it does NOT shrink when
                        // a DotProduct/Cosine distance happens to land near zero. A
                        // purely relative tolerance therefore false-fails on those
                        // near-zero ties (~3% of runs; reproduces identically on the
                        // pre-SIMD code). Require 2% relative agreement OR a small
                        // absolute floor (~10× the observed residual); a genuinely
                        // a wrong prepare_stored diverges by O(distance), far above the floor.
                        let diff = (old_d - new_d).abs();
                        assert!(
                            diff <= 0.02 * old_d.abs() + 5e-3,
                            "prepare_stored diverged: dims={dims} bits={bits} \
                             metric={metric:?} old={old_d} new={new_d} diff={diff}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn quantizer_roundtrip_preserves_distances() {
        let dims = 32;
        let config = TurboQuantConfig::new(dims, 3);
        let quantizer = TurboQuantizer::new(config);

        let mut buf = Vec::new();
        quantizer.write_to(&mut buf).unwrap();
        let restored = TurboQuantizer::read_from(&mut &buf[..]).unwrap();

        let mut rng = rand::rng();
        let query: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
        let target: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();

        let compressed = quantizer.compress(&target);
        let ip_original = quantizer.inner_product_estimate(&query, &compressed);
        let ip_restored = restored.inner_product_estimate(&query, &compressed);
        assert_eq!(ip_original, ip_restored);
    }

    /// The SIMD `dot` and the portable `dot_scalar` must both agree with a naive
    /// left-fold dot product to within floating-point reassociation error — they
    /// reassociate (and the AVX2 path fuses multiply-add), so they are not
    /// bit-identical to the naive sum, but must stay far inside the quantization
    /// noise floor. Covers lengths that exercise the 32-/8-wide and scalar-tail
    /// paths, plus the empty and sub-lane cases.
    #[test]
    fn dot_matches_naive_within_tolerance() {
        let mut rng = rand::rng();
        for &len in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 127, 384, 1536] {
            for _ in 0..20 {
                let a: Vec<f32> = (0..len).map(|_| rng.random_range(-3.0..3.0)).collect();
                let b: Vec<f32> = (0..len).map(|_| rng.random_range(-3.0..3.0)).collect();
                let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
                // Conditioning bound: rounding error scales with Σ|a_i·b_i|.
                let mag: f32 = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| x.abs() * y.abs())
                    .sum::<f32>()
                    .max(1e-6);
                let tol = 1e-3 * mag;
                let simd = dot(&a, &b);
                let scal = dot_scalar(&a, &b);
                assert!(
                    (simd - naive).abs() <= tol,
                    "dot diverged from naive: len={len} simd={simd} naive={naive} tol={tol}"
                );
                assert!(
                    (scal - naive).abs() <= tol,
                    "dot_scalar diverged from naive: len={len} scal={scal} naive={naive} tol={tol}"
                );
            }
        }
    }

    /// `prepare_stored_into` reuses a caller buffer; it must produce exactly the
    /// same `PreparedQuery` as the allocating `prepare_stored` (they share the
    /// code path), and must fully overwrite a pre-dirtied buffer (no stale tail).
    #[test]
    fn prepare_stored_into_matches_prepare_stored() {
        let mut rng = rand::rng();
        for &dims in &[64usize, 128, 384] {
            for &bits in &[2u8, 3, 4] {
                let q = TurboQuantizer::new(TurboQuantConfig::new(dims as u32, bits));
                // Pre-dirty with wrong lengths + sentinel values to prove a full
                // overwrite, then reuse the same buffer across iterations.
                let mut buf = PreparedQuery {
                    rq: vec![123.0; dims + 5],
                    q_norm: 999.0,
                };
                for _ in 0..10 {
                    let v: Vec<f32> = (0..dims).map(|_| rng.random_range(-2.0..2.0)).collect();
                    let cv = q.compress(&v);
                    let fresh = q.prepare_stored(&cv);
                    q.prepare_stored_into(&cv, &mut buf);
                    assert_eq!(fresh.rq, buf.rq, "rq mismatch dims={dims} bits={bits}");
                    assert_eq!(
                        fresh.q_norm.to_bits(),
                        buf.q_norm.to_bits(),
                        "q_norm mismatch dims={dims} bits={bits}"
                    );
                }
            }
        }
    }
}
