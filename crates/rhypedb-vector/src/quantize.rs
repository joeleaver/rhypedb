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

    pub fn qjl_size(&self) -> usize {
        (self.dimensions as usize).div_ceil(8)
    }

    pub fn total_size(&self) -> usize {
        // quantized data + QJL signs + norm (f32) + residual norm (f32)
        self.compressed_size() + self.qjl_size() + 4 + 4
    }
}

/// A trained TurboQuant quantizer for a specific vector collection.
///
/// Implements the TurboQuant pipeline from arXiv:2504.19874:
/// 1. Normalize to unit sphere, store norm
/// 2. Random orthogonal rotation (QR of Gaussian matrix)
/// 3. Lloyd-Max scalar quantization with Beta-distribution codebook
/// 4. QJL residual: project residual through Gaussian matrix, store signs
///
/// The combined estimator is unbiased: E[estimated inner product] = true inner product.
pub struct TurboQuantizer {
    config: TurboQuantConfig,
    rotation_matrix: Vec<f32>, // d x d, row-major — orthogonal via QR
    codebook: Codebook,
    qjl_matrix: Vec<f32>, // d x d, i.i.d. N(0,1)
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
        let qjl_matrix = generate_qjl_matrix(dims);
        let codebook = compute_beta_codebook(dims, num_centroids);

        Self {
            config,
            rotation_matrix,
            codebook,
            qjl_matrix,
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
        let mut reconstructed = Vec::with_capacity(dims);

        for &val in &rotated {
            let idx = self.codebook.quantize(val);
            indices.push(idx as u8);
            reconstructed.push(self.codebook.centroids[idx]);
        }

        // Step 4: QJL residual.
        let residual: Vec<f32> = rotated
            .iter()
            .zip(reconstructed.iter())
            .map(|(r, q)| r - q)
            .collect();
        let residual_norm = residual.iter().map(|x| x * x).sum::<f32>().sqrt();

        let projected = mat_vec_mul(&self.qjl_matrix, &residual, dims);
        let signs: Vec<bool> = projected.iter().map(|&v| v >= 0.0).collect();

        CompressedVector {
            data: bit_pack(&indices, self.config.bits),
            qjl_signs: pack_bools(&signs),
            norm,
            residual_norm,
        }
    }

    /// Decompress to approximate f32 values (MSE reconstruction only, no QJL correction).
    pub fn decompress(&self, compressed: &CompressedVector) -> Vec<f32> {
        let dims = self.config.dimensions as usize;

        let indices = bit_unpack(&compressed.data, self.config.bits, dims);
        let rotated: Vec<f32> = indices
            .iter()
            .map(|&i| self.codebook.centroids[i as usize])
            .collect();

        // Inverse rotation (transpose of orthogonal matrix).
        let unit = mat_vec_mul_transpose(&self.rotation_matrix, &rotated, dims);

        // Rescale by original norm.
        unit.iter().map(|x| x * compressed.norm).collect()
    }

    /// Project a full-precision query through both matrices ONCE so every
    /// candidate distance can reuse them. This is the key optimization: it turns
    /// the per-distance cost from two O(d²) matrix multiplies (the inverse
    /// rotation inside `decompress`, and the QJL `S^T·signs`) into two O(d) dot
    /// products, via the identities
    ///   ⟨q, R^T·cv⟩ = ⟨R·q, cv⟩   and   ⟨S^T·signs, q⟩ = ⟨signs, S·q⟩.
    pub fn prepare_query(&self, query: &[f32]) -> PreparedQuery {
        let dims = self.config.dimensions as usize;
        PreparedQuery {
            rq: mat_vec_mul(&self.rotation_matrix, query, dims),
            sq: mat_vec_mul(&self.qjl_matrix, query, dims),
            q_norm: query.iter().map(|x| x * x).sum::<f32>().sqrt(),
        }
    }

    /// Unbiased inner-product estimate reusing a [`PreparedQuery`]. O(d) per
    /// call — no matrix multiplies and no full `decompress`.
    pub fn inner_product_estimate_prepared(
        &self,
        prepared: &PreparedQuery,
        compressed: &CompressedVector,
    ) -> f32 {
        let dims = self.config.dimensions as usize;

        debug_assert_eq!(prepared.rq.len(), dims);
        debug_assert_eq!(prepared.sq.len(), dims);

        // MSE term: ⟨q, x̂_mse⟩ = ‖x‖ · ⟨R·q, cv⟩, where cv[i] is the centroid for
        // the quantized rotated coordinate i (decompress's pre-inverse-rotation
        // values). Avoids materializing x̂_mse (which would need the R^T multiply).
        //
        // The decode is fused into the dot product: instead of `bit_unpack`-ing
        // the indices into a heap `Vec<u8>` and then zipping, we walk the packed
        // bytes and accumulate `centroids[idx] · rq[i]` in one allocation-free
        // pass. Summation is strictly position-ordered, so the result is
        // bit-identical to the previous materialized path.
        let mse_dot =
            mse_dot_fused(&compressed.data, self.config.bits, &self.codebook.centroids, &prepared.rq)
                * compressed.norm;

        // QJL correction: ‖r‖ · ‖x‖ · √(π/2)/d · ⟨signs, S·q⟩ (signs are ±1, so a
        // signed sum over the pre-projected sq). Likewise fused: the sign bits are
        // applied to `sq` in place with no intermediate `Vec<bool>`.
        let correction_dot = correction_dot_fused(&compressed.qjl_signs, &prepared.sq);

        let qjl_scale = (std::f32::consts::FRAC_PI_2).sqrt() / dims as f32;
        let correction = compressed.residual_norm * compressed.norm * qjl_scale * correction_dot;

        mse_dot + correction
    }

    /// Compute the unbiased inner product estimate between a full-precision
    /// query and a compressed vector. Convenience wrapper that prepares the
    /// query then delegates; prefer preparing once and reusing
    /// [`Self::inner_product_estimate_prepared`] across many candidates.
    pub fn inner_product_estimate(
        &self,
        query: &[f32],
        compressed: &CompressedVector,
    ) -> f32 {
        let prepared = self.prepare_query(query);
        self.inner_product_estimate_prepared(&prepared, compressed)
    }

    /// Compute approximate distance using the unbiased estimator. Wrapper that
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
        self.codebook.write_to(w)?;
        write_f32_slice(w, &self.qjl_matrix)
    }

    pub fn read_from(r: &mut dyn io::Read) -> io::Result<Self> {
        let dimensions = read_u32(r)?;
        let bits = read_u8(r)?;
        let config = TurboQuantConfig::new(dimensions, bits);
        let dims = dimensions as usize;
        let rotation_matrix = read_f32_vec(r, dims * dims)?;
        let codebook = Codebook::read_from(r)?;
        let qjl_matrix = read_f32_vec(r, dims * dims)?;
        Ok(Self {
            config,
            rotation_matrix,
            codebook,
            qjl_matrix,
        })
    }
}

/// A query projected through the rotation and QJL matrices once, so that many
/// candidate distances can be estimated against it in O(d) each. Built by
/// [`TurboQuantizer::prepare_query`].
#[derive(Debug, Clone)]
pub struct PreparedQuery {
    /// R · query — the rotated query (reused by the MSE term).
    pub rq: Vec<f32>,
    /// S · query — the QJL-projected query (reused by the correction term).
    pub sq: Vec<f32>,
    /// ‖query‖ — cached for the cosine/L2 conversions.
    pub q_norm: f32,
}

/// A compressed vector with all components needed for the unbiased estimator.
#[derive(Debug, Clone)]
pub struct CompressedVector {
    pub data: Vec<u8>,      // bit-packed quantized indices
    pub qjl_signs: Vec<u8>, // packed sign bits from QJL projection
    pub norm: f32,           // original vector L2 norm
    pub residual_norm: f32,  // L2 norm of quantization residual (in rotated space)
}

impl CompressedVector {
    pub fn total_bytes(&self) -> usize {
        self.data.len() + self.qjl_signs.len() + 8 // + 2 f32 norms
    }

    pub fn write_to(&self, w: &mut dyn io::Write) -> io::Result<()> {
        write_byte_vec(w, &self.data)?;
        write_byte_vec(w, &self.qjl_signs)?;
        write_f32(w, self.norm)?;
        write_f32(w, self.residual_norm)
    }

    pub fn read_from(r: &mut dyn io::Read) -> io::Result<Self> {
        let data = read_byte_vec(r)?;
        let qjl_signs = read_byte_vec(r)?;
        let norm = read_f32(r)?;
        let residual_norm = read_f32(r)?;
        Ok(Self {
            data,
            qjl_signs,
            norm,
            residual_norm,
        })
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

/// Generate QJL projection matrix: i.i.d. N(0,1) entries.
fn generate_qjl_matrix(dims: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..dims * dims)
        .map(|_| sample_normal(&mut rng))
        .collect()
}

fn mat_vec_mul(matrix: &[f32], vector: &[f32], dims: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dims];
    for (i, res) in result.iter_mut().enumerate() {
        *res = (0..dims).map(|j| matrix[i * dims + j] * vector[j]).sum();
    }
    result
}

fn mat_vec_mul_transpose(matrix: &[f32], vector: &[f32], dims: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dims];
    for (i, res) in result.iter_mut().enumerate() {
        *res = (0..dims).map(|j| matrix[j * dims + i] * vector[j]).sum();
    }
    result
}

// --- Fused decode + dot product (allocation-free hot kernel) ---

/// `Σ_i centroids[idx_i] · rq[i]`, decoding `bits`-wide quantized indices
/// straight out of the packed byte stream — no intermediate `Vec`.
///
/// Byte-aligned widths (2 and 4 bits) get specialized loops that the optimizer
/// can keep in registers and unroll; the odd width (3) falls back to a generic
/// bit-walker. All three accumulate strictly in position order, so the sum is
/// bit-identical to `bit_unpack` followed by a zipped `.sum()`.
#[inline]
fn mse_dot_fused(data: &[u8], bits: u8, centroids: &[f32], rq: &[f32]) -> f32 {
    let dims = rq.len();
    match bits {
        4 => {
            // Each byte holds two indices: high nibble first (even position),
            // low nibble second (odd position).
            let pairs = dims / 2;
            let mut acc = 0.0f32;
            for j in 0..pairs {
                let b = data[j];
                acc += centroids[(b >> 4) as usize] * rq[2 * j];
                acc += centroids[(b & 0x0F) as usize] * rq[2 * j + 1];
            }
            if dims & 1 == 1 {
                // Trailing odd dimension lives in the high nibble of its byte.
                let b = data[pairs];
                acc += centroids[(b >> 4) as usize] * rq[dims - 1];
            }
            acc
        }
        2 => {
            // Each byte holds four indices, MSB-first.
            let quads = dims / 4;
            let mut acc = 0.0f32;
            for j in 0..quads {
                let b = data[j];
                acc += centroids[((b >> 6) & 0x03) as usize] * rq[4 * j];
                acc += centroids[((b >> 4) & 0x03) as usize] * rq[4 * j + 1];
                acc += centroids[((b >> 2) & 0x03) as usize] * rq[4 * j + 2];
                acc += centroids[(b & 0x03) as usize] * rq[4 * j + 3];
            }
            let done = quads * 4;
            if done < dims {
                let b = data[quads];
                for (r, &rqi) in rq[done..].iter().enumerate() {
                    let shift = 6 - 2 * r;
                    acc += centroids[((b >> shift) & 0x03) as usize] * rqi;
                }
            }
            acc
        }
        _ => mse_dot_generic(data, bits, centroids, rq),
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

/// `Σ_i sign_i · sq[i]` where `sign_i = +1` if the packed bit is set, else `-1`.
/// Walks the packed sign bytes MSB-first with no intermediate `Vec<bool>`.
/// Bit-identical to `unpack_bools` followed by a zipped `if b { s } else { -s }`.
#[inline]
fn correction_dot_fused(signs: &[u8], sq: &[f32]) -> f32 {
    let dims = sq.len();
    let full = dims / 8;
    let mut acc = 0.0f32;
    for (byte_i, &b) in signs[..full].iter().enumerate() {
        let base = byte_i * 8;
        acc += sign_apply((b >> 7) & 1, sq[base]);
        acc += sign_apply((b >> 6) & 1, sq[base + 1]);
        acc += sign_apply((b >> 5) & 1, sq[base + 2]);
        acc += sign_apply((b >> 4) & 1, sq[base + 3]);
        acc += sign_apply((b >> 3) & 1, sq[base + 4]);
        acc += sign_apply((b >> 2) & 1, sq[base + 5]);
        acc += sign_apply((b >> 1) & 1, sq[base + 6]);
        acc += sign_apply(b & 1, sq[base + 7]);
    }
    let rem = dims % 8;
    if rem > 0 {
        let b = signs[full];
        let base = full * 8;
        for (k, &s) in sq[base..].iter().enumerate() {
            acc += sign_apply((b >> (7 - k)) & 1, s);
        }
    }
    acc
}

/// `bit == 1 → +s`, `bit == 0 → -s`, branchlessly by XOR-ing the f32 sign bit.
/// Bit-identical to `if bit { s } else { -s }` (negation also just flips the
/// sign bit, including for ±0 and ±∞).
#[inline(always)]
fn sign_apply(bit: u8, s: f32) -> f32 {
    let flip = ((bit ^ 1) as u32) << 31;
    f32::from_bits(s.to_bits() ^ flip)
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

fn pack_bools(bools: &[bool]) -> Vec<u8> {
    let num_bytes = bools.len().div_ceil(8);
    let mut packed = vec![0u8; num_bytes];
    for (i, &b) in bools.iter().enumerate() {
        if b {
            packed[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    packed
}

/// Reference (materializing) sign unpack. The hot path no longer uses this —
/// [`correction_dot_fused`] decodes signs in place — so it is retained only as
/// the test oracle that the fused decoder is checked against.
#[cfg(test)]
fn unpack_bools(packed: &[u8], count: usize) -> Vec<bool> {
    let mut bools = Vec::with_capacity(count);
    for i in 0..count {
        let bit = (packed[i / 8] >> (7 - (i % 8))) & 1;
        bools.push(bit == 1);
    }
    bools
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance;

    #[test]
    fn config_sizes() {
        let c2 = TurboQuantConfig::new(1536, 2);
        assert_eq!(c2.compressed_size(), 384);
        assert_eq!(c2.qjl_size(), 192);

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
    fn bool_pack_roundtrip() {
        let bools = vec![true, false, true, true, false, false, true, false, true];
        let packed = pack_bools(&bools);
        let unpacked = unpack_bools(&packed, bools.len());
        assert_eq!(bools, unpacked);
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
    fn unbiased_inner_product_estimator() {
        // The key property: over many random vectors, the estimated inner product
        // should converge to the true inner product (unbiased).
        let dims = 64;
        let config = TurboQuantConfig::new(dims, 3);
        let quantizer = TurboQuantizer::new(config);

        let mut rng = rand::rng();
        let query: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();

        let n_trials = 200;
        let mut total_error = 0.0f32;

        for _ in 0..n_trials {
            let target: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
            let true_ip = distance::dot_product(&query, &target);
            let compressed = quantizer.compress(&target);
            let estimated_ip = quantizer.inner_product_estimate(&query, &compressed);
            total_error += estimated_ip - true_ip;
        }

        let mean_error = total_error / n_trials as f32;
        let true_ip_scale: f32 = {
            let target: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
            distance::dot_product(&query, &target).abs()
        };

        // Mean error should be small relative to typical inner product magnitude.
        // With 200 trials, bias should be clearly visible if present.
        let relative_bias = mean_error.abs() / (true_ip_scale + 1.0);
        assert!(
            relative_bias < 0.3,
            "estimator appears biased: mean_error={mean_error}, relative_bias={relative_bias}"
        );
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

        assert_eq!(compressed.data, restored.data);
        assert_eq!(compressed.qjl_signs, restored.qjl_signs);
        assert_eq!(compressed.norm, restored.norm);
        assert_eq!(compressed.residual_norm, restored.residual_norm);
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
        assert_eq!(quantizer.qjl_matrix, restored.qjl_matrix);
        assert_eq!(quantizer.codebook.centroids, restored.codebook.centroids);
        assert_eq!(quantizer.codebook.boundaries, restored.codebook.boundaries);
    }

    /// The fused, allocation-free kernel must produce *bit-identical* results to
    /// the original materialize-then-zip implementation, for every bit width and
    /// for dims that are odd / not a multiple of 8 (exercising the tail paths).
    /// Bit-identity is what guarantees recall is unchanged by the optimization.
    #[test]
    fn fused_kernel_bit_identical_to_materialized() {
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

                    // Reference: the exact computation the kernel used to perform.
                    let indices = bit_unpack(&cv.data, bits, dims);
                    let ref_mse: f32 = indices
                        .iter()
                        .zip(prepared.rq.iter())
                        .map(|(&i, &rqi)| quantizer.codebook.centroids[i as usize] * rqi)
                        .sum::<f32>()
                        * cv.norm;
                    let sgns = unpack_bools(&cv.qjl_signs, dims);
                    let ref_corr_dot: f32 = sgns
                        .iter()
                        .zip(prepared.sq.iter())
                        .map(|(&b, &s)| if b { s } else { -s })
                        .sum();
                    let qjl_scale = (std::f32::consts::FRAC_PI_2).sqrt() / dims as f32;
                    let ref_ip = ref_mse + cv.residual_norm * cv.norm * qjl_scale * ref_corr_dot;

                    let got = quantizer.inner_product_estimate_prepared(&prepared, &cv);
                    assert_eq!(
                        got.to_bits(),
                        ref_ip.to_bits(),
                        "fused kernel diverged: dims={dims} bits={bits} got={got} ref={ref_ip}"
                    );
                }
            }
        }
    }

    /// The fused helpers in isolation must match the materialized helpers exactly.
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
                let fused = mse_dot_fused(&packed, bits, &centroids, &rq);
                assert_eq!(
                    fused.to_bits(),
                    reference.to_bits(),
                    "mse_dot_fused mismatch: dims={dims} bits={bits}"
                );
            }

            // Correction: random signs + random sq.
            let bools: Vec<bool> = (0..dims).map(|_| rng.random_bool(0.5)).collect();
            let sq: Vec<f32> = (0..dims).map(|_| rng.random_range(-2.0..2.0)).collect();
            let packed_signs = pack_bools(&bools);
            let reference: f32 = bools
                .iter()
                .zip(sq.iter())
                .map(|(&b, &s)| if b { s } else { -s })
                .sum();
            let fused = correction_dot_fused(&packed_signs, &sq);
            assert_eq!(
                fused.to_bits(),
                reference.to_bits(),
                "correction_dot_fused mismatch: dims={dims}"
            );
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
}
