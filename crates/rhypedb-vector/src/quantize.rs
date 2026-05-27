use rand::Rng;

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

    /// Bytes needed per compressed vector (quantized data only, no QJL).
    pub fn compressed_size(&self) -> usize {
        let total_bits = self.dimensions as usize * self.bits as usize;
        total_bits.div_ceil(8)
    }

    /// Bytes needed for the QJL residual (1 bit per dimension).
    pub fn qjl_size(&self) -> usize {
        (self.dimensions as usize).div_ceil(8)
    }

    /// Total bytes per compressed vector.
    pub fn total_size(&self) -> usize {
        self.compressed_size() + self.qjl_size()
    }
}

/// A trained TurboQuant quantizer for a specific vector collection.
///
/// Stores the rotation matrix and Lloyd-Max codebook calibrated to the
/// distribution of vectors in this collection.
pub struct TurboQuantizer {
    config: TurboQuantConfig,
    rotation_matrix: Vec<f32>, // dims x dims, row-major
    codebook: Vec<f32>,        // 2^bits centroids
    qjl_matrix: Vec<f32>,     // dims x dims, random projection for residuals
}

impl TurboQuantizer {
    /// Train a quantizer on a sample of vectors.
    pub fn train(config: TurboQuantConfig, samples: &[&[f32]]) -> Self {
        let dims = config.dimensions as usize;
        assert!(!samples.is_empty(), "need at least one sample");
        assert!(samples[0].len() == dims, "sample dimension mismatch");

        let rotation_matrix = generate_orthogonal_matrix(dims);
        let qjl_matrix = generate_random_projection(dims);

        // Rotate all samples and compute distribution stats for Lloyd-Max.
        let mut all_rotated_values = Vec::with_capacity(samples.len() * dims);
        for sample in samples {
            let rotated = matrix_vector_multiply(&rotation_matrix, sample, dims);
            all_rotated_values.extend_from_slice(&rotated);
        }

        let num_centroids = 1 << config.bits;
        let codebook = lloyd_max_quantize(&all_rotated_values, num_centroids);

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

        // Step 1: Orthogonal rotation.
        let rotated = matrix_vector_multiply(&self.rotation_matrix, vector, dims);

        // Step 2: Lloyd-Max scalar quantization.
        let mut quantized_indices = Vec::with_capacity(dims);
        let mut reconstructed = Vec::with_capacity(dims);

        for &val in &rotated {
            let idx = find_nearest_centroid(&self.codebook, val);
            quantized_indices.push(idx as u8);
            reconstructed.push(self.codebook[idx]);
        }

        // Step 3: QJL residual encoding (sign bits of projected residual).
        let residual: Vec<f32> = rotated
            .iter()
            .zip(reconstructed.iter())
            .map(|(r, q)| r - q)
            .collect();
        let projected = matrix_vector_multiply(&self.qjl_matrix, &residual, dims);
        let qjl_signs: Vec<bool> = projected.iter().map(|&v| v >= 0.0).collect();

        // Step 4: Bit-pack.
        let packed_data = bit_pack(&quantized_indices, self.config.bits);
        let packed_qjl = pack_bools(&qjl_signs);

        CompressedVector {
            data: packed_data,
            qjl: packed_qjl,
        }
    }

    /// Decompress a vector to approximate f32 values (for reranking).
    pub fn decompress(&self, compressed: &CompressedVector) -> Vec<f32> {
        let dims = self.config.dimensions as usize;

        let indices = bit_unpack(&compressed.data, self.config.bits, dims);
        let rotated: Vec<f32> = indices.iter().map(|&i| self.codebook[i as usize]).collect();

        // Inverse rotation (transpose of orthogonal matrix).
        matrix_vector_multiply_transpose(&self.rotation_matrix, &rotated, dims)
    }

    /// Compute approximate distance between a full-precision query and a
    /// compressed stored vector (asymmetric search).
    pub fn asymmetric_distance(
        &self,
        query: &[f32],
        compressed: &CompressedVector,
        metric: crate::distance::Metric,
    ) -> f32 {
        let approx = self.decompress(compressed);
        crate::distance::compute_distance(metric, query, &approx)
    }

    pub fn config(&self) -> &TurboQuantConfig {
        &self.config
    }
}

/// A compressed vector: quantized data + QJL residual bits.
#[derive(Debug, Clone)]
pub struct CompressedVector {
    pub data: Vec<u8>,
    pub qjl: Vec<u8>,
}

impl CompressedVector {
    pub fn total_bytes(&self) -> usize {
        self.data.len() + self.qjl.len()
    }
}

/// Generate a random orthogonal matrix using QR decomposition of a random matrix.
/// Uses a simplified Gram-Schmidt process.
fn generate_orthogonal_matrix(dims: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    let mut matrix = vec![0.0f32; dims * dims];

    // Fill with random values.
    for val in &mut matrix {
        *val = rng.random_range(-1.0..1.0);
    }

    // Gram-Schmidt orthogonalization.
    for i in 0..dims {
        // Subtract projections onto previous vectors.
        for j in 0..i {
            let dot = (0..dims)
                .map(|k| matrix[i * dims + k] * matrix[j * dims + k])
                .sum::<f32>();
            for k in 0..dims {
                matrix[i * dims + k] -= dot * matrix[j * dims + k];
            }
        }

        // Normalize.
        let norm = (0..dims)
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

/// Generate a random projection matrix for QJL residual encoding.
fn generate_random_projection(dims: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    let scale = 1.0 / (dims as f32).sqrt();
    let mut matrix = vec![0.0f32; dims * dims];
    for val in &mut matrix {
        *val = if rng.random_bool(0.5) { scale } else { -scale };
    }
    matrix
}

fn matrix_vector_multiply(matrix: &[f32], vector: &[f32], dims: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dims];
    for i in 0..dims {
        let mut sum = 0.0f32;
        for j in 0..dims {
            sum += matrix[i * dims + j] * vector[j];
        }
        result[i] = sum;
    }
    result
}

fn matrix_vector_multiply_transpose(matrix: &[f32], vector: &[f32], dims: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dims];
    for i in 0..dims {
        let mut sum = 0.0f32;
        for j in 0..dims {
            sum += matrix[j * dims + i] * vector[j];
        }
        result[i] = sum;
    }
    result
}

/// Lloyd-Max optimal scalar quantizer: find centroids that minimize MSE
/// for the given distribution of scalar values.
fn lloyd_max_quantize(values: &[f32], num_centroids: usize) -> Vec<f32> {
    if values.is_empty() {
        return vec![0.0; num_centroids];
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Initialize centroids uniformly across the value range.
    let min_val = sorted[0];
    let max_val = sorted[sorted.len() - 1];
    let mut centroids: Vec<f32> = (0..num_centroids)
        .map(|i| min_val + (max_val - min_val) * (i as f32 + 0.5) / num_centroids as f32)
        .collect();

    // Lloyd's algorithm: iterate assignment + update.
    for _ in 0..50 {
        // Assign each value to nearest centroid.
        let mut sums = vec![0.0f32; num_centroids];
        let mut counts = vec![0u32; num_centroids];

        for &val in values {
            let idx = find_nearest_centroid(&centroids, val);
            sums[idx] += val;
            counts[idx] += 1;
        }

        // Update centroids.
        let mut changed = false;
        for i in 0..num_centroids {
            if counts[i] > 0 {
                let new_centroid = sums[i] / counts[i] as f32;
                if (new_centroid - centroids[i]).abs() > 1e-8 {
                    changed = true;
                }
                centroids[i] = new_centroid;
            }
        }

        if !changed {
            break;
        }
    }

    centroids
}

fn find_nearest_centroid(centroids: &[f32], value: f32) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (value - **a).abs();
            let db = (value - **b).abs();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap()
        .0
}

/// Pack quantized indices (each `bits` wide) into bytes.
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

/// Unpack quantized indices from packed bytes.
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

/// Pack boolean values (1 bit each) into bytes.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn config_sizes() {
        let c2 = TurboQuantConfig::new(1536, 2);
        assert_eq!(c2.compressed_size(), 384); // 1536 * 2 / 8
        assert_eq!(c2.qjl_size(), 192); // 1536 / 8

        let c3 = TurboQuantConfig::new(1536, 3);
        assert_eq!(c3.compressed_size(), 576); // 1536 * 3 / 8

        let c4 = TurboQuantConfig::new(1536, 4);
        assert_eq!(c4.compressed_size(), 768); // 1536 * 4 / 8
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
    fn lloyd_max_produces_correct_centroid_count() {
        let values: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let centroids = lloyd_max_quantize(&values, 8);
        assert_eq!(centroids.len(), 8);
    }

    #[test]
    fn lloyd_max_centroids_are_sorted_roughly() {
        let values: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let centroids = lloyd_max_quantize(&values, 4);
        // Centroids should roughly span the range.
        assert!(centroids.iter().any(|&c| c < 0.3));
        assert!(centroids.iter().any(|&c| c > 0.7));
    }

    #[test]
    fn compress_decompress_preserves_direction() {
        let dims = 32;
        let config = TurboQuantConfig::new(dims, 4);

        // Generate random vectors.
        let mut rng = rand::rng();
        let samples: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        let sample_refs: Vec<&[f32]> = samples.iter().map(|v| v.as_slice()).collect();

        let quantizer = TurboQuantizer::train(config, &sample_refs);

        // Compress and decompress a vector.
        let original = &samples[0];
        let compressed = quantizer.compress(original);
        let decompressed = quantizer.decompress(&compressed);

        // The decompressed vector should be in roughly the same direction.
        let dot: f32 = original
            .iter()
            .zip(decompressed.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_orig: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_decomp: f32 = decompressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine_sim = dot / (norm_orig * norm_decomp);

        // At 4-bit with 32 dims, should be quite good.
        assert!(cosine_sim > 0.8, "cosine similarity too low: {cosine_sim}");
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
    fn asymmetric_distance_reasonable() {
        let dims = 64;
        let config = TurboQuantConfig::new(dims, 4);

        let mut rng = rand::rng();
        let samples: Vec<Vec<f32>> = (0..200)
            .map(|_| (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        let sample_refs: Vec<&[f32]> = samples.iter().map(|v| v.as_slice()).collect();

        let quantizer = TurboQuantizer::train(config, &sample_refs);

        let query = &samples[0];
        let similar = &samples[1];

        let compressed_similar = quantizer.compress(similar);

        let approx_dist =
            quantizer.asymmetric_distance(query, &compressed_similar, Metric::L2);
        let exact_dist = crate::distance::l2_squared(query, similar);

        // Approximate distance should be in the right ballpark.
        let ratio = approx_dist / exact_dist;
        assert!(
            ratio > 0.3 && ratio < 3.0,
            "distance ratio {ratio} out of range (approx={approx_dist}, exact={exact_dist})"
        );
    }
}
