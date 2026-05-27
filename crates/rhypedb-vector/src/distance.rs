/// Distance metric for vector similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    L2,
    Cosine,
    DotProduct,
}

/// Squared L2 (Euclidean) distance.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

/// Dot product (higher = more similar).
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cosine distance = 1 - cosine_similarity.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = dot_product(a, b);
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    if denom == 0.0 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

/// Compute distance between two vectors using the given metric.
/// Lower values = more similar for all metrics.
pub fn compute_distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => l2_squared(a, b),
        Metric::Cosine => cosine_distance(a, b),
        Metric::DotProduct => -dot_product(a, b), // negate so lower = more similar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(l2_squared(&v, &v), 0.0);
    }

    #[test]
    fn l2_known_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_eq!(l2_squared(&a, &b), 25.0);
    }

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        assert!(cosine_distance(&v, &v) < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b), 32.0);
    }
}
