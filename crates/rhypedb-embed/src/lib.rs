use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("embedding model error: {0}")]
    Model(String),

    #[error("unsupported model: {0}")]
    UnsupportedModel(String),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

pub type EmbedResult<T> = Result<T, EmbedError>;

/// Trait for text-to-vector encoding.
pub trait Embedder: Send + Sync {
    fn embed(&mut self, texts: &[&str]) -> EmbedResult<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
    fn model_name(&self) -> &str;
}

/// Embedder backed by fastembed (ONNX Runtime, CPU-only).
pub struct FastEmbedder {
    model: fastembed::TextEmbedding,
    dimensions: usize,
    model_name: String,
}

impl FastEmbedder {
    pub fn new(model_name: &str) -> EmbedResult<Self> {
        let model_type = match model_name {
            "all-MiniLM-L6-v2" => fastembed::EmbeddingModel::AllMiniLML6V2,
            "BAAI/bge-small-en-v1.5" | "bge-small-en-v1.5" => {
                fastembed::EmbeddingModel::BGESmallENV15
            }
            "BAAI/bge-base-en-v1.5" | "bge-base-en-v1.5" => {
                fastembed::EmbeddingModel::BGEBaseENV15
            }
            "BAAI/bge-large-en-v1.5" | "bge-large-en-v1.5" => {
                fastembed::EmbeddingModel::BGELargeENV15
            }
            _ => return Err(EmbedError::UnsupportedModel(model_name.into())),
        };

        let dimensions = match model_name {
            "all-MiniLM-L6-v2" => 384,
            "BAAI/bge-small-en-v1.5" | "bge-small-en-v1.5" => 384,
            "BAAI/bge-base-en-v1.5" | "bge-base-en-v1.5" => 768,
            "BAAI/bge-large-en-v1.5" | "bge-large-en-v1.5" => 1024,
            _ => 384,
        };

        let mut init_options = fastembed::InitOptions::default();
        init_options.model_name = model_type;
        init_options.show_download_progress = false;

        let model = fastembed::TextEmbedding::try_new(init_options)
        .map_err(|e| EmbedError::Model(e.to_string()))?;

        Ok(Self {
            model,
            dimensions,
            model_name: model_name.to_string(),
        })
    }

    pub fn with_default_model() -> EmbedResult<Self> {
        Self::new("all-MiniLM-L6-v2")
    }
}

impl Embedder for FastEmbedder {
    fn embed(&mut self, texts: &[&str]) -> EmbedResult<Vec<Vec<f32>>> {
        let documents: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let embeddings = self
            .model
            .embed(documents, None)
            .map_err(|e| EmbedError::Model(e.to_string()))?;

        for emb in &embeddings {
            if emb.len() != self.dimensions {
                return Err(EmbedError::DimensionMismatch {
                    expected: self.dimensions,
                    got: emb.len(),
                });
            }
        }

        Ok(embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_single_text() {
        let mut embedder = FastEmbedder::with_default_model().unwrap();
        let result = embedder.embed(&["hello world"]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 384);
    }

    #[test]
    fn embed_batch() {
        let mut embedder = FastEmbedder::with_default_model().unwrap();
        let texts = vec!["hello", "world", "foo bar"];
        let result = embedder.embed(&texts).unwrap();
        assert_eq!(result.len(), 3);
        for emb in &result {
            assert_eq!(emb.len(), 384);
        }
    }

    #[test]
    fn similar_texts_have_closer_embeddings() {
        let mut embedder = FastEmbedder::with_default_model().unwrap();
        let result = embedder
            .embed(&[
                "the cat sat on the mat",
                "a kitten rested on the rug",
                "quantum chromodynamics explains quark confinement",
            ])
            .unwrap();

        fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (norm_a * norm_b)
        }

        let sim_related = cosine_sim(&result[0], &result[1]);
        let sim_unrelated = cosine_sim(&result[0], &result[2]);

        assert!(
            sim_related > sim_unrelated,
            "related texts should be more similar: {sim_related} vs {sim_unrelated}"
        );
    }

    #[test]
    fn dimensions_correct() {
        let embedder = FastEmbedder::with_default_model().unwrap();
        assert_eq!(embedder.dimensions(), 384);
        assert_eq!(embedder.model_name(), "all-MiniLM-L6-v2");
    }

    #[test]
    fn unsupported_model_errors() {
        let result = FastEmbedder::new("nonexistent-model");
        assert!(matches!(result, Err(EmbedError::UnsupportedModel(_))));
    }
}
