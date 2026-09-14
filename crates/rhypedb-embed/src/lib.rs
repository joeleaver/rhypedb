use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("embedding model error: {0}")]
    Model(String),

    #[error("unsupported model: {0}")]
    UnsupportedModel(String),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// The model could not be loaded (e.g. a failed download or a corrupt
    /// cache entry). Distinct from `UnsupportedModel`: the model name was
    /// valid, but the load itself failed, so a caller may want to retry
    /// rather than treat this as a permanent configuration error.
    #[error("model unavailable: {0}")]
    Unavailable(String),
}

pub type EmbedResult<T> = Result<T, EmbedError>;

/// How the fastembed/ONNX embedder is built. All fields have defaults chosen
/// for an application that embeds in the background next to a UI.
///
/// This type has no fastembed dependency and is always available, regardless
/// of whether this crate's `fastembed` feature is enabled, so callers can
/// build and pass one through even when they don't (yet) depend on the
/// `FastEmbedder` implementation itself.
#[derive(Debug, Clone)]
pub struct EmbedOptions {
    /// Where model files are cached. `None` = fastembed's default
    /// (`FASTEMBED_CACHE_PATH` env var, else `./.fastembed_cache`).
    pub cache_dir: Option<PathBuf>,
    /// Token limit per text. Default 256 (MiniLM/BGE-small were trained at
    /// 256; attention memory grows with the square of this).
    pub max_length: usize,
    /// ONNX intra-op threads. Default `max(1, available_parallelism / 2)`.
    pub intra_threads: usize,
    /// Prefer the int8-quantized variant of the model when fastembed has one
    /// (`AllMiniLML6V2Q`, `BGESmallENV15Q`); otherwise the fp32 model.
    /// Default true.
    pub quantized: bool,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            cache_dir: None,
            max_length: 256,
            intra_threads: (cores / 2).max(1),
            quantized: true,
        }
    }
}

/// How the fastembed cross-encoder reranker is built. Like [`EmbedOptions`]
/// this has no fastembed dependency and is always available.
///
/// Each field's `None`/`false` means "consult the legacy environment
/// variable, else the default" — the env vars predate this struct and are
/// kept as fallbacks so an existing deployment keeps working:
/// `RHYPEDB_RERANKER_DIR` (a local model directory) and
/// `RHYPEDB_RERANKER_FP32` (the full-precision model).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RerankOptions {
    /// Load the reranker from this directory (`model.onnx` plus the four
    /// tokenizer files) instead of downloading it. `None` = the
    /// `RHYPEDB_RERANKER_DIR` env var if set, else download.
    pub model_dir: Option<PathBuf>,
    /// Use the full-precision `bge-reranker-base` (~1.1 GB) instead of the
    /// int8-quantized build (~280 MB, equivalent quality). `false` = the
    /// `RHYPEDB_RERANKER_FP32` env var if set, else quantized.
    pub fp32: bool,
    /// Where a downloaded reranker is cached. `None` = fastembed's default
    /// cache dir (shared with the embedding models).
    pub cache_dir: Option<PathBuf>,
}

impl RerankOptions {
    /// The effective model directory: the option, else the env var.
    pub fn effective_model_dir(&self) -> Option<PathBuf> {
        self.model_dir
            .clone()
            .or_else(|| std::env::var_os("RHYPEDB_RERANKER_DIR").map(PathBuf::from))
    }

    /// The effective precision choice: the option, else the env var.
    pub fn effective_fp32(&self) -> bool {
        self.fp32 || std::env::var_os("RHYPEDB_RERANKER_FP32").is_some()
    }
}

/// Trait for text-to-vector encoding.
pub trait Embedder: Send + Sync {
    fn embed(&mut self, texts: &[&str]) -> EmbedResult<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
    fn model_name(&self) -> &str;
}

/// Serializes fastembed model construction across this process.
///
/// hf-hub's local cache uses a per-blob lock file while a file is being
/// downloaded. When two `FastEmbedder`s are constructed concurrently in the
/// same process (even for different models, since they can share a cache
/// dir), the second `TextEmbedding::try_new` has been observed to lose that
/// race and fail outright with "Failed to retrieve model.onnx" rather than
/// wait for the first download to finish. Holding this mutex around
/// `TextEmbedding::try_new` ensures at most one model load is ever in flight
/// per process, so callers never hit that race.
#[cfg(feature = "fastembed")]
static MODEL_LOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Embedder backed by fastembed (ONNX Runtime, CPU-only).
#[cfg(feature = "fastembed")]
pub struct FastEmbedder {
    model: fastembed::TextEmbedding,
    dimensions: usize,
    model_name: String,
    options: EmbedOptions,
}

#[cfg(feature = "fastembed")]
impl FastEmbedder {
    pub fn new(model_name: &str) -> EmbedResult<Self> {
        Self::with_options(model_name, &EmbedOptions::default())
    }

    pub fn with_options(model_name: &str, options: &EmbedOptions) -> EmbedResult<Self> {
        let (model_type, dimensions) = Self::resolve_model(model_name, options.quantized)?;

        let mut init_options = fastembed::InitOptions::new(model_type)
            .with_max_length(options.max_length)
            .with_intra_threads(options.intra_threads)
            .with_show_download_progress(false);
        if let Some(cache_dir) = &options.cache_dir {
            init_options = init_options.with_cache_dir(cache_dir.clone());
        }

        let model = {
            // See `MODEL_LOAD`'s doc comment for why this is serialized.
            let _guard = MODEL_LOAD.lock().unwrap_or_else(|e| e.into_inner());
            fastembed::TextEmbedding::try_new(init_options)
                .map_err(|e| EmbedError::Unavailable(e.to_string()))?
        };

        Ok(Self {
            model,
            dimensions,
            model_name: model_name.to_string(),
            options: options.clone(),
        })
    }

    pub fn with_default_model() -> EmbedResult<Self> {
        Self::new("all-MiniLM-L6-v2")
    }

    pub fn options(&self) -> &EmbedOptions {
        &self.options
    }

    /// Map a supported model name (and its aliases) plus the `quantized`
    /// preference to the fastembed model variant and its embedding
    /// dimension.
    ///
    /// fastembed ships true int8-quantized ONNX builds only for
    /// `all-MiniLM-L6-v2` and `bge-small-en-v1.5` (`AllMiniLML6V2Q`,
    /// `BGESmallENV15Q`); the "Q" variants it has for the base/large BGE
    /// models are graph-optimized rather than int8-quantized. So for those
    /// two, `quantized` has no matching variant to prefer and this falls
    /// back to the fp32 model regardless of the option's value.
    fn resolve_model(
        model_name: &str,
        quantized: bool,
    ) -> EmbedResult<(fastembed::EmbeddingModel, usize)> {
        use fastembed::EmbeddingModel::*;

        let (fp32, int8, dimensions): (
            fastembed::EmbeddingModel,
            Option<fastembed::EmbeddingModel>,
            usize,
        ) = match model_name {
            "all-MiniLM-L6-v2" => (AllMiniLML6V2, Some(AllMiniLML6V2Q), 384),
            "BAAI/bge-small-en-v1.5" | "bge-small-en-v1.5" => {
                (BGESmallENV15, Some(BGESmallENV15Q), 384)
            }
            "BAAI/bge-base-en-v1.5" | "bge-base-en-v1.5" => (BGEBaseENV15, None, 768),
            "BAAI/bge-large-en-v1.5" | "bge-large-en-v1.5" => (BGELargeENV15, None, 1024),
            _ => return Err(EmbedError::UnsupportedModel(model_name.into())),
        };

        let model = if quantized {
            int8.unwrap_or(fp32)
        } else {
            fp32
        };
        Ok((model, dimensions))
    }
}

#[cfg(feature = "fastembed")]
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

/// A scored document from reranking.
#[derive(Debug, Clone)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}

/// Trait for cross-encoder reranking.
pub trait Reranker: Send + Sync {
    fn rerank(
        &mut self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> EmbedResult<Vec<RerankResult>>;
}

/// Cross-encoder reranker backed by fastembed.
#[cfg(feature = "fastembed")]
pub struct FastReranker {
    model: fastembed::TextRerank,
}

#[cfg(feature = "fastembed")]
impl FastReranker {
    /// [`Self::with_options`] with [`RerankOptions::default`] — i.e. the
    /// legacy env vars, else the int8-quantized download.
    pub fn new() -> EmbedResult<Self> {
        Self::with_options(&RerankOptions::default())
    }

    pub fn with_options(options: &RerankOptions) -> EmbedResult<Self> {
        // Reranker selection (each option falls back to its legacy env var,
        // see `RerankOptions`):
        //   - a model directory: load a user-supplied reranker (model.onnx +
        //     the four tokenizer files) from local disk. Used by bundled
        //     deployments that ship their own model and must not touch the
        //     network.
        //   - fp32: the full-precision bge-reranker-base (~1.1GB) — an
        //     escape hatch.
        //   - otherwise (DEFAULT): the int8-quantized bge-reranker-base
        //     (~280MB, equivalent quality, ~4x smaller and lower memory),
        //     downloaded and cached on first use.
        // Serialized under `MODEL_LOAD` like the embedder: a reranker load
        // racing an embedder load on the same hf-hub cache hit the same
        // per-blob lock failure.
        let _guard = MODEL_LOAD.lock().unwrap_or_else(|e| e.into_inner());
        let model = if let Some(dir) = options.effective_model_dir() {
            Self::load_from_dir(&dir)?
        } else if options.effective_fp32() {
            let mut init_options = fastembed::RerankInitOptions::default();
            init_options.show_download_progress = false;
            if let Some(cache_dir) = &options.cache_dir {
                init_options.cache_dir = cache_dir.clone();
            }
            fastembed::TextRerank::try_new(init_options)
                .map_err(|e| EmbedError::Unavailable(e.to_string()))?
        } else {
            Self::load_quantized_default(options.cache_dir.as_deref())?
        };

        Ok(Self { model })
    }

    /// Load a user-defined reranker (model.onnx + tokenizer files) from a dir.
    fn load_from_dir(dir: &std::path::Path) -> EmbedResult<fastembed::TextRerank> {
        let read = |name: &str| {
            std::fs::read(dir.join(name))
                .map_err(|e| EmbedError::Model(format!("reranker file '{name}': {e}")))
        };
        let tokenizer_files = fastembed::TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        };
        Self::build(
            fastembed::OnnxSource::File(dir.join("model.onnx")),
            tokenizer_files,
        )
    }

    /// Default reranker: the int8-quantized bge-reranker-base, fetched from the
    /// HuggingFace hub (Xenova/bge-reranker-base) and cached alongside the
    /// embedding models. ~280MB vs the 1.1GB fp32 build, equivalent quality.
    fn load_quantized_default(
        cache_dir: Option<&std::path::Path>,
    ) -> EmbedResult<fastembed::TextRerank> {
        use hf_hub::api::sync::ApiBuilder;
        let cache_dir = cache_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from(fastembed::get_cache_dir()));
        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .with_progress(false)
            .build()
            .map_err(|e| EmbedError::Model(format!("hf-hub init: {e}")))?;
        let repo = api.model("Xenova/bge-reranker-base".to_string());
        let fetch = |name: &str| {
            repo.get(name)
                .map_err(|e| EmbedError::Model(format!("download '{name}': {e}")))
        };
        let read = |name: &str| -> EmbedResult<Vec<u8>> {
            std::fs::read(fetch(name)?)
                .map_err(|e| EmbedError::Model(format!("read '{name}': {e}")))
        };
        let tokenizer_files = fastembed::TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        };
        Self::build(
            fastembed::OnnxSource::File(fetch("onnx/model_quantized.onnx")?),
            tokenizer_files,
        )
    }

    fn build(
        onnx: fastembed::OnnxSource,
        tokenizer_files: fastembed::TokenizerFiles,
    ) -> EmbedResult<fastembed::TextRerank> {
        let user_model = fastembed::UserDefinedRerankingModel::new(onnx, tokenizer_files);
        fastembed::TextRerank::try_new_from_user_defined(
            user_model,
            fastembed::RerankInitOptionsUserDefined::default(),
        )
        .map_err(|e| EmbedError::Model(e.to_string()))
    }
}

#[cfg(feature = "fastembed")]
impl Reranker for FastReranker {
    fn rerank(
        &mut self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> EmbedResult<Vec<RerankResult>> {
        let results = self
            .model
            .rerank(query, documents, false, None)
            .map_err(|e| EmbedError::Model(e.to_string()))?;

        let mut scored: Vec<RerankResult> = results
            .iter()
            .map(|r| RerankResult {
                index: r.index,
                score: r.score,
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored.truncate(top_k);
        Ok(scored)
    }
}

// These tests exercise the fastembed-backed impls (they download/load ONNX
// models), so they only build when the `fastembed` feature is on.
#[cfg(all(test, feature = "fastembed"))]
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

    #[test]
    fn reranker_scores_relevant_higher() {
        let mut reranker = FastReranker::new().unwrap();

        let query = "What is machine learning?";
        let documents = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "The weather forecast predicts rain tomorrow in Seattle.",
            "Deep neural networks are used in modern AI systems for pattern recognition.",
            "My favorite recipe for chocolate cake requires three eggs.",
        ];

        let results = reranker.rerank(query, &documents, 4).unwrap();

        // The ML-related documents (0 and 2) should score higher than unrelated ones.
        assert_eq!(results.len(), 4);
        let top_2_indices: Vec<usize> = results.iter().take(2).map(|r| r.index).collect();
        assert!(
            top_2_indices.contains(&0) && top_2_indices.contains(&2),
            "expected ML docs in top 2, got indices {:?}",
            top_2_indices
        );
    }

    #[test]
    fn reranker_respects_top_k() {
        let mut reranker = FastReranker::new().unwrap();

        let documents = ["doc1", "doc2", "doc3", "doc4", "doc5"];
        let results = reranker.rerank("query", &documents, 2).unwrap();
        assert_eq!(results.len(), 2);
    }
}

// Pure-Rust tests for `EmbedOptions` and `EmbedError`: no fastembed
// dependency, no network. These compile and run regardless of the
// `fastembed` feature, since the types themselves are not feature-gated.
#[cfg(test)]
mod options_tests {
    use super::*;

    #[test]
    fn default_options_have_expected_values() {
        let opts = EmbedOptions::default();
        assert_eq!(opts.cache_dir, None);
        assert_eq!(opts.max_length, 256);
        assert!(opts.intra_threads >= 1);
        assert!(opts.quantized);
    }

    #[test]
    fn rerank_options_fall_back_to_env_then_default() {
        // No option, no env → download the quantized default.
        let o = RerankOptions::default();
        // (env may be set on a developer machine; only assert the option path)
        let with_dir = RerankOptions { model_dir: Some(PathBuf::from("/m")), ..o.clone() };
        assert_eq!(with_dir.effective_model_dir(), Some(PathBuf::from("/m")));
        let with_fp32 = RerankOptions { fp32: true, ..o };
        assert!(with_fp32.effective_fp32());
    }

    #[test]
    fn unavailable_error_formats_message() {
        let err = EmbedError::Unavailable("model.onnx missing".to_string());
        assert_eq!(err.to_string(), "model unavailable: model.onnx missing");
    }
}

// Model-name resolution tests: these need the `fastembed` dependency for
// `fastembed::EmbeddingModel`, but exercise only `FastEmbedder::resolve_model`
// and the name-validation path of `with_options`, neither of which
// constructs a `TextEmbedding` — so no model download and no network access,
// unlike the tests in `mod tests` above.
#[cfg(all(test, feature = "fastembed"))]
mod no_network_tests {
    use super::*;
    use fastembed::EmbeddingModel;

    #[test]
    fn resolve_model_prefers_int8_when_available() {
        let (model, dims) = FastEmbedder::resolve_model("all-MiniLM-L6-v2", true).unwrap();
        assert!(matches!(model, EmbeddingModel::AllMiniLML6V2Q));
        assert_eq!(dims, 384);

        let (model, dims) = FastEmbedder::resolve_model("BAAI/bge-small-en-v1.5", true).unwrap();
        assert!(matches!(model, EmbeddingModel::BGESmallENV15Q));
        assert_eq!(dims, 384);
    }

    #[test]
    fn resolve_model_uses_fp32_when_quantized_is_false() {
        let (model, dims) = FastEmbedder::resolve_model("all-MiniLM-L6-v2", false).unwrap();
        assert!(matches!(model, EmbeddingModel::AllMiniLML6V2));
        assert_eq!(dims, 384);

        let (model, dims) = FastEmbedder::resolve_model("bge-small-en-v1.5", false).unwrap();
        assert!(matches!(model, EmbeddingModel::BGESmallENV15));
        assert_eq!(dims, 384);
    }

    #[test]
    fn resolve_model_falls_back_to_fp32_for_base_and_large() {
        // bge-base and bge-large have no true int8-quantized variant in
        // fastembed (their "Q" builds are graph-optimized, not
        // int8-quantized), so `quantized: true` still resolves to the fp32
        // model rather than erroring or silently picking the wrong build.
        let (model, dims) = FastEmbedder::resolve_model("BAAI/bge-base-en-v1.5", true).unwrap();
        assert!(matches!(model, EmbeddingModel::BGEBaseENV15));
        assert_eq!(dims, 768);

        let (model, dims) = FastEmbedder::resolve_model("bge-large-en-v1.5", true).unwrap();
        assert!(matches!(model, EmbeddingModel::BGELargeENV15));
        assert_eq!(dims, 1024);

        // quantized: false resolves the same way (there's nothing to fall
        // back from).
        let (model, _) = FastEmbedder::resolve_model("BAAI/bge-large-en-v1.5", false).unwrap();
        assert!(matches!(model, EmbeddingModel::BGELargeENV15));
    }

    #[test]
    fn resolve_model_rejects_unknown_names() {
        let result = FastEmbedder::resolve_model("nonexistent-model", true);
        assert!(matches!(result, Err(EmbedError::UnsupportedModel(_))));
    }

    #[test]
    fn with_options_rejects_unknown_model_before_touching_network() {
        // If this reached `TextEmbedding::try_new` it would try to download
        // a model, which would hang or fail in a network-less sandbox.
        // Asserting the error variant here proves `with_options` returns
        // from name resolution instead of ever reaching that call.
        let result = FastEmbedder::with_options("nonexistent-model", &EmbedOptions::default());
        assert!(matches!(result, Err(EmbedError::UnsupportedModel(_))));
    }
}
