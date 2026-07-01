use thiserror::Error;

pub type QueryResult<T> = Result<T, QueryError>;

#[derive(Debug, Error)]
pub enum QueryError {
    #[error("parse error at position {pos}: {message}")]
    Parse { pos: usize, message: String },

    #[error("engine error: {0}")]
    Engine(#[from] rhypedb_engine::EngineError),

    #[error("type error: {0}")]
    Type(String),

    #[error("unknown function: {0}")]
    UnknownFunction(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A per-query resource cap (rows examined, traversal depth, result size, or
    /// the wall-clock budget) was exceeded — the query governor fails closed
    /// rather than let one request exhaust a small deployment VM. See
    /// [`crate::governor`].
    #[error("query resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
}
