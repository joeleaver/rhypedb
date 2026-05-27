use thiserror::Error;

pub type SchemaResult<T> = Result<T, SchemaError>;

#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("parse error at line {line}, col {col}: {message}")]
    Parse {
        line: usize,
        col: usize,
        message: String,
    },

    #[error("unknown type: {0}")]
    UnknownType(String),

    #[error("duplicate type name: {0}")]
    DuplicateType(String),

    #[error("duplicate field '{field}' in type '{type_name}'")]
    DuplicateField { type_name: String, field: String },

    #[error("invalid @inverse: {0}")]
    InvalidInverse(String),

    #[error("invalid @on_delete policy: {0}")]
    InvalidOnDelete(String),

    #[error("validation error: {0}")]
    Validation(String),
}
