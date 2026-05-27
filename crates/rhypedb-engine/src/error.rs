use thiserror::Error;

pub type EngineResult<T> = Result<T, EngineError>;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("storage error: {0}")]
    Storage(#[from] rhypedb_storage::Error),

    #[error("schema error: {0}")]
    Schema(#[from] rhypedb_schema::SchemaError),

    #[error("type not found: {0}")]
    TypeNotFound(String),

    #[error("object not found: {type_name}:{object_id}")]
    ObjectNotFound { type_name: String, object_id: u64 },

    #[error("field not found: '{field}' on type '{type_name}'")]
    FieldNotFound { type_name: String, field: String },

    #[error("type mismatch for field '{field}': expected {expected}, got {got}")]
    TypeMismatch {
        field: String,
        expected: String,
        got: String,
    },

    #[error("unique constraint violated: {type_name}.{field} = {value}")]
    UniqueViolation {
        type_name: String,
        field: String,
        value: String,
    },

    #[error("delete denied: {type_name}:{object_id} is referenced by {referencing_type}.{referencing_field}")]
    DeleteDenied {
        type_name: String,
        object_id: u64,
        referencing_type: String,
        referencing_field: String,
    },

    #[error("write conflict")]
    WriteConflict,
}
