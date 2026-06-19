pub(crate) mod catalog;
pub mod database;
pub mod error;
pub mod logical;
pub mod object;
pub mod vectorizer;

pub use catalog::{
    AppliedMigration, ErrorPolicy, FieldTypeChangePair, Migration, MigrationContext,
    MigrationLogReport, MigrationPhase, MigrationReport, MigrationStatus, RenamePair,
};
pub use error::{CatalogError, EngineError, EngineResult};

// Re-export the storage-layer `CompareOp` so query callers can construct
// `filter_scan` arguments without taking a direct dependency on the storage
// crate. The enum lives in storage because that's where zone-map evaluation
// happens, but the only thing query code does with it is forward query AST
// comparison operators.
pub use rhypedb_storage::zone::CompareOp;
