use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("WAL corrupted: {0}")]
    WalCorrupted(String),

    #[error("SST corrupted: {0}")]
    SstCorrupted(String),

    #[error("key not found")]
    NotFound,

    #[error("write conflict: key modified by concurrent transaction")]
    WriteConflict,

    #[error("transaction aborted")]
    TransactionAborted,
}
