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

    /// The data dir is already opened by another live process on this host
    /// (advisory `flock` held). Refused at open to prevent two writers from
    /// corrupting one LSM (colliding SST ids + a shared, truncated WAL).
    #[error("data directory is locked: {0}")]
    DataDirLocked(String),

    /// A second process stamped a different owner token over this data dir
    /// after we opened it — i.e. the advisory lock did not exclude it (likely a
    /// network/overlay filesystem where `flock` is a no-op). Writes are halted
    /// to avoid silently overwriting the other writer's data.
    #[error("data directory ownership lost: {0}")]
    DataDirOwnershipLost(String),
}
