//! Single-writer guard for an LSM data directory.
//!
//! Two processes opening the same data dir at once corrupts the LSM: each
//! seeds its own in-memory `next_sst_id` and they collide on `NNNNNNNN.sst`
//! filenames (second writer's `File::create` truncates the first's), and both
//! append to / truncate the single shared `wal.log`. The damage is silent.
//!
//! This guard layers two protections, deliberately scoped:
//!
//! 1. **Advisory `flock` (same-host guard).** [`DataDirGuard::acquire`] takes a
//!    non-blocking exclusive `flock(2)` on `<data_dir>/LOCK`, held for the life
//!    of the guard and released by the kernel when the process exits — so a
//!    crash drops it automatically and there is no stale-lock problem (unlike a
//!    bare pid-file). This is correct and complete for a second opener on the
//!    SAME host, including another container sharing the host kernel.
//!
//! 2. **Owner-fence token (defence in depth).** `flock` is unreliable — and
//!    sometimes a silent no-op — on network/overlay filesystems (NFS &c.),
//!    exactly where a cross-host double-open is possible. So `acquire` also
//!    stamps a random `owner_id` into the LOCK file, and [`DataDirGuard::verify_owner`]
//!    re-reads it at every DESTRUCTIVE boundary (SST flush, compaction). A
//!    second process that got past a no-op flock will have stamped its own
//!    `owner_id`; we detect the mismatch and fail loud (poisoning the guard)
//!    BEFORE overwriting its SSTs / truncating its WAL.
//!
//! Layer 2 is NOT a cross-host exclusion guarantee — only the cluster/ownership
//! layer (leader election + write-boundary fencing) can provide that. It turns
//! silent corruption into a detectable, halting error on substrates where the
//! advisory lock does nothing.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{Error, Result};

const LOCK_FILE: &str = "LOCK";

/// Held for the lifetime of an open [`crate::lsm::LsmTree`]. Dropping it closes
/// the LOCK file, which releases the OS advisory lock.
pub struct DataDirGuard {
    /// Kept open purely to hold the `flock`; closing it releases the lock.
    _file: File,
    lock_path: PathBuf,
    /// Our fence token, stamped into the LOCK file at acquire time.
    owner_id: u128,
    /// Latched once a `verify_owner` mismatch is seen, so every subsequent
    /// write/flush fails fast without re-reading the file.
    fenced: AtomicBool,
}

impl DataDirGuard {
    /// Acquire the single-writer guard for `data_dir`. The directory must
    /// already exist. Returns [`Error::DataDirLocked`] if another live process
    /// on this host holds the lock.
    pub fn acquire(data_dir: &Path) -> Result<Self> {
        let lock_path = data_dir.join(LOCK_FILE);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;

        if !try_lock_exclusive(&file)? {
            // Held by a live process on this host. Surface whatever the current
            // holder stamped, to make the message actionable.
            let detail = read_holder_summary(&lock_path)
                .unwrap_or_else(|| "held by another process".to_string());
            return Err(Error::DataDirLocked(format!(
                "{} is locked ({detail}). Stop the server or other tool using \
                 this data directory before opening it.",
                lock_path.display(),
            )));
        }

        // We own the lock. Stamp our fence token + diagnostics over any stale
        // metadata left by a previous (now-dead) holder.
        let owner_id = random_owner_id();
        write_metadata(&file, owner_id)?;

        Ok(Self {
            _file: file,
            lock_path,
            owner_id,
            fenced: AtomicBool::new(false),
        })
    }

    /// Cheap poisoned-state check (atomic load only) for the commit hot path.
    pub fn check_fenced(&self) -> Result<()> {
        if self.fenced.load(Ordering::Acquire) {
            return Err(self.ownership_lost_err());
        }
        Ok(())
    }

    /// Confirm the on-disk owner token is still ours. Call at destructive
    /// boundaries (flush / compaction) BEFORE any irreversible write. On
    /// mismatch (a second process stamped the dir), latches the poison flag and
    /// returns [`Error::DataDirOwnershipLost`] so the caller aborts before it
    /// can clobber the other writer.
    pub fn verify_owner(&self) -> Result<()> {
        if self.fenced.load(Ordering::Acquire) {
            return Err(self.ownership_lost_err());
        }
        match read_owner_id(&self.lock_path) {
            Some(id) if id == self.owner_id => Ok(()),
            _ => {
                self.fenced.store(true, Ordering::Release);
                Err(self.ownership_lost_err())
            }
        }
    }

    fn ownership_lost_err(&self) -> Error {
        Error::DataDirOwnershipLost(format!(
            "{}: another process stamped a different owner token over this data \
             directory (the advisory lock did not exclude it — likely a \
             network/overlay filesystem). Writes are halted to avoid corrupting \
             the other writer.",
            self.lock_path.display(),
        ))
    }
}

/// 128-bit fence token, forced non-zero so a missing/unparseable LOCK file
/// (read as "no owner") never compares equal to a live token.
fn random_owner_id() -> u128 {
    rand::random::<u128>() | 1
}

/// Rewrite the LOCK file with our owner token plus diagnostics. We hold the
/// advisory lock, so no concurrent writer races this.
fn write_metadata(file: &File, owner_id: u128) -> Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let body = format!(
        "owner_id={owner_id:032x}\npid={}\nhost={}\n",
        std::process::id(),
        hostname(),
    );
    file.set_len(0)?;
    let mut f: &File = file;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(body.as_bytes())?;
    f.flush()?;
    file.sync_all()?;
    Ok(())
}

fn read_owner_id(path: &Path) -> Option<u128> {
    let s = std::fs::read_to_string(path).ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("owner_id=") {
            return u128::from_str_radix(rest.trim(), 16).ok();
        }
    }
    None
}

/// A one-line "pid N on host H" summary of the current holder, for the
/// refusal message. `None` if the LOCK file has no readable metadata.
fn read_holder_summary(path: &Path) -> Option<String> {
    let s = std::fs::read_to_string(path).ok()?;
    let mut pid = None;
    let mut host = None;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("pid=") {
            pid = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("host=") {
            host = Some(rest.trim().to_string());
        }
    }
    match (pid, host) {
        (Some(p), Some(h)) => Some(format!("held by pid {p} on host {h}")),
        (Some(p), None) => Some(format!("held by pid {p}")),
        _ => None,
    }
}

#[cfg(unix)]
fn try_lock_exclusive(file: &File) -> Result<bool> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    // SAFETY: `fd` is a valid open descriptor owned by `file` for the duration
    // of this call. `flock` only inspects it.
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if ret == 0 {
        return Ok(true);
    }
    let err = std::io::Error::last_os_error();
    match err.raw_os_error() {
        // The lock is held by someone else (EWOULDBLOCK == EAGAIN on Linux).
        Some(libc::EWOULDBLOCK) => Ok(false),
        _ => Err(Error::Io(err)),
    }
}

#[cfg(not(unix))]
fn try_lock_exclusive(_file: &File) -> Result<bool> {
    // No portable advisory lock here; layer 2 (the owner fence) still makes a
    // concurrent open detectable rather than silently destructive.
    Ok(true)
}

#[cfg(unix)]
fn hostname() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: writing up to `buf.len()` bytes into a buffer we own.
    let ret = unsafe {
        libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len())
    };
    if ret != 0 {
        return "unknown".to_string();
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

#[cfg(not(unix))]
fn hostname() -> String {
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[cfg(unix)]
    #[test]
    fn second_acquire_refused_while_first_held() {
        let dir = tempdir().unwrap();
        let g1 = DataDirGuard::acquire(dir.path()).unwrap();
        // `DataDirGuard` isn't `Debug`, so match rather than `unwrap_err`.
        let err = match DataDirGuard::acquire(dir.path()) {
            Err(e) => e,
            Ok(_) => panic!("second acquire should be refused while first held"),
        };
        assert!(matches!(err, Error::DataDirLocked(_)), "got {err:?}");
        drop(g1);
        // Lock released on drop → reacquire succeeds.
        let _g2 = DataDirGuard::acquire(dir.path()).unwrap();
    }

    #[test]
    fn verify_owner_detects_foreign_stamp() {
        let dir = tempdir().unwrap();
        let g = DataDirGuard::acquire(dir.path()).unwrap();
        assert!(g.verify_owner().is_ok());
        assert!(g.check_fenced().is_ok());

        // Simulate a second process (that got past a no-op flock) stamping its
        // own token over the LOCK file.
        std::fs::write(
            dir.path().join(LOCK_FILE),
            "owner_id=0000000000000000000000000000dead\npid=1\nhost=other\n",
        )
        .unwrap();

        let err = g.verify_owner().unwrap_err();
        assert!(matches!(err, Error::DataDirOwnershipLost(_)), "got {err:?}");
        // Poison is sticky: the cheap path now fails too.
        assert!(g.check_fenced().is_err());
    }

    #[test]
    fn holder_summary_reports_pid_and_host() {
        let dir = tempdir().unwrap();
        let _g = DataDirGuard::acquire(dir.path()).unwrap();
        let summary = read_holder_summary(&dir.path().join(LOCK_FILE)).unwrap();
        assert!(summary.contains("pid"), "got {summary}");
    }
}
