use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use bytes::Bytes;

use crate::crash_inject::{self, Site};
use crate::{Error, Result};

/// Record type tags for the WAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RecordType {
    Put = 1,
    Delete = 2,
    /// Commit FOOTER, written after a transaction's data records by
    /// `append_txn`. Carries the committing version and the data-record count
    /// (in its value). It marks the batch boundary so replay applies a
    /// transaction all-or-nothing: a crash mid-write loses the footer, so the
    /// unterminated trailing batch is discarded rather than applied as a
    /// partial transaction. Never carried into the rebuilt memtable.
    Commit = 3,
}

impl TryFrom<u8> for RecordType {
    type Error = Error;
    fn try_from(v: u8) -> Result<Self> {
        match v {
            1 => Ok(Self::Put),
            2 => Ok(Self::Delete),
            3 => Ok(Self::Commit),
            _ => Err(Error::WalCorrupted(format!("unknown record type: {v}"))),
        }
    }
}

/// A single WAL record.
#[derive(Debug, Clone)]
pub struct WalRecord {
    pub record_type: RecordType,
    pub key: Bytes,
    pub value: Bytes,
    pub version: u64,
}

/// WAL record on-disk format:
///
/// ```text
/// [crc32: 4 bytes]
/// [version: 8 bytes]
/// [record_type: 1 byte]
/// [key_len: 4 bytes]
/// [value_len: 4 bytes]
/// [key: key_len bytes]
/// [value: value_len bytes]
/// ```
const HEADER_SIZE: usize = 4 + 8 + 1 + 4 + 4; // 21 bytes

/// Encode one on-disk record into `buf` (CRC over everything after the CRC
/// field). Shared by `WalRecord::encode_into` and the batch/txn writers so the
/// wire format has a single definition.
fn encode_record_into(
    buf: &mut Vec<u8>,
    record_type: RecordType,
    version: u64,
    key: &[u8],
    value: &[u8],
) {
    let start = buf.len();
    // Placeholder for CRC — filled after encoding the rest.
    buf.extend_from_slice(&0u32.to_be_bytes());
    buf.extend_from_slice(&version.to_be_bytes());
    buf.push(record_type as u8);
    buf.extend_from_slice(&(key.len() as u32).to_be_bytes());
    buf.extend_from_slice(&(value.len() as u32).to_be_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(value);

    let crc = crc32fast::hash(&buf[start + 4..]);
    buf[start..start + 4].copy_from_slice(&crc.to_be_bytes());
}

impl WalRecord {
    // Write side of an individual record — production writes go through
    // `encode_record_into` via `Wal::append_txn`; these remain for tests that
    // construct raw records (round-trip, legacy-file simulation).
    #[cfg(test)]
    fn encode(&self) -> Bytes {
        let mut buf = Vec::with_capacity(HEADER_SIZE + self.key.len() + self.value.len());
        self.encode_into(&mut buf);
        Bytes::from(buf)
    }

    #[cfg(test)]
    pub(crate) fn encode_into(&self, buf: &mut Vec<u8>) {
        encode_record_into(buf, self.record_type, self.version, &self.key, &self.value);
    }

    fn decode(data: &[u8]) -> Result<(Self, usize)> {
        if data.len() < HEADER_SIZE {
            return Err(Error::WalCorrupted("record too short".into()));
        }

        let stored_crc = u32::from_be_bytes(data[0..4].try_into().unwrap());
        let version = u64::from_be_bytes(data[4..12].try_into().unwrap());
        let record_type = RecordType::try_from(data[12])?;
        let key_len = u32::from_be_bytes(data[13..17].try_into().unwrap()) as usize;
        let value_len = u32::from_be_bytes(data[17..21].try_into().unwrap()) as usize;

        let total = HEADER_SIZE + key_len + value_len;
        if data.len() < total {
            return Err(Error::WalCorrupted("record truncated".into()));
        }

        let computed_crc = crc32fast::hash(&data[4..total]);
        if stored_crc != computed_crc {
            return Err(Error::WalCorrupted(format!(
                "CRC mismatch: stored={stored_crc:#x}, computed={computed_crc:#x}"
            )));
        }

        let key = Bytes::copy_from_slice(&data[HEADER_SIZE..HEADER_SIZE + key_len]);
        let value =
            Bytes::copy_from_slice(&data[HEADER_SIZE + key_len..HEADER_SIZE + key_len + value_len]);

        Ok((
            Self {
                record_type,
                key,
                value,
                version,
            },
            total,
        ))
    }
}

/// Magic + format byte written at the START of every WAL file created by this
/// binary. Its PRESENCE is what tells `replay` the file is commit-framed (so a
/// trailing un-footered batch is a torn transaction to DISCARD, not a legacy
/// per-record prefix to apply). A pre-framing (legacy) WAL has no magic and is
/// replayed with old per-record semantics. The trailing byte is a format
/// version for future evolution.
const WAL_MAGIC: &[u8; 8] = b"RHYPWAL\x01";

/// Append-only write-ahead log.
pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl Wal {
    /// Open the WAL for appending. A freshly-created (empty) file gets the
    /// framing magic header written immediately, so every WAL this binary
    /// originates is self-describing as framed. A non-empty file is left as-is
    /// (it already carries a magic, or is a legacy file that `LsmTree::open`
    /// converts after recovery).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let mut writer = BufWriter::new(file);
        if std::fs::metadata(&path)?.len() == 0 {
            writer.write_all(WAL_MAGIC)?;
            writer.flush()?;
        }
        Ok(Self { path, writer })
    }

    /// True if `path` is a pre-framing (legacy) WAL: it exists, is non-empty,
    /// and does NOT start with the framing magic. `LsmTree::open` uses this to
    /// decide whether to convert the WAL to framed (flush + recreate) after
    /// recovery, so no un-magic'd file ever receives framed appends.
    pub fn file_is_legacy(path: impl AsRef<Path>) -> Result<bool> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(false);
        }
        let mut f = File::open(path)?;
        let mut head = [0u8; 8];
        match f.read_exact(&mut head) {
            Ok(()) => Ok(&head != WAL_MAGIC),
            // Fewer than 8 bytes: empty or a torn header — empty is not legacy,
            // and a sub-8-byte non-empty file can't be a valid magic, so treat a
            // non-empty short file as legacy (it predates framing).
            Err(_) => Ok(std::fs::metadata(path)?.len() > 0),
        }
    }

    /// Append a whole transaction's writes as ONE framed batch: the data
    /// records (`Put` for `Some`, `Delete` for `None`), all stamped with
    /// `commit_version`, followed by a `Commit` FOOTER record. One
    /// `write_all` + flush for the entire transaction.
    ///
    /// Replay applies the batch all-or-nothing (see [`Wal::replay`]): the footer
    /// is written LAST, so a crash mid-`write_all` necessarily loses (or tears)
    /// the footer, and replay then discards the whole unterminated batch instead
    /// of resurrecting a partial transaction (primary row without its
    /// secondary-index/cover rows, etc.).
    pub fn append_txn(
        &mut self,
        writes: &[(Bytes, Option<Bytes>)],
        commit_version: u64,
    ) -> Result<()> {
        if writes.is_empty() {
            return Ok(());
        }
        let mut buf = Vec::with_capacity(
            writes
                .iter()
                .map(|(k, v)| HEADER_SIZE + k.len() + v.as_ref().map_or(0, |v| v.len()))
                .sum::<usize>()
                + HEADER_SIZE
                + 8,
        );
        for (key, value) in writes {
            let (record_type, val): (RecordType, &[u8]) = match value {
                Some(v) => (RecordType::Put, v),
                None => (RecordType::Delete, &[]),
            };
            encode_record_into(&mut buf, record_type, commit_version, key, val);
        }
        // Footer LAST: version = commit_version, empty key, value = data count.
        let count = (writes.len() as u64).to_be_bytes();
        encode_record_into(&mut buf, RecordType::Commit, commit_version, &[], &count);

        self.writer.write_all(&buf)?;
        // CRASH BOUNDARY: a batch < the BufWriter capacity sits in the
        // user-space buffer (lost on crash); a batch >= the capacity is written
        // straight through to the OS page cache (footer may already be durable,
        // so it can survive). Both are faithful to a real kill here — the oracle
        // predicts the outcome from the on-disk WAL framing, not the payload.
        crash_inject::hit(Site::WalAfterWriteBeforeFlush);
        self.writer.flush()?;
        Ok(())
    }

    pub fn sync(&mut self) -> Result<()> {
        self.writer.flush()?;
        // CRASH BOUNDARY: bytes flushed to the OS page cache but not fsync'd —
        // survive an in-process kill, lost only on a real power loss.
        crash_inject::hit(Site::WalAfterFlushBeforeFsync);
        self.writer.get_ref().sync_all()?;
        // CRASH BOUNDARY: the txn's WAL records are fully durable on the platter.
        crash_inject::hit(Site::WalAfterFsync);
        Ok(())
    }

    /// Crash-fuzz teardown ONLY: drop the `BufWriter`'s un-flushed user-space
    /// buffer WITHOUT flushing it, modelling a process that died with dirty
    /// buffers. The file is reopened (append) so a subsequent normal `Drop`
    /// closes a clean, empty-buffer writer; the old fd is closed without a flush
    /// via `BufWriter::into_parts` (which, unlike `into_inner`, never flushes).
    /// Bytes already `write`n by a completed `append_txn` remain in the OS page
    /// cache, as they would across a SIGKILL.
    #[cfg(feature = "crash-fuzz")]
    pub fn forget_unflushed_buffer(&mut self) {
        let fresh = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .expect("crash-fuzz: reopen WAL during simulated-crash teardown");
        let old = std::mem::replace(&mut self.writer, BufWriter::new(fresh));
        // `into_parts` returns the inner File + the buffered bytes WITHOUT
        // flushing; both drop here, closing the old fd and discarding the buffer.
        let _ = old.into_parts();
    }

    /// Replay all committed records from the WAL file. Used for crash recovery.
    ///
    /// Transactions are framed: `append_txn` writes N data records then a
    /// `Commit` footer. Replay buffers data records into a `pending` batch and
    /// commits them when their footer is decoded.
    ///
    /// Whether a trailing un-footered `pending` batch is APPLIED or DISCARDED is
    /// decided by the file's framing magic, NOT by whether a footer was seen:
    ///   * Framed file (starts with [`WAL_MAGIC`]): a trailing un-footered batch
    ///     is a torn (crash-mid-write) transaction — DISCARD it whole, so a
    ///     partial transaction is never resurrected. This holds even for the
    ///     FIRST/only transaction in the file (e.g. the first commit after a
    ///     flush re-created the WAL), which a "have I seen a footer?" heuristic
    ///     would mishandle.
    ///   * Legacy file (no magic, pre-framing): records were committed per-record,
    ///     so the valid prefix is applied (old semantics). `LsmTree::open`
    ///     converts such a file to framed after recovery, so framed appends are
    ///     never mixed into an un-magic'd file.
    pub fn replay(path: impl AsRef<Path>) -> Result<Vec<WalRecord>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        // Detect framing from the magic header; skip it before decoding records.
        let framed = data.len() >= WAL_MAGIC.len() && &data[..WAL_MAGIC.len()] == WAL_MAGIC;
        let mut offset = if framed { WAL_MAGIC.len() } else { 0 };

        let mut records = Vec::new();
        let mut pending: Vec<WalRecord> = Vec::new();

        while offset < data.len() {
            match WalRecord::decode(&data[offset..]) {
                Ok((record, size)) => {
                    offset += size;
                    if record.record_type == RecordType::Commit {
                        // Footer terminates the current batch. The footer value
                        // carries the data-record count; a mismatch means the
                        // batch is corrupt → drop it and stop (defense in depth).
                        let count = if record.value.len() == 8 {
                            u64::from_be_bytes(record.value[..8].try_into().unwrap()) as usize
                        } else {
                            usize::MAX
                        };
                        if count != pending.len() {
                            pending.clear();
                            break;
                        }
                        records.append(&mut pending);
                    } else {
                        pending.push(record);
                    }
                }
                Err(Error::WalCorrupted(_)) => {
                    // Truncated/corrupted tail — stop. Expected after a crash
                    // mid-write. The unterminated `pending` batch is handled below.
                    break;
                }
                Err(e) => return Err(e),
            }
        }

        if !pending.is_empty() && !framed {
            // Legacy file only: apply the valid per-record prefix. In a framed
            // file an unterminated trailing batch is a torn txn — discarded.
            records.append(&mut pending);
        }

        Ok(records)
    }

    /// Create a fresh empty WAL, replacing any existing file. Writes the framing
    /// magic header so the new file is self-describing as framed (every flush
    /// re-creates the WAL via this path). Used after flush to discard
    /// already-persisted records.
    pub fn create_fresh(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(WAL_MAGIC)?;
        writer.flush()?;
        Ok(Self { path, writer })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_roundtrip() {
        let record = WalRecord {
            record_type: RecordType::Put,
            key: Bytes::from_static(b"hello"),
            value: Bytes::from_static(b"world"),
            version: 42,
        };
        let encoded = record.encode();
        let (decoded, size) = WalRecord::decode(&encoded).unwrap();
        assert_eq!(size, encoded.len());
        assert_eq!(decoded.record_type, RecordType::Put);
        assert_eq!(decoded.key, record.key);
        assert_eq!(decoded.value, record.value);
        assert_eq!(decoded.version, 42);
    }

    #[test]
    fn corrupted_crc_detected() {
        let record = WalRecord {
            record_type: RecordType::Put,
            key: Bytes::from_static(b"key"),
            value: Bytes::from_static(b"val"),
            version: 1,
        };
        let mut encoded = record.encode().to_vec();
        encoded[5] ^= 0xFF; // corrupt a byte
        assert!(WalRecord::decode(&encoded).is_err());
    }

    #[test]
    fn wal_write_and_replay() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            for i in 0..10u64 {
                wal.append_txn(
                    &[(Bytes::from(format!("key{i}")), Some(Bytes::from(format!("val{i}"))))],
                    i + 1,
                )
                .unwrap();
            }
            wal.sync().unwrap();
        }

        let records = Wal::replay(&wal_path).unwrap();
        assert_eq!(records.len(), 10);
        assert_eq!(records[0].key, Bytes::from("key0"));
        assert_eq!(records[9].version, 10);
    }

    #[test]
    fn framed_txn_replay_applies_all_records() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append_txn(
                &[
                    (Bytes::from_static(b"a"), Some(Bytes::from_static(b"1"))),
                    (Bytes::from_static(b"b"), None), // tombstone
                    (Bytes::from_static(b"c"), Some(Bytes::from_static(b"3"))),
                ],
                7,
            )
            .unwrap();
            wal.sync().unwrap();
        }
        let records = Wal::replay(&wal_path).unwrap();
        // Footer is resolved internally; only the 3 data records come back.
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].key, Bytes::from_static(b"a"));
        assert_eq!(records[0].record_type, RecordType::Put);
        assert_eq!(records[1].record_type, RecordType::Delete);
        assert_eq!(records[2].version, 7);
    }

    // Commit framing: a crash that tears off (or never wrote) the footer of the
    // trailing transaction must drop that WHOLE transaction on replay, not apply
    // it as a partial prefix. Here two framed txns are written, then the last
    // footer is truncated away — replay keeps txn 1 and discards txn 2 entirely.
    #[test]
    fn torn_batch_replay_drops_whole_unterminated_txn() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append_txn(&[(Bytes::from_static(b"k1"), Some(Bytes::from_static(b"v1")))], 1)
                .unwrap();
            wal.append_txn(
                &[
                    (Bytes::from_static(b"k2"), Some(Bytes::from_static(b"v2"))),
                    (Bytes::from_static(b"k3"), Some(Bytes::from_static(b"v3"))),
                ],
                2,
            )
            .unwrap();
            wal.sync().unwrap();
        }
        // Truncate off the second txn's footer (Commit record: 21B header + 0 key
        // + 8B count = 29 bytes), leaving its data records present but unterminated.
        let len = std::fs::metadata(&wal_path).unwrap().len();
        let f = OpenOptions::new().write(true).open(&wal_path).unwrap();
        f.set_len(len - 29).unwrap();
        drop(f);

        let records = Wal::replay(&wal_path).unwrap();
        // Only txn 1 survives; txn 2 (k2, k3) is dropped wholesale.
        assert_eq!(records.len(), 1, "partial txn must not be applied");
        assert_eq!(records[0].key, Bytes::from_static(b"k1"));
    }

    // BLOCKER regression: a torn footer on the FIRST (and only) txn in a framed
    // file must still drop the whole txn — the framing magic, not "have I seen a
    // footer", drives the decision. (flush() re-creates the WAL after every
    // flush, so "first txn in the file" recurs constantly.)
    #[test]
    fn torn_first_txn_in_framed_file_drops_whole_txn() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        {
            let mut wal = Wal::open(&wal_path).unwrap(); // writes the magic header
            wal.append_txn(
                &[
                    (Bytes::from_static(b"k1"), Some(Bytes::from_static(b"v1"))),
                    (Bytes::from_static(b"k2"), Some(Bytes::from_static(b"v2"))),
                ],
                1,
            )
            .unwrap();
            wal.sync().unwrap();
        }
        // Truncate off the only footer (29 bytes), leaving magic + 2 data records.
        let len = std::fs::metadata(&wal_path).unwrap().len();
        let f = OpenOptions::new().write(true).open(&wal_path).unwrap();
        f.set_len(len - 29).unwrap();
        drop(f);

        let records = Wal::replay(&wal_path).unwrap();
        assert!(
            records.is_empty(),
            "torn first/only txn must be dropped wholesale, got {records:?}"
        );
    }

    // A pre-framing (legacy, un-magic'd) WAL replays with old per-record
    // semantics: the valid prefix is applied even past a torn tail.
    #[test]
    fn legacy_no_magic_applies_valid_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        // Write raw un-framed records (NO magic header) directly, then a torn tail.
        let mut raw = Vec::new();
        for i in 0..3u64 {
            WalRecord {
                record_type: RecordType::Put,
                key: Bytes::from(format!("k{i}")),
                value: Bytes::from_static(b"v"),
                version: i + 1,
            }
            .encode_into(&mut raw);
        }
        raw.extend_from_slice(b"torn garbage tail");
        std::fs::write(&wal_path, &raw).unwrap();

        assert!(Wal::file_is_legacy(&wal_path).unwrap());
        let records = Wal::replay(&wal_path).unwrap();
        assert_eq!(records.len(), 3, "legacy prefix must be applied");
        assert_eq!(records[2].key, Bytes::from_static(b"k2"));
    }

    #[test]
    fn wal_handles_truncated_tail() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append_txn(&[(Bytes::from_static(b"good"), Some(Bytes::from_static(b"record")))], 1)
                .unwrap();
            wal.sync().unwrap();
        }

        // Append garbage to simulate a crash mid-write of the NEXT txn.
        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            file.write_all(b"truncated garbage").unwrap();
        }

        let records = Wal::replay(&wal_path).unwrap();
        // The first (footered) txn survives; the torn trailing garbage is dropped.
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].key, Bytes::from_static(b"good"));
    }
}
