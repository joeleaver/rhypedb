use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use bytes::{BufMut, Bytes, BytesMut};

use crate::{Error, Result};

/// Record type tags for the WAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RecordType {
    Put = 1,
    Delete = 2,
}

impl TryFrom<u8> for RecordType {
    type Error = Error;
    fn try_from(v: u8) -> Result<Self> {
        match v {
            1 => Ok(Self::Put),
            2 => Ok(Self::Delete),
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

impl WalRecord {
    fn encode(&self) -> Bytes {
        let total = HEADER_SIZE + self.key.len() + self.value.len();
        let mut buf = BytesMut::with_capacity(total);

        // Placeholder for CRC — filled after encoding the rest.
        buf.put_u32(0);
        buf.put_u64(self.version);
        buf.put_u8(self.record_type as u8);
        buf.put_u32(self.key.len() as u32);
        buf.put_u32(self.value.len() as u32);
        buf.put_slice(&self.key);
        buf.put_slice(&self.value);

        let crc = crc32fast::hash(&buf[4..]);
        buf[0..4].copy_from_slice(&crc.to_be_bytes());

        buf.freeze()
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

/// Append-only write-ahead log.
pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl Wal {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
        })
    }

    pub fn append(&mut self, record: &WalRecord) -> Result<()> {
        let encoded = record.encode();
        self.writer.write_all(&encoded)?;
        self.writer.flush()?;
        Ok(())
    }

    pub fn sync(&mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Replay all records from the WAL file. Used for crash recovery.
    pub fn replay(path: impl AsRef<Path>) -> Result<Vec<WalRecord>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        let mut records = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            match WalRecord::decode(&data[offset..]) {
                Ok((record, size)) => {
                    records.push(record);
                    offset += size;
                }
                Err(Error::WalCorrupted(_)) => {
                    // Truncated/corrupted tail — stop replay here.
                    // This is expected after a crash mid-write.
                    break;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(records)
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
                wal.append(&WalRecord {
                    record_type: RecordType::Put,
                    key: Bytes::from(format!("key{i}")),
                    value: Bytes::from(format!("val{i}")),
                    version: i,
                })
                .unwrap();
            }
            wal.sync().unwrap();
        }

        let records = Wal::replay(&wal_path).unwrap();
        assert_eq!(records.len(), 10);
        assert_eq!(records[0].key, Bytes::from("key0"));
        assert_eq!(records[9].version, 9);
    }

    #[test]
    fn wal_handles_truncated_tail() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(&WalRecord {
                record_type: RecordType::Put,
                key: Bytes::from_static(b"good"),
                value: Bytes::from_static(b"record"),
                version: 1,
            })
            .unwrap();
            wal.sync().unwrap();
        }

        // Append garbage to simulate a crash mid-write.
        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            file.write_all(b"truncated garbage").unwrap();
        }

        let records = Wal::replay(&wal_path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].key, Bytes::from_static(b"good"));
    }
}
