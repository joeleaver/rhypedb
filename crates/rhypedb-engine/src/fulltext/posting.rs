//! Posting payload codec + per-document term map.
//!
//! One posting row per (field, term, object):
//!
//! ```text
//! varint(doc_len) varint(tf) [varint(pos₀) varint(pos₁ − pos₀) … ]   // positions on
//! varint(doc_len) varint(tf)                                          // positions off
//! ```
//!
//! `doc_len` (the document's token count) rides on every posting so BM25's
//! length normalization needs no second lookup per candidate. Positions are
//! delta-encoded, ascending. Varints are unsigned LEB128.

use std::collections::BTreeMap;

use bytes::Bytes;

use super::analyzer::Token;

/// A decoded posting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Posting {
    /// Token count of the whole field value the posting belongs to.
    pub doc_len: u32,
    /// Occurrences of the term in that value.
    pub tf: u32,
    /// Ascending term positions; empty when the field stores no positions.
    pub positions: Vec<u32>,
}

/// Why a posting payload could not be decoded — always corruption (the
/// codec never produces these shapes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostingError {
    /// Ran out of bytes mid-varint or before the mandatory fields.
    Truncated,
    /// A varint did not fit in 32 bits / exceeded 5 bytes.
    Overflow,
    /// `tf == 0`, or a position delta list whose length disagrees with `tf`.
    Inconsistent,
    /// Bytes remain after the last expected field.
    TrailingBytes,
}

impl std::fmt::Display for PostingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "truncated posting payload"),
            Self::Overflow => write!(f, "posting varint overflow"),
            Self::Inconsistent => write!(f, "posting tf/positions disagree"),
            Self::TrailingBytes => write!(f, "trailing bytes after posting payload"),
        }
    }
}

/// Group an analyzed token stream by term: term → ascending positions. The
/// map is ordered so the write path emits postings in key order.
pub fn term_map(tokens: &[Token]) -> BTreeMap<&str, Vec<u32>> {
    let mut map: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
    for t in tokens {
        map.entry(t.term.as_str()).or_default().push(t.position);
    }
    map
}

/// Encode one posting. `positions` MUST be non-empty and ascending (it is
/// the term's occurrence list from [`term_map`]); `tf` is its length.
pub fn encode_posting(doc_len: u32, positions: &[u32], store_positions: bool) -> Bytes {
    debug_assert!(!positions.is_empty());
    let mut out = Vec::with_capacity(10 + if store_positions { positions.len() * 2 } else { 0 });
    put_varint(&mut out, doc_len);
    put_varint(&mut out, positions.len() as u32);
    if store_positions {
        let mut prev = 0u32;
        for (i, &p) in positions.iter().enumerate() {
            let delta = if i == 0 { p } else { p - prev };
            put_varint(&mut out, delta);
            prev = p;
        }
    }
    Bytes::from(out)
}

/// Decode one posting payload. A payload holding only `doc_len tf` decodes
/// with empty positions (a positions-off field); anything else must carry
/// exactly `tf` deltas and nothing more.
pub fn decode_posting(bytes: &[u8]) -> Result<Posting, PostingError> {
    let mut cur = bytes;
    let doc_len = get_varint(&mut cur)?;
    let tf = get_varint(&mut cur)?;
    if tf == 0 {
        return Err(PostingError::Inconsistent);
    }
    if cur.is_empty() {
        return Ok(Posting {
            doc_len,
            tf,
            positions: Vec::new(),
        });
    }
    let mut positions = Vec::with_capacity(tf as usize);
    let mut prev = 0u32;
    for i in 0..tf {
        let delta = get_varint(&mut cur)?;
        let p = if i == 0 {
            delta
        } else {
            // A zero delta would mean a duplicate position — the analyzer
            // never emits one, so treat it as corruption like an overflow.
            if delta == 0 {
                return Err(PostingError::Inconsistent);
            }
            prev.checked_add(delta).ok_or(PostingError::Overflow)?
        };
        positions.push(p);
        prev = p;
    }
    if !cur.is_empty() {
        return Err(PostingError::TrailingBytes);
    }
    Ok(Posting {
        doc_len,
        tf,
        positions,
    })
}

/// Decode only `(doc_len, tf)` — what BM25 needs for a non-phrase term —
/// without materializing the position list (the per-posting allocation
/// that dominates a high-df scan). Validates the header the same way as
/// [`decode_posting`]; the position bytes, if any, are left unread.
pub fn decode_posting_header(bytes: &[u8]) -> Result<(u32, u32), PostingError> {
    let mut cur = bytes;
    let doc_len = get_varint(&mut cur)?;
    let tf = get_varint(&mut cur)?;
    if tf == 0 {
        return Err(PostingError::Inconsistent);
    }
    Ok((doc_len, tf))
}

/// Unsigned LEB128 (u32).
pub fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Unsigned LEB128 (u32); advances `cur` past the varint.
pub fn get_varint(cur: &mut &[u8]) -> Result<u32, PostingError> {
    let mut result: u32 = 0;
    for i in 0..5 {
        let Some(&b) = cur.first() else {
            return Err(PostingError::Truncated);
        };
        *cur = &cur[1..];
        let payload = (b & 0x7f) as u32;
        // The 5th byte may only carry 4 bits.
        if i == 4 && payload > 0x0f {
            return Err(PostingError::Overflow);
        }
        result |= payload << (7 * i);
        if b & 0x80 == 0 {
            return Ok(result);
        }
    }
    Err(PostingError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fulltext::Analyzer;

    #[test]
    fn varint_round_trips_boundaries() {
        for v in [0u32, 1, 127, 128, 255, 16_383, 16_384, 1 << 21, (1 << 28) - 1, 1 << 28, u32::MAX] {
            let mut buf = Vec::new();
            put_varint(&mut buf, v);
            let mut cur = buf.as_slice();
            assert_eq!(get_varint(&mut cur).unwrap(), v);
            assert!(cur.is_empty());
        }
        // Truncated + overflowing shapes.
        assert_eq!(get_varint(&mut &[0x80u8][..]), Err(PostingError::Truncated));
        assert_eq!(get_varint(&mut &[][..]), Err(PostingError::Truncated));
        assert_eq!(
            get_varint(&mut &[0xff, 0xff, 0xff, 0xff, 0xff, 0x01][..]),
            Err(PostingError::Overflow)
        );
        // 5th byte carrying more than 4 payload bits overflows u32.
        assert_eq!(
            get_varint(&mut &[0xff, 0xff, 0xff, 0xff, 0x10][..]),
            Err(PostingError::Overflow)
        );
    }

    #[test]
    fn posting_round_trips_with_and_without_positions() {
        let positions = vec![3u32, 4, 10, 1_000_000];
        let with = encode_posting(42, &positions, true);
        assert_eq!(
            decode_posting(&with).unwrap(),
            Posting {
                doc_len: 42,
                tf: 4,
                positions: positions.clone()
            }
        );
        let without = encode_posting(42, &positions, false);
        assert_eq!(without.len(), 2);
        assert_eq!(
            decode_posting(&without).unwrap(),
            Posting {
                doc_len: 42,
                tf: 4,
                positions: vec![]
            }
        );
        // Position 0 is a legal first position.
        let z = encode_posting(1, &[0], true);
        assert_eq!(decode_posting(&z).unwrap().positions, vec![0]);
    }

    #[test]
    fn header_decode_matches_full_decode() {
        let full = encode_posting(9, &[1, 5, 6], true);
        assert_eq!(decode_posting_header(&full).unwrap(), (9, 3));
        let no_pos = encode_posting(9, &[1, 5, 6], false);
        assert_eq!(decode_posting_header(&no_pos).unwrap(), (9, 3));
        assert_eq!(decode_posting_header(&[5, 0]), Err(PostingError::Inconsistent));
        assert_eq!(decode_posting_header(&[5]), Err(PostingError::Truncated));
    }

    #[test]
    fn posting_decode_rejects_corruption() {
        assert_eq!(decode_posting(&[]), Err(PostingError::Truncated));
        assert_eq!(decode_posting(&[5]), Err(PostingError::Truncated));
        // tf == 0 never occurs.
        assert_eq!(decode_posting(&[5, 0]), Err(PostingError::Inconsistent));
        // tf = 2 but only one delta present.
        assert_eq!(decode_posting(&[5, 2, 1]), Err(PostingError::Truncated));
        // tf = 1 but two deltas present.
        assert_eq!(decode_posting(&[5, 1, 1, 1]), Err(PostingError::TrailingBytes));
        // Zero delta after the first = duplicate position.
        assert_eq!(decode_posting(&[5, 2, 1, 0]), Err(PostingError::Inconsistent));
        // Position overflow.
        let mut buf = vec![5, 2];
        put_varint(&mut buf, u32::MAX);
        put_varint(&mut buf, 1);
        assert_eq!(decode_posting(&buf), Err(PostingError::Overflow));
    }

    #[test]
    fn term_map_groups_positions_in_order() {
        let tokens = Analyzer::Simple.analyze("b a b c a b");
        let map = term_map(&tokens);
        assert_eq!(map["a"], vec![1, 4]);
        assert_eq!(map["b"], vec![0, 2, 5]);
        assert_eq!(map["c"], vec![3]);
        assert_eq!(map.keys().copied().collect::<Vec<_>>(), vec!["a", "b", "c"]);
    }
}
