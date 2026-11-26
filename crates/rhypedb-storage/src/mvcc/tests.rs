//! Unit tests for [`super`]: version assignment, conflict detection and
//! snapshot-hold bookkeeping.

use super::*;

// A no-op apply: the manager-level tests exercise version + conflict logic,
// not the WAL/memtable apply (that is covered at the LsmTree level).
fn noop_apply(_v: u64, _w: &[(Bytes, Option<Bytes>)]) -> Result<()> {
    Ok(())
}

#[test]
fn basic_commit() {
    let tm = TransactionManager::new();
    let mut txn = tm.begin();
    txn.record_put(Bytes::from("key1"), Bytes::from("v"));
    let version = tm.commit(&mut txn, noop_apply).unwrap();
    assert!(version > 0);
}

#[test]
fn no_conflict_on_disjoint_writes() {
    let tm = TransactionManager::new();

    let mut txn1 = tm.begin();
    let mut txn2 = tm.begin();

    txn1.record_put(Bytes::from("key1"), Bytes::from("v"));
    txn2.record_put(Bytes::from("key2"), Bytes::from("v"));

    tm.commit(&mut txn1, noop_apply).unwrap();
    tm.commit(&mut txn2, noop_apply).unwrap(); // different keys → ok
}

#[test]
fn conflict_on_overlapping_writes() {
    let tm = TransactionManager::new();

    let mut txn1 = tm.begin();
    let mut txn2 = tm.begin();

    txn1.record_put(Bytes::from("key1"), Bytes::from("a"));
    txn2.record_put(Bytes::from("key1"), Bytes::from("b")); // same key

    tm.commit(&mut txn1, noop_apply).unwrap();
    let result = tm.commit(&mut txn2, noop_apply);
    assert!(matches!(result, Err(Error::WriteConflict)));
    tm.abort(&mut txn2);
}

#[test]
fn later_snapshot_sees_committed_writes() {
    let tm = TransactionManager::new();

    let mut txn1 = tm.begin();
    txn1.record_put(Bytes::from("key1"), Bytes::from("v"));
    let v1 = tm.commit(&mut txn1, noop_apply).unwrap();

    let mut txn2 = tm.begin();
    txn2.record_put(Bytes::from("key1"), Bytes::from("v"));
    let v2 = tm.commit(&mut txn2, noop_apply).unwrap();
    assert!(v2 > v1);
}

#[test]
fn read_only_transactions_always_commit() {
    let tm = TransactionManager::new();
    let mut txn = tm.begin();
    // No writes buffered.
    let version = tm.commit(&mut txn, noop_apply).unwrap();
    assert_eq!(version, txn.snapshot());
    // A read-only commit consumes no version.
    assert_eq!(tm.current_version(), 0);
}

#[test]
fn visible_version_only_advances_on_a_writing_commit() {
    let tm = TransactionManager::new();
    assert_eq!(tm.current_version(), 0);
    let mut t = tm.begin();
    t.record_put(Bytes::from("k"), Bytes::from("v"));
    let v = tm.commit(&mut t, noop_apply).unwrap();
    assert_eq!(tm.current_version(), v);
}

#[test]
fn last_write_wins_coalesces_per_key() {
    let tm = TransactionManager::new();
    let mut t = tm.begin();
    t.record_put(Bytes::from("k"), Bytes::from("v1"));
    t.record_delete(Bytes::from("k"));
    t.record_put(Bytes::from("k"), Bytes::from("v2"));
    assert_eq!(t.writes().len(), 1, "one entry per key");
    assert_eq!(t.writes()[0].1.as_deref(), Some(&b"v2"[..]));
}

#[test]
fn min_active_snapshot_tracking() {
    let tm = TransactionManager::new();
    let txn1 = tm.begin();
    let _txn2 = tm.begin();
    assert_eq!(tm.min_active_snapshot(), txn1.snapshot());
}

#[test]
fn abort_releases_snapshot() {
    let tm = TransactionManager::new();
    let mut txn = tm.begin();
    tm.abort(&mut txn);
    assert_eq!(tm.min_active_snapshot(), u64::MAX);
}

#[test]
fn dropped_read_only_txn_releases_snapshot() {
    let tm = TransactionManager::new();
    let txn = tm.begin();
    assert_eq!(tm.min_active_snapshot(), 0);
    drop(txn);
    assert_eq!(tm.min_active_snapshot(), u64::MAX);
    assert_eq!(tm.active_snapshot_stats(), (0, 0));
}

#[test]
fn dropped_read_write_txn_releases_snapshot_and_applies_nothing() {
    let tm = TransactionManager::new();
    let mut txn = tm.begin();
    txn.record_put(Bytes::from("k"), Bytes::from("v"));
    drop(txn);
    assert_eq!(tm.min_active_snapshot(), u64::MAX);
    // Nothing was committed: no version consumed or published.
    assert_eq!(tm.current_version(), 0);
    let mut t2 = tm.begin();
    t2.record_put(Bytes::from("k"), Bytes::from("v2"));
    assert_eq!(tm.commit(&mut t2, noop_apply).unwrap(), 1);
}

#[test]
fn shared_snapshot_refcount_survives_one_drop() {
    let tm = TransactionManager::new();
    let a = tm.begin();
    let b = tm.begin();
    assert_eq!(a.snapshot(), b.snapshot());
    assert_eq!(tm.active_snapshot_stats(), (1, 2));
    drop(a);
    // b still holds the shared snapshot value.
    assert_eq!(tm.min_active_snapshot(), b.snapshot());
    assert_eq!(tm.active_snapshot_stats(), (1, 1));
    drop(b);
    assert_eq!(tm.min_active_snapshot(), u64::MAX);
}

#[test]
fn commit_or_abort_then_drop_releases_exactly_once() {
    let tm = TransactionManager::new();
    // Three txns share snapshot 0; finishing-then-dropping two of them must
    // not over-decrement the count the third still relies on.
    let mut ro = tm.begin();
    let mut rw = tm.begin();
    let witness = tm.begin();
    let mut aborted = tm.begin();
    assert_eq!(tm.active_snapshot_stats(), (1, 4));

    tm.commit(&mut ro, noop_apply).unwrap(); // read-only fast path
    drop(ro);
    rw.record_put(Bytes::from("k"), Bytes::from("v"));
    tm.commit(&mut rw, noop_apply).unwrap(); // writing commit
    drop(rw);
    tm.abort(&mut aborted);
    tm.abort(&mut aborted); // a second abort is a no-op too
    drop(aborted);

    assert_eq!(tm.active_snapshot_stats(), (1, 1));
    assert_eq!(tm.min_active_snapshot(), witness.snapshot());
    drop(witness);
    assert_eq!(tm.min_active_snapshot(), u64::MAX);
}

/// A transaction that stays open while more commits land than the conflict
/// log keeps must not slip a write-write conflict through: the conflicting
/// commit's entry may have been trimmed, so the log can no longer prove the
/// commit safe — it must refuse (conservatively), never silently overwrite.
#[test]
fn long_txn_conflict_is_detected_after_log_overflow() {
    let tm = TransactionManager::new();
    let mut long = tm.begin();
    long.record_put(Bytes::from("K"), Bytes::from("long"));

    let mut rival = tm.begin();
    rival.record_put(Bytes::from("K"), Bytes::from("rival"));
    tm.commit(&mut rival, noop_apply).unwrap();

    // Far more unrelated commits than the log's capacity. Each writes many
    // keys so a key-count cap is crossed too.
    for i in 0..1_100u32 {
        let mut t = tm.begin();
        for j in 0..2_000u32 {
            t.record_put(Bytes::from(format!("other-{i}-{j}")), Bytes::from("x"));
        }
        tm.commit(&mut t, noop_apply).unwrap();
    }

    assert!(
        matches!(tm.commit(&mut long, noop_apply), Err(Error::WriteConflict)),
        "lost update: the rival's write to K was overwritten"
    );
}

/// The trim must not make a long transaction that touched NOTHING in
/// common fail... unless the log really had to drop entries it needed.
/// Without overflow, a disjoint long transaction commits fine.
#[test]
fn long_disjoint_txn_commits_without_overflow() {
    let tm = TransactionManager::new();
    let mut long = tm.begin();
    long.record_put(Bytes::from("mine"), Bytes::from("v"));
    for i in 0..2_000u32 {
        let mut t = tm.begin();
        t.record_put(Bytes::from(format!("other-{i}")), Bytes::from("x"));
        tm.commit(&mut t, noop_apply).unwrap();
    }
    tm.commit(&mut long, noop_apply).unwrap();
}

#[test]
fn conflict_loser_dropped_without_abort_releases() {
    let tm = TransactionManager::new();
    let mut t1 = tm.begin();
    let mut t2 = tm.begin();
    t1.record_put(Bytes::from("k"), Bytes::from("a"));
    t2.record_put(Bytes::from("k"), Bytes::from("b"));
    tm.commit(&mut t1, noop_apply).unwrap();
    assert!(matches!(
        tm.commit(&mut t2, noop_apply),
        Err(Error::WriteConflict)
    ));
    drop(t2); // the `?`-propagation shape: no explicit abort
    assert_eq!(tm.min_active_snapshot(), u64::MAX);
}

#[test]
fn apply_error_burns_version_but_does_not_publish() {
    let tm = TransactionManager::new();
    let mut t = tm.begin();
    t.record_put(Bytes::from("k"), Bytes::from("v"));
    let r = tm.commit(&mut t, |_v, _w| Err(Error::WalCorrupted("boom".into())));
    assert!(r.is_err());
    // The failed version (1) is NOT published — readers don't see it.
    assert_eq!(tm.current_version(), 0);
    tm.abort(&mut t);
    // The failed version is BURNED, not reused: the next good commit gets 2,
    // so it can never alias a page-cached-but-abandoned batch at version 1.
    let mut t2 = tm.begin();
    t2.record_put(Bytes::from("k2"), Bytes::from("v"));
    assert_eq!(tm.commit(&mut t2, noop_apply).unwrap(), 2);
}

#[test]
fn read_guard_conflicts_with_a_later_commit_to_the_guarded_key() {
    let tm = TransactionManager::new();
    let mut guarded = tm.begin();
    let mut writer = tm.begin();
    // `guarded` decides on "src" (reads it) and writes only "edge".
    guarded.record_read_guard(Bytes::from("src"));
    guarded.record_put(Bytes::from("edge"), Bytes::from("e"));
    writer.record_put(Bytes::from("src"), Bytes::from("changed"));
    tm.commit(&mut writer, noop_apply).unwrap();
    // Plain snapshot isolation would let this commit (disjoint write sets);
    // the guard turns the write skew into a conflict.
    assert!(matches!(
        tm.commit(&mut guarded, noop_apply),
        Err(Error::WriteConflict)
    ));
}

#[test]
fn read_guard_ignores_commits_before_the_snapshot() {
    let tm = TransactionManager::new();
    let mut writer = tm.begin();
    writer.record_put(Bytes::from("src"), Bytes::from("v"));
    tm.commit(&mut writer, noop_apply).unwrap();
    let mut guarded = tm.begin();
    guarded.record_read_guard(Bytes::from("src"));
    guarded.record_put(Bytes::from("edge"), Bytes::from("e"));
    tm.commit(&mut guarded, noop_apply).unwrap();
}

#[test]
fn read_guards_are_not_logged_as_writes() {
    let tm = TransactionManager::new();
    let mut guarded = tm.begin();
    let mut later = tm.begin();
    guarded.record_read_guard(Bytes::from("src"));
    guarded.record_put(Bytes::from("edge"), Bytes::from("e"));
    tm.commit(&mut guarded, noop_apply).unwrap();
    // A concurrent writer of the guarded key does not conflict with the
    // guarding transaction's commit: only its writes are logged.
    later.record_put(Bytes::from("src"), Bytes::from("v"));
    tm.commit(&mut later, noop_apply).unwrap();
}

#[test]
fn conflict_token_fails_an_older_reader_that_guarded_it() {
    let tm = TransactionManager::new();
    let mut reader = tm.begin();
    let mut writer = tm.begin();
    // `reader` decided on a set (guarding its token); `writer` adds a member
    // under a key `reader` never saw, and claims the set's token.
    reader.record_read_guard(Bytes::from("set"));
    reader.record_put(Bytes::from("obj"), Bytes::from("v"));
    writer.record_put(Bytes::from("member-new"), Bytes::from("m"));
    writer.record_conflict_token(Bytes::from("set"));
    tm.commit(&mut writer, noop_apply).unwrap();
    assert!(matches!(
        tm.commit(&mut reader, noop_apply),
        Err(Error::WriteConflict)
    ));
}

#[test]
fn conflict_tokens_are_never_applied() {
    let tm = TransactionManager::new();
    let mut txn = tm.begin();
    txn.record_put(Bytes::from("k"), Bytes::from("v"));
    txn.record_conflict_token(Bytes::from("token"));
    tm.commit(&mut txn, |_v, writes| {
        assert_eq!(writes.len(), 1, "a token is not a write");
        assert_eq!(writes[0].0, Bytes::from("k"));
        Ok(())
    })
    .unwrap();
}

#[test]
fn conflict_token_ignores_readers_with_a_later_snapshot() {
    let tm = TransactionManager::new();
    let mut writer = tm.begin();
    writer.record_put(Bytes::from("member"), Bytes::from("m"));
    writer.record_conflict_token(Bytes::from("set"));
    tm.commit(&mut writer, noop_apply).unwrap();
    let mut reader = tm.begin();
    reader.record_read_guard(Bytes::from("set"));
    reader.record_put(Bytes::from("obj"), Bytes::from("v"));
    tm.commit(&mut reader, noop_apply).unwrap();
}
