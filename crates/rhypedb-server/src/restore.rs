//! Restore-on-boot: restore a physical backup snapshot into the `--data-dir`
//! at startup, BEFORE the database is opened, so a managed platform (jkbase) can
//! wake an instance from a backup without shelling into the offline CLI.
//!
//! The flow mirrors the CLI `restore` (validate the snapshot's `MANIFEST.json`,
//! clear stale data, copy the SSTs + WAL + `schema.rhype` + `hnsw_*.bin`), with
//! three additions the boot context demands:
//!
//! * **Single-writer guard held across the destructive window.** A `flock` alone
//!   only excludes a second *opener*; it does not protect the gap between "clear
//!   the data dir" and "open re-acquires the lock". So restore holds a
//!   [`DataDirGuard`] across clear+copy and drops it before `Database::open`
//!   re-acquires — a concurrent opener fails loud instead of racing a half-cleared
//!   dir.
//! * **Idempotency via a `RESTORE_DONE` sentinel.** A managed platform leaves
//!   `RHYPEDB_RESTORE_FROM` set across restarts; without a sentinel every restart
//!   would wipe live data and re-restore the (now stale) snapshot. The sentinel
//!   records the snapshot identity (`created_at_ms` + path); a boot whose snapshot
//!   matches it is a no-op.
//! * **Crash recovery without staging.** The sentinel is removed FIRST and written
//!   LAST (after fsync), so a crash mid-copy leaves no sentinel and the next boot
//!   cleanly re-restores. (Full staging+atomic-rename is deferred.)
//!
//! The snapshot's own `schema.rhype` is authoritative: the SSTs carry a catalog
//! written for that schema, and `Database::open` reconciles a *different* schema
//! against the on-disk catalog (which can mutate/shrink it). `run()` therefore
//! reads the schema from the restored data dir, not from a possibly-divergent
//! `--schema`.

use rhypedb_storage::lock::DataDirGuard;
use std::path::Path;

const SENTINEL: &str = "RESTORE_DONE";

/// Outcome of a restore-on-boot.
#[derive(Debug)]
pub(crate) struct RestoreReport {
    /// True when the snapshot was already restored (sentinel matched) and nothing
    /// was touched — the live data dir is served as-is.
    pub skipped: bool,
    pub sst_count: u64,
    pub hnsw_count: u64,
    /// `(plan_id, converter)` for any migration in flight when the backup was
    /// taken — the operator must have those converters registered before serving.
    pub in_flight: Vec<(u64, String)>,
}

/// Files a complete snapshot must contain per its manifest. Empty = complete.
/// (Mirrors the CLI `backup_missing_files`; the CLI is engine/storage-free so the
/// logic can't be shared without a new crate — accepted for this phase.)
fn backup_missing_files(dir: &Path, manifest: &serde_json::Value) -> Vec<String> {
    let mut missing = Vec::new();
    if let Some(ssts) = manifest.get("ssts").and_then(|v| v.as_array()) {
        for s in ssts.iter().filter_map(|s| s.as_str()) {
            if !dir.join("sst").join(s).is_file() {
                missing.push(format!("sst/{s}"));
            }
        }
    }
    if !dir.join("wal.log").is_file() {
        missing.push("wal.log".to_string());
    }
    if let Some(sf) = manifest.get("schema_file").and_then(|v| v.as_str())
        && !dir.join(sf).is_file()
    {
        missing.push(sf.to_string());
    }
    missing
}

/// True if `data_dir` already holds load-bearing DB state — any of `sst/*.sst`,
/// `wal.log`, `hnsw_*.bin`, or `schema.rhype`. The single-writer `LOCK` file and
/// our own `RESTORE_DONE` sentinel are IGNORED, so a stale LOCK from a crashed
/// instance doesn't spuriously demand `--restore-force`.
fn data_dir_non_empty(data_dir: &Path) -> bool {
    let sst_has_files = std::fs::read_dir(data_dir.join("sst"))
        .map(|mut d| {
            d.any(|e| {
                e.ok()
                    .map(|e| e.path().extension().and_then(|x| x.to_str()) == Some("sst"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if sst_has_files || data_dir.join("wal.log").is_file() || data_dir.join("schema.rhype").is_file()
    {
        return true;
    }
    std::fs::read_dir(data_dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                let n = e.file_name();
                let n = n.to_string_lossy();
                n.starts_with("hnsw_") && n.ends_with(".bin")
            })
        })
        .unwrap_or(false)
}

fn fsync_file(p: &Path) -> std::io::Result<()> {
    std::fs::File::open(p)?.sync_all()
}

/// fsync a directory's entries so renames/creates/unlinks within it persist.
fn fsync_dir(p: &Path) {
    if let Ok(f) = std::fs::File::open(p) {
        let _ = f.sync_all();
    }
}

/// The snapshot identity we stamp into the sentinel: the manifest `created_at_ms`
/// plus the absolute snapshot path. Both must match for a boot to skip restore.
fn sentinel_value(snapshot_dir: &Path, created_at_ms: u64) -> serde_json::Value {
    let abs = std::fs::canonicalize(snapshot_dir)
        .unwrap_or_else(|_| snapshot_dir.to_path_buf());
    serde_json::json!({
        "snapshot": abs.to_string_lossy(),
        "created_at_ms": created_at_ms,
    })
}

/// True if `data_dir/RESTORE_DONE` records exactly this snapshot — i.e. it was
/// already restored, so this boot is a no-op (preserving any post-restore writes).
fn already_restored(data_dir: &Path, want: &serde_json::Value) -> bool {
    let Ok(text) = std::fs::read_to_string(data_dir.join(SENTINEL)) else {
        return false;
    };
    let Ok(have) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    have.get("created_at_ms") == want.get("created_at_ms")
        && have.get("snapshot") == want.get("snapshot")
}

/// Restore the physical backup in `snapshot_dir` into `data_dir`, returning a
/// report. Destructive (clears stale data) unless the sentinel shows the same
/// snapshot is already restored, in which case it is a no-op regardless of
/// `force`. Refuses a real pre-existing DB unless `force`. Errors are returned as
/// strings for the caller to print + exit on; nothing is cleared until the source
/// is validated.
pub(crate) fn restore_from_snapshot(
    snapshot_dir: &Path,
    data_dir: &Path,
    force: bool,
) -> Result<RestoreReport, String> {
    // 1. Read + validate the manifest. NON-DESTRUCTIVE — a typo'd source must
    //    never reach the clear step.
    let manifest_text = std::fs::read_to_string(snapshot_dir.join("MANIFEST.json"))
        .map_err(|_| {
            format!(
                "not a valid backup: {}/MANIFEST.json missing",
                snapshot_dir.display()
            )
        })?;
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_text).map_err(|e| format!("invalid MANIFEST.json: {e}"))?;
    let missing = backup_missing_files(snapshot_dir, &manifest);
    if !missing.is_empty() {
        return Err(format!(
            "source backup is INCOMPLETE — missing: {}",
            missing.join(", ")
        ));
    }
    let created_at_ms = manifest
        .get("created_at_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let in_flight = parse_in_flight(&manifest);

    // 2. Ensure the data dir exists, then take the single-writer guard. A live
    //    process on this dir makes acquire() fail loud (DataDirLocked) — we never
    //    clear a dir another instance is serving.
    std::fs::create_dir_all(data_dir).map_err(|e| format!("create data dir: {e}"))?;
    let guard = DataDirGuard::acquire(data_dir).map_err(|e| e.to_string())?;

    let want = sentinel_value(snapshot_dir, created_at_ms);

    // 3. Idempotency: this exact snapshot is already restored → no-op (do NOT
    //    re-clobber; preserves writes made since the restore). Independent of
    //    `force`, so a persistent RHYPEDB_RESTORE_FROM doesn't wipe on every boot.
    if already_restored(data_dir, &want) {
        drop(guard);
        return Ok(RestoreReport {
            skipped: true,
            sst_count: 0,
            hnsw_count: 0,
            in_flight,
        });
    }

    // 4. A real pre-existing DB (not our sentinel'd restore) requires --restore-force.
    if data_dir_non_empty(data_dir) && !force {
        return Err(format!(
            "{} already contains a database (use --restore-force / \
             RHYPEDB_RESTORE_FROM_FORCE=1 to overwrite)",
            data_dir.display()
        ));
    }

    // 5. Destructive phase, all under the held guard. Remove the sentinel FIRST so
    //    a crash anywhere below leaves NO sentinel → the next boot re-restores.
    let _ = std::fs::remove_file(data_dir.join(SENTINEL));
    fsync_dir(data_dir);

    // Clear stale data: LsmTree::open loads EVERY *.sst (no manifest), so a
    // leftover foreign SST would corrupt the restore.
    let _ = std::fs::remove_dir_all(data_dir.join("sst"));
    let _ = std::fs::remove_file(data_dir.join("wal.log"));
    remove_hnsw_bins(data_dir);
    std::fs::create_dir_all(data_dir.join("sst")).map_err(|e| format!("create sst dir: {e}"))?;

    // 6. Copy ONLY load-bearing files. COPY (not link) so the restored dir is
    //    independent of the snapshot. Never copy MANIFEST.json / LOCK / *.tmp.
    let mut sst_count = 0u64;
    for entry in std::fs::read_dir(snapshot_dir.join("sst")).map_err(|e| format!("read src/sst: {e}"))? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.path().extension().and_then(|e| e.to_str()) == Some("sst") {
            let dst = data_dir.join("sst").join(entry.file_name());
            std::fs::copy(entry.path(), &dst).map_err(|e| format!("copy SST: {e}"))?;
            sst_count += 1;
        }
    }
    std::fs::copy(snapshot_dir.join("wal.log"), data_dir.join("wal.log"))
        .map_err(|e| format!("copy wal.log: {e}"))?;
    std::fs::copy(snapshot_dir.join("schema.rhype"), data_dir.join("schema.rhype"))
        .map_err(|e| format!("copy schema.rhype: {e}"))?;
    // hnsw_*.bin: a performance optimization (skips the rebuild on open). Copy
    // every file the manifest lists; a missing listed one is a WARN, not a
    // failure — the Vectorizer delta-rebuilds from the LSM f32 vectors.
    let mut hnsw_count = 0u64;
    if let Some(files) = manifest.get("hnsw_files").and_then(|v| v.as_array()) {
        for f in files.iter().filter_map(|f| f.as_str()) {
            let src = snapshot_dir.join(f);
            if src.is_file() {
                std::fs::copy(&src, data_dir.join(f)).map_err(|e| format!("copy {f}: {e}"))?;
                hnsw_count += 1;
            } else {
                eprintln!(
                    "WARNING: manifest lists HNSW snapshot {f} but it is missing from the \
                     backup; the index will be rebuilt from the LSM on open (slower wake)."
                );
            }
        }
    }

    // 7. Durability: fsync every copied file + the dirs, THEN write the sentinel
    //    LAST + fsync it. Sentinel present ⟹ the restore is complete + durable.
    sync_restored_files(data_dir);
    std::fs::write(
        data_dir.join(SENTINEL),
        serde_json::to_vec_pretty(&want).unwrap_or_default(),
    )
    .map_err(|e| format!("write sentinel: {e}"))?;
    let _ = fsync_file(&data_dir.join(SENTINEL));
    fsync_dir(data_dir);

    // 8. Drop the guard so Database::open cleanly re-acquires it.
    drop(guard);

    Ok(RestoreReport {
        skipped: false,
        sst_count,
        hnsw_count,
        in_flight,
    })
}

fn parse_in_flight(manifest: &serde_json::Value) -> Vec<(u64, String)> {
    manifest
        .get("in_flight_migrations")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|m| {
                    (
                        m.get("plan_id").and_then(|v| v.as_u64()).unwrap_or(0),
                        m.get("converter")
                            .and_then(|v| v.as_str())
                            .unwrap_or("?")
                            .to_string(),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn remove_hnsw_bins(data_dir: &Path) {
    if let Ok(rd) = std::fs::read_dir(data_dir) {
        for entry in rd.flatten() {
            let n = entry.file_name();
            let n = n.to_string_lossy();
            if n.starts_with("hnsw_") && n.ends_with(".bin") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

fn sync_restored_files(data_dir: &Path) {
    if let Ok(rd) = std::fs::read_dir(data_dir.join("sst")) {
        for e in rd.flatten() {
            let _ = fsync_file(&e.path());
        }
    }
    let _ = fsync_file(&data_dir.join("wal.log"));
    let _ = fsync_file(&data_dir.join("schema.rhype"));
    if let Ok(rd) = std::fs::read_dir(data_dir) {
        for e in rd.flatten() {
            let n = e.file_name();
            let n = n.to_string_lossy();
            if n.starts_with("hnsw_") && n.ends_with(".bin") {
                let _ = fsync_file(&e.path());
            }
        }
    }
    fsync_dir(&data_dir.join("sst"));
    fsync_dir(data_dir);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Fabricate a backup snapshot dir. Fake SST/hnsw bytes are fine — restore
    /// only copies them. `present_ssts`/`present_hnsw` control which manifest-listed
    /// files actually exist on disk (for the incomplete / missing-hnsw cases).
    fn make_backup(
        dir: &Path,
        created_at_ms: u64,
        ssts: &[&str],
        present_ssts: &[&str],
        hnsw: &[&str],
        present_hnsw: &[&str],
    ) {
        std::fs::create_dir_all(dir.join("sst")).unwrap();
        for s in present_ssts {
            std::fs::write(dir.join("sst").join(s), b"fake-sst").unwrap();
        }
        std::fs::write(dir.join("wal.log"), b"fake-wal").unwrap();
        std::fs::write(dir.join("schema.rhype"), b"type Doc { x: u32 }\n").unwrap();
        for h in present_hnsw {
            std::fs::write(dir.join(h), b"fake-hnsw").unwrap();
        }
        let manifest = serde_json::json!({
            "created_at_ms": created_at_ms,
            "wal_bytes": 8,
            "ssts": ssts,
            "hnsw_files": hnsw,
            "schema_file": "schema.rhype",
            "in_flight_migrations": [],
        });
        std::fs::write(
            dir.join("MANIFEST.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    fn temp() -> (tempfile::TempDir, PathBuf) {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().to_path_buf();
        (d, p)
    }

    #[test]
    fn restore_into_empty_dir_copies_files_and_writes_sentinel() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 111, &["1.sst", "2.sst"], &["1.sst", "2.sst"], &["hnsw_Doc.v.bin"], &["hnsw_Doc.v.bin"]);

        let r = restore_from_snapshot(&snap, &data, false).unwrap();
        assert!(!r.skipped);
        assert_eq!(r.sst_count, 2);
        assert_eq!(r.hnsw_count, 1);
        assert!(data.join("sst/1.sst").is_file());
        assert!(data.join("sst/2.sst").is_file());
        assert!(data.join("wal.log").is_file());
        assert!(data.join("schema.rhype").is_file());
        assert!(data.join("hnsw_Doc.v.bin").is_file());
        assert!(data.join(SENTINEL).is_file());
        // The snapshot's metadata files must NOT be copied into the data dir.
        assert!(!data.join("MANIFEST.json").exists());
    }

    #[test]
    fn refuses_non_empty_without_force_then_succeeds_with_force() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 1, &["1.sst"], &["1.sst"], &[], &[]);
        // Pre-existing DB state (a foreign SST) in the data dir.
        std::fs::create_dir_all(data.join("sst")).unwrap();
        std::fs::write(data.join("sst/foreign.sst"), b"old").unwrap();

        let err = restore_from_snapshot(&snap, &data, false).unwrap_err();
        assert!(err.contains("already contains a database"), "{err}");
        // Untouched without force.
        assert!(data.join("sst/foreign.sst").is_file());

        // With force: clears the foreign SST and restores cleanly.
        restore_from_snapshot(&snap, &data, true).unwrap();
        assert!(!data.join("sst/foreign.sst").exists());
        assert!(data.join("sst/1.sst").is_file());
    }

    #[test]
    fn incomplete_source_refuses_before_touching_data_dir() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        // Manifest lists 2.sst but only 1.sst exists on disk.
        make_backup(&snap, 1, &["1.sst", "2.sst"], &["1.sst"], &[], &[]);
        // A marker that must survive (restore must validate before any clear).
        std::fs::write(data.join("MARKER"), b"keep").unwrap();

        let err = restore_from_snapshot(&snap, &data, true).unwrap_err();
        assert!(err.contains("INCOMPLETE"), "{err}");
        assert!(data.join("MARKER").is_file(), "data dir was touched before validation");
        assert!(!data.join("sst/1.sst").exists());
    }

    #[test]
    fn stale_lock_file_is_not_treated_as_non_empty() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 1, &["1.sst"], &["1.sst"], &[], &[]);
        // A leftover LOCK file from a crashed instance (not flock-held) must not
        // count as a pre-existing DB, so no --force is needed.
        std::fs::write(data.join("LOCK"), b"").unwrap();

        let r = restore_from_snapshot(&snap, &data, false).unwrap();
        assert!(!r.skipped);
        assert!(data.join("sst/1.sst").is_file());
    }

    #[test]
    fn missing_listed_hnsw_warns_but_succeeds() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        // Manifest lists an hnsw file that is NOT present in the backup.
        make_backup(&snap, 1, &["1.sst"], &["1.sst"], &["hnsw_Doc.v.bin"], &[]);

        let r = restore_from_snapshot(&snap, &data, false).unwrap();
        assert!(!r.skipped);
        assert_eq!(r.hnsw_count, 0); // missing → not copied, but restore still OK
        assert!(data.join("sst/1.sst").is_file());
        assert!(!data.join("hnsw_Doc.v.bin").exists());
    }

    #[test]
    fn second_restore_of_same_snapshot_is_a_noop() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 777, &["1.sst"], &["1.sst"], &[], &[]);

        restore_from_snapshot(&snap, &data, false).unwrap();
        // Simulate a write made AFTER the restore (e.g. live serving).
        std::fs::write(data.join("sst/post.sst"), b"new-data").unwrap();

        // Same snapshot again (e.g. a restart with RHYPEDB_RESTORE_FROM still set):
        // must skip and preserve the post-restore write — NOT re-clobber. No force.
        let r = restore_from_snapshot(&snap, &data, false).unwrap();
        assert!(r.skipped);
        assert!(data.join("sst/post.sst").is_file(), "skip must not clobber live data");
    }

    #[test]
    fn different_snapshot_over_restored_dir_needs_force() {
        let (_s1, snap1) = temp();
        let (_s2, snap2) = temp();
        let (_d, data) = temp();
        make_backup(&snap1, 100, &["a.sst"], &["a.sst"], &[], &[]);
        make_backup(&snap2, 200, &["b.sst"], &["b.sst"], &[], &[]);

        restore_from_snapshot(&snap1, &data, false).unwrap();
        // A DIFFERENT snapshot (different created_at_ms) over a restored dir is a
        // real overwrite — refused without force.
        let err = restore_from_snapshot(&snap2, &data, false).unwrap_err();
        assert!(err.contains("already contains a database"), "{err}");
        restore_from_snapshot(&snap2, &data, true).unwrap();
        assert!(data.join("sst/b.sst").is_file());
        assert!(!data.join("sst/a.sst").exists());
    }

    #[test]
    fn refuses_when_data_dir_is_locked_by_a_live_process() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 1, &["1.sst"], &["1.sst"], &[], &[]);
        std::fs::create_dir_all(&data).unwrap();

        // Hold the single-writer guard — restore must fail loud, not clear under it.
        let guard = DataDirGuard::acquire(&data).unwrap();
        let err = restore_from_snapshot(&snap, &data, true).unwrap_err();
        assert!(err.to_lowercase().contains("locked"), "{err}");
        assert!(!data.join("sst/1.sst").exists());

        // Released → restore proceeds.
        drop(guard);
        restore_from_snapshot(&snap, &data, true).unwrap();
        assert!(data.join("sst/1.sst").is_file());
    }

    #[test]
    fn missing_manifest_is_not_a_backup() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        std::fs::create_dir_all(snap.join("sst")).unwrap();
        let err = restore_from_snapshot(&snap, &data, false).unwrap_err();
        assert!(err.contains("MANIFEST.json missing"), "{err}");
    }
}
