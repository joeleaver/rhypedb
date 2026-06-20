//! Restore-on-boot: restore a physical backup snapshot into the `--data-dir`
//! at startup, BEFORE the database is opened, so a managed platform (jkbase) can
//! wake an instance from a backup without shelling into the offline CLI.
//!
//! The flow mirrors the CLI `restore` (validate the snapshot's `MANIFEST.json`,
//! clear stale data, copy the SSTs + WAL + `schema.rhype` + `hnsw_*.bin`), with
//! the additions the boot context demands:
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
//!   records the snapshot identity (`created_at_ms` + `max_version`, both
//!   mount-path independent); a boot whose snapshot matches it is a no-op.
//! * **Crash recovery via a `RESTORE_IN_PROGRESS` marker.** The marker is written
//!   (and fsync'd) BEFORE the destructive clear and removed only after the
//!   sentinel is durable. A crash mid-restore leaves the marker behind; the next
//!   boot sees it and re-restores (bypassing the "non-empty needs --force" gate,
//!   since the leftover files are this restore's own partial output) — so an
//!   unattended restart recovers without operator intervention.
//!
//! The snapshot's own `schema.rhype` is authoritative: the SSTs carry a catalog
//! written for that schema, and `Database::open` reconciles a *different* schema
//! against the on-disk catalog (which can mutate/shrink it). `run()` therefore
//! reads the schema from the restored data dir, not from a possibly-divergent
//! `--schema`.

use rhypedb_storage::lock::DataDirGuard;
use std::path::Path;

const SENTINEL: &str = "RESTORE_DONE";
const IN_PROGRESS: &str = "RESTORE_IN_PROGRESS";

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
    /// Empty on a skip (the live data dir, not the snapshot, defines current state).
    pub in_flight: Vec<(u64, String)>,
}

/// A plain filename: non-empty, no path separators, not `.`/`..`, not absolute.
/// Manifest-listed names are joined onto the data dir, so anything else could
/// write outside it (path traversal from a tampered backup).
fn is_safe_filename(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains('/')
        && !name.contains('\\')
        && !Path::new(name).is_absolute()
}

fn manifest_strs<'a>(manifest: &'a serde_json::Value, key: &str) -> Vec<&'a str> {
    manifest
        .get(key)
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|s| s.as_str()).collect())
        .unwrap_or_default()
}

/// Files a complete snapshot must contain per its manifest. Empty = complete.
/// (Mirrors the CLI `backup_missing_files`; the CLI is engine/storage-free so the
/// logic can't be shared without a new crate — accepted for this phase.)
fn backup_missing_files(dir: &Path, manifest: &serde_json::Value) -> Vec<String> {
    let mut missing = Vec::new();
    for s in manifest_strs(manifest, "ssts") {
        if !dir.join("sst").join(s).is_file() {
            missing.push(format!("sst/{s}"));
        }
    }
    if !dir.join("wal.log").is_file() {
        missing.push("wal.log".to_string());
    }
    // schema.rhype is MANDATORY regardless of whether the manifest names it (a
    // restored dir must be self-describingly openable).
    if !dir.join("schema.rhype").is_file() {
        missing.push("schema.rhype".to_string());
    }
    missing
}

/// True if `data_dir` already holds load-bearing DB state — any of `sst/*.sst`,
/// `wal.log`, `hnsw_*.bin`, or `schema.rhype`. The single-writer `LOCK` file, the
/// `RESTORE_DONE` sentinel, and the `RESTORE_IN_PROGRESS` marker are IGNORED, so a
/// stale LOCK from a crashed instance doesn't spuriously demand `--restore-force`.
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
    if sst_has_files
        || data_dir.join("wal.log").is_file()
        || data_dir.join("schema.rhype").is_file()
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

fn fsync_dir(p: &Path) -> std::io::Result<()> {
    std::fs::File::open(p)?.sync_all()
}

/// The snapshot identity stamped into the sentinel + marker: the manifest
/// `created_at_ms` plus `max_version`. Both are intrinsic to the backup and
/// independent of where the platform mounted it, so a snapshot fetched to a fresh
/// path each wake still matches (and two distinct backups don't collide on a
/// same-millisecond timestamp alone).
fn snapshot_identity(manifest: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "created_at_ms": manifest.get("created_at_ms").and_then(|v| v.as_u64()).unwrap_or(0),
        "max_version": manifest.get("max_version").and_then(|v| v.as_u64()).unwrap_or(0),
    })
}

/// True if `data_dir/RESTORE_DONE` records exactly this snapshot identity — i.e.
/// it was already restored, so this boot is a no-op (preserving post-restore
/// writes).
fn already_restored(data_dir: &Path, want: &serde_json::Value) -> bool {
    let Ok(text) = std::fs::read_to_string(data_dir.join(SENTINEL)) else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .map(|have| &have == want)
        .unwrap_or(false)
}

/// The snapshot identity recorded in a leftover `RESTORE_IN_PROGRESS` marker, if
/// present and parseable. Used to resume ONLY an interrupted restore of the SAME
/// snapshot (a marker for a different snapshot grants no force bypass).
fn marker_identity(data_dir: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(data_dir.join(IN_PROGRESS)).ok()?;
    serde_json::from_str::<serde_json::Value>(&text).ok()
}

/// Restore the physical backup in `snapshot_dir` into `data_dir`, returning a
/// report. Destructive (clears stale data) unless the sentinel shows the same
/// snapshot is already restored, in which case it is a no-op regardless of
/// `force`. Refuses a real pre-existing DB unless `force` (or an interrupted
/// restore is being resumed). Errors are returned as strings for the caller to
/// print + exit on; NOTHING is cleared until the source is fully validated.
pub(crate) fn restore_from_snapshot(
    snapshot_dir: &Path,
    data_dir: &Path,
    force: bool,
) -> Result<RestoreReport, String> {
    // 1. Read + validate the manifest. NON-DESTRUCTIVE — a typo'd / incomplete /
    //    tampered source must never reach the clear step.
    let manifest_text = std::fs::read_to_string(snapshot_dir.join("MANIFEST.json"))
        .map_err(|_| {
            format!(
                "not a valid backup: {}/MANIFEST.json missing",
                snapshot_dir.display()
            )
        })?;
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_text).map_err(|e| format!("invalid MANIFEST.json: {e}"))?;

    // Reject manifest-listed names that aren't plain filenames (path traversal)
    // before they are joined onto either directory.
    for name in manifest_strs(&manifest, "ssts")
        .into_iter()
        .chain(manifest_strs(&manifest, "hnsw_files"))
    {
        if !is_safe_filename(name) {
            return Err(format!("backup manifest lists an unsafe filename: {name:?}"));
        }
    }
    let missing = backup_missing_files(snapshot_dir, &manifest);
    if !missing.is_empty() {
        return Err(format!(
            "source backup is INCOMPLETE — missing: {}",
            missing.join(", ")
        ));
    }
    let in_flight = parse_in_flight(&manifest);
    let identity = snapshot_identity(&manifest);

    // 2. Ensure the data dir exists (non-destructive), then refuse if the snapshot
    //    IS the data dir or lives inside it — the clear would destroy the source.
    //    Fail CLOSED: a canonicalize error on a source-safety check is fatal, not
    //    a silent skip. (Both dirs exist here — data_dir was just created, the
    //    snapshot's MANIFEST.json already read — so this never spuriously fires.)
    std::fs::create_dir_all(data_dir).map_err(|e| format!("create data dir: {e}"))?;
    let snap_canon =
        std::fs::canonicalize(snapshot_dir).map_err(|e| format!("canonicalize snapshot dir: {e}"))?;
    let data_canon =
        std::fs::canonicalize(data_dir).map_err(|e| format!("canonicalize data dir: {e}"))?;
    if snap_canon.starts_with(&data_canon) {
        return Err(format!(
            "--restore-from {} is the data dir (or inside it); refusing to clear the source",
            snapshot_dir.display()
        ));
    }

    // 3. Take the single-writer guard. A live process on this dir makes acquire()
    //    fail loud (DataDirLocked) — we never clear a dir another instance serves.
    let guard = DataDirGuard::acquire(data_dir).map_err(|e| e.to_string())?;

    // 4. Idempotency: this exact snapshot is already fully restored → no-op (do
    //    NOT re-clobber; preserves writes made since the restore). The sentinel is
    //    written LAST, so a matching sentinel means the restore COMPLETED even if a
    //    stale in-progress marker survived a failed cleanup — so this takes priority
    //    over the resume path, and we sweep the stale marker.
    if already_restored(data_dir, &identity) {
        let _ = std::fs::remove_file(data_dir.join(IN_PROGRESS));
        let _ = fsync_dir(data_dir);
        drop(guard);
        return Ok(RestoreReport {
            skipped: true,
            sst_count: 0,
            hnsw_count: 0,
            in_flight: Vec::new(),
        });
    }

    // A marker for THIS SAME snapshot means a previous restore of it was interrupted
    // mid-copy (sentinel absent above); the load-bearing files present are that
    // restore's own partial output, so re-restoring is safe and bypasses the
    // "non-empty needs --force" gate. A marker for a DIFFERENT snapshot is stale and
    // grants no such bypass (the operator must --force to switch snapshots).
    let resuming = marker_identity(data_dir).as_ref() == Some(&identity);

    // 5. A real pre-existing DB (not our sentinel'd restore, not an interrupted
    //    restore of this same snapshot) requires --restore-force.
    if data_dir_non_empty(data_dir) && !force && !resuming {
        return Err(format!(
            "{} already contains a database (use --restore-force / \
             RHYPEDB_RESTORE_FROM_FORCE=1 to overwrite)",
            data_dir.display()
        ));
    }

    // 6. Destructive phase, all under the held guard. Write the IN_PROGRESS marker
    //    FIRST (durably) and remove the stale sentinel, so any crash below leaves
    //    marker-present + sentinel-absent → the next boot resumes the restore.
    std::fs::write(data_dir.join(IN_PROGRESS), identity.to_string())
        .map_err(|e| format!("write in-progress marker: {e}"))?;
    fsync_file(&data_dir.join(IN_PROGRESS)).map_err(|e| format!("fsync in-progress marker: {e}"))?;
    let _ = std::fs::remove_file(data_dir.join(SENTINEL));
    let _ = fsync_dir(data_dir);

    // Clear stale data: LsmTree::open loads EVERY *.sst (no manifest), so a
    // leftover foreign SST would corrupt the restore.
    let _ = std::fs::remove_dir_all(data_dir.join("sst"));
    let _ = std::fs::remove_file(data_dir.join("wal.log"));
    remove_hnsw_bins(data_dir);
    std::fs::create_dir_all(data_dir.join("sst")).map_err(|e| format!("create sst dir: {e}"))?;

    // 7. Copy ONLY load-bearing files. COPY (not link) so the restored dir is
    //    independent of the snapshot. Never copy MANIFEST.json / LOCK / *.tmp.
    let mut sst_count = 0u64;
    for entry in
        std::fs::read_dir(snapshot_dir.join("sst")).map_err(|e| format!("read src/sst: {e}"))?
    {
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
    for f in manifest_strs(&manifest, "hnsw_files") {
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

    // 8. Durability: fsync every copied file + the dirs (PROPAGATE errors — a
    //    silent fsync failure would let us claim completeness we can't back), THEN
    //    write the sentinel + fsync it. Sentinel present ⟹ restore complete+durable.
    sync_restored_files(data_dir).map_err(|e| format!("fsync restored files: {e}"))?;
    std::fs::write(data_dir.join(SENTINEL), identity.to_string())
        .map_err(|e| format!("write sentinel: {e}"))?;
    fsync_file(&data_dir.join(SENTINEL)).map_err(|e| format!("fsync sentinel: {e}"))?;
    let _ = fsync_dir(data_dir);

    // 9. Restore is durable — drop the in-progress marker, then the guard so
    //    Database::open cleanly re-acquires it.
    let _ = std::fs::remove_file(data_dir.join(IN_PROGRESS));
    let _ = fsync_dir(data_dir);
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

fn sync_restored_files(data_dir: &Path) -> std::io::Result<()> {
    for e in std::fs::read_dir(data_dir.join("sst"))? {
        fsync_file(&e?.path())?;
    }
    fsync_file(&data_dir.join("wal.log"))?;
    fsync_file(&data_dir.join("schema.rhype"))?;
    for e in std::fs::read_dir(data_dir)?.flatten() {
        let n = e.file_name();
        let n = n.to_string_lossy();
        if n.starts_with("hnsw_") && n.ends_with(".bin") {
            fsync_file(&e.path())?;
        }
    }
    fsync_dir(&data_dir.join("sst"))?;
    fsync_dir(data_dir)?;
    Ok(())
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
        max_version: u64,
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
            "max_version": max_version,
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
        make_backup(&snap, 111, 5, &["1.sst", "2.sst"], &["1.sst", "2.sst"], &["hnsw_Doc.v.bin"], &["hnsw_Doc.v.bin"]);

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
        // The in-progress marker is removed on success; metadata never copied.
        assert!(!data.join(IN_PROGRESS).exists());
        assert!(!data.join("MANIFEST.json").exists());
    }

    #[test]
    fn refuses_non_empty_without_force_then_succeeds_with_force() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 1, 1, &["1.sst"], &["1.sst"], &[], &[]);
        std::fs::create_dir_all(data.join("sst")).unwrap();
        std::fs::write(data.join("sst/foreign.sst"), b"old").unwrap();

        let err = restore_from_snapshot(&snap, &data, false).unwrap_err();
        assert!(err.contains("already contains a database"), "{err}");
        assert!(data.join("sst/foreign.sst").is_file());

        restore_from_snapshot(&snap, &data, true).unwrap();
        assert!(!data.join("sst/foreign.sst").exists());
        assert!(data.join("sst/1.sst").is_file());
    }

    #[test]
    fn incomplete_source_refuses_before_touching_data_dir() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 1, 1, &["1.sst", "2.sst"], &["1.sst"], &[], &[]);
        std::fs::write(data.join("MARKER"), b"keep").unwrap();

        let err = restore_from_snapshot(&snap, &data, true).unwrap_err();
        assert!(err.contains("INCOMPLETE"), "{err}");
        assert!(data.join("MARKER").is_file(), "data dir was touched before validation");
        assert!(!data.join("sst/1.sst").exists());
    }

    #[test]
    fn missing_schema_rhype_refuses_before_clear() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 1, 1, &["1.sst"], &["1.sst"], &[], &[]);
        std::fs::remove_file(snap.join("schema.rhype")).unwrap();
        std::fs::write(data.join("MARKER"), b"keep").unwrap();

        let err = restore_from_snapshot(&snap, &data, true).unwrap_err();
        assert!(err.contains("schema.rhype"), "{err}");
        assert!(data.join("MARKER").is_file());
    }

    #[test]
    fn stale_lock_file_is_not_treated_as_non_empty() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 1, 1, &["1.sst"], &["1.sst"], &[], &[]);
        std::fs::write(data.join("LOCK"), b"").unwrap();

        let r = restore_from_snapshot(&snap, &data, false).unwrap();
        assert!(!r.skipped);
        assert!(data.join("sst/1.sst").is_file());
    }

    #[test]
    fn missing_listed_hnsw_warns_but_succeeds() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 1, 1, &["1.sst"], &["1.sst"], &["hnsw_Doc.v.bin"], &[]);

        let r = restore_from_snapshot(&snap, &data, false).unwrap();
        assert!(!r.skipped);
        assert_eq!(r.hnsw_count, 0);
        assert!(data.join("sst/1.sst").is_file());
        assert!(!data.join("hnsw_Doc.v.bin").exists());
    }

    #[test]
    fn second_restore_of_same_snapshot_is_a_noop() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 777, 9, &["1.sst"], &["1.sst"], &[], &[]);

        let first = restore_from_snapshot(&snap, &data, false).unwrap();
        assert!(first.in_flight.is_empty());
        std::fs::write(data.join("sst/post.sst"), b"new-data").unwrap();

        let r = restore_from_snapshot(&snap, &data, false).unwrap();
        assert!(r.skipped);
        assert!(r.in_flight.is_empty(), "skip must not surface stale snapshot migration state");
        assert!(data.join("sst/post.sst").is_file(), "skip must not clobber live data");
    }

    #[test]
    fn identity_matches_by_version_not_path() {
        // The SAME backup fetched to a DIFFERENT path each wake must still skip.
        let (_s1, snap1) = temp();
        let (_s2, snap2) = temp();
        let (_d, data) = temp();
        make_backup(&snap1, 555, 42, &["1.sst"], &["1.sst"], &[], &[]);
        make_backup(&snap2, 555, 42, &["1.sst"], &["1.sst"], &[], &[]); // identical identity, other path

        restore_from_snapshot(&snap1, &data, false).unwrap();
        let r = restore_from_snapshot(&snap2, &data, false).unwrap();
        assert!(r.skipped, "same identity at a new path must still be a no-op");
    }

    #[test]
    fn crash_mid_restore_resumes_without_force() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 100, 1, &["1.sst"], &["1.sst"], &[], &[]);

        restore_from_snapshot(&snap, &data, false).unwrap();
        // Simulate a crash mid-restore: sentinel gone, partial files + a marker
        // stamped with THIS snapshot's identity left behind (as the real code does).
        std::fs::remove_file(data.join(SENTINEL)).unwrap();
        std::fs::write(
            data.join(IN_PROGRESS),
            serde_json::json!({"created_at_ms": 100, "max_version": 1}).to_string(),
        )
        .unwrap();

        // Next boot, force OFF (the documented managed config) must RESUME, not
        // refuse with "already contains a database".
        let r = restore_from_snapshot(&snap, &data, false).unwrap();
        assert!(!r.skipped);
        assert!(data.join("sst/1.sst").is_file());
        assert!(data.join(SENTINEL).is_file());
        assert!(!data.join(IN_PROGRESS).exists(), "marker cleared after a successful resume");
    }

    #[test]
    fn surviving_marker_after_complete_restore_still_skips() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 100, 1, &["1.sst"], &["1.sst"], &[], &[]);

        restore_from_snapshot(&snap, &data, false).unwrap();
        // A completed restore (sentinel present + matching) whose marker-removal
        // failed: the marker survives with the matching identity. The next boot must
        // SKIP (sentinel-match wins), NOT re-restore + clobber, and sweep the marker.
        std::fs::write(data.join("sst/post.sst"), b"live-write").unwrap();
        std::fs::write(
            data.join(IN_PROGRESS),
            serde_json::json!({"created_at_ms": 100, "max_version": 1}).to_string(),
        )
        .unwrap();

        let r = restore_from_snapshot(&snap, &data, false).unwrap();
        assert!(r.skipped, "a matching sentinel means restore completed — must skip");
        assert!(data.join("sst/post.sst").is_file(), "skip must not clobber live writes");
        assert!(!data.join(IN_PROGRESS).exists(), "stale marker swept on skip");
    }

    #[test]
    fn stale_marker_for_a_different_snapshot_does_not_bypass_force() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        make_backup(&snap, 200, 2, &["b.sst"], &["b.sst"], &[], &[]);
        // A real pre-existing DB plus a leftover marker from an interrupted restore
        // of a DIFFERENT snapshot (identity 100/1). Restoring snapshot 200/2 must NOT
        // treat the foreign marker as a resume — it still needs --force.
        std::fs::create_dir_all(data.join("sst")).unwrap();
        std::fs::write(data.join("sst/foreign.sst"), b"real-db").unwrap();
        std::fs::write(
            data.join(IN_PROGRESS),
            serde_json::json!({"created_at_ms": 100, "max_version": 1}).to_string(),
        )
        .unwrap();

        let err = restore_from_snapshot(&snap, &data, false).unwrap_err();
        assert!(err.contains("already contains a database"), "{err}");
        assert!(data.join("sst/foreign.sst").is_file(), "must not clobber without force");
    }

    #[test]
    fn different_snapshot_over_restored_dir_needs_force() {
        let (_s1, snap1) = temp();
        let (_s2, snap2) = temp();
        let (_d, data) = temp();
        make_backup(&snap1, 100, 1, &["a.sst"], &["a.sst"], &[], &[]);
        make_backup(&snap2, 200, 2, &["b.sst"], &["b.sst"], &[], &[]);

        restore_from_snapshot(&snap1, &data, false).unwrap();
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
        make_backup(&snap, 1, 1, &["1.sst"], &["1.sst"], &[], &[]);
        std::fs::create_dir_all(&data).unwrap();

        let guard = DataDirGuard::acquire(&data).unwrap();
        let err = restore_from_snapshot(&snap, &data, true).unwrap_err();
        assert!(err.to_lowercase().contains("locked"), "{err}");
        assert!(!data.join("sst/1.sst").exists());

        drop(guard);
        restore_from_snapshot(&snap, &data, true).unwrap();
        assert!(data.join("sst/1.sst").is_file());
    }

    #[test]
    fn refuses_snapshot_equal_to_data_dir() {
        let (_s, snap) = temp();
        make_backup(&snap, 1, 1, &["1.sst"], &["1.sst"], &[], &[]);
        // snapshot_dir == data_dir: clearing would destroy the source.
        let err = restore_from_snapshot(&snap, &snap, true).unwrap_err();
        assert!(err.contains("is the data dir"), "{err}");
        assert!(snap.join("sst/1.sst").is_file(), "source must be untouched");
    }

    #[test]
    fn rejects_unsafe_manifest_filename() {
        let (_s, snap) = temp();
        let (_d, data) = temp();
        std::fs::create_dir_all(snap.join("sst")).unwrap();
        std::fs::write(snap.join("wal.log"), b"x").unwrap();
        std::fs::write(snap.join("schema.rhype"), b"type Doc { x: u32 }\n").unwrap();
        let manifest = serde_json::json!({
            "created_at_ms": 1, "max_version": 1, "ssts": ["../escape.sst"],
            "hnsw_files": [], "schema_file": "schema.rhype", "in_flight_migrations": [],
        });
        std::fs::write(snap.join("MANIFEST.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        std::fs::write(data.join("MARKER"), b"keep").unwrap();

        let err = restore_from_snapshot(&snap, &data, true).unwrap_err();
        assert!(err.contains("unsafe filename"), "{err}");
        assert!(data.join("MARKER").is_file());
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
