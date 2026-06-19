//! Offline logical IMPORT — reconstruct a database from a `rhypedb-logical-export`
//! NDJSON dump (Overboard cmqikqug4), the counterpart of `Database::logical_export_stream`.
//!
//! Unlike export (an HTTP endpoint + thin CLI client), import is an ENGINE
//! operation: it opens the data dir directly and re-inserts every row through
//! the write path. So it ships as the offline `rhypedb-import` binary (the
//! server must be STOPPED), not the HTTP CLI — the CLI does not link the engine.
//!
//! All-or-nothing at the dir level: the whole import is built in a sibling
//! STAGING dir and atomically renamed into place only after it fully succeeds,
//! so a failure (a bad line, an I/O error, an unparseable schema) leaves the
//! target data dir completely untouched — never a wiped-and-partial dir.
//!
//! Flow:
//! 1. Validate the file (header format + trailer + per-type counts) AND parse
//!    the embedded schema — all non-destructive — and guard the target dir
//!    (refuse a non-empty one without `--force`). Nothing is touched yet.
//! 2. `Database::open` a fresh STAGING dir (a sibling of the target) and stream
//!    the dependency-safe sections into it: object lines → `restore_objects`
//!    (id-preserving, chunked per type); edge lines → `Database::link` (every
//!    endpoint exists by then — two-pass-by-stream-order handles cycles, no topo
//!    sort); vector lines → `restore_vectors` (verbatim raw f32, chunked per
//!    field; the HNSW graph rebuilds from those `v:` keys on the next start).
//!    Write `schema.rhype` so the operator can start the server.
//! 3. Atomically swap the staging dir into the target; the staging dir is
//!    removed on any earlier failure, leaving the target intact.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use bytes::Bytes;
use rhypedb_engine::database::Database;
use rhypedb_engine::logical::{self, FORMAT_TAG};
use rhypedb_engine::object::FieldMap;
use rhypedb_schema::parser::parse_schema;
use serde_json::Value as Json;

/// Objects / vectors re-inserted per engine call. Bounds peak memory.
const IMPORT_CHUNK: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorImportMode {
    /// Import the raw f32 vectors (default); the HNSW rebuilds on next open.
    Raw,
    /// Skip vector lines entirely.
    None,
}

pub struct ImportOptions {
    pub force: bool,
    pub vectors: VectorImportMode,
}

#[derive(Debug, Default)]
pub struct ImportReport {
    pub types: usize,
    pub objects: u64,
    pub edges: u64,
    pub vectors: u64,
}

/// Run an offline logical import of `src` into `data_dir`.
pub fn run_import(
    src: &Path,
    data_dir: &Path,
    opts: &ImportOptions,
) -> Result<ImportReport, String> {
    // 1. Validate the file + parse the embedded schema (both NON-destructive),
    //    and guard the target — all BEFORE anything is touched, so a truncated
    //    dump or an unparseable schema refuses with the target dir intact.
    let validated = validate_file(src)?;
    let schema = parse_schema(&validated.sdl)
        .map_err(|e| format!("embedded schema is invalid: {e}"))?;
    guard_target(data_dir, opts.force)?;

    // 2. Build the ENTIRE import in a sibling staging dir (same filesystem, so
    //    the final rename is atomic). The staging dir is removed on ANY failure
    //    below, leaving the target untouched — the import is all-or-nothing.
    let staging = staging_path(data_dir)?;
    let mut guard = StagingGuard::new(staging.clone());
    std::fs::create_dir_all(&staging).map_err(|e| format!("create staging dir: {e}"))?;

    let report = {
        let db = Database::open(schema, &staging)
            .map_err(|e| format!("open staging dir: {e}"))?;
        let report = import_pass(src, &db, opts)?;
        // The validation above checked the file's INTERNAL counts; re-check what
        // we actually imported against the trailer to catch a file that changed
        // between the validate read and the import read (TOCTOU).
        verify_counts(&report, &validated, opts)?;
        // Materialize the memtable to SSTs before handing the dir over.
        db.storage()
            .flush()
            .map_err(|e| format!("flush staging dir: {e}"))?;
        report
        // `db` drops here, releasing the LSM on the staging dir.
    };

    // Make the dir self-contained + startable: the server needs a --schema file
    // (the dump embeds the schema instead).
    std::fs::write(staging.join("schema.rhype"), &validated.sdl)
        .map_err(|e| format!("write schema.rhype: {e}"))?;

    // 3. Atomically swap the staging dir into place. Only NOW is the target
    //    touched; on success the staging dir was moved, so disarm cleanup.
    swap_into_place(&staging, data_dir)?;
    guard.disarm();
    Ok(report)
}

struct Validated {
    sdl: String,
    /// Total [objects, edges, vectors] across all types, from the trailer.
    totals: [u64; 3],
}

/// Refuse a non-empty target dir without `--force`. The target is NOT modified
/// here — the import stages into a sibling dir and only swaps in at the end.
fn guard_target(dir: &Path, force: bool) -> Result<(), String> {
    if dir.exists() {
        let non_empty = std::fs::read_dir(dir)
            .map(|mut d| d.next().is_some())
            .unwrap_or(true);
        if non_empty && !force {
            return Err(format!(
                "{} exists and is not empty (use --force to overwrite)",
                dir.display()
            ));
        }
    }
    Ok(())
}

fn verify_counts(
    report: &ImportReport,
    validated: &Validated,
    opts: &ImportOptions,
) -> Result<(), String> {
    let [objects, edges, vectors] = validated.totals;
    if report.objects != objects || report.edges != edges {
        return Err(format!(
            "imported counts diverge from the trailer (objects {}/{objects}, edges {}/{edges}) \
             — did the source file change during import?",
            report.objects, report.edges
        ));
    }
    if opts.vectors == VectorImportMode::Raw && report.vectors != vectors {
        return Err(format!(
            "imported vectors {} != trailer {vectors} — did the source file change during import?",
            report.vectors
        ));
    }
    Ok(())
}

/// A staging dir under the target's parent (same filesystem → atomic rename).
fn staging_path(data_dir: &Path) -> Result<PathBuf, String> {
    let parent = parent_or_cwd(data_dir);
    std::fs::create_dir_all(&parent)
        .map_err(|e| format!("create parent of {}: {e}", data_dir.display()))?;
    Ok(parent.join(format!(".rhypedb-import-{}", unique_suffix(data_dir))))
}

/// Atomically install the fully-built `staging` dir at `data_dir`. If the target
/// already exists it is moved aside first (and restored on a mid-swap failure),
/// so the operation never leaves a half-installed dir.
fn swap_into_place(staging: &Path, data_dir: &Path) -> Result<(), String> {
    if data_dir.exists() {
        let backup = parent_or_cwd(data_dir).join(format!(".rhypedb-old-{}", unique_suffix(data_dir)));
        std::fs::rename(data_dir, &backup)
            .map_err(|e| format!("move {} aside: {e}", data_dir.display()))?;
        match std::fs::rename(staging, data_dir) {
            Ok(()) => {
                let _ = std::fs::remove_dir_all(&backup);
                Ok(())
            }
            Err(e) => {
                // Restore the original so a failed swap is not destructive.
                let _ = std::fs::rename(&backup, data_dir);
                Err(format!("install import at {}: {e}", data_dir.display()))
            }
        }
    } else {
        std::fs::rename(staging, data_dir)
            .map_err(|e| format!("install import at {}: {e}", data_dir.display()))
    }
}

fn parent_or_cwd(p: &Path) -> PathBuf {
    p.parent()
        .filter(|x| !x.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn unique_suffix(data_dir: &Path) -> String {
    let name = data_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "data".into());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{name}-{}-{nanos}", std::process::id())
}

/// Removes the staging dir on drop unless disarmed — so any early return from
/// `run_import` cleans up the partial staging dir and leaves the target intact.
struct StagingGuard {
    path: PathBuf,
    armed: bool,
}
impl StagingGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }
    fn disarm(&mut self) {
        self.armed = false;
    }
}
impl Drop for StagingGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// Validate the export file's structure: a `header` first line with the
/// expected format tag, a `schema` line, a `trailer` LAST line marked
/// `complete:true`, and per-type object/edge/vector counts that match the
/// actual lines. Streams the file (bounded memory).
fn validate_file(path: &Path) -> Result<Validated, String> {
    let file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut lines = std::io::BufReader::new(file).lines();

    let header_line = loop {
        match lines.next() {
            Some(Ok(l)) if l.trim().is_empty() => continue,
            Some(Ok(l)) => break l,
            Some(Err(e)) => return Err(format!("read error: {e}")),
            None => return Err("empty export file".into()),
        }
    };
    let header: Json =
        serde_json::from_str(&header_line).map_err(|e| format!("first line is not JSON: {e}"))?;
    if header.get("kind").and_then(Json::as_str) != Some("header") {
        return Err("first line is not a header".into());
    }
    if header.get("format").and_then(Json::as_str) != Some(FORMAT_TAG) {
        return Err(format!("unknown/incompatible format (expected {FORMAT_TAG})"));
    }

    let mut sdl: Option<String> = None;
    let mut tally: BTreeMap<String, [u64; 3]> = BTreeMap::new();
    let mut trailer: Option<Json> = None;
    for line in lines {
        let line = line.map_err(|e| format!("read error: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        if trailer.is_some() {
            return Err("malformed export: lines present after the trailer".into());
        }
        let v: Json = serde_json::from_str(&line).map_err(|e| format!("invalid JSON line: {e}"))?;
        match v.get("kind").and_then(Json::as_str) {
            Some("schema") => {
                sdl = Some(
                    v.get("sdl")
                        .and_then(Json::as_str)
                        .ok_or("schema line missing string `sdl`")?
                        .to_owned(),
                );
            }
            Some(kind @ ("object" | "edge" | "vector")) => {
                let t = v.get("type").and_then(Json::as_str).unwrap_or("").to_string();
                let slot = tally.entry(t).or_default();
                match kind {
                    "object" => slot[0] += 1,
                    "edge" => slot[1] += 1,
                    _ => slot[2] += 1,
                }
            }
            Some("trailer") => trailer = Some(v),
            Some("header") => return Err("unexpected second header line".into()),
            other => return Err(format!("unknown line kind: {other:?}")),
        }
    }

    let sdl = sdl.ok_or("export has no schema line")?;
    let trailer = trailer.ok_or("export INCOMPLETE: no trailer line (truncated)")?;
    if trailer.get("complete").and_then(Json::as_bool) != Some(true) {
        return Err("export trailer is not marked complete".into());
    }
    let counts_obj = trailer
        .get("counts")
        .and_then(Json::as_object)
        .ok_or("trailer is missing per-type counts")?;

    let mut totals = [0u64; 3];
    for (t, c) in counts_obj {
        let want = [
            c.get("objects").and_then(Json::as_u64).unwrap_or(0),
            c.get("edges").and_then(Json::as_u64).unwrap_or(0),
            c.get("vectors").and_then(Json::as_u64).unwrap_or(0),
        ];
        let got = tally.remove(t).unwrap_or([0, 0, 0]);
        if got != want {
            return Err(format!(
                "count mismatch for type {t}: trailer says objects={} edges={} vectors={}, \
                 file has objects={} edges={} vectors={}",
                want[0], want[1], want[2], got[0], got[1], got[2]
            ));
        }
        for i in 0..3 {
            totals[i] += want[i];
        }
    }
    if let Some((t, _)) = tally.iter().next() {
        return Err(format!("type {t} has data lines but is absent from the trailer counts"));
    }

    Ok(Validated { sdl, totals })
}


fn flush_objects(
    db: &Database,
    ty: &Option<String>,
    buf: &mut Vec<(u64, FieldMap)>,
) -> Result<(), String> {
    if let Some(t) = ty
        && !buf.is_empty()
    {
        db.restore_objects(t, std::mem::take(buf))
            .map_err(|e| format!("restore objects of {t}: {e}"))?;
    }
    Ok(())
}

fn flush_vectors(
    db: &Database,
    key: &Option<(String, String)>,
    buf: &mut Vec<(u64, Bytes)>,
) -> Result<(), String> {
    if let Some((t, f)) = key
        && !buf.is_empty()
    {
        db.restore_vectors(t, f, &std::mem::take(buf))
            .map_err(|e| format!("restore vectors of {t}.{f}: {e}"))?;
    }
    Ok(())
}

fn import_pass(
    path: &Path,
    db: &Database,
    opts: &ImportOptions,
) -> Result<ImportReport, String> {
    let file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let lines = std::io::BufReader::new(file).lines();
    let mut report = ImportReport::default();

    let mut obj_type: Option<String> = None;
    let mut obj_buf: Vec<(u64, FieldMap)> = Vec::new();
    let mut vec_key: Option<(String, String)> = None;
    let mut vec_buf: Vec<(u64, Bytes)> = Vec::new();

    for line in lines {
        let line = line.map_err(|e| format!("read error: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Json = serde_json::from_str(&line).map_err(|e| format!("invalid JSON line: {e}"))?;
        match v.get("kind").and_then(Json::as_str) {
            // Header + schema were consumed up front (validate + open).
            Some("header") | Some("schema") => {}

            Some("object") => {
                let t = str_field(&v, "type")?;
                if obj_type.as_deref() != Some(t) {
                    // Type changed → flush the previous type's buffer.
                    flush_objects(db, &obj_type, &mut obj_buf)?;
                    obj_type = Some(t.to_owned());
                }
                let id = u64_field(&v, "id")?;
                let fields = logical::fields_from_json(field_obj(&v, "fields")?)
                    .map_err(|e| format!("object {t}:{id} fields: {e}"))?;
                obj_buf.push((id, fields));
                if obj_buf.len() >= IMPORT_CHUNK {
                    flush_objects(db, &obj_type, &mut obj_buf)?;
                }
                report.objects += 1;
            }

            Some("edge") => {
                // All objects precede all edges, so every endpoint now exists.
                flush_objects(db, &obj_type, &mut obj_buf)?;
                obj_type = None;
                let src_type = str_field(&v, "type")?;
                let src = u64_field(&v, "src")?;
                let field = str_field(&v, "field")?;
                let dst = u64_field(&v, "dst")?;
                let edge_fields = logical::fields_from_json(field_obj(&v, "edge_fields")?)
                    .map_err(|e| format!("edge {src_type}:{src}.{field}->{dst}: {e}"))?;
                let edge_fields = if edge_fields.is_empty() {
                    None
                } else {
                    Some(edge_fields)
                };
                db.link(src_type, src, field, dst, edge_fields)
                    .map_err(|e| format!("link {src_type}:{src}.{field}->{dst}: {e}"))?;
                report.edges += 1;
            }

            Some("vector") => {
                flush_objects(db, &obj_type, &mut obj_buf)?;
                obj_type = None;
                if opts.vectors == VectorImportMode::None {
                    continue;
                }
                let t = str_field(&v, "type")?;
                let field = str_field(&v, "field")?;
                let key = (t.to_owned(), field.to_owned());
                if vec_key.as_ref() != Some(&key) {
                    flush_vectors(db, &vec_key, &mut vec_buf)?;
                    vec_key = Some(key);
                }
                let id = u64_field(&v, "id")?;
                let raw = logical::decode_bytes(
                    v.get("f32").and_then(Json::as_str).ok_or("vector missing `f32`")?,
                )
                .map_err(|e| format!("vector {t}:{id}.{field}: {e}"))?;
                vec_buf.push((id, Bytes::from(raw)));
                if vec_buf.len() >= IMPORT_CHUNK {
                    flush_vectors(db, &vec_key, &mut vec_buf)?;
                }
                report.vectors += 1;
            }

            Some("trailer") => {
                flush_objects(db, &obj_type, &mut obj_buf)?;
                obj_type = None;
                flush_vectors(db, &vec_key, &mut vec_buf)?;
                vec_key = None;
                if let Some(c) = v.get("counts").and_then(Json::as_object) {
                    report.types = c.len();
                }
            }

            other => return Err(format!("unknown line kind: {other:?}")),
        }
    }

    // The trailer was validated to be present + last, so the buffers above are
    // already flushed; this is a belt-and-suspenders final flush.
    flush_objects(db, &obj_type, &mut obj_buf)?;
    flush_vectors(db, &vec_key, &mut vec_buf)?;
    Ok(report)
}

fn str_field<'a>(v: &'a Json, name: &str) -> Result<&'a str, String> {
    v.get(name)
        .and_then(Json::as_str)
        .ok_or_else(|| format!("line missing string field `{name}`"))
}

fn u64_field(v: &Json, name: &str) -> Result<u64, String> {
    let s = str_field(v, name)?;
    s.parse::<u64>()
        .map_err(|e| format!("invalid `{name}` {s:?}: {e}"))
}

fn field_obj<'a>(v: &'a Json, name: &str) -> Result<&'a Json, String> {
    let f = v.get(name).ok_or_else(|| format!("line missing `{name}`"))?;
    if !f.is_object() {
        return Err(format!("`{name}` must be a JSON object"));
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhypedb_engine::logical::LogicalExportOptions;
    use rhypedb_engine::object::Value;

    fn export_to_string(db: &Database) -> String {
        let mut out = Vec::new();
        db.logical_export_stream(&mut out, &LogicalExportOptions::default())
            .unwrap();
        String::from_utf8(out).unwrap()
    }

    fn vec_bytes(base: f32) -> Bytes {
        let mut b = bytes::BytesMut::new();
        for i in 0..4 {
            b.extend_from_slice(&(base + i as f32).to_be_bytes());
        }
        b.freeze()
    }

    #[test]
    fn import_round_trips_full_export() {
        // Source DB: scalars, a forward relation with edge fields, an @inverse
        // back-ref, a self-relation CYCLE, and raw vectors.
        let src_dir = tempfile::tempdir().unwrap();
        let sdl = r#"
            type User {
                name: String @unique
                age: u32
                posts: [Post] @inverse(Post.author)
                buddy: User
                embedding: Vector<4>
            }
            type Post {
                title: String
                rating: f64
                author: User { weight: f32 }
            }
        "#;
        let db = Database::open(parse_schema(sdl).unwrap(), src_dir.path()).unwrap();

        let mk = |db: &Database, ty: &str, pairs: Vec<(&str, Value)>| {
            let mut f = FieldMap::new();
            for (k, v) in pairs {
                f.insert(k.into(), v);
            }
            db.create(ty, f).unwrap().id
        };
        let u1 = mk(&db, "User", vec![("name", Value::String("Ada".into())), ("age", Value::U32(36))]);
        let u2 = mk(&db, "User", vec![("name", Value::String("Alan".into())), ("age", Value::U32(41))]);
        let p1 = mk(&db, "Post", vec![("title", Value::String("OCN".into())), ("rating", Value::F64(4.5))]);
        let p2 = mk(&db, "Post", vec![("title", Value::String("Mind".into())), ("rating", Value::F64(3.0))]);
        let mut weight = FieldMap::new();
        weight.insert("weight".into(), Value::F32(0.5));
        db.link("Post", p1, "author", u1, Some(weight)).unwrap();
        db.link("Post", p2, "author", u2, None).unwrap();
        // Self-relation CYCLE: u1.buddy -> u2 -> u1.
        db.link("User", u1, "buddy", u2, None).unwrap();
        db.link("User", u2, "buddy", u1, None).unwrap();
        db.restore_vectors("User", "embedding", &[(u1, vec_bytes(1.0)), (u2, vec_bytes(5.0))])
            .unwrap();

        let export1 = export_to_string(&db);
        let export_file = src_dir.path().join("dump.ndjson");
        std::fs::write(&export_file, &export1).unwrap();

        // Import into a fresh dir.
        let imp_dir = tempfile::tempdir().unwrap();
        let report = run_import(
            &export_file,
            imp_dir.path(),
            &ImportOptions { force: false, vectors: VectorImportMode::Raw },
        )
        .unwrap();
        assert_eq!(report.objects, 4);
        assert_eq!(report.edges, 4, "2 author + 2 buddy; @inverse posts not emitted");
        assert_eq!(report.vectors, 2);
        assert_eq!(report.types, 2);
        assert!(imp_dir.path().join("schema.rhype").is_file(), "schema written for startup");

        // Re-open + re-export: the data lines (everything past the header's
        // timestamp/snapshot) must match the original byte-for-byte — proving
        // objects (ids + scalars), forward edges (+ edge_fields + the cycle), and
        // raw vectors all round-tripped.
        let db2 = Database::open(parse_schema(sdl).unwrap(), imp_dir.path()).unwrap();
        let export2 = export_to_string(&db2);
        let lines1: Vec<&str> = export1.lines().skip(1).collect();
        let lines2: Vec<&str> = export2.lines().skip(1).collect();
        assert_eq!(lines1, lines2, "re-export of the imported DB must match the original");

        // Explicit spot-checks for readability.
        assert_eq!(
            db2.get("User", u1).unwrap().fields.get("name"),
            Some(&Value::String("Ada".into()))
        );
        let buddy = |id| -> Vec<u64> {
            db2.get_links("User", id, "buddy").unwrap().into_iter().map(|(i, _)| i).collect()
        };
        assert_eq!(buddy(u1), vec![u2], "cycle edge u1->u2 preserved");
        assert_eq!(buddy(u2), vec![u1], "cycle edge u2->u1 preserved");
        let author = db2.get_links("Post", p1, "author").unwrap();
        assert_eq!(author[0].0, u1);
        assert_eq!(author[0].1.get("weight"), Some(&Value::F32(0.5)), "edge field preserved");
        let posts: Vec<u64> = db2.get_links("User", u1, "posts").unwrap().into_iter().map(|(i, _)| i).collect();
        assert_eq!(posts, vec![p1], "@inverse reconstructed from the forward edge");

        // @unique still enforced; next_object_id seeded past the restored max.
        let mut dup = FieldMap::new();
        dup.insert("name".into(), Value::String("Ada".into()));
        assert!(db2.create("User", dup).is_err());
        let mut nf = FieldMap::new();
        nf.insert("name".into(), Value::String("New".into()));
        nf.insert("age".into(), Value::U32(1));
        assert!(db2.create("User", nf).unwrap().id > p2, "ids continue past the restored max");
    }

    #[test]
    fn import_refuses_truncated_and_nonempty() {
        let dir = tempfile::tempdir().unwrap();
        let sdl = r#"type T { n: u32 }"#;
        let db = Database::open(parse_schema(sdl).unwrap(), dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("n".into(), Value::U32(1));
        db.create("T", f).unwrap();
        let full = export_to_string(&db);
        let file = dir.path().join("dump.ndjson");

        // Truncated (drop the trailer) → refused, nothing materialized.
        let truncated: String = full
            .lines()
            .filter(|l| !l.contains("\"trailer\""))
            .map(|l| format!("{l}\n"))
            .collect();
        std::fs::write(&file, &truncated).unwrap();
        let target = tempfile::tempdir().unwrap();
        let r = run_import(
            &file,
            target.path(),
            &ImportOptions { force: false, vectors: VectorImportMode::Raw },
        );
        assert!(r.is_err() && r.unwrap_err().contains("trailer"));

        // A complete file into a NON-EMPTY dir without --force is refused.
        std::fs::write(&file, &full).unwrap();
        std::fs::write(target.path().join("stray.txt"), b"x").unwrap();
        let r = run_import(
            &file,
            target.path(),
            &ImportOptions { force: false, vectors: VectorImportMode::Raw },
        );
        assert!(r.is_err() && r.unwrap_err().contains("not empty"));
        // With --force it succeeds and FULLY REPLACES the target (the stray file
        // is gone — the staging dir is swapped in, not merged).
        let r = run_import(
            &file,
            target.path(),
            &ImportOptions { force: true, vectors: VectorImportMode::Raw },
        );
        assert!(r.is_ok(), "got {r:?}");
        assert!(
            !target.path().join("stray.txt").exists(),
            "--force replaces the dir wholesale; no stale files survive"
        );
    }

    #[test]
    fn import_bad_schema_leaves_target_untouched() {
        // A dump that passes structural validation but whose embedded SDL is
        // unparseable must be refused BEFORE the target dir is touched (the
        // destructive swap happens only after a full successful import).
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(parse_schema("type T { n: u32 }").unwrap(), dir.path()).unwrap();
        let mut f = FieldMap::new();
        f.insert("n".into(), Value::U32(1));
        db.create("T", f).unwrap();
        let full = export_to_string(&db);

        // Corrupt the schema line's SDL but keep counts + trailer valid.
        let bad: String = full
            .lines()
            .map(|l| {
                if l.contains("\"kind\":\"schema\"") {
                    r#"{"kind":"schema","sdl":"type type !!! not valid"}"#.to_string()
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        let file = dir.path().join("bad.ndjson");
        std::fs::write(&file, &bad).unwrap();

        // Pre-populate the target (inside a controlled base so we can check for a
        // leaked staging dir as a sibling).
        let base = tempfile::tempdir().unwrap();
        let target = base.path().join("data");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("marker.txt"), b"keep me").unwrap();

        let r = run_import(
            &file,
            &target,
            &ImportOptions { force: true, vectors: VectorImportMode::Raw },
        );
        assert!(r.is_err() && r.unwrap_err().contains("schema"), "bad SDL must be rejected");
        assert!(target.join("marker.txt").is_file(), "target untouched on failure");
        let leaked = std::fs::read_dir(base.path())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with(".rhypedb-"))
            .count();
        assert_eq!(leaked, 0, "no staging dir leaked");
    }
}
