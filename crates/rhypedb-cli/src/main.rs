
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "rhypedb-cli", about = "rhypedb interactive client")]
struct Cli {
    /// Server URL.
    #[arg(short = 'H', long, default_value = "http://127.0.0.1:4200")]
    host: String,

    /// Execute a single query and exit.
    #[arg(short, long)]
    execute: Option<String>,

    /// Admin token for `migrate` subcommands (sent as `Authorization: Bearer`).
    #[arg(long, env = "RHYPEDB_ADMIN_TOKEN")]
    admin_token: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Field-type migration admin (talks to the server's /admin/migrations API).
    Migrate {
        #[command(subcommand)]
        action: MigrateAction,
    },
    /// Physical backup of the live database (talks to /admin/backup).
    Backup {
        /// Write the snapshot to this path ON THE SERVER's filesystem.
        #[arg(long, conflicts_with = "download")]
        dest: Option<String>,
        /// Stream the backup and unpack it into this LOCAL directory.
        #[arg(long, conflicts_with = "dest")]
        download: Option<PathBuf>,
        /// Optional label folded into the server-side snapshot dir name (--dest).
        #[arg(long, requires = "dest")]
        label: Option<String>,
    },
    /// Restore a snapshot directory into a fresh data dir (server must be STOPPED).
    Restore {
        /// A snapshot directory (downloaded via `backup --download`, or copied
        /// from a server `backup --dest`).
        src: PathBuf,
        /// The data dir to materialize (must be empty unless --force).
        data_dir: PathBuf,
        /// Overwrite even if the data dir is non-empty.
        #[arg(long)]
        force: bool,
    },
    /// Verify a local snapshot directory (MANIFEST + listed files present).
    Verify {
        /// The snapshot directory to check.
        dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum MigrateAction {
    /// Start a migration.
    Start {
        #[arg(long = "type")]
        type_name: String,
        #[arg(long)]
        field: String,
        /// Target scalar kind (e.g. f64, i64, String).
        #[arg(long)]
        to: String,
        /// Built-in converter name (e.g. widen_int_to_f64).
        #[arg(long)]
        converter: String,
        #[arg(long, default_value_t = 1)]
        converter_version: u32,
        #[arg(long, default_value_t = 0)]
        chunk: u64,
        #[arg(long)]
        parallel: Option<u8>,
        /// Per-row failure policy: stop | skip | quarantine.
        #[arg(long)]
        policy: Option<String>,
        /// Quarantine/error cap (0 = engine default 100K); exceeding it auto-stops.
        #[arg(long, default_value_t = 0)]
        quarantine_cap: u64,
        #[arg(long)]
        dry_run: bool,
    },
    /// Show a tabular list of migrations, or detail for one id.
    Status { id: Option<u64> },
    /// Pause a running migration (resumable).
    Pause { id: u64 },
    /// Resume a paused / parked migration.
    Resume { id: u64 },
    /// Terminally cancel a migration (rolls back partial shadows).
    Cancel { id: u64 },
    /// Explicitly run the cutover for a backfilled migration.
    Cutover { id: u64 },
    /// Quarantine triage.
    Quarantine {
        #[command(subcommand)]
        action: QuarantineAction,
    },
    /// Stream the live event feed (SSE) until the migration settles.
    Events { id: u64 },
}

#[derive(Subcommand)]
enum QuarantineAction {
    /// List a plan's quarantined rows.
    List { id: u64 },
    /// Re-run a fixed converter over the quarantined rows.
    Retry {
        id: u64,
        #[arg(long)]
        new_converter: String,
    },
}

#[derive(Serialize)]
struct QueryRequest {
    query: String,
}

#[derive(Deserialize)]
struct QueryResponse {
    objects: Option<Vec<ObjectJson>>,
    object: Option<ObjectJson>,
    ok: Option<bool>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct ObjectJson {
    #[serde(rename = "type")]
    type_name: String,
    id: u64,
    fields: std::collections::HashMap<String, serde_json::Value>,
}

fn send_query(host: &str, query: &str) -> Result<QueryResponse, String> {
    let url = format!("{host}/query");
    let req_body = QueryRequest {
        query: query.to_string(),
    };

    let mut response = ureq::post(&url)
        .send_json(&req_body)
        .map_err(|e| format!("connection error: {e}"))?;

    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("read error: {e}"))?;

    serde_json::from_str(&text).map_err(|e| format!("parse error: {e}"))
}

fn print_response(resp: &QueryResponse) {
    if let Some(error) = &resp.error {
        eprintln!("ERROR: {error}");
        return;
    }

    if let Some(objects) = &resp.objects {
        if objects.is_empty() {
            println!("(no results)");
            return;
        }
        for obj in objects {
            print_object(obj);
        }
        println!("({} object{})", objects.len(), if objects.len() == 1 { "" } else { "s" });
    } else if let Some(obj) = &resp.object {
        print_object(obj);
    } else if resp.ok == Some(true) {
        println!("OK");
    }
}

fn print_object(obj: &ObjectJson) {
    println!("{}:{}", obj.type_name, obj.id);
    let mut keys: Vec<_> = obj.fields.keys().collect();
    keys.sort();
    for key in keys {
        let val = &obj.fields[key];
        let formatted = match val {
            serde_json::Value::String(s) => format!("\"{s}\""),
            other => other.to_string(),
        };
        println!("  {key}: {formatted}");
    }
}

// ---------------------------------------------------------------------------
// Migration admin client (card 5)
// ---------------------------------------------------------------------------

/// GET a JSON admin endpoint, returning `(status, body)`. HTTP status is not
/// treated as an error so a 4xx body (the server's `{"error": ...}`) is readable.
fn admin_get(cli: &Cli, path: &str) -> Result<(u16, serde_json::Value), String> {
    let url = format!("{}{path}", cli.host);
    let mut req = ureq::get(&url)
        .config()
        .http_status_as_error(false)
        .timeout_global(Some(std::time::Duration::from_secs(30)))
        .build();
    if let Some(t) = &cli.admin_token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let mut resp = req.call().map_err(|e| format!("connection error: {e}"))?;
    let status = resp.status().as_u16();
    let text = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("read error: {e}"))?;
    let json = serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
    Ok((status, json))
}

/// POST a JSON body to an admin endpoint, returning `(status, body)`.
fn admin_post(
    cli: &Cli,
    path: &str,
    body: serde_json::Value,
) -> Result<(u16, serde_json::Value), String> {
    let url = format!("{}{path}", cli.host);
    let mut req = ureq::post(&url)
        .config()
        .http_status_as_error(false)
        .timeout_global(Some(std::time::Duration::from_secs(30)))
        .build();
    if let Some(t) = &cli.admin_token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let mut resp = req
        .send_json(&body)
        .map_err(|e| format!("connection error: {e}"))?;
    let status = resp.status().as_u16();
    let text = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("read error: {e}"))?;
    let json = serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
    Ok((status, json))
}

/// Render a non-2xx response as an error string: the server's JSON `error`
/// field, or a plain-text body (the auth 401/403 responses aren't JSON), else
/// the bare status.
fn http_error(status: u16, body: &serde_json::Value) -> String {
    let msg = body
        .get("error")
        .and_then(|e| e.as_str())
        .or_else(|| body.as_str().map(str::trim).filter(|s| !s.is_empty()))
        .unwrap_or("(no detail)");
    format!("server returned {status}: {msg}")
}

fn run_migrate(cli: &Cli, action: &MigrateAction) -> Result<(), String> {
    match action {
        MigrateAction::Start {
            type_name,
            field,
            to,
            converter,
            converter_version,
            chunk,
            parallel,
            policy,
            quarantine_cap,
            dry_run,
        } => {
            let mut body = serde_json::json!({
                "type": type_name, "field": field, "to": to,
                "converter": converter, "converter_version": converter_version,
                "chunk": chunk, "quarantine_cap": quarantine_cap, "dry_run": dry_run,
            });
            if let Some(p) = parallel {
                body["parallel"] = serde_json::json!(p);
            }
            if let Some(p) = policy {
                body["policy"] = serde_json::json!(p);
            }
            let (status, resp) = admin_post(cli, "/admin/migrations", body)?;
            if status != 200 {
                return Err(http_error(status, &resp));
            }
            println!(
                "started migration {} (created_at_ms {})",
                resp["migration_id"], resp["created_at_ms"]
            );
        }
        MigrateAction::Status { id: Some(id) } => {
            let (status, resp) = admin_get(cli, &format!("/admin/migrations/{id}"))?;
            if status != 200 {
                return Err(http_error(status, &resp));
            }
            print_progress(&resp);
        }
        MigrateAction::Status { id: None } => {
            let (status, resp) = admin_get(cli, "/admin/migrations")?;
            if status != 200 {
                return Err(http_error(status, &resp));
            }
            print_migration_table(resp["migrations"].as_array().unwrap_or(&vec![]));
        }
        MigrateAction::Pause { id } => verb(cli, &format!("/admin/migrations/{id}/pause"), "paused")?,
        MigrateAction::Resume { id } => {
            verb(cli, &format!("/admin/migrations/{id}/resume"), "resuming")?
        }
        MigrateAction::Cancel { id } => {
            verb(cli, &format!("/admin/migrations/{id}/cancel"), "cancelling")?
        }
        MigrateAction::Cutover { id } => {
            verb(cli, &format!("/admin/migrations/{id}/cutover"), "cutting over")?
        }
        MigrateAction::Quarantine {
            action: QuarantineAction::List { id },
        } => {
            let (status, resp) = admin_get(cli, &format!("/admin/migrations/{id}/quarantine"))?;
            if status != 200 {
                return Err(http_error(status, &resp));
            }
            let rows = resp["quarantined"].as_array().cloned().unwrap_or_default();
            if rows.is_empty() {
                println!("(no quarantined rows)");
            }
            for r in &rows {
                println!(
                    "object {}: {} (converter {}, at {})",
                    r["object_id"], r["error"], r["attempted_converter"], r["errored_at_ms"]
                );
            }
        }
        MigrateAction::Quarantine {
            action: QuarantineAction::Retry { id, new_converter },
        } => {
            let (status, resp) = admin_post(
                cli,
                &format!("/admin/migrations/{id}/quarantine/retry"),
                serde_json::json!({ "new_converter": new_converter }),
            )?;
            if status != 200 {
                return Err(http_error(status, &resp));
            }
            println!("resolved {} quarantined row(s)", resp["resolved"]);
        }
        MigrateAction::Events { id } => stream_events(cli, *id)?,
    }
    Ok(())
}

fn verb(cli: &Cli, path: &str, label: &str) -> Result<(), String> {
    let (status, resp) = admin_post(cli, path, serde_json::json!({}))?;
    if status != 200 {
        return Err(http_error(status, &resp));
    }
    println!("OK ({label})");
    Ok(())
}

fn print_migration_table(rows: &[serde_json::Value]) {
    if rows.is_empty() {
        println!("(no migrations)");
        return;
    }
    println!(
        "{:>5}  {:<20} {:<16} {:>10}  CONVERTER",
        "ID", "TYPE.FIELD", "STATUS", "CONVERTED"
    );
    for r in rows {
        let tf = format!(
            "{}.{}",
            r["type"].as_str().unwrap_or("?"),
            r["field"].as_str().unwrap_or("?")
        );
        // Pull numbers out as primitives — serde_json::Value's Display ignores the
        // formatter width, so format them as integers for the columns to align.
        println!(
            "{:>5}  {:<20} {:<16} {:>10}  {}",
            r["plan_id"].as_u64().unwrap_or(0),
            tf,
            r["status"].as_str().unwrap_or("?"),
            r["objects_converted"].as_u64().unwrap_or(0),
            r["converter"].as_str().unwrap_or("?"),
        );
    }
}

fn print_progress(p: &serde_json::Value) {
    println!(
        "migration {} — {}.{}",
        p["plan_id"],
        p["type"].as_str().unwrap_or("?"),
        p["field"].as_str().unwrap_or("?")
    );
    println!("  status:    {}", p["status"].as_str().unwrap_or("?"));
    println!("  phase:     {}", p["phase"].as_str().unwrap_or("?"));
    println!(
        "  progress:  {} / {} objects ({} errors)",
        p["objects_converted"], p["total_objects"], p["errors"]
    );
    if let Some(rate) = p["objects_per_sec"].as_f64() {
        println!("  rate:      {rate:.1} obj/s");
    }
    if let Some(eta) = p["eta_unix_ms"].as_u64() {
        let now = p["now_ms"].as_u64().unwrap_or(eta);
        let remaining_s = eta.saturating_sub(now) as f64 / 1000.0;
        println!("  eta:       ~{remaining_s:.0}s (unix_ms {eta})");
    }
    if let Some(parts) = p["partitions"].as_array() {
        for pp in parts {
            println!(
                "  partition {}: [{}, {}) cursor {} converted {} done {}",
                pp["idx"], pp["lo"], pp["hi"], pp["cursor"], pp["objects_converted"], pp["done"]
            );
        }
    }
}

/// Consume the SSE stream, printing one line per event, until a terminal event
/// (cutover_done / failed / status_changed to a settled status) or EOF.
fn stream_events(cli: &Cli, id: u64) -> Result<(), String> {
    use std::io::BufRead;
    let url = format!("{}/admin/migrations/{id}/events", cli.host);
    let mut req = ureq::get(&url).config().http_status_as_error(false).build();
    if let Some(t) = &cli.admin_token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req.call().map_err(|e| format!("connection error: {e}"))?;
    let status = resp.status().as_u16();
    if status != 200 {
        return Err(format!("server returned {status}"));
    }
    let reader = std::io::BufReader::new(resp.into_body().into_reader());
    let mut saw_terminal = false;
    for line in reader.lines() {
        let line = line.map_err(|e| format!("stream error: {e}"))?;
        // SSE data frames: `data: {json}`. Skip event:/id:/keep-alive (`:`) lines.
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() {
            continue;
        }
        let ev: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let kind = ev["type"].as_str().unwrap_or("?");
        println!("{kind}: {ev}");
        // Failed / a settled status_changed ends the stream. (cutover_done is
        // always followed by a Completed status_changed, so it is not terminal.)
        let terminal = kind == "failed"
            || (kind == "status_changed"
                && matches!(
                    ev["status"].as_str(),
                    Some("Cancelled" | "Completed" | "DryRunCompleted")
                ));
        if terminal {
            saw_terminal = true;
            break;
        }
    }
    if !saw_terminal {
        eprintln!("(stream closed before the migration settled — re-run `migrate status {id}`)");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Backup / restore (Overboard cmqiioa2y)
// ---------------------------------------------------------------------------

fn run_backup(
    cli: &Cli,
    dest: &Option<String>,
    download: &Option<PathBuf>,
    label: &Option<String>,
) -> Result<(), String> {
    match (dest, download) {
        (Some(d), _) => {
            let mut body = serde_json::json!({ "dest": d });
            if let Some(l) = label {
                body["label"] = serde_json::Value::String(l.clone());
            }
            let (status, resp) = admin_post(cli, "/admin/backup", body)?;
            if !(200..300).contains(&status) {
                return Err(http_error(status, &resp));
            }
            let path = resp.get("path").and_then(|p| p.as_str()).unwrap_or("?");
            println!("backup written on server: {path}");
            if let Some(m) = resp.get("manifest") {
                println!(
                    "  ssts: {}  max_version: {}",
                    m.get("ssts").and_then(|s| s.as_array()).map_or(0, |a| a.len()),
                    m.get("max_version").and_then(|v| v.as_u64()).unwrap_or(0),
                );
            }
            Ok(())
        }
        (None, Some(dir)) => {
            download_backup(cli, dir)?;
            println!("backup downloaded + unpacked to {}", dir.display());
            println!(
                "restore with: rhypedb-cli restore {} <new-data-dir>",
                dir.display()
            );
            Ok(())
        }
        (None, None) => Err("specify --dest <server-path> or --download <local-dir>".into()),
    }
}

/// Stream `GET /admin/backup/stream` and unpack the tar into `into`.
fn download_backup(cli: &Cli, into: &Path) -> Result<(), String> {
    let url = format!("{}/admin/backup/stream", cli.host);
    let mut req = ureq::get(&url)
        .config()
        .http_status_as_error(false)
        .build();
    if let Some(t) = &cli.admin_token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let mut resp = req.call().map_err(|e| format!("connection error: {e}"))?;
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        let text = resp.body_mut().read_to_string().unwrap_or_default();
        return Err(format!("server returned {status}: {}", text.trim()));
    }
    let existed = into.exists();
    std::fs::create_dir_all(into).map_err(|e| format!("create {}: {e}", into.display()))?;
    let reader = resp.body_mut().as_reader();
    if let Err(e) = tar::Archive::new(reader).unpack(into) {
        // A truncated stream leaves a half-unpacked dir — remove what WE created.
        if !existed {
            let _ = std::fs::remove_dir_all(into);
        }
        return Err(format!("unpack tar (incomplete download): {e}"));
    }
    Ok(())
}

/// Files a complete snapshot must contain per its manifest. Empty = complete.
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

fn run_restore(src: &Path, data_dir: &Path, force: bool) -> Result<(), String> {
    // A complete backup has a MANIFEST.json (written last by the server).
    let manifest_text = std::fs::read_to_string(src.join("MANIFEST.json"))
        .map_err(|_| format!("not a valid backup: {}/MANIFEST.json missing", src.display()))?;
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_text).map_err(|e| format!("invalid MANIFEST.json: {e}"))?;

    // Refuse an INCOMPLETE source — else the restored DB silently loses data
    // (LsmTree::open has no manifest; it loads whatever SSTs are present).
    let missing = backup_missing_files(src, &manifest);
    if !missing.is_empty() {
        return Err(format!(
            "source backup is INCOMPLETE — missing: {}",
            missing.join(", ")
        ));
    }

    if data_dir.exists() {
        // A read_dir failure means we CAN'T confirm it's empty → require --force.
        let non_empty = std::fs::read_dir(data_dir)
            .map(|mut d| d.next().is_some())
            .unwrap_or(true);
        if non_empty && !force {
            return Err(format!(
                "{} exists and is not empty (use --force to overwrite)",
                data_dir.display()
            ));
        }
    }

    // Clear any stale data first so a --force restore over a DIFFERENT database
    // can't mix the two: LsmTree::open loads EVERY *.sst in sst/ (no manifest
    // file), so a leftover foreign SST would corrupt the restore.
    let _ = std::fs::remove_dir_all(data_dir.join("sst"));
    let _ = std::fs::remove_file(data_dir.join("wal.log"));
    if let Ok(rd) = std::fs::read_dir(data_dir) {
        for entry in rd.flatten() {
            let n = entry.file_name();
            let n = n.to_string_lossy();
            if n.starts_with("hnsw_") && n.ends_with(".bin") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    std::fs::create_dir_all(data_dir.join("sst")).map_err(|e| format!("create data dir: {e}"))?;

    // COPY (not link) so the restored dir is independent of the snapshot.
    let mut sst_count = 0u64;
    for entry in std::fs::read_dir(src.join("sst")).map_err(|e| format!("read src/sst: {e}"))? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.path().extension().and_then(|e| e.to_str()) == Some("sst") {
            std::fs::copy(entry.path(), data_dir.join("sst").join(entry.file_name()))
                .map_err(|e| e.to_string())?;
            sst_count += 1;
        }
    }
    std::fs::copy(src.join("wal.log"), data_dir.join("wal.log"))
        .map_err(|e| format!("copy wal.log: {e}"))?;
    // hnsw_*.bin + the schema, so the data dir is self-contained + startable.
    for entry in std::fs::read_dir(src).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        let n = name.to_string_lossy();
        if (n.starts_with("hnsw_") && n.ends_with(".bin")) || n == "schema.rhype" {
            std::fs::copy(entry.path(), data_dir.join(&*n)).map_err(|e| e.to_string())?;
        }
    }

    println!("restored {sst_count} SSTs + WAL to {}", data_dir.display());
    if let Some(arr) = manifest
        .get("in_flight_migrations")
        .and_then(|v| v.as_array())
        .filter(|a| !a.is_empty())
    {
        println!(
            "NOTE: this backup is MID-MIGRATION ({} plan(s)); register these converters before starting:",
            arr.len()
        );
        for m in arr {
            println!(
                "  plan {} converter {}",
                m.get("plan_id").and_then(|v| v.as_u64()).unwrap_or(0),
                m.get("converter").and_then(|v| v.as_str()).unwrap_or("?"),
            );
        }
    }
    println!("start the server with:");
    println!(
        "  rhypedb-server --schema {} --data-dir {}",
        data_dir.join("schema.rhype").display(),
        data_dir.display()
    );
    Ok(())
}

fn run_verify(dir: &Path) -> Result<(), String> {
    let manifest_text = std::fs::read_to_string(dir.join("MANIFEST.json"))
        .map_err(|_| "not a valid backup: MANIFEST.json missing".to_string())?;
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_text).map_err(|e| format!("invalid MANIFEST.json: {e}"))?;

    let missing = backup_missing_files(dir, &manifest);
    if !missing.is_empty() {
        return Err(format!("backup INCOMPLETE — missing: {}", missing.join(", ")));
    }
    println!(
        "backup OK: {} SSTs, max_version {}, created_at_ms {}",
        manifest.get("ssts").and_then(|v| v.as_array()).map_or(0, |a| a.len()),
        manifest.get("max_version").and_then(|v| v.as_u64()).unwrap_or(0),
        manifest.get("created_at_ms").and_then(|v| v.as_u64()).unwrap_or(0),
    );
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    if let Some(Commands::Migrate { action }) = &cli.command {
        if let Err(e) = run_migrate(&cli, action) {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
        return;
    }

    if let Some(cmd) = &cli.command {
        let result = match cmd {
            Commands::Backup {
                dest,
                download,
                label,
            } => run_backup(&cli, dest, download, label),
            Commands::Restore {
                src,
                data_dir,
                force,
            } => run_restore(src, data_dir, *force),
            Commands::Verify { dir } => run_verify(dir),
            Commands::Migrate { .. } => unreachable!("handled above"),
        };
        if let Err(e) = result {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
        return;
    }

    if let Some(query) = &cli.execute {
        match send_query(&cli.host, query) {
            Ok(resp) => print_response(&resp),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // REPL mode.
    println!("rhypedb client connected to {}", cli.host);
    println!("Type queries, or 'quit' to exit.\n");

    let mut rl = rustyline::DefaultEditor::new().unwrap();

    loop {
        let line = match rl.readline("rhypedb> ") {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Interrupted | rustyline::error::ReadlineError::Eof) => {
                println!("bye");
                break;
            }
            Err(e) => {
                eprintln!("readline error: {e}");
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "quit" || trimmed == "exit" {
            break;
        }

        let _ = rl.add_history_entry(trimmed);

        match send_query(&cli.host, trimmed) {
            Ok(resp) => print_response(&resp),
            Err(e) => eprintln!("{e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_fake_backup(dir: &Path) {
        std::fs::create_dir_all(dir.join("sst")).unwrap();
        std::fs::write(dir.join("sst").join("00000001.sst"), b"dummy-sst").unwrap();
        std::fs::write(dir.join("wal.log"), b"dummy-wal").unwrap();
        std::fs::write(dir.join("schema.rhype"), b"type T { x: i64 }").unwrap();
        let manifest = serde_json::json!({
            "format": "rhypedb-physical-backup-v1",
            "max_version": 7,
            "created_at_ms": 123,
            "ssts": ["00000001.sst"],
            "hnsw_files": [],
            "schema_file": "schema.rhype",
            "in_flight_migrations": [],
        });
        std::fs::write(
            dir.join("MANIFEST.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn verify_accepts_complete_and_rejects_missing_sst() {
        let dir = tempfile::tempdir().unwrap();
        write_fake_backup(dir.path());
        assert!(run_verify(dir.path()).is_ok());
        std::fs::remove_file(dir.path().join("sst").join("00000001.sst")).unwrap();
        let err = run_verify(dir.path()).unwrap_err();
        assert!(err.contains("INCOMPLETE"), "got: {err}");
    }

    #[test]
    fn restore_copies_into_fresh_dir_and_guards_nonempty() {
        let src = tempfile::tempdir().unwrap();
        write_fake_backup(src.path());
        let out = tempfile::tempdir().unwrap();
        let data_dir = out.path().join("restored");
        run_restore(src.path(), &data_dir, false).unwrap();
        assert!(data_dir.join("sst").join("00000001.sst").is_file());
        assert!(data_dir.join("wal.log").is_file());
        assert!(data_dir.join("schema.rhype").is_file());
        // Re-restoring into the now-non-empty dir is refused without --force.
        let err = run_restore(src.path(), &data_dir, false).unwrap_err();
        assert!(err.contains("not empty"), "got: {err}");
        // --force overrides.
        run_restore(src.path(), &data_dir, true).unwrap();
    }

    #[test]
    fn restore_rejects_dir_without_manifest() {
        let src = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(src.path().join("sst")).unwrap();
        let out = tempfile::tempdir().unwrap();
        let err = run_restore(src.path(), &out.path().join("x"), false).unwrap_err();
        assert!(err.contains("not a valid backup"), "got: {err}");
    }

    #[test]
    fn restore_force_clears_stale_ssts() {
        let src = tempfile::tempdir().unwrap();
        write_fake_backup(src.path()); // sst/00000001.sst
        let out = tempfile::tempdir().unwrap();
        let data_dir = out.path().join("restored");
        // Pre-populate the target with a STALE foreign SST + wal.
        std::fs::create_dir_all(data_dir.join("sst")).unwrap();
        std::fs::write(data_dir.join("sst").join("99999999.sst"), b"foreign").unwrap();
        std::fs::write(data_dir.join("wal.log"), b"old").unwrap();

        run_restore(src.path(), &data_dir, true).unwrap();
        assert!(data_dir.join("sst").join("00000001.sst").is_file());
        assert!(
            !data_dir.join("sst").join("99999999.sst").exists(),
            "a --force restore must clear the stale foreign SST (else open() mixes both DBs)"
        );
    }

    #[test]
    fn restore_refuses_incomplete_source() {
        let src = tempfile::tempdir().unwrap();
        write_fake_backup(src.path());
        // Remove the SST the manifest lists → the source is incomplete.
        std::fs::remove_file(src.path().join("sst").join("00000001.sst")).unwrap();
        let out = tempfile::tempdir().unwrap();
        let err = run_restore(src.path(), &out.path().join("x"), false).unwrap_err();
        assert!(err.contains("INCOMPLETE"), "got: {err}");
    }
}
