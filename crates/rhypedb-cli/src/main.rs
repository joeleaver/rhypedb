
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
    let mut req = ureq::get(&url).config().http_status_as_error(false).build();
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
    let mut req = ureq::post(&url).config().http_status_as_error(false).build();
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

/// Render a non-2xx response as an error string (the server's `error` field).
fn http_error(status: u16, body: &serde_json::Value) -> String {
    let msg = body
        .get("error")
        .and_then(|e| e.as_str())
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
            dry_run,
        } => {
            let mut body = serde_json::json!({
                "type": type_name, "field": field, "to": to,
                "converter": converter, "converter_version": converter_version,
                "chunk": chunk, "dry_run": dry_run,
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
        println!(
            "{:>5}  {:<20} {:<16} {:>10}  {}",
            r["plan_id"],
            tf,
            r["status"].as_str().unwrap_or("?"),
            r["objects_converted"],
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
        let terminal = matches!(kind, "cutover_done" | "failed")
            || (kind == "status_changed"
                && matches!(
                    ev["status"].as_str(),
                    Some("Cancelled" | "Completed" | "DryRunCompleted")
                ));
        if terminal {
            break;
        }
    }
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
