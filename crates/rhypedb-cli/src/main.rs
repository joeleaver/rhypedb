
use clap::Parser;
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

fn main() {
    let cli = Cli::parse();

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
