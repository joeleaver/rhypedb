//! A minimal first-party Rust subscription client for rhypedb's binary TCP
//! protocol. It opens a DEDICATED subscription connection, registers one filter,
//! and prints each pushed change event (one JSON line per event) until Ctrl-C.
//!
//! It reuses the server's own `protocol` codec, so there is no risk of the wire
//! format drifting between server and client.
//!
//! ```text
//! cargo run -p rhypedb-server --example subscribe -- 127.0.0.1:4201
//! cargo run -p rhypedb-server --example subscribe -- 127.0.0.1:4201 --type User
//! cargo run -p rhypedb-server --example subscribe -- 127.0.0.1:4201 --type User --id 42
//! cargo run -p rhypedb-server --example subscribe -- 127.0.0.1:4201 --kind create,delete
//! ```
//!
//! Each printed line is the canonical `WireEvent` JSON, e.g.
//! `{"v":"rhypedb-event-v1","kind":"create","type":"User","id":"42","version":"7","fields":{...}}`.
//! `id`/`version` are decimal strings (lossless even past 2^53); `fields` are
//! best-effort (Bytes elided, no fields on delete) — re-query for authoritative
//! state, and treat a `[lagged]` notice as "reconcile by re-query".

use std::env;

use rhypedb_server::protocol;
use rhypedb_subscribe::{ChangeKind, SubscriptionFilter};
use tokio::io::{BufReader, BufWriter};
use tokio::net::TcpStream;

const USAGE: &str =
    "usage: subscribe <host:port> [--type T] [--id N] [--kind create,update,delete]";

#[tokio::main]
async fn main() {
    let mut args = env::args().skip(1);
    let addr = match args.next() {
        Some(a) => a,
        None => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    };

    let mut type_name: Option<String> = None;
    let mut object_id: Option<u64> = None;
    let mut kinds: Vec<ChangeKind> = Vec::new();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--type" => type_name = args.next(),
            "--id" => {
                object_id = match args.next().and_then(|s| s.parse().ok()) {
                    Some(id) => Some(id),
                    None => {
                        eprintln!("--id needs a u64\n{USAGE}");
                        std::process::exit(2);
                    }
                }
            }
            "--kind" => {
                let Some(list) = args.next() else {
                    eprintln!("--kind needs a comma list\n{USAGE}");
                    std::process::exit(2);
                };
                for k in list.split(',') {
                    match k.trim() {
                        "create" => kinds.push(ChangeKind::Create),
                        "update" => kinds.push(ChangeKind::Update),
                        "delete" => kinds.push(ChangeKind::Delete),
                        other => {
                            eprintln!("unknown kind '{other}' (create|update|delete)");
                            std::process::exit(2);
                        }
                    }
                }
            }
            other => {
                eprintln!("unknown argument '{other}'\n{USAGE}");
                std::process::exit(2);
            }
        }
    }

    let filter = SubscriptionFilter {
        type_name,
        object_id,
        kinds,
    };

    let stream = TcpStream::connect(&addr).await.unwrap_or_else(|e| {
        eprintln!("connect {addr}: {e}");
        std::process::exit(1);
    });
    let (read, write) = stream.into_split();
    let mut reader = BufReader::new(read);
    let mut writer = BufWriter::new(write);

    // The subscribe frame's req_id is the subscription HANDLE: every pushed
    // Event/SubLagged echoes it, and an Unsubscribe would target it.
    const HANDLE: u32 = 1;
    let payload = protocol::encode_subscribe_filter(&filter);
    if let Err(e) =
        protocol::write_frame(&mut writer, HANDLE, protocol::REQ_SUBSCRIBE, &payload).await
    {
        eprintln!("send subscribe: {e}");
        std::process::exit(1);
    }
    eprintln!("subscribed to {addr} (Ctrl-C to exit)");

    // Demultiplex by KIND (never by req_id): Event/SubLagged are server-initiated.
    loop {
        let frame = tokio::select! {
            r = protocol::read_frame(&mut reader) => match r {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("connection closed: {e}");
                    break;
                }
            },
            _ = tokio::signal::ctrl_c() => {
                eprintln!("bye");
                break;
            }
        };

        match frame.kind {
            protocol::RESP_SUBSCRIBED => {
                eprintln!("[subscribed] handle={}", frame.req_id);
            }
            protocol::RESP_EVENT => {
                // The payload is the canonical WireEvent JSON; print it verbatim.
                println!("{}", String::from_utf8_lossy(&frame.payload));
            }
            protocol::RESP_SUBLAGGED => {
                eprintln!(
                    "[lagged] handle={} — the buffer overflowed and events were dropped; \
                     re-query to reconcile",
                    frame.req_id
                );
            }
            protocol::RESP_ERROR => {
                let msg = protocol::decode_error_payload(&frame.payload).unwrap_or_default();
                eprintln!("[error] {msg}");
                std::process::exit(1);
            }
            other => eprintln!("[unexpected frame kind 0x{other:02x}]"),
        }
    }
}
