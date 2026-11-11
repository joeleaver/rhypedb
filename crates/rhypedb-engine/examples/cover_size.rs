//! GH #26 measurement: bytes per keyspace for a mail-shaped store, plus the
//! cost of the inverse-link listing an embedder (Postrinch) runs most.
//!
//! The shape follows the real report: `Message` has two to-one relations
//! (`thread`, `account`), three to-many (`labels`, `from`, `recipients`),
//! five `@indexed` fields, and several KB of text. Each reverse edge of a
//! to-many link used to carry a full copy of the message, and each index
//! entry another.
//!
//! ```text
//! cargo run --release -p rhypedb-engine --example cover_size -- [messages] [cover_max_bytes]
//! ```
//! `cover_max_bytes` defaults to the engine default; pass `max` to copy every
//! object (the pre-cap index behaviour).

use rhypedb_engine::database::{Database, OpenOptions};
use rhypedb_engine::object::{FieldMap, Value};
use rhypedb_schema::parser::parse_schema;
use std::time::Instant;

const SCHEMA: &str = r#"
type Account { email: String  messages: [Thread] @inverse(Thread.account) }
type Thread { subject: String  account: Account  messages: [Message] @inverse(Message.thread) }
type Label { name: String  messages: [Message] @inverse(Message.labels) }
type Contact { address: String  name: String }
type Message {
    uid: u64 @indexed
    date: u64 @indexed
    size: u32 @indexed
    flags: u32 @indexed
    message_id: String @indexed
    subject: String
    body_text: String
    search_text: String
    headers: String
    thread: Thread
    account: Account
    labels: [Label]
    from: [Contact]
    recipients: [Contact]
}
"#;

fn proc_status(key: &str) -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(key)).map(|l| l.to_string()))
        .unwrap_or_else(|| format!("{key} n/a"))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: u64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(3_715);
    let cap = match args.next().as_deref() {
        None => OpenOptions::default().cover_max_bytes,
        Some("max") => usize::MAX,
        Some(s) => s.parse().unwrap(),
    };
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_options(
        parse_schema(SCHEMA).unwrap(),
        dir.path(),
        OpenOptions { cover_max_bytes: cap, sync_on_commit: false, ..Default::default() },
    )
    .unwrap();

    let mk = |t: &str, f: Vec<(&str, Value)>| {
        let mut m = FieldMap::new();
        for (k, v) in f {
            m.insert(k.into(), v);
        }
        db.create(t, m).unwrap().id
    };
    let account = mk("Account", vec![("email", Value::String("me@example.com".into()))]);
    let labels: Vec<u64> = ["INBOX", "Archive", "Work", "Family", "Receipts"]
        .iter()
        .map(|l| mk("Label", vec![("name", Value::String((*l).into()))]))
        .collect();
    let contacts: Vec<u64> = (0..200)
        .map(|i| {
            mk(
                "Contact",
                vec![
                    ("address", Value::String(format!("person{i}@example.org"))),
                    ("name", Value::String(format!("Person {i}"))),
                ],
            )
        })
        .collect();
    let threads: Vec<u64> = (0..n / 3 + 1)
        .map(|i| {
            let t = mk("Thread", vec![("subject", Value::String(format!("Thread {i}")))]);
            db.link("Thread", t, "account", account, None).unwrap();
            t
        })
        .collect();

    // ~7 KB per message, like the reported 26.5 MB / 3,715.
    let para = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod \
                tempor incididunt ut labore et dolore magna aliqua. ";
    let t0 = Instant::now();
    for i in 0..n {
        let body = format!("{i} {}", para.repeat(25));
        let m = mk(
            "Message",
            vec![
                ("uid", Value::U64(i)),
                ("date", Value::U64(1_700_000_000 + i * 37)),
                ("size", Value::U32(7000 + (i % 900) as u32)),
                ("flags", Value::U32((i % 4) as u32)),
                ("message_id", Value::String(format!("<{i}@mail.example.org>"))),
                ("subject", Value::String(format!("Subject {i}"))),
                ("search_text", Value::String(body.clone())),
                ("body_text", Value::String(body)),
                ("headers", Value::String("Received: from mx.example.org\r\n".repeat(30))),
            ],
        );
        db.link("Message", m, "thread", threads[(i / 3) as usize], None).unwrap();
        db.link("Message", m, "account", account, None).unwrap();
        db.link("Message", m, "labels", labels[0], None).unwrap();
        db.link("Message", m, "labels", labels[1 + (i % 4) as usize], None).unwrap();
        db.link("Message", m, "from", contacts[(i % 200) as usize], None).unwrap();
        for r in 0..3 {
            db.link("Message", m, "recipients", contacts[((i * 7 + r) % 200) as usize], None)
                .unwrap();
        }
    }
    let load = t0.elapsed();
    db.storage().compact().unwrap();

    let snap = db.storage().read_snapshot();
    let mut total = 0u64;
    println!("messages {n}, cover_max_bytes {cap}, load {:.2?}", load);
    for (prefix, name) in [
        (b'o', "objects     o:"),
        (b'r', "rev edges   r:"),
        (b'i', "field index i:"),
        (b'e', "edges       e:"),
    ] {
        let rows = db.storage().scan_prefix_at(snap, &[prefix]).unwrap();
        let bytes: u64 = rows.iter().map(|(k, v)| (k.len() + v.len()) as u64).sum();
        total += bytes;
        println!("{name}  {:>8} rows  {:>9.1} MB", rows.len(), bytes as f64 / 1e6);
    }
    let disk: u64 = std::fs::read_dir(dir.path().join("sst"))
        .unwrap()
        .filter_map(|e| e.ok()?.metadata().ok())
        .map(|m| m.len())
        .sum();
    println!("keyspaces above {:>9.1} MB, SST files on disk {:.1} MB", total as f64 / 1e6, disk as f64 / 1e6);

    // Postrinch's listing: every label's messages, then each message's thread.
    let t0 = Instant::now();
    let mut seen = 0usize;
    for &l in &labels {
        for (m, _) in db.get_links("Label", l, "messages").unwrap() {
            seen += db.get_links("Message", m, "thread").unwrap().len();
        }
    }
    println!("label listing: {seen} links in {:.2?}", t0.elapsed());
    let t0 = Instant::now();
    let hits = db
        .filter_scan("Message", "date", rhypedb_storage::zone::CompareOp::Ge, 0, None)
        .unwrap()
        .len();
    println!("indexed full range read: {hits} rows in {:.2?}", t0.elapsed());
    println!("{}  {}  {}", proc_status("VmRSS"), proc_status("RssAnon"), proc_status("RssFile"));
}
