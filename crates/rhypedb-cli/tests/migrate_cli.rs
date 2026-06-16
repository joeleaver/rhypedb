//! Card 5 (5e) AC `cli_migrate_start_and_status_round_trip`: drive the
//! `rhypedb-cli migrate` subcommands (as a real subprocess) against a tiny mock
//! admin server, asserting the CLI sends the right request (method, path, auth
//! header, body) and formats the response in house style.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::sync::mpsc;

/// Handle one HTTP/1.1 connection: read the request (line + headers + body),
/// pick a canned JSON response by path, write it, and return the full raw
/// request for assertions.
fn handle_one(listener: &TcpListener) -> String {
    let (mut stream, _) = listener.accept().unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request = String::new();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap() == 0 {
            break;
        }
        if line == "\r\n" {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
        request.push_str(&line);
    }
    if content_length > 0 {
        let mut body = vec![0u8; content_length];
        reader.read_exact(&mut body).unwrap();
        request.push_str("\r\n");
        request.push_str(&String::from_utf8_lossy(&body));
    }
    let first = request.lines().next().unwrap_or("");
    let body = if first.starts_with("POST /admin/migrations ") {
        r#"{"migration_id":7,"created_at_ms":123}"#
    } else if first.contains("GET /admin/migrations/7 ") {
        r#"{"plan_id":7,"type":"User","field":"score","status":"Running","phase":"Converting","dry_run":false,"parallel_degree":4,"total_objects":100,"objects_converted":40,"errors":0,"created_at_ms":123,"now_ms":1123,"objects_per_sec":40.0,"eta_unix_ms":2623,"partitions":[]}"#
    } else {
        r#"{"migrations":[]}"#
    };
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(resp.as_bytes()).unwrap();
    stream.flush().unwrap();
    request
}

fn run_cli(host: &str, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rhypedb-cli"))
        .args(["-H", host])
        .args(["--admin-token", "tok"])
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn cli_migrate_start_and_status_round_trip() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let host = format!("http://{}", listener.local_addr().unwrap());

    // Mock server handles exactly two requests (start, then status) and reports
    // each raw request back over a channel.
    let (tx, rx) = mpsc::channel();
    let server = std::thread::spawn(move || {
        for _ in 0..2 {
            tx.send(handle_one(&listener)).unwrap();
        }
    });

    // --- start ---
    let out = run_cli(
        &host,
        &[
            "migrate", "start", "--type", "User", "--field", "score", "--to", "f64",
            "--converter", "widen_int_to_f64", "--chunk", "4", "--parallel", "8",
        ],
    );
    assert!(out.status.success(), "start failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("started migration 7"), "got: {stdout}");
    let start_req = rx.recv().unwrap();
    assert!(start_req.starts_with("POST /admin/migrations "), "{start_req}");
    assert!(
        start_req.to_ascii_lowercase().contains("authorization: bearer tok"),
        "missing auth header: {start_req}"
    );
    let compact: String = start_req.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(compact.contains("\"converter\":\"widen_int_to_f64\""), "{start_req}");
    assert!(compact.contains("\"parallel\":8"), "{start_req}");

    // --- status <id> ---
    let out = run_cli(&host, &["migrate", "status", "7"]);
    assert!(out.status.success(), "status failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("migration 7"), "got: {stdout}");
    assert!(stdout.contains("status:    Running"), "got: {stdout}");
    assert!(stdout.contains("40 / 100 objects"), "got: {stdout}");
    let status_req = rx.recv().unwrap();
    assert!(status_req.starts_with("GET /admin/migrations/7 "), "{status_req}");

    server.join().unwrap();
}
