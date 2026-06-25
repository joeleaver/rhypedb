//! Increment-6 capstone — the SHIP GATE for the crash-recovery-from-Pause epic.
//!
//! Increments 1–5 fuzzed the recovery boundaries *in process* via the `crash-fuzz`
//! feature: a thread-local injector panics at a wired site, the harness tears the
//! engine down without flushing, then reopens it and an oracle asserts no committed
//! data was lost or resurrected. That model is only trustworthy if it matches what
//! a *real* hard kill does. This test closes that loop end-to-end:
//!
//!   1. Spawn the actual `rhypedb-server` binary on a fresh data dir.
//!   2. Drive a workload over the real binary TCP protocol with the official client,
//!      so every write is server-acked (= committed + fsync'd) before we continue.
//!   3. Force a flush+compaction so part of the data lands in SSTs while the rest
//!      stays in the WAL — the realistic mixed on-disk state a cold boot must merge.
//!   4. `SIGKILL` the process (`Child::kill()` on unix). No SIGTERM, no graceful
//!      flush, no `Drop` — exactly the jkbase Firecracker Pause+SIGKILL route, and
//!      exactly what kill -9 / OOM does to any DB.
//!   5. Cold-restart the binary on the SAME data dir and assert, through a brand-new
//!      client, that EVERY committed write survived: scalar objects, the @unique and
//!      @indexed secondary indexes, and BYO vectors searchable via a rebuilt HNSW.
//!
//! No embedder / no network: the schema has a bare `Vector<4>` (no `@vectorize`), so
//! vectors are caller-supplied and nothing pulls in ONNX. Run the no-egress build with
//!     cargo test -p rhypedb-server --no-default-features --test crash_recovery_e2e
//! (the test is config-agnostic; `--no-default-features` only avoids the ONNX download
//! at build time — at runtime this test never embeds).

use std::collections::BTreeMap;
use std::fs::File;
use std::net::TcpListener as StdTcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use rhypedb_client::{Client, Error, Query, Row};
use serde_json::Value;

/// The schema both server runs are opened under. `User` exercises the @unique and
/// @indexed secondary-index keyspaces; `Doc` carries a bare (caller-supplied)
/// `Vector<4>` so the BYO-vector ingest + HNSW-rebuild-on-cold-boot path is live
/// without an embedder.
const SCHEMA: &str = r#"type User {
  name: String @unique
  age: u32 @indexed
  active: Bool
}
type Doc {
  label: String
  embedding: Vector<4>
}
"#;

const ADMIN_TOKEN: &str = "crash-e2e-token";

/// A spawned `rhypedb-server` process plus the bookkeeping to drive, diagnose, and
/// reap it.
struct ServerProc {
    child: Child,
    /// The binary-protocol address clients connect to.
    tcp_addr: String,
    /// The HTTP address (used for `/admin/compact`).
    http_addr: String,
    stdout_log: PathBuf,
    stderr_log: PathBuf,
}

impl ServerProc {
    /// Spawn the real server binary on `data_dir`, with freshly-chosen ephemeral
    /// ports (the binary binds the exact `--tcp-listen` string and only echoes
    /// that back, so `:0` would hide the real port — we pick concrete free ports
    /// instead). stdout/stderr go to files in `log_dir` so a full pipe can never
    /// deadlock the child and so failures can surface its output.
    fn spawn(data_dir: &Path, schema: &Path, log_dir: &Path, tag: &str) -> Self {
        let http_addr = format!("127.0.0.1:{}", free_port());
        let tcp_addr = format!("127.0.0.1:{}", free_port());
        let stdout_log = log_dir.join(format!("{tag}.out"));
        let stderr_log = log_dir.join(format!("{tag}.err"));
        let out = File::create(&stdout_log).expect("create stdout log");
        let err = File::create(&stderr_log).expect("create stderr log");

        let child = Command::new(env!("CARGO_BIN_EXE_rhypedb-server"))
            .arg("--schema")
            .arg(schema)
            .arg("--data-dir")
            .arg(data_dir)
            .arg("--listen")
            .arg(&http_addr)
            .arg("--tcp-listen")
            .arg(&tcp_addr)
            // Admin is gated by this token; it enables /admin/compact below.
            .env("RHYPEDB_ADMIN_TOKEN", ADMIN_TOKEN)
            // Defensive: never let an ambient config file or a stray restore env
            // var redirect this run (CLI flags already win over env for the
            // addresses/schema/data dir, but restore has no flag here). Durability
            // needs no env scrubbing: we never pass --no-sync, so sync_on_commit
            // stays on and every acked write is fsync'd before the kill.
            .env_remove("RHYPEDB_CONFIG")
            .env_remove("RHYPEDB_RESTORE_FROM")
            .env_remove("RHYPEDB_RESTORE_FROM_FORCE")
            .stdout(Stdio::from(out))
            .stderr(Stdio::from(err))
            .spawn()
            .expect("spawn rhypedb-server binary");

        ServerProc {
            child,
            tcp_addr,
            http_addr,
            stdout_log,
            stderr_log,
        }
    }

    /// Block until the server accepts a client and answers a ping, or panic with the
    /// captured server output. Fails fast if the process dies during startup (bad
    /// bind, schema error, …).
    fn connect_when_ready(&mut self) -> Client {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = self.child.try_wait().expect("poll child") {
                panic!(
                    "server exited before becoming ready (status {status})\n{}",
                    self.read_logs()
                );
            }
            match Client::connect(self.tcp_addr.as_str()).and_then(|c| c.ping().map(|()| c)) {
                Ok(client) => return client,
                Err(_) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => panic!(
                    "server never became ready within 30s: {e}\n{}",
                    self.read_logs()
                ),
            }
        }
    }

    /// Hard-kill the process (SIGKILL on unix) and reap it. Returns the exit status
    /// so callers can prove it was a signal kill, not a graceful exit.
    fn sigkill(&mut self) -> ExitStatus {
        self.child.kill().expect("kill server");
        self.child.wait().expect("reap server")
    }

    fn read_logs(&self) -> String {
        let out = std::fs::read_to_string(&self.stdout_log).unwrap_or_default();
        let err = std::fs::read_to_string(&self.stderr_log).unwrap_or_default();
        format!("--- server stdout ---\n{out}\n--- server stderr ---\n{err}")
    }
}

/// Discover a currently-free TCP port. There's a tiny TOCTOU window before the
/// server binds it, but it's the standard, good-enough approach for an out-of-process
/// test (and each spawn re-discovers, so a restart never reuses the dead run's port).
fn free_port() -> u16 {
    StdTcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("local_addr")
        .port()
}

/// POST `/admin/compact` (force-flush + full compaction) and assert it ran. After
/// this returns, everything written so far is durably in SSTs and the WAL is reset,
/// so subsequent writes land in a fresh WAL — giving the cold boot a genuine
/// SST-plus-WAL-replay merge to perform.
fn force_compact(http_addr: &str) {
    let body = ureq::post(&format!("http://{http_addr}/admin/compact"))
        .header("Authorization", &format!("Bearer {ADMIN_TOKEN}"))
        .send_empty()
        .expect("compact request")
        .body_mut()
        .read_json::<Value>()
        .expect("compact response json");
    assert_eq!(body["flush_ok"], true, "compact must force-flush");
    assert_eq!(body["compact_ok"], true, "compact must compact");
}

/// Count `.sst` files under `<data_dir>/sst`. Used to prove the flush actually
/// landed batch A on disk, so the cold boot is genuinely merging SST data with the
/// WAL tail (not silently testing an all-in-WAL state).
fn sst_count(data_dir: &Path) -> usize {
    let sst_dir = data_dir.join("sst");
    std::fs::read_dir(&sst_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "sst"))
                .count()
        })
        .unwrap_or(0)
}

/// `User.create(...)` over the real protocol; returns the new id.
fn create_user(c: &Client, name: &str, age: u32, active: bool) -> u64 {
    c.create(&Query::<Value>::raw(format!(
        "User.create({{ name: \"{name}\", age: {age}, active: {active} }})"
    )))
    .unwrap_or_else(|e| panic!("create user {name}: {e}"))
    .id
}

/// `Doc.create(...)` + a BYO vector ingest in one step; returns the new id.
fn create_doc_with_vector(c: &Client, label: &str, vector: [f32; 4]) -> u64 {
    let id = c
        .create(&Query::<Value>::raw(format!(
            "Doc.create({{ label: \"{label}\" }})"
        )))
        .unwrap_or_else(|e| panic!("create doc {label}: {e}"))
        .id;
    let ingested = c
        .ingest_vectors("Doc", "embedding", &[(id, vector)])
        .unwrap_or_else(|e| panic!("ingest vector for {label}: {e}"));
    assert_eq!(
        ingested, 1,
        "ingest_vectors must report one row for {label}"
    );
    id
}

/// Run a raw mutation (update/delete) and unwrap, panicking with context.
fn exec_raw(c: &Client, q: &str) {
    c.execute(&Query::<Value>::raw(q))
        .unwrap_or_else(|e| panic!("execute `{q}`: {e}"));
}

/// All `User` rows as a name → (age, active) map.
fn users_by_name(c: &Client) -> BTreeMap<String, (i64, bool)> {
    let rows: Vec<Row<Value>> = c.fetch(&Query::<Value>::all("User")).expect("fetch users");
    rows.into_iter()
        .map(|r| {
            let name = r.data["name"].as_str().expect("name").to_string();
            let age = r.data["age"].as_i64().expect("age");
            let active = r.data["active"].as_bool().expect("active");
            (name, (age, active))
        })
        .collect()
}

/// The single nearest `Doc` label to `probe` via the HNSW index.
fn nearest_doc_label(c: &Client, probe: &str) -> String {
    let hits: Vec<Row<Value>> = c
        .fetch(&Query::<Value>::raw(format!(
            "Doc.similar(.embedding, {probe}, k: 1)"
        )))
        .expect("similar query");
    assert_eq!(
        hits.len(),
        1,
        "expected exactly one nearest neighbour for {probe}"
    );
    hits[0].data["label"].as_str().expect("label").to_string()
}

/// On unix a real `Child::kill()` is SIGKILL — prove the process was signal-killed
/// (signal 9), not that it somehow exited cleanly (which would invalidate the
/// "matches a real kill" premise). No-op on non-unix.
fn assert_signal_killed(status: ExitStatus) {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(
            status.signal(),
            Some(9),
            "server should have been SIGKILLed, but exited with {status}"
        );
        assert!(
            !status.success(),
            "a SIGKILLed process must not report success"
        );
    }
    #[cfg(not(unix))]
    {
        let _ = status;
    }
}

/// THE capstone: a real binary, a real SIGKILL across a real SST+WAL boundary, a real
/// cold boot — every committed write recovered (scalars, @unique, @indexed, vectors).
#[test]
fn real_binary_sigkill_recovers_committed_data() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let logs = tempfile::tempdir().expect("log tempdir");
    let data_dir = tmp.path().join("data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let schema_path = tmp.path().join("schema.rhype");
    std::fs::write(&schema_path, SCHEMA).expect("write schema");

    // --- run #1: populate, force part to disk, then leave a WAL tail ---
    let mut server1 = ServerProc::spawn(&data_dir, &schema_path, logs.path(), "run1");
    let c1 = server1.connect_when_ready();

    // Batch A — flushed to SSTs by the compact below.
    let alice = create_user(&c1, "alice", 30, true);
    create_user(&c1, "bob", 70, false);
    let carol = create_user(&c1, "carol", 25, true);
    create_doc_with_vector(&c1, "doc-x", [1.0, 0.0, 0.0, 0.0]);
    create_doc_with_vector(&c1, "doc-y", [0.0, 1.0, 0.0, 0.0]);
    create_doc_with_vector(&c1, "doc-z", [0.0, 0.0, 1.0, 0.0]);

    // Drain batch A to SSTs and reset the WAL.
    force_compact(&server1.http_addr);
    assert!(
        sst_count(&data_dir) >= 1,
        "after a force-flush+compact, batch A must be persisted as at least one SST \
         (so the cold boot genuinely merges SST data with the WAL tail)"
    );

    // Batch B — lands in the fresh WAL only; cold boot must replay this over the SSTs.
    // The update/delete target batch-A rows by id (the e2e-proven `.get(id)` form), so
    // the cold boot must merge a WAL-resident mutation over an SST-resident base row.
    create_user(&c1, "dave", 80, true);
    create_user(&c1, "erin", 18, false);
    exec_raw(&c1, &format!("User.get({alice}).update({{ age: 99 }})"));
    exec_raw(&c1, &format!("User.get({carol}).delete()"));
    create_doc_with_vector(&c1, "doc-w", [0.0, 0.0, 0.0, 1.0]);

    // Sanity before the kill: the live server sees the post-batch-B state.
    let pre = users_by_name(&c1);
    assert_eq!(
        pre.len(),
        4,
        "live state pre-crash should be 4 users, got {pre:?}"
    );
    assert_eq!(
        pre["alice"],
        (99, true),
        "alice's update must be visible pre-crash"
    );
    assert!(!pre.contains_key("carol"), "carol was deleted pre-crash");

    // Hard kill — no graceful shutdown, no Drop, no flush.
    drop(c1);
    let status = server1.sigkill();
    assert_signal_killed(status);

    // --- run #2: cold boot on the same data dir; the oracle ---
    let mut server2 = ServerProc::spawn(&data_dir, &schema_path, logs.path(), "run2");
    let c2 = server2.connect_when_ready();

    // (1) Scalar objects: SST-resident (batch A) + WAL-replayed (batch B), with the
    // in-WAL update and delete applied. carol gone; alice carries her updated age.
    let recovered = users_by_name(&c2);
    let expected: BTreeMap<String, (i64, bool)> = [
        ("alice".to_string(), (99, true)),
        ("bob".to_string(), (70, false)),
        ("dave".to_string(), (80, true)),
        ("erin".to_string(), (18, false)),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        recovered,
        expected,
        "cold-boot recovery must reproduce the exact committed user state\n{}",
        server2.read_logs()
    );

    // (2) @indexed secondary index survives recovery: ages > 50 → alice(99) bob(70)
    // dave(80). Proves the i: keyspace replayed (and reflects the in-WAL update to
    // alice, who started at 30).
    let adults: Vec<Row<Value>> = c2
        .fetch(&Query::<Value>::filter_on("User", ".age > 50"))
        .expect("indexed filter");
    let mut adult_names: Vec<String> = adults
        .iter()
        .map(|r| r.data["name"].as_str().unwrap().to_string())
        .collect();
    adult_names.sort();
    assert_eq!(
        adult_names,
        vec!["alice", "bob", "dave"],
        "the @indexed age filter must return the recovered high-age users"
    );

    // (3) @unique secondary index survives recovery: re-creating an existing name is
    // rejected with a UNIQUENESS error specifically (not just any server error — a
    // corrupted catalog could fail the create for an unrelated reason and still look
    // "rejected"). The engine renders `unique constraint violated: User.name = …`,
    // surfaced to the client as `Error::Server`. Proves the u: keyspace replayed.
    let dup = c2.execute(&Query::<Value>::raw(
        "User.create({ name: \"bob\", age: 1, active: true })",
    ));
    match &dup {
        Err(Error::Server(msg)) => assert!(
            msg.to_lowercase().contains("unique"),
            "a duplicate @unique name must be rejected as a uniqueness violation, got: {msg}"
        ),
        other => panic!("expected a uniqueness violation after recovery, got {other:?}"),
    }

    // (4) BYO vectors survive recovery via an HNSW rebuilt from the recovered `v:`
    // keyspace (no graceful Drop ran, so no fresh snapshot — this is the full rebuild
    // path). doc-x is SST-resident (batch A); doc-w is WAL-replayed (batch B).
    assert_eq!(nearest_doc_label(&c2, "[1.0, 0.0, 0.0, 0.0]"), "doc-x");
    assert_eq!(nearest_doc_label(&c2, "[0.0, 1.0, 0.0, 0.0]"), "doc-y");
    assert_eq!(nearest_doc_label(&c2, "[0.0, 0.0, 1.0, 0.0]"), "doc-z");
    assert_eq!(
        nearest_doc_label(&c2, "[0.0, 0.0, 0.0, 1.0]"),
        "doc-w",
        "the WAL-replayed BYO vector must be searchable in the rebuilt HNSW"
    );

    drop(c2);
    assert_signal_killed(server2.sigkill());
}

/// Focused complement: the pure WAL-replay path through the real binary. Nothing is
/// flushed to an SST before the kill, so cold boot recovers entirely from replaying
/// the WAL the binary fsync'd at each commit. Pins the minimal recovery path on its
/// own for clear diagnosis if the comprehensive test regresses.
#[test]
fn real_binary_sigkill_recovers_wal_only() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let logs = tempfile::tempdir().expect("log tempdir");
    let data_dir = tmp.path().join("data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let schema_path = tmp.path().join("schema.rhype");
    std::fs::write(&schema_path, SCHEMA).expect("write schema");

    let mut server1 = ServerProc::spawn(&data_dir, &schema_path, logs.path(), "wal-run1");
    let c1 = server1.connect_when_ready();
    // All committed, all in the WAL — no compact.
    create_user(&c1, "ada", 36, true);
    create_user(&c1, "grace", 45, false);
    let linus = create_user(&c1, "linus", 22, true);
    exec_raw(&c1, &format!("User.get({linus}).update({{ age: 23 }})"));
    drop(c1);
    assert_signal_killed(server1.sigkill());

    let mut server2 = ServerProc::spawn(&data_dir, &schema_path, logs.path(), "wal-run2");
    let c2 = server2.connect_when_ready();
    let recovered = users_by_name(&c2);
    let expected: BTreeMap<String, (i64, bool)> = [
        ("ada".to_string(), (36, true)),
        ("grace".to_string(), (45, false)),
        ("linus".to_string(), (23, true)),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        recovered,
        expected,
        "pure-WAL-replay recovery must reproduce the exact committed state\n{}",
        server2.read_logs()
    );
    drop(c2);
    assert_signal_killed(server2.sigkill());
}
