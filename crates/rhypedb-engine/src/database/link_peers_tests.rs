//! `get_link_peer_ids`, `get_link_peers_top` and `get_many_columns` (GH #52):
//! a mail client listing a label's threads, newest first. Every answer is
//! checked against the same question asked the slow way — the edge index and
//! full object reads — including on a store whose reverse-edge copies are
//! stale or not trusted.

use super::*;
use rhypedb_schema::parser::parse_schema;

fn schema() -> Schema {
    parse_schema(
        r#"
        type Label {
            name: String
            messages: [Message] @inverse(Message.labels)
            pinned: [Message]
        }
        type Message {
            subject: String
            thread: Thread @on_delete(remove)
            labels: [Label]
        }
        type Thread {
            key: String
            latest: DateTime
            note: String
        }
        "#,
    )
    .unwrap()
}

fn no_sweeper() -> OpenOptions {
    OpenOptions {
        background_cover_refresh: false,
        ..Default::default()
    }
}

fn open(dir: &std::path::Path) -> Arc<Database> {
    Database::open_with_options(schema(), dir, no_sweeper()).unwrap()
}

fn label(db: &Database) -> u64 {
    let mut f = FieldMap::new();
    f.insert("name".into(), Value::String("inbox".into()));
    db.create("Label", f).unwrap().id
}

fn message(db: &Database) -> u64 {
    let mut f = FieldMap::new();
    f.insert("subject".into(), Value::String("hi".into()));
    db.create("Message", f).unwrap().id
}

fn thread(db: &Database, key: &str, latest: Option<i64>) -> u64 {
    let mut f = FieldMap::new();
    f.insert("key".into(), Value::String(key.into()));
    if let Some(t) = latest {
        f.insert("latest".into(), Value::DateTime(t));
    }
    db.create("Thread", f).unwrap().id
}

fn set_latest(db: &Database, t: u64, latest: i64) {
    let mut f = FieldMap::new();
    f.insert("latest".into(), Value::DateTime(latest));
    db.update("Thread", t, f).unwrap();
}

/// `get_link_peer_ids` the slow way: the label's messages, then each one's
/// thread edge.
fn peers_reference(db: &Database, l: u64, field: &str) -> Vec<(u64, Option<u64>)> {
    db.get_link_ids("Label", l, field)
        .unwrap()
        .into_iter()
        .map(|m| (m, db.get_link_ids("Message", m, "thread").unwrap().first().copied()))
        .collect()
}

/// `get_link_peers_top` the slow way: distinct threads, full reads, sorted.
fn top_reference(db: &Database, l: u64, q: &LinkPeersTop<'_>) -> Vec<(u64, FieldMap)> {
    let mut threads: Vec<u64> = peers_reference(db, l, "messages")
        .into_iter()
        .filter_map(|(_, t)| t)
        .collect();
    threads.sort_unstable();
    threads.dedup();
    let mut rows: Vec<(Option<Value>, u64, FieldMap)> = threads
        .into_iter()
        .filter_map(|t| db.get("Thread", t).ok())
        .map(|o| {
            let key = match o.fields.get(q.order_by) {
                None | Some(Value::Null) => None,
                Some(v) => Some(v.clone()),
            };
            let fields = q
                .fields
                .iter()
                .filter_map(|f| o.fields.get(*f).map(|v| (f.to_string(), v.clone())))
                .collect();
            (key, o.id, fields)
        })
        .collect();
    rows.sort_by(|a, b| peer_order(&a.0, a.1, &b.0, b.1, q.descending));
    rows.into_iter()
        .skip(q.offset)
        .take(q.limit)
        .map(|(_, id, f)| (id, f))
        .collect()
}

fn top(db: &Database, l: u64, q: &LinkPeersTop<'_>) -> Vec<(u64, FieldMap)> {
    db.get_link_peers_top("Label", l, "messages", q)
        .unwrap()
        .into_iter()
        .map(|o| {
            assert_eq!(o.type_name, "Thread");
            (o.id, o.fields)
        })
        .collect()
}

fn newest(limit: usize) -> LinkPeersTop<'static> {
    LinkPeersTop {
        peer: "thread",
        order_by: "latest",
        descending: true,
        offset: 0,
        limit,
        fields: &["key", "latest"],
    }
}

#[test]
fn peer_ids_match_the_edge_index_whatever_the_link_order() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let l = label(&db);
    let (t1, t2) = (thread(&db, "a", Some(1)), thread(&db, "b", Some(2)));
    let (m1, m2, m3, m4) = (message(&db), message(&db), message(&db), message(&db));
    // m1: thread before label. m2: label before thread. m3: no thread.
    // m4: relinked to another thread.
    db.link("Message", m1, "thread", t1, None).unwrap();
    db.link("Message", m1, "labels", l, None).unwrap();
    db.link("Message", m2, "labels", l, None).unwrap();
    db.link("Message", m2, "thread", t2, None).unwrap();
    db.link("Message", m3, "labels", l, None).unwrap();
    db.link("Message", m4, "thread", t1, None).unwrap();
    db.link("Message", m4, "labels", l, None).unwrap();
    db.link("Message", m4, "thread", t2, None).unwrap();
    for m in [m1, m2, m3, m4] {
        db.link("Label", l, "pinned", m, None).unwrap();
    }

    let want = vec![(m1, Some(t1)), (m2, Some(t2)), (m3, None), (m4, Some(t2))];
    assert!(db.cover_peers_sound());
    // The fast path has something to read: the label's reverse edges carry
    // each threaded message's thread id and a copy of the thread.
    let rel = db.rel_ids["Message.labels"];
    for m in [m1, m2, m4] {
        let v = db
            .storage
            .get_at(db.storage.read_snapshot(), &KeyBuilder::reverse_edge(l, rel, m))
            .unwrap()
            .unwrap();
        assert!(crate::object::find_u64_field_in_raw(&v, "thread").is_some());
        assert!(crate::object::find_bytes_field_in_raw(&v, "thread__cover").is_some());
    }
    assert_eq!(db.get_link_peer_ids("Label", l, "messages", "thread").unwrap(), want);
    assert_eq!(peers_reference(&db, l, "messages"), want);
    // A forward field carries no copies: every peer comes from the edge index.
    assert_eq!(db.get_link_peer_ids("Label", l, "pinned", "thread").unwrap(), want);
}

#[test]
fn the_peer_must_be_a_forward_to_one_relation() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let l = label(&db);
    for bad in ["labels", "subject"] {
        assert!(matches!(
            db.get_link_peer_ids("Label", l, "messages", bad),
            Err(EngineError::TypeMismatch { .. })
        ));
    }
    assert!(matches!(
        db.get_link_peer_ids("Label", l, "messages", "nope"),
        Err(EngineError::FieldNotFound { .. })
    ));
    assert!(matches!(
        db.get_link_peer_ids("Label", l, "name", "thread"),
        Err(EngineError::TypeMismatch { .. })
    ));
    // A relation is not a field to read off a peer.
    assert!(matches!(
        db.get_many_columns("Message", &[1], &["thread"]),
        Err(EngineError::TypeMismatch { .. })
    ));
    let q = LinkPeersTop { order_by: "nope", ..newest(5) };
    assert!(matches!(
        db.get_link_peers_top("Label", l, "messages", &q),
        Err(EngineError::FieldNotFound { .. })
    ));
}

/// Make the store look like an older rhypedb wrote it: no soundness marker,
/// and `stale` back as the reverse edge of message `m` under label `l`.
fn age_store(db: &Database, l: u64, m: u64, stale: Bytes) {
    let rel = db.rel_ids["Message.labels"];
    let mut txn = db.storage.begin_txn();
    db.storage.delete(&mut txn, &KeyBuilder::catalog_cover_peers()).unwrap();
    db.storage.put(&mut txn, &KeyBuilder::reverse_edge(l, rel, m), stale).unwrap();
    db.storage.commit(&mut txn).unwrap();
}

#[test]
fn an_untrusted_store_answers_from_the_edge_index() {
    let dir = tempfile::tempdir().unwrap();
    let (l, m, t1, t2);
    {
        let db = open(dir.path());
        l = label(&db);
        m = message(&db);
        t1 = thread(&db, "a", Some(10));
        t2 = thread(&db, "b", Some(5));
        db.link("Message", m, "thread", t1, None).unwrap();
        db.link("Message", m, "labels", l, None).unwrap();
        let rel = db.rel_ids["Message.labels"];
        let old = db
            .storage
            .get_at(db.storage.read_snapshot(), &KeyBuilder::reverse_edge(l, rel, m))
            .unwrap()
            .unwrap();
        db.link("Message", m, "thread", t2, None).unwrap();
        // What an older rhypedb could leave behind: the copy still names t1.
        age_store(&db, l, m, old);
    }
    let db = open(dir.path());
    assert!(!db.cover_peers_sound());
    assert_eq!(db.get_link_peer_ids("Label", l, "messages", "thread").unwrap(), vec![(m, Some(t2))]);
    let got = top(&db, l, &newest(5));
    assert_eq!(got.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![t2]);
    assert_eq!(got, top_reference(&db, l, &newest(5)));
}

#[test]
fn top_threads_order_window_and_project() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let l = label(&db);
    // Thread latest values: two ties (30), one without a value.
    let ts = [
        thread(&db, "t0", Some(10)),
        thread(&db, "t1", Some(30)),
        thread(&db, "t2", None),
        thread(&db, "t3", Some(30)),
        thread(&db, "t4", Some(20)),
    ];
    // Several messages per thread, interleaved; one thread only via an
    // unlabelled message (must not appear).
    let other = thread(&db, "unlabelled", Some(99));
    for round in 0..3 {
        for (i, t) in ts.iter().enumerate() {
            if round > 0 && i % 2 == 0 {
                continue;
            }
            let m = message(&db);
            db.link("Message", m, "thread", *t, None).unwrap();
            db.link("Message", m, "labels", l, None).unwrap();
        }
    }
    let m = message(&db);
    db.link("Message", m, "thread", other, None).unwrap();

    let ids = |q: &LinkPeersTop<'_>| top(&db, l, q).into_iter().map(|(id, _)| id).collect::<Vec<_>>();
    // Descending: ties by id descending; no value last.
    assert_eq!(ids(&newest(10)), vec![ts[3], ts[1], ts[4], ts[0], ts[2]]);
    // Ascending: ties by id ascending; no value still last.
    let asc = LinkPeersTop { descending: false, ..newest(10) };
    assert_eq!(ids(&asc), vec![ts[0], ts[4], ts[1], ts[3], ts[2]]);
    // Windows.
    assert_eq!(ids(&newest(2)), vec![ts[3], ts[1]]);
    assert_eq!(ids(&LinkPeersTop { offset: 2, ..newest(2) }), vec![ts[4], ts[0]]);
    assert_eq!(ids(&LinkPeersTop { offset: 4, ..newest(2) }), vec![ts[2]]);
    assert!(ids(&LinkPeersTop { offset: 9, ..newest(2) }).is_empty());
    assert!(ids(&newest(0)).is_empty());

    // Projection: exactly the requested fields (`note` is never asked for).
    let got = top(&db, l, &newest(1));
    assert_eq!(got[0].1.len(), 2);
    assert_eq!(got[0].1["key"], Value::String("t3".into()));
    assert_eq!(got[0].1["latest"], Value::DateTime(30));
    for q in [newest(10), asc, LinkPeersTop { offset: 1, ..newest(3) }] {
        assert_eq!(top(&db, l, &q), top_reference(&db, l, &q));
    }
}

#[test]
fn stale_copies_are_looked_up_and_deleted_threads_drop_out() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let l = label(&db);
    let (t1, t2, t3) = (thread(&db, "a", Some(1)), thread(&db, "b", Some(2)), thread(&db, "c", Some(3)));
    for t in [t1, t2, t3] {
        let m = message(&db);
        db.link("Message", m, "thread", t, None).unwrap();
        db.link("Message", m, "labels", l, None).unwrap();
    }
    assert_eq!(top(&db, l, &newest(5)).iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![t3, t2, t1]);

    // No sweeper: the copies of t1 in the label's reverse edges are now stale
    // (older generation) and must not be believed.
    set_latest(&db, t1, 100);
    let got = top(&db, l, &newest(5));
    assert_eq!(got.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![t1, t3, t2]);
    assert_eq!(got[0].1["latest"], Value::DateTime(100));
    assert_eq!(got, top_reference(&db, l, &newest(5)));

    // Message.thread is @on_delete(remove): the message stays labelled but
    // loses its thread, and the thread leaves the listing.
    db.delete("Thread", t3).unwrap();
    assert_eq!(top(&db, l, &newest(5)).iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![t1, t2]);

    // After a full repair every copy is current again; same answer.
    db.rewrite_covers().unwrap();
    assert_eq!(top(&db, l, &newest(5)), top_reference(&db, l, &newest(5)));
}

/// Random links, relinks, unlinks, updates and deletes; after every few steps
/// both calls must agree with the slow way.
#[test]
fn random_histories_agree_with_the_slow_way() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
    let mut rand = move |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    let labels: Vec<u64> = (0..2).map(|_| label(&db)).collect();
    let mut threads: Vec<u64> = (0..8)
        .map(|i| thread(&db, &format!("t{i}"), (i % 3 != 0).then_some(i as i64 * 7 % 5)))
        .collect();
    let messages: Vec<u64> = (0..30).map(|_| message(&db)).collect();
    for step in 0..400 {
        let m = messages[rand(30) as usize];
        match rand(8) {
            0..=2 => {
                if !threads.is_empty() {
                    let t = threads[rand(threads.len() as u64) as usize];
                    db.link("Message", m, "thread", t, None).unwrap();
                }
            }
            3 => {
                let l = labels[rand(2) as usize];
                db.link("Message", m, "labels", l, None).unwrap();
            }
            4 => {
                let l = labels[rand(2) as usize];
                db.unlink("Message", m, "labels", l).unwrap();
            }
            5 => {
                if let Some(&t) = db.get_link_ids("Message", m, "thread").unwrap().first() {
                    db.unlink("Message", m, "thread", t).unwrap();
                }
            }
            6 => {
                if !threads.is_empty() {
                    let t = threads[rand(threads.len() as u64) as usize];
                    set_latest(&db, t, rand(6) as i64);
                }
            }
            _ => {
                if rand(5) == 0 && threads.len() > 2 {
                    let t = threads.remove(rand(threads.len() as u64) as usize);
                    db.delete("Thread", t).unwrap();
                } else {
                    let l = labels[rand(2) as usize];
                    db.link("Message", m, "labels", l, None).unwrap();
                }
            }
        }
        if step % 7 == 0 {
            for &l in &labels {
                assert_eq!(
                    db.get_link_peer_ids("Label", l, "messages", "thread").unwrap(),
                    peers_reference(&db, l, "messages"),
                    "step {step}"
                );
                for q in [
                    newest(3),
                    LinkPeersTop { descending: false, offset: 1, ..newest(4) },
                ] {
                    assert_eq!(top(&db, l, &q), top_reference(&db, l, &q), "step {step}");
                }
            }
        }
    }
}

#[test]
fn get_many_columns_reads_values_by_position() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let t1 = thread(&db, "a", Some(5));
    let t2 = thread(&db, "b", None);
    let cols = db
        .get_many_columns("Thread", &[t2, 999_999, t1, t2], &["latest", "key"])
        .unwrap();
    assert_eq!(cols.ids, vec![t1, t2]);
    assert_eq!(cols.row(0), &[Some(Value::DateTime(5)), Some(Value::String("a".into()))]);
    assert_eq!(cols.row(1), &[None, Some(Value::String("b".into()))]);
    assert_eq!(cols.rows().count(), 2);
    // A repeated name fills each of its slots.
    let dup = db.get_many_columns("Thread", &[t1], &["key", "latest", "key"]).unwrap();
    let a = Some(Value::String("a".into()));
    assert_eq!(dup.row(0), &[a.clone(), Some(Value::DateTime(5)), a]);
    assert!(db.get_many_columns("Thread", &[], &["key"]).unwrap().is_empty());
    assert!(matches!(
        db.get_many_columns("Thread", &[t1], &["nope"]),
        Err(EngineError::FieldNotFound { .. })
    ));
    // Same values as the map form.
    let maps = db.get_many_projected("Thread", &[t1, t2], &["latest", "key"]).unwrap();
    for (obj, (id, row)) in maps.iter().zip(cols.rows()) {
        assert_eq!(obj.id, id);
        assert_eq!(obj.fields.get("latest"), row[0].as_ref());
        assert_eq!(obj.fields.get("key"), row[1].as_ref());
    }
}
