//! Guarded writes are compare-and-set end to end: many [`AsyncClient`]
//! connections race the same filter-guarded update against a real server,
//! and exactly one may apply per round (see `docs/src/queries.md`, "Guarded
//! writes"). The executor and engine tests cover the mechanism; this proves
//! the guarantee survives the wire and the server's concurrent dispatch.

use super::*;

/// Concurrent connections per round (the task floor is 16).
const RACERS: usize = 16;
const ROUNDS: u32 = 25;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_guarded_updates_over_the_wire_apply_exactly_once() {
    let state = build_state();
    let (addr, accept_task) = start_server(state).await;

    let mut clients = Vec::with_capacity(RACERS);
    for _ in 0..RACERS {
        clients.push(Arc::new(AsyncClient::connect(addr).await.unwrap()));
    }
    clients[0]
        .query(r#"User.create({ name: "cas", age: 0, active: true })"#)
        .await
        .unwrap();

    for round in 0..ROUNDS {
        let barrier = Arc::new(tokio::sync::Barrier::new(RACERS));
        let racers: Vec<_> = clients
            .iter()
            .map(|client| {
                let (client, barrier) = (client.clone(), barrier.clone());
                let q = format!(
                    r#"User.filter(.name == "cas" && .age == {round}).update({{ age: {} }})"#,
                    round + 1
                );
                tokio::spawn(async move {
                    barrier.wait().await;
                    client.query(&q).await
                })
            })
            .collect();
        let mut applied = 0;
        for racer in racers {
            let result = racer
                .await
                .unwrap()
                .expect("a guarded update must not fail");
            applied += result.into_objects().len();
        }
        assert_eq!(
            applied, 1,
            "round {round}: {applied} of {RACERS} guarded updates applied"
        );
    }

    let mut rows = clients[0]
        .query(r#"User.filter(.name == "cas")"#)
        .await
        .unwrap()
        .into_objects();
    assert_eq!(rows.len(), 1);
    rows[0].ensure_fields_deserialized();
    assert_eq!(
        rows[0].fields.get("age"),
        Some(&rhypedb_wire::object::Value::U32(ROUNDS))
    );

    accept_task.abort();
}
