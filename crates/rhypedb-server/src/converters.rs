//! Card 5: a small registry of built-in, lossless field-type converters,
//! registered on the server's `Database` at startup so an operator can start a
//! migration by NAMING a converter over HTTP/CLI (the engine itself only knows
//! converters as Rust closures). WASM / user-supplied converters are deferred to
//! a follow-on card.
//!
//! Each converter is `Fn(object_id, &Value) -> EngineResult<Value>` at version 1.
//! A converter that receives an unexpected source kind returns a hard error (so a
//! wrong-kind row surfaces under the plan's ErrorPolicy instead of silently
//! coercing).

use std::sync::Arc;

use rhypedb_engine::database::Database;
use rhypedb_engine::object::Value;
use rhypedb_engine::{CatalogError, EngineError, MigrationStatus};

fn convert_err(name: &str, oid: u64, got: &Value) -> EngineError {
    EngineError::Catalog(CatalogError::FieldTypeChangeConverterFailed {
        qualified: name.to_string(),
        object_id: oid,
        reason: format!("unexpected source kind {}", got.type_name()),
    })
}

/// The built-in converters available by name. Exposed for the CLI `--converter`
/// help text and validation.
pub(crate) const BUILTIN_CONVERTER_NAMES: &[&str] = &[
    "widen_int_to_f64",
    "lossless_widen_u32_to_u64",
    "lossless_widen_i32_to_i64",
    "int_to_string",
    "parse_string_as_i64",
];

/// Register every built-in converter (version 1) on `db`. Idempotent per name
/// (the engine's registry is keyed by name).
pub(crate) fn register_builtins(db: &Database) {
    // Any integer kind → f64 (e.g. an `i64`/`i32`/`u32`/`u64` column widened to
    // a float). Lossy beyond 2^53 for 64-bit ints, but that is the operator's
    // choice when they pick this converter.
    db.register_converter("widen_int_to_f64", 1, |oid, v| match v {
        Value::I64(i) => Ok(Value::F64(*i as f64)),
        Value::I32(i) => Ok(Value::F64(*i as f64)),
        Value::U64(i) => Ok(Value::F64(*i as f64)),
        Value::U32(i) => Ok(Value::F64(*i as f64)),
        Value::F32(f) => Ok(Value::F64(*f as f64)),
        other => Err(convert_err("widen_int_to_f64", oid, other)),
    });

    // u32 → u64 (always lossless).
    db.register_converter("lossless_widen_u32_to_u64", 1, |oid, v| match v {
        Value::U32(i) => Ok(Value::U64(*i as u64)),
        other => Err(convert_err("lossless_widen_u32_to_u64", oid, other)),
    });

    // i32 → i64 (always lossless).
    db.register_converter("lossless_widen_i32_to_i64", 1, |oid, v| match v {
        Value::I32(i) => Ok(Value::I64(*i as i64)),
        other => Err(convert_err("lossless_widen_i32_to_i64", oid, other)),
    });

    // Any integer kind → its decimal string.
    db.register_converter("int_to_string", 1, |oid, v| match v {
        Value::I64(i) => Ok(Value::String(i.to_string())),
        Value::I32(i) => Ok(Value::String(i.to_string())),
        Value::U64(i) => Ok(Value::String(i.to_string())),
        Value::U32(i) => Ok(Value::String(i.to_string())),
        other => Err(convert_err("int_to_string", oid, other)),
    });

    // Decimal string → i64 (a row that doesn't parse errors under the policy).
    db.register_converter("parse_string_as_i64", 1, |oid, v| match v {
        Value::String(s) => s.trim().parse::<i64>().map(Value::I64).map_err(|_| {
            EngineError::Catalog(CatalogError::FieldTypeChangeConverterFailed {
                qualified: "parse_string_as_i64".to_string(),
                object_id: oid,
                reason: format!("{s:?} is not an integer"),
            })
        }),
        other => Err(convert_err("parse_string_as_i64", oid, other)),
    });
}

/// Resume every drivable in-flight migration left by a prior run (card 5 —
/// `cli_migrate_resume_after_server_restart`). `Database::open` already armed the
/// hooks and inline-drove the passes that need no converter (a crashed cutover, a
/// terminal-cancel rollback); a `Converting` plan, however, was armed but NOT
/// driven because the converter registry is empty until `register_builtins`
/// (just above). Each is resumed on its OWN background thread so a large backfill
/// doesn't block startup. Best-effort: a plan whose schema still validates at the
/// source kind (the F3 guard) or whose converter is unknown logs + stays
/// resumable for an explicit operator `resume`.
pub(crate) fn resume_inflight(db: &Arc<Database>) {
    let plans = match db.list_migrations() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("migration resume: failed to list plans: {e}");
            return;
        }
    };
    for plan in plans {
        // Only Pending/Running need a post-open drive; Completed/Cancelled/Failed/
        // DryRunCompleted are settled or operator-gated.
        if !matches!(plan.status, MigrationStatus::Pending | MigrationStatus::Running) {
            continue;
        }
        let db = Arc::clone(db);
        let plan_id = plan.plan_id;
        std::thread::Builder::new()
            .name(format!("rhypedb-resume-{plan_id}"))
            .spawn(move || {
                if let Err(e) = db.resume_field_type_migration(plan_id) {
                    eprintln!("migration {plan_id} resume deferred: {e}");
                }
            })
            .ok();
    }
}
