//! Server configuration.
//!
//! A `rhypedb.toml` config file is an OPTIONAL fallback layer beneath the CLI
//! flags and env vars that already exist. Precedence, most-specific wins:
//!
//! ```text
//! CLI flag  >  env var  >  config file  >  built-in default
//! ```
//!
//! With no `--config` / `RHYPEDB_CONFIG`, behaviour is exactly as before — the
//! file is purely additive. The file is loaded only from an explicit path (no
//! auto-discovery). Unknown keys are rejected so a typo fails loud.
//!
//! [`resolve`] takes the three layers and returns the effective [`ServerConfig`].
//! It does not read env, touch the filesystem, or exit (the caller owns I/O +
//! `process::exit`) — it only emits stderr WARNINGs for out-of-range tuning values.
//! That keeps the precedence logic exhaustively unit-testable on the return value.

use rhypedb_query::GovernorLimits;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Duration;

// Defaults. These are the SINGLE source of truth for the effective values; the
// clap `default_value`s on the flags are kept only so `--help` shows them and
// MUST equal these.
pub const DEFAULT_DATA_DIR: &str = "./rhypedb-data";
pub const DEFAULT_LISTEN: &str = "127.0.0.1:4200";
pub const DEFAULT_TCP_LISTEN: &str = "127.0.0.1:4201";
pub const DEFAULT_CACHE_MAX_ENTRIES: usize = crate::query_cache::DEFAULT_CACHE_SIZE;
pub const DEFAULT_GRACEFUL_DRAIN_SECS: u64 = 20;
pub const DEFAULT_WORKER_QUIESCE_SECS: u64 = 10;

/// The on-disk `rhypedb.toml`. Every field optional (a file may set any subset);
/// unknown keys are rejected (a typo like `admin_tocken` fails loud). `ef`/`rerank`/
/// `cache_max_entries` are `i64` so a NEGATIVE value is a range check in `resolve`
/// (warn + ignore), while a wrong TYPE (e.g. `ef = "x"`) is a loud parse error.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    pub schema: Option<PathBuf>,
    pub data_dir: Option<PathBuf>,
    pub listen: Option<String>,
    pub tcp_listen: Option<String>,
    pub no_sync: Option<bool>,
    pub restore_from: Option<PathBuf>,
    pub restore_from_force: Option<bool>,
    pub admin_token: Option<String>,
    pub ef: Option<i64>,
    pub rerank: Option<i64>,
    pub cache_max_entries: Option<i64>,
    // i64 (not u64) so a negative value range-checks to warn+default in `resolve`,
    // consistent with the other tuning knobs, instead of being a fatal type error.
    pub graceful_drain_secs: Option<i64>,
    pub worker_quiesce_budget_secs: Option<i64>,
    /// `"lz4"` (default) or `"none"`. Per-block SST compression for newly
    /// written files. An unrecognized value warns and falls through.
    pub block_compression: Option<String>,
}

/// A snapshot of the relevant env vars (raw, unparsed). Built once via
/// [`EnvLayer::from_env`]; `resolve` interprets these so it never reads env itself.
#[derive(Debug, Default)]
pub struct EnvLayer {
    pub admin_token: Option<String>,
    pub ef: Option<String>,
    pub rerank: Option<String>,
    // PathBuf via var_os (not var) so a non-UTF-8 restore path is preserved, as
    // the original run() did.
    pub restore_from: Option<PathBuf>,
    pub restore_from_force: Option<String>,
    pub block_compression: Option<String>,
    /// Query-governor tuning (raw env strings; parsed in [`resolve`]). See
    /// [`rhypedb_query::GovernorLimits`]. `RHYPEDB_QUERY_GOVERNOR` set to a falsey
    /// value disables the governor entirely; each `RHYPEDB_MAX_QUERY_*` overrides
    /// one dimension (a `0` turns that dimension off).
    pub query_governor: Option<String>,
    pub max_query_rows: Option<String>,
    pub max_query_limit: Option<String>,
    pub max_query_depth: Option<String>,
    pub max_query_result_rows: Option<String>,
    pub max_query_ms: Option<String>,
}

impl EnvLayer {
    pub fn from_env() -> Self {
        Self {
            admin_token: std::env::var("RHYPEDB_ADMIN_TOKEN").ok(),
            ef: std::env::var("RHYPEDB_EF").ok(),
            rerank: std::env::var("RHYPEDB_RERANK").ok(),
            restore_from: std::env::var_os("RHYPEDB_RESTORE_FROM").map(PathBuf::from),
            restore_from_force: std::env::var("RHYPEDB_RESTORE_FROM_FORCE").ok(),
            block_compression: std::env::var("RHYPEDB_BLOCK_COMPRESSION").ok(),
            query_governor: std::env::var("RHYPEDB_QUERY_GOVERNOR").ok(),
            max_query_rows: std::env::var("RHYPEDB_MAX_QUERY_ROWS").ok(),
            max_query_limit: std::env::var("RHYPEDB_MAX_QUERY_LIMIT").ok(),
            max_query_depth: std::env::var("RHYPEDB_MAX_QUERY_DEPTH").ok(),
            max_query_result_rows: std::env::var("RHYPEDB_MAX_QUERY_RESULT_ROWS").ok(),
            max_query_ms: std::env::var("RHYPEDB_MAX_QUERY_MS").ok(),
        }
    }
}

/// The CLI flag layer. For flags WITH a clap default (`data_dir`/`listen`/
/// `tcp_listen`) the field is `Some` ONLY when the user actually typed the flag
/// (decided via `ArgMatches::value_source`), so an unspecified flag's default
/// does not shadow env/file. Bool flags are plain bools (store_true can only push
/// toward `true`).
#[derive(Debug, Default)]
pub struct CliLayer {
    pub schema: Option<PathBuf>,
    pub data_dir: Option<PathBuf>,
    pub listen: Option<String>,
    pub tcp_listen: Option<String>,
    pub no_sync: bool,
    pub restore_from: Option<PathBuf>,
    pub restore_force: bool,
    pub block_compression: Option<String>,
}

/// The effective, resolved configuration the server runs with.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerConfig {
    pub schema: Option<PathBuf>,
    pub data_dir: PathBuf,
    pub listen: String,
    pub tcp_listen: String,
    pub no_sync: bool,
    pub admin_token: Option<String>,
    pub default_ef: Option<usize>,
    pub default_rerank: Option<usize>,
    pub restore_from: Option<PathBuf>,
    pub restore_force: bool,
    pub cache_max_entries: usize,
    pub graceful_drain: Duration,
    pub worker_quiesce_budget: Duration,
    /// Per-block SST compression for newly written files (flush + compaction).
    pub block_compression: rhypedb_storage::SstCompression,
    /// Per-query resource governor limits (`Some` = on, the default with generous
    /// caps; `None` = disabled via `RHYPEDB_QUERY_GOVERNOR`). Threaded into every
    /// query's `ExecContext`. See [`rhypedb_query::GovernorLimits`].
    pub query_governor: Option<GovernorLimits>,
}

fn env_truthy(v: &str) -> bool {
    matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

/// Parse an env string to an integer, warning (and dropping to `None`) if present
/// but unparseable — matching the existing `RHYPEDB_EF`/`RHYPEDB_RERANK` contract.
fn env_int(name: &str, raw: Option<&str>) -> Option<i64> {
    let s = raw.map(str::trim).filter(|s| !s.is_empty())?;
    match s.parse::<i64>() {
        Ok(n) => Some(n),
        Err(_) => {
            eprintln!("WARNING: {name}=\"{s}\" is not an integer; ignoring it.");
            None
        }
    }
}

/// Validate a resolved `ef` (must be >= 1) — warn + drop otherwise.
fn finalize_ef(raw: Option<i64>) -> Option<usize> {
    match raw {
        Some(n) if n >= 1 => Some(n as usize),
        Some(n) => {
            eprintln!("WARNING: ef must be >= 1 (got {n}); ignoring it.");
            None
        }
        None => None,
    }
}

/// Validate a resolved `rerank` — `0` (or negative) means off → `None`.
fn finalize_rerank(raw: Option<i64>) -> Option<usize> {
    match raw {
        Some(n) if n > 0 => Some(n as usize),
        _ => None,
    }
}

/// Parse a block-compression setting (`"lz4"`/`"none"`, case-insensitive). An
/// empty/whitespace value is treated as unset (`None` → fall through to the
/// lower layer); an unrecognized value warns and falls through.
fn parse_compression(
    source: &str,
    raw: Option<&str>,
) -> Option<rhypedb_storage::SstCompression> {
    let s = raw.map(str::trim).filter(|s| !s.is_empty())?;
    match s.to_ascii_lowercase().as_str() {
        "lz4" => Some(rhypedb_storage::SstCompression::Lz4),
        "none" | "off" => Some(rhypedb_storage::SstCompression::None),
        other => {
            eprintln!(
                "WARNING: {source}=\"{other}\" is not a valid block compression \
                 (expected \"lz4\" or \"none\"); ignoring it."
            );
            None
        }
    }
}

/// Resolve one governor knob (rows / limit / depth / result-rows) from its env var:
/// absent → `default`; `>= 0` → that value (`0` disables THAT dimension, per
/// [`GovernorLimits`]); negative or unparseable → warn + keep `default`.
fn governor_knob(name: &str, raw: Option<&str>, default: usize) -> usize {
    match env_int(name, raw) {
        None => default,
        Some(n) if n >= 0 => n as usize,
        Some(n) => {
            eprintln!("WARNING: {name} must be >= 0 (got {n}); using the default {default}.");
            default
        }
    }
}

/// A positive-or-default knob (cache size, drain/quiesce seconds): warn + use the
/// default if the file value is < 1.
fn positive_or_default(name: &str, v: Option<i64>, default: u64) -> u64 {
    match v {
        Some(n) if n >= 1 => n as u64,
        Some(n) => {
            eprintln!("WARNING: {name} must be >= 1 (got {n}); using the default {default}.");
            default
        }
        None => default,
    }
}

/// Resolve the effective config from the three layers. No env / fs / exit (the
/// caller owns those); only emits stderr WARNINGs for out-of-range tuning knobs,
/// which fall back to the lower layer / default. The return value is fully
/// determined by the inputs, so the precedence matrix is unit-tested on it.
pub fn resolve(
    cli: &CliLayer,
    env: &EnvLayer,
    file: Option<&FileConfig>,
) -> ServerConfig {
    let f_schema = file.and_then(|f| f.schema.clone());
    let f_data_dir = file.and_then(|f| f.data_dir.clone());
    let f_listen = file.and_then(|f| f.listen.clone());
    let f_tcp = file.and_then(|f| f.tcp_listen.clone());
    let f_no_sync = file.and_then(|f| f.no_sync);
    let f_restore_from = file.and_then(|f| f.restore_from.clone());
    let f_restore_force = file.and_then(|f| f.restore_from_force);
    let f_admin = file.and_then(|f| f.admin_token.clone());
    let f_ef = file.and_then(|f| f.ef);
    let f_rerank = file.and_then(|f| f.rerank);
    let f_cache = file.and_then(|f| f.cache_max_entries);
    let f_drain = file.and_then(|f| f.graceful_drain_secs);
    let f_quiesce = file.and_then(|f| f.worker_quiesce_budget_secs);

    // Block compression: cli > env > file > default (None). Each layer parses to
    // an enum; an invalid value warns and falls through to the next layer.
    let block_compression = parse_compression("--block-compression", cli.block_compression.as_deref())
        .or_else(|| parse_compression("RHYPEDB_BLOCK_COMPRESSION", env.block_compression.as_deref()))
        .or_else(|| {
            parse_compression(
                "block_compression (config file)",
                file.and_then(|f| f.block_compression.as_deref()),
            )
        })
        .unwrap_or(rhypedb_storage::SstCompression::None);

    // ef / rerank: cli (none today) > env (parsed string) > file, validated ONCE
    // on the resolved value. A valid env value wins over the file.
    let ef_raw = env_int("RHYPEDB_EF", env.ef.as_deref()).or(f_ef);
    let rerank_raw = env_int("RHYPEDB_RERANK", env.rerank.as_deref()).or(f_rerank);

    // Query governor: ON by default with generous caps. A non-empty falsey
    // `RHYPEDB_QUERY_GOVERNOR` (off/0/false/no) disables it wholesale; otherwise
    // each dimension starts from `GovernorLimits::DEFAULT` and is overridable by
    // its own `RHYPEDB_MAX_QUERY_*` (a `0` turns that one dimension off). Env-only
    // (an operational per-deployment cap set by the orchestrator).
    let query_governor = {
        let governor_off = env
            .query_governor
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(|v| !env_truthy(v))
            .unwrap_or(false);
        if governor_off {
            None
        } else {
            let d = GovernorLimits::DEFAULT;
            Some(GovernorLimits {
                max_limit: governor_knob(
                    "RHYPEDB_MAX_QUERY_LIMIT",
                    env.max_query_limit.as_deref(),
                    d.max_limit,
                ),
                max_rows_scanned: governor_knob(
                    "RHYPEDB_MAX_QUERY_ROWS",
                    env.max_query_rows.as_deref(),
                    d.max_rows_scanned,
                ),
                max_depth: governor_knob(
                    "RHYPEDB_MAX_QUERY_DEPTH",
                    env.max_query_depth.as_deref(),
                    d.max_depth,
                ),
                max_result_rows: governor_knob(
                    "RHYPEDB_MAX_QUERY_RESULT_ROWS",
                    env.max_query_result_rows.as_deref(),
                    d.max_result_rows,
                ),
                max_duration: Duration::from_millis(governor_knob(
                    "RHYPEDB_MAX_QUERY_MS",
                    env.max_query_ms.as_deref(),
                    d.max_duration.as_millis() as usize,
                ) as u64),
            })
        }
    };

    ServerConfig {
        schema: cli.schema.clone().or(f_schema),
        data_dir: cli
            .data_dir
            .clone()
            .or(f_data_dir)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_DATA_DIR)),
        listen: cli
            .listen
            .clone()
            .or(f_listen)
            .unwrap_or_else(|| DEFAULT_LISTEN.to_string()),
        tcp_listen: cli
            .tcp_listen
            .clone()
            .or(f_tcp)
            .unwrap_or_else(|| DEFAULT_TCP_LISTEN.to_string()),
        // Bool Model A: a store_true flag can only force `true`; otherwise fall to
        // env (truthy string) then file then default-false. OR is exactly this.
        no_sync: cli.no_sync || f_no_sync.unwrap_or(false),
        // An empty env token is treated as "unset" so it falls through to the file
        // rather than shadowing a real file token; an empty file token is also None.
        admin_token: env
            .admin_token
            .clone()
            .filter(|t| !t.is_empty())
            .or(f_admin)
            .filter(|t| !t.is_empty()),
        default_ef: finalize_ef(ef_raw),
        default_rerank: finalize_rerank(rerank_raw),
        restore_from: cli
            .restore_from
            .clone()
            .or_else(|| env.restore_from.clone())
            .or(f_restore_from),
        restore_force: cli.restore_force
            || env.restore_from_force.as_deref().map(env_truthy).unwrap_or(false)
            || f_restore_force.unwrap_or(false),
        cache_max_entries: positive_or_default(
            "cache_max_entries",
            f_cache,
            DEFAULT_CACHE_MAX_ENTRIES as u64,
        ) as usize,
        graceful_drain: Duration::from_secs(positive_or_default(
            "graceful_drain_secs",
            f_drain,
            DEFAULT_GRACEFUL_DRAIN_SECS,
        )),
        worker_quiesce_budget: Duration::from_secs(positive_or_default(
            "worker_quiesce_budget_secs",
            f_quiesce,
            DEFAULT_WORKER_QUIESCE_SECS,
        )),
        block_compression,
        query_governor,
    }
}

/// Strip the source-snippet lines (the ones containing `|`) from a rendered toml
/// error, keeping the `line N, column M` location and the trailing message. A toml
/// error echoes the offending SOURCE LINE, which for a malformed `admin_token = …`
/// would leak the secret to stderr — so it must never be printed verbatim.
fn redact_toml_error(rendered: &str) -> String {
    let kept: Vec<&str> = rendered
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.contains('|'))
        .collect();
    if kept.is_empty() {
        "parse error".to_string()
    } else {
        kept.join(": ")
    }
}

/// Load + parse a `rhypedb.toml`. Every failure (unreadable, invalid TOML,
/// unknown key, wrong-typed value) is a fatal, legible error naming the FILE. The
/// parse error is REDACTED of its source snippet so file contents (e.g. a secret
/// on the offending line) never reach the log.
pub fn load_config_file(path: &std::path::Path) -> Result<FileConfig, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read config file {}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| {
        format!(
            "invalid config file {}: {}",
            path.display(),
            redact_toml_error(&e.to_string())
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(toml_src: &str) -> FileConfig {
        toml::from_str(toml_src).unwrap()
    }

    #[test]
    fn no_file_no_env_yields_defaults() {
        let c = resolve(&CliLayer::default(), &EnvLayer::default(), None);
        assert_eq!(c.data_dir, PathBuf::from(DEFAULT_DATA_DIR));
        assert_eq!(c.listen, DEFAULT_LISTEN);
        assert_eq!(c.tcp_listen, DEFAULT_TCP_LISTEN);
        assert!(!c.no_sync);
        assert_eq!(c.admin_token, None);
        assert_eq!(c.default_ef, None);
        assert_eq!(c.default_rerank, None);
        assert_eq!(c.restore_from, None);
        assert!(!c.restore_force);
        assert_eq!(c.cache_max_entries, DEFAULT_CACHE_MAX_ENTRIES);
        assert_eq!(c.graceful_drain, Duration::from_secs(DEFAULT_GRACEFUL_DRAIN_SECS));
        assert_eq!(c.schema, None);
        // SST compression is OFF by default (the v6 LZ4 default was flipped back
        // to None after a benchmark showed it cost ~3.7x on multi-hop traversal).
        assert_eq!(c.block_compression, rhypedb_storage::SstCompression::None);
        // The query governor is ON by default with the generous DoS caps.
        assert_eq!(c.query_governor, Some(GovernorLimits::DEFAULT));
    }

    #[test]
    fn query_governor_env_resolution() {
        // Off switch (falsey, non-empty) disables the governor entirely.
        for off in ["off", "0", "false", "no"] {
            let env = EnvLayer { query_governor: Some(off.into()), ..Default::default() };
            assert_eq!(
                resolve(&CliLayer::default(), &env, None).query_governor,
                None,
                "RHYPEDB_QUERY_GOVERNOR={off} must disable the governor"
            );
        }
        // An empty value is treated as unset -> default ON.
        let env = EnvLayer { query_governor: Some("".into()), ..Default::default() };
        assert_eq!(
            resolve(&CliLayer::default(), &env, None).query_governor,
            Some(GovernorLimits::DEFAULT)
        );
        // Per-dimension overrides; `0` turns that one dimension off, the rest keep
        // their defaults.
        let env = EnvLayer {
            max_query_rows: Some("5000".into()),
            max_query_ms: Some("2500".into()),
            max_query_depth: Some("0".into()),
            ..Default::default()
        };
        let g = resolve(&CliLayer::default(), &env, None).query_governor.unwrap();
        assert_eq!(g.max_rows_scanned, 5000);
        assert_eq!(g.max_duration, Duration::from_millis(2500));
        assert_eq!(g.max_depth, 0, "0 disables the depth dimension");
        assert_eq!(g.max_limit, GovernorLimits::DEFAULT.max_limit, "unset knob keeps default");
        // A negative value warns and falls back to the default for that knob.
        let env = EnvLayer { max_query_rows: Some("-1".into()), ..Default::default() };
        let g = resolve(&CliLayer::default(), &env, None).query_governor.unwrap();
        assert_eq!(g.max_rows_scanned, GovernorLimits::DEFAULT.max_rows_scanned);
    }

    #[test]
    fn block_compression_precedence() {
        // File sets lz4; nothing above overrides -> lz4.
        let f = file(r#"block_compression = "lz4""#);
        assert_eq!(
            resolve(&CliLayer::default(), &EnvLayer::default(), Some(&f)).block_compression,
            rhypedb_storage::SstCompression::Lz4
        );
        // Env "none" wins over a file "lz4".
        let env = EnvLayer { block_compression: Some("none".into()), ..Default::default() };
        assert_eq!(
            resolve(&CliLayer::default(), &env, Some(&f)).block_compression,
            rhypedb_storage::SstCompression::None
        );
        // CLI wins over both.
        let cli = CliLayer { block_compression: Some("lz4".into()), ..Default::default() };
        let env_none = EnvLayer { block_compression: Some("none".into()), ..Default::default() };
        assert_eq!(
            resolve(&cli, &env_none, Some(&f)).block_compression,
            rhypedb_storage::SstCompression::Lz4
        );
        // An invalid value falls through to the default (None).
        let bad = file(r#"block_compression = "zstd""#);
        assert_eq!(
            resolve(&CliLayer::default(), &EnvLayer::default(), Some(&bad)).block_compression,
            rhypedb_storage::SstCompression::None
        );
    }

    #[test]
    fn file_overrides_default() {
        let f = file(
            r#"
            data_dir = "/var/lib/rhypedb"
            listen = "0.0.0.0:9000"
            no_sync = true
            ef = 200
            cache_max_entries = 1024
            graceful_drain_secs = 45
        "#,
        );
        let c = resolve(&CliLayer::default(), &EnvLayer::default(), Some(&f));
        assert_eq!(c.data_dir, PathBuf::from("/var/lib/rhypedb"));
        assert_eq!(c.listen, "0.0.0.0:9000");
        assert!(c.no_sync);
        assert_eq!(c.default_ef, Some(200));
        assert_eq!(c.cache_max_entries, 1024);
        assert_eq!(c.graceful_drain, Duration::from_secs(45));
    }

    #[test]
    fn env_wins_over_file() {
        let f = file(r#"ef = 200
            rerank = 5
            admin_token = "file-token""#);
        let env = EnvLayer {
            ef: Some("100".into()),
            admin_token: Some("env-token".into()),
            ..Default::default()
        };
        let c = resolve(&CliLayer::default(), &env, Some(&f));
        assert_eq!(c.default_ef, Some(100), "env ef wins over file");
        assert_eq!(c.default_rerank, Some(5), "file rerank used when env absent");
        assert_eq!(c.admin_token, Some("env-token".into()));
    }

    #[test]
    fn flag_wins_over_env_and_file() {
        let f = file(r#"data_dir = "/from/file"
            listen = "1.1.1.1:1""#);
        let env = EnvLayer {
            restore_from: Some("/from/env".into()),
            ..Default::default()
        };
        let cli = CliLayer {
            data_dir: Some(PathBuf::from("/from/flag")),
            restore_from: Some(PathBuf::from("/from/flag-restore")),
            ..Default::default()
        };
        let c = resolve(&cli, &env, Some(&f));
        assert_eq!(c.data_dir, PathBuf::from("/from/flag"));
        assert_eq!(c.listen, "1.1.1.1:1", "file used when no flag");
        assert_eq!(c.restore_from, Some(PathBuf::from("/from/flag-restore")));
    }

    #[test]
    fn bool_model_a_or_across_layers() {
        // Only flag.
        let cli = CliLayer { no_sync: true, restore_force: true, ..Default::default() };
        let c = resolve(&cli, &EnvLayer::default(), None);
        assert!(c.no_sync && c.restore_force);

        // Only env (restore_from_force truthy string).
        let env = EnvLayer { restore_from_force: Some("yes".into()), ..Default::default() };
        let c = resolve(&CliLayer::default(), &env, None);
        assert!(c.restore_force);
        let env = EnvLayer { restore_from_force: Some("0".into()), ..Default::default() };
        let c = resolve(&CliLayer::default(), &env, None);
        assert!(!c.restore_force, "env '0' is falsey");

        // Only file.
        let f = file("no_sync = true\nrestore_from_force = true");
        let c = resolve(&CliLayer::default(), &EnvLayer::default(), Some(&f));
        assert!(c.no_sync && c.restore_force);

        // None.
        let c = resolve(&CliLayer::default(), &EnvLayer::default(), None);
        assert!(!c.no_sync && !c.restore_force);
    }

    #[test]
    fn ef_rerank_range_validation_warns_and_drops() {
        // ef < 1 and negative -> None (heuristic); rerank 0 -> off.
        let f = file("ef = 0\nrerank = 0");
        let c = resolve(&CliLayer::default(), &EnvLayer::default(), Some(&f));
        assert_eq!(c.default_ef, None);
        assert_eq!(c.default_rerank, None);

        let f = file("ef = -5");
        let c = resolve(&CliLayer::default(), &EnvLayer::default(), Some(&f));
        assert_eq!(c.default_ef, None);

        // A bad env string is ignored, falling through to the file value.
        let f = file("ef = 64");
        let env = EnvLayer { ef: Some("abc".into()), ..Default::default() };
        let c = resolve(&CliLayer::default(), &env, Some(&f));
        assert_eq!(c.default_ef, Some(64));
    }

    #[test]
    fn cache_and_drain_below_one_fall_back_to_default() {
        let f = file("cache_max_entries = 0\ngraceful_drain_secs = 0");
        let c = resolve(&CliLayer::default(), &EnvLayer::default(), Some(&f));
        assert_eq!(c.cache_max_entries, DEFAULT_CACHE_MAX_ENTRIES);
        assert_eq!(c.graceful_drain, Duration::from_secs(DEFAULT_GRACEFUL_DRAIN_SECS));
    }

    #[test]
    fn empty_admin_token_is_none() {
        let f = file(r#"admin_token = """#);
        let c = resolve(&CliLayer::default(), &EnvLayer::default(), Some(&f));
        assert_eq!(c.admin_token, None);
    }

    #[test]
    fn schema_from_file_satisfies_resolution() {
        let f = file(r#"schema = "/etc/rhypedb/schema.rhype""#);
        let c = resolve(&CliLayer::default(), &EnvLayer::default(), Some(&f));
        assert_eq!(c.schema, Some(PathBuf::from("/etc/rhypedb/schema.rhype")));
    }

    #[test]
    fn load_rejects_unknown_key_and_bad_type() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.toml");

        std::fs::write(&p, "admin_tocken = \"x\"").unwrap();
        let err = load_config_file(&p).unwrap_err();
        assert!(err.contains("invalid config file"), "{err}");

        std::fs::write(&p, "ef = \"not-an-int\"").unwrap();
        assert!(load_config_file(&p).is_err());

        std::fs::write(&p, "data_dir = \"/ok\"\nlisten = \"a:1\"").unwrap();
        let ok = load_config_file(&p).unwrap();
        assert_eq!(ok.data_dir, Some(PathBuf::from("/ok")));
    }

    #[test]
    fn load_missing_file_is_error() {
        assert!(load_config_file(std::path::Path::new("/no/such/rhypedb.toml")).is_err());
    }

    #[test]
    fn toml_parse_error_does_not_leak_secret() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.toml");
        // Unterminated string on the admin_token line — toml renders the source line.
        std::fs::write(&p, "admin_token = \"SUPERSECRET").unwrap();
        let err = load_config_file(&p).unwrap_err();
        assert!(!err.contains("SUPERSECRET"), "secret leaked into error: {err}");
        assert!(err.contains("invalid config file"), "{err}");
        // A wrong-typed token value must also not echo the value.
        std::fs::write(&p, "admin_token = 12345\nlisten = [\"oops\"]").unwrap();
        let err = load_config_file(&p).unwrap_err();
        assert!(!err.contains("oops"), "value leaked: {err}");
    }

    #[test]
    fn negative_drain_warns_and_defaults() {
        let f = file("graceful_drain_secs = -3\nworker_quiesce_budget_secs = -1");
        let c = resolve(&CliLayer::default(), &EnvLayer::default(), Some(&f));
        assert_eq!(c.graceful_drain, Duration::from_secs(DEFAULT_GRACEFUL_DRAIN_SECS));
        assert_eq!(c.worker_quiesce_budget, Duration::from_secs(DEFAULT_WORKER_QUIESCE_SECS));
    }

    #[test]
    fn empty_env_admin_token_falls_through_to_file() {
        let f = file(r#"admin_token = "file-tok""#);
        let env = EnvLayer { admin_token: Some(String::new()), ..Default::default() };
        let c = resolve(&CliLayer::default(), &env, Some(&f));
        assert_eq!(c.admin_token, Some("file-tok".into()), "empty env must not shadow the file");
    }

    #[test]
    fn env_restore_from_is_used() {
        let env = EnvLayer {
            restore_from: Some(PathBuf::from("/snap/from/env")),
            ..Default::default()
        };
        let c = resolve(&CliLayer::default(), &env, None);
        assert_eq!(c.restore_from, Some(PathBuf::from("/snap/from/env")));
    }
}
