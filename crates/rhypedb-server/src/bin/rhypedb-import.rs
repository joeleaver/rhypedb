//! `rhypedb-import` — offline logical import of a `rhypedb-logical-export` NDJSON
//! dump into a fresh data dir. The server must be STOPPED (this opens the data
//! dir directly). Counterpart of the CLI's `export` / `verify-export`.

use std::path::PathBuf;

use clap::Parser;
use rhypedb_server::import::{ImportOptions, VectorImportMode, run_import};

#[derive(Parser)]
#[command(
    name = "rhypedb-import",
    about = "Offline logical import of a rhypedb NDJSON export (server must be stopped)"
)]
struct Cli {
    /// The `.ndjson` export file to import (from `rhypedb-cli export --download`).
    src: PathBuf,
    /// The data dir to materialize (must be empty unless --force).
    #[arg(long)]
    data_dir: PathBuf,
    /// Overwrite even if the data dir is non-empty (clears stale LSM state first).
    #[arg(long)]
    force: bool,
    /// Vector handling: raw (default — import raw f32, HNSW rebuilds on start) | none.
    #[arg(long, default_value = "raw")]
    vectors: String,
}

fn main() {
    let cli = Cli::parse();
    let vectors = match cli.vectors.as_str() {
        "raw" => VectorImportMode::Raw,
        "none" => VectorImportMode::None,
        other => {
            eprintln!("ERROR: unknown --vectors '{other}' (expected raw|none)");
            std::process::exit(1);
        }
    };
    let opts = ImportOptions {
        force: cli.force,
        vectors,
    };

    match run_import(&cli.src, &cli.data_dir, &opts) {
        Ok(r) => {
            println!(
                "imported {} type(s): {} objects, {} edges, {} vectors",
                r.types, r.objects, r.edges, r.vectors
            );
            println!(
                "start the server with:\n  rhypedb-server --schema {} --data-dir {}",
                cli.data_dir.join("schema.rhype").display(),
                cli.data_dir.display()
            );
            if r.vectors > 0 {
                println!("(the HNSW vector index rebuilds from the imported vectors on first start)");
            }
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    }
}
