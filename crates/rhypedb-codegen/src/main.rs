//! `rhypedb-codegen` — generate a typed client from a rhypedb SDL schema.
//!
//! ```text
//! rhypedb-codegen --schema schema.rhype --out client.rs
//! rhypedb-codegen --schema schema.rhype        # writes to stdout
//! ```

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(name = "rhypedb-codegen", about = "Generate a typed client from a rhypedb SDL schema")]
struct Cli {
    /// Path to the SDL schema file.
    #[arg(short, long)]
    schema: PathBuf,

    /// Output file. Defaults to stdout.
    #[arg(short, long)]
    out: Option<PathBuf>,

    /// Target language: `rust` or `ts` (`typescript`).
    #[arg(short, long, default_value = "rust")]
    lang: String,
}

fn main() {
    let cli = Cli::parse();

    let sdl = std::fs::read_to_string(&cli.schema).unwrap_or_else(|e| {
        eprintln!("failed to read schema {:?}: {e}", cli.schema);
        std::process::exit(1);
    });

    let schema = rhypedb_schema::parser::parse_schema(&sdl).unwrap_or_else(|e| {
        eprintln!("schema error: {e}");
        std::process::exit(1);
    });

    let code = match cli.lang.as_str() {
        "rust" => rhypedb_codegen::generate_rust(&schema),
        "ts" | "typescript" => rhypedb_codegen::generate_typescript(&schema),
        other => {
            eprintln!("unsupported --lang '{other}': expected 'rust' or 'ts'");
            std::process::exit(2);
        }
    };

    match &cli.out {
        Some(path) => {
            std::fs::write(path, &code).unwrap_or_else(|e| {
                eprintln!("failed to write {path:?}: {e}");
                std::process::exit(1);
            });
            eprintln!("wrote {} type(s) to {path:?}", schema.types.len());
        }
        None => print!("{code}"),
    }
}
