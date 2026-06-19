//! Emit a [`Schema`] back to SDL text — the structural inverse of
//! [`crate::parser::parse_schema`].
//!
//! The logical-export feature embeds a portable, version-independent schema
//! description in its dump. The correct source for that description is the
//! *live* schema (`Database::schema()`), not the operator's on-disk SDL file:
//! after a completed `rename_type` / `rename_field` / `change_field_type`
//! migration + hot reload, the on-disk file goes stale relative to the values
//! actually stored, so embedding it would emit field/type names that mismatch
//! the exported objects. This emitter renders the live schema instead.
//!
//! Round-trip contract (structural, not textual): for any schema `s`,
//! `parse_schema(&emit_schema(&s)).unwrap() == s`. Comments and the original
//! whitespace are not preserved, but every type, field, field-type, edge
//! field, and directive does. Types are emitted sorted by name and fields in
//! declaration order, so the output is deterministic.

use std::fmt::Write as _;

use crate::types::*;

/// Serialize a whole schema to SDL text that [`crate::parser::parse_schema`]
/// round-trips.
pub fn emit_schema(schema: &Schema) -> String {
    let mut names: Vec<&str> = schema.types.keys().map(String::as_str).collect();
    names.sort_unstable();

    let mut out = String::new();
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        emit_type(&mut out, &schema.types[*name]);
    }
    out
}

fn emit_type(out: &mut String, td: &TypeDef) {
    let _ = writeln!(out, "type {} {{", td.name);
    for field in &td.fields {
        out.push_str("    ");
        emit_field(out, field);
        out.push('\n');
    }
    out.push_str("}\n");
}

fn emit_field(out: &mut String, field: &FieldDef) {
    let _ = write!(out, "{}: ", field.name);
    emit_field_type(out, &field.field_type);
    for directive in &field.directives {
        out.push(' ');
        emit_directive(out, directive);
    }
}

fn emit_field_type(out: &mut String, ft: &FieldType) {
    match ft {
        FieldType::Scalar(s) => out.push_str(scalar_name(s)),
        FieldType::Vector(v) => {
            let _ = write!(out, "Vector<{}>", v.dimensions);
        }
        FieldType::Relation(rel) => {
            if rel.is_many {
                let _ = write!(out, "[{}]", rel.target_type);
            } else {
                out.push_str(&rel.target_type);
            }
            // Edge fields are parsed immediately after the relation type and
            // before any directive, so they must be emitted in that position.
            if !rel.edge_fields.is_empty() {
                out.push_str(" { ");
                for (i, ef) in rel.edge_fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{}: {}", ef.name, scalar_name(&ef.scalar_type));
                }
                out.push_str(" }");
            }
        }
    }
}

fn emit_directive(out: &mut String, d: &Directive) {
    match d {
        Directive::Unique => out.push_str("@unique"),
        Directive::Indexed => out.push_str("@indexed"),
        Directive::OnDelete(policy) => {
            let name = match policy {
                OnDeletePolicy::Remove => "remove",
                OnDeletePolicy::Cascade => "cascade",
                OnDeletePolicy::Deny => "deny",
            };
            let _ = write!(out, "@on_delete({name})");
        }
        Directive::Inverse(inv) => {
            let _ = write!(out, "@inverse({}.{})", inv.type_name, inv.field_name);
        }
        Directive::Index(idx) => emit_index(out, idx),
        Directive::Vectorize(v) => {
            let _ = write!(
                out,
                "@vectorize(source: \"{}\", model: \"{}\")",
                v.source_field, v.model
            );
        }
    }
}

fn emit_index(out: &mut String, idx: &IndexDef) {
    out.push_str("@index(");
    match idx.index_type {
        IndexType::Hnsw => out.push_str("hnsw"),
    }
    if let Some(metric) = &idx.metric {
        let name = match metric {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::L2 => "l2",
            DistanceMetric::DotProduct => "dot_product",
        };
        let _ = write!(out, ", metric: {name}");
    }
    if let Some(q) = &idx.quantization {
        // `none` is rejected by `validate_schema`, so a live schema never
        // carries it; emitted only for completeness / symmetry.
        let name = match q {
            QuantizationType::TurboQuant2Bit => "turboquant_2bit",
            QuantizationType::TurboQuant3Bit => "turboquant_3bit",
            QuantizationType::TurboQuant4Bit => "turboquant_4bit",
            QuantizationType::None => "none",
        };
        let _ = write!(out, ", quantization: {name}");
    }
    if let Some(m) = idx.m {
        let _ = write!(out, ", m: {m}");
    }
    if let Some(ef) = idx.ef_construction {
        let _ = write!(out, ", ef_construction: {ef}");
    }
    out.push(')');
}

/// The SDL spelling of a scalar type — exactly what
/// [`crate::parser`]'s `parse_scalar_type` accepts.
fn scalar_name(s: &ScalarType) -> &'static str {
    match s {
        ScalarType::String => "String",
        ScalarType::U32 => "u32",
        ScalarType::U64 => "u64",
        ScalarType::I32 => "i32",
        ScalarType::I64 => "i64",
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
        ScalarType::Bool => "Bool",
        ScalarType::DateTime => "DateTime",
        ScalarType::Bytes => "Bytes",
        ScalarType::Json => "Json",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_schema;

    /// Parse `sdl`, emit it back, re-parse, and assert structural identity.
    /// Also asserts the emitted text itself parses cleanly.
    fn assert_round_trips(sdl: &str) -> Schema {
        let original = parse_schema(sdl).expect("fixture SDL parses");
        let emitted = emit_schema(&original);
        let reparsed = parse_schema(&emitted)
            .unwrap_or_else(|e| panic!("emitted SDL must re-parse, got {e:?}\n---\n{emitted}\n---"));
        assert_eq!(original, reparsed, "emitted SDL:\n{emitted}");
        reparsed
    }

    #[test]
    fn every_scalar_type_round_trips() {
        assert_round_trips(
            r#"
            type Everything {
                s: String
                a: u32
                b: u64
                c: i32
                d: i64
                e: f32
                f: f64
                g: Bool
                h: DateTime
                i: Bytes
                j: Json
            }
            "#,
        );
    }

    #[test]
    fn vectors_relations_and_edge_fields_round_trip() {
        assert_round_trips(
            r#"
            type User {
                name: String @unique
                best_friend: User
                manager: User { since: DateTime }
                posts: [Post] @inverse(Post.author) @on_delete(remove)
                favorites: [Post] {
                    rating: f32,
                    added_at: DateTime
                } @on_delete(cascade)
                embedding: Vector<384> @vectorize(source: "name", model: "all-MiniLM-L6-v2") @index(hnsw, metric: cosine, quantization: turboquant_3bit, m: 16, ef_construction: 200)
            }

            type Post {
                title: String @indexed
                author: User @on_delete(deny)
                vec: Vector<8>
            }
            "#,
        );
    }

    #[test]
    fn partial_index_params_round_trip() {
        // Only some @index params present — the emitter must omit the rest,
        // not emit defaults.
        assert_round_trips(r#"type T { e: Vector<8> @index(hnsw, m: 8) }"#);
        assert_round_trips(r#"type T { e: Vector<8> @index(hnsw, metric: l2) }"#);
        assert_round_trips(r#"type T { e: Vector<8> @index(hnsw) }"#);
    }

    #[test]
    fn on_delete_each_policy_round_trips() {
        assert_round_trips(
            r#"
            type A { b: B @on_delete(remove) }
            type B { c: A @on_delete(cascade)  d: A @on_delete(deny) }
            "#,
        );
    }

    #[test]
    fn output_is_deterministic() {
        let s = parse_schema(
            r#"
            type Zebra { name: String }
            type Apple { core: i32 }
            type Mango { sweet: Bool }
            "#,
        )
        .unwrap();
        let a = emit_schema(&s);
        let b = emit_schema(&s);
        assert_eq!(a, b, "emit must be deterministic");
        // Sorted by type name: Apple before Mango before Zebra.
        let apple = a.find("type Apple").unwrap();
        let mango = a.find("type Mango").unwrap();
        let zebra = a.find("type Zebra").unwrap();
        assert!(apple < mango && mango < zebra, "types emitted sorted by name:\n{a}");
    }
}
