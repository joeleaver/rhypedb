use std::collections::HashMap;

/// A complete schema — all types and their relationships.
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub types: HashMap<String, TypeDef>,
}

impl Schema {
    pub fn get_type(&self, name: &str) -> Option<&TypeDef> {
        self.types.get(name)
    }

    pub fn type_names(&self) -> impl Iterator<Item = &str> {
        self.types.keys().map(|s| s.as_str())
    }

    /// Return a clone of this schema with `type_name.field_name`'s scalar/field
    /// type replaced by `field_type`, leaving every directive
    /// (`@indexed`/`@unique`/...) intact. Returns `None` if the type or field
    /// does not exist. Used by the server to build the post-cutover target
    /// schema for an in-place hot-reload after a `change_field_type` migration —
    /// the one field's kind is flipped, everything else carries verbatim.
    pub fn with_field_type(&self, type_name: &str, field_name: &str, field_type: FieldType) -> Option<Schema> {
        let mut next = self.clone();
        let td = next.types.get_mut(type_name)?;
        let fd = td.fields.iter_mut().find(|f| f.name == field_name)?;
        fd.field_type = field_type;
        Some(next)
    }
}

/// Definition of a single object type.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
}

impl TypeDef {
    pub fn get_field(&self, name: &str) -> Option<&FieldDef> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn scalar_fields(&self) -> impl Iterator<Item = &FieldDef> {
        self.fields.iter().filter(|f| matches!(f.field_type, FieldType::Scalar(_)))
    }

    pub fn relationship_fields(&self) -> impl Iterator<Item = &FieldDef> {
        self.fields.iter().filter(|f| matches!(f.field_type, FieldType::Relation(_)))
    }

    pub fn vector_fields(&self) -> impl Iterator<Item = &FieldDef> {
        self.fields.iter().filter(|f| matches!(f.field_type, FieldType::Vector(_)))
    }
}

/// Definition of a single field within a type.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    pub name: String,
    pub field_type: FieldType,
    pub directives: Vec<Directive>,
}

impl FieldDef {
    pub fn is_unique(&self) -> bool {
        self.directives.iter().any(|d| matches!(d, Directive::Unique))
    }

    pub fn is_indexed(&self) -> bool {
        self.directives.iter().any(|d| matches!(d, Directive::Indexed))
    }

    pub fn on_delete(&self) -> Option<&OnDeletePolicy> {
        self.directives.iter().find_map(|d| match d {
            Directive::OnDelete(policy) => Some(policy),
            _ => None,
        })
    }

    pub fn inverse(&self) -> Option<&InverseDef> {
        self.directives.iter().find_map(|d| match d {
            Directive::Inverse(inv) => Some(inv),
            _ => None,
        })
    }

    pub fn vectorize(&self) -> Option<&VectorizeDef> {
        self.directives.iter().find_map(|d| match d {
            Directive::Vectorize(v) => Some(v),
            _ => None,
        })
    }

    /// The `@index(hnsw, ...)` directive configuring this field's vector index
    /// (metric / quantization bit-width / m / ef_construction), if present.
    /// Only meaningful on `Vector` fields; `validate_schema` rejects it
    /// elsewhere.
    pub fn index(&self) -> Option<&IndexDef> {
        self.directives.iter().find_map(|d| match d {
            Directive::Index(i) => Some(i),
            _ => None,
        })
    }
}

/// The type of a field — scalar, relationship, or vector.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    Scalar(ScalarType),
    Relation(RelationType),
    Vector(VectorType),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarType {
    String,
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
    Bool,
    DateTime,
    Bytes,
    Json,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelationType {
    pub target_type: String,
    pub is_many: bool,
    pub edge_fields: Vec<EdgeFieldDef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeFieldDef {
    pub name: String,
    pub scalar_type: ScalarType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorType {
    pub dimensions: u32,
}

/// Schema directives (annotations on fields).
#[derive(Debug, Clone, PartialEq)]
pub enum Directive {
    Unique,
    /// Non-unique secondary index on a scalar integer field. Enables
    /// `Type.filter(.field op N)` to skip the full type scan in favor of an
    /// `idx:` prefix scan that returns matching object IDs directly.
    Indexed,
    OnDelete(OnDeletePolicy),
    Inverse(InverseDef),
    Index(IndexDef),
    Vectorize(VectorizeDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorizeDef {
    pub source_field: String,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OnDeletePolicy {
    Remove,
    Cascade,
    Deny,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InverseDef {
    pub type_name: String,
    pub field_name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexDef {
    pub index_type: IndexType,
    pub metric: Option<DistanceMetric>,
    pub quantization: Option<QuantizationType>,
    pub m: Option<u32>,
    pub ef_construction: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexType {
    Hnsw,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    Cosine,
    L2,
    DotProduct,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationType {
    TurboQuant2Bit,
    TurboQuant3Bit,
    TurboQuant4Bit,
    None,
}

#[cfg(test)]
mod with_field_type_tests {
    use crate::parser::parse_schema;
    use crate::types::{FieldType, ScalarType};

    #[test]
    fn flips_one_field_kind_preserving_directives_and_other_fields() {
        let s = parse_schema(r#"type User { id: i64 @unique  score: i64 @indexed  name: String }"#)
            .unwrap();
        let out = s
            .with_field_type("User", "score", FieldType::Scalar(ScalarType::F64))
            .expect("type+field exist");
        let user = out.get_type("User").unwrap();
        // The migrated field flipped to F64 but kept its @indexed directive.
        let score = user.get_field("score").unwrap();
        assert_eq!(score.field_type, FieldType::Scalar(ScalarType::F64));
        assert!(score.is_indexed(), "directives are preserved across the flip");
        // Sibling fields are untouched.
        assert_eq!(
            user.get_field("id").unwrap().field_type,
            FieldType::Scalar(ScalarType::I64)
        );
        assert!(user.get_field("id").unwrap().is_unique());
        assert_eq!(
            user.get_field("name").unwrap().field_type,
            FieldType::Scalar(ScalarType::String)
        );
    }

    #[test]
    fn returns_none_for_unknown_type_or_field() {
        let s = parse_schema(r#"type User { score: i64 }"#).unwrap();
        assert!(s
            .with_field_type("Nope", "score", FieldType::Scalar(ScalarType::F64))
            .is_none());
        assert!(s
            .with_field_type("User", "nope", FieldType::Scalar(ScalarType::F64))
            .is_none());
    }
}
