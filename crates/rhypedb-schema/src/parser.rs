use std::collections::HashMap;

use crate::error::{SchemaError, SchemaResult};
use crate::types::*;

/// Parse an SDL string into a validated Schema.
pub fn parse_schema(input: &str) -> SchemaResult<Schema> {
    let mut parser = Parser::new(input);
    let mut types = HashMap::new();

    parser.skip_whitespace();
    while !parser.is_eof() {
        let type_def = parser.parse_type_def()?;
        if types.contains_key(&type_def.name) {
            return Err(SchemaError::DuplicateType(type_def.name));
        }
        types.insert(type_def.name.clone(), type_def);
        parser.skip_whitespace();
    }

    let schema = Schema { types };
    validate_schema(&schema)?;
    Ok(schema)
}

fn validate_schema(schema: &Schema) -> SchemaResult<()> {
    for type_def in schema.types.values() {
        let mut field_names = std::collections::HashSet::new();
        for field in &type_def.fields {
            if !field_names.insert(&field.name) {
                return Err(SchemaError::DuplicateField {
                    type_name: type_def.name.clone(),
                    field: field.name.clone(),
                });
            }

            // Validate relationship targets exist.
            if let FieldType::Relation(rel) = &field.field_type
                && !schema.types.contains_key(&rel.target_type) {
                    return Err(SchemaError::UnknownType(rel.target_type.clone()));
                }

            // Validate @inverse references.
            if let Some(inv) = field.inverse() {
                let target_type = schema
                    .types
                    .get(&inv.type_name)
                    .ok_or_else(|| SchemaError::InvalidInverse(format!(
                        "type '{}' not found", inv.type_name
                    )))?;
                if target_type.get_field(&inv.field_name).is_none() {
                    return Err(SchemaError::InvalidInverse(format!(
                        "field '{}.{}' not found",
                        inv.type_name, inv.field_name
                    )));
                }
            }
        }
    }
    Ok(())
}

struct Parser<'a> {
    input: &'a str,
    pos: usize,
    line: usize,
    col: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else if ch == '/' && self.input[self.pos..].starts_with("//") {
                // Line comment.
                while let Some(c) = self.advance() {
                    if c == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, expected: char) -> SchemaResult<()> {
        self.skip_whitespace();
        match self.advance() {
            Some(ch) if ch == expected => Ok(()),
            Some(ch) => Err(self.error(format!("expected '{expected}', got '{ch}'"))),
            None => Err(self.error(format!("expected '{expected}', got EOF"))),
        }
    }

    fn parse_ident(&mut self) -> SchemaResult<String> {
        self.skip_whitespace();
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(self.error("expected identifier".into()));
        }
        Ok(self.input[start..self.pos].to_string())
    }

    fn parse_u32(&mut self) -> SchemaResult<u32> {
        self.skip_whitespace();
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(self.error("expected number".into()));
        }
        self.input[start..self.pos]
            .parse()
            .map_err(|_| self.error("invalid number".into()))
    }

    fn error(&self, message: String) -> SchemaError {
        SchemaError::Parse {
            line: self.line,
            col: self.col,
            message,
        }
    }

    fn parse_type_def(&mut self) -> SchemaResult<TypeDef> {
        self.skip_whitespace();
        let keyword = self.parse_ident()?;
        if keyword != "type" {
            return Err(self.error(format!("expected 'type', got '{keyword}'")));
        }

        let name = self.parse_ident()?;
        self.expect('{')?;

        let mut fields = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek() == Some('}') {
                self.advance();
                break;
            }
            fields.push(self.parse_field_def()?);
        }

        Ok(TypeDef { name, fields })
    }

    fn parse_field_def(&mut self) -> SchemaResult<FieldDef> {
        let name = self.parse_ident()?;
        self.expect(':')?;

        let field_type = self.parse_field_type()?;

        // Parse optional edge fields for relationships: { rating: f32, ... }
        let field_type = if let FieldType::Relation(mut rel) = field_type {
            self.skip_whitespace();
            if self.peek() == Some('{') {
                rel.edge_fields = self.parse_edge_fields()?;
            }
            FieldType::Relation(rel)
        } else {
            field_type
        };

        // Parse directives.
        let mut directives = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek() == Some('@') {
                self.advance();
                directives.push(self.parse_directive()?);
            } else {
                break;
            }
        }

        Ok(FieldDef {
            name,
            field_type,
            directives,
        })
    }

    fn parse_field_type(&mut self) -> SchemaResult<FieldType> {
        self.skip_whitespace();

        // Check for array type: [TypeName]
        if self.peek() == Some('[') {
            self.advance();
            let inner = self.parse_ident()?;
            self.expect(']')?;

            // Could be [Vector<N>] but that doesn't make sense — vectors aren't arrays of types.
            // Arrays are always relationships.
            return Ok(FieldType::Relation(RelationType {
                target_type: inner,
                is_many: true,
                edge_fields: Vec::new(),
            }));
        }

        let type_name = self.parse_ident()?;

        // Check for Vector<N>
        if type_name == "Vector" {
            self.expect('<')?;
            let dims = self.parse_u32()?;
            self.expect('>')?;
            return Ok(FieldType::Vector(VectorType { dimensions: dims }));
        }

        // Check if this is a scalar type.
        if let Some(scalar) = parse_scalar_type(&type_name) {
            return Ok(FieldType::Scalar(scalar));
        }

        // Otherwise it's a singular relationship.
        Ok(FieldType::Relation(RelationType {
            target_type: type_name,
            is_many: false,
            edge_fields: Vec::new(),
        }))
    }

    fn parse_edge_fields(&mut self) -> SchemaResult<Vec<EdgeFieldDef>> {
        self.expect('{')?;
        let mut fields = Vec::new();

        loop {
            self.skip_whitespace();
            if self.peek() == Some('}') {
                self.advance();
                break;
            }

            let name = self.parse_ident()?;
            self.expect(':')?;
            let type_name = self.parse_ident()?;
            let scalar_type = parse_scalar_type(&type_name).ok_or_else(|| {
                self.error(format!("edge field must be a scalar type, got '{type_name}'"))
            })?;

            fields.push(EdgeFieldDef { name, scalar_type });

            // Optional comma.
            self.skip_whitespace();
            if self.peek() == Some(',') {
                self.advance();
            }
        }

        Ok(fields)
    }

    fn parse_directive(&mut self) -> SchemaResult<Directive> {
        let name = self.parse_ident()?;

        match name.as_str() {
            "unique" => Ok(Directive::Unique),

            "on_delete" => {
                self.expect('(')?;
                let policy_name = self.parse_ident()?;
                self.expect(')')?;
                let policy = match policy_name.as_str() {
                    "remove" => OnDeletePolicy::Remove,
                    "cascade" => OnDeletePolicy::Cascade,
                    "deny" => OnDeletePolicy::Deny,
                    _ => {
                        return Err(SchemaError::InvalidOnDelete(policy_name));
                    }
                };
                Ok(Directive::OnDelete(policy))
            }

            "inverse" => {
                self.expect('(')?;
                let type_name = self.parse_ident()?;
                self.expect('.')?;
                let field_name = self.parse_ident()?;
                self.expect(')')?;
                Ok(Directive::Inverse(InverseDef {
                    type_name,
                    field_name,
                }))
            }

            "index" => {
                self.expect('(')?;
                let index_type_name = self.parse_ident()?;
                let index_type = match index_type_name.as_str() {
                    "hnsw" => IndexType::Hnsw,
                    _ => return Err(self.error(format!("unknown index type: {index_type_name}"))),
                };

                let mut metric = None;
                let mut quantization = None;
                let mut m = None;
                let mut ef_construction = None;

                // Parse optional key-value parameters.
                loop {
                    self.skip_whitespace();
                    if self.peek() == Some(')') {
                        self.advance();
                        break;
                    }
                    self.expect(',')?;
                    let key = self.parse_ident()?;
                    self.expect(':')?;

                    match key.as_str() {
                        "metric" => {
                            let val = self.parse_ident()?;
                            metric = Some(match val.as_str() {
                                "cosine" => DistanceMetric::Cosine,
                                "l2" => DistanceMetric::L2,
                                "dot_product" => DistanceMetric::DotProduct,
                                _ => return Err(self.error(format!("unknown metric: {val}"))),
                            });
                        }
                        "quantization" => {
                            let val = self.parse_ident()?;
                            quantization = Some(match val.as_str() {
                                "turboquant_2bit" => QuantizationType::TurboQuant2Bit,
                                "turboquant_3bit" => QuantizationType::TurboQuant3Bit,
                                "turboquant_4bit" => QuantizationType::TurboQuant4Bit,
                                "none" => QuantizationType::None,
                                _ => {
                                    return Err(
                                        self.error(format!("unknown quantization: {val}"))
                                    )
                                }
                            });
                        }
                        "m" => {
                            m = Some(self.parse_u32()?);
                        }
                        "ef_construction" => {
                            ef_construction = Some(self.parse_u32()?);
                        }
                        _ => return Err(self.error(format!("unknown index parameter: {key}"))),
                    }
                }

                Ok(Directive::Index(IndexDef {
                    index_type,
                    metric,
                    quantization,
                    m,
                    ef_construction,
                }))
            }

            _ => Err(self.error(format!("unknown directive: @{name}"))),
        }
    }
}

fn parse_scalar_type(name: &str) -> Option<ScalarType> {
    match name {
        "String" => Some(ScalarType::String),
        "u32" => Some(ScalarType::U32),
        "u64" => Some(ScalarType::U64),
        "i32" => Some(ScalarType::I32),
        "i64" => Some(ScalarType::I64),
        "f32" => Some(ScalarType::F32),
        "f64" => Some(ScalarType::F64),
        "Bool" => Some(ScalarType::Bool),
        "DateTime" => Some(ScalarType::DateTime),
        "Bytes" => Some(ScalarType::Bytes),
        "Json" => Some(ScalarType::Json),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_type() {
        let schema = parse_schema(
            r#"
            type User {
                name: String
                age: u32
                active: Bool
            }
            "#,
        )
        .unwrap();

        let user = schema.get_type("User").unwrap();
        assert_eq!(user.fields.len(), 3);
        assert_eq!(user.fields[0].name, "name");
        assert_eq!(
            user.fields[0].field_type,
            FieldType::Scalar(ScalarType::String)
        );
        assert_eq!(user.fields[1].field_type, FieldType::Scalar(ScalarType::U32));
        assert_eq!(
            user.fields[2].field_type,
            FieldType::Scalar(ScalarType::Bool)
        );
    }

    #[test]
    fn parse_relationships() {
        let schema = parse_schema(
            r#"
            type User {
                name: String
                best_friend: User
                posts: [Post]
            }

            type Post {
                title: String
                author: User
            }
            "#,
        )
        .unwrap();

        let user = schema.get_type("User").unwrap();

        // Singular relationship.
        let best_friend = user.get_field("best_friend").unwrap();
        match &best_friend.field_type {
            FieldType::Relation(rel) => {
                assert_eq!(rel.target_type, "User");
                assert!(!rel.is_many);
            }
            _ => panic!("expected relation"),
        }

        // To-many relationship.
        let posts = user.get_field("posts").unwrap();
        match &posts.field_type {
            FieldType::Relation(rel) => {
                assert_eq!(rel.target_type, "Post");
                assert!(rel.is_many);
            }
            _ => panic!("expected relation"),
        }
    }

    #[test]
    fn parse_vector_type() {
        let schema = parse_schema(
            r#"
            type Doc {
                embedding: Vector<1536>
            }
            "#,
        )
        .unwrap();

        let doc = schema.get_type("Doc").unwrap();
        let emb = doc.get_field("embedding").unwrap();
        assert_eq!(
            emb.field_type,
            FieldType::Vector(VectorType { dimensions: 1536 })
        );
    }

    #[test]
    fn parse_directives() {
        let schema = parse_schema(
            r#"
            type User {
                email: String @unique
                posts: [Post] @on_delete(cascade) @inverse(Post.author)
            }

            type Post {
                author: User @on_delete(deny)
            }
            "#,
        )
        .unwrap();

        let user = schema.get_type("User").unwrap();

        assert!(user.get_field("email").unwrap().is_unique());

        let posts = user.get_field("posts").unwrap();
        assert_eq!(
            posts.on_delete(),
            Some(&OnDeletePolicy::Cascade)
        );
        let inv = posts.inverse().unwrap();
        assert_eq!(inv.type_name, "Post");
        assert_eq!(inv.field_name, "author");

        let post = schema.get_type("Post").unwrap();
        assert_eq!(
            post.get_field("author").unwrap().on_delete(),
            Some(&OnDeletePolicy::Deny)
        );
    }

    #[test]
    fn parse_edge_fields() {
        let schema = parse_schema(
            r#"
            type User {
                favorite_movies: [Movie] {
                    rating: f32,
                    added_at: DateTime
                } @on_delete(remove)
            }

            type Movie {
                title: String
            }
            "#,
        )
        .unwrap();

        let user = schema.get_type("User").unwrap();
        let fav = user.get_field("favorite_movies").unwrap();

        match &fav.field_type {
            FieldType::Relation(rel) => {
                assert_eq!(rel.edge_fields.len(), 2);
                assert_eq!(rel.edge_fields[0].name, "rating");
                assert_eq!(rel.edge_fields[0].scalar_type, ScalarType::F32);
                assert_eq!(rel.edge_fields[1].name, "added_at");
                assert_eq!(rel.edge_fields[1].scalar_type, ScalarType::DateTime);
            }
            _ => panic!("expected relation"),
        }
    }

    #[test]
    fn parse_index_directive() {
        let schema = parse_schema(
            r#"
            type Product {
                embedding: Vector<768> @index(hnsw, metric: cosine, quantization: turboquant_3bit, m: 16, ef_construction: 200)
            }
            "#,
        )
        .unwrap();

        let product = schema.get_type("Product").unwrap();
        let emb = product.get_field("embedding").unwrap();
        let idx = emb.directives.iter().find_map(|d| match d {
            Directive::Index(i) => Some(i),
            _ => None,
        });
        let idx = idx.unwrap();
        assert_eq!(idx.index_type, IndexType::Hnsw);
        assert_eq!(idx.metric, Some(DistanceMetric::Cosine));
        assert_eq!(idx.quantization, Some(QuantizationType::TurboQuant3Bit));
        assert_eq!(idx.m, Some(16));
        assert_eq!(idx.ef_construction, Some(200));
    }

    #[test]
    fn parse_comments() {
        let schema = parse_schema(
            r#"
            // This is a comment
            type User {
                // Name of the user
                name: String
                age: u32  // inline comment
            }
            "#,
        )
        .unwrap();

        let user = schema.get_type("User").unwrap();
        assert_eq!(user.fields.len(), 2);
    }

    #[test]
    fn reject_duplicate_type() {
        let result = parse_schema(
            r#"
            type User { name: String }
            type User { email: String }
            "#,
        );
        assert!(matches!(result, Err(SchemaError::DuplicateType(_))));
    }

    #[test]
    fn reject_unknown_relationship_target() {
        let result = parse_schema(
            r#"
            type User {
                posts: [NonExistent]
            }
            "#,
        );
        assert!(matches!(result, Err(SchemaError::UnknownType(_))));
    }

    #[test]
    fn reject_invalid_inverse() {
        let result = parse_schema(
            r#"
            type User {
                posts: [Post] @inverse(Post.nonexistent)
            }
            type Post {
                author: User
            }
            "#,
        );
        assert!(matches!(result, Err(SchemaError::InvalidInverse(_))));
    }

    #[test]
    fn full_schema_example() {
        let schema = parse_schema(
            r#"
            type User {
                name: String
                email: String @unique
                reputation: u32
                friends: [User] @on_delete(remove)
                posts: [Post] @inverse(Post.author) @on_delete(remove)
                embedding: Vector<1536>
            }

            type Post {
                title: String
                body: String
                author: User @on_delete(cascade)
                tags: [Tag] @on_delete(remove)
                embedding: Vector<1536>
            }

            type Tag {
                name: String @unique
            }
            "#,
        )
        .unwrap();

        assert_eq!(schema.types.len(), 3);
        assert!(schema.get_type("User").is_some());
        assert!(schema.get_type("Post").is_some());
        assert!(schema.get_type("Tag").is_some());

        let user = schema.get_type("User").unwrap();
        assert_eq!(user.scalar_fields().count(), 3); // name, email, reputation
        assert_eq!(user.relationship_fields().count(), 2); // friends, posts
        assert_eq!(user.vector_fields().count(), 1); // embedding
    }
}
