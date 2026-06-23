//! The typed query builder ([`Query`]) and result row ([`Row`]).
//!
//! These are schema-agnostic, so they live in the client (not in generated
//! code). The codegen emits thin per-type seed constructors (`User::all()`,
//! `User::get(id)`, …) that return a `Query<User>`; `User` only needs to
//! `derive(Deserialize)` for [`Row::from_object`] to materialize it.

use std::marker::PhantomData;

use rhypedb_wire::object::{value_to_query_json, Object};
use serde::de::DeserializeOwned;

use crate::Error;

/// A typed query string targeting one object type `T`.
///
/// Build it from a seed constructor ([`Query::all`] / [`Query::get`] /
/// [`Query::filter_on`]) or [`Query::raw`], optionally chain [`filter`](Self::filter)
/// / [`limit`](Self::limit), then hand it to a [`Client`](crate::Client) method
/// (`fetch`, `fetch_one`, `create`, `execute`). This is a thin string builder;
/// the server validates query composition.
#[derive(Debug, Clone)]
pub struct Query<T> {
    text: String,
    _row: PhantomData<fn() -> T>,
}

impl<T> Query<T> {
    /// Wrap a raw query string. Escape hatch + the primitive the generated
    /// seed constructors build on.
    pub fn raw(text: impl Into<String>) -> Self {
        Query {
            text: text.into(),
            _row: PhantomData,
        }
    }

    /// `Type` — all objects of the type (a bare type name is "all" in the QL).
    pub fn all(type_name: &str) -> Self {
        Query::raw(type_name.to_string())
    }

    /// `Type.get(id)` — a single object by id.
    pub fn get(type_name: &str, id: u64) -> Self {
        Query::raw(format!("{type_name}.get({id})"))
    }

    /// `Type.filter(<predicate>)` — objects matching a predicate.
    pub fn filter_on(type_name: &str, predicate: &str) -> Self {
        Query::raw(format!("{type_name}.filter({predicate})"))
    }

    /// Append `.filter(<predicate>)`, e.g. `.filter(".age > 18")`.
    #[must_use]
    pub fn filter(mut self, predicate: &str) -> Self {
        self.text.push_str(".filter(");
        self.text.push_str(predicate);
        self.text.push(')');
        self
    }

    /// Append `.limit(<n>)`.
    #[must_use]
    pub fn limit(mut self, n: u64) -> Self {
        self.text.push_str(&format!(".limit({n})"));
        self
    }

    /// Borrow the query string (what the client sends).
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Consume the builder, returning the query string.
    pub fn build(self) -> String {
        self.text
    }
}

impl<T> std::fmt::Display for Query<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

/// A query-result row: the object `id` plus its typed scalar fields.
#[derive(Debug, Clone)]
pub struct Row<T> {
    pub id: u64,
    pub data: T,
}

impl<T: DeserializeOwned> Row<T> {
    /// Materialize a typed row from a decoded wire [`Object`].
    ///
    /// Fields are rendered through the canonical [`value_to_query_json`] (Bytes
    /// → base64, DateTime → RFC 3339, Json inline) — identical to the `/query`
    /// JSON form — then deserialized into `T`, so a struct generated for the
    /// HTTP path deserializes unchanged here.
    pub fn from_object(obj: &Object) -> Result<Self, Error> {
        let mut map = serde_json::Map::with_capacity(obj.fields.len());
        for (name, value) in &obj.fields {
            map.insert(name.clone(), value_to_query_json(value));
        }
        let data = serde_json::from_value(serde_json::Value::Object(map))
            .map_err(|e| Error::Deserialize(e.to_string()))?;
        Ok(Row { id: obj.id, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhypedb_wire::object::{FieldMap, Value};

    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct User {
        name: Option<String>,
        age: Option<u32>,
    }

    #[test]
    fn query_builder_composes_expected_strings() {
        assert_eq!(Query::<User>::all("User").as_str(), "User");
        assert_eq!(Query::<User>::get("User", 7).as_str(), "User.get(7)");
        assert_eq!(
            Query::<User>::filter_on("User", ".age > 18").as_str(),
            "User.filter(.age > 18)"
        );
        assert_eq!(
            Query::<User>::all("User")
                .filter(".age > 18")
                .limit(5)
                .build(),
            "User.filter(.age > 18).limit(5)"
        );
        assert_eq!(Query::<User>::raw("anything").as_str(), "anything");
    }

    #[test]
    fn row_from_object_materializes_typed_fields() {
        let mut fields = FieldMap::new();
        fields.insert("name".into(), Value::String("Ada".into()));
        fields.insert("age".into(), Value::U32(36));
        let obj = Object {
            type_name: "User".into(),
            id: 42,
            fields,
            raw_fields: None,
        };
        let row = Row::<User>::from_object(&obj).unwrap();
        assert_eq!(row.id, 42);
        assert_eq!(
            row.data,
            User {
                name: Some("Ada".into()),
                age: Some(36),
            }
        );
    }

    #[test]
    fn row_from_object_tolerates_missing_fields() {
        // Column-projected results omit fields; Option<T> fields stay None.
        let obj = Object {
            type_name: "User".into(),
            id: 1,
            fields: FieldMap::new(),
            raw_fields: None,
        };
        let row = Row::<User>::from_object(&obj).unwrap();
        assert_eq!(row.data, User { name: None, age: None });
    }
}
