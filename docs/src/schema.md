# Schema Reference

A rhypedb schema is a text file (conventionally `*.rhype`) that declares your object types, their fields, and the relationships between them. The server reads it at startup (`--schema`), and it is the single source of truth for what the database will store and enforce.

```
type User {
    name: String
    email: String @unique
    age: u32
    friends: [User] @on_delete(remove)
    posts: [Post] @inverse(Post.author)
    embedding: Vector<384> @vectorize(source: "name", model: "all-MiniLM-L6-v2")
}
```

## Types

A schema is a collection of `type` declarations. Each `type` has a name (used as-is in queries) and a brace-delimited block of fields:

```
type TypeName {
    field_name: FieldType
    another_field: FieldType @directive
}
```

Fields are separated by newlines (commas are also accepted). Field names may **not** contain a double underscore (`__`) — that sequence is reserved for the engine's internal sidecar keys.

## Field types

### Scalars

| Type | Meaning |
| --- | --- |
| `String` | UTF-8 text |
| `Bool` | boolean |
| `i32`, `i64` | signed integers |
| `u32`, `u64` | unsigned integers |
| `f32`, `f64` | floating point |
| `DateTime` | timestamp |
| `Bytes` | raw byte blob |
| `Json` | arbitrary JSON value |

> Note the casing: the named types (`String`, `Bool`, `DateTime`, `Bytes`, `Json`) are capitalized, while the numeric primitives (`i64`, `u32`, `f64`, …) are lowercase. This matches how you refer to them in queries and migrations.

### Relations

A field whose type is another type name is a **relationship**:

```
author: User          // to-one
posts: [Post]         // to-many
```

Relationships are stored and traversed by the database — there are no foreign-key columns or junction tables in your schema. You create and remove links at query time with `.link(...)` / `.unlink(...)` (see the [Query Language](queries.md)).

### Vectors

```
embedding: Vector<384>
```

`Vector<N>` declares an `N`-dimensional float vector. On its own it stores raw vectors you supply; add an `@index(hnsw, ...)` directive to make it searchable and `@vectorize(...)` to have the server compute embeddings for you. See **[Vector Search](vectors.md)**.

## Edge fields (metadata on relationships)

A relationship can carry its own scalar fields — data that belongs to the *link*, not to either object. Declare them in a block after the relation:

```
type User {
    favorite_movies: [Movie] {
        rating: f32
        added_at: DateTime
    }
}
```

You set edge fields when linking: `User.get(1).favorite_movies.link(Movie.get(42), { rating: 4.5 })`.

## Directives

Directives annotate a field with `@name` or `@name(args)`.

### `@unique`

On a scalar field, enforces that no two objects of the type share a value. A violating insert is rejected atomically.

```
email: String @unique
```

### `@indexed`

On a scalar field, builds a secondary index so `filter` queries on that field don't scan the whole type. Use it for fields you filter or look up by frequently.

```
year: i64 @indexed
```

### `@on_delete(policy)`

On a relationship, declares what happens to the link (and possibly the target) when an object is deleted. Policies:

| Policy | Behavior |
| --- | --- |
| `remove` | Remove the link only; leave the target object. |
| `cascade` | Delete the target object too (recursively). |
| `deny` | Refuse the delete while the link exists. |

```
posts: [Post] @on_delete(cascade)
friends: [User] @on_delete(remove)
```

### `@inverse(Type.field)`

Marks two relationship fields as the two ends of a single bidirectional relationship. Linking one side makes the other side reflect it automatically.

```
type User {
    posts: [Post] @inverse(Post.author)
}
type Post {
    author: User
}
```

### `@index(hnsw, ...)`

On a `Vector<N>` field, configures the HNSW index used for similarity search. All parameters are optional and have defaults.

```
embedding: Vector<768> @index(hnsw, metric: cosine, quantization: turboquant_4bit, m: 16, ef_construction: 200)
```

| Parameter | Values | Default |
| --- | --- | --- |
| `metric` | `cosine`, `l2`, `dot_product` | `cosine` |
| `quantization` | `turboquant_2bit`, `turboquant_3bit`, `turboquant_4bit` | `turboquant_4bit` |
| `m` | HNSW graph fan-out (≥ 1) | engine default |
| `ef_construction` | build-time search width (≥ 1) | engine default |

Fewer quantization bits means a smaller index and faster search at some recall cost; pair it with per-query `rerank` (see [Vector Search](vectors.md)) to recover precision. `quantization: none` is rejected — vectors are always quantized in the index (the raw vectors are still kept losslessly at rest).

### `@vectorize(source, model)`

On a `Vector<N>` field, tells the server to **compute the embedding for you** from a `String` field on the same object whenever it is created or updated.

```
text: String
embedding: Vector<384> @vectorize(source: "text", model: "all-MiniLM-L6-v2")
```

- `source` — the name of a `String` field on this type.
- `model` — the embedding model identifier (e.g. `all-MiniLM-L6-v2`). The vector dimension `N` must match the model's output size.

Combine it with `@index(hnsw, ...)` on the same field to both auto-embed and index. See **[Vector Search](vectors.md)**.

## A complete example

```
type User {
    name: String
    email: String @unique
    reputation: u32 @indexed
    friends: [User] @on_delete(remove)
    posts: [Post] @inverse(Post.author) @on_delete(remove)
    favorite_movies: [Movie] {
        rating: f32
        added_at: DateTime
    }
}

type Post {
    title: String
    body: String
    author: User @on_delete(cascade)
    tags: [Tag] @on_delete(remove)
    embedding: Vector<384> @vectorize(source: "body", model: "all-MiniLM-L6-v2") @index(hnsw, metric: cosine)
}

type Movie {
    title: String
    year: i64 @indexed
}

type Tag {
    name: String @unique
}
```

## Evolving the schema

Changing the schema of a database that already holds data is a real operation with rules — adding fields is free, removing them is gated, and renames or type changes go through a migration system. See **[Schema Migrations](migrations.md)**.
