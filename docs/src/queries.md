# Query Language

rhypedb has a single, path-based query language used for reads, writes, relationship traversal, and vector search. A query reads left to right: it starts from a **source** and then applies zero or more **steps**, each separated by a dot.

```
User.get(1).posts.filter(.title == "Hello").limit(10)
└─ source ─┘└ step ┘└────── step ──────────┘└─ step ─┘
```

You send a query as a string — via the CLI (`-e '<query>'` or the REPL), `POST /query`, or the binary TCP protocol. The result comes back as JSON. See the **[API Reference](api-reference.md)** for response shapes.

## Sources

Every query begins with one of these:

| Source | Meaning |
| --- | --- |
| `Type.get(id)` | Fetch one object by numeric id. |
| `Type.filter(predicate)` | All objects of `Type` matching the predicate. |
| `Type` | All objects of `Type`. |
| `Type.create({ ... })` | Create one object; returns it (with its assigned `id`). |
| `Type.create_batch([{ ... }, ...])` | Create many objects in one call. |

```
User.get(42)
User.filter(.age >= 18)
User
User.create({ name: "Alice", email: "alice@example.com", age: 30 })
User.create_batch([{ name: "Bob" }, { name: "Carol" }])
```

Object ids are assigned by the server. You reference other objects by embedding a `get` in a create:

```
Post.create({ title: "Hello", author: User.get(1) })
```

## Steps

Steps transform or act on the current set of objects.

### Traverse a relationship — `.field`

Follow a relationship field to the objects on the other side. Traversals chain:

```
User.get(1).posts                       // posts authored by user 1
User.get(1).friends.posts               // posts by user 1's friends
User.get(1).posts.tags                  // tags on user 1's posts
```

### Filter — `.filter(predicate)`

Keep only objects matching a predicate.

```
User.filter(.age > 18)
User.get(1).posts.filter(.title == "Hello")
```

Inside a predicate, `.field` refers to a field of the current object.

**Comparison operators:** `==`, `!=`, `<`, `<=`, `>`, `>=`

**Boolean operators:** `&&` (or `and`), `||` (or `or`), with parentheses for grouping. `&&` binds tighter than `||`.

```
User.filter(.age >= 18 && .active == true)
User.filter(.age < 13 || .age >= 65)
User.filter((.role == "admin" || .role == "mod") && .active == true)
```

### Vector similarity — `.similar(.field, query, k: N, ...)`

Find the nearest neighbours of a query in a `Vector` field. Covered in depth in **[Vector Search](vectors.md)**.

```
Post.similar(.embedding, "distributed systems", k: 10)        // text query (auto-embedded)
Post.similar(.embedding, [0.1, 0.2, 0.3], k: 10)              // raw vector query
Post.similar(.embedding, "databases", k: 10, ef: 200, rerank: 50)
```

| Argument | Meaning |
| --- | --- |
| `.field` | the `Vector` field to search |
| query | a `"text"` string (requires `@vectorize`) or a raw `[f32, ...]` vector |
| `k:` | number of results to return |
| `ef:` | (optional) HNSW search width — higher is more accurate, slower |
| `rerank:` | (optional) re-score this many candidates with full-precision vectors; `0`/omitted = off |

`.similar` can also follow a filter, narrowing the candidate set first:

```
Post.filter(.published == true).similar(.embedding, "rust", k: 5)
```

### Mutations — `.update`, `.delete`

```
User.get(1).update({ name: "Alice B.", age: 31 })
Post.get(7).delete()
```

`delete` honours the `@on_delete` policies declared in your schema (`remove` / `cascade` / `deny`).

### Links — `.link`, `.unlink`

Create or remove a relationship between objects. Edge-field values (if the relationship declares any) go in the optional object literal.

```
User.get(1).friends.link(User.get(2))
User.get(1).favorite_movies.link(Movie.get(42), { rating: 4.5 })
User.get(1).friends.unlink(User.get(2))
```

### Pagination — `.offset`, `.limit`

```
User.filter(.age > 18).limit(10)
User.filter(.age > 18).offset(20).limit(10)
```

Apply `offset` **before** `limit`. Writing `.limit(N).offset(M)` is rejected — that order silently drops rows and is treated as a mistake.

## Literals

| Kind | Examples |
| --- | --- |
| String | `"hello"` |
| Integer | `42`, `-10` |
| Float | `3.14`, `-2.5` |
| Bool | `true`, `false` |
| Null | `null` |
| Vector | `[1.0, 2.0, 3.0]` |
| JSON | `{ "k": 1, "tags": ["a", "b"] }` |

How a literal is interpreted depends on the field's **declared type** — the
language has no inline type annotations. A string literal becomes a `DateTime`
when written to a `DateTime` field, base64-decodes into a `Bytes` field, and a
raw JSON object/array literal is stored in a `Json` field:

```text
Event.create({
  created: "2026-06-20T12:00:00Z",   # DateTime  (RFC 3339; or an integer epoch-millis)
  blob:    "aGVsbG8=",               # Bytes     (base64)
  meta:    { "k": 1, "tags": ["a"] } # Json      (raw JSON value)
})
```

These read back faithfully — `created` as an RFC 3339 string, `blob` as base64,
`meta` as inline JSON. A malformed value (a non-RFC-3339 `DateTime`, invalid
base64, or unparseable JSON) is a clear error, not a silent default.

## Examples

```
// Create
User.create({ name: "Alice", email: "alice@example.com", age: 30 })

// Read by id and by filter
User.get(1)
User.filter(.reputation >= 100 && .active == true).offset(0).limit(20)

// Traverse
User.get(1).friends.posts.filter(.title != "")

// Mutate
User.get(1).update({ reputation: 150 })
Post.get(7).delete()

// Relate
User.get(1).favorite_movies.link(Movie.get(42), { rating: 5.0 })

// Vector search, narrowed by a filter
Post.filter(.published == true).similar(.embedding, "vector databases", k: 10, rerank: 50)
```
