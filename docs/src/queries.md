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

**Substring — `.field.contains("text")`:** true when a `String` field contains the literal text. It is **case-sensitive**, never uses an index (the type is scanned like any other unindexed filter), and composes with the other operators. A `null` value never matches. Applying it to a non-`String` field is an error.

```
Post.filter(.title.contains("invoice"))
Post.filter(.published == true && .body.contains("4471"))
```

For ranked keyword search that *does* use an index, see [Full-text search](#full-text-search--matchesfield-query-k-n).

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

`.similar` returns a **ranked** result: each object carries a `score` and the rows come back in rank order. Ordinarily `score` is the index's distance under the field's metric (lower is closer); when the [cross-encoder](vectors.md#cross-encoder-reranking) actually scored a row it is instead the cross-encoder's relevance score (higher is better). See [Ranked results](#ranked-results) for exactly when that is.

### Full-text search — `.matches(.field, "query", k: N)`

Ranked keyword search over a `String` field declared `@fulltext` (see [Schema → `@fulltext`](schema.md#fulltext--fulltextanalyzer-simple-positions-false)). Sits where `.similar` sits: bare on a type, or after a filter or traversal that narrows the candidates.

```
Post.matches(.body, "invoice 4471", k: 20)
Post.filter(.published == true).matches(.body, "distributed consensus", k: 10)
User.get(1).posts.matches(.title, "+draft budget", k: 5)
```

| Argument | Meaning |
| --- | --- |
| `.field` | the `@fulltext` String field to search |
| `"query"` | the search text (syntax below) |
| `k:` | number of results to return (most relevant first) |

**Query syntax.** The text is analyzed with the field's analyzer (lowercased, diacritics folded, split into words; stemmed too under `analyzer: "english"`), then:

| Form | Meaning |
| --- | --- |
| `invoice 4471` | terms are **OR-ed**: a document matches if it contains any of them, and every term it contains adds to its score |
| `+invoice` | a `+` prefix makes the term **required** |
| `"distributed consensus"` | quoted words are a **phrase**: they must appear consecutively (the field must store positions — the default) |
| `camera*` | a trailing `*` makes a **prefix term**: it matches every indexed term that starts with `camera` (`camera`, `cameras`, `camerawork`, …) and scores as one term — the type-ahead case |
| `+"exact phrase" cam* extra` | modifiers combine: the phrase is required, the prefix and `extra` are optional |

Results are ranked by **BM25** (k1 = 1.2, b = 0.75): rarer terms count more, repeated terms count more with diminishing returns, and shorter documents win ties. Each returned object carries its `score` (higher is better) — see [Ranked results](#ranked-results).

A prefix term is analyzed like any other term before it is expanded — lowercased and folded, and under the `english` analyzer **stemmed**: `cameras*` searches for the stem `camera`, which is what the index holds. Be aware of what that means for type-ahead over a stemmed field: the index stores stems, which are often *shorter* than the word being typed, so a partial word that runs past the stem finds nothing until the word is complete (`securit*` matches nothing because `security` is indexed as `secur`; `secur*` and `security*` both match), and a prefix the stemmer shortens matches a little more than typed (`agre*` searches `agr*`). Stemming the prefix is still the better single strategy (a dictionary sweep found it matches more partial prefixes than leaving the prefix unstemmed, and it is what makes `cameras*` find `camera`), but if you need character-by-character completion, put it on a `simple` field. Text that analyzes to several words before the `*` (`e-mail*`) gives ordinary terms for all but the last, which becomes the prefix.

The prefix needs at least two characters after analysis (`*` and `c*` are errors — the error names the analyzed word, so `東京*`, one ideograph per word under Unicode segmentation, reports the one-character prefix `京`), it cannot appear inside a phrase, and an expansion of more than **64 distinct indexed terms** is refused with an error asking for a longer prefix — so a one- or two-letter prefix over a large vocabulary fails fast instead of scanning half the index. The cap counts the field's whole vocabulary under the prefix, not only the terms of the candidates a preceding filter narrowed to. A `*` anywhere but the end of a word is ordinary punctuation and splits the word.

A query with no searchable terms (only punctuation), an unterminated quote, or a phrase against a field declared `@fulltext(positions: false)` is an error. So is `.matches` on a field without `@fulltext`; for an unindexed substring test use [`.contains`](#filter--filterpredicate) instead — `.contains` is literal and never stems, whatever the field's analyzer. While the index of a freshly-declared (or reconfigured) field is still being built for existing objects, `.matches` on it is an error that reports the progress — see [`@fulltext`](schema.md#fulltext--fulltextanalyzer-simple-positions-false).

`.matches` after a filter or traversal restricts the ranking to those candidates *before* taking the top `k`, so you always get up to `k` results from within the narrowed set.

### Ranked results

`.similar` and `.matches` produce a **ranked** result: the same objects as any other query, in rank order, each carrying a `score`. Over `POST /query` the score is an extra `"score"` member on every object; over the binary protocol the result is a `Scored` frame ([API Reference](api-reference.md#server-response-types)); the Rust and TypeScript clients expose it as `Row.score`. The score's meaning depends on the step:

| Step | `score` |
| --- | --- |
| `.matches` | BM25 relevance — higher is better |
| `.similar` (default) | the index distance under the field's metric (cosine distance, squared L2, or negated dot product) — lower is closer |
| `.similar`, row scored by the cross-encoder | the cross-encoder's relevance score — higher is better |

The cross-encoder scores a row only when all of these hold: `[vectorizer] cross_encoder` names a model, the query was a **text** query (a raw-vector `.similar` has no text to score against), the reranker model is currently loaded (a load failure is fail-soft — see `reranker_error` on `GET /status`), and the row's object has readable source text. A row that misses any of these carries a distance, and one result set can mix the two: with the cross-encoder on, rows the model scored come first (relevance, descending), then any candidates whose source text was unreadable (distance, ascending). So treat `score` as opaque for ordering purposes — the rows are already in rank order — and, if you read it directly, know which of the two your deployment produces. A `.similar` step's `score` is only comparable across queries against the SAME field with the SAME vectorizer configuration.

A `.filter`, `.limit` or `.offset` after a ranked step keeps the order and the scores; a traversal or a mutation drops them.

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
