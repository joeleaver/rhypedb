# Getting Started

This walkthrough takes you from a clean checkout to a running server you can query — in about five minutes.

## 1. Build

rhypedb is a Cargo workspace. Build the server and CLI in release mode:

```bash
cargo build --release -p rhypedb-server -p rhypedb-cli
```

The binaries land in `target/release/`:

- `target/release/rhypedb-server`
- `target/release/rhypedb-cli`

> The first build compiles the embedding runtime (ONNX), so it takes a while. Subsequent builds are fast.

## 2. Write a schema

Create a file `blog.rhype`:

```
type User {
    name: String
    email: String @unique
    age: u32
    friends: [User] @on_delete(remove)
    posts: [Post] @inverse(Post.author)
}

type Post {
    title: String
    body: String
    author: User @on_delete(cascade)
    tags: [Tag] @on_delete(remove)
}

type Tag {
    name: String @unique
}
```

This declares three object types and the relationships between them. `@unique` enforces a uniqueness constraint, `@inverse` makes `User.posts` and `Post.author` two ends of the same relationship, and `@on_delete` declares what happens to a relationship when an object is deleted. See the **[Schema Reference](schema.md)** for everything you can write here.

## 3. Run the server

```bash
target/release/rhypedb-server --schema blog.rhype --data-dir ./data
```

You'll see:

```
rhypedb HTTP listening on 127.0.0.1:4200
rhypedb binary TCP listening on 127.0.0.1:4201
```

The `./data` directory now holds the database (LSM-tree files, the write-ahead log, a copy of the schema, and a single-writer `LOCK`). Leave the server running and open a second terminal.

## 4. Run queries

### With the CLI

Execute a single query with `-e`:

```bash
target/release/rhypedb-cli -e 'User.create({ name: "Alice", email: "alice@example.com", age: 30 })'
```

```json
{ "object": { "id": 1, "name": "Alice", "email": "alice@example.com", "age": 30 } }
```

Create another user and a post, then link them:

```bash
rhypedb-cli -e 'User.create({ name: "Bob", email: "bob@example.com", age: 25 })'
rhypedb-cli -e 'Post.create({ title: "Hello", body: "first post", author: User.get(1) })'
```

Read it back, including a relationship traversal:

```bash
rhypedb-cli -e 'User.get(1).posts'
rhypedb-cli -e 'User.filter(.age > 26)'
```

Run the CLI with no arguments to drop into an interactive REPL:

```
$ rhypedb-cli
rhypedb> User.get(1)
{ "object": { "id": 1, "name": "Alice", ... } }
rhypedb> quit
```

### With HTTP

The same queries work over HTTP — `POST /query` with a JSON body:

```bash
curl -s http://127.0.0.1:4200/query \
  -H 'content-type: application/json' \
  -d '{"query": "User.get(1)"}'
```

```json
{ "object": { "id": 1, "name": "Alice", "email": "alice@example.com", "age": 30 } }
```

This is all you need to build a client in any language: send a query string, get JSON back. See the **[API Reference](api-reference.md)** for the full response shapes and the faster binary TCP protocol.

## 5. Check server health

```bash
curl -s http://127.0.0.1:4200/status
```

```json
{ "subscriptions": 0, "vectorizer": { "pending": 0, "indexes": [] } }
```

## Next steps

- **[Query Language](queries.md)** — filters, traversal, mutations, pagination.
- **[Vector Search](vectors.md)** — add a `Vector<N>` field and search it (with or without built-in text embedding).
- **[Schema Reference](schema.md)** — the full schema DSL.
- **[Running rhypedb](operations.md)** — configuration, admin auth, and production concerns.
