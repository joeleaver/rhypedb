//! The typed query builder ([`Query`]) and result row ([`Row`]).
//!
//! Schema-agnostic, so they live in the client (not in generated code). The
//! codegen emits thin per-type seed constructors (`User.all()`, `User.get(id)`,
//! …) that return a `Query<User>`; materializing a `Row<User>` just casts the
//! decoded scalar fields to the generated `User` interface.

/**
 * A typed query string targeting one object type `T`. A thin string builder;
 * the server validates query composition. Build it from a seed
 * ([`all`](Query.all) / [`get`](Query.get) / [`filterOn`](Query.filterOn)) or
 * [`raw`](Query.raw), optionally chain [`filter`](Query#filter) /
 * [`limit`](Query#limit), then hand it to a client method.
 *
 * `T` is a compile-time phantom (it shapes the returned [`Row`]); it has no
 * runtime footprint.
 */
export class Query<T> {
  readonly text: string;
  /** Phantom marker for `T`; never read at runtime. */
  declare private readonly _row: (x: T) => void;

  private constructor(text: string) {
    this.text = text;
  }

  /** Wrap a raw query string (escape hatch + the seed primitive). */
  static raw<T>(text: string): Query<T> {
    return new Query<T>(text);
  }

  /** `Type` — all objects of the type (a bare type name is "all" in the QL). */
  static all<T>(typeName: string): Query<T> {
    return new Query<T>(typeName);
  }

  /** `Type.get(id)` — a single object by id. */
  static get<T>(typeName: string, id: bigint | number): Query<T> {
    return new Query<T>(`${typeName}.get(${id})`);
  }

  /** `Type.filter(<predicate>)` — objects matching a predicate. */
  static filterOn<T>(typeName: string, predicate: string): Query<T> {
    return new Query<T>(`${typeName}.filter(${predicate})`);
  }

  /** Append `.filter(<predicate>)`, e.g. `.filter(".age > 18")`. */
  filter(predicate: string): Query<T> {
    return new Query<T>(`${this.text}.filter(${predicate})`);
  }

  /** Append `.limit(<n>)`. */
  limit(n: bigint | number): Query<T> {
    return new Query<T>(`${this.text}.limit(${n})`);
  }

  /**
   * Append `.matches(.<field>, "<text>", k: <k>)` — ranked full-text search
   * over a `@fulltext` field. `text` is escaped as a query-language string
   * literal; the rows come back with [`Row.score`](Row) set (BM25).
   */
  matches(field: string, text: string, k: bigint | number): Query<T> {
    return new Query<T>(`${this.text}.matches(.${field}, ${qlStringLiteral(text)}, k: ${k})`);
  }

  toString(): string {
    return this.text;
  }
}

/**
 * A query-result row: the object `id` (a lossless `bigint`) plus its typed
 * scalar fields. The fields are rendered in the canonical form — `Bytes` as a
 * base64 string, `DateTime` as an RFC 3339 string, `Json` inline, `u64`/`i64`
 * as `bigint` — so a generated interface `T` describes them directly.
 */
export interface Row<T> {
  id: bigint;
  data: T;
  /**
   * Set only for rows of a ranked result, which arrive in rank order:
   * `.matches` → the BM25 score (higher is better); `.similar` → the index
   * distance under the field's metric (lower is closer) — always the
   * distance, even when the server reranked the row.
   */
  score?: number;
  /**
   * The cross-encoder's relevance score (higher is better). Set only on a
   * `.similar` row the server's cross-encoder actually scored
   * (`[vectorizer] cross_encoder` on); such rows are ordered by it.
   */
  rerankScore?: number;
}

/** Escape `s` as a query-language string literal (`\"` and `\\`). */
function qlStringLiteral(s: string): string {
  return `"${s.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}
