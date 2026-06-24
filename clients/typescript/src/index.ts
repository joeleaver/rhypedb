//! @rhypedb/client — the official TypeScript client for rhypedb.
//!
//! Speaks the binary TCP protocol over a single persistent connection, sharing
//! the wire format with the Rust client and the server. Node/Bun/Deno only
//! (it opens raw TCP sockets, which browsers can't).
//!
//! ```ts
//! import { AsyncClient, Query } from "@rhypedb/client";
//!
//! interface User { name: string | null; age: number | null }
//!
//! const client = await AsyncClient.connect("127.0.0.1:4201");
//! const adults = await client.fetch(Query.all<User>("User").filter(".age > 18"));
//! for (const row of adults) console.log(row.id, row.data.name);
//! ```

export { AsyncClient, Prepared } from "./client.ts";
export type { ClientOptions, QueryResult } from "./client.ts";
export { Query } from "./query.ts";
export type { Row } from "./query.ts";
export { RhypedbError } from "./errors.ts";
export type { ErrorCode } from "./errors.ts";
export type { DecodedObject, DecodedValue, Frame } from "./wire.ts";
