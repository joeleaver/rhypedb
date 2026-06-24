//! The asynchronous rhypedb client over a single persistent TCP connection.
//!
//! The binary protocol is strictly request/response on a (non-subscription)
//! connection and the server replies in request order, so requests are matched
//! to replies FIFO. Writes are ordered by the socket, so concurrent callers
//! pipeline safely. Any socket/framing error latches the connection closed
//! (fail-fast) rather than risk a mis-correlated reply — reconnect.

import { connect as tcpConnect, Socket } from "node:net";

import { RhypedbError } from "./errors.ts";
import { type Query, type Row } from "./query.ts";
import {
  FrameParser,
  MAX_FRAME_PAYLOAD,
  Req,
  Resp,
  encodeFrame,
  encodeQueryPayload,
  encodePreparePayload,
  encodeExecutePayload,
  encodeVectorBatchPayload,
  decodeObjectsPayload,
  decodeSinglePayload,
  decodePreparedPayload,
  decodeErrorPayload,
  type DecodedObject,
  type Frame,
} from "./wire.ts";

/** Connection options. */
export interface ClientOptions {
  /** Bound on the initial TCP connect, in milliseconds. Omit for no bound. */
  connectTimeoutMs?: number;
}

/** The decoded result of a query, before typing. */
export type QueryResult =
  | { kind: "objects"; objects: DecodedObject[] }
  | { kind: "single"; object: DecodedObject }
  | { kind: "done" };

interface Pending {
  resolve: (frame: Frame) => void;
  reject: (err: RhypedbError) => void;
}

/** Process-wide source of per-client brands for [`Prepared`] statements. */
let nextClientId = 1;

/**
 * A statement prepared on an [`AsyncClient`]'s connection (see
 * [`prepare`](AsyncClient#prepare)). The server parses + caches the query once
 * and hands back this handle; re-running it via the `*Prepared` methods re-runs
 * it with no re-parse and without re-sending the query text. There are no bind
 * parameters — a prepared statement is a *fixed* query.
 *
 * Statement ids are scoped to the connection that created them. A `Prepared` is
 * branded with its issuing client, so using it on a *different* client is
 * rejected with a `foreign_statement` error (before any I/O) rather than running
 * that client's unrelated statement of the same id.
 */
export class Prepared<T> {
  readonly stmtId: bigint;
  readonly clientId: number;
  /** @internal — created by [`AsyncClient.prepare`]; do not construct directly. */
  constructor(stmtId: bigint, clientId: number) {
    this.stmtId = stmtId;
    this.clientId = clientId;
  }
  declare private readonly _row: (x: T) => void;
}

/** Parse a `"host:port"` string or `{ host, port }` into connect options. */
function parseAddr(addr: string | { host: string; port: number }): { host: string; port: number } {
  if (typeof addr !== "string") return addr;
  // Bracketed IPv6 literal: `[::1]:4201`. Split on the `]:` so the colons inside
  // the address aren't mistaken for the port separator.
  let host: string;
  let portStr: string;
  if (addr.startsWith("[")) {
    const end = addr.indexOf("]:");
    if (end < 0) throw new RhypedbError("connect", `invalid IPv6 address ${JSON.stringify(addr)} (want [host]:port)`);
    host = addr.slice(1, end);
    portStr = addr.slice(end + 2);
  } else {
    const i = addr.lastIndexOf(":");
    if (i < 0) throw new RhypedbError("connect", `invalid address ${JSON.stringify(addr)} (want host:port)`);
    host = addr.slice(0, i) || "127.0.0.1";
    portStr = addr.slice(i + 1);
  }
  const port = Number(portStr);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new RhypedbError("connect", `invalid port in ${JSON.stringify(addr)}`);
  }
  return { host, port };
}

/**
 * A connected rhypedb client. Construct with [`connect`](AsyncClient.connect).
 * Safe to share and call concurrently — calls pipeline on the one connection.
 */
export class AsyncClient {
  readonly #socket: Socket;
  readonly #parser = new FrameParser();
  readonly #clientId = nextClientId++;
  #pending: Pending[] = [];
  #dead = false;
  #nextReqId = 1;

  private constructor(socket: Socket) {
    this.#socket = socket;
    socket.on("data", (chunk: Buffer) => this.#onData(chunk));
    socket.on("error", (e: Error) => this.#fail(new RhypedbError("io", e.message)));
    socket.on("close", () => {
      // A close while requests are in flight (or unexpectedly) is an I/O failure;
      // an intentional `close()` has already drained `#pending`.
      this.#fail(new RhypedbError("io", "connection closed by peer"));
    });
  }

  /** Connect to a rhypedb server's binary TCP port (`"host:port"`). */
  static connect(
    addr: string | { host: string; port: number },
    opts: ClientOptions = {},
  ): Promise<AsyncClient> {
    const { host, port } = parseAddr(addr);
    return new Promise<AsyncClient>((resolve, reject) => {
      const socket = tcpConnect({ host, port });
      socket.setNoDelay(true);
      let settled = false;
      const timer =
        opts.connectTimeoutMs !== undefined
          ? setTimeout(() => {
              if (settled) return;
              settled = true;
              socket.destroy();
              reject(new RhypedbError("connect", "connect timed out"));
            }, opts.connectTimeoutMs)
          : null;
      const onError = (e: Error) => {
        if (settled) return;
        settled = true;
        if (timer) clearTimeout(timer);
        reject(new RhypedbError("connect", e.message));
      };
      socket.once("error", onError);
      socket.once("connect", () => {
        if (settled) return;
        settled = true;
        if (timer) clearTimeout(timer);
        socket.removeListener("error", onError);
        resolve(new AsyncClient(socket));
      });
    });
  }

  // --- request/response surface ------------------------------------------

  /** Liveness check: `PING` → `PONG`. */
  async ping(): Promise<void> {
    const frame = await this.#roundTrip(Req.Ping, Buffer.alloc(0));
    if (frame.kind === Resp.Pong) return;
    if (frame.kind === Resp.Error) throw serverError(frame.payload);
    throw unexpected(frame.kind);
  }

  /** Run a raw query string, returning the untyped [`QueryResult`]. */
  async query(query: string): Promise<QueryResult> {
    const frame = await this.#roundTrip(Req.Query, encodeQueryPayload(query));
    return decodeQueryFrame(frame);
  }

  /** Run a typed [`Query`] and materialize every row into `T`. */
  async fetch<T>(query: Query<T>): Promise<Row<T>[]> {
    return flattenRows<T>(await this.query(query.text));
  }

  /** Run a typed [`Query`] and return the first row, if any. */
  async fetchOne<T>(query: Query<T>): Promise<Row<T> | null> {
    const rows = await this.fetch(query);
    return rows.length > 0 ? rows[0]! : null;
  }

  /** Run a typed write that returns the affected object (`create`/`update`). */
  async create<T>(query: Query<T>): Promise<Row<T>> {
    return singleRow<T>(await this.query(query.text));
  }

  /** Run a query for its effect, returning the untyped result shape. */
  execute<T>(query: Query<T>): Promise<QueryResult> {
    return this.query(query.text);
  }

  /**
   * Prepare a typed [`Query`] for repeated execution on this connection. The
   * server parses + caches it and returns a [`Prepared`] handle; run it with the
   * `*Prepared` methods (no re-parse, no re-sent text). Valid until this client
   * is closed.
   */
  async prepare<T>(query: Query<T>): Promise<Prepared<T>> {
    const frame = await this.#roundTrip(Req.Prepare, encodePreparePayload(query.text));
    if (frame.kind === Resp.Prepared) {
      return new Prepared<T>(decode(() => decodePreparedPayload(frame.payload)), this.#clientId);
    }
    if (frame.kind === Resp.Error) throw serverError(frame.payload);
    throw unexpected(frame.kind);
  }

  /** Run a [`Prepared`] statement, returning the untyped [`QueryResult`]. */
  async executePrepared<T>(stmt: Prepared<T>): Promise<QueryResult> {
    return decodeQueryFrame(await this.#runPrepared(stmt));
  }

  /** Run a [`Prepared`] statement and materialize every row into `T`. */
  async fetchPrepared<T>(stmt: Prepared<T>): Promise<Row<T>[]> {
    return flattenRows<T>(await this.executePrepared(stmt));
  }

  /** Run a [`Prepared`] statement and return the first row, if any. */
  async fetchOnePrepared<T>(stmt: Prepared<T>): Promise<Row<T> | null> {
    const rows = await this.fetchPrepared(stmt);
    return rows.length > 0 ? rows[0]! : null;
  }

  /** Run a [`Prepared`] write that returns the affected object. */
  async createPrepared<T>(stmt: Prepared<T>): Promise<Row<T>> {
    return singleRow<T>(await this.executePrepared(stmt));
  }

  /**
   * Bulk-load precomputed vectors into one `Vector` field. `rows` is `[id,
   * vector]` pairs; the server validates and applies the whole batch atomically
   * (returns the row count). One frame per call — throws a `batch_too_large`
   * error (before any write) if the encoded batch would exceed the protocol
   * frame cap; split it across calls (each call is its own atomic batch).
   */
  async ingestVectors(
    typeName: string,
    fieldName: string,
    rows: ReadonlyArray<readonly [bigint, ArrayLike<number>]>,
  ): Promise<number> {
    const payload = encodeVectorBatchPayload(typeName, fieldName, rows);
    if (payload.length > MAX_FRAME_PAYLOAD) {
      throw new RhypedbError(
        "batch_too_large",
        `vector batch too large: ${rows.length} rows encode to ${payload.length} bytes (frame cap ${MAX_FRAME_PAYLOAD})`,
      );
    }
    const frame = await this.#roundTrip(Req.VectorBatch, payload);
    if (frame.kind === Resp.Done) return rows.length;
    if (frame.kind === Resp.Error) throw serverError(frame.payload);
    throw unexpected(frame.kind);
  }

  /** Verify a [`Prepared`] belongs to this client (before any I/O), then run it. */
  async #runPrepared<T>(stmt: Prepared<T>): Promise<Frame> {
    if (stmt.clientId !== this.#clientId) {
      throw new RhypedbError(
        "foreign_statement",
        "prepared statement belongs to a different client; prepare it on this client first",
      );
    }
    return this.#roundTrip(Req.Execute, encodeExecutePayload(stmt.stmtId));
  }

  /** Close the connection. In-flight requests reject with a closed error. */
  close(): void {
    this.#fail(new RhypedbError("closed", "connection closed by client"));
  }

  // --- internals ----------------------------------------------------------

  #onData(chunk: Buffer): void {
    this.#parser.push(chunk);
    try {
      for (let frame = this.#parser.next(); frame !== null; frame = this.#parser.next()) {
        const pending = this.#pending.shift();
        // A non-subscription connection never receives an unsolicited frame; if
        // one arrives with nothing pending, the stream is desynced — fail fast.
        if (pending === undefined) {
          this.#fail(new RhypedbError("io", "unsolicited frame with no pending request"));
          return;
        }
        pending.resolve(frame);
      }
    } catch (e) {
      // A framing error (bad length) desyncs the stream irrecoverably → latch.
      this.#fail(new RhypedbError("io", `framing error: ${(e as Error).message}`));
    }
  }

  #fail(err: RhypedbError): void {
    if (this.#dead) return;
    this.#dead = true;
    const pending = this.#pending;
    this.#pending = [];
    for (const p of pending) p.reject(err);
    this.#socket.destroy();
  }

  #roundTrip(kind: number, payload: Buffer): Promise<Frame> {
    if (this.#dead) {
      return Promise.reject(
        new RhypedbError("closed", "connection is closed; reconnect"),
      );
    }
    const reqId = this.#nextReqId;
    this.#nextReqId = (this.#nextReqId + 1) >>> 0; // wrap as u32
    return new Promise<Frame>((resolve, reject) => {
      this.#pending.push({ resolve, reject });
      this.#socket.write(encodeFrame(reqId, kind, payload), (err) => {
        if (err) this.#fail(new RhypedbError("io", err.message));
      });
    });
  }
}

/** Map a query/execute reply frame to a [`QueryResult`] or throw. */
function decodeQueryFrame(frame: Frame): QueryResult {
  switch (frame.kind) {
    case Resp.Objects:
      return { kind: "objects", objects: decode(() => decodeObjectsPayload(frame.payload)) };
    case Resp.Single:
      return { kind: "single", object: decode(() => decodeSinglePayload(frame.payload)) };
    case Resp.Done:
      return { kind: "done" };
    case Resp.Error:
      throw serverError(frame.payload);
    default:
      throw unexpected(frame.kind);
  }
}

/** Wrap a decode so a malformed (but framed) payload surfaces as `decode`. */
function decode<T>(f: () => T): T {
  try {
    return f();
  } catch (e) {
    throw new RhypedbError("decode", (e as Error).message);
  }
}

function objectToRow<T>(obj: DecodedObject): Row<T> {
  return { id: obj.id, data: obj.fields as unknown as T };
}

/** `objects` → rows; `single` → one row; `done` → empty (the `fetch` shape). */
function flattenRows<T>(result: QueryResult): Row<T>[] {
  switch (result.kind) {
    case "objects":
      return result.objects.map((o) => objectToRow<T>(o));
    case "single":
      return [objectToRow<T>(result.object)];
    case "done":
      return [];
  }
}

/** A result that must be exactly one object (the `create`/`update` shape). */
function singleRow<T>(result: QueryResult): Row<T> {
  if (result.kind === "single") return objectToRow<T>(result.object);
  if (result.kind === "objects" && result.objects.length === 1) {
    return objectToRow<T>(result.objects[0]!);
  }
  throw new RhypedbError("unexpected_shape", `expected a single object, got ${result.kind}`);
}

function serverError(payload: Buffer): RhypedbError {
  try {
    return new RhypedbError("server", decodeErrorPayload(payload));
  } catch {
    return new RhypedbError("server", payload.toString("utf8"));
  }
}

function unexpected(kind: number): RhypedbError {
  return new RhypedbError("unexpected_response", `unexpected response frame kind 0x${kind.toString(16)}`);
}
