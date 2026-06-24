//! Change-event subscriptions over a dedicated connection.
//!
//! A subscription runs on its **own** TCP connection, separate from the
//! [`AsyncClient`](./client.ts)'s request/response connection: the server turns
//! a connection that issues `SUBSCRIBE` into a subscription connection that
//! rejects queries, so pushed events never race a query reply. Open one with
//! [`AsyncClient#subscribe`]; it returns an [`AsyncSubscription`], an
//! `AsyncIterableIterator` of [`Notification`]s.
//!
//! Events are NOTIFICATIONS, not authoritative state: `fields` are best-effort
//! (the `/query` read form) and large 64-bit scalars may lose precision; a
//! [`lagged`](Notification) notification means the server's bounded buffer
//! overflowed (or it is shutting down) and ≥1 event was dropped — reconcile by
//! re-querying with the [`AsyncClient`].

import { type Socket } from "node:net";

import { dialSocket } from "./dial.ts";
import { RhypedbError } from "./errors.ts";
import {
  EVENT_FORMAT_TAG,
  FrameParser,
  Req,
  Resp,
  encodeFrame,
  encodeSubscribeFilter,
  encodeUnsubscribePayload,
  decodeEventPayload,
  decodeErrorPayload,
  type ChangeKind,
  type Frame,
  type WireEvent,
} from "./wire.ts";

export type { ChangeKind } from "./wire.ts";

/** A filter selecting which change events a subscription receives. */
export class SubscriptionFilter {
  typeName?: string;
  objectId?: bigint;
  kinds: ChangeKind[];

  private constructor(typeName: string | undefined, objectId: bigint | undefined, kinds: ChangeKind[]) {
    this.typeName = typeName;
    this.objectId = objectId;
    this.kinds = kinds;
  }

  /** All change events. */
  static all(): SubscriptionFilter {
    return new SubscriptionFilter(undefined, undefined, []);
  }

  /** Only events for one object type. */
  static forType(typeName: string): SubscriptionFilter {
    return new SubscriptionFilter(typeName, undefined, []);
  }

  /** Only events for one object (by type + id). */
  static forObject(typeName: string, objectId: bigint): SubscriptionFilter {
    return new SubscriptionFilter(typeName, objectId, []);
  }

  /** Narrow to specific change kinds (returns a new filter; empty = all kinds). */
  withKinds(...kinds: ChangeKind[]): SubscriptionFilter {
    return new SubscriptionFilter(this.typeName, this.objectId, [...kinds]);
  }
}

/** A single change event (best-effort scalar `fields`; re-query for authority). */
export interface ChangeNotification {
  kind: ChangeKind;
  typeName: string;
  /** The object id (lossless from the wire's decimal-string form). */
  id: bigint;
  /** The commit version (lossless from the wire's decimal-string form). */
  version: bigint;
  fields?: Record<string, unknown>;
}

/** One message delivered over a [`AsyncSubscription`]. */
export type Notification =
  | { type: "change"; change: ChangeNotification }
  | { type: "lagged" };

/** Decode + validate a wire event envelope into a typed change notification. */
function changeFromWire(w: WireEvent): ChangeNotification {
  if (w.v !== EVENT_FORMAT_TAG) {
    throw new RhypedbError(
      "decode",
      `unsupported event format tag ${JSON.stringify(w.v)} (expected ${EVENT_FORMAT_TAG}) — upgrade the client`,
    );
  }
  if (w.kind !== "create" && w.kind !== "update" && w.kind !== "delete") {
    throw new RhypedbError("decode", `unknown event kind ${JSON.stringify(w.kind)}`);
  }
  let id: bigint;
  let version: bigint;
  try {
    id = BigInt(w.id);
    version = BigInt(w.version);
  } catch {
    throw new RhypedbError("decode", `invalid event id/version (${JSON.stringify(w.id)}/${JSON.stringify(w.version)})`);
  }
  const out: ChangeNotification = { kind: w.kind, typeName: w.type, id, version };
  if (w.fields !== undefined) out.fields = w.fields;
  return out;
}

function serverErrorFrom(payload: Buffer): RhypedbError {
  try {
    return new RhypedbError("server", decodeErrorPayload(payload));
  } catch {
    return new RhypedbError("server", payload.toString("utf8"));
  }
}

type Item = { ok: Notification } | { err: RhypedbError };
interface Waiter {
  resolve: (r: IteratorResult<Notification>) => void;
  reject: (e: unknown) => void;
}

/**
 * A live change-event subscription on a dedicated connection — an
 * `AsyncIterableIterator` of [`Notification`]s. Iterate with `for await`, or call
 * [`next`](AsyncSubscription#next) directly.
 *
 * A socket/framing error or an unexpected frame ends the stream, surfaced once as
 * a rejected `next()` then `{ done: true }`. A malformed *event payload* rejects
 * that one `next()` with a `decode` error but leaves the subscription live (the
 * framing is intact) — call `next()` again to continue. Breaking a `for await`
 * loop calls [`return`](AsyncSubscription#return), which closes the connection
 * (the server treats that as an implicit unsubscribe); for an acknowledged
 * teardown use [`unsubscribe`](AsyncSubscription#unsubscribe).
 */
export class AsyncSubscription implements AsyncIterableIterator<Notification> {
  readonly #socket: Socket;
  readonly #parser = new FrameParser();
  readonly #handle = 1; // one subscription per connection → a fixed handle suffices
  #items: Item[] = [];
  #waiters: Waiter[] = [];
  #ended = false; // clean end: drain remaining items then yield done
  #terminated = false; // socket torn down; no more frames will arrive
  #fatal: RhypedbError | null = null; // a terminal error to surface exactly once
  #awaitingAck = true;
  #ackResolve!: () => void;
  #ackReject!: (e: unknown) => void;
  readonly #ackReady: Promise<void>;
  #unsub: { resolve: () => void; reject: (e: unknown) => void } | null = null;

  private constructor(socket: Socket, filter: SubscriptionFilter) {
    this.#socket = socket;
    this.#ackReady = new Promise<void>((resolve, reject) => {
      this.#ackResolve = resolve;
      this.#ackReject = reject;
    });
    socket.on("data", (chunk: Buffer) => this.#onData(chunk));
    socket.on("error", (e: Error) => this.#terminate(new RhypedbError("io", e.message)));
    socket.on("close", () => this.#terminate(new RhypedbError("io", "subscription connection closed")));
    // The SUBSCRIBE frame's req_id is the subscription handle.
    socket.write(encodeFrame(this.#handle, Req.Subscribe, encodeSubscribeFilter(filter)), (err) => {
      if (err) this.#terminate(new RhypedbError("io", err.message));
    });
  }

  /** Dial a fresh connection and run the SUBSCRIBE handshake (crate-internal). */
  static async open(
    host: string,
    port: number,
    connectTimeoutMs: number | undefined,
    filter: SubscriptionFilter,
  ): Promise<AsyncSubscription> {
    const socket = await dialSocket(host, port, connectTimeoutMs);
    const sub = new AsyncSubscription(socket, filter);
    await sub.#ackReady; // rejects on RESP_ERROR / unexpected / socket error
    return sub;
  }

  [Symbol.asyncIterator](): AsyncIterableIterator<Notification> {
    return this;
  }

  /** Await the next [`Notification`]. See the class docs for error semantics. */
  next(): Promise<IteratorResult<Notification>> {
    if (this.#items.length > 0) {
      const item = this.#items.shift()!;
      return "ok" in item
        ? Promise.resolve({ value: item.ok, done: false })
        : Promise.reject(item.err); // non-terminal (malformed event) — stays live
    }
    if (this.#fatal) {
      const e = this.#fatal;
      this.#fatal = null;
      this.#ended = true;
      return Promise.reject(e);
    }
    if (this.#ended) return Promise.resolve({ value: undefined, done: true });
    return new Promise<IteratorResult<Notification>>((resolve, reject) => {
      this.#waiters.push({ resolve, reject });
    });
  }

  /** `for await` early-exit hook — closes the subscription. */
  return(): Promise<IteratorResult<Notification>> {
    this.close();
    return Promise.resolve({ value: undefined, done: true });
  }

  /**
   * Cleanly close: send `UNSUBSCRIBE`, then resolve once the server's `DONE` ack
   * arrives (events queued ahead of it are still delivered to the iterator).
   * Rejects if the connection fails first.
   */
  unsubscribe(): Promise<void> {
    if (this.#ended || this.#terminated) return Promise.resolve();
    return new Promise<void>((resolve, reject) => {
      this.#unsub = { resolve, reject };
      this.#socket.write(
        encodeFrame(this.#handle, Req.Unsubscribe, encodeUnsubscribePayload(this.#handle)),
        (err) => {
          if (err) this.#terminate(new RhypedbError("io", err.message));
        },
      );
    });
  }

  /** Hard-close the connection (the server treats it as an implicit unsubscribe). */
  close(): void {
    if (this.#terminated) return;
    this.#terminated = true;
    this.#ended = true;
    this.#fatal = null; // intentional close — don't surface a stale error
    if (this.#unsub) {
      this.#unsub.reject(new RhypedbError("closed", "subscription closed"));
      this.#unsub = null;
    }
    if (this.#awaitingAck) {
      this.#awaitingAck = false;
      this.#ackReject(new RhypedbError("closed", "subscription closed before ack"));
    }
    try {
      this.#socket.destroy();
    } catch {
      /* already gone */
    }
    this.#pump();
  }

  // --- internals ----------------------------------------------------------

  #onData(chunk: Buffer): void {
    this.#parser.push(chunk);
    try {
      for (let frame = this.#parser.next(); frame !== null; frame = this.#parser.next()) {
        this.#classify(frame);
      }
    } catch (e) {
      this.#terminate(new RhypedbError("io", `framing error: ${(e as Error).message}`));
      return;
    }
    this.#pump();
  }

  #classify(frame: Frame): void {
    if (this.#terminated) return; // torn down — ignore any trailing frames in the batch
    if (this.#awaitingAck) {
      this.#awaitingAck = false;
      if (frame.kind === Resp.Subscribed) {
        this.#ackResolve();
      } else if (frame.kind === Resp.Error) {
        const e = serverErrorFrom(frame.payload);
        this.#ackReject(e);
        this.#endClean();
      } else {
        const e = new RhypedbError("unexpected_response", `expected SUBSCRIBED, got 0x${frame.kind.toString(16)}`);
        this.#ackReject(e);
        this.#endClean();
      }
      return;
    }
    switch (frame.kind) {
      case Resp.Event:
        try {
          this.#items.push({ ok: { type: "change", change: changeFromWire(decodeEventPayload(frame.payload)) } });
        } catch (e) {
          this.#items.push({ err: e instanceof RhypedbError ? e : new RhypedbError("decode", (e as Error).message) });
        }
        break;
      case Resp.SubLagged:
        this.#items.push({ ok: { type: "lagged" } });
        break;
      case Resp.Done:
        if (this.#unsub) {
          this.#unsub.resolve();
          this.#unsub = null;
          this.#endClean();
        } else {
          // A terminal frame tears the connection down (so the socket can't leak
          // and the process can exit); already-queued items still drain first.
          this.#terminate(new RhypedbError("unexpected_response", "unexpected DONE"));
        }
        break;
      case Resp.Error:
        this.#terminate(serverErrorFrom(frame.payload));
        break;
      default:
        this.#terminate(new RhypedbError("unexpected_response", `unexpected frame 0x${frame.kind.toString(16)}`));
    }
  }

  /** Satisfy waiting `next()` calls from the item queue / terminal state. */
  #pump(): void {
    while (this.#waiters.length > 0) {
      if (this.#items.length > 0) {
        const w = this.#waiters.shift()!;
        const item = this.#items.shift()!;
        if ("ok" in item) w.resolve({ value: item.ok, done: false });
        else w.reject(item.err);
      } else if (this.#fatal) {
        const w = this.#waiters.shift()!;
        const e = this.#fatal;
        this.#fatal = null;
        this.#ended = true;
        w.reject(e);
      } else if (this.#ended) {
        const w = this.#waiters.shift()!;
        w.resolve({ value: undefined, done: true });
      } else {
        break; // nothing to deliver yet — keep waiting
      }
    }
  }

  /** A socket/framing failure: tear down, surface the error once, fail any waiters. */
  #terminate(err: RhypedbError): void {
    if (this.#terminated || this.#ended) return;
    this.#terminated = true;
    if (this.#unsub) {
      this.#unsub.reject(err);
      this.#unsub = null;
    }
    if (this.#awaitingAck) {
      this.#awaitingAck = false;
      this.#ackReject(err);
    }
    if (this.#fatal === null) this.#fatal = err;
    try {
      this.#socket.destroy();
    } catch {
      /* already gone */
    }
    this.#pump();
  }

  /** A clean end (server DONE / ack rejection): drain remaining items then done. */
  #endClean(): void {
    this.#ended = true;
    this.#terminated = true;
    try {
      this.#socket.destroy();
    } catch {
      /* already gone */
    }
    this.#pump();
  }
}
