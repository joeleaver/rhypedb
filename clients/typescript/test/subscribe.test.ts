//! Subscription tests against a wire-faithful mock. `AsyncClient.subscribe`
//! opens a SECOND, dedicated connection (separate from the query connection), so
//! each mock here accepts two connections: the idle query connection first, then
//! the subscription connection it drives.
//!
//! The real server guarantees RESP_SUBSCRIBED precedes any event, so the mock
//! acks first and only then pushes events (via `afterAck`).

import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer, type Server, type Socket } from "node:net";

import { AsyncClient } from "../src/client.ts";
import { SubscriptionFilter, type Notification } from "../src/subscription.ts";
import { RhypedbError } from "../src/errors.ts";
import {
  EVENT_FORMAT_TAG,
  FrameParser,
  Req,
  Resp,
  encodeFrame,
  type Frame,
} from "../src/wire.ts";

function eventPayload(kind: string, typeName: string, id: bigint, version: bigint, withFields = true): Buffer {
  const we: Record<string, unknown> = {
    v: EVENT_FORMAT_TAG,
    kind,
    type: typeName,
    id: id.toString(),
    version: version.toString(),
  };
  if (withFields) we.fields = { name: "Ada" };
  return Buffer.from(JSON.stringify(we), "utf8");
}

/** Spawn a mock accepting the (idle) query conn, then the subscription conn. */
function spawnSubServer(script: (sub: Socket) => void): Promise<{ port: number; server: Server }> {
  let nthConn = 0;
  const server = createServer((sock) => {
    nthConn += 1;
    sock.on("error", () => {});
    if (nthConn === 1) {
      sock.on("data", () => {}); // the client's request/response connection — idle here
      return;
    }
    script(sock);
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const a = server.address();
      if (a === null || typeof a === "string") throw new Error("no port");
      resolve({ port: a.port, server });
    });
  });
}

/** Ack the SUBSCRIBE, run `afterAck` (push events), route later frames to `onFrame`. */
function ackSubscribe(
  sub: Socket,
  opts: { afterAck?: () => void; onFrame?: (f: Frame) => void } = {},
): void {
  const parser = new FrameParser();
  let acked = false;
  sub.on("data", (chunk: Buffer) => {
    parser.push(chunk);
    for (let f = parser.next(); f !== null; f = parser.next()) {
      if (!acked) {
        acked = true;
        assert.equal(f.kind, Req.Subscribe);
        sub.write(encodeFrame(f.reqId, Resp.Subscribed, Buffer.alloc(0)));
        opts.afterAck?.();
      } else {
        opts.onFrame?.(f);
      }
    }
  });
}

test("iterates create/lagged/delete then ends on close", async () => {
  const { port, server } = await spawnSubServer((sub) => {
    ackSubscribe(sub, {
      afterAck: () => {
        sub.write(encodeFrame(1, Resp.Event, eventPayload("create", "User", 42n, 7n)));
        sub.write(encodeFrame(1, Resp.SubLagged, Buffer.alloc(0)));
        sub.write(encodeFrame(1, Resp.Event, eventPayload("delete", "User", 42n, 9n)));
        setTimeout(() => sub.destroy(), 30); // then EOF
      },
    });
  });

  const client = await AsyncClient.connect({ host: "127.0.0.1", port });
  try {
    const sub = await client.subscribe(SubscriptionFilter.forType("User"));
    const got: Notification[] = [];
    try {
      for await (const n of sub) got.push(n);
    } catch (e) {
      assert.ok(e instanceof RhypedbError && e.code === "io"); // EOF surfaces once
    }
    assert.equal(got.length, 3);
    const c0 = got[0]!;
    assert.ok(c0.type === "change");
    assert.equal(c0.change.kind, "create");
    assert.equal(c0.change.id, 42n);
    assert.equal(c0.change.version, 7n);
    assert.deepEqual(c0.change.fields, { name: "Ada" });
    assert.equal(got[1]!.type, "lagged");
    assert.equal(got[2]!.type, "change");
  } finally {
    client.close();
    server.close();
  }
});

test("u64 id/version decode losslessly past 2^53", async () => {
  const bigId = 9_007_199_254_740_993n; // 2^53 + 1
  const { port, server } = await spawnSubServer((sub) => {
    ackSubscribe(sub, {
      afterAck: () => sub.write(encodeFrame(1, Resp.Event, eventPayload("update", "User", bigId, bigId + 1n))),
    });
  });
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });
  try {
    const sub = await client.subscribe(SubscriptionFilter.all());
    const n = (await sub.next()).value as Notification;
    assert.ok(n.type === "change");
    assert.equal(n.change.id, bigId);
    assert.equal(n.change.version, bigId + 1n);
    sub.close();
  } finally {
    client.close();
    server.close();
  }
});

test("origin decodes losslessly past 2^53; absent → undefined; malformed rejects", async () => {
  const bigOrigin = 9_007_199_254_740_993n; // 2^53 + 1
  const evt = (extra: Record<string, unknown>): Buffer =>
    Buffer.from(JSON.stringify({ v: EVENT_FORMAT_TAG, type: "User", ...extra }), "utf8");
  const { port, server } = await spawnSubServer((sub) => {
    ackSubscribe(sub, {
      afterAck: () => {
        // (a) tagged event with an origin.
        sub.write(encodeFrame(1, Resp.Event, evt({ kind: "update", id: "5", version: "9", origin: bigOrigin.toString() })));
        // (b) untagged event (no origin key) → undefined.
        sub.write(encodeFrame(1, Resp.Event, eventPayload("create", "User", 6n, 10n)));
        // (c) malformed origin → decode error, subscription stays live. ("abc"
        // is rejected by BigInt(); note JS BigInt() DOES accept hex like "0x2a",
        // unlike Rust's decimal-only parse, so use a clearly non-numeric string.)
        sub.write(encodeFrame(1, Resp.Event, evt({ kind: "create", id: "7", version: "11", origin: "abc" })));
        // (d) a good event still arrives after the bad one.
        sub.write(encodeFrame(1, Resp.Event, eventPayload("create", "User", 8n, 12n)));
      },
    });
  });
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });
  try {
    const sub = await client.subscribe(SubscriptionFilter.all());
    const n1 = (await sub.next()).value as Notification;
    assert.ok(n1.type === "change" && n1.change.origin === bigOrigin);
    const n2 = (await sub.next()).value as Notification;
    assert.ok(n2.type === "change" && n2.change.origin === undefined);
    await assert.rejects(sub.next(), (e: unknown) => e instanceof RhypedbError && e.code === "decode"); // bad origin
    const n4 = (await sub.next()).value as Notification;
    assert.ok(n4.type === "change" && n4.change.id === 8n); // subscription stayed live
    sub.close();
  } finally {
    client.close();
    server.close();
  }
});

test("excludingOrigin drops own-origin events client-side", async () => {
  const evt = (id: string, origin?: string): Buffer =>
    Buffer.from(
      JSON.stringify({ v: EVENT_FORMAT_TAG, kind: "create", type: "User", id, version: "1", ...(origin !== undefined ? { origin } : {}) }),
      "utf8",
    );
  const { port, server } = await spawnSubServer((sub) => {
    ackSubscribe(sub, {
      afterAck: () => {
        sub.write(encodeFrame(1, Resp.Event, evt("1", "42"))); // ours → dropped client-side
        sub.write(encodeFrame(1, Resp.Event, evt("2", "7"))); // other origin → delivered
        sub.write(encodeFrame(1, Resp.Event, evt("3"))); // untagged → delivered
      },
    });
  });
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });
  try {
    const sub = await client.subscribe(SubscriptionFilter.forType("User").excludingOrigin(42n));
    const n1 = (await sub.next()).value as Notification;
    assert.ok(n1.type === "change" && n1.change.id === 2n && n1.change.origin === 7n);
    const n2 = (await sub.next()).value as Notification;
    assert.ok(n2.type === "change" && n2.change.id === 3n && n2.change.origin === undefined);
    sub.close();
  } finally {
    client.close();
    server.close();
  }
});

test("a malformed event rejects that next() but keeps the subscription live", async () => {
  const { port, server } = await spawnSubServer((sub) => {
    ackSubscribe(sub, {
      afterAck: () => {
        sub.write(encodeFrame(1, Resp.Event, Buffer.from("not json")));
        sub.write(
          encodeFrame(
            1,
            Resp.Event,
            Buffer.from(JSON.stringify({ v: "wrong-tag", kind: "create", type: "User", id: "1", version: "1" })),
          ),
        );
        sub.write(encodeFrame(1, Resp.Event, eventPayload("create", "User", 9n, 3n)));
      },
    });
  });
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });
  try {
    const sub = await client.subscribe(SubscriptionFilter.all());
    await assert.rejects(sub.next(), (e: unknown) => e instanceof RhypedbError && e.code === "decode"); // bad json
    await assert.rejects(sub.next(), (e: unknown) => e instanceof RhypedbError && e.code === "decode"); // bad tag
    const n = (await sub.next()).value as Notification; // the good event still arrives
    assert.ok(n.type === "change" && n.change.id === 9n);
    sub.close();
  } finally {
    client.close();
    server.close();
  }
});

test("an unexpected terminal frame ends the stream and tears down the socket (no leak)", async () => {
  // A post-ack frame that isn't EVENT/SUBLAGGED is terminal. The subscription
  // must surface it once, then yield done, AND destroy its socket without an
  // explicit close() — otherwise the process would not exit (this suite asserts
  // a clean exit, so a leak here would hang it).
  const { port, server } = await spawnSubServer((sub) => {
    ackSubscribe(sub, { afterAck: () => sub.write(encodeFrame(1, Resp.Done, Buffer.alloc(0))) });
  });
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });
  try {
    const sub = await client.subscribe(SubscriptionFilter.all());
    await assert.rejects(sub.next(), (e: unknown) => e instanceof RhypedbError && e.code === "unexpected_response");
    assert.deepEqual(await sub.next(), { value: undefined, done: true });
    // deliberately NO sub.close() — the terminal frame must have torn it down.
  } finally {
    client.close();
    server.close();
  }
});

test("a server SUBSCRIBE rejection surfaces from open", async () => {
  const { port, server } = await spawnSubServer((sub) => {
    const parser = new FrameParser();
    sub.on("data", (chunk: Buffer) => {
      parser.push(chunk);
      for (let f = parser.next(); f !== null; f = parser.next()) {
        const msg = Buffer.from("subscriptions not supported");
        const lh = Buffer.allocUnsafe(4);
        lh.writeUInt32BE(msg.length, 0);
        sub.write(encodeFrame(f.reqId, Resp.Error, Buffer.concat([lh, msg])));
      }
    });
  });
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });
  try {
    await assert.rejects(
      client.subscribe(SubscriptionFilter.all()),
      (e: unknown) => e instanceof RhypedbError && e.code === "server",
    );
  } finally {
    client.close();
    server.close();
  }
});

test("unsubscribe drains an in-flight event then resolves on DONE", async () => {
  const { port, server } = await spawnSubServer((sub) => {
    ackSubscribe(sub, {
      onFrame: (f) => {
        if (f.kind === Req.Unsubscribe) {
          assert.equal(f.payload.readUInt32BE(0), 1); // the handle
          sub.write(encodeFrame(1, Resp.Event, eventPayload("create", "User", 5n, 1n))); // queued ahead of DONE
          sub.write(encodeFrame(f.reqId, Resp.Done, Buffer.alloc(0)));
        }
      },
    });
  });
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });
  try {
    const sub = await client.subscribe(SubscriptionFilter.all());
    await sub.unsubscribe(); // resolves when DONE arrives
  } finally {
    client.close();
    server.close();
  }
});

test("the query connection is unaffected by subscription pushes", async () => {
  // Unlike the other tests, conn1 (the query connection) must answer pings here.
  let nthConn = 0;
  const server = createServer((sock) => {
    nthConn += 1;
    sock.on("error", () => {});
    if (nthConn === 1) {
      const parser = new FrameParser();
      sock.on("data", (chunk: Buffer) => {
        parser.push(chunk);
        for (let f = parser.next(); f !== null; f = parser.next()) {
          if (f.kind === Req.Ping) sock.write(encodeFrame(f.reqId, Resp.Pong, Buffer.alloc(0)));
        }
      });
    } else {
      ackSubscribe(sock, {
        afterAck: () => sock.write(encodeFrame(1, Resp.Event, eventPayload("create", "User", 1n, 1n))),
      });
    }
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");

  const client = await AsyncClient.connect({ host: "127.0.0.1", port: addr.port });
  try {
    const sub = await client.subscribe(SubscriptionFilter.all());
    await client.ping(); // the query connection answers correctly despite the sub stream
    const n = (await sub.next()).value as Notification;
    assert.ok(n.type === "change");
    sub.close();
  } finally {
    client.close();
    server.close();
  }
});
