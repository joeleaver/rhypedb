//! Client-core tests against a wire-faithful mock server (a `net.Server` that
//! speaks the binary protocol via the same codec). Because the wire format is
//! single-sourced, exercising the client against the mock proves its full I/O
//! path — connect, frame, request/response correlation, decode-to-typed, error
//! mapping, dead-latch — without a running rhypedb server.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer, type Server, type Socket } from "node:net";

import { AsyncClient } from "../src/client.ts";
import { Query } from "../src/query.ts";
import { RhypedbError } from "../src/errors.ts";
import {
  FrameParser,
  Req,
  Resp,
  encodeFrame,
  encodeObject,
  encodeObjectsPayload,
  encodeErrorPayload,
  type EncValue,
  type Frame,
} from "../src/wire.ts";

interface User {
  name: string | null;
  age: number | null;
}

function userObj(id: bigint, name: string, age: number): Buffer {
  return encodeObject("User", id, [
    ["name", { tag: "string", value: name } satisfies EncValue],
    ["age", { tag: "u32", value: age } satisfies EncValue],
  ]);
}

/** Reply to a query string with the wire-faithful response shape. */
function respondQuery(sock: Socket, reqId: number, q: string): void {
  const send = (kind: number, payload: Buffer) => sock.write(encodeFrame(reqId, kind, payload));
  if (q === "User") {
    send(Resp.Objects, encodeObjectsPayload([userObj(1n, "Alice", 30), userObj(2n, "Bob", 25)]));
  } else if (q === "User.get(1)") {
    send(Resp.Single, userObj(1n, "Alice", 30));
  } else if (q === "User.get(99)") {
    send(Resp.Objects, encodeObjectsPayload([]));
  } else if (q.startsWith("User.create")) {
    send(Resp.Single, userObj(2n, "Bob", 25));
  } else if (q.includes("delete")) {
    send(Resp.Done, Buffer.alloc(0));
  } else {
    send(Resp.Error, encodeErrorPayload("no such type: Bad"));
  }
}

/** Start a mock server; `onFrame` drives each inbound frame. Returns its port. */
async function startMock(onFrame: (sock: Socket, frame: Frame) => void): Promise<{ port: number; server: Server }> {
  const server = createServer((sock) => {
    const parser = new FrameParser();
    sock.on("data", (chunk: Buffer) => {
      parser.push(chunk);
      try {
        for (let f = parser.next(); f !== null; f = parser.next()) onFrame(sock, f);
      } catch {
        sock.destroy();
      }
    });
    sock.on("error", () => {});
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return { port: addr.port, server };
}

/** The standard request/response mock used by most tests. */
function standardHandler(sock: Socket, frame: Frame): void {
  switch (frame.kind) {
    case Req.Ping:
      sock.write(encodeFrame(frame.reqId, Resp.Pong, Buffer.alloc(0)));
      break;
    case Req.Query: {
      const len = frame.payload.readUInt32BE(0);
      const q = frame.payload.subarray(4, 4 + len).toString("utf8");
      respondQuery(sock, frame.reqId, q);
      break;
    }
    default:
      sock.write(encodeFrame(frame.reqId, Resp.Error, encodeErrorPayload("mock: unsupported")));
  }
}

test("full request/response surface", async () => {
  const { port, server } = await startMock(standardHandler);
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });

  await client.ping();

  const one = await client.fetchOne(Query.get<User>("User", 1));
  assert.ok(one);
  assert.equal(one.id, 1n);
  assert.deepEqual(one.data, { name: "Alice", age: 30 });

  const all = await client.fetch(Query.all<User>("User"));
  assert.equal(all.length, 2);
  assert.deepEqual(all.map((r) => r.id), [1n, 2n]);

  assert.equal(await client.fetchOne(Query.get<User>("User", 99)), null);

  const created = await client.create(Query.raw<User>('User.create({ name: "Bob" })'));
  assert.equal(created.id, 2n);
  assert.equal(created.data.name, "Bob");

  const del = await client.execute(Query.raw<User>("User.get(1).delete()"));
  assert.equal(del.kind, "done");

  await assert.rejects(
    client.query("Bad.thing"),
    (e: unknown) => e instanceof RhypedbError && e.code === "server" && e.message.includes("no such type"),
  );

  client.close();
  server.close();
});

test("concurrent calls pipeline and stay correctly correlated (FIFO)", async () => {
  const { port, server } = await startMock(standardHandler);
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });

  // Fire a mix of queries without awaiting between them; each must resolve with
  // its own correctly-correlated reply despite sharing one connection.
  const [list, one, pong, created] = await Promise.all([
    client.fetch(Query.all<User>("User")),
    client.fetchOne(Query.get<User>("User", 1)),
    client.ping(),
    client.create(Query.raw<User>("User.create({})")),
  ]);
  assert.equal(list.length, 2);
  assert.equal(one?.id, 1n);
  assert.equal(pong, undefined); // ping resolves void
  assert.equal(created.id, 2n);

  client.close();
  server.close();
});

test("a malformed (but framed) reply rejects that call without latching the connection", async () => {
  const { port, server } = await startMock((sock, frame) => {
    if (frame.kind === Req.Ping) {
      sock.write(encodeFrame(frame.reqId, Resp.Pong, Buffer.alloc(0)));
      return;
    }
    // A RESP_SINGLE whose object body is truncated → a decode error, framing intact.
    const truncated = Buffer.from([0x00, 0x01, 0x55]); // type_len=1, "U", then nothing
    sock.write(encodeFrame(frame.reqId, Resp.Single, truncated));
  });
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });

  await assert.rejects(
    client.fetchOne(Query.get<User>("User", 1)),
    (e: unknown) => e instanceof RhypedbError && e.code === "decode",
  );
  // The framing was intact, so the connection is still usable.
  await client.ping();

  client.close();
  server.close();
});

test("a peer hangup mid-request latches the connection closed", async () => {
  const { port, server } = await startMock((sock) => {
    sock.destroy(); // read the request, then drop without replying
  });
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });

  // The in-flight request fails with an I/O error...
  await assert.rejects(client.ping(), (e: unknown) => e instanceof RhypedbError && e.code === "io");
  // ...and every subsequent call fails fast as closed (no mis-correlation).
  await assert.rejects(client.ping(), (e: unknown) => e instanceof RhypedbError && e.code === "closed");
  await assert.rejects(
    client.query("User"),
    (e: unknown) => e instanceof RhypedbError && e.code === "closed",
  );

  server.close();
});

test("connect to a closed port rejects with a connect error", async () => {
  // Bind then immediately close to get a port nothing is listening on.
  const { port, server } = await startMock(() => {});
  await new Promise<void>((resolve) => server.close(() => resolve()));
  await assert.rejects(
    AsyncClient.connect({ host: "127.0.0.1", port }, { connectTimeoutMs: 1000 }),
    (e: unknown) => e instanceof RhypedbError && e.code === "connect",
  );
});
