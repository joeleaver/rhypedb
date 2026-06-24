//! Prepared-statement + vector-ingest tests against a stateful wire-faithful
//! mock (per-connection `stmt_id -> query` map, mirroring the real server).

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
  encodePreparedPayload,
  type EncValue,
  type Frame,
} from "../src/wire.ts";

interface User {
  name: string | null;
}

function userObj(id: bigint, name: string): Buffer {
  return encodeObject("User", id, [["name", { tag: "string", value: name } satisfies EncValue]]);
}

function readQuery(payload: Buffer): string {
  const len = payload.readUInt32BE(0);
  return payload.subarray(4, 4 + len).toString("utf8");
}

function respondQuery(sock: Socket, reqId: number, q: string): void {
  const send = (kind: number, payload: Buffer) => sock.write(encodeFrame(reqId, kind, payload));
  if (q === "User") send(Resp.Objects, encodeObjectsPayload([userObj(1n, "Alice"), userObj(2n, "Bob")]));
  else if (q === "User.get(1)") send(Resp.Single, userObj(1n, "Alice"));
  else if (q.startsWith("User.create")) send(Resp.Single, userObj(3n, "Carol"));
  else send(Resp.Error, encodeErrorPayload(`no such type for ${q}`));
}

/** A stateful mock: prepared statements are per-connection state. */
function startMock(): Promise<{ port: number; server: Server }> {
  const server = createServer((sock) => {
    const parser = new FrameParser();
    const prepared = new Map<bigint, string>();
    let nextStmt = 1n;
    const onFrame = (frame: Frame) => {
      switch (frame.kind) {
        case Req.Ping:
          sock.write(encodeFrame(frame.reqId, Resp.Pong, Buffer.alloc(0)));
          break;
        case Req.Query:
          respondQuery(sock, frame.reqId, readQuery(frame.payload));
          break;
        case Req.Prepare: {
          const q = readQuery(frame.payload);
          if (q === "PARSEFAIL") {
            sock.write(encodeFrame(frame.reqId, Resp.Error, encodeErrorPayload("parse error")));
          } else {
            const id = nextStmt++;
            prepared.set(id, q);
            sock.write(encodeFrame(frame.reqId, Resp.Prepared, encodePreparedPayload(id)));
          }
          break;
        }
        case Req.Execute: {
          const stmtId = frame.payload.readBigUInt64BE(0);
          const q = prepared.get(stmtId);
          if (q === undefined) {
            sock.write(encodeFrame(frame.reqId, Resp.Error, encodeErrorPayload(`unknown statement id ${stmtId}`)));
          } else {
            respondQuery(sock, frame.reqId, q);
          }
          break;
        }
        case Req.VectorBatch: {
          // Decode just the type name to allow an error case.
          const tlen = frame.payload.readUInt16BE(0);
          const typeName = frame.payload.subarray(2, 2 + tlen).toString("utf8");
          if (typeName === "Nope") {
            sock.write(encodeFrame(frame.reqId, Resp.Error, encodeErrorPayload("no vector index")));
          } else {
            sock.write(encodeFrame(frame.reqId, Resp.Done, Buffer.alloc(0)));
          }
          break;
        }
        default:
          sock.write(encodeFrame(frame.reqId, Resp.Error, encodeErrorPayload("mock: unsupported")));
      }
    };
    sock.on("data", (chunk: Buffer) => {
      parser.push(chunk);
      try {
        for (let f = parser.next(); f !== null; f = parser.next()) onFrame(f);
      } catch {
        sock.destroy();
      }
    });
    sock.on("error", () => {});
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const a = server.address();
      if (a === null || typeof a === "string") throw new Error("no port");
      resolve({ port: a.port, server });
    });
  });
}

test("prepared statements: the full family re-runs a fixed query", async () => {
  const { port, server } = await startMock();
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });

  const all = await client.prepare(Query.all<User>("User"));
  assert.equal((await client.fetchPrepared(all)).length, 2);
  assert.equal((await client.fetchPrepared(all)).length, 2); // re-runs, no re-parse
  const res = await client.executePrepared(all);
  assert.equal(res.kind === "objects" ? res.objects.length : -1, 2);

  const get1 = await client.prepare(Query.get<User>("User", 1));
  assert.equal((await client.fetchOnePrepared(get1))?.id, 1n);

  const mk = await client.prepare(Query.raw<User>('User.create({ name: "Carol" })'));
  assert.equal((await client.createPrepared(mk)).id, 3n);

  // A prepare-time server error surfaces.
  await assert.rejects(
    client.prepare(Query.raw<User>("PARSEFAIL")),
    (e: unknown) => e instanceof RhypedbError && e.code === "server",
  );
  // An execute-time server error (handled at prepare-fine/execute-fail) stays usable.
  const bad = await client.prepare(Query.raw<User>("Bad.thing"));
  await assert.rejects(
    client.executePrepared(bad),
    (e: unknown) => e instanceof RhypedbError && e.code === "server",
  );
  await client.ping();

  client.close();
  server.close();
});

test("a prepared statement from another client is rejected before any I/O", async () => {
  const { port, server } = await startMock();
  const a = await AsyncClient.connect({ host: "127.0.0.1", port });
  const b = await AsyncClient.connect({ host: "127.0.0.1", port });

  const stmt = await a.prepare(Query.all<User>("User"));
  await assert.rejects(
    b.executePrepared(stmt),
    (e: unknown) => e instanceof RhypedbError && e.code === "foreign_statement",
  );
  // b is untouched and still works; a still runs its own statement.
  await b.ping();
  assert.equal((await a.fetchPrepared(stmt)).length, 2);

  a.close();
  b.close();
  server.close();
});

test("ingestVectors: owned + typed-array rows, server error, oversized-batch guard", async () => {
  const { port, server } = await startMock();
  const client = await AsyncClient.connect({ host: "127.0.0.1", port });

  assert.equal(await client.ingestVectors("Doc", "embedding", [[1n, [0.5, -1, 2]], [7n, [3, 4, 5]]]), 2);
  // A Float32Array row encodes identically (ArrayLike<number>).
  assert.equal(await client.ingestVectors("Doc", "embedding", [[2n, new Float32Array([0.1, 0.2])]]), 1);
  // Empty batch → 0.
  assert.equal(await client.ingestVectors("Doc", "embedding", []), 0);

  // Server-side error surfaces, connection stays usable.
  await assert.rejects(
    client.ingestVectors("Nope", "embedding", [[1n, [1, 2]]]),
    (e: unknown) => e instanceof RhypedbError && e.code === "server",
  );
  await client.ping();

  // Oversized batch is refused client-side (before any write); connection pristine.
  const huge: Array<[bigint, number[]]> = [[1n, new Array(16 * 1024 * 1024 / 4 + 1).fill(0)]];
  await assert.rejects(
    client.ingestVectors("Doc", "embedding", huge),
    (e: unknown) => e instanceof RhypedbError && e.code === "batch_too_large",
  );
  await client.ping();

  client.close();
  server.close();
});
