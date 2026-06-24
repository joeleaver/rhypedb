//! Real-server end-to-end: spawn an actual rhypedb server binary and drive the
//! TypeScript client against it over a real TCP socket. This is the definitive
//! proof that the TS wire codec is byte-faithful to the server — the mock-server
//! tests share our own codec, but this talks to the real engine.
//!
//! Opt-in (it needs the compiled server): set RHYPEDB_SERVER_BIN to the binary
//! path. Build a no-ONNX server with:
//!   cargo build -p rhypedb-server --no-default-features --bin rhypedb-server
//! then run:
//!   RHYPEDB_SERVER_BIN=../../target/debug/rhypedb-server node --test test/realserver.e2e.ts
//! Without the env var the suite is skipped (so a TS-only CI stays green).

import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "node:net";

import { AsyncClient } from "../src/client.ts";
import { Query } from "../src/query.ts";

const BIN = process.env.RHYPEDB_SERVER_BIN;

interface User {
  name: string | null;
  age: number | null;
  active: boolean | null;
}

const SCHEMA = `type User {
  name: String @unique
  age: u32 @indexed
  active: Bool
}
`;

/** Ask the OS for a free TCP port (small TOCTOU window, fine for a local test). */
function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createServer();
    s.listen(0, "127.0.0.1", () => {
      const a = s.address();
      if (a === null || typeof a === "string") return reject(new Error("no port"));
      const p = a.port;
      s.close(() => resolve(p));
    });
    s.on("error", reject);
  });
}

/** Connect with retries while the freshly-spawned server is still binding. */
async function connectWithRetry(port: number, deadlineMs: number): Promise<AsyncClient> {
  const start = Date.now();
  let lastErr: unknown;
  while (Date.now() - start < deadlineMs) {
    try {
      return await AsyncClient.connect({ host: "127.0.0.1", port }, { connectTimeoutMs: 500 });
    } catch (e) {
      lastErr = e;
      await new Promise((r) => setTimeout(r, 50));
    }
  }
  throw lastErr ?? new Error("server did not come up");
}

test(
  "TS client drives a real rhypedb server over a socket",
  { skip: BIN ? false : "set RHYPEDB_SERVER_BIN to run the real-server E2E" },
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "rhypedb-ts-e2e-"));
    const schemaPath = join(dir, "schema.rhype");
    writeFileSync(schemaPath, SCHEMA);
    const httpPort = await freePort();
    const tcpPort = await freePort();

    const server: ChildProcess = spawn(
      BIN!,
      [
        "--schema", schemaPath,
        "--data-dir", join(dir, "data"),
        "--listen", `127.0.0.1:${httpPort}`,
        "--tcp-listen", `127.0.0.1:${tcpPort}`,
      ],
      { stdio: ["ignore", "pipe", "pipe"] },
    );
    let serverLog = "";
    server.stdout?.on("data", (b: Buffer) => (serverLog += b.toString()));
    server.stderr?.on("data", (b: Buffer) => (serverLog += b.toString()));

    let client: AsyncClient | null = null;
    try {
      client = await connectWithRetry(tcpPort, 15_000);

      // --- liveness ---
      await client.ping();

      // --- typed create + read-back (proves create-literal + binary decode agree) ---
      const ada = await client.create(
        Query.raw<User>('User.create({ name: "Ada", age: 30, active: true })'),
      );
      assert.equal(ada.data.name, "Ada");
      assert.equal(ada.data.age, 30);
      assert.equal(ada.data.active, true);
      assert.equal(typeof ada.id, "bigint");

      await client.create(Query.raw<User>('User.create({ name: "Bob", age: 20, active: false })'));

      // --- typed fetch (list) ---
      const all = await client.fetch(Query.all<User>("User"));
      assert.equal(all.length, 2);

      // --- indexed filter ---
      const adult = await client.fetchOne(Query.all<User>("User").filter(".age > 25"));
      assert.equal(adult?.data.name, "Ada");

      // --- get by id (round-trips the bigint id) ---
      const got = await client.fetchOne(Query.get<User>("User", ada.id));
      assert.equal(got?.id, ada.id);
      assert.equal(got?.data.name, "Ada");

      // --- raw update, then confirm ---
      await client.execute(Query.raw<User>(`User.get(${ada.id}).update({ age: 31 })`));
      const updated = await client.fetchOne(Query.get<User>("User", ada.id));
      assert.equal(updated?.data.age, 31);

      // --- delete, then confirm it's gone ---
      await client.execute(Query.raw<User>(`User.get(${ada.id}).delete()`));
      const remaining = await client.fetch(Query.all<User>("User"));
      assert.equal(remaining.length, 1);
      assert.ok(remaining.every((r) => r.id !== ada.id));
    } catch (e) {
      throw new Error(`${(e as Error).message}\n--- server log ---\n${serverLog}`);
    } finally {
      client?.close();
      server.kill("SIGKILL");
      rmSync(dir, { recursive: true, force: true });
    }
  },
);
