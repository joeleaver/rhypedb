//! Generated-client end-to-end: run `rhypedb-codegen --lang ts` over a fixture
//! schema, then drive the GENERATED typed seeds (`Account.all()`,
//! `Account.create({..})`, …) against a real rhypedb server over a socket. This
//! is the definitive proof that the codegen retarget (Inc5) is correct: the
//! emitted interfaces type-check against `@rhypedb/client`, the seed constructors
//! produce queries the server accepts, and the decoded rows match the generated
//! types (64-bit ints → `bigint`, `DateTime`/`Bytes` → `string`, `Json` parsed).
//!
//! Opt-in (it needs the compiled server AND codegen binaries):
//!   cargo build -p rhypedb-server --no-default-features --bin rhypedb-server
//!   cargo build -p rhypedb-codegen
//! then run (codegen path defaults to ../../target/debug/rhypedb-codegen):
//!   RHYPEDB_SERVER_BIN=../../target/debug/rhypedb-server node --test test/generated.e2e.ts
//! Without RHYPEDB_SERVER_BIN the suite is skipped (TS-only CI stays green).
//!
//! The package isn't installed in-repo, so the one `@rhypedb/client` import in
//! the generated source is rewritten to the local `src/index.ts` before import —
//! the analog of the Rust E2E resolving `rhypedb-client` via a dev-dependency.

import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createServer } from "node:net";

import { AsyncClient } from "../src/client.ts";
import { type Query, type Row } from "../src/query.ts";

const BIN = process.env.RHYPEDB_SERVER_BIN;
const CODEGEN =
  process.env.RHYPEDB_CODEGEN_BIN ??
  fileURLToPath(new URL("../../../target/debug/rhypedb-codegen", import.meta.url));

const SCHEMA = `type Account {
  name: String @unique
  balance: i64 @indexed
  visits: u32
  ratio: f64
  active: Bool
  created: DateTime
  meta: Json
  embedding: Vector<4>
}
`;

/** The shape the generated `Account` interface yields, as the client returns it. */
interface Account {
  name: string | null;
  balance: bigint | null;
  visits: number | null;
  ratio: number | null;
  active: boolean | null;
  created: string | null;
  meta: unknown;
}

/** The slice of the generated value namespace this test drives. */
interface GeneratedAccount {
  readonly TYPE_NAME: "Account";
  readonly FIELDS: readonly string[];
  all(): Query<Account>;
  get(id: bigint | number): Query<Account>;
  filter(predicate: string): Query<Account>;
  create(row: Partial<Account>): Query<Account>;
}

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
  "generated TS client drives a real rhypedb server",
  { skip: BIN ? false : "set RHYPEDB_SERVER_BIN to run the generated-client E2E" },
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "rhypedb-ts-gen-e2e-"));
    const schemaPath = join(dir, "schema.rhype");
    writeFileSync(schemaPath, SCHEMA);

    // --- generate the typed client from the schema, then point its sole
    //     `@rhypedb/client` import at the local source so it resolves in-repo ---
    if (!existsSync(CODEGEN)) {
      throw new Error(`codegen binary not found at ${CODEGEN}; build it with \`cargo build -p rhypedb-codegen\` or set RHYPEDB_CODEGEN_BIN`);
    }
    const gen = spawnSync(CODEGEN, ["--schema", schemaPath, "--lang", "ts"], { encoding: "utf8" });
    if (gen.status !== 0) {
      throw new Error(`codegen failed (status ${gen.status}): ${gen.stderr}`);
    }
    const srcIndex = fileURLToPath(new URL("../src/index.ts", import.meta.url));
    const generated = gen.stdout.replaceAll('"@rhypedb/client"', JSON.stringify(srcIndex));
    const genPath = join(dir, "generated_client.ts");
    writeFileSync(genPath, generated);
    const mod = (await import(pathToFileURL(genPath).href)) as { Account: GeneratedAccount };
    const Account = mod.Account;
    assert.equal(Account.TYPE_NAME, "Account");
    // The vector field is documented, not a scalar column.
    assert.deepEqual([...Account.FIELDS], ["name", "balance", "visits", "ratio", "active", "created", "meta"]);

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

      // --- typed create via the generated seed, then verify the decode matches
      //     the generated types exactly ---
      const created = "2021-06-15T12:00:00Z";
      const ada: Row<Account> = await client.create(
        Account.create({
          name: "Ada",
          balance: 100n,
          visits: 3,
          ratio: 1.5,
          active: true,
          created,
          meta: { tier: "gold", level: 3 },
        }),
      );
      assert.equal(typeof ada.id, "bigint");
      assert.equal(ada.data.name, "Ada");
      assert.equal(typeof ada.data.balance, "bigint", "i64 must decode as bigint");
      assert.equal(ada.data.balance, 100n);
      assert.equal(typeof ada.data.visits, "number", "u32 must decode as number");
      assert.equal(ada.data.visits, 3);
      assert.equal(ada.data.ratio, 1.5);
      assert.equal(ada.data.active, true);
      assert.equal(typeof ada.data.created, "string", "DateTime must decode as string");
      assert.equal(
        new Date(ada.data.created!).getTime(),
        new Date(created).getTime(),
        "DateTime must round-trip to the same instant",
      );
      assert.deepEqual(ada.data.meta, { tier: "gold", level: 3 }, "Json must round-trip parsed");

      await client.create(
        Account.create({ name: "Bob", balance: 20n, visits: 1, ratio: 0.5, active: false }),
      );

      // --- typed list fetch ---
      const all = await client.fetch(Account.all());
      assert.equal(all.length, 2);

      // --- indexed i64 filter pushdown through the generated seed ---
      const rich = await client.fetchOne(Account.all().filter(".balance > 50"));
      assert.equal(rich?.data.name, "Ada");
      assert.equal(rich?.data.balance, 100n);

      // --- get by the lossless bigint id ---
      const got = await client.fetchOne(Account.get(ada.id));
      assert.equal(got?.id, ada.id);
      assert.equal(got?.data.name, "Ada");

      // --- prepared statement built from a generated seed ---
      const allAccounts = await client.prepare(Account.all());
      assert.equal((await client.fetchPrepared(allAccounts)).length, 2);

      // --- the doc-commented vector field still works on the server ---
      assert.equal(await client.ingestVectors("Account", "embedding", [[ada.id, [1, 0, 0, 0]]]), 1);
    } catch (e) {
      throw new Error(`${(e as Error).message}\n--- server log ---\n${serverLog}`);
    } finally {
      client?.close();
      server.kill("SIGKILL");
      rmSync(dir, { recursive: true, force: true });
    }
  },
);
