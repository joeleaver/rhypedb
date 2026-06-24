import { test } from "node:test";
import assert from "node:assert/strict";

import {
  Req,
  Resp,
  MAX_FRAME_PAYLOAD,
  WireError,
  FrameParser,
  encodeFrame,
  encodeObject,
  encodeObjectsPayload,
  encodeErrorPayload,
  encodePreparedPayload,
  encodeQueryPayload,
  encodeExecutePayload,
  encodeVectorBatchPayload,
  decodeObjectsPayload,
  decodeSinglePayload,
  decodePreparedPayload,
  decodeErrorPayload,
  rfc3339FromMillis,
  type EncValue,
} from "../src/wire.ts";

// ---- value round-trips (encode a single-field object, decode it back) -------

function roundTripField(name: string, value: EncValue): unknown {
  const obj = encodeObject("T", 7n, [[name, value]]);
  const decoded = decodeSinglePayload(obj);
  assert.equal(decoded.typeName, "T");
  assert.equal(decoded.id, 7n);
  return decoded.fields[name];
}

test("every value tag round-trips through encode/decode", () => {
  assert.equal(roundTripField("s", { tag: "string", value: "héllo" }), "héllo");
  assert.equal(roundTripField("u32", { tag: "u32", value: 4_000_000_000 }), 4_000_000_000);
  assert.equal(roundTripField("u64", { tag: "u64", value: 18_000_000_000_000_000_000n }), 18_000_000_000_000_000_000n);
  assert.equal(roundTripField("i32", { tag: "i32", value: -2_000_000_000 }), -2_000_000_000);
  assert.equal(roundTripField("i64", { tag: "i64", value: -9_000_000_000_000_000_000n }), -9_000_000_000_000_000_000n);
  assert.equal(roundTripField("f32", { tag: "f32", value: 0.5 }), 0.5); // exact in f32
  assert.equal(roundTripField("f64", { tag: "f64", value: 3.141592653589793 }), 3.141592653589793);
  assert.equal(roundTripField("b", { tag: "bool", value: true }), true);
  assert.equal(roundTripField("b0", { tag: "bool", value: false }), false);
  assert.equal(roundTripField("n", { tag: "null" }), null);
  // Bytes → base64 string.
  assert.equal(roundTripField("blob", { tag: "bytes", value: new Uint8Array([0, 1, 2]) }), "AAEC");
  // Json → parsed value.
  assert.deepEqual(roundTripField("meta", { tag: "json", value: { k: 1, tags: ["a", "b"] } }), {
    k: 1,
    tags: ["a", "b"],
  });
});

test("u64 fields decode losslessly as bigint past 2^53", () => {
  const big = 9_007_199_254_740_993n; // 2^53 + 1, not representable as a JS number
  assert.equal(roundTripField("v", { tag: "u64", value: big }), big);
});

test("DateTime decodes to RFC 3339 matching the server's variable-precision form", () => {
  // 2021-01-01T00:00:00Z — whole seconds, no fractional part (matches Rust `time`).
  const ms = BigInt(Date.UTC(2021, 0, 1, 0, 0, 0, 0));
  assert.equal(roundTripField("t", { tag: "datetime", millis: ms }), "2021-01-01T00:00:00Z");
  // Trailing zeros trimmed: .120 → .12, .100 → .1, .123 stays.
  assert.equal(rfc3339FromMillis(BigInt(Date.UTC(2021, 0, 1, 0, 0, 0, 123))), "2021-01-01T00:00:00.123Z");
  assert.equal(rfc3339FromMillis(BigInt(Date.UTC(2021, 0, 1, 0, 0, 0, 120))), "2021-01-01T00:00:00.12Z");
  assert.equal(rfc3339FromMillis(BigInt(Date.UTC(2021, 0, 1, 0, 0, 0, 100))), "2021-01-01T00:00:00.1Z");
});

// ---- byte-exact fixtures (lock the wire format, not just self-consistency) --

test("object encodes to the exact documented byte layout", () => {
  // Type "T" (len 1), id 7, one field "a" = U32(1).
  const obj = encodeObject("T", 7n, [["a", { tag: "u32", value: 1 }]]);
  const expected = Buffer.from([
    0x00, 0x01, // type_len = 1
    0x54, // "T"
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, // id = 7 (u64 BE)
    0x00, 0x01, // field count = 1
    0x00, 0x01, // name_len = 1
    0x61, // "a"
    0x02, // tag U32
    0x00, 0x00, 0x00, 0x01, // value 1 (u32 BE)
  ]);
  assert.deepEqual(obj, expected);
});

test("query / execute / error / prepared payloads are byte-exact", () => {
  assert.deepEqual(
    encodeQueryPayload("User"),
    Buffer.from([0x00, 0x00, 0x00, 0x04, 0x55, 0x73, 0x65, 0x72]),
  );
  assert.deepEqual(encodeExecutePayload(1n), Buffer.from([0, 0, 0, 0, 0, 0, 0, 1]));
  assert.equal(decodePreparedPayload(encodePreparedPayload(42n)), 42n);
  assert.equal(decodeErrorPayload(encodeErrorPayload("boom")), "boom");
});

test("vector batch payload is byte-exact (BE header, LE f32 components)", () => {
  const payload = encodeVectorBatchPayload("Doc", "embedding", [[5n, [1.0, 0.0]]]);
  const expected = Buffer.concat([
    Buffer.from([0x00, 0x03]), // type_len 3
    Buffer.from("Doc"),
    Buffer.from([0x00, 0x09]), // field_len 9
    Buffer.from("embedding"),
    Buffer.from([0x00, 0x00, 0x00, 0x01]), // count 1
    Buffer.from([0, 0, 0, 0, 0, 0, 0, 5]), // id 5 (u64 BE)
    Buffer.from([0x00, 0x00, 0x00, 0x02]), // dim 2 (u32 BE)
    (() => { const b = Buffer.allocUnsafe(8); b.writeFloatLE(1.0, 0); b.writeFloatLE(0.0, 4); return b; })(),
  ]);
  assert.deepEqual(payload, expected);
});

// ---- objects payload --------------------------------------------------------

test("objects payload round-trips a list", () => {
  const a = encodeObject("User", 1n, [["name", { tag: "string", value: "Ada" }]]);
  const b = encodeObject("User", 2n, [["name", { tag: "string", value: "Bob" }]]);
  const objs = decodeObjectsPayload(encodeObjectsPayload([a, b]));
  assert.equal(objs.length, 2);
  assert.deepEqual(objs.map((o) => [o.id, o.fields.name]), [[1n, "Ada"], [2n, "Bob"]]);
});

// ---- frame framing ----------------------------------------------------------

test("FrameParser reassembles a frame split across arbitrary chunk boundaries", () => {
  const payload = encodeQueryPayload("User.filter(.age > 18)");
  const frame = encodeFrame(9, Req.Query, payload);
  const parser = new FrameParser();
  // Feed one byte at a time — the parser must retain partial state and only
  // yield the frame once every byte has arrived.
  for (let i = 0; i < frame.length; i++) {
    assert.equal(parser.next(), null, `no frame until byte ${i + 1}/${frame.length}`);
    parser.push(frame.subarray(i, i + 1));
  }
  const got = parser.next();
  assert.ok(got);
  assert.equal(got.reqId, 9);
  assert.equal(got.kind, Req.Query);
  assert.deepEqual(got.payload, payload);
  assert.equal(parser.next(), null);
});

test("FrameParser yields multiple frames coalesced in one chunk", () => {
  const f1 = encodeFrame(1, Resp.Pong, Buffer.alloc(0));
  const f2 = encodeFrame(2, Resp.Done, Buffer.alloc(0));
  const parser = new FrameParser();
  parser.push(Buffer.concat([f1, f2]));
  assert.equal(parser.next()?.reqId, 1);
  assert.equal(parser.next()?.reqId, 2);
  assert.equal(parser.next(), null);
});

test("FrameParser rejects an out-of-bounds frame length", () => {
  const parser = new FrameParser();
  const bad = Buffer.alloc(4);
  bad.writeUInt32BE(2, 0); // len < 5
  parser.push(bad);
  assert.throws(() => parser.next(), WireError);

  const huge = Buffer.alloc(4);
  huge.writeUInt32BE(MAX_FRAME_PAYLOAD + 6, 0);
  const p2 = new FrameParser();
  p2.push(huge);
  assert.throws(() => p2.next(), WireError);
});

test("decoding truncated payloads throws WireError, never a silent wrong value", () => {
  assert.throws(() => decodeObjectsPayload(Buffer.from([0x00])), WireError); // count truncated
  // count says 1 object but no object bytes follow
  assert.throws(() => decodeObjectsPayload(Buffer.from([0, 0, 0, 1])), WireError);
  assert.throws(() => decodeErrorPayload(Buffer.from([0, 0, 0, 5, 0x61])), WireError); // body short
});
