//! The rhypedb binary wire format — frame framing, the value/object codec, and
//! the request/response payload codecs. A byte-for-byte port of the Rust
//! `rhypedb-wire` crate (the single source of truth for the protocol), so this
//! client interoperates with the same server.
//!
//! Everything is big-endian except the per-component f32s in a vector batch,
//! which are little-endian (matching the engine's vector storage).

/** Maximum payload size per frame (16 MiB). Mirrors `MAX_FRAME_PAYLOAD`. */
export const MAX_FRAME_PAYLOAD = 16 * 1024 * 1024;

/** Format tag stamped into every change-event envelope. */
export const EVENT_FORMAT_TAG = "rhypedb-event-v1";

/** Request opcodes. */
export const Req = {
  Query: 0x01,
  Ping: 0x02,
  VectorBatch: 0x03,
  Prepare: 0x04,
  Execute: 0x05,
  Subscribe: 0x06,
  Unsubscribe: 0x07,
} as const;

/** Response opcodes. */
export const Resp = {
  Objects: 0x80,
  Single: 0x81,
  Done: 0x82,
  Error: 0x83,
  Pong: 0x84,
  Prepared: 0x85,
  Subscribed: 0x86,
  Event: 0x87,
  SubLagged: 0x88,
} as const;

/** Value serialization tags (the `ValueTag` enum on the Rust side). */
const Tag = {
  Null: 0,
  String: 1,
  U32: 2,
  U64: 3,
  I32: 4,
  I64: 5,
  F32: 6,
  F64: 7,
  Bool: 8,
  Bytes: 9,
  DateTime: 10,
  Json: 11,
} as const;

/** A change-event kind. Matches the lowercase wire form. */
export type ChangeKind = "create" | "update" | "delete";

const KIND_BIT: Record<ChangeKind, number> = { create: 0x01, update: 0x02, delete: 0x04 };
const FILTER_FLAG_TYPE = 0x01;
const FILTER_FLAG_OBJECT = 0x02;

/** An error decoding malformed wire bytes (framing or payload). */
export class WireError extends Error {
  override readonly name = "WireError";
}

/** A parsed inbound frame. */
export interface Frame {
  reqId: number;
  kind: number;
  payload: Buffer;
}

/**
 * A decoded value as the client surfaces it (the same render form as the Rust
 * client's `value_to_query_json` → `Row<T>` path, so generated row types match):
 * `u64`/`i64` → `bigint` (lossless — rhypedb ids/counters are 64-bit), other
 * numerics → `number`, `Bytes` → base64 `string`, `DateTime` → RFC 3339 `string`,
 * `Json` → the parsed value, `Null` → `null`.
 */
export type DecodedValue =
  | string
  | number
  | bigint
  | boolean
  | null
  | { [k: string]: unknown }
  | unknown[];

/** A decoded object: its type, id, and rendered scalar fields. */
export interface DecodedObject {
  typeName: string;
  id: bigint;
  fields: Record<string, DecodedValue>;
}

/**
 * A value to ENCODE into a FieldMap. The client itself never encodes objects
 * (writes go through query strings), but this drives the test suite and any
 * wire-faithful mock server. The discriminant names the exact tag.
 */
export type EncValue =
  | { tag: "null" }
  | { tag: "string"; value: string }
  | { tag: "u32"; value: number }
  | { tag: "u64"; value: bigint }
  | { tag: "i32"; value: number }
  | { tag: "i64"; value: bigint }
  | { tag: "f32"; value: number }
  | { tag: "f64"; value: number }
  | { tag: "bool"; value: boolean }
  | { tag: "bytes"; value: Uint8Array }
  | { tag: "datetime"; millis: bigint }
  | { tag: "json"; value: unknown };

// ---------------------------------------------------------------------------
// DateTime rendering: epoch millis (i64) → RFC 3339, matching the Rust `time`
// crate's variable-precision form (no fractional part when zero; trailing
// zeros trimmed otherwise) so the string equals what the server's JSON/HTTP
// path produces.
// ---------------------------------------------------------------------------
export function rfc3339FromMillis(ms: bigint): string {
  const n = Number(ms);
  // The server (Rust `time`, no `large-dates`) can only format years 0000-9999;
  // anything else — beyond JS `Date`'s ±8.64e15 ms range, or an extended-year
  // (`±YYYYYY`) `toISOString` — falls back to the raw decimal millis string, so
  // mirror that exactly (and avoid a `RangeError` on out-of-range input).
  if (!Number.isFinite(n) || Number.isNaN(new Date(n).getTime())) return ms.toString();
  const iso = new Date(n).toISOString(); // "2021-01-01T00:00:00.000Z"
  if (iso[0] === "+" || iso[0] === "-") return ms.toString(); // year outside 0000-9999
  return iso.replace(/\.(\d{3})Z$/, (_m, frac: string) => {
    const trimmed = frac.replace(/0+$/, "");
    return trimmed ? `.${trimmed}Z` : "Z";
  });
}

// ---------------------------------------------------------------------------
// Frame framing
// ---------------------------------------------------------------------------

/** Build a single frame: `[len:u32][req_id:u32][kind:u8][payload]`. */
export function encodeFrame(reqId: number, kind: number, payload: Buffer): Buffer {
  const len = 5 + payload.length;
  const out = Buffer.allocUnsafe(4 + len);
  out.writeUInt32BE(len, 0);
  out.writeUInt32BE(reqId, 4);
  out[8] = kind;
  payload.copy(out, 9);
  return out;
}

/**
 * Accumulates inbound bytes and yields whole frames. A TCP read can deliver a
 * partial frame or several frames at once; feed every chunk with [`push`] and
 * drain complete frames with [`next`]. Partial bytes are retained, so this is
 * safe across arbitrary chunk boundaries.
 */
export class FrameParser {
  #buf: Buffer = Buffer.alloc(0);

  push(chunk: Buffer): void {
    this.#buf = this.#buf.length === 0 ? chunk : Buffer.concat([this.#buf, chunk]);
  }

  /** Pull the next complete frame, or `null` if more bytes are needed. */
  next(): Frame | null {
    if (this.#buf.length < 4) return null;
    const len = this.#buf.readUInt32BE(0);
    if (len < 5) throw new WireError(`frame too small: ${len} bytes`);
    if (len > MAX_FRAME_PAYLOAD + 5) throw new WireError(`frame too large: ${len} bytes`);
    if (this.#buf.length < 4 + len) return null;
    const reqId = this.#buf.readUInt32BE(4);
    const kind = this.#buf[8]!;
    // Copy the payload out so it doesn't alias the retained accumulation buffer.
    const payload = Buffer.from(this.#buf.subarray(9, 4 + len));
    this.#buf = this.#buf.subarray(4 + len);
    return { reqId, kind, payload };
  }
}

// ---------------------------------------------------------------------------
// A bounds-checked cursor reader over a payload buffer.
// ---------------------------------------------------------------------------
class Reader {
  #buf: Buffer;
  pos: number;
  constructor(buf: Buffer, pos = 0) {
    this.#buf = buf;
    this.pos = pos;
  }
  #need(n: number, what: string): void {
    if (this.pos + n > this.#buf.length) throw new WireError(`${what}: truncated`);
  }
  u8(what: string): number {
    this.#need(1, what);
    return this.#buf[this.pos++]!;
  }
  u16(what: string): number {
    this.#need(2, what);
    const v = this.#buf.readUInt16BE(this.pos);
    this.pos += 2;
    return v;
  }
  u32(what: string): number {
    this.#need(4, what);
    const v = this.#buf.readUInt32BE(this.pos);
    this.pos += 4;
    return v;
  }
  i32(what: string): number {
    this.#need(4, what);
    const v = this.#buf.readInt32BE(this.pos);
    this.pos += 4;
    return v;
  }
  u64(what: string): bigint {
    this.#need(8, what);
    const v = this.#buf.readBigUInt64BE(this.pos);
    this.pos += 8;
    return v;
  }
  i64(what: string): bigint {
    this.#need(8, what);
    const v = this.#buf.readBigInt64BE(this.pos);
    this.pos += 8;
    return v;
  }
  f32(what: string): number {
    this.#need(4, what);
    const v = this.#buf.readFloatBE(this.pos);
    this.pos += 4;
    return v;
  }
  f64(what: string): number {
    this.#need(8, what);
    const v = this.#buf.readDoubleBE(this.pos);
    this.pos += 8;
    return v;
  }
  bytes(n: number, what: string): Buffer {
    this.#need(n, what);
    const v = this.#buf.subarray(this.pos, this.pos + n);
    this.pos += n;
    return v;
  }
  utf8(n: number, what: string): string {
    return this.bytes(n, what).toString("utf8");
  }
  atEnd(): boolean {
    return this.pos >= this.#buf.length;
  }
}

// ---------------------------------------------------------------------------
// Value + FieldMap decode (the form the client returns to callers)
// ---------------------------------------------------------------------------
function decodeValue(r: Reader): DecodedValue {
  const tag = r.u8("value tag");
  switch (tag) {
    case Tag.Null:
      return null;
    case Tag.String:
      return r.utf8(r.u32("string len"), "string");
    case Tag.U32:
      return r.u32("u32");
    case Tag.U64:
      return r.u64("u64");
    case Tag.I32:
      return r.i32("i32");
    case Tag.I64:
      return r.i64("i64");
    case Tag.F32:
      return r.f32("f32");
    case Tag.F64:
      return r.f64("f64");
    case Tag.Bool:
      return r.u8("bool") !== 0;
    case Tag.Bytes:
      return r.bytes(r.u32("bytes len"), "bytes").toString("base64");
    case Tag.DateTime:
      return rfc3339FromMillis(r.i64("datetime"));
    case Tag.Json: {
      const text = r.utf8(r.u32("json len"), "json");
      return JSON.parse(text) as DecodedValue;
    }
    default:
      throw new WireError(`unknown value tag ${tag}`);
  }
}

function decodeFields(r: Reader): Record<string, DecodedValue> {
  const count = r.u16("fields count");
  const fields: Record<string, DecodedValue> = {};
  for (let i = 0; i < count; i++) {
    const name = r.utf8(r.u16("field name len"), "field name");
    fields[name] = decodeValue(r);
  }
  return fields;
}

/** Decode one object at the reader's cursor: `[type_len:u16][type][id:u64][fields]`. */
function decodeObjectAt(r: Reader): DecodedObject {
  const typeName = r.utf8(r.u16("type len"), "type name");
  const id = r.u64("object id");
  const fields = decodeFields(r);
  return { typeName, id, fields };
}

/** Decode a `RESP_OBJECTS` payload: `[count:u32]` then that many objects. */
export function decodeObjectsPayload(payload: Buffer): DecodedObject[] {
  const r = new Reader(payload);
  const count = r.u32("objects count");
  const out: DecodedObject[] = [];
  for (let i = 0; i < count; i++) out.push(decodeObjectAt(r));
  return out;
}

/** Decode a `RESP_SINGLE` payload: exactly one object. */
export function decodeSinglePayload(payload: Buffer): DecodedObject {
  return decodeObjectAt(new Reader(payload));
}

/** Decode a `RESP_PREPARED` payload: `[stmt_id:u64]`. */
export function decodePreparedPayload(payload: Buffer): bigint {
  return new Reader(payload).u64("prepared stmt id");
}

/** Decode a `RESP_ERROR` payload: `[len:u32][utf8 message]`. */
export function decodeErrorPayload(payload: Buffer): string {
  const r = new Reader(payload);
  return r.utf8(r.u32("error len"), "error message");
}

// ---------------------------------------------------------------------------
// Request payload encoders
// ---------------------------------------------------------------------------

/** `[len:u32][utf8]` — shared by `Query` and `Prepare`. */
export function encodeQueryPayload(query: string): Buffer {
  const q = Buffer.from(query, "utf8");
  const out = Buffer.allocUnsafe(4 + q.length);
  out.writeUInt32BE(q.length, 0);
  q.copy(out, 4);
  return out;
}

/** Identical to a Query payload. */
export const encodePreparePayload = encodeQueryPayload;

/** `[stmt_id:u64]`. */
export function encodeExecutePayload(stmtId: bigint): Buffer {
  const out = Buffer.allocUnsafe(8);
  out.writeBigUInt64BE(stmtId, 0);
  return out;
}

/**
 * `[type_len:u16][type][field_len:u16][field][count:u32]` then `count` rows of
 * `[object_id:u64][dim:u32][f32 x dim, little-endian]`.
 */
export function encodeVectorBatchPayload(
  typeName: string,
  fieldName: string,
  rows: ReadonlyArray<readonly [bigint, ArrayLike<number>]>,
): Buffer {
  const t = Buffer.from(typeName, "utf8");
  const f = Buffer.from(fieldName, "utf8");
  let bytes = 2 + t.length + 2 + f.length + 4;
  for (const [, vec] of rows) bytes += 12 + vec.length * 4;
  const out = Buffer.allocUnsafe(bytes);
  let p = 0;
  out.writeUInt16BE(t.length, p); p += 2;
  t.copy(out, p); p += t.length;
  out.writeUInt16BE(f.length, p); p += 2;
  f.copy(out, p); p += f.length;
  out.writeUInt32BE(rows.length, p); p += 4;
  for (const [id, vec] of rows) {
    out.writeBigUInt64BE(id, p); p += 8;
    out.writeUInt32BE(vec.length, p); p += 4;
    for (let i = 0; i < vec.length; i++) {
      out.writeFloatLE(vec[i]!, p); p += 4;
    }
  }
  return out;
}

/** The structural shape `encodeSubscribeFilter` reads (the `SubscriptionFilter`). */
export interface SubscribeFilterParts {
  typeName?: string | undefined;
  objectId?: bigint | undefined;
  kinds: readonly ChangeKind[];
}

/**
 * Encode a `Subscribe` filter payload: `[flags:u8 bit0=has_type bit1=has_object]`
 * then (if set) `[type_len:u16][type]` and `[object_id:u64]`, then a
 * `[kinds:u8 bitmask]` (bit0 Create / bit1 Update / bit2 Delete; 0 = all kinds).
 * An empty type name is treated as "no type filter" (the server rejects a
 * zero-length name), keeping encode round-trippable.
 */
export function encodeSubscribeFilter(filter: SubscribeFilterParts): Buffer {
  const typeName = filter.typeName && filter.typeName.length > 0 ? filter.typeName : undefined;
  const parts: Buffer[] = [];
  let flags = 0;
  if (typeName !== undefined) flags |= FILTER_FLAG_TYPE;
  if (filter.objectId !== undefined) flags |= FILTER_FLAG_OBJECT;
  parts.push(Buffer.from([flags]));
  if (typeName !== undefined) {
    const b = Buffer.from(typeName, "utf8");
    const lh = Buffer.allocUnsafe(2);
    lh.writeUInt16BE(b.length, 0);
    parts.push(lh, b);
  }
  if (filter.objectId !== undefined) {
    const o = Buffer.allocUnsafe(8);
    o.writeBigUInt64BE(filter.objectId, 0);
    parts.push(o);
  }
  let kinds = 0;
  for (const k of filter.kinds) kinds |= KIND_BIT[k];
  parts.push(Buffer.from([kinds]));
  return Buffer.concat(parts);
}

/** Encode an `Unsubscribe` request payload: `[sub_req_id:u32]`. */
export function encodeUnsubscribePayload(handle: number): Buffer {
  const out = Buffer.allocUnsafe(4);
  out.writeUInt32BE(handle, 0);
  return out;
}

/**
 * The wire form of a change event (the `RESP_EVENT` JSON). `id`/`version` are
 * decimal strings (lossless past 2^53 through JSON), `type` is the type name,
 * `v` is the format tag, `fields` is best-effort scalars in the `/query` form.
 */
export interface WireEvent {
  v: string;
  kind: string;
  type: string;
  id: string;
  version: string;
  fields?: Record<string, unknown>;
}

/** Decode a `RESP_EVENT` payload (JSON). Throws `WireError` if it isn't an object. */
export function decodeEventPayload(payload: Buffer): WireEvent {
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload.toString("utf8"));
  } catch (e) {
    throw new WireError(`event json: ${(e as Error).message}`);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new WireError("event json: not an object");
  }
  return parsed as WireEvent;
}

// ---------------------------------------------------------------------------
// Value + object ENCODE — not used by the client (writes are query strings),
// but exercised by the test suite and available for a wire-faithful mock.
// ---------------------------------------------------------------------------
function encodeValueInto(v: EncValue, parts: Buffer[]): void {
  const head = (tag: number, extra = 0): Buffer => {
    const b = Buffer.allocUnsafe(1 + extra);
    b[0] = tag;
    return b;
  };
  switch (v.tag) {
    case "null":
      parts.push(head(Tag.Null));
      break;
    case "string": {
      const s = Buffer.from(v.value, "utf8");
      const h = head(Tag.String, 4);
      h.writeUInt32BE(s.length, 1);
      parts.push(h, s);
      break;
    }
    case "u32": {
      const h = head(Tag.U32, 4);
      h.writeUInt32BE(v.value, 1);
      parts.push(h);
      break;
    }
    case "u64": {
      const h = head(Tag.U64, 8);
      h.writeBigUInt64BE(v.value, 1);
      parts.push(h);
      break;
    }
    case "i32": {
      const h = head(Tag.I32, 4);
      h.writeInt32BE(v.value, 1);
      parts.push(h);
      break;
    }
    case "i64": {
      const h = head(Tag.I64, 8);
      h.writeBigInt64BE(v.value, 1);
      parts.push(h);
      break;
    }
    case "f32": {
      const h = head(Tag.F32, 4);
      h.writeFloatBE(v.value, 1);
      parts.push(h);
      break;
    }
    case "f64": {
      const h = head(Tag.F64, 8);
      h.writeDoubleBE(v.value, 1);
      parts.push(h);
      break;
    }
    case "bool": {
      const h = head(Tag.Bool, 1);
      h[1] = v.value ? 1 : 0;
      parts.push(h);
      break;
    }
    case "bytes": {
      const h = head(Tag.Bytes, 4);
      h.writeUInt32BE(v.value.length, 1);
      parts.push(h, Buffer.from(v.value));
      break;
    }
    case "datetime": {
      const h = head(Tag.DateTime, 8);
      h.writeBigInt64BE(v.millis, 1);
      parts.push(h);
      break;
    }
    case "json": {
      const s = Buffer.from(JSON.stringify(v.value), "utf8");
      const h = head(Tag.Json, 4);
      h.writeUInt32BE(s.length, 1);
      parts.push(h, s);
      break;
    }
  }
}

/** Encode a FieldMap: `[count:u16]` then `[name_len:u16][name][tag][value]` each. */
export function encodeFields(fields: ReadonlyArray<readonly [string, EncValue]>): Buffer {
  const parts: Buffer[] = [];
  const count = Buffer.allocUnsafe(2);
  count.writeUInt16BE(fields.length, 0);
  parts.push(count);
  for (const [name, value] of fields) {
    const n = Buffer.from(name, "utf8");
    const nh = Buffer.allocUnsafe(2);
    nh.writeUInt16BE(n.length, 0);
    parts.push(nh, n);
    encodeValueInto(value, parts);
  }
  return Buffer.concat(parts);
}

/** Encode one object: `[type_len:u16][type][id:u64][fields]`. */
export function encodeObject(
  typeName: string,
  id: bigint,
  fields: ReadonlyArray<readonly [string, EncValue]>,
): Buffer {
  const t = Buffer.from(typeName, "utf8");
  const head = Buffer.allocUnsafe(2 + t.length + 8);
  head.writeUInt16BE(t.length, 0);
  t.copy(head, 2);
  head.writeBigUInt64BE(id, 2 + t.length);
  return Buffer.concat([head, encodeFields(fields)]);
}

/** Encode a `RESP_OBJECTS` payload: `[count:u32]` then the objects. */
export function encodeObjectsPayload(objects: ReadonlyArray<Buffer>): Buffer {
  const head = Buffer.allocUnsafe(4);
  head.writeUInt32BE(objects.length, 0);
  return Buffer.concat([head, ...objects]);
}

/** Encode a `RESP_ERROR` payload: `[len:u32][utf8]`. */
export function encodeErrorPayload(message: string): Buffer {
  const m = Buffer.from(message, "utf8");
  const out = Buffer.allocUnsafe(4 + m.length);
  out.writeUInt32BE(m.length, 0);
  m.copy(out, 4);
  return out;
}

/** Encode a `RESP_PREPARED` payload: `[stmt_id:u64]`. */
export function encodePreparedPayload(stmtId: bigint): Buffer {
  const out = Buffer.allocUnsafe(8);
  out.writeBigUInt64BE(stmtId, 0);
  return out;
}
