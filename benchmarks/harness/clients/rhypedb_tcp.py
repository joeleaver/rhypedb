"""rhypedb binary TCP client.

Matches the wire format defined in crates/rhypedb-server/src/protocol.rs.

Frame: [len: u32 BE][req_id: u32 BE][type: u8][payload]

Reuses the engine's serialize_fields format for object fields.
"""

import socket
import struct


# Request types
REQ_QUERY = 0x01
REQ_PING = 0x02
REQ_VECTOR_BATCH = 0x03

# Response types
RESP_OBJECTS = 0x80
RESP_SINGLE = 0x81
RESP_DONE = 0x82
RESP_ERROR = 0x83
RESP_PONG = 0x84

# Value tags (from engine::object)
TAG_NULL = 0
TAG_STRING = 1
TAG_U32 = 2
TAG_U64 = 3
TAG_I32 = 4
TAG_I64 = 5
TAG_F32 = 6
TAG_F64 = 7
TAG_BOOL = 8
TAG_BYTES = 9


class RhypedbError(Exception):
    pass


class RhypedbTcpClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 4201):
        self.host = host
        self.port = port
        self.sock = None
        self.req_id = 0
        self._buf = b""

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.host, self.port))

    def close(self) -> None:
        if self.sock:
            self.sock.close()
            self.sock = None

    def _next_req_id(self) -> int:
        self.req_id = (self.req_id + 1) & 0xFFFFFFFF
        return self.req_id

    def _send_frame(self, req_id: int, kind: int, payload: bytes) -> None:
        body = struct.pack(">IB", req_id, kind) + payload
        frame = struct.pack(">I", len(body)) + body
        self.sock.sendall(frame)

    def _recv_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                raise RhypedbError("connection closed mid-frame")
            self._buf += chunk
        result, self._buf = self._buf[:n], self._buf[n:]
        return result

    def _recv_frame(self) -> tuple:
        len_bytes = self._recv_exact(4)
        (body_len,) = struct.unpack(">I", len_bytes)
        body = self._recv_exact(body_len)
        req_id, kind = struct.unpack(">IB", body[:5])
        return req_id, kind, body[5:]

    def ping(self) -> None:
        rid = self._next_req_id()
        self._send_frame(rid, REQ_PING, b"")
        resp_id, kind, _payload = self._recv_frame()
        if resp_id != rid:
            raise RhypedbError(f"ping req_id mismatch: {resp_id} != {rid}")
        if kind != RESP_PONG:
            raise RhypedbError(f"expected RESP_PONG, got 0x{kind:02x}")

    def vector_batch(self, type_name: str, field_name: str, ids, vectors) -> dict:
        """Bulk-ingest precomputed vectors for one Vector field via the binary
        REQ_VECTOR_BATCH op.

        `ids` is a sequence of int object ids; `vectors` is an (N, D) float32
        array (numpy or sequence-of-sequences). One frame per call — callers
        chunk large loads. Payload:
        [type_len:u16][type][field_len:u16][field][count:u32] then count rows of
        [object_id:u64 BE][dim:u32 BE][f32 x dim little-endian].
        """
        import numpy as np

        rid = self._next_req_id()
        tb = type_name.encode("utf-8")
        fb = field_name.encode("utf-8")
        arr = np.ascontiguousarray(vectors, dtype="<f4")
        n, dim = arr.shape
        if len(ids) != n:
            raise RhypedbError(f"ids/vectors length mismatch: {len(ids)} != {n}")

        parts = [
            struct.pack(">H", len(tb)), tb,
            struct.pack(">H", len(fb)), fb,
            struct.pack(">I", n),
        ]
        row_hdr = struct.Struct(">QI")
        for i in range(n):
            parts.append(row_hdr.pack(int(ids[i]), dim))
            parts.append(arr[i].tobytes())
        payload = b"".join(parts)

        self._send_frame(rid, REQ_VECTOR_BATCH, payload)
        resp_id, kind, body = self._recv_frame()
        if resp_id != rid:
            raise RhypedbError(f"vector_batch req_id mismatch: {resp_id} != {rid}")
        if kind == RESP_DONE:
            return {"ok": True}
        if kind == RESP_ERROR:
            (msg_len,) = struct.unpack(">I", body[:4])
            return {"error": body[4:4 + msg_len].decode("utf-8", errors="replace")}
        raise RhypedbError(f"unexpected vector_batch response 0x{kind:02x}")

    def query(self, q: str) -> dict:
        """Execute a query. Returns a dict matching the HTTP JSON shape so the
        harness doesn't care which transport it's using."""
        rid = self._next_req_id()
        q_bytes = q.encode("utf-8")
        payload = struct.pack(">I", len(q_bytes)) + q_bytes
        self._send_frame(rid, REQ_QUERY, payload)

        resp_id, kind, body = self._recv_frame()
        if resp_id != rid:
            raise RhypedbError(f"query req_id mismatch: {resp_id} != {rid}")

        if kind == RESP_OBJECTS:
            return {"objects": _decode_objects(body)}
        if kind == RESP_SINGLE:
            obj, _ = _decode_object(body, 0)
            return {"object": obj}
        if kind == RESP_DONE:
            return {"ok": True}
        if kind == RESP_ERROR:
            (msg_len,) = struct.unpack(">I", body[:4])
            msg = body[4:4 + msg_len].decode("utf-8", errors="replace")
            return {"error": msg}
        raise RhypedbError(f"unknown response type 0x{kind:02x}")


def _decode_objects(data: bytes) -> list:
    (count,) = struct.unpack(">I", data[:4])
    pos = 4
    out = []
    for _ in range(count):
        obj, pos = _decode_object(data, pos)
        out.append(obj)
    return out


def _decode_object(data: bytes, pos: int) -> tuple:
    (type_len,) = struct.unpack(">H", data[pos:pos + 2])
    pos += 2
    type_name = data[pos:pos + type_len].decode("utf-8")
    pos += type_len
    (obj_id,) = struct.unpack(">Q", data[pos:pos + 8])
    pos += 8
    fields, pos = _decode_fields(data, pos)
    return {"type": type_name, "id": obj_id, "fields": fields}, pos


def _decode_fields(data: bytes, pos: int) -> tuple:
    (count,) = struct.unpack(">H", data[pos:pos + 2])
    pos += 2
    fields = {}
    for _ in range(count):
        (name_len,) = struct.unpack(">H", data[pos:pos + 2])
        pos += 2
        name = data[pos:pos + name_len].decode("utf-8")
        pos += name_len
        tag = data[pos]
        pos += 1
        value, pos = _decode_value(data, pos, tag)
        fields[name] = value
    return fields, pos


def _decode_value(data: bytes, pos: int, tag: int) -> tuple:
    if tag == TAG_NULL:
        return None, pos
    if tag == TAG_STRING:
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        pos += 4
        return data[pos:pos + n].decode("utf-8"), pos + n
    if tag == TAG_U32:
        (v,) = struct.unpack(">I", data[pos:pos + 4])
        return v, pos + 4
    if tag == TAG_U64:
        (v,) = struct.unpack(">Q", data[pos:pos + 8])
        return v, pos + 8
    if tag == TAG_I32:
        (v,) = struct.unpack(">i", data[pos:pos + 4])
        return v, pos + 4
    if tag == TAG_I64:
        (v,) = struct.unpack(">q", data[pos:pos + 8])
        return v, pos + 8
    if tag == TAG_F32:
        (v,) = struct.unpack(">f", data[pos:pos + 4])
        return v, pos + 4
    if tag == TAG_F64:
        (v,) = struct.unpack(">d", data[pos:pos + 8])
        return v, pos + 8
    if tag == TAG_BOOL:
        return data[pos] != 0, pos + 1
    if tag == TAG_BYTES:
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        pos += 4
        return data[pos:pos + n], pos + n
    raise RhypedbError(f"unknown value tag {tag}")
