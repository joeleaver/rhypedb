//! Shared TCP dialer: connect a `net.Socket` with TCP_NODELAY and an optional
//! connect deadline. Used by both the request/response client and the dedicated
//! subscription connection.

import { connect as tcpConnect, type Socket } from "node:net";

import { RhypedbError } from "./errors.ts";

export function dialSocket(host: string, port: number, connectTimeoutMs?: number): Promise<Socket> {
  return new Promise<Socket>((resolve, reject) => {
    const socket = tcpConnect({ host, port });
    socket.setNoDelay(true);
    let settled = false;
    const timer =
      connectTimeoutMs !== undefined
        ? setTimeout(() => {
            if (settled) return;
            settled = true;
            socket.destroy();
            reject(new RhypedbError("connect", "connect timed out"));
          }, connectTimeoutMs)
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
      resolve(socket);
    });
  });
}
