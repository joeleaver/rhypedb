//! The client's error type. A single class with a discriminating `code` keeps
//! `catch` ergonomic while still letting callers branch on the cause.

export type ErrorCode =
  /** Failed to establish the TCP connection. */
  | "connect"
  /** Socket / framing I/O error during a request or response. */
  | "io"
  /** The server returned an error response (parse / validation / engine error). */
  | "server"
  /** The server replied with a frame kind that is not valid for this request. */
  | "unexpected_response"
  /** A query expected exactly one object but the server returned another shape. */
  | "unexpected_shape"
  /** A previous I/O error latched the connection closed; reconnect. */
  | "closed"
  /** A response payload could not be decoded into the requested type. */
  | "decode"
  /** A vector batch's encoded payload exceeds the protocol frame cap. */
  | "batch_too_large"
  /** A prepared statement was used with a different client than prepared it. */
  | "foreign_statement";

/** An error from the rhypedb client or server. Branch on [`code`](RhypedbError.code). */
export class RhypedbError extends Error {
  override readonly name = "RhypedbError";
  readonly code: ErrorCode;
  constructor(code: ErrorCode, message: string) {
    super(message);
    this.code = code;
  }
}
