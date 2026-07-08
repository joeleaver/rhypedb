//! Data-plane authorization for rhypedb (managed-DB P4 — the "Firestore core").
//!
//! Two halves:
//! - **Identity** ([`jose`] + [`principal`]): the engine verifies a per-project jkbase-Auth EdDSA
//!   JWT **offline** (alg-pinned, against a public JWKS) and resolves it to a [`Principal`]. A
//!   missing/invalid token fails closed to the anonymous principal — never fail-open.
//! - **Rules** (added in P4b): a default-deny, relationship-aware security-rules program parsed from
//!   a project's `rules.rhype`, evaluated per op/type/object against the principal.
//!
//! Enforcement lives in `rhypedb-query`'s executor (which threads a [`Principal`] + an optional
//! rules program on its `ExecContext`); this crate is the pure, engine-independent authz logic so it
//! stays dependency-light and unit-testable in isolation. **Rules-off ⇒ the engine is byte-unchanged**
//! (the executor takes its current path), so every existing deployment is unaffected until a project
//! opts in.
//!
//! See `docs/managed-rhypedb-p4-design.md` (jkbase repo) for the full design + the P0-DBA-* invariants.

pub mod jose;
pub mod principal;

pub use jose::{verify, Claims, Jwk, Jwks, VerifiedToken, VerifyError, VerifyOptions, ALG_EDDSA};
pub use principal::{JwtSource, Principal, PrincipalSource};
