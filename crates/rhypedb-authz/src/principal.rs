//! The verified end-user identity the rules layer authorizes against, and the [`PrincipalSource`]
//! seam that decouples *how identity arrives* from *how it's enforced*.
//!
//! At P4 the only source is [`JwtSource`]: the engine verifies a jkbase-Auth EdDSA JWT itself
//! (offline, alg-pinned) against the project's public JWKS. The seam exists so a future
//! gateway-injected-HMAC-header source (the parent design's P0-DB-5) can be added **without touching
//! the rules engine or the executor gates** — the executor only ever sees a [`Principal`].
//!
//! Every failure resolves to the **anonymous** principal (`authenticated = false`), never an error
//! that fails open: under a rules program the anonymous principal is subject to default-deny, so a
//! missing/expired/forged token simply can't read or write anything a rule doesn't explicitly open.

use crate::jose::{self, Jwks, VerifiedToken, VerifyOptions};

/// A verified (or anonymous) end-user identity. The rules language maps onto exactly this surface:
/// `request.auth` ⇔ [`Principal::authenticated`], `request.auth.uid` ⇔ [`Principal::uid`],
/// `request.auth.claims.*` ⇔ [`Principal::claim`].
#[derive(Debug, Clone, PartialEq)]
pub struct Principal {
    /// True iff a token verified. The anonymous principal has this `false` and `uid = None`.
    authenticated: bool,
    /// The token `sub` — the tenant-chosen end-user id. `None` when anonymous.
    uid: Option<String>,
    /// The token's nested custom claims object (`claims` in the JWT). `Null` when anonymous or when
    /// the issuer supplied none. Namespaced by the issuer so it can never shadow `iss`/`aud`/`sub`.
    claims: serde_json::Value,
}

impl Principal {
    /// The unauthenticated identity: no token, no uid, no claims. First-class — rules evaluate
    /// against it (`request.auth == null` is true), so absence of a token is default-denied, not
    /// fail-open.
    pub fn anonymous() -> Self {
        Self {
            authenticated: false,
            uid: None,
            claims: serde_json::Value::Null,
        }
    }

    /// Build from a verified token: `uid = sub`, `claims = the nested custom-claims object`.
    pub fn from_verified(v: VerifiedToken) -> Self {
        Self {
            authenticated: true,
            uid: Some(v.claims.sub),
            claims: v.claims.claims.unwrap_or(serde_json::Value::Null),
        }
    }

    pub fn is_authenticated(&self) -> bool {
        self.authenticated
    }

    pub fn uid(&self) -> Option<&str> {
        self.uid.as_deref()
    }

    /// The whole custom-claims object (`Null` when absent).
    pub fn claims(&self) -> &serde_json::Value {
        &self.claims
    }

    /// A single custom claim by key (`request.auth.claims.<key>`). `None` if unauthenticated, the
    /// claims aren't an object, or the key is absent.
    pub fn claim(&self, key: &str) -> Option<&serde_json::Value> {
        self.claims.get(key)
    }

    /// Test-only: build an authenticated principal directly. Production principals come ONLY from a
    /// verified token ([`Principal::from_verified`]) — this constructor is unavailable outside test
    /// builds, which is what makes "a principal exists ⇒ a token verified" a type-level guarantee.
    #[cfg(any(test, feature = "test-support"))]
    pub fn new_authenticated(uid: impl Into<String>, claims: serde_json::Value) -> Self {
        Self {
            authenticated: true,
            uid: Some(uid.into()),
            claims,
        }
    }
}

/// Resolves a request's bearer token into a [`Principal`]. Isolates the trust seam (DECISION 2):
/// implementors decide how identity is authenticated; the enforcement layer is oblivious.
pub trait PrincipalSource: Send + Sync {
    /// Resolve the optional bearer token (the raw `Authorization: Bearer …` value, or the
    /// `REQ_AUTH` frame payload) into a principal. MUST fail closed to [`Principal::anonymous`].
    fn principal(&self, bearer: Option<&str>) -> Principal;
}

/// The P4 source: verify a jkbase-Auth EdDSA JWT offline against the project's public JWKS.
///
/// Holds the project's key set plus an optional issuer/audience pin (defense-in-depth on top of the
/// per-project `kid` scoping) and the allowed clock skew. Clock-free at the core
/// ([`JwtSource::principal_at`]); the [`PrincipalSource`] impl stamps `SystemTime::now()`.
#[derive(Debug, Clone)]
pub struct JwtSource {
    jwks: Jwks,
    leeway_secs: u64,
    expected_iss: Option<String>,
    expected_aud: Option<String>,
}

impl JwtSource {
    pub fn new(jwks: Jwks) -> Self {
        Self {
            jwks,
            leeway_secs: 60,
            expected_iss: None,
            expected_aud: None,
        }
    }

    pub fn with_leeway(mut self, secs: u64) -> Self {
        self.leeway_secs = secs;
        self
    }
    pub fn expect_iss(mut self, iss: impl Into<String>) -> Self {
        self.expected_iss = Some(iss.into());
        self
    }
    pub fn expect_aud(mut self, aud: impl Into<String>) -> Self {
        self.expected_aud = Some(aud.into());
        self
    }

    pub fn jwks(&self) -> &Jwks {
        &self.jwks
    }

    /// Clock-explicit resolution — the testable core. Any verify failure (or absent token) →
    /// anonymous.
    pub fn principal_at(&self, bearer: Option<&str>, now: u64) -> Principal {
        let Some(token) = bearer else {
            return Principal::anonymous();
        };
        let mut opts = VerifyOptions::at(now);
        opts.leeway_secs = self.leeway_secs;
        opts.expected_iss = self.expected_iss.clone();
        opts.expected_aud = self.expected_aud.clone();
        match jose::verify(token, &self.jwks, &opts) {
            Ok(v) => Principal::from_verified(v),
            Err(_) => Principal::anonymous(),
        }
    }
}

impl PrincipalSource for JwtSource {
    fn principal(&self, bearer: Option<&str>) -> Principal {
        self.principal_at(bearer, unix_now())
    }
}

/// Seconds since the Unix epoch, saturating at 0 if the clock is somehow before it.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jose::test_support::SigningKeypair;
    use crate::jose::Claims;

    fn mint(kid: &str, seed: u8, now: u64, sub: &str, claims: serde_json::Value) -> String {
        SigningKeypair::from_seed(kid, [seed; 32]).sign(&Claims {
            iss: "https://auth.jkbase.app/v1/projects/proj".into(),
            sub: sub.into(),
            aud: "proj".into(),
            iat: now,
            exp: now + 3600,
            jti: "j".into(),
            claims: Some(claims),
        })
    }

    #[test]
    fn anonymous_when_no_token() {
        let src = JwtSource::new(Jwks::new(vec![]));
        let p = src.principal_at(None, 1_000_000);
        assert!(!p.is_authenticated());
        assert_eq!(p.uid(), None);
        assert_eq!(p.claims(), &serde_json::Value::Null);
    }

    #[test]
    fn valid_token_yields_authenticated_principal() {
        let signer = SigningKeypair::from_seed("proj.1", [7; 32]);
        let src = JwtSource::new(Jwks::new(vec![signer.jwk()]));
        let now = 1_000_000;
        let tok = mint("proj.1", 7, now, "user-42", serde_json::json!({ "role": "admin" }));
        let p = src.principal_at(Some(&tok), now + 5);
        assert!(p.is_authenticated());
        assert_eq!(p.uid(), Some("user-42"));
        assert_eq!(p.claim("role").unwrap(), "admin");
        assert_eq!(p.claim("missing"), None);
    }

    #[test]
    fn invalid_or_expired_token_falls_to_anonymous() {
        let signer = SigningKeypair::from_seed("proj.1", [7; 32]);
        let src = JwtSource::new(Jwks::new(vec![signer.jwk()]));
        let now = 1_000_000;
        // Garbage token → anonymous, not an error.
        assert!(!src.principal_at(Some("not.a.jwt"), now).is_authenticated());
        // Expired token → anonymous (fail-closed to default-deny under rules).
        let tok = mint("proj.1", 7, now, "u", serde_json::Value::Null);
        assert!(
            !src
                .principal_at(Some(&tok), now + 3600 + 120)
                .is_authenticated()
        );
    }

    #[test]
    fn wrong_project_key_falls_to_anonymous() {
        // Engine holds project B's key; a token minted for project A must NOT authenticate.
        let projb = SigningKeypair::from_seed("projB.1", [9; 32]);
        let src = JwtSource::new(Jwks::new(vec![projb.jwk()]));
        let now = 1_000_000;
        let tok_a = mint("projA.1", 7, now, "spoofer", serde_json::Value::Null);
        assert!(!src.principal_at(Some(&tok_a), now).is_authenticated());
    }

    #[test]
    fn iss_aud_pin_enforced_via_source() {
        let signer = SigningKeypair::from_seed("proj.1", [7; 32]);
        let now = 1_000_000;
        let tok = mint("proj.1", 7, now, "u", serde_json::Value::Null);
        // Correct pins → authenticated.
        let ok = JwtSource::new(Jwks::new(vec![signer.jwk()]))
            .expect_iss("https://auth.jkbase.app/v1/projects/proj")
            .expect_aud("proj");
        assert!(ok.principal_at(Some(&tok), now).is_authenticated());
        // Wrong audience pin → anonymous.
        let bad = JwtSource::new(Jwks::new(vec![signer.jwk()])).expect_aud("other-project");
        assert!(!bad.principal_at(Some(&tok), now).is_authenticated());
    }
}
