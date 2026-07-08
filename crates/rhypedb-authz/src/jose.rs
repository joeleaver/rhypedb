//! **Verify-only** EdDSA (Ed25519) JWT + JWKS (RFC 7517) core for the rhypedb data plane.
//!
//! This is the engine's half of jkbase-Auth (P3): the tenant's issuer (`auth.jkbase.app`) mints
//! per-project EdDSA JWTs and publishes a per-project JWKS; the engine holds only the **public**
//! keys and verifies offline. It is a faithful port of the issuer's crypto core
//! (`jkbase-control/src/jose.rs`) with the signing side removed — a token minted there verifies here
//! byte-for-byte (same `{alg:EdDSA,typ:JWT,kid}` header, same `{iss,sub,aud,iat,exp,jti,claims}`
//! claims, same OKP/Ed25519 JWK). **Keep the wire formats in lockstep with that file.**
//!
//! Invariants (P0-DBA-2/3/8, mirroring the issuer's P0-AUTH-5/8):
//! - **PIN `alg = EdDSA`** and `kty = OKP / crv = Ed25519`; the key is selected from the *trusted*
//!   JWKS by `kid`, and the attacker-supplied header `alg` is used ONLY to reject anything that
//!   isn't `EdDSA` — never to pick the verification scheme. Closes `alg:none` + HS256-substitution.
//! - **Fail-closed**: every failure path returns [`VerifyError`] — malformed, unknown `kid`, bad key,
//!   bad signature, expired, not-yet-valid, `iss`/`aud` mismatch.
//! - The engine never holds a private key: no `sign` in the production surface (a test-only
//!   [`SigningKeypair`] exists under `cfg(test)`/`feature="test-support"` so downstream tests can
//!   mint tokens). The private seed lives host-side in jkbase and never enters a VM.
//!
//! Clock-free by construction: [`verify`] takes `now` in [`VerifyOptions`], so the whole surface is
//! unit-testable without a wall clock (the server passes `SystemTime::now()`).

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};

/// The only JWS algorithm minted or accepted. Pinned at verify time (P0-DBA-2).
pub const ALG_EDDSA: &str = "EdDSA";

/// The JOSE header. `kid` selects the verifying key; `alg` must be `EdDSA`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Header {
    alg: String,
    #[serde(default)]
    typ: String,
    kid: String,
}

/// Registered + custom claims, matching the jkbase issuer exactly. The registered fields are set by
/// the issuer and cannot be forged by a tenant; tenant-supplied custom claims live NESTED under
/// [`Claims::claims`] so they can never shadow `iss`/`aud`/`exp`/`sub`. The rules layer reads `sub`
/// (= `request.auth.uid`) and `claims.*` (= `request.auth.claims.*`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Claims {
    pub iss: String,
    pub sub: String,
    pub aud: String,
    pub iat: u64,
    pub exp: u64,
    pub jti: String,
    /// Opaque tenant-supplied custom claims (namespaced so they can't shadow registered claims).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claims: Option<serde_json::Value>,
}

/// A single public JWK (RFC 7517, OKP/Ed25519). Deserialized from the project's baked JWKS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Jwk {
    pub kty: String,
    pub crv: String,
    /// Serialized as the reserved JWK member name `use`.
    #[serde(rename = "use", default)]
    pub use_: String,
    #[serde(default)]
    pub alg: String,
    pub kid: String,
    /// base64url(no-pad) of the 32-byte Ed25519 public key.
    pub x: String,
}

impl Jwk {
    /// Decode `x` into a strict Ed25519 verifying key (validates the point) and enforce the
    /// OKP/Ed25519 key type, so a JWK of the wrong family can't be coerced into an Ed25519 verify.
    fn verifying_key(&self) -> Result<VerifyingKey, VerifyError> {
        if self.kty != "OKP" || self.crv != "Ed25519" {
            return Err(VerifyError::BadKey);
        }
        let raw = B64
            .decode(self.x.as_bytes())
            .map_err(|_| VerifyError::BadKey)?;
        let bytes: [u8; 32] = raw.try_into().map_err(|_| VerifyError::BadKey)?;
        VerifyingKey::from_bytes(&bytes).map_err(|_| VerifyError::BadKey)
    }
}

/// A project's published key set: the CURRENT key plus any rotating-out keys still inside the
/// overlap window (P3's P0-AUTH-4), so a token minted just before a rotation still verifies.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Jwks {
    pub keys: Vec<Jwk>,
}

impl Jwks {
    pub fn new(keys: Vec<Jwk>) -> Self {
        Self { keys }
    }

    /// Parse a JWKS from the raw JSON the platform bakes into the DB VM (or an inline config value).
    pub fn from_json(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    fn find(&self, kid: &str) -> Option<&Jwk> {
        self.keys.iter().find(|k| k.kid == kid)
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// Verification policy. `iss`/`aud` are checked only when supplied (a JWKS-only verifier may not know
/// them); `leeway_secs` bounds clock skew for both `exp` and `iat`.
#[derive(Debug, Clone)]
pub struct VerifyOptions {
    pub now: u64,
    pub leeway_secs: u64,
    pub expected_iss: Option<String>,
    pub expected_aud: Option<String>,
}

impl VerifyOptions {
    /// A permissive-but-safe default: 60s skew, no iss/aud pinning (caller sets those when known).
    pub fn at(now: u64) -> Self {
        Self {
            now,
            leeway_secs: 60,
            expected_iss: None,
            expected_aud: None,
        }
    }
    pub fn expect_iss(mut self, iss: impl Into<String>) -> Self {
        self.expected_iss = Some(iss.into());
        self
    }
    pub fn expect_aud(mut self, aud: impl Into<String>) -> Self {
        self.expected_aud = Some(aud.into());
        self
    }
}

/// A validated token: the `kid` that signed it and its decoded claims.
#[derive(Debug, Clone)]
pub struct VerifiedToken {
    pub kid: String,
    pub claims: Claims,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VerifyError {
    /// Not three base64url segments, or a segment failed to decode/parse.
    #[error("malformed token")]
    Malformed,
    /// Header `alg` is not `EdDSA` (rejects `none` and alg-substitution) — P0-DBA-2.
    #[error("unsupported alg (only EdDSA accepted)")]
    UnsupportedAlg,
    /// No JWK in the project's JWKS matched the header `kid`.
    #[error("unknown key id")]
    UnknownKid,
    /// The JWK is not a usable OKP/Ed25519 public key.
    #[error("unusable verifying key")]
    BadKey,
    /// Signature did not verify against the selected key.
    #[error("bad signature")]
    BadSignature,
    /// `now > exp + leeway`.
    #[error("token expired")]
    Expired,
    /// `iat > now + leeway` (token minted implausibly in the future).
    #[error("token not yet valid")]
    NotYetValid,
    #[error("issuer mismatch")]
    IssuerMismatch,
    #[error("audience mismatch")]
    AudienceMismatch,
}

/// Verify a compact JWT against `jwks` under `opts`, **fail-closed** (P0-DBA-8). The algorithm is
/// pinned to EdDSA and the key is chosen from the trusted JWKS by `kid` — the token header's `alg`
/// is used ONLY to reject anything that isn't `EdDSA`, never to select the verification scheme
/// (P0-DBA-2). The signature is checked over the exact received `header.claims` bytes (no
/// re-encoding) with `verify_strict` (rejects the small-order/malleable edge cases).
pub fn verify(token: &str, jwks: &Jwks, opts: &VerifyOptions) -> Result<VerifiedToken, VerifyError> {
    let mut parts = token.split('.');
    let (header_b64, claims_b64, sig_b64) =
        match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some(h), Some(c), Some(s), None) if !h.is_empty() && !c.is_empty() => (h, c, s),
            _ => return Err(VerifyError::Malformed),
        };

    let header: Header = decode_json(header_b64).ok_or(VerifyError::Malformed)?;
    // Pin the algorithm BEFORE touching the key or signature (P0-DBA-2).
    if header.alg != ALG_EDDSA {
        return Err(VerifyError::UnsupportedAlg);
    }

    let jwk = jwks.find(&header.kid).ok_or(VerifyError::UnknownKid)?;
    let vk = jwk.verifying_key()?;

    let sig_bytes = B64
        .decode(sig_b64.as_bytes())
        .map_err(|_| VerifyError::Malformed)?;
    let sig = Signature::from_slice(&sig_bytes).map_err(|_| VerifyError::Malformed)?;

    let signing_input = format!("{header_b64}.{claims_b64}");
    vk.verify_strict(signing_input.as_bytes(), &sig)
        .map_err(|_| VerifyError::BadSignature)?;

    // Signature is good — now the claims can be trusted enough to parse and time/scope-check.
    let claims: Claims = decode_json(claims_b64).ok_or(VerifyError::Malformed)?;
    if opts.now > claims.exp.saturating_add(opts.leeway_secs) {
        return Err(VerifyError::Expired);
    }
    if claims.iat > opts.now.saturating_add(opts.leeway_secs) {
        return Err(VerifyError::NotYetValid);
    }
    if let Some(iss) = &opts.expected_iss
        && &claims.iss != iss
    {
        return Err(VerifyError::IssuerMismatch);
    }
    if let Some(aud) = &opts.expected_aud
        && &claims.aud != aud
    {
        return Err(VerifyError::AudienceMismatch);
    }

    Ok(VerifiedToken {
        kid: header.kid,
        claims,
    })
}

fn decode_json<T: for<'de> Deserialize<'de>>(seg: &str) -> Option<T> {
    let raw = B64.decode(seg.as_bytes()).ok()?;
    serde_json::from_slice(&raw).ok()
}

/// Test-only signer. The production engine NEVER signs (it has no private key); this exists so the
/// crate's own tests and downstream crates' tests (`feature = "test-support"`) can mint tokens that
/// exercise [`verify`]. It intentionally mirrors the issuer's `SigningKeypair` so a test token is
/// wire-identical to a real jkbase-Auth one.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey as DalekSigningKey};

    #[derive(Clone)]
    pub struct SigningKeypair {
        kid: String,
        seed: [u8; 32],
        public: [u8; 32],
    }

    impl SigningKeypair {
        pub fn from_seed(kid: impl Into<String>, seed: [u8; 32]) -> Self {
            let dalek = DalekSigningKey::from_bytes(&seed);
            let public = dalek.verifying_key().to_bytes();
            Self {
                kid: kid.into(),
                seed,
                public,
            }
        }

        pub fn kid(&self) -> &str {
            &self.kid
        }

        pub fn jwk(&self) -> Jwk {
            Jwk {
                kty: "OKP".into(),
                crv: "Ed25519".into(),
                use_: "sig".into(),
                alg: ALG_EDDSA.into(),
                kid: self.kid.clone(),
                x: B64.encode(self.public),
            }
        }

        pub fn sign(&self, claims: &Claims) -> String {
            let header = Header {
                alg: ALG_EDDSA.into(),
                typ: "JWT".into(),
                kid: self.kid.clone(),
            };
            let header_b64 = B64.encode(serde_json::to_vec(&header).unwrap());
            let claims_b64 = B64.encode(serde_json::to_vec(claims).unwrap());
            let signing_input = format!("{header_b64}.{claims_b64}");
            let dalek = DalekSigningKey::from_bytes(&self.seed);
            let sig: Signature = dalek.sign(signing_input.as_bytes());
            format!("{signing_input}.{}", B64.encode(sig.to_bytes()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::SigningKeypair;
    use super::*;

    fn kp(kid: &str, seed_byte: u8) -> SigningKeypair {
        SigningKeypair::from_seed(kid, [seed_byte; 32])
    }

    fn claims(now: u64) -> Claims {
        Claims {
            iss: "https://auth.jkbase.app/v1/projects/proj".into(),
            sub: "user-42".into(),
            aud: "proj".into(),
            iat: now,
            exp: now + 3600,
            jti: "jti-abc".into(),
            claims: Some(serde_json::json!({ "role": "admin" })),
        }
    }

    #[test]
    fn sign_then_verify_roundtrips() {
        let k = kp("proj.1", 7);
        let now = 1_000_000;
        let tok = k.sign(&claims(now));
        let jwks = Jwks::new(vec![k.jwk()]);
        let v = verify(&tok, &jwks, &VerifyOptions::at(now + 10)).unwrap();
        assert_eq!(v.kid, "proj.1");
        assert_eq!(v.claims.sub, "user-42");
        assert_eq!(v.claims.claims.unwrap()["role"], "admin");
    }

    #[test]
    fn iss_and_aud_pinning() {
        let k = kp("proj.1", 7);
        let now = 1_000_000;
        let tok = k.sign(&claims(now));
        let jwks = Jwks::new(vec![k.jwk()]);
        let ok = VerifyOptions::at(now)
            .expect_iss("https://auth.jkbase.app/v1/projects/proj")
            .expect_aud("proj");
        assert!(verify(&tok, &jwks, &ok).is_ok());
        assert_eq!(
            verify(&tok, &jwks, &VerifyOptions::at(now).expect_iss("other")).unwrap_err(),
            VerifyError::IssuerMismatch
        );
        assert_eq!(
            verify(&tok, &jwks, &VerifyOptions::at(now).expect_aud("other")).unwrap_err(),
            VerifyError::AudienceMismatch
        );
    }

    #[test]
    fn rejects_alg_none() {
        let hdr = B64.encode(br#"{"alg":"none","typ":"JWT","kid":"proj.1"}"#);
        let now = 1_000_000u64;
        let payload = B64.encode(serde_json::to_vec(&claims(now)).unwrap());
        let forged = format!("{hdr}.{payload}.");
        let jwks = Jwks::new(vec![kp("proj.1", 7).jwk()]);
        assert_eq!(
            verify(&forged, &jwks, &VerifyOptions::at(now)).unwrap_err(),
            VerifyError::UnsupportedAlg
        );
    }

    #[test]
    fn rejects_alg_substitution_hs256() {
        // alg:HS256 with the public key used as an HMAC secret — the classic confusion attack. We
        // must reject at the alg pin, never treat the token as HMAC.
        let hdr = B64.encode(br#"{"alg":"HS256","typ":"JWT","kid":"proj.1"}"#);
        let now = 1_000_000u64;
        let payload = B64.encode(serde_json::to_vec(&claims(now)).unwrap());
        let forged = format!("{hdr}.{payload}.AAAA");
        let jwks = Jwks::new(vec![kp("proj.1", 7).jwk()]);
        assert_eq!(
            verify(&forged, &jwks, &VerifyOptions::at(now)).unwrap_err(),
            VerifyError::UnsupportedAlg
        );
    }

    #[test]
    fn rejects_tampered_payload() {
        let k = kp("proj.1", 7);
        let now = 1_000_000;
        let tok = k.sign(&claims(now));
        let mut parts: Vec<&str> = tok.split('.').collect();
        let mut evil = claims(now);
        evil.sub = "root".into();
        let evil_b64 = B64.encode(serde_json::to_vec(&evil).unwrap());
        parts[1] = &evil_b64;
        let tampered = parts.join(".");
        let jwks = Jwks::new(vec![k.jwk()]);
        assert_eq!(
            verify(&tampered, &jwks, &VerifyOptions::at(now)).unwrap_err(),
            VerifyError::BadSignature
        );
    }

    #[test]
    fn rejects_unknown_kid_and_cross_project_key() {
        let signer = kp("projA.1", 7);
        let now = 1_000_000;
        let tok = signer.sign(&claims(now));
        // JWKS holds a DIFFERENT project's key under a different kid → unknown kid, fail-closed.
        let other = kp("projB.1", 9);
        let jwks = Jwks::new(vec![other.jwk()]);
        assert_eq!(
            verify(&tok, &jwks, &VerifyOptions::at(now)).unwrap_err(),
            VerifyError::UnknownKid
        );
        // Same kid but the wrong key material → bad signature (a forged JWKS entry can't validate).
        let impostor = Jwk {
            kid: "projA.1".into(),
            ..kp("projB.1", 9).jwk()
        };
        let jwks2 = Jwks::new(vec![impostor]);
        assert_eq!(
            verify(&tok, &jwks2, &VerifyOptions::at(now)).unwrap_err(),
            VerifyError::BadSignature
        );
    }

    #[test]
    fn rejects_expired_and_future() {
        let k = kp("proj.1", 7);
        let now = 1_000_000;
        let tok = k.sign(&claims(now)); // exp = now+3600
        let jwks = Jwks::new(vec![k.jwk()]);
        assert_eq!(
            verify(&tok, &jwks, &VerifyOptions::at(now + 3600 + 61)).unwrap_err(),
            VerifyError::Expired
        );
        assert_eq!(
            verify(&tok, &jwks, &VerifyOptions::at(now - 3600)).unwrap_err(),
            VerifyError::NotYetValid
        );
        assert!(verify(&tok, &jwks, &VerifyOptions::at(now + 3600 + 30)).is_ok());
    }

    #[test]
    fn rotation_window_old_kid_still_verifies() {
        let prev = kp("proj.1", 7);
        let cur = kp("proj.2", 11);
        let now = 1_000_000;
        let tok = prev.sign(&claims(now));
        let jwks = Jwks::new(vec![cur.jwk(), prev.jwk()]); // overlap window
        assert!(verify(&tok, &jwks, &VerifyOptions::at(now)).is_ok());
        // After the window closes (prev dropped) the same token no longer verifies.
        let closed = Jwks::new(vec![cur.jwk()]);
        assert_eq!(
            verify(&tok, &closed, &VerifyOptions::at(now)).unwrap_err(),
            VerifyError::UnknownKid
        );
    }

    #[test]
    fn malformed_shapes_rejected() {
        let jwks = Jwks::new(vec![kp("proj.1", 7).jwk()]);
        for bad in ["", "a.b", "a.b.c.d", "...", "notb64.notb64.notb64"] {
            assert!(verify(bad, &jwks, &VerifyOptions::at(1)).is_err());
        }
    }

    #[test]
    fn accepts_a_real_jkbase_issuer_token() {
        // Golden vector minted by the ACTUAL jkbase-Auth issuer (jkbase repo,
        // jkbase-control/src/jose.rs) — locks cross-implementation format compatibility so a token
        // from auth.jkbase.app verifies in the engine byte-for-byte. Ed25519 signing is
        // deterministic, so this vector is stable (seed=[7;32], kid=goldenproj.1, iat=1_700_000_000).
        const TOKEN: &str = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCIsImtpZCI6ImdvbGRlbnByb2ouMSJ9.eyJpc3MiOiJodHRwczovL2F1dGguamtiYXNlLmFwcC92MS9wcm9qZWN0cy9wcm9qIiwic3ViIjoidXNlci00MiIsImF1ZCI6InByb2oiLCJpYXQiOjE3MDAwMDAwMDAsImV4cCI6MTcwMDAwMzYwMCwianRpIjoianRpLWFiYyIsImNsYWltcyI6eyJyb2xlIjoiYWRtaW4ifX0.tH93W-TeRTOa_0VPZslaplxwtJgH36rd6fq_3-wHpklaAFzcT3H9utpKUMq5dQAGAxl8YCPwBSuu43aO4PNmAw";
        const JWKS: &str = r#"{"keys":[{"kty":"OKP","crv":"Ed25519","use":"sig","alg":"EdDSA","kid":"goldenproj.1","x":"6kpsY-KcUgq-9VB7Ey7F-ZVHdq6-vnuSQh7qaRRG0iw"}]}"#;

        let jwks = Jwks::from_json(JWKS.as_bytes()).unwrap();
        let now = 1_700_000_100; // within [iat, iat+3600]
        // Full verify including the issuer/audience pins the engine will use.
        let opts = VerifyOptions::at(now)
            .expect_iss("https://auth.jkbase.app/v1/projects/proj")
            .expect_aud("proj");
        let v = verify(TOKEN, &jwks, &opts).expect("a real jkbase token must verify in the engine");
        assert_eq!(v.kid, "goldenproj.1");
        assert_eq!(v.claims.sub, "user-42");
        assert_eq!(v.claims.claims.as_ref().unwrap()["role"], "admin");

        // A one-char corruption of the real issuer's token fails closed.
        let tampered = format!("{}X", &TOKEN[..TOKEN.len() - 1]);
        assert!(verify(&tampered, &jwks, &VerifyOptions::at(now)).is_err());
        // And it is expired past its window.
        assert_eq!(
            verify(TOKEN, &jwks, &VerifyOptions::at(1_700_003_600 + 3600)).unwrap_err(),
            VerifyError::Expired
        );
    }

    #[test]
    fn jwks_parses_from_platform_json() {
        // The exact RFC 7517 shape jkbase's `GET …/jwks.json` emits — must round-trip in.
        let k = kp("proj.7", 3);
        let doc = serde_json::to_vec(&Jwks::new(vec![k.jwk()])).unwrap();
        let parsed = Jwks::from_json(&doc).unwrap();
        assert_eq!(parsed.keys.len(), 1);
        assert_eq!(parsed.keys[0].kid, "proj.7");
        let now = 1_000_000;
        let tok = k.sign(&claims(now));
        assert!(verify(&tok, &parsed, &VerifyOptions::at(now)).is_ok());
    }
}
