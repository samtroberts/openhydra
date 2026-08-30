//! M5 remote-rig control: a signed `REMOTE_SCOPE_SET` command that a control-capable swarm member
//! sends to an owner's rig to flip a model's share scope (e.g. Private↔Global) from another device.
//!
//! Authenticity + integrity already ride the libp2p Noise channel — the provider binds the presented
//! credential to the connection's authenticated peer id and checks [`CAP_CONTROL`] via
//! [`SwarmAuthorizer::authorize_control`](crate::swarms::SwarmAuthorizer::authorize_control). On top
//! of that, this command is **member-signed, domain-separated, and timestamped** for non-repudiation
//! (an auditable "this device asked for exactly this scope change") and an application-layer replay
//! window independent of the transport. It carries **no secret** — only the public credential + a
//! signature over public fields.

use crate::membership::MembershipCredential;
use serde::{Deserialize, Serialize};

/// Domain-separation header for the command preimage (bump on any layout change).
const CMD_DOMAIN: &str = "openhydra-remote-scope-set-v1";

/// Reject a command whose `issued_at_ms` is more than this far from the provider's clock (either
/// direction) — a belt-and-suspenders replay/skew bound on top of the transport's own per-session
/// freshness. Generous enough for real clock skew, tight enough that a captured command can't be
/// replayed hours later.
pub const REMOTE_CMD_MAX_SKEW_MS: u64 = 5 * 60 * 1000;

fn one() -> u32 {
    1
}

/// A signed request to set one model's share scope on a remote rig (M5).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteScopeSet {
    #[serde(default = "one")]
    pub schema_version: u32,
    /// The model (engine ref / clean handle) to re-scope.
    pub model_id: String,
    /// Requested scope: `"global" | "private" | "device"` (parsed provider-side).
    pub scope: String,
    /// Command creation time (unix ms) — the replay/skew window anchor.
    pub issued_at_ms: u64,
    /// The control credential (M3, [`CAP_CONTROL`]) proving the sender is an owner-authorised
    /// controller. The provider verifies it owner-authoritatively before acting.
    pub credential: MembershipCredential,
    /// base64url signature by the member's node key over [`Self::command_canonical_bytes`].
    #[serde(default)]
    pub command_sig: String,
}

impl RemoteScopeSet {
    /// The deterministic signing preimage. Domain-separated and **bound to the credential's swarm**
    /// (so a signed command can't be replayed against a different swarm the member also controls) and
    /// to `issued_at_ms` (so it can't be replayed later).
    pub fn command_canonical_bytes(&self) -> Vec<u8> {
        format!(
            "{CMD_DOMAIN}\nschema_version={}\nswarm_public_key={}\nmodel_id={}\nscope={}\nissued_at_ms={}",
            self.schema_version,
            self.credential.swarm_public_key,
            self.model_id,
            self.scope,
            self.issued_at_ms,
        )
        .into_bytes()
    }

    /// Build + sign a command with the sender's node keypair — which MUST be the credential's member
    /// key (the provider checks the member signature against `credential.member_public_key`).
    pub fn signed(
        model_id: impl Into<String>,
        scope: impl Into<String>,
        issued_at_ms: u64,
        credential: MembershipCredential,
        node_keypair: &libp2p::identity::Keypair,
    ) -> Result<Self, String> {
        let mut cmd = Self {
            schema_version: 1,
            model_id: model_id.into(),
            scope: scope.into(),
            issued_at_ms,
            credential,
            command_sig: String::new(),
        };
        let sig = node_keypair
            .sign(&cmd.command_canonical_bytes())
            .map_err(|e| format!("sign command: {e}"))?;
        cmd.command_sig = b64(&sig);
        Ok(cmd)
    }

    /// Verify the member signature over the preimage against the credential's `member_public_key`.
    /// Does **not** verify the credential itself — that's the SwarmAuthorizer's owner-authoritative
    /// job; this only proves the live sender holds the member key and authorised *this* command.
    pub fn verify_command_sig(&self) -> Result<(), String> {
        if self.command_sig.is_empty() {
            return Err("missing command signature".into());
        }
        let pk_bytes = hex::decode(&self.credential.member_public_key)
            .map_err(|e| format!("bad member key hex: {e}"))?;
        let ed_pk = libp2p::identity::ed25519::PublicKey::try_from_bytes(&pk_bytes)
            .map_err(|e| format!("bad member ed25519 key: {e}"))?;
        let sig = unb64(&self.command_sig)?;
        if ed_pk.verify(&self.command_canonical_bytes(), &sig) {
            Ok(())
        } else {
            Err("command signature does not verify against the credential's member key".into())
        }
    }

    /// `issued_at_ms` within [`REMOTE_CMD_MAX_SKEW_MS`] of `now_ms` (either direction).
    pub fn within_replay_window(&self, now_ms: u64) -> bool {
        now_ms.abs_diff(self.issued_at_ms) <= REMOTE_CMD_MAX_SKEW_MS
    }

    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|e| format!("decode remote-scope-set: {e}"))
    }
}

/// The provider's reply to a [`RemoteScopeSet`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RemoteScopeAck {
    /// Applied — the model's scope is now this (and, for `global`, consent was recorded).
    Applied { model_id: String, scope: String },
    /// Refused, with a human-readable reason (authz, unknown scope, stale command, no policy file).
    Refused(String),
}

impl RemoteScopeAck {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|e| format!("decode ack: {e}"))
    }
}

fn b64(d: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE.encode(d)
}
fn unb64(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE
        .decode(s)
        .map_err(|e| format!("base64: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::membership::{sign_credential, MembershipCredential, CAP_CONTROL, CAP_SERVE};

    fn control_cred(
        group: &libp2p::identity::Keypair,
        member_pk_hex: &str,
    ) -> MembershipCredential {
        sign_credential(
            MembershipCredential::new_unsigned(member_pk_hex, "oh", "Home rig", 1_000, 9_999_999_999_000)
                .granting(CAP_SERVE | CAP_CONTROL),
            group,
        )
        .unwrap()
    }
    fn pk_hex(kp: &libp2p::identity::Keypair) -> String {
        hex::encode(kp.public().try_into_ed25519().unwrap().to_bytes())
    }

    #[test]
    fn sign_then_verify_command_round_trips() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let cred = control_cred(&group, &pk_hex(&member));
        let cmd = RemoteScopeSet::signed("llama3.1:8b", "global", 5_000, cred, &member).unwrap();
        cmd.verify_command_sig().unwrap();
        assert!(cmd.within_replay_window(5_000));
        // Round-trips through the wire encoding.
        let back = RemoteScopeSet::decode(&cmd.encode()).unwrap();
        assert_eq!(back, cmd);
        back.verify_command_sig().unwrap();
    }

    #[test]
    fn a_command_signed_by_the_wrong_key_fails() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let imposter = libp2p::identity::Keypair::generate_ed25519();
        // Credential names `member`, but the command is signed by `imposter`.
        let cred = control_cred(&group, &pk_hex(&member));
        let cmd = RemoteScopeSet::signed("m", "global", 5_000, cred, &imposter).unwrap();
        assert!(cmd.verify_command_sig().is_err());
    }

    #[test]
    fn tampering_the_scope_after_signing_breaks_the_command_signature() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let cred = control_cred(&group, &pk_hex(&member));
        let mut cmd = RemoteScopeSet::signed("m", "private", 5_000, cred, &member).unwrap();
        cmd.scope = "global".into(); // escalate after signing
        assert!(cmd.verify_command_sig().is_err());
    }

    #[test]
    fn a_stale_command_is_outside_the_replay_window() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let cred = control_cred(&group, &pk_hex(&member));
        let cmd = RemoteScopeSet::signed("m", "global", 1_000, cred, &member).unwrap();
        assert!(!cmd.within_replay_window(1_000 + REMOTE_CMD_MAX_SKEW_MS + 1));
        assert!(cmd.within_replay_window(1_000 + REMOTE_CMD_MAX_SKEW_MS));
    }
}
