//! Load OpenHydra Ed25519 identity keys and derive libp2p PeerId.
//!
//! OpenHydra stores keys as JSON at `~/.openhydra/identity.key`:
//! ```json
//! {"public_key": "<64 hex chars>", "private_key": "<64 hex chars>", "peer_id": "<16 hex chars>"}
//! ```
//!
//! The 32-byte raw Ed25519 keys are directly compatible with
//! `libp2p::identity::ed25519::SecretKey`.

use std::path::Path;

use libp2p::identity;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use zeroize::Zeroize;

/// JSON format of `~/.openhydra/identity.key`.
#[derive(Deserialize)]
#[allow(dead_code)]
struct IdentityFile {
    public_key: String,
    private_key: String,
    peer_id: String,
}

/// Loaded identity with both libp2p and OpenHydra peer IDs.
pub struct Identity {
    pub keypair: identity::Keypair,
    pub libp2p_peer_id: libp2p::PeerId,
    /// The original OpenHydra peer_id (SHA256(pubkey)[:16] hex).
    pub openhydra_peer_id: String,
}

impl Identity {
    /// Load an existing identity from the OpenHydra JSON key file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // Warn if the key file has overly permissive permissions.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(path)?.permissions().mode();
            if mode & 0o077 != 0 {
                tracing::warn!(
                    path = %path.display(),
                    mode = format!("{:o}", mode),
                    "identity key file has overly permissive permissions (expected 0600)"
                );
            }
        }

        let mut data = std::fs::read_to_string(path)?;
        let mut file: IdentityFile = serde_json::from_str(&data)?;

        let mut private_bytes = hex::decode(&file.private_key)?;
        if private_bytes.len() != 32 {
            let n = private_bytes.len();
            private_bytes.zeroize();
            file.private_key.zeroize();
            data.zeroize();
            return Err(format!("expected 32-byte private key, got {n} bytes").into());
        }

        // Build libp2p Ed25519 keypair from the raw 32-byte secret. `try_from_bytes`
        // takes the buffer by `&mut` and zeroizes it on success.
        let mut key_bytes: [u8; 32] = private_bytes.as_slice().try_into()
            .map_err(|_| "private key must be exactly 32 bytes")?;
        let secret = identity::ed25519::SecretKey::try_from_bytes(&mut key_bytes)?;
        let ed25519_keypair = identity::ed25519::Keypair::from(secret);
        let keypair = identity::Keypair::from(ed25519_keypair);
        let libp2p_peer_id = keypair.public().to_peer_id();

        // PQC0.2: scrub every transient copy of the secret (libp2p already zeroized
        // `key_bytes`). The `SigningKey` inside `keypair` keeps the live secret and
        // zeroizes itself on drop.
        private_bytes.zeroize();
        file.private_key.zeroize();
        data.zeroize();

        Ok(Identity {
            keypair,
            libp2p_peer_id,
            openhydra_peer_id: file.peer_id,
        })
    }

    /// Generate a new identity and save it to the given path.
    pub fn generate(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let keypair = identity::Keypair::generate_ed25519();

        // Extract raw 32-byte keys for the JSON file.
        let ed25519_kp = keypair
            .clone()
            .try_into_ed25519()
            .map_err(|_| "not ed25519")?;
        let secret_obj = ed25519_kp.secret();
        let secret_bytes = secret_obj.as_ref();
        let public_bytes = ed25519_kp.public().to_bytes();

        // OpenHydra peer_id = SHA256(public_key_bytes)[:16] as hex.
        let openhydra_peer_id = {
            let mut hasher = Sha256::new();
            hasher.update(&public_bytes);
            let hash = hasher.finalize();
            hex::encode(&hash[..8]) // 8 bytes = 16 hex chars
        };

        // Write JSON file.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::json!({
            "public_key": hex::encode(&public_bytes),
            "private_key": hex::encode(secret_bytes),
            "peer_id": &openhydra_peer_id,
        });
        // PQC0.2: zeroize the serialized secret-bearing string after it's written. (The
        // hex secret inside the `json` Value is a transient residual covered by the
        // process-level core-dump/swap hardening applied at agent startup.)
        let mut serialized = serde_json::to_string_pretty(&json)?;
        std::fs::write(path, &serialized)?;
        serialized.zeroize();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        }

        let libp2p_peer_id = keypair.public().to_peer_id();
        Ok(Identity {
            keypair,
            libp2p_peer_id,
            openhydra_peer_id,
        })
    }

    /// Load from path if it exists, otherwise generate and save.
    pub fn load_or_create(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let existed = path.exists();
        let identity = if existed {
            Self::load(path)?
        } else {
            Self::generate(path)?
        };
        tracing::info!(
            path = %path.display(),
            peer_id = %identity.libp2p_peer_id,
            openhydra_id = %identity.openhydra_peer_id,
            created = !existed,
            "identity loaded"
        );
        Ok(identity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_generate_and_load_roundtrip() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        // Remove the file so generate() creates it.
        std::fs::remove_file(path).ok();

        let gen = Identity::generate(path).unwrap();
        assert!(!gen.openhydra_peer_id.is_empty());
        assert_eq!(gen.openhydra_peer_id.len(), 16);

        let loaded = Identity::load(path).unwrap();
        assert_eq!(gen.openhydra_peer_id, loaded.openhydra_peer_id);
        assert_eq!(gen.libp2p_peer_id, loaded.libp2p_peer_id);
    }

    #[test]
    fn test_load_existing_openhydra_key() {
        // Simulate an existing OpenHydra identity.key file.
        let tmp = NamedTempFile::new().unwrap();
        let json = r#"{
            "public_key": "fec7e292a1d53e7b07c633c3ac827ca1232dc1e2d381476970adf4531c67e92d",
            "private_key": "4553afd57d4c9d372fbdf2ee081a21eb14e0ce77e1b360061660b7c84d8d3e92",
            "peer_id": "919cafdbe635606c"
        }"#;
        std::fs::write(tmp.path(), json).unwrap();

        let id = Identity::load(tmp.path()).unwrap();
        assert_eq!(id.openhydra_peer_id, "919cafdbe635606c");
        // libp2p PeerId is different (multihash) but deterministic.
        assert!(!id.libp2p_peer_id.to_string().is_empty());
    }
}
