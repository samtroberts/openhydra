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
/// The OpenHydra peer id of an Ed25519 public key: the first 8 bytes of its SHA-256, as 16 hex
/// characters. Always derived from the key, never taken on trust: reputation, receipts, vouchers
/// and attestations are keyed by it, so a peer must not be able to claim another's.
pub fn openhydra_peer_id(public_key: &[u8]) -> String {
    hex::encode(&Sha256::digest(public_key)[..8])
}

/// The 128-bit id a peer's standing is kept under: the first 16 bytes of its public key's SHA-256
/// (32 hex characters, beginning with its [`openhydra_peer_id`]). The 64-bit peer id is a display
/// and routing handle; two keys sharing one is a ~2^32 search, so anything a peer earns or is owed —
/// reputation, verified authenticity, attestation, payment — is keyed by this instead (~2^64).
pub fn openhydra_state_id(public_key: &[u8]) -> String {
    hex::encode(&Sha256::digest(public_key)[..16])
}

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

        // The id is derived from the key; the file's copy is only a convenience for people reading
        // it. One that disagrees (edited, or from a different derivation) is ignored, since every
        // other node derives the id from the key and would reject records carrying the file's.
        let public = keypair
            .public()
            .try_into_ed25519()
            .map_err(|_| "identity key is not ed25519")?
            .to_bytes();
        let openhydra_peer_id = openhydra_peer_id(&public);
        if file.peer_id != openhydra_peer_id {
            tracing::warn!(
                path = %path.display(),
                file = %file.peer_id,
                derived = %openhydra_peer_id,
                "identity file's peer_id doesn't match its key; using the one derived from the key"
            );
        }
        Ok(Identity {
            keypair,
            libp2p_peer_id,
            openhydra_peer_id,
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

        let openhydra_peer_id = openhydra_peer_id(&public_bytes);

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
        let written = write_new_key_file(path, serialized.as_bytes(), &openhydra_peer_id);
        serialized.zeroize();
        written?;

        let libp2p_peer_id = keypair.public().to_peer_id();
        Ok(Identity {
            keypair,
            libp2p_peer_id,
            openhydra_peer_id,
        })
    }

    /// Load from path if it exists, otherwise generate and save.
    pub fn load_or_create(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut existed = path.exists();
        let identity = if existed {
            Self::load(path)?
        } else {
            match Self::generate(path) {
                Ok(identity) => identity,
                // Another process (the desktop's provider and gateway start together) created it
                // first: use that one, so both agree on the identity.
                Err(e) if e.downcast_ref::<std::io::Error>().is_some_and(|e| e.kind() == std::io::ErrorKind::AlreadyExists) => {
                    existed = true;
                    Self::load(path)?
                }
                Err(e) => return Err(e),
            }
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

/// Create the key file at `path` with `contents`, never replacing one that exists (an
/// `AlreadyExists` error then) and never visible half-written or readable by others: written to a
/// temporary file beside it (owner-only from the start, named by `tag`, unique to this key), then
/// linked into place.
fn write_new_key_file(path: &Path, contents: &[u8], tag: &str) -> std::io::Result<()> {
    use std::io::Write;
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("key");
    let tmp = dir.join(format!(".{name}.{tag}.tmp"));
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let result = opts.open(&tmp).and_then(|mut f| {
        f.write_all(contents)?;
        f.sync_all()?;
        std::fs::hard_link(&tmp, path)
    });
    let _ = std::fs::remove_file(&tmp);
    match result {
        // A filesystem without hard links: create it in place, still never over an existing one.
        Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => opts.open(path).and_then(|mut f| {
            f.write_all(contents)?;
            f.sync_all()
        }),
        r => r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn concurrent_creators_agree_on_one_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");
        let ids: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8).map(|_| s.spawn(|| Identity::load_or_create(&path).unwrap().libp2p_peer_id.to_string())).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(ids.windows(2).all(|w| w[0] == w[1]), "one key, whoever created it: {ids:?}");
        // Nothing left behind but the key, readable by its owner only.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        // Creating over an existing key is refused.
        let err = Identity::generate(&path).err().unwrap();
        assert_eq!(err.downcast_ref::<std::io::Error>().map(|e| e.kind()), Some(std::io::ErrorKind::AlreadyExists));
    }

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

    #[test]
    fn an_edited_peer_id_in_the_file_is_ignored() {
        let tmp = NamedTempFile::new().unwrap();
        let json = r#"{
            "public_key": "fec7e292a1d53e7b07c633c3ac827ca1232dc1e2d381476970adf4531c67e92d",
            "private_key": "4553afd57d4c9d372fbdf2ee081a21eb14e0ce77e1b360061660b7c84d8d3e92",
            "peer_id": "0123456789abcdef"
        }"#;
        std::fs::write(tmp.path(), json).unwrap();
        assert_eq!(Identity::load(tmp.path()).unwrap().openhydra_peer_id, "919cafdbe635606c");
    }
}
