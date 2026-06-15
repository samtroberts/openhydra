//! R-DHT-6: persistent Kademlia routing table.
//!
//! BitTorrent's DHT survives churn partly because every node caches its routing
//! table to disk and reloads it on restart, so a fresh process rejoins the
//! network with hundreds of known-good contacts instead of re-bootstrapping from
//! a handful of well-known nodes. OpenHydra previously rebuilt its routing table
//! from the 3–4 bootstrap IPs on every start — a cold start that amplifies churn
//! (everyone leans on the relays) rather than absorbing it.
//!
//! This module snapshots `(PeerId, [Multiaddr])` contacts to a small JSON file
//! beside the identity key and reloads them at startup (fed back via
//! `kademlia.add_address`). It is intentionally best-effort: a missing,
//! truncated, or partially-corrupt cache degrades gracefully to "fewer warm
//! contacts", never an error — the bootstrap peers remain the source of truth.

use std::path::{Path, PathBuf};

use libp2p::{Multiaddr, PeerId};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// One persisted routing-table contact.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedNode {
    peer_id: String,
    addrs: Vec<String>,
}

/// The on-disk snapshot. Versioned so the format can evolve without misreading
/// an older file as the current shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedRoutingTable {
    version: u32,
    nodes: Vec<PersistedNode>,
}

const FORMAT_VERSION: u32 = 1;

/// Derive the routing-cache path from the identity-key path: a sibling
/// `routing_cache.json`. Keeping it next to the identity ties the cache to the
/// node's own data dir without introducing a new config knob.
pub fn cache_path_for(identity_path: &Path) -> PathBuf {
    match identity_path.parent() {
        Some(dir) => dir.join("routing_cache.json"),
        None => PathBuf::from("routing_cache.json"),
    }
}

/// Load persisted contacts as flat `(PeerId, Multiaddr)` pairs ready for
/// `kademlia.add_address`. Unparseable peer-ids / multiaddrs are skipped
/// individually; a missing or corrupt file yields an empty list.
pub fn load(path: &Path) -> Vec<(PeerId, Multiaddr)> {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            warn!(%e, ?path, "routing_cache: read failed; cold-starting routing table");
            return Vec::new();
        }
    };
    let table: PersistedRoutingTable = match serde_json::from_slice(&data) {
        Ok(t) => t,
        Err(e) => {
            warn!(%e, ?path, "routing_cache: parse failed; ignoring cache");
            return Vec::new();
        }
    };
    if table.version != FORMAT_VERSION {
        warn!(found = table.version, expected = FORMAT_VERSION, "routing_cache: version mismatch; ignoring cache");
        return Vec::new();
    }
    let mut out = Vec::new();
    for node in table.nodes {
        let peer_id = match node.peer_id.parse::<PeerId>() {
            Ok(p) => p,
            Err(e) => {
                debug!(%e, peer_id = %node.peer_id, "routing_cache: skipping invalid peer_id");
                continue;
            }
        };
        for addr_str in node.addrs {
            // Don't reload relay-circuit addresses: a `/p2p-circuit` contact is
            // only valid while that specific relay reservation is live, which it
            // won't be across a restart. Direct addresses are what's worth warming.
            if addr_str.contains("/p2p-circuit") {
                continue;
            }
            match addr_str.parse::<Multiaddr>() {
                Ok(addr) => out.push((peer_id, addr)),
                Err(e) => debug!(%e, addr = %addr_str, "routing_cache: skipping invalid multiaddr"),
            }
        }
    }
    out
}

/// Atomically persist the current routing-table contacts. Writes to a temp file
/// and renames over the target so a crash mid-write can't truncate the cache.
pub fn save(path: &Path, entries: &[(PeerId, Vec<Multiaddr>)]) -> std::io::Result<()> {
    let nodes: Vec<PersistedNode> = entries
        .iter()
        .filter(|(_, addrs)| !addrs.is_empty())
        .map(|(peer_id, addrs)| PersistedNode {
            peer_id: peer_id.to_base58(),
            addrs: addrs
                .iter()
                // Persist only direct addresses (see `load`): circuit addrs are
                // ephemeral and would just churn the file.
                .filter(|a| !a.to_string().contains("/p2p-circuit"))
                .map(|a| a.to_string())
                .collect(),
        })
        .filter(|n| !n.addrs.is_empty())
        .collect();
    let table = PersistedRoutingTable {
        version: FORMAT_VERSION,
        nodes,
    };
    let bytes = serde_json::to_vec_pretty(&table)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_path_sibling_of_identity() {
        let p = cache_path_for(Path::new("/opt/openhydra/.identity.key"));
        assert_eq!(p, PathBuf::from("/opt/openhydra/routing_cache.json"));
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing_cache.json");
        let p1 = PeerId::random();
        let p2 = PeerId::random();
        let a1: Multiaddr = "/ip4/45.79.190.172/tcp/4001".parse().unwrap();
        let a2: Multiaddr = "/ip6/2a03:4000:41:ed1::1/udp/4001/quic-v1".parse().unwrap();
        let entries = vec![(p1, vec![a1.clone(), a2.clone()]), (p2, vec![a1.clone()])];
        save(&path, &entries).unwrap();

        let loaded = load(&path);
        // p1 contributes 2 addrs, p2 contributes 1 → 3 pairs.
        assert_eq!(loaded.len(), 3);
        assert!(loaded.contains(&(p1, a1.clone())));
        assert!(loaded.contains(&(p1, a2)));
        assert!(loaded.contains(&(p2, a1)));
    }

    #[test]
    fn test_load_missing_file_is_empty() {
        let loaded = load(Path::new("/nonexistent/routing_cache.json"));
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_load_corrupt_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing_cache.json");
        std::fs::write(&path, b"{ not valid json ]").unwrap();
        assert!(load(&path).is_empty());
    }

    #[test]
    fn test_circuit_addrs_not_persisted_or_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing_cache.json");
        let p = PeerId::random();
        let direct: Multiaddr = "/ip4/8.8.8.8/tcp/4001".parse().unwrap();
        let circuit: Multiaddr = format!(
            "/ip4/45.79.190.172/tcp/4001/p2p/{}/p2p-circuit",
            PeerId::random().to_base58()
        )
        .parse()
        .unwrap();
        save(&path, &[(p, vec![direct.clone(), circuit])]).unwrap();
        let loaded = load(&path);
        assert_eq!(loaded, vec![(p, direct)]);
    }
}
