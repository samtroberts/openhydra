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

/// 2 (27 Oct 2026): caches written before then may hold addresses a peer only claimed (identify
/// went straight into the routing table), so a version-1 file is ignored once and the node rejoins
/// through the bootstraps. Since then the table only holds addresses Kademlia dialled itself.
const FORMAT_VERSION: u32 = 2;
/// Addresses kept per contact, on save and on load.
const MAX_ADDRS_PER_NODE: usize = 4;

/// Derive the routing-cache path from the identity-key path: a sibling
/// `routing_cache.json`. Keeping it next to the identity ties the cache to the
/// node's own data dir without introducing a new config knob. A private
/// deployment (`private_scope`: which one, see [`private_scope`]) keeps its own,
/// `routing_cache.<scope>.json`, so it never dials the public peers a node
/// remembers from the public network, and going back keeps those (review
/// 2026-10-14, M1); and one deployment's never dials another's, which a single
/// `routing_cache.private.json` for them all did when a node moved between them.
pub fn cache_path_for(identity_path: &Path, private_scope: Option<&str>) -> PathBuf {
    let name = match private_scope {
        Some(scope) => format!("routing_cache.{scope}.json"),
        None => "routing_cache.json".to_string(),
    };
    match identity_path.parent() {
        Some(dir) => dir.join(name),
        None => PathBuf::from(name),
    }
}

/// Which private deployment a routing cache belongs to, safe as part of a file name: on a private
/// network (a network key), the first 16 hex digits of the key's fingerprint, never the key; else
/// the network id ([`crate::relay::network_id`]: a hash of its bootstraps, or its own name). The key
/// comes first because it is what decides who can be reached: two deployments can't share one, and
/// a node given one deployment's key can't use the other's peers whatever its bootstraps.
pub fn private_scope(network_id: &str, key_fingerprint: Option<&str>) -> String {
    match key_fingerprint {
        Some(fp) => {
            let short: String = fp.chars().filter(char::is_ascii_hexdigit).take(16).collect();
            format!("private-key-{short}")
        }
        // Network ids are hex or a checked name (letters, digits, `.`, `_`, `-`) already; anything
        // else is replaced, so no id can name another directory (with no `/`, `..` can't).
        None => network_id
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' })
            .collect(),
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
        for addr_str in node.addrs.into_iter().take(MAX_ADDRS_PER_NODE) {
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
                .take(MAX_ADDRS_PER_NODE)
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
        let p = cache_path_for(Path::new("/opt/openhydra/.identity.key"), None);
        assert_eq!(p, PathBuf::from("/opt/openhydra/routing_cache.json"));
        // A private deployment keeps its own.
        let p = cache_path_for(Path::new("/opt/openhydra/.identity.key"), Some("private-0123456789abcdef"));
        assert_eq!(p, PathBuf::from("/opt/openhydra/routing_cache.private-0123456789abcdef.json"));
    }

    #[test]
    fn each_private_deployment_keeps_its_own_cache() {
        let key = |text: &str| crate::transport::private_network_fingerprint(&crate::transport::parse_private_network_key(text).unwrap());
        let (a, b) = (key(&crate::transport::new_private_network_key_file()), key(&crate::transport::new_private_network_key_file()));
        // Keyed: by the key's fingerprint, whatever the network id; another key, another file.
        let sa = private_scope("private-0123456789abcdef", Some(&a));
        assert_eq!(sa, format!("private-key-{}", &a[..16]));
        assert_eq!(sa, private_scope("private-name-lab", Some(&a)));
        assert_ne!(sa, private_scope("private-0123456789abcdef", Some(&b)));
        // Unkeyed: by the network id.
        assert_eq!(private_scope("private-0123456789abcdef", None), "private-0123456789abcdef");
        assert_ne!(private_scope("private-0123456789abcdef", None), private_scope("private-fedcba9876543210", None));
        assert_eq!(private_scope("private-name-acme-lab.1", None), "private-name-acme-lab.1");
        // Never a path, whatever it is given.
        assert_eq!(private_scope("../x/y", None), ".._x_y");
        let p = cache_path_for(Path::new("/opt/openhydra/.identity.key"), Some(&private_scope("../../etc", None)));
        assert_eq!(p.parent(), Some(Path::new("/opt/openhydra")));
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

    #[test]
    fn a_version_1_cache_is_ignored() {
        // Written before 27 Oct 2026, when claimed addresses could reach the routing table.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing_cache.json");
        let v1 = format!(r#"{{"version":1,"nodes":[{{"peer_id":"{}","addrs":["/ip4/8.8.8.8/tcp/4001"]}}]}}"#, PeerId::random());
        std::fs::write(&path, v1).unwrap();
        assert!(load(&path).is_empty());
    }

    #[test]
    fn at_most_four_addresses_per_contact_are_saved_or_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing_cache.json");
        let p = PeerId::random();
        let many: Vec<Multiaddr> = (1..=10).map(|i| format!("/ip4/8.8.8.{i}/tcp/4001").parse().unwrap()).collect();
        save(&path, &[(p, many.clone())]).unwrap();
        assert_eq!(load(&path).len(), 4);
        // And a hand-edited file with more is cut on load too.
        let addrs: Vec<String> = many.iter().map(|a| format!("\"{a}\"")).collect();
        std::fs::write(&path, format!(r#"{{"version":2,"nodes":[{{"peer_id":"{p}","addrs":[{}]}}]}}"#, addrs.join(","))).unwrap();
        assert_eq!(load(&path).len(), 4);
    }
}
