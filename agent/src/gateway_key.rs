// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The local gateway key: a random secret in `~/.openhydra/gateway.key` (owner-only) that the
//! desktop's gateway requires on `/v1/*` and that `connect` / `launch` write into the coding tools'
//! configs. Without it, any local process or web page that could reach `127.0.0.1:16527` could spend
//! the user's credit and read their answers.

use std::io::Write;
use std::path::{Path, PathBuf};

/// Generated keys: this prefix + 64 hex digits (32 random bytes).
pub const KEY_PREFIX: &str = "oh-";

/// The placeholder older builds wrote into tool configs (the loopback gateway was open then).
pub const LEGACY_PLACEHOLDER: &str = "oh-local";

/// `~/.openhydra/gateway.key` (`%USERPROFILE%` on Windows).
pub fn default_path() -> PathBuf {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".openhydra").join("gateway.key")
}

/// Is `k` a key this module generates?
pub fn is_generated(k: &str) -> bool {
    k.strip_prefix(KEY_PREFIX).is_some_and(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Is `k` a key OpenHydra wrote into a tool's config (a generated key, or the old placeholder)?
pub fn is_ours(k: &str) -> bool {
    k == LEGACY_PLACEHOLDER || is_generated(k)
}

fn generate() -> String {
    use rand::RngCore;
    let mut b = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut b);
    format!("{KEY_PREFIX}{}", hex::encode(b))
}

/// The key stored at `path`, if there is a valid one.
pub fn load(path: &Path) -> Option<String> {
    let k = std::fs::read_to_string(path).ok()?.trim().to_string();
    is_generated(&k).then_some(k)
}

/// The key at `path`, creating it (owner-only, 0600 on unix) when missing or invalid.
pub fn load_or_create(path: &Path) -> std::io::Result<String> {
    if let Some(k) = load(path) {
        return Ok(k);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let key = generate();
    // Write to a temp file created owner-only, then rename over: never a moment where the key sits
    // in a file others can read, and a concurrent creator can't leave half a key.
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp)?;
        f.write_all(key.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    // Another process may have won the race: use whatever is there now.
    Ok(load(path).unwrap_or(key))
}

/// The key a tool should present to the local gateway: `OPENHYDRA_API_KEY` if set, else the stored
/// key, else the old placeholder (a CLI gateway without a key is open and ignores it).
pub fn for_tools() -> String {
    std::env::var("OPENHYDRA_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty())
        .or_else(|| load(&default_path()))
        .unwrap_or_else(|| LEGACY_PLACEHOLDER.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_created_key_is_random_owner_only_and_stable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("gateway.key");
        let k = load_or_create(&path).unwrap();
        assert!(is_generated(&k) && is_ours(&k), "{k}");
        assert_eq!(load_or_create(&path).unwrap(), k, "loaded, not regenerated");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let other = tempfile::tempdir().unwrap();
        assert_ne!(load_or_create(&other.path().join("gateway.key")).unwrap(), k);
        // No temp file left behind.
        assert_eq!(std::fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn an_invalid_key_file_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gateway.key");
        std::fs::write(&path, "oh-local").unwrap();
        assert_eq!(load(&path), None);
        let k = load_or_create(&path).unwrap();
        assert!(is_generated(&k));
    }

    #[test]
    fn only_our_keys_are_ours() {
        assert!(is_ours("oh-local"));
        assert!(is_ours(&format!("oh-{}", "a".repeat(64))));
        assert!(!is_ours(&format!("oh-{}", "a".repeat(63))));
        assert!(!is_ours("sk-ant-user-key"));
        assert!(!is_ours(""));
    }
}
