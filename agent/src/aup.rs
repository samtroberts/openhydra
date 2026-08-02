// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Acceptable-Use Policy floor — a minimal, operator-configured request guard.
//!
//! This is a **mechanism, not a mandated policy**: the default is fully permissive, and the
//! operator opts in to limits via flags. It exists so a node exposed to an open, untrusted
//! network has a *floor* — a way to bound request size and refuse content it won't serve —
//! without baking anyone's content rules into the protocol.
//!
//! Applied at two points (both drive the same pure [`AupPolicy::evaluate`]):
//! - the **provider** serve loop — the security-critical one, since a provider serves
//!   strangers from the open network;
//! - the **gateway** front door — the operator's own ingress.
//!
//! Rate limiting is intentionally *not* here: per-consumer rate is already the M2.3
//! give/take credit throttle on the provider; gateway DoS protection is a separate concern.
//! The floor is about *size* and *content*, the two things credit can't express.

use crate::adapter::ChatMessage;

/// An operator's acceptable-use limits. A `0` numeric limit means "unlimited"; an empty
/// `denied_substrings` means "no content filter". The all-zero/empty value
/// ([`permissive`](AupPolicy::permissive)) allows everything — the default.
#[derive(Debug, Clone, Default)]
pub struct AupPolicy {
    /// Reject a request with more than this many messages (`0` = unlimited).
    pub max_messages: usize,
    /// Reject a request whose total prompt content exceeds this many characters
    /// (`0` = unlimited).
    pub max_prompt_chars: usize,
    /// Reject a request whose explicit `max_tokens` exceeds this (`0` = unlimited). A request
    /// that sets no `max_tokens` is not capped here — that is the engine's own default.
    pub max_completion_tokens: u32,
    /// Refuse a request whose any message contains one of these (case-insensitive) — the
    /// operator's content blocklist. Empty = no content filter.
    pub denied_substrings: Vec<String>,
}

/// The outcome of an [`AupPolicy::evaluate`]. `Deny` carries a human-readable reason (size
/// reasons are specific to help a legitimate client; a content match is deliberately generic
/// so the blocklist can't be enumerated by probing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AupDecision {
    Allow,
    Deny(String),
}

impl AupPolicy {
    /// The default floor: allow everything (a pure mechanism until the operator sets limits).
    pub fn permissive() -> Self {
        Self::default()
    }

    /// Whether any limit is configured. When `false`, callers can skip
    /// [`evaluate`](Self::evaluate) entirely (the common, zero-overhead path).
    pub fn is_active(&self) -> bool {
        self.max_messages > 0
            || self.max_prompt_chars > 0
            || self.max_completion_tokens > 0
            || !self.denied_substrings.is_empty()
    }

    /// Evaluate `messages` (and the request's explicit `max_tokens`, if any) against the
    /// policy. The first rule that trips wins; otherwise [`Allow`](AupDecision::Allow).
    pub fn evaluate(&self, messages: &[ChatMessage], max_tokens: Option<u32>) -> AupDecision {
        if self.max_messages > 0 && messages.len() > self.max_messages {
            return AupDecision::Deny(format!(
                "too many messages ({} > limit {})",
                messages.len(),
                self.max_messages
            ));
        }

        if self.max_prompt_chars > 0 {
            let total: usize = messages.iter().map(|m| m.content.chars().count()).sum();
            if total > self.max_prompt_chars {
                return AupDecision::Deny(format!(
                    "prompt too large ({total} chars > limit {})",
                    self.max_prompt_chars
                ));
            }
        }

        if self.max_completion_tokens > 0 {
            if let Some(want) = max_tokens {
                if want > self.max_completion_tokens {
                    return AupDecision::Deny(format!(
                        "max_tokens {want} exceeds limit {}",
                        self.max_completion_tokens
                    ));
                }
            }
        }

        if !self.denied_substrings.is_empty() {
            for msg in messages {
                let haystack = msg.content.to_lowercase();
                if self
                    .denied_substrings
                    .iter()
                    .any(|needle| haystack.contains(&needle.to_lowercase()))
                {
                    // Generic on purpose — don't reveal which pattern matched.
                    return AupDecision::Deny("request rejected by content policy".to_string());
                }
            }
        }

        AupDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(content: &str) -> ChatMessage {
        ChatMessage { role: "user".to_string(), content: content.to_string(), ..Default::default() }
    }

    #[test]
    fn permissive_allows_everything() {
        let p = AupPolicy::permissive();
        assert!(!p.is_active());
        assert_eq!(p.evaluate(&[msg("anything at all")], Some(100_000)), AupDecision::Allow);
    }

    #[test]
    fn too_many_messages_denied() {
        let p = AupPolicy { max_messages: 2, ..Default::default() };
        assert_eq!(p.evaluate(&[msg("a"), msg("b")], None), AupDecision::Allow);
        assert!(matches!(p.evaluate(&[msg("a"), msg("b"), msg("c")], None), AupDecision::Deny(_)));
    }

    #[test]
    fn oversized_prompt_denied_counting_all_messages() {
        let p = AupPolicy { max_prompt_chars: 10, ..Default::default() };
        assert_eq!(p.evaluate(&[msg("hello")], None), AupDecision::Allow); // 5 chars
        // 6 + 6 = 12 > 10, summed across messages.
        assert!(matches!(p.evaluate(&[msg("abcdef"), msg("ghijkl")], None), AupDecision::Deny(_)));
    }

    #[test]
    fn prompt_chars_counts_unicode_scalars_not_bytes() {
        let p = AupPolicy { max_prompt_chars: 3, ..Default::default() };
        // "héllo" worth of multibyte: 3 chars allowed, 4 denied (bytes would over-count).
        assert_eq!(p.evaluate(&[msg("é€ñ")], None), AupDecision::Allow); // 3 scalars
        assert!(matches!(p.evaluate(&[msg("é€ñx")], None), AupDecision::Deny(_)));
    }

    #[test]
    fn max_tokens_over_limit_denied_but_unset_allowed() {
        let p = AupPolicy { max_completion_tokens: 256, ..Default::default() };
        assert_eq!(p.evaluate(&[msg("hi")], Some(256)), AupDecision::Allow);
        assert_eq!(p.evaluate(&[msg("hi")], None), AupDecision::Allow); // unset → engine default
        assert!(matches!(p.evaluate(&[msg("hi")], Some(257)), AupDecision::Deny(_)));
    }

    #[test]
    fn content_blocklist_is_case_insensitive_and_generic() {
        let p = AupPolicy {
            denied_substrings: vec!["FORBIDDEN".to_string()],
            ..Default::default()
        };
        assert_eq!(p.evaluate(&[msg("a clean request")], None), AupDecision::Allow);
        // Match regardless of case, and the reason must not echo the matched term.
        match p.evaluate(&[msg("this is forbidden content")], None) {
            AupDecision::Deny(reason) => assert!(!reason.to_lowercase().contains("forbidden")),
            AupDecision::Allow => panic!("should have been denied"),
        }
    }
}
