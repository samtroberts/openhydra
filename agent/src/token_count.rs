// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Counting a delivered completion's tokens with the model's own tokenizer — the check behind a
//! provider's billed count (P7). Billing is bounded by the delivered text's *bytes*
//! ([`crate::serve::billable_tokens`]), which is about 4× the tokens of typical text; with the
//! tokenizer a gateway can see whether a provider billed for tokens it didn't deliver.
//!
//! The tokenizer is a vocabulary-only GGUF (the model's metadata, no weights; a few MB) made with
//! `openhydra-scorer --export-vocab` and named by a log-prob reference file (`"vocab"`). Counting
//! runs `openhydra-scorer --count-tokens <vocab>`, the text on stdin.
//!
//! What it can't tell apart: an engine that bills for reasoning it never streams (some hosted
//! APIs) bills more than its delivered text tokenizes to, as a padded count would.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::adapter::ToolCall;
use crate::serve::{BILLABLE_TOKEN_SLACK, TOOL_CALL_TOKEN_ALLOWANCE};

/// Share of a provider's serves whose billed count is checked, when a tokenizer covers its model.
pub const COUNT_SAMPLE_RATE: f64 = 0.25;
/// Counts running at once: each is a scorer process loading the vocabulary (~0.25 s); past this a
/// sampled serve is just not checked, so a busy gateway can't pile them up.
pub const MAX_CONCURRENT_COUNTS: usize = 4;
/// The longest delivered text kept for a count; a longer one isn't checked (its copy isn't kept).
pub const MAX_COUNTED_BYTES: usize = 1 << 20;

/// A place among [`MAX_CONCURRENT_COUNTS`]: given back when dropped.
pub struct CountSlot(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl CountSlot {
    /// A slot, if fewer than [`MAX_CONCURRENT_COUNTS`] counts hold one.
    pub fn try_take(running: &std::sync::Arc<std::sync::atomic::AtomicUsize>) -> Option<Self> {
        use std::sync::atomic::Ordering;
        running
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| (n < MAX_CONCURRENT_COUNTS).then_some(n + 1))
            .ok()
            .map(|_| CountSlot(std::sync::Arc::clone(running)))
    }
}

impl Drop for CountSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}
/// A count takes milliseconds; past this the scorer is stuck and killed.
const COUNT_TIMEOUT: Duration = Duration::from_secs(15);
/// The billed count may exceed the tokenized count by this share (an engine's sampled tokens don't
/// always re-tokenize to the same split) plus [`BILLABLE_TOKEN_SLACK`] (end-of-turn markers).
const COUNT_TOLERANCE: f64 = 0.10;

/// How many tokens `text` is under the tokenizer in `vocab`, using the scorer binary `scorer`.
pub fn count_tokens(scorer: &Path, vocab: &Path, text: &str) -> Result<u64, String> {
    let mut child = Command::new(scorer)
        .arg("--count-tokens")
        .arg(vocab)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{}: {e}", scorer.display()))?;
    // Write on another thread: a large text can fill the pipe before the scorer starts reading.
    let mut stdin = child.stdin.take().ok_or("no stdin")?;
    let input = text.as_bytes().to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&input));
    let started = Instant::now();
    let status = loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => break status,
            None if started.elapsed() > COUNT_TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("the scorer didn't finish counting".into());
            }
            None => std::thread::sleep(Duration::from_millis(5)),
        }
    };
    let _ = writer.join();
    let mut out = String::new();
    child.stdout.take().ok_or("no stdout")?.read_to_string(&mut out).map_err(|e| e.to_string())?;
    if !status.success() {
        let mut err = String::new();
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        return Err(format!("count-tokens failed ({status}): {}", err.trim()));
    }
    out.trim().parse().map_err(|_| format!("count-tokens printed {:?}", out.trim()))
}

/// The text a serve's tokens are counted on: the delivered text, then each tool call's name and
/// arguments (on lines of their own), and how many of the calls earn the template allowance (calls to
/// an `offered` tool, at most [`crate::serve::MAX_ALLOWED_TOOL_CALLS`] — as billing grants it).
pub fn countable_text(delivered: &str, tool_calls: &[ToolCall], offered: &std::collections::HashSet<String>) -> (String, usize) {
    let mut text = delivered.to_string();
    for c in tool_calls {
        text.push('\n');
        text.push_str(&c.function.name);
        text.push('\n');
        text.push_str(&c.function.arguments);
    }
    let allowed = tool_calls.iter().filter(|c| offered.contains(&c.function.name)).count();
    (text, allowed.min(crate::serve::MAX_ALLOWED_TOOL_CALLS))
}

/// Whether `billed` tokens is more than `counted` (the delivered text's tokens, `calls` tool calls
/// among them) can account for: each tool call's template tokens ([`TOOL_CALL_TOKEN_ALLOWANCE`]),
/// end-of-turn markers ([`BILLABLE_TOKEN_SLACK`]) and [`COUNT_TOLERANCE`].
pub fn over_billed(billed: u64, counted: u64, calls: usize) -> bool {
    let expected = counted.saturating_add(calls as u64 * TOOL_CALL_TOKEN_ALLOWANCE);
    let allowed = expected as f64 * (1.0 + COUNT_TOLERANCE) + BILLABLE_TOKEN_SLACK as f64;
    billed as f64 > allowed
}

/// The scorer binary a gateway counts with: `OPENHYDRA_SCORER` if set, else `openhydra-scorer` next
/// to the agent.
pub fn locate_count_scorer() -> Option<PathBuf> {
    let explicit = std::env::var_os("OPENHYDRA_SCORER").map(PathBuf::from);
    crate::side_scorer::locate_scorer(explicit.as_deref(), crate::side_scorer::SCORER_BIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bill_within_the_counted_tokens_passes_and_padding_does_not() {
        // Engines bill what they generated: the text's tokens, maybe one end-of-turn marker more.
        assert!(!over_billed(121, 120, 0));
        assert!(!over_billed(120 + 12 + BILLABLE_TOKEN_SLACK, 120, 0), "within the tolerance");
        // The byte bound allows ~4× typical text: that much is over-billing.
        assert!(over_billed(480, 120, 0));
        // Tool calls carry template tokens of their own.
        assert!(!over_billed(20 + 2 * TOOL_CALL_TOKEN_ALLOWANCE, 20, 2));
        assert!(over_billed(20 + 2 * TOOL_CALL_TOKEN_ALLOWANCE, 20, 0));
    }

    #[test]
    fn tool_calls_are_counted_after_the_text() {
        let call = ToolCall {
            id: "c".into(),
            kind: "function".into(),
            function: crate::adapter::ToolCallFunction { name: "ls".into(), arguments: "{\"p\":\"/\"}".into() },
        };
        let ls: std::collections::HashSet<String> = ["ls".to_string()].into();
        assert_eq!(countable_text("hi", &[call.clone()], &ls), ("hi\nls\n{\"p\":\"/\"}".to_string(), 1));
        assert_eq!(countable_text("hi", &[call], &Default::default()).1, 0, "a tool nobody offered");
        assert_eq!(countable_text("hi", &[], &ls), ("hi".to_string(), 0));
    }

    #[test]
    fn counts_are_bounded() {
        let running = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let slots: Vec<CountSlot> = (0..MAX_CONCURRENT_COUNTS).filter_map(|_| CountSlot::try_take(&running)).collect();
        assert_eq!(slots.len(), MAX_CONCURRENT_COUNTS);
        assert!(CountSlot::try_take(&running).is_none(), "full: the serve just isn't checked");
        drop(slots);
        assert!(CountSlot::try_take(&running).is_some());
    }

    #[test]
    fn a_missing_scorer_is_an_error_not_a_hang() {
        assert!(count_tokens(Path::new("/nonexistent/openhydra-scorer"), Path::new("/x.gguf"), "hi").is_err());
    }
}
