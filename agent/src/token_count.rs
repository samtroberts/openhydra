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
/// The most delivered text kept for a count. A paid serve can't come near it (at most 65,536
/// tokens); past it, an unpaid serve's count is estimated from what was kept (review B5).
pub const MAX_COUNTED_BYTES: usize = 8 << 20;
/// The text one scorer run counts: a longer text is counted in pieces of at most this, summed.
pub const COUNT_CHUNK_BYTES: usize = 1 << 20;

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

/// `text` in pieces of at most `max` bytes, cut at a whitespace where one is near the end (so few
/// words are split: a split word may count a token more, never fewer).
pub fn chunks(text: &str, max: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = text;
    while rest.len() > max {
        let mut cut = max;
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        if let Some(ws) = rest[..cut].rfind(char::is_whitespace).filter(|&i| i > cut / 2) {
            cut = ws + rest[ws..].chars().next().map_or(1, char::len_utf8);
        }
        out.push(&rest[..cut]);
        rest = &rest[cut..];
    }
    if !rest.is_empty() || out.is_empty() {
        out.push(rest);
    }
    out
}

/// [`count_tokens`] over a text of any length kept for counting, in [`COUNT_CHUNK_BYTES`] pieces.
pub fn count_tokens_chunked(scorer: &Path, vocab: &Path, text: &str) -> Result<u64, String> {
    chunks(text, COUNT_CHUNK_BYTES).into_iter().try_fold(0u64, |sum, c| Ok(sum + count_tokens(scorer, vocab, c)?))
}

/// The tokens of a whole text estimated from a counted part of it: `counted` tokens in `part_units`
/// ([`crate::serve::DeliveredText`] units), scaled to `total_units`.
pub fn estimate_from_part(counted: u64, part_units: u64, total_units: u64) -> u64 {
    if part_units == 0 {
        return counted;
    }
    ((counted as u128 * total_units as u128) / part_units as u128) as u64
}

/// Cut `text` (a serve's delivered text) so it fits beside `tool_calls` within [`MAX_COUNTED_BYTES`],
/// at a character boundary; whether it was cut (the count is then estimated from what's left, review
/// round 4 S4-9: an answer padded past the cap isn't skipped).
pub fn fit_for_counting(text: &mut String, tool_calls: &[ToolCall]) -> bool {
    let calls_len: usize = tool_calls.iter().map(|c| c.function.name.len() + c.function.arguments.len() + 2).sum();
    let room = MAX_COUNTED_BYTES.saturating_sub(calls_len);
    if text.len() <= room {
        return false;
    }
    let mut cut = room;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    text.truncate(cut);
    true
}

/// The text a serve's tokens are counted on: the delivered text, then each tool call's name and
/// arguments (on lines of their own), and how many of the calls earn the template allowance
/// ([`crate::serve::allowance_earners`], as billing grants it). `None` past [`MAX_COUNTED_BYTES`]
/// in all: too long to count.
pub fn countable_text(
    delivered: &str,
    tool_calls: &[ToolCall],
    offered: &std::collections::HashSet<String>,
) -> Option<(String, usize)> {
    let calls_len: usize = tool_calls.iter().map(|c| c.function.name.len() + c.function.arguments.len() + 2).sum();
    if delivered.len().saturating_add(calls_len) > MAX_COUNTED_BYTES {
        return None;
    }
    let mut text = delivered.to_string();
    for c in tool_calls {
        text.push('\n');
        text.push_str(&c.function.name);
        text.push('\n');
        text.push_str(&c.function.arguments);
    }
    let allowed = crate::serve::allowance_earners(tool_calls, offered).into_iter().filter(|e| *e).count();
    Some((text, allowed))
}

/// Whether `billed` tokens is more than `counted` (the delivered text's tokens, `calls` tool calls
/// among them) can account for: each tool call's template tokens ([`TOOL_CALL_TOKEN_ALLOWANCE`]),
/// end-of-turn markers ([`BILLABLE_TOKEN_SLACK`]) and [`COUNT_TOLERANCE`].
pub fn over_billed(billed: u64, counted: u64, calls: usize) -> bool {
    billed > billable_by_count(counted, calls)
}

/// The most a serve may bill given the tokenizer's count of what it delivered (`counted`, plus
/// `calls` tool calls): the count with [`COUNT_TOLERANCE`] and [`BILLABLE_TOKEN_SLACK`] on top, so an
/// honest provider's count (a different tokenizer build, a template token) always fits.
pub fn billable_by_count(counted: u64, calls: usize) -> u64 {
    let expected = counted.saturating_add(calls as u64 * TOOL_CALL_TOKEN_ALLOWANCE);
    (expected as f64 * (1.0 + COUNT_TOLERANCE) + BILLABLE_TOKEN_SLACK as f64).floor() as u64
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
        // What an over-billing provider is paid instead: the largest count that isn't over-billed.
        for (counted, calls) in [(120u64, 0usize), (20, 2), (0, 0), (1_000_000, 16)] {
            let fair = billable_by_count(counted, calls);
            assert!(!over_billed(fair, counted, calls) && over_billed(fair + 1, counted, calls), "{counted} {calls}");
        }
        // The 3-4x a byte-bound bill allows (one token per character) is cut to the count.
        assert!(billable_by_count(120, 0) < 480 / 3);
    }

    #[test]
    fn tool_calls_are_counted_after_the_text() {
        let call = ToolCall {
            id: "c".into(),
            kind: "function".into(),
            function: crate::adapter::ToolCallFunction { name: "ls".into(), arguments: "{\"p\":\"/\"}".into() },
        };
        let ls: std::collections::HashSet<String> = ["ls".to_string()].into();
        assert_eq!(countable_text("hi", &[call.clone()], &ls), Some(("hi\nls\n{\"p\":\"/\"}".to_string(), 1)));
        assert_eq!(countable_text("hi", &[call.clone()], &Default::default()).unwrap().1, 0, "a tool nobody offered");
        assert_eq!(countable_text("hi", &[], &ls), Some(("hi".to_string(), 0)));
        // Text and tool calls together are bounded: a huge argument can't make the count unbounded.
        let mut big = call.clone();
        big.function.arguments = "x".repeat(MAX_COUNTED_BYTES);
        assert_eq!(countable_text("hi", &[big.clone()], &ls), None);
        // A long answer is cut to fit beside the calls, not skipped.
        let mut long = "é".repeat(MAX_COUNTED_BYTES);
        assert!(fit_for_counting(&mut long, &[call.clone()]));
        assert!(countable_text(&long, &[call.clone()], &ls).is_some());
        let mut short = "hi".to_string();
        assert!(!fit_for_counting(&mut short, &[call]));
    }

    #[test]
    fn long_texts_are_counted_in_pieces_cut_between_words() {
        let text = "alpha beta gamma delta ".repeat(10);
        let pieces = chunks(&text, 50);
        assert!(pieces.iter().all(|p| p.len() <= 50));
        assert_eq!(pieces.concat(), text, "nothing lost or doubled");
        assert!(pieces[..pieces.len() - 1].iter().all(|p| p.ends_with(' ')), "cut after a space");
        // Multi-byte characters are never split.
        let wide = "日本語のテキスト".repeat(20);
        assert_eq!(chunks(&wide, 31).concat(), wide);
        assert_eq!(chunks("", 10), vec![""]);
        // An estimate scales the counted part to the whole.
        assert_eq!(estimate_from_part(250, 1_000, 4_000), 1_000);
        assert_eq!(estimate_from_part(7, 0, 100), 7);
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
