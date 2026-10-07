// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Prompts for the audits a consumer sends when it has no real traffic to replay (review round 3,
//! D1): ordinary chat and coding requests, composed from templates and subjects into a few thousand
//! variations, so an audit reads like a user's request rather than a fixed probe.
//!
//! This is camouflage, not secrecy: the list is public, so a provider that checks every prompt
//! against it can still spot these. Replaying real traffic (where the consumer allows it) is the
//! audit a provider can't tell apart.

use rand::seq::SliceRandom;
use rand::Rng;

/// Coding requests: `{lang}` and `{task}` are filled in.
const CODE_TEMPLATES: &[&str] = &[
    "Write a {lang} function that {task}.",
    "How do I write a function in {lang} that {task}?",
    "Can you show me {lang} code that {task}? Keep it short.",
    "I need a {lang} snippet that {task}. Please add a couple of comments.",
    "What's the cleanest way in {lang} to write something that {task}?",
    "Give me an example in {lang} of a function that {task}, and explain how it works.",
    "Write a small {lang} program that {task}.",
    "In {lang}, implement a helper that {task}. Include one test.",
];

const LANGS: &[&str] = &[
    "Python", "JavaScript", "TypeScript", "Rust", "Go", "Java", "C#", "C++", "Ruby", "Kotlin", "Swift", "PHP", "Bash",
];

const TASKS: &[&str] = &[
    "reverses a string",
    "checks whether a number is prime",
    "removes duplicates from a list while keeping the order",
    "counts how often each word appears in a text",
    "merges two sorted lists into one sorted list",
    "parses a date in YYYY-MM-DD format",
    "finds the largest number in an array",
    "validates an email address",
    "converts Celsius to Fahrenheit",
    "reads a CSV file and prints the number of rows",
    "flattens a nested list",
    "returns the nth Fibonacci number",
    "checks whether a string is a palindrome",
    "sorts a list of people by age",
    "computes the average of a list of numbers",
    "groups a list of items by a key",
    "retries a failing network call with exponential backoff",
    "splits a list into chunks of a given size",
    "turns a title into a URL slug",
    "finds the intersection of two arrays",
    "converts a number to Roman numerals",
    "checks whether two strings are anagrams",
    "rotates a matrix by 90 degrees",
    "debounces a function call",
    "generates a random password of a given length",
    "computes the factorial of a number",
    "finds the longest common prefix of a list of strings",
    "counts the vowels in a sentence",
    "implements binary search on a sorted array",
    "pretty-prints a JSON object",
];

/// General requests: `{topic}` is filled in.
const CHAT_TEMPLATES: &[&str] = &[
    "Explain {topic} in simple terms.",
    "Can you give me a short overview of {topic}?",
    "What are the main ideas behind {topic}?",
    "I'm new to {topic}. Where should I start?",
    "Summarize {topic} in a few paragraphs.",
    "What are common misconceptions about {topic}?",
    "Explain {topic} to a high school student.",
    "Give me three key facts about {topic}.",
    "What are the pros and cons of {topic}?",
    "Write a short introduction to {topic} for a blog post.",
    "How would you explain {topic} to someone who has never heard of it?",
    "List some practical tips related to {topic}.",
];

const TOPICS: &[&str] = &[
    "photosynthesis",
    "the water cycle",
    "compound interest",
    "how vaccines work",
    "black holes",
    "the French Revolution",
    "machine learning",
    "climate change",
    "the stock market",
    "plate tectonics",
    "public key cryptography",
    "how the internet works",
    "the immune system",
    "supply and demand",
    "the Renaissance",
    "renewable energy",
    "DNA replication",
    "the theory of relativity",
    "inflation",
    "how airplanes fly",
    "sleep and memory",
    "the Roman Empire",
    "recycling",
    "meditation",
    "container gardening",
    "time management",
    "budgeting for students",
    "learning a second language",
    "remote work",
    "healthy breakfast ideas",
    "how batteries store energy",
    "the history of jazz",
    "volcanoes",
    "electric cars",
    "the scientific method",
    "probability",
    "the Silk Road",
    "coral reefs",
    "how search engines rank pages",
    "writing a good resume",
];

/// Requests that stand on their own.
const STANDALONE: &[&str] = &[
    "Write a short poem about autumn leaves.",
    "Suggest five names for a small coffee shop.",
    "Write a polite email asking my manager for a day off next Friday.",
    "What's a good weekly workout plan for a beginner?",
    "Give me a simple recipe for vegetable soup.",
    "Write a haiku about the ocean.",
    "What should I pack for a three-day hiking trip?",
    "Write a short story opening about a lost key.",
    "How can I improve my focus while studying?",
    "Write a thank-you note for a birthday gift.",
    "What are some fun things to do on a rainy day?",
    "Explain the difference between a list and a tuple in Python.",
    "What is the difference between TCP and UDP?",
    "Explain what a REST API is.",
    "What does Big O notation mean?",
    "How does garbage collection work?",
    "What is the difference between a process and a thread?",
    "Explain recursion with a simple example.",
    "What is a hash table and why is it fast?",
    "How do Git branches work?",
];

/// One audit prompt, drawn at random from a few thousand variations.
pub fn random_prompt<R: Rng + ?Sized>(rng: &mut R) -> String {
    let pick = |items: &[&'static str], rng: &mut R| *items.choose(rng).expect("non-empty list");
    match rng.gen_range(0..10) {
        0..=3 => pick(CODE_TEMPLATES, rng).replace("{lang}", pick(LANGS, rng)).replace("{task}", pick(TASKS, rng)),
        4..=8 => pick(CHAT_TEMPLATES, rng).replace("{topic}", pick(TOPICS, rng)),
        _ => pick(STANDALONE, rng).to_string(),
    }
}

/// How many distinct prompts [`random_prompt`] can produce.
#[cfg(test)]
fn variations() -> usize {
    CODE_TEMPLATES.len() * LANGS.len() * TASKS.len() + CHAT_TEMPLATES.len() * TOPICS.len() + STANDALONE.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_are_varied_and_fully_filled_in() {
        assert!(variations() > 3_000);
        let mut rng = rand::thread_rng();
        let drawn: std::collections::HashSet<String> = (0..500).map(|_| random_prompt(&mut rng)).collect();
        assert!(drawn.len() > 300, "500 draws give mostly distinct prompts, got {}", drawn.len());
        for p in &drawn {
            assert!(!p.contains('{') && !p.contains('}'), "unfilled template: {p}");
            assert!(!p.to_lowercase().contains("verification") && !p.to_lowercase().contains("probe"));
        }
    }
}
