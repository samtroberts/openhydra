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
//!
//! Fresher (review P1): questions written for this gateway by a model ([`writer_request`],
//! [`parse_questions`]), then changed here before use ([`finish`]), so no prompt an audit sends is
//! on any list, and the model that wrote the raw question never sees the final one. The templates
//! are the fallback.

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

/// Ways a person asks, for [`writer_request`].
const STYLES: &[&str] = &[
    "casual and short",
    "detailed, with some background about their situation",
    "a beginner's question",
    "an expert's precise question",
    "slightly informal, with a typo or two",
    "polite and formal",
    "a request for step-by-step help",
    "a comparison between two options",
];

/// What a gateway asks a model, to write questions it can audit with: ordinary questions about a
/// random subject, in random styles, one per line.
pub fn writer_request<R: Rng + ?Sized>(rng: &mut R) -> String {
    let subject = if rng.gen_bool(0.4) {
        format!("programming in {}", LANGS.choose(rng).expect("non-empty"))
    } else {
        TOPICS.choose(rng).expect("non-empty").to_string()
    };
    let style = STYLES.choose(rng).expect("non-empty");
    format!(
        "Write 8 different questions a person might ask an AI assistant about {subject}. Make them {style}, \
         and vary their length. Write only the questions, one per line, without numbering or quotes."
    )
}

/// The most questions taken from one writer's answer (as many as it was asked for): a writer that
/// sends more can't fill the pool with its own (review round 4 H6).
pub const MAX_QUESTIONS_PER_ANSWER: usize = 8;

/// The questions in a writer's answer: one per line, list marks and quotes dropped, each 15 to 400
/// characters, no repeats, at most [`MAX_QUESTIONS_PER_ANSWER`]. Only visible text is kept: control
/// and invisible characters (zero-width marks, say) are dropped, so a writer can't hide a mark in a
/// question that a colluding provider would spot in the probe (review round 4 H6).
pub fn parse_questions(text: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    text.lines()
        .map(|l| {
            // Every kind of space becomes a plain one (a run of odd spaces can't carry a mark, R7-6).
            let visible: String = l.chars().filter(|c| is_visible(*c)).map(|c| if c.is_whitespace() { ' ' } else { c }).collect();
            visible
                .trim()
                .trim_start_matches(|c: char| c.is_ascii_digit() || matches!(c, '.' | ')' | '-' | '*' | '•' | ' '))
                .trim_matches(|c: char| is_quote(c) || c.is_whitespace())
                .to_string()
        })
        .filter(|q| (15..=400).contains(&q.chars().count()) && !q.starts_with("Here are") && seen.insert(q.to_lowercase()))
        .take(MAX_QUESTIONS_PER_ANSWER)
        .collect()
}

/// Whether `c` shows as text: not a control character, and not one of the invisible characters a
/// mark could hide in (zero-width spaces and joiners, direction marks, word joiners, the byte-order
/// mark, the soft hyphen, variation selectors, the combining grapheme joiner, fillers, tag
/// characters; review rounds 4 and 5). Scripts' own combining signs (a Devanagari virama, say) stay,
/// so probe questions in any language read normally (R6-9).
fn is_visible(c: char) -> bool {
    !c.is_control()
        && !matches!(
            c,
            '\u{00AD}'
                | '\u{034F}'
                | '\u{061C}'
                | '\u{115F}'
                | '\u{1160}'
                | '\u{17B4}'
                | '\u{17B5}'
                | '\u{180B}'..='\u{180F}'
                | '\u{200B}'..='\u{200F}'
                | '\u{202A}'..='\u{202E}'
                | '\u{2060}'..='\u{206F}'
                | '\u{3164}'
                | '\u{FE00}'..='\u{FE0F}'
                | '\u{FEFF}'
                | '\u{FFA0}'
                | '\u{FFF9}'..='\u{FFFC}'
                | '\u{1BCA0}'..='\u{1BCA3}'
                | '\u{1D173}'..='\u{1D17A}'
                | '\u{E0000}'..='\u{E007F}'
                | '\u{E0100}'..='\u{E01EF}'
        )
}

/// Quotation marks a question may come wrapped in.
fn is_quote(c: char) -> bool {
    matches!(c, '"' | '\'' | '“' | '”' | '‘' | '’' | '«' | '»' | '„' | '‚' | '`')
}

/// System prompts commonly sent with requests (for probes, where real requests carry one).
const GENERIC_SYSTEM: &[&str] = &[
    "You are a helpful assistant.",
    "You are a helpful, concise assistant.",
    "You are a knowledgeable assistant. Answer clearly and accurately.",
    "You are an AI assistant that helps people find information.",
    "You are a friendly assistant. Keep answers short unless asked for detail.",
];

/// A system prompt of the kind clients commonly send.
pub fn generic_system_prompt<R: Rng + ?Sized>(rng: &mut R) -> &'static str {
    GENERIC_SYSTEM.choose(rng).expect("non-empty")
}

/// Openings a person might put before a question.
const OPENINGS: &[&str] = &["", "", "", "Quick question: ", "Hi! ", "Hey, ", "Can you help me? ", "I was wondering: "];
/// Details a person might add after it.
const DETAILS: &[&str] = &[
    "",
    "",
    " I'm fairly new to this.",
    " Please keep it short.",
    " An example would help.",
    " This is for a project I'm working on.",
    " I've read a bit about it already.",
    " Thanks in advance!",
];

/// A prompt finished here from the raw questions: one of them, or two joined, with an opening and a
/// detail of our own, so the model that wrote the raw questions never sees the final text. Takes the
/// questions it uses from `raw`; `None` when it is empty.
pub fn finish<R: Rng + ?Sized>(raw: &mut std::collections::VecDeque<String>, rng: &mut R) -> Option<String> {
    let first = raw.pop_front()?;
    let body = match raw.front() {
        Some(_) if rng.gen_bool(0.35) => {
            let second = raw.pop_front().expect("checked");
            // Lower-cased by its first character, not byte (it may not be ASCII, review round 4 M7).
            let mut chars = second.chars();
            let head: String = chars.next().map(|c| c.to_lowercase().collect()).unwrap_or_default();
            format!("{first} Also, {head}{}", chars.as_str())
        }
        _ => first,
    };
    let opening = OPENINGS.choose(rng).expect("non-empty");
    let mut detail = *DETAILS.choose(rng).expect("non-empty");
    // Never exactly what the writer wrote.
    while opening.is_empty() && detail.is_empty() && !body.contains(" Also, ") {
        detail = DETAILS.choose(rng).expect("non-empty");
    }
    Some(format!("{opening}{body}{detail}"))
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

    #[test]
    fn a_writers_questions_are_parsed_and_finished_into_new_text() {
        let answer = "Here are 8 questions:\n1. How do I sort a list in Python?\n2) What's a decorator in Python, really?\n\
                      - \"Why is my loop so slow?\"\n* ok\n1. How do I sort a list in Python?";
        let qs = parse_questions(answer);
        assert_eq!(qs, vec!["How do I sort a list in Python?", "What's a decorator in Python, really?", "Why is my loop so slow?"]);
        let mut rng = rand::thread_rng();
        let mut raw: std::collections::VecDeque<String> = qs.iter().cloned().collect();
        let mut finished = Vec::new();
        while let Some(p) = finish(&mut raw, &mut rng) {
            finished.push(p);
        }
        assert!(!finished.is_empty() && raw.is_empty(), "every raw question used once");
        // Each finished prompt contains a raw question but is never just one (no prompt the writer saw).
        for _ in 0..500 {
            let mut raw: std::collections::VecDeque<String> = qs.iter().cloned().collect();
            let p = finish(&mut raw, &mut rng).unwrap();
            assert!(qs.iter().any(|q| p.contains(q.as_str()) || p.contains(&q[1..])));
            assert!(!qs.contains(&p), "unchanged from the writer's: {p}");
        }
        let req = writer_request(&mut rng);
        assert!(req.contains("one per line") && !req.to_lowercase().contains("audit"));
    }

    /// Review round 4 H6/M7: a writer can't hide marks in its questions or flood the pool, and a
    /// question starting with a non-ASCII character is joined without a panic.
    #[test]
    fn a_writers_answer_is_cleaned_capped_and_joined_safely() {
        let marked = "How do\u{200B} I sort\u{2060} a list\u{FE0F} in\u{3164} Rust?\u{FEFF}\n“¿Qué es un cierre en Rust, exactamente?”";
        let qs = parse_questions(marked);
        assert_eq!(qs, vec!["How do I sort a list in Rust?", "¿Qué es un cierre en Rust, exactamente?"]);
        // Scripts' own combining signs stay (a Hindi question reads as written).
        assert_eq!(parse_questions("क्या आप पायथन में सूची को क्रमबद्ध करना समझा सकते हैं?"), vec!["क्या आप पायथन में सूची को क्रमबद्ध करना समझा सकते हैं?"]);
        let many: String = (0..50).map(|i| format!("Question number {i} about something?\n")).collect();
        assert_eq!(parse_questions(&many).len(), MAX_QUESTIONS_PER_ANSWER);
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let mut raw: std::collections::VecDeque<String> =
                ["What is a closure in Rust?".to_string(), "Émile asked: what is ownership?".to_string()].into();
            let p = finish(&mut raw, &mut rng).unwrap();
            assert!(!p.is_empty());
        }
    }
}
