// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Model cost-weight registry (CREDIT_TIERING_PLAN_v2, Phase 1).
//!
//! Today the ledger is **flat**: 1 token served = 1 credit, so a `tinyllama:1b` earns the
//! same as a `qwen3-coder:30b`. This module derives the **cost-to-serve weight** of a model
//! so credit can track it (`credits = weight(model) × (prompt·w_in + completion·w_out)`).
//!
//! **Cost basis = memory footprint (VRAM occupied), not compute** (Decision #1). The binding,
//! honest scarcity for a provider is *"can I even fit this?"* — the unified-memory / VRAM the
//! weights occupy — not tokens/sec. So:
//!
//! ```text
//! footprint_GB = total_params × bytes_per_param(quant)
//! ```
//!
//! Two consequences the plan leans on:
//! * **MoE is weighted by *total* params, not active** — every expert must be resident because
//!   routing can pick any expert for any token; active params only describe compute *per token*.
//!   A `qwen3-coder:30b-a3b` occupies ~30B of memory, so it weighs as 30B (the `-a3b` suffix is
//!   ignored for footprint).
//! * **Quantisation is first-class** — a `70b-Q4` (~35 GB) weighs *less* than a `70b-Q8`
//!   (~70 GB). A param-only table cannot express that; the GB basis makes the quant correction
//!   real.
//!
//! ## Where the weight comes from (anti-gaming)
//! The weight is looked up from the **canonical model id** (`family/params/quant/template_hash`),
//! which the provider does **not** get to self-declare into the ledger — that is the attack
//! surface. `params` and `quant` are already parsed by [`crate::model_id`]. When either is
//! missing or unparseable, this module falls back to the **nano** tier (the conservative floor):
//! an unknown model never defaults *high*, so a "claim big, serve small" attempt can only ever
//! *under*-award itself here (the real defence — model-authenticity verification — is
//! Phase 4 / `PROOF_OF_INFERENCE_DESIGN.md`; this module is deliberately unwired until then).
//!
//! **Purity.** No I/O, no clock, no provider input — a canonical id in, a weight out. This is the
//! shared registry the auto-router also needs (`AUTO_ROUTER_IMPROVEMENT_PLAN_v1` §1); build once.
//!
//! **Status:** Phase 1 — pure registry only. Nothing here is wired into [`crate::credit`] or the
//! ledger yet; that is Phase 3, which must not ship ahead of the Phase 4 verification gate.

use crate::model_id::{normalize_quant, CanonicalModelId};

/// Input-token weight (prefill: one parallel, compute-bound pass). The unit; `w_in = 1`.
pub const W_IN: u64 = 1;

/// Output-token weight relative to input (Decision #10). Decode is autoregressive — one pass
/// **plus a full KV-cache read per token**, bandwidth-bound — so output tokens cost ~3–5× input.
/// `4×` is the market median (GPT-4o 4×; Anthropic 5×; Gemini/DeepSeek ~3×). Global default; a
/// future refinement may derive the ratio per-model from published input/output prices.
pub const W_OUT: u64 = 4;

/// Bytes of memory one parameter occupies at a given (normalised) quantisation. Drives the
/// footprint: `footprint_GB = total_params_billions × bytes_per_param`.
///
/// `quant` is a **normalised** label as produced by [`normalize_quant`] (`fp16`, `int4`, …). An
/// unrecognised label returns the **conservative** `int4` cost (0.5) so a novel/unparseable quant
/// can only *under*-state footprint, never inflate it — the weight registry never takes the
/// provider's word for a bigger number.
pub fn bytes_per_param(normalized_quant: &str) -> f64 {
    match normalized_quant {
        "fp32" => 4.0,
        "fp16" | "bf16" => 2.0,
        "fp8" | "int8" => 1.0,
        "int4" => 0.5,
        // GGUF k-/legacy quants that `normalize_quant` keeps as-is (it folds only 4- and 8-bit):
        // their average bits per weight, so a Q6_K model isn't priced like a 4-bit one.
        q if q.starts_with("q6") => 6.5 / 8.0,
        q if q.starts_with("q5") => 5.5 / 8.0,
        q if q.starts_with("q3") => 3.5 / 8.0,
        q if q.starts_with("q2") => 2.6 / 8.0,
        // Unknown → assume the cheapest realistic packing (int4). Conservative: never over-award.
        _ => 0.5,
    }
}

/// The [`CostTier`] for an explicit `(params, quant)` — e.g. a model identity verified against a
/// log-prob reference. `None` when `params` is unparseable (the caller keeps its fallback).
pub fn cost_tier_for_params_quant(params: &str, quant: &str) -> Option<CostTier> {
    footprint_gb(params, quant).map(cost_tier_for_footprint)
}

/// Parse the **total** parameter count (in billions) from a canonical-id `params` component
/// (`35b-a3b`, `2b`, `0.5b`, `360m`, `e4b`), or `None` if it is not parseable (e.g. `"unknown"`).
///
/// * The leading number is the total scale; a trailing MoE `-a3b` (active params) is **ignored**
///   — footprint is by total params (all experts resident).
/// * A single leading letter (gemma's `E2B`/`E4B` "effective params") is skipped; the number is
///   taken as the billions count. These niche models are long-tail — a slightly-low estimate is
///   acceptable (conservative), consistent with "unrated long-tail → cost-to-serve only".
/// * Unit `b` ⇒ billions (×1); `m` ⇒ millions (×0.001).
pub fn total_params_billions(params: &str) -> Option<f64> {
    // Take the part before any MoE active-params suffix: "35b-a3b" → "35b".
    let head = params.split('-').next()?.trim();
    if head.is_empty() {
        return None;
    }
    let bytes = head.as_bytes();
    let mut i = 0;
    // Skip a single leading letter (gemma E2B/E4B effective-params marker).
    if bytes[i].is_ascii_alphabetic() {
        i += 1;
    }
    let num_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        i += 1;
    }
    if i == num_start {
        return None; // no numeric part → not a params token (e.g. "unknown")
    }
    let value: f64 = head[num_start..i].parse().ok()?;
    // The unit is whatever follows the number (`b` or `m`); default to billions if absent.
    let unit = head[i..].trim();
    let scale = match unit.to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "m" => 0.001,
        _ => return None, // an unexpected unit → let the caller fall back conservatively
    };
    let billions = value * scale;
    if billions > 0.0 && billions.is_finite() {
        Some(billions)
    } else {
        None
    }
}

/// A model's discrete **cost-tier** — a bucket of memory footprint (VRAM held), *not* the
/// router's `tier: u8` (a latency-vs-balanced scoring profile, unrelated to model size).
///
/// Buckets and multipliers are the v2.2 table (Decision #11): the multiplier ≈ the representative
/// GB of the bucket, so credit is roughly **linear in VRAM held**. The buckets are the shippable
/// v1 discretisation of the continuous target [`continuous_weight`] (`clamp(footprint_GB, min 1)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CostTier {
    /// ≤ 2 GB — e.g. `llama3.2:1b` Q8 (~1 GB), `tinyllama` Q4 (~0.6 GB).
    Nano,
    /// 2–6 GB — e.g. `qwen2.5:7b` Q4 (~3.5 GB), `llama3:8b` Q4 (~4 GB).
    Small,
    /// 6–18 GB — e.g. `qwen3:14b` Q4 (~7 GB), 8b FP16 (~16 GB), `qwen3-coder:30b-a3b` Q4 (~15 GB).
    Medium,
    /// 18–45 GB — e.g. 30b Q8 (~30 GB), 70b Q4 (~35 GB).
    Large,
    /// 45–110 GB — e.g. 70b Q8 (~70 GB), 120b Q4 (~60 GB).
    Frontier,
    /// > 110 GB — e.g. 405b Q4 (~200 GB), deepseek-671b, 235b Q8.
    Mega,
}

impl CostTier {
    /// The credit multiplier for this tier (Decision #11): `1 / 4 / 12 / 30 / 75 / 150×`,
    /// each ≈ the representative GB of the bucket.
    pub fn weight(self) -> f64 {
        match self {
            CostTier::Nano => 1.0,
            CostTier::Small => 4.0,
            CostTier::Medium => 12.0,
            CostTier::Large => 30.0,
            CostTier::Frontier => 75.0,
            CostTier::Mega => 150.0,
        }
    }

    /// Every tier, ascending (nano → mega). The canonical order for indexing / ascension.
    pub const ALL: [CostTier; 6] = [
        CostTier::Nano,
        CostTier::Small,
        CostTier::Medium,
        CostTier::Large,
        CostTier::Frontier,
        CostTier::Mega,
    ];

    /// This tier's position in [`ALL`](Self::ALL) (`nano = 0 … mega = 5`) — the "tier index" the
    /// `max_awardable_tier` ascension curve steps through (credit-tiering Phase 4).
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|&t| t == self).expect("every CostTier is in ALL") as u8
    }

    /// The tier at `index`, saturating at [`Mega`](Self::Mega) for an out-of-range index — so
    /// `floor_index + steps` can be clamped by construction without a bounds panic.
    pub fn from_index(index: u8) -> CostTier {
        *Self::ALL.get(index as usize).unwrap_or(&CostTier::Mega)
    }

    /// A stable lower-case label for display / telemetry / the disclosed cost-tier.
    pub fn label(self) -> &'static str {
        match self {
            CostTier::Nano => "nano",
            CostTier::Small => "small",
            CostTier::Medium => "medium",
            CostTier::Large => "large",
            CostTier::Frontier => "frontier",
            CostTier::Mega => "mega",
        }
    }
}

/// Bucket a memory footprint (GB) into its [`CostTier`]. Boundaries are inclusive-low / exclusive-
/// high (`2.0 GB` → nano, `2.0001 GB` → small), matching the table's `≤ 2 / 2–6 / …` ranges.
pub fn cost_tier_for_footprint(footprint_gb: f64) -> CostTier {
    match footprint_gb {
        f if f <= 2.0 => CostTier::Nano,
        f if f <= 6.0 => CostTier::Small,
        f if f <= 18.0 => CostTier::Medium,
        f if f <= 45.0 => CostTier::Large,
        f if f <= 110.0 => CostTier::Frontier,
        _ => CostTier::Mega,
    }
}

/// The continuous cost-weight target (Decision #11): `clamp(footprint_GB, min 1)` — credit
/// linear in VRAM held, floored at 1 so a sub-GB model still earns the base unit. This is the
/// calibration reference the discrete [`CostTier`] table approximates; **post-MVP** the ledger may
/// switch to it directly (plan Phase 8). Provided now so tests can anchor the buckets to it.
pub fn continuous_weight(footprint_gb: f64) -> f64 {
    footprint_gb.max(1.0)
}

/// The memory footprint (GB) of a `(params, quant)` pair, or `None` if `params` is unparseable.
/// `footprint_GB = total_params_billions × bytes_per_param(normalize_quant(quant))`. GB is decimal
/// (1e9 bytes), matching the plan's table figures.
pub fn footprint_gb(params: &str, quant: &str) -> Option<f64> {
    let billions = total_params_billions(params)?;
    Some(billions * bytes_per_param(&normalize_quant(quant)))
}

/// The [`CostTier`] of a parsed canonical model id — the ledger's lookup entry point.
///
/// Falls back to the **conservative** [`CostTier::Nano`] when `params` is missing/unparseable
/// (an unrated or malformed id never defaults high). `quant` degrades gracefully inside
/// [`bytes_per_param`] (unknown → int4 cost), so a concrete `params` with a novel quant still
/// tiers by footprint rather than collapsing to the floor.
pub fn cost_tier_for_model(id: &CanonicalModelId) -> CostTier {
    match footprint_gb(&id.params, &id.quant) {
        Some(gb) => cost_tier_for_footprint(gb),
        None => CostTier::Nano,
    }
}

/// The cost-**weight** multiplier of a parsed canonical model id (its [`CostTier::weight`]).
/// Shorthand for `cost_tier_for_model(id).weight()`.
pub fn weight_for_model(id: &CanonicalModelId) -> f64 {
    cost_tier_for_model(id).weight()
}

/// The [`CostTier`] of a raw model-id string. Parses the canonical form; a non-canonical or
/// unparseable id (e.g. a raw engine handle like `qwen2.5:7b`, or a legacy provider that advertised
/// no canonical id) falls back to the conservative [`CostTier::Nano`] — the ledger never awards a
/// high tier on the provider's unvalidated word. Pass the provider's **canonical** id when known.
pub fn cost_tier_for_model_id(model_id: &str) -> CostTier {
    match CanonicalModelId::parse(model_id) {
        Ok(id) => cost_tier_for_model(&id),
        Err(_) => CostTier::Nano,
    }
}

/// The cost-weight of a raw model-id string (its [`cost_tier_for_model_id`] weight). A non-canonical
/// or unparseable id falls back to the conservative [`CostTier::Nano`] weight (`1×`).
pub fn weight_for_model_id(model_id: &str) -> f64 {
    cost_tier_for_model_id(model_id).weight()
}

/// The [`CostTier`] implied by a **raw engine handle** (`qwen2.5:7b`, `llama3:70b`, an HF id, or a
/// `.gguf` path) — the *commitment-bound* identity the provider actually served and the consumer
/// validated (`validate_and_consume_commitment`), as opposed to the self-declared `canonical_model_id`
/// which no audit verifies. Used as the **ceiling** on cost-weighted credit until model-authenticity
/// fingerprinting (PROOF_OF_INFERENCE) lands: a provider can't inflate its tier by mislabelling the
/// canonical, because the handle caps it.
///
/// Heuristic and **conservative**: it extracts the parameter count from the handle (reusing the
/// catalog heuristic in [`crate::model_id::parse_hf_model_name`], treating an Ollama `:tag` like a
/// `-` so a size tag such as `7b` is seen), and — since a handle carries no quantisation — assumes
/// the **cheapest** packing (`int4`), so the footprint is a floor, never inflated. An unparseable
/// handle (a bare `:latest` tag, an opaque alias) → [`CostTier::Nano`].
pub fn cost_tier_for_handle(handle: &str) -> CostTier {
    // Strip any directory prefix (llama.cpp serves by file path) to the basename.
    let name = handle.rsplit(['/', '\\']).next().unwrap_or(handle);
    // Ollama handles are `name:tag`; treat ':' like '-' so a size tag (`7b`) reaches the parser.
    let normalized = name.replace(':', "-");
    let (_family, params, _variants) = crate::model_id::parse_hf_model_name(&normalized);
    match total_params_billions(&params) {
        // Quant is unknown from a handle → assume the cheapest (int4) so the tier is a floor.
        Some(billions) => cost_tier_for_footprint(billions * bytes_per_param("int4")),
        None => CostTier::Nano,
    }
}

/// The input/output-weighted token count for a request: `prompt·w_in + completion·w_out`
/// (Decision #10). The cost-tier multiplier is applied on top (see [`credits_for`]).
///
/// ⚠️ The prompt/completion split is **not in the co-signed receipt yet** (Phase 2 adds
/// `completion_tokens`); until then the ledger has only a total and must weight legacy receipts at
/// a neutral blended rate. This helper is the forward-looking primitive Phase 3 will call once the
/// split is signed.
pub fn weighted_token_units(prompt_tokens: u64, completion_tokens: u64) -> u64 {
    prompt_tokens.saturating_mul(W_IN).saturating_add(completion_tokens.saturating_mul(W_OUT))
}

/// The full credit for one receipt: `weight × (prompt·w_in + completion·w_out)`, the core model
/// that replaces the flat `credits = tokens`. Pure arithmetic over an already-resolved `weight`
/// (from [`weight_for_model`]) and the signed token split. Rounded down to whole credits.
pub fn credits_for(weight: f64, prompt_tokens: u64, completion_tokens: u64) -> u64 {
    (weight * weighted_token_units(prompt_tokens, completion_tokens) as f64) as u64
}

/// The **neutral blended** input/output weighting for a receipt whose prompt/completion split is
/// unknown — a legacy total-only receipt, or any receipt before Phase 2b populates the split. The
/// split can't be recovered, so weight every token at the midpoint of `w_in` and `w_out`
/// (`(1 + 4)/2 = 2.5×`): honest on average, and it never assumes the expensive all-output case.
/// When the true split *is* known, use [`weighted_token_units`] instead — it is exact.
pub fn blended_weighted_units(total_tokens: u64) -> u64 {
    total_tokens.saturating_mul(W_IN + W_OUT) / 2
}

/// The input/output-weighted token count for one receipt, using the **exact** split when known
/// (`Some(completion)`, with `prompt = total − completion`) and the [`blended_weighted_units`]
/// neutral rate when not (`None` — a legacy / pre-Phase-2b receipt). The cost-tier `weight`
/// multiplies this (see [`credits_for_receipt`]).
pub fn receipt_weighted_units(total_tokens: u64, completion_tokens: Option<u64>) -> u64 {
    match completion_tokens {
        Some(completion) => {
            let prompt = total_tokens.saturating_sub(completion);
            weighted_token_units(prompt, completion.min(total_tokens))
        }
        None => blended_weighted_units(total_tokens),
    }
}

/// The full credit for one co-signed receipt: `weight × receipt_weighted_units(total, completion)`
/// — the ledger's earn/spend amount (Phase 3). `weight` is the applicable [`CostTier::weight`]
/// (the *claimed* tier for a spend; the gated *awarded* tier for a withdrawable earn — see
/// [`crate::award`]). Whole credits, rounded down.
pub fn credits_for_receipt(weight: f64, total_tokens: u64, completion_tokens: Option<u64>) -> u64 {
    (weight * receipt_weighted_units(total_tokens, completion_tokens) as f64) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_k_quants_are_priced_by_their_bit_width() {
        assert_eq!(bytes_per_param(&normalize_quant("Q6_K")), 6.5 / 8.0);
        assert_eq!(bytes_per_param(&normalize_quant("Q5_K_M")), 5.5 / 8.0);
        assert_eq!(bytes_per_param(&normalize_quant("Q3_K_M")), 3.5 / 8.0);
        assert_eq!(bytes_per_param(&normalize_quant("Q4_K_M")), 0.5); // folded to int4
        assert_eq!(bytes_per_param(&normalize_quant("Q8_0")), 1.0); // folded to int8
        assert_eq!(bytes_per_param("iq4_xs"), 0.5); // unknown stays conservative
        assert_eq!(cost_tier_for_params_quant("9b", "bf16"), Some(cost_tier_for_footprint(18.0)));
        assert!(cost_tier_for_params_quant("9b", "q6_k") > cost_tier_for_params_quant("9b", "q4_k_m"));
        assert_eq!(cost_tier_for_params_quant("unknown", "bf16"), None);
    }

    fn id(params: &str, quant: &str) -> CanonicalModelId {
        CanonicalModelId {
            family: "test".into(),
            params: params.into(),
            quant: quant.into(),
            template_hash: "deadbeefdeadbeef".into(),
        }
    }

    // ── bytes_per_param ──

    #[test]
    fn bytes_per_param_covers_the_common_quants() {
        assert_eq!(bytes_per_param("fp32"), 4.0);
        assert_eq!(bytes_per_param("fp16"), 2.0);
        assert_eq!(bytes_per_param("bf16"), 2.0);
        assert_eq!(bytes_per_param("fp8"), 1.0);
        assert_eq!(bytes_per_param("int8"), 1.0);
        assert_eq!(bytes_per_param("int4"), 0.5);
    }

    #[test]
    fn unknown_quant_is_conservative_not_inflated() {
        // A novel/unparseable quant must assume the CHEAPEST packing, never a big one — the
        // registry never lets an unknown label inflate footprint.
        assert_eq!(bytes_per_param("int2"), 0.5);
        assert_eq!(bytes_per_param("some-future-quant"), 0.5);
        assert!(bytes_per_param("mystery") <= bytes_per_param("fp16"));
    }

    // ── total_params_billions ──

    #[test]
    fn parses_billions_and_millions() {
        assert_eq!(total_params_billions("2b"), Some(2.0));
        assert_eq!(total_params_billions("0.5b"), Some(0.5));
        assert_eq!(total_params_billions("70b"), Some(70.0));
        assert_eq!(total_params_billions("360m"), Some(0.36));
        assert_eq!(total_params_billions("15m"), Some(0.015));
    }

    #[test]
    fn moe_uses_total_params_not_active() {
        // qwen3-coder:30b-a3b → 30B total resident (all experts), NOT 3B active.
        assert_eq!(total_params_billions("30b-a3b"), Some(30.0));
        assert_eq!(total_params_billions("235b-a22b"), Some(235.0));
    }

    #[test]
    fn skips_gemma_effective_params_leading_letter() {
        assert_eq!(total_params_billions("e4b"), Some(4.0));
        assert_eq!(total_params_billions("e2b"), Some(2.0));
    }

    #[test]
    fn unparseable_params_return_none() {
        assert_eq!(total_params_billions("unknown"), None);
        assert_eq!(total_params_billions(""), None);
        assert_eq!(total_params_billions("b"), None);
    }

    // ── footprint + tiering: anchored to the v2.2 plan table ──

    #[test]
    fn footprint_matches_the_plan_table() {
        // (params, quant) → footprint GB, from CREDIT_TIERING_PLAN_v2 Decision #11.
        assert_eq!(footprint_gb("1b", "q8_0"), Some(1.0)); // llama3.2:1b Q8 ~1 GB
        assert_eq!(footprint_gb("7b", "q4_k_m"), Some(3.5)); // qwen2.5:7b Q4 ~3.5 GB
        assert_eq!(footprint_gb("8b", "q4_k_m"), Some(4.0)); // llama3:8b Q4 ~4 GB
        assert_eq!(footprint_gb("8b", "fp16"), Some(16.0)); // 8b FP16 ~16 GB
        assert_eq!(footprint_gb("30b-a3b", "int4"), Some(15.0)); // qwen3-coder MoE ~15 GB
        assert_eq!(footprint_gb("30b", "int8"), Some(30.0)); // 30b Q8 ~30 GB
        assert_eq!(footprint_gb("70b", "int4"), Some(35.0)); // 70b Q4 ~35 GB
        assert_eq!(footprint_gb("70b", "int8"), Some(70.0)); // 70b Q8 ~70 GB
        assert_eq!(footprint_gb("405b", "int4"), Some(202.5)); // 405b Q4 ~200 GB
    }

    #[test]
    fn tiers_match_the_plan_table() {
        let tier = |p: &str, q: &str| cost_tier_for_model(&id(p, q));
        assert_eq!(tier("1b", "q8_0"), CostTier::Nano);
        assert_eq!(tier("7b", "q4_k_m"), CostTier::Small);
        assert_eq!(tier("8b", "q4_k_m"), CostTier::Small);
        assert_eq!(tier("8b", "fp16"), CostTier::Medium);
        assert_eq!(tier("30b-a3b", "int4"), CostTier::Medium);
        assert_eq!(tier("30b", "int8"), CostTier::Large);
        assert_eq!(tier("70b", "int4"), CostTier::Large);
        assert_eq!(tier("70b", "int8"), CostTier::Frontier);
        assert_eq!(tier("405b", "int4"), CostTier::Mega);
    }

    #[test]
    fn quant_correction_is_real_same_params_different_vram() {
        // The headline the GB basis buys: a 70b-Q4 earns LESS than a 70b-Q8, which a
        // param-only table could never express.
        let q4 = weight_for_model(&id("70b", "int4")); // large  → 30×
        let q8 = weight_for_model(&id("70b", "int8")); // frontier → 75×
        assert!(q4 < q8, "70b-Q4 (35 GB) must weigh less than 70b-Q8 (70 GB)");
        assert_eq!(q4, 30.0);
        assert_eq!(q8, 75.0);
    }

    #[test]
    fn tier_index_roundtrips_and_orders_ascending() {
        for (i, &t) in CostTier::ALL.iter().enumerate() {
            assert_eq!(t.index() as usize, i);
            assert_eq!(CostTier::from_index(i as u8), t);
        }
        assert!(CostTier::Nano.index() < CostTier::Mega.index());
        assert_eq!(CostTier::from_index(99), CostTier::Mega, "out-of-range saturates at mega");
    }

    #[test]
    fn tier_multipliers_are_the_locked_numbers() {
        assert_eq!(CostTier::Nano.weight(), 1.0);
        assert_eq!(CostTier::Small.weight(), 4.0);
        assert_eq!(CostTier::Medium.weight(), 12.0);
        assert_eq!(CostTier::Large.weight(), 30.0);
        assert_eq!(CostTier::Frontier.weight(), 75.0);
        assert_eq!(CostTier::Mega.weight(), 150.0);
    }

    #[test]
    fn tier_boundaries_are_inclusive_low_exclusive_high() {
        assert_eq!(cost_tier_for_footprint(2.0), CostTier::Nano);
        assert_eq!(cost_tier_for_footprint(2.0001), CostTier::Small);
        assert_eq!(cost_tier_for_footprint(6.0), CostTier::Small);
        assert_eq!(cost_tier_for_footprint(18.0), CostTier::Medium);
        assert_eq!(cost_tier_for_footprint(45.0), CostTier::Large);
        assert_eq!(cost_tier_for_footprint(110.0), CostTier::Frontier);
        assert_eq!(cost_tier_for_footprint(110.01), CostTier::Mega);
    }

    #[test]
    fn tier_multiplier_tracks_the_representative_gb() {
        // Each tier's multiplier is ≈ the representative GB of its bucket, so the discrete table
        // stays within a small factor of the continuous target across the range.
        for (gb, tier) in [(1.0, CostTier::Nano), (4.0, CostTier::Small), (12.0, CostTier::Medium),
            (30.0, CostTier::Large), (70.0, CostTier::Frontier), (150.0, CostTier::Mega)] {
            assert_eq!(cost_tier_for_footprint(gb), tier);
            let ratio = tier.weight() / continuous_weight(gb);
            assert!((0.5..=2.0).contains(&ratio), "tier {} weight {} vs GB {}", tier.label(), tier.weight(), gb);
        }
    }

    // ── conservative fallback (the anti-gaming floor) ──

    #[test]
    fn unknown_model_falls_back_to_nano_never_high() {
        // The load-bearing safety property: an unparseable/unrated model defaults to the FLOOR.
        assert_eq!(cost_tier_for_model(&id("unknown", "fp16")), CostTier::Nano);
        assert_eq!(weight_for_model(&id("unknown", "int4")), 1.0);
        // A non-canonical raw id → floor.
        assert_eq!(weight_for_model_id("just-some-alias"), 1.0);
        assert_eq!(weight_for_model_id(""), 1.0);
    }

    #[test]
    fn weight_for_model_id_parses_canonical_form() {
        assert_eq!(weight_for_model_id("qwen2.5/7b/q4_k_m/abcd0123abcd0123"), 4.0); // small
        assert_eq!(weight_for_model_id("llama/70b/int8/abcd0123abcd0123"), 75.0); // frontier
    }

    #[test]
    fn cost_tier_for_handle_extracts_size_conservatively() {
        // Ollama `name:tag` — the size tag is read; quant assumed int4 (cheapest → conservative).
        assert_eq!(cost_tier_for_handle("qwen2.5:7b"), CostTier::Small); // 7×0.5=3.5GB
        assert_eq!(cost_tier_for_handle("llama3:70b"), CostTier::Large); // 70×0.5=35GB
        assert_eq!(cost_tier_for_handle("llama3.2:1b"), CostTier::Nano); // 1×0.5=0.5GB
        // A `.gguf` path — basename parsed, size extracted.
        assert_eq!(cost_tier_for_handle("/home/u/Qwen3.5-9B-UD-Q4_K_XL.gguf"), CostTier::Small);
        // No parseable size → conservative nano (never inflate on an opaque handle).
        assert_eq!(cost_tier_for_handle("tinyllama:latest"), CostTier::Nano);
        assert_eq!(cost_tier_for_handle("some-alias"), CostTier::Nano);
        assert_eq!(cost_tier_for_handle(""), CostTier::Nano);
    }

    #[test]
    fn handle_tier_caps_a_mislabelled_canonical() {
        // The H1 defence: a provider serving a 7b handle can't earn the 70b rate by declaring a big
        // canonical — the handle-implied tier is the ceiling.
        let handle_tier = cost_tier_for_handle("qwen2.5:7b"); // Small (4×)
        let claimed = cost_tier_for_model_id("qwen2.5/70b/int8/abcd0123abcd0123"); // Frontier (75×)
        let capped = claimed.min(handle_tier);
        assert_eq!(capped, CostTier::Small, "handle caps the inflated canonical claim");
    }

    // ── input/output weighting + credit math ──

    #[test]
    fn output_tokens_weigh_four_times_input() {
        assert_eq!(W_IN, 1);
        assert_eq!(W_OUT, 4);
        // 100 prompt + 100 completion = 100·1 + 100·4 = 500 weighted units.
        assert_eq!(weighted_token_units(100, 100), 500);
        assert_eq!(weighted_token_units(0, 0), 0);
    }

    #[test]
    fn credits_for_applies_tier_then_io_split() {
        // A frontier (75×) request of 100 in / 100 out = 75 × 500 = 37_500 credits.
        assert_eq!(credits_for(75.0, 100, 100), 37_500);
        // The same token counts on a nano model earn 1 × 500 = 500 — the 75× spread the plan wants.
        assert_eq!(credits_for(1.0, 100, 100), 500);
    }

    #[test]
    fn weighted_units_saturate_rather_than_overflow() {
        // Defensive: a pathological token count must not panic on multiply/add overflow.
        assert_eq!(weighted_token_units(u64::MAX, u64::MAX), u64::MAX);
    }

    #[test]
    fn blended_rate_is_the_in_out_midpoint() {
        // Split unknown → weight every token at (1 + 4)/2 = 2.5×.
        assert_eq!(blended_weighted_units(100), 250);
        assert_eq!(blended_weighted_units(0), 0);
    }

    #[test]
    fn receipt_units_use_exact_split_when_known_else_blended() {
        // Known split: 40 prompt + 60 completion = 40·1 + 60·4 = 280.
        assert_eq!(receipt_weighted_units(100, Some(60)), 280);
        // Unknown split → blended 2.5× = 250.
        assert_eq!(receipt_weighted_units(100, None), 250);
        // A completion larger than the total is clamped (defensive against a bad receipt).
        assert_eq!(receipt_weighted_units(10, Some(999)), weighted_token_units(0, 10));
    }

    #[test]
    fn credits_for_receipt_multiplies_tier_weight_by_units() {
        // Frontier (75×), 100 tokens, unknown split (blended 250 units) → 75 × 250 = 18_750.
        assert_eq!(credits_for_receipt(CostTier::Frontier.weight(), 100, None), 18_750);
        // Same tokens with a known 60-completion split (280 units) → 75 × 280 = 21_000.
        assert_eq!(credits_for_receipt(CostTier::Frontier.weight(), 100, Some(60)), 21_000);
        // Nano (1×) blended → 250.
        assert_eq!(credits_for_receipt(CostTier::Nano.weight(), 100, None), 250);
    }
}
