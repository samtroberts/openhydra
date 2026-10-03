// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Qwen3 (dense), affine-quantized MLX weights, computed exactly as mlx_lm 0.31 does — verified
//! bit-identical to mlx_lm (as LM Studio's MLX engine runs it) on 26,429 positions.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::error::Exception;
use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::ops;
use mlx_rs::Array;
use serde::Deserialize;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct Quantization {
    group_size: i32,
    bits: i32,
}

#[derive(Deserialize)]
struct Config {
    model_type: String,
    hidden_size: i32,
    num_hidden_layers: usize,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    rms_norm_eps: f32,
    rope_theta: f32,
    tie_word_embeddings: bool,
    quantization: Quantization,
}

/// A quantized (affine) weight: packed values, per-group scales and biases.
struct QWeight {
    w: Array,
    scales: Array,
    biases: Array,
}

struct Layer {
    input_norm: Array,
    post_norm: Array,
    q: QWeight,
    k: QWeight,
    v: QWeight,
    o: QWeight,
    q_norm: Array,
    k_norm: Array,
    gate: QWeight,
    up: QWeight,
    down: QWeight,
}

/// `silu(gate) * up` as one compiled (fused) kernel, as mlx_lm runs it: `@mx.compile(shapeless=True)`
/// on `swiglu` and on `nn.silu`. Unfused, the intermediate rounding differs (~0.03 nats per token).
type Swiglu = Box<dyn for<'a> FnMut((&'a Array, &'a Array)) -> std::result::Result<Array, Exception>>;

fn compiled_swiglu() -> Swiglu {
    Box::new(mlx_rs::transforms::compile::compile(
        |(gate, up): (&Array, &Array)| -> std::result::Result<Array, Exception> { gate.multiply(ops::sigmoid(gate)?)?.multiply(up) },
        true,
    ))
}

pub struct Model {
    swiglu: Swiglu,
    cfg: Config,
    embed: QWeight,
    layers: Vec<Layer>,
    norm: Array,
    lm_head: Option<QWeight>,
}

fn load_weights(dir: &Path) -> Result<HashMap<String, Array>> {
    let index = dir.join("model.safetensors.index.json");
    let files: Vec<String> = if index.is_file() {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
        let mut f: Vec<String> = v["weight_map"].as_object().ok_or("index without weight_map")?.values().filter_map(|x| x.as_str().map(String::from)).collect();
        f.sort();
        f.dedup();
        f
    } else {
        vec!["model.safetensors".into()]
    };
    let mut all = HashMap::new();
    for f in files {
        all.extend(Array::load_safetensors(dir.join(&f))?);
    }
    Ok(all)
}

impl Model {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        if cfg.model_type != "qwen3" {
            return Err(format!("unsupported model_type {}", cfg.model_type).into());
        }
        let mut w = load_weights(dir)?;
        let mut take = |k: &str| w.remove(k).ok_or_else(|| format!("missing tensor {k}"));
        let mut q = |p: &str| -> Result<QWeight> {
            Ok(QWeight { w: take(&format!("{p}.weight"))?, scales: take(&format!("{p}.scales"))?, biases: take(&format!("{p}.biases"))? })
        };
        let embed = q("model.embed_tokens")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                q: q(&format!("{p}.self_attn.q_proj"))?,
                k: q(&format!("{p}.self_attn.k_proj"))?,
                v: q(&format!("{p}.self_attn.v_proj"))?,
                o: q(&format!("{p}.self_attn.o_proj"))?,
                gate: q(&format!("{p}.mlp.gate_proj"))?,
                up: q(&format!("{p}.mlp.up_proj"))?,
                down: q(&format!("{p}.mlp.down_proj"))?,
                input_norm: Array::from_slice::<f32>(&[], &[0]), // replaced below
                post_norm: Array::from_slice::<f32>(&[], &[0]),
                q_norm: Array::from_slice::<f32>(&[], &[0]),
                k_norm: Array::from_slice::<f32>(&[], &[0]),
            });
        }
        for (i, l) in layers.iter_mut().enumerate() {
            let p = format!("model.layers.{i}");
            l.input_norm = take(&format!("{p}.input_layernorm.weight"))?;
            l.post_norm = take(&format!("{p}.post_attention_layernorm.weight"))?;
            l.q_norm = take(&format!("{p}.self_attn.q_norm.weight"))?;
            l.k_norm = take(&format!("{p}.self_attn.k_norm.weight"))?;
        }
        let norm = take("model.norm.weight")?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(QWeight { w: take("lm_head.weight")?, scales: take("lm_head.scales")?, biases: take("lm_head.biases")? })
        };
        Ok(Self { swiglu: compiled_swiglu(), cfg, embed, layers, norm, lm_head })
    }

    fn qmm(&self, x: &Array, q: &QWeight) -> std::result::Result<Array, Exception> {
        ops::quantized_matmul(x, &q.w, &q.scales, &q.biases, true, self.cfg.quantization.group_size, self.cfg.quantization.bits)
    }

    /// Log-softmax logits for every position of `ids`, as `[L, vocab]` f32.
    pub fn log_probs(&mut self, ids: &[u32]) -> std::result::Result<Array, Exception> {
        let c = &self.cfg;
        let (gs, bits) = (c.quantization.group_size, c.quantization.bits);
        let l = ids.len() as i32;
        let idx = Array::from_slice(ids, &[l]);
        // Embedding lookup = dequantize the selected rows (QuantizedEmbedding.__call__).
        let e = ops::dequantize(&self.embed.w.take_axis(&idx, 0)?, &self.embed.scales.take_axis(&idx, 0)?, &self.embed.biases.take_axis(&idx, 0)?, gs, bits)?;
        let mut x = e.reshape(&[1, l, c.hidden_size])?;
        let scale = (c.head_dim as f32).powf(-0.5);
        for layer in &self.layers {
            let h = fast::rms_norm(&x, Some(&layer.input_norm), c.rms_norm_eps)?;
            let q = self.qmm(&h, &layer.q)?.reshape(&[1, l, c.num_attention_heads, c.head_dim])?;
            let k = self.qmm(&h, &layer.k)?.reshape(&[1, l, c.num_key_value_heads, c.head_dim])?;
            let v = self.qmm(&h, &layer.v)?.reshape(&[1, l, c.num_key_value_heads, c.head_dim])?;
            let q = fast::rms_norm(&q, Some(&layer.q_norm), c.rms_norm_eps)?.transpose_axes(&[0, 2, 1, 3])?;
            let k = fast::rms_norm(&k, Some(&layer.k_norm), c.rms_norm_eps)?.transpose_axes(&[0, 2, 1, 3])?;
            let v = v.transpose_axes(&[0, 2, 1, 3])?;
            let q = fast::rope(&q, c.head_dim, false, c.rope_theta, 1.0, 0, None)?;
            let k = fast::rope(&k, c.head_dim, false, c.rope_theta, 1.0, 0, None)?;
            let o = fast::scaled_dot_product_attention(&q, &k, &v, scale, ScaledDotProductAttentionMask::Causal, None)?;
            let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, l, c.num_attention_heads * c.head_dim])?;
            let h2 = x.add(self.qmm(&o, &layer.o)?)?;
            let m = fast::rms_norm(&h2, Some(&layer.post_norm), c.rms_norm_eps)?;
            let g = self.qmm(&m, &layer.gate)?;
            let u = self.qmm(&m, &layer.up)?;
            let a = (self.swiglu)((&g, &u))?;
            x = h2.add(self.qmm(&a, &layer.down)?)?;
        }
        let x = fast::rms_norm(&x, Some(&self.norm), c.rms_norm_eps)?;
        let logits = self.qmm(&x, self.lm_head.as_ref().unwrap_or(&self.embed))?.reshape(&[l, -1])?.as_type::<f32>()?;
        let lse = logits.logsumexp_axis(-1, true)?;
        logits.subtract(&lse)
    }

    /// Vocabulary size (rows of the token embedding).
    pub fn vocab_size(&self) -> usize {
        self.embed.w.shape()[0] as usize
    }
}
