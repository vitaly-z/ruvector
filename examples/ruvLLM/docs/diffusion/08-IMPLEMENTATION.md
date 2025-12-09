# Implementation Plan

## Overview

This document provides a detailed step-by-step implementation plan for RuvDLLM. The plan is organized into phases with clear milestones, dependencies, and deliverables.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Implementation Timeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Foundation ────────────────────────────────────────────►          │
│  │  • Core diffusion model                                                  │
│  │  • Q4 quantization                                                       │
│  │  • MDLM sampler                                                          │
│  │                                                                          │
│  Phase 2: Novel Contributions ───────────────────────────────────►          │
│  │  • TALoRA implementation                                                 │
│  │  • DGR implementation                                                    │
│  │  • RuVector integration                                                  │
│  │                                                                          │
│  Phase 3: Federation ────────────────────────────────────────────►          │
│  │  • DAF protocol                                                          │
│  │  • Privacy tiers                                                         │
│  │  • Gossip sync                                                           │
│  │                                                                          │
│  Phase 4: Optimization ──────────────────────────────────────────►          │
│  │  • GPU kernels                                                           │
│  │  • Benchmarking                                                          │
│  │  • Production hardening                                                  │
│  │                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Foundation

### 1.1 Project Setup

```bash
# Create module structure
mkdir -p examples/ruvLLM/src/diffusion
mkdir -p examples/ruvLLM/src/federation
mkdir -p examples/ruvLLM/src/micro_lora
mkdir -p examples/ruvLLM/src/gpu
```

**Files to create:**

```
examples/ruvLLM/src/
├── diffusion/
│   ├── mod.rs           # Module exports
│   ├── model.rs         # Diffusion model struct
│   ├── sampler.rs       # MDLM/BD3LM samplers
│   ├── scheduler.rs     # Noise schedules
│   └── simd/
│       ├── mod.rs
│       ├── denoise.rs
│       └── attention.rs
├── federation/
│   ├── mod.rs
│   └── (empty initially)
├── micro_lora/
│   ├── mod.rs
│   └── (empty initially)
└── gpu/
    ├── mod.rs
    └── (empty initially)
```

**Cargo.toml additions:**

```toml
[features]
default = ["simd"]
simd = []
cuda = ["cudarc"]
metal = ["metal-rs"]
vulkan = ["ash", "gpu-allocator"]
federation = ["tokio", "quinn"]

[dependencies]
# Existing dependencies...

# New dependencies for diffusion
half = "2.3"  # f16 support
bytemuck = "1.14"  # Safe casting
memmap2 = "0.9"  # Memory-mapped model loading

[target.'cfg(target_arch = "x86_64")'.dependencies]
# AVX2/AVX-512 intrinsics (part of std)

[target.'cfg(target_arch = "aarch64")'.dependencies]
# NEON intrinsics (part of std)
```

### 1.2 Core Diffusion Model

**File: `src/diffusion/model.rs`**

```rust
//! Core diffusion language model with Q4 quantization

use crate::simd_inference::{Q4Weights, dot_product_avx2};

/// Quantized diffusion language model
pub struct DiffusionModel {
    /// Model configuration
    pub config: DiffusionConfig,
    /// Embedding layer
    pub embed_tokens: Embedding,
    /// Transformer blocks
    pub layers: Vec<TransformerBlock>,
    /// Output projection (tied with embeddings)
    pub lm_head: Option<Linear>,
    /// Time embedding MLP
    pub time_embed: TimeEmbedding,
    /// Noise schedule
    pub schedule: NoiseSchedule,
}

#[derive(Clone)]
pub struct DiffusionConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Diffusion-specific
    pub num_timesteps: u32,
    pub noise_schedule: NoiseScheduleType,
}

impl DiffusionModel {
    /// Load from quantized safetensors
    pub fn load_q4(path: &Path) -> Result<Self, ModelError> {
        // Memory-map the file
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Parse safetensors header
        let tensors = SafeTensors::deserialize(&mmap)?;

        // Load each component
        let config = Self::load_config(&tensors)?;
        let embed_tokens = Self::load_embedding(&tensors, "embed_tokens")?;
        let layers = Self::load_layers(&tensors, config.num_hidden_layers)?;
        let time_embed = TimeEmbedding::new(config.hidden_size);
        let schedule = NoiseSchedule::new(config.noise_schedule, config.num_timesteps);

        Ok(Self {
            config,
            embed_tokens,
            layers,
            lm_head: None, // Tied with embed_tokens
            time_embed,
            schedule,
        })
    }

    /// Forward pass with timestep conditioning
    pub fn forward(
        &self,
        input_ids: &[u32],
        timestep: u32,
        kv_cache: Option<&mut KVCache>,
    ) -> Tensor {
        // 1. Token embeddings
        let mut hidden = self.embed_tokens.forward(input_ids);

        // 2. Add time embedding
        let time_emb = self.time_embed.forward(timestep);
        hidden.add_broadcast(&time_emb);

        // 3. Transformer blocks (bidirectional attention)
        for layer in &self.layers {
            hidden = layer.forward(&hidden, kv_cache, AttentionMask::Bidirectional);
        }

        // 4. Output logits
        self.output_projection(&hidden)
    }
}
```

### 1.3 MDLM Sampler

**File: `src/diffusion/sampler.rs`**

```rust
//! MDLM (Masked Diffusion Language Model) sampler

pub struct MDLMSampler {
    /// Number of denoising steps
    pub num_steps: u32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p for nucleus sampling
    pub top_p: f32,
    /// Noise schedule
    schedule: NoiseSchedule,
}

impl MDLMSampler {
    /// Generate text via iterative denoising
    pub fn generate(
        &self,
        model: &DiffusionModel,
        prompt_ids: &[u32],
        max_length: usize,
        adapters: Option<&[&MicroLoRA]>,
    ) -> Vec<u32> {
        // Initialize with noise
        let mut x_t = self.initialize_noisy(prompt_ids, max_length);

        // Iterative denoising
        for step in 0..self.num_steps {
            let t = self.num_steps - step - 1;
            let timestep = self.schedule.timestep_at_step(t);

            // Model prediction
            let logits = model.forward(&x_t, timestep, None);

            // Apply adapters if present
            let logits = if let Some(adapters) = adapters {
                self.apply_adapters(&logits, adapters, timestep)
            } else {
                logits
            };

            // Sample next state
            x_t = self.denoise_step(&x_t, &logits, timestep);
        }

        x_t
    }

    /// Single denoising step
    fn denoise_step(
        &self,
        x_t: &[u32],
        logits: &Tensor,
        timestep: u32,
    ) -> Vec<u32> {
        let alpha_t = self.schedule.alpha(timestep);
        let alpha_t_prev = self.schedule.alpha(timestep.saturating_sub(1));

        let mut result = Vec::with_capacity(x_t.len());

        for (i, &token) in x_t.iter().enumerate() {
            // Compute transition probabilities
            let token_logits = logits.slice(i);
            let probs = softmax_with_temperature(&token_logits, self.temperature);

            // MDLM update rule
            if self.should_update(token, alpha_t, alpha_t_prev) {
                // Sample new token
                let new_token = self.sample_token(&probs);
                result.push(new_token);
            } else {
                // Keep current token
                result.push(token);
            }
        }

        result
    }

    /// Determine if token should be updated at this step
    fn should_update(&self, token: u32, alpha_t: f32, alpha_t_prev: f32) -> bool {
        // MDLM: unmask tokens according to schedule
        let mask_token = MASK_TOKEN_ID;

        if token == mask_token {
            // Masked token: unmask with probability based on schedule
            let unmask_prob = (alpha_t_prev - alpha_t) / (1.0 - alpha_t);
            rand::random::<f32>() < unmask_prob
        } else {
            // Already unmasked: small probability of re-masking for correction
            false
        }
    }
}
```

### 1.4 Noise Schedule

**File: `src/diffusion/scheduler.rs`**

```rust
//! Noise schedules for diffusion

pub enum NoiseScheduleType {
    Linear,
    Cosine,
    Sqrt,
    Custom(Vec<f32>),
}

pub struct NoiseSchedule {
    /// Alpha values (signal retention)
    pub alphas: Vec<f32>,
    /// Sigma values (noise level)
    pub sigmas: Vec<f32>,
    /// Cumulative alpha products
    pub alpha_cumprod: Vec<f32>,
    /// Number of timesteps
    pub num_timesteps: u32,
}

impl NoiseSchedule {
    pub fn new(schedule_type: NoiseScheduleType, num_timesteps: u32) -> Self {
        let betas = match schedule_type {
            NoiseScheduleType::Linear => Self::linear_schedule(num_timesteps),
            NoiseScheduleType::Cosine => Self::cosine_schedule(num_timesteps),
            NoiseScheduleType::Sqrt => Self::sqrt_schedule(num_timesteps),
            NoiseScheduleType::Custom(betas) => betas,
        };

        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        let alpha_cumprod = Self::cumulative_product(&alphas);
        let sigmas: Vec<f32> = alpha_cumprod.iter().map(|a| (1.0 - a).sqrt()).collect();

        Self {
            alphas,
            sigmas,
            alpha_cumprod,
            num_timesteps,
        }
    }

    /// Cosine schedule (better for text)
    fn cosine_schedule(num_timesteps: u32) -> Vec<f32> {
        let s = 0.008; // Small offset to prevent singularity
        let steps: Vec<f32> = (0..=num_timesteps)
            .map(|i| i as f32 / num_timesteps as f32)
            .collect();

        let alphas_cumprod: Vec<f32> = steps
            .iter()
            .map(|t| {
                let angle = (t + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2;
                angle.cos().powi(2)
            })
            .collect();

        // Convert to betas
        let mut betas = Vec::with_capacity(num_timesteps as usize);
        for i in 1..=num_timesteps as usize {
            let beta = 1.0 - alphas_cumprod[i] / alphas_cumprod[i - 1];
            betas.push(beta.clamp(0.0001, 0.999));
        }

        betas
    }

    /// Get noise level at timestep
    pub fn alpha(&self, timestep: u32) -> f32 {
        self.alpha_cumprod[timestep as usize]
    }

    pub fn sigma(&self, timestep: u32) -> f32 {
        self.sigmas[timestep as usize]
    }
}
```

### 1.5 SIMD Denoising Kernels

**File: `src/diffusion/simd/denoise.rs`**

```rust
//! SIMD-optimized denoising operations

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD denoising step
#[target_feature(enable = "avx2,fma")]
pub unsafe fn denoise_step_avx2(
    x_t: &[f32],           // Current noisy state
    predicted: &[f32],     // Model prediction
    alpha: f32,            // Signal retention
    sigma: f32,            // Noise level
    output: &mut [f32],    // Denoised output
) {
    let n = x_t.len();
    assert_eq!(n, predicted.len());
    assert_eq!(n, output.len());

    let alpha_v = _mm256_set1_ps(alpha);
    let sigma_v = _mm256_set1_ps(sigma);
    let inv_alpha_v = _mm256_set1_ps(1.0 / alpha);

    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;

        // Load
        let x = _mm256_loadu_ps(x_t.as_ptr().add(offset));
        let pred = _mm256_loadu_ps(predicted.as_ptr().add(offset));

        // x_{t-1} = (x_t - sigma * pred) / alpha
        let noise_scaled = _mm256_mul_ps(sigma_v, pred);
        let denoised = _mm256_sub_ps(x, noise_scaled);
        let result = _mm256_mul_ps(denoised, inv_alpha_v);

        // Store
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        output[i] = (x_t[i] - sigma * predicted[i]) / alpha;
    }
}

/// SIMD-optimized attention for bidirectional diffusion
#[target_feature(enable = "avx2,fma")]
pub unsafe fn bidirectional_attention_avx2(
    q: &[f32],             // [seq_len, head_dim]
    k: &[f32],             // [seq_len, head_dim]
    v: &[f32],             // [seq_len, head_dim]
    seq_len: usize,
    head_dim: usize,
    output: &mut [f32],    // [seq_len, head_dim]
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scale_v = _mm256_set1_ps(scale);

    // Compute attention scores
    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];

        // Compute attention weights for this query
        let mut scores = vec![0.0f32; seq_len];
        let mut max_score = f32::NEG_INFINITY;

        for j in 0..seq_len {
            // No mask - bidirectional attention
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            let dot = dot_product_avx2(q_row, k_row);
            let score = dot * scale;
            scores[j] = score;
            max_score = max_score.max(score);
        }

        // Softmax
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }

        // Weighted sum of values
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        out_row.fill(0.0);

        for j in 0..seq_len {
            let v_row = &v[j * head_dim..(j + 1) * head_dim];
            let weight = scores[j];

            // out += weight * v
            let weight_v = _mm256_set1_ps(weight);
            let chunks = head_dim / 8;
            for c in 0..chunks {
                let offset = c * 8;
                let v_chunk = _mm256_loadu_ps(v_row.as_ptr().add(offset));
                let out_chunk = _mm256_loadu_ps(out_row.as_ptr().add(offset));
                let result = _mm256_fmadd_ps(weight_v, v_chunk, out_chunk);
                _mm256_storeu_ps(out_row.as_mut_ptr().add(offset), result);
            }
        }
    }
}
```

## Phase 2: Novel Contributions

### 2.1 TALoRA Implementation

**File: `src/micro_lora/talora.rs`**

```rust
//! TALoRA: Timestep-Aware LoRA

use super::{MicroLoRA, LoRABank};

/// TALoRA manager with timestep-specific adapter banks
pub struct TALoRAManager {
    /// Adapter banks for each timestep group
    /// Index 0: Coarse (t=700-1000), Index 1: Domain (t=300-700), Index 2: Fine (t=0-300)
    banks: [LoRABank; 3],
    /// Timestep boundaries
    boundaries: [u32; 2],
    /// Transition smoothing width
    transition_width: u32,
    /// RuVector index for each bank
    indices: [HNSWIndex; 3],
}

impl TALoRAManager {
    pub fn new(config: TALoRAConfig) -> Self {
        Self {
            banks: [
                LoRABank::new(config.coarse_capacity, config.coarse_rank),
                LoRABank::new(config.domain_capacity, config.domain_rank),
                LoRABank::new(config.fine_capacity, config.fine_rank),
            ],
            boundaries: [700, 300],
            transition_width: config.transition_width,
            indices: [
                HNSWIndex::new(config.hidden_dim, 16, 200),
                HNSWIndex::new(config.hidden_dim, 16, 200),
                HNSWIndex::new(config.hidden_dim, 16, 200),
            ],
        }
    }

    /// Get bank index for timestep
    pub fn get_bank_index(&self, timestep: u32) -> usize {
        if timestep > self.boundaries[0] {
            0 // Coarse
        } else if timestep > self.boundaries[1] {
            1 // Domain
        } else {
            2 // Fine
        }
    }

    /// Retrieve adapters for timestep with smooth blending
    pub fn retrieve(
        &self,
        query: &[f32],
        timestep: u32,
        top_k: usize,
    ) -> TALoRARetrievalResult {
        let bank_idx = self.get_bank_index(timestep);
        let primary_results = self.indices[bank_idx].search(query, top_k);

        // Check if we're near a boundary
        let blend_info = self.compute_blend(timestep);

        match blend_info {
            BlendInfo::Pure => {
                let adapters: Vec<_> = primary_results
                    .into_iter()
                    .map(|(id, sim)| (self.banks[bank_idx].get(id), sim))
                    .collect();
                TALoRARetrievalResult::Single { adapters }
            }
            BlendInfo::Blend { secondary_bank, factor } => {
                let secondary_results = self.indices[secondary_bank].search(query, top_k);

                let primary: Vec<_> = primary_results
                    .into_iter()
                    .map(|(id, sim)| (self.banks[bank_idx].get(id), sim))
                    .collect();
                let secondary: Vec<_> = secondary_results
                    .into_iter()
                    .map(|(id, sim)| (self.banks[secondary_bank].get(id), sim))
                    .collect();

                TALoRARetrievalResult::Blended {
                    primary,
                    secondary,
                    blend_factor: factor,
                }
            }
        }
    }

    /// Store new adapter in appropriate bank
    pub fn store(
        &mut self,
        adapter: MicroLoRA,
        query_embedding: &[f32],
        timestep: u32,
    ) -> AdapterId {
        let bank_idx = self.get_bank_index(timestep);
        let id = self.banks[bank_idx].add(adapter);
        self.indices[bank_idx].insert(id, query_embedding);
        id
    }
}
```

### 2.2 DGR Implementation

**File: `src/micro_lora/dgr.rs`**

```rust
//! DGR: Denoising-Guided Retrieval

use super::TALoRAManager;

/// Denoising-Guided Retrieval system
pub struct DGRSystem {
    /// TALoRA manager for timestep-aware retrieval
    talora: TALoRAManager,
    /// Uncertainty threshold for triggering retrieval
    uncertainty_threshold: f32,
    /// Maximum retrievals per step
    max_retrievals: usize,
    /// Uncertainty metric
    metric: UncertaintyMetric,
}

impl DGRSystem {
    /// Compute uncertainty and retrieve adapters for uncertain positions
    pub fn retrieve_for_uncertain(
        &self,
        logits: &Tensor,          // [batch, seq, vocab]
        hidden_states: &Tensor,    // [batch, seq, hidden]
        timestep: u32,
    ) -> DGRResult {
        // 1. Compute per-position uncertainty
        let uncertainty = self.compute_uncertainty(logits, timestep);

        // 2. Identify positions exceeding threshold
        let mut uncertain_positions = Vec::new();
        for batch in 0..uncertainty.size(0) {
            for seq in 0..uncertainty.size(1) {
                let u = uncertainty[[batch, seq]];
                if u > self.uncertainty_threshold {
                    uncertain_positions.push((batch, seq, u));
                }
            }
        }

        // 3. Sort by uncertainty and limit
        uncertain_positions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        uncertain_positions.truncate(self.max_retrievals);

        // 4. Retrieve adapters for each uncertain position
        let mut position_adapters = HashMap::new();
        for (batch, seq, _uncertainty) in uncertain_positions {
            let query = hidden_states.slice_2d(batch, seq);
            let adapters = self.talora.retrieve(&query, timestep, TOP_K);
            position_adapters.insert((batch, seq), adapters);
        }

        DGRResult { position_adapters }
    }

    /// Compute uncertainty from logits
    fn compute_uncertainty(&self, logits: &Tensor, timestep: u32) -> Tensor {
        let raw_uncertainty = match self.metric {
            UncertaintyMetric::Entropy => {
                // H(p) = -sum(p * log(p))
                let probs = logits.softmax(-1);
                let log_probs = probs.log();
                -(probs * log_probs).sum(-1)
            }
            UncertaintyMetric::ConfidenceGap => {
                // 1 - (max_prob - second_max_prob)
                let sorted = logits.sort(-1, true);
                let gap = sorted.select(-1, 0) - sorted.select(-1, 1);
                1.0 - gap.sigmoid()
            }
        };

        // Normalize by timestep (higher t = naturally more uncertain)
        let t_factor = (timestep as f32 / 1000.0).sqrt();
        raw_uncertainty / t_factor
    }

    /// Apply DGR adapters to specific positions
    pub fn apply(
        &self,
        output: &mut Tensor,
        dgr_result: &DGRResult,
    ) {
        for ((batch, seq), adapters) in &dgr_result.position_adapters {
            let position_delta = self.compose_adapters(adapters);
            output.slice_2d_mut(*batch, *seq).add_(&position_delta);
        }
    }
}
```

### 2.3 RuVector Integration

**File: `src/micro_lora/ruvector_integration.rs`**

```rust
//! Integration with RuVector for pattern storage

use ruvector::{VectorDb, HNSWConfig, SearchResult};

/// RuVector-backed adapter storage
pub struct RuVectorAdapterStore {
    /// Vector database
    db: VectorDb,
    /// Adapter data (stored separately from vectors)
    adapters: HashMap<u64, MicroLoRA>,
    /// Next adapter ID
    next_id: AtomicU64,
}

impl RuVectorAdapterStore {
    pub fn new(dimension: usize) -> Result<Self, RuVectorError> {
        let config = HNSWConfig {
            m: 16,                    // Connections per node
            ef_construction: 200,     // Construction-time search width
            ef_search: 50,            // Query-time search width
            max_elements: 100_000,    // Maximum patterns
        };

        let db = VectorDb::new(dimension, config)?;

        Ok(Self {
            db,
            adapters: HashMap::new(),
            next_id: AtomicU64::new(0),
        })
    }

    /// Store adapter with embedding
    pub fn store(&mut self, embedding: &[f32], adapter: MicroLoRA) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.db.insert(id, embedding);
        self.adapters.insert(id, adapter);
        id
    }

    /// Search for similar adapters
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, &MicroLoRA, f32)> {
        let results = self.db.search(query, k);
        results
            .into_iter()
            .filter_map(|SearchResult { id, distance }| {
                let adapter = self.adapters.get(&id)?;
                Some((id, adapter, 1.0 - distance)) // Convert distance to similarity
            })
            .collect()
    }

    /// Remove adapter
    pub fn remove(&mut self, id: u64) -> Option<MicroLoRA> {
        self.db.remove(id);
        self.adapters.remove(&id)
    }

    /// Persistence
    pub fn save(&self, path: &Path) -> Result<(), IoError> {
        // Save vector index
        self.db.save(&path.join("vectors.bin"))?;

        // Save adapter data
        let adapter_data = bincode::serialize(&self.adapters)?;
        std::fs::write(path.join("adapters.bin"), adapter_data)?;

        Ok(())
    }

    pub fn load(path: &Path, dimension: usize) -> Result<Self, IoError> {
        let db = VectorDb::load(&path.join("vectors.bin"), dimension)?;
        let adapter_data = std::fs::read(path.join("adapters.bin"))?;
        let adapters: HashMap<u64, MicroLoRA> = bincode::deserialize(&adapter_data)?;
        let max_id = adapters.keys().max().copied().unwrap_or(0);

        Ok(Self {
            db,
            adapters,
            next_id: AtomicU64::new(max_id + 1),
        })
    }
}
```

## Phase 3: Federation

### 3.1 DAF Protocol

**File: `src/federation/daf.rs`**

```rust
//! DAF: Diffusion-Aware Federation

/// DAF aggregator with timestep-aware strategies
pub struct DAFAggregator {
    /// Per-group strategies
    strategies: [GroupStrategy; 3],
    /// Privacy manager
    privacy: PrivacyManager,
}

impl DAFAggregator {
    /// Aggregate updates from multiple clients
    pub fn aggregate(&self, updates: Vec<ClientUpdate>) -> Result<AggregatedUpdate, DAFError> {
        // Group updates by timestep bank
        let mut grouped: [Vec<BankUpdate>; 3] = Default::default();

        for update in updates {
            for (bank_idx, bank_update) in update.bank_updates {
                grouped[bank_idx].push(bank_update);
            }
        }

        // Aggregate each group with appropriate strategy
        let mut result = AggregatedUpdate::default();

        for (bank_idx, bank_updates) in grouped.iter().enumerate() {
            if bank_updates.is_empty() {
                continue;
            }

            let aggregated = match &self.strategies[bank_idx] {
                GroupStrategy::WeightedAverage => {
                    self.weighted_average(bank_updates)
                }
                GroupStrategy::QualityWeighted { metric } => {
                    self.quality_weighted(bank_updates, metric)
                }
                GroupStrategy::Conservative { min_k, threshold } => {
                    self.conservative(bank_updates, *min_k, *threshold)?
                }
            };

            // Apply differential privacy
            let private = self.privacy.privatize(&aggregated)?;
            result.bank_updates.insert(bank_idx, private);
        }

        Ok(result)
    }
}
```

### 3.2 Privacy Tiers

**File: `src/federation/privacy.rs`**

See 05-PRIVACY.md for full implementation.

### 3.3 Gossip Protocol

**File: `src/federation/gossip.rs`**

See 04-FEDERATION.md for full implementation.

## Phase 4: Optimization

### 4.1 GPU Kernels

See 06-SIMD-GPU.md for CUDA, Metal, and Vulkan implementations.

### 4.2 Benchmarking Suite

**File: `benches/diffusion_bench.rs`**

```rust
//! Benchmarks for diffusion model performance

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_denoise_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("denoise_step");

    for seq_len in [128, 256, 512, 1024].iter() {
        let model = setup_test_model(*seq_len);
        let x_t = random_tensor(*seq_len);

        group.bench_with_input(
            BenchmarkId::new("scalar", seq_len),
            seq_len,
            |b, _| b.iter(|| model.denoise_step_scalar(&x_t, 500)),
        );

        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(
            BenchmarkId::new("avx2", seq_len),
            seq_len,
            |b, _| b.iter(|| model.denoise_step_avx2(&x_t, 500)),
        );

        #[cfg(feature = "cuda")]
        group.bench_with_input(
            BenchmarkId::new("cuda", seq_len),
            seq_len,
            |b, _| b.iter(|| model.denoise_step_cuda(&x_t, 500)),
        );
    }

    group.finish();
}

fn bench_talora_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("talora_retrieval");

    let talora = setup_talora_with_patterns(10000);
    let query = random_embedding(4096);

    for k in [1, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("top_k", k),
            k,
            |b, &k| b.iter(|| talora.retrieve(&query, 500, k)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_denoise_step, bench_talora_retrieval);
criterion_main!(benches);
```

## Milestone Checklist

### Phase 1 Milestones
- [ ] Load Q4 quantized model
- [ ] Basic forward pass
- [ ] MDLM sampling loop
- [ ] SIMD-optimized attention
- [ ] End-to-end generation

### Phase 2 Milestones
- [ ] TALoRA bank structure
- [ ] Timestep-aware retrieval
- [ ] DGR uncertainty computation
- [ ] Position-specific application
- [ ] RuVector integration

### Phase 3 Milestones
- [ ] DAF aggregation strategies
- [ ] Privacy tier system
- [ ] Gossip protocol
- [ ] Consent management
- [ ] Audit logging

### Phase 4 Milestones
- [ ] CUDA kernels
- [ ] Metal kernels
- [ ] Benchmark suite
- [ ] Performance regression tests
- [ ] Documentation complete

---

**Previous**: [07-NOVEL-CONTRIBUTIONS.md](./07-NOVEL-CONTRIBUTIONS.md) - Original research
**Next**: [09-API-REFERENCE.md](./09-API-REFERENCE.md) - Module API design
