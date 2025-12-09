# MicroLoRA: Real-Time Adaptation System

## Overview

MicroLoRA enables per-request adaptation with sub-millisecond overhead by retrieving and composing small LoRA adapters from RuVector storage.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MicroLoRA Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          ┌─────────────────┐                                │
│                          │  Query Input    │                                │
│                          └────────┬────────┘                                │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌─────────────────┐                                │
│                          │  SIMD Embed     │ 0.02ms                         │
│                          └────────┬────────┘                                │
│                                   │                                          │
│                    ┌──────────────┼──────────────┐                          │
│                    │              │              │                          │
│                    ▼              ▼              ▼                          │
│            ┌───────────┐  ┌───────────┐  ┌───────────┐                     │
│            │ TALoRA    │  │ TALoRA    │  │ TALoRA    │                     │
│            │ Bank 0    │  │ Bank 1    │  │ Bank 2    │                     │
│            │ t∈[0,0.3] │  │ t∈[0.3,0.7]│ │ t∈[0.7,1] │                     │
│            └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                     │
│                  │              │              │                            │
│                  └──────────────┼──────────────┘                            │
│                                 │                                            │
│                                 ▼                                            │
│                          ┌─────────────────┐                                │
│                          │  LoRA Composer  │ 0.05ms                         │
│                          └────────┬────────┘                                │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌─────────────────┐                                │
│                          │ Merged MicroLoRA │                               │
│                          └─────────────────┘                                │
│                                                                              │
│  Total Overhead: <0.1ms per request                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Three-Tier LoRA Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LoRA Hierarchy                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TIER 1: QLoRA Base (AR→Diffusion Conversion)                              │
│  ─────────────────────────────────────────────                              │
│  • Rank: 32-64                                                              │
│  • Trained once during model conversion                                     │
│  • Frozen during operation                                                  │
│  • Size: ~100MB                                                             │
│                                                                              │
│  TIER 2: BaseLoRA (Background Learning)                                     │
│  ───────────────────────────────────────                                    │
│  • Rank: 8-16                                                               │
│  • Updated hourly/daily via SONA Loop C                                    │
│  • EWC++ protected against forgetting                                      │
│  • Size: ~50MB                                                              │
│                                                                              │
│  TIER 3: MicroLoRA (Real-Time Adaptation)                                   │
│  ─────────────────────────────────────────                                  │
│  • Rank: 1-2                                                                │
│  • Retrieved per-request from RuVector                                     │
│  • Composed from stored patterns                                           │
│  • Size per pattern: ~4KB                                                  │
│                                                                              │
│  EFFECTIVE WEIGHTS:                                                         │
│  W' = W_base + ΔW_qlora + ΔW_base_lora + α × ΔW_micro_lora                 │
│       (frozen)  (frozen)  (periodic)     (per-request)                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Data Structures

```rust
/// MicroLoRA pattern stored in RuVector
#[derive(Clone, Serialize, Deserialize)]
pub struct MicroLoraPattern {
    /// Pattern identifier
    pub id: PatternId,

    /// Query embedding for retrieval
    pub embedding: Vec<f32>,

    /// Compressed LoRA weights (rank 1-2)
    pub lora: CompressedMicroLora,

    /// Timestep range this pattern is optimized for
    pub timestep_range: (f32, f32),

    /// Quality metrics
    pub metrics: PatternMetrics,

    /// Sharing policy
    pub policy: SharingPolicy,

    /// Provenance tracking
    pub provenance: Provenance,
}

/// Compressed MicroLoRA (8-bit quantized)
#[derive(Clone, Serialize, Deserialize)]
pub struct CompressedMicroLora {
    /// Quantized A matrix
    pub a_quantized: Vec<u8>,
    pub a_scale: f32,
    pub a_zero_point: i8,

    /// Quantized B matrix
    pub b_quantized: Vec<u8>,
    pub b_scale: f32,
    pub b_zero_point: i8,

    /// LoRA rank
    pub rank: u8,

    /// Hidden dimension
    pub hidden_dim: u16,

    /// Per-layer presence flags (which layers have this pattern)
    pub layer_mask: u64,
}

impl CompressedMicroLora {
    /// Compress from f32 LoRA (~32x compression)
    pub fn compress(lora: &MicroLora) -> Self {
        let (a_quantized, a_scale, a_zero_point) = quantize_symmetric_i8(&lora.a);
        let (b_quantized, b_scale, b_zero_point) = quantize_symmetric_i8(&lora.b);

        Self {
            a_quantized,
            a_scale,
            a_zero_point,
            b_quantized,
            b_scale,
            b_zero_point,
            rank: lora.rank as u8,
            hidden_dim: lora.hidden_dim as u16,
            layer_mask: lora.layer_mask,
        }
    }

    /// Decompress to f32 (SIMD optimized)
    pub fn decompress(&self) -> MicroLora {
        let a = dequantize_symmetric_i8_simd(&self.a_quantized, self.a_scale, self.a_zero_point);
        let b = dequantize_symmetric_i8_simd(&self.b_quantized, self.b_scale, self.b_zero_point);

        MicroLora {
            a,
            b,
            rank: self.rank as usize,
            hidden_dim: self.hidden_dim as usize,
            layer_mask: self.layer_mask,
        }
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        self.a_quantized.len() + self.b_quantized.len() + 16 // metadata
    }
}

/// Quality metrics for a pattern
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct PatternMetrics {
    /// Success rate from user feedback
    pub success_rate: f32,

    /// Number of times used
    pub usage_count: u32,

    /// Average latency when using this pattern
    pub avg_latency_ms: f32,

    /// Denoising quality score
    pub denoising_quality: f32,

    /// Last used timestamp
    pub last_used: DateTime<Utc>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}
```

## MicroLoRA Bank

```rust
/// Bank of MicroLoRA patterns organized by timestep range
pub struct MicroLoraBank {
    /// RuVector index for pattern retrieval
    index: RuVectorIndex,

    /// Timestep range this bank covers
    timestep_range: (f32, f32),

    /// Maximum patterns to store
    max_patterns: usize,

    /// LRU eviction when full
    lru: LruCache<PatternId, ()>,

    /// Statistics
    stats: BankStats,
}

impl MicroLoraBank {
    /// Create new bank for timestep range
    pub fn new(timestep_range: (f32, f32), max_patterns: usize) -> Self {
        let index = RuVectorIndex::new(RuVectorConfig {
            dimension: 768,  // Embedding dimension
            metric: DistanceMetric::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 64,
        });

        Self {
            index,
            timestep_range,
            max_patterns,
            lru: LruCache::new(max_patterns),
            stats: BankStats::default(),
        }
    }

    /// Retrieve top-k patterns for query
    pub fn retrieve(&self, query_embedding: &[f32], k: usize) -> Vec<RetrievedPattern> {
        let start = Instant::now();

        // HNSW search
        let candidates = self.index.search(query_embedding, k * 2);

        // Filter and score
        let results: Vec<RetrievedPattern> = candidates
            .into_iter()
            .filter_map(|candidate| {
                let pattern = self.index.get_pattern(candidate.id)?;

                // Compute final score
                let similarity_score = 1.0 / (1.0 + candidate.distance);
                let quality_score = pattern.metrics.success_rate;
                let recency_score = self.compute_recency_score(&pattern);

                let final_score =
                    similarity_score * 0.5 +
                    quality_score * 0.3 +
                    recency_score * 0.2;

                Some(RetrievedPattern {
                    pattern,
                    similarity: similarity_score,
                    score: final_score,
                })
            })
            .take(k)
            .collect();

        self.stats.record_retrieval(start.elapsed());

        results
    }

    /// Insert new pattern
    pub fn insert(&mut self, pattern: MicroLoraPattern) -> Result<()> {
        // Check capacity
        if self.index.len() >= self.max_patterns {
            // Evict least recently used
            if let Some((evict_id, _)) = self.lru.pop_lru() {
                self.index.remove(evict_id)?;
            }
        }

        // Insert into index
        self.index.insert(&pattern.embedding, pattern.clone())?;

        // Update LRU
        self.lru.put(pattern.id, ());

        Ok(())
    }
}
```

## LoRA Composition

```rust
/// Composes multiple MicroLoRA patterns into single adapter
pub struct LoraComposer {
    /// Composition strategy
    strategy: CompositionStrategy,

    /// SIMD operations
    simd: SimdOps,
}

#[derive(Clone, Debug)]
pub enum CompositionStrategy {
    /// Weighted average based on scores
    WeightedAverage,

    /// Attention-based composition
    Attention { temperature: f32 },

    /// Top-1 only
    TopK { k: usize },

    /// Task-arithmetic style composition
    TaskArithmetic { scaling: f32 },
}

impl LoraComposer {
    /// Compose multiple patterns into single MicroLoRA
    pub fn compose(&self, patterns: &[RetrievedPattern]) -> MicroLora {
        if patterns.is_empty() {
            return MicroLora::zero();
        }

        if patterns.len() == 1 {
            return patterns[0].pattern.lora.decompress();
        }

        match &self.strategy {
            CompositionStrategy::WeightedAverage => {
                self.weighted_average_compose(patterns)
            }
            CompositionStrategy::Attention { temperature } => {
                self.attention_compose(patterns, *temperature)
            }
            CompositionStrategy::TopK { k } => {
                self.topk_compose(patterns, *k)
            }
            CompositionStrategy::TaskArithmetic { scaling } => {
                self.task_arithmetic_compose(patterns, *scaling)
            }
        }
    }

    /// Weighted average composition (SIMD optimized)
    fn weighted_average_compose(&self, patterns: &[RetrievedPattern]) -> MicroLora {
        // Normalize scores to weights
        let total_score: f32 = patterns.iter().map(|p| p.score).sum();
        let weights: Vec<f32> = patterns.iter()
            .map(|p| p.score / total_score)
            .collect();

        // Initialize output
        let first = patterns[0].pattern.lora.decompress();
        let mut a_sum = vec![0.0f32; first.a.len()];
        let mut b_sum = vec![0.0f32; first.b.len()];

        // Weighted sum (SIMD)
        for (pattern, &weight) in patterns.iter().zip(weights.iter()) {
            let lora = pattern.pattern.lora.decompress();

            SimdOps::accumulate_scaled(&mut a_sum, &lora.a, weight);
            SimdOps::accumulate_scaled(&mut b_sum, &lora.b, weight);
        }

        MicroLora {
            a: a_sum,
            b: b_sum,
            rank: first.rank,
            hidden_dim: first.hidden_dim,
            layer_mask: patterns.iter().fold(0, |acc, p| acc | p.pattern.lora.layer_mask),
        }
    }

    /// Attention-based composition
    fn attention_compose(&self, patterns: &[RetrievedPattern], temperature: f32) -> MicroLora {
        // Compute attention weights from scores
        let mut scores: Vec<f32> = patterns.iter()
            .map(|p| p.score / temperature)
            .collect();

        // Softmax
        SimdOps::softmax(&mut scores);

        // Same as weighted average with softmax weights
        self.weighted_sum_with_weights(patterns, &scores)
    }

    /// Top-k composition
    fn topk_compose(&self, patterns: &[RetrievedPattern], k: usize) -> MicroLora {
        let top_patterns: Vec<_> = patterns.iter().take(k).collect();

        // Equal weights for top-k
        let weight = 1.0 / top_patterns.len() as f32;
        let weights = vec![weight; top_patterns.len()];

        self.weighted_sum_with_weights_ref(&top_patterns, &weights)
    }
}
```

## TALoRA Manager

```rust
/// Manages Timestep-Aware LoRA banks
pub struct TALoraManager {
    /// Banks for different timestep ranges
    banks: Vec<MicroLoraBank>,

    /// Timestep boundaries
    boundaries: Vec<f32>,

    /// Interpolation config
    interpolation: InterpolationConfig,

    /// Composer
    composer: LoraComposer,
}

impl TALoraManager {
    /// Create with default 3-bank configuration
    pub fn new_default(max_patterns_per_bank: usize) -> Self {
        Self::new(
            vec![0.0, 0.33, 0.67, 1.0],  // 3 equal ranges
            max_patterns_per_bank,
            InterpolationConfig::default(),
        )
    }

    /// Create with custom boundaries
    pub fn new(boundaries: Vec<f32>, max_patterns_per_bank: usize, interpolation: InterpolationConfig) -> Self {
        let banks: Vec<_> = boundaries.windows(2)
            .map(|w| MicroLoraBank::new((w[0], w[1]), max_patterns_per_bank))
            .collect();

        Self {
            banks,
            boundaries,
            interpolation,
            composer: LoraComposer::new(CompositionStrategy::WeightedAverage),
        }
    }

    /// Get adapted LoRA for query at specific timestep
    pub fn get_adapter(&self, query_embedding: &[f32], timestep: f32) -> MicroLora {
        match self.interpolation.strategy {
            InterpolationStrategy::Discrete => {
                let bank_idx = self.get_bank_index(timestep);
                let patterns = self.banks[bank_idx].retrieve(query_embedding, 5);
                self.composer.compose(&patterns)
            }
            InterpolationStrategy::Linear => {
                let (bank_a_idx, bank_b_idx, alpha) = self.get_interpolation_params(timestep);

                let patterns_a = self.banks[bank_a_idx].retrieve(query_embedding, 3);
                let patterns_b = self.banks[bank_b_idx].retrieve(query_embedding, 3);

                let lora_a = self.composer.compose(&patterns_a);
                let lora_b = self.composer.compose(&patterns_b);

                self.interpolate_loras(&lora_a, &lora_b, alpha)
            }
            InterpolationStrategy::SoftBoundary { temperature } => {
                // Soft selection across all banks
                let mut all_patterns = Vec::new();

                for (idx, bank) in self.banks.iter().enumerate() {
                    let bank_center = (self.boundaries[idx] + self.boundaries[idx + 1]) / 2.0;
                    let distance = (timestep - bank_center).abs();
                    let bank_weight = (-distance / temperature).exp();

                    let patterns = bank.retrieve(query_embedding, 3);
                    for p in patterns {
                        all_patterns.push((p, bank_weight));
                    }
                }

                self.weighted_compose_with_bank_weights(&all_patterns)
            }
        }
    }

    /// Insert pattern into appropriate bank
    pub fn insert(&mut self, pattern: MicroLoraPattern) -> Result<()> {
        let bank_idx = self.get_bank_index(pattern.timestep_range.0);
        self.banks[bank_idx].insert(pattern)
    }

    /// Get bank index for timestep
    fn get_bank_index(&self, timestep: f32) -> usize {
        for (idx, window) in self.boundaries.windows(2).enumerate() {
            if timestep >= window[0] && timestep < window[1] {
                return idx;
            }
        }
        self.banks.len() - 1
    }

    /// Interpolate between two LoRAs (SIMD)
    fn interpolate_loras(&self, a: &MicroLora, b: &MicroLora, alpha: f32) -> MicroLora {
        MicroLora {
            a: SimdOps::lerp(&a.a, &b.a, alpha),
            b: SimdOps::lerp(&a.b, &b.b, alpha),
            rank: a.rank,
            hidden_dim: a.hidden_dim,
            layer_mask: a.layer_mask | b.layer_mask,
        }
    }
}
```

## Integration with Inference

```rust
/// Complete MicroLoRA-enhanced inference
pub struct MicroLoraInference {
    /// Base diffusion model
    model: DiffusionModel,

    /// TALoRA manager
    talora: TALoraManager,

    /// DGR system
    dgr: DenoisingGuidedRetrieval,

    /// SONA for learning
    sona: LoopCoordinator,
}

impl MicroLoraInference {
    /// Generate with real-time adaptation
    pub async fn generate(&mut self, query: &str, config: &GenerationConfig) -> Response {
        let start = Instant::now();

        // 1. Embed query
        let embedding = self.model.embed_query(query);

        // 2. Initialize generation
        let mut tokens = self.model.tokenize(query);
        let target_len = tokens.len() + config.max_new_tokens;
        let mut masked = vec![self.model.mask_token(); target_len];
        masked[..tokens.len()].copy_from_slice(&tokens);

        // 3. Diffusion loop with TALoRA
        for step in (0..config.num_diffusion_steps).rev() {
            let t = step as f32 / config.num_diffusion_steps as f32;

            // Get timestep-appropriate adapter
            let micro_lora = self.talora.get_adapter(&embedding, t);

            // Forward with adapted model
            let logits = self.model.forward_with_lora(&masked, t, &micro_lora);

            // Check if DGR should retrieve more adapters
            if self.dgr.should_retrieve(&logits, t, step) {
                let additional = self.dgr.retrieve(&embedding, &masked, t);
                let augmented_lora = self.merge_dgr_adapters(&micro_lora, &additional);

                // Re-forward with augmented adapter
                let logits = self.model.forward_with_lora(&masked, t, &augmented_lora);
            }

            // Denoise step
            masked = self.model.denoise_step(&masked, &logits, t, config);
        }

        // 4. Record trajectory for learning (async)
        let trajectory = self.create_trajectory(&embedding, &masked, start.elapsed());
        self.sona.record_trajectory(trajectory);

        // 5. Decode and return
        Response {
            text: self.model.decode(&masked),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            tokens_generated: target_len - tokens.len(),
        }
    }
}
```

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Query embedding | 0.02ms | SIMD optimized |
| HNSW search (k=5) | 0.03ms | RuVector |
| Pattern decompression | 0.02ms | 8-bit → f32 |
| LoRA composition | 0.03ms | Weighted avg |
| **Total overhead** | **<0.1ms** | Per request |

---

**Next**: [04-FEDERATION.md](./04-FEDERATION.md) - Secure federated learning
