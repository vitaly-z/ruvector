# RuvDLLM Architecture

## System Overview

RuvDLLM is built on a layered architecture that separates concerns while maintaining high performance through zero-cost abstractions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 5: API & Integration                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  HTTP Server │ NAPI Bindings │ WASM │ CLI │ ruvLLM Integration     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                         LAYER 4: Federation & Privacy                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DAF Protocol │ Gossip Sync │ Privacy Tiers │ Encryption │ Consent │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                         LAYER 3: Learning & Adaptation                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TALoRA │ DGR │ SONA Loops │ EWC++ │ Pattern Clustering │ Training │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                         LAYER 2: Model & Inference                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Q4 Diffusion Model │ MDLM/BD3LM Sampler │ QLoRA │ Tokenizer       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                         LAYER 1: Core Infrastructure                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RuVector (HNSW) │ SIMD Ops │ GPU Kernels │ Memory Management      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Diffusion Model Core

```rust
/// Core diffusion model with Q4 quantization
pub struct DiffusionModel {
    /// Frozen base weights (Q4 quantized)
    base_weights: Q4Weights,

    /// QLoRA adapters (AR→Diffusion conversion)
    qlora: QLoraAdapters,

    /// Time embedding projection
    time_embed: TimeEmbedding,

    /// Attention mask (bidirectional for diffusion)
    attention_type: AttentionType,

    /// Model configuration
    config: ModelConfig,
}

/// Model configuration
pub struct ModelConfig {
    pub hidden_dim: usize,        // 4096 for 7B
    pub num_layers: usize,        // 32 for 7B
    pub num_heads: usize,         // 32 for 7B
    pub vocab_size: usize,        // 32000 typical
    pub max_seq_len: usize,       // 2048-8192
    pub qlora_rank: usize,        // 32-64 for conversion
}
```

### 2. Timestep-Aware LoRA (TALoRA) - NOVEL

```rust
/// TALoRA: Different adapters for different denoising stages
///
/// NOVEL CONTRIBUTION: Existing work uses static LoRA for all timesteps.
/// We observe that early denoising (high noise) requires different
/// adaptations than late denoising (refinement).
pub struct TALoRA {
    /// LoRA banks for different timestep ranges
    /// e.g., [0.0-0.3], [0.3-0.6], [0.6-1.0]
    timestep_banks: Vec<LoraBank>,

    /// Interpolation strategy between banks
    interpolation: InterpolationStrategy,

    /// Learned routing weights
    routing_weights: Option<RoutingNetwork>,
}

/// Timestep-to-LoRA mapping
pub enum InterpolationStrategy {
    /// Hard switch at boundaries
    Discrete,
    /// Linear interpolation between adjacent banks
    Linear,
    /// Learned soft routing
    Attention,
    /// Smooth transition with learnable boundaries
    SoftBoundary { temperature: f32 },
}

impl TALoRA {
    /// Get adapted LoRA for specific timestep
    pub fn get_adapter(&self, t: f32, query_embedding: &[f32]) -> MergedLora {
        match self.interpolation {
            InterpolationStrategy::Discrete => {
                let bank_idx = self.get_bank_index(t);
                self.timestep_banks[bank_idx].retrieve(query_embedding)
            }
            InterpolationStrategy::Linear => {
                let (bank_a, bank_b, alpha) = self.get_adjacent_banks(t);
                let lora_a = bank_a.retrieve(query_embedding);
                let lora_b = bank_b.retrieve(query_embedding);
                self.interpolate_lora(&lora_a, &lora_b, alpha)
            }
            InterpolationStrategy::Attention => {
                let weights = self.routing_weights.as_ref().unwrap()
                    .compute_weights(t, query_embedding);
                self.weighted_merge(&weights)
            }
            InterpolationStrategy::SoftBoundary { temperature } => {
                self.soft_boundary_merge(t, query_embedding, temperature)
            }
        }
    }
}
```

### 3. Denoising-Guided Retrieval (DGR) - NOVEL

```rust
/// DGR: Use model uncertainty to guide adapter retrieval
///
/// NOVEL CONTRIBUTION: Instead of retrieving adapters based only on
/// input query, we use the model's uncertainty during denoising to
/// dynamically retrieve more specialized adapters when needed.
pub struct DenoisingGuidedRetrieval {
    /// RuVector index for pattern storage
    pattern_index: RuVectorIndex,

    /// Uncertainty threshold for triggering retrieval
    uncertainty_threshold: f32,

    /// Maximum retrievals per generation
    max_retrievals: usize,

    /// Retrieval strategy
    strategy: DGRStrategy,
}

pub enum DGRStrategy {
    /// Retrieve when entropy exceeds threshold
    EntropyBased { threshold: f32 },
    /// Retrieve when top-k probability gap is small
    ConfidenceGap { min_gap: f32 },
    /// Retrieve at fixed intervals + uncertainty spikes
    Hybrid { interval: usize, spike_threshold: f32 },
    /// Adaptive based on running statistics
    Adaptive,
}

impl DenoisingGuidedRetrieval {
    /// Check if retrieval is needed based on current denoising state
    pub fn should_retrieve(
        &self,
        logits: &[f32],
        timestep: f32,
        step: usize,
    ) -> bool {
        match &self.strategy {
            DGRStrategy::EntropyBased { threshold } => {
                let entropy = self.compute_entropy(logits);
                entropy > *threshold
            }
            DGRStrategy::ConfidenceGap { min_gap } => {
                let gap = self.compute_confidence_gap(logits);
                gap < *min_gap
            }
            DGRStrategy::Hybrid { interval, spike_threshold } => {
                step % interval == 0 || self.is_uncertainty_spike(logits, *spike_threshold)
            }
            DGRStrategy::Adaptive => {
                self.adaptive_decision(logits, timestep, step)
            }
        }
    }

    /// Retrieve adapters based on current generation state
    pub fn retrieve(
        &self,
        query_embedding: &[f32],
        current_tokens: &[u32],
        uncertainty_context: &UncertaintyContext,
    ) -> Vec<RetrievedAdapter> {
        // Combine query embedding with uncertainty signal
        let augmented_query = self.augment_with_uncertainty(
            query_embedding,
            uncertainty_context,
        );

        // Search RuVector with augmented query
        let candidates = self.pattern_index.search(&augmented_query, k: 10);

        // Filter by relevance to current uncertainty type
        self.filter_by_uncertainty_type(candidates, uncertainty_context)
    }
}
```

### 4. Diffusion-Aware Federation (DAF) - NOVEL

```rust
/// DAF: Federation protocol aware of diffusion model semantics
///
/// NOVEL CONTRIBUTION: Standard federated averaging doesn't account for
/// the fact that different timesteps in diffusion models serve different
/// purposes. We propose schedule-aligned aggregation.
pub struct DiffusionAwareFederation {
    /// Local node state
    node: FederatedNode,

    /// Aggregation strategy
    aggregation: DAFAggregation,

    /// Privacy engine
    privacy: DiffusionPrivacyEngine,

    /// Sync protocol
    sync: DAFSync,
}

pub enum DAFAggregation {
    /// Aggregate TALoRA banks separately by timestep range
    TimestepAligned,
    /// Weight by denoising quality at each timestep
    QualityWeighted,
    /// Use noise schedule similarity for aggregation weights
    ScheduleSimilarity,
    /// Combine all strategies
    Hybrid {
        timestep_weight: f32,
        quality_weight: f32,
        schedule_weight: f32,
    },
}

impl DiffusionAwareFederation {
    /// Aggregate updates with diffusion awareness
    pub fn aggregate(
        &self,
        updates: Vec<FederatedUpdate>,
    ) -> AggregatedUpdate {
        match &self.aggregation {
            DAFAggregation::TimestepAligned => {
                // Aggregate each timestep bank separately
                let mut aggregated_banks = Vec::new();

                for bank_idx in 0..self.num_timestep_banks() {
                    let bank_updates: Vec<_> = updates.iter()
                        .map(|u| &u.talora_banks[bank_idx])
                        .collect();

                    aggregated_banks.push(self.fedavg_bank(&bank_updates));
                }

                AggregatedUpdate { talora_banks: aggregated_banks }
            }
            DAFAggregation::QualityWeighted => {
                // Weight each update by its denoising quality metrics
                let weights: Vec<f32> = updates.iter()
                    .map(|u| u.denoising_quality_score())
                    .collect();

                self.weighted_aggregate(&updates, &weights)
            }
            DAFAggregation::ScheduleSimilarity => {
                // Group updates by noise schedule similarity
                let clusters = self.cluster_by_schedule(&updates);

                // Aggregate within clusters, then merge
                self.hierarchical_aggregate(&clusters)
            }
            DAFAggregation::Hybrid { timestep_weight, quality_weight, schedule_weight } => {
                self.hybrid_aggregate(&updates, *timestep_weight, *quality_weight, *schedule_weight)
            }
        }
    }
}
```

## Data Flow

### Inference Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE DATA FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. INPUT                                                                   │
│     │                                                                       │
│     ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ Tokenize + Embed (SIMD)                              [0.02ms]   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│     │                                                                       │
│     ▼                                                                       │
│  2. INITIAL RETRIEVAL                                                       │
│     │                                                                       │
│     ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ RuVector HNSW Search (query embedding)               [0.05ms]   │      │
│  │   → Retrieve initial TALoRA candidates                          │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│     │                                                                       │
│     ▼                                                                       │
│  3. DIFFUSION LOOP (8 steps typical)                                       │
│     │                                                                       │
│     ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ FOR step in (0..num_steps).rev():                                │      │
│  │   │                                                              │      │
│  │   ├─► t = step / num_steps                                       │      │
│  │   │                                                              │      │
│  │   ├─► TALoRA: Get timestep-appropriate adapter        [0.02ms]  │      │
│  │   │                                                              │      │
│  │   ├─► Forward: Q4 base + merged LoRA                 [5-10ms]   │      │
│  │   │                                                              │      │
│  │   ├─► DGR: Check uncertainty, retrieve if needed     [0-0.1ms]  │      │
│  │   │                                                              │      │
│  │   └─► Denoise: Sample next tokens                    [0.5ms]    │      │
│  │                                                                  │      │
│  │ Total per step: ~6-11ms                                         │      │
│  │ Total 8 steps: ~50-90ms                                         │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│     │                                                                       │
│     ▼                                                                       │
│  4. OUTPUT                                                                  │
│     │                                                                       │
│     ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ Detokenize + Return                                  [0.01ms]   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  TOTAL INFERENCE: 50-100ms (competitive with AR models)                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Learning Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LEARNING DATA FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LOOP A: INSTANT (Every Request)                                            │
│  ────────────────────────────────                                           │
│  Request → Record trajectory → Update retrieval scores → Done               │
│  Latency: <0.1ms (async, non-blocking)                                      │
│                                                                              │
│  LOOP B: BACKGROUND (Every N requests or M minutes)                         │
│  ──────────────────────────────────────────────────                         │
│  Trajectories → Cluster → Extract patterns → Update TALoRA banks           │
│  Latency: ~100ms (background thread)                                        │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 1. Collect high-quality trajectories (rating > 0.7)              │      │
│  │ 2. Cluster by timestep range + task type                        │      │
│  │ 3. For each cluster:                                            │      │
│  │    a. Compute centroid embedding                                │      │
│  │    b. Train MicroLoRA (rank 2, SIMD)                           │      │
│  │    c. Add to appropriate TALoRA bank                           │      │
│  │ 4. Update RuVector index                                        │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  LOOP C: DEEP (Every H hours)                                               │
│  ─────────────────────────────                                              │
│  All patterns → EWC++ consolidation → BaseLoRA training → Federation       │
│  Latency: ~10-60 minutes (CPU) or ~1-5 minutes (GPU)                       │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 1. Aggregate all successful patterns                            │      │
│  │ 2. Compute EWC++ Fisher information                             │      │
│  │ 3. Train BaseLoRA with EWC regularization                       │      │
│  │ 4. If consented: Prepare federated update                       │      │
│  │    a. Classify patterns by sharing tier                         │      │
│  │    b. Apply differential privacy                                │      │
│  │    c. Submit to DAF protocol                                    │      │
│  │ 5. Receive federated updates                                    │      │
│  │ 6. Merge with local model                                       │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY LAYOUT (7B Model)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STATIC (loaded once):                                                      │
│  ─────────────────────                                                      │
│  Base model weights (Q4)           3.5 GB                                  │
│  QLoRA adapters (FP16)             100 MB                                  │
│  Time embeddings                   16 MB                                   │
│  Tokenizer                         5 MB                                    │
│  ────────────────────────────────────────                                   │
│  Subtotal:                         ~3.6 GB                                  │
│                                                                              │
│  DYNAMIC (per-session):                                                     │
│  ──────────────────────                                                     │
│  TALoRA banks (3 banks × 1000 patterns × 4KB)   12 MB                     │
│  RuVector index overhead                        50 MB                      │
│  KV cache (2048 tokens)                         256 MB                     │
│  Working memory                                 100 MB                     │
│  ────────────────────────────────────────                                   │
│  Subtotal:                         ~420 MB                                  │
│                                                                              │
│  TRAINING (background):                                                     │
│  ──────────────────────                                                     │
│  Gradient accumulation             200 MB                                  │
│  Optimizer state (Adam)            400 MB                                  │
│  EWC Fisher matrices               200 MB                                  │
│  ────────────────────────────────────────                                   │
│  Subtotal:                         ~800 MB                                  │
│                                                                              │
│  ════════════════════════════════════════                                   │
│  TOTAL (inference only):           ~4 GB                                   │
│  TOTAL (with training):            ~4.8 GB                                 │
│  ════════════════════════════════════════                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Threading Model

```rust
/// Threading architecture for maximum performance
pub struct ThreadingModel {
    /// Inference thread pool (CPU-bound)
    inference_pool: ThreadPool,

    /// Background learning (separate thread)
    learning_thread: JoinHandle<()>,

    /// Federation sync (async runtime)
    federation_runtime: tokio::Runtime,

    /// GPU dispatch (if available)
    gpu_stream: Option<GpuStream>,
}

impl ThreadingModel {
    pub fn new(config: &ThreadingConfig) -> Self {
        // Inference: Use all physical cores minus 1
        let inference_threads = num_cpus::get_physical() - 1;

        // Learning: Single dedicated thread
        // Federation: Tokio async runtime

        Self {
            inference_pool: ThreadPool::new(inference_threads),
            learning_thread: spawn_learning_thread(),
            federation_runtime: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .build()
                .unwrap(),
            gpu_stream: GpuStream::try_new(),
        }
    }
}
```

---

**Next**: [02-QLORA-DIFFUSION.md](./02-QLORA-DIFFUSION.md) - QLoRA AR→Diffusion conversion
