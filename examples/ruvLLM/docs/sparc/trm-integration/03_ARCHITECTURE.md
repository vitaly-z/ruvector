# SPARC Phase 3: Architecture

## TinyRecursiveModels + RuvLLM System Design

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RuvLLM v2.0 + TRM Architecture                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Public API Layer                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │  query() │  │ reason() │  │ learn()  │  │ export() │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                        Orchestration Layer                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │  │  RuvLLM    │  │   Session   │  │   Metrics   │                  │   │
│  │  │ Coordinator │  │   Manager   │  │   Tracker   │                  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                         Core Engine Layer                            │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                    TRM Recursive Engine                        │  │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │  │   │
│  │  │  │   Latent    │  │   Answer    │  │  Confidence │           │  │   │
│  │  │  │   Updater   │  │   Refiner   │  │   Scorer    │           │  │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘           │  │   │
│  │  │  ┌─────────────┐  ┌─────────────┐                             │  │   │
│  │  │  │  MLP Mode   │  │ Attn Mode   │                             │  │   │
│  │  │  └─────────────┘  └─────────────┘                             │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                      SONA Engine                               │  │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │  │   │
│  │  │  │  MicroLoRA  │  │  BaseLoRA   │  │   EWC++     │           │  │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘           │  │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │  │   │
│  │  │  │ Reasoning   │  │ Trajectory  │  │    Loop     │           │  │   │
│  │  │  │    Bank     │  │   Buffer    │  │ Coordinator │           │  │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘           │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌────────────────────┐  ┌────────────────────┐                     │   │
│  │  │    FastGRNN        │  │   HNSW Memory      │                     │   │
│  │  │    Router          │  │   Index            │                     │   │
│  │  └────────────────────┘  └────────────────────┘                     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                      Infrastructure Layer                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │  │   SIMD      │  │   Memory    │  │   Tensor    │                  │   │
│  │  │   Engine    │  │   Pool      │  │   Ops       │                  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │  │   Embed     │  │  Quantize   │  │   WASM      │                  │   │
│  │  │   Service   │  │   Engine    │  │   Bridge    │                  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Design

### 2.1 TRM Recursive Engine

```rust
// src/trm/mod.rs

pub mod config;
pub mod latent;
pub mod attention;
pub mod mlp;
pub mod refiner;
pub mod engine;

// Re-exports
pub use config::TrmConfig;
pub use engine::TrmEngine;
```

#### 2.1.1 TRM Configuration

```rust
// src/trm/config.rs

/// Configuration for TRM recursive engine
#[derive(Clone, Debug)]
pub struct TrmConfig {
    /// Hidden dimension for latent state
    pub hidden_dim: usize,

    /// Embedding dimension (input/output)
    pub embedding_dim: usize,

    /// Maximum answer sequence length
    pub max_answer_len: usize,

    /// Maximum recursion depth (K)
    pub max_k: usize,

    /// Default recursion depth
    pub default_k: usize,

    /// Latent updates per K iteration (n)
    pub latent_iterations: usize,

    /// Use attention variant (true) or MLP variant (false)
    pub use_attention: bool,

    /// Number of attention heads (if using attention)
    pub num_heads: usize,

    /// Dropout probability
    pub dropout: f32,

    /// Confidence threshold for early stopping
    pub confidence_threshold: f32,

    /// Enable SIMD optimizations
    pub use_simd: bool,
}

impl Default for TrmConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            embedding_dim: 256,
            max_answer_len: 64,
            max_k: 20,
            default_k: 5,
            latent_iterations: 3,
            use_attention: false,  // MLP is faster
            num_heads: 8,
            dropout: 0.1,
            confidence_threshold: 0.95,
            use_simd: true,
        }
    }
}
```

#### 2.1.2 TRM Engine Core

```rust
// src/trm/engine.rs

use crate::trm::{TrmConfig, LatentUpdater, AnswerRefiner, ConfidenceScorer};
use crate::sona::types::QueryTrajectory;

/// Main TRM recursive reasoning engine
pub struct TrmEngine {
    config: TrmConfig,
    latent_updater: LatentUpdater,
    answer_refiner: AnswerRefiner,
    confidence_scorer: ConfidenceScorer,

    // Pre-allocated buffers
    latent_buffer: Vec<f32>,
    answer_buffer: Vec<f32>,
}

/// Result of TRM recursive reasoning
#[derive(Clone, Debug)]
pub struct TrmResult {
    /// Final answer embedding
    pub answer: Vec<f32>,

    /// Confidence score [0, 1]
    pub confidence: f32,

    /// Number of iterations actually used
    pub iterations_used: usize,

    /// Whether early stopping was triggered
    pub early_stopped: bool,

    /// Trajectory for SONA learning
    pub trajectory: TrmTrajectory,

    /// Total latency in microseconds
    pub latency_us: u64,
}

/// Single iteration state for trajectory
#[derive(Clone, Debug)]
pub struct TrmIterationState {
    pub iteration: usize,
    pub latent_state: Vec<f32>,
    pub answer_state: Vec<f32>,
    pub confidence: f32,
    pub latency_us: u64,
}

/// Full trajectory of TRM reasoning
#[derive(Clone, Debug)]
pub struct TrmTrajectory {
    pub states: Vec<TrmIterationState>,
    pub optimal_k: usize,
    pub total_latency_us: u64,
}

impl TrmEngine {
    pub fn new(config: TrmConfig) -> Self {
        let latent_updater = if config.use_attention {
            LatentUpdater::attention(
                config.hidden_dim,
                config.embedding_dim,
                config.num_heads,
            )
        } else {
            LatentUpdater::mlp(
                config.hidden_dim,
                config.embedding_dim,
            )
        };

        Self {
            latent_buffer: vec![0.0; config.hidden_dim],
            answer_buffer: vec![0.0; config.max_answer_len * config.embedding_dim],
            latent_updater,
            answer_refiner: AnswerRefiner::new(&config),
            confidence_scorer: ConfidenceScorer::new(&config),
            config,
        }
    }

    /// Run recursive reasoning
    pub fn reason(
        &mut self,
        question: &[f32],     // [q_len * dim]
        initial_answer: &[f32], // [a_len * dim]
        k: Option<usize>,
    ) -> TrmResult {
        let start = std::time::Instant::now();
        let max_k = k.unwrap_or(self.config.default_k).min(self.config.max_k);

        // Initialize
        self.latent_buffer.fill(0.0);
        self.answer_buffer[..initial_answer.len()].copy_from_slice(initial_answer);

        let mut trajectory = TrmTrajectory {
            states: Vec::with_capacity(max_k),
            optimal_k: 0,
            total_latency_us: 0,
        };

        let mut final_confidence = 0.0;

        // Main recursive loop
        for k_iter in 0..max_k {
            let iter_start = std::time::Instant::now();

            // Phase 1: Latent updates (n iterations)
            for _ in 0..self.config.latent_iterations {
                self.latent_updater.update(
                    question,
                    &self.answer_buffer,
                    &mut self.latent_buffer,
                );
            }

            // Phase 2: Answer refinement
            self.answer_refiner.refine(
                question,
                &self.latent_buffer,
                &mut self.answer_buffer,
            );

            // Compute confidence
            let confidence = self.confidence_scorer.score(&self.answer_buffer);

            let iter_latency = iter_start.elapsed().as_micros() as u64;

            // Record state
            trajectory.states.push(TrmIterationState {
                iteration: k_iter,
                latent_state: self.latent_buffer.clone(),
                answer_state: self.answer_buffer.clone(),
                confidence,
                latency_us: iter_latency,
            });

            final_confidence = confidence;

            // Early stopping
            if confidence >= self.config.confidence_threshold {
                trajectory.optimal_k = k_iter + 1;
                break;
            }

            trajectory.optimal_k = k_iter + 1;
        }

        let total_latency = start.elapsed().as_micros() as u64;
        trajectory.total_latency_us = total_latency;

        TrmResult {
            answer: self.answer_buffer.clone(),
            confidence: final_confidence,
            iterations_used: trajectory.optimal_k,
            early_stopped: final_confidence >= self.config.confidence_threshold,
            trajectory,
            latency_us: total_latency,
        }
    }

    /// Reason with predicted optimal K from SONA
    pub fn reason_adaptive(
        &mut self,
        question: &[f32],
        initial_answer: &[f32],
        predicted_k: usize,
    ) -> TrmResult {
        // Add 20% buffer for safety
        let k = (predicted_k as f32 * 1.2).ceil() as usize;
        self.reason(question, initial_answer, Some(k))
    }
}
```

### 2.2 Latent Updater Variants

#### 2.2.1 MLP Variant

```rust
// src/trm/mlp.rs

use crate::simd::SimdOps;

/// MLP-based latent updater (faster, default)
pub struct MlpLatentUpdater {
    // Linear 1: combined_dim -> hidden_dim * 4
    w1: Vec<f32>,
    b1: Vec<f32>,

    // Linear 2: hidden_dim * 4 -> hidden_dim
    w2: Vec<f32>,
    b2: Vec<f32>,

    // Gate: hidden_dim -> hidden_dim
    w_gate: Vec<f32>,
    b_gate: Vec<f32>,

    // Layer norm
    ln_gamma: Vec<f32>,
    ln_beta: Vec<f32>,

    hidden_dim: usize,
    combined_dim: usize,

    // Scratch buffers
    combined_buffer: Vec<f32>,
    hidden_buffer: Vec<f32>,
    gate_buffer: Vec<f32>,
}

impl MlpLatentUpdater {
    pub fn new(hidden_dim: usize, embedding_dim: usize) -> Self {
        let combined_dim = hidden_dim + embedding_dim * 2; // latent + question + answer

        Self {
            w1: Self::init_weights(combined_dim, hidden_dim * 4),
            b1: vec![0.0; hidden_dim * 4],
            w2: Self::init_weights(hidden_dim * 4, hidden_dim),
            b2: vec![0.0; hidden_dim],
            w_gate: Self::init_weights(hidden_dim, hidden_dim),
            b_gate: vec![0.0; hidden_dim],
            ln_gamma: vec![1.0; hidden_dim],
            ln_beta: vec![0.0; hidden_dim],
            hidden_dim,
            combined_dim,
            combined_buffer: vec![0.0; combined_dim],
            hidden_buffer: vec![0.0; hidden_dim * 4],
            gate_buffer: vec![0.0; hidden_dim],
        }
    }

    fn init_weights(in_dim: usize, out_dim: usize) -> Vec<f32> {
        // Xavier initialization
        let std = (2.0 / (in_dim + out_dim) as f32).sqrt();
        (0..in_dim * out_dim)
            .map(|i| {
                let x = (i as f32 * 0.618033988749) % 1.0; // Golden ratio hash
                (x - 0.5) * 2.0 * std
            })
            .collect()
    }

    pub fn update(
        &mut self,
        question_pooled: &[f32],  // [dim]
        answer_pooled: &[f32],    // [dim]
        latent: &mut [f32],       // [hidden_dim]
    ) {
        let h = self.hidden_dim;

        // Combine inputs
        self.combined_buffer[..question_pooled.len()].copy_from_slice(question_pooled);
        self.combined_buffer[question_pooled.len()..question_pooled.len() + answer_pooled.len()]
            .copy_from_slice(answer_pooled);
        self.combined_buffer[question_pooled.len() + answer_pooled.len()..].copy_from_slice(latent);

        // Linear 1 + GELU
        SimdOps::matmul_add(
            &self.combined_buffer,
            &self.w1,
            &self.b1,
            &mut self.hidden_buffer,
            1,
            self.combined_dim,
            h * 4,
        );
        SimdOps::gelu_inplace(&mut self.hidden_buffer);

        // Linear 2
        let mut delta = vec![0.0; h];
        SimdOps::matmul_add(
            &self.hidden_buffer,
            &self.w2,
            &self.b2,
            &mut delta,
            1,
            h * 4,
            h,
        );

        // Gate
        SimdOps::matmul_add(
            latent,
            &self.w_gate,
            &self.b_gate,
            &mut self.gate_buffer,
            1,
            h,
            h,
        );
        SimdOps::sigmoid_inplace(&mut self.gate_buffer);

        // Gated residual: latent = gate * latent + (1 - gate) * delta
        for i in 0..h {
            latent[i] = self.gate_buffer[i] * latent[i] +
                       (1.0 - self.gate_buffer[i]) * delta[i];
        }

        // Layer norm
        SimdOps::layer_norm_inplace(latent, &self.ln_gamma, &self.ln_beta, 1e-5);
    }
}
```

#### 2.2.2 Attention Variant

```rust
// src/trm/attention.rs

/// Attention-based latent updater (more expressive)
pub struct AttentionLatentUpdater {
    num_heads: usize,
    head_dim: usize,
    hidden_dim: usize,

    // Query projection for latent
    w_q: Vec<f32>,

    // Key/Value projection for context
    w_k: Vec<f32>,
    w_v: Vec<f32>,

    // Output projection
    w_o: Vec<f32>,

    // Layer norm
    ln_gamma: Vec<f32>,
    ln_beta: Vec<f32>,

    // Buffers
    q_buffer: Vec<f32>,
    k_buffer: Vec<f32>,
    v_buffer: Vec<f32>,
    scores_buffer: Vec<f32>,
    attended_buffer: Vec<f32>,
}

impl AttentionLatentUpdater {
    pub fn new(hidden_dim: usize, embedding_dim: usize, num_heads: usize) -> Self {
        assert!(hidden_dim % num_heads == 0);
        let head_dim = hidden_dim / num_heads;

        Self {
            num_heads,
            head_dim,
            hidden_dim,
            w_q: Self::init_weights(hidden_dim, hidden_dim),
            w_k: Self::init_weights(embedding_dim * 2, hidden_dim),
            w_v: Self::init_weights(embedding_dim * 2, hidden_dim),
            w_o: Self::init_weights(hidden_dim, hidden_dim),
            ln_gamma: vec![1.0; hidden_dim],
            ln_beta: vec![0.0; hidden_dim],
            q_buffer: vec![0.0; hidden_dim],
            k_buffer: vec![0.0; hidden_dim * 2], // For question + answer
            v_buffer: vec![0.0; hidden_dim * 2],
            scores_buffer: vec![0.0; num_heads * 2],
            attended_buffer: vec![0.0; hidden_dim],
        }
    }

    fn init_weights(in_dim: usize, out_dim: usize) -> Vec<f32> {
        let std = (2.0 / (in_dim + out_dim) as f32).sqrt();
        (0..in_dim * out_dim)
            .map(|i| ((i as f32 * 0.618033988749) % 1.0 - 0.5) * 2.0 * std)
            .collect()
    }

    pub fn update(
        &mut self,
        question_pooled: &[f32],
        answer_pooled: &[f32],
        latent: &mut [f32],
    ) {
        // ... attention computation
        // Similar to pseudocode in 02_PSEUDOCODE.md
    }
}
```

### 2.3 Integration with SONA

```rust
// src/trm/sona_bridge.rs

use crate::sona::{SonaEngine, QueryTrajectory, TrajectoryStep};
use crate::trm::{TrmResult, TrmTrajectory};

/// Bridge between TRM and SONA for learning
pub struct TrmSonaBridge {
    sona: SonaEngine,
}

impl TrmSonaBridge {
    pub fn new(sona: SonaEngine) -> Self {
        Self { sona }
    }

    /// Convert TRM trajectory to SONA format and submit for learning
    pub fn learn_from_trm(&self, query_embedding: &[f32], result: &TrmResult, quality: f32) {
        let trajectory = self.convert_trajectory(query_embedding, result, quality);
        self.sona.submit_trajectory(trajectory);
    }

    /// Predict optimal K for a query
    pub fn predict_optimal_k(&self, query_embedding: &[f32], max_k: usize) -> usize {
        let patterns = self.sona.find_patterns(query_embedding, 5);

        if patterns.is_empty() {
            return 5; // Default
        }

        let mut weighted_k = 0.0;
        let mut total_weight = 0.0;

        for pattern in &patterns {
            let similarity = cosine_similarity(query_embedding, &pattern.centroid);
            if let Some(k) = pattern.metadata.get("optimal_k") {
                if let Ok(k_val) = k.parse::<f32>() {
                    weighted_k += similarity * k_val;
                    total_weight += similarity;
                }
            }
        }

        if total_weight > 0.0 {
            (weighted_k / total_weight).round().clamp(1.0, max_k as f32) as usize
        } else {
            5
        }
    }

    fn convert_trajectory(
        &self,
        query_embedding: &[f32],
        result: &TrmResult,
        quality: f32,
    ) -> QueryTrajectory {
        let mut trajectory = QueryTrajectory {
            id: self.sona.next_trajectory_id(),
            embedding: query_embedding.to_vec(),
            steps: Vec::with_capacity(result.trajectory.states.len()),
            quality,
            model_route: Some("trm_recursive".to_string()),
            context_tags: vec!["reasoning".to_string()],
        };

        for state in &result.trajectory.states {
            trajectory.steps.push(TrajectoryStep {
                token_id: state.iteration as u32,
                activations: state.latent_state.clone(),
                attention_weights: vec![], // No token attention in TRM
                confidence: state.confidence,
                latency_us: state.latency_us,
            });
        }

        // Store optimal K in metadata
        trajectory.context_tags.push(format!("k:{}", result.iterations_used));

        trajectory
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-8)
}
```

---

## 3. Module Structure

```
examples/ruvLLM/
├── src/
│   ├── lib.rs                    # Library entry point
│   ├── config.rs                 # Global configuration
│   ├── error.rs                  # Error types
│   ├── types.rs                  # Core domain types
│   │
│   ├── trm/                      # TRM Recursive Engine (NEW)
│   │   ├── mod.rs                # Module exports
│   │   ├── config.rs             # TRM configuration
│   │   ├── engine.rs             # Main TRM engine
│   │   ├── latent.rs             # Latent updater enum/trait
│   │   ├── mlp.rs                # MLP latent updater
│   │   ├── attention.rs          # Attention latent updater
│   │   ├── refiner.rs            # Answer refinement
│   │   ├── confidence.rs         # Confidence scoring
│   │   ├── sona_bridge.rs        # SONA integration
│   │   └── tests.rs              # Unit tests
│   │
│   ├── sona/                     # SONA Engine (existing)
│   │   ├── mod.rs
│   │   ├── types.rs
│   │   ├── engine.rs
│   │   ├── lora.rs
│   │   ├── ewc.rs
│   │   ├── reasoning_bank.rs
│   │   ├── trajectory.rs
│   │   └── loops/
│   │       ├── instant.rs
│   │       ├── background.rs
│   │       └── coordinator.rs
│   │
│   ├── orchestrator.rs           # RuvLLM coordinator (updated)
│   ├── router.rs                 # FastGRNN router (updated for K prediction)
│   ├── memory.rs                 # HNSW memory service
│   ├── embedding.rs              # Embedding service
│   ├── attention.rs              # Graph attention
│   ├── simd_inference.rs         # SIMD operations
│   │
│   └── bin/
│       ├── demo.rs               # Interactive demo
│       ├── bench.rs              # Quick benchmarks
│       ├── benchmark_suite.rs    # Full benchmark suite
│       ├── trm_demo.rs           # TRM-specific demo (NEW)
│       ├── trm_bench.rs          # TRM benchmarks (NEW)
│       └── export.rs             # HuggingFace export
│
├── tests/
│   ├── integration.rs
│   ├── trm_integration.rs        # TRM integration tests (NEW)
│   └── trm_sona_integration.rs   # TRM+SONA tests (NEW)
│
├── benches/
│   ├── pipeline.rs
│   ├── trm_recursive.rs          # TRM benchmarks (NEW)
│   └── trm_sona.rs               # Combined benchmarks (NEW)
│
└── docs/
    └── sparc/
        └── trm-integration/      # This documentation
```

---

## 4. Interface Contracts

### 4.1 TRM Engine Interface

```rust
/// Main TRM reasoning interface
pub trait RecursiveReasoner {
    /// Perform recursive reasoning
    fn reason(
        &mut self,
        question: &[f32],
        initial_answer: &[f32],
        max_k: Option<usize>,
    ) -> TrmResult;

    /// Get configuration
    fn config(&self) -> &TrmConfig;

    /// Reset internal state
    fn reset(&mut self);
}

/// Latent update interface
pub trait LatentUpdate {
    fn update(
        &mut self,
        question_pooled: &[f32],
        answer_pooled: &[f32],
        latent: &mut [f32],
    );

    fn hidden_dim(&self) -> usize;
}

/// Answer refinement interface
pub trait AnswerRefine {
    fn refine(
        &mut self,
        question: &[f32],
        latent: &[f32],
        answer: &mut [f32],
    );
}
```

### 4.2 SONA Bridge Interface

```rust
/// Bridge for TRM-SONA integration
pub trait TrmLearning {
    /// Submit TRM result for learning
    fn learn_from_trm(&self, query: &[f32], result: &TrmResult, quality: f32);

    /// Predict optimal K for query
    fn predict_optimal_k(&self, query: &[f32], max_k: usize) -> usize;

    /// Get learning statistics
    fn learning_stats(&self) -> TrmLearningStats;
}

#[derive(Clone, Debug)]
pub struct TrmLearningStats {
    pub patterns_learned: usize,
    pub avg_predicted_k: f32,
    pub prediction_accuracy: f32,
    pub cache_hit_rate: f32,
}
```

### 4.3 Updated Router Interface

```rust
/// Extended routing decision for TRM
#[derive(Clone, Debug)]
pub struct TrmRoutingDecision {
    /// Use TRM recursive reasoning
    pub use_trm: bool,

    /// Predicted optimal K
    pub k_value: usize,

    /// Latent iterations per K
    pub n_value: usize,

    /// Use attention variant
    pub use_attention: bool,

    /// Confidence in routing decision
    pub confidence: f32,

    /// Reasoning for decision
    pub reason: String,
}
```

---

## 5. Data Flow

### 5.1 Query Processing Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Query Processing Flow                            │
└──────────────────────────────────────────────────────────────────────────┘

1. Input Query
   ↓
2. Embedding Service → [query_embedding: f32[256]]
   ↓
3. Memory Search (HNSW)
   ├── Cache Hit (>0.95 similarity) → Return cached result
   └── Cache Miss → Continue
   ↓
4. K Predictor (SONA)
   ├── Find similar patterns in ReasoningBank
   └── Weighted average → predicted_k
   ↓
5. Router Decision
   ├── Complexity estimation
   ├── K prediction confidence
   └── → TrmRoutingDecision
   ↓
6. TRM Recursive Engine
   ├── Initialize: latent=0, answer=initial
   ├── FOR k=1 to K:
   │   ├── Latent Update (n times)
   │   ├── Answer Refinement
   │   ├── Confidence Check
   │   └── Record Trajectory
   └── Return TrmResult
   ↓
7. SONA Learning
   ├── Convert trajectory
   ├── MicroLoRA update
   └── Store pattern in ReasoningBank
   ↓
8. Memory Storage
   ├── Cache result with embedding key
   └── Store optimal_k metadata
   ↓
9. Return Response
```

### 5.2 Learning Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Learning Flow                                    │
└──────────────────────────────────────────────────────────────────────────┘

TRM Trajectory Completion
   ↓
Quality Assessment (user feedback or automatic)
   ↓
┌─ Loop A: Instant (<100μs) ──────────────────────────────────────────────┐
│  • Extract features: query_embedding, optimal_k, convergence_pattern    │
│  • MicroLoRA gradient update for K prediction                           │
│  • Buffer trajectory for background processing                          │
└─────────────────────────────────────────────────────────────────────────┘
   ↓
┌─ Loop B: Background (hourly) ───────────────────────────────────────────┐
│  • K-means++ clustering on trajectory embeddings                        │
│  • Extract patterns: common K values per cluster                        │
│  • BaseLoRA update from successful patterns                             │
│  • Store patterns in ReasoningBank                                      │
└─────────────────────────────────────────────────────────────────────────┘
   ↓
┌─ Loop C: Deep (weekly) ─────────────────────────────────────────────────┐
│  • EWC++ consolidation                                                  │
│  • Archive old patterns                                                 │
│  • Update concept hierarchies                                           │
│  • Cross-validate K predictions                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Performance Targets

| Component | Latency Target | Memory Target |
|-----------|---------------|---------------|
| Single Latent Update | <500μs | 1KB |
| Single K Iteration | <2ms | 10KB |
| Full K=10 Reasoning | <50ms | 100KB |
| K Prediction | <1ms | 10KB |
| SONA Learning | <5ms | 50KB |
| Memory Lookup | <1ms | - |

---

## 7. Error Handling

```rust
/// TRM-specific errors
#[derive(Debug, thiserror::Error)]
pub enum TrmError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid K value: {0} (max: {1})")]
    InvalidK(usize, usize),

    #[error("Latent update failed: {0}")]
    LatentUpdateFailed(String),

    #[error("Answer refinement failed: {0}")]
    RefinementFailed(String),

    #[error("SIMD operation failed: {0}")]
    SimdError(String),

    #[error("Memory allocation failed")]
    AllocationFailed,
}
```

---

**Next**: [04_REFINEMENT.md](./04_REFINEMENT.md) - TDD Implementation Plan
