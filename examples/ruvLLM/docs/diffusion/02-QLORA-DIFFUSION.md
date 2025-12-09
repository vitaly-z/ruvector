# QLoRA AR→Diffusion Conversion

## Overview

Converting autoregressive (AR) language models to diffusion models enables parallel generation and bidirectional context. This module implements efficient conversion using QLoRA to minimize compute requirements.

## Theoretical Foundation

### Why AR→Diffusion Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AR vs Diffusion Paradigms                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  AUTOREGRESSIVE (AR):                                                       │
│  ────────────────────                                                       │
│  P(x) = ∏ᵢ P(xᵢ | x₁, ..., xᵢ₋₁)                                          │
│                                                                              │
│  • Generates left-to-right                                                  │
│  • Causal attention mask                                                    │
│  • O(n) sequential generation                                               │
│                                                                              │
│  DIFFUSION (MDLM):                                                          │
│  ─────────────────                                                          │
│  P(x) = P(x₀) ∏ₜ P(xₜ₋₁ | xₜ)                                              │
│                                                                              │
│  • Generates by iterative denoising                                         │
│  • Bidirectional attention mask                                             │
│  • O(1) parallel token prediction per step                                  │
│                                                                              │
│  KEY INSIGHT:                                                               │
│  Both use same transformer backbone - only attention mask differs!          │
│  AR weights can be adapted to diffusion with minimal training.              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Conversion Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AR→Diffusion Conversion Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: Load AR Model (Frozen, Q4)                                         │
│  ───────────────────────────────────                                        │
│  • Load LLaMA/Qwen/etc in Q4 format                                        │
│  • Freeze all weights                                                       │
│  • ~4GB memory for 7B model                                                │
│                                                                              │
│  STEP 2: Add QLoRA Adapters                                                 │
│  ──────────────────────────────                                             │
│  • Inject LoRA into Q, K, V, O projections                                 │
│  • Rank 32-64 for conversion                                               │
│  • ~100MB trainable parameters                                             │
│                                                                              │
│  STEP 3: Add Time Embedding                                                 │
│  ──────────────────────────────                                             │
│  • Sinusoidal time embedding                                               │
│  • Project to hidden dimension                                             │
│  • Add to input embeddings                                                 │
│                                                                              │
│  STEP 4: Attention Mask Annealing                                          │
│  ─────────────────────────────────                                          │
│  • Gradually transition from causal → bidirectional                        │
│  • Over 10K-50K training steps                                             │
│  • Prevents catastrophic forgetting                                        │
│                                                                              │
│  STEP 5: Train with Masked Diffusion Objective                             │
│  ──────────────────────────────────────────────                             │
│  • Randomly mask tokens                                                    │
│  • Predict original tokens                                                 │
│  • Cross-entropy loss                                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Core Structures

```rust
/// QLoRA configuration for AR→Diffusion conversion
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QLoraConversionConfig {
    /// LoRA rank for conversion (32-64 recommended)
    pub rank: usize,

    /// LoRA alpha (scaling factor, typically 2x rank)
    pub alpha: f32,

    /// Target modules for LoRA injection
    pub target_modules: Vec<TargetModule>,

    /// Quantization bits for base model
    pub quantization_bits: u8,

    /// Attention mask annealing schedule
    pub mask_annealing: MaskAnnealingConfig,

    /// Time embedding configuration
    pub time_embedding: TimeEmbeddingConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TargetModule {
    QueryProj,
    KeyProj,
    ValueProj,
    OutputProj,
    GateProj,
    UpProj,
    DownProj,
}

impl Default for QLoraConversionConfig {
    fn default() -> Self {
        Self {
            rank: 32,
            alpha: 64.0,
            target_modules: vec![
                TargetModule::QueryProj,
                TargetModule::KeyProj,
                TargetModule::ValueProj,
                TargetModule::OutputProj,
            ],
            quantization_bits: 4,
            mask_annealing: MaskAnnealingConfig::default(),
            time_embedding: TimeEmbeddingConfig::default(),
        }
    }
}

/// Attention mask annealing configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaskAnnealingConfig {
    /// Total annealing steps
    pub total_steps: usize,

    /// Annealing schedule type
    pub schedule: AnnealingSchedule,

    /// Initial causal ratio (1.0 = fully causal)
    pub initial_causal_ratio: f32,

    /// Final causal ratio (0.0 = fully bidirectional)
    pub final_causal_ratio: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnnealingSchedule {
    Linear,
    Cosine,
    Exponential { decay: f32 },
    StepWise { steps: Vec<(usize, f32)> },
}

impl Default for MaskAnnealingConfig {
    fn default() -> Self {
        Self {
            total_steps: 20_000,
            schedule: AnnealingSchedule::Cosine,
            initial_causal_ratio: 1.0,
            final_causal_ratio: 0.0,
        }
    }
}
```

### Time Embedding

```rust
/// Sinusoidal time embedding for diffusion
pub struct TimeEmbedding {
    /// Embedding dimension
    dim: usize,

    /// Maximum timesteps
    max_timesteps: usize,

    /// Precomputed embeddings
    embeddings: Array2<f32>,

    /// Projection to hidden dimension
    proj: LinearLayer,
}

impl TimeEmbedding {
    pub fn new(dim: usize, hidden_dim: usize, max_timesteps: usize) -> Self {
        // Sinusoidal embedding
        let half_dim = dim / 2;
        let emb_scale = -(f32::ln(10000.0)) / (half_dim as f32);

        let mut embeddings = Array2::zeros((max_timesteps, dim));

        for t in 0..max_timesteps {
            let t_f = t as f32;
            for i in 0..half_dim {
                let freq = (i as f32 * emb_scale).exp();
                embeddings[[t, i]] = (t_f * freq).sin();
                embeddings[[t, half_dim + i]] = (t_f * freq).cos();
            }
        }

        Self {
            dim,
            max_timesteps,
            embeddings,
            proj: LinearLayer::new(dim, hidden_dim),
        }
    }

    /// Get time embedding for continuous timestep [0, 1]
    pub fn forward(&self, t: f32) -> Array1<f32> {
        // Interpolate between discrete timesteps
        let t_scaled = t * (self.max_timesteps - 1) as f32;
        let t_low = t_scaled.floor() as usize;
        let t_high = (t_low + 1).min(self.max_timesteps - 1);
        let alpha = t_scaled - t_low as f32;

        let emb_low = self.embeddings.row(t_low);
        let emb_high = self.embeddings.row(t_high);

        // Linear interpolation
        let emb = &emb_low * (1.0 - alpha) + &emb_high * alpha;

        // Project to hidden dimension
        self.proj.forward(&emb.to_owned())
    }
}
```

### Attention Mask Annealing

```rust
/// Manages gradual transition from causal to bidirectional attention
pub struct AttentionMaskAnnealer {
    config: MaskAnnealingConfig,
    current_step: usize,
}

impl AttentionMaskAnnealer {
    pub fn new(config: MaskAnnealingConfig) -> Self {
        Self {
            config,
            current_step: 0,
        }
    }

    /// Get current causal ratio [0, 1]
    pub fn get_causal_ratio(&self) -> f32 {
        let progress = (self.current_step as f32) / (self.config.total_steps as f32);
        let progress = progress.clamp(0.0, 1.0);

        let ratio = match &self.config.schedule {
            AnnealingSchedule::Linear => {
                1.0 - progress
            }
            AnnealingSchedule::Cosine => {
                0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
            }
            AnnealingSchedule::Exponential { decay } => {
                (-decay * progress).exp()
            }
            AnnealingSchedule::StepWise { steps } => {
                let mut ratio = self.config.initial_causal_ratio;
                for &(step, r) in steps {
                    if self.current_step >= step {
                        ratio = r;
                    }
                }
                ratio
            }
        };

        // Interpolate between initial and final ratios
        self.config.final_causal_ratio +
            ratio * (self.config.initial_causal_ratio - self.config.final_causal_ratio)
    }

    /// Generate attention mask with current causal ratio
    pub fn get_mask(&self, seq_len: usize) -> Array2<f32> {
        let causal_ratio = self.get_causal_ratio();

        let mut mask = Array2::ones((seq_len, seq_len));

        if causal_ratio > 0.0 {
            // Apply causal masking with probability = causal_ratio
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    // Probabilistic masking during training
                    // Deterministic boundary during inference
                    mask[[i, j]] = 1.0 - causal_ratio;
                }
            }
        }

        mask
    }

    /// Advance to next step
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Check if annealing is complete
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.config.total_steps
    }
}
```

### Conversion Trainer

```rust
/// Trainer for AR→Diffusion conversion
pub struct ConversionTrainer {
    /// Base model (Q4, frozen)
    base_model: Q4Model,

    /// QLoRA adapters (trainable)
    qlora: QLoraAdapters,

    /// Time embedding
    time_embed: TimeEmbedding,

    /// Mask annealer
    mask_annealer: AttentionMaskAnnealer,

    /// Optimizer
    optimizer: AdamW,

    /// Configuration
    config: ConversionTrainingConfig,

    /// Noise schedule for masking
    noise_schedule: NoiseSchedule,
}

impl ConversionTrainer {
    /// Single training step (SIMD optimized)
    pub fn train_step(&mut self, batch: &TokenBatch) -> TrainMetrics {
        let batch_size = batch.len();
        let mut total_loss = 0.0;

        // Sample random timesteps
        let timesteps: Vec<f32> = (0..batch_size)
            .map(|_| rand::random::<f32>())
            .collect();

        // Get current attention mask
        let causal_ratio = self.mask_annealer.get_causal_ratio();

        // Forward pass with gradient tracking
        let mut gradients = self.qlora.zero_grad();

        for (tokens, t) in batch.iter().zip(timesteps.iter()) {
            // Add noise (masking) according to schedule
            let mask_prob = self.noise_schedule.mask_probability(*t);
            let noisy_tokens = self.add_noise(tokens, mask_prob);

            // Get time embedding
            let time_emb = self.time_embed.forward(*t);

            // Forward through model
            let logits = self.forward_with_qlora(
                &noisy_tokens,
                &time_emb,
                causal_ratio,
            );

            // Compute loss (only on masked positions)
            let loss = self.masked_cross_entropy(&logits, tokens, &noisy_tokens);
            total_loss += loss;

            // Backward pass (SIMD optimized)
            self.backward_qlora(&logits, tokens, &noisy_tokens, &mut gradients);
        }

        // Optimizer step
        self.optimizer.step(&mut self.qlora, &gradients);

        // Advance annealer
        self.mask_annealer.step();

        TrainMetrics {
            loss: total_loss / batch_size as f32,
            causal_ratio,
            step: self.mask_annealer.current_step,
        }
    }

    /// Add noise by masking tokens
    fn add_noise(&self, tokens: &[u32], mask_prob: f32) -> Vec<u32> {
        let mask_token = self.config.mask_token_id;

        tokens.iter().map(|&token| {
            if rand::random::<f32>() < mask_prob {
                mask_token
            } else {
                token
            }
        }).collect()
    }

    /// Forward with QLoRA adapters
    fn forward_with_qlora(
        &self,
        tokens: &[u32],
        time_emb: &Array1<f32>,
        causal_ratio: f32,
    ) -> Array2<f32> {
        // Embed tokens
        let mut hidden = self.base_model.embed(tokens);

        // Add time embedding
        for row in hidden.rows_mut() {
            row.zip_mut_with(time_emb, |h, t| *h += t);
        }

        // Generate attention mask
        let attn_mask = self.generate_attn_mask(tokens.len(), causal_ratio);

        // Forward through layers
        for layer_idx in 0..self.base_model.num_layers() {
            hidden = self.forward_layer_with_qlora(
                layer_idx,
                hidden,
                &attn_mask,
            );
        }

        // LM head
        self.base_model.lm_head(&hidden)
    }
}
```

### CPU SIMD Training

```rust
impl ConversionTrainer {
    /// SIMD-optimized backward pass for QLoRA
    #[inline]
    pub fn backward_qlora(
        &self,
        logits: &Array2<f32>,
        targets: &[u32],
        noisy: &[u32],
        gradients: &mut QLoraGradients,
    ) {
        let mask_token = self.config.mask_token_id;

        // Compute loss gradient (softmax - one_hot)
        let mut loss_grad = self.softmax_2d(logits);
        for (i, &target) in targets.iter().enumerate() {
            // Only backprop through masked positions
            if noisy[i] == mask_token {
                loss_grad[[i, target as usize]] -= 1.0;
            } else {
                loss_grad.row_mut(i).fill(0.0);
            }
        }

        // Backward through layers (reverse order)
        let mut grad = loss_grad;

        for layer_idx in (0..self.base_model.num_layers()).rev() {
            grad = self.backward_layer_qlora(layer_idx, grad, gradients);
        }
    }

    /// SIMD-optimized layer backward
    fn backward_layer_qlora(
        &self,
        layer: usize,
        grad_out: Array2<f32>,
        gradients: &mut QLoraGradients,
    ) -> Array2<f32> {
        // Get cached activations
        let activation = self.activation_cache.get(layer);

        // Compute LoRA gradients
        // grad_B = activation.T @ grad_out
        // grad_A = (grad_out @ B.T).T @ activation

        let lora = &self.qlora.layers[layer];

        // SIMD matrix multiply for gradients
        let grad_b = SimdOps::matmul_tn(&activation, &grad_out);
        SimdOps::accumulate(&mut gradients.layers[layer].b, &grad_b, 1.0);

        let grad_through_b = SimdOps::matmul_nt(&grad_out, &lora.b);
        let grad_a = SimdOps::matmul_tn(&activation, &grad_through_b);
        SimdOps::accumulate(&mut gradients.layers[layer].a, &grad_a, lora.scale);

        // Propagate gradient through base model (frozen, just pass through)
        self.base_model.backward_frozen(layer, grad_out)
    }
}
```

## Training Recipe

### Recommended Hyperparameters

```toml
# conversion_config.toml

[model]
base_model = "llama-7b-q4.gguf"
qlora_rank = 32
qlora_alpha = 64
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

[training]
batch_size = 8
gradient_accumulation = 4
learning_rate = 1e-4
warmup_steps = 1000
total_steps = 50000
weight_decay = 0.01

[mask_annealing]
total_steps = 20000
schedule = "cosine"
initial_causal_ratio = 1.0
final_causal_ratio = 0.0

[noise_schedule]
type = "cosine"
num_timesteps = 1000

[time_embedding]
dim = 256
max_timesteps = 1000

[hardware]
device = "cpu"  # or "cuda", "metal"
num_threads = 16
use_simd = true
```

### Training Timeline (CPU)

| Phase | Steps | Time (16-core) | Description |
|-------|-------|----------------|-------------|
| Warmup | 0-1K | ~2 hours | LR warmup, stable gradients |
| Annealing | 1K-20K | ~40 hours | Mask transition |
| Refinement | 20K-50K | ~60 hours | Final tuning |
| **Total** | 50K | **~100 hours** | 4-5 days on good CPU |

### Training Timeline (GPU)

| Phase | Steps | Time (A100) | Description |
|-------|-------|-------------|-------------|
| Warmup | 0-1K | ~10 min | LR warmup |
| Annealing | 1K-20K | ~3 hours | Mask transition |
| Refinement | 20K-50K | ~5 hours | Final tuning |
| **Total** | 50K | **~8 hours** | Single GPU |

## Checkpoint Format

```rust
/// Checkpoint structure for converted model
#[derive(Serialize, Deserialize)]
pub struct DiffusionCheckpoint {
    /// Model configuration
    pub config: DiffusionModelConfig,

    /// QLoRA weights (safetensors format)
    pub qlora_weights: SafeTensors,

    /// Time embedding weights
    pub time_embed_weights: SafeTensors,

    /// Training metadata
    pub metadata: CheckpointMetadata,

    /// Noise schedule parameters
    pub noise_schedule: NoiseScheduleParams,
}

#[derive(Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub base_model: String,
    pub conversion_steps: usize,
    pub final_causal_ratio: f32,
    pub training_tokens: usize,
    pub created_at: DateTime<Utc>,
    pub ruvdllm_version: String,
}
```

## Usage

### Converting a Model

```rust
use ruvdllm::diffusion::{ConversionTrainer, QLoraConversionConfig};

// Load base model
let base_model = Q4Model::from_gguf("llama-7b-q4.gguf")?;

// Create conversion config
let config = QLoraConversionConfig {
    rank: 32,
    alpha: 64.0,
    ..Default::default()
};

// Create trainer
let mut trainer = ConversionTrainer::new(base_model, config)?;

// Training loop
let dataloader = TextDataLoader::new("training_data/")?;

for epoch in 0..num_epochs {
    for batch in dataloader.iter() {
        let metrics = trainer.train_step(&batch);

        if metrics.step % 1000 == 0 {
            println!("Step {}: loss={:.4}, causal_ratio={:.3}",
                metrics.step, metrics.loss, metrics.causal_ratio);
        }
    }

    // Save checkpoint
    trainer.save_checkpoint(&format!("checkpoint_epoch_{}.safetensors", epoch))?;
}

// Export final model
trainer.export("diffusion-llama-7b.safetensors")?;
```

### Loading Converted Model

```rust
use ruvdllm::diffusion::DiffusionModel;

// Load converted model
let model = DiffusionModel::load(
    "llama-7b-q4.gguf",           // Base weights
    "diffusion-llama-7b.safetensors", // Conversion weights
)?;

// Generate text
let response = model.generate("What is machine learning?", &GenerationConfig::default())?;
```

---

**Next**: [03-MICRO-LORA.md](./03-MICRO-LORA.md) - Real-time MicroLoRA adaptation
