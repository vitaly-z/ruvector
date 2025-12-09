# API Reference

## Overview

This document provides the complete API reference for RuvDLLM modules. The API is designed for both high-level ease of use and low-level control when needed.

## Module Structure

```rust
// Top-level re-exports
pub mod diffusion {
    pub mod model;
    pub mod sampler;
    pub mod scheduler;
    pub mod simd;
}

pub mod micro_lora {
    pub mod bank;
    pub mod talora;
    pub mod dgr;
    pub mod composition;
}

pub mod federation {
    pub mod daf;
    pub mod gossip;
    pub mod privacy;
    pub mod tiers;
}

pub mod gpu {
    pub mod cuda;
    pub mod metal;
    pub mod vulkan;
}
```

## Diffusion Model API

### DiffusionModel

```rust
/// Core diffusion language model
pub struct DiffusionModel {
    // Private fields
}

impl DiffusionModel {
    /// Load model from Q4 quantized safetensors
    ///
    /// # Arguments
    /// * `path` - Path to model directory containing safetensors and config
    ///
    /// # Example
    /// ```rust
    /// let model = DiffusionModel::load_q4("models/llama-7b-diffusion-q4")?;
    /// ```
    pub fn load_q4(path: impl AsRef<Path>) -> Result<Self, ModelError>;

    /// Load model from unquantized weights
    pub fn load_f16(path: impl AsRef<Path>) -> Result<Self, ModelError>;

    /// Forward pass with timestep conditioning
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch_size, seq_len]
    /// * `timestep` - Current diffusion timestep (0-1000)
    /// * `attention_mask` - Optional attention mask
    ///
    /// # Returns
    /// Logits tensor [batch_size, seq_len, vocab_size]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        timestep: u32,
        attention_mask: Option<&Tensor>,
    ) -> Tensor;

    /// Forward with hidden states (for DGR)
    pub fn forward_with_hidden(
        &self,
        input_ids: &Tensor,
        timestep: u32,
    ) -> (Tensor, Tensor); // (hidden_states, logits)

    /// Get model configuration
    pub fn config(&self) -> &DiffusionConfig;

    /// Get noise schedule
    pub fn schedule(&self) -> &NoiseSchedule;
}
```

### DiffusionConfig

```rust
/// Model configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiffusionConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// FFN intermediate dimension
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_key_value_heads: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// RoPE theta
    pub rope_theta: f32,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// Number of diffusion timesteps
    pub num_timesteps: u32,
    /// Noise schedule type
    pub noise_schedule: NoiseScheduleType,
}

impl DiffusionConfig {
    /// Load from JSON file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError>;

    /// Create LLaMA-7B config
    pub fn llama_7b() -> Self;

    /// Create LLaMA-13B config
    pub fn llama_13b() -> Self;

    /// Create Qwen-7B config
    pub fn qwen_7b() -> Self;
}
```

### MDLMSampler

```rust
/// MDLM (Masked Diffusion Language Model) sampler
pub struct MDLMSampler {
    // Private fields
}

impl MDLMSampler {
    /// Create new sampler with default settings
    pub fn new() -> Self;

    /// Create sampler with custom configuration
    pub fn with_config(config: SamplerConfig) -> Self;

    /// Generate text from prompt
    ///
    /// # Arguments
    /// * `model` - The diffusion model
    /// * `prompt` - Text prompt
    /// * `max_length` - Maximum output length
    ///
    /// # Returns
    /// Generated text
    ///
    /// # Example
    /// ```rust
    /// let sampler = MDLMSampler::new();
    /// let output = sampler.generate(&model, "Once upon a time", 100)?;
    /// println!("{}", output);
    /// ```
    pub fn generate(
        &self,
        model: &DiffusionModel,
        prompt: &str,
        max_length: usize,
    ) -> Result<String, GenerationError>;

    /// Generate with TALoRA adapters
    pub fn generate_with_talora(
        &self,
        model: &DiffusionModel,
        prompt: &str,
        max_length: usize,
        talora: &TALoRAManager,
    ) -> Result<String, GenerationError>;

    /// Generate with full DGR system
    pub fn generate_with_dgr(
        &self,
        model: &DiffusionModel,
        prompt: &str,
        max_length: usize,
        dgr: &DGRSystem,
    ) -> Result<String, GenerationError>;

    /// Stream generation with callback
    pub fn generate_stream<F>(
        &self,
        model: &DiffusionModel,
        prompt: &str,
        max_length: usize,
        callback: F,
    ) -> Result<(), GenerationError>
    where
        F: FnMut(GenerationStep) -> bool;
}

/// Sampler configuration
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    /// Number of denoising steps (default: 8)
    pub num_steps: u32,
    /// Sampling temperature (default: 1.0)
    pub temperature: f32,
    /// Top-p nucleus sampling (default: 0.9)
    pub top_p: f32,
    /// Top-k sampling (default: 50)
    pub top_k: usize,
    /// Repetition penalty (default: 1.0)
    pub repetition_penalty: f32,
}
```

## TALoRA API

### TALoRAManager

```rust
/// Timestep-Aware LoRA manager
pub struct TALoRAManager {
    // Private fields
}

impl TALoRAManager {
    /// Create new TALoRA manager
    ///
    /// # Arguments
    /// * `config` - TALoRA configuration
    ///
    /// # Example
    /// ```rust
    /// let config = TALoRAConfig::default();
    /// let talora = TALoRAManager::new(config);
    /// ```
    pub fn new(config: TALoRAConfig) -> Self;

    /// Retrieve adapters for current context and timestep
    ///
    /// # Arguments
    /// * `query` - Query embedding (from hidden states)
    /// * `timestep` - Current diffusion timestep
    /// * `top_k` - Number of adapters to retrieve
    ///
    /// # Returns
    /// Retrieved adapters with similarity scores
    pub fn retrieve(
        &self,
        query: &[f32],
        timestep: u32,
        top_k: usize,
    ) -> TALoRARetrievalResult;

    /// Store new adapter
    ///
    /// # Arguments
    /// * `adapter` - MicroLoRA adapter to store
    /// * `query_embedding` - Context embedding for indexing
    /// * `timestep` - Timestep this adapter is trained for
    ///
    /// # Returns
    /// Adapter ID for later retrieval/removal
    pub fn store(
        &mut self,
        adapter: MicroLoRA,
        query_embedding: &[f32],
        timestep: u32,
    ) -> AdapterId;

    /// Remove adapter by ID
    pub fn remove(&mut self, id: AdapterId) -> Option<MicroLoRA>;

    /// Apply adapters to model output
    pub fn apply(
        &self,
        output: &mut Tensor,
        result: &TALoRARetrievalResult,
        timestep: u32,
    );

    /// Get bank statistics
    pub fn stats(&self) -> TALoRAStats;

    /// Save to disk
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), IoError>;

    /// Load from disk
    pub fn load(path: impl AsRef<Path>) -> Result<Self, IoError>;
}

/// TALoRA configuration
#[derive(Clone, Debug)]
pub struct TALoRAConfig {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Coarse bank capacity (default: 10000)
    pub coarse_capacity: usize,
    /// Coarse bank rank (default: 8)
    pub coarse_rank: usize,
    /// Domain bank capacity (default: 50000)
    pub domain_capacity: usize,
    /// Domain bank rank (default: 4)
    pub domain_rank: usize,
    /// Fine bank capacity (default: 100000)
    pub fine_capacity: usize,
    /// Fine bank rank (default: 2)
    pub fine_rank: usize,
    /// Timestep boundaries (default: [700, 300])
    pub boundaries: [u32; 2],
    /// Transition smoothing width (default: 50)
    pub transition_width: u32,
}
```

### MicroLoRA

```rust
/// Lightweight LoRA adapter
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MicroLoRA {
    // Private fields
}

impl MicroLoRA {
    /// Create new MicroLoRA adapter
    ///
    /// # Arguments
    /// * `rank` - LoRA rank (typically 1-4 for micro)
    /// * `hidden_dim` - Model hidden dimension
    ///
    /// # Example
    /// ```rust
    /// let lora = MicroLoRA::new(2, 4096);
    /// ```
    pub fn new(rank: usize, hidden_dim: usize) -> Self;

    /// Create from A and B matrices
    pub fn from_matrices(a: Tensor, b: Tensor) -> Self;

    /// Apply LoRA to input
    pub fn apply(&self, input: &Tensor) -> Tensor;

    /// Get rank
    pub fn rank(&self) -> usize;

    /// Get memory size in bytes
    pub fn size_bytes(&self) -> usize;

    /// Merge with another LoRA
    pub fn merge(&mut self, other: &MicroLoRA, weight: f32);

    /// Compress to 8-bit
    pub fn compress(&self) -> CompressedMicroLoRA;
}
```

## DGR API

### DGRSystem

```rust
/// Denoising-Guided Retrieval system
pub struct DGRSystem {
    // Private fields
}

impl DGRSystem {
    /// Create new DGR system
    ///
    /// # Arguments
    /// * `talora` - TALoRA manager for retrieval
    /// * `config` - DGR configuration
    pub fn new(talora: TALoRAManager, config: DGRConfig) -> Self;

    /// Process step with DGR
    ///
    /// # Arguments
    /// * `logits` - Model output logits
    /// * `hidden_states` - Hidden states for retrieval queries
    /// * `timestep` - Current timestep
    ///
    /// # Returns
    /// DGR result with position-specific adapters
    pub fn process(
        &self,
        logits: &Tensor,
        hidden_states: &Tensor,
        timestep: u32,
    ) -> DGRResult;

    /// Apply DGR adapters to output
    pub fn apply(&self, output: &mut Tensor, result: &DGRResult);

    /// Get uncertainty map
    pub fn get_uncertainty_map(
        &self,
        logits: &Tensor,
        timestep: u32,
    ) -> Tensor;

    /// Set uncertainty threshold
    pub fn set_threshold(&mut self, threshold: f32);

    /// Set maximum retrievals per step
    pub fn set_max_retrievals(&mut self, max: usize);
}

/// DGR configuration
#[derive(Clone, Debug)]
pub struct DGRConfig {
    /// Uncertainty threshold (default: 0.5)
    pub uncertainty_threshold: f32,
    /// Maximum retrievals per step (default: 10)
    pub max_retrievals: usize,
    /// Uncertainty metric (default: Entropy)
    pub metric: UncertaintyMetric,
    /// Top-k for each position (default: 3)
    pub top_k: usize,
}
```

## Federation API

### DAFAggregator

```rust
/// Diffusion-Aware Federation aggregator
pub struct DAFAggregator {
    // Private fields
}

impl DAFAggregator {
    /// Create new DAF aggregator
    pub fn new(config: DAFConfig) -> Self;

    /// Aggregate updates from clients
    ///
    /// # Arguments
    /// * `updates` - Client updates
    ///
    /// # Returns
    /// Aggregated update
    pub fn aggregate(&self, updates: Vec<ClientUpdate>) -> Result<AggregatedUpdate, DAFError>;

    /// Set aggregation strategy for bank
    pub fn set_strategy(&mut self, bank: usize, strategy: AggregationStrategy);

    /// Get current strategies
    pub fn strategies(&self) -> &[AggregationStrategy; 3];
}

/// DAF configuration
#[derive(Clone, Debug)]
pub struct DAFConfig {
    /// Strategy for coarse bank
    pub coarse_strategy: AggregationStrategy,
    /// Strategy for domain bank
    pub domain_strategy: AggregationStrategy,
    /// Strategy for fine bank
    pub fine_strategy: AggregationStrategy,
    /// Differential privacy epsilon
    pub dp_epsilon: f32,
    /// Differential privacy delta
    pub dp_delta: f32,
}

/// Aggregation strategies
#[derive(Clone, Debug)]
pub enum AggregationStrategy {
    WeightedAverage,
    QualityWeighted { metric: QualityMetric },
    Conservative { min_k: usize, threshold: f32 },
    Selective { method: ClusteringMethod },
}
```

### FederationClient

```rust
/// Federation client for participating in federated learning
pub struct FederationClient {
    // Private fields
}

impl FederationClient {
    /// Create new federation client
    pub fn new(config: FederationClientConfig) -> Result<Self, FederationError>;

    /// Connect to federation network
    pub async fn connect(&mut self) -> Result<(), FederationError>;

    /// Submit local update
    pub async fn submit_update(&self, update: LocalUpdate) -> Result<(), FederationError>;

    /// Receive aggregated update
    pub async fn receive_update(&self) -> Result<AggregatedUpdate, FederationError>;

    /// Get client status
    pub fn status(&self) -> ClientStatus;

    /// Disconnect from network
    pub async fn disconnect(&mut self) -> Result<(), FederationError>;
}
```

### PrivacyManager

```rust
/// Privacy management for tiered data sharing
pub struct PrivacyManager {
    // Private fields
}

impl PrivacyManager {
    /// Create new privacy manager
    pub fn new(config: PrivacyConfig) -> Self;

    /// Store pattern with privacy tier
    pub async fn store_pattern(
        &mut self,
        pattern: RawPattern,
        tier: PrivacyTier,
    ) -> Result<PatternId, PrivacyError>;

    /// Get consent for tier
    pub async fn request_consent(&self, tier: PrivacyTier) -> Result<ConsentRecord, ConsentError>;

    /// Revoke consent
    pub async fn revoke_consent(&mut self, tier: PrivacyTier) -> Result<(), ConsentError>;

    /// Promote pattern to higher tier
    pub async fn promote_pattern(
        &mut self,
        id: PatternId,
        target_tier: PrivacyTier,
    ) -> Result<(), PrivacyError>;

    /// Get privacy report
    pub async fn generate_report(&self) -> Result<PrivacyReport, AuditError>;
}

/// Privacy tiers
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PrivacyTier {
    /// Never leaves device
    Private,
    /// Shared within group
    Group,
    /// Organization-wide
    Tenant,
    /// Global with DP
    Public,
}
```

## GPU Backend API

### ComputeBackend

```rust
/// Unified compute backend
pub enum ComputeBackend {
    CUDA(CUDABackend),
    Metal(MetalBackend),
    Vulkan(VulkanBackend),
    SIMD(SIMDBackend),
    Scalar(ScalarBackend),
}

impl ComputeBackend {
    /// Auto-detect best backend
    pub fn auto_detect() -> Result<Self, BackendError>;

    /// Create specific backend
    pub fn cuda(device_id: usize) -> Result<Self, BackendError>;
    pub fn metal() -> Result<Self, BackendError>;
    pub fn vulkan() -> Result<Self, BackendError>;
    pub fn simd() -> Self;
    pub fn scalar() -> Self;

    /// Get backend name
    pub fn name(&self) -> &'static str;

    /// Get device info
    pub fn device_info(&self) -> DeviceInfo;

    /// Run denoising step
    pub fn denoise_step(
        &mut self,
        x_t: &Tensor,
        timestep: u32,
        model: &DiffusionModel,
    ) -> Result<Tensor, BackendError>;

    /// Batch inference
    pub fn batch_forward(
        &mut self,
        inputs: &[Tensor],
        timesteps: &[u32],
        model: &DiffusionModel,
    ) -> Result<Vec<Tensor>, BackendError>;
}
```

## High-Level API

### RuvDLLM

```rust
/// High-level RuvDLLM interface
pub struct RuvDLLM {
    // Private fields
}

impl RuvDLLM {
    /// Create with default settings
    ///
    /// # Example
    /// ```rust
    /// let ruvdllm = RuvDLLM::load("models/llama-7b-diffusion")?;
    /// let output = ruvdllm.generate("Hello, world!", 100)?;
    /// ```
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, RuvDLLMError>;

    /// Create with configuration
    pub fn with_config(
        model_path: impl AsRef<Path>,
        config: RuvDLLMConfig,
    ) -> Result<Self, RuvDLLMError>;

    /// Generate text
    pub fn generate(&self, prompt: &str, max_length: usize) -> Result<String, RuvDLLMError>;

    /// Generate with options
    pub fn generate_with_options(
        &self,
        prompt: &str,
        options: GenerationOptions,
    ) -> Result<String, RuvDLLMError>;

    /// Stream generation
    pub fn generate_stream<F>(&self, prompt: &str, max_length: usize, callback: F) -> Result<(), RuvDLLMError>
    where
        F: FnMut(&str) -> bool;

    /// Enable learning from interaction
    pub fn enable_learning(&mut self, config: LearningConfig);

    /// Disable learning
    pub fn disable_learning(&mut self);

    /// Join federation
    pub async fn join_federation(&mut self, config: FederationJoinConfig) -> Result<(), FederationError>;

    /// Leave federation
    pub async fn leave_federation(&mut self) -> Result<(), FederationError>;

    /// Get model info
    pub fn info(&self) -> ModelInfo;

    /// Get statistics
    pub fn stats(&self) -> RuvDLLMStats;
}

/// High-level configuration
#[derive(Clone, Debug)]
pub struct RuvDLLMConfig {
    /// Compute backend (auto-detect if None)
    pub backend: Option<BackendType>,
    /// Enable TALoRA
    pub enable_talora: bool,
    /// Enable DGR
    pub enable_dgr: bool,
    /// TALoRA config
    pub talora_config: Option<TALoRAConfig>,
    /// DGR config
    pub dgr_config: Option<DGRConfig>,
    /// Sampler config
    pub sampler_config: SamplerConfig,
    /// Privacy tier for learning
    pub privacy_tier: PrivacyTier,
}
```

## Error Types

```rust
/// Model errors
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Failed to load model: {0}")]
    LoadError(String),
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    #[error("Quantization error: {0}")]
    QuantizationError(String),
}

/// Generation errors
#[derive(Debug, thiserror::Error)]
pub enum GenerationError {
    #[error("Tokenization failed: {0}")]
    TokenizationError(String),
    #[error("Context length exceeded")]
    ContextLengthExceeded,
    #[error("Generation interrupted")]
    Interrupted,
}

/// Federation errors
#[derive(Debug, thiserror::Error)]
pub enum FederationError {
    #[error("Connection failed: {0}")]
    ConnectionError(String),
    #[error("Authentication failed")]
    AuthError,
    #[error("Insufficient peers for aggregation")]
    InsufficientPeers,
    #[error("Privacy violation detected")]
    PrivacyViolation,
}

/// Privacy errors
#[derive(Debug, thiserror::Error)]
pub enum PrivacyError {
    #[error("PII detected in pattern")]
    PIIDetected,
    #[error("Consent required for tier {0:?}")]
    ConsentRequired(PrivacyTier),
    #[error("Privacy budget exhausted")]
    BudgetExhausted,
}
```

---

**Previous**: [08-IMPLEMENTATION.md](./08-IMPLEMENTATION.md) - Implementation plan
**Next**: [10-BENCHMARKS.md](./10-BENCHMARKS.md) - Performance benchmarks
