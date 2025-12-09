# Novel Contributions

## Overview

RuvDLLM introduces three genuinely novel contributions to the field of diffusion language models. While individual components like diffusion LLMs, LoRA adapters, and federated learning exist separately, our specific combinations and implementations are new.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Novel Contributions Summary                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. TALoRA (Timestep-Aware LoRA)                                            │
│     • Different adapters for different denoising stages                     │
│     • Matches semantic content of each timestep                             │
│     • First application to diffusion text generation                        │
│                                                                              │
│  2. DGR (Denoising-Guided Retrieval)                                        │
│     • Uses model uncertainty during denoising to guide retrieval            │
│     • Dynamic adapter selection based on token confidence                   │
│     • Novel integration of RAG concepts with diffusion                      │
│                                                                              │
│  3. DAF (Diffusion-Aware Federation)                                        │
│     • Federated aggregation aligned to noise schedule semantics             │
│     • Per-timestep-group aggregation strategies                             │
│     • First diffusion-specific federated learning protocol                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## TALoRA: Timestep-Aware LoRA

### Motivation

In diffusion models, different timesteps serve different semantic purposes:

| Timestep Range | Noise Level | Semantic Role |
|----------------|-------------|---------------|
| t=1000→700 | High (σ≈1.0) | Global structure, syntax patterns |
| t=700→300 | Medium (σ≈0.5) | Domain semantics, topic coherence |
| t=300→0 | Low (σ≈0.1) | Fine details, style, word choice |

Standard LoRA applies the same adapter across all timesteps, ignoring this semantic structure. TALoRA uses different adapters for different stages.

### Comparison to Prior Work

| Approach | Timestep Awareness | Text Generation | Dynamic Selection |
|----------|-------------------|-----------------|-------------------|
| Standard LoRA | No | Yes | No |
| DiffuLLaMA | No | Yes | No |
| ControlNet (images) | Per-block | No (images) | No |
| T2I-Adapter (images) | Partial | No (images) | No |
| **TALoRA (ours)** | **Full** | **Yes** | **Yes** |

### Implementation

```rust
/// TALoRA: Timestep-Aware LoRA
pub struct TALoRA {
    /// Adapter banks for each timestep group
    banks: [LoRABank; 3],  // Coarse, Domain, Fine
    /// Timestep boundaries
    boundaries: [u32; 2],  // [700, 300]
    /// Transition smoothing
    transition_width: u32,
}

impl TALoRA {
    /// Get adapters for current timestep with smooth blending
    pub fn get_adapters(&self, timestep: u32, context: &QueryContext) -> TALoRAResult {
        // Determine primary and secondary banks
        let (primary_bank, secondary_bank, blend_factor) = self.get_banks_and_blend(timestep);

        // Retrieve from primary bank
        let primary_adapters = primary_bank.retrieve(context, TOP_K)?;

        // Optionally blend with secondary near boundaries
        if blend_factor > 0.0 {
            let secondary_adapters = secondary_bank.retrieve(context, TOP_K)?;
            return TALoRAResult::Blended {
                primary: primary_adapters,
                secondary: secondary_adapters,
                blend_factor,
            };
        }

        TALoRAResult::Single(primary_adapters)
    }

    /// Compute bank selection with smooth transitions
    fn get_banks_and_blend(&self, timestep: u32) -> (&LoRABank, &LoRABank, f32) {
        let [boundary1, boundary2] = self.boundaries;

        if timestep > boundary1 + self.transition_width {
            // Pure coarse region
            (&self.banks[0], &self.banks[0], 0.0)
        } else if timestep > boundary1 - self.transition_width {
            // Coarse-Domain transition
            let blend = (boundary1 + self.transition_width - timestep) as f32
                / (2 * self.transition_width) as f32;
            (&self.banks[0], &self.banks[1], blend)
        } else if timestep > boundary2 + self.transition_width {
            // Pure domain region
            (&self.banks[1], &self.banks[1], 0.0)
        } else if timestep > boundary2 - self.transition_width {
            // Domain-Fine transition
            let blend = (boundary2 + self.transition_width - timestep) as f32
                / (2 * self.transition_width) as f32;
            (&self.banks[1], &self.banks[2], blend)
        } else {
            // Pure fine region
            (&self.banks[2], &self.banks[2], 0.0)
        }
    }

    /// Apply TALoRA to model output
    pub fn apply(
        &self,
        base_output: &mut Tensor,
        adapters: &TALoRAResult,
        timestep: u32,
    ) -> Result<(), TALoRAError> {
        match adapters {
            TALoRAResult::Single(loras) => {
                for (lora, weight) in loras {
                    base_output.add_scaled(&lora.apply()?, *weight);
                }
            }
            TALoRAResult::Blended { primary, secondary, blend_factor } => {
                // Apply primary
                for (lora, weight) in primary {
                    base_output.add_scaled(&lora.apply()?, weight * (1.0 - blend_factor));
                }
                // Apply secondary
                for (lora, weight) in secondary {
                    base_output.add_scaled(&lora.apply()?, weight * blend_factor);
                }
            }
        }
        Ok(())
    }
}
```

### Training TALoRA Banks

```rust
/// Train timestep-specific adapters
pub struct TALoRATrainer {
    /// Per-bank optimizers
    optimizers: [AdamW; 3],
    /// Loss aggregators
    losses: [RunningMean; 3],
}

impl TALoRATrainer {
    /// Training step with timestep routing
    pub fn train_step(
        &mut self,
        batch: &TrainingBatch,
        model: &mut DiffusionModel,
        talora: &mut TALoRA,
    ) -> TrainingMetrics {
        let mut metrics = TrainingMetrics::default();

        for sample in batch.samples() {
            // Sample random timestep
            let t = self.sample_timestep();

            // Get bank index for this timestep
            let bank_idx = talora.get_bank_index(t);

            // Forward pass with LoRA
            let noise_pred = model.forward_with_lora(
                &sample.noisy,
                t,
                &talora.banks[bank_idx].current_adapter(),
            );

            // Compute loss
            let loss = self.diffusion_loss(&noise_pred, &sample.noise);

            // Backward pass (only update active bank)
            self.optimizers[bank_idx].zero_grad();
            loss.backward();
            self.optimizers[bank_idx].step();

            // Track metrics
            metrics.add_loss(bank_idx, loss.item());
        }

        metrics
    }

    /// Sample timestep with curriculum (start with coarse, progress to fine)
    fn sample_timestep(&self) -> u32 {
        // During early training, bias toward coarse timesteps
        // Gradually include more fine timesteps
        let progress = self.training_progress();

        if progress < 0.3 {
            // Phase 1: Focus on coarse (structure)
            rand::thread_rng().gen_range(600..1000)
        } else if progress < 0.6 {
            // Phase 2: Include domain
            rand::thread_rng().gen_range(200..1000)
        } else {
            // Phase 3: Full range
            rand::thread_rng().gen_range(0..1000)
        }
    }
}
```

## DGR: Denoising-Guided Retrieval

### Motivation

During diffusion denoising, model uncertainty varies across tokens:
- High uncertainty tokens need more guidance
- Low uncertainty tokens are already well-predicted

DGR uses this uncertainty signal to dynamically retrieve relevant adapters for uncertain tokens.

### Comparison to Prior Work

| Approach | Uncertainty-Based | Dynamic Retrieval | Diffusion-Specific |
|----------|------------------|-------------------|-------------------|
| RAMoLE | No | Yes | No |
| LoRARetriever | No | Yes | No |
| Entropy-based AR | Yes | Yes | No (AR only) |
| **DGR (ours)** | **Yes** | **Yes** | **Yes** |

### Implementation

```rust
/// DGR: Denoising-Guided Retrieval
pub struct DGR {
    /// Retrieval index
    index: HNSWIndex,
    /// Uncertainty threshold for retrieval
    uncertainty_threshold: f32,
    /// Retrieval budget per step
    max_retrievals_per_step: usize,
}

impl DGR {
    /// Compute uncertainty from denoising prediction
    pub fn compute_uncertainty(
        &self,
        logits: &Tensor,       // [batch, seq, vocab]
        timestep: u32,
    ) -> Tensor {
        // Entropy of predicted distribution
        let probs = logits.softmax(-1);
        let log_probs = probs.log();
        let entropy = -(probs * log_probs).sum(-1);  // [batch, seq]

        // Normalize by timestep (higher t = naturally higher uncertainty)
        let t_factor = (timestep as f32 / 1000.0).sqrt();
        entropy / t_factor
    }

    /// Identify tokens needing retrieval
    pub fn identify_uncertain_tokens(
        &self,
        uncertainty: &Tensor,
    ) -> Vec<(usize, usize)> {  // (batch_idx, seq_idx)
        let mut uncertain = Vec::new();

        for batch_idx in 0..uncertainty.size(0) {
            for seq_idx in 0..uncertainty.size(1) {
                let u = uncertainty[[batch_idx, seq_idx]];
                if u > self.uncertainty_threshold {
                    uncertain.push((batch_idx, seq_idx));
                }
            }
        }

        // Budget constraint: only retrieve for top-k most uncertain
        uncertain.sort_by(|a, b| {
            let ua = uncertainty[[a.0, a.1]];
            let ub = uncertainty[[b.0, b.1]];
            ub.partial_cmp(&ua).unwrap()
        });

        uncertain.truncate(self.max_retrievals_per_step);
        uncertain
    }

    /// Retrieve adapters for uncertain tokens
    pub fn retrieve_for_uncertain(
        &self,
        hidden_states: &Tensor,
        uncertain_positions: &[(usize, usize)],
        talora: &TALoRA,
        timestep: u32,
    ) -> HashMap<(usize, usize), Vec<WeightedLoRA>> {
        let mut retrievals = HashMap::new();

        for &(batch_idx, seq_idx) in uncertain_positions {
            // Extract hidden state for this position
            let query_vec = hidden_states.slice(batch_idx, seq_idx);

            // Get appropriate bank for timestep
            let bank_idx = talora.get_bank_index(timestep);

            // Retrieve from bank
            let results = self.index.search(
                &query_vec,
                bank_idx,
                TOP_K_RETRIEVAL,
            );

            // Weight by relevance and uncertainty
            let weighted: Vec<WeightedLoRA> = results
                .into_iter()
                .map(|(lora_id, similarity)| {
                    WeightedLoRA {
                        lora_id,
                        weight: similarity * self.compute_weight_factor(timestep),
                    }
                })
                .collect();

            retrievals.insert((batch_idx, seq_idx), weighted);
        }

        retrievals
    }

    /// Apply position-specific adapters
    pub fn apply_position_specific(
        &self,
        base_output: &mut Tensor,
        retrievals: &HashMap<(usize, usize), Vec<WeightedLoRA>>,
        lora_bank: &LoRABank,
    ) {
        for (&(batch_idx, seq_idx), weighted_loras) in retrievals {
            // Compose adapters for this position
            let composed = self.compose_loras(weighted_loras, lora_bank);

            // Apply only to this position
            let position_output = base_output.slice_mut(batch_idx, seq_idx);
            position_output.add_(&composed);
        }
    }

    /// Compose multiple LoRAs with learned/fixed weights
    fn compose_loras(
        &self,
        weighted: &[WeightedLoRA],
        bank: &LoRABank,
    ) -> Tensor {
        let mut result = Tensor::zeros_like(&bank.get(weighted[0].lora_id).output());

        for wl in weighted {
            let lora = bank.get(wl.lora_id);
            let output = lora.apply();
            result.add_scaled(&output, wl.weight);
        }

        result
    }
}

/// Full DGR-enhanced denoising step
pub fn dgr_denoise_step(
    model: &DiffusionModel,
    dgr: &DGR,
    talora: &TALoRA,
    x_t: &Tensor,
    timestep: u32,
) -> Tensor {
    // 1. Base model forward pass
    let (hidden, logits) = model.forward_with_hidden(x_t, timestep);

    // 2. Compute uncertainty
    let uncertainty = dgr.compute_uncertainty(&logits, timestep);

    // 3. Identify uncertain tokens
    let uncertain_positions = dgr.identify_uncertain_tokens(&uncertainty);

    // 4. Get timestep-appropriate adapters (TALoRA)
    let context = QueryContext::from_hidden(&hidden);
    let base_adapters = talora.get_adapters(timestep, &context);

    // 5. Retrieve position-specific adapters for uncertain tokens (DGR)
    let position_adapters = dgr.retrieve_for_uncertain(
        &hidden,
        &uncertain_positions,
        talora,
        timestep,
    );

    // 6. Apply base TALoRA adapters to all positions
    let mut output = logits.clone();
    talora.apply(&mut output, &base_adapters, timestep);

    // 7. Apply position-specific DGR adapters
    dgr.apply_position_specific(&mut output, &position_adapters, talora.get_bank(timestep));

    // 8. Denoise
    model.denoise_from_logits(x_t, &output, timestep)
}
```

### Uncertainty Metrics

```rust
/// Multiple uncertainty metrics for DGR
pub enum UncertaintyMetric {
    /// Entropy of predicted distribution
    Entropy,
    /// Variance across ensemble (if available)
    Variance,
    /// Confidence gap (max prob - second max)
    ConfidenceGap,
    /// Learned uncertainty head
    LearnedHead,
}

impl UncertaintyMetric {
    pub fn compute(&self, logits: &Tensor, model: Option<&UncertaintyHead>) -> Tensor {
        match self {
            Self::Entropy => {
                let probs = logits.softmax(-1);
                -(probs.clone() * probs.log()).sum(-1)
            }
            Self::Variance => {
                // Requires ensemble predictions
                logits.var(-1)
            }
            Self::ConfidenceGap => {
                let sorted = logits.sort(-1, true).0;
                sorted.select(-1, 0) - sorted.select(-1, 1)
            }
            Self::LearnedHead => {
                model.unwrap().forward(logits)
            }
        }
    }
}
```

## DAF: Diffusion-Aware Federation

### Motivation

Standard federated learning (FedAvg) treats all parameters equally. For diffusion models with TALoRA:
- Coarse timestep adapters learn global patterns → benefit from broad aggregation
- Fine timestep adapters learn user-specific style → need selective aggregation

DAF aggregates updates with awareness of their semantic role.

### Comparison to Prior Work

| Approach | Diffusion Model | Timestep Aware | Semantic Grouping |
|----------|-----------------|----------------|-------------------|
| FedAvg | Any | No | No |
| FedProx | Any | No | No |
| FedEx-LoRA | LLM (AR) | No | No |
| **DAF (ours)** | **Diffusion** | **Yes** | **Yes** |

### Implementation

```rust
/// DAF: Diffusion-Aware Federation
pub struct DAF {
    /// Per-timestep-group aggregation strategies
    strategies: [AggregationStrategy; 3],
    /// Semantic importance weights
    semantic_weights: [f32; 3],
}

#[derive(Clone)]
pub enum AggregationStrategy {
    /// Standard weighted average (for coarse/global patterns)
    WeightedAverage,
    /// Quality-weighted (for domain patterns)
    QualityWeighted { metric: QualityMetric },
    /// Conservative (for fine/personal patterns)
    Conservative { min_contributors: usize, similarity_threshold: f32 },
    /// Selective (only aggregate similar updates)
    Selective { clustering: ClusteringMethod },
}

impl DAF {
    /// Aggregate updates with diffusion awareness
    pub fn aggregate(
        &self,
        updates: Vec<ClientUpdate>,
        timestep_group: TimestepGroup,
    ) -> AggregatedUpdate {
        let strategy = &self.strategies[timestep_group as usize];

        match strategy {
            AggregationStrategy::WeightedAverage => {
                self.weighted_average_aggregate(&updates)
            }
            AggregationStrategy::QualityWeighted { metric } => {
                self.quality_weighted_aggregate(&updates, metric)
            }
            AggregationStrategy::Conservative { min_contributors, similarity_threshold } => {
                self.conservative_aggregate(&updates, *min_contributors, *similarity_threshold)
            }
            AggregationStrategy::Selective { clustering } => {
                self.selective_aggregate(&updates, clustering)
            }
        }
    }

    /// Conservative aggregation for fine timesteps
    /// Only aggregates if updates are sufficiently similar (shared patterns)
    fn conservative_aggregate(
        &self,
        updates: &[ClientUpdate],
        min_contributors: usize,
        similarity_threshold: f32,
    ) -> AggregatedUpdate {
        if updates.len() < min_contributors {
            // Not enough contributors - return zero update
            return AggregatedUpdate::zero();
        }

        // Cluster updates by similarity
        let clusters = self.cluster_by_similarity(updates, similarity_threshold);

        // Only aggregate the largest cluster that meets threshold
        let valid_clusters: Vec<_> = clusters
            .into_iter()
            .filter(|c| c.len() >= min_contributors)
            .collect();

        if valid_clusters.is_empty() {
            return AggregatedUpdate::zero();
        }

        // Take largest cluster
        let largest = valid_clusters.into_iter().max_by_key(|c| c.len()).unwrap();

        // Average within cluster
        self.weighted_average_aggregate(&largest)
    }

    /// Selective aggregation based on clustering
    fn selective_aggregate(
        &self,
        updates: &[ClientUpdate],
        clustering: &ClusteringMethod,
    ) -> AggregatedUpdate {
        // Cluster updates
        let clusters = match clustering {
            ClusteringMethod::KMeans { k } => self.kmeans_cluster(updates, *k),
            ClusteringMethod::DBSCAN { eps, min_samples } => {
                self.dbscan_cluster(updates, *eps, *min_samples)
            }
            ClusteringMethod::Hierarchical { distance_threshold } => {
                self.hierarchical_cluster(updates, *distance_threshold)
            }
        };

        // Aggregate each cluster separately
        let cluster_updates: Vec<_> = clusters
            .into_iter()
            .map(|c| self.weighted_average_aggregate(&c))
            .collect();

        // Return multiple updates for different patterns
        // Client can select based on their context
        AggregatedUpdate::multi(cluster_updates)
    }

    /// Quality-weighted aggregation based on validation metrics
    fn quality_weighted_aggregate(
        &self,
        updates: &[ClientUpdate],
        metric: &QualityMetric,
    ) -> AggregatedUpdate {
        // Compute quality scores
        let scores: Vec<f32> = updates
            .iter()
            .map(|u| metric.compute(&u.validation_metrics))
            .collect();

        // Softmax normalization
        let weights = softmax(&scores);

        // Weighted average
        let mut result = AggregatedUpdate::zero_like(&updates[0]);
        for (update, weight) in updates.iter().zip(weights.iter()) {
            result.add_scaled(update, *weight);
        }

        result
    }
}

/// Quality metrics for weighting
pub enum QualityMetric {
    /// Perplexity improvement on validation set
    PerplexityImprovement,
    /// Task-specific accuracy
    TaskAccuracy { task: String },
    /// Composite score
    Composite { metrics: Vec<(QualityMetric, f32)> },
}

impl QualityMetric {
    pub fn compute(&self, validation: &ValidationMetrics) -> f32 {
        match self {
            Self::PerplexityImprovement => {
                // Higher is better: inverse perplexity reduction
                let ppl_before = validation.perplexity_before;
                let ppl_after = validation.perplexity_after;
                (ppl_before - ppl_after) / ppl_before
            }
            Self::TaskAccuracy { task } => {
                validation.task_accuracy.get(task).copied().unwrap_or(0.0)
            }
            Self::Composite { metrics } => {
                metrics.iter()
                    .map(|(m, w)| m.compute(validation) * w)
                    .sum()
            }
        }
    }
}
```

### DAF Round Protocol

```rust
/// DAF-enhanced federation round
pub struct DAFRound {
    round_id: u64,
    daf: DAF,
    privacy: DifferentialPrivacy,
}

impl DAFRound {
    /// Execute DAF federation round
    pub async fn execute(&mut self, clients: &[ClientProxy]) -> RoundResult {
        // 1. Collect updates from all clients
        let updates = self.collect_updates(clients).await?;

        // 2. Separate by timestep group
        let grouped = self.group_by_timestep(&updates);

        // 3. Aggregate each group with appropriate strategy
        let mut aggregated = HashMap::new();
        for (group, group_updates) in grouped {
            let agg = self.daf.aggregate(group_updates, group);

            // Apply differential privacy
            let private_agg = self.privacy.privatize(&agg)?;

            aggregated.insert(group, private_agg);
        }

        // 4. Distribute back to clients
        self.distribute_updates(clients, &aggregated).await?;

        Ok(RoundResult {
            round_id: self.round_id,
            participants: updates.len(),
            aggregations_per_group: aggregated.iter()
                .map(|(g, a)| (*g, a.num_contributors()))
                .collect(),
        })
    }

    /// Group updates by timestep semantic group
    fn group_by_timestep(&self, updates: &[ClientUpdate]) -> HashMap<TimestepGroup, Vec<ClientUpdate>> {
        let mut grouped = HashMap::new();

        for update in updates {
            // Each update may contain multiple TALoRA bank updates
            for (group, bank_update) in &update.talora_updates {
                grouped.entry(*group).or_insert_with(Vec::new).push(ClientUpdate {
                    client_id: update.client_id,
                    talora_updates: vec![(*group, bank_update.clone())].into_iter().collect(),
                    validation_metrics: update.validation_metrics.clone(),
                });
            }
        }

        grouped
    }
}
```

## Research Validation Plan

### Experimental Setup

```rust
/// Benchmarking suite for novel contributions
pub struct NoveltyBenchmark {
    // Baselines
    baseline_lora: LoRA,
    baseline_federation: FedAvg,

    // Our methods
    talora: TALoRA,
    dgr: DGR,
    daf: DAF,

    // Datasets
    datasets: Vec<BenchmarkDataset>,
}

impl NoveltyBenchmark {
    /// Run ablation studies
    pub async fn run_ablations(&mut self) -> AblationResults {
        let mut results = AblationResults::default();

        // 1. TALoRA vs Standard LoRA
        results.talora = self.ablate_talora().await?;

        // 2. DGR vs No Retrieval vs Static Retrieval
        results.dgr = self.ablate_dgr().await?;

        // 3. DAF vs FedAvg vs FedProx
        results.daf = self.ablate_daf().await?;

        // 4. Full system vs each component removed
        results.full_system = self.ablate_full_system().await?;

        results
    }

    /// TALoRA ablation
    async fn ablate_talora(&mut self) -> TALoRAResults {
        let configs = vec![
            ("standard_lora", LoRAConfig::standard()),
            ("talora_2_banks", TALoRAConfig::two_banks()),
            ("talora_3_banks", TALoRAConfig::three_banks()),
            ("talora_5_banks", TALoRAConfig::five_banks()),
        ];

        let mut results = Vec::new();
        for (name, config) in configs {
            let metrics = self.evaluate_lora_config(name, &config).await?;
            results.push((name, metrics));
        }

        TALoRAResults { configs: results }
    }
}
```

### Expected Results

| Method | Perplexity | Latency | Memory | Federation Benefit |
|--------|------------|---------|--------|-------------------|
| Standard LoRA | Baseline | Baseline | Baseline | +5% |
| TALoRA | -8-12% | +5% | +20% | +8% |
| DGR | -5-8% | +10% | +10% | +3% |
| DAF | 0% | +2% | 0% | +15-20% |
| Full System | -12-18% | +15% | +30% | +25% |

## Publication Roadmap

### Venue Targets

1. **Primary**: NeurIPS / ICML / ICLR (ML venues)
2. **Secondary**: EMNLP / ACL (NLP venues)
3. **System**: MLSys / OSDI (Systems venues)

### Paper Structure

```
Title: TALoRA: Timestep-Aware Adaptation for Diffusion Language Models
       with Denoising-Guided Retrieval and Federated Learning

Abstract: [Novel contributions summary]

1. Introduction
   - Diffusion LLMs potential
   - Limitation: static adaptation
   - Our solution: TALoRA + DGR + DAF

2. Background
   - Diffusion LLMs (MDLM, BD3LM)
   - LoRA adaptation
   - Federated learning

3. Method
   3.1 TALoRA: Timestep-Aware LoRA
   3.2 DGR: Denoising-Guided Retrieval
   3.3 DAF: Diffusion-Aware Federation

4. Experiments
   4.1 Setup
   4.2 Main Results
   4.3 Ablation Studies
   4.4 Analysis

5. Related Work
   - Diffusion LLMs
   - LoRA and Adapters
   - Retrieval-Augmented Generation
   - Federated Learning for LLMs

6. Conclusion
```

---

**Previous**: [06-SIMD-GPU.md](./06-SIMD-GPU.md) - Hardware acceleration
**Next**: [08-IMPLEMENTATION.md](./08-IMPLEMENTATION.md) - Step-by-step implementation plan
