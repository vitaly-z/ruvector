# Diffusion-Aware Federation (DAF)

## Overview

RuvDLLM introduces **Diffusion-Aware Federation (DAF)**, a novel federated learning protocol designed specifically for diffusion language models. Unlike standard FedAvg, DAF understands the semantic meaning of different timesteps in the denoising process and aligns aggregation accordingly.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DAF Federation Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │ Node A  │    │ Node B  │    │ Node C  │    │ Node N  │                  │
│  │ Private │    │ Group   │    │ Tenant  │    │ Public  │                  │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘                  │
│       │              │              │              │                        │
│       │              │              │              │                        │
│       ▼              ▼              ▼              ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Gossip Protocol Layer                            │   │
│  │  • Encrypted delta exchange                                         │   │
│  │  • Privacy tier filtering                                           │   │
│  │  • Reputation tracking                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DAF Aggregation Engine                           │   │
│  │  • Timestep-aligned weight averaging                                │   │
│  │  • Noise schedule semantic grouping                                 │   │
│  │  • Quality-weighted contribution                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Global Model Updates                             │   │
│  │  • Differential privacy (ε=1.0)                                     │   │
│  │  • Byzantine fault tolerance                                        │   │
│  │  • Gradient clipping                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## DAF: The Novel Contribution

### Why Standard Federation Fails for Diffusion

Standard FedAvg treats all model parameters equally. For diffusion models, this is suboptimal:

1. **Early timesteps** (t=1000→700): High noise, coarse structure - learns global patterns
2. **Middle timesteps** (t=700→300): Medium noise, detail formation - learns domain specifics
3. **Late timesteps** (t=300→0): Low noise, fine refinement - learns user preferences

Averaging weights across nodes without timestep awareness loses this semantic structure.

### DAF Algorithm

```rust
/// Diffusion-Aware Federation aggregation
pub struct DAFAggregator {
    /// Timestep groupings for semantic alignment
    timestep_groups: Vec<TimestepGroup>,
    /// Per-group aggregation weights
    group_weights: HashMap<TimestepGroupId, Vec<f32>>,
    /// Noise schedule for semantic mapping
    noise_schedule: NoiseSchedule,
}

#[derive(Clone)]
pub struct TimestepGroup {
    pub id: TimestepGroupId,
    pub range: Range<u32>,
    pub semantic_role: SemanticRole,
    pub aggregation_strategy: AggregationStrategy,
}

#[derive(Clone, Copy)]
pub enum SemanticRole {
    /// t=1000→700: Global structure, syntax
    CoarseStructure,
    /// t=700→300: Domain patterns, semantics
    DomainPatterns,
    /// t=300→0: Fine details, style
    FineRefinement,
}

#[derive(Clone)]
pub enum AggregationStrategy {
    /// Standard weighted average
    WeightedAverage { weights: Vec<f32> },
    /// Quality-weighted (higher score = more influence)
    QualityWeighted { metric: QualityMetric },
    /// Robust aggregation (Byzantine-tolerant)
    TrimmedMean { trim_fraction: f32 },
    /// Attention-based (learns optimal combination)
    LearnedAttention { temperature: f32 },
}

impl DAFAggregator {
    pub fn new(config: DAFConfig) -> Self {
        let timestep_groups = vec![
            TimestepGroup {
                id: TimestepGroupId(0),
                range: 700..1000,
                semantic_role: SemanticRole::CoarseStructure,
                aggregation_strategy: AggregationStrategy::WeightedAverage {
                    weights: vec![], // Computed dynamically
                },
            },
            TimestepGroup {
                id: TimestepGroupId(1),
                range: 300..700,
                semantic_role: SemanticRole::DomainPatterns,
                aggregation_strategy: AggregationStrategy::QualityWeighted {
                    metric: QualityMetric::PerplexityImprovement,
                },
            },
            TimestepGroup {
                id: TimestepGroupId(2),
                range: 0..300,
                semantic_role: SemanticRole::FineRefinement,
                aggregation_strategy: AggregationStrategy::TrimmedMean {
                    trim_fraction: 0.1, // Remove top/bottom 10%
                },
            },
        ];

        Self {
            timestep_groups,
            group_weights: HashMap::new(),
            noise_schedule: config.noise_schedule,
        }
    }

    /// Aggregate TALoRA updates from multiple nodes
    pub fn aggregate_talora_updates(
        &mut self,
        updates: Vec<NodeUpdate>,
        privacy_filter: &PrivacyFilter,
    ) -> Result<AggregatedUpdate, FederationError> {
        // Filter updates by privacy tier
        let filtered_updates: Vec<_> = updates
            .into_iter()
            .filter(|u| privacy_filter.allows_aggregation(&u.privacy_tier))
            .collect();

        if filtered_updates.len() < MIN_NODES_FOR_AGGREGATION {
            return Err(FederationError::InsufficientNodes);
        }

        let mut aggregated = AggregatedUpdate::new();

        // Aggregate each timestep group separately
        for group in &self.timestep_groups {
            let group_updates: Vec<_> = filtered_updates
                .iter()
                .filter_map(|u| u.get_group_update(group.id))
                .collect();

            let aggregated_group = match &group.aggregation_strategy {
                AggregationStrategy::WeightedAverage { .. } => {
                    self.weighted_average_aggregate(&group_updates)
                }
                AggregationStrategy::QualityWeighted { metric } => {
                    self.quality_weighted_aggregate(&group_updates, metric)
                }
                AggregationStrategy::TrimmedMean { trim_fraction } => {
                    self.trimmed_mean_aggregate(&group_updates, *trim_fraction)
                }
                AggregationStrategy::LearnedAttention { temperature } => {
                    self.attention_aggregate(&group_updates, *temperature)
                }
            };

            aggregated.set_group(group.id, aggregated_group);
        }

        // Apply differential privacy
        aggregated.apply_differential_privacy(EPSILON, DELTA)?;

        Ok(aggregated)
    }

    /// Quality-weighted aggregation based on perplexity improvement
    fn quality_weighted_aggregate(
        &self,
        updates: &[GroupUpdate],
        metric: &QualityMetric,
    ) -> GroupUpdate {
        // Compute quality scores
        let scores: Vec<f32> = updates
            .iter()
            .map(|u| metric.compute_score(u))
            .collect();

        // Softmax normalization
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        // Weighted average of deltas
        let mut result = GroupUpdate::zeros_like(&updates[0]);
        for (update, weight) in updates.iter().zip(weights.iter()) {
            result.add_scaled(update, *weight);
        }

        result
    }

    /// Byzantine-tolerant trimmed mean aggregation
    fn trimmed_mean_aggregate(
        &self,
        updates: &[GroupUpdate],
        trim_fraction: f32,
    ) -> GroupUpdate {
        let n = updates.len();
        let trim_count = (n as f32 * trim_fraction) as usize;

        // For each parameter, sort values and trim extremes
        let param_count = updates[0].params.len();
        let mut result = vec![0.0f32; param_count];

        for i in 0..param_count {
            let mut values: Vec<f32> = updates.iter().map(|u| u.params[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Trim top and bottom
            let trimmed = &values[trim_count..n - trim_count];
            result[i] = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
        }

        GroupUpdate { params: result }
    }
}
```

## Gossip Protocol

### Secure Delta Exchange

```rust
/// Gossip-based federation protocol
pub struct GossipProtocol {
    /// Local node ID
    node_id: NodeId,
    /// Known peers
    peers: HashMap<NodeId, PeerInfo>,
    /// Pending updates to share
    pending_updates: Vec<EncryptedDelta>,
    /// Received updates awaiting aggregation
    received_updates: HashMap<RoundId, Vec<NodeUpdate>>,
    /// Cryptographic keys
    crypto: FederationCrypto,
}

pub struct PeerInfo {
    pub node_id: NodeId,
    pub address: SocketAddr,
    pub public_key: PublicKey,
    pub reputation: f32,
    pub privacy_tier: PrivacyTier,
    pub last_seen: Instant,
}

impl GossipProtocol {
    /// Start gossip round
    pub async fn start_round(&mut self, round_id: RoundId) -> Result<(), GossipError> {
        // Select random subset of peers (fan-out)
        let fan_out = (self.peers.len() as f32).sqrt().ceil() as usize;
        let selected_peers = self.select_peers(fan_out);

        // Prepare encrypted deltas
        for peer in selected_peers {
            let delta = self.prepare_delta_for_peer(&peer)?;
            self.send_delta(peer.node_id, delta).await?;
        }

        Ok(())
    }

    /// Prepare delta encrypted for specific peer's privacy tier
    fn prepare_delta_for_peer(&self, peer: &PeerInfo) -> Result<EncryptedDelta, GossipError> {
        // Filter local updates by peer's privacy tier
        let allowed_updates = self.pending_updates
            .iter()
            .filter(|u| peer.privacy_tier.can_receive(&u.source_tier))
            .cloned()
            .collect::<Vec<_>>();

        if allowed_updates.is_empty() {
            return Err(GossipError::NoUpdatesToShare);
        }

        // Serialize and encrypt
        let serialized = bincode::serialize(&allowed_updates)?;
        let encrypted = self.crypto.encrypt_for_peer(&serialized, &peer.public_key)?;

        Ok(EncryptedDelta {
            source: self.node_id,
            round_id: self.current_round,
            ciphertext: encrypted,
            signature: self.crypto.sign(&encrypted)?,
        })
    }

    /// Handle received delta
    pub async fn handle_delta(&mut self, delta: EncryptedDelta) -> Result<(), GossipError> {
        // Verify signature
        let peer = self.peers.get(&delta.source)
            .ok_or(GossipError::UnknownPeer)?;

        if !self.crypto.verify(&delta.ciphertext, &delta.signature, &peer.public_key)? {
            // Update reputation negatively
            self.update_reputation(delta.source, -0.1);
            return Err(GossipError::InvalidSignature);
        }

        // Decrypt
        let decrypted = self.crypto.decrypt(&delta.ciphertext)?;
        let updates: Vec<EncryptedDelta> = bincode::deserialize(&decrypted)?;

        // Store for aggregation
        let round_updates = self.received_updates.entry(delta.round_id).or_default();
        for update in updates {
            round_updates.push(update.into_node_update()?);
        }

        // Update reputation positively
        self.update_reputation(delta.source, 0.01);

        // Propagate to other peers (with decay)
        self.propagate_delta(delta).await?;

        Ok(())
    }

    /// Select peers based on reputation and diversity
    fn select_peers(&self, count: usize) -> Vec<&PeerInfo> {
        let mut candidates: Vec<_> = self.peers.values()
            .filter(|p| p.last_seen.elapsed() < PEER_TIMEOUT)
            .collect();

        // Sort by reputation with some randomness
        let mut rng = thread_rng();
        candidates.sort_by(|a, b| {
            let score_a = a.reputation + rng.gen::<f32>() * 0.2;
            let score_b = b.reputation + rng.gen::<f32>() * 0.2;
            score_b.partial_cmp(&score_a).unwrap()
        });

        candidates.into_iter().take(count).collect()
    }
}
```

### Federation Round Protocol

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Federation Round Timeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  T=0        T=10s       T=30s       T=60s       T=90s       T=120s         │
│   │           │           │           │           │           │             │
│   │           │           │           │           │           │             │
│   ▼           ▼           ▼           ▼           ▼           ▼             │
│ ┌────┐     ┌────┐     ┌────┐     ┌────┐     ┌────┐     ┌────┐            │
│ │Init│────►│Gossip────►│Collect───►│Aggregate──►│Apply───►│Done│           │
│ └────┘     └────┘     └────┘     └────┘     └────┘     └────┘            │
│                                                                              │
│  Phases:                                                                    │
│  1. Init (T=0): Prepare local updates, announce participation              │
│  2. Gossip (T=10s): Exchange encrypted deltas with √N peers                │
│  3. Collect (T=30s): Wait for propagation, handle late arrivals            │
│  4. Aggregate (T=60s): Run DAF aggregation on collected updates            │
│  5. Apply (T=90s): Merge aggregated updates into local model               │
│  6. Done (T=120s): Commit, start next round                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Security Measures

### Byzantine Fault Tolerance

```rust
/// Byzantine-tolerant update validation
pub struct ByzantineDetector {
    /// Historical update statistics per node
    node_stats: HashMap<NodeId, NodeStatistics>,
    /// Anomaly detection threshold
    anomaly_threshold: f32,
}

impl ByzantineDetector {
    /// Validate update before aggregation
    pub fn validate_update(&mut self, update: &NodeUpdate) -> ValidationResult {
        let stats = self.node_stats.entry(update.source).or_default();

        // Check 1: Gradient magnitude bounds
        let magnitude = update.compute_magnitude();
        if magnitude > stats.mean_magnitude * 10.0 {
            return ValidationResult::Rejected(RejectReason::ExcessiveMagnitude);
        }

        // Check 2: Direction consistency (cosine similarity with historical)
        if let Some(prev_direction) = &stats.last_direction {
            let similarity = update.cosine_similarity(prev_direction);
            if similarity < -0.5 {
                stats.flip_count += 1;
                if stats.flip_count > MAX_FLIPS {
                    return ValidationResult::Rejected(RejectReason::DirectionAnomaly);
                }
            }
        }

        // Check 3: Timestamp-specific bounds (DAF-aware)
        for (group_id, group_update) in &update.group_updates {
            let expected_range = self.get_expected_range(*group_id);
            if !expected_range.contains(&group_update.compute_norm()) {
                return ValidationResult::Suspicious(SuspicionLevel::Medium);
            }
        }

        // Update statistics
        stats.update(update);

        ValidationResult::Accepted
    }

    /// Detect potential Sybil attacks
    pub fn detect_sybil(&self, updates: &[NodeUpdate]) -> Vec<NodeId> {
        let mut suspicious = Vec::new();

        // Cluster updates by similarity
        let clusters = self.cluster_updates(updates);

        for cluster in clusters {
            // If cluster is too similar and too large, likely Sybil
            if cluster.similarity > 0.99 && cluster.nodes.len() > 3 {
                suspicious.extend(cluster.nodes.iter().skip(1).cloned());
            }
        }

        suspicious
    }
}
```

### Gradient Clipping and Noise

```rust
/// Differential privacy for gradient protection
pub struct DifferentialPrivacy {
    /// Privacy budget epsilon
    epsilon: f32,
    /// Delta parameter
    delta: f32,
    /// Clipping bound
    clip_bound: f32,
}

impl DifferentialPrivacy {
    /// Apply differential privacy to gradients
    pub fn apply(&self, gradients: &mut [f32]) -> Result<(), PrivacyError> {
        // Clip gradients to bound
        let norm = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
        if norm > self.clip_bound {
            let scale = self.clip_bound / norm;
            for g in gradients.iter_mut() {
                *g *= scale;
            }
        }

        // Add Gaussian noise calibrated to privacy budget
        let sigma = self.compute_noise_scale();
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, sigma).unwrap();

        for g in gradients.iter_mut() {
            *g += normal.sample(&mut rng);
        }

        Ok(())
    }

    /// Compute noise scale for given privacy parameters
    fn compute_noise_scale(&self) -> f32 {
        // Gaussian mechanism: σ = Δf * √(2 * ln(1.25/δ)) / ε
        let sensitivity = self.clip_bound;
        let factor = (2.0 * (1.25 / self.delta).ln()).sqrt();
        sensitivity * factor / self.epsilon
    }
}
```

## Multi-Tier Federation

```rust
/// Multi-tier federation manager
pub struct TieredFederation {
    /// Local node tier
    local_tier: PrivacyTier,
    /// Tier-specific aggregators
    aggregators: HashMap<PrivacyTier, DAFAggregator>,
    /// Cross-tier promotion rules
    promotion_rules: Vec<PromotionRule>,
}

impl TieredFederation {
    /// Federate updates respecting privacy tiers
    pub async fn federate(
        &mut self,
        local_update: LocalUpdate,
    ) -> Result<FederationResult, FederationError> {
        // Determine which tiers this update can participate in
        let allowed_tiers = self.get_allowed_tiers(&local_update);

        let mut results = FederationResult::default();

        for tier in allowed_tiers {
            match tier {
                PrivacyTier::Private => {
                    // Private: Only local consolidation
                    results.private = Some(local_update.clone());
                }
                PrivacyTier::Group => {
                    // Group: Federate within group key holders
                    let group_update = self.federate_within_group(&local_update).await?;
                    results.group = Some(group_update);
                }
                PrivacyTier::Tenant => {
                    // Tenant: Federate within organization
                    let tenant_update = self.federate_within_tenant(&local_update).await?;
                    results.tenant = Some(tenant_update);
                }
                PrivacyTier::Public => {
                    // Public: Global federation with full privacy protection
                    let public_update = self.federate_public(&local_update).await?;
                    results.public = Some(public_update);
                }
            }
        }

        Ok(results)
    }

    /// Check if update should be promoted to higher tier
    pub fn check_promotion(&self, update: &LocalUpdate) -> Option<PrivacyTier> {
        for rule in &self.promotion_rules {
            if rule.matches(update) {
                return Some(rule.target_tier);
            }
        }
        None
    }
}

/// Rules for automatic promotion between tiers
pub struct PromotionRule {
    pub name: String,
    pub condition: PromotionCondition,
    pub target_tier: PrivacyTier,
    pub requires_consent: bool,
}

pub enum PromotionCondition {
    /// Pattern seen by N+ distinct users
    MinimumUsers(usize),
    /// Pattern older than duration
    AgeThreshold(Duration),
    /// Pattern matches public category
    CategoryMatch(Vec<String>),
    /// Manual approval
    Explicit,
}
```

## Network Topology

### Adaptive Peer Selection

```rust
/// Adaptive peer selection for optimal federation
pub struct AdaptivePeerSelector {
    /// Peer performance history
    peer_history: HashMap<NodeId, PeerHistory>,
    /// Network latency estimates
    latency_matrix: HashMap<(NodeId, NodeId), Duration>,
    /// Current topology
    topology: FederationTopology,
}

impl AdaptivePeerSelector {
    /// Select optimal peer set for current round
    pub fn select_peers(&mut self, count: usize) -> Vec<NodeId> {
        let mut candidates: Vec<_> = self.peer_history.iter()
            .map(|(id, history)| {
                let score = self.compute_peer_score(id, history);
                (*id, score)
            })
            .collect();

        // Multi-objective selection:
        // 1. High reputation (reliable updates)
        // 2. Low latency (fast rounds)
        // 3. Diverse privacy tiers (broader coverage)
        // 4. Geographic diversity (partition tolerance)

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Ensure diversity
        let mut selected = Vec::new();
        let mut tier_counts: HashMap<PrivacyTier, usize> = HashMap::new();

        for (id, _score) in candidates {
            let tier = self.peer_history[&id].privacy_tier;
            let tier_count = tier_counts.entry(tier).or_insert(0);

            // Allow max 40% from any single tier
            if *tier_count < (count * 2 / 5) {
                selected.push(id);
                *tier_count += 1;
            }

            if selected.len() >= count {
                break;
            }
        }

        selected
    }

    fn compute_peer_score(&self, id: &NodeId, history: &PeerHistory) -> f32 {
        let reputation_weight = 0.4;
        let latency_weight = 0.3;
        let freshness_weight = 0.2;
        let diversity_weight = 0.1;

        let latency_score = 1.0 / (1.0 + history.avg_latency.as_secs_f32());
        let freshness_score = 1.0 / (1.0 + history.last_seen.elapsed().as_secs_f32() / 3600.0);
        let diversity_score = self.compute_diversity_score(id);

        reputation_weight * history.reputation
            + latency_weight * latency_score
            + freshness_weight * freshness_score
            + diversity_weight * diversity_score
    }
}
```

## Performance Optimizations

### Compressed Delta Exchange

```rust
/// Delta compression for efficient transmission
pub struct DeltaCompressor {
    /// Sparsity threshold (below this, send as sparse)
    sparsity_threshold: f32,
    /// Quantization bits
    quantization_bits: u8,
}

impl DeltaCompressor {
    /// Compress delta for transmission
    pub fn compress(&self, delta: &[f32]) -> CompressedDelta {
        // Count non-zero elements
        let non_zero_count = delta.iter().filter(|&&v| v.abs() > 1e-8).count();
        let sparsity = 1.0 - (non_zero_count as f32 / delta.len() as f32);

        if sparsity > self.sparsity_threshold {
            // Sparse encoding
            self.compress_sparse(delta)
        } else {
            // Dense quantized encoding
            self.compress_dense(delta)
        }
    }

    /// Sparse compression (indices + values)
    fn compress_sparse(&self, delta: &[f32]) -> CompressedDelta {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &v) in delta.iter().enumerate() {
            if v.abs() > 1e-8 {
                indices.push(i as u32);
                values.push(self.quantize(v));
            }
        }

        CompressedDelta::Sparse {
            total_len: delta.len(),
            indices,
            values,
        }
    }

    /// Dense quantization to reduce bandwidth
    fn compress_dense(&self, delta: &[f32]) -> CompressedDelta {
        let min_val = delta.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = delta.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        let levels = (1 << self.quantization_bits) as f32;
        let quantized: Vec<u8> = delta
            .iter()
            .map(|&v| ((v - min_val) / range * (levels - 1.0)) as u8)
            .collect();

        CompressedDelta::Dense {
            min_val,
            max_val,
            quantization_bits: self.quantization_bits,
            data: quantized,
        }
    }
}
```

### Async Federation Pipeline

```rust
/// Asynchronous federation pipeline
pub struct AsyncFederationPipeline {
    /// Compression stage
    compressor: DeltaCompressor,
    /// Encryption stage
    crypto: FederationCrypto,
    /// Network stage
    network: FederationNetwork,
    /// Aggregation stage
    aggregator: DAFAggregator,
}

impl AsyncFederationPipeline {
    /// Run federation round asynchronously
    pub async fn run_round(&mut self, round_id: RoundId) -> Result<RoundResult, FederationError> {
        // Stage 1: Compress local updates (CPU-bound)
        let compress_task = tokio::task::spawn_blocking({
            let updates = self.local_updates.clone();
            let compressor = self.compressor.clone();
            move || compressor.compress_batch(&updates)
        });

        // Stage 2: Encrypt (CPU-bound, parallelizable)
        let compressed = compress_task.await??;
        let encrypt_tasks: Vec<_> = compressed
            .into_iter()
            .map(|c| {
                let crypto = self.crypto.clone();
                tokio::task::spawn_blocking(move || crypto.encrypt(&c))
            })
            .collect();

        let encrypted: Vec<_> = futures::future::try_join_all(encrypt_tasks).await?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        // Stage 3: Network exchange (IO-bound)
        let received = self.network.exchange_round(round_id, encrypted).await?;

        // Stage 4: Decrypt (CPU-bound, parallelizable)
        let decrypt_tasks: Vec<_> = received
            .into_iter()
            .map(|r| {
                let crypto = self.crypto.clone();
                tokio::task::spawn_blocking(move || crypto.decrypt(&r))
            })
            .collect();

        let decrypted: Vec<_> = futures::future::try_join_all(decrypt_tasks).await?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        // Stage 5: Aggregate (CPU-bound)
        let aggregated = tokio::task::spawn_blocking({
            let aggregator = self.aggregator.clone();
            let privacy_filter = self.privacy_filter.clone();
            move || aggregator.aggregate_talora_updates(decrypted, &privacy_filter)
        }).await??;

        Ok(RoundResult {
            round_id,
            aggregated,
            participants: decrypted.len(),
            latency: round_start.elapsed(),
        })
    }
}
```

## Configuration

```toml
# federation.toml
[federation]
enabled = true
round_interval_secs = 120
min_nodes_for_aggregation = 3
max_nodes_per_round = 50

[federation.gossip]
fan_out = 4
max_hops = 3
propagation_timeout_secs = 30

[federation.daf]
# Timestep groupings
coarse_range = [700, 1000]
domain_range = [300, 700]
fine_range = [0, 300]
# Aggregation strategies per group
coarse_strategy = "weighted_average"
domain_strategy = "quality_weighted"
fine_strategy = "trimmed_mean"
trim_fraction = 0.1

[federation.security]
byzantine_detection = true
max_magnitude_multiplier = 10.0
direction_flip_threshold = 5

[federation.privacy]
epsilon = 1.0
delta = 1e-5
clip_bound = 1.0
min_k_anonymity = 5

[federation.compression]
sparsity_threshold = 0.9
quantization_bits = 8
```

---

**Previous**: [03-MICRO-LORA.md](./03-MICRO-LORA.md) - Real-time adaptation
**Next**: [05-PRIVACY.md](./05-PRIVACY.md) - Privacy tiers and encryption
