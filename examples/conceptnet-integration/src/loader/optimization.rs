//! Optimization Strategies for RuVector and RuvLLM
//!
//! Uses the full ConceptNet graph to optimize:
//! - RuVector: Better similarity search via commonsense relationships
//! - RuvLLM: Grounded context augmentation and hallucination detection
//!
//! ## Key Optimizations
//!
//! ### RuVector Optimizations
//! 1. **Semantic Graph Augmentation**: Use ConceptNet edges to enhance similarity
//! 2. **Relation-Weighted Distance**: Weight distances by relation type
//! 3. **Multi-hop Expansion**: Find related concepts through graph traversal
//! 4. **Numberbatch Integration**: Pre-trained 300-dim semantic embeddings
//!
//! ### RuvLLM Optimizations
//! 1. **Fact-Grounded Generation**: Inject verified commonsense facts
//! 2. **Hallucination Detection**: Verify claims against knowledge graph
//! 3. **Reasoning Chains**: Generate explainable multi-hop reasoning
//! 4. **Domain-Specific Context**: Load domain subgraphs for focused queries

use crate::api::RelationType;
use crate::graph::ConceptNetGraphBuilder;
use crate::gnn::{CommonsenseGNN, GNNConfig};
use crate::numberbatch::Numberbatch;
use crate::ruvllm::{CommonsenseAugmenter, RuvLLMConfig, AugmentedContext};
use crate::sona::{CommonsenseSona, CommonsenseSonaConfig, PatternType};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Configuration for optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Use GNN for enhanced reasoning
    pub use_gnn: bool,
    /// GNN configuration
    pub gnn_config: GNNConfig,
    /// Enable SONA self-learning
    pub enable_sona: bool,
    /// SONA configuration
    pub sona_config: CommonsenseSonaConfig,
    /// Minimum confidence for fact inclusion
    pub min_confidence: f32,
    /// Maximum reasoning depth
    pub max_reasoning_depth: usize,
    /// Enable relation-weighted distances
    pub relation_weighted: bool,
    /// Relation weights for distance calculation
    pub relation_weights: HashMap<RelationType, f32>,
    /// Cache reasoning results
    pub cache_results: bool,
    /// Cache size
    pub cache_size: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        let mut relation_weights = HashMap::new();
        // Higher weight = stronger semantic relationship
        relation_weights.insert(RelationType::Synonym, 1.0);
        relation_weights.insert(RelationType::IsA, 0.9);
        relation_weights.insert(RelationType::InstanceOf, 0.85);
        relation_weights.insert(RelationType::PartOf, 0.8);
        relation_weights.insert(RelationType::HasA, 0.75);
        relation_weights.insert(RelationType::UsedFor, 0.7);
        relation_weights.insert(RelationType::CapableOf, 0.7);
        relation_weights.insert(RelationType::Causes, 0.65);
        relation_weights.insert(RelationType::HasProperty, 0.6);
        relation_weights.insert(RelationType::AtLocation, 0.55);
        relation_weights.insert(RelationType::RelatedTo, 0.5);
        relation_weights.insert(RelationType::Antonym, 0.1); // Opposite meaning

        Self {
            use_gnn: true,
            gnn_config: GNNConfig::default(),
            enable_sona: true,
            sona_config: CommonsenseSonaConfig::default(),
            min_confidence: 0.5,
            max_reasoning_depth: 3,
            relation_weighted: true,
            relation_weights,
            cache_results: true,
            cache_size: 10_000,
        }
    }
}

/// Metrics from optimization
#[derive(Debug, Clone, Default)]
pub struct OptimizationMetrics {
    /// Number of concepts processed
    pub concepts_processed: usize,
    /// Number of relationships used
    pub relationships_used: usize,
    /// Average similarity improvement
    pub avg_similarity_improvement: f32,
    /// Hallucinations detected
    pub hallucinations_detected: usize,
    /// Reasoning chains generated
    pub reasoning_chains_generated: usize,
    /// Patterns learned (SONA)
    pub patterns_learned: usize,
    /// Cache hit rate
    pub cache_hit_rate: f32,
}

/// Callback for training progress
pub trait TrainingCallback: Send + Sync {
    fn on_progress(&self, metrics: &OptimizationMetrics);
    fn on_batch_complete(&self, batch_idx: usize, total_batches: usize);
    fn on_error(&self, error: &str);
}

/// Default no-op callback
pub struct NoOpCallback;
impl TrainingCallback for NoOpCallback {
    fn on_progress(&self, _: &OptimizationMetrics) {}
    fn on_batch_complete(&self, _: usize, _: usize) {}
    fn on_error(&self, _: &str) {}
}

/// RuVector optimizer using ConceptNet
pub struct RuVectorOptimizer {
    graph: Arc<ConceptNetGraphBuilder>,
    embeddings: Arc<Numberbatch>,
    gnn: Option<CommonsenseGNN>,
    config: OptimizationConfig,
    metrics: OptimizationMetrics,
}

impl RuVectorOptimizer {
    /// Create optimizer from graph and embeddings
    pub fn new(
        graph: Arc<ConceptNetGraphBuilder>,
        embeddings: Arc<Numberbatch>,
        config: OptimizationConfig,
    ) -> Self {
        let gnn = if config.use_gnn {
            Some(CommonsenseGNN::new(config.gnn_config.clone()))
        } else {
            None
        };

        Self {
            graph,
            embeddings,
            gnn,
            config,
            metrics: OptimizationMetrics::default(),
        }
    }

    /// Get commonsense-enhanced similarity between two concepts
    pub fn semantic_similarity(&self, concept1: &str, concept2: &str) -> f32 {
        // Base embedding similarity
        let base_sim = self.embeddings
            .similarity(concept1, concept2)
            .unwrap_or(0.0);

        // Graph-based similarity (shared neighbors, paths)
        let graph_sim = self.graph_similarity(concept1, concept2);

        // GNN-enhanced similarity
        let gnn_sim = if let (Some(gnn), Some(emb1), Some(emb2)) = (
            &self.gnn,
            self.embeddings.get(concept1),
            self.embeddings.get(concept2),
        ) {
            gnn.compute_similarity(&emb1, &emb2)
        } else {
            base_sim
        };

        // Weighted combination
        0.4 * base_sim + 0.3 * graph_sim + 0.3 * gnn_sim
    }

    /// Graph-based similarity using ConceptNet structure
    fn graph_similarity(&self, concept1: &str, concept2: &str) -> f32 {
        let uri1 = self.normalize_uri(concept1);
        let uri2 = self.normalize_uri(concept2);

        // Check direct connection
        if let Some(neighbors) = self.get_neighbors_with_relations(&uri1) {
            for (neighbor, relation, weight) in &neighbors {
                if neighbor == &uri2 {
                    let rel_weight = self.config.relation_weights
                        .get(relation)
                        .copied()
                        .unwrap_or(0.5);
                    return (*weight as f32 * rel_weight).min(1.0);
                }
            }
        }

        // Check common neighbors (Jaccard-like)
        let neighbors1: HashSet<String> = self.get_neighbor_set(&uri1);
        let neighbors2: HashSet<String> = self.get_neighbor_set(&uri2);

        if neighbors1.is_empty() || neighbors2.is_empty() {
            return 0.0;
        }

        let intersection = neighbors1.intersection(&neighbors2).count();
        let union = neighbors1.union(&neighbors2).count();

        if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        }
    }

    /// Find semantically similar concepts using graph + embeddings
    pub fn find_similar(&self, concept: &str, k: usize) -> Vec<(String, f32)> {
        let uri = self.normalize_uri(concept);
        let mut candidates = HashMap::new();

        // Get embedding-based candidates
        let similar = self.embeddings.most_similar(&uri, k * 2);
        for (c, score) in similar {
            candidates.insert(c, score);
        }

        // Add graph neighbors
        if let Some(neighbors) = self.get_neighbors_with_relations(&uri) {
            for (neighbor, relation, weight) in neighbors {
                let rel_weight = self.config.relation_weights
                    .get(&relation)
                    .copied()
                    .unwrap_or(0.5);
                let score = weight as f32 * rel_weight;
                candidates
                    .entry(neighbor)
                    .and_modify(|s| *s = (*s + score) / 2.0)
                    .or_insert(score);
            }
        }

        // Sort by score and take top k
        let mut results: Vec<(String, f32)> = candidates.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }

    /// Expand query with related concepts
    pub fn expand_query(&self, query: &[f32], depth: usize) -> Vec<Vec<f32>> {
        let mut expanded = vec![query.to_vec()];

        // Find similar concepts in embedding space
        let similar = self.embeddings.most_similar_to_vector(query, 5);
        for (concept, _) in similar {
            if let Some(emb) = self.embeddings.get(&concept) {
                expanded.push(emb.clone());
            }

            // Add related concepts from graph
            if depth > 0 {
                let uri = self.normalize_uri(&concept);
                for neighbor in self.get_neighbor_set(&uri).into_iter().take(3) {
                    if let Some(emb) = self.embeddings.get(&neighbor) {
                        expanded.push(emb.clone());
                    }
                }
            }
        }

        expanded
    }

    /// Get relation-weighted distance
    pub fn relation_weighted_distance(&self, concept1: &str, concept2: &str, base_distance: f32) -> f32 {
        if !self.config.relation_weighted {
            return base_distance;
        }

        let uri1 = self.normalize_uri(concept1);
        let uri2 = self.normalize_uri(concept2);

        // Check if directly connected
        if let Some(neighbors) = self.get_neighbors_with_relations(&uri1) {
            for (neighbor, relation, _) in &neighbors {
                if neighbor == &uri2 {
                    let rel_weight = self.config.relation_weights
                        .get(relation)
                        .copied()
                        .unwrap_or(0.5);
                    // Reduce distance for strongly related concepts
                    return base_distance * (1.0 - rel_weight * 0.5);
                }
            }
        }

        base_distance
    }

    // Helper methods

    fn normalize_uri(&self, concept: &str) -> String {
        if concept.starts_with("/c/") {
            concept.to_string()
        } else {
            format!("/c/en/{}", concept.to_lowercase().replace(' ', "_"))
        }
    }

    fn get_neighbors_with_relations(&self, uri: &str) -> Option<Vec<(String, RelationType, f64)>> {
        let node = self.graph.get_node(uri)?;
        let neighbors = self.graph.get_neighbors(uri);

        Some(
            neighbors
                .into_iter()
                .map(|n| (n.uri.clone(), RelationType::RelatedTo, 1.0))
                .collect()
        )
    }

    fn get_neighbor_set(&self, uri: &str) -> HashSet<String> {
        self.graph
            .get_neighbors(uri)
            .into_iter()
            .map(|n| n.uri.clone())
            .collect()
    }

    /// Get metrics
    pub fn metrics(&self) -> &OptimizationMetrics {
        &self.metrics
    }
}

/// RuvLLM optimizer using ConceptNet
pub struct RuvLLMOptimizer<'a> {
    augmenter: CommonsenseAugmenter<'a>,
    sona: Option<CommonsenseSona>,
    config: OptimizationConfig,
    metrics: OptimizationMetrics,
}

impl<'a> RuvLLMOptimizer<'a> {
    /// Create optimizer
    pub fn new(
        graph: &'a ConceptNetGraphBuilder,
        config: OptimizationConfig,
    ) -> Self {
        let ruvllm_config = RuvLLMConfig {
            max_facts: 15,
            min_relevance: config.min_confidence,
            include_reasoning: true,
            max_reasoning_depth: config.max_reasoning_depth,
            detect_hallucinations: true,
            grounding_threshold: 0.6,
        };

        let augmenter = CommonsenseAugmenter::new(graph, ruvllm_config);

        let sona = if config.enable_sona {
            Some(CommonsenseSona::new(config.sona_config.clone()))
        } else {
            None
        };

        Self {
            augmenter,
            sona,
            config,
            metrics: OptimizationMetrics::default(),
        }
    }

    /// Generate optimized context for a query
    pub fn augment(&mut self, query: &str) -> AugmentedContext {
        let context = self.augmenter.augment(query);
        self.metrics.concepts_processed += context.concepts.len();
        self.metrics.relationships_used += context.facts.len();

        // Learn from successful augmentations
        if let Some(sona) = &mut self.sona {
            if !context.facts.is_empty() {
                // Record pattern
                let mut builder = sona.begin_trajectory(query);
                for fact in &context.facts {
                    builder.add_step(
                        &fact.subject,
                        vec![0.0; 300], // Would use actual embeddings
                        fact.relation,
                        fact.confidence,
                    );
                }
                let trajectory = builder.build(0.8);
                sona.end_trajectory(trajectory);
                self.metrics.patterns_learned = sona.export_patterns().len();
            }
        }

        context
    }

    /// Verify a statement against the knowledge graph
    pub fn verify_statement(&mut self, statement: &str) -> VerificationResult {
        let result = self.augmenter.check_grounding(statement);

        if !result.is_grounded {
            self.metrics.hallucinations_detected += 1;
        }

        VerificationResult {
            is_grounded: result.is_grounded,
            confidence: result.confidence,
            supporting_facts: result.supporting_facts.iter().map(|f| f.to_natural_language()).collect(),
            contradicting_facts: result.contradicting_facts.iter().map(|f| f.to_natural_language()).collect(),
        }
    }

    /// Generate a grounded response
    pub fn generate_grounded_prompt(&self, query: &str, system_prompt: Option<&str>) -> String {
        self.augmenter.generate_prompt(query, system_prompt)
    }

    /// Answer a question with reasoning
    pub fn answer_with_reasoning(&self, question: &str) -> AnswerWithReasoning {
        let answer = self.augmenter.answer_question(question);

        AnswerWithReasoning {
            answer: answer.answer,
            confidence: answer.confidence,
            facts_used: answer.supporting_facts.iter().map(|f| f.to_natural_language()).collect(),
            reasoning_chain: String::new(), // QuestionAnswer doesn't have a reasoning_chain field
        }
    }

    /// Get learned patterns from SONA
    pub fn get_learned_patterns(&self) -> Vec<LearnedPattern> {
        self.sona
            .as_ref()
            .map(|s| {
                s.export_patterns()
                    .into_iter()
                    .map(|p| LearnedPattern {
                        pattern_type: format!("{:?}", p.pattern_type),
                        relations: p.relations.iter().map(|r| format!("{:?}", r)).collect(),
                        success_rate: p.success_rate,
                        usage_count: p.usage_count,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get metrics
    pub fn metrics(&self) -> &OptimizationMetrics {
        &self.metrics
    }
}

/// Result of statement verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub is_grounded: bool,
    pub confidence: f32,
    pub supporting_facts: Vec<String>,
    pub contradicting_facts: Vec<String>,
}

/// Answer with full reasoning chain
#[derive(Debug, Clone)]
pub struct AnswerWithReasoning {
    pub answer: String,
    pub confidence: f32,
    pub facts_used: Vec<String>,
    pub reasoning_chain: String,
}

/// Learned pattern from SONA
#[derive(Debug, Clone)]
pub struct LearnedPattern {
    pub pattern_type: String,
    pub relations: Vec<String>,
    pub success_rate: f32,
    pub usage_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_config_defaults() {
        let config = OptimizationConfig::default();
        assert!(config.use_gnn);
        assert!(config.enable_sona);
        assert!(config.relation_weights.contains_key(&RelationType::IsA));
    }
}
