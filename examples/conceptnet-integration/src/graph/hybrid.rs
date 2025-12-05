//! Hybrid Vector-Graph Query Engine
//!
//! Combines vector similarity search with graph traversal for semantic reasoning.

use super::builder::{ConceptNetGraphBuilder, GraphNode};
use crate::api::RelationType;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

/// Configuration for hybrid queries
#[derive(Debug, Clone)]
pub struct HybridQueryConfig {
    /// Weight for vector similarity (0-1)
    pub vector_weight: f32,
    /// Weight for graph distance (0-1)
    pub graph_weight: f32,
    /// Maximum vector candidates
    pub vector_k: usize,
    /// Maximum graph hops
    pub max_hops: usize,
    /// Minimum similarity threshold
    pub min_similarity: f32,
    /// Enable reranking
    pub rerank: bool,
}

impl Default for HybridQueryConfig {
    fn default() -> Self {
        Self {
            vector_weight: 0.6,
            graph_weight: 0.4,
            vector_k: 50,
            max_hops: 2,
            min_similarity: 0.3,
            rerank: true,
        }
    }
}

/// Result of a hybrid query
#[derive(Debug, Clone)]
pub struct HybridResult {
    pub uri: String,
    pub label: String,
    pub vector_score: f32,
    pub graph_score: f32,
    pub combined_score: f32,
    pub path_relations: Vec<RelationType>,
}

impl PartialEq for HybridResult {
    fn eq(&self, other: &Self) -> bool {
        self.uri == other.uri
    }
}

impl Eq for HybridResult {}

impl PartialOrd for HybridResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.combined_score.partial_cmp(&other.combined_score)
    }
}

impl Ord for HybridResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Hybrid query engine combining vector and graph search
pub struct HybridQueryEngine<'a> {
    graph: &'a ConceptNetGraphBuilder,
    config: HybridQueryConfig,
}

impl<'a> HybridQueryEngine<'a> {
    /// Create a new hybrid query engine
    pub fn new(graph: &'a ConceptNetGraphBuilder, config: HybridQueryConfig) -> Self {
        Self { graph, config }
    }

    /// Execute a hybrid query using vector similarity + graph structure
    pub fn query(&self, query_embedding: &[f32], k: usize) -> Vec<HybridResult> {
        // Phase 1: Vector similarity search
        let vector_candidates = self.vector_search(query_embedding);

        // Phase 2: Graph expansion
        let expanded = self.graph_expand(&vector_candidates);

        // Phase 3: Score combination
        let mut scored = self.combine_scores(query_embedding, expanded, &vector_candidates);

        // Phase 4: Reranking (optional)
        if self.config.rerank {
            self.rerank(&mut scored, query_embedding);
        }

        // Return top-k results
        scored.truncate(k);
        scored
    }

    /// Semantic search with graph-aware reranking
    pub fn semantic_search(
        &self,
        query_embedding: &[f32],
        anchor_concepts: &[String],
        k: usize,
    ) -> Vec<HybridResult> {
        // Get initial vector candidates
        let mut candidates = self.vector_search(query_embedding);

        // Boost candidates connected to anchor concepts
        for candidate in &mut candidates {
            let boost = self.compute_anchor_boost(&candidate.uri, anchor_concepts);
            candidate.combined_score *= 1.0 + boost;
        }

        candidates.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        candidates.truncate(k);
        candidates
    }

    /// Query by example: find concepts similar to the example in vector and graph space
    pub fn query_by_example(&self, example_uri: &str, k: usize) -> Vec<HybridResult> {
        let node = match self.graph.get_node(example_uri) {
            Some(n) => n,
            None => return vec![],
        };

        let embedding = match &node.embedding {
            Some(e) => e,
            None => return self.graph_only_search(example_uri, k),
        };

        self.query(embedding, k)
    }

    /// Find concepts related through multiple relation types
    pub fn multi_hop_query(
        &self,
        start_uri: &str,
        relation_path: &[RelationType],
        k: usize,
    ) -> Vec<HybridResult> {
        let mut current_set: HashSet<String> = HashSet::new();
        current_set.insert(start_uri.to_string());

        for relation in relation_path {
            let mut next_set = HashSet::new();
            for uri in &current_set {
                for edge in self.graph.get_outgoing_edges(uri) {
                    if edge.relation == *relation {
                        next_set.insert(edge.target_id.clone());
                    }
                }
            }
            current_set = next_set;
            if current_set.is_empty() {
                break;
            }
        }

        let mut results: Vec<HybridResult> = current_set
            .into_iter()
            .filter_map(|uri| {
                self.graph.get_node(&uri).map(|node| HybridResult {
                    uri: uri.clone(),
                    label: node.label.clone(),
                    vector_score: 1.0,
                    graph_score: 1.0,
                    combined_score: 1.0,
                    path_relations: relation_path.to_vec(),
                })
            })
            .collect();

        results.truncate(k);
        results
    }

    /// Analogical reasoning: A is to B as C is to ?
    pub fn analogy(
        &self,
        a_uri: &str,
        b_uri: &str,
        c_uri: &str,
        k: usize,
    ) -> Vec<HybridResult> {
        // Get embeddings
        let a_emb = self.get_embedding(a_uri);
        let b_emb = self.get_embedding(b_uri);
        let c_emb = self.get_embedding(c_uri);

        if a_emb.is_none() || b_emb.is_none() || c_emb.is_none() {
            return vec![];
        }

        let a_emb = a_emb.unwrap();
        let b_emb = b_emb.unwrap();
        let c_emb = c_emb.unwrap();

        // Compute analogy vector: D = B - A + C
        let target_embedding: Vec<f32> = a_emb
            .iter()
            .zip(b_emb.iter())
            .zip(c_emb.iter())
            .map(|((&a, &b), &c)| b - a + c)
            .collect();

        // Normalize
        let norm: f32 = target_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 0.0 {
            target_embedding.iter().map(|x| x / norm).collect()
        } else {
            target_embedding
        };

        // Search for the target
        let mut results = self.query(&normalized, k + 3);

        // Filter out input concepts
        results.retain(|r| r.uri != a_uri && r.uri != b_uri && r.uri != c_uri);
        results.truncate(k);
        results
    }

    // Private methods

    fn vector_search(&self, query_embedding: &[f32]) -> Vec<HybridResult> {
        let mut results = Vec::new();

        for node in self.graph.nodes() {
            if let Some(ref embedding) = node.embedding {
                let similarity = self.cosine_similarity(query_embedding, embedding);
                if similarity >= self.config.min_similarity {
                    results.push(HybridResult {
                        uri: node.uri.clone(),
                        label: node.label.clone(),
                        vector_score: similarity,
                        graph_score: 0.0,
                        combined_score: similarity,
                        path_relations: vec![],
                    });
                }
            }
        }

        results.sort_by(|a, b| b.vector_score.partial_cmp(&a.vector_score).unwrap());
        results.truncate(self.config.vector_k);
        results
    }

    fn graph_expand(&self, candidates: &[HybridResult]) -> Vec<HybridResult> {
        let mut expanded: HashMap<String, HybridResult> = HashMap::new();
        let candidate_uris: HashSet<_> = candidates.iter().map(|c| &c.uri).collect();

        for candidate in candidates {
            expanded.insert(candidate.uri.clone(), candidate.clone());

            // Expand to neighbors
            for hop in 1..=self.config.max_hops {
                let decay = 0.5f32.powi(hop as i32);
                for edge in self.graph.get_node_edges(&candidate.uri) {
                    let neighbor_uri = if edge.source_id == candidate.uri {
                        &edge.target_id
                    } else {
                        &edge.source_id
                    };

                    // Don't overwrite direct candidates
                    if candidate_uris.contains(neighbor_uri) {
                        continue;
                    }

                    let graph_score = candidate.vector_score * decay * (edge.weight as f32 / 10.0);

                    expanded
                        .entry(neighbor_uri.clone())
                        .and_modify(|e| {
                            e.graph_score = e.graph_score.max(graph_score);
                            if !e.path_relations.contains(&edge.relation) {
                                e.path_relations.push(edge.relation);
                            }
                        })
                        .or_insert_with(|| {
                            let node = self.graph.get_node(neighbor_uri);
                            HybridResult {
                                uri: neighbor_uri.clone(),
                                label: node.map(|n| n.label.clone()).unwrap_or_default(),
                                vector_score: 0.0,
                                graph_score,
                                combined_score: 0.0,
                                path_relations: vec![edge.relation],
                            }
                        });
                }
            }
        }

        expanded.into_values().collect()
    }

    fn combine_scores(
        &self,
        query_embedding: &[f32],
        mut candidates: Vec<HybridResult>,
        original_candidates: &[HybridResult],
    ) -> Vec<HybridResult> {
        let original_map: HashMap<_, _> = original_candidates
            .iter()
            .map(|c| (&c.uri, c.vector_score))
            .collect();

        for candidate in &mut candidates {
            // Update vector score if we have embedding
            if candidate.vector_score == 0.0 {
                if let Some(embedding) = self.get_embedding(&candidate.uri) {
                    candidate.vector_score = self.cosine_similarity(query_embedding, &embedding);
                }
            }

            // Use original score if available
            if let Some(&orig_score) = original_map.get(&candidate.uri) {
                candidate.vector_score = orig_score;
            }

            // Combine scores
            candidate.combined_score = self.config.vector_weight * candidate.vector_score
                + self.config.graph_weight * candidate.graph_score;
        }

        candidates.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        candidates
    }

    fn rerank(&self, results: &mut Vec<HybridResult>, query_embedding: &[f32]) {
        // Diversify results using MMR-like reranking
        let lambda = 0.5;
        let mut reranked = Vec::with_capacity(results.len());
        let mut remaining: Vec<_> = results.drain(..).collect();

        while !remaining.is_empty() && reranked.len() < results.capacity() {
            let mut best_idx = 0;
            let mut best_score = f32::MIN;

            for (i, candidate) in remaining.iter().enumerate() {
                let relevance = candidate.combined_score;

                // Compute max similarity to already selected items
                let max_sim = reranked
                    .iter()
                    .filter_map(|r: &HybridResult| {
                        let emb1 = self.get_embedding(&candidate.uri)?;
                        let emb2 = self.get_embedding(&r.uri)?;
                        Some(self.cosine_similarity(&emb1, &emb2))
                    })
                    .fold(0.0f32, |a, b| a.max(b));

                let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim;
                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = i;
                }
            }

            reranked.push(remaining.remove(best_idx));
        }

        *results = reranked;
    }

    fn compute_anchor_boost(&self, uri: &str, anchors: &[String]) -> f32 {
        let mut total_boost = 0.0;

        for anchor in anchors {
            // Check direct connection
            for edge in self.graph.get_node_edges(uri) {
                if edge.source_id == *anchor || edge.target_id == *anchor {
                    total_boost += 0.5 * edge.weight as f32 / 10.0;
                }
            }

            // Check shared neighbors
            let uri_neighbors: HashSet<_> = self
                .graph
                .get_neighbors(uri)
                .iter()
                .map(|n| &n.uri)
                .collect();
            let anchor_neighbors: HashSet<_> = self
                .graph
                .get_neighbors(anchor)
                .iter()
                .map(|n| &n.uri)
                .collect();

            let shared = uri_neighbors.intersection(&anchor_neighbors).count();
            total_boost += shared as f32 * 0.1;
        }

        total_boost.min(1.0)
    }

    fn graph_only_search(&self, uri: &str, k: usize) -> Vec<HybridResult> {
        let mut results = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(uri.to_string());

        for edge in self.graph.get_node_edges(uri) {
            let neighbor_uri = if edge.source_id == uri {
                &edge.target_id
            } else {
                &edge.source_id
            };

            if visited.insert(neighbor_uri.clone()) {
                if let Some(node) = self.graph.get_node(neighbor_uri) {
                    results.push(HybridResult {
                        uri: neighbor_uri.clone(),
                        label: node.label.clone(),
                        vector_score: 0.0,
                        graph_score: edge.weight as f32 / 10.0,
                        combined_score: edge.weight as f32 / 10.0,
                        path_relations: vec![edge.relation],
                    });
                }
            }
        }

        results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        results.truncate(k);
        results
    }

    fn get_embedding(&self, uri: &str) -> Option<Vec<f32>> {
        self.graph.get_node(uri).and_then(|n| n.embedding.clone())
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let graph = ConceptNetGraphBuilder::default_config();
        let engine = HybridQueryEngine::new(
            &graph,
            HybridQueryConfig::default(),
        );

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((engine.cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((engine.cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }
}
