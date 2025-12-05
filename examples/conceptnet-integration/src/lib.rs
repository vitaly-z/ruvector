//! # ConceptNet Integration with RuVector
//!
//! Comprehensive integration of ConceptNet knowledge graph with RuVector's
//! vector database, graph database, GNN, attention mechanisms, and SONA
//! self-learning capabilities.
//!
//! ## Features
//!
//! - **API Client**: High-performance async client for ConceptNet 5.7 API
//! - **Graph Integration**: Map ConceptNet edges to RuVector graph database
//! - **GNN Reasoning**: Neural reasoning over knowledge graph
//! - **Attention Mechanisms**: Relation-aware, hierarchical, and causal attention
//! - **SONA Learning**: Self-optimizing learning from commonsense patterns
//! - **Numberbatch**: Semantic embeddings integration
//! - **RuvLLM**: Commonsense augmentation for language models
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use conceptnet_integration::{
//!     api::ConceptNetClient,
//!     graph::ConceptNetGraphBuilder,
//!     gnn::{CommonsenseGNN, GNNConfig},
//!     ruvllm::CommonsenseAugmenter,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create API client
//!     let client = ConceptNetClient::new();
//!
//!     // Lookup a concept
//!     let response = client.lookup("/c/en/artificial_intelligence").await?;
//!     println!("Found {} edges", response.edges.len());
//!
//!     // Build graph
//!     let mut builder = ConceptNetGraphBuilder::default_config();
//!     builder.add_edges(&response.edges)?;
//!
//!     // Create GNN for reasoning
//!     let gnn = CommonsenseGNN::new(GNNConfig::default());
//!
//!     // Augment LLM with commonsense
//!     let augmenter = CommonsenseAugmenter::new(&builder, Default::default());
//!     let context = augmenter.augment("What is machine learning?");
//!
//!     println!("Augmented context:\n{}", context.formatted_context);
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    ConceptNet API                           │
//! │              (api.conceptnet.io)                            │
//! └─────────────────────────────────────────────────────────────┘
//!                           ↓
//! ┌─────────────────────────────────────────────────────────────┐
//! │                 ConceptNet Client                           │
//! │     (rate-limited, cached, async)                           │
//! └─────────────────────────────────────────────────────────────┘
//!                           ↓
//! ┌─────────────────────────────────────────────────────────────┐
//! │              ConceptNet Graph Builder                       │
//! │    (nodes, edges, hyperedges, Cypher queries)               │
//! └─────────────────────────────────────────────────────────────┘
//!                           ↓
//! ┌───────────────────┬─────────────────────┬───────────────────┐
//! │   Numberbatch     │    GNN Layers       │    Attention      │
//! │   (embeddings)    │  (reasoning)        │   (weighting)     │
//! └───────────────────┴─────────────────────┴───────────────────┘
//!                           ↓
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    SONA Learning                            │
//! │     (patterns, trajectories, ReasoningBank)                 │
//! └─────────────────────────────────────────────────────────────┘
//!                           ↓
//! ┌─────────────────────────────────────────────────────────────┐
//! │                 RuvLLM Integration                          │
//! │   (context augmentation, hallucination detection, QA)       │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Examples
//!
//! See the `examples/` directory for complete working examples:
//!
//! - `basic_usage.rs` - Simple API queries and graph building
//! - `gnn_reasoning.rs` - Neural reasoning over knowledge graph
//! - `llm_augmentation.rs` - Augmenting LLM with commonsense
//! - `semantic_search.rs` - Hybrid vector-graph search
//! - `analogical_reasoning.rs` - Analogy completion (A:B::C:?)
//!

#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]

pub mod api;
pub mod graph;
pub mod gnn;
pub mod attention;
pub mod numberbatch;
pub mod sona;
pub mod ruvllm;
pub mod loader;

/// Re-exports for convenience
pub mod prelude {
    pub use crate::api::{ConceptNetClient, Edge, ConceptNode, RelationType, QueryParams};
    pub use crate::graph::{ConceptNetGraphBuilder, GraphBuildConfig, CommonsenseQuery};
    pub use crate::gnn::{CommonsenseGNN, GNNConfig, CommonsenseReasoner, ReasoningQuery};
    pub use crate::attention::{RelationAttention, CommonsenseAttentionConfig};
    pub use crate::numberbatch::Numberbatch;
    pub use crate::sona::{CommonsenseSona, CommonsenseSonaConfig};
    pub use crate::ruvllm::{CommonsenseAugmenter, RuvLLMConfig, AugmentedContext};
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library metadata
pub fn version() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
}
