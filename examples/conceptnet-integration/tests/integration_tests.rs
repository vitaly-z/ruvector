//! Integration tests for ConceptNet-RuVector integration

use conceptnet_integration::api::{ConceptNode, Edge, Relation, RelationType};
use conceptnet_integration::graph::{ConceptNetGraphBuilder, GraphBuildConfig};
use conceptnet_integration::gnn::{CommonsenseGNN, GNNConfig, CommonsenseReasoner, ReasoningQuery};
use conceptnet_integration::attention::{RelationAttention, CommonsenseAttentionConfig};
use conceptnet_integration::numberbatch::{Numberbatch, MockNumberbatch};
use conceptnet_integration::sona::{CommonsenseSona, CommonsenseSonaConfig};
use conceptnet_integration::ruvllm::{CommonsenseAugmenter, RuvLLMConfig};

fn mock_edge(start: &str, end: &str, rel: &str, weight: f64) -> Edge {
    Edge {
        id: format!("/a/[{},{},{}]", rel, start, end),
        start: ConceptNode {
            id: format!("/c/en/{}", start),
            label: Some(start.to_string()),
            language: Some("en".to_string()),
            term: Some(start.to_string()),
            sense_label: None,
        },
        end: ConceptNode {
            id: format!("/c/en/{}", end),
            label: Some(end.to_string()),
            language: Some("en".to_string()),
            term: Some(end.to_string()),
            sense_label: None,
        },
        rel: Relation {
            id: format!("/r/{}", rel),
            label: Some(rel.to_string()),
        },
        weight,
        surface_text: None,
        license: None,
        dataset: None,
        sources: vec![],
    }
}

// ============================================================================
// Graph Builder Tests
// ============================================================================

#[test]
fn test_graph_builder_add_edges() {
    let mut builder = ConceptNetGraphBuilder::default_config();

    let edges = vec![
        mock_edge("dog", "animal", "IsA", 2.0),
        mock_edge("cat", "animal", "IsA", 2.0),
        mock_edge("dog", "mammal", "IsA", 1.5),
    ];

    let added = builder.add_edges(&edges).unwrap();
    assert_eq!(added, 3);

    let stats = builder.stats();
    assert!(stats.total_nodes >= 4); // dog, cat, animal, mammal
    assert_eq!(stats.total_edges, 3);
}

#[test]
fn test_graph_shortest_path() {
    let mut builder = ConceptNetGraphBuilder::default_config();

    let edges = vec![
        mock_edge("dog", "mammal", "IsA", 2.0),
        mock_edge("mammal", "animal", "IsA", 2.0),
        mock_edge("animal", "living_thing", "IsA", 2.0),
    ];

    builder.add_edges(&edges).unwrap();

    let path = builder.shortest_path("/c/en/dog", "/c/en/living_thing", 5);
    assert!(path.is_some());
    let path = path.unwrap();
    assert_eq!(path.len(), 4); // dog -> mammal -> animal -> living_thing
}

#[test]
fn test_graph_neighbors() {
    let mut builder = ConceptNetGraphBuilder::default_config();

    let edges = vec![
        mock_edge("dog", "animal", "IsA", 2.0),
        mock_edge("dog", "pet", "IsA", 2.0),
        mock_edge("dog", "loyal", "HasProperty", 1.5),
    ];

    builder.add_edges(&edges).unwrap();

    let neighbors = builder.get_neighbors("/c/en/dog");
    assert_eq!(neighbors.len(), 3);
}

#[test]
fn test_graph_stats() {
    let mut builder = ConceptNetGraphBuilder::default_config();

    let edges = vec![
        mock_edge("a", "b", "IsA", 2.0),
        mock_edge("b", "c", "IsA", 2.0),
        mock_edge("c", "d", "IsA", 2.0),
    ];

    builder.add_edges(&edges).unwrap();

    let stats = builder.stats();
    assert_eq!(stats.total_nodes, 4);
    assert_eq!(stats.total_edges, 3);
    assert!(stats.avg_degree > 0.0);
}

// ============================================================================
// GNN Tests
// ============================================================================

#[test]
fn test_gnn_forward() {
    let config = GNNConfig {
        input_dim: 64,
        hidden_dim: 32,
        output_dim: 16,
        num_heads: 2,
        num_layers: 2,
        ..Default::default()
    };

    let gnn = CommonsenseGNN::new(config);

    let embeddings = vec![vec![0.1; 64], vec![0.2; 64], vec![0.3; 64]];

    let adjacency = vec![
        (0, 1, RelationType::IsA, 1.0),
        (1, 2, RelationType::HasA, 0.8),
    ];

    let outputs = gnn.forward(&embeddings, &adjacency);
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].len(), 16);
}

#[test]
fn test_gnn_link_prediction() {
    let config = GNNConfig::default();
    let gnn = CommonsenseGNN::new(config);

    let src = vec![0.5; 300];
    let dst = vec![0.5; 300];

    let score = gnn.predict_link(&src, &dst, &RelationType::IsA);
    assert!(score >= 0.0 && score <= 1.0);
}

#[test]
fn test_gnn_similarity() {
    let config = GNNConfig::default();
    let gnn = CommonsenseGNN::new(config);

    let a = vec![1.0; 300];
    let b = vec![1.0; 300];

    let sim = gnn.compute_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 0.001);
}

// ============================================================================
// Attention Tests
// ============================================================================

#[test]
fn test_relation_attention() {
    let config = CommonsenseAttentionConfig {
        hidden_dim: 64,
        num_heads: 4,
        ..Default::default()
    };

    let attention = RelationAttention::new(config);

    let query = vec![0.1; 64];
    let keys = vec![vec![0.2; 64], vec![0.3; 64]];
    let values = vec![vec![0.4; 64], vec![0.5; 64]];
    let relations = vec![RelationType::IsA, RelationType::HasA];

    let output = attention.forward(&query, &keys, &values, &relations);
    assert_eq!(output.len(), 64);
}

#[test]
fn test_attention_weights() {
    let config = CommonsenseAttentionConfig {
        hidden_dim: 64,
        num_heads: 4,
        ..Default::default()
    };

    let attention = RelationAttention::new(config);

    let query = vec![0.1; 64];
    let keys = vec![vec![0.2; 64], vec![0.3; 64], vec![0.4; 64]];
    let relations = vec![RelationType::IsA, RelationType::HasA, RelationType::UsedFor];

    let weights = attention.get_attention_weights(&query, &keys, &relations);
    assert_eq!(weights.len(), 3);

    // Weights should sum to 1 (softmax)
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

// ============================================================================
// Numberbatch Tests
// ============================================================================

#[test]
fn test_numberbatch_operations() {
    let mut nb = Numberbatch::new(300);

    nb.set("/c/en/dog", vec![0.1; 300]).unwrap();
    nb.set("/c/en/cat", vec![0.2; 300]).unwrap();

    assert!(nb.contains("/c/en/dog"));
    assert!(nb.contains("dog")); // Should find with prefix
    assert!(!nb.contains("/c/en/xyz"));

    let sim = nb.similarity("/c/en/dog", "/c/en/cat");
    assert!(sim.is_some());
}

#[test]
fn test_mock_numberbatch() {
    let mock = MockNumberbatch::new(300);

    let emb1 = mock.get("/c/en/dog");
    let emb2 = mock.get("/c/en/dog");
    let emb3 = mock.get("/c/en/cat");

    // Same concept should produce same embedding
    assert_eq!(emb1, emb2);
    // Different concepts should produce different embeddings
    assert_ne!(emb1, emb3);
}

#[test]
fn test_numberbatch_centroid() {
    let mut nb = Numberbatch::new(3);

    nb.set("a", vec![1.0, 0.0, 0.0]).unwrap();
    nb.set("b", vec![0.0, 1.0, 0.0]).unwrap();
    nb.set("c", vec![0.0, 0.0, 1.0]).unwrap();

    let centroid = nb.centroid(&["a", "b", "c"]);
    assert!(centroid.is_some());
}

// ============================================================================
// SONA Tests
// ============================================================================

#[test]
fn test_sona_trajectory() {
    let config = CommonsenseSonaConfig::default();
    let mut sona = CommonsenseSona::new(config);

    let mut builder = sona.begin_trajectory("test query");
    builder.add_step("/c/en/dog", vec![0.1; 300], RelationType::IsA, 0.9);
    builder.add_step("/c/en/animal", vec![0.2; 300], RelationType::IsA, 0.8);

    let trajectory = builder.build(0.85);
    sona.end_trajectory(trajectory);

    // Patterns should be extracted
    let patterns = sona.export_patterns();
    assert!(!patterns.is_empty());
}

#[test]
fn test_sona_transform() {
    let config = CommonsenseSonaConfig::default();
    let sona = CommonsenseSona::new(config);

    let input = vec![0.5; 300];
    let output = sona.transform(&input);

    assert_eq!(output.len(), 300);
}

// ============================================================================
// RuvLLM Tests
// ============================================================================

#[test]
fn test_ruvllm_augmentation() {
    let builder = ConceptNetGraphBuilder::default_config();
    let config = RuvLLMConfig::default();
    let augmenter = CommonsenseAugmenter::new(&builder, config);

    let context = augmenter.augment("What is a dog?");
    assert_eq!(context.query, "What is a dog?");
}

#[test]
fn test_ruvllm_prompt_generation() {
    let builder = ConceptNetGraphBuilder::default_config();
    let config = RuvLLMConfig::default();
    let augmenter = CommonsenseAugmenter::new(&builder, config);

    let prompt = augmenter.generate_prompt("Hello", None);
    assert!(prompt.contains("Hello"));
}

#[test]
fn test_fact_natural_language() {
    use conceptnet_integration::ruvllm::CommonsenseFact;

    let fact = CommonsenseFact {
        subject: "/c/en/dog".to_string(),
        relation: RelationType::IsA,
        object: "/c/en/animal".to_string(),
        confidence: 0.9,
        natural_language: String::new(),
    };

    let nl = fact.to_natural_language();
    assert!(nl.contains("dog"));
    assert!(nl.contains("animal"));
}

// ============================================================================
// API Types Tests
// ============================================================================

#[test]
fn test_relation_type_from_str() {
    assert_eq!(RelationType::from("/r/IsA"), RelationType::IsA);
    assert_eq!(RelationType::from("PartOf"), RelationType::PartOf);
    assert_eq!(RelationType::from("/r/HasPrerequisite"), RelationType::HasPrerequisite);
}

#[test]
fn test_relation_type_weights() {
    assert!(RelationType::IsA.reasoning_weight() > RelationType::RelatedTo.reasoning_weight());
    assert!(RelationType::Synonym.reasoning_weight() > 0.9);
}

#[test]
fn test_concept_node_term_extraction() {
    let node = ConceptNode {
        id: "/c/en/artificial_intelligence".to_string(),
        label: None,
        language: Some("en".to_string()),
        term: None,
        sense_label: None,
    };

    assert_eq!(node.term_from_uri(), "artificial intelligence");
    assert_eq!(node.language_code(), Some("en"));
}
