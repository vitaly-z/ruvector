//! ConceptNet Data Loaders
//!
//! Efficient loaders for the full ConceptNet dataset:
//! - Assertions CSV (~34M edges, ~2.5GB uncompressed)
//! - Numberbatch embeddings (~400K concepts, 300-dim)
//!
//! ## Data Sources
//! - Assertions: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
//! - Numberbatch: https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz

mod assertions;
mod numberbatch_loader;
mod streaming;
mod optimization;

pub use assertions::{AssertionsLoader, AssertionsConfig, LoadProgress};
pub use numberbatch_loader::{NumberbatchLoader, NumberbatchConfig};
pub use streaming::{StreamingLoader, BatchProcessor};
pub use optimization::{
    RuVectorOptimizer, RuvLLMOptimizer, OptimizationConfig,
    OptimizationMetrics, TrainingCallback,
};
