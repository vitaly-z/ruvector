//! ConceptNet 5.7 API Client
//!
//! High-performance async client for querying the ConceptNet knowledge graph.
//! Implements all core endpoints: lookup, query, related, and relatedness.
//!
//! ## Features
//! - Rate-limited requests (3600/hour, 120/minute burst)
//! - Response caching with LRU eviction
//! - Automatic retry with exponential backoff
//! - Connection pooling
//!
//! ## Example
//! ```rust,no_run
//! # async fn example() -> anyhow::Result<()> {
//! use conceptnet_integration::api::ConceptNetClient;
//!
//! let client = ConceptNetClient::new();
//! let edges = client.lookup("/c/en/artificial_intelligence").await?;
//! # Ok(())
//! # }
//! ```

mod client;
mod types;
mod cache;
mod rate_limiter;

pub use client::ConceptNetClient;
pub use types::*;
pub use cache::ResponseCache;
pub use rate_limiter::RateLimiter;
