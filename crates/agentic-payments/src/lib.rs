//! # Agentic Payments
//!
//! High-performance payment processing library for WASM App Store.
//! Supports credits, subscriptions, micropayments, and blockchain integration.
//!
//! ## Features
//!
//! - **Credit System**: Manage user credits with atomic operations
//! - **Micropayments**: Sub-cent transactions for chip-sized apps
//! - **Subscriptions**: Tiered subscription management (Free, Pro, Enterprise)
//! - **Transaction Processing**: ACID-compliant transaction handling
//! - **WASM Support**: Full browser and embedded support
//! - **Blockchain Ready**: Optional Ed25519 signatures and verification
//!
//! ## Example
//!
//! ```rust
//! use agentic_payments::{PaymentEngine, CreditTransactionType};
//!
//! let engine = PaymentEngine::new();
//! let account = engine.create_account(Some("user_123".to_string())).unwrap();
//!
//! // Add credits
//! engine.add_credits(&account.id, 1000, CreditTransactionType::Purchase, "Purchase".to_string()).unwrap();
//!
//! // Check balance
//! let balance = engine.get_balance(&account.id).unwrap();
//! assert_eq!(balance.available, 1000);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec, boxed::Box};

pub mod credits;
pub mod micropayments;
pub mod subscriptions;
pub mod transactions;
pub mod accounts;
pub mod engine;
pub mod error;
pub mod types;

#[cfg(feature = "blockchain")]
pub mod crypto;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports
pub use credits::*;
pub use micropayments::*;
pub use subscriptions::*;
pub use transactions::*;
pub use accounts::*;
pub use engine::*;
pub use error::*;
pub use types::*;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Maximum chip app size in bytes (8KB)
pub const CHIP_APP_MAX_SIZE: usize = 8 * 1024;

/// Minimum credit purchase amount in cents
pub const MIN_PURCHASE_CENTS: u64 = 1000; // $10

/// Credit to cents conversion rate (1 credit = 1 cent)
pub const CREDITS_PER_CENT: u64 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_constants() {
        assert_eq!(CHIP_APP_MAX_SIZE, 8192);
        assert_eq!(MIN_PURCHASE_CENTS, 1000);
    }
}
