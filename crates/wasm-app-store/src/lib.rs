//! # WASM App Store
//!
//! A comprehensive marketplace for WebAssembly applications supporting both
//! chip-sized (≤8KB) and full-sized applications for browser, mobile, and embedded targets.
//!
//! ## Features
//!
//! - **Chip Apps**: Ultra-small WASM modules (≤8KB) with micropayment support
//! - **Full Apps**: Larger applications with rich feature sets
//! - **App Registry**: Publish, discover, and manage WASM applications
//! - **Version Control**: Semantic versioning and update management
//! - **Payment Integration**: Built-in payment processing for app monetization
//! - **Verification**: Cryptographic signing and integrity verification
//!
//! ## Example
//!
//! ```rust
//! use wasm_app_store::{AppStore, ChipApp, FullApp, AppMetadata};
//!
//! let store = AppStore::new();
//!
//! // Publish a chip app (≤8KB)
//! let chip_app = ChipApp::new("calculator", wasm_bytes, metadata);
//! store.publish_chip_app(chip_app).unwrap();
//!
//! // Search for apps
//! let results = store.search("calculator", 10);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod apps;
pub mod chip;
pub mod full;
pub mod registry;
pub mod store;
pub mod category;
pub mod version;
pub mod error;

#[cfg(feature = "payments")]
pub mod payments_integration;

#[cfg(feature = "verification")]
pub mod verification;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports
pub use apps::*;
pub use chip::*;
pub use full::*;
pub use registry::*;
pub use store::*;
pub use category::*;
pub use version::*;
pub use error::*;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Maximum chip app size (8KB)
pub const MAX_CHIP_APP_SIZE: usize = 8 * 1024;

/// Maximum micro app size (64KB)
pub const MAX_MICRO_APP_SIZE: usize = 64 * 1024;

/// Maximum small app size (512KB)
pub const MAX_SMALL_APP_SIZE: usize = 512 * 1024;

/// Maximum medium app size (2MB)
pub const MAX_MEDIUM_APP_SIZE: usize = 2 * 1024 * 1024;

/// Maximum large app size (10MB)
pub const MAX_LARGE_APP_SIZE: usize = 10 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_size_limits() {
        assert_eq!(MAX_CHIP_APP_SIZE, 8192);
        assert!(MAX_MICRO_APP_SIZE > MAX_CHIP_APP_SIZE);
        assert!(MAX_SMALL_APP_SIZE > MAX_MICRO_APP_SIZE);
    }
}
