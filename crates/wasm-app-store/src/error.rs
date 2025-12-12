//! Error types for the WASM App Store

use thiserror::Error;

/// App store errors
#[derive(Error, Debug)]
pub enum AppStoreError {
    #[error("App not found: {0}")]
    AppNotFound(String),

    #[error("App already exists: {0}")]
    AppAlreadyExists(String),

    #[error("Invalid app: {0}")]
    InvalidApp(String),

    #[error("App size exceeds limit: {size} bytes > {limit} bytes")]
    SizeExceeded { size: usize, limit: usize },

    #[error("Invalid WASM module: {0}")]
    InvalidWasm(String),

    #[error("Version conflict: {0}")]
    VersionConflict(String),

    #[error("Category not found: {0}")]
    CategoryNotFound(String),

    #[error("Publisher not authorized: {0}")]
    Unauthorized(String),

    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    #[error("Payment required: {0}")]
    PaymentRequired(String),

    #[error("Download failed: {0}")]
    DownloadFailed(String),

    #[error("Compression error: {0}")]
    CompressionError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for app store operations
pub type AppStoreResult<T> = Result<T, AppStoreError>;

impl From<serde_json::Error> for AppStoreError {
    fn from(err: serde_json::Error) -> Self {
        AppStoreError::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for AppStoreError {
    fn from(err: std::io::Error) -> Self {
        AppStoreError::Internal(err.to_string())
    }
}

#[cfg(feature = "wasm")]
impl From<AppStoreError> for wasm_bindgen::JsValue {
    fn from(err: AppStoreError) -> Self {
        wasm_bindgen::JsValue::from_str(&err.to_string())
    }
}
