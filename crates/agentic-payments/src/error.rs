//! Error types for the payment system

use thiserror::Error;
use crate::types::{AccountId, TransactionId, AppId, Credits};

/// Payment system errors
#[derive(Error, Debug)]
pub enum PaymentError {
    #[error("Account not found: {0}")]
    AccountNotFound(AccountId),

    #[error("Transaction not found: {0}")]
    TransactionNotFound(TransactionId),

    #[error("App not found: {0}")]
    AppNotFound(AppId),

    #[error("Insufficient credits: required {required}, available {available}")]
    InsufficientCredits { required: Credits, available: Credits },

    #[error("Invalid amount: {0}")]
    InvalidAmount(String),

    #[error("Transaction failed: {0}")]
    TransactionFailed(String),

    #[error("Duplicate transaction: {0}")]
    DuplicateTransaction(TransactionId),

    #[error("Account already exists: {0}")]
    AccountAlreadyExists(AccountId),

    #[error("Account suspended: {0}")]
    AccountSuspended(AccountId),

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Subscription expired")]
    SubscriptionExpired,

    #[error("Invalid subscription tier: {0}")]
    InvalidSubscriptionTier(String),

    #[error("Payment method declined: {0}")]
    PaymentDeclined(String),

    #[error("Refund not allowed: {0}")]
    RefundNotAllowed(String),

    #[error("App size exceeds limit: {size} bytes > {limit} bytes")]
    AppSizeExceeded { size: usize, limit: usize },

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for payment operations
pub type PaymentResult<T> = Result<T, PaymentError>;

impl From<serde_json::Error> for PaymentError {
    fn from(err: serde_json::Error) -> Self {
        PaymentError::Serialization(err.to_string())
    }
}

impl From<std::io::Error> for PaymentError {
    fn from(err: std::io::Error) -> Self {
        PaymentError::Internal(err.to_string())
    }
}

#[cfg(feature = "wasm")]
impl From<PaymentError> for wasm_bindgen::JsValue {
    fn from(err: PaymentError) -> Self {
        wasm_bindgen::JsValue::from_str(&err.to_string())
    }
}
