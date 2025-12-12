//! Core types for the payment system

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Unique identifier for accounts
pub type AccountId = String;

/// Unique identifier for transactions
pub type TransactionId = String;

/// Unique identifier for apps
pub type AppId = String;

/// Credit amount (1 credit = 1 cent)
pub type Credits = u64;

/// Currency amount in smallest unit (cents for USD)
pub type CurrencyAmount = u64;

/// Supported currencies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Currency {
    USD,
    EUR,
    GBP,
    JPY,
    /// rUv native credits
    RUV,
}

impl Default for Currency {
    fn default() -> Self {
        Currency::USD
    }
}

/// Payment method types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaymentMethod {
    /// Credit card payment
    CreditCard {
        last_four: String,
        brand: String,
        exp_month: u8,
        exp_year: u16,
    },
    /// Cryptocurrency payment
    Crypto {
        chain: String,
        address: String,
    },
    /// Bank transfer
    BankTransfer {
        bank_name: String,
        account_last_four: String,
    },
    /// Platform credits
    Credits,
    /// External payment provider
    External {
        provider: String,
        reference: String,
    },
}

/// App size category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppSizeCategory {
    /// Chip-sized app: <= 8KB WASM
    Chip,
    /// Micro app: <= 64KB WASM
    Micro,
    /// Small app: <= 512KB WASM
    Small,
    /// Medium app: <= 2MB WASM
    Medium,
    /// Large app: <= 10MB WASM
    Large,
    /// Full app: > 10MB WASM
    Full,
}

impl AppSizeCategory {
    /// Get the maximum size in bytes for this category
    pub fn max_bytes(&self) -> usize {
        match self {
            AppSizeCategory::Chip => 8 * 1024,          // 8KB
            AppSizeCategory::Micro => 64 * 1024,        // 64KB
            AppSizeCategory::Small => 512 * 1024,       // 512KB
            AppSizeCategory::Medium => 2 * 1024 * 1024, // 2MB
            AppSizeCategory::Large => 10 * 1024 * 1024, // 10MB
            AppSizeCategory::Full => usize::MAX,
        }
    }

    /// Get the base price in credits for this category
    pub fn base_price(&self) -> Credits {
        match self {
            AppSizeCategory::Chip => 1,      // 1 cent per use
            AppSizeCategory::Micro => 5,     // 5 cents
            AppSizeCategory::Small => 10,    // 10 cents
            AppSizeCategory::Medium => 25,   // 25 cents
            AppSizeCategory::Large => 50,    // 50 cents
            AppSizeCategory::Full => 100,    // $1
        }
    }

    /// Determine category from byte size
    pub fn from_size(bytes: usize) -> Self {
        if bytes <= 8 * 1024 {
            AppSizeCategory::Chip
        } else if bytes <= 64 * 1024 {
            AppSizeCategory::Micro
        } else if bytes <= 512 * 1024 {
            AppSizeCategory::Small
        } else if bytes <= 2 * 1024 * 1024 {
            AppSizeCategory::Medium
        } else if bytes <= 10 * 1024 * 1024 {
            AppSizeCategory::Large
        } else {
            AppSizeCategory::Full
        }
    }
}

/// Pricing model for apps
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PricingModel {
    /// Free app
    Free,
    /// One-time purchase
    OneTime { price: Credits },
    /// Pay per use
    PayPerUse { price_per_use: Credits },
    /// Subscription based
    Subscription {
        monthly_price: Credits,
        annual_price: Credits,
    },
    /// Usage-based metering
    Metered {
        base_price: Credits,
        per_unit_price: Credits,
        unit_name: String,
    },
    /// Revenue share model
    RevenueShare {
        creator_percentage: u8, // 0-100
        platform_percentage: u8,
    },
}

impl Default for PricingModel {
    fn default() -> Self {
        PricingModel::Free
    }
}

/// Revenue split configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevenueSplit {
    /// Creator's share (percentage)
    pub creator_share: u8,
    /// Platform's share (percentage)
    pub platform_share: u8,
    /// Referrer's share (percentage, if applicable)
    pub referrer_share: u8,
}

impl Default for RevenueSplit {
    fn default() -> Self {
        RevenueSplit {
            creator_share: 70,
            platform_share: 25,
            referrer_share: 5,
        }
    }
}

/// Timestamp wrapper for cross-platform compatibility
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timestamp(pub DateTime<Utc>);

impl Timestamp {
    pub fn now() -> Self {
        Timestamp(Utc::now())
    }

    pub fn unix_millis(&self) -> i64 {
        self.0.timestamp_millis()
    }

    pub fn from_unix_millis(millis: i64) -> Self {
        Timestamp(DateTime::from_timestamp_millis(millis).unwrap_or_else(Utc::now))
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Timestamp::now()
    }
}

/// Generate a new unique ID
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

/// Generate a short ID for chip apps (8 chars)
pub fn generate_short_id() -> String {
    let uuid = Uuid::new_v4();
    let bytes = uuid.as_bytes();
    hex::encode(&bytes[0..4])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_size_category() {
        assert_eq!(AppSizeCategory::from_size(1000), AppSizeCategory::Chip);
        assert_eq!(AppSizeCategory::from_size(8192), AppSizeCategory::Chip);
        assert_eq!(AppSizeCategory::from_size(8193), AppSizeCategory::Micro);
        assert_eq!(AppSizeCategory::from_size(64 * 1024), AppSizeCategory::Micro);
    }

    #[test]
    fn test_generate_ids() {
        let id = generate_id();
        assert_eq!(id.len(), 36); // UUID format

        let short = generate_short_id();
        assert_eq!(short.len(), 8);
    }

    #[test]
    fn test_revenue_split() {
        let split = RevenueSplit::default();
        assert_eq!(split.creator_share + split.platform_share + split.referrer_share, 100);
    }
}
