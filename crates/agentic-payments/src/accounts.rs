//! Account management for the payment system

use serde::{Deserialize, Serialize};
use crate::types::{AccountId, Timestamp, generate_id};
use crate::credits::CreditBalance;
use crate::subscriptions::Subscription;
use crate::error::{PaymentError, PaymentResult};

/// Account status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountStatus {
    /// Active account
    Active,
    /// Suspended account
    Suspended,
    /// Pending verification
    Pending,
    /// Closed account
    Closed,
}

impl Default for AccountStatus {
    fn default() -> Self {
        AccountStatus::Active
    }
}

/// Account type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountType {
    /// Individual user
    User,
    /// App developer
    Developer,
    /// Organization
    Organization,
    /// Platform/system account
    Platform,
}

impl Default for AccountType {
    fn default() -> Self {
        AccountType::User
    }
}

/// Account settings
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccountSettings {
    /// Enable auto-refill
    pub auto_refill_enabled: bool,
    /// Auto-refill threshold (refill when balance drops below)
    pub auto_refill_threshold: u64,
    /// Auto-refill amount
    pub auto_refill_amount: u64,
    /// Notification preferences
    pub notifications: NotificationSettings,
    /// Security settings
    pub security: SecuritySettings,
}

/// Notification settings
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NotificationSettings {
    /// Email notifications
    pub email_enabled: bool,
    /// Low balance alerts
    pub low_balance_alert: bool,
    /// Low balance threshold
    pub low_balance_threshold: u64,
    /// Transaction notifications
    pub transaction_notifications: bool,
}

/// Security settings
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecuritySettings {
    /// Two-factor authentication enabled
    pub two_factor_enabled: bool,
    /// Transaction signing required
    pub transaction_signing: bool,
    /// Allowed IP addresses (empty = all allowed)
    pub allowed_ips: Vec<String>,
}

/// Payment account
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    /// Unique account ID
    pub id: AccountId,
    /// External user ID (e.g., from auth provider)
    pub external_id: Option<String>,
    /// Account type
    pub account_type: AccountType,
    /// Account status
    pub status: AccountStatus,
    /// Credit balance
    pub credits: CreditBalance,
    /// Active subscription
    pub subscription: Option<Subscription>,
    /// Account settings
    pub settings: AccountSettings,
    /// Public key for transaction signing (optional)
    pub public_key: Option<String>,
    /// Account metadata
    pub metadata: Option<serde_json::Value>,
    /// Created timestamp
    pub created_at: Timestamp,
    /// Updated timestamp
    pub updated_at: Timestamp,
}

impl Account {
    /// Create a new account
    pub fn new(external_id: Option<String>) -> Self {
        let id = generate_id();
        Account {
            id: id.clone(),
            external_id,
            account_type: AccountType::User,
            status: AccountStatus::Active,
            credits: CreditBalance::new(id),
            subscription: None,
            settings: AccountSettings::default(),
            public_key: None,
            metadata: None,
            created_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        }
    }

    /// Create a developer account
    pub fn new_developer(external_id: Option<String>) -> Self {
        let mut account = Self::new(external_id);
        account.account_type = AccountType::Developer;
        account
    }

    /// Create an organization account
    pub fn new_organization(external_id: Option<String>) -> Self {
        let mut account = Self::new(external_id);
        account.account_type = AccountType::Organization;
        account
    }

    /// Check if account is active
    pub fn is_active(&self) -> bool {
        self.status == AccountStatus::Active
    }

    /// Suspend account
    pub fn suspend(&mut self, reason: &str) -> PaymentResult<()> {
        if self.status == AccountStatus::Closed {
            return Err(PaymentError::AccountSuspended(self.id.clone()));
        }
        self.status = AccountStatus::Suspended;
        self.updated_at = Timestamp::now();
        // Store reason in metadata
        let mut meta = self.metadata.clone().unwrap_or(serde_json::json!({}));
        meta["suspension_reason"] = serde_json::json!(reason);
        meta["suspended_at"] = serde_json::json!(Timestamp::now().unix_millis());
        self.metadata = Some(meta);
        Ok(())
    }

    /// Reactivate suspended account
    pub fn reactivate(&mut self) -> PaymentResult<()> {
        if self.status != AccountStatus::Suspended {
            return Err(PaymentError::Internal("Account is not suspended".to_string()));
        }
        self.status = AccountStatus::Active;
        self.updated_at = Timestamp::now();
        Ok(())
    }

    /// Close account
    pub fn close(&mut self) -> PaymentResult<()> {
        self.status = AccountStatus::Closed;
        self.updated_at = Timestamp::now();
        Ok(())
    }

    /// Set public key for transaction signing
    pub fn set_public_key(&mut self, public_key: String) {
        self.public_key = Some(public_key);
        self.updated_at = Timestamp::now();
    }

    /// Configure auto-refill
    pub fn configure_auto_refill(&mut self, enabled: bool, threshold: u64, amount: u64) {
        self.settings.auto_refill_enabled = enabled;
        self.settings.auto_refill_threshold = threshold;
        self.settings.auto_refill_amount = amount;
        self.updated_at = Timestamp::now();
    }

    /// Check if auto-refill should be triggered
    pub fn should_auto_refill(&self) -> bool {
        self.settings.auto_refill_enabled
            && self.credits.available < self.settings.auto_refill_threshold
    }
}

/// Developer profile for app creators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeveloperProfile {
    /// Associated account ID
    pub account_id: AccountId,
    /// Developer name
    pub name: String,
    /// Developer description
    pub description: Option<String>,
    /// Website URL
    pub website: Option<String>,
    /// Verified developer
    pub verified: bool,
    /// Published app count
    pub app_count: u32,
    /// Total downloads
    pub total_downloads: u64,
    /// Average rating (0-5)
    pub average_rating: f32,
    /// Revenue earned (in credits)
    pub total_revenue: u64,
    /// Created timestamp
    pub created_at: Timestamp,
}

impl DeveloperProfile {
    /// Create a new developer profile
    pub fn new(account_id: AccountId, name: String) -> Self {
        DeveloperProfile {
            account_id,
            name,
            description: None,
            website: None,
            verified: false,
            app_count: 0,
            total_downloads: 0,
            average_rating: 0.0,
            total_revenue: 0,
            created_at: Timestamp::now(),
        }
    }

    /// Add revenue to developer profile
    pub fn add_revenue(&mut self, amount: u64) {
        self.total_revenue = self.total_revenue.saturating_add(amount);
    }

    /// Increment download count
    pub fn add_download(&mut self) {
        self.total_downloads = self.total_downloads.saturating_add(1);
    }

    /// Update rating
    pub fn update_rating(&mut self, new_rating: f32, total_ratings: u32) {
        // Weighted average
        if total_ratings > 0 {
            self.average_rating =
                (self.average_rating * (total_ratings - 1) as f32 + new_rating)
                / total_ratings as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_creation() {
        let account = Account::new(Some("user_ext_123".to_string()));
        assert!(account.is_active());
        assert_eq!(account.account_type, AccountType::User);
        assert_eq!(account.credits.available, 0);
    }

    #[test]
    fn test_account_suspension() {
        let mut account = Account::new(None);
        account.suspend("Test suspension").unwrap();
        assert_eq!(account.status, AccountStatus::Suspended);
        assert!(!account.is_active());

        account.reactivate().unwrap();
        assert!(account.is_active());
    }

    #[test]
    fn test_auto_refill() {
        let mut account = Account::new(None);
        account.credits.add(50).unwrap();
        account.configure_auto_refill(true, 100, 500);

        assert!(account.should_auto_refill()); // Balance (50) < threshold (100)

        account.credits.add(100).unwrap();
        assert!(!account.should_auto_refill()); // Balance (150) >= threshold (100)
    }

    #[test]
    fn test_developer_profile() {
        let mut profile = DeveloperProfile::new("acc_123".to_string(), "Test Dev".to_string());

        profile.add_revenue(1000);
        assert_eq!(profile.total_revenue, 1000);

        profile.add_download();
        profile.add_download();
        assert_eq!(profile.total_downloads, 2);
    }
}
