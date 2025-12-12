//! Credit management system

use serde::{Deserialize, Serialize};
use crate::types::{AccountId, Credits, Timestamp, generate_id};
use crate::error::{PaymentError, PaymentResult};

/// Credit balance for an account
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditBalance {
    /// Account identifier
    pub account_id: AccountId,
    /// Available credits
    pub available: Credits,
    /// Reserved credits (pending transactions)
    pub reserved: Credits,
    /// Lifetime credits earned
    pub lifetime_earned: Credits,
    /// Lifetime credits spent
    pub lifetime_spent: Credits,
    /// Last updated timestamp
    pub updated_at: Timestamp,
}

impl CreditBalance {
    /// Create a new credit balance for an account
    pub fn new(account_id: AccountId) -> Self {
        CreditBalance {
            account_id,
            available: 0,
            reserved: 0,
            lifetime_earned: 0,
            lifetime_spent: 0,
            updated_at: Timestamp::now(),
        }
    }

    /// Get total balance (available + reserved)
    pub fn total(&self) -> Credits {
        self.available.saturating_add(self.reserved)
    }

    /// Check if sufficient credits are available
    pub fn has_sufficient(&self, amount: Credits) -> bool {
        self.available >= amount
    }

    /// Add credits to the balance
    pub fn add(&mut self, amount: Credits) -> PaymentResult<()> {
        self.available = self.available.saturating_add(amount);
        self.lifetime_earned = self.lifetime_earned.saturating_add(amount);
        self.updated_at = Timestamp::now();
        Ok(())
    }

    /// Reserve credits for a pending transaction
    pub fn reserve(&mut self, amount: Credits) -> PaymentResult<()> {
        if self.available < amount {
            return Err(PaymentError::InsufficientCredits {
                required: amount,
                available: self.available,
            });
        }
        self.available = self.available.saturating_sub(amount);
        self.reserved = self.reserved.saturating_add(amount);
        self.updated_at = Timestamp::now();
        Ok(())
    }

    /// Release reserved credits (cancel transaction)
    pub fn release(&mut self, amount: Credits) -> PaymentResult<()> {
        let release_amount = amount.min(self.reserved);
        self.reserved = self.reserved.saturating_sub(release_amount);
        self.available = self.available.saturating_add(release_amount);
        self.updated_at = Timestamp::now();
        Ok(())
    }

    /// Commit reserved credits (complete transaction)
    pub fn commit(&mut self, amount: Credits) -> PaymentResult<()> {
        let commit_amount = amount.min(self.reserved);
        self.reserved = self.reserved.saturating_sub(commit_amount);
        self.lifetime_spent = self.lifetime_spent.saturating_add(commit_amount);
        self.updated_at = Timestamp::now();
        Ok(())
    }

    /// Deduct credits directly (for immediate transactions)
    pub fn deduct(&mut self, amount: Credits) -> PaymentResult<()> {
        if self.available < amount {
            return Err(PaymentError::InsufficientCredits {
                required: amount,
                available: self.available,
            });
        }
        self.available = self.available.saturating_sub(amount);
        self.lifetime_spent = self.lifetime_spent.saturating_add(amount);
        self.updated_at = Timestamp::now();
        Ok(())
    }
}

/// Credit transaction types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreditTransactionType {
    /// Credits purchased
    Purchase,
    /// Credits earned from activity
    Earned,
    /// Credits spent on app
    Spent,
    /// Credits refunded
    Refund,
    /// Credits transferred
    Transfer,
    /// Bonus credits
    Bonus,
    /// Promotional credits
    Promo,
    /// Subscription credits
    Subscription,
    /// Challenge reward
    ChallengeReward,
    /// Referral bonus
    ReferralBonus,
    /// Revenue share payout
    RevenueShare,
}

/// Credit transaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditTransaction {
    /// Unique transaction ID
    pub id: String,
    /// Account ID
    pub account_id: AccountId,
    /// Transaction type
    pub transaction_type: CreditTransactionType,
    /// Amount (positive for credit, negative for debit)
    pub amount: i64,
    /// Balance after transaction
    pub balance_after: Credits,
    /// Description
    pub description: String,
    /// Related app ID (if applicable)
    pub app_id: Option<String>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
    /// Created timestamp
    pub created_at: Timestamp,
}

impl CreditTransaction {
    /// Create a new credit transaction
    pub fn new(
        account_id: AccountId,
        transaction_type: CreditTransactionType,
        amount: i64,
        balance_after: Credits,
        description: String,
    ) -> Self {
        CreditTransaction {
            id: generate_id(),
            account_id,
            transaction_type,
            amount,
            balance_after,
            description,
            app_id: None,
            metadata: None,
            created_at: Timestamp::now(),
        }
    }

    /// Set app ID for this transaction
    pub fn with_app(mut self, app_id: String) -> Self {
        self.app_id = Some(app_id);
        self
    }

    /// Set metadata for this transaction
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Credit pricing tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditPricing {
    /// Credits per dollar
    pub credits_per_dollar: Credits,
    /// Minimum purchase amount in cents
    pub min_purchase_cents: u64,
    /// Maximum purchase amount in cents
    pub max_purchase_cents: u64,
    /// Bulk discount tiers
    pub bulk_discounts: Vec<BulkDiscount>,
}

impl Default for CreditPricing {
    fn default() -> Self {
        CreditPricing {
            credits_per_dollar: 100, // $1 = 100 credits
            min_purchase_cents: 1000, // $10 minimum
            max_purchase_cents: 1000000, // $10,000 maximum
            bulk_discounts: vec![
                BulkDiscount { min_amount: 5000, bonus_percentage: 5 },   // $50+ = 5% bonus
                BulkDiscount { min_amount: 10000, bonus_percentage: 10 }, // $100+ = 10% bonus
                BulkDiscount { min_amount: 50000, bonus_percentage: 15 }, // $500+ = 15% bonus
                BulkDiscount { min_amount: 100000, bonus_percentage: 20 }, // $1000+ = 20% bonus
            ],
        }
    }
}

/// Bulk discount tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkDiscount {
    /// Minimum purchase amount in cents
    pub min_amount: u64,
    /// Bonus percentage
    pub bonus_percentage: u8,
}

impl CreditPricing {
    /// Calculate credits for a given purchase amount
    pub fn calculate_credits(&self, amount_cents: u64) -> Credits {
        let base_credits = (amount_cents * self.credits_per_dollar) / 100;
        let bonus_percentage = self.get_bonus_percentage(amount_cents);
        let bonus_credits = (base_credits * bonus_percentage as u64) / 100;
        base_credits + bonus_credits
    }

    /// Get bonus percentage for amount
    fn get_bonus_percentage(&self, amount_cents: u64) -> u8 {
        self.bulk_discounts
            .iter()
            .rev()
            .find(|d| amount_cents >= d.min_amount)
            .map(|d| d.bonus_percentage)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credit_balance_operations() {
        let mut balance = CreditBalance::new("user_123".to_string());

        // Add credits
        balance.add(1000).unwrap();
        assert_eq!(balance.available, 1000);
        assert_eq!(balance.lifetime_earned, 1000);

        // Reserve credits
        balance.reserve(300).unwrap();
        assert_eq!(balance.available, 700);
        assert_eq!(balance.reserved, 300);

        // Commit reserved
        balance.commit(300).unwrap();
        assert_eq!(balance.reserved, 0);
        assert_eq!(balance.lifetime_spent, 300);

        // Insufficient credits
        let result = balance.deduct(1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_credit_pricing() {
        let pricing = CreditPricing::default();

        // Basic purchase: $10 = 1000 credits
        assert_eq!(pricing.calculate_credits(1000), 1000);

        // Bulk purchase: $100 = 10,000 + 10% = 11,000 credits
        assert_eq!(pricing.calculate_credits(10000), 11000);

        // Large purchase: $1000 = 100,000 + 20% = 120,000 credits
        assert_eq!(pricing.calculate_credits(100000), 120000);
    }
}
