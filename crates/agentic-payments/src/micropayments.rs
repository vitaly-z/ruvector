//! Micropayment system for chip-sized WASM apps (≤8KB)

use serde::{Deserialize, Serialize};
use crate::types::{AccountId, AppId, Credits, Timestamp, AppSizeCategory, generate_id, generate_short_id};
use crate::transactions::{Transaction, TransactionType};
use crate::error::{PaymentError, PaymentResult};

/// Micropayment channel for frequent small transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicropaymentChannel {
    /// Channel ID
    pub id: String,
    /// User account ID
    pub user_account: AccountId,
    /// Developer account ID
    pub developer_account: AccountId,
    /// App ID
    pub app_id: AppId,
    /// Deposited amount (escrow)
    pub deposited: Credits,
    /// Spent amount
    pub spent: Credits,
    /// Remaining balance
    pub balance: Credits,
    /// Number of micropayments
    pub payment_count: u64,
    /// Channel status
    pub status: ChannelStatus,
    /// Expiration timestamp
    pub expires_at: Timestamp,
    /// Created timestamp
    pub created_at: Timestamp,
    /// Last payment timestamp
    pub last_payment_at: Option<Timestamp>,
}

/// Micropayment channel status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelStatus {
    /// Channel is active
    Active,
    /// Channel is closed
    Closed,
    /// Channel has expired
    Expired,
    /// Channel is being settled
    Settling,
}

impl MicropaymentChannel {
    /// Create a new micropayment channel
    pub fn new(
        user_account: AccountId,
        developer_account: AccountId,
        app_id: AppId,
        deposit: Credits,
        duration_hours: u64,
    ) -> Self {
        let now = Timestamp::now();
        let expires_at = Timestamp::from_unix_millis(
            now.unix_millis() + (duration_hours as i64 * 3600 * 1000),
        );

        MicropaymentChannel {
            id: generate_id(),
            user_account,
            developer_account,
            app_id,
            deposited: deposit,
            spent: 0,
            balance: deposit,
            payment_count: 0,
            status: ChannelStatus::Active,
            expires_at,
            created_at: now,
            last_payment_at: None,
        }
    }

    /// Process a micropayment through the channel
    pub fn process_payment(&mut self, amount: Credits) -> PaymentResult<MicropaymentReceipt> {
        // Check channel status
        if self.status != ChannelStatus::Active {
            return Err(PaymentError::TransactionFailed(
                format!("Channel is not active: {:?}", self.status)
            ));
        }

        // Check expiration
        if Timestamp::now().unix_millis() > self.expires_at.unix_millis() {
            self.status = ChannelStatus::Expired;
            return Err(PaymentError::TransactionFailed("Channel has expired".to_string()));
        }

        // Check balance
        if self.balance < amount {
            return Err(PaymentError::InsufficientCredits {
                required: amount,
                available: self.balance,
            });
        }

        // Process payment
        self.balance = self.balance.saturating_sub(amount);
        self.spent = self.spent.saturating_add(amount);
        self.payment_count += 1;
        self.last_payment_at = Some(Timestamp::now());

        Ok(MicropaymentReceipt {
            id: generate_short_id(),
            channel_id: self.id.clone(),
            amount,
            total_spent: self.spent,
            remaining_balance: self.balance,
            payment_number: self.payment_count,
            timestamp: Timestamp::now(),
        })
    }

    /// Top up the channel with additional credits
    pub fn top_up(&mut self, amount: Credits) -> PaymentResult<()> {
        if self.status != ChannelStatus::Active {
            return Err(PaymentError::TransactionFailed(
                "Cannot top up inactive channel".to_string(),
            ));
        }

        self.deposited = self.deposited.saturating_add(amount);
        self.balance = self.balance.saturating_add(amount);
        Ok(())
    }

    /// Close the channel and return remaining balance
    pub fn close(&mut self) -> PaymentResult<Credits> {
        if self.status == ChannelStatus::Closed {
            return Err(PaymentError::TransactionFailed(
                "Channel already closed".to_string(),
            ));
        }

        self.status = ChannelStatus::Settling;
        let refund = self.balance;
        self.balance = 0;
        self.status = ChannelStatus::Closed;

        Ok(refund)
    }

    /// Check if channel is expired
    pub fn is_expired(&self) -> bool {
        Timestamp::now().unix_millis() > self.expires_at.unix_millis()
    }

    /// Get channel utilization percentage
    pub fn utilization_percentage(&self) -> f64 {
        if self.deposited == 0 {
            return 0.0;
        }
        (self.spent as f64 / self.deposited as f64) * 100.0
    }
}

/// Micropayment receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicropaymentReceipt {
    /// Receipt ID (short for space efficiency)
    pub id: String,
    /// Channel ID
    pub channel_id: String,
    /// Payment amount
    pub amount: Credits,
    /// Total spent in channel
    pub total_spent: Credits,
    /// Remaining balance
    pub remaining_balance: Credits,
    /// Payment number
    pub payment_number: u64,
    /// Timestamp
    pub timestamp: Timestamp,
}

impl MicropaymentReceipt {
    /// Serialize to compact binary format
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        // Compact format: id (8) + amount (8) + number (8) + timestamp (8) = 32 bytes
        let mut bytes = Vec::with_capacity(32);
        bytes.extend_from_slice(self.id.as_bytes());
        bytes.extend_from_slice(&self.amount.to_le_bytes());
        bytes.extend_from_slice(&self.payment_number.to_le_bytes());
        bytes.extend_from_slice(&self.timestamp.unix_millis().to_le_bytes());
        bytes
    }
}

/// Micropayment pricing for chip apps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChipAppPricing {
    /// Price per use (in credits, 1 credit = 1 cent)
    pub price_per_use: Credits,
    /// Free uses per day
    pub free_daily_uses: u32,
    /// Bulk discount threshold
    pub bulk_threshold: u32,
    /// Bulk discount percentage
    pub bulk_discount_percentage: u8,
}

impl Default for ChipAppPricing {
    fn default() -> Self {
        ChipAppPricing {
            price_per_use: 1, // 1 cent per use
            free_daily_uses: 10,
            bulk_threshold: 100,
            bulk_discount_percentage: 20,
        }
    }
}

impl ChipAppPricing {
    /// Calculate price for a number of uses
    pub fn calculate_price(&self, uses: u32) -> Credits {
        if uses <= self.free_daily_uses {
            return 0;
        }

        let paid_uses = uses - self.free_daily_uses;
        let base_price = paid_uses as u64 * self.price_per_use;

        if paid_uses >= self.bulk_threshold {
            // Apply bulk discount
            let discount = (base_price * self.bulk_discount_percentage as u64) / 100;
            base_price - discount
        } else {
            base_price
        }
    }
}

/// Micropayment aggregator for batching small transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicropaymentAggregator {
    /// Developer account ID
    pub developer_account: AccountId,
    /// Pending payments by user
    pub pending_payments: Vec<PendingMicropayment>,
    /// Total pending amount
    pub total_pending: Credits,
    /// Settlement threshold
    pub settlement_threshold: Credits,
    /// Last settlement timestamp
    pub last_settlement: Option<Timestamp>,
}

/// Pending micropayment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingMicropayment {
    /// User account ID
    pub user_account: AccountId,
    /// App ID
    pub app_id: AppId,
    /// Amount
    pub amount: Credits,
    /// Timestamp
    pub timestamp: Timestamp,
}

impl MicropaymentAggregator {
    /// Create a new aggregator
    pub fn new(developer_account: AccountId, settlement_threshold: Credits) -> Self {
        MicropaymentAggregator {
            developer_account,
            pending_payments: Vec::new(),
            total_pending: 0,
            settlement_threshold,
            last_settlement: None,
        }
    }

    /// Add a pending payment
    pub fn add_payment(&mut self, payment: PendingMicropayment) {
        self.total_pending = self.total_pending.saturating_add(payment.amount);
        self.pending_payments.push(payment);
    }

    /// Check if settlement threshold is reached
    pub fn should_settle(&self) -> bool {
        self.total_pending >= self.settlement_threshold
    }

    /// Generate settlement transaction
    pub fn settle(&mut self) -> Option<Transaction> {
        if self.pending_payments.is_empty() {
            return None;
        }

        let total = self.total_pending;
        let count = self.pending_payments.len();

        // Create aggregated transaction
        let tx = Transaction::new(
            TransactionType::Payout,
            "platform".to_string(),
            total,
            format!("Aggregated micropayment settlement: {} payments", count),
        );

        // Clear pending payments
        self.pending_payments.clear();
        self.total_pending = 0;
        self.last_settlement = Some(Timestamp::now());

        Some(tx)
    }
}

/// Daily usage tracker for free tier management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyUsageTracker {
    /// User account ID
    pub user_account: AccountId,
    /// App ID
    pub app_id: AppId,
    /// Usage count for today
    pub today_usage: u32,
    /// Date (YYYYMMDD format)
    pub date: u32,
    /// Total lifetime usage
    pub lifetime_usage: u64,
}

impl DailyUsageTracker {
    /// Create a new usage tracker
    pub fn new(user_account: AccountId, app_id: AppId) -> Self {
        let now = chrono::Utc::now();
        let date = (now.format("%Y%m%d").to_string())
            .parse::<u32>()
            .unwrap_or(0);

        DailyUsageTracker {
            user_account,
            app_id,
            today_usage: 0,
            date,
            lifetime_usage: 0,
        }
    }

    /// Record a usage
    pub fn record_usage(&mut self) {
        let now = chrono::Utc::now();
        let current_date = (now.format("%Y%m%d").to_string())
            .parse::<u32>()
            .unwrap_or(0);

        // Reset if new day
        if current_date != self.date {
            self.today_usage = 0;
            self.date = current_date;
        }

        self.today_usage += 1;
        self.lifetime_usage += 1;
    }

    /// Check if free uses remain
    pub fn has_free_uses(&self, free_daily_limit: u32) -> bool {
        let now = chrono::Utc::now();
        let current_date = (now.format("%Y%m%d").to_string())
            .parse::<u32>()
            .unwrap_or(0);

        if current_date != self.date {
            return free_daily_limit > 0;
        }

        self.today_usage < free_daily_limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micropayment_channel() {
        let mut channel = MicropaymentChannel::new(
            "user_123".to_string(),
            "dev_456".to_string(),
            "chip_app".to_string(),
            100,
            24, // 24 hours
        );

        // Process payments
        let receipt1 = channel.process_payment(5).unwrap();
        assert_eq!(receipt1.amount, 5);
        assert_eq!(channel.balance, 95);

        let receipt2 = channel.process_payment(10).unwrap();
        assert_eq!(receipt2.payment_number, 2);
        assert_eq!(channel.spent, 15);

        // Check utilization
        assert!((channel.utilization_percentage() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_channel_insufficient_balance() {
        let mut channel = MicropaymentChannel::new(
            "user_123".to_string(),
            "dev_456".to_string(),
            "chip_app".to_string(),
            10,
            24,
        );

        let result = channel.process_payment(20);
        assert!(result.is_err());
    }

    #[test]
    fn test_chip_app_pricing() {
        let pricing = ChipAppPricing::default();

        // Free uses
        assert_eq!(pricing.calculate_price(5), 0);
        assert_eq!(pricing.calculate_price(10), 0);

        // Paid uses
        assert_eq!(pricing.calculate_price(15), 5); // 5 paid uses * 1 credit

        // Bulk discount (20% off after 100 paid uses)
        // 120 total uses = 110 paid uses = 110 credits - 20% = 88 credits
        assert_eq!(pricing.calculate_price(120), 88);
    }

    #[test]
    fn test_daily_usage_tracker() {
        let mut tracker = DailyUsageTracker::new(
            "user_123".to_string(),
            "chip_app".to_string(),
        );

        // Should have free uses
        assert!(tracker.has_free_uses(10));

        // Record usage
        for _ in 0..5 {
            tracker.record_usage();
        }

        assert_eq!(tracker.today_usage, 5);
        assert!(tracker.has_free_uses(10));

        // Use up free tier
        for _ in 0..5 {
            tracker.record_usage();
        }

        assert!(!tracker.has_free_uses(10));
        assert_eq!(tracker.lifetime_usage, 10);
    }

    #[test]
    fn test_micropayment_aggregator() {
        let mut aggregator = MicropaymentAggregator::new(
            "dev_456".to_string(),
            100, // Settlement threshold
        );

        // Add payments
        for i in 0..5 {
            aggregator.add_payment(PendingMicropayment {
                user_account: format!("user_{}", i),
                app_id: "chip_app".to_string(),
                amount: 10,
                timestamp: Timestamp::now(),
            });
        }

        assert_eq!(aggregator.total_pending, 50);
        assert!(!aggregator.should_settle());

        // Add more to reach threshold
        for i in 5..15 {
            aggregator.add_payment(PendingMicropayment {
                user_account: format!("user_{}", i),
                app_id: "chip_app".to_string(),
                amount: 10,
                timestamp: Timestamp::now(),
            });
        }

        assert!(aggregator.should_settle());

        // Settle
        let tx = aggregator.settle();
        assert!(tx.is_some());
        assert_eq!(tx.unwrap().amount, 150);
        assert!(aggregator.pending_payments.is_empty());
    }
}
