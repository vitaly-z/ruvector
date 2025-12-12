//! Payment engine - core processing logic

use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::types::{AccountId, TransactionId, AppId, Credits, Timestamp, RevenueSplit, generate_id};
use crate::accounts::{Account, AccountStatus, DeveloperProfile};
use crate::credits::{CreditBalance, CreditTransaction, CreditTransactionType, CreditPricing};
use crate::transactions::{Transaction, TransactionStatus, TransactionType, TransactionReceipt};
use crate::micropayments::{MicropaymentChannel, ChipAppPricing, MicropaymentReceipt, DailyUsageTracker};
use crate::subscriptions::{Subscription, SubscriptionTier, BillingPeriod};
use crate::error::{PaymentError, PaymentResult};

/// Payment engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentEngineConfig {
    /// Credit pricing configuration
    pub credit_pricing: CreditPricing,
    /// Default revenue split
    pub default_revenue_split: RevenueSplit,
    /// Chip app pricing
    pub chip_app_pricing: ChipAppPricing,
    /// Platform fee percentage (0-100)
    pub platform_fee_percentage: u8,
    /// Minimum payout amount
    pub min_payout_amount: Credits,
    /// Micropayment channel default duration (hours)
    pub channel_default_duration_hours: u64,
    /// Enable blockchain signatures
    pub enable_signatures: bool,
}

impl Default for PaymentEngineConfig {
    fn default() -> Self {
        PaymentEngineConfig {
            credit_pricing: CreditPricing::default(),
            default_revenue_split: RevenueSplit::default(),
            chip_app_pricing: ChipAppPricing::default(),
            platform_fee_percentage: 25,
            min_payout_amount: 1000, // $10 minimum payout
            channel_default_duration_hours: 24,
            enable_signatures: false,
        }
    }
}

/// Payment engine - thread-safe payment processor
pub struct PaymentEngine {
    /// Configuration
    config: PaymentEngineConfig,
    /// Accounts by ID
    accounts: DashMap<AccountId, Account>,
    /// Developer profiles by account ID
    developers: DashMap<AccountId, DeveloperProfile>,
    /// Transactions by ID
    transactions: DashMap<TransactionId, Transaction>,
    /// Credit transactions by account ID
    credit_transactions: DashMap<AccountId, Vec<CreditTransaction>>,
    /// Micropayment channels by ID
    channels: DashMap<String, MicropaymentChannel>,
    /// Usage trackers by (user, app) key
    usage_trackers: DashMap<String, DailyUsageTracker>,
    /// Idempotency keys to prevent duplicates
    idempotency_keys: DashMap<String, TransactionId>,
}

impl PaymentEngine {
    /// Create a new payment engine
    pub fn new() -> Self {
        Self::with_config(PaymentEngineConfig::default())
    }

    /// Create a payment engine with custom configuration
    pub fn with_config(config: PaymentEngineConfig) -> Self {
        PaymentEngine {
            config,
            accounts: DashMap::new(),
            developers: DashMap::new(),
            transactions: DashMap::new(),
            credit_transactions: DashMap::new(),
            channels: DashMap::new(),
            usage_trackers: DashMap::new(),
            idempotency_keys: DashMap::new(),
        }
    }

    // ==================== Account Management ====================

    /// Create a new account
    pub fn create_account(&self, external_id: Option<String>) -> PaymentResult<Account> {
        let account = Account::new(external_id);
        let id = account.id.clone();

        if self.accounts.contains_key(&id) {
            return Err(PaymentError::AccountAlreadyExists(id));
        }

        self.accounts.insert(id.clone(), account.clone());
        self.credit_transactions.insert(id, Vec::new());

        Ok(account)
    }

    /// Create a developer account
    pub fn create_developer_account(
        &self,
        external_id: Option<String>,
        name: String,
    ) -> PaymentResult<(Account, DeveloperProfile)> {
        let account = Account::new_developer(external_id);
        let id = account.id.clone();

        if self.accounts.contains_key(&id) {
            return Err(PaymentError::AccountAlreadyExists(id));
        }

        let profile = DeveloperProfile::new(id.clone(), name);

        self.accounts.insert(id.clone(), account.clone());
        self.developers.insert(id.clone(), profile.clone());
        self.credit_transactions.insert(id, Vec::new());

        Ok((account, profile))
    }

    /// Get account by ID
    pub fn get_account(&self, account_id: &AccountId) -> PaymentResult<Account> {
        self.accounts
            .get(account_id)
            .map(|a| a.clone())
            .ok_or_else(|| PaymentError::AccountNotFound(account_id.clone()))
    }

    /// Update account
    pub fn update_account(&self, account: &Account) -> PaymentResult<()> {
        if !self.accounts.contains_key(&account.id) {
            return Err(PaymentError::AccountNotFound(account.id.clone()));
        }
        self.accounts.insert(account.id.clone(), account.clone());
        Ok(())
    }

    // ==================== Credit Management ====================

    /// Add credits to an account
    pub fn add_credits(
        &self,
        account_id: &AccountId,
        amount: Credits,
        transaction_type: CreditTransactionType,
        description: String,
    ) -> PaymentResult<CreditTransaction> {
        let mut account = self.get_account(account_id)?;

        if account.status != AccountStatus::Active {
            return Err(PaymentError::AccountSuspended(account_id.clone()));
        }

        account.credits.add(amount)?;
        self.accounts.insert(account_id.clone(), account.clone());

        let tx = CreditTransaction::new(
            account_id.clone(),
            transaction_type,
            amount as i64,
            account.credits.available,
            description,
        );

        self.credit_transactions
            .entry(account_id.clone())
            .or_insert_with(Vec::new)
            .push(tx.clone());

        Ok(tx)
    }

    /// Purchase credits
    pub fn purchase_credits(
        &self,
        account_id: &AccountId,
        amount_cents: u64,
    ) -> PaymentResult<(Credits, CreditTransaction)> {
        if amount_cents < self.config.credit_pricing.min_purchase_cents {
            return Err(PaymentError::InvalidAmount(format!(
                "Minimum purchase is {} cents",
                self.config.credit_pricing.min_purchase_cents
            )));
        }

        let credits = self.config.credit_pricing.calculate_credits(amount_cents);
        let tx = self.add_credits(
            account_id,
            credits,
            CreditTransactionType::Purchase,
            format!("Purchased {} credits for ${:.2}", credits, amount_cents as f64 / 100.0),
        )?;

        Ok((credits, tx))
    }

    /// Get credit balance
    pub fn get_balance(&self, account_id: &AccountId) -> PaymentResult<CreditBalance> {
        let account = self.get_account(account_id)?;
        Ok(account.credits)
    }

    /// Get credit history
    pub fn get_credit_history(
        &self,
        account_id: &AccountId,
        limit: usize,
    ) -> PaymentResult<Vec<CreditTransaction>> {
        self.credit_transactions
            .get(account_id)
            .map(|txs| {
                txs.iter()
                    .rev()
                    .take(limit)
                    .cloned()
                    .collect()
            })
            .ok_or_else(|| PaymentError::AccountNotFound(account_id.clone()))
    }

    // ==================== Transaction Processing ====================

    /// Process a payment transaction
    pub fn process_transaction(&self, mut transaction: Transaction) -> PaymentResult<TransactionReceipt> {
        // Check idempotency
        if let Some(key) = &transaction.idempotency_key {
            if let Some(existing_id) = self.idempotency_keys.get(key) {
                return Err(PaymentError::DuplicateTransaction(existing_id.clone()));
            }
        }

        // Validate accounts
        let mut from_account = self.get_account(&transaction.from_account)?;
        if from_account.status != AccountStatus::Active {
            return Err(PaymentError::AccountSuspended(transaction.from_account.clone()));
        }

        // Start processing
        transaction.start_processing();

        // Check balance
        if from_account.credits.available < transaction.amount {
            transaction.fail("Insufficient credits");
            self.transactions.insert(transaction.id.clone(), transaction.clone());
            return Err(PaymentError::InsufficientCredits {
                required: transaction.amount,
                available: from_account.credits.available,
            });
        }

        // Deduct credits
        from_account.credits.deduct(transaction.amount)?;
        self.accounts.insert(from_account.id.clone(), from_account);

        // Credit destination account (if applicable)
        if let Some(to_account_id) = &transaction.to_account {
            if let Ok(mut to_account) = self.get_account(to_account_id) {
                to_account.credits.add(transaction.net_amount)?;
                self.accounts.insert(to_account_id.clone(), to_account);

                // Update developer profile if applicable
                if let Some(mut profile) = self.developers.get_mut(to_account_id) {
                    profile.add_revenue(transaction.net_amount);
                }
            }
        }

        // Complete transaction
        transaction.complete();

        // Store idempotency key
        if let Some(key) = &transaction.idempotency_key {
            self.idempotency_keys.insert(key.clone(), transaction.id.clone());
        }

        // Record credit transactions
        self.add_credits(
            &transaction.from_account,
            0, // Already deducted
            CreditTransactionType::Spent,
            format!("Payment: {}", transaction.description),
        ).ok();

        // Store transaction
        self.transactions.insert(transaction.id.clone(), transaction.clone());

        Ok(TransactionReceipt::from_transaction(&transaction))
    }

    /// Process an app payment
    pub fn process_app_payment(
        &self,
        user_account_id: &AccountId,
        developer_account_id: &AccountId,
        app_id: &AppId,
        amount: Credits,
    ) -> PaymentResult<TransactionReceipt> {
        let transaction = Transaction::app_payment(
            user_account_id.clone(),
            developer_account_id.clone(),
            app_id.clone(),
            amount,
            self.config.platform_fee_percentage,
        );

        self.process_transaction(transaction)
    }

    /// Get transaction by ID
    pub fn get_transaction(&self, transaction_id: &TransactionId) -> PaymentResult<Transaction> {
        self.transactions
            .get(transaction_id)
            .map(|t| t.clone())
            .ok_or_else(|| PaymentError::TransactionNotFound(transaction_id.clone()))
    }

    // ==================== Micropayments ====================

    /// Create a micropayment channel
    pub fn create_micropayment_channel(
        &self,
        user_account_id: &AccountId,
        developer_account_id: &AccountId,
        app_id: &AppId,
        deposit: Credits,
    ) -> PaymentResult<MicropaymentChannel> {
        // Verify accounts
        let mut user_account = self.get_account(user_account_id)?;
        let _dev_account = self.get_account(developer_account_id)?;

        // Reserve credits from user
        user_account.credits.reserve(deposit)?;
        self.accounts.insert(user_account_id.clone(), user_account);

        // Create channel
        let channel = MicropaymentChannel::new(
            user_account_id.clone(),
            developer_account_id.clone(),
            app_id.clone(),
            deposit,
            self.config.channel_default_duration_hours,
        );

        self.channels.insert(channel.id.clone(), channel.clone());

        Ok(channel)
    }

    /// Process a micropayment through a channel
    pub fn process_micropayment(
        &self,
        channel_id: &str,
        amount: Credits,
    ) -> PaymentResult<MicropaymentReceipt> {
        let mut channel = self.channels
            .get_mut(channel_id)
            .ok_or_else(|| PaymentError::TransactionNotFound(channel_id.to_string()))?;

        let receipt = channel.process_payment(amount)?;
        Ok(receipt)
    }

    /// Process a chip app use (with free tier handling)
    pub fn process_chip_app_use(
        &self,
        user_account_id: &AccountId,
        developer_account_id: &AccountId,
        app_id: &AppId,
    ) -> PaymentResult<Option<MicropaymentReceipt>> {
        let tracker_key = format!("{}:{}", user_account_id, app_id);

        // Get or create usage tracker
        let mut tracker = self.usage_trackers
            .entry(tracker_key.clone())
            .or_insert_with(|| DailyUsageTracker::new(user_account_id.clone(), app_id.clone()));

        // Check free tier
        if tracker.has_free_uses(self.config.chip_app_pricing.free_daily_uses) {
            tracker.record_usage();
            return Ok(None);
        }

        // Create or get micropayment channel
        let channel_key = format!("channel:{}:{}", user_account_id, app_id);
        let channel = if let Some(c) = self.channels.get(&channel_key) {
            if c.status == crate::micropayments::ChannelStatus::Active {
                c.clone()
            } else {
                // Create new channel with default deposit
                let deposit = self.config.chip_app_pricing.price_per_use * 100; // 100 uses
                self.create_micropayment_channel(
                    user_account_id,
                    developer_account_id,
                    app_id,
                    deposit,
                )?
            }
        } else {
            let deposit = self.config.chip_app_pricing.price_per_use * 100;
            let channel = self.create_micropayment_channel(
                user_account_id,
                developer_account_id,
                app_id,
                deposit,
            )?;
            self.channels.insert(channel_key.clone(), channel.clone());
            channel
        };

        // Process micropayment
        let receipt = self.process_micropayment(
            &channel.id,
            self.config.chip_app_pricing.price_per_use,
        )?;

        tracker.record_usage();

        Ok(Some(receipt))
    }

    // ==================== Subscriptions ====================

    /// Create a subscription
    pub fn create_subscription(
        &self,
        account_id: &AccountId,
        tier: SubscriptionTier,
        billing_period: BillingPeriod,
    ) -> PaymentResult<Subscription> {
        let mut account = self.get_account(account_id)?;

        // Create subscription
        let subscription = Subscription::new(account_id.clone(), tier, billing_period);

        // Add subscription credits
        account.credits.add(subscription.monthly_credits)?;
        account.subscription = Some(subscription.clone());

        self.accounts.insert(account_id.clone(), account);

        // Record credit transaction
        self.add_credits(
            account_id,
            0, // Already added above
            CreditTransactionType::Subscription,
            format!("Subscription credits: {:?}", tier),
        ).ok();

        Ok(subscription)
    }

    /// Create a trial subscription
    pub fn create_trial_subscription(
        &self,
        account_id: &AccountId,
        trial_days: u32,
    ) -> PaymentResult<Subscription> {
        let mut account = self.get_account(account_id)?;

        // Check if already has subscription
        if account.subscription.is_some() {
            return Err(PaymentError::Internal(
                "Account already has a subscription".to_string(),
            ));
        }

        let subscription = Subscription::new_trial(account_id.clone(), trial_days);
        account.credits.add(subscription.monthly_credits)?;
        account.subscription = Some(subscription.clone());

        self.accounts.insert(account_id.clone(), account);

        Ok(subscription)
    }

    /// Upgrade subscription
    pub fn upgrade_subscription(
        &self,
        account_id: &AccountId,
        new_tier: SubscriptionTier,
    ) -> PaymentResult<Credits> {
        let mut account = self.get_account(account_id)?;

        let subscription = account.subscription
            .as_mut()
            .ok_or_else(|| PaymentError::SubscriptionExpired)?;

        let added_credits = subscription.upgrade(new_tier)?;
        account.credits.add(added_credits)?;

        self.accounts.insert(account_id.clone(), account);

        Ok(added_credits)
    }

    /// Cancel subscription
    pub fn cancel_subscription(&self, account_id: &AccountId) -> PaymentResult<()> {
        let mut account = self.get_account(account_id)?;

        let subscription = account.subscription
            .as_mut()
            .ok_or_else(|| PaymentError::SubscriptionExpired)?;

        subscription.cancel();
        self.accounts.insert(account_id.clone(), account);

        Ok(())
    }

    // ==================== Statistics ====================

    /// Get engine statistics
    pub fn get_stats(&self) -> PaymentEngineStats {
        PaymentEngineStats {
            total_accounts: self.accounts.len(),
            total_developers: self.developers.len(),
            total_transactions: self.transactions.len(),
            active_channels: self.channels.iter()
                .filter(|c| c.status == crate::micropayments::ChannelStatus::Active)
                .count(),
        }
    }
}

impl Default for PaymentEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Payment engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentEngineStats {
    pub total_accounts: usize,
    pub total_developers: usize,
    pub total_transactions: usize,
    pub active_channels: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = PaymentEngine::new();
        let stats = engine.get_stats();
        assert_eq!(stats.total_accounts, 0);
    }

    #[test]
    fn test_account_management() {
        let engine = PaymentEngine::new();

        // Create account
        let account = engine.create_account(Some("ext_123".to_string())).unwrap();
        assert_eq!(account.credits.available, 0);

        // Get account
        let retrieved = engine.get_account(&account.id).unwrap();
        assert_eq!(retrieved.id, account.id);
    }

    #[test]
    fn test_credit_purchase() {
        let engine = PaymentEngine::new();
        let account = engine.create_account(None).unwrap();

        // Purchase credits
        let (credits, _tx) = engine.purchase_credits(&account.id, 1000).unwrap();
        assert_eq!(credits, 1000); // $10 = 1000 credits

        // Check balance
        let balance = engine.get_balance(&account.id).unwrap();
        assert_eq!(balance.available, 1000);
    }

    #[test]
    fn test_app_payment() {
        let engine = PaymentEngine::new();

        // Create accounts
        let user_account = engine.create_account(None).unwrap();
        let (dev_account, _profile) = engine.create_developer_account(
            None,
            "Test Developer".to_string(),
        ).unwrap();

        // Add credits to user
        engine.add_credits(
            &user_account.id,
            1000,
            CreditTransactionType::Purchase,
            "Test purchase".to_string(),
        ).unwrap();

        // Process payment
        let receipt = engine.process_app_payment(
            &user_account.id,
            &dev_account.id,
            &"test_app".to_string(),
            100,
        ).unwrap();

        assert_eq!(receipt.amount, 100);

        // Check balances
        let user_balance = engine.get_balance(&user_account.id).unwrap();
        assert_eq!(user_balance.available, 900);

        let dev_balance = engine.get_balance(&dev_account.id).unwrap();
        assert_eq!(dev_balance.available, 75); // 100 - 25% platform fee
    }

    #[test]
    fn test_subscription() {
        let engine = PaymentEngine::new();
        let account = engine.create_account(None).unwrap();

        // Create subscription
        let sub = engine.create_subscription(
            &account.id,
            SubscriptionTier::Pro,
            BillingPeriod::Monthly,
        ).unwrap();

        assert_eq!(sub.tier, SubscriptionTier::Pro);

        // Check credits were added
        let balance = engine.get_balance(&account.id).unwrap();
        assert_eq!(balance.available, 1000);
    }

    #[test]
    fn test_micropayment_channel() {
        let engine = PaymentEngine::new();

        // Create accounts
        let user_account = engine.create_account(None).unwrap();
        let (dev_account, _) = engine.create_developer_account(None, "Dev".to_string()).unwrap();

        // Add credits
        engine.add_credits(&user_account.id, 1000, CreditTransactionType::Purchase, "Test".to_string()).unwrap();

        // Create channel
        let channel = engine.create_micropayment_channel(
            &user_account.id,
            &dev_account.id,
            &"chip_app".to_string(),
            100,
        ).unwrap();

        assert_eq!(channel.deposited, 100);
        assert_eq!(channel.balance, 100);

        // Process micropayment
        let receipt = engine.process_micropayment(&channel.id, 5).unwrap();
        assert_eq!(receipt.amount, 5);
    }
}
