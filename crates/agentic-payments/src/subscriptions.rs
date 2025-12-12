//! Subscription management for tiered access

use serde::{Deserialize, Serialize};
use crate::types::{AccountId, Credits, Timestamp, generate_id};
use crate::error::{PaymentError, PaymentResult};

/// Subscription tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SubscriptionTier {
    /// Free tier with limited credits
    Free,
    /// Pro tier with standard features
    Pro,
    /// Enterprise tier with unlimited access
    Enterprise,
    /// Custom tier for specific needs
    Custom,
}

impl Default for SubscriptionTier {
    fn default() -> Self {
        SubscriptionTier::Free
    }
}

impl SubscriptionTier {
    /// Get monthly credit allocation
    pub fn monthly_credits(&self) -> Credits {
        match self {
            SubscriptionTier::Free => 100,
            SubscriptionTier::Pro => 1000,
            SubscriptionTier::Enterprise => 10000,
            SubscriptionTier::Custom => 0, // Configured separately
        }
    }

    /// Get monthly price in cents
    pub fn monthly_price_cents(&self) -> u64 {
        match self {
            SubscriptionTier::Free => 0,
            SubscriptionTier::Pro => 2900, // $29
            SubscriptionTier::Enterprise => 9900, // $99
            SubscriptionTier::Custom => 0, // Configured separately
        }
    }

    /// Get annual price in cents (with discount)
    pub fn annual_price_cents(&self) -> u64 {
        // 2 months free for annual
        (self.monthly_price_cents() * 10)
    }

    /// Get priority level (higher = more priority)
    pub fn priority_level(&self) -> u8 {
        match self {
            SubscriptionTier::Free => 0,
            SubscriptionTier::Pro => 5,
            SubscriptionTier::Enterprise => 10,
            SubscriptionTier::Custom => 8,
        }
    }

    /// Get rate limit (requests per minute)
    pub fn rate_limit(&self) -> u32 {
        match self {
            SubscriptionTier::Free => 10,
            SubscriptionTier::Pro => 100,
            SubscriptionTier::Enterprise => 1000,
            SubscriptionTier::Custom => 500,
        }
    }

    /// Get max concurrent executions
    pub fn max_concurrent(&self) -> u32 {
        match self {
            SubscriptionTier::Free => 1,
            SubscriptionTier::Pro => 5,
            SubscriptionTier::Enterprise => 50,
            SubscriptionTier::Custom => 20,
        }
    }
}

/// Billing period
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BillingPeriod {
    Monthly,
    Annual,
}

impl Default for BillingPeriod {
    fn default() -> Self {
        BillingPeriod::Monthly
    }
}

/// Subscription status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionStatus {
    /// Subscription is active
    Active,
    /// Payment pending
    PastDue,
    /// Subscription cancelled but still active until period end
    Cancelled,
    /// Subscription has expired
    Expired,
    /// Trial period
    Trial,
    /// Paused subscription
    Paused,
}

impl Default for SubscriptionStatus {
    fn default() -> Self {
        SubscriptionStatus::Active
    }
}

/// User subscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    /// Subscription ID
    pub id: String,
    /// Account ID
    pub account_id: AccountId,
    /// Subscription tier
    pub tier: SubscriptionTier,
    /// Billing period
    pub billing_period: BillingPeriod,
    /// Status
    pub status: SubscriptionStatus,
    /// Monthly credits allocated
    pub monthly_credits: Credits,
    /// Credits remaining this period
    pub credits_remaining: Credits,
    /// Custom features (for Custom tier)
    pub custom_features: Option<CustomFeatures>,
    /// Current period start
    pub current_period_start: Timestamp,
    /// Current period end
    pub current_period_end: Timestamp,
    /// Trial end date (if in trial)
    pub trial_end: Option<Timestamp>,
    /// Cancellation date
    pub cancelled_at: Option<Timestamp>,
    /// Created timestamp
    pub created_at: Timestamp,
    /// Updated timestamp
    pub updated_at: Timestamp,
}

/// Custom features for Custom tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomFeatures {
    /// Custom monthly credits
    pub monthly_credits: Credits,
    /// Custom rate limit
    pub rate_limit: u32,
    /// Custom max concurrent
    pub max_concurrent: u32,
    /// Custom monthly price in cents
    pub monthly_price_cents: u64,
    /// Additional features
    pub features: Vec<String>,
}

impl Subscription {
    /// Create a new subscription
    pub fn new(account_id: AccountId, tier: SubscriptionTier, billing_period: BillingPeriod) -> Self {
        let now = Timestamp::now();
        let period_end = Self::calculate_period_end(&now, billing_period);
        let monthly_credits = tier.monthly_credits();

        Subscription {
            id: generate_id(),
            account_id,
            tier,
            billing_period,
            status: SubscriptionStatus::Active,
            monthly_credits,
            credits_remaining: monthly_credits,
            custom_features: None,
            current_period_start: now.clone(),
            current_period_end: period_end,
            trial_end: None,
            cancelled_at: None,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    /// Create a trial subscription
    pub fn new_trial(account_id: AccountId, trial_days: u32) -> Self {
        let now = Timestamp::now();
        let trial_end = Timestamp::from_unix_millis(
            now.unix_millis() + (trial_days as i64 * 24 * 3600 * 1000),
        );

        let mut subscription = Self::new(account_id, SubscriptionTier::Pro, BillingPeriod::Monthly);
        subscription.status = SubscriptionStatus::Trial;
        subscription.trial_end = Some(trial_end.clone());
        subscription.current_period_end = trial_end;
        subscription
    }

    /// Create a custom subscription
    pub fn new_custom(account_id: AccountId, features: CustomFeatures, billing_period: BillingPeriod) -> Self {
        let mut subscription = Self::new(account_id, SubscriptionTier::Custom, billing_period);
        subscription.monthly_credits = features.monthly_credits;
        subscription.credits_remaining = features.monthly_credits;
        subscription.custom_features = Some(features);
        subscription
    }

    /// Check if subscription is active
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            SubscriptionStatus::Active | SubscriptionStatus::Trial | SubscriptionStatus::PastDue
        )
    }

    /// Check if subscription is expired
    pub fn is_expired(&self) -> bool {
        self.status == SubscriptionStatus::Expired
            || Timestamp::now().unix_millis() > self.current_period_end.unix_millis()
    }

    /// Use credits
    pub fn use_credits(&mut self, amount: Credits) -> PaymentResult<()> {
        if !self.is_active() {
            return Err(PaymentError::SubscriptionExpired);
        }

        if self.credits_remaining < amount {
            return Err(PaymentError::InsufficientCredits {
                required: amount,
                available: self.credits_remaining,
            });
        }

        self.credits_remaining = self.credits_remaining.saturating_sub(amount);
        self.updated_at = Timestamp::now();
        Ok(())
    }

    /// Add bonus credits
    pub fn add_credits(&mut self, amount: Credits) {
        self.credits_remaining = self.credits_remaining.saturating_add(amount);
        self.updated_at = Timestamp::now();
    }

    /// Renew subscription for next period
    pub fn renew(&mut self) -> PaymentResult<()> {
        if self.status == SubscriptionStatus::Cancelled {
            return Err(PaymentError::SubscriptionExpired);
        }

        let now = Timestamp::now();
        self.current_period_start = now.clone();
        self.current_period_end = Self::calculate_period_end(&now, self.billing_period);
        self.credits_remaining = self.monthly_credits;
        self.status = SubscriptionStatus::Active;
        self.updated_at = now;

        Ok(())
    }

    /// Cancel subscription
    pub fn cancel(&mut self) {
        self.status = SubscriptionStatus::Cancelled;
        self.cancelled_at = Some(Timestamp::now());
        self.updated_at = Timestamp::now();
    }

    /// Pause subscription
    pub fn pause(&mut self) -> PaymentResult<()> {
        if self.status != SubscriptionStatus::Active {
            return Err(PaymentError::Internal(
                "Can only pause active subscriptions".to_string(),
            ));
        }

        self.status = SubscriptionStatus::Paused;
        self.updated_at = Timestamp::now();
        Ok(())
    }

    /// Resume paused subscription
    pub fn resume(&mut self) -> PaymentResult<()> {
        if self.status != SubscriptionStatus::Paused {
            return Err(PaymentError::Internal(
                "Can only resume paused subscriptions".to_string(),
            ));
        }

        self.status = SubscriptionStatus::Active;
        self.updated_at = Timestamp::now();
        Ok(())
    }

    /// Upgrade tier
    pub fn upgrade(&mut self, new_tier: SubscriptionTier) -> PaymentResult<Credits> {
        if !self.is_active() {
            return Err(PaymentError::SubscriptionExpired);
        }

        // Calculate prorated credit adjustment
        let remaining_millis = self.current_period_end.unix_millis() - Timestamp::now().unix_millis();
        let total_millis = self.current_period_end.unix_millis() - self.current_period_start.unix_millis();
        let remaining_ratio = remaining_millis as f64 / total_millis as f64;

        let old_monthly = self.tier.monthly_credits();
        let new_monthly = new_tier.monthly_credits();
        let credit_diff = new_monthly.saturating_sub(old_monthly);
        let prorated_credits = (credit_diff as f64 * remaining_ratio) as Credits;

        self.tier = new_tier;
        self.monthly_credits = new_monthly;
        self.credits_remaining = self.credits_remaining.saturating_add(prorated_credits);
        self.updated_at = Timestamp::now();

        Ok(prorated_credits)
    }

    /// Get rate limit for this subscription
    pub fn get_rate_limit(&self) -> u32 {
        self.custom_features
            .as_ref()
            .map(|f| f.rate_limit)
            .unwrap_or_else(|| self.tier.rate_limit())
    }

    /// Get max concurrent for this subscription
    pub fn get_max_concurrent(&self) -> u32 {
        self.custom_features
            .as_ref()
            .map(|f| f.max_concurrent)
            .unwrap_or_else(|| self.tier.max_concurrent())
    }

    fn calculate_period_end(start: &Timestamp, period: BillingPeriod) -> Timestamp {
        let days = match period {
            BillingPeriod::Monthly => 30,
            BillingPeriod::Annual => 365,
        };
        Timestamp::from_unix_millis(start.unix_millis() + (days * 24 * 3600 * 1000))
    }
}

/// Subscription plan for display/selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionPlan {
    /// Plan ID
    pub id: String,
    /// Plan name
    pub name: String,
    /// Description
    pub description: String,
    /// Tier
    pub tier: SubscriptionTier,
    /// Monthly price in cents
    pub monthly_price_cents: u64,
    /// Annual price in cents
    pub annual_price_cents: u64,
    /// Monthly credits
    pub monthly_credits: Credits,
    /// Features list
    pub features: Vec<String>,
    /// Is popular/recommended
    pub is_popular: bool,
}

impl SubscriptionPlan {
    /// Get all available plans
    pub fn all_plans() -> Vec<SubscriptionPlan> {
        vec![
            SubscriptionPlan {
                id: "free".to_string(),
                name: "Free".to_string(),
                description: "Perfect for trying out chip apps".to_string(),
                tier: SubscriptionTier::Free,
                monthly_price_cents: 0,
                annual_price_cents: 0,
                monthly_credits: 100,
                features: vec![
                    "100 credits/month".to_string(),
                    "10 free chip app uses/day".to_string(),
                    "Community support".to_string(),
                    "Basic analytics".to_string(),
                ],
                is_popular: false,
            },
            SubscriptionPlan {
                id: "pro".to_string(),
                name: "Pro".to_string(),
                description: "For developers and power users".to_string(),
                tier: SubscriptionTier::Pro,
                monthly_price_cents: 2900,
                annual_price_cents: 29000,
                monthly_credits: 1000,
                features: vec![
                    "1,000 credits/month".to_string(),
                    "Unlimited chip app uses".to_string(),
                    "Priority support".to_string(),
                    "Advanced analytics".to_string(),
                    "5 concurrent executions".to_string(),
                    "Revenue sharing for published apps".to_string(),
                ],
                is_popular: true,
            },
            SubscriptionPlan {
                id: "enterprise".to_string(),
                name: "Enterprise".to_string(),
                description: "For teams and organizations".to_string(),
                tier: SubscriptionTier::Enterprise,
                monthly_price_cents: 9900,
                annual_price_cents: 99000,
                monthly_credits: 10000,
                features: vec![
                    "10,000 credits/month".to_string(),
                    "Unlimited everything".to_string(),
                    "Dedicated support".to_string(),
                    "Custom analytics".to_string(),
                    "50 concurrent executions".to_string(),
                    "Higher revenue share".to_string(),
                    "SLA guarantee".to_string(),
                    "Custom integrations".to_string(),
                ],
                is_popular: false,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscription_creation() {
        let sub = Subscription::new(
            "user_123".to_string(),
            SubscriptionTier::Pro,
            BillingPeriod::Monthly,
        );

        assert!(sub.is_active());
        assert_eq!(sub.tier, SubscriptionTier::Pro);
        assert_eq!(sub.monthly_credits, 1000);
        assert_eq!(sub.credits_remaining, 1000);
    }

    #[test]
    fn test_subscription_credits() {
        let mut sub = Subscription::new(
            "user_123".to_string(),
            SubscriptionTier::Pro,
            BillingPeriod::Monthly,
        );

        // Use credits
        sub.use_credits(500).unwrap();
        assert_eq!(sub.credits_remaining, 500);

        // Add bonus credits
        sub.add_credits(200);
        assert_eq!(sub.credits_remaining, 700);

        // Insufficient credits
        let result = sub.use_credits(1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_subscription_trial() {
        let sub = Subscription::new_trial("user_123".to_string(), 14);

        assert_eq!(sub.status, SubscriptionStatus::Trial);
        assert_eq!(sub.tier, SubscriptionTier::Pro);
        assert!(sub.trial_end.is_some());
    }

    #[test]
    fn test_subscription_cancel() {
        let mut sub = Subscription::new(
            "user_123".to_string(),
            SubscriptionTier::Pro,
            BillingPeriod::Monthly,
        );

        sub.cancel();
        assert_eq!(sub.status, SubscriptionStatus::Cancelled);
        assert!(sub.cancelled_at.is_some());
    }

    #[test]
    fn test_tier_properties() {
        assert_eq!(SubscriptionTier::Free.monthly_credits(), 100);
        assert_eq!(SubscriptionTier::Pro.monthly_credits(), 1000);
        assert_eq!(SubscriptionTier::Enterprise.monthly_credits(), 10000);

        assert_eq!(SubscriptionTier::Pro.monthly_price_cents(), 2900);
        assert!(SubscriptionTier::Pro.annual_price_cents() < SubscriptionTier::Pro.monthly_price_cents() * 12);
    }

    #[test]
    fn test_subscription_plans() {
        let plans = SubscriptionPlan::all_plans();
        assert_eq!(plans.len(), 3);

        let pro_plan = plans.iter().find(|p| p.tier == SubscriptionTier::Pro).unwrap();
        assert!(pro_plan.is_popular);
    }
}
